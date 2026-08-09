"""Answering the results-based image selection filters from the index.

The selection filters ask three questions about each candidate image: whether
its metadata document exists under the results root, whether its summary PNG
exists beside it, and -- for the images whose document exists -- whether that
document records a fatal error of a particular kind.  Asked of the tree, those
cost one directory walk per selected volume plus a batched download of every
document an error filter has to look inside.  Asked of the index, they cost the
one query this module issues, whose answer is three sets of stubs the filter
then tests membership in.

Both tables that record a file are read.  A document the ingest refused is a
file that exists, and the tree answers a presence filter with the file rather
than with its contents, so a refusal recorded in ``failed_files`` counts towards
presence exactly as an ingested document does, and carries the volume it lives
under and the summary PNG the walk saw beside it for the same reason.

What the index answers differently
----------------------------------

The index holds what one ingest pass could read and record, so the answers below
are bounded by that rather than by this query.  Each is stated here, in the
plan, and in a test of its own, and one found later is added in the same three
places rather than left to be rediscovered.

- **A summary PNG with no document beside it** is recorded nowhere: the flag
  lives on the row of the file it was found beside, and a PNG on its own has no
  file to be beside.  It reads as absent, where the tree reads it as present,
  which makes ``--has-no-offset-file --has-png-file`` empty under an index.
  Recording it needs a row keyed by a stub no document backs, and every other
  reader of these tables takes such a row as evidence that a document exists.
- **A document that is valid JSON and carries ``status``, but is not a
  navigation document,** is refused by the ingest and so records no status of
  its own.  It matches no error filter, where the tree reads ``status`` and
  ``status_error`` out of any JSON object it can parse.
- **A document whose top-level ``status`` is absent, empty, or not a string**
  takes its recorded status from ``navigation_result.status``, which is where
  the rest of the index reads an outcome from.  The tree reads the top-level
  field alone, so such a document can match an error filter here and not there.
- **A file that exists and has no row at all** reads as absent, which is what
  the absence filters read as "this image was never navigated".  Three passes
  end that way, and the first two do so deliberately, because a recorded row
  would be skipped for as long as the file did not change and the next pass
  would never retry it:

  - a file the pass could not retrieve;
  - a document the pass read whose rows the database would not store;
  - a file under a directory the walk did not list, either because it could not
    be listed or because it had already been listed under another name.  That
    one is counted rather than invisible: the count is on the run row, it is
    returned as :attr:`ResultStubs.directories_missed`, and the caller reports
    it rather than reading absence under it in silence.

- **A document the tree no longer holds** keeps its row, and reads as present
  where the tree reads it as absent, which also makes ``--has-no-offset-file``
  skip an image nothing has been written for.  A row leaves the index only when
  a pass that listed the whole root finds no file for it, and a pass that missed
  a single directory anywhere under the root removes no row at all: the stubs it
  did not see are the ones it has no evidence about.  So one unlistable
  subdirectory holds every stale row of the root for as long as it stays
  unlistable, however many passes complete in the meantime, and the count of
  missed directories is what says so.

The index is also a snapshot: it answers as of the last ingest over the root, so
a document written since is one it does not hold and a document deleted since is
one it still holds.  When that pass finished is returned as
:attr:`ResultStubs.ingested_utc` and reported with the answer, because how old
the answer is decides whether it is the answer the tree would give.
"""

import contextlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy
from filecache import FCPath

from spindoctor.results_index.engine import masked_url, open_index
from spindoctor.results_index.roots import (
    newest_pass,
    normalize_root_url,
    require_ingested_roots,
)
from spindoctor.results_index.schema import FAILED_FILES, IMAGES

__all__ = ['FATAL_STATUS', 'SPICE_STATUS_ERROR', 'ResultStubs', 'read_result_stubs']

FATAL_STATUS = 'error'
"""Value of ``status`` that the error filters select on.

An image whose navigation failed outright.  The other statuses describe a run
that finished, whatever it concluded, and no error filter selects one.
"""

SPICE_STATUS_ERROR = 'missing_spice_data'
"""Value of ``status_error`` that the SPICE error filters tell apart.

Matched verbatim, which is what makes ``status_error`` a column of its own: it
is the navigator's machine-readable classification of a fatal error, distinct
from the prose of ``status_reason``.
"""


@dataclass(frozen=True)
class ResultStubs:
    """What one root holds, in the terms the selection filters ask about.

    Parameters:
        with_metadata: Stubs the index holds a metadata document for, an
            ingested one and a refused one alike, since both are a file that
            exists under the root.
        with_summary_png: Stubs the ingest walk saw a summary PNG beside.
        matching_error: Stubs whose document satisfies the error filters that
            were asked for, and empty when none were.
        directories_missed: How many directories the newest pass over the root
            did not list, so a caller reading absence as an answer can say that
            the answer does not cover all of the root.
        ingested_utc: When that pass finished, so a caller can say how old the
            answer is, and None when the index recorded no finish time.
    """

    with_metadata: frozenset[str]
    with_summary_png: frozenset[str]
    matching_error: frozenset[str]
    directories_missed: int = 0
    ingested_utc: str | None = None


def _error_condition(
    *, has_offset_error: bool, has_offset_spice_error: bool, has_offset_nonspice_error: bool
) -> sqlalchemy.ColumnElement[bool]:
    """Build the condition an image row must satisfy to match the error filters.

    Parameters:
        has_offset_error: Whether any fatal error is wanted.
        has_offset_spice_error: Whether only a missing-SPICE-data error is
            wanted.
        has_offset_nonspice_error: Whether only a fatal error other than
            missing SPICE data is wanted.

    Returns:
        The condition, which is false for every row when no error filter is
        active.
    """
    if not (has_offset_error or has_offset_spice_error or has_offset_nonspice_error):
        return sqlalchemy.false()
    conditions: list[sqlalchemy.ColumnElement[bool]] = [IMAGES.c.status == FATAL_STATUS]
    if has_offset_spice_error:
        conditions.append(IMAGES.c.status_error == SPICE_STATUS_ERROR)
    if has_offset_nonspice_error:
        # A document that records no status_error at all is not a SPICE error,
        # and an inequality alone would drop it: comparing NULL with anything
        # yields NULL, which is not true, so the row would fail a filter it
        # satisfies.
        conditions.append(
            sqlalchemy.or_(
                IMAGES.c.status_error.is_(None), IMAGES.c.status_error != SPICE_STATUS_ERROR
            )
        )
    return sqlalchemy.and_(*conditions)


def _stub_query(
    root_url: str, volumes: Sequence[str], error_condition: sqlalchemy.ColumnElement[bool]
) -> sqlalchemy.CompoundSelect[tuple[str, bool | None, bool]]:
    """Build the one query the filters are answered from.

    Every term filters on the root, because the index is keyed by root and stub
    together and one database serves several roots: a query that asked about the
    stub alone would answer with another root's images.

    Both arms are restricted to the selected volumes, which is the restriction
    the tree walk applies by walking only those volumes' directories.  A stub
    with no volume above it -- a bare scene name -- is under no walked directory
    and is matched by neither arm, because SQL's ``IN`` is false for NULL.

    Parameters:
        root_url: The normalized root the candidates live under.
        volumes: Volume names the enumeration selected.
        error_condition: What makes an image row match the error filters.

    Returns:
        A query yielding one row per recorded file of the selected volumes,
        carrying its stub, whether a summary PNG was recorded beside it, and
        whether it matches the error filters.
    """
    documents = sqlalchemy.select(
        IMAGES.c.results_path_stub,
        IMAGES.c.has_summary_png,
        error_condition.label('matches_error'),
    ).where(IMAGES.c.root_url == root_url, IMAGES.c.volume.in_(volumes))
    # A refused file records no status, so it matches no error filter; the walk
    # saw its summary PNG all the same, because a PNG is found beside a file
    # rather than read out of it.
    refusals = sqlalchemy.select(
        FAILED_FILES.c.results_path_stub,
        FAILED_FILES.c.has_summary_png,
        sqlalchemy.false(),
    ).where(FAILED_FILES.c.root_url == root_url, FAILED_FILES.c.volume.in_(volumes))
    return documents.union_all(refusals)


@contextlib.contextmanager
def _reporting_a_failed_read(url: str) -> Iterator[None]:
    """Report a database failure as the refusal every consumer already catches.

    :func:`~spindoctor.results_index.engine.open_index` goes to some length to
    make every way of failing to open the index a ``ValueError``, so that a
    consumer reporting the cause rather than crashing catches one type.  The
    queries issued afterwards are outside that guarantee on their own: a table
    the account may not read, a database holding part of the schema, or a
    connection lost between the open and the query raises the database layer's
    own exception, which a caller that deliberately never imports SQLAlchemy
    cannot name in an ``except`` clause.

    Parameters:
        url: The index URL, masked here so that the report names which index
            was asked without printing its password.

    Yields:
        Nothing; the queries run inside the translation.

    Raises:
        ValueError: If the block raises anything the database layer raised.
    """
    try:
        yield
    except sqlalchemy.exc.SQLAlchemyError as exc:
        raise ValueError(
            f'{masked_url(url)}: the results index could not be read '
            f'({type(exc).__name__}: {exc}). Check that this URL names an index '
            f'sd_stats_ingest wrote and that the account it is opened with may read '
            f'every table of it.'
        ) from exc


def read_result_stubs(
    url: str,
    nav_results_root: str | Path | FCPath,
    volumes: Iterable[str],
    *,
    has_offset_error: bool = False,
    has_offset_spice_error: bool = False,
    has_offset_nonspice_error: bool = False,
) -> ResultStubs:
    """Read what a results root holds, for one enumeration's selection filters.

    One query per enumeration, rather than a walk per volume and a download per
    document.  The index is opened, asked, and closed here, so the caller holds
    no database object and no connection outlives the answer.

    A root the index has no completed ingest of is refused rather than answered:
    absence of a row is what the absence filters read as "this image was never
    navigated", and under a root nobody ingested that reading is simply false.

    Parameters:
        url: Connection URL of the results index.
        nav_results_root: The results root the candidates live under, in
            whatever spelling its holder has; it is normalized here, so a
            relative path or a trailing separator names the same root the
            ingest recorded.
        volumes: Volume names the enumeration selected.  Only images under
            these are read, exactly as only these are walked in the tree.
        has_offset_error: Whether any fatal error is wanted.
        has_offset_spice_error: Whether only a missing-SPICE-data error is
            wanted.
        has_offset_nonspice_error: Whether only a fatal error other than
            missing SPICE data is wanted.

    Returns:
        The stubs the root holds, in the three sets the filters test membership
        in, with how much of the root the pass that recorded them missed and
        when it finished.

    Raises:
        ValueError: If the index cannot be opened, is stamped with another
            schema version, holds no completed ingest of this root, or fails
            the queries this asks it.  A caller never falls back to reading
            files: a run that resolved a URL and could not use it is
            misconfigured, and reading the tree instead would answer the same
            questions far more slowly and silently differently.
    """
    root_url = normalize_root_url(nav_results_root)
    query = _stub_query(
        root_url,
        list(volumes),
        _error_condition(
            has_offset_error=has_offset_error,
            has_offset_spice_error=has_offset_spice_error,
            has_offset_nonspice_error=has_offset_nonspice_error,
        ),
    )
    with_metadata: set[str] = set()
    with_summary_png: set[str] = set()
    matching_error: set[str] = set()
    engine = open_index(url)
    try:
        with _reporting_a_failed_read(url), engine.connect() as connection:
            require_ingested_roots(connection, [root_url], url=url)
            newest = newest_pass(connection, root_url)
            for stub, has_summary_png, matches_error in connection.execute(query):
                stub_text = str(stub)
                with_metadata.add(stub_text)
                if has_summary_png:
                    with_summary_png.add(stub_text)
                if matches_error:
                    matching_error.add(stub_text)
    finally:
        engine.dispose()
    return ResultStubs(
        with_metadata=frozenset(with_metadata),
        with_summary_png=frozenset(with_summary_png),
        matching_error=frozenset(matching_error),
        directories_missed=newest.directories_missed,
        ingested_utc=newest.finished_utc,
    )
