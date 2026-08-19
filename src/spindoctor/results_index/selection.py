"""Answering the results-based image selection filters from the index.

The selection filters ask two questions about each candidate image: whether its
metadata document exists under the results root, and -- for the images whose
document exists -- what fatal error, if any, that document records.  Asked of
the tree, those cost one directory walk per selected subtree plus
a batched download of every document an error filter has to look inside.  Asked
of the index, they cost the one query this module issues, whose answer is two
sets of stubs the filter then tests membership in.

Both tables that record a file are read.  A document the ingest refused is a
file that exists, and the tree answers a presence filter with the file rather
than with its contents, so a refusal recorded in ``failed_files`` counts towards
presence exactly as an ingested document does, and carries the subtree it lives
under for the same reason.

What the index answers differently
----------------------------------

The index holds what one ingest pass could read and record, so the answers below
are bounded by that rather than by this query.  Each is stated here, in the
plan, in the navigation guide's account of ``--results-db``, and in a test of
its own, and one found later is added in the same four places rather than left
to be rediscovered.  The guide is one of them because an operator reading it is
the person a silently short selection is served to: an enumeration a user is
never shown answers nobody's question about the selection they got.

- **A document the ingest refused** -- a JSON object this index will not accept
  -- records no status of its own.  It matches no error filter, the one for a
  document recording no fatal error included, where the tree reads ``status``
  and ``status_error`` out of any JSON object it can parse and answers every one
  of them from what it read, an object carrying no ``status`` at all included.
  A file no JSON object came out of is refused too and is not one of these: the
  tree excludes such a file from every error filter as well, so the two answer
  alike about it.
- **A file the pass could not retrieve** has no row at all in the index and
  reads as absent, which is what the absence filters read as "this image was
  never navigated".  Nothing is recorded for it deliberately: a recorded row
  would be skipped for as long as the file did not change, and a download that
  failed once says nothing that will still be true next pass.

  Two other ways a file could go unrecorded are not divergences, because
  neither leaves a completed pass behind it: an ingest that cannot list a
  directory stops there, and one whose writer the database refuses a document's
  rows stops there.  So a root with a completed pass is a root every directory
  of which was listed and every document of which was stored.

- **A document rewritten in place, keeping the length and the modification time
  it had before,** keeps the row the document before it produced, so an error
  filter answers from what that one recorded.  Those two metrics are everything
  a listing supplies about a file, and reading the file to find out whether it
  needs reading is the retrieval the skip exists to avoid, so no number of
  completed passes corrects this one: an ingest told to read every document
  regardless is what puts the row right.  A tree restored by a copy that
  preserves times, a document patched and stamped back from a sibling, and a
  backend reporting one modification time for two writes all produce it; an
  ordinary re-navigation writes a different length at a later time and does not.

The index is also a snapshot: it answers as of the last ingest over the root, so
a document written since is one it does not hold and a document deleted since is
one it still holds.  When that pass finished is returned as
:attr:`ResultStubs.ingested_utc` and reported with the answer, because outside
the members above that is what decides whether the answer is the answer the tree
would give.  Inside them the age decides nothing: each of those survives a pass
that finished a second ago, which is why each is stated here rather than left to
be read off the stamp.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy
from filecache import FCPath

from spindoctor.results_index.engine import reporting_a_failed_read
from spindoctor.results_index.roots import (
    newest_finish_time,
    normalize_root_url,
    open_index_for_roots,
)
from spindoctor.results_index.schema import FAILED_FILES, IMAGES

__all__ = [
    'FATAL_STATUS',
    'SPICE_STATUS_ERROR',
    'ResultStubs',
    'read_result_stubs',
]

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
        matching_error: Stubs whose document satisfies the error filters that
            were asked for, and empty when none were.
        ingested_utc: When the newest pass over the root finished, so a caller
            can say how old the answer is, and None when the index recorded no
            finish time.
    """

    with_metadata: frozenset[str]
    matching_error: frozenset[str]
    ingested_utc: str | None = None


def _error_condition(
    *,
    has_offset_error: bool,
    has_no_offset_error: bool,
    has_offset_spice_error: bool,
    has_offset_nonspice_error: bool,
) -> sqlalchemy.ColumnElement[bool]:
    """Build the condition an image row must satisfy to match the error filters.

    The filters are conjoined here as they are everywhere else, so a caller
    asking both for a fatal error and for none gets the empty selection that
    describes, rather than one of the two.

    Parameters:
        has_offset_error: Whether any fatal error is wanted.
        has_no_offset_error: Whether a document recording no fatal error is
            wanted.
        has_offset_spice_error: Whether only a missing-SPICE-data error is
            wanted.
        has_offset_nonspice_error: Whether only a fatal error other than
            missing SPICE data is wanted.

    Returns:
        The condition, which is false for every row when no error filter is
        active.
    """
    conditions: list[sqlalchemy.ColumnElement[bool]] = []
    if has_offset_error or has_offset_spice_error or has_offset_nonspice_error:
        conditions.append(IMAGES.c.status == FATAL_STATUS)
    if has_no_offset_error:
        # No guard against NULL, unlike status_error below: the column forbids
        # one, and a document naming no outcome anywhere is recorded with the
        # status the ingest gives it in place of a missing one, so the
        # inequality answers for every row there is.
        conditions.append(IMAGES.c.status != FATAL_STATUS)
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
    if not conditions:
        return sqlalchemy.false()
    return sqlalchemy.and_(*conditions)


def _stub_query(
    root_url: str, subtrees: Sequence[str], error_condition: sqlalchemy.ColumnElement[bool]
) -> sqlalchemy.CompoundSelect[tuple[str, bool]]:
    """Build the one query the filters are answered from.

    Every term filters on the root, because the index is keyed by root and stub
    together and one database serves several roots: a query that asked about the
    stub alone would answer with another root's images.

    Both arms are restricted to the selected subtrees, which is the restriction
    the tree walk applies by walking only those subtrees' directories.  A stub
    with no subtree above it -- a bare scene name -- is under no walked directory
    and is matched by neither arm, because SQL's ``IN`` is false for NULL.

    Parameters:
        root_url: The normalized root the candidates live under.
        subtrees: Top-level directories of the root the enumeration selected.
        error_condition: What makes an image row match the error filters.

    Returns:
        A query yielding one row per recorded file of the selected subtrees,
        carrying its stub and whether it matches the error filters.
    """
    documents = sqlalchemy.select(
        IMAGES.c.results_path_stub,
        error_condition.label('matches_error'),
    ).where(IMAGES.c.root_url == root_url, IMAGES.c.subtree.in_(subtrees))
    # A refused file records no status, so it matches no error filter -- the
    # one for a document recording no fatal error included, since what such a
    # file records is unknown rather than known to be an outcome, which is also
    # how the tree path reads a document nothing can be parsed out of.
    refusals = sqlalchemy.select(
        FAILED_FILES.c.results_path_stub,
        sqlalchemy.false(),
    ).where(FAILED_FILES.c.root_url == root_url, FAILED_FILES.c.subtree.in_(subtrees))
    return documents.union_all(refusals)


def read_result_stubs(
    url: str,
    nav_results_root: str | Path | FCPath,
    subtrees: Iterable[str],
    *,
    has_offset_error: bool = False,
    has_no_offset_error: bool = False,
    has_offset_spice_error: bool = False,
    has_offset_nonspice_error: bool = False,
) -> ResultStubs:
    """Read what a results root holds, for one enumeration's selection filters.

    One query per enumeration, rather than a walk per subtree and a download per
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
        subtrees: Top-level directories of the root the enumeration selected.
            Only images under these are read, exactly as only these are walked
            in the tree.
        has_offset_error: Whether any fatal error is wanted.
        has_no_offset_error: Whether a document recording no fatal error is
            wanted.  A file the ingest refused records no outcome at all and
            satisfies this no more than it satisfies the others.
        has_offset_spice_error: Whether only a missing-SPICE-data error is
            wanted.
        has_offset_nonspice_error: Whether only a fatal error other than
            missing SPICE data is wanted.

    Returns:
        The stubs the root holds, in the two sets the filters test membership
        in, with when the pass that recorded them finished.

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
        list(subtrees),
        _error_condition(
            has_offset_error=has_offset_error,
            has_no_offset_error=has_no_offset_error,
            has_offset_spice_error=has_offset_spice_error,
            has_offset_nonspice_error=has_offset_nonspice_error,
        ),
    )
    with_metadata: set[str] = set()
    matching_error: set[str] = set()
    engine = open_index_for_roots(url, [root_url])
    try:
        with reporting_a_failed_read(url), engine.connect() as connection:
            ingested_utc = newest_finish_time(connection, root_url)
            for stub, matches_error in connection.execute(query):
                stub_text = str(stub)
                with_metadata.add(stub_text)
                if matches_error:
                    matching_error.add(stub_text)
    finally:
        engine.dispose()
    return ResultStubs(
        with_metadata=frozenset(with_metadata),
        matching_error=frozenset(matching_error),
        ingested_utc=ingested_utc,
    )
