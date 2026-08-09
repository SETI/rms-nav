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
presence exactly as an ingested document does.

What the index answers differently
----------------------------------

The index holds what one ingest pass could read, so three answers are bounded by
that rather than by this query:

- A summary PNG with no document beside it is not recorded anywhere: the flag
  lives on the row of the document it was found beside.  Such a PNG reads as
  absent, where the tree reads it as present.
- A document the ingest refused, because it is not a per-image navigation
  document, records no status, so it matches no error filter.  The tree reads
  ``status`` out of any JSON object it can parse, and a file that is not a
  navigation document but does carry the field matches there.
- A file a pass could not download is recorded nowhere at all, deliberately, so
  that the next pass tries it again.  It reads as absent until a pass reads it.

The index is also a snapshot: it answers as of the last ingest over the root,
and a document written since is one the index does not hold.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy
from filecache import FCPath

from spindoctor.results_index.engine import open_index
from spindoctor.results_index.roots import normalize_root_url, require_ingested_roots
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
    """

    with_metadata: frozenset[str]
    with_summary_png: frozenset[str]
    matching_error: frozenset[str]


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


def _volume_of(stub: str) -> str | None:
    """Return the volume a results path stub names, as the index records it.

    Parameters:
        stub: The results path stub.

    Returns:
        Its first path segment, or None when it has no separator at all, which
        is what a scene name with no volume above it produces.
    """
    volume, separator, _rest = stub.partition('/')
    return volume if separator else None


def _stub_query(
    root_url: str, volumes: Sequence[str], error_condition: sqlalchemy.ColumnElement[bool]
) -> sqlalchemy.CompoundSelect[tuple[str, bool | None, bool]]:
    """Build the one query the filters are answered from.

    Every term filters on the root, because the index is keyed by root and stub
    together and one database serves several roots: a query that asked about the
    stub alone would answer with another root's images.

    The images are restricted to the selected volumes, which is the restriction
    the tree walk applies by walking only those volumes' directories.  The
    refusals cannot be, because ``failed_files`` records no volume; the caller
    holds them to the same restriction as it reads them.

    Parameters:
        root_url: The normalized root the candidates live under.
        volumes: Volume names the enumeration selected.
        error_condition: What makes an image row match the error filters.

    Returns:
        A query yielding one row per recorded file, carrying its stub, whether a
        summary PNG was recorded beside it, and whether it matches the error
        filters.
    """
    documents = sqlalchemy.select(
        IMAGES.c.results_path_stub,
        IMAGES.c.has_summary_png,
        error_condition.label('matches_error'),
    ).where(IMAGES.c.root_url == root_url, IMAGES.c.volume.in_(volumes))
    # A refused file has no status to match an error filter with, and no summary
    # PNG was recorded for it: the flag belongs to an image row, which a file
    # that is not a navigation document deliberately does not get.
    refusals = sqlalchemy.select(
        FAILED_FILES.c.results_path_stub, sqlalchemy.false(), sqlalchemy.false()
    ).where(FAILED_FILES.c.root_url == root_url)
    return documents.union_all(refusals)


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
        in.

    Raises:
        ValueError: If the index cannot be opened, is stamped with another
            schema version, or holds no completed ingest of this root.  A
            caller never falls back to reading files: a run that resolved a URL
            and could not use it is misconfigured, and reading the tree instead
            would answer the same questions far more slowly and silently
            differently.
    """
    root_url = normalize_root_url(nav_results_root)
    selected = list(volumes)
    query = _stub_query(
        root_url,
        selected,
        _error_condition(
            has_offset_error=has_offset_error,
            has_offset_spice_error=has_offset_spice_error,
            has_offset_nonspice_error=has_offset_nonspice_error,
        ),
    )
    # The volume restriction is applied to every row rather than to the refusals
    # the query could not restrict, so that one rule bounds the answer: what the
    # tree path holds is what a walk of these volumes' directories found, and a
    # stub above them or beside them is not part of that answer.
    selected_volumes = frozenset(selected)
    with_metadata: set[str] = set()
    with_summary_png: set[str] = set()
    matching_error: set[str] = set()
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, [root_url], url=url)
            for stub, has_summary_png, matches_error in connection.execute(query):
                stub_text = str(stub)
                if _volume_of(stub_text) not in selected_volumes:
                    continue
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
    )
