"""Tests for the one spelling of a results root, and for the refusal of an unknown one.

Every row of the index names the root it was ingested from and every consumer
filters on the root it was itself pointed at, so the two only meet if both spell
it the same way. A root reaches a program as a command-line value, a
configuration key or an environment variable, and those three routinely differ
by a trailing slash or by being relative to the working directory.

These are assertions about the functions' contracts rather than about any one
backend's behavior: that when a pass finished is read from the runs over the
root it was asked about, and that the refusal of a root nobody ingested names
its index without its password and its roots exactly as they were given.

How a root is spelled is one rule for both storages, so it is asserted where it
lives, in ``tests/spindoctor/nav_records/test_roots.py``.
"""

from pathlib import Path

import pytest
from tests.spindoctor.results_index.conftest import opened, sqlite_url_for

from spindoctor import nav_records
from spindoctor.results_index import (
    INGEST_RUNS,
    SCHEMA_VERSION,
    ingested_roots,
    newest_finish_time,
    normalize_root_url,
    open_index,
    require_ingested_roots,
    unfinished_roots,
)

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

SERVER_URL = f'postgresql+psycopg://svc:{PASSWORD}@db.example/spindoctor'
"""An index URL carrying a password, as a consumer's own resolution produces it."""

FIRST_ROOT = '/data/nav-results'
"""The root the per-root queries are asked about."""

SECOND_ROOT = '/data/other-nav-results'
"""A second root of the same index, passed over after the first."""

FIRST_FINISHED = '2026-02-03T04:05:06+00:00'
"""When the pass over the first root finished."""

SECOND_FINISHED = '2026-03-04T05:06:07+00:00'
"""When the pass over the second root finished, which is later and is not the first."""


def test_the_index_re_exports_the_one_spelling_of_a_results_root() -> None:
    """The rule lives with the record seam, and a consumer of the index reads it here.

    A results root is not a database concept, so the spelling rule is kept where
    a reader of documents can reach it without reaching a database.  It is
    re-exported from the index so that a consumer of rows reads the root and the
    bookkeeping about it out of one place, and so that the two can never come to
    hold different versions of what one root is.
    """
    assert normalize_root_url is nav_records.normalize_root_url


def _refusal_of_an_unknown_root(tmp_path: Path) -> str:
    """Ask an empty index for a root nobody ingested and return the refusal text.

    The index URL handed in is not the one the engine was opened with: the point
    is what the refusal does with the URL its caller names, and a consumer's
    resolved URL is a server URL carrying a password far more often than the
    local file a test can build.

    Parameters:
        tmp_path: Directory the index file lives under.

    Returns:
        The refusal message.
    """
    engine = open_index(sqlite_url_for(tmp_path / 'index.sqlite3'), create=True)
    try:
        with (
            engine.connect() as connection,
            pytest.raises(ValueError, match='no completed ingest') as excinfo,
        ):
            require_ingested_roots(connection, ['/data/nav-results'], url=SERVER_URL)
    finally:
        engine.dispose()
    return str(excinfo.value)


def test_the_refusal_masks_its_index_and_leaves_its_roots_alone(tmp_path: Path) -> None:
    """Three things are true of the refusal, and one call shows all three.

    It is printed to a terminal and written to run logs, so the index password
    may not survive into it; it is masked inside this function rather than by
    each consumer, because a consumer that forgets is a leak in a program nobody
    thought to check. Which of the three resolution levels supplied the URL is
    half of what the message is for, so the rest of the URL has to survive. And
    the root is printed exactly as it was given, because a results root has
    nothing to hide and is the string the reader has to correct.
    """
    refusal = _refusal_of_an_unknown_root(tmp_path)
    assert PASSWORD not in refusal
    assert 'postgresql+psycopg://svc:***@db.example/spindoctor' in refusal
    assert '/data/nav-results' in refusal


def test_the_refusal_leaves_a_credential_shaped_root_alone(tmp_path: Path) -> None:
    """Masking a root would corrupt the one string the message exists to deliver."""
    engine = open_index(sqlite_url_for(tmp_path / 'index.sqlite3'), create=True)
    try:
        with (
            engine.connect() as connection,
            pytest.raises(ValueError, match='no completed ingest') as excinfo,
        ):
            require_ingested_roots(connection, ['//store:8443/nav@results'], url=SERVER_URL)
    finally:
        engine.dispose()
    assert '//store:8443/nav@results' in str(excinfo.value)


def _index_with_two_passed_over_roots(tmp_path: Path) -> str:
    """Build an index whose two roots were each passed over, the second one last.

    The second root's pass is therefore the newest run in the index, and it
    records a finish time of its own.  A query that read the newest run of the
    table rather than the newest run of the root it was asked about would
    answer with that one.

    Parameters:
        tmp_path: Directory the index file is written into.

    Returns:
        The connection URL of the index.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': finished,
                    'finished_utc': finished,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, finished in (
                    (FIRST_ROOT, FIRST_FINISHED),
                    (SECOND_ROOT, SECOND_FINISHED),
                )
            ],
        )
    return url


def test_the_finish_time_is_read_from_the_root_it_was_asked_about(tmp_path: Path) -> None:
    """How old one root's answer is has nothing to do with when another was walked."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_finish_time(connection, FIRST_ROOT) == FIRST_FINISHED


def test_the_other_root_is_answered_for_on_the_same_terms(tmp_path: Path) -> None:
    """And the root whose pass is the newest in the table is answered for as itself."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_finish_time(connection, SECOND_ROOT) == SECOND_FINISHED


def test_a_root_with_no_run_row_has_no_finish_time(tmp_path: Path) -> None:
    """Nothing was recorded for it, which is a different thing from a recorded time."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_finish_time(connection, '/data/never-ingested') is None


def test_a_run_that_never_finished_has_no_finish_time(tmp_path: Path) -> None:
    """The column is nullable, and a NULL there is a pass that did not reach the end.

    A run row is written when the pass starts and stamped when the pass ends, so
    a NULL is a pass still in flight or one that died -- which the
    completed-ingest check refuses before any of this is read.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': FIRST_ROOT,
                    'started_utc': FIRST_FINISHED,
                    'finished_utc': None,
                    'schema_version': SCHEMA_VERSION,
                }
            ],
        )
    with opened(url) as engine, engine.connect() as connection:
        assert newest_finish_time(connection, FIRST_ROOT) is None


def _index_with_one_unfinished_root(tmp_path: Path) -> str:
    """Build an index whose first root was walked to the end and whose second was not.

    The second root's run is the newest in the table and carries no finish time,
    which is what an ingest that started and died leaves behind.  The two roots
    therefore differ in exactly the value the two queries below divide them on.

    Parameters:
        tmp_path: Directory the index file is written into.

    Returns:
        The connection URL of the index.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert(),
            [
                {
                    'root_url': root_url,
                    'started_utc': FIRST_FINISHED,
                    'finished_utc': finished,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, finished in (
                    (FIRST_ROOT, FIRST_FINISHED),
                    (SECOND_ROOT, None),
                )
            ],
        )
    return url


def test_a_root_whose_newest_run_never_finished_is_named_as_unfinished(
    tmp_path: Path,
) -> None:
    """A consumer bound to the roots it may read has to be able to name the rest.

    A narrowing nobody can name reads as an answer about the whole index, so the
    roots left out are asked for by name rather than inferred from the ones that
    were kept.
    """
    url = _index_with_one_unfinished_root(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert unfinished_roots(connection) == [SECOND_ROOT]


def test_a_root_whose_newest_run_finished_is_not_named_as_unfinished(
    tmp_path: Path,
) -> None:
    """The two queries divide the index between them, so neither may claim both roots."""
    url = _index_with_one_unfinished_root(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert ingested_roots(connection) == [FIRST_ROOT]


def test_a_root_walked_to_the_end_after_a_run_that_died_is_not_unfinished(
    tmp_path: Path,
) -> None:
    """The newest run decides, in both directions, and a later good pass repairs a root.

    A root with a dead run followed by a completed one is a root a consumer may
    read absence under, so it is named by neither this query nor a consumer's
    report of what it left out.
    """
    url = _index_with_one_unfinished_root(tmp_path)
    with opened(url, create=False) as engine, engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert().values(
                root_url=SECOND_ROOT,
                started_utc=SECOND_FINISHED,
                finished_utc=SECOND_FINISHED,
                schema_version=SCHEMA_VERSION,
            )
        )
    with opened(url) as engine, engine.connect() as connection:
        assert unfinished_roots(connection) == []
