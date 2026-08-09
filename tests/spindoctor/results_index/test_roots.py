"""Tests for the one spelling of a results root, and for the refusal of an unknown one.

Every row of the index names the root it was ingested from and every consumer
filters on the root it was itself pointed at, so the two only meet if both spell
it the same way. A root reaches a program as a command-line value, a
configuration key or an environment variable, and those three routinely differ
by a trailing slash or by being relative to the working directory.

These are assertions about the functions' contracts rather than about any one
backend's behavior: that two spellings of one root produce one string, that the
filesystem root -- the one root whose separator is its whole name -- survives
intact, that what one pass recorded about its own reach is read from the runs
over the root it was asked about, and that the refusal of a root nobody ingested
names its index without its password and its roots exactly as they were given.
"""

from pathlib import Path

import pytest
from filecache import FCPath
from tests.spindoctor.results_index.conftest import opened, sqlite_url_for

from spindoctor.results_index import (
    INGEST_RUNS,
    SCHEMA_VERSION,
    newest_pass,
    normalize_root_url,
    open_index,
    require_ingested_roots,
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


def test_a_trailing_separator_does_not_make_two_roots() -> None:
    """One program writes the trailing slash and another does not."""
    assert normalize_root_url('/data/nav-results/') == normalize_root_url('/data/nav-results')


def test_a_repeated_separator_does_not_make_two_roots() -> None:
    """A root pasted together from two settings arrives with the join doubled."""
    assert normalize_root_url('/data//nav-results') == '/data/nav-results'


def test_a_relative_root_becomes_an_absolute_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root named relatively on one run and absolutely on the next is one root."""
    monkeypatch.chdir(tmp_path)
    assert normalize_root_url('results') == (tmp_path / 'results').as_posix()


def test_the_filesystem_root_keeps_its_separator() -> None:
    """It is the one root whose trailing separator is the whole of its name."""
    assert normalize_root_url('/') == '/'


def test_a_cloud_root_keeps_its_scheme() -> None:
    """A results root is any URL the file layer accepts, not only a local path."""
    assert normalize_root_url('gs://rms-nav-results/coiss/') == 'gs://rms-nav-results/coiss'


def test_an_fcpath_normalizes_the_same_way_as_its_text() -> None:
    """A caller holding the path object must not get a second spelling of it."""
    assert normalize_root_url(FCPath('/data/nav-results/')) == normalize_root_url(
        '/data/nav-results'
    )


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
    records a finish time and a missed count the first root's never records.  A
    query that read the newest run of the table rather than the newest run of
    the root it was asked about would answer with that one.

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
                    'directories_missed': missed,
                    'schema_version': SCHEMA_VERSION,
                }
                for root_url, finished, missed in (
                    (FIRST_ROOT, FIRST_FINISHED, 0),
                    (SECOND_ROOT, SECOND_FINISHED, 4),
                )
            ],
        )
    return url


def test_the_missed_count_is_read_from_the_root_it_was_asked_about(tmp_path: Path) -> None:
    """One index serves several roots, and the newest run in it is routinely another's."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, FIRST_ROOT).directories_missed == 0


def test_the_finish_time_is_read_from_the_root_it_was_asked_about(tmp_path: Path) -> None:
    """How old one root's answer is has nothing to do with when another was walked."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, FIRST_ROOT).finished_utc == FIRST_FINISHED


def test_the_other_root_is_answered_for_on_the_same_terms(tmp_path: Path) -> None:
    """The gap the second root does have is reported when the second root is asked about."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, SECOND_ROOT).directories_missed == 4


def test_a_root_with_no_run_row_has_no_gap(tmp_path: Path) -> None:
    """A root this index never passed over borrows no other root's coverage."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, '/data/never-ingested').directories_missed == 0


def test_a_root_with_no_run_row_has_no_finish_time(tmp_path: Path) -> None:
    """Nothing was recorded for it, which is a different thing from a recorded zero."""
    url = _index_with_two_passed_over_roots(tmp_path)
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, '/data/never-ingested').finished_utc is None


def test_a_pass_that_recorded_no_count_reads_as_no_gap(tmp_path: Path) -> None:
    """The column is nullable, and a NULL there is not a gap of unknown size.

    A run row is written when the pass starts and its count is stamped when the
    pass ends, so a NULL is a pass that did not reach the end -- which the
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
                    'directories_missed': None,
                    'schema_version': SCHEMA_VERSION,
                }
            ],
        )
    with opened(url) as engine, engine.connect() as connection:
        assert newest_pass(connection, FIRST_ROOT).directories_missed == 0
