"""Tests for the worker that ingests one share of a results root.

``sd_stats_ingest_cloud_tasks`` is the middle of the three steps: it is handed
the files of a share and no root to walk, and what is pinned here is the split
of authority that makes that safe.  It creates no schema, because a worker that
did would answer a mistyped URL by building an empty index beside the real one
and every consumer would read absence of a row in it as "this image was never
navigated".  It has no run log either, so its whole account of itself -- what it
ingested, what it skipped, and every file it could not read -- is its return
value, which is what the program that adds the shares up reads.

The command lines that divide a root up and put it back together are in
``test_ingest_cloud_tasks_driver``.
"""

from pathlib import Path

import pytest
from tests.spindoctor.conftest import (
    index_url,
    metadata_document,
    write_metadata,
)

from spindoctor.cli import sd_stats_ingest, sd_stats_ingest_cloud_tasks

from .ingest_driver_helpers import (
    STUB,
    fanned_out,
    process,
    run_driver,
    tasks_of,
    worker_data,
)

# ---------------------------------------------------------------------------
# Who creates the schema
# ---------------------------------------------------------------------------


def test_the_fan_out_creates_the_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One program makes the schema, before any worker is given a task."""
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    database = tmp_path / 'index.sqlite3'
    run_driver(
        [
            '--results-db',
            index_url(database),
            '--nav-results-root',
            root.as_posix(),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert database.exists()


def test_a_worker_refuses_an_index_that_is_not_there(tmp_path: Path) -> None:
    """A worker that created one would answer a wrong URL with an empty index.

    Every consumer would then read absence of a row under a fully navigated root
    as "this image was never navigated", which is the one thing the run
    bookkeeping exists to prevent.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    _retry, result = process(
        {
            'run_id': 1,
            'root_url': root.as_posix(),
            'force': False,
            'has_file_metrics': False,
            'files': [
                {
                    'results_path_stub': STUB,
                    'mtime_ns': None,
                    'size_bytes': None,
                }
            ],
        },
        index_url(tmp_path / 'index.sqlite3'),
    )
    assert result['status_error'] == 'index_unopenable'


def test_a_worker_leaves_no_index_behind(tmp_path: Path) -> None:
    """Refusing is not enough if the refusal creates the file on the way out."""
    database = tmp_path / 'index.sqlite3'
    process(
        {
            'run_id': 1,
            'root_url': str(tmp_path / 'results'),
            'force': False,
            'has_file_metrics': False,
            'files': [],
        },
        index_url(database),
    )
    assert not database.exists()


LEAKING_PASSWORD = 'se@cr:etlongsecretpassword'
"""A password whose tail a URL parser quotes back as the port it could not read."""

LEAKING_INDEX_URL = f'postgresql+psycopg://user:{LEAKING_PASSWORD}@dbhost/spindoctor'
"""An index URL whose refusal is where that tail would otherwise appear."""


def test_a_worker_that_cannot_open_the_index_names_no_password() -> None:
    """A task result travels further than a log line, so a leak in one travels too.

    What a worker returns is written verbatim into its event log, and an
    operator collects those logs, concatenates them and hands the file to the
    program that completes the ingest. A refusal that masks the URL and then
    quotes the parser's own complaint about it puts a run of the password in
    that file.
    """
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db=LEAKING_INDEX_URL)
    )
    assert 'etlongsecretpassword' not in result['status_exception']


def test_a_worker_that_cannot_open_the_index_still_says_why() -> None:
    """And keeps the diagnosis, which is the whole of what the result is for."""
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db=LEAKING_INDEX_URL)
    )
    assert result['status_error'] == 'index_unopenable'


def test_a_worker_with_no_index_url_reports_it() -> None:
    """A worker has no run log, so the missing setting comes back in the result."""
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db=None)
    )
    assert result['status_error'] == 'no_results_db'


def test_a_worker_told_to_use_no_index_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``none`` is the documented opt-out, and there is nothing for a worker to do.

    Every other program answers it by reading files; ingest into no index is not
    a mode that exists.
    """
    monkeypatch.setenv('NAV_RESULTS_DB', index_url(tmp_path / 'index.sqlite3'))
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db='none')
    )
    assert result['status_error'] == 'no_results_db'


# ---------------------------------------------------------------------------
# What a worker returns
# ---------------------------------------------------------------------------


def test_a_worker_reports_what_its_share_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task result is the only channel a worker has."""
    url = fanned_out(tmp_path, monkeypatch, count=2)
    tasks = tasks_of(tmp_path / 'tasks.json')
    _retry, result = process(tasks[0]['data'], url)
    assert result['files_ingested'] == 2


def test_a_worker_never_asks_for_a_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A share that failed halfway is re-run by an operator, not by the queue.

    Its files are unaccounted for either way, and the run it belongs to stays
    unfinished until they are.
    """
    url = fanned_out(tmp_path, monkeypatch)
    tasks = tasks_of(tmp_path / 'tasks.json')
    retry, _result = process(tasks[0]['data'], url)
    assert retry is False


def test_a_worker_handed_a_task_it_cannot_read_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed task is an error result, not a traceback out of the worker."""
    url = fanned_out(tmp_path, monkeypatch)
    _retry, result = process({'run_id': 1}, url)
    assert result['status_error'] == 'malformed_task'


def test_a_worker_says_what_was_wrong_with_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal nobody can act on is only half a refusal."""
    url = fanned_out(tmp_path, monkeypatch)
    _retry, result = process({'run_id': 1}, url)
    assert 'root_url' in result['status_exception']


def test_the_worker_shares_its_interactive_siblings_identity() -> None:
    """One ``logging.programs`` block governs both forms of a program."""
    assert sd_stats_ingest_cloud_tasks.PROGRAM_NAME == sd_stats_ingest.PROGRAM_NAME
