"""Tests for the two command lines that divide an ingest up and put it together.

``sd_stats_ingest`` gains two modes and ``sd_stats_ingest_cloud_tasks`` is the
worker between them.  What is pinned here is the split of authority: the mode
that divides the work up is the only one that creates the schema, a worker
refuses an index that is not there rather than building one beside it, and a
worker's whole account of itself is its return value, because it has no run log
to write one in.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import sqlalchemy
from cloud_tasks.worker import WorkerData

from spindoctor.cli import sd_stats_ingest, sd_stats_ingest_cloud_tasks
from spindoctor.config import MAIN_LOGGER
from spindoctor.results_index import IMAGES, open_index

from .conftest import index_url, metadata_document, write_metadata

STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the document every tree below holds."""


class _StubWorkerData:
    """Stands in for the cloud_tasks worker's data object."""

    def __init__(self, **kwargs: object) -> None:
        """Build worker data carrying only the given CLI arguments.

        Parameters:
            **kwargs: Argument names and values for the parsed namespace.
        """
        self.args = argparse.Namespace(config_file=None, log_root=None, **kwargs)


def worker_data(**kwargs: object) -> WorkerData:
    """Build the worker data a driver reads its CLI arguments from.

    Parameters:
        **kwargs: Argument names and values for the parsed namespace.

    Returns:
        The stub, typed as the worker data a driver expects.  A driver reads
        only ``args`` from it, so building the real thing would mean standing up
        a worker for no benefit.
    """
    return cast(WorkerData, _StubWorkerData(**kwargs))


def run_driver(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[int | None, list[str]]:
    """Run ``sd_stats_ingest`` and return its exit status and its main log.

    Parameters:
        argv: Arguments, without the program name.
        monkeypatch: Fixture the argument vector and logger are replaced through.
        tmp_path: Directory the run's log files are written under.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    written: list[str] = []

    def recording(message: Any, *args: Any) -> None:
        written.append(str(message) % args if args else str(message))

    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    monkeypatch.setattr(
        sys, 'argv', ['sd_stats_ingest', '--log-root', str(tmp_path / 'logs'), *argv]
    )
    for level in ('info', 'warning', 'error', 'fatal', 'exception'):
        monkeypatch.setattr(MAIN_LOGGER, level, recording)
    with pytest.raises(SystemExit) as caught:
        sd_stats_ingest.main()
    status = caught.value.code
    return (status if status is None or isinstance(status, int) else 1), written


def tasks_of(path: Path) -> list[dict[str, Any]]:
    """Read a written cloud-tasks file.

    Parameters:
        path: The file the driver wrote.

    Returns:
        The task descriptions.
    """
    return cast(list[dict[str, Any]], json.loads(path.read_text(encoding='utf-8')))


def process(task_data: dict[str, Any], url: str) -> tuple[bool, Any]:
    """Run one ingest task through the worker driver.

    Parameters:
        task_data: The task's data.
        url: The index URL to hand the worker.

    Returns:
        What ``process_task`` returned.
    """
    return sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', task_data, worker_data(results_db=url)
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


def test_a_worker_refuses_an_index_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that created one would answer a wrong URL with an empty index.

    Every consumer would then read absence of a row under a fully navigated root
    as "this image was never navigated", which is the one thing the run
    bookkeeping exists to prevent.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
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
                    'has_summary_png': False,
                }
            ],
        },
        index_url(tmp_path / 'index.sqlite3'),
    )
    assert result['status_error'] == 'index_unopenable'


def test_a_worker_leaves_no_index_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusing is not enough if the refusal creates the file on the way out."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
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


def test_a_worker_with_no_index_url_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker has no run log, so the missing setting comes back in the result."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
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


def fanned_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 1) -> str:
    """Write a tree, fan it out through the driver, and return the index URL.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.
        count: How many documents to write.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    for index in range(count):
        name = f'N{1454725799 + index}_1_CALIB'
        write_metadata(root, f'VOL/{name}', metadata_document(image_name=f'{name}.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            root.as_posix(),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    return url


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


# ---------------------------------------------------------------------------
# The command line that divides the work up
# ---------------------------------------------------------------------------


def test_the_driver_writes_the_tasks_it_divided_the_root_into(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file is what an operator loads into a queue."""
    fanned_out(tmp_path, monkeypatch, count=3)
    handed = [
        entry['results_path_stub']
        for task in tasks_of(tmp_path / 'tasks.json')
        for entry in task['data']['files']
    ]
    assert len(handed) == 3


def test_the_driver_reads_no_document_when_it_is_dividing_the_work_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The work is the workers'; this mode lists, removes and hands out."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES)
            ).scalar()
    finally:
        engine.dispose()
    assert found == 0


def test_dividing_a_root_that_is_not_there_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule as a pass that reads the documents: a listing failed."""
    status, _written = run_driver(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_the_two_cloud_modes_cannot_be_asked_for_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dividing the work up and adding it together are different runs.

    Asking for both would write a tasks file and then complete the run it had
    just created, before a single worker had read anything.
    """
    monkeypatch.setattr(sys, 'argv', ['sd_stats_ingest'])
    with pytest.raises(SystemExit) as caught:
        sd_stats_ingest.parse_args(
            [
                '--output-cloud-tasks-file',
                str(tmp_path / 'tasks.json'),
                '--complete-cloud-tasks-file',
                str(tmp_path / 'events.log'),
            ]
        )
    assert caught.value.code == 2


# ---------------------------------------------------------------------------
# The command line that puts it back together
# ---------------------------------------------------------------------------


def write_event_log(path: Path, results: list[Any]) -> Path:
    """Write a cloud-tasks event log holding the given task results.

    Parameters:
        path: Where to write it.
        results: What each task returned.

    Returns:
        The path written.
    """
    lines = [
        json.dumps({'event_type': 'task_completed', 'task_id': f'ingest-{n}', 'result': result})
        for n, result in enumerate(results)
    ]
    path.write_text(''.join(f'{line}\n' for line in lines), encoding='utf-8')
    return path


def test_the_driver_completes_the_run_from_an_event_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full cycle through the two command lines and the worker between them."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    results = [process(task['data'], url)[1] for task in tasks_of(tmp_path / 'tasks.json')]
    write_event_log(tmp_path / 'events.log', results)
    status, _written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 0


def test_completing_a_run_the_tasks_did_not_cover_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stamped without every share would license a wrong answer."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log(tmp_path / 'events.log', [])
    status, _written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_completing_a_run_the_tasks_did_not_cover_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shortfall is named, with the root, so an operator knows what to re-run."""
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log(tmp_path / 'events.log', [])
    _status, written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('0 of 3 file(s) accounted for' in line for line in written)


def test_completing_a_root_nobody_divided_up_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no run to stamp, and saying so is the whole of the diagnosis."""
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = fanned_out(tmp_path, monkeypatch)
    write_event_log(tmp_path / 'events.log', [])
    status, _written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            str(tmp_path / 'other-results'),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_completing_against_an_index_that_is_not_there_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creating an empty one would report every root as never divided up.

    The runs the operator meant to complete are sitting in the index they meant
    to name, and an empty index beside it answers the question wrongly.
    """
    write_event_log(tmp_path / 'events.log', [])
    database = tmp_path / 'index.sqlite3'
    status, _written = run_driver(
        [
            '--results-db',
            index_url(database),
            '--nav-results-root',
            str(tmp_path / 'results'),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_completing_against_an_index_that_is_not_there_creates_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And leaves nothing behind on the way out."""
    write_event_log(tmp_path / 'events.log', [])
    database = tmp_path / 'index.sqlite3'
    run_driver(
        [
            '--results-db',
            index_url(database),
            '--nav-results-root',
            str(tmp_path / 'results'),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert not database.exists()
