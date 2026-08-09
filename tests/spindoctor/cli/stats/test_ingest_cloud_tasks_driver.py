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
from spindoctor.results_index import IMAGES, INGEST_RUNS, open_index

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


LEAKING_PASSWORD = 'se@cr:etlongsecretpassword'
"""A password whose tail a URL parser quotes back as the port it could not read."""

LEAKING_INDEX_URL = f'postgresql+psycopg://user:{LEAKING_PASSWORD}@dbhost/spindoctor'
"""An index URL whose refusal is where that tail would otherwise appear."""


def test_a_worker_that_cannot_open_the_index_names_no_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task result travels further than a log line, so a leak in one travels too.

    What a worker returns is written verbatim into its event log, and an
    operator collects those logs, concatenates them and hands the file to the
    program that completes the ingest. A refusal that masks the URL and then
    quotes the parser's own complaint about it puts a run of the password in
    that file.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db=LEAKING_INDEX_URL)
    )
    assert 'etlongsecretpassword' not in result['status_exception']


def test_a_worker_that_cannot_open_the_index_still_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And keeps the diagnosis, which is the whole of what the result is for."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    _retry, result = sd_stats_ingest_cloud_tasks.process_task(
        'ingest-1-000000', {}, worker_data(results_db=LEAKING_INDEX_URL)
    )
    assert result['status_error'] == 'index_unopenable'


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


def fanned_out_with_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a tree holding one document and one file that is not one, and fan it out.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
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


def dividing_a_root_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Divide up a root the walk cannot list.

    Parameters:
        tmp_path: Directory the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(
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


def test_dividing_a_root_that_is_not_there_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule as a pass that reads the documents: a listing failed."""
    status, _written = dividing_a_root_that_is_not_there(tmp_path, monkeypatch)
    assert status == 1


def test_dividing_a_root_that_is_not_there_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status alone is what the catch-all produces for anything at all.

    What tells this refusal from a failure nobody enumerated is the message
    naming the thing that went wrong, so the message is what is asserted.
    """
    _status, written = dividing_a_root_that_is_not_there(tmp_path, monkeypatch)
    assert any('Roots that could not be listed' in line for line in written)


def refusing_both_cloud_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SystemExit:
    """Parse a command line asking for both cloud modes at once.

    Parameters:
        tmp_path: Directory the two named files would live under.
        monkeypatch: Fixture the argument vector is replaced through.

    Returns:
        The exit the parser raised.
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
    return caught.value


def test_the_two_cloud_modes_cannot_be_asked_for_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dividing the work up and adding it together are different runs.

    Asking for both would write a tasks file and then complete the run it had
    just created, before a single worker had read anything.
    """
    assert refusing_both_cloud_modes(tmp_path, monkeypatch).code == 2


def test_the_two_cloud_modes_are_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A status of 2 is what argparse exits with for any command line at all.

    A renamed option, a missing value, a typo: all of them are status 2, so what
    says this refusal is the one meant is the message naming the two options
    that cannot be asked for together.
    """
    refusing_both_cloud_modes(tmp_path, monkeypatch)
    assert '--complete-cloud-tasks-file' in capsys.readouterr().err


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


def test_the_completion_summary_says_why_a_file_was_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker has no run log, so its reasons reach this summary or nowhere.

    A results tree holds many ``*_metadata.json`` files that were never
    navigation documents; several hundred of those are ordinary, and several
    hundred navigation results that would not parse are not. The tally with one
    example file per reason is what tells the two apart, and a divided ingest
    must not be the configuration that loses it.
    """
    url = fanned_out_with_a_refusal(tmp_path, monkeypatch)
    results = [process(task['data'], url)[1] for task in tasks_of(tmp_path / 'tasks.json')]
    write_event_log(tmp_path / 'events.log', results)
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
    examples = [line for line in written if 'for example' in line]
    assert any('edges_metadata.json' in line for line in examples)


def test_a_result_written_under_another_root_is_named_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run number is only unique inside the index that minted it.

    A task file that outlived its index names a run of whatever was built next,
    and its shares add up to that run's listing while having written their rows
    somewhere else entirely. Saying so is what tells an operator which of the
    two they are looking at.
    """
    url = fanned_out(tmp_path, monkeypatch, count=3)
    write_event_log(
        tmp_path / 'events.log',
        [
            {
                'status': 'ok',
                'run_id': 1,
                'root_url': str(tmp_path / 'elsewhere'),
                'files_ingested': 3,
                'files_skipped': 0,
                'files_failed': 0,
            }
        ],
    )
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
    assert any('reporting rows under a different root' in line for line in written)


def completing_a_mistyped_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Divide up a root that is not there, then complete it from an empty log.

    Parameters:
        tmp_path: Directory the index, the tasks file and the log live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status of the completion, and one entry per line it wrote to
        the main log.
    """
    mistyped = tmp_path / 'nav-offset-reuslts'
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            str(mistyped),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    write_event_log(tmp_path / 'events.log', [])
    return run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            str(mistyped),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_completing_a_root_whose_listing_was_never_recorded_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sequence an operator reaches it by: a mistyped root, then a completion.

    The fan-out refuses the root and records nothing about it, so the completion
    has nothing to measure its tasks against. Read as zero files, the mistyped
    root completes as a fully ingested empty tree and every consumer then reports
    the images under the real one as never navigated.
    """
    status, _written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert status == 1


def test_completing_a_root_whose_listing_was_never_recorded_says_which_refusal_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status of 1 is what the catch-all produces for anything whatever.

    This refusal is one the pass enumerates, and what tells it apart is the
    message naming the root and the correction to make -- divide it up again,
    rather than re-run the outstanding tasks, which is what a shortfall needs.
    """
    _status, written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert any('never recorded what its listing found' in line for line in written)


def test_completing_a_root_whose_listing_was_never_recorded_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And it must not reach the catch-all, whose traceback replaces the message."""
    _status, written = completing_a_mistyped_root(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_a_root_whose_listing_was_never_recorded_keeps_its_unfinished_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what a consumer reads, and the reason the status is 1."""
    url = index_url(tmp_path / 'index.sqlite3')
    completing_a_mistyped_root(tmp_path, monkeypatch)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            finished = list(connection.execute(sqlalchemy.select(INGEST_RUNS.c.finished_utc)))
    finally:
        engine.dispose()
    assert [row.finished_utc for row in finished] == [None]


def missing_event_log_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a fanned-out root against an event log that is not there.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    url = fanned_out(tmp_path, monkeypatch)
    return run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'nowhere.log'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_an_event_log_that_is_not_there_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped path is an ordinary operator error, and is charged to the file.

    The message is what says which failure this is: a status of 1 alone is the
    same status the catch-all produces for a failure nobody enumerated.
    """
    _status, written = missing_event_log_run(tmp_path, monkeypatch)
    assert any('Cannot read the task event log' in line for line in written)


def test_an_event_log_that_is_not_there_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass charges every failure it expects to one file or one root.

    A path that names no file is one it can charge, so it must not reach the
    catch-all, whose message says the run could not complete and whose traceback
    is what an operator gets instead of a correction to make.
    """
    _status, written = missing_event_log_run(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_an_event_log_that_is_not_there_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the status says the run did not complete, as it does for any refusal."""
    status, _written = missing_event_log_run(tmp_path, monkeypatch)
    assert status == 1


def binary_event_log_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a fanned-out root against a file that is not text.

    Parameters:
        tmp_path: Directory the tree, the index and the tasks file live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    url = fanned_out(tmp_path, monkeypatch)
    (tmp_path / 'events.log.gz').write_bytes(b'\x1f\x8b\x08\x00\xff\xfe\x00\x00')
    return run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log.gz'),
        ],
        monkeypatch,
        tmp_path,
    )


def test_an_event_log_that_is_not_text_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path naming a compressed log or a database is the same operator error.

    It is a path that names the wrong thing, which the pass charges to the file,
    and the decoding failure it raises is a ValueError rather than an OSError --
    so a guard written for the missing-file case alone lets this one past.
    """
    _status, written = binary_event_log_run(tmp_path, monkeypatch)
    assert any('Cannot read the task event log' in line for line in written)


def test_an_event_log_that_is_not_text_is_not_an_unhandled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch-all's message and traceback are what it must not produce."""
    _status, written = binary_event_log_run(tmp_path, monkeypatch)
    assert not any('Ingest could not complete' in line for line in written)


def test_an_event_log_that_is_not_text_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the run is not completed, since nothing was read to complete it with."""
    status, _written = binary_event_log_run(tmp_path, monkeypatch)
    assert status == 1


def test_forcing_a_completion_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is read here, so --force could only ever be ignored.

    An operator who typed it meant the documents to be read again, which is a
    property of the fan-out that cut the shares and is decided one step earlier.
    """
    url = fanned_out(tmp_path, monkeypatch)
    write_event_log(tmp_path / 'events.log', [])
    status, _written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
            '--force',
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_forcing_a_completion_says_what_to_do_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal nobody can act on is only half a refusal."""
    url = fanned_out(tmp_path, monkeypatch)
    write_event_log(tmp_path / 'events.log', [])
    _status, written = run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            (tmp_path / 'results').as_posix(),
            '--complete-cloud-tasks-file',
            str(tmp_path / 'events.log'),
            '--force',
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('--output-cloud-tasks-file with --force' in line for line in written)


def completing_a_root_nobody_divided_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int | None, list[str]]:
    """Complete a root that no fan-out ever covered.

    Parameters:
        tmp_path: Directory the tree, the index and the log live under.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = fanned_out(tmp_path, monkeypatch)
    write_event_log(tmp_path / 'events.log', [])
    return run_driver(
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


def test_completing_a_root_nobody_divided_up_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no run to stamp, and saying so is the whole of the diagnosis."""
    status, _written = completing_a_root_nobody_divided_up(tmp_path, monkeypatch)
    assert status == 1


def test_completing_a_root_nobody_divided_up_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming the root and the step to run over it is that whole diagnosis.

    The status is the same one every other refusal exits with, so the message is
    the only thing that distinguishes this from a failure nobody enumerated.
    """
    _status, written = completing_a_root_nobody_divided_up(tmp_path, monkeypatch)
    assert any('No unfinished ingest run to complete' in line for line in written)


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
