"""Tests for what a completion reads out of the workers' own results.

Two things travel back from a worker and nowhere else.  Why the files it could
not read were refused: a worker has no run log, so the per-reason tally reaches
the closing summary in the task result or not at all, and a count of unreadable
files with nothing to say about them reads the same whether a tree holds many
documents that were never navigation results or the ingest went wrong.  And the
results themselves, which are read back out of the JSON-Lines event log the
queue's workers wrote.

Both are read defensively.  A log is appended to while it is being read, is
concatenated from several machines by hand, and is a plain text file an operator
can edit: a line of it that is not an event, an event that is not a result, and
a per-reason map of another shape each cost their own line and nothing else,
because what licenses a run's stamp is the three counts and they are complete
without any of this.

What the counts license is in ``test_ingest_cloud_tasks_completion``.
"""

import json
from pathlib import Path
from typing import Any

import pdslogger
from filecache import FCPath

from spindoctor.cli.stats.ingest import task_results_from_event_log

from .conftest import (
    build_tree,
    complete,
    fan_out,
    index_url,
    reported,
    run_rows,
    run_shares,
)


def test_the_completion_tallies_the_reasons_the_shares_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A worker has no run log, so its reasons reach the summary or nowhere.

    A tree holds many documents that were never navigation results, and a count
    of unreadable files with nothing to say about them reads the same whether
    that is what happened or the ingest went wrong.
    """
    root = tmp_path / 'results'
    build_tree(root, 1)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert sum(outcome.counts.failures_by_reason.values()) == 1


def test_the_completion_keeps_one_example_of_each_reason(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A reason is a field-level diagnosis; one real file is what explains it."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    examples = [Path(name).name for name in outcome.counts.example_by_reason.values()]
    assert examples == ['edges_metadata.json']


def test_the_completion_tallies_the_reasons_of_every_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Two shares refusing for one reason are one reason with two files."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger, share_size=1)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert sum(outcome.counts.failures_by_reason.values()) == 2


def test_a_reason_tally_of_another_shape_costs_only_the_reasons(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The reasons are the diagnosis; the counts beside them are the account.

    A share whose reasons cannot be read has still said what became of its
    files, so its run completes and only the diagnosis is lost.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    mangled = [
        reported(str(found.task_id), {**found.result, 'failures_by_reason': ['not a map']})
        for found in results
    ]
    complete(url, [root], mangled, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def completed_with_reasons(
    tmp_path: Path, reasons: dict[Any, Any], key: str, *, logger: pdslogger.PdsLogger
) -> Any:
    """Complete a two-document root whose one share reports the given reason map.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        reasons: The map to put in the share's result, whatever shape it is.
        key: Which of the share's two per-reason maps to replace.
        logger: Logger every stage reports through.

    Returns:
        The completion outcome.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=logger)
    results = run_shares(url, tasks, logger=logger)
    mangled = [reported(str(found.task_id), {**found.result, key: reasons}) for found in results]
    return complete(url, [root], mangled, logger=logger)


def test_a_reason_whose_count_is_not_a_number_is_dropped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A count is added to the tally, so anything else has to be refused there.

    Carried through, it reaches the addition that folds one share's reasons into
    the pass's and raises out of the completion -- a diagnostic map taking down
    the accounting it is written beside.
    """
    outcome = completed_with_reasons(
        tmp_path, {'not a navigation result': 'lots'}, 'failures_by_reason', logger=quiet_logger
    )
    assert outcome.counts.failures_by_reason == {}


def test_a_reason_whose_count_is_not_a_number_leaves_the_run_completed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is the point: the account is complete without the diagnosis."""
    completed_with_reasons(
        tmp_path, {'not a navigation result': 'lots'}, 'failures_by_reason', logger=quiet_logger
    )
    assert run_rows(index_url(tmp_path / 'index.sqlite3'))[0].finished_utc is not None


def test_a_reason_whose_count_is_a_flag_is_dropped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A flag is a number to Python, and would be tallied as one file."""
    outcome = completed_with_reasons(
        tmp_path, {'not a navigation result': True}, 'failures_by_reason', logger=quiet_logger
    )
    assert outcome.counts.failures_by_reason == {}


def test_a_reason_that_is_not_text_is_dropped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The reasons are sorted to be reported, and mixed types do not sort.

    A map arriving from anywhere but a worker of this version is free to be keyed
    by anything, and one number among the reasons is enough to raise out of the
    closing summary after every run has already been stamped.
    """
    outcome = completed_with_reasons(tmp_path, {7: 2}, 'failures_by_reason', logger=quiet_logger)
    assert outcome.counts.failures_by_reason == {}


def test_an_example_that_is_not_a_file_name_is_dropped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An example is printed as the one real file a reason means, so it is text."""
    outcome = completed_with_reasons(
        tmp_path, {'not a navigation result': 7}, 'example_by_reason', logger=quiet_logger
    )
    assert outcome.counts.example_by_reason == {}


# ---------------------------------------------------------------------------
# Reading the workers' results back out of an event log
# ---------------------------------------------------------------------------


def write_event_log(path: Path, events: list[Any]) -> Path:
    """Write a cloud-tasks event log, one JSON object per line.

    Parameters:
        path: Where to write it.
        events: The events, each serialized on its own line.

    Returns:
        The path written.
    """
    path.write_text(''.join(f'{json.dumps(event)}\n' for event in events), encoding='utf-8')
    return path


def completed_event(result: Any, *, task_id: str | None = 'ingest-1-000000') -> dict[str, Any]:
    """Return the event a worker's return value is written under.

    Parameters:
        result: What the worker returned.
        task_id: The task it ran under; None writes an event carrying none.

    Returns:
        The event.
    """
    event: dict[str, Any] = {'event_type': 'task_completed', 'result': result}
    if task_id is not None:
        event['task_id'] = task_id
    return event


def test_the_event_log_yields_what_the_workers_returned(tmp_path: Path) -> None:
    """The log is the one channel a worker's tally travels back on."""
    log = write_event_log(
        tmp_path / 'events.log',
        [completed_event({'status': 'ok', 'run_id': 1}), completed_event({'status': 'ok'})],
    )
    found = task_results_from_event_log(FCPath(log))
    reported_values = [found.result for found in found.results]
    assert reported_values == [{'status': 'ok', 'run_id': 1}, {'status': 'ok'}]


def test_an_event_about_something_other_than_a_task_is_passed_over(tmp_path: Path) -> None:
    """A worker logs its own lifecycle into the same file."""
    log = write_event_log(
        tmp_path / 'events.log',
        [{'event_type': 'spot_termination'}, completed_event({'status': 'ok'})],
    )
    found = task_results_from_event_log(FCPath(log))
    assert len(found.results) == 1


def test_a_task_that_ended_without_a_value_is_counted(tmp_path: Path) -> None:
    """Its documents were never read, so its run cannot be stamped."""
    log = write_event_log(
        tmp_path / 'events.log',
        [{'event_type': 'task_exception', 'task_id': 'ingest-1-000000'}],
    )
    found = task_results_from_event_log(FCPath(log))
    assert found.tasks_unfinished == 1


def test_a_completed_task_whose_value_is_not_an_object_is_counted_the_same_way(
    tmp_path: Path,
) -> None:
    """A worker that returned a bare string reported no share."""
    log = write_event_log(tmp_path / 'events.log', [completed_event('done')])
    found = task_results_from_event_log(FCPath(log))
    assert found.tasks_unfinished == 1


def test_a_line_that_is_not_json_is_counted(tmp_path: Path) -> None:
    """An event log is appended to while it is read, so a partial line is ordinary."""
    (tmp_path / 'events.log').write_text('{"event_type": "task_comp', encoding='utf-8')
    found = task_results_from_event_log(FCPath(tmp_path / 'events.log'))
    assert found.lines_unread == 1


def test_a_line_that_is_json_but_not_an_object_is_counted(tmp_path: Path) -> None:
    """A file of JSON arrays parses line by line and holds no event at all."""
    log = write_event_log(tmp_path / 'events.log', [[1, 2, 3]])
    found = task_results_from_event_log(FCPath(log))
    assert found.lines_unread == 1


def test_a_blank_line_is_not_an_unread_line(tmp_path: Path) -> None:
    """A log that ends in a newline would otherwise report a broken last line."""
    (tmp_path / 'events.log').write_text('\n\n', encoding='utf-8')
    found = task_results_from_event_log(FCPath(tmp_path / 'events.log'))
    assert found.lines_unread == 0


def test_a_result_keeps_the_task_that_reported_it(tmp_path: Path) -> None:
    """One report per task is what stops a repeat covering for a task that failed."""
    log = write_event_log(
        tmp_path / 'events.log', [completed_event({'status': 'ok'}, task_id='ingest-7-000003')]
    )
    found = task_results_from_event_log(FCPath(log))
    assert found.results[0].task_id == 'ingest-7-000003'


def test_a_result_whose_event_names_no_task_keeps_none(tmp_path: Path) -> None:
    """A task nothing identifies is reported as such rather than invented."""
    log = write_event_log(
        tmp_path / 'events.log', [completed_event({'status': 'ok'}, task_id=None)]
    )
    found = task_results_from_event_log(FCPath(log))
    assert found.results[0].task_id is None


def test_a_task_identity_that_is_not_text_is_kept_as_none(tmp_path: Path) -> None:
    """An identity of another shape identifies nothing, so it is not one.

    The fan-out mints a string, and every identity is compared against the
    others as one; a number kept verbatim would sort into the same table as a
    string and match nothing there. Kept as no identity at all, such a result is
    counted toward no run rather than credited to a task it cannot be told from.
    """
    log = write_event_log(
        tmp_path / 'events.log',
        [{'event_type': 'task_completed', 'task_id': 7, 'result': {'status': 'ok'}}],
    )
    found = task_results_from_event_log(FCPath(log))
    assert found.results[0].task_id is None


def test_an_event_whose_type_is_not_text_is_passed_over(tmp_path: Path) -> None:
    """Nothing about a number says which task, if any, it belongs to."""
    log = write_event_log(tmp_path / 'events.log', [{'event_type': 7}])
    found = task_results_from_event_log(FCPath(log))
    assert found.tasks_unfinished == 0


def test_the_shares_survive_the_round_trip_through_an_event_log(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole cycle, with the tallies going out to a file and coming back."""
    root = tmp_path / 'results'
    build_tree(root, 5)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    log = write_event_log(
        tmp_path / 'events.log',
        [completed_event(found.result, task_id=found.task_id) for found in results],
    )
    found = task_results_from_event_log(FCPath(log))
    complete(url, [root], found.results, logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 5
