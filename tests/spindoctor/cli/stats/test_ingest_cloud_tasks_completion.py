"""Tests for adding the shares of a divided ingest back up.

A root must stay unreadable until every share is accounted for, because a task
that never reported leaves documents unread and absence of their rows is what
every consumer reads as "this image was never navigated".  What licenses the
stamp is arithmetic: the listing found so many files, and the shares must say
what became of at least that many.  Dividing the root up and ingesting one share
are in ``test_ingest_cloud_tasks``.
"""

import json
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.cli.stats.ingest import task_results_from_event_log
from spindoctor.results_index import (
    normalize_root_url,
    open_index,
    require_ingested_roots,
)

from .conftest import (
    build_tree,
    complete,
    cycle,
    fan_out,
    index_url,
    run_rows,
    run_shares,
)

# ---------------------------------------------------------------------------
# Adding the shares up
# ---------------------------------------------------------------------------


def test_a_root_is_unreadable_between_the_fan_out_and_the_completion(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Half-written rows must not be read as an account of a root.

    This is what the deferred finish time buys: while the workers are running,
    a consumer asking about the root is told nobody has ingested it rather than
    being handed the shares that happen to have landed.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(root)], url=url)
    finally:
        engine.dispose()


def test_a_completed_root_is_readable(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """And afterwards it is, which is the other half of the same guarantee."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, [normalize_root_url(root)], url=url)
    finally:
        engine.dispose()


def test_the_shares_are_added_into_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Three tasks of two files each are six ingested documents on one run row."""
    root = tmp_path / 'results'
    build_tree(root, 6)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 6


def test_the_failures_of_every_share_are_added_into_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal is part of the account, and is what makes the arithmetic add up."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_failed == 1


def test_the_run_keeps_what_only_the_fan_out_saw(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """No share knows how many files the root holds, so the fan-out records it."""
    root = tmp_path / 'results'
    build_tree(root, 5)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    assert run_rows(url)[0].files_seen == 5


def test_the_run_keeps_the_rows_the_fan_out_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The removal happens before any worker runs, and is reported on the run row."""
    root = tmp_path / 'results'
    stubs = build_tree(root, 2)
    url = cycle(tmp_path, [root], logger=quiet_logger)
    (root / f'{stubs[0]}_metadata.json').unlink()
    tasks = fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], run_shares(url, tasks, logger=quiet_logger), logger=quiet_logger)
    assert run_rows(url)[-1].files_removed == 1


def test_a_share_that_never_reported_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A task that failed read none of its documents.

    Stamping the run anyway would tell every consumer that absence of those
    images' rows means they were never navigated, which is exactly the claim
    the run row exists to license.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[:-1], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_share_that_never_reported_names_its_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unfinished run with nothing said about it is nothing anyone can act on."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], results[:-1], logger=quiet_logger)
    assert any(normalize_root_url(root) in named for named in outcome.roots_unaccounted)


def test_a_worker_that_reported_an_error_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An error result carries no tally, so its files are unaccounted for."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    broken = [*results[:-1], {'status': 'error', 'status_error': 'index_unopenable'}]
    complete(url, [root], broken, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_worker_error_is_counted(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """And is told apart from a result of a shape nobody recognizes."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(
        url,
        [root],
        [*results, {'status': 'error', 'status_error': 'no_results_db'}],
        logger=quiet_logger,
    )
    assert outcome.results_failed == 1


@pytest.mark.parametrize(
    'result',
    [
        {'status': 'ok'},
        {
            'status': 'ok',
            'run_id': True,
            'files_ingested': 1,
            'files_skipped': 0,
            'files_failed': 0,
        },
        {
            'status': 'ok',
            'run_id': 1,
            'files_ingested': 'many',
            'files_skipped': 0,
            'files_failed': 0,
        },
        {'status': 'ok', 'run_id': 1, 'files_ingested': 1.5, 'files_skipped': 0, 'files_failed': 0},
        {
            'status': 'ok',
            'run_id': 1,
            'files_ingested': True,
            'files_skipped': 0,
            'files_failed': 0,
        },
    ],
    ids=[
        'no-run-id',
        'run-id-is-a-flag',
        'count-is-text',
        'count-is-fractional',
        'count-is-a-flag',
    ],
)
def test_a_result_of_another_shape_is_not_counted_as_a_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, result: dict[str, Any]
) -> None:
    """A value that says nothing usable must not be read as files accounted for.

    A flag is a number to Python, so ``True`` would otherwise be counted as one
    file ingested and one run identified.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    outcome = complete(url, [root], [result], logger=quiet_logger)
    assert outcome.results_unreadable == 1


def test_a_share_counted_twice_does_not_short_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A retried task reports its share a second time, as skipped.

    The account then runs past what the walk saw, which is not a shortfall: every
    file is covered, some of them twice.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    retried = run_shares(url, tasks[:1], logger=quiet_logger)
    complete(url, [root], [*results, *retried], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_root_with_no_unfinished_run_is_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Completing a root nobody fanned out is a mistake worth reporting."""
    root = tmp_path / 'results'
    build_tree(root, 1)
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    engine.dispose()
    outcome = complete(url, [root], [], logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]


def test_completing_a_root_twice_names_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The second completion has nothing outstanding to stamp."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results, logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]


def test_completion_leaves_another_fan_outs_results_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A result naming a run this completion is not waiting on is not thrown away.

    It belongs to whoever completes that root, and is counted here so that a
    completion pointed at the wrong event log says so.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [first], results, logger=quiet_logger)
    assert outcome.results_unclaimed == 1


def test_a_run_is_credited_only_with_its_own_shares(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A tally belongs to the run its task named, not to whichever run is being stamped.

    Both roots are fanned out together here and both sets of shares are handed
    to a completion of the first alone, so a run credited with everything it was
    shown would record twice the files its own listing found.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [first], results, logger=quiet_logger)
    ingested = {str(row.root_url): row.files_ingested for row in run_rows(url)}
    assert ingested[normalize_root_url(first)] == 2


def test_completion_finishes_only_the_root_it_was_given(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One root's completion says nothing about another's, however they were fanned out."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 2)
    build_tree(second, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [first, second], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [first], results, logger=quiet_logger)
    finished = {str(row.root_url): row.finished_utc for row in run_rows(url)}
    assert finished[normalize_root_url(second)] is None


def test_an_empty_root_completes(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A root that holds nothing yields no task, and must still finish its run.

    Otherwise a consumer would refuse it forever, reporting a root that exists
    and is empty as one nobody has ingested.
    """
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


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


def completed_event(result: Any) -> dict[str, Any]:
    """Return the event a worker's return value is written under.

    Parameters:
        result: What the worker returned.

    Returns:
        The event.
    """
    return {'event_type': 'task_completed', 'task_id': 'ingest-1-000000', 'result': result}


def test_the_event_log_yields_what_the_workers_returned(tmp_path: Path) -> None:
    """The log is the one channel a worker's tally travels back on."""
    log = write_event_log(
        tmp_path / 'events.log',
        [completed_event({'status': 'ok', 'run_id': 1}), completed_event({'status': 'ok'})],
    )
    found = task_results_from_event_log(FCPath(log))
    assert found.results == [{'status': 'ok', 'run_id': 1}, {'status': 'ok'}]


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
    log = write_event_log(tmp_path / 'events.log', [completed_event(r) for r in results])
    found = task_results_from_event_log(FCPath(log))
    complete(url, [root], found.results, logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 5
