"""Tests for adding the shares of a divided ingest back up.

A root must stay unreadable until every share is accounted for, because a task
that never reported leaves documents unread and absence of their rows is what
every consumer reads as "this image was never navigated".  What licenses the
stamp is arithmetic: the listing found so many files, and this run's own shares
must say what became of exactly that many.  Dividing the root up and ingesting
one share are in ``test_ingest_cloud_tasks``.
"""

import json
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.cli.stats.ingest import TaskResult, task_results_from_event_log
from spindoctor.results_index import (
    INGEST_RUNS,
    SCHEMA_VERSION,
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
    ingest_tree,
    metadata_document,
    reported,
    run_rows,
    run_shares,
    write_metadata,
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


def unlistable_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the volume named ``VOL2`` refuse to be listed.

    A directory the walk cannot enumerate is the ordinary case on a shared tree
    -- a permission the run does not hold, a mount that stopped answering -- and
    it is the case where absence of a row means nothing at all.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
    """
    real_iterdir = FCPath.iterdir_metadata

    def refusing_vol2(self: FCPath) -> Any:
        if self.name == 'VOL2':
            raise PermissionError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing_vol2)


def test_the_run_keeps_the_directories_the_fan_out_could_not_list(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the fan-out ever saw them, and the completion rewrites the row.

    Under a directory nobody enumerated, absence of a row is not evidence that
    an image was never navigated, and the count on the run row is the only place
    a consumer can read that.  A completion that did not carry the fan-out's
    count across would replace it with a zero, and the row would then say the
    whole root was listed.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    unlistable_volume(monkeypatch)
    tasks = fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], run_shares(url, tasks, logger=quiet_logger), logger=quiet_logger)
    assert run_rows(url)[-1].directories_missed == 1


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
    failure = reported('ingest-1-000009', {'status': 'error', 'status_error': 'index_unopenable'})
    broken = [*results[:-1], failure]
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
        [*results, reported('ingest-1-000009', {'status': 'error', 'status_error': 'no_db'})],
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
    outcome = complete(url, [root], [reported('ingest-1-000000', result)], logger=quiet_logger)
    assert outcome.results_unreadable == 1


def test_a_share_counted_twice_does_not_short_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A retried task reports its share a second time, as skipped.

    Its later report stands in for the earlier one, so the account is still one
    report per task and still covers every file the walk saw.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    retried = run_shares(url, tasks[:1], logger=quiet_logger)
    complete(url, [root], [*results, *retried], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_share_reported_twice_does_not_cover_for_one_that_never_ran(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The arithmetic must not let over- and under-accounting cancel.

    A queue redelivers a task whenever it could not see the delivery
    acknowledged, so one share reported twice while another never ran is an
    ordinary sequence.  Summed by files alone it reaches the number the walk
    found, and the run is stamped with two documents nobody read -- which every
    consumer would then read as two images that were never navigated.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    first = run_shares(url, tasks[:1], logger=quiet_logger)
    again = run_shares(url, tasks[:1], logger=quiet_logger)
    complete(url, [root], [*first, *again], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_share_reported_twice_leaves_its_roots_shortfall_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the shortfall is the one the tasks that never ran left behind."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    first = run_shares(url, tasks[:1], logger=quiet_logger)
    again = run_shares(url, tasks[:1], logger=quiet_logger)
    outcome = complete(url, [root], [*first, *again], logger=quiet_logger)
    assert any('2 of 4 file(s)' in named for named in outcome.roots_unaccounted)


def test_a_repeat_of_one_task_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Counted once, and said out loud, so the log records that it happened."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root], [*results, *again], logger=quiet_logger)
    assert outcome.results_superseded == 1


def test_the_later_report_of_a_task_is_the_one_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A task that failed and was re-run reports its failure first.

    Reading the earlier report would leave a run unfinished though its documents
    are in the index, so the last thing a task said is what counts.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    failed = reported(str(tasks[0]['task_id']), {'status': 'error', 'status_error': 'no_index'})
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], [failed, *results], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_result_naming_no_task_is_not_counted_toward_a_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Nothing tells such a result from a repeat of another, so it counts nowhere.

    Counting it could only ever inflate an account, and an inflated account is
    what stamps a run whose documents were never read.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    anonymous = [TaskResult(task_id=None, result=found.result) for found in results]
    complete(url, [root], anonymous, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_result_naming_no_task_is_counted_as_one(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And is reported, since a log full of them means the run cannot complete."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    anonymous = [TaskResult(task_id=None, result=found.result) for found in results]
    outcome = complete(url, [root], anonymous, logger=quiet_logger)
    assert outcome.results_unidentified == 1


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


def test_a_run_abandoned_under_a_newer_finished_one_is_not_completed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only the newest run of a root is a candidate for a completion.

    A fan-out given up on and then made good by an ordinary pass leaves an older
    unfinished run under a newer finished one.  Stamping the older one would put
    a finish time on a walk nothing came back from, and would date the root's
    ingest to a pass that was abandoned.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    ingest_tree(url, [root], logger=quiet_logger)
    outcome = complete(url, [root], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == [normalize_root_url(root)]


def test_a_run_abandoned_under_a_newer_finished_one_keeps_its_null_finish(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the abandoned run keeps saying so, which is what a consumer reads.

    A consumer takes the newest run of a root, so the stamp on an older one
    would not mislead it; what it would do is tell an operator reading the table
    that a pass nothing came back from finished.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    ingest_tree(url, [root], logger=quiet_logger)
    complete(url, [root], results, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_two_spellings_of_one_root_are_completed_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing slash is not another root, here or at the fan-out.

    Completed twice, the second pass finds the run it has just stamped and
    reports the root as one nobody divided up.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    outcome = complete(url, [root, f'{root.as_posix()}/'], results, logger=quiet_logger)
    assert outcome.roots_without_a_run == []


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
    and is empty as one nobody has ingested.  This root was listed and found to
    hold nothing, which is what no shares can account for; a root whose listing
    was never recorded is the case below, and must not complete.
    """
    root = tmp_path / 'results'
    root.mkdir()
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    complete(url, [root], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


def test_a_shortfall_records_what_the_shares_did(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The run stays unfinished, but the work the shares did is not lost.

    Their documents are in the index, and an operator reading the run row after
    a partial pass needs to see how far it got rather than the zeros the fan-out
    left there.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[:-1], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 2


def test_a_shortfall_records_what_its_shares_skipped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A pass whose shares mostly skipped got just as far as one that ingested.

    Recording only the documents read would write a row of zeros for a re-run of
    an unchanged tree, which is the case the incremental skip makes ordinary.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    run_shares(url, tasks, logger=quiet_logger)
    again = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], again[:-1], logger=quiet_logger)
    assert run_rows(url)[0].files_skipped == 2


def test_a_shortfall_records_what_its_shares_could_not_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And the refusals, which are the half an operator is most likely to chase."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger)
    results = run_shares(url, tasks, logger=quiet_logger)
    complete(url, [root], results[1:], logger=quiet_logger)
    assert run_rows(url)[0].files_failed == 2


# ---------------------------------------------------------------------------
# An account that runs past the listing
# ---------------------------------------------------------------------------


def test_an_account_past_the_listing_leaves_the_run_unfinished(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Each task counts once, so the sum can only exceed the listing wrongly.

    Whatever produced it -- a hand-edited log, a result from somewhere else --
    is not an account of this run, and stamping on it would license the one
    claim the run bookkeeping exists to license from evidence that cannot be
    right.
    """
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    inflated = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 1000000,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [inflated], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_an_account_past_the_listing_names_its_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """With both numbers, since the excess is the whole of what is wrong."""
    root = tmp_path / 'results'
    build_tree(root, 4)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    inflated = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 1000000,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    outcome = complete(url, [root], [inflated], logger=quiet_logger)
    assert any('1000000 of 4 file(s)' in named for named in outcome.roots_unaccounted)


def test_a_count_too_large_for_the_run_row_is_not_a_share_tally(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A number no share could report is refused before it reaches the row.

    What the shares reported is written to the run row on a shortfall, and a
    count larger than that column holds fails the write -- ending the whole
    completion in the driver's own error, for one corrupt or foreign line of a
    concatenated event log.  Refused here it costs its own result, like every
    other value of a shape a worker does not return.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    enormous = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 10**30,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    outcome = complete(url, [root], [enormous], logger=quiet_logger)
    assert outcome.results_unreadable == 1


def test_a_count_below_zero_is_not_a_share_tally(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A negative count is not a number of files, and it cancels a real one.

    Read as an account it subtracts from the shares that did report, so a run
    left short by one task is stamped by another claiming minus its files.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    negative = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': -100,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    outcome = complete(url, [root], [negative], logger=quiet_logger)
    assert outcome.results_unreadable == 1


# ---------------------------------------------------------------------------
# Shares written under another root
# ---------------------------------------------------------------------------


def discard_index(path: Path) -> None:
    """Delete a SQLite index and the side files it writes beside itself.

    This is the documented remedy for an index stamped with another schema
    version, and it is the step that makes a surrogate run identifier start
    again at one.

    Parameters:
        path: The database file.
    """
    for name in (path.name, f'{path.name}-wal', f'{path.name}-shm'):
        (path.parent / name).unlink(missing_ok=True)


def stale_shares_of_a_rebuilt_index(
    tmp_path: Path, logger: pdslogger.PdsLogger
) -> tuple[str, Path, list[TaskResult]]:
    """Run one root's tasks against an index rebuilt around another root.

    The sequence takes no hand-editing of anything.  A root is divided up, the
    index is deleted and rebuilt -- which is what a schema version mismatch is
    remedied by -- another root is divided into the fresh index and takes the
    run identifier the first one had, and the queue still holds the first root's
    tasks.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        logger: Logger every stage reports through.

    Returns:
        The index URL, the root that was divided up second, and what the first
        root's tasks reported.
    """
    first = tmp_path / 'root-a'
    second = tmp_path / 'root-b'
    build_tree(first, 4)
    build_tree(second, 4)
    database = tmp_path / 'index.sqlite3'
    url = index_url(database)
    stale = fan_out(url, [first], logger=logger)
    discard_index(database)
    fan_out(url, [second], logger=logger)
    return url, second, run_shares(url, stale, logger=logger)


def test_a_run_is_not_stamped_by_shares_of_another_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run number is unique only inside the index that minted it.

    The shares wrote real rows and reported real counts, and they add up to
    exactly what this run's listing found -- under a different root. Credited by
    run number alone they stamp a root with nothing under it, and every consumer
    then reads absence of a row there as "this image was never navigated".
    """
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    complete(url, [second], results, logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_left_to_the_shares_of_another_is_named_short(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Named with nothing accounted for, which is what its own tasks reported."""
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    outcome = complete(url, [second], results, logger=quiet_logger)
    assert any('0 of 4 file(s)' in named for named in outcome.roots_unaccounted)


def test_shares_written_under_another_root_are_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Told apart from a result belonging to a fan-out nobody is completing here.

    That one is somebody else's to complete; this one names the very run being
    completed and is still not its share, which is a different thing to say and
    a different thing to do about it.
    """
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    outcome = complete(url, [second], results, logger=quiet_logger)
    assert outcome.results_of_another_root == 2


def test_a_root_left_to_the_shares_of_another_stays_unreadable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what a consumer asks, and the whole reason for not stamping it."""
    url, second, results = stale_shares_of_a_rebuilt_index(tmp_path, quiet_logger)
    complete(url, [second], results, logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(second)], url=url)
    finally:
        engine.dispose()


def test_a_share_naming_no_root_is_not_counted_as_one(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A tally that names no root says nothing about which root was written.

    The root is half the key of every row a share writes, so a value carrying
    counts and a run number alone cannot be attributed to anything.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    rootless = reported(
        'ingest-1-000000',
        {'status': 'ok', 'run_id': 1, 'files_ingested': 2, 'files_skipped': 0, 'files_failed': 0},
    )
    outcome = complete(url, [root], [rootless], logger=quiet_logger)
    assert outcome.results_unreadable == 1


# ---------------------------------------------------------------------------
# A run that never recorded what its root holds
# ---------------------------------------------------------------------------


def begin_a_run(url: str, root: Path) -> None:
    """Record that an ingest of a root began, and nothing else about it.

    This is the row a pass leaves behind when it dies between starting and
    listing, and the row a fan-out leaves for a root it could not list at all.

    Parameters:
        url: The index URL to create or add to.
        root: The results root the run covers.
    """
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.insert().values(
                    root_url=normalize_root_url(root),
                    started_utc='2026-01-01T00:00:00+00:00',
                    schema_version=SCHEMA_VERSION,
                )
            )
    finally:
        engine.dispose()


def test_a_root_whose_listing_was_never_recorded_is_not_stamped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run that never established what its root holds cannot be accounted for.

    No files seen is not zero files seen.  Read as zero, a mistyped root
    completes with nothing under it, and every consumer then reads absence of a
    row as "this image was never navigated" for a whole tree nobody listed.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    complete(url, [root], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_whose_listing_was_never_recorded_is_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Told apart from a shortfall, because what to do about it is different.

    A shortfall is re-run the outstanding tasks; this is divide the root up
    again, because no task was ever cut from it.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    outcome = complete(url, [root], [], logger=quiet_logger)
    assert outcome.roots_unlisted == [normalize_root_url(root)]


def test_a_root_whose_listing_was_never_recorded_records_what_its_shares_did(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Such a run can still hold real work, and the row is where it shows.

    The run is not stamped and the root stays unreadable, but a share may have
    written rows under it all the same -- a task file that outlived the listing
    it was cut from is exactly that -- and an operator reading a row of zeros
    would conclude nothing was written.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    begin_a_run(url, root)
    share = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [share], logger=quiet_logger)
    assert run_rows(url)[0].files_ingested == 2


def test_a_root_the_fan_out_could_not_list_is_not_stamped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same rule, reached the way an operator reaches it: a mistyped root.

    The fan-out refuses it and records nothing on the run, so the completion has
    nothing to measure against and must not stamp it -- otherwise the mistyped
    root reads as a fully ingested empty tree.
    """
    absent = tmp_path / 'nav-offset-reuslts'
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [absent], logger=quiet_logger)
    complete(url, [absent], [], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_root_the_fan_out_could_not_list_stays_unreadable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what a consumer asks about, and the whole point of not stamping."""
    absent = tmp_path / 'nav-offset-reuslts'
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [absent], logger=quiet_logger)
    complete(url, [absent], [], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection, pytest.raises(ValueError, match='no completed ingest'):
            require_ingested_roots(connection, [normalize_root_url(absent)], url=url)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Why the files a share could not read were refused
# ---------------------------------------------------------------------------


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
