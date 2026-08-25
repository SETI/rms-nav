"""Tests for which task results a completion reads as one share's tally.

A completion is handed whatever the event log held: values workers returned,
lines of another run's log, a file an operator edited, a file that was truncated
or concatenated with somebody else's.  Only some of those are a share's tally,
and telling them apart is load-bearing rather than defensive.  A value read as a
tally counts toward its run's account, an account that reaches the listing stamps
the run, and a stamped run is what tells every consumer that absence of a row
under that root means the image was never navigated.  So each guard here is
tested by what breaking it costs: a run stamped on documents nothing read, a run
number that is not a run number, a count the row cannot hold, or a completion
that ends in an exception nobody enumerated.  Adding up the tallies that do pass
is in ``test_ingest_cloud_tasks_completion``.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest
from tests.spindoctor.cli.results_index.conftest import (
    build_tree,
    complete,
    fan_out,
    reported,
    run_rows,
)
from tests.spindoctor.conftest import (
    index_url,
)

from spindoctor.cli.results_index.tasks import _LARGEST_RUN_ROW_COUNT, _share_tally
from spindoctor.results_index import normalize_root_url

# ---------------------------------------------------------------------------
# The shape of a tally
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The root a tally names
# ---------------------------------------------------------------------------


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


def test_a_root_no_storage_layer_can_render_costs_only_its_own_result(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One spelling nothing can make absolute must not end the whole completion.

    A completion reads a log that has been concatenated from several machines,
    and a line naming a root the storage layer refuses -- a bare ``//``, which is
    a UNC path with no share in it -- would otherwise take every other root down
    with it, in an exception nobody enumerated and with a traceback in place of
    a message.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    unrenderable = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': '//',
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    outcome = complete(url, [root], [unrenderable], logger=quiet_logger)
    assert outcome.results_unreadable == 1


def test_a_share_naming_its_root_another_way_is_still_its_share(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A spelling is not a root here either, and the tally is normalized too.

    Two spellings of one root are one root everywhere else in the pass -- at the
    fan-out, in the roots the completion is given, in the rows a worker writes --
    and a tally naming this run's root with a trailing separator is this run's
    share.  Compared as it was spelled it is charged to another root instead, and
    the run whose documents were read stays unreadable forever.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    spelled = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': f'{root.as_posix()}/',
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [spelled], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is not None


# ---------------------------------------------------------------------------
# The run a tally names
# ---------------------------------------------------------------------------


def test_a_flag_where_the_run_number_belongs_does_not_stamp_run_one(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """``True`` is the same dictionary key as ``1``, and run 1 is the first run.

    A share is credited to a run by the pair it names, and a flag hashes and
    compares equal to the whole number one -- so a line carrying ``true`` where
    a run number belongs is credited to the first run of the index, accounts for
    its listing exactly, and stamps it.  Every consumer then reads absence of a
    row under that root as "this image was never navigated", on the strength of
    a value that identifies nothing.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    flagged = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': True,
            'root_url': normalize_root_url(root),
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [flagged], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_fractional_run_number_does_not_stamp_the_run_it_rounds_to(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And ``1.0`` is that same key, by the same equality.

    A run identifier is a whole number the index minted, and a JSON document
    that went through anything treating every number as a float is exactly where
    ``1.0`` comes from.  It is not the identifier the index issued, and reading
    it as one stamps a run on a line whose provenance nothing here can vouch for.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    fractional = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1.0,
            'root_url': normalize_root_url(root),
            'files_ingested': 2,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [fractional], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


# ---------------------------------------------------------------------------
# The counts a tally carries
# ---------------------------------------------------------------------------


def test_a_fractional_count_does_not_stamp_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A number of files is a whole number, and ``2.0`` is not a report of two.

    It compares equal to the two files the listing found, so read as a tally it
    accounts for the root exactly and stamps the run -- on a line no worker of
    this run wrote, for documents nothing read.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    fractional = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': 2.0,
            'files_skipped': 0,
            'files_failed': 0,
        },
    )
    complete(url, [root], [fractional], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_flags_where_the_counts_belong_do_not_stamp_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Two flags are two files to any arithmetic that adds them up.

    ``true`` is one and ``false`` is zero, so a pair of them where an ingested
    and a skipped count belong accounts for a two-file listing exactly.  A run
    stamped on that has had none of its documents read.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    flagged = reported(
        'ingest-1-000000',
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(root),
            'files_ingested': True,
            'files_skipped': True,
            'files_failed': 0,
        },
    )
    complete(url, [root], [flagged], logger=quiet_logger)
    assert run_rows(url)[0].finished_utc is None


def test_a_count_that_is_not_a_number_is_not_a_share_tally(tmp_path: Path) -> None:
    """NaN defeats every comparison it is put through, and is then written down.

    It is neither short of a listing nor past one, so the arithmetic that
    decides whether a run is stamped says nothing about it, and what the
    shortfall records on the run row is the value itself: the column holds no
    number at all afterwards, and an operator reading that row to see how far a
    partial pass got is told nothing was written under the root.  Asked of the
    reader directly, because the account it would join is bounded as well and a
    tally is refused here before either arithmetic sees it.
    """
    tally = _share_tally(
        {
            'status': 'ok',
            'run_id': 1,
            'root_url': normalize_root_url(tmp_path / 'results'),
            'files_ingested': float('nan'),
            'files_skipped': 0,
            'files_failed': 0,
        }
    )
    assert tally is None


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


def test_two_counts_the_run_row_cannot_hold_between_them_are_not_an_account(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The bound is on the running total, because the total is what is written.

    Each of these two lines carries a count the guard accepts on its own; what
    the run row is written from is their sum, which is twice what the column
    holds.  Bounding each count and not the total leaves two lines of a
    concatenated log to overflow the write between them, and the whole
    completion ends in the database driver's own error rather than the second
    line costing itself.  Refused, it is counted like every other result nobody
    can read, and the run is left short.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    url = index_url(tmp_path / 'index.sqlite3')
    fan_out(url, [root], logger=quiet_logger)
    huge = [
        reported(
            f'ingest-1-00000{index}',
            {
                'status': 'ok',
                'run_id': 1,
                'root_url': normalize_root_url(root),
                'files_ingested': _LARGEST_RUN_ROW_COUNT,
                'files_skipped': 0,
                'files_failed': 0,
            },
        )
        for index in range(2)
    ]
    outcome = complete(url, [root], huge, logger=quiet_logger)
    assert outcome.results_unreadable == 1
