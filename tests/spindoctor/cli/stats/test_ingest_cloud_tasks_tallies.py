"""Tests for which task results a completion reads as one share's tally.

A completion is handed whatever the event log held: values workers returned,
lines of another run's log, a file an operator edited, a file that was truncated
or concatenated with somebody else's.  Only some of those are a share's tally,
and telling them apart is load-bearing rather than defensive.  A value read as a
tally counts toward its run's account, an account that reaches the listing stamps
the run, and a stamped run is what tells every consumer that absence of a row
under that root means the image was never navigated.  Adding up the tallies that
do pass is in ``test_ingest_cloud_tasks_completion``.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest

from spindoctor.results_index import normalize_root_url

from .conftest import build_tree, complete, fan_out, index_url, reported

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


# ---------------------------------------------------------------------------
# The counts a tally carries
# ---------------------------------------------------------------------------


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
