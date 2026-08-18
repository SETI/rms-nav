"""Tests for the standing count of refusals a pass reports at the end of a root.

The number exists to tell an operator how much shorter an error filter answered
from the index comes than the same filter answered from the results tree, so
what it counts has to be the refusals that actually make that difference.  Two
of the things a refusal can be do not.  A file no JSON object came out of is one
the tree excludes from every error filter as well, so the two agree about it --
the parity of the two implementations over such a file is pinned in
``tests/spindoctor/dataset/test_results_filter_index.py`` -- and a refusal under
no subtree is in no selection's answer at all, because a selection enumerates
subtrees and both arms of the query are restricted to them.  Counting either
would report a gap where there is none, and an operator reading a plausible
number falls back to reading the tree, or distrusts a selection that was exact.

The report is also informational, and the pass around it is not: the count is
taken after the root's rows are written and its run is stamped, so a database
that goes away in between must cost the sentence and not the run.

Both drivers report it, and both are exercised here.  The single-process one is
also read in ``test_ingest_driver`` for what a second pass says, and in
``test_ingest_two_roots`` for the root half of the query.
"""

from pathlib import Path

import pdslogger
import pytest
import sqlalchemy

from spindoctor.cli.stats.ingest import store
from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.results_index import (
    FAILED_FILES,
    IMAGES,
    INGEST_RUNS,
    normalize_root_url,
    open_index,
)

from .conftest import (
    FIRST_STUB,
    REFUSAL_REPORT_LEAD,
    build_tree,
    complete,
    cycle,
    fan_out,
    index_url,
    ingest_tree,
    recorded_lines,
    refusal_report,
    run_shares,
    write_refusal,
)

SECOND_STUB = 'VOL/N1454725800_1_CALIB'
"""A second stub under the same subtree, for a root that holds two refusals."""

THIRD_STUB = 'VOL/N1454725801_1_CALIB'
"""A third stub under the same subtree, for the refusal that parses to a list."""

ROOTLESS_STUB = 'N1454725802_1_CALIB'
"""A stub above every subtree, which no enumeration walks or queries."""


def write_unparseable(root: Path, stub: str) -> Path:
    """Write a file under a root that no reader gets a JSON value out of.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status": "error"', encoding='utf-8')
    return path


def write_not_an_object(root: Path, stub: str) -> Path:
    """Write a file under a root that parses to a JSON value of another kind.

    Parameters:
        root: The results root to write under.
        stub: The document's results path stub under that root.

    Returns:
        The path written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[1, 2, 3]', encoding='utf-8')
    return path


def reports_in(written: list[str]) -> list[str]:
    """Return the standing-refusal lines of a recorded log.

    Parameters:
        written: Every line the log was told.

    Returns:
        The lines this report writes, in the order they were written.
    """
    return [line for line in written if line.startswith(REFUSAL_REPORT_LEAD)]


def refusals_recorded(url: str, root: Path) -> int:
    """Return how many rows ``failed_files`` holds for one root, of any kind.

    Parameters:
        url: The index URL.
        root: The results root to count under.

    Returns:
        The rows, whatever their reason and whatever subtree they are under.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            total = (
                sqlalchemy.select(sqlalchemy.func.count())
                .select_from(FAILED_FILES)
                .where(FAILED_FILES.c.root_url == normalize_root_url(root))
            )
            return int(connection.execute(total).scalar_one())
    finally:
        engine.dispose()


def roots_holding_an_image(url: str) -> list[str]:
    """Return the roots an index holds an ``images`` row under, in name order.

    Parameters:
        url: The index URL.

    Returns:
        The normalized roots.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            found = connection.execute(sqlalchemy.select(IMAGES.c.root_url).distinct())
            return sorted(str(row.root_url) for row in found)
    finally:
        engine.dispose()


def roots_whose_newest_run_finished(url: str) -> list[str]:
    """Return the roots whose newest ingest run carries a finish time.

    An ``images`` row says a document was read; it says nothing about whether
    the pass that read it ever finished, and a run left unfinished is what
    every consumer reads as "this root is not ingested".  The two facts are
    asserted apart because a failure between them leaves exactly the first.

    Parameters:
        url: The index URL.

    Returns:
        The normalized roots, in name order.
    """
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            newest = connection.execute(
                sqlalchemy.select(
                    INGEST_RUNS.c.root_url,
                    INGEST_RUNS.c.finished_utc,
                    INGEST_RUNS.c.run_id,
                ).order_by(INGEST_RUNS.c.root_url, INGEST_RUNS.c.run_id.desc())
            )
            finished: dict[str, bool] = {}
            for row in newest:
                finished.setdefault(str(row.root_url), row.finished_utc is not None)
            return sorted(root for root, done in finished.items() if done)
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# What the number counts
# ---------------------------------------------------------------------------


def _tree_holding_one_of_each_refusal(root: Path) -> None:
    """Write a root holding one refusal of every kind a pass records.

    One of the four is a divergence and three are not.  The tree reads a
    ``status`` out of any JSON object it can parse, whatever else is wrong with
    the object, so only the document refused for its schema is answered
    differently; a half-written file and a file holding a JSON list are ones the
    tree excludes from every error filter too; and a document above every subtree
    is outside the subtrees any enumeration walks or queries.

    Parameters:
        root: The results root to write under.
    """
    write_refusal(root, FIRST_STUB)
    write_unparseable(root, SECOND_STUB)
    write_not_an_object(root, THIRD_STUB)
    write_refusal(root, ROOTLESS_STUB)


def _report_over_one_of_each(
    tmp_path: Path, logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, list[str]]:
    """Ingest a root holding one refusal of every kind and return what was said.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the pass reports through.
        monkeypatch: Fixture the logger's method is replaced through.

    Returns:
        The results root, the index URL, and the standing-refusal lines.
    """
    root = tmp_path / 'results'
    _tree_holding_one_of_each_refusal(root)
    written = recorded_lines(logger, monkeypatch)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    return root, url, reports_in(written)


def test_the_count_is_the_refusals_the_tree_answers_an_error_filter_for(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One of the four refusals is a gap, and the number reported is one.

    Counting the other three would tell an operator their selection is four
    documents short of the tree's when it matches it exactly, which is the
    conclusion that sends them back to reading the tree over a root the index
    answers perfectly well.
    """
    root, _url, reports = _report_over_one_of_each(tmp_path, quiet_logger, monkeypatch)
    assert reports == [refusal_report(root, 1)]


def test_every_refusal_of_the_root_is_recorded_whatever_the_count_says(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count narrows what is reported, not what the pass writes down.

    Every refused file is still recorded, because the record is what stops the
    next pass paying to download and parse it again.  What the reported number
    leaves out is a refusal that costs a selection nothing, not a refusal.
    """
    root, url, _reports = _report_over_one_of_each(tmp_path, quiet_logger, monkeypatch)
    assert refusals_recorded(url, root) == 4


def test_a_pass_that_refuses_nothing_of_the_kind_reports_none(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root whose documents all ingest is a root with no gap to report."""
    root = tmp_path / 'results'
    build_tree(root, 2)
    written = recorded_lines(quiet_logger, monkeypatch)
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert reports_in(written) == [refusal_report(root, 0)]


# ---------------------------------------------------------------------------
# The fan-out reports it too
# ---------------------------------------------------------------------------


def test_a_completed_fan_out_reports_the_refusals_of_each_root_it_stamped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A divided ingest tells an operator the same thing a single pass does.

    It is the path where the number matters most: a worker has no run log, so
    the completion is the only place the standing count of a fanned-out root is
    ever said.  The two roots are built to disagree about it, so a report of the
    wrong root -- or of no root -- cannot pass by naming a plausible number.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 1)
    build_tree(second, 1)
    write_refusal(first, FIRST_STUB)
    write_refusal(second, FIRST_STUB)
    write_refusal(second, SECOND_STUB)
    written = recorded_lines(quiet_logger, monkeypatch)
    cycle(tmp_path, [first, second], logger=quiet_logger)
    assert reports_in(written) == [refusal_report(first, 1), refusal_report(second, 2)]


def test_a_fan_out_whose_shares_do_not_add_up_reports_no_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count belongs to a completed root, and an unaccounted one has none.

    Reported for a root the completion left unfinished, the number would read as
    an account of a root every consumer refuses.
    """
    root = tmp_path / 'results'
    build_tree(root, 2)
    write_refusal(root, 'VOL/N1454725900_1_CALIB')
    url = index_url(tmp_path / 'index.sqlite3')
    tasks = fan_out(url, [root], logger=quiet_logger, share_size=1)
    results = run_shares(url, tasks[:-1], logger=quiet_logger)
    written = recorded_lines(quiet_logger, monkeypatch)
    complete(url, [root], results, logger=quiet_logger)
    assert reports_in(written) == []


# ---------------------------------------------------------------------------
# A failure of the report costs the report
# ---------------------------------------------------------------------------


def count_failure_warning(root: Path) -> str:
    """Return the line a pass writes when the standing count cannot be taken.

    Parameters:
        root: The results root the pass covered.

    Returns:
        The line, spelled as the pass writes it, carrying the driver's own
        sentence and not the statement and parameters around it.
    """
    return (
        f'Could not count the refused documents under {normalize_root_url(root)} '
        f'(OperationalError: server closed). The pass itself is unaffected; query '
        f'failed_files for the count.'
    )


def _failing_count(*_args: object, **_kwargs: object) -> int:
    """Fail the way a server that went away between two statements does.

    Parameters:
        _args: Whatever the caller passed positionally.
        _kwargs: Whatever the caller passed by keyword.

    Returns:
        Nothing; it always raises.

    Raises:
        sqlalchemy.exc.OperationalError: Always.
    """
    raise sqlalchemy.exc.OperationalError('SELECT count(*)', {}, Exception('server closed'))


def _two_roots_with_a_failing_report(
    tmp_path: Path, logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, list[str], IngestCounts]:
    """Ingest two roots with the standing count failing on each of them.

    Parameters:
        tmp_path: Directory the trees and the index live under.
        logger: Logger the pass reports through.
        monkeypatch: Fixture the count and the logger are replaced through.

    Returns:
        The index URL, the warnings the pass wrote, and what it counted.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    build_tree(first, 1)
    build_tree(second, 1)
    monkeypatch.setattr(store, '_refusals_the_tree_answers_for', _failing_count)
    warnings = recorded_lines(logger, monkeypatch, level='warning')
    url = index_url(tmp_path / 'index.sqlite3')
    counts = ingest_tree(url, [first, second], logger=logger)
    return url, warnings, counts


def test_a_failing_report_leaves_every_root_ingested(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is one log line, and it must never cost a root its pass.

    Raised out of the loop it sits in, a failure of this query skips every root
    after the one it fired on, and those roots keep a run row with no finish
    time -- so every consumer afterwards refuses them.
    """
    url, _warnings, _counts = _two_roots_with_a_failing_report(tmp_path, quiet_logger, monkeypatch)
    both = [normalize_root_url(tmp_path / 'first'), normalize_root_url(tmp_path / 'second')]
    assert roots_holding_an_image(url) == both
    assert roots_whose_newest_run_finished(url) == both


def test_a_failing_report_keeps_the_counts_of_the_root_it_fired_on(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass's own account of what it did survives the sentence about it.

    Raised, the failure discards the counts of the root that had just finished
    as well as the roots after it, so the run ends with no closing summary at
    all over work that was completed and written down.
    """
    _url, _warnings, counts = _two_roots_with_a_failing_report(tmp_path, quiet_logger, monkeypatch)
    assert counts.files_ingested == 2


def test_a_failing_report_says_which_root_it_could_not_count(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contained is not silent: the operator is told which number is missing."""
    _url, warnings, _counts = _two_roots_with_a_failing_report(tmp_path, quiet_logger, monkeypatch)
    assert [line for line in warnings if line.startswith('Could not count')] == [
        count_failure_warning(tmp_path / 'first'),
        count_failure_warning(tmp_path / 'second'),
    ]
