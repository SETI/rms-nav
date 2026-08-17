"""Reporting over a results tree, with the report from an index as the control.

``sd_stats_report`` answers every question with a query, so a report over a tree
is a report over an index of that tree: the run ingests it into a temporary index
of its own and reports from that.  The parity that buys is a property of the
design rather than of a test -- there is one implementation of every statistic --
so what is tested here is that the design holds end to end and that the temporary
index really is temporary.

The frozen fixture tree is what both runs read: eight documents across two
missions and a simulated scene, which is the same tree the report's own
regression is measured over, so the comparison covers every section the report
writes.  What the pass could not read is tested over a tree written for it, since
the frozen one is deliberately all readable.
"""

import re
import tempfile
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from sqlalchemy.engine import Engine

from spindoctor import results_index
from spindoctor.cli.stats import report as report_module
from spindoctor.cli.stats.report import TEMPORARY_INDEX_NAME, main_report

from .conftest import RESULTS_TREE, index_url, ingest_tree, metadata_document, write_metadata

TEMPORARY_DIRECTORIES = 'sd_stats_report_*'
"""What this program's throwaway directories are named, for the tests that count them."""


@pytest.fixture
def temporary_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Move where temporary directories are made to a directory of this test's own.

    The tests that assert nothing was left behind would otherwise look in the
    shared temporary directory, where a report running in another worker at that
    moment has one open -- so they would fail on somebody else's work in progress
    and pass when run alone.

    Parameters:
        tmp_path: The test's own directory.
        monkeypatch: Fixture the override is installed through.

    Returns:
        The directory the program's temporary directories will be made in.
    """
    root = tmp_path / 'tmp'
    root.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(root))
    return root


@pytest.fixture
def opened_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every index URL the run opens, in the order it opens them.

    Parameters:
        monkeypatch: Fixture the recording opener is installed through.

    Returns:
        The list, filled as the run opens indexes.
    """
    recorded: list[str] = []
    real_open = results_index.open_index

    def recording_open(url: str, **options: Any) -> Engine:
        recorded.append(url)
        return real_open(url, **options)

    monkeypatch.setattr(report_module, 'open_index', recording_open)
    return recorded


@pytest.fixture
def directory_when_the_report_failed(
    monkeypatch: pytest.MonkeyPatch, temporary_root: Path
) -> list[list[str]]:
    """Fail the report with the temporary index open, recording what was there then.

    The failure is the only moment the directory certainly exists, so it is what
    lets a test tell "nothing was left behind" from "nothing was ever made here".

    Parameters:
        monkeypatch: Fixture the failing report is installed through.
        temporary_root: The directory the program makes its own directory in.

    Returns:
        One entry, holding the names in that directory at the moment the report
        was asked for.
    """
    recorded: list[list[str]] = []

    def failing_report(*_args: Any, **_options: Any) -> Path:
        recorded.append(sorted(entry.name for entry in temporary_root.iterdir()))
        raise ValueError('no report today')

    monkeypatch.setattr(report_module, 'build_report', failing_report)
    return recorded


def _from_the_tree(out: Path, *options: str) -> int:
    """Run the report over the frozen fixture tree.

    Parameters:
        out: Directory receiving the report.
        options: Further command-line options.

    Returns:
        The exit code.
    """
    return main_report(
        ['--nav-results-root', str(RESULTS_TREE), '--output-dir', str(out), *options]
    )


def _from_an_index(tmp_path: Path, out: Path, logger: pdslogger.PdsLogger, *options: str) -> int:
    """Ingest the frozen fixture tree and run the report from that index.

    Parameters:
        tmp_path: Directory the index file is written into.
        out: Directory receiving the report.
        logger: Logger the ingest reports through.
        options: Further command-line options.

    Returns:
        The exit code.
    """
    url = index_url(tmp_path / 'control.sqlite3')
    ingest_tree(url, [RESULTS_TREE], logger=logger)
    return main_report(['--results-db', url, '--output-dir', str(out), *options])


@pytest.fixture
def both_reports(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> tuple[Path, Path]:
    """Write the same report twice, once over the tree and once from an index.

    Parameters:
        tmp_path: Directory both reports and the control index are written under.
        quiet_logger: Logger the control ingest reports through.

    Returns:
        The tree-read report directory and the index-read one, in that order.
    """
    from_tree = tmp_path / 'from-tree'
    from_index = tmp_path / 'from-index'
    assert _from_the_tree(from_tree, '--csv', '--top-n', '3') == 0
    assert _from_an_index(tmp_path, from_index, quiet_logger, '--csv', '--top-n', '3') == 0
    return from_tree, from_index


def test_the_report_is_byte_identical_to_one_from_an_index(
    both_reports: tuple[Path, Path],
) -> None:
    """The acceptance criterion: one set of statements answers either way.

    Byte-identical rather than equivalent, because the report is deterministic by
    design and a difference of any kind is a difference in the data or a defect.
    """
    from_tree, from_index = both_reports
    assert (from_tree / 'report.md').read_bytes() == (from_index / 'report.md').read_bytes()


def test_the_csv_export_is_byte_identical_too(both_reports: tuple[Path, Path]) -> None:
    """The flattened one-row-per-image export is the other half of the output."""
    from_tree, from_index = both_reports
    assert (from_tree / 'images.csv').read_bytes() == (from_index / 'images.csv').read_bytes()


def test_the_report_is_not_empty(both_reports: tuple[Path, Path]) -> None:
    """Two empty reports would be byte-identical and would prove nothing.

    The comparisons above are equalities, so they would pass over a tree that
    ingested nothing at all.  This is what says the tree was read.
    """
    from_tree, _from_index = both_reports
    assert 'N1294561202' in (from_tree / 'report.md').read_text(encoding='utf-8')


def test_the_report_covers_every_image_the_tree_holds(both_reports: tuple[Path, Path]) -> None:
    """And the count is the tree's, not one image of it.

    The fixture tree holds eight documents; the CSV is one row per image plus its
    header, so this is the whole tree arriving rather than whatever the walk
    happened to reach first.
    """
    from_tree, _from_index = both_reports
    rows = (from_tree / 'images.csv').read_text(encoding='utf-8').splitlines()
    assert len(rows) == 9


def test_the_temporary_index_is_gone_when_the_run_ends(
    tmp_path: Path, temporary_root: Path
) -> None:
    """The index is this program's own and nobody is invited to keep it.

    Left behind, it would be an index of the tree as it was at some past moment,
    in a temporary directory, that a later run of anything could be pointed at.
    """
    assert _from_the_tree(tmp_path / 'report') == 0
    assert list(temporary_root.glob(TEMPORARY_DIRECTORIES)) == []


def test_the_temporary_index_was_made_where_these_tests_are_watching(
    tmp_path: Path, directory_when_the_report_failed: list[list[str]]
) -> None:
    """Otherwise "nothing was left behind" is "nothing was ever made here".

    Every other test in this group asserts an empty directory, and an empty
    directory is also what a program that put its index somewhere else leaves.
    """
    with pytest.raises(SystemExit):
        _from_the_tree(tmp_path / 'report')
    assert directory_when_the_report_failed[0] != []


def test_the_temporary_index_is_gone_when_the_report_fails(
    tmp_path: Path,
    temporary_root: Path,
    directory_when_the_report_failed: list[list[str]],
) -> None:
    """Including when the run ends the other way, which is when litter accumulates.

    The report is made to fail after the ingest has built the index, which is the
    only window in which there is anything to leave behind.
    """
    with pytest.raises(SystemExit):
        _from_the_tree(tmp_path / 'report')
    assert list(temporary_root.glob(TEMPORARY_DIRECTORIES)) == []


def test_naming_a_root_with_no_index_is_refused(tmp_path: Path) -> None:
    """``--root`` selects among the roots an index holds, and there is no index.

    Read as a second spelling of ``--nav-results-root`` it would quietly report
    on a tree the operator did not name; refused, it says which flag to use.
    """
    with pytest.raises(SystemExit) as caught:
        main_report(['--root', str(RESULTS_TREE), '--output-dir', str(tmp_path / 'report')])
    assert caught.value.code == 2


def test_the_refusal_names_the_flag_that_does_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that does not say what to type instead is one nobody can act on."""
    with pytest.raises(SystemExit):
        main_report(['--root', str(RESULTS_TREE), '--output-dir', str(tmp_path / 'report')])
    assert '--nav-results-root' in capsys.readouterr().err


def test_a_root_that_cannot_be_listed_fails_the_run(tmp_path: Path) -> None:
    """A report over a tree nobody could list would cover less and not say so."""
    assert _from_the_tree_at(tmp_path / 'absent', tmp_path / 'report') == 1


def _from_the_tree_at(root: Path, out: Path) -> int:
    """Run the report over one named tree.

    Parameters:
        root: The results root to read.
        out: Directory receiving the report.

    Returns:
        The exit code.
    """
    return main_report(['--nav-results-root', str(root), '--output-dir', str(out)])


def test_a_root_that_cannot_be_listed_writes_no_report(tmp_path: Path) -> None:
    """The exit status and the output agree: nothing was reported on."""
    out = tmp_path / 'report'
    _from_the_tree_at(tmp_path / 'absent', out)
    assert not (out / 'report.md').exists()


def _tree_holding_a_file_that_is_not_a_document(tmp_path: Path) -> Path:
    """Write a results tree holding one document and one file that is not one.

    Every real results tree holds many of the second, which is why the pass
    tallies them by reason rather than failing on them.

    Parameters:
        tmp_path: Directory the tree is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'mixed'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'VOL' / 'notes_metadata.json').write_text('{not json')
    return root


def test_what_the_pass_could_not_read_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A report over a tree says what it could not read rather than covering less.

    Silence here would read as a report over the whole tree, and the operator
    would have no way to tell a tree full of files that were never navigation
    results from a tree whose results will not parse.
    """
    root = _tree_holding_a_file_that_is_not_a_document(tmp_path)
    assert _from_the_tree_at(root, tmp_path / 'report') == 0
    assert re.search(r'Not ingestible: [1-9]', capsys.readouterr().out)


def test_the_tally_names_a_reason_and_an_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """In the same words the ingest logs them in, since it is the same pass."""
    root = _tree_holding_a_file_that_is_not_a_document(tmp_path)
    _from_the_tree_at(root, tmp_path / 'report')
    assert 'file(s), for example' in capsys.readouterr().out


def test_a_file_that_is_not_a_document_does_not_fail_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It is counted, not fatal: a tree of them is the ordinary case.

    Paired with the tally above so that "reported" and "survived" are both
    asserted -- a run that failed would also have printed the tally.
    """
    root = _tree_holding_a_file_that_is_not_a_document(tmp_path)
    out = tmp_path / 'report'
    assert _from_the_tree_at(root, out) == 0
    assert (out / 'report.md').exists()


def test_the_run_says_which_tree_it_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One full read of every document is worth announcing before it starts."""
    _from_the_tree(tmp_path / 'report')
    assert str(RESULTS_TREE) in capsys.readouterr().out


def test_two_trees_are_reported_on_together(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """``--nav-results-root`` repeats, as it does for the ingest.

    Held against an index ingested from both, so the second root is read rather
    than merely accepted.
    """
    second = tmp_path / 'second'
    write_metadata(second, 'VOL/N1666666666_1_CALIB', metadata_document(image_name='N1666666666'))
    from_tree = tmp_path / 'from-tree'
    from_index = tmp_path / 'from-index'
    assert (
        main_report(
            [
                '--nav-results-root',
                str(RESULTS_TREE),
                '--nav-results-root',
                str(second),
                '--output-dir',
                str(from_tree),
            ]
        )
        == 0
    )
    url = index_url(tmp_path / 'control.sqlite3')
    ingest_tree(url, [RESULTS_TREE, second], logger=quiet_logger)
    assert main_report(['--results-db', url, '--output-dir', str(from_index)]) == 0
    assert (from_tree / 'report.md').read_bytes() == (from_index / 'report.md').read_bytes()


def test_the_temporary_index_is_a_file_under_the_temporary_directory(
    tmp_path: Path, temporary_root: Path, opened_urls: list[str]
) -> None:
    """An archive-scale root does not fit in memory, so the index is a file.

    The in-memory URL is also a special case in the opener that nothing else
    asks for.  What the run opens is observed where the choice is expressed, so
    that "a file" is asserted rather than assumed from the run having worked --
    an in-memory index would report perfectly well and only fail on a tree too
    big for a test.
    """
    assert _from_the_tree(tmp_path / 'report') == 0
    assert opened_urls != []
    assert opened_urls[0].endswith(TEMPORARY_INDEX_NAME)


def test_the_temporary_index_is_not_the_in_memory_database(
    tmp_path: Path, temporary_root: Path, opened_urls: list[str]
) -> None:
    """Stated as its own assertion, since it is the mistake being guarded against."""
    assert _from_the_tree(tmp_path / 'report') == 0
    assert ':memory:' not in opened_urls[0]
