"""Tests for the ``sd_stats_report`` command line, reading an index it was named.

What the driver does with an index is mostly refuse: an index that is not there,
an index it must not create, and a root the index holds no completed ingest of.
The last is the load-bearing one -- absence of rows under a root is not evidence
that nothing was navigated -- and it is decided by the root's newest ingest run
alone, however many earlier ones finished.

Naming no index is not a refusal: it reads the results tree, which is
``test_report_over_a_tree.py``. What is refused here is naming neither, which is
a run with nothing to report on.
"""

from pathlib import Path

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)

from spindoctor.cli.stats.report import main_report
from spindoctor.results_index import INGEST_RUNS, SCHEMA_VERSION, TECHNIQUES, open_index

# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_report_writes_a_report(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver opens the index it was named and writes the report."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(['--results-index-db', url, '--output-dir', str(out)])
    assert exit_code == 0


def test_main_report_accepts_the_drill_down_flags(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The range, suspect, and CSV flags parse and take effect."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    main_report(
        [
            '--results-index-db',
            url,
            '--output-dir',
            str(out),
            '--top-n',
            '3',
            '--filelists',
            '--csv',
            '--suspect-fraction',
            '0.8',
            '--min-image',
            '1',
        ]
    )
    text = (out / 'report.md').read_text(encoding='utf-8')
    assert 'at least 0.80 of the per-axis maximum expected pointing' in text


def test_main_report_accepts_a_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root is normalized the way ingest normalized it, so it matches."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(
        ['--results-index-db', url, '--root', f'{root.as_posix()}/', '--output-dir', str(out)]
    )
    assert exit_code == 0


def test_main_report_refuses_a_root_nobody_ingested(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of rows under a root is not evidence that nothing was navigated."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit) as caught:
        main_report(
            [
                '--results-index-db',
                url,
                '--root',
                str(tmp_path / 'never-ingested'),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert caught.value.code == 2


def test_main_report_names_the_roots_it_does_hold(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The message has to be actionable, so it says what the index does cover."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit):
        main_report(
            [
                '--results-index-db',
                url,
                '--root',
                str(tmp_path / 'never-ingested'),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert root.as_posix() in capsys.readouterr().err


def test_main_report_with_neither_an_index_nor_a_tree_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming no index reads a tree, and naming no tree either leaves nothing."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    exit_code = main_report(['--output-dir', str(tmp_path / 'report')])
    assert exit_code == 1


def test_main_report_with_neither_names_the_index_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that does not say what to type is a refusal nobody can act on."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert '--results-index-db' in capsys.readouterr().err


def test_main_report_with_neither_names_the_tree_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both ways forward are named, since either would let the run proceed."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert '--nav-results-root' in capsys.readouterr().err


def _empty_variable_over_a_readable_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, Path]:
    """Run the driver with an empty index variable and a tree it could report over.

    The tree is what makes the refusal mean something: read as naming no index,
    the empty value would send this run to the documents and it would exit 0
    with a report written, which is exactly the outcome the refusal exists to
    prevent.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.

    Returns:
        The exit code, and where the report would have been written.
    """
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', '')
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    output_dir = tmp_path / 'report'
    exit_code = main_report(['--nav-results-root', str(root), '--output-dir', str(output_dir)])
    return exit_code, output_dir


def test_main_report_refuses_an_empty_index_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported variable carrying no URL stops the run rather than steering it.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.
    """
    exit_code, _output_dir = _empty_variable_over_a_readable_tree(tmp_path, monkeypatch)
    assert exit_code == 1


def test_main_report_says_an_empty_index_variable_named_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator who exported the variable is told the value carries no URL.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.
        capsys: Fixture the refusal is read back from.
    """
    _empty_variable_over_a_readable_tree(tmp_path, monkeypatch)
    assert 'is set to an empty value' in capsys.readouterr().err


def test_main_report_names_the_variable_the_empty_value_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One unset fixes it, and only the named level says which one to unset.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.
        capsys: Fixture the refusal is read back from.
    """
    _empty_variable_over_a_readable_tree(tmp_path, monkeypatch)
    assert 'NAV_RESULTS_INDEX_DB' in capsys.readouterr().err


def test_main_report_says_it_through_its_own_output_rather_than_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """This program's output is terminal text, and the refusal is part of it.

    Written through the main log instead, the line arrives wrapped in run-log
    machinery and on the stream the report itself is announced on.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.
        capsys: Fixture the streams are read back from.
    """
    _empty_variable_over_a_readable_tree(tmp_path, monkeypatch)
    assert 'is set to an empty value' not in capsys.readouterr().out


def test_main_report_writes_no_report_from_an_empty_index_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused run produces nothing, so no report is mistaken for an answer.

    Parameters:
        tmp_path: Directory the tree and the report live under.
        monkeypatch: Fixture the exported variable is set through.
    """
    _exit_code, output_dir = _empty_variable_over_a_readable_tree(tmp_path, monkeypatch)
    assert not output_dir.exists()


def test_main_report_refuses_an_index_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer never creates an index; it reports that there is none."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    exit_code = main_report(
        ['--results-index-db', index_url(missing), '--output-dir', str(tmp_path / 'report')]
    )
    assert exit_code == 1


def test_main_report_leaves_no_database_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty database would answer every question with "not navigated"."""
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    main_report(
        ['--results-index-db', index_url(missing), '--output-dir', str(tmp_path / 'report')]
    )
    assert not missing.exists()


def test_a_read_failure_while_streaming_fails_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An index that stops answering mid-pass is a failed run, not a bad command line.

    The stream issues its queries while the report reads them, so an index that
    can be opened and then cannot be read fails from inside the pass.  Reported
    as a usage error it would exit 2 and print a usage line over a database
    failure no command line could have avoided.  What kind of failure it is does
    not matter, only that it lands while the pass is reading, so a table dropped
    after the open stands in for a lost connection.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP TABLE {TECHNIQUES.name}'))
    finally:
        engine.dispose()
    exit_code = main_report(['--results-index-db', url, '--output-dir', str(tmp_path / 'report')])
    assert exit_code == 1


def test_a_read_failure_while_streaming_says_the_index_could_not_be_read(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exit code says a run failed and the message has to say what failed.

    Nothing else on the command line is at fault, so a line that did not name
    the index would leave an operator looking at their own flags.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP TABLE {TECHNIQUES.name}'))
    finally:
        engine.dispose()
    main_report(['--results-index-db', url, '--output-dir', str(tmp_path / 'report')])
    assert 'Cannot read the results index' in capsys.readouterr().err


def test_an_image_bound_with_no_digits_is_a_usage_error(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound naming no number is a value on the command line, and exits 2 for it.

    The bound is read before any storage is opened, which is what leaves a
    failure raised later in the pass meaning one thing.  Read after the source
    is open, it would arrive at the same place as an index that could not be
    read, and one of the two would take the other's exit code.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit) as caught:
        main_report(
            [
                '--results-index-db',
                url,
                '--min-image',
                'nodigits',
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert caught.value.code == 2


def test_main_report_honors_the_none_sentinel(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported index URL is overridden on the command line, and the tree answers.

    The sentinel means here what it means everywhere: read the tree -- which is
    what a machine with an index configured says to ask for a report over the
    documents as they are now.  The tree therefore holds an image written after
    the ingest, which the index carries no row for, and the report names it as
    the highest-numbered image it selected.  Measured by exit code alone this
    would pass on a run that read the exported index instead, since that index
    reports on the same root and exits 0 doing it.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(
        root, 'VOL/N1595336177_1_CALIB', metadata_document(image_name='N1595336177_1_CALIB.IMG')
    )
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)
    out = tmp_path / 'report'
    main_report(
        [
            '--results-index-db',
            'none',
            '--nav-results-root',
            str(root),
            '--output-dir',
            str(out),
        ]
    )
    assert 'N1595336177' in (out / 'report.md').read_text(encoding='utf-8')


def test_main_report_reads_the_environment_variable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with one index need not name it on every invocation."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_INDEX_DB', url)
    exit_code = main_report(['--output-dir', str(tmp_path / 'report')])
    assert exit_code == 0


# ---------------------------------------------------------------------------
# A root whose newest run did not finish
# ---------------------------------------------------------------------------


def test_main_report_refuses_a_root_whose_newest_run_did_not_finish(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest that started and died leaves a root nothing may read absence from.

    However many earlier runs finished, the tree the dead run half-walked is the
    tree a consumer would be answering from, so the newest run is the only one
    that decides.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert().values(
                root_url=root.as_posix(),
                started_utc='2026-08-07T00:00:00+00:00',
                finished_utc=None,
                schema_version=SCHEMA_VERSION,
            )
        )
    engine.dispose()
    with pytest.raises(SystemExit) as caught:
        main_report(
            [
                '--results-index-db',
                url,
                '--root',
                root.as_posix(),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert caught.value.code == 2
