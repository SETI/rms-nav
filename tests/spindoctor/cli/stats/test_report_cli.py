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

from spindoctor.cli.stats.report import main_report
from spindoctor.results_index import INGEST_RUNS, SCHEMA_VERSION, open_index

from .conftest import index_url, ingest_tree, metadata_document, write_metadata

# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_main_report_writes_a_report(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver opens the index it was named and writes the report."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(['--results-db', url, '--output-dir', str(out)])
    assert exit_code == 0


def test_main_report_accepts_the_drill_down_flags(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The range, suspect, and CSV flags parse and take effect."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    main_report(
        [
            '--results-db',
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
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    out = tmp_path / 'report'
    exit_code = main_report(
        ['--results-db', url, '--root', f'{root.as_posix()}/', '--output-dir', str(out)]
    )
    assert exit_code == 0


def test_main_report_refuses_a_root_nobody_ingested(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence of rows under a root is not evidence that nothing was navigated."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit) as caught:
        main_report(
            [
                '--results-db',
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
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with pytest.raises(SystemExit):
        main_report(
            [
                '--results-db',
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
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    exit_code = main_report(['--output-dir', str(tmp_path / 'report')])
    assert exit_code == 1


def test_main_report_with_neither_names_the_index_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that does not say what to type is a refusal nobody can act on."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert '--results-db' in capsys.readouterr().err


def test_main_report_with_neither_names_the_tree_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both ways forward are named, since either would let the run proceed."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert '--nav-results-root' in capsys.readouterr().err


def test_main_report_says_an_empty_index_variable_named_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator who exported the variable is told the value carries no URL.

    Without it the refusal beside this line says to name an index with
    ``NAV_RESULTS_DB``, which is what they believe they did.
    """
    monkeypatch.setenv('NAV_RESULTS_DB', '')
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert 'is set to an empty value' in capsys.readouterr().err


def test_main_report_says_it_through_its_own_output_rather_than_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """This program's output is terminal text, and both lines are part of it.

    Written through the main log instead, the line arrives wrapped in run-log
    machinery and after the refusal it explains, because the two then travel by
    different routes.
    """
    monkeypatch.setenv('NAV_RESULTS_DB', '')
    main_report(['--output-dir', str(tmp_path / 'report')])
    assert 'NAV_RESULTS_DB' not in capsys.readouterr().out


def test_main_report_refuses_an_index_that_is_not_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer never creates an index; it reports that there is none."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    exit_code = main_report(
        ['--results-db', index_url(missing), '--output-dir', str(tmp_path / 'report')]
    )
    assert exit_code == 1


def test_main_report_leaves_no_database_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty database would answer every question with "not navigated"."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    missing = tmp_path / 'absent.sqlite3'
    main_report(['--results-db', index_url(missing), '--output-dir', str(tmp_path / 'report')])
    assert not missing.exists()


def test_main_report_honors_the_none_sentinel(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported index URL can be overridden on the command line.

    The sentinel means here what it means everywhere: read the tree.  So it is
    the tree that answers, and the exported index is not read at all -- which is
    what a machine with an index configured says to ask for a report over the
    documents as they are now.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_DB', url)
    exit_code = main_report(
        [
            '--results-db',
            'none',
            '--nav-results-root',
            str(root),
            '--output-dir',
            str(tmp_path / 'report'),
        ]
    )
    assert exit_code == 0


def test_main_report_reads_the_environment_variable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine with one index need not name it on every invocation."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    monkeypatch.setenv('NAV_RESULTS_DB', url)
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
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
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
                '--results-db',
                url,
                '--root',
                root.as_posix(),
                '--output-dir',
                str(tmp_path / 'report'),
            ]
        )
    assert caught.value.code == 2
