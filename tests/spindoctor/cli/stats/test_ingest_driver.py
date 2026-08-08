"""Tests for the ``sd_stats_ingest`` command line.

Two things the driver does are worth pinning on their own. It reports a
configuration failure -- no index named, no results root resolvable -- through
the log an operator is already reading, rather than as a traceback. And what it
writes to that log is masked: a results root can be a signed cloud URL and an
index URL can carry a database password, and a run log is read by whoever is
handed one.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from spindoctor.cli import sd_stats_ingest
from spindoctor.config import MAIN_LOGGER

from .conftest import index_url, metadata_document, write_metadata

PASSWORD = 'sup3rs3cr3t'
"""A password distinctive enough that finding it anywhere is proof of a leak."""

SERVER_URL = f'postgresql+psycopg:/svc:{PASSWORD}@db.example/spindoctor'
"""An index URL carrying a password, in the one-slash form a parser rejects."""


def _run(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[int | None, list[str]]:
    """Run the driver and return its exit status and what it told the main log.

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

    monkeypatch.setattr(
        sys, 'argv', ['sd_stats_ingest', '--log-root', str(tmp_path / 'logs'), *argv]
    )
    monkeypatch.setattr(MAIN_LOGGER, 'info', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'error', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'fatal', recording)
    with pytest.raises(SystemExit) as caught:
        sd_stats_ingest.main()
    status = caught.value.code
    return (status if status is None or isinstance(status, int) else 1), written


def test_no_navigation_root_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configuration failure is an operator's mistake, not a traceback."""
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    status, _written = _run(
        ['--results-db', index_url(tmp_path / 'index.sqlite3')], monkeypatch, tmp_path
    )
    assert status == 1


def test_no_navigation_root_says_which_settings_supply_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal that does not say what to set is a refusal nobody can act on."""
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    _status, written = _run(
        ['--results-db', index_url(tmp_path / 'index.sqlite3')], monkeypatch, tmp_path
    )
    assert any('--nav-results-root' in line for line in written)


def test_the_run_log_does_not_carry_a_database_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command line is logged, and a password can be one of its words."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        ['--results-db', SERVER_URL, '--nav-results-root', root.as_posix()], monkeypatch, tmp_path
    )
    assert not any(PASSWORD in line for line in written)


def test_the_run_log_still_names_the_index_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Masking must not cost the identification the log line exists for.

    Which of the command line, the configuration file and the environment
    supplied a bad URL is exactly what the logged arguments answer.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        ['--results-db', SERVER_URL, '--nav-results-root', root.as_posix()], monkeypatch, tmp_path
    )
    assert any('db.example/spindoctor' in line for line in written)


def test_the_run_log_names_the_roots_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary local root carries no credentials and reaches the log whole."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any(f'Roots: {root.as_posix()}' == line for line in written)


def test_a_root_that_is_not_there_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit status 1, because nothing was accounted for under that root."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    status, _written = _run(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_a_root_that_is_not_there_is_named_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped root reads as a root that is empty unless the summary says so."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    _status, written = _run(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            str(tmp_path / 'absent'),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('could not be listed' in line for line in written)


def test_a_failure_reason_names_one_example_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reason is a field-level diagnosis until one real file is named beside it."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert any('for example' in line and 'edges_metadata.json' in line for line in written)
