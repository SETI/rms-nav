"""Tests for the ``sd_stats_ingest`` command line.

Three things the driver does are worth pinning on their own. It reports a
configuration failure -- no index named, no results root resolvable -- through
the log an operator is already reading, rather than as a traceback. Of what it
writes to that log, the index URL is masked and nothing else is: an index URL
can carry a database password and a run log is read by whoever is handed one,
while a results root carries no credentials and is the one word of the command
line the reader is there to correct. And its exit status says whether the run
completed, not what the run found, so a scheduled invocation reads the same
status from the same tree every time.
"""

import os
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

CREDENTIAL_SHAPED_ROOT = '//store:8443/nav@results'
"""A results root the masking rule would read as credentials if it saw one.

Everything between the first colon and the at-sign is a password to that rule.
It is a path, and a run log that printed ``//store:***@results`` would have
hidden the only thing an operator needs from the line.
"""


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
    monkeypatch.setattr(MAIN_LOGGER, 'warning', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'error', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'fatal', recording)
    monkeypatch.setattr(MAIN_LOGGER, 'exception', recording)
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


def test_no_index_is_reported_and_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ingest with nowhere to write is the other configuration failure.

    Both ambient sources of an index URL are closed for every test, so naming
    none on the command line is a program run with no index at all -- which for
    this one program is a mistake rather than the ordinary case it is everywhere
    else in the pipeline.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    status, _written = _run(['--nav-results-root', str(root)], monkeypatch, tmp_path)
    assert status == 1


def test_no_index_says_which_settings_supply_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And names all three, since which one an operator meant to set is theirs."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    _status, written = _run(['--nav-results-root', str(root)], monkeypatch, tmp_path)
    assert any('NAV_RESULTS_DB' in line for line in written)


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


def test_a_results_root_reaches_the_arguments_whole() -> None:
    """The one word of the line an operator is reading it to correct."""
    masked = sd_stats_ingest.masked_command_line(
        ['--nav-results-root', CREDENTIAL_SHAPED_ROOT, '--force']
    )
    assert masked == ['--nav-results-root', CREDENTIAL_SHAPED_ROOT, '--force']


def test_an_index_url_is_masked_in_the_arguments() -> None:
    """A password can be a word of the command line, and the line is logged."""
    masked = sd_stats_ingest.masked_command_line(['--results-db', SERVER_URL])
    assert masked == ['--results-db', 'postgresql+psycopg:/svc:***@db.example/spindoctor']


def test_an_index_url_joined_by_an_equals_sign_is_masked_too() -> None:
    """Both spellings argparse accepts put the same password on the same line."""
    masked = sd_stats_ingest.masked_command_line([f'--results-db={SERVER_URL}'])
    assert masked == ['--results-db=postgresql+psycopg:/svc:***@db.example/spindoctor']


def test_an_abbreviated_option_carries_the_same_url() -> None:
    """argparse accepts any distinguishing prefix and consumes the URL after it."""
    masked = sd_stats_ingest.masked_command_line(['--results-d', SERVER_URL])
    assert masked == ['--results-d', 'postgresql+psycopg:/svc:***@db.example/spindoctor']


def test_an_abbreviated_option_joined_by_an_equals_sign_carries_it_too() -> None:
    """The two spellings multiply: an abbreviation joined to its value is a third."""
    masked = sd_stats_ingest.masked_command_line([f'--results-d={SERVER_URL}'])
    assert masked == ['--results-d=postgresql+psycopg:/svc:***@db.example/spindoctor']


def test_the_argument_separator_does_not_swallow_the_word_after_it() -> None:
    """Every long option starts with the separator, and it names none of them."""
    masked = sd_stats_ingest.masked_command_line(['--', CREDENTIAL_SHAPED_ROOT])
    assert masked == ['--', CREDENTIAL_SHAPED_ROOT]


def test_a_root_that_is_not_there_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit status 1, because the run could not walk a root it was given."""
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


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_missed_directory_is_named_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed run that saw part of a root has to say which part it missed.

    Every consumer of the index reads a missing row as "this image was never
    navigated". Under a directory nobody listed that reading is wrong, and the
    summary is where an operator finds out before acting on it.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL1/N1454725799_1_CALIB', metadata_document())
    write_metadata(root, 'VOL2/N1454725800_1_CALIB', metadata_document())
    closed = root / 'VOL2'
    closed.chmod(0o000)
    try:
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
    finally:
        closed.chmod(0o755)
    assert any('Directories not listed' in line for line in written)


def test_a_failure_nobody_enumerated_exits_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A console entry point owes its caller a status, not a traceback.

    The pass charges every failure it expects to one file or one root. What is
    left is a failure nobody enumerated, and the driver's contract is that it
    exits either way -- otherwise a caller reading the status gets an exception
    instead, and the roots the pass never reached keep their unfinished runs
    with nothing said about why.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('a failure nobody enumerated')

    monkeypatch.setattr(sd_stats_ingest, 'ingest_metadata_files', exploding)
    status, _written = _run(
        [
            '--results-db',
            index_url(tmp_path / 'index.sqlite3'),
            '--nav-results-root',
            root.as_posix(),
        ],
        monkeypatch,
        tmp_path,
    )
    assert status == 1


def test_a_failure_nobody_enumerated_says_what_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exit status with nothing in the log leaves nobody anything to act on."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def exploding(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('a failure nobody enumerated')

    monkeypatch.setattr(sd_stats_ingest, 'ingest_metadata_files', exploding)
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
    assert any('a failure nobody enumerated' in line for line in written)


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
    examples = [line for line in written if 'for example' in line]
    assert any('edges_metadata.json' in line for line in examples)


def _statuses_of_two_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Run the driver twice over a tree of files that are not navigation documents.

    Parameters:
        tmp_path: Directory the tree, the index and the logs live under.
        monkeypatch: Fixture the argument vector and logger are replaced through.

    Returns:
        The exit status of each pass.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'rings_metadata.json').write_text('{"rings": []}', encoding='utf-8')
    argv = [
        '--results-db',
        index_url(tmp_path / 'index.sqlite3'),
        '--nav-results-root',
        root.as_posix(),
    ]
    first, _written = _run(argv, monkeypatch, tmp_path)
    second, _also = _run(argv, monkeypatch, tmp_path)
    return [first, second]


def test_two_passes_over_one_tree_exit_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A status that flips on an unchanged tree tells a scheduled run nothing.

    The first pass refuses both files and the second skips them as unchanged, so
    a status read from what was ingested or skipped reports a failure once and
    never again.  The pass completed both times, which is what the status says.
    """
    assert _statuses_of_two_passes(tmp_path, monkeypatch) == [0, 0]
