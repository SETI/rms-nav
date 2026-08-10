"""Tests for ``sd_stats_ingest --drop-index``.

The destructive command, so what is asked of it is mostly what it does *not* do:
it does not drop without an answer, it does not treat a closed standard input as
consent, it does not walk a tree, it does not print a password, and it does not
leave behind a database anything reads differently from one nobody ever built.

The last of those is asked of the real consumer entry points rather than
asserted about the schema, because "indistinguishable to every consumer" is a
statement about them.
"""

import io
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy

from spindoctor.cli.stats.report import main_report
from spindoctor.results_index import IMAGES, SCHEMA_VERSION, index_table_names, read_result_stubs

from .conftest import index_url, ingest_tree, metadata_document, rows_of, write_metadata
from .ingest_driver_helpers import run_driver

STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the document the trees below hold."""

SECRET = 'p@ss:w/rd?x#y'
"""A password carrying every character that means something to a URL.

Each of them is a place a rule reading the URL by eye stops early, and the
at-sign in the user name beside it is the other one.  A message that carries
this string carries a working password.
"""

CREDENTIALED_URL = f'postgresql+psycopg://ad@min:{SECRET}@127.0.0.1:1/spindoctor'
"""A server URL nothing answers on, carrying credentials worth hiding.

Port 1 refuses at once, so the refusal this drives is the connection failing
rather than a name lookup waiting.
"""


def _tree_with_an_index(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Write a one-document results tree, ingest it, and return the index URL.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    return url


def _answering(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Put an answer where the confirmation will read one.

    Parameters:
        monkeypatch: Fixture standard input is replaced through.
        answer: What is typed, including any newline.  An empty string is a
            standard input at its end, which is what a scheduled run has.
    """
    monkeypatch.setattr('sys.stdin', io.StringIO(answer))


def _drop(
    url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> tuple[int | None, list[str]]:
    """Run a drop of one index and return its status and main log.

    Parameters:
        url: The index URL to drop.
        monkeypatch: Fixture the driver is run through.
        tmp_path: Directory the run's log files are written under.
        extra: Further arguments, such as ``--yes``.

    Returns:
        The exit status, and one entry per line written to the main log.
    """
    return run_driver(['--results-db', url, '--drop-index', *extra], monkeypatch, tmp_path)


def _tables(url: str) -> list[str]:
    """Return every table a database holds.

    Parameters:
        url: The database URL.

    Returns:
        The table names, sorted.
    """
    engine = sqlalchemy.create_engine(url)
    try:
        return sorted(sqlalchemy.inspect(engine).get_table_names())
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The confirmation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('answer', ['y\n', 'yes\n', 'Y\n', ' yes \n'])
def test_an_answer_of_yes_drops_the_tables(
    answer: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Case and surrounding space are not part of the answer.

    Parameters:
        answer: What is typed at the prompt.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, answer)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == []


@pytest.mark.parametrize('answer', ['n\n', '\n', 'yes please\n', 'drop\n'])
def test_any_other_answer_leaves_the_index_alone(
    answer: str, tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything that is not yes is no, including nothing at all.

    Parameters:
        answer: What is typed at the prompt.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, answer)
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == sorted(index_table_names())


def test_an_answer_that_is_not_yes_exits_nonzero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script must not read success from a drop that did not happen.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    status, _written = _drop(url, monkeypatch, tmp_path)
    assert status == 1


def test_a_declined_drop_says_what_the_answer_was(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run log has to distinguish an answer of no from a failure to ask.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('rather than yes' in line for line in written)


def test_a_drop_with_nobody_to_ask_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standard input at its end is not consent.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _drop(url, monkeypatch, tmp_path)
    assert _tables(url) == sorted(index_table_names())


def test_a_drop_with_nobody_to_ask_names_the_flag_that_would_have_worked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduled run's operator reads the log to find out what to add.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('--yes' in line for line in written)


def test_yes_drops_without_reading_an_answer(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Told not to ask, it does not ask, and does not obey what was typed anyway.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path, '--yes')
    assert _tables(url) == []


def test_the_question_is_put_to_the_terminal(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary is read in the second before an answer is typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert 'Drop 6 table(s) and ' in capsys.readouterr().out


def test_the_question_names_the_index_it_is_about(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The question has to stand on its own, for a run whose log went to a file.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert f'from {url}?' in capsys.readouterr().out


def test_the_account_of_each_table_is_logged(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is at stake is rows, and the tree above holds exactly one image.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('images: 1 row(s)' in line for line in written)


def test_a_run_told_not_to_ask_still_records_what_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary goes to the log whether or not anybody was shown it.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _answering(monkeypatch, '')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('images: 1 row(s)' in line for line in written)


def test_the_drop_names_the_tables_it_removed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Says what it removed" is a list of names, not a count.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    dropped = [line for line in written if line.startswith('Dropped from ')]
    assert dropped[0].endswith(', '.join(index_table_names()))


def test_an_unfinished_ingest_run_is_reported_before_the_question(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drop under a live pass is allowed, so the person answering is told.

    The run is left unfinished the way a real one is: a fan-out records what its
    listing found and waits for workers that have not run.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            root.as_posix(),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    _answering(monkeypatch, 'n\n')
    _status, written = _drop(url, monkeypatch, tmp_path)
    assert any('1 ingest run(s) have begun and not finished' in line for line in written)


def test_an_unfinished_ingest_run_is_named_in_the_question_itself(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one fact a person answering could not have known from what they typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
        capsys: Fixture the terminal output is captured through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    run_driver(
        [
            '--results-db',
            url,
            '--nav-results-root',
            root.as_posix(),
            '--output-cloud-tasks-file',
            str(tmp_path / 'tasks.json'),
        ],
        monkeypatch,
        tmp_path,
    )
    _answering(monkeypatch, 'n\n')
    _drop(url, monkeypatch, tmp_path)
    assert '1 ingest run(s) have not finished.' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Dropping nothing
# ---------------------------------------------------------------------------


def test_a_database_holding_no_index_exits_zero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state asked for is the state arrived at, so it is not a failure.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 0


def test_a_database_holding_no_index_says_it_removed_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idempotent drop has to be visibly idempotent rather than silent.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert any('holds none of the results index tables' in line for line in written)


def test_nobody_is_asked_when_there_is_nothing_to_drop(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A question about nothing is noise, so it is not asked at all.

    Standard input is at its end, which would refuse a drop that had something
    to remove.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver and standard input are run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    _answering(monkeypatch, '')
    status, _written = _drop(url, monkeypatch, tmp_path)
    assert status == 0


# ---------------------------------------------------------------------------
# What a drop refuses to be combined with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('extra', 'named'),
    [
        (['--force'], '--force'),
        (['--output-cloud-tasks-file', 'tasks.json'], '--output-cloud-tasks-file'),
        (['--complete-cloud-tasks-file', 'events.log'], '--complete-cloud-tasks-file'),
    ],
)
def test_a_drop_refuses_the_ingest_options(
    extra: list[str],
    named: str,
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused rather than ignored: a program may not choose which half to do.

    Parameters:
        extra: The ingest option the command line also carries.
        named: What the refusal has to name.
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, *extra)
    assert any(named in line for line in written)


def test_a_refused_combination_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is before the open, so nothing is opened and nothing goes.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--force', '--yes')
    assert _tables(url) == sorted(index_table_names())


def test_a_refused_combination_exits_nonzero(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status has to say the command did not run, not that it ran.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    status, _written = _drop(url, monkeypatch, tmp_path, '--force', '--yes')
    assert status == 1


def test_yes_without_a_drop_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It answers a question only the drop asks, so on its own it means nothing.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    status, _written = run_driver(['--results-db', url, '--yes'], monkeypatch, tmp_path)
    assert status == 1


def test_the_refusal_of_yes_alone_says_what_to_add(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who typed it meant something, so the refusal says what.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = run_driver(['--results-db', url, '--yes'], monkeypatch, tmp_path)
    assert any('--drop-index' in line for line in written)


# ---------------------------------------------------------------------------
# What a drop does not need, and what it will not do
# ---------------------------------------------------------------------------


def test_a_drop_needs_no_results_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drop is about the database alone, on a machine that may not hold the tree.

    Neither the option nor the environment variable names one here, which is
    what stops every other mode of this program.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    monkeypatch.delenv('NAV_RESULTS_ROOT', raising=False)
    url = _tree_with_an_index(tmp_path, quiet_logger)
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 0


def test_a_drop_ingests_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It stops where it stops: a mistyped URL costs one command, not a walk.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    url = _tree_with_an_index(tmp_path, quiet_logger)
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert not any(line.startswith('Metadata files seen') for line in written)


def test_a_database_that_is_not_there_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typed path gets the answer a server gives for a database that is absent.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    url = index_url(tmp_path / 'absent.sqlite3')
    status, _written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert status == 1


def test_the_index_being_dropped_is_recorded_before_it_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log saying only that a database would not open leaves out which one.

    The database here is one that cannot be opened, so a line naming it is a
    line written before the attempt.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    url = index_url(tmp_path / 'absent.sqlite3')
    _status, written = _drop(url, monkeypatch, tmp_path, '--yes')
    assert f'Results index to drop the tables of: {url}' in written


def test_a_database_that_is_not_there_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped URL must not leave an empty database where it pointed.

    Parameters:
        tmp_path: Directory the path names a file in.
        monkeypatch: Fixture the driver is run through.
    """
    path = tmp_path / 'absent.sqlite3'
    _drop(index_url(path), monkeypatch, tmp_path, '--yes')
    assert not path.exists()


# ---------------------------------------------------------------------------
# The password
# ---------------------------------------------------------------------------


def _credentialed_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run a drop of an unreachable server URL and return everything it said.

    Parameters:
        monkeypatch: Fixture the driver is run through.
        tmp_path: Directory the run's log files are written under.
        capsys: Fixture the terminal output is captured through.

    Returns:
        The main log and the terminal output, joined.
    """
    _status, written = _drop(CREDENTIALED_URL, monkeypatch, tmp_path, '--yes')
    captured = capsys.readouterr()
    return '\n'.join([*written, captured.out, captured.err])


def test_the_password_survives_nowhere_a_drop_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asserted on the bytes, over a password carrying every character that ends one.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    assert SECRET not in _credentialed_drop(monkeypatch, tmp_path, capsys)


def test_no_run_of_the_password_survives_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A driver quotes back the fragment it stopped on, not the field it read.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    said = _credentialed_drop(monkeypatch, tmp_path, capsys)
    assert not any(SECRET[start : start + 4] in said for start in range(len(SECRET) - 3))


def test_the_masked_url_still_names_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Masking must not cost the identification the message exists for.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    assert '127.0.0.1' in _credentialed_drop(monkeypatch, tmp_path, capsys)


def test_the_command_line_the_drop_logs_is_masked_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run banner records the arguments, and one of those words is the URL.

    Parameters:
        tmp_path: Directory the run's log files are written under.
        monkeypatch: Fixture the driver is run through.
        capsys: Fixture the terminal output is captured through.
    """
    said = _credentialed_drop(monkeypatch, tmp_path, capsys)
    assert 'ad@min:***@127.0.0.1' in said


# ---------------------------------------------------------------------------
# A dropped index and an index that never existed
# ---------------------------------------------------------------------------


def _dropped_and_never_built(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, str]:
    """Return the URL of an index that was dropped and of one never built.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.

    Returns:
        The dropped index's URL and the never-built one's.
    """
    dropped = _tree_with_an_index(tmp_path, quiet_logger)
    _drop(dropped, monkeypatch, tmp_path, '--yes')
    return dropped, index_url(tmp_path / 'never-built.sqlite3')


def test_neither_a_dropped_index_nor_an_absent_one_answers_a_selection(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The filters read absence of a row as "never navigated", so both must refuse.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch):
        with pytest.raises(ValueError):
            read_result_stubs(url, tmp_path / 'results', ['VOL'])


def test_both_send_the_reader_to_the_ingest(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two states, one remedy, so the two refusals prescribe the same thing.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    said = []
    for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch):
        with pytest.raises(ValueError) as excinfo:
            read_result_stubs(url, tmp_path / 'results', ['VOL'])
        said.append('sd_stats_ingest' in str(excinfo.value))
    assert said == [True, True]


def test_the_report_refuses_both(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one consumer that requires an index treats the two the same way.

    Parameters:
        tmp_path: Directory the trees and the indexes live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    statuses = [
        main_report(['--results-db', url, '--output-dir', str(tmp_path / 'report')])
        for url in _dropped_and_never_built(tmp_path, quiet_logger, monkeypatch)
    ]
    assert statuses == [1, 1]


def test_an_ingest_over_a_dropped_index_writes_the_rows_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starting from scratch, which is the operation the whole command is for.

    The rows are compared against the ones the first ingest wrote, so a drop
    that had left something behind, or one an ingest could only half-rebuild
    over, shows up as a difference rather than as a count.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    before = rows_of(url, IMAGES)
    _drop(url, monkeypatch, tmp_path, '--yes')
    ingest_tree(url, [root], logger=quiet_logger)
    assert rows_of(url, IMAGES) == before


def test_a_rebuilt_index_carries_this_schema_version(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp goes with the tables and comes back with them.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        quiet_logger: Logger the ingest reports through.
        monkeypatch: Fixture the driver is run through.
    """
    root = tmp_path / 'results'
    write_metadata(root, STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    _drop(url, monkeypatch, tmp_path, '--yes')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            stamped: Any = connection.exec_driver_sql(
                'SELECT schema_version FROM schema_meta'
            ).scalar()
    finally:
        engine.dispose()
    assert stamped == SCHEMA_VERSION
