"""Tests for the command line a run log is allowed to record.

Every program that logs its run environment logs the words it was invoked
with, and one of those words can be a database password.  What is pinned here
is that the value of a connection-URL option never reaches a log in any of the
spellings argparse accepts for it, that nothing else on the line is touched,
and that both the console and the file a run persists carry the masked form --
because an unattended run's log file is where a password would sit longest.
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import pdslogger
import pytest

from spindoctor.support.command_line import masked_command_line
from spindoctor.support.misc import log_run_environment

LEFT = 'sup3r'
"""First half of the password, distinctive enough that finding it is a leak."""

RIGHT = 's3cr3t'
"""Second half, so a rule that hides only part of a password is still caught."""

PASSWORDS = [
    f'{LEFT}{RIGHT}',
    f'{LEFT}@{RIGHT}',
    f'{LEFT}:{RIGHT}',
    f'{LEFT}/{RIGHT}',
    f'{LEFT}?{RIGHT}',
    f'{LEFT}#{RIGHT}',
    f'{LEFT}@:/?#{RIGHT}',
]
"""Passwords carrying every character that also delimits part of a URL.

Each of them makes some reading of the URL end early: an ``@`` ends the
credentials, a ``:`` starts a port, a ``/`` starts a path, a ``?`` starts a
query and a ``#`` starts a fragment.  A rule that stops at the first of them
leaves the rest of the password in the log.
"""

USER = 'us@er'
"""A user name carrying the character that also ends the credentials."""


def _url(password: str, *, user: str = 'svc') -> str:
    """Return a server index URL carrying the given credentials.

    Parameters:
        password: The password to put in the authority.
        user: The user name to put in front of it.

    Returns:
        A ``postgresql+psycopg:`` URL naming a host, a port and a database.
    """
    return f'postgresql+psycopg://{user}:{password}@db.example:5432/spindoctor'


@pytest.mark.parametrize('password', PASSWORDS)
def test_a_password_of_any_punctuation_does_not_reach_the_line(password: str) -> None:
    """A password is a password whatever URL syntax it happens to spell."""
    line = ' '.join(masked_command_line(['--results-db', _url(password)]))
    assert LEFT not in line


@pytest.mark.parametrize('password', PASSWORDS)
def test_no_tail_of_such_a_password_survives_either(password: str) -> None:
    """Masking that stops at the first delimiter leaves a working password."""
    line = ' '.join(masked_command_line(['--results-db', _url(password)]))
    assert RIGHT not in line


@pytest.mark.parametrize('password', PASSWORDS)
def test_a_user_name_carrying_an_at_sign_does_not_shelter_the_password(password: str) -> None:
    """The credentials end at the last at-sign, not the first."""
    line = ' '.join(masked_command_line(['--results-db', _url(password, user=USER)]))
    assert RIGHT not in line


@pytest.mark.parametrize('password', PASSWORDS)
def test_the_value_joined_to_its_option_is_masked_too(password: str) -> None:
    """Both spellings argparse accepts put the same password on the same line."""
    line = ' '.join(masked_command_line([f'--results-db={_url(password)}']))
    assert RIGHT not in line


@pytest.mark.parametrize('option', ['--r', '--res', '--results-d', '--results-db'])
def test_every_abbreviation_argparse_accepts_is_masked(option: str) -> None:
    """A distinguishing prefix is the option, and consumes the URL after it."""
    line = ' '.join(masked_command_line([option, _url(PASSWORDS[-1])]))
    assert LEFT not in line


@pytest.mark.parametrize('option', ['--r', '--res', '--results-d', '--results-db'])
def test_every_abbreviation_is_masked_when_joined_by_an_equals_sign(option: str) -> None:
    """The two spellings multiply: an abbreviation joined to its value is a third."""
    line = ' '.join(masked_command_line([f'{option}={_url(PASSWORDS[-1])}']))
    assert LEFT not in line


def test_the_index_is_still_named_after_it_is_masked() -> None:
    """Which URL a failed run was given is what the logged line is read for."""
    line = ' '.join(masked_command_line(['--results-db', _url(PASSWORDS[0])]))
    assert 'db.example:5432/spindoctor' in line


def test_the_option_is_still_named_after_its_value_is_masked() -> None:
    """Which of the resolution levels supplied the URL is the other half of it."""
    line = ' '.join(masked_command_line(['--results-db', _url(PASSWORDS[0])]))
    assert line.startswith('--results-db ')


def test_a_command_line_naming_no_index_is_returned_word_for_word() -> None:
    """Nothing but a connection URL is a secret, and nothing else is touched."""
    given = ['coiss', '--volumes', 'COISS_2001', '--has-offset-file', '--dry-run']
    assert masked_command_line(given) == given


def test_a_results_root_reaches_the_line_whole() -> None:
    """The one word of the line an operator is reading it to correct.

    Everything between its colon and its at-sign reads as a password to the
    masking rule, and a log that printed the mangled form would have hidden the
    only thing the reader needs.
    """
    given = ['--nav-results-root', '//store:8443/nav@results']
    assert masked_command_line(given) == given


def test_the_argument_separator_does_not_swallow_the_word_after_it() -> None:
    """Every long option starts with the separator, and it names none of them."""
    given = ['--', '//store:8443/nav@results']
    assert masked_command_line(given) == given


def test_a_second_index_url_on_the_same_line_is_masked_as_well() -> None:
    """argparse keeps the last of a repeated option; the log carries both."""
    line = ' '.join(
        masked_command_line(
            ['--results-db', _url(PASSWORDS[0]), '--results-db', _url(PASSWORDS[3])]
        )
    )
    assert LEFT not in line


def _console_banner(name: str, argv: list[str]) -> str:
    """Write one run banner to a console sink and return what it wrote.

    The stream sink is the shape the main logger's console handler has, and it
    is on by default, so what it carries reaches every terminal a run is
    started from.

    Parameters:
        name: A logger name unique to the calling test, so that two tests never
            share a logger and its handlers.
        argv: The command line the banner records.

    Returns:
        The text the console sink received.
    """
    console = io.StringIO()
    logger = pdslogger.PdsLogger(name)
    logger.add_handler(pdslogger.stream_handler(level='info', stream=console))
    try:
        log_run_environment(logger, argv)
    finally:
        logger.remove_all_handlers()
    return console.getvalue()


def _persisted_banner(name: str, argv: list[str], log_path: Path) -> str:
    """Write one run banner to a file and return what the file holds.

    Parameters:
        name: A logger name unique to the calling test.
        argv: The command line the banner records.
        log_path: Where the run's log file is written.

    Returns:
        The text of the log file.
    """
    logger = pdslogger.PdsLogger(name)
    handler = pdslogger.file_handler(log_path, level='info')
    logger.add_handler(handler)
    try:
        log_run_environment(logger, argv)
    finally:
        logger.remove_all_handlers()
        handler.close()
    return log_path.read_text(encoding='utf-8')


def test_the_run_banner_does_not_print_a_password() -> None:
    """The console sink of the main logger is on by default, so this reaches a terminal."""
    written = _console_banner('test_banner_console', ['coiss', '--results-db', _url(PASSWORDS[-1])])
    assert LEFT not in written


def test_the_run_banner_still_records_the_command_line() -> None:
    """A banner that hid the whole line would cost what the banner is written for."""
    written = _console_banner('test_banner_named', ['coiss', '--results-db', _url(PASSWORDS[-1])])
    assert 'db.example:5432/spindoctor' in written


def test_a_persisted_run_log_does_not_carry_a_password(tmp_path: Path) -> None:
    """An unattended run writes its log to a file, which is where one would sit longest."""
    written = _persisted_banner(
        'test_banner_file',
        ['coiss', '--results-db', _url(PASSWORDS[-1])],
        tmp_path / 'run.log',
    )
    assert LEFT not in written


def test_a_persisted_run_log_still_records_the_command_line(tmp_path: Path) -> None:
    """The file sink is the copy a reader is handed, and it identifies the run."""
    written = _persisted_banner(
        'test_banner_file_named',
        ['coiss', '--results-db', _url(PASSWORDS[-1])],
        tmp_path / 'run.log',
    )
    assert 'db.example:5432/spindoctor' in written


def test_masking_a_line_with_no_url_imports_no_database_layer() -> None:
    """The banner is written by every run, and most of them name no index.

    The masking rule lives beside the URL parsing it mirrors, which is in the
    package that imports SQLAlchemy, so it is imported only for a command line
    that actually carries a URL.  This runs in a subprocess because the
    assertion is about a fresh interpreter: anything else in the test session
    has already imported SQLAlchemy.
    """
    probe = (
        'import json, sys\n'
        'from spindoctor.support.command_line import masked_command_line\n'
        'masked_command_line(["coiss", "--volumes", "COISS_2001", "--dry-run"])\n'
        'print(json.dumps(sorted(name for name in sys.modules '
        'if name.split(".")[0] == "sqlalchemy")))\n'
    )
    completed = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
