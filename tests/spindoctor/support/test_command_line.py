"""Tests for the command line a run log is allowed to record.

Every program that logs its run environment logs the words it was invoked
with, and one of those words can be a database password.  What is pinned here
is that the value of a connection-URL option never reaches a log in any of the
spellings argparse accepts for it, that nothing else on the line is touched,
and that both the console and the file a run persists carry the masked form --
because an unattended run's log file is where a password would sit longest.
"""

import ast
import io
import json
import subprocess
import sys
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.support.command_line import masked_command_line
from spindoctor.support.misc import log_run_environment

_LEFT = 'sup3r'
"""First half of the password, distinctive enough that finding it is a leak."""

_RIGHT = 's3cr3t'
"""Second half, so a rule that hides only part of a password is still caught."""

_PASSWORDS = [
    f'{_LEFT}{_RIGHT}',
    f'{_LEFT}@{_RIGHT}',
    f'{_LEFT}:{_RIGHT}',
    f'{_LEFT}/{_RIGHT}',
    f'{_LEFT}?{_RIGHT}',
    f'{_LEFT}#{_RIGHT}',
    f'{_LEFT}@:/?#{_RIGHT}',
]
"""Passwords carrying every character that also delimits part of a URL.

Each of them makes some reading of the URL end early: an ``@`` ends the
credentials, a ``:`` starts a port, a ``/`` starts a path, a ``?`` starts a
query and a ``#`` starts a fragment.  A rule that stops at the first of them
leaves the rest of the password in the log.
"""

_USER = 'us@er'
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


@pytest.mark.parametrize('password', _PASSWORDS)
def test_a_password_of_any_punctuation_does_not_reach_the_line(password: str) -> None:
    """A password is a password whatever URL syntax it happens to spell."""
    line = ' '.join(masked_command_line(['--results-db', _url(password)]))
    assert _LEFT not in line


@pytest.mark.parametrize('password', _PASSWORDS)
def test_no_tail_of_such_a_password_survives_either(password: str) -> None:
    """Masking that stops at the first delimiter leaves a working password."""
    line = ' '.join(masked_command_line(['--results-db', _url(password)]))
    assert _RIGHT not in line


@pytest.mark.parametrize('password', _PASSWORDS)
def test_a_user_name_carrying_an_at_sign_does_not_shelter_the_password(password: str) -> None:
    """The credentials end at the last at-sign, not the first.

    Both halves are asserted: masking that stopped at the at-sign inside the
    user name would leave the head of the password on the line, and masking
    that stopped at the one inside the password would leave its tail, so a test
    naming one half passes against a defect that exposes the other.
    """
    line = ' '.join(masked_command_line(['--results-db', _url(password, user=_USER)]))
    assert _LEFT not in line
    assert _RIGHT not in line


@pytest.mark.parametrize('password', _PASSWORDS)
def test_the_value_joined_to_its_option_is_masked_too(password: str) -> None:
    """Both spellings argparse accepts put the same password on the same line."""
    line = ' '.join(masked_command_line([f'--results-db={_url(password)}']))
    assert _RIGHT not in line


@pytest.mark.parametrize('option', ['--r', '--res', '--results-d', '--results-db'])
def test_every_abbreviation_argparse_accepts_is_masked(option: str) -> None:
    """A distinguishing prefix is the option, and consumes the URL after it."""
    line = ' '.join(masked_command_line([option, _url(_PASSWORDS[-1])]))
    assert _LEFT not in line


@pytest.mark.parametrize('option', ['--r', '--res', '--results-d', '--results-db'])
def test_every_abbreviation_is_masked_when_joined_by_an_equals_sign(option: str) -> None:
    """The two spellings multiply: an abbreviation joined to its value is a third."""
    line = ' '.join(masked_command_line([f'{option}={_url(_PASSWORDS[-1])}']))
    assert _LEFT not in line


def test_the_index_is_still_named_after_it_is_masked() -> None:
    """Which URL a failed run was given is what the logged line is read for."""
    line = ' '.join(masked_command_line(['--results-db', _url(_PASSWORDS[0])]))
    assert 'db.example:5432/spindoctor' in line


def test_the_option_is_still_named_after_its_value_is_masked() -> None:
    """Which of the resolution levels supplied the URL is the other half of it."""
    line = ' '.join(masked_command_line(['--results-db', _url(_PASSWORDS[0])]))
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


def test_a_slash_before_the_first_colon_ends_the_authority() -> None:
    """A value of this shape carries a path where a password would look to be.

    In ``//svc/corp:x@db.example/spindoctor`` the authority is ``svc`` and
    everything after the slash is the path, so there is no user name, no colon
    introducing a password, and nothing to hide.  That is the reading a URL's
    own grammar gives and the reading ``make_url`` gives -- it parses this as
    host ``svc`` and a database name carrying the rest -- so the value is
    recorded as it was written, and an operator reading the log sees the string
    that failed to connect.
    """
    given = ['--results-db', f'postgresql+psycopg://svc/corp:{_LEFT}{_RIGHT}@db.example/spindoctor']
    assert masked_command_line(given) == given


def test_a_second_index_url_on_the_same_line_is_masked_as_well() -> None:
    """argparse keeps the last of a repeated option; the log carries both.

    Both halves are asserted, because masking that reached only as far as the
    first occurrence would leave the second password whole and a test naming
    one half of it would pass.
    """
    line = ' '.join(
        masked_command_line(
            ['--results-db', _url(_PASSWORDS[0]), '--results-db', _url(_PASSWORDS[3])]
        )
    )
    assert _LEFT not in line
    assert _RIGHT not in line


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


def _persisted_banner(name: str, argv: list[str], log_path: FCPath) -> str:
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
    with log_path.open('r') as stream:
        return str(stream.read())


def test_the_run_banner_does_not_print_a_password() -> None:
    """The console sink of the main logger is on by default, so this reaches a terminal."""
    written = _console_banner(
        'test_banner_console', ['coiss', '--results-db', _url(_PASSWORDS[-1])]
    )
    assert _LEFT not in written


def test_no_tail_of_that_password_is_printed_either() -> None:
    """Masking that stopped at the first URL delimiter would print a working password."""
    written = _console_banner('test_banner_tail', ['coiss', '--results-db', _url(_PASSWORDS[-1])])
    assert _RIGHT not in written


def test_the_run_banner_still_records_the_command_line() -> None:
    """A banner that hid the whole line would cost what the banner is written for."""
    written = _console_banner('test_banner_named', ['coiss', '--results-db', _url(_PASSWORDS[-1])])
    assert 'db.example:5432/spindoctor' in written


def test_a_persisted_run_log_does_not_carry_a_password(tmp_path: Path) -> None:
    """An unattended run writes its log to a file, which is where one would sit longest."""
    written = _persisted_banner(
        'test_banner_file',
        ['coiss', '--results-db', _url(_PASSWORDS[-1])],
        FCPath(tmp_path) / 'run.log',
    )
    assert _LEFT not in written


def test_no_tail_of_that_password_reaches_the_persisted_log_either(tmp_path: Path) -> None:
    """The other half of the password, which the same partial rule would leave behind."""
    written = _persisted_banner(
        'test_banner_file_tail',
        ['coiss', '--results-db', _url(_PASSWORDS[-1])],
        FCPath(tmp_path) / 'run.log',
    )
    assert _RIGHT not in written


def test_a_persisted_run_log_still_records_the_command_line(tmp_path: Path) -> None:
    """The file sink is the copy a reader is handed, and it identifies the run."""
    written = _persisted_banner(
        'test_banner_file_named',
        ['coiss', '--results-db', _url(_PASSWORDS[-1])],
        FCPath(tmp_path) / 'run.log',
    )
    assert 'db.example:5432/spindoctor' in written


_CLI_DIR = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor' / 'cli'

# The interactive programs that both declare a connection-URL option and write
# a run log, which is the pair of properties that puts a password in a file.
_PROGRAMS_LOGGING_A_URL_BEARING_LINE = ['sd_backplanes', 'sd_mosaic']

# What each of them is invoked as, with the dataset positional its parser reads
# before the option, so the line masked here is the shape a run really carries.
_PROGRAM_ARGV = {
    'sd_backplanes': ['coiss_saturn', '--nav-results-root', '/data/nav'],
    'sd_mosaic': ['rings', 'coiss_saturn', '--planet', 'SATURN'],
}


_ARGUMENT_VECTORS = {'command_list', 'sys.argv[1:]'}
"""How a program names the words it was invoked with.

One reads them from the list its own ``main`` was handed and the other from the
interpreter; either is the whole line the operator typed, which is what has to
reach the masking rule.  A call given anything else -- a rebuilt list, a subset,
one option's value -- records something other than the invocation.
"""


def _module_tree(program: str) -> ast.Module:
    """Parse one dispatch module.

    Read rather than imported: what is asserted is the call the program makes,
    which is a property of its source and not of running it.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The parsed module.
    """
    with (_CLI_DIR / f'{program}.py').open('r') as stream:
        return ast.parse(str(stream.read()))


def _banner_calls(program: str) -> list[ast.Call]:
    """Return every call the program makes to the run-banner function.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The calls, in source order.
    """
    return [
        node
        for node in ast.walk(_module_tree(program))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'log_run_environment'
    ]


def _joins_of_the_argument_vector(program: str) -> list[str]:
    """Return every expression joining the program's own arguments into one string.

    Any ``join`` over a value naming ``argv`` or ``command_list`` is a second
    copy of the command line, and a second copy is one the masking rule never
    saw.  Matched on the joined expression rather than on two literal
    spellings, so a differently written formatter is caught too.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The source of each offending call.
    """
    found = []
    for node in ast.walk(_module_tree(program)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'join' or len(node.args) != 1:
            continue
        joined = ast.unparse(node.args[0])
        if 'argv' in joined or 'command_list' in joined:
            found.append(ast.unparse(node))
    return found


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_a_program_carrying_the_option_records_its_line_exactly_once(program: str) -> None:
    """A program with both properties records its command line the one masked way.

    The banner is where a command line is written down, and a program that
    formatted its own would put the URL it was handed into a log file without
    passing the rule that hides the password in it.

    Parameters:
        program: Which of the two programs is under test.
    """
    assert len(_banner_calls(program)) == 1


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_that_call_is_given_the_words_the_program_was_invoked_with(program: str) -> None:
    """Naming the function is not enough: what it is handed is what reaches the log.

    A call given a rebuilt or filtered list would satisfy a search for the
    function's name and still record a line that is not the one typed.

    Parameters:
        program: Which of the two programs is under test.
    """
    given = ast.unparse(_banner_calls(program)[0].args[1])
    assert given in _ARGUMENT_VECTORS


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_such_a_program_formats_no_command_line_of_its_own(program: str) -> None:
    """The guard on that: one place records the line, so nothing else joins it.

    Parameters:
        program: Which of the two programs is under test.
    """
    assert _joins_of_the_argument_vector(program) == []


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_such_a_programs_own_command_line_reaches_a_log_file_masked(
    program: str, tmp_path: Path
) -> None:
    """And the bytes that reach the file carry no password.

    Written with the argv the program is really invoked with, dataset
    positional included, so a rule that only held for a line beginning with the
    option would be caught here.
    """
    argv = [*_PROGRAM_ARGV[program], '--results-db', _url(_PASSWORDS[-1], user=_USER)]
    written = _persisted_banner(f'test_banner_{program}', argv, FCPath(tmp_path) / 'run.log')
    assert _LEFT not in written


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_no_tail_of_that_password_reaches_the_file_either(program: str, tmp_path: Path) -> None:
    """A rule stopping at the first URL delimiter would leave a working password.

    Parameters:
        program: Which of the two programs is under test.
        tmp_path: pytest-provided temporary directory.
    """
    argv = [*_PROGRAM_ARGV[program], '--results-db', _url(_PASSWORDS[-1], user=_USER)]
    written = _persisted_banner(f'test_banner_tail_{program}', argv, FCPath(tmp_path) / 'run.log')
    assert _RIGHT not in written


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_that_line_still_names_the_index(program: str, tmp_path: Path) -> None:
    """The control: a banner that hid the line would pass both assertions above.

    Parameters:
        program: Which of the two programs is under test.
        tmp_path: pytest-provided temporary directory.
    """
    argv = [*_PROGRAM_ARGV[program], '--results-db', _url(_PASSWORDS[-1], user=_USER)]
    written = _persisted_banner(f'test_banner_named_{program}', argv, FCPath(tmp_path) / 'run.log')
    assert 'db.example:5432/spindoctor' in written


@pytest.mark.parametrize('program', _PROGRAMS_LOGGING_A_URL_BEARING_LINE)
def test_that_line_still_names_the_dataset(program: str, tmp_path: Path) -> None:
    """And the rest of the line, which is masked by neither rule nor accident.

    The index URL is the only word the banner is allowed to alter, so the
    positional that says which dataset the run was over survives whole.

    Parameters:
        program: Which of the two programs is under test.
        tmp_path: pytest-provided temporary directory.
    """
    argv = [*_PROGRAM_ARGV[program], '--results-db', _url(_PASSWORDS[-1], user=_USER)]
    written = _persisted_banner(
        f'test_banner_dataset_{program}', argv, FCPath(tmp_path) / 'run.log'
    )
    assert 'coiss_saturn' in written


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
