"""Which options each ``sd_results_index`` subcommand takes, and which it will not.

The program's four modes are subcommands, so the parser is what makes them
exclusive: a mode is named once and each carries the options that mode acts on.
Everything asserted here is an exclusion the program would otherwise have had to
police itself, and each is asserted twice -- that the option is refused where it
does not belong, and that it is accepted where it does.  An exclusion asserted
only in the negative is satisfied by a parser that refuses everything.

The refusals are read from the parser's own error, because that is what an
operator meets: argparse writes a usage message naming the option to standard
error and exits 2, and a message that stopped naming the option would leave
somebody guessing which word of their command line was the problem.

One test runs a refused command line all the way through ``main`` over a real
index, because "refused" has to mean the database was never opened rather than
that a message was printed after it was.
"""

from pathlib import Path

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.cli.results_index.ingest_driver_helpers import run_driver
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)

from spindoctor.cli import sd_results_index
from spindoctor.results_index import index_table_names

STUB = 'VOL/N1454725799_1_CALIB'
"""The stub of the document the tree below holds."""

_URL = 'sqlite:///tmp/index.sqlite3'
"""A well-formed index URL, which parsing never connects to."""

_READ_A_TREE = [sd_results_index.INGEST, sd_results_index.DIVIDE, sd_results_index.COMPLETE]
"""The subcommands that walk a navigation results tree."""

_READ_DOCUMENTS = [sd_results_index.INGEST, sd_results_index.DIVIDE]
"""The subcommands that read documents and remove rows."""

_EVERY_SUBCOMMAND = [*_READ_A_TREE, sd_results_index.DROP]
"""Every subcommand the program offers."""

# The arguments each subcommand needs before it will parse at all, so that a
# refusal under test is the option it is about rather than a missing path.
_REQUIRED_OF: dict[str, list[str]] = {
    sd_results_index.INGEST: [],
    sd_results_index.DIVIDE: ['--tasks-file', 'tasks.json'],
    sd_results_index.COMPLETE: ['--events-log', 'events.log'],
    sd_results_index.DROP: [],
}


def _refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Parse a command line that must not parse, and return what was said about it.

    Parameters:
        argv: The whole command line, without the program name.
        capsys: Fixture the parser's error is captured through.

    Returns:
        The parser's message on standard error.
    """
    with pytest.raises(SystemExit) as caught:
        sd_results_index.parse_args(argv)
    assert caught.value.code == 2
    return capsys.readouterr().err


def _line(subcommand: str, *extra: str) -> list[str]:
    """Return a command line for one subcommand, carrying whatever it requires.

    Parameters:
        subcommand: The subcommand to run.
        *extra: Further arguments, appended after the required ones.

    Returns:
        The arguments, without the program name.
    """
    return [subcommand, '--results-index-db', _URL, *_REQUIRED_OF[subcommand], *extra]


# ---------------------------------------------------------------------------
# One subcommand, named once
# ---------------------------------------------------------------------------


def test_a_command_line_naming_no_subcommand_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The mode is the first thing typed, so a line without one asks for nothing."""
    assert 'COMMAND' in _refused([], capsys)


def test_an_unknown_subcommand_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """And the refusal lists the four, which is what somebody who mistyped needs."""
    assert "invalid choice: 'summarize'" in _refused(['summarize'], capsys)


def test_the_refusal_of_an_unknown_subcommand_names_the_known_ones(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A list of what is on offer is the whole use of that refusal."""
    said = _refused(['summarize'], capsys)
    assert all(subcommand in said for subcommand in _EVERY_SUBCOMMAND)


@pytest.mark.parametrize('subcommand', _EVERY_SUBCOMMAND)
def test_a_second_subcommand_is_refused(
    subcommand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two modes cannot be asked for at once, which is the point of the change."""
    assert subcommand in _refused([sd_results_index.INGEST, subcommand], capsys)


def test_dividing_and_completing_cannot_be_asked_for_at_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dividing the work up and adding it together are different runs.

    Asking for both would write a tasks file and then complete the run it had
    just created, before a single worker had read anything.  Each is its own
    subcommand carrying its own path, so a divide handed the completion's path
    is refused by the name of that path.
    """
    said = _refused(
        [sd_results_index.DIVIDE, '--tasks-file', 'tasks.json', '--events-log', 'events.log'],
        capsys,
    )
    assert '--events-log' in said


@pytest.mark.parametrize('subcommand', _EVERY_SUBCOMMAND)
def test_each_subcommand_parses_on_its_own(subcommand: str) -> None:
    """The control for all of the above, which a parser refusing everything passes."""
    assert sd_results_index.parse_args(_line(subcommand)).command == subcommand


# ---------------------------------------------------------------------------
# The confirmation, which only the drop asks for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('subcommand', _READ_A_TREE)
def test_the_confirmation_flag_is_refused_by_every_other_subcommand(
    subcommand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--yes`` answers a question only the drop asks, so elsewhere it means nothing."""
    assert '--yes' in _refused(_line(subcommand, '--yes'), capsys)


def test_the_drop_takes_the_confirmation_flag() -> None:
    """The control: the one subcommand that asks a question can be answered."""
    assert sd_results_index.parse_args(_line(sd_results_index.DROP, '--yes')).yes is True


def test_a_drop_without_the_flag_still_intends_to_ask() -> None:
    """The default is to ask, so the flag is an opt-out rather than the only path."""
    assert sd_results_index.parse_args(_line(sd_results_index.DROP)).yes is False


# ---------------------------------------------------------------------------
# The results root, which the drop never reads
# ---------------------------------------------------------------------------


def test_the_drop_is_refused_a_results_root(capsys: pytest.CaptureFixture[str]) -> None:
    """A drop walks no tree, so a root named on its command line meant another one."""
    said = _refused(_line(sd_results_index.DROP, '--nav-results-root', '/data/nav'), capsys)
    assert '--nav-results-root' in said


@pytest.mark.parametrize('subcommand', _READ_A_TREE)
def test_every_subcommand_that_reads_a_tree_takes_a_results_root(subcommand: str) -> None:
    """The control: the three that walk one are told which one."""
    arguments = sd_results_index.parse_args(_line(subcommand, '--nav-results-root', '/data/nav'))
    assert arguments.nav_results_roots == ['/data/nav']


@pytest.mark.parametrize('subcommand', _READ_A_TREE)
def test_a_results_root_may_be_named_more_than_once(subcommand: str) -> None:
    """One pass may cover several roots, which is what makes this program different."""
    arguments = sd_results_index.parse_args(
        _line(subcommand, '--nav-results-root', '/data/one', '--nav-results-root', '/data/two')
    )
    assert arguments.nav_results_roots == ['/data/one', '/data/two']


# ---------------------------------------------------------------------------
# What a pass reads and what it removes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('option', ['--force', '--no-prune'])
def test_a_completion_is_refused_the_reading_options(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A completion reads no document and removes no row, so neither asks anything of it.

    Whether the shares were read again, and whether the rows of documents that
    have left the tree went, were both settled by the divide that cut them.
    """
    assert option in _refused(_line(sd_results_index.COMPLETE, option), capsys)


@pytest.mark.parametrize('option', ['--force', '--no-prune'])
def test_a_drop_is_refused_the_reading_options(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drop removes the index and stops, so it has no documents and no rows to spare."""
    assert option in _refused(_line(sd_results_index.DROP, option), capsys)


@pytest.mark.parametrize('subcommand', _READ_DOCUMENTS)
def test_the_two_reading_subcommands_take_force(subcommand: str) -> None:
    """The control for the refusals above, on the option that says what is read."""
    assert sd_results_index.parse_args(_line(subcommand, '--force')).force is True


@pytest.mark.parametrize('subcommand', _READ_DOCUMENTS)
def test_the_two_reading_subcommands_take_no_prune(subcommand: str) -> None:
    """The control for them on the option that says what is removed."""
    assert sd_results_index.parse_args(_line(subcommand, '--no-prune')).prune is False


@pytest.mark.parametrize('subcommand', _READ_DOCUMENTS)
def test_a_pass_prunes_unless_it_is_told_not_to(subcommand: str) -> None:
    """Presence of a row meaning what absence means is the default, not an option."""
    assert sd_results_index.parse_args(_line(subcommand)).prune is True


# ---------------------------------------------------------------------------
# The two paths, each belonging to one subcommand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'subcommand', [sd_results_index.INGEST, sd_results_index.COMPLETE, sd_results_index.DROP]
)
def test_the_tasks_file_belongs_to_the_divide_alone(
    subcommand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the pass that cuts the shares writes them out."""
    assert '--tasks-file' in _refused(_line(subcommand, '--tasks-file', 'tasks.json'), capsys)


@pytest.mark.parametrize(
    'subcommand', [sd_results_index.INGEST, sd_results_index.DIVIDE, sd_results_index.DROP]
)
def test_the_events_log_belongs_to_the_completion_alone(
    subcommand: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the pass that adds the shares up reads what the workers wrote."""
    assert '--events-log' in _refused(_line(subcommand, '--events-log', 'events.log'), capsys)


def test_a_divide_takes_the_tasks_file() -> None:
    """The control: the shares are written where the command line says."""
    arguments = sd_results_index.parse_args([sd_results_index.DIVIDE, '--tasks-file', 'tasks.json'])
    assert arguments.tasks_file == 'tasks.json'


def test_a_completion_takes_the_events_log() -> None:
    """The control: the log is read from where the command line says."""
    arguments = sd_results_index.parse_args(
        [sd_results_index.COMPLETE, '--events-log', 'events.log']
    )
    assert arguments.events_log == 'events.log'


def test_a_divide_without_a_tasks_file_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """The shares have to go somewhere, so the path is the subcommand's own subject."""
    assert '--tasks-file' in _refused([sd_results_index.DIVIDE], capsys)


def test_a_completion_without_an_events_log_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """And there is nothing to add up without one."""
    assert '--events-log' in _refused([sd_results_index.COMPLETE], capsys)


# ---------------------------------------------------------------------------
# What every subcommand shares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('subcommand', _EVERY_SUBCOMMAND)
def test_every_subcommand_takes_the_index_url(subcommand: str) -> None:
    """The index is the one thing all four are about."""
    assert sd_results_index.parse_args(_line(subcommand)).results_index_db == _URL


@pytest.mark.parametrize('subcommand', _EVERY_SUBCOMMAND)
def test_every_subcommand_takes_a_configuration_file(subcommand: str) -> None:
    """A machine's settings reach every mode, including the one that reads no tree."""
    arguments = sd_results_index.parse_args(_line(subcommand, '--config-file', 'nav.yaml'))
    assert arguments.config_file == ['nav.yaml']


@pytest.mark.parametrize('subcommand', _EVERY_SUBCOMMAND)
def test_every_subcommand_takes_the_logging_options(subcommand: str) -> None:
    """A run that fails partway has to appear in a log rather than only in a status."""
    arguments = sd_results_index.parse_args(_line(subcommand, '--log-root', '/var/log/nav'))
    assert arguments.log_root == '/var/log/nav'


# ---------------------------------------------------------------------------
# A refused command line reaches nothing
# ---------------------------------------------------------------------------


def _index_of_one_document(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
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


def _table_names(url: str) -> list[str]:
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


def test_a_refused_command_line_exits_two(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status of a command line the program never ran is a usage error."""
    url = _index_of_one_document(tmp_path, quiet_logger)
    status, _written = run_driver(
        [sd_results_index.DROP, '--results-index-db', url, '--yes', '--force'],
        monkeypatch,
        tmp_path,
    )
    assert status == 2


@pytest.mark.parametrize(
    'command',
    [sd_results_index.INGEST, sd_results_index.DIVIDE, sd_results_index.COMPLETE],
)
def test_the_run_header_names_the_subcommand_that_is_running(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log that opens by naming a pass nobody asked for is worse than none."""
    url = index_url(tmp_path / 'index.sqlite3')
    argv = [command, '--results-index-db', url, '--nav-results-root', str(tmp_path / 'results')]
    if command == sd_results_index.DIVIDE:
        argv += ['--tasks-file', str(tmp_path / 'tasks.json')]
    if command == sd_results_index.COMPLETE:
        argv += ['--events-log', str(tmp_path / 'events.jsonl')]
    _status, written = run_driver(argv, monkeypatch, tmp_path)
    headers = [line for line in written if line.startswith('Starting results index ')]
    assert headers[0] == f'Starting results index {command}'


def test_a_drop_says_it_is_dropping(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fourth subcommand takes a different path to its own header."""
    url = _index_of_one_document(tmp_path, quiet_logger)
    _status, written = run_driver(
        [sd_results_index.DROP, '--results-index-db', url, '--yes'], monkeypatch, tmp_path
    )
    headers = [line for line in written if line.startswith('Starting results index ')]
    assert headers[0] == f'Starting results index {sd_results_index.DROP}'


def test_a_refused_command_line_leaves_the_index_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before anything is opened, so the tables a real drop removes stay."""
    url = _index_of_one_document(tmp_path, quiet_logger)
    run_driver(
        [sd_results_index.DROP, '--results-index-db', url, '--yes', '--force'],
        monkeypatch,
        tmp_path,
    )
    assert _table_names(url) == sorted(index_table_names())


def test_the_same_command_line_without_the_refused_option_empties_the_index(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: without ``--force`` the drop runs and the tables really go."""
    url = _index_of_one_document(tmp_path, quiet_logger)
    run_driver([sd_results_index.DROP, '--results-index-db', url, '--yes'], monkeypatch, tmp_path)
    assert _table_names(url) == []
