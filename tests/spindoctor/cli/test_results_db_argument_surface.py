"""Which programs read the results index, and how they are told to.

A program becomes index-backed by declaring ``--results-db``, never by
inheriting an exported variable.  That is the whole policy, and it has two
halves: a declaring program honors the three-level ladder the option belongs to,
and a program that declares nothing resolves no URL at all however the machine
it runs on is configured.  The second half is what a scan enforces -- an
exported ``NAV_RESULTS_DB`` reaches every program on a machine, so a program
that quietly started resolving it would change behavior nobody asked to change.

The flags are asserted against the parser each program builds for itself, so a
group of options moving between shared helpers cannot make this pass by
accident.
"""

import argparse
import contextlib
import io
from collections.abc import Callable
from pathlib import Path

import pytest
from filecache import FCPath
from tests.spindoctor.cli.program_parsers import cloud_task_parser, program_help_text

from spindoctor.cli import (
    sd_backplanes,
    sd_consolidate_metadata,
    sd_create_bundle,
    sd_mosaic,
    sd_offset,
)
from spindoctor.config import DEFAULT_CONFIG, get_results_db_url

_CLI_DIR = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor' / 'cli'

_URL = 'sqlite:///tmp/index.sqlite3'

# The interactive programs that read their records through the index, named by
# the argv each one's own parser is driven with.  The option is asserted by
# parsing it rather than by finding it in help text, so that a differently
# spelled option carrying the same prefix cannot satisfy the assertion.
_INDEX_BACKED: list[tuple[str, list[str]]] = [
    ('sd_backplanes', ['coiss_saturn']),
    ('sd_mosaic', ['rings', 'coiss_saturn', '--output-dir', 'out', '--planet', 'SATURN']),
    ('sd_mosaic', ['body', 'coiss_saturn', '--output-dir', 'out', '--body-name', 'RHEA']),
]

_INDEX_BACKED_CLOUD_TASK_DRIVERS = [
    'sd_backplanes_cloud_tasks',
    'sd_mosaic_cloud_tasks',
]

# The programs that read a results index in any capacity, as the paths of the
# modules that resolve the URL, relative to the CLI package.  The statistics
# programs read it as their whole subject; the pipeline stages read one row per
# image.  A module absent from this list must not resolve a URL: it would then
# pick one up from an exported NAV_RESULTS_DB without ever offering the option
# that opts back out.
_RESOLVING_MODULES = {
    'sd_backplanes.py',
    'sd_backplanes_cloud_tasks.py',
    'sd_mosaic.py',
    'sd_mosaic_cloud_tasks.py',
    'sd_stats_ingest.py',
    'stats/report.py',
}

# Programs deliberately left reading files, each with its own parser and the
# argv that reaches it.  They write the documents, consume a whole document
# rather than a few fields of it, or copy its bytes, so a column schema serves
# none of them.
_FILE_ONLY: list[tuple[str, list[str]]] = [
    ('sd_offset', ['coiss_saturn']),
    ('sd_consolidate_metadata', ['coiss_saturn', '--dest-dir', 'out', '--copy-all']),
    ('sd_create_bundle', ['coiss_saturn']),
]

_FILE_ONLY_PARSERS: dict[str, Callable[[list[str]], argparse.Namespace]] = {
    'sd_offset': sd_offset.parse_args,
    'sd_consolidate_metadata': sd_consolidate_metadata.parse_args,
    'sd_create_bundle': sd_create_bundle.parse_args_labels,
}


def _modules_resolving_a_url() -> set[str]:
    """Return every CLI module that resolves a results index URL.

    Returns:
        Paths relative to the CLI package, using forward slashes.
    """
    root = Path(str(_CLI_DIR))
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob('*.py')
        if 'get_results_db_url' in path.read_text(encoding='utf-8')
    }


def test_only_the_declaring_programs_resolve_an_index_url() -> None:
    """A program that offers no option must not inherit one from the machine.

    Adding a program here means giving it ``--results-db`` and its ``none``
    sentinel in the same change; without them a run on a machine that exports
    the variable could not be told to read files.
    """
    assert _modules_resolving_a_url() == _RESOLVING_MODULES


def _parsed(program: str, argv: list[str]) -> argparse.Namespace:
    """Parse one program's command line with its own parser.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: The whole command line, without the program name.

    Returns:
        The parsed namespace.
    """
    if program == 'sd_mosaic':
        return sd_mosaic.parse_args(argv)[1]
    return sd_backplanes.parse_args(argv)


@pytest.mark.parametrize(('program', 'argv'), _INDEX_BACKED)
def test_an_index_backed_program_parses_the_option(program: str, argv: list[str]) -> None:
    """The stages that read one row per image accept ``--results-db``.

    Asserted on the value the parser produced, which is what the program then
    resolves: an option merely printed in a help text proves neither that it
    parses nor that it lands where the program reads it from.
    """
    arguments = _parsed(program, [*argv, '--results-db', _URL])
    assert arguments.results_db == _URL


@pytest.mark.parametrize(('program', 'argv'), _INDEX_BACKED)
def test_an_index_backed_program_documents_the_opt_out(program: str, argv: list[str]) -> None:
    """And say in their own help how to read the files anyway."""
    assert '"none"' in program_help_text(program, argv)


@pytest.mark.parametrize('program', _INDEX_BACKED_CLOUD_TASK_DRIVERS)
def test_an_index_backed_worker_offers_the_option(program: str) -> None:
    """So do their cloud-task workers, whose own command line is all they have."""
    options = [
        option for action in cloud_task_parser(program)._actions for option in action.option_strings
    ]
    assert '--results-db' in options


@pytest.mark.parametrize(('program', 'argv'), _FILE_ONLY)
def test_a_file_reading_program_refuses_the_option(program: str, argv: list[str]) -> None:
    """Offering it would promise an index the program never reads.

    Asserted by handing the option to the program's own parser and watching it
    exit: an option absent from a help text could also be an option the help
    text does not mention.
    """
    parse = _FILE_ONLY_PARSERS[program]
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse([*argv, '--results-db', _URL])


@pytest.mark.parametrize(('program', 'argv'), _FILE_ONLY)
def test_a_file_reading_programs_parser_accepts_its_own_options(
    program: str, argv: list[str]
) -> None:
    """A guard on the refusal above, which a parser rejecting everything passes."""
    parse = _FILE_ONLY_PARSERS[program]
    assert parse([*argv, '--nav-results-root', 'somewhere']).nav_results_root == 'somewhere'


def test_the_option_resolves_to_the_url_it_names() -> None:
    """The command-line value is the first level of the ladder."""
    arguments = argparse.Namespace(results_db=_URL)
    assert get_results_db_url(arguments, DEFAULT_CONFIG) == _URL


def test_the_sentinel_overrides_an_exported_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--results-db none`` reads the files on a machine that exports a URL.

    Without the opt-out, exporting the variable would make a file-mode run
    impossible on that machine.
    """
    monkeypatch.setenv('NAV_RESULTS_DB', _URL)
    arguments = argparse.Namespace(results_db='none')
    assert get_results_db_url(arguments, DEFAULT_CONFIG) is None


def test_a_declaring_program_inherits_the_exported_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no option given, the exported URL is the level that answers."""
    monkeypatch.setenv('NAV_RESULTS_DB', _URL)
    arguments = argparse.Namespace(results_db=None)
    assert get_results_db_url(arguments, DEFAULT_CONFIG) == _URL


def test_a_url_containing_the_sentinel_is_still_a_url() -> None:
    """The sentinel is the exact string, not a substring of a path."""
    url = 'sqlite:///data/none/index.sqlite3'
    arguments = argparse.Namespace(results_db=url)
    assert get_results_db_url(arguments, DEFAULT_CONFIG) == url


def test_no_option_and_no_variable_means_no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading the files is every program's default and is not an error."""
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    arguments = argparse.Namespace(results_db=None)
    assert get_results_db_url(arguments, DEFAULT_CONFIG) is None
