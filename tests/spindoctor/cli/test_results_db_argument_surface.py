"""Which programs read the results index, and how they are told to.

A program becomes index-backed by declaring ``--results-db``, never by
inheriting an exported variable.  That is the whole policy, and it has two
halves: a declaring program honors the three-level ladder the option belongs to,
and a program that declares nothing resolves no URL at all however the machine
it runs on is configured.  The second half is what a scan enforces -- an
exported ``NAV_RESULTS_DB`` reaches every program on a machine, so a program
that quietly started resolving it would change behavior nobody asked to change.

A declaring program that also has a file-reading mode says in its own help that
``--results-db none`` is how to ask for it, so that a machine which exports the
URL can still be told to read files.  The programs that deliberately keep
reading files -- the bundle builder, the metadata consolidator, the backplane
viewer -- must not quietly stop, which they would if a resolved URL reached
whatever happened to be running.

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
from tests.spindoctor.cli.conftest import cloud_task_parser, help_text

from spindoctor.cli import (
    sd_backplanes,
    sd_consolidate_metadata,
    sd_create_bundle,
    sd_mosaic,
)
from spindoctor.config import DEFAULT_CONFIG, get_results_db_url

_SOURCE_ROOT = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

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

# Every program whose questions the index answers, named by the argv that
# reaches its parser.  Beyond the per-image row readers above: the navigator
# answers its image selection from one query, and the statistics programs read
# and write the index as their whole subject.
_CONSUMERS: list[tuple[str, list[str]]] = [
    *_INDEX_BACKED,
    ('sd_offset', ['coiss_saturn']),
    ('sd_stats_ingest', []),
    ('sd_stats_report', []),
]

# The consumers that also have a file-reading mode, which is what the sentinel
# opts back into.  The statistics programs read and write the index by
# construction and have no such mode, so naming the sentinel to them would
# advertise an answer they refuse.
_WITH_A_FILE_MODE: list[tuple[str, list[str]]] = [
    *_INDEX_BACKED,
    ('sd_offset', ['coiss_saturn']),
]

_INDEX_BACKED_CLOUD_TASK_DRIVERS = [
    'sd_backplanes_cloud_tasks',
    'sd_mosaic_cloud_tasks',
    'sd_stats_ingest_cloud_tasks',
]

# The programs that read a results index in any capacity, as the paths of the
# modules that resolve the URL, relative to the package root.  The statistics
# programs read it as their whole subject; the pipeline stages read one row per
# image; the navigator's dataset layer answers one selection query from it.  A
# module absent from this list must not resolve a URL: it would then pick one up
# from an exported NAV_RESULTS_DB without ever offering the option that opts
# back out.  The scan covers the whole package rather than the dispatch modules
# alone, because a resolution that moved into a library module would otherwise
# leave the scan looking at the wrong place.
_RESOLVING_MODULES = {
    'cli/sd_backplanes.py',
    'cli/sd_backplanes_cloud_tasks.py',
    'cli/sd_mosaic.py',
    'cli/sd_mosaic_cloud_tasks.py',
    'cli/sd_stats_ingest.py',
    'cli/sd_stats_ingest_cloud_tasks.py',
    'cli/stats/report.py',
    'dataset/dataset_pds3.py',
}

# The resolver's own home and the package that re-exports it are not
# resolutions, so they are excluded from the scan rather than listed above as
# though they were programs.
_RESOLVER_HOME = {'config/config_helper.py', 'config/__init__.py'}

# Programs deliberately left reading files, each with its own parser and the
# argv that reaches it.  They write the documents, consume a whole document
# rather than a few fields of it, or copy its bytes, so a column schema serves
# none of them.
_FILE_ONLY: list[tuple[str, list[str]]] = [
    ('sd_consolidate_metadata', ['coiss_saturn', '--dest-dir', 'out', '--copy-all']),
    ('sd_create_bundle', ['coiss_saturn']),
]

_FILE_ONLY_PARSERS: dict[str, Callable[[list[str]], argparse.Namespace]] = {
    'sd_consolidate_metadata': sd_consolidate_metadata.parse_args,
    'sd_create_bundle': sd_create_bundle.parse_args_labels,
}

# The backplane viewer opens the products of one image through a GUI and builds
# its parser inside its own main, so its surface is read from the help text it
# prints.  Its image selection reads the tree like the rest of them, which is
# only true while it names no index.
_FILE_ONLY_HELP: list[tuple[str, list[str]]] = [
    ('sd_create_bundle', ['labels', 'coiss_saturn']),
    ('sd_consolidate_metadata', ['coiss_saturn']),
    ('sd_backplane_viewer', ['coiss_saturn']),
]


def _modules_resolving_a_url() -> set[str]:
    """Return every package module that resolves a results index URL.

    Returns:
        Paths relative to the package root, using forward slashes, excluding
        the resolver's own home.
    """
    root = Path(str(_SOURCE_ROOT))
    found = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*.py')
        if 'get_results_db_url' in path.read_text(encoding='utf-8')
    }
    return found - _RESOLVER_HOME


def _one_line(program: str, argv: list[str]) -> str:
    """Return a program's help with its wrapping removed.

    argparse rewraps every help string to the terminal width, so a phrase is
    only reliably searchable once the line breaks are gone.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.

    Returns:
        The help text as one space-separated line.
    """
    return ' '.join(help_text(program, argv).split())


def test_only_the_declaring_programs_resolve_an_index_url() -> None:
    """A program that offers no option must not inherit one from the machine.

    Adding a program here means giving it ``--results-db`` and its ``none``
    sentinel in the same change; without them a run on a machine that exports
    the variable could not be told to read files.
    """
    assert _modules_resolving_a_url() == _RESOLVING_MODULES


def test_the_resolver_home_is_excluded_from_something_that_names_it() -> None:
    """A guard on the exclusion above, which an empty scan would also satisfy."""
    root = Path(str(_SOURCE_ROOT))
    named = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*.py')
        if 'get_results_db_url' in path.read_text(encoding='utf-8')
    }
    assert _RESOLVER_HOME <= named


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


@pytest.mark.parametrize(('program', 'argv'), _CONSUMERS)
def test_a_consuming_program_accepts_the_option(program: str, argv: list[str]) -> None:
    """The URL is named on the command line of every program that reads it."""
    assert '--results-db' in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _WITH_A_FILE_MODE)
def test_a_consuming_program_documents_the_opt_out(program: str, argv: list[str]) -> None:
    """The sentinel is the only opt-out visible to somebody reading --help.

    Spelled the same way by every program that has a file-reading mode, and
    asserted as the whole phrase: an operator who has to learn one program's
    wording to use the next has not been told the same thing twice.
    """
    assert '--results-db none' in _one_line(program, argv)


@pytest.mark.parametrize('program', _INDEX_BACKED_CLOUD_TASK_DRIVERS)
def test_an_index_backed_worker_offers_the_option(program: str) -> None:
    """So do the cloud-task workers, whose own command line is all they have."""
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


@pytest.mark.parametrize(('program', 'argv'), _FILE_ONLY_HELP)
def test_a_program_that_keeps_reading_files_does_not_offer_the_option(
    program: str, argv: list[str]
) -> None:
    """Declaring the option is what makes a program index-backed, so these do not."""
    assert '--results-db' not in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _FILE_ONLY_HELP)
def test_the_help_the_absence_is_read_from_is_the_program_s_own(
    program: str, argv: list[str]
) -> None:
    """The control for the assertion above, which is otherwise unfalsifiable.

    A program that reads its dataset from argv before parsing prints a usage
    error instead of its help when it is not given one, and every option is then
    absent from a string that names none of them.  Each of these programs writes
    to the navigation results root, so its help says so.
    """
    assert '--nav-results-root' in _one_line(program, argv)


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
