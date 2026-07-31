"""Tests that the logging command-line surface is the same everywhere.

Someone who learns these flags for one program should not have to relearn
them for the next, so the check is over the whole set of programs rather than
one at a time: every program with a logger accepts the same four main flags,
every program that processes images individually accepts the three image
flags as well, and a program with no logger accepts none of them.

Each program's parser is driven the way the program itself builds it, by
running ``main`` with ``--help``, so what is asserted is the surface a user
actually meets rather than a reconstruction of it.
"""

import argparse
import contextlib
import importlib
import io
import sys
from pathlib import Path

import pytest
from filecache import FCPath

from spindoctor.cli.logging_args import add_logging_arguments

_MAIN_FLAGS = ('--log-root', '--log-main-to-console', '--log-main-to-file', '--log-level')
_IMAGE_FLAGS = ('--log-image-to-console', '--log-image-to-file', '--log-level-image')

# The sink flags, which argparse gives a --no- form; the level and root flags
# take a value and have none.
_NEGATABLE_MAIN_FLAGS = ('--log-main-to-console', '--log-main-to-file')
_NEGATABLE_IMAGE_FLAGS = ('--log-image-to-console', '--log-image-to-file')

_MAIN_DESTINATIONS = ('log_root', 'log_main_to_console', 'log_main_to_file', 'log_level')
_IMAGE_DESTINATIONS = ('log_image_to_console', 'log_image_to_file', 'log_level_image')

_CLI_DIR = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor' / 'cli'

# Programs with a logger, and the argv that reaches their parser.  A program
# reading its dataset or mode from argv before parsing needs it supplied.
_WITH_IMAGE_LOGGER = [
    ('sd_offset', ['coiss_saturn']),
    ('sd_backplanes', ['coiss_saturn']),
    ('sd_mosaic', ['rings', 'coiss_saturn']),
    ('sd_mosaic', ['body', 'coiss_saturn']),
]

_WITHOUT_IMAGE_LOGGER = [
    ('sd_consolidate_metadata', ['coiss_saturn']),
    ('sd_create_bundle', ['labels', 'coiss_saturn']),
    ('sd_create_bundle', ['summary', 'coiss_saturn']),
]

_WITH_ANY_LOGGER = _WITH_IMAGE_LOGGER + _WITHOUT_IMAGE_LOGGER

# Programs that carry no logger and write to the terminal with print().  These
# are checked by source rather than by running them: the GUI programs import
# PyQt6 at module scope, and importing it to prove a program has no logging
# flags is a poor trade.
_WITHOUT_LOGGER = [
    'sd_stats_ingest',
    'sd_stats_report',
    'sd_create_simulated_image',
    'sd_backplane_viewer',
    'sd_mosaic_display',
]

# The cloud-task drivers deliberately have no logging surface: every flag here
# configures a logger they are not allowed to have, or a console they must not
# write to.
_CLOUD_TASK_DRIVERS = [
    'sd_offset_cloud_tasks',
    'sd_backplanes_cloud_tasks',
    'sd_mosaic_cloud_tasks',
]


def _help_text(program: str, argv: list[str]) -> str:
    """Return what ``program --help`` prints.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``, for a program that reads its
            dataset or mode from argv before parsing.

    Returns:
        The help text.
    """
    module = importlib.import_module(f'spindoctor.cli.{program}')
    buffer = io.StringIO()
    saved = sys.argv
    sys.argv = [program, *argv, '--help']
    try:
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            module.main()
    finally:
        sys.argv = saved
    return buffer.getvalue()


def _source(program: str) -> str:
    """Return the source of a dispatch module.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The module's text.
    """
    return str((_CLI_DIR / f'{program}.py').read_text())


@pytest.mark.parametrize(('program', 'argv'), _WITH_ANY_LOGGER)
@pytest.mark.parametrize('flag', _MAIN_FLAGS)
def test_a_program_with_a_logger_accepts_the_main_flags(
    program: str, argv: list[str], flag: str
) -> None:
    """Every program that logs accepts the same main-logger flags."""
    assert flag in _help_text(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _WITH_IMAGE_LOGGER)
@pytest.mark.parametrize('flag', _IMAGE_FLAGS)
def test_a_program_that_processes_images_accepts_the_image_flags(
    program: str, argv: list[str], flag: str
) -> None:
    """A program with a per-image backend accepts the image-logger flags."""
    assert flag in _help_text(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _WITHOUT_IMAGE_LOGGER)
@pytest.mark.parametrize('flag', _IMAGE_FLAGS)
def test_a_program_without_images_rejects_the_image_flags(
    program: str, argv: list[str], flag: str
) -> None:
    """A program with no image backend does not offer the image flags.

    Offering them would leave someone believing they had changed something.
    """
    assert flag not in _help_text(program, argv)


@pytest.mark.parametrize('destination', [*_MAIN_DESTINATIONS, *_IMAGE_DESTINATIONS])
def test_a_logging_flag_defaults_to_unset(destination: str) -> None:
    """Naming no flag leaves every destination None, so the configuration decides.

    A flag defaulting to a concrete value would override the configuration
    just by existing, and there would be no way to ask on the command line for
    the behavior the configuration was set up to give.  Parsed rather than read
    out of the help text: a default that silently overrode the configuration
    would be spelled the same way in ``--help`` as one that did not.
    """
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    assert getattr(parser.parse_args([]), destination) is None


@pytest.mark.parametrize(('program', 'argv'), _WITH_ANY_LOGGER)
@pytest.mark.parametrize('flag', _NEGATABLE_MAIN_FLAGS)
def test_a_main_sink_can_be_turned_off(program: str, argv: list[str], flag: str) -> None:
    """Each main sink can be turned off as well as on, from the command line."""
    assert f'--no-{flag.removeprefix("--")}' in _help_text(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _WITH_IMAGE_LOGGER)
@pytest.mark.parametrize('flag', _NEGATABLE_IMAGE_FLAGS)
def test_an_image_sink_can_be_turned_off(program: str, argv: list[str], flag: str) -> None:
    """So can each image sink, on the programs that have one."""
    assert f'--no-{flag.removeprefix("--")}' in _help_text(program, argv)


@pytest.mark.parametrize('flag', [*_NEGATABLE_MAIN_FLAGS, *_NEGATABLE_IMAGE_FLAGS])
def test_turning_a_sink_off_is_distinguishable_from_saying_nothing(flag: str) -> None:
    """The negated form resolves to False rather than back to None.

    None means "the configuration decides", so a negation that produced it
    would silently ask for the default it was trying to override.
    """
    parser = argparse.ArgumentParser()
    add_logging_arguments(parser)
    destination = flag.removeprefix('--').replace('-', '_')
    assert getattr(parser.parse_args([f'--no-{flag.removeprefix("--")}']), destination) is False


@pytest.mark.parametrize('program', _WITHOUT_LOGGER)
def test_a_program_with_no_logger_has_no_logging_flags(program: str) -> None:
    """The statistics and GUI programs take none of the logging surface."""
    assert 'add_logging_arguments' not in _source(program)


@pytest.mark.parametrize('program', _CLOUD_TASK_DRIVERS)
def test_a_cloud_task_driver_has_no_logging_flags(program: str) -> None:
    """A worker's logging is not the operator's to configure per invocation.

    Every flag configures a main logger a cloud task must not have, or a
    console it must not write to.
    """
    assert 'add_logging_arguments' not in _source(program)
