"""Tests for logging invariants that hold across the source rather than in one run.

Some of what this design guarantees is an absence, and an absence is not
observable by exercising the code: a logger that should not be there does
nothing until the day something logs to it. These read the source instead, so
a conversion cannot be quietly undone.

Reading the source rather than importing the module is what lets these cover
whole packages, so a file added to one of them tomorrow is covered without
being named here, and lets them cover the GUI packages without importing
PyQt6 to do it. The complement is in ``test_log_scope.py``, which imports a
handful of named modules and inspects their attributes: that catches a logger
bound at run time, which no amount of reading imports will find.
"""

import ast
from pathlib import Path

import pytest
from filecache import FCPath

_SRC = FCPath(Path(__file__).resolve().parents[3]) / 'src' / 'spindoctor'

# The statistics and GUI programs carry no logger and write to the terminal
# with print(): neither is a batch pipeline, and both are read as they run.
_PRINT_ONLY = [
    'cli/stats',
    'cli/sd_stats_ingest.py',
    'cli/sd_stats_report.py',
    'cli/sd_backplane_viewer.py',
    'cli/sd_create_simulated_image.py',
    'cli/sim_editor',
    'cli/sd_mosaic_display.py',
    'ui/mosaic_viewer',
]

_LOGGER_NAMES = frozenset({'IMAGE_LOGGER', 'MAIN_LOGGER'})

# Core packages route through NavBase.logger; the stdlib logging module is
# never imported there.
_NO_STDLIB_LOGGING = [
    'feature',
    'nav_model',
    'nav_orchestrator',
    'nav_technique',
    'support',
]

# The results index is a data-access layer with no voice of its own: what it did
# is reported by whichever program called it, in that program's log.  It is also
# built on a third-party library that logs through the stdlib module, so a logger
# here would be the one place those two ladders could be wired together.
_NO_LOGGER_AT_ALL = [
    'results_index',
]


def _python_files(relative: str) -> list[FCPath]:
    """Return the Python files a target names.

    Parameters:
        relative: Path under the package, either a module or a directory.

    Returns:
        The files to inspect.
    """
    target = _SRC / relative
    return sorted(target.rglob('*.py')) if target.is_dir() else [target]


def _imported_names(path: FCPath) -> set[str]:
    """Return every name a module imports.

    Parsed rather than matched textually, so a name in a docstring or a
    comment explaining why it is absent does not count as a use.

    Parameters:
        path: The module to read.

    Returns:
        The imported names, including module names.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
            if node.module is not None:
                names.update(_dotted_prefixes(node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.update(_dotted_prefixes(alias.name))
                # An aliased module hides the real name behind the alias, and
                # what it is later used for is an attribute access rather than
                # an import, so the alias is recorded as well.
                if alias.asname is not None:
                    names.add(alias.asname)
        elif isinstance(node, ast.Attribute):
            # "from spindoctor import config" then "config.MAIN_LOGGER" never
            # imports the logger by name; the attribute is where it surfaces.
            names.add(node.attr)
    return names


def _dotted_prefixes(dotted: str) -> set[str]:
    """Return a dotted module name and every package leading to it.

    ``import logging.handlers`` records only the full dotted name, so a check
    for ``logging`` would miss it.

    Parameters:
        dotted: A possibly dotted module name.

    Returns:
        The name and each of its prefixes.
    """
    parts = dotted.split('.')
    return {'.'.join(parts[: index + 1]) for index in range(len(parts))}


@pytest.mark.parametrize('relative', _PRINT_ONLY + _NO_STDLIB_LOGGING + _NO_LOGGER_AT_ALL)
def test_every_target_of_these_checks_exists(relative: str) -> None:
    """A target that has moved would make its check pass by finding nothing.

    Every assertion in this module is that a search came back empty, so a path
    that names nothing is indistinguishable from a path that is clean.
    """
    assert (_SRC / relative).exists()


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_logger(relative: str) -> None:
    """A program that reports through print() holds no logger to drift back to."""
    offenders = [
        f'{path.name}:{sorted(_LOGGER_NAMES & _imported_names(path))}'
        for path in _python_files(relative)
        if _LOGGER_NAMES & _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_pdslogger(relative: str) -> None:
    """Nor does it reach around the loggers to pdslogger itself."""
    offenders = [
        path.name for path in _python_files(relative) if 'pdslogger' in _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('relative', _PRINT_ONLY)
def test_a_print_only_program_imports_no_stdlib_logging(relative: str) -> None:
    """Nor the stdlib module, which is the third way back to having a logger.

    Checking only for the two loggers and for pdslogger would let
    ``logging.getLogger(__name__)`` reintroduce exactly what these programs
    were converted away from, configured by nothing this design controls.
    """
    offenders = [
        path.name for path in _python_files(relative) if 'logging' in _imported_names(path)
    ]
    assert offenders == []


@pytest.mark.parametrize('package', _NO_STDLIB_LOGGING)
def test_core_code_does_not_import_stdlib_logging(package: str) -> None:
    """Core navigation code logs through pdslogger, never the stdlib module.

    The two do not share a level ladder or a handler set, so a stdlib logger
    here would be configured by nothing this design controls -- and in a cloud
    task it would reach the worker's terminal by a path isolation never sees.
    """
    offenders = [path.name for path in _python_files(package) if 'logging' in _imported_names(path)]
    assert offenders == []


@pytest.mark.parametrize('package', _NO_LOGGER_AT_ALL)
def test_a_data_access_layer_holds_no_logger_of_any_kind(package: str) -> None:
    """Neither of the two loggers, nor pdslogger, nor the stdlib module.

    Reaching for any of the three would give a library layer a voice its caller
    did not configure, and the stdlib one would additionally turn on whatever the
    database library it sits over decided to say.

    Parameters:
        package: Path, relative to the source root, of the package to scan.
    """
    forbidden = _LOGGER_NAMES | {'logging', 'pdslogger'}
    offenders = [
        f'{path.name}:{sorted(found)}'
        for path in _python_files(package)
        if (found := forbidden & _imported_names(path))
    ]
    assert offenders == []


def test_the_loggers_are_the_only_two() -> None:
    """No third logger has been constructed anywhere in the package.

    Every record belongs to the run or to one image.  A logger outside those
    two is configured by none of the command line, the configuration, or the
    cloud-task isolation.
    """
    constructed = sorted(
        path.relative_to(_SRC).as_posix()
        for path in _SRC.rglob('*.py')
        if _constructs_a_logger(path)
    )
    assert constructed == ['config/log_scope.py', 'config/logger.py']


def _constructs_a_logger(path: FCPath) -> bool:
    """Whether a module calls the PdsLogger constructor.

    Matched in the AST rather than the text: a docstring mentioning the call
    is not one, and an aliased import (``import pdslogger as _pl``) is.

    Parameters:
        path: The module to read.

    Returns:
        True when the module constructs a logger.
    """
    tree = ast.parse(path.read_text())
    # ``import pdslogger as p`` then ``p.PdsLogger(...)``.
    module_aliases = {'pdslogger'} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == 'pdslogger'
    }
    # ``from pdslogger import PdsLogger as P`` then ``P(...)``: the name the
    # constructor is called by is whatever the import bound it to.
    constructor_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'pdslogger'
        for alias in node.names
        if alias.name == 'PdsLogger'
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in constructor_names:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == 'PdsLogger'
            and isinstance(func.value, ast.Name)
            and func.value.id in module_aliases
        ):
            return True
    return False
