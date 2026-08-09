"""Tests for the closure that keeps an operator's results index out of the suite.

A results index URL resolves from an argument, then from the
``environment.results_db`` configuration variable, then from ``NAV_RESULTS_DB``,
so a suite run on a machine that sets either ambient one opens a real index in
every test that names none -- which for SQLite means a write-lock probe against
a file an ingest may be holding.  Both are closed in ``tests/conftest.py``, and
what is pinned here is the half that is easy to lose: the closure has to be in
place before a fixture of a broader scope is built, not only before a test body
runs, since a module-scoped fixture that ingests a tree or builds a report is
exactly the kind that would open one.
"""

import os
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope='module')
def ambient_levels_at_module_scope() -> dict[str, Any]:
    """Record both ambient levels as a module-scoped fixture finds them.

    Module-scoped deliberately: pytest builds a fixture of this scope before
    any function-scoped fixture of the test that first asks for it, so this is
    the state a per-test closure cannot reach.

    Returns:
        The working directory and the environment variable, as they were when
        this fixture was built.
    """
    return {'directory': Path.cwd(), 'variable': os.environ.get('NAV_RESULTS_DB')}


def test_a_module_scoped_fixture_runs_where_no_configuration_names_an_index(
    ambient_levels_at_module_scope: dict[str, Any], directory_naming_no_index: Path
) -> None:
    """The configuration level is a file beside the process, so it is a directory.

    A fixture built in the directory the suite was started from resolves
    whatever ``nav_default_config.yaml`` is there, which on an operator's
    machine is the one naming the index they use.
    """
    assert ambient_levels_at_module_scope['directory'] == directory_naming_no_index


def test_a_module_scoped_fixture_finds_no_index_in_the_environment(
    ambient_levels_at_module_scope: dict[str, Any],
) -> None:
    """And the other level, which travels into every subprocess a test starts."""
    assert ambient_levels_at_module_scope['variable'] is None
