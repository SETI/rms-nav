"""Pytest configuration and shared fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pdslogger
import pytest

from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    set_log_levels,
    set_strict_scope,
    strict_scope_override,
)
from spindoctor.config.log_scope import _reset_reported_call_sites


@pytest.fixture(autouse=True)
def config_fixture() -> None:
    """Load bundled default config before each test if not already loaded."""
    DEFAULT_CONFIG.ensure_loaded()


@pytest.fixture(scope='session')
def directory_naming_no_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a directory holding no user configuration file.

    One directory for the whole session rather than one per test: nothing is
    ever written into it, and what makes it useful is what it does not hold.

    Parameters:
        tmp_path_factory: Factory the directory is made under.

    Returns:
        The directory.
    """
    return tmp_path_factory.mktemp('naming_no_index')


@pytest.fixture(autouse=True)
def no_ambient_results_index(
    monkeypatch: pytest.MonkeyPatch, directory_naming_no_index: Path
) -> None:
    """Close both ways a test could reach a results index nobody named.

    A results index URL is resolved from three places in order: the argument,
    the ``environment.results_db`` configuration variable, and the
    ``NAV_RESULTS_DB`` environment variable.  A test that names none of them is
    testing what a program does with no index, and on a machine that sets either
    ambient one it instead opens a real one -- for SQLite a write-lock probe
    against a file an ingest may be holding, and for a report a read of every
    row in it.  Both are closed here rather than in each test, because the level
    an author forgets is the level nothing then tests.

    The configuration level is closed by moving the working directory.  The user
    override file is ``nav_default_config.yaml`` beside the process, so a
    directory holding none is a configuration naming no index, whatever the
    directory the suite was started from holds.

    A test that wants either level sets it up for itself: what it does through
    the same fixture is undone before what is done here.

    Parameters:
        monkeypatch: Fixture the working directory and the environment are moved
            through.
        directory_naming_no_index: The working directory to run under.
    """
    monkeypatch.chdir(directory_naming_no_index)
    monkeypatch.delenv('NAV_RESULTS_DB', raising=False)
    # A configuration merged before this test ran -- by a test that named its own
    # override file, or by one that ran before the working directory moved --
    # outlives it: the merge is into a process-global configuration and nothing
    # takes it back out.
    monkeypatch.delitem(DEFAULT_CONFIG.environment, 'results_db', raising=False)


@pytest.fixture(autouse=True)
def restore_loggers_fixture() -> Iterator[None]:
    """Put both loggers back as they were found.

    Configuring a logger changes process state, and a cloud task deliberately
    reconfigures both of them for good -- a worker never un-isolates itself.
    Without this, one test resolving a cloud task's logging would silence every
    later test in the same worker, and the failures would land far from the
    cause.

    The restore is checked rather than assumed, and a test it could not undo
    fails here.  A logger holding an unexpected handler stops falling back to
    printing, so every later test in the same worker that reads its output
    through ``capsys`` sees nothing -- a failure with no visible connection to
    the test that caused it, appearing only in the worker that happened to run
    them in that order.  Failing at the source names the culprit instead.
    """
    main_handlers = list(MAIN_LOGGER.handlers)
    image_handlers = list(IMAGE_LOGGER.handlers)
    main_propagate = MAIN_LOGGER.propagate
    image_propagate = IMAGE_LOGGER.propagate
    main_level = MAIN_LOGGER.level
    image_level = IMAGE_LOGGER.level
    strict_override = strict_scope_override()
    yield
    left_behind: list[str] = []
    for name, logger, baseline in (
        ('main', MAIN_LOGGER, main_handlers),
        ('image', IMAGE_LOGGER, image_handlers),
    ):
        # remove_all_handlers only detaches, so a handler the test attached
        # would keep its log file open for the rest of the session.  Only what
        # the test added is closed; the baseline is put back as it was, and
        # NULL_HANDLER is a process-wide singleton nobody here owns.
        for handler in logger.handlers:
            if handler not in baseline and handler is not pdslogger.NULL_HANDLER:
                handler.close()
        logger.remove_all_handlers()
        for handler in baseline:
            logger.add_handler(handler)
        # NULL_HANDLER is excluded for the same reason the close loop above
        # excludes it: pdslogger can reinstate the process-wide singleton on
        # its own, and nobody here owns it, so finding it attached is not a
        # test leaving state behind.
        left_behind += [
            f'{name} logger: {handler!r}'
            for handler in logger.handlers
            if handler not in baseline and handler is not pdslogger.NULL_HANDLER
        ]
    MAIN_LOGGER.propagate = main_propagate
    IMAGE_LOGGER.propagate = image_propagate
    # Restored to what was found rather than to a level named here: a test that
    # sets one and puts back what it assumed the default was pins that
    # assumption on every test after it.
    MAIN_LOGGER.set_level(main_level)
    IMAGE_LOGGER.set_level(image_level)
    # The override, not the resolved value: saving the resolved boolean would
    # pin it and lose the deferral to the configuration.
    set_strict_scope(strict_override)
    # A test that logs out of scope otherwise leaves its call site in the
    # process-wide dedup set, so a later test asserting on that warning sees
    # nothing and fails somewhere unrelated to the cause.
    _reset_reported_call_sites()
    if len(left_behind) > 0:
        pytest.fail(
            'This test left a handler attached that could not be detached: '
            + '; '.join(left_behind)
            + '. One known cause: pdslogger identifies an open log file by the absolute '
            'path the working directory gives, so a handler built from a relative path '
            'cannot be found again once the working directory moves; build log handlers '
            'from an absolute path. A handler that was closed and then re-attached '
            'produces the same symptom.'
        )


@pytest.fixture(autouse=True)
def reset_log_levels_fixture() -> Iterator[None]:
    """Discard any resolved levels a test installs.

    The resolved set is process state, memoized on first use, so without this
    one test's levels would govern every later test in the same worker.
    """
    yield
    set_log_levels(None)


@pytest.fixture
def strict_log_scope() -> Iterator[None]:
    """Make an out-of-scope image log raise for the duration of a test.

    Opt-in rather than automatic.  A unit test exercising a model or technique
    in isolation calls it outside any image scope by design, which is correct
    practice and not the mis-binding this switch exists to catch, so enabling
    it for the whole suite would fail hundreds of legitimate tests.  Request it
    from a test that drives a real pipeline, where a scope genuinely should be
    open.

    Clears the override on exit rather than forcing it off, so behavior returns
    to whatever the configuration says.
    """
    set_strict_scope(True)
    yield
    set_strict_scope(None)
