"""Pytest configuration and shared fixtures."""

from collections.abc import Iterator

import pytest

from spindoctor.config import DEFAULT_CONFIG, set_strict_scope


@pytest.fixture(autouse=True)
def config_fixture() -> None:
    """Load bundled default config before each test if not already loaded."""
    DEFAULT_CONFIG.ensure_loaded()


@pytest.fixture
def strict_log_scope() -> Iterator[None]:
    """Make an out-of-scope image log raise for the duration of a test.

    Opt-in rather than automatic.  A unit test exercising a model or technique
    in isolation calls it outside any image scope by design, which is correct
    practice and not the mis-binding this switch exists to catch, so enabling
    it for the whole suite would fail hundreds of legitimate tests.  Request it
    from a test that drives a real pipeline, where a scope genuinely should be
    open.
    """
    set_strict_scope(True)
    yield
    set_strict_scope(False)
