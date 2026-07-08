"""Pytest configuration and shared fixtures."""

import pytest

from spindoctor.config import DEFAULT_CONFIG


@pytest.fixture(autouse=True)
def config_fixture() -> None:
    """Load bundled default config before each test if not already loaded."""
    DEFAULT_CONFIG.ensure_loaded()
