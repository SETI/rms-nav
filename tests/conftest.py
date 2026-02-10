"""Pytest configuration and shared fixtures."""

from collections.abc import Generator

import pytest
from nav.config import DEFAULT_CONFIG


def ensure_config_loaded() -> None:
    """Ensure DEFAULT_CONFIG is loaded.

    This function can be called to ensure the configuration is loaded
    without causing side effects during test collection.
    """
    # Only load config if it hasn't been loaded yet to avoid side effects
    if not DEFAULT_CONFIG._config_dict:
        DEFAULT_CONFIG.read_config()


@pytest.fixture(autouse=True)
def config_fixture() -> Generator[None, None, None]:
    """Ensure DEFAULT_CONFIG is loaded before each test.

    This fixture automatically runs before each test to ensure the configuration
    is loaded without causing side effects during test collection.
    """
    ensure_config_loaded()
    yield
    # Clean up after test if needed
    # Note: We don't reset the config as it might be used by other tests
