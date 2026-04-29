"""Shared fixtures for the integration test layer (Part 0 §56).

A single ``pds3_holdings_dir`` fixture replaces every ad-hoc
``pytest.skip(...)`` so every integration test gates on the same
environment variable consistently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def pds3_holdings_dir() -> Path:
    """Return the PDS3 holdings root, or skip the test when unset."""
    p = os.environ.get('PDS3_HOLDINGS_DIR')
    if not p:
        pytest.skip('PDS3_HOLDINGS_DIR unset; integration tests skipped')
    return Path(p)
