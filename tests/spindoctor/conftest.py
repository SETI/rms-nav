"""Shared fixtures for the ``tests/spindoctor`` subtree."""

import pytest


@pytest.fixture
def fakes_report_as_simulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report a module's fake observations as simulated.

    ``obs_class_to_inst_name`` cannot identify a test fake and returns
    ``'unknown'``, which the orchestrator treats as a build defect and warns
    about.  The fakes in the modules requesting this fixture stand in for an
    observation carrying no SPICE camera frame, which is exactly what a
    simulated image is, so they report that instead of shaping the production
    set around the test suite.

    Deliberately not autouse: each module whose fakes reach the orchestrator
    opts in with its own one-line autouse wrapper, so the patch never touches
    tests that exercise the real instrument registry.
    """
    monkeypatch.setattr(
        'spindoctor.nav_orchestrator.orchestrator.obs_class_to_inst_name', lambda cls: 'sim'
    )
