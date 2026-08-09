"""Shared fixtures for the ``tests/spindoctor`` subtree."""

import numpy as np
import pytest

from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution


@pytest.fixture
def sentinel_pointing() -> PointingSolution:
    """Build a PointingSolution a wiring test can identify by reference.

    Returns:
        A solution whose attitudes are identity matrices and whose baseline
        names the Cassini narrow angle camera.
    """
    baseline = AttitudeBaseline(
        cmatrix_original=np.eye(3),
        oops_from_spice=np.eye(3),
        camera_frame='CASSINI_ISS_NAC',
        camera_frame_id=-82360,
        ck_frame_id=-82000,
        start_et=1.0,
        stop_et=2.0,
        midtime_et=1.5,
        exposure_s=1.0,
        sclk_start='1/1.000',
        sclk_midtime='1/1.500',
        sclk_stop='1/2.000',
    )
    return PointingSolution(baseline=baseline, cmatrix=np.eye(3))


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
