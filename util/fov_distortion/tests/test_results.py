"""Unit tests for per-instrument aggregation over synthetic frames."""

from __future__ import annotations

import pytest
from util.fov_distortion.measure import FrameMeasurement
from util.fov_distortion.results import summarize_instrument
from util.fov_distortion.tests._synthetic import make_frame


def test_summary_flags_common_twist_consistent() -> None:
    frames = [make_frame(twist_deg=0.05, k1=0.002, seed=i, name=f'F{i}') for i in range(8)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    assert summary.n_frames_ok == 8
    assert summary.consistency is not None
    assert summary.consistency.consistent
    assert summary.recommendation is not None
    assert not summary.recommendation.fit_camera_rotation


def test_summary_flags_scattered_twist_inconsistent() -> None:
    twists = [0.5, -0.4, 0.8, -0.7, 0.3, -0.5, 0.6, -0.3]
    frames = [make_frame(twist_deg=t, k1=0.002, seed=i, name=f'F{i}') for i, t in enumerate(twists)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    assert summary.consistency is not None
    assert not summary.consistency.consistent
    assert summary.recommendation is not None
    assert summary.recommendation.fit_camera_rotation


def test_summary_recovers_aggregate_radial() -> None:
    frames = [make_frame(twist_deg=0.05, k1=0.01, seed=i, name=f'F{i}') for i in range(6)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    assert summary.pooled_radial is not None
    assert summary.pooled_radial.model.k_sim[0] == pytest.approx(0.01, abs=2e-3)


def test_summary_handles_no_ok_frames() -> None:
    failed = FrameMeasurement(
        image_name='X',
        url='mem://x',
        inst_id='syn',
        image_shape=(1024, 1024),
        offset_vu=None,
        center_vu=(511.5, 511.5),
        rho_ref_px=724.0,
        status='nav_failed',
    )
    summary = summarize_instrument('syn', 'Synthetic', [failed])
    assert summary.n_frames_ok == 0
    assert summary.consistency is None
    assert summary.recommendation is None
    assert summary.pooled_radial is None
