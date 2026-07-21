"""Unit tests for the per-instrument twist aggregation."""

from __future__ import annotations

import numpy as np
import pytest
from util.fov_distortion.aggregate import (
    recommend_rotation_fitting,
    twist_consistency,
)

# A representative field-corner radius (half the diagonal of a 1024 image).
RHO_REF = 724.0


def test_tight_twists_are_flagged_consistent() -> None:
    # A common small twist: the corner scatter is far below threshold even
    # though the formal per-frame sigmas are tiny.
    twists = np.array([0.10, 0.12, 0.09, 0.11, 0.10])
    sigmas = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    result = twist_consistency(twists, sigmas, RHO_REF)
    assert result.consistent
    assert result.weighted_mean_deg == pytest.approx(0.104, abs=0.02)
    assert result.scatter_corner_px < 0.15


def test_scattered_twists_are_flagged_inconsistent() -> None:
    # Corner scatter is large: genuine per-frame variation.
    twists = np.array([0.5, -0.3, 0.9, -0.6, 0.2])
    sigmas = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    result = twist_consistency(twists, sigmas, RHO_REF)
    assert not result.consistent
    assert result.scatter_corner_px > 0.15


def test_precise_but_operationally_tiny_scatter_stays_consistent() -> None:
    # Chi-square explodes (tiny sigmas, 0.002 deg scatter) but the corner
    # displacement is ~0.03 px: operationally one common twist.
    twists = np.array([0.010, 0.012, 0.009, 0.013, 0.011])
    sigmas = np.array([0.0005, 0.0005, 0.0005, 0.0005, 0.0005])
    result = twist_consistency(twists, sigmas, RHO_REF)
    assert result.reduced_chi_square > 2.0
    assert result.consistent


def test_corner_conversion_matches_small_angle() -> None:
    twists = np.array([0.1, 0.1])
    sigmas = np.array([0.01, 0.01])
    result = twist_consistency(twists, sigmas, RHO_REF)
    assert result.mean_corner_px == pytest.approx(np.radians(0.1) * RHO_REF, rel=1e-3)


def test_weighted_mean_favors_low_sigma_frame() -> None:
    twists = np.array([0.0, 1.0])
    sigmas = np.array([0.01, 1.0])
    result = twist_consistency(twists, sigmas, RHO_REF)
    assert result.weighted_mean_deg < 0.1


def test_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match='strictly positive'):
        twist_consistency(np.array([0.1, 0.2]), np.array([0.0, 0.1]), RHO_REF)


def test_rejects_empty() -> None:
    with pytest.raises(ValueError, match='at least one frame'):
        twist_consistency(np.array([]), np.array([]), RHO_REF)


def test_rejects_nonpositive_rho_ref() -> None:
    with pytest.raises(ValueError, match='rho_ref_px must be positive'):
        twist_consistency(np.array([0.1, 0.2]), np.array([0.1, 0.1]), 0.0)


def test_recommend_fit_rotation_when_inconsistent() -> None:
    twists = np.array([0.5, -0.3, 0.9, -0.6, 0.2])
    sigmas = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
    rec = recommend_rotation_fitting(twist_consistency(twists, sigmas, RHO_REF))
    assert rec.fit_camera_rotation
    assert rec.kernel_twist_correction_deg is None
    assert 'per frame' in rec.rationale


def test_recommend_kernel_correction_for_consistent_significant_twist() -> None:
    # Consistent twist whose corner displacement (0.2 deg -> ~2.5 px) matters.
    twists = np.array([0.20, 0.21, 0.19, 0.20, 0.205])
    sigmas = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    rec = recommend_rotation_fitting(twist_consistency(twists, sigmas, RHO_REF))
    assert not rec.fit_camera_rotation
    assert rec.kernel_twist_correction_deg == pytest.approx(0.2, abs=0.02)
    assert 'kernel' in rec.rationale


def test_recommend_nothing_for_consistent_negligible_twist() -> None:
    twists = np.array([0.001, -0.002, 0.0, 0.001, -0.001])
    sigmas = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    rec = recommend_rotation_fitting(twist_consistency(twists, sigmas, RHO_REF))
    assert not rec.fit_camera_rotation
    assert rec.kernel_twist_correction_deg is None
    assert 'negligible' in rec.rationale
