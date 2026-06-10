"""End-to-end tests for ``BodyBlobNav``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.nav.nav_technique.conftest import (
    DiscImageFactory,
    NavContextFactory,
)

from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import BodyBlobFlags
from nav.feature.geometry import BodyBlobGeometry
from nav.nav_technique.diagnostics import BodyBlobDiagnostics
from nav.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE
from nav.nav_technique.nav_technique_body_blob import BodyBlobNav, _joint_covariance
from nav.support.filters import NavFilterKind, NavFilterSpec


def _make_blob_feature(
    body_name: str,
    *,
    predicted_center_vu: tuple[float, float],
    predicted_diameter_px: float,
    bbox_pad: int = 4,
    phase_angle_deg: float = 0.0,
    phase_irregularity_factor: float = 0.0,
) -> NavFeature:
    """Build a BODY_BLOB feature whose bbox tightly bounds the predicted disc."""
    radius = predicted_diameter_px / 2.0
    v_min = int(np.floor(predicted_center_vu[0] - radius - bbox_pad))
    u_min = int(np.floor(predicted_center_vu[1] - radius - bbox_pad))
    v_max = int(np.ceil(predicted_center_vu[0] + radius + bbox_pad))
    u_max = int(np.ceil(predicted_center_vu[1] + radius + bbox_pad))
    sigma_centroid = max(predicted_diameter_px / 6.0, 0.5)
    cov = (sigma_centroid * sigma_centroid) * np.eye(2, dtype=np.float64)
    return NavFeature(
        feature_id=f'body_blob:{body_name}',
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body',
        geometry=BodyBlobGeometry(
            predicted_center_vu=predicted_center_vu,
            bbox_extfov_vu=(v_min, u_min, v_max, u_max),
            predicted_diameter_px=predicted_diameter_px,
        ),
        subject_range_km=5.0e5,
        position_cov_px=cov,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.4,
        reliability_reasons=NavReliabilityBreakdown(
            blob_snr=0.5,
            blob_extent_px=predicted_diameter_px / 30.0,
        ),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(
            body_name=body_name,
            predicted_diameter_px=predicted_diameter_px,
            phase_angle_deg=phase_angle_deg,
            phase_irregularity_factor=phase_irregularity_factor,
        ),
    )


def test_body_blob_recovers_planted_offset_single_blob(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A single bright disc + blob feature predicting an offset center recovers offset."""
    shape = (200, 200)
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    planted_dv, planted_du = 2.0, -3.0
    pred_center = (actual_center[0] - planted_dv, actual_center[1] - planted_du)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(planted_dv, abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted_du, abs=0.5)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 1
    # Hard cap at 0.4 — the technique cannot dominate the ensemble
    # even with perfect inputs.
    assert result.confidence <= 0.4 + 1e-12


def test_body_blob_multi_body_least_squares_average(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Two blob features with the same planted offset average to that offset."""
    shape = (220, 220)
    radius = 7.0
    actual_centers = [(60.0, 70.0), (150.0, 140.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in actual_centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=(c[0] - planted[0], c[1] - planted[1]),
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(actual_centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 2


def test_body_blob_returns_zero_confidence_when_image_blank(
    make_nav_context: NavContextFactory,
) -> None:
    """A blob feature whose predicted bbox lies in a blank image is dropped."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=(100.0, 100.0),
        predicted_diameter_px=20.0,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.blob_count == 0


def test_body_blob_infeasible_on_empty_input() -> None:
    """``is_feasible([])`` reports infeasibility with a no-features reason."""
    technique = BodyBlobNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_body_blob_features' in report.reason


def test_body_blob_infeasible_on_zero_diameter() -> None:
    """A BODY_BLOB feature with ``predicted_diameter_px == 0`` is infeasible.

    Asserts the reason names the predicted-diameter requirement so a
    regression that returns a generic infeasibility message is caught.
    """
    feature = NavFeature(
        feature_id='body_blob:zero',
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body',
        geometry=BodyBlobGeometry(
            predicted_center_vu=(50.0, 50.0),
            bbox_extfov_vu=(40, 40, 60, 60),
            predicted_diameter_px=0.0,
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.1,
        reliability_reasons=NavReliabilityBreakdown(blob_snr=0.0),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(body_name='zero', predicted_diameter_px=0.0),
    )
    technique = BodyBlobNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False
    assert 'predicted_diameter' in report.reason


def test_body_blob_confidence_capped_at_0_4(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Even an ideal multi-blob fit cannot exceed 0.4 confidence."""
    shape = (240, 240)
    radius = 12.0
    centers = [(60.0, 60.0), (60.0, 180.0), (180.0, 60.0), (180.0, 180.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=c,
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.confidence == pytest.approx(0.4, abs=1e-12)


def test_body_blob_registered_with_navtechnique_registry() -> None:
    """``BodyBlobNav`` is auto-registered in ``NavTechnique._registry`` on import."""
    from nav.nav_technique.nav_technique import NavTechnique

    assert BodyBlobNav in NavTechnique._registry


def test_body_blob_marks_at_edge_when_centroid_hits_window(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A converged offset within ``at_edge_tolerance_px`` of the search-window
    edge is flagged ``at_edge=True`` and forced to zero confidence by the
    ``hard_zero_if`` gate.
    """
    shape = (200, 200)
    margin_v = 6
    margin_u = 6
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    # Plant the predicted center exactly ``margin_v`` rows above the
    # actual center so the recovered offset lands on the search-window
    # boundary.
    pred_center = (actual_center[0] - float(margin_v), actual_center[1])
    feature = _make_blob_feature(
        'edge_moon',
        predicted_center_vu=pred_center,
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image, extfov_margin_vu=(margin_v, margin_u))
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_body_blob_diagnostics_records_residual_and_snr(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Multi-blob agreement reports sub-pixel residual scatter and high SNR.

    The two blobs share the same planted offset so per-blob residuals
    around the joint mean must be sub-pixel; the bright synthetic discs
    push mean SNR well above 5.
    """
    shape = (220, 220)
    radius = 7.0
    actual_centers = [(60.0, 70.0), (150.0, 140.0)]
    image = np.zeros(shape, dtype=np.float64)
    for c in actual_centers:
        image += disc_image(shape, c, radius)
    image = np.clip(image, 0.0, 100.0)
    planted = (1.0, 1.5)
    features = [
        _make_blob_feature(
            f'moon_{i}',
            predicted_center_vu=(c[0] - planted[0], c[1] - planted[1]),
            predicted_diameter_px=2.0 * radius,
        )
        for i, c in enumerate(actual_centers)
    ]
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.residual_px < 0.5
    assert result.diagnostics.body_snr_inside_predicted_bbox > 5.0


def test_body_blob_3dof_rotation_unobservable(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """When ``fit_camera_rotation=True`` the blob technique reports a 3x3 covariance.

    The rotation slot carries the ``ROTATION_UNOBSERVABLE_VARIANCE``
    sentinel because a centroid is rotation-invariant about itself; the
    ensemble combine treats this as no-information in the rotation
    direction.
    """
    shape = (200, 200)
    actual_center = (100.0, 100.0)
    radius = 8.0
    image = disc_image(shape, actual_center, radius)
    feature = _make_blob_feature(
        'moonA',
        predicted_center_vu=(98.0, 102.0),
        predicted_diameter_px=2.0 * radius,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image, fit_camera_rotation=True)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad == pytest.approx(0.0)
    assert result.sigma_rotation_rad == pytest.approx(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE))
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)


def test_body_blob_diagnostics_records_max_phase_irregularity(
    disc_image: DiscImageFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The technique surfaces the worst per-blob irregularity factor.

    Two blobs in the same scene with different irregularity factors:
    the diagnostics record the maximum so the confidence-formula
    penalty is driven by the worst blob (regular bodies have factor ~
    0.005, irregular ~ 0.05+).
    """
    shape = (200, 200)
    image = np.maximum(
        disc_image(shape, (60.0, 60.0), 8.0),
        disc_image(shape, (140.0, 140.0), 8.0),
    )
    low_phase_blob = _make_blob_feature(
        'regular_moon',
        predicted_center_vu=(60.0, 60.0),
        predicted_diameter_px=16.0,
        phase_angle_deg=15.0,
        phase_irregularity_factor=0.003,
    )
    high_phase_blob = _make_blob_feature(
        'irregular_moon',
        predicted_center_vu=(140.0, 140.0),
        predicted_diameter_px=16.0,
        phase_angle_deg=130.0,
        phase_irregularity_factor=0.075,
    )
    technique = BodyBlobNav()
    context = make_nav_context(image)
    result = technique.navigate([low_phase_blob, high_phase_blob], context)
    assert isinstance(result.diagnostics, BodyBlobDiagnostics)
    assert result.diagnostics.max_phase_angle_deg == pytest.approx(130.0)
    assert result.diagnostics.max_phase_irregularity_factor == pytest.approx(0.075)


def test_body_blob_flags_reject_phase_outside_valid_range() -> None:
    """``BodyBlobFlags`` validates ``phase_angle_deg`` is in [0, 180]."""
    with pytest.raises(ValueError, match='phase_angle_deg'):
        BodyBlobFlags(body_name='X', predicted_diameter_px=10.0, phase_angle_deg=-1.0)
    with pytest.raises(ValueError, match='phase_angle_deg'):
        BodyBlobFlags(body_name='X', predicted_diameter_px=10.0, phase_angle_deg=181.0)


def test_body_blob_flags_reject_negative_phase_irregularity() -> None:
    """``BodyBlobFlags`` validates ``phase_irregularity_factor`` is non-negative."""
    with pytest.raises(ValueError, match='phase_irregularity_factor'):
        BodyBlobFlags(
            body_name='X',
            predicted_diameter_px=10.0,
            phase_irregularity_factor=-0.01,
        )


def test_joint_covariance_two_point_reduced_chi_square() -> None:
    """Two-blob covariance matches the analytic reduced-chi-square weighted mean.

    Derivation (per-axis: var = chi2_nu / sum(w), with
    chi2_nu = sum(w * r^2) / max(N - p, 1), p = 2):

    w = [1, 3], sum(w) = 4, N = 2, dof = max(2 - 2, 1) = 1.

    V axis: offsets = [2, 4], mean = (1*2 + 3*4)/4 = 3.5,
            r = [-1.5, 0.5], sum(w r^2) = 1*2.25 + 3*0.25 = 3.0,
            chi2_nu = 3.0/1 = 3.0, var = 3.0/4 = 0.75 (> floor 1/4 = 0.25).
    U axis: offsets = [-1, 1], mean = (1*-1 + 3*1)/4 = 0.5,
            r = [-1.5, 0.5], sum(w r^2) = 3.0, var = 0.75 (identical).
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.5
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 0.5
    cov = _joint_covariance(offsets_v=offsets_v, offsets_u=offsets_u, weights=weights, dv=dv, du=du)
    assert cov[0, 0] == pytest.approx(0.75, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.75, abs=1e-9)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert cov[1, 0] == pytest.approx(0.0, abs=1e-9)


def test_joint_covariance_n_point_reduced_chi_square() -> None:
    """Four-blob covariance matches the analytic reduced-chi-square weighted mean.

    w = [1, 1, 1, 1], sum(w) = 4, N = 4, dof = max(4 - 2, 1) = 2.

    V axis: offsets = [0, 2, 4, 6], mean = 12/4 = 3.0,
            r = [-3, -1, 1, 3], sum(w r^2) = 9 + 1 + 1 + 9 = 20,
            chi2_nu = 20/2 = 10, var = 10/4 = 2.5 (> floor 0.25).
    U axis: offsets = [1, 1, 1, 1], mean = 1.0, r = 0 everywhere,
            sum(w r^2) = 0, chi2_nu = 0, candidate var = 0 -> floored at
            1/sum(w) = 1/4 = 0.25.
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    offsets_v = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    offsets_u = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    dv = float(np.sum(weights * offsets_v) / weights.sum())  # 3.0
    du = float(np.sum(weights * offsets_u) / weights.sum())  # 1.0
    cov = _joint_covariance(offsets_v=offsets_v, offsets_u=offsets_u, weights=weights, dv=dv, du=du)
    assert cov[0, 0] == pytest.approx(2.5, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)


def test_joint_covariance_single_blob_is_large_not_overconfident() -> None:
    """NAV-005: a single blob yields the (large) inverse-precision floor.

    One point cannot constrain two translation parameters, so the
    covariance must be large rather than a tiny over-confident value.
    With a single small-weight blob (w = 0.01) the per-axis variance is
    the pure inverse precision 1/sum(w) = 1/0.01 = 100.0 -- there is no
    residual scatter (the lone residual is zero by construction).  The
    old ``sum(w r^2)/(sum w)^2`` form would have reported 0 here (no
    floor at the correct power), which the over-confidence fix corrects.
    """
    weights = np.array([0.01], dtype=np.float64)
    offsets_v = np.array([3.0], dtype=np.float64)
    offsets_u = np.array([-4.0], dtype=np.float64)
    cov = _joint_covariance(
        offsets_v=offsets_v, offsets_u=offsets_u, weights=weights, dv=3.0, du=-4.0
    )
    # 1 / sum(w) = 1 / 0.01 = 100.0 on each axis.
    assert cov[0, 0] == pytest.approx(100.0, abs=1e-9)
    assert cov[1, 1] == pytest.approx(100.0, abs=1e-9)
    # Far larger than a well-determined multi-blob fit (~O(1)).
    assert cov[0, 0] > 1.0


def test_joint_covariance_model_error_floor_inflates_diagonal_by_square() -> None:
    """ORCH-001: model_error_floor_px>0 adds exactly its square to the diagonal.

    Reuses the two-point fixture (var = 0.75 per axis with no floor).
    With model_error_floor_px = 2.0 each diagonal grows by exactly
    2.0**2 = 4.0 -> 0.75 + 4.0 = 4.75; the off-diagonal stays zero.
    """
    weights = np.array([1.0, 3.0], dtype=np.float64)
    offsets_v = np.array([2.0, 4.0], dtype=np.float64)
    offsets_u = np.array([-1.0, 1.0], dtype=np.float64)
    dv = 3.5
    du = 0.5
    base = _joint_covariance(
        offsets_v=offsets_v, offsets_u=offsets_u, weights=weights, dv=dv, du=du
    )
    floored = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
        model_error_floor_px=2.0,
    )
    assert floored[0, 0] == pytest.approx(4.75, abs=1e-9)
    assert floored[1, 1] == pytest.approx(4.75, abs=1e-9)
    assert floored[0, 0] - base[0, 0] == pytest.approx(4.0, abs=1e-9)
    assert floored[1, 1] - base[1, 1] == pytest.approx(4.0, abs=1e-9)
    assert floored[0, 1] == pytest.approx(0.0, abs=1e-9)
