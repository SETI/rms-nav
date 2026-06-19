"""End-to-end tests for ``StarUniqueMatchNav``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.nav.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from nav.nav_technique.diagnostics import StarUniqueMatchDiagnostics
from nav.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
)
from nav.nav_technique.nav_technique_star_unique_match import StarUniqueMatchNav


def test_star_unique_match_one_star_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A single uniquely-bright star recovers the planted translation."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (100.0, 100.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    planted = (2.0, -3.0)
    pred_vu = (actual_vu[0] - planted[0], actual_vu[1] - planted[1])
    feature = make_star_feature('star:UCAC4:1', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    feas = technique.is_feasible([feature])
    assert feas.feasible is True
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'one_star'
    assert result.confidence <= 0.7 + 1e-12


def test_star_unique_match_two_star_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The 2-star path picks the smaller-residual assignment."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=180.0, sigma=1.2)
    planted = (1.0, 1.5)
    pred_a = (actual_a[0] - planted[0], actual_a[1] - planted[1])
    pred_b = (actual_b[0] - planted[0], actual_b[1] - planted[1])
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.4)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.4)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'two_star'
    # No third predictable star, so the brightness margin diagnostic is +inf.
    assert result.diagnostics.brightness_margin_mag == float('inf')
    assert result.confidence <= 0.8 + 1e-12


def test_star_unique_match_one_star_fails_when_brightness_margin_too_small(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two near-equal-brightness stars fail the 1-star uniqueness test.

    With two stars whose predicted SNRs are close and the 2-star path
    fails (e.g. wrong-position prediction such that one detection misses
    its window entirely), the technique reports a spurious result with
    a brightness-margin reason rather than picking the wrong star.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual = (100.0, 100.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    planted = (1.0, 1.0)
    pred_a = (actual[0] - planted[0], actual[1] - planted[1])
    # Second prediction in a region with no signal so its detection
    # fails — forcing the one-star fallback.
    pred_b = (10.0, 10.0)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=20.0)
    feat_b = make_star_feature(
        'star:UCAC4:B',
        predicted_vu=pred_b,
        predicted_snr=18.0,  # too close to A — no margin
    )
    technique = StarUniqueMatchNav()
    context = make_nav_context(image)
    result = technique.navigate([feat_a, feat_b], context)
    # Cannot trigger the 2-star path because pred_b has no detection.
    # The 1-star fallback rejects because brightness margin to feat_b
    # is below the 1.5 mag floor.
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)


def test_star_unique_match_infeasible_on_empty_input() -> None:
    """``is_feasible([])`` reports infeasibility."""
    technique = StarUniqueMatchNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_usable_star_features' in report.reason


def test_star_unique_match_skips_occluded_stars(
    make_star_feature: NavFeatureFactory,
) -> None:
    """STAR features marked in_body_silhouette are filtered out."""
    feature = make_star_feature(
        'star:UCAC4:occluded',
        predicted_vu=(100.0, 100.0),
        predicted_snr=30.0,
        in_body_silhouette=True,
    )
    technique = StarUniqueMatchNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False
    assert 'no_usable_star_features' in report.reason


def test_star_unique_match_registered_with_navtechnique_registry() -> None:
    """``StarUniqueMatchNav`` self-registers on import."""
    assert StarUniqueMatchNav in NavTechnique._registry


def test_star_unique_match_marks_at_edge_when_offset_hits_margin(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """An offset that hits the search-window edge is flagged + zero confidence."""
    shape = (220, 220)
    image = np.zeros(shape, dtype=np.float64)
    actual_vu = (110.0, 110.0)
    draw_gaussian_star(image, actual_vu, peak_dn=200.0, sigma=1.2)
    margin = (8, 8)
    # Plant the offset exactly on the search-window edge.
    pred_vu = (actual_vu[0] - float(margin[0]), actual_vu[1])
    feature = make_star_feature('star:UCAC4:edge', predicted_vu=pred_vu, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, extfov_margin_vu=margin)
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_star_unique_match_two_star_3dof_recovers_planted_rotation(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The 2-star Procrustes path recovers a planted rotation around the catalog midpoint.

    Translation is reported in the catalog frame: the test plants a 1
    deg rotation between the catalog cohort and the detection cohort,
    which the 2-star similarity-transform fit must recover within the
    expected sub-degree tolerance.
    """
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual_a = (60.0, 60.0)
    actual_b = (140.0, 140.0)
    draw_gaussian_star(image, actual_a, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(image, actual_b, peak_dn=180.0, sigma=1.2)
    # Choose catalog predictions such that rotating them by the planted
    # angle about their midpoint and applying a tiny translation lands
    # on the detected positions.  Pivot is the midpoint of the catalog
    # predictions.
    planted_offset = (0.5, 0.5)
    planted_theta = math.radians(1.0)
    cat_a = (actual_a[0] - planted_offset[0], actual_a[1] - planted_offset[1])
    cat_b = (actual_b[0] - planted_offset[0], actual_b[1] - planted_offset[1])
    pivot = (0.5 * (cat_a[0] + cat_b[0]), 0.5 * (cat_a[1] + cat_b[1]))
    cos_t = math.cos(-planted_theta)
    sin_t = math.sin(-planted_theta)

    def _rotate(p: tuple[float, float]) -> tuple[float, float]:
        rv = p[0] - pivot[0]
        ru = p[1] - pivot[1]
        return (
            pivot[0] + cos_t * rv - sin_t * ru,
            pivot[1] + sin_t * rv + cos_t * ru,
        )

    pred_a = _rotate(cat_a)
    pred_b = _rotate(cat_b)
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=pred_a, predicted_snr=60.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=pred_b, predicted_snr=40.0)
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(planted_theta, abs=math.radians(0.3))
    # The Procrustes translation is ``t = det_centroid - R @ cat_centroid``,
    # which equals ``planted_offset + (pivot - R @ pivot)`` for the test
    # setup.  When the pivot sits far from the image origin (here at
    # ~ (100, 100)) that pivot-rotation correction dominates the raw
    # ``planted_offset`` term, so the assertion compares against the
    # full expected Procrustes output rather than the planted shift.
    cos_t = math.cos(planted_theta)
    sin_t = math.sin(planted_theta)
    rotated_pivot_v = cos_t * pivot[0] - sin_t * pivot[1]
    rotated_pivot_u = sin_t * pivot[0] + cos_t * pivot[1]
    expected_offset_v = planted_offset[0] + (pivot[0] - rotated_pivot_v)
    expected_offset_u = planted_offset[1] + (pivot[1] - rotated_pivot_u)
    assert result.offset_px[0] == pytest.approx(expected_offset_v, abs=0.4)
    assert result.offset_px[1] == pytest.approx(expected_offset_u, abs=0.4)
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.mode == 'two_star'


def test_star_unique_match_one_star_3dof_rotation_unobservable(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A 1-star path with fit_camera_rotation reports rotation as unobservable."""
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    actual = (100.0, 100.0)
    draw_gaussian_star(image, actual, peak_dn=200.0, sigma=1.2)
    feature = make_star_feature(
        'star:UCAC4:bright',
        predicted_vu=(99.0, 101.0),
        predicted_snr=60.0,
    )
    technique = StarUniqueMatchNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad == 0.0
    assert result.covariance_px2[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)
