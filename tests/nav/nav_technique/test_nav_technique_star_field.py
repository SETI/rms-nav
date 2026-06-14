"""End-to-end and helper-level tests for ``StarFieldFromCatalogNav``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.nav.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from nav.feature.feature import NavFeature
from nav.nav_technique.diagnostics import StarFieldDiagnostics
from nav.nav_technique.nav_technique import ROTATION_UNOBSERVABLE_VARIANCE, NavTechnique
from nav.nav_technique.nav_technique_star_field import (
    StarFieldFromCatalogNav,
    _detect_image_sources,
    _enumerate_triplets,
    _greedy_inlier_count,
    _hash_distance_sq,
    _solve_translation,
    _triplet_hash,
)


def _star_field_image(
    star_centers: list[tuple[float, float]],
    *,
    draw: DrawGaussianStarFactory,
    shape: tuple[int, int] = (300, 300),
    peak_dn: float = 200.0,
    sigma: float = 1.2,
) -> np.ndarray:
    """Return an image with planted Gaussian stars at the supplied centers."""
    image = np.zeros(shape, dtype=np.float64)
    for cv, cu in star_centers:
        draw(image, (cv, cu), peak_dn=peak_dn, sigma=sigma)
    return image


def _make_star_field_features(
    centers: list[tuple[float, float]],
    *,
    make_feature: NavFeatureFactory,
    offset: tuple[float, float],
    snrs: list[float] | None = None,
) -> list[NavFeature]:
    """Build a STAR feature per planted center, predicted at ``center - offset``.

    ``offset`` is the planted ``(dv, du)`` translation: predicted at
    position ``center - offset`` so that the technique recovers
    ``offset``.
    """
    if snrs is None:
        snrs = [40.0 - 0.5 * i for i in range(len(centers))]
    features: list[NavFeature] = []
    for i, (cv, cu) in enumerate(centers):
        pred = (cv - offset[0], cu - offset[1])
        features.append(
            make_feature(
                f'star:UCAC4:{i}',
                predicted_vu=pred,
                predicted_snr=snrs[i],
            )
        )
    return features


# ---------------------------------------------------------------------------
# Sub-piece 1 — source detection.
# ---------------------------------------------------------------------------


def test_detect_image_sources_finds_planted_stars(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The matched-filter detector recovers planted Gaussian peaks."""
    centers = [(50.0, 60.0), (120.0, 80.0), (200.0, 220.0)]
    image = _star_field_image(centers, draw=draw_gaussian_star)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=30,
    )
    assert len(detected) == 3
    detected_centers = sorted((s.v, s.u) for s in detected)
    expected = sorted(centers)
    for got, want in zip(detected_centers, expected, strict=True):
        assert got[0] == pytest.approx(want[0], abs=0.5)
        assert got[1] == pytest.approx(want[1], abs=0.5)


def test_detect_image_sources_caps_at_max_sources(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """``max_sources`` keeps the brightest peaks first."""
    image = _star_field_image(
        [(50.0, 50.0), (100.0, 100.0), (150.0, 150.0), (200.0, 200.0)],
        draw=draw_gaussian_star,
        peak_dn=200.0,
    )
    # Plant a fainter fifth peak — it must rank below the four brighter
    # ones and get dropped when max_sources=4.
    draw_gaussian_star(image, (250.0, 250.0), peak_dn=80.0, sigma=1.2)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=4,
    )
    assert len(detected) == 4
    assert all(s.peak_dn >= detected[-1].peak_dn for s in detected)


def test_detect_image_sources_returns_empty_when_blank() -> None:
    """A blank image yields no detections."""
    image = np.zeros((100, 100), dtype=np.float64)
    detected = _detect_image_sources(
        image,
        image_noise_sigma=1.0,
        sigma_px=1.2,
        detection_sigma=4.0,
        centroid_box_half_px=3,
        max_sources=30,
    )
    assert detected == []


# ---------------------------------------------------------------------------
# Sub-piece 2 — triplet hashing.
# ---------------------------------------------------------------------------


def test_triplet_hash_is_translation_invariant() -> None:
    """The hash does not change when every vertex is translated by the same offset."""
    a = (10.0, 20.0)
    b = (12.0, 35.0)
    c = (40.0, 22.0)
    h1 = _triplet_hash(a, b, c)
    h2 = _triplet_hash(
        (a[0] + 7.0, a[1] - 3.0),
        (b[0] + 7.0, b[1] - 3.0),
        (c[0] + 7.0, c[1] - 3.0),
    )
    assert h1 is not None
    assert h2 is not None
    for v1, v2 in zip(h1, h2, strict=True):
        assert v1 == pytest.approx(v2)


def test_triplet_hash_is_uniform_scale_invariant() -> None:
    """Multiplying every vertex by a constant scale leaves the hash unchanged."""
    a = (10.0, 20.0)
    b = (12.0, 35.0)
    c = (40.0, 22.0)
    s = 3.7
    h1 = _triplet_hash(a, b, c)
    h2 = _triplet_hash(
        (a[0] * s, a[1] * s),
        (b[0] * s, b[1] * s),
        (c[0] * s, c[1] * s),
    )
    assert h1 is not None
    assert h2 is not None
    for v1, v2 in zip(h1, h2, strict=True):
        assert v1 == pytest.approx(v2)


def test_triplet_hash_drops_collinear() -> None:
    """A triplet with two coincident points is rejected (None)."""
    h = _triplet_hash((10.0, 10.0), (10.0, 10.0), (20.0, 20.0))
    assert h is None


def test_enumerate_triplets_yields_one_per_unordered_set() -> None:
    """Each unordered triplet appears exactly once with A=brightest."""
    points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (5.0, 5.0)]
    # Brightness rank 0 = brightest; rank by index here.
    brightness_rank = [0, 1, 2, 3]
    triplets = _enumerate_triplets(points, brightness_rank)
    # C(4,3) = 4 triplets.
    assert len(triplets) == 4
    # The brightest of each unordered set is correctly identified as A.
    seen_keys: set[tuple[int, int, int]] = set()
    for t in triplets:
        # b and c indices are sorted ascending.
        assert t.idx_b < t.idx_c
        # A index has the lowest brightness rank.
        ranks = (
            brightness_rank[t.idx_a],
            brightness_rank[t.idx_b],
            brightness_rank[t.idx_c],
        )
        assert ranks[0] == min(ranks)
        # Unordered triplet seen once.
        sorted_indices = sorted([t.idx_a, t.idx_b, t.idx_c])
        key = (sorted_indices[0], sorted_indices[1], sorted_indices[2])
        assert key not in seen_keys
        seen_keys.add(key)


def test_hash_distance_sq_zero_for_identical_hashes() -> None:
    """``_hash_distance_sq(h, h, ...)`` is exactly zero."""
    h = (1.5, 2.5, 0.7)
    d_sq = _hash_distance_sq(h, h, ratio_weight=1.0, angle_weight=1.0)
    assert d_sq == 0.0


# ---------------------------------------------------------------------------
# Helper-level translation solve and inlier counting.
# ---------------------------------------------------------------------------


def test_solve_translation_recovers_constant_offset() -> None:
    """Constant per-correspondence offset is recovered as the weighted mean."""
    catalog = np.asarray([[10.0, 20.0], [30.0, 50.0], [70.0, 110.0]], dtype=np.float64)
    planted = (4.0, -2.5)
    detection = catalog + np.asarray(planted)
    weights = np.ones(3, dtype=np.float64)
    dv, du = _solve_translation(detection, catalog, weights)
    assert dv == pytest.approx(planted[0])
    assert du == pytest.approx(planted[1])


def test_greedy_inlier_count_under_perfect_offset() -> None:
    """Every detection is an inlier when the offset perfectly aligns the catalog."""
    catalog = np.asarray([[10.0, 20.0], [30.0, 50.0], [70.0, 110.0]], dtype=np.float64)
    offset = (4.0, -2.5)
    detections = catalog + np.asarray(offset)
    n_inliers, pairs = _greedy_inlier_count(detections, catalog, offset_vu=offset, tolerance_px=0.5)
    assert n_inliers == 3
    assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)]


def test_greedy_inlier_count_does_not_double_match() -> None:
    """Each catalog star matches at most one detection."""
    catalog = np.asarray([[10.0, 20.0]], dtype=np.float64)
    detections = np.asarray([[10.0, 20.0], [10.5, 19.9]], dtype=np.float64)
    n_inliers, pairs = _greedy_inlier_count(
        detections, catalog, offset_vu=(0.0, 0.0), tolerance_px=2.0
    )
    assert n_inliers == 1
    assert pairs == [(0, 0)]


# ---------------------------------------------------------------------------
# Sub-piece 3 / 4 — RANSAC + verification (technique-level).
# ---------------------------------------------------------------------------


def test_star_field_recovers_planted_offset(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """RANSAC + Tukey refit recovers a planted translation on a clean field."""
    centers = [
        (60.0, 80.0),
        (110.0, 130.0),
        (180.0, 90.0),
        (220.0, 200.0),
        (90.0, 240.0),
        (250.0, 70.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (3.0, -2.5)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=(32, 32))
    feasibility = technique.is_feasible(features)
    assert feasibility.feasible is True
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.n_inliers >= 6


def test_star_field_is_deterministic_across_two_invocations(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two back-to-back invocations on the same obs return bit-identical fits."""
    centers = [
        (60.0, 80.0),
        (110.0, 130.0),
        (180.0, 90.0),
        (220.0, 200.0),
        (90.0, 240.0),
        (250.0, 70.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (1.5, 2.0)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image)
    result_a = technique.navigate(features, context)
    result_b = technique.navigate(features, context)
    assert result_a.offset_px == result_b.offset_px
    assert np.array_equal(result_a.covariance_px2, result_b.covariance_px2)
    assert result_a.confidence == result_b.confidence
    assert result_a.diagnostics == result_b.diagnostics


def test_star_field_infeasible_with_fewer_than_three_features(
    make_star_feature: NavFeatureFactory,
) -> None:
    """``is_feasible`` requires >= 3 usable STAR features."""
    feat_a = make_star_feature('star:UCAC4:A', predicted_vu=(50.0, 50.0), predicted_snr=40.0)
    feat_b = make_star_feature('star:UCAC4:B', predicted_vu=(80.0, 80.0), predicted_snr=35.0)
    technique = StarFieldFromCatalogNav()
    report = technique.is_feasible([feat_a, feat_b])
    assert report.feasible is False
    assert 'fewer_than_3_predicted_stars' in report.reason


def test_star_field_infeasible_when_all_stars_occluded(
    make_star_feature: NavFeatureFactory,
) -> None:
    """Stars marked in_body_silhouette are filtered out before feasibility."""
    features = [
        make_star_feature(
            f'star:UCAC4:{i}',
            predicted_vu=(50.0 + 30.0 * i, 60.0 + 25.0 * i),
            predicted_snr=30.0,
            in_body_silhouette=True,
        )
        for i in range(5)
    ]
    technique = StarFieldFromCatalogNav()
    report = technique.is_feasible(features)
    assert report.feasible is False
    assert 'fewer_than_3_predicted_stars' in report.reason


def test_star_field_returns_spurious_when_too_few_detections(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Image with only 2 bright peaks fails the >= 3 detected-sources check."""
    image = _star_field_image(
        [(50.0, 50.0), (120.0, 130.0)], draw=draw_gaussian_star, shape=(200, 200)
    )
    centers = [(50.0, 50.0), (120.0, 130.0), (160.0, 80.0)]
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=(0.0, 0.0))
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image)
    result = technique.navigate(features, context)
    assert result.spurious is True
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.n_detected_sources == 2


def test_star_field_marks_at_edge_when_offset_hits_margin(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """An offset that hits the search-window edge is flagged + zero confidence."""
    centers = [
        (80.0, 80.0),
        (110.0, 130.0),
        (140.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    margin = (8, 8)
    # Plant the offset on the V-axis margin.
    planted = (float(margin[0]), 0.0)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, extfov_margin_vu=margin)
    result = technique.navigate(features, context)
    # Translation hit the v-axis margin: at_edge fires and the
    # confidence formula's hard_zero_if pushes confidence to zero.
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0)


def test_star_field_registered_with_navtechnique_registry() -> None:
    """``StarFieldFromCatalogNav`` self-registers on import."""
    assert StarFieldFromCatalogNav in NavTechnique._registry


def test_star_field_min_inliers_less_than_three_raises() -> None:
    """Construction rejects a ``pattern_match_min_inliers`` below 3."""
    technique_class = StarFieldFromCatalogNav
    bad_tuning = dict(technique_class.tuning)
    bad_tuning['pattern_match_min_inliers'] = 2
    original_tuning = technique_class.tuning
    try:
        technique_class.tuning = bad_tuning
        with pytest.raises(ValueError, match='pattern_match_min_inliers'):
            technique_class()
    finally:
        technique_class.tuning = original_tuning


# ---------------------------------------------------------------------------
# 3-DoF (Procrustes / similarity-transform) tests.
# ---------------------------------------------------------------------------


def _rotate_about(
    points: list[tuple[float, float]],
    pivot: tuple[float, float],
    theta_rad: float,
) -> list[tuple[float, float]]:
    """Rotate ``points`` about ``pivot`` by ``theta_rad``.

    Parameters:
        points: Sequence of ``(v, u)`` pixel coordinates to rotate.
        pivot: ``(v, u)`` pivot in the same coordinate frame as
            ``points``.  The rotation is applied about this point with
            no translation.
        theta_rad: Rotation angle in radians, counter-clockwise in the
            image-frame ``(v, u)`` axes (``v`` increasing downward,
            ``u`` increasing rightward) — i.e. the standard rotation
            matrix ``[[cos, -sin], [sin, cos]]`` applied to the
            ``(v, u)`` offset from the pivot.

    Returns:
        New list of ``(v, u)`` tuples, one per input point, in the
        same order as ``points``.
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    pv, pu = pivot
    rotated: list[tuple[float, float]] = []
    for v, u in points:
        rv = v - pv
        ru = u - pu
        rotated.append(
            (
                pv + cos_t * rv - sin_t * ru,
                pu + sin_t * rv + cos_t * ru,
            )
        )
    return rotated


def test_star_field_3dof_recovers_planted_rotation(
    draw_gaussian_star: DrawGaussianStarFactory,
    make_star_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A planted rotation + translation is recovered by the Procrustes path."""
    centers = [
        (80.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
        (60.0, 60.0),
        (260.0, 250.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted_offset = (1.5, -2.0)
    # 0.5 deg rotation: displacement at ~150 px radius from the pivot is
    # ~1.3 px, comfortably under the 2.0 px inlier-matching tolerance.
    planted_theta = math.radians(0.5)
    pivot = (
        sum(c[0] for c in centers) / len(centers),
        sum(c[1] for c in centers) / len(centers),
    )
    # Rotate detection-frame centres back into the catalog frame.  The
    # technique should report (planted_offset, planted_theta) as the
    # similarity transform that maps catalog -> detection.
    catalog_centers = _rotate_about(
        [(c[0] - planted_offset[0], c[1] - planted_offset[1]) for c in centers],
        pivot=(pivot[0] - planted_offset[0], pivot[1] - planted_offset[1]),
        theta_rad=-planted_theta,
    )
    snrs = [40.0 - 0.5 * i for i in range(len(catalog_centers))]
    features: list[NavFeature] = [
        make_star_feature(
            f'star:UCAC4:{i}',
            predicted_vu=catalog_centers[i],
            predicted_snr=snrs[i],
        )
        for i in range(len(catalog_centers))
    ]
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate(features, context)
    assert result.spurious is False
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.rotation_rad == pytest.approx(planted_theta, abs=math.radians(0.2))
    assert result.sigma_rotation_rad is not None
    assert result.sigma_rotation_rad > 0.0


def test_star_field_3dof_zero_rotation_path_remains_close_to_planted_offset(
    draw_gaussian_star: DrawGaussianStarFactory,
    make_star_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A zero-rotation scene under fit_camera_rotation still recovers the offset."""
    centers = [
        (80.0, 90.0),
        (170.0, 200.0),
        (200.0, 70.0),
        (90.0, 240.0),
        (250.0, 130.0),
        (60.0, 60.0),
        (260.0, 250.0),
    ]
    image = _star_field_image(centers, draw=draw_gaussian_star, shape=(320, 320))
    planted = (1.0, -2.5)
    features = _make_star_field_features(centers, make_feature=make_star_feature, offset=planted)
    technique = StarFieldFromCatalogNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate(features, context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.5)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.5)
    assert result.rotation_rad == pytest.approx(0.0, abs=math.radians(0.5))


def test_build_covariance_two_point_reduced_chi_square() -> None:
    """Translation-only covariance matches the analytic reduced-chi-square mean.

    Per-axis var = chi2_nu / sum(w), chi2_nu = sum(w r^2)/max(N - p, 1),
    p = 2.  w = [1, 3], sum(w) = 4, N = 2, dof = max(2 - 2, 1) = 1.

    V residuals = [-1.5, 0.5]: sum(w r^2) = 1*2.25 + 3*0.25 = 3.0,
        chi2_nu = 3.0/1 = 3.0, var = 3.0/4 = 0.75 (> floor 1/4 = 0.25).
    U residuals = [-1.5, 0.5]: identical -> var = 0.75.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 3.0], dtype=np.float64)
    residuals = np.array([[-1.5, -1.5], [0.5, 0.5]], dtype=np.float64)
    cov = technique._build_covariance(weights=weights, residuals=residuals)
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(0.75, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.75, abs=1e-9)
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_n_point_reduced_chi_square() -> None:
    """Four-point translation covariance matches the analytic reduced chi-square.

    w = [1, 1, 1, 1], sum(w) = 4, N = 4, p = 2, dof = max(4 - 2, 1) = 2.

    V residuals = [-3, -1, 1, 3]: sum(w r^2) = 9 + 1 + 1 + 9 = 20,
        chi2_nu = 20/2 = 10, var = 10/4 = 2.5 (> floor 0.25).
    U residuals = [0, 0, 0, 0]: sum(w r^2) = 0, candidate var 0 ->
        floored at 1/sum(w) = 0.25.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    cov = technique._build_covariance(weights=weights, residuals=residuals)
    assert cov[0, 0] == pytest.approx(2.5, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)


def test_build_covariance_model_error_floor_inflates_by_square() -> None:
    """ORCH-001: model_error_floor_px>0 adds exactly its square to the diagonal.

    Reuses the two-point fixture (var 0.75/axis with no floor).  Setting
    model_error_floor_px = 2.0 adds 2.0**2 = 4.0 -> 4.75 on each axis;
    the off-diagonal stays zero.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 3.0], dtype=np.float64)
    residuals = np.array([[-1.5, -1.5], [0.5, 0.5]], dtype=np.float64)
    base = technique._build_covariance(weights=weights, residuals=residuals)
    technique._model_error_floor_px = 2.0
    floored = technique._build_covariance(weights=weights, residuals=residuals)
    assert floored[0, 0] == pytest.approx(4.75, abs=1e-9)
    assert floored[1, 1] == pytest.approx(4.75, abs=1e-9)
    assert floored[0, 0] - base[0, 0] == pytest.approx(4.0, abs=1e-9)
    assert floored[1, 1] - base[1, 1] == pytest.approx(4.0, abs=1e-9)
    assert floored[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_3dof_rotation_variance_analytic() -> None:
    """3-DoF rotation variance matches the analytic lever-arm reduced chi-square.

    Four equal-weight points (w = 1), p = 3, dof = max(4 - 3, 1) = 1.

    Catalog positions (0,0),(0,4),(4,0),(4,4): weighted centroid (2,2),
    each lever arm |cat - cc|**2 = 8, so spread = sum(w * 8) = 32.

    Residuals: r_v = [1, -1, 1, -1], r_u = [0, 0, 0, 0].
        sum(w r_v^2) = 4, sum(w r_u^2) = 0,
        chi2_nu_residual = 0.5 * (4 + 0) / 1 = 2.0,
        sigma_theta^2 = max(2.0/32, 1/32) = 0.0625.

    Translation block (p = 3, dof = 1):
        V: chi2_nu = 4/1 = 4, var = 4/4 = 1.0 (> floor 0.25).
        U: chi2_nu = 0 -> floored at 0.25.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    cat_inliers = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 4.0]], dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov.shape == (3, 3)
    assert cov[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert cov[1, 1] == pytest.approx(0.25, abs=1e-9)
    assert cov[2, 2] == pytest.approx(0.0625, abs=1e-9)
    # Cross-terms are zero by construction.
    assert cov[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert cov[1, 2] == pytest.approx(0.0, abs=1e-9)


def test_build_covariance_3dof_coincident_catalog_is_rotation_unobservable() -> None:
    """Coincident catalog inliers (zero spread) report rotation unobservable.

    With all catalog points at the same location the lever-arm spread is
    zero, so the rotation slot collapses to the unobservable sentinel.
    """
    technique = StarFieldFromCatalogNav()
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    residuals = np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]], dtype=np.float64)
    cat_inliers = np.full((4, 2), 7.0, dtype=np.float64)
    cov = technique._build_covariance_3dof(
        weights=weights, residuals=residuals, cat_inliers=cat_inliers
    )
    assert cov[2, 2] == pytest.approx(ROTATION_UNOBSERVABLE_VARIANCE)
