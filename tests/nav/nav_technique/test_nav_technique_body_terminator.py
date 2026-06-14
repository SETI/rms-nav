"""End-to-end tests for ``BodyTerminatorNav``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.nav.nav_technique.conftest import (
    ArcPolylineFactory,
    DiscImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import TerminatorArcFlags
from nav.feature.geometry import TerminatorPolyline
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_technique.diagnostics import BodyTerminatorDiagnostics
from nav.nav_technique.nav_technique_body_terminator import (
    BodyTerminatorNav,
    _aggregate_terminator_features,
)
from nav.support.filters import NavFilterKind, NavFilterSpec

# Tests pull this value from the technique's loaded YAML tuning so a
# config edit retunes the assertions automatically.  ``BodyTerminatorNav``
# instantiation populates ``cls.tuning`` via ``Config.read_config`` —
# see ``test_nav_technique_body_limb.py`` for the same pattern.
BodyTerminatorNav()
TERMINATOR_MIN_ARC_PX = BodyTerminatorNav.tuning['min_arc_vertices']
TERMINATOR_SPURIOUS_MIN_INLIER_FRACTION = BodyTerminatorNav.tuning['spurious_min_inlier_fraction']
TERMINATOR_SPURIOUS_DT_FLOOR_PX = BodyTerminatorNav.tuning['spurious_dt_floor_px']

# Terminator tests always use a right-side crescent: a half-arc spanning
# [-pi/2, pi/2] around the body centre.  Other techniques use different
# angle ranges, so the bounds are a per-test parameter rather than a
# shared fixture default.
_TERMINATOR_ANGLE_START: float = -np.pi / 2.0
_TERMINATOR_ANGLE_END: float = np.pi / 2.0


def test_body_terminator_nav_recovers_planted_offset(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    model_center = (image_center[0] - 0.7, image_center[1] - 1.3)
    vertices, outward = arc_polyline(
        model_center, radius, 80, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('moonA', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(0.7, abs=0.1)
    assert result.offset_px[1] == pytest.approx(1.3, abs=0.1)
    assert isinstance(result.diagnostics, BodyTerminatorDiagnostics)
    assert result.diagnostics.lm_iterations >= 1


def test_body_terminator_nav_per_body_uniform_weighting() -> None:
    """Two bodies with very different sigma means must each carry uniform weight.

    The technique must collapse each body's per-vertex sigmas to a single
    per-body scalar via the body's mean — that's the design's
    "per-body uniform weight" rule (cross-body weighting reflects albedo
    variation; intra-body weighting is uniform).
    """
    angles = np.linspace(-np.pi / 4, np.pi / 4, 30)
    body_a_vs = 50.0 + 20.0 * np.sin(angles)
    body_a_us = 50.0 + 20.0 * np.cos(angles)
    body_a_vertices = np.stack([body_a_vs, body_a_us], axis=-1)
    body_a_normals = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    n = 30
    feature_a = NavFeature(
        feature_id='terminator_arc:bodyA',
        feature_type=NavFeatureType.TERMINATOR_ARC,
        source_model='body',
        geometry=TerminatorPolyline(
            vertices_vu=body_a_vertices,
            normals_vu=body_a_normals,
            sigma_normal_per_vertex_px=np.linspace(0.5, 2.5, n).astype(np.float64),
            sigma_tangent_per_vertex_px=np.full(n, 0.5, dtype=np.float64),
            bbox_extfov_vu=(40, 40, 80, 80),
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.7,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0, albedo_penalty=0.05),
        usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
        flags=TerminatorArcFlags(
            body_name='bodyA', visible_arc_fraction=1.0, phase_angle_factor=0.8
        ),
    )
    body_b_vs = 110.0 + 20.0 * np.sin(angles)
    body_b_us = 110.0 + 20.0 * np.cos(angles)
    body_b_vertices = np.stack([body_b_vs, body_b_us], axis=-1)
    body_b_normals = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    feature_b = NavFeature(
        feature_id='terminator_arc:bodyB',
        feature_type=NavFeatureType.TERMINATOR_ARC,
        source_model='body',
        geometry=TerminatorPolyline(
            vertices_vu=body_b_vertices,
            normals_vu=body_b_normals,
            sigma_normal_per_vertex_px=np.full(n, 5.0, dtype=np.float64),
            sigma_tangent_per_vertex_px=np.full(n, 0.5, dtype=np.float64),
            bbox_extfov_vu=(100, 100, 140, 140),
        ),
        subject_range_km=1.0e6,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.4,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0, albedo_penalty=0.6),
        usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
        flags=TerminatorArcFlags(
            body_name='bodyB', visible_arc_fraction=1.0, phase_angle_factor=0.4
        ),
    )
    _, _, sigmas, ids, _, _ = _aggregate_terminator_features([feature_a, feature_b])
    # bodyA's per-body sigma is the mean of its per-vertex sigmas (1.5);
    # bodyB's is the constant 5.0.  The aggregator must use each body's
    # mean uniformly across its own vertices.
    assert ids == ['terminator_arc:bodyA', 'terminator_arc:bodyB']
    assert np.allclose(sigmas[:n], np.full(n, 1.5))
    assert np.allclose(sigmas[n:], np.full(n, 5.0))


def test_body_terminator_nav_infeasible_on_empty_input() -> None:
    technique = BodyTerminatorNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_terminator_arc_features' in report.reason


def test_body_terminator_nav_infeasible_on_short_arc(
    make_terminator_feature: NavFeatureFactory,
) -> None:
    short_n = int(TERMINATOR_MIN_ARC_PX) - 1
    angles = np.linspace(-np.pi / 4, np.pi / 4, short_n)
    vs = 50.0 + 20.0 * np.sin(angles)
    us = 50.0 + 20.0 * np.cos(angles)
    vertices = np.stack([vs, us], axis=-1)
    outward = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    feature = make_terminator_feature('shortmoon', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    report = technique.is_feasible([feature])
    assert report.feasible is False


def test_body_terminator_nav_marks_spurious_when_image_lacks_terminator(
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    image = np.zeros(shape, dtype=np.float64)
    image[5:15, 5:15] = 100.0
    angles = np.linspace(-np.pi / 4, np.pi / 4, 80)
    vs = 100.0 + 30.0 * np.sin(angles)
    us = 100.0 + 30.0 * np.cos(angles)
    vertices = np.stack([vs, us], axis=-1)
    outward = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    feature = make_terminator_feature('lonely', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert result.spurious is True


def test_body_terminator_nav_marks_spurious_when_inlier_fraction_collapses(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An LM convergence retaining a small minority of vertices is spurious.

    Models the production failure mode (Voyager Enceladus C4400436): the
    terminator polyline can lock onto crater shadows or surface boundary
    features and report a low RMS while rejecting most vertices as
    outliers — a high-confidence wrong answer if the spurious flag does
    not fire.  This mirrors ``BodyLimbNav``'s inlier-fraction guard.
    """
    from nav.nav_technique import dt_fitting, nav_technique_body_terminator

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = arc_polyline(
        (100.0, 100.0), 30.0, 1000, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('confused', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)

    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(7.0, -3.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64),
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.zeros(vertices.shape[0], dtype=np.float64),
        rms_px=2.5,
        raw_rms_px=2.5,
        iterations=5,
        converged=True,
        inlier_count=10,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_terminator,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyTerminatorDiagnostics)
    inlier_fraction = result.diagnostics.tukey_inlier_count / float(vertices.shape[0])
    assert inlier_fraction < TERMINATOR_SPURIOUS_MIN_INLIER_FRACTION
    assert result.spurious is True


def test_body_terminator_nav_does_not_mark_spurious_when_inlier_fraction_healthy(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A healthy inlier fraction does not trip the spurious flag."""
    from nav.nav_technique import dt_fitting, nav_technique_body_terminator

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = arc_polyline(
        (100.0, 100.0), 30.0, 1000, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('healthy', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)

    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=np.zeros(vertices.shape[0], dtype=np.float64),
        weights=np.ones(vertices.shape[0], dtype=np.float64),
        rms_px=0.4,
        raw_rms_px=0.4,
        iterations=8,
        converged=True,
        inlier_count=500,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_terminator,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyTerminatorDiagnostics)
    inlier_fraction = result.diagnostics.tukey_inlier_count / float(vertices.shape[0])
    assert inlier_fraction >= TERMINATOR_SPURIOUS_MIN_INLIER_FRACTION
    assert result.spurious is False


def test_body_terminator_nav_raw_rms_gate_catches_tukey_masked_bad_arc(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A bad arc that Tukey down-weights to zero is still flagged spurious.

    One arc half fits cleanly (~0 px); the other half is offset by ~10 px
    and reweighted to zero by Tukey.  The weighted ``rms_px`` collapses to
    ~0 (below the floor, so the weighted gate would pass), but the
    unweighted ``raw_rms_px`` retains the offset half and exceeds the
    threshold, so the raw-RMS gate marks the fit spurious.
    """
    from nav.nav_technique import dt_fitting, nav_technique_body_terminator

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = arc_polyline(
        (100.0, 100.0), 30.0, 1000, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('masked_bad_arc', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)

    n_total = vertices.shape[0]
    n_half = n_total // 2
    residuals = np.zeros(n_total, dtype=np.float64)
    residuals[n_half:] = 10.0
    weights = np.zeros(n_total, dtype=np.float64)
    weights[:n_half] = 1.0
    raw_rms = float(np.sqrt(np.mean(residuals**2)))
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.0,
        raw_rms_px=raw_rms,
        iterations=10,
        converged=True,
        inlier_count=n_half,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_terminator,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyTerminatorDiagnostics)
    assert forged_result.rms_px <= TERMINATOR_SPURIOUS_DT_FLOOR_PX
    assert raw_rms > TERMINATOR_SPURIOUS_DT_FLOOR_PX
    assert result.spurious is True


def test_body_terminator_nav_clean_fit_has_small_raw_rms_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A clean fit has a small ``raw_rms_px`` and is not flagged spurious."""
    from nav.nav_technique import dt_fitting, nav_technique_body_terminator

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, outward = arc_polyline(
        (100.0, 100.0), 30.0, 1000, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('clean_arc', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)

    n_total = vertices.shape[0]
    residuals = np.full(n_total, 0.3, dtype=np.float64)
    weights = np.ones(n_total, dtype=np.float64)
    raw_rms = float(np.sqrt(np.mean(residuals**2)))
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.5, 0.5),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=raw_rms,
        raw_rms_px=raw_rms,
        iterations=8,
        converged=True,
        inlier_count=n_total,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_body_terminator,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, BodyTerminatorDiagnostics)
    assert raw_rms < TERMINATOR_SPURIOUS_DT_FLOOR_PX
    assert result.spurious is False


def test_body_terminator_nav_at_edge_when_offset_hits_search_window(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image_center = (80.0, 80.0)
    radius = 25.0
    image = disc_image(shape, image_center, radius)
    margin_v, margin_u = 6, 6
    model_center = (image_center[0] - float(margin_v), image_center[1] - float(margin_u))
    vertices, outward = arc_polyline(
        model_center, radius, 80, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('atedge', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image, extfov_margin_vu=(margin_v, margin_u))
    result = technique.navigate([feature], context)
    assert result.at_edge is True
    assert result.confidence == pytest.approx(0.0, abs=1.0e-12)


def test_body_terminator_nav_registered_with_navtechnique_registry() -> None:
    from nav.nav_technique.nav_technique import NavTechnique

    assert BodyTerminatorNav in NavTechnique._registry


def test_body_terminator_nav_raises_when_navcontext_lacks_derivatives(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image = disc_image(shape, (80.0, 80.0), 25.0)
    vertices, outward = arc_polyline(
        (80.0, 80.0), 25.0, 80, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('moon', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image)
    bare_context = NavContext(
        obs=context.obs,
        image_ext=context.image_ext,
        sensor_mask_ext=context.sensor_mask_ext,
        image_noise_sigma=context.image_noise_sigma,
        saturation_mask_ext=context.saturation_mask_ext,
        cosmic_ray_mask_ext=context.cosmic_ray_mask_ext,
        image_classifier=context.image_classifier,
        provenance=context.provenance,
    )
    with pytest.raises(RuntimeError, match='image_edge_dt_ext'):
        technique.navigate([feature], bare_context)


def test_body_terminator_nav_3dof_emits_3x3_covariance(
    disc_image: DiscImageFactory,
    arc_polyline: ArcPolylineFactory,
    make_terminator_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """With ``fit_camera_rotation=True`` the result carries a 3x3 covariance + rotation."""
    shape = (200, 200)
    image_center = (100.0, 100.0)
    radius = 30.0
    image = disc_image(shape, image_center, radius)
    model_center = (image_center[0] - 0.7, image_center[1] - 1.3)
    vertices, outward = arc_polyline(
        model_center, radius, 80, _TERMINATOR_ANGLE_START, _TERMINATOR_ANGLE_END
    )
    feature = make_terminator_feature('moonA', vertices=vertices, outward_normals=outward)
    technique = BodyTerminatorNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.sigma_rotation_rad is not None
    # No rotation planted; convergence stays well inside the 5 degree cap.
    assert abs(result.rotation_rad) < np.deg2rad(5.0)
