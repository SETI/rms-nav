"""End-to-end tests for ``RingEdgeNav``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    CirclePolylineFactory,
    DiscImageFactory,
    FlatPolylineFactory,
    HorizontalStepImageFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.nav_technique_ring_edge import (
    _RANK1_NULL_RELATIVE_THRESHOLD,
    RingEdgeNav,
)


def test_ring_edge_nav_recovers_planted_offset_curved(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    cv = 100.0
    cu = 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    # Plant model 0.7 v-down, 1.3 u-right.
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    feature = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    feasibility = technique.is_feasible([feature])
    assert feasibility.feasible is True
    result = technique.navigate([feature], context)
    assert result.offset_px[0] == pytest.approx(0.7, abs=0.3)
    assert result.offset_px[1] == pytest.approx(1.3, abs=0.3)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is False
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    assert float(eigvals.min()) > 1.0e-12


def test_ring_edge_nav_returns_rank_1_for_all_flat_input(
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (200, 200)
    # A simple half-image step: bright above row 100, dark below.  The
    # gradient peak (one pixel wide) sits at row 100, giving a clean,
    # single minimum that the LM can converge to.
    image = horizontal_step_image(shape, 100.0)
    # Polyline sits 1.5 px off the actual edge.  The LM should drive
    # the offset back to ~ -1.5 in v but only along the radial axis;
    # along the edge tangent (u) the cost has no slope, so the
    # information matrix is rank-1.
    vertices, outward = flat_polyline(101.5, 20.0, 180.0, 120)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    result = technique.navigate([feature], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    null_eigval = float(eigvals.min())
    observed_eigval = float(eigvals.max())
    assert observed_eigval > 0.0
    # Null direction is along the edge tangent (u axis); rank-deficient
    # covariance is expressed via the eigenvalue ratio.  Use the same
    # threshold the production code's ``_is_rank_1`` classifier uses, so
    # this test exercises the same boundary the technique itself draws
    # rather than a looser independent constant.
    assert (
        null_eigval == pytest.approx(0.0, abs=1.0e-9)
        or null_eigval / observed_eigval < _RANK1_NULL_RELATIVE_THRESHOLD
    )


def test_ring_edge_nav_mixed_curved_and_flat_full_rank(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (220, 220)
    cv_disc, cu_disc = 60.0, 110.0
    image = disc_image(shape, (cv_disc, cu_disc), 24.0)
    # Add a horizontal step below the disc so the two image features do
    # not overlap; the disc occupies rows 36..84 and the step lives at
    # row 150.
    bar_image = np.zeros(shape, dtype=np.float64)
    bar_image[150:152, 30:190] = 100.0
    image = np.clip(image + bar_image, 0.0, 100.0)
    # Both polylines share offset (-1.5, 0): curved centre at
    # (cv_disc + 1.5, cu_disc) and flat polyline at v = 151 + 1.5.
    curved_v, curved_n = circle_polyline((cv_disc + 1.5, cu_disc), 24.0, 120)
    flat_v, flat_n = flat_polyline(152.5, 40.0, 180.0, 80)
    curved_feature = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    flat_feature = make_ring_feature(
        'flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)
    result = technique.navigate([curved_feature, flat_feature], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is False
    eigvals = np.linalg.eigvalsh(result.covariance_px2)
    assert float(eigvals.min()) > 1.0e-12


def test_ring_edge_nav_infeasible_on_empty_input() -> None:
    technique = RingEdgeNav()
    report = technique.is_feasible([])
    assert report.feasible is False
    assert 'no_ring_edge_features' in report.reason


def test_ring_edge_nav_marks_spurious_when_per_edge_rms_collapses_to_one(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Per-edge RMS exposes a single-edge mis-convergence Tukey would mask.

    Models the production failure mode (Cassini Tethys N1572471790):
    three RING_EDGE features at distinct radii feed the LM, and the
    fit walks onto one of them while the other two contribute pure
    outliers.  Tukey rejects the outliers so ``rms_px`` is near zero
    even though the offset is far from the true joint solution; the
    per-edge sum is the only signal that recovers the correct
    spurious flag.

    The test forges an LM result whose post-Tukey ``rms_px`` is
    zero but whose per-vertex residuals are bimodal so that the
    per-edge RMS averages to a value far above the spurious
    threshold.  ``RingEdgeNav.navigate`` must then mark the result
    spurious so the ensemble can drop it.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 60)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 60)
    vertices_c, normals_c = circle_polyline((100.0, 100.0), 50.0, 60)
    feat_a = make_ring_feature(
        'inner', vertices=vertices_a, outward_normals=normals_a, is_straight_line=False
    )
    feat_b = make_ring_feature(
        'middle', vertices=vertices_b, outward_normals=normals_b, is_straight_line=False
    )
    feat_c = make_ring_feature(
        'outer', vertices=vertices_c, outward_normals=normals_c, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Bimodal residuals: edge A fits cleanly (~0 px), edges B and C are
    # off by ~50 px so their per-edge RMS is large.  Tukey reweights
    # the bad edges to zero — ``rms_px`` collapses to ~0 but the raw
    # per-edge sum makes the mis-convergence visible.
    n_total = vertices_a.shape[0] + vertices_b.shape[0] + vertices_c.shape[0]
    residuals = np.zeros(n_total, dtype=np.float64)
    residuals[vertices_a.shape[0] :] = 50.0
    weights = np.zeros(n_total, dtype=np.float64)
    weights[: vertices_a.shape[0]] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-27.0, -18.0),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.0,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=vertices_a.shape[0],
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )

    result = technique.navigate([feat_a, feat_b, feat_c], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    # Average per-edge RMS = (0 + 50 + 50) / 3 ≈ 33.3, far above any
    # plausible spurious threshold derived from sub-pixel sigmas.
    assert result.diagnostics.per_edge_dt_rms_summed > 50.0
    assert result.spurious is True


def test_ring_edge_nav_registered_with_navtechnique_registry() -> None:
    from spindoctor.nav_technique.nav_technique import NavTechnique

    assert RingEdgeNav in NavTechnique._registry


def test_ring_edge_nav_raises_when_navcontext_lacks_derivatives(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    shape = (160, 160)
    image = disc_image(shape, (80.0, 80.0), 25.0)
    vertices, outward = circle_polyline((80.0, 80.0), 25.0, 80)
    feature = make_ring_feature(
        'mid', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
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


def test_ring_edge_nav_3dof_emits_3x3_covariance(
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A curved ring edge with ``fit_camera_rotation=True`` produces a 3x3 covariance."""
    shape = (200, 200)
    cv = 100.0
    cu = 100.0
    radius = 32.0
    image = disc_image(shape, (cv, cu), radius)
    vertices, outward = circle_polyline((cv - 0.7, cu - 1.3), radius, 120)
    feature = make_ring_feature(
        'outer', vertices=vertices, outward_normals=outward, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert result.rotation_rad is not None
    assert result.sigma_rotation_rad is not None
    # No rotation planted; the LM should converge to near zero.  Allow
    # up to 3 sigma of the reported rotation uncertainty as the
    # tolerance so tighter sigma estimates also tighten the test.
    assert np.isclose(result.rotation_rad, 0.0, atol=3.0 * result.sigma_rotation_rad)


def test_ring_edge_nav_3dof_flat_edge_remains_rank_deficient(
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """An all-flat scene under fit_camera_rotation reports rank-1 in the translation block.

    The 3x3 covariance is rank-deficient on multiple axes (along-edge
    plus rotation when geometry is uninformative); the technique still
    flags ``is_rank_1`` based on the 2x2 translation block so the
    diagnostic stays comparable to the 2-DoF case.
    """
    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    vertices, outward = flat_polyline(99.5, 30.0, 170.0, 200)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image, fit_camera_rotation=True, max_rotation_deg=5.0)
    result = technique.navigate([feature], context)
    assert result.covariance_px2.shape == (3, 3)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True
