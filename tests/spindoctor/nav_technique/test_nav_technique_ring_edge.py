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
    aggregate_edge_normal_angle_deg,
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
    # The forged fit anchors only edge A — one third of the model
    # vertices — so the inlier fraction sits far below the 0.5 gate.
    # The per-edge diagnostics record the misalignment signature.
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(50.0)
    assert result.diagnostics.per_edge_dt_rms_summed > 50.0
    assert result.spurious is True


def test_ring_edge_nav_multi_edge_with_undetected_dominant_edges_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Absent edges waive the aggregate inlier-fraction veto.

    Models the production false flag (Cassini C ring N1467344214): three
    RING_EDGE features feed the LM, one curved edge fits at sub-pixel
    RMS with every vertex an inlier, and the other two -- holding the
    majority of the model vertices -- are undetectable in the image:
    their vertices sit tens of pixels from every detected edge pixel and
    are fully Tukey-rejected.  The aggregate inlier fraction falls below
    the 0.5 gate even though the fused offset is correct.  Because the
    rejected edges are absent (median DT residual far above the waiver
    threshold) rather than mis-aligned against detected structure, the
    mis-convergence veto must be waived and the result kept; the
    per-edge diagnostics still record the missing edges' residual
    signature unchanged.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 120)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 120)
    vertices_c, normals_c = circle_polyline((100.0, 100.0), 50.0, 120)
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

    # Edge A fits at 0.2 px with every vertex an inlier; edges B and C
    # sit 17 and 46 px from every detected edge -- absent -- and are fully
    # Tukey-rejected.  Aggregate inlier fraction is 120 / 360 = 0.33,
    # below the 0.5 gate.
    n_per_edge = 120
    residuals = np.full(3 * n_per_edge, 0.2, dtype=np.float64)
    residuals[n_per_edge : 2 * n_per_edge] = 17.0
    residuals[2 * n_per_edge :] = 46.0
    weights = np.zeros(3 * n_per_edge, dtype=np.float64)
    weights[:n_per_edge] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=n_per_edge,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    # Pin the coarse seed next to the forged LM offset so the unrelated
    # LM-displacement gate stays quiet.
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (1, -1),
    )

    result = technique.navigate([feat_a, feat_b, feat_c], context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    # The diagnostics still expose the missing edges (stats unchanged);
    # only the gate decision is robust to them.
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(46.0)
    assert result.spurious is False


def test_ring_edge_nav_rejected_edge_on_detected_structure_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A rejected edge lying ON a detected image edge blocks the waiver.

    Models the wrong-ring lock (Cassini Tethys N1572472169): the fused
    fit anchors one edge cleanly, but another rejected edge has a
    sub-pixel median DT residual -- it sits on a detected image edge the
    robust fit disagrees with.  That internal inconsistency is the
    mis-convergence signature the veto exists for, so the result must
    stay spurious even though one edge clears the per-edge gate.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices_a, normals_a = circle_polyline((100.0, 100.0), 30.0, 60)
    vertices_b, normals_b = circle_polyline((100.0, 100.0), 40.0, 200)
    feat_a = make_ring_feature(
        'locked', vertices=vertices_a, outward_normals=normals_a, is_straight_line=False
    )
    feat_b = make_ring_feature(
        'disputed', vertices=vertices_b, outward_normals=normals_b, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Edge A: all 60 vertices inliers at 0.2 px.  Edge B: median DT
    # residual 0.4 px (it lies along detected structure) but only 60 of
    # its 200 vertices survive Tukey.  Aggregate inlier fraction is
    # 120 / 260 = 0.46, below the 0.5 gate; edge B is not well-fit and
    # not absent, so the veto must stand.
    residuals = np.full(260, 0.4, dtype=np.float64)
    residuals[:60] = 0.2
    weights = np.zeros(260, dtype=np.float64)
    weights[:120] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=120,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (1, -1),
    )

    result = technique.navigate([feat_a, feat_b], context)
    assert result.spurious is True


def test_ring_edge_nav_rank1_well_fit_subset_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """The waiver is rank-aware: a rank-1 constrained fit cannot carry it.

    When the surviving vertices constrain only one offset axis (the
    translation covariance is rank-1), the well-fit subset cannot vouch
    for the full 2-D offset on its own; the aggregate inlier-fraction
    veto must stand even when the rejected curved edge is absent from
    the image.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    flat_v, flat_n = flat_polyline(130.5, 20.0, 180.0, 60)
    curved_v, curved_n = circle_polyline((100.0, 100.0), 50.0, 200)
    flat_feat = make_ring_feature(
        'flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True
    )
    curved_feat = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # The straight edge fits cleanly (60 / 60); the curved edge is
    # absent (median 46 px, zero inliers).  Aggregate fraction is
    # 60 / 260 = 0.23, below the gate; the surviving vertices only
    # constrain the normal axis (rank-1 covariance), so the waiver
    # must not fire.
    residuals = np.full(260, 0.2, dtype=np.float64)
    residuals[60:] = 46.0
    weights = np.zeros(260, dtype=np.float64)
    weights[:60] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=60,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (1, -1),
    )

    result = technique.navigate([flat_feat, curved_feat], context)
    assert result.spurious is True


def test_ring_edge_nav_marks_spurious_when_every_edge_fits_poorly(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """No edge clears the per-edge gate, so the fraction veto stands.

    A genuinely mis-converged multi-edge fit where every edge retains
    only a few stray inliers must stay spurious: the well-fit-edge
    quorum is zero, so the aggregate inlier-fraction veto applies
    exactly as before the absent-edge exemption.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    features = []
    for name, radius in (('inner', 30.0), ('middle', 40.0), ('outer', 50.0)):
        vertices, normals = circle_polyline((100.0, 100.0), radius, 60)
        features.append(
            make_ring_feature(
                name, vertices=vertices, outward_normals=normals, is_straight_line=False
            )
        )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Every edge keeps only 5 of its 60 vertices as inliers: aggregate
    # fraction 15 / 180 = 0.083 and no edge reaches the 0.5 per-edge
    # gate (nor the 6-inlier per-edge minimum).
    n_per_edge = 60
    residuals = np.full(3 * n_per_edge, 25.0, dtype=np.float64)
    weights = np.zeros(3 * n_per_edge, dtype=np.float64)
    for edge_index in range(3):
        start = edge_index * n_per_edge
        residuals[start : start + 5] = 0.2
        weights[start : start + 5] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=15,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (1, -1),
    )

    result = technique.navigate(features, context)
    assert result.spurious is True


def test_ring_edge_nav_single_edge_low_inlier_fraction_stays_spurious(
    monkeypatch: pytest.MonkeyPatch,
    disc_image: DiscImageFactory,
    circle_polyline: CirclePolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A single-edge fit below the inlier-fraction gate is spurious, unchanged.

    One edge can never reach the two-edge well-fit quorum, so the
    absent-edge exemption never applies to a single-edge fit: the aggregate
    inlier-fraction gate behaves exactly as before.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = disc_image(shape, (100.0, 100.0), 30.0)
    vertices, normals = circle_polyline((100.0, 100.0), 30.0, 60)
    feature = make_ring_feature(
        'only', vertices=vertices, outward_normals=normals, is_straight_line=False
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # 10 of 60 vertices survive Tukey: fraction 0.167, below the gate.
    residuals = np.full(60, 25.0, dtype=np.float64)
    residuals[:10] = 0.2
    weights = np.zeros(60, dtype=np.float64)
    weights[:10] = 1.0
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(0.7, -1.3),
        rotation_rad=0.0,
        covariance=np.eye(2, dtype=np.float64) * 0.25,
        residuals_px=residuals,
        weights=weights,
        rms_px=0.2,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=10,
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (1, -1),
    )

    result = technique.navigate([feature], context)
    assert result.spurious is True


def test_ring_edge_nav_flat_parallel_edges_with_minority_snaps_not_spurious(
    monkeypatch: pytest.MonkeyPatch,
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """A correct flat multi-edge fit with an outlier minority passes the gate.

    Models the ``ring_only_flat`` production regression (#203, Cassini
    N1863267799): parallel straight Keeler-gap edges fit cleanly while a
    minority of vertices (an edge too faint to detect, vertices snapping
    to a neighbouring parallel edge 9-30 px away) are Tukey outliers.
    Those outliers inflate every raw per-edge residual statistic past any
    sigma-derived threshold even though the joint fit is correct — which is
    exactly what used to gate every flat ansa frame as spurious.  The fit
    still anchors 85% of the model vertices, so the inlier-fraction gate
    passes and the result survives with ``is_rank_1=True`` for the ensemble
    to surface as ``rank_1_only``.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    features = []
    for name, row in (('inner', 80.5), ('middle', 100.5), ('outer', 120.5)):
        vertices, outward = flat_polyline(row, 20.0, 180.0, 60)
        features.append(
            make_ring_feature(
                name, vertices=vertices, outward_normals=outward, is_straight_line=True
            )
        )
    technique = RingEdgeNav()
    context = make_nav_context(image)

    # Per-edge residuals: 85% of vertices at the ~1 px fit residual, 15%
    # snapped to a parallel neighbour 20 px away.  Raw per-edge RMS is
    # sqrt(0.85*1 + 0.15*400) ~ 7.8 px — any sigma-derived residual gate
    # would fire — but the inlier fraction is 0.85, well above the gate.
    n_per_edge = 60
    per_edge = np.full(n_per_edge, 1.0, dtype=np.float64)
    per_edge[: int(0.15 * n_per_edge)] = 20.0
    residuals = np.concatenate([per_edge] * 3)
    weights = np.ones(residuals.size, dtype=np.float64)
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-1.5, 0.0),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=weights,
        rms_px=1.0,
        raw_rms_px=float(np.sqrt(np.mean(residuals**2))),
        iterations=10,
        converged=True,
        inlier_count=int(residuals.size * 0.85),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    # Pin the coarse seed next to the forged LM offset so the unrelated
    # LM-displacement gate stays quiet; this test is about the per-edge gate.
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (-2, 0),
    )

    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.edge_count == 3
    assert result.diagnostics.per_edge_dt_median_max == pytest.approx(1.0)
    assert result.diagnostics.per_edge_dt_rms_mean > 3.0
    assert result.spurious is False
    assert result.diagnostics.is_rank_1 is True


def test_ring_edge_nav_rank1_tangent_slide_not_gated(
    monkeypatch: pytest.MonkeyPatch,
    horizontal_step_image: HorizontalStepImageFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
    make_nav_context: NavContextFactory,
) -> None:
    """Tangent slide on a rank-1 scene trips neither at_edge nor displacement.

    Nothing constrains the along-edge axis of an all-straight fit, so the
    LM may drift to the search-window boundary along the tangent (Cassini
    N1863267979: dv slid to the margin while the observable normal
    component was mid-window).  Both the at-edge check and the
    LM-displacement spurious gate must therefore be evaluated on the
    edge-normal component only.
    """
    from spindoctor.nav_technique import dt_fitting, nav_technique_ring_edge

    shape = (200, 200)
    image = horizontal_step_image(shape, 100.0)
    vertices, outward = flat_polyline(101.5, 20.0, 180.0, 60)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    technique = RingEdgeNav()
    context = make_nav_context(image)  # extfov margins (32, 32)

    # Horizontal edge: normal is +v, tangent is +u.  The forged fit slid
    # 31.5 px along the tangent — at the (32 - 1) px per-axis at-edge
    # boundary and far past the 4 px displacement gate — while the
    # observable normal component stays a benign -1.5 px.
    residuals = np.full(vertices.shape[0], 0.5, dtype=np.float64)
    forged_result = dt_fitting.LMRefineResult(
        offset_vu=(-1.5, 31.5),
        rotation_rad=0.0,
        covariance=np.array([[0.04, 0.0], [0.0, 1.0e9]], dtype=np.float64),
        residuals_px=residuals,
        weights=np.ones(residuals.size, dtype=np.float64),
        rms_px=0.5,
        raw_rms_px=0.5,
        iterations=10,
        converged=True,
        inlier_count=int(residuals.size),
        degenerate=False,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'lm_subpixel_refine',
        lambda **_kwargs: forged_result,
    )
    monkeypatch.setattr(
        nav_technique_ring_edge,
        'coarse_ncc_search',
        lambda *_args, **_kwargs: (-2, 0),
    )

    result = technique.navigate([feature], context)
    assert result.at_edge is False
    assert result.spurious is False
    assert isinstance(result.diagnostics, RingEdgeDiagnostics)
    assert result.diagnostics.is_rank_1 is True


def test_aggregate_edge_normal_angle_all_straight_horizontal(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """Horizontal straight edges have a +v-aligned normal: angle 0 deg."""
    vertices, outward = flat_polyline(100.5, 20.0, 180.0, 60)
    feature = make_ring_feature(
        'flat', vertices=vertices, outward_normals=outward, is_straight_line=True
    )
    angle = aggregate_edge_normal_angle_deg([feature])
    assert angle is not None
    assert angle == pytest.approx(0.0, abs=1.0e-6)


def test_aggregate_edge_normal_angle_polarity_sign_independent(
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """Two parallel edges with opposite normal senses do not cancel."""
    vertices_a, outward_a = flat_polyline(80.5, 20.0, 180.0, 60)
    vertices_b, outward_b = flat_polyline(120.5, 20.0, 180.0, 60)
    feat_a = make_ring_feature(
        'inner', vertices=vertices_a, outward_normals=outward_a, is_straight_line=True
    )
    feat_b = make_ring_feature(
        'outer', vertices=vertices_b, outward_normals=-outward_b, is_straight_line=True
    )
    angle = aggregate_edge_normal_angle_deg([feat_a, feat_b])
    assert angle is not None
    assert angle == pytest.approx(0.0, abs=1.0e-6)


def test_aggregate_edge_normal_angle_none_when_any_edge_curved(
    circle_polyline: CirclePolylineFactory,
    flat_polyline: FlatPolylineFactory,
    make_ring_feature: NavFeatureFactory,
) -> None:
    """A mixed straight + curved scene is full-rank: no constraint seed."""
    curved_v, curved_n = circle_polyline((100.0, 100.0), 30.0, 60)
    flat_v, flat_n = flat_polyline(150.5, 20.0, 180.0, 60)
    curved = make_ring_feature(
        'curved', vertices=curved_v, outward_normals=curved_n, is_straight_line=False
    )
    flat = make_ring_feature('flat', vertices=flat_v, outward_normals=flat_n, is_straight_line=True)
    assert aggregate_edge_normal_angle_deg([curved, flat]) is None


def test_aggregate_edge_normal_angle_none_without_ring_edges() -> None:
    """No ring-edge features means no seed."""
    assert aggregate_edge_normal_angle_deg([]) is None


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
