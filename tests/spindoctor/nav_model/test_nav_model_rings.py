"""Tests for ``spindoctor.nav_model.nav_model_rings`` helpers and emission gates.

The full ``NavModelRings.create_model`` path requires a live ``oops``
backplane and so is exercised by integration tests against the image
library.  These unit tests cover the polyline-extraction helper, the
straight-line classifier, the bbox helper, and the per-feature
reliability sigmoids.
"""

from __future__ import annotations

import numpy as np
import pytest

from spindoctor.config.config import Config
from spindoctor.nav_model.nav_model_rings import (
    NavModelRings,
    _require_positive_finite_planet_scalar,
    _ring_annulus_emission_params,
)
from spindoctor.nav_model.ring_emission import (
    RING_EDGE_DEFAULT_RELIABILITY,
    RING_EDGE_SIGMA_ALONG_PX,
    _aggregate_annulus_orbit_terms,
    _ring_annulus_reliability,
    _ring_edge_reliability,
)
from spindoctor.nav_model.ring_polyline import (
    FLAT_CURVATURE_THRESHOLD_PX,
    _composite_ring_renderings,
    _is_straight_line,
    _mask_bbox,
    _polyline_from_edge_mask,
    _radial_extent_px,
)


def test_constants_have_design_values() -> None:
    """Module-level constants match their design defaults."""
    assert pytest.approx(0.7) == RING_EDGE_DEFAULT_RELIABILITY
    assert pytest.approx(0.5) == RING_EDGE_SIGMA_ALONG_PX
    assert pytest.approx(1.0) == FLAT_CURVATURE_THRESHOLD_PX


def test_polyline_from_edge_mask_returns_one_vertex_per_true_pixel() -> None:
    """Each True pixel in the mask becomes one polyline vertex."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 4:9] = True  # 5 vertices along a horizontal line
    vertices, normals = _polyline_from_edge_mask(mask)
    assert vertices.shape == (5, 2)
    assert normals.shape == (5, 2)


def test_polyline_from_edge_mask_empty_returns_empty_arrays() -> None:
    """An all-False mask returns ``(0, 2)``-shaped arrays."""
    mask = np.zeros((5, 5), dtype=bool)
    vertices, normals = _polyline_from_edge_mask(mask)
    assert vertices.shape == (0, 2)
    assert normals.shape == (0, 2)


def test_polyline_normals_are_unit_length() -> None:
    """Each normal has length 1 (rounded to floating tolerance)."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 4:9] = True
    _, normals = _polyline_from_edge_mask(mask)
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0)


def test_radial_extent_zero_for_point_polyline() -> None:
    """A polyline with only one vertex has zero radial extent."""
    vertices = np.array([[1.0, 1.0]])
    normals = np.array([[1.0, 0.0]])
    assert _radial_extent_px(vertices, normals) == pytest.approx(0.0)


def test_radial_extent_along_normal_axis() -> None:
    """Extent equals the projection range onto the mean normal."""
    vertices = np.array([[0.0, 0.0], [3.0, 0.0]])
    normals = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert _radial_extent_px(vertices, normals) == pytest.approx(3.0)


def test_is_straight_line_for_collinear_points() -> None:
    """Three collinear points are classified as a straight line."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert _is_straight_line(vertices)


def test_is_straight_line_for_curved_points() -> None:
    """Points with curvature above the threshold are not flagged straight.

    The SVD-based test projects the centred points onto the smallest
    singular direction and compares the maximum deviation to
    ``FLAT_CURVATURE_THRESHOLD_PX = 1.0``.  We construct a triangle whose
    apex is far enough from the line connecting the endpoints that the
    deviation comfortably clears the threshold.
    """
    vertices = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    assert not _is_straight_line(vertices)


def test_is_straight_line_short_polyline() -> None:
    """Polylines with < 3 vertices are trivially straight."""
    assert _is_straight_line(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_mask_bbox_for_empty_mask() -> None:
    """An all-False mask returns ``(0, 0, 0, 0)``."""
    mask = np.zeros((5, 5), dtype=bool)
    assert _mask_bbox(mask) == (0, 0, 0, 0)


def test_mask_bbox_for_simple_rectangle() -> None:
    """The bbox is the half-open envelope of the True pixels."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    assert _mask_bbox(mask) == (2, 3, 5, 7)


def test_ring_edge_reliability_caps_at_one() -> None:
    """Pathologically generous inputs are clipped to 1.0."""
    out = _ring_edge_reliability(
        catalog_default=10.0,
        visible_arc_fraction=10.0,
        shadow_occluded_fraction=0.0,
        is_straight_line=False,
    )
    assert out == pytest.approx(1.0)


def test_ring_edge_reliability_penalises_straight_lines() -> None:
    """Straight-line edges receive a 0.7 multiplier."""
    curved = _ring_edge_reliability(
        catalog_default=0.7,
        visible_arc_fraction=1.0,
        shadow_occluded_fraction=0.0,
        is_straight_line=False,
    )
    straight = _ring_edge_reliability(
        catalog_default=0.7,
        visible_arc_fraction=1.0,
        shadow_occluded_fraction=0.0,
        is_straight_line=True,
    )
    assert straight == pytest.approx(curved * 0.7)


def test_ring_edge_reliability_drops_with_shadow() -> None:
    """Higher shadow occlusion drives reliability down."""
    low = _ring_edge_reliability(
        catalog_default=0.7,
        visible_arc_fraction=1.0,
        shadow_occluded_fraction=0.0,
        is_straight_line=False,
    )
    high = _ring_edge_reliability(
        catalog_default=0.7,
        visible_arc_fraction=1.0,
        shadow_occluded_fraction=0.5,
        is_straight_line=False,
    )
    assert high < low


def test_ring_annulus_reliability_zero_for_empty_annulus() -> None:
    """An annulus with zero constituent edges has zero reliability."""
    assert _ring_annulus_reliability(constituent_count=0, radial_extent_px=100.0) == 0.0


def test_ring_annulus_reliability_returns_clamped_value() -> None:
    """The annulus reliability is in ``[0, 1]`` for sensible inputs."""
    out = _ring_annulus_reliability(constituent_count=3, radial_extent_px=20.0)
    assert 0.0 <= out <= 1.0


def test_ring_annulus_reliability_increases_with_constituent_count() -> None:
    """More constituent edges -> higher reliability (up to the cap)."""
    low = _ring_annulus_reliability(constituent_count=1, radial_extent_px=100.0)
    high = _ring_annulus_reliability(constituent_count=5, radial_extent_px=100.0)
    assert high > low


def test_ring_annulus_reliability_increases_with_radial_extent() -> None:
    """A wider annulus has higher reliability via the sigmoid-extent factor."""
    narrow = _ring_annulus_reliability(constituent_count=3, radial_extent_px=10.0)
    wide = _ring_annulus_reliability(constituent_count=3, radial_extent_px=200.0)
    assert wide > narrow


def test_composite_ring_renderings_unions_masks_and_takes_max_image() -> None:
    """The composite is the OR of input masks and the per-pixel max of input images."""
    extfov_shape = (4, 4)
    # (2, 2) is shared between the two inputs at different intensities to
    # force the per-pixel-max path (rather than last-writer-wins).
    img_a = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    mask_a = img_a > 0.0
    img_b = np.array(
        [[0.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    mask_b = img_b > 0.0
    composite_img, composite_mask = _composite_ring_renderings(
        [(img_a, mask_a, 'a', 1.0), (img_b, mask_b, 'b', 1.0)],
        extfov_shape=extfov_shape,
    )
    # Both rings' pixels appear in the union mask.
    assert bool(composite_mask[0, 1])
    assert bool(composite_mask[0, 0])
    assert bool(composite_mask[1, 1])
    assert bool(composite_mask[2, 2])
    # Non-overlapping pixels keep whichever input had them set.
    assert composite_img[0, 1] == pytest.approx(1.0)
    assert composite_img[0, 0] == pytest.approx(0.5)
    assert composite_img[1, 1] == pytest.approx(2.0)
    # Overlapping pixel takes the brighter input (3.0 from img_a, not 1.5
    # from img_b) — this is the per-pixel-max contract, not last-writer-wins.
    assert composite_img[2, 2] == pytest.approx(3.0)


def test_composite_ring_renderings_empty_input_returns_zero_arrays() -> None:
    """An empty list returns zero-shaped composite of the requested extfov shape."""
    composite_img, composite_mask = _composite_ring_renderings([], extfov_shape=(8, 8))
    assert composite_img.shape == (8, 8)
    assert composite_mask.shape == (8, 8)
    assert not composite_mask.any()


def test_ring_annulus_emission_params_loads_saturn_block() -> None:
    """Saturn's planet-specific block in the bundled YAML is loaded."""
    config = Config()
    max_radial_px, kmpp_threshold = _ring_annulus_emission_params(config, 'SATURN')
    assert max_radial_px == pytest.approx(5.0)
    assert kmpp_threshold == pytest.approx(1000.0)


def test_ring_annulus_emission_params_loads_jupiter_block() -> None:
    """Jupiter's planet-specific block uses a tighter km/px threshold."""
    config = Config()
    max_radial_px, kmpp_threshold = _ring_annulus_emission_params(config, 'JUPITER')
    assert max_radial_px == pytest.approx(5.0)
    # Jupiter's main ring is ~10x narrower than Saturn's; threshold lower.
    assert kmpp_threshold == pytest.approx(200.0)


def test_ring_annulus_emission_params_falls_back_to_default_block() -> None:
    """An unknown planet falls back to the ``default`` block."""
    config = Config()
    max_radial_px, kmpp_threshold = _ring_annulus_emission_params(config, 'UNKNOWN')
    assert max_radial_px == pytest.approx(5.0)
    assert kmpp_threshold == pytest.approx(1000.0)


def test_require_positive_finite_planet_scalar_returns_value() -> None:
    """The validator returns the float when input is finite and positive."""
    out = _require_positive_finite_planet_scalar(
        'SATURN', 'fade_width_pix', {'fade_width_pix': 4.0}
    )
    assert out == pytest.approx(4.0)


def test_require_positive_finite_planet_scalar_rejects_missing_key() -> None:
    """A missing key raises ``ValueError`` naming the key."""
    with pytest.raises(ValueError, match='Missing required ring configuration key'):
        _require_positive_finite_planet_scalar('SATURN', 'fade_width_pix', {})


def test_require_positive_finite_planet_scalar_rejects_zero() -> None:
    """A zero value is invalid (non-positive)."""
    with pytest.raises(ValueError, match=r'Invalid fade_width_pix 0\.0'):
        _require_positive_finite_planet_scalar('SATURN', 'fade_width_pix', {'fade_width_pix': 0.0})


def test_require_positive_finite_planet_scalar_rejects_inf() -> None:
    """A non-finite value raises ``ValueError``."""
    with pytest.raises(ValueError, match='must be finite'):
        _require_positive_finite_planet_scalar(
            'SATURN', 'fade_width_pix', {'fade_width_pix': float('inf')}
        )


def test_require_positive_finite_planet_scalar_rejects_bool() -> None:
    """A bool value is not accepted as a numeric scalar."""
    with pytest.raises(ValueError, match='got bool'):
        _require_positive_finite_planet_scalar('SATURN', 'fade_width_pix', {'fade_width_pix': True})


def test_require_positive_finite_planet_scalar_rejects_non_numeric() -> None:
    """A non-numeric value raises ``ValueError``."""
    with pytest.raises(ValueError, match='expected a finite numeric value'):
        _require_positive_finite_planet_scalar(
            'SATURN', 'fade_width_pix', {'fade_width_pix': 'big'}
        )


def test_polyline_normals_are_radially_signed_with_a_radius_backplane() -> None:
    """A rasterized closed ring emits outward-radial normals, not scan-order ones.

    The mask-neighbour test fixes only the normal AXIS: it probes ``v - 1``
    before ``v + 1`` and ``u - 1`` before ``u + 1``, so on a closed ring the
    emitted signs follow scan order and rasterization.  Measured on this
    fixture without a radius backplane, the mean dot with the true outward
    radial is ~0.00 (the axis is right, the sense is random) even though the
    mean ABSOLUTE dot is ~0.83.

    That matters beyond tidiness: ``RingEdgeNav``'s orbit-uncertainty channel
    sums the normals, so random signs fabricate coherence on geometry that
    should cancel -- a closed ring measured a sensitivity of ~0.85 where the
    true value is ~0, pointing along the rasterizer's quadrant bias.
    """
    from spindoctor.nav_technique.ring_edge_geometry import _absorbed_orbit_sensitivity

    size = 401
    center = (size - 1) / 2.0
    vs, us = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    radius = np.hypot(vs - center, us - center)
    mask = np.abs(radius - 120.0) <= 0.5
    ring_radius = radius.astype(np.float64)

    vertices, normals = _polyline_from_edge_mask(mask, ring_radius)
    true_radial = np.stack([vertices[:, 0] - center, vertices[:, 1] - center], axis=-1)
    true_radial /= np.linalg.norm(true_radial, axis=1, keepdims=True)
    dots = np.sum(normals * true_radial, axis=1)
    # Every normal points outward: the signed mean matches the absolute mean
    # (which is below 1 only because the axis is quantized to eight directions).
    assert float(np.mean(dots)) == pytest.approx(float(np.mean(np.abs(dots))), abs=1.0e-9)
    assert float(np.min(dots)) > 0.0

    # And the closed-ring limit the orbit channel relies on is now reachable.
    sensitivity = _absorbed_orbit_sensitivity(normals, np.ones(len(normals)))
    assert float(np.linalg.norm(sensitivity)) < 0.02


def test_polyline_normals_unsigned_without_a_radius_backplane() -> None:
    """Without the backplane the historical unsigned axis is emitted.

    The distinguishing property is the SIGN, not the length: a fully
    radially-signed result is also unit length, so asserting length alone
    would pass either way.  Without the backplane the sign comes from the
    mask-neighbour scan order, which on a closed ring puts a large share of
    the normals on the inward side -- a signed result would have none.
    """
    size = 201
    center = (size - 1) / 2.0
    vs, us = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    radius = np.hypot(vs - center, us - center)
    mask = np.abs(radius - 60.0) <= 0.5
    vertices, normals = _polyline_from_edge_mask(mask)
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0)

    true_radial = np.stack([vertices[:, 0] - center, vertices[:, 1] - center], axis=-1)
    true_radial /= np.linalg.norm(true_radial, axis=1, keepdims=True)
    dots = np.sum(normals * true_radial, axis=1)
    # Exactly half the ring's vertices come out pointing INWARD: the scan
    # tests v - 1 before v + 1 and u - 1 before u + 1, so the sign is set by
    # which neighbour is probed first, not by the geometry.
    assert int(np.count_nonzero(dots < 0.0)) == 190
    assert int(np.count_nonzero(dots > 0.0)) == 190

    # The same fixture through the signed path leaves none pointing inward,
    # which is the property that separates the two branches.
    _signed_vertices, signed_normals = _polyline_from_edge_mask(mask, radius.astype(np.float64))
    signed_dots = np.sum(signed_normals * true_radial, axis=1)
    assert int(np.count_nonzero(signed_dots < 0.0)) == 0


def test_aggregate_annulus_orbit_terms_vertex_weighted_mean() -> None:
    """Per-edge orbit sigmas combine by vertex share; normals concatenate."""
    n_a = np.tile(np.array([1.0, 0.0]), (30, 1))
    n_b = np.tile(np.array([0.0, 1.0]), (10, 1))
    normals, sigma = _aggregate_annulus_orbit_terms([(n_a, 30, 2.0), (n_b, 10, 6.0)])
    assert normals.shape == (40, 2)
    # (30 * 2 + 10 * 6) / 40 = 3.0
    assert sigma == pytest.approx(3.0)


def test_aggregate_annulus_orbit_terms_empty_is_no_channel() -> None:
    """No terms yields an empty normal set and a zero sigma."""
    normals, sigma = _aggregate_annulus_orbit_terms([])
    assert normals.shape == (0, 2)
    assert sigma == 0.0


###############################################################################
#
# _drop_occluded_edges — planet-occlusion filtering of ring-edge overlays
#
###############################################################################


def _rings_model_with_occlusion(
    occluded: np.ndarray | None,
) -> NavModelRings:
    """Build a bare NavModelRings carrying only an occlusion mask.

    ``_drop_occluded_edges`` touches no other state, so the model is
    constructed without an obs or the full render pipeline.

    Parameters:
        occluded: Ext-FOV occlusion mask (or None for "no signal").

    Returns:
        A NavModelRings instance with ``_ring_occluded_ext`` set.
    """
    model = NavModelRings.__new__(NavModelRings)
    model._ring_occluded_ext = occluded
    return model


def test_drop_occluded_edges_removes_fully_hidden_edge() -> None:
    """An edge entirely behind the planet is dropped from the overlay."""
    occluded = np.zeros((10, 10), dtype=bool)
    occluded[:, 5:] = True  # planet hides the right half
    hidden = np.zeros((10, 10), dtype=bool)
    hidden[2:4, 6:8] = True  # wholly inside the occluded region
    model = _rings_model_with_occlusion(occluded)
    kept = model._drop_occluded_edges([(hidden, 'A ring', 'a_outer')])
    assert kept == []


def test_drop_occluded_edges_keeps_visible_edge_intact() -> None:
    """An edge entirely in front of the planet survives unchanged."""
    occluded = np.zeros((10, 10), dtype=bool)
    occluded[:, 5:] = True
    visible = np.zeros((10, 10), dtype=bool)
    visible[2:4, 1:3] = True  # wholly outside the occluded region
    model = _rings_model_with_occlusion(occluded)
    kept = model._drop_occluded_edges([(visible, 'A ring', 'a_outer')])
    assert len(kept) == 1


def test_drop_occluded_edges_trims_straddling_edge() -> None:
    """A straddling edge keeps only its visible pixels."""
    occluded = np.zeros((10, 10), dtype=bool)
    occluded[:, 5:] = True
    straddle = np.zeros((10, 10), dtype=bool)
    straddle[4, 3:8] = True  # columns 3,4 visible; 5,6,7 occluded
    model = _rings_model_with_occlusion(occluded)
    kept = model._drop_occluded_edges([(straddle, 'A ring', 'a_outer')])
    trimmed_mask = kept[0][0]
    assert bool(trimmed_mask[4, 3])
    assert bool(trimmed_mask[4, 4])


def test_drop_occluded_edges_trims_away_hidden_pixels() -> None:
    """The occluded pixels of a straddling edge are cleared."""
    occluded = np.zeros((10, 10), dtype=bool)
    occluded[:, 5:] = True
    straddle = np.zeros((10, 10), dtype=bool)
    straddle[4, 3:8] = True
    model = _rings_model_with_occlusion(occluded)
    kept = model._drop_occluded_edges([(straddle, 'A ring', 'a_outer')])
    trimmed_mask = kept[0][0]
    assert not bool(trimmed_mask[4, 6])


def test_drop_occluded_edges_no_mask_returns_input() -> None:
    """With no occlusion signal the edge list is returned unchanged."""
    edge = np.zeros((10, 10), dtype=bool)
    edge[4, 3:8] = True
    edge_info = [(edge, 'A ring', 'a_outer')]
    model = _rings_model_with_occlusion(None)
    kept = model._drop_occluded_edges(edge_info)
    assert kept is edge_info
