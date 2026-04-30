"""Tests for ``nav.nav_model.nav_model_rings`` helpers and emission gates.

The full ``NavModelRings.create_model`` path requires a live ``oops``
backplane and so is exercised by integration tests against the image
library.  These unit tests cover the polyline-extraction helper, the
straight-line classifier, the bbox helper, and the per-feature
reliability sigmoids.
"""

from __future__ import annotations

import numpy as np
import pytest

from nav.config.config import Config
from nav.nav_model.nav_model_rings import (
    FLAT_CURVATURE_THRESHOLD_PX,
    RING_EDGE_DEFAULT_RELIABILITY,
    RING_EDGE_SIGMA_ALONG_PX,
    _composite_ring_renderings,
    _is_straight_line,
    _mask_bbox,
    _polyline_from_edge_mask,
    _radial_extent_px,
    _require_positive_finite_planet_scalar,
    _ring_annulus_emission_params,
    _ring_annulus_reliability,
    _ring_edge_reliability,
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
    img_a = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    mask_a = img_a > 0.0
    img_b = np.array(
        [[0.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
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
    # Per-pixel max keeps the brighter contribution where they overlap
    # would have collided (no overlap here, so each pixel's value is
    # whichever input had it set).
    assert composite_img[0, 1] == pytest.approx(1.0)
    assert composite_img[0, 0] == pytest.approx(0.5)
    assert composite_img[1, 1] == pytest.approx(2.0)


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
