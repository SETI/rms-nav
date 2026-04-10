"""Unit tests for nav.nav_model.rings.ring_math.

Tests cover compute_antialiasing, compute_fade_integral, and compute_edge_fade.
Adapted from test_nav_model_rings_antialiasing.py and test_nav_model_rings_edge_fade.py
with the new per-pixel width API.
"""

import numpy as np
import pytest

from nav.nav_model.rings.ring_math import (
    compute_antialiasing,
    compute_edge_fade,
    compute_fade_integral,
)

# ---------------------------------------------------------------------------
# Helper: compute expected fade integral for a single pixel
# ---------------------------------------------------------------------------


def _expected_fade_integral_shade_above(
    edge_radius: float,
    width: float,  # per-pixel width for this pixel
    pixel_lower: float,
    pixel_upper: float,
    resolution: float,
) -> float:
    """Expected shade_above fade integral for one pixel (shade_sign = +1)."""
    fade_start = edge_radius
    fade_end = edge_radius + width

    eq2 = pixel_lower <= fade_start < pixel_upper
    eq3 = pixel_lower <= fade_end < pixel_upper

    if eq2 and eq3:
        a0, a1 = fade_start, fade_end
    elif fade_start < pixel_lower and fade_end > pixel_upper:
        a0, a1 = pixel_lower, pixel_upper
    elif eq2:
        a0, a1 = fade_start, pixel_upper
    elif eq3:
        a0, a1 = pixel_lower, fade_end
    else:
        return 0.0

    if a0 >= a1:
        return 0.0

    return ((1.0 + edge_radius / width) * (a1 - a0) + (a0**2 - a1**2) / (2.0 * width)) / resolution


def _expected_fade_integral_shade_below(
    edge_radius: float,
    width: float,
    pixel_lower: float,
    pixel_upper: float,
    resolution: float,
) -> float:
    """Expected shade_below fade integral for one pixel (shade_sign = -1)."""
    fade_start = edge_radius - width
    fade_end = edge_radius

    eq2 = pixel_lower < fade_end <= pixel_upper
    eq3 = pixel_lower < fade_start <= pixel_upper

    if eq2 and eq3:
        a0, a1 = fade_start, fade_end
    elif fade_end > pixel_upper and fade_start < pixel_lower:
        a0, a1 = pixel_lower, pixel_upper
    elif eq2:
        a0, a1 = pixel_lower, fade_end
    elif eq3:
        a0, a1 = fade_start, pixel_upper
    else:
        return 0.0

    if a0 >= a1:
        return 0.0

    return ((1.0 - edge_radius / width) * (a1 - a0) + (a1**2 - a0**2) / (2.0 * width)) / resolution


# ---------------------------------------------------------------------------
# compute_antialiasing
# ---------------------------------------------------------------------------


def test_antialiasing_pixel_center_at_edge() -> None:
    """Pixel center exactly at edge gives shade 0.5."""
    radii = np.array([100.0])
    resolutions = np.array([10.0])
    shade = compute_antialiasing(
        radii=radii, edge_radius=100.0, shade_above=True, resolutions=resolutions
    )
    assert shade[0] == pytest.approx(0.5, abs=1e-6)


def test_antialiasing_edge_half_resolution_above_shade_above() -> None:
    """Edge 0.5*res above pixel center: shade 1.0 for shade_above."""
    radii = np.array([100.0])
    resolutions = np.array([10.0])
    shade = compute_antialiasing(
        radii=radii, edge_radius=105.0, shade_above=True, resolutions=resolutions
    )
    assert shade[0] == pytest.approx(1.0, abs=1e-6)


def test_antialiasing_edge_half_resolution_below_shade_below() -> None:
    """Edge 0.5*res below pixel center: shade 1.0 for shade_below."""
    radii = np.array([100.0])
    resolutions = np.array([10.0])
    shade = compute_antialiasing(
        radii=radii, edge_radius=95.0, shade_above=False, resolutions=resolutions
    )
    assert shade[0] == pytest.approx(1.0, abs=1e-6)


def test_antialiasing_shade_above_direction() -> None:
    """shade_above shades the region below the edge (object above edge)."""
    radii = np.array([95.0, 100.0, 105.0])
    resolutions = np.full(3, 10.0)
    shade = compute_antialiasing(
        radii=radii, edge_radius=100.0, shade_above=True, resolutions=resolutions
    )
    assert shade[0] == pytest.approx(1.0, abs=1e-6)  # well below edge
    assert shade[1] == pytest.approx(0.5, abs=1e-6)  # at edge
    assert shade[2] == pytest.approx(0.0, abs=1e-6)  # above edge


def test_antialiasing_shade_below_direction() -> None:
    """shade_above=False shades the region above the edge."""
    radii = np.array([95.0, 100.0, 105.0])
    resolutions = np.full(3, 10.0)
    shade = compute_antialiasing(
        radii=radii, edge_radius=100.0, shade_above=False, resolutions=resolutions
    )
    assert shade[0] == pytest.approx(0.0, abs=1e-6)  # below edge
    assert shade[1] == pytest.approx(0.5, abs=1e-6)  # at edge
    assert shade[2] == pytest.approx(1.0, abs=1e-6)  # above edge


def test_antialiasing_value_range() -> None:
    """Anti-aliasing values are always in [0, max_value]."""
    radii = np.linspace(80.0, 120.0, 100)
    resolutions = np.full(100, 10.0)
    shade = compute_antialiasing(
        radii=radii, edge_radius=100.0, shade_above=True, resolutions=resolutions
    )
    assert np.all(shade >= 0.0)
    assert np.all(shade <= 1.0)


def test_antialiasing_max_value() -> None:
    """Custom max_value scales the output."""
    radii = np.array([100.0])
    resolutions = np.array([10.0])
    shade = compute_antialiasing(
        radii=radii,
        edge_radius=100.0,
        shade_above=True,
        resolutions=resolutions,
        max_value=0.5,
    )
    assert shade[0] == pytest.approx(0.25, abs=1e-6)  # 0.5 * 0.5


def test_antialiasing_2d_array() -> None:
    """Handles 2-D array inputs with correct output shape."""
    radii = np.array([[100.0, 105.0], [95.0, 100.0]])
    resolutions = np.full((2, 2), 10.0)
    shade = compute_antialiasing(
        radii=radii, edge_radius=100.0, shade_above=True, resolutions=resolutions
    )
    assert shade.shape == (2, 2)
    assert shade[0, 0] == pytest.approx(0.5, abs=1e-6)  # at edge
    assert shade[0, 1] == pytest.approx(0.0, abs=1e-6)  # above edge
    assert shade[1, 0] == pytest.approx(1.0, abs=1e-6)  # below edge by 0.5*res


# ---------------------------------------------------------------------------
# compute_fade_integral
# ---------------------------------------------------------------------------


def _direct_integral(
    a0: float,
    a1: float,
    edge_radius: float,
    width: float,
    resolution: float,
    shade_sign: float,
) -> float:
    """Direct evaluation of the unified fade integral formula."""
    return (
        (1.0 + shade_sign * edge_radius / width) * (a1 - a0)
        + shade_sign * (a0**2 - a1**2) / (2.0 * width)
    ) / resolution


def test_fade_integral_shade_above_known_value() -> None:
    """Shade-above integral produces correct value for known inputs."""
    # Integration over [100, 110], edge=90, width=20, res=10, sign=+1
    # Formula: ((1+90/20)*(10) + (100^2-110^2)/(40)) / 10
    # = (5.5*10 - 2100/40) / 10 = (55 - 52.5) / 10 = 0.25
    a0 = np.array([100.0])
    a1 = np.array([110.0])
    width = np.array([20.0])
    resolutions = np.array([10.0])
    result = compute_fade_integral(
        a0, a1, edge_radius=90.0, width=width, resolutions=resolutions, shade_sign=1.0
    )
    expected = _direct_integral(100.0, 110.0, 90.0, 20.0, 10.0, 1.0)
    assert result[0] == pytest.approx(expected, abs=1e-6)
    assert result[0] == pytest.approx(0.25, abs=1e-6)


def test_fade_integral_shade_below_known_value() -> None:
    """Shade-below integral produces correct value for known inputs."""
    # Integration over [90, 100], edge=110, width=20, res=10, sign=-1
    a0 = np.array([90.0])
    a1 = np.array([100.0])
    width = np.array([20.0])
    resolutions = np.array([10.0])
    result = compute_fade_integral(
        a0, a1, edge_radius=110.0, width=width, resolutions=resolutions, shade_sign=-1.0
    )
    expected = _direct_integral(90.0, 100.0, 110.0, 20.0, 10.0, -1.0)
    assert result[0] == pytest.approx(expected, abs=1e-6)
    assert result[0] == pytest.approx(0.25, abs=1e-6)


def test_fade_integral_zero_integration_range() -> None:
    """When a0 == a1, integral is exactly zero."""
    a0 = np.array([100.0])
    a1 = np.array([100.0])
    width = np.array([20.0])
    resolutions = np.array([10.0])
    result = compute_fade_integral(
        a0, a1, edge_radius=90.0, width=width, resolutions=resolutions, shade_sign=1.0
    )
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_fade_integral_per_pixel_width() -> None:
    """Different width values per pixel produce different integrals."""
    a0 = np.array([100.0, 100.0])
    a1 = np.array([110.0, 110.0])
    width = np.array([20.0, 30.0])
    resolutions = np.array([10.0, 10.0])
    result = compute_fade_integral(
        a0, a1, edge_radius=90.0, width=width, resolutions=resolutions, shade_sign=1.0
    )
    expected0 = _direct_integral(100.0, 110.0, 90.0, 20.0, 10.0, 1.0)
    expected1 = _direct_integral(100.0, 110.0, 90.0, 30.0, 10.0, 1.0)
    assert result[0] == pytest.approx(expected0, abs=1e-6)
    assert result[1] == pytest.approx(expected1, abs=1e-6)
    assert result[0] != pytest.approx(result[1], abs=1e-6)


# ---------------------------------------------------------------------------
# compute_edge_fade - basic fade with per-pixel width
# ---------------------------------------------------------------------------


def test_edge_fade_shade_above_uniform_resolution() -> None:
    """Uniform resolution produces fade matching per-pixel width calculation."""
    model = np.zeros((5, 5), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 2.0
    resolutions = np.full((5, 5), 10.0)
    # expected fade_width_km per pixel = 2.0 * 10.0 = 20.0 km

    radii = np.array(
        [
            [90.0, 95.0, 100.0, 105.0, 110.0],
            [92.5, 97.5, 100.0, 102.5, 107.5],
            [95.0, 100.0, 100.0, 115.0, 120.0],
            [100.0, 105.0, 110.0, 115.0, 125.0],
            [110.0, 115.0, 120.0, 125.0, 130.0],
        ]
    )

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    # Per-pixel width = fade_width_pix * resolution = 20.0 km for all pixels
    for i in range(5):
        for j in range(5):
            pixel_center = radii[i, j]
            res = resolutions[i, j]
            pixel_lower = pixel_center - res / 2.0
            pixel_upper = pixel_center + res / 2.0
            per_pixel_width = fade_width_pix * res
            expected_shade = _expected_fade_integral_shade_above(
                edge_radius, per_pixel_width, pixel_lower, pixel_upper, res
            )
            expected = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_shade_below_uniform_resolution() -> None:
    """shade_above=False with uniform resolution."""
    model = np.zeros((5, 5), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 2.0
    resolutions = np.full((5, 5), 10.0)

    radii = np.array(
        [
            [70.0, 80.0, 85.0, 90.0, 95.0],
            [75.0, 85.0, 90.0, 95.0, 100.0],
            [80.0, 90.0, 100.0, 100.0, 110.0],
            [92.5, 97.5, 100.0, 102.5, 107.5],
            [95.0, 100.0, 105.0, 110.0, 115.0],
        ]
    )

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=False,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    for i in range(5):
        for j in range(5):
            pixel_center = radii[i, j]
            res = resolutions[i, j]
            pixel_lower = pixel_center - res / 2.0
            pixel_upper = pixel_center + res / 2.0
            per_pixel_width = fade_width_pix * res
            expected_shade = _expected_fade_integral_shade_below(
                edge_radius, per_pixel_width, pixel_lower, pixel_upper, res
            )
            expected = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_varying_resolution() -> None:
    """Per-pixel varying resolution results in varying fade widths in km."""
    model = np.zeros((2, 2), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 2.0
    # Two different resolutions
    resolutions = np.array([[5.0, 10.0], [10.0, 20.0]])

    radii = np.array(
        [
            [100.0, 100.0],
            [100.0, 100.0],
        ]
    )

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    # Each pixel has different expected fade width
    for i in range(2):
        for j in range(2):
            res = resolutions[i, j]
            per_pixel_width = fade_width_pix * res
            pixel_lower = radii[i, j] - res / 2.0
            pixel_upper = radii[i, j] + res / 2.0
            expected_shade = _expected_fade_integral_shade_above(
                edge_radius, per_pixel_width, pixel_lower, pixel_upper, res
            )
            expected = np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_always_returns_result() -> None:
    """compute_edge_fade always returns an array (no None return)."""
    model = np.zeros((5, 5), dtype=np.float64)
    radii = np.full((5, 5), 100.0)
    resolutions = np.full((5, 5), 1.0)
    # Very narrow fade width pix (would have caused None in old code)
    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=100.0,
        shade_above=True,
        fade_width_pix=0.1,
        resolutions=resolutions,
        all_edge_radii=(),
    )
    assert result is not None
    assert result.shape == (5, 5)


def test_edge_fade_adds_to_existing_model() -> None:
    """Fade value is added to the existing model values."""
    model = np.ones((10, 10), dtype=np.float64) * 0.3
    edge_radius = 100.0
    fade_width_pix = 2.0
    resolutions = np.full((10, 10), 10.0)
    radii = np.full((10, 10), 100.0)

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    # All pixels at edge center: expect model + fade_integral(at edge)
    res = 10.0
    per_pixel_width = fade_width_pix * res
    pixel_lower = edge_radius - res / 2.0
    pixel_upper = edge_radius + res / 2.0
    expected_shade = _expected_fade_integral_shade_above(
        edge_radius, per_pixel_width, pixel_lower, pixel_upper, res
    )
    expected = 0.3 + np.clip(expected_shade, 0.0, 1.0)
    assert result[0, 0] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_value_range() -> None:
    """Result values per pixel: model[i,j] + shade[i,j] is always >= 0."""
    model = np.zeros((10, 10), dtype=np.float64)
    radii = np.linspace(80.0, 120.0, 100).reshape(10, 10)
    resolutions = np.full((10, 10), 10.0)

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=100.0,
        shade_above=True,
        fade_width_pix=2.0,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# compute_edge_fade - conflict detection (width reduction)
# ---------------------------------------------------------------------------


def test_edge_fade_conflict_reduces_width() -> None:
    """A nearby feature edge in the fade zone reduces the fade width."""
    model = np.zeros((20, 20), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 20.0  # large: with res=1.0, fade_width_km=20.0
    resolutions = np.full((20, 20), 1.0)

    # Create radii that span the fade region
    u_coords = np.arange(20)
    v_coords = np.arange(20)
    u_grid, _ = np.meshgrid(u_coords, v_coords)
    radii = 100.0 + (u_grid - 10) * 0.5  # 90 to 110 km

    # Feature at 110 km - within fade_width_km=20, so conflict
    all_edge_radii = ((110.0, 'IER'),)

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=all_edge_radii,
    )

    # Adjusted width = abs(110 - 100) / 2 = 5.0 km (scalar reduction)
    adjusted_km = 5.0
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            res = resolutions[i, j]
            pixel_lower = pixel_center - res / 2.0
            pixel_upper = pixel_center + res / 2.0
            # After conflict, effective width = min(original_width_km, adjusted_km)
            # original_width_km = fade_width_pix * res = 20.0 * 1.0 = 20.0
            # adjusted = min(20.0, 5.0) = 5.0
            effective_width = min(fade_width_pix * res, adjusted_km)
            expected_shade = _expected_fade_integral_shade_above(
                edge_radius, effective_width, pixel_lower, pixel_upper, res
            )
            expected = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_no_conflict_when_outside_fade_zone() -> None:
    """Feature outside the fade zone does not reduce width."""
    model = np.zeros((30, 30), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 2.0  # with res=1.0, fade_width_km=2.0
    resolutions = np.full((30, 30), 1.0)

    u_coords = np.arange(30)
    v_coords = np.arange(30)
    u_grid, _ = np.meshgrid(u_coords, v_coords)
    radii = 100.0 + (u_grid - 15) * 0.5

    # Feature at 150 km - well outside fade zone of 2.0 km
    all_edge_radii = ((150.0, 'IER'),)

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=all_edge_radii,
    )

    # No adjustment: width = fade_width_pix * res = 2.0 km
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            res = resolutions[i, j]
            pixel_lower = pixel_center - res / 2.0
            pixel_upper = pixel_center + res / 2.0
            effective_width = fade_width_pix * res
            expected_shade = _expected_fade_integral_shade_above(
                edge_radius, effective_width, pixel_lower, pixel_upper, res
            )
            expected = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)


def test_edge_fade_conflict_on_wrong_side_ignored() -> None:
    """A feature on the wrong side (opposite to shade direction) does not reduce width."""
    model = np.zeros((20, 20), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 20.0
    resolutions = np.full((20, 20), 1.0)

    u_coords = np.arange(20)
    u_grid, _ = np.meshgrid(u_coords, np.arange(20))
    radii = 100.0 + (u_grid - 10) * 0.5

    # Feature at 90 km - shade_above=True means we shade toward larger radii,
    # so a feature at 90 km (below edge) should NOT conflict
    all_edge_radii = ((90.0, 'IER'),)

    result_with_conflict = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=all_edge_radii,
    )

    result_no_conflict = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=(),
    )

    np.testing.assert_array_almost_equal(result_with_conflict, result_no_conflict)


def test_edge_fade_output_shape_preserved() -> None:
    """Output shape matches model/radii input shape."""
    model = np.zeros((7, 13), dtype=np.float64)
    radii = np.full((7, 13), 100.0)
    resolutions = np.full((7, 13), 5.0)

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=100.0,
        shade_above=True,
        fade_width_pix=1.0,
        resolutions=resolutions,
        all_edge_radii=(),
    )
    assert result.shape == (7, 13)


def test_edge_fade_multiple_conflicts_takes_tightest() -> None:
    """When multiple features conflict, the closest one sets the adjusted width."""
    model = np.zeros((20, 20), dtype=np.float64)
    edge_radius = 100.0
    fade_width_pix = 30.0  # wide: fade_width_km=30 @ res=1
    resolutions = np.full((20, 20), 1.0)

    u_grid, _ = np.meshgrid(np.arange(20), np.arange(20))
    radii = 100.0 + (u_grid - 10) * 0.5

    # Features at 115 (distance=15, half=7.5) and 108 (distance=8, half=4)
    # Tightest: 108 km gives half_dist=4
    all_edge_radii = ((115.0, 'OER'), (108.0, 'IER'))

    result = compute_edge_fade(
        model=model,
        radii=radii,
        edge_radius=edge_radius,
        shade_above=True,
        fade_width_pix=fade_width_pix,
        resolutions=resolutions,
        all_edge_radii=all_edge_radii,
    )

    # Effective adjusted_km = min(7.5, 4.0) = 4.0
    adjusted_km = 4.0
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            res = resolutions[i, j]
            pixel_lower = pixel_center - res / 2.0
            pixel_upper = pixel_center + res / 2.0
            effective_width = min(fade_width_pix * res, adjusted_km)
            expected_shade = _expected_fade_integral_shade_above(
                edge_radius, effective_width, pixel_lower, pixel_upper, res
            )
            expected = np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected, abs=1e-6)
