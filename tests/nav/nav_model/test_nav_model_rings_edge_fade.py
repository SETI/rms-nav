"""Unit tests for ring model edge fade computation.

Tests the _compute_edge_fade() method with different fade widths, directions,
and conflict detection.
"""

import numpy as np
import pytest

from nav.nav_model.nav_model_rings import NavModelRings


class MockObservation:
    """Mock observation for testing."""

    def __init__(self):
        self.closest_planet = 'SATURN'
        self.midtime = 0.0
        self.extdata_shape_vu = (100, 100)

    def make_extfov_zeros(self):
        return np.zeros((100, 100), dtype=np.float64)

    def make_extfov_false(self):
        return np.zeros((100, 100), dtype=bool)


@pytest.fixture
def ring_model():
    """Create a NavModelRings instance for testing."""
    obs = MockObservation()
    return NavModelRings('test_rings', obs)


def _calculate_fade_integral_shade_above(edge_radius, width, pixel_lower, pixel_upper, resolution):
    """Calculate fade integral for shade_above case matching implementation logic.

    Matches the case-based logic in the implementation:
    - Case 1: Edge and fade end both within pixel
    - Case 2: Edge within pixel, fade end extends beyond
    - Case 3: Edge before pixel, fade end within pixel
    - Case 4: Edge before pixel, fade end after pixel (full coverage)
    """
    fade_start = edge_radius
    fade_end = edge_radius + width

    # Match implementation boundary conditions exactly
    # Case 1: Edge and fade end both within pixel
    eq2 = pixel_lower <= fade_start < pixel_upper
    eq3 = pixel_lower <= fade_end < pixel_upper
    if eq2 and eq3:
        # Both boundaries within pixel
        integration_start = fade_start
        integration_end = fade_end
    # Case 4: Edge before pixel, fade end after pixel (full coverage)
    elif fade_start < pixel_lower and fade_end > pixel_upper:
        integration_start = pixel_lower
        integration_end = pixel_upper
    # Case 2: Edge within pixel, fade end extends beyond
    elif eq2:
        integration_start = fade_start
        integration_end = pixel_upper
    # Case 3: Edge before pixel, fade end within pixel
    elif eq3:
        integration_start = pixel_lower
        integration_end = fade_end
    # No overlap
    else:
        return 0.0

    if integration_start >= integration_end:
        return 0.0

    # Integral formula matching implementation: [(1+a0/w)*a - a^2/(2w)] from a0 to a1
    integral = (((1.0 + edge_radius / width) * (integration_end - integration_start) +
                (integration_start**2 - integration_end**2) / (2.0 * width)) /
                resolution)
    return integral


def _calculate_fade_integral_shade_below(edge_radius, width, pixel_lower, pixel_upper, resolution):
    """Calculate fade integral for shade_below case matching implementation logic.

    Matches the case-based logic in the implementation:
    - Case 1: Fade start and edge both within pixel
    - Case 2: Edge within pixel, fade start before pixel
    - Case 3: Fade start within pixel, edge after pixel
    - Case 4: Fade start before pixel, edge after pixel (full coverage)
    """
    fade_start = edge_radius - width
    fade_end = edge_radius

    # Match implementation boundary conditions exactly
    # Case 1: Fade start and edge both within pixel
    eq2 = pixel_lower < fade_end <= pixel_upper
    eq3 = pixel_lower < fade_start <= pixel_upper
    if eq2 and eq3:
        # Both boundaries within pixel
        integration_start = fade_start
        integration_end = fade_end
    # Case 4: Fade start before pixel, edge after pixel (full coverage)
    elif fade_end > pixel_upper and fade_start < pixel_lower:
        integration_start = pixel_lower
        integration_end = pixel_upper
    # Case 2: Edge within pixel, fade start before pixel
    elif eq2:
        integration_start = pixel_lower
        integration_end = fade_end
    # Case 3: Fade start within pixel, edge after pixel
    elif eq3:
        integration_start = fade_start
        integration_end = pixel_upper
    # No overlap
    else:
        return 0.0

    if integration_start >= integration_end:
        return 0.0

    # Integral formula matching implementation: [(1-a0/w)*a + a^2/(2w)] from a0 to a1
    integral = (((1.0 - edge_radius / width) * (integration_end - integration_start) +
                (integration_end**2 - integration_start**2) / (2.0 * width)) /
                resolution)
    return integral


def test_edge_fade_shade_above_basic(ring_model):
    """Test basic edge fade with shade_above=True using multiple radii and edge cases."""
    # Create a 5x5 grid with various radii to test different scenarios
    model = np.zeros((5, 5), dtype=np.float64)
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    feature_list_by_a = []

    # Test various pixel positions:
    # - Pixel exactly at edge (100.0)
    # - Pixel below edge (90.0, 95.0)
    # - Pixel above edge but within fade (105.0, 110.0, 115.0)
    # - Pixel above fade region (125.0, 130.0)
    # - Pixel with non-integer center (97.5, 102.5)
    # - Pixel with odd resolution (7.0, 13.0) to test non-even division
    radii = np.array([
        [90.0, 95.0, 100.0, 105.0, 110.0],
        [92.5, 97.5, 100.0, 102.5, 107.5],
        [95.0, 100.0, 100.0, 115.0, 120.0],
        [100.0, 105.0, 110.0, 115.0, 125.0],
        [110.0, 115.0, 120.0, 125.0, 130.0]
    ])

    # Use different resolutions to test edge cases
    resolutions = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [7.0, 10.0, 10.0, 10.0, 13.0],  # Odd resolutions
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0, 10.0]
    ])

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Calculate expected value for each pixel individually
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            pixel_lower = pixel_center - resolutions[i, j] / 2.0
            pixel_upper = pixel_center + resolutions[i, j] / 2.0
            expected_shade = _calculate_fade_integral_shade_above(
                edge_radius, radius_width_km, pixel_lower, pixel_upper, resolutions[i, j])
            expected_value = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6), (
                f'Pixel at ({i}, {j}) with center={pixel_center:.1f}, '
                f'resolution={resolutions[i, j]:.1f} failed')


def test_edge_fade_shade_above_false(ring_model):
    """Test basic edge fade with shade_above=False using multiple radii and edge cases."""
    # Create a 5x5 grid with various radii to test different scenarios
    model = np.zeros((5, 5), dtype=np.float64)
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    feature_list_by_a = []

    # Test various pixel positions:
    # - Pixel exactly at edge (100.0)
    # - Pixel above edge (110.0, 115.0)
    # - Pixel below edge but within fade (85.0, 90.0, 95.0)
    # - Pixel below fade region (70.0, 75.0, 80.0)
    # - Pixel with non-integer center (97.5, 102.5)
    # - Pixel with odd resolution (7.0, 13.0) to test non-even division
    radii = np.array([
        [70.0, 80.0, 85.0, 90.0, 95.0],
        [75.0, 85.0, 90.0, 95.0, 100.0],
        [80.0, 90.0, 100.0, 100.0, 110.0],
        [92.5, 97.5, 100.0, 102.5, 107.5],
        [95.0, 100.0, 105.0, 110.0, 115.0]
    ])

    # Use different resolutions to test edge cases
    resolutions = np.array([
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [7.0, 10.0, 10.0, 10.0, 13.0],  # Odd resolutions
        [10.0, 10.0, 10.0, 10.0, 10.0]
    ])

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=False,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Calculate expected value for each pixel individually
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            pixel_lower = pixel_center - resolutions[i, j] / 2.0
            pixel_upper = pixel_center + resolutions[i, j] / 2.0
            expected_shade = _calculate_fade_integral_shade_below(
                edge_radius, radius_width_km, pixel_lower, pixel_upper, resolutions[i, j])
            expected_value = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6), (
                f'Pixel at ({i}, {j}) with center={pixel_center:.1f}, '
                f'resolution={resolutions[i, j]:.1f} failed')


def test_edge_fade_too_narrow(ring_model):
    """Test edge fade when width is too narrow."""
    model = np.zeros((10, 10), dtype=np.float64)
    radii = np.full((10, 10), 100.0)
    edge_radius = 100.0
    radius_width_km = 2.0
    min_radius_width_km = 5.0  # Larger than width
    resolutions = np.full((10, 10), 10.0)
    feature_list_by_a = []

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is None


def test_edge_fade_conflict_detection(ring_model):
    """Test edge fade conflict detection with nearby feature."""
    # Create a model with varying radii to properly test fade width
    model = np.zeros((20, 20), dtype=np.float64)
    # Create radii array that varies from 90 to 115 km
    u_coords = np.arange(20)
    v_coords = np.arange(20)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    # Map to radial distances: center at 100 km, extend outward
    radii = 100.0 + (u_grid - 10) * 0.5  # 90 to 110 km range
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    resolutions = np.full((20, 20), 1.0)  # Small resolution for better precision
    # Feature at 110 km (within fade width) - should adjust width to 5.0
    feature_list_by_a = [(110.0, 'IER')]

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Width should be adjusted to abs(110 - 100) / 2 = 5.0
    adjusted_width = abs(110.0 - edge_radius) / 2.0

    # Calculate expected value for each pixel using adjusted width
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            pixel_lower = pixel_center - resolutions[i, j] / 2.0
            pixel_upper = pixel_center + resolutions[i, j] / 2.0
            expected_shade = _calculate_fade_integral_shade_above(
                edge_radius, adjusted_width, pixel_lower, pixel_upper, resolutions[i, j])
            expected_value = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6)


def test_edge_fade_no_conflict(ring_model):
    """Test edge fade when no conflicts exist."""
    # Create a model with varying radii to properly test fade width
    model = np.zeros((30, 30), dtype=np.float64)
    # Create radii array that varies from 80 to 130 km to cover full fade extent
    u_coords = np.arange(30)
    v_coords = np.arange(30)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)
    # Map to radial distances: center at 100 km, extend outward
    # Fade extends from 100 to 120 km, so we need coverage up to at least 120 km
    radii = 100.0 + (u_grid - 15) * 1.5  # 77.5 to 122.5 km range
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    resolutions = np.full((30, 30), 1.0)  # Small resolution for better precision
    # Feature at 150 km (outside fade width, so no conflict)
    feature_list_by_a = [(150.0, 'IER')]

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Width should not be adjusted (no conflict)
    adjusted_width = radius_width_km

    # Calculate expected value for each pixel using original width
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            pixel_lower = pixel_center - resolutions[i, j] / 2.0
            pixel_upper = pixel_center + resolutions[i, j] / 2.0
            expected_shade = _calculate_fade_integral_shade_above(
                edge_radius, adjusted_width, pixel_lower, pixel_upper, resolutions[i, j])
            expected_value = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6)


def test_edge_fade_value_range(ring_model):
    """Test that fade values are always in [0, 1] range."""
    model = np.zeros((10, 10), dtype=np.float64)
    radii = np.linspace(80.0, 120.0, 100).reshape(10, 10)
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    resolutions = np.full((10, 10), 10.0)
    feature_list_by_a = []

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Calculate expected value for each pixel using the mathematical formula
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            pixel_center = radii[i, j]
            pixel_lower = pixel_center - resolutions[i, j] / 2.0
            pixel_upper = pixel_center + resolutions[i, j] / 2.0
            expected_shade = _calculate_fade_integral_shade_above(
                edge_radius, radius_width_km, pixel_lower, pixel_upper, resolutions[i, j])
            expected_value = model[i, j] + np.clip(expected_shade, 0.0, 1.0)
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6)


def test_edge_fade_adds_to_model(ring_model):
    """Test that fade adds to existing model values."""
    model = np.ones((10, 10), dtype=np.float64) * 0.5
    radii = np.full((10, 10), 100.0)
    edge_radius = 100.0
    radius_width_km = 20.0
    min_radius_width_km = 5.0
    resolutions = np.full((10, 10), 10.0)
    feature_list_by_a = []

    result = ring_model._compute_edge_fade(
        model=model, radii=radii, edge_radius=edge_radius, shade_above=True,
        radius_width_km=radius_width_km, min_radius_width_km=min_radius_width_km,
        resolutions=resolutions, feature_list_by_a=feature_list_by_a)

    assert result is not None

    # Calculate expected value: model value + fade value
    pixel_lower = 100.0 - 10.0 / 2.0  # 95.0
    pixel_upper = 100.0 + 10.0 / 2.0  # 105.0
    expected_shade = _calculate_fade_integral_shade_above(
        edge_radius, radius_width_km, pixel_lower, pixel_upper, 10.0)
    expected_value = model[0, 0] + np.clip(expected_shade, 0.0, 1.0)

    # All pixels should have the same value
    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            assert result[i, j] == pytest.approx(expected_value, abs=1e-6)
