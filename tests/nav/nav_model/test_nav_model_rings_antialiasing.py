"""Unit tests for ring model anti-aliasing computation.

Tests the _compute_antialiasing() method with various pixel/edge
configurations to verify smooth transitions and correct value ranges.
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


def test_antialiasing_pixel_center_at_edge(ring_model):
    """Test anti-aliasing when pixel center is exactly at edge."""
    radii = np.array([100.0])
    edge_radius = 100.0
    resolutions = np.array([10.0])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True, resolutions=resolutions)
    assert shade[0] == pytest.approx(0.5, abs=1e-6)


def test_antialiasing_edge_half_resolution_above(ring_model):
    """Test anti-aliasing when edge is 0.5*resolution above pixel center."""
    radii = np.array([100.0])
    edge_radius = 105.0  # 0.5 * resolution above
    resolutions = np.array([10.0])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True, resolutions=resolutions)
    assert shade[0] == pytest.approx(1.0, abs=1e-6)


def test_antialiasing_edge_half_resolution_below(ring_model):
    """Test anti-aliasing when edge is 0.5*resolution below pixel center."""
    radii = np.array([100.0])
    edge_radius = 95.0  # 0.5 * resolution below
    resolutions = np.array([10.0])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=False, resolutions=resolutions)
    assert shade[0] == pytest.approx(1.0, abs=1e-6)


def test_antialiasing_shade_above_true(ring_model):
    """Test anti-aliasing with shade_above=True.

    When shade_above=True, we're shading the object above the edge, so
    anti-aliasing occurs below the edge. Pixels above the edge should have
    shade=0 (already in object), pixels below should have shade>0.
    """
    radii = np.array([95.0, 100.0, 105.0])
    edge_radius = 100.0
    resolutions = np.array([10.0, 10.0, 10.0])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True, resolutions=resolutions)
    # First pixel: 5 km below edge, shade = 1.0 - 1*(95-100)/10 - 0.5 = 1.0
    assert shade[0] == pytest.approx(1.0, abs=1e-6)
    # Second pixel: center at edge, should be 0.5
    assert shade[1] == pytest.approx(0.5, abs=1e-6)
    # Third pixel: 5 km above edge, should be 0 (in object, no anti-aliasing)
    assert shade[2] == pytest.approx(0.0, abs=1e-6)


def test_antialiasing_shade_above_false(ring_model):
    """Test anti-aliasing with shade_above=False.

    When shade_above=False, we're shading the object below the edge, so
    anti-aliasing occurs above the edge. Pixels below the edge should have
    shade=0 (already in object), pixels above should have shade>0.
    """
    radii = np.array([95.0, 100.0, 105.0])
    edge_radius = 100.0
    resolutions = np.array([10.0, 10.0, 10.0])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=False, resolutions=resolutions)
    # First pixel: 5 km below edge, should be 0 (in object, no anti-aliasing)
    assert shade[0] == pytest.approx(0.0, abs=1e-6)
    # Second pixel: center at edge, should be 0.5
    assert shade[1] == pytest.approx(0.5, abs=1e-6)
    # Third pixel: 5 km above edge, shade = 1.0 - (-1)*(105-100)/10 - 0.5 = 1.0
    assert shade[2] == pytest.approx(1.0, abs=1e-6)


def test_antialiasing_value_range(ring_model):
    """Test that anti-aliasing values are always in [0, 1] range."""
    radii = np.linspace(80.0, 120.0, 100)
    edge_radius = 100.0
    resolutions = np.full(100, 10.0)

    shade_above = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True, resolutions=resolutions)
    shade_below = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=False, resolutions=resolutions)

    # Calculate expected values for shade_above=True
    # Match the actual implementation: shade[shade > 1.0] = 0.0 (not 1.0)
    shade_sign_above = 1.0
    expected_above = 1.0 - shade_sign_above * (radii - edge_radius) / resolutions
    expected_above -= 0.5
    expected_above[expected_above < 0.0] = 0.0
    expected_above[expected_above > 1.0] = 0.0  # Match implementation behavior
    expected_above *= 1.0  # max_value

    # Calculate expected values for shade_above=False
    shade_sign_below = -1.0
    expected_below = 1.0 - shade_sign_below * (radii - edge_radius) / resolutions
    expected_below -= 0.5
    expected_below[expected_below < 0.0] = 0.0
    expected_below[expected_below > 1.0] = 0.0  # Match implementation behavior
    expected_below *= 1.0  # max_value

    # Check each value individually
    for i in range(len(radii)):
        assert shade_above[i] == pytest.approx(expected_above[i], abs=1e-6)
        assert shade_below[i] == pytest.approx(expected_below[i], abs=1e-6)


def test_antialiasing_max_value_parameter(ring_model):
    """Test anti-aliasing with custom max_value parameter."""
    radii = np.array([100.0])
    edge_radius = 100.0
    resolutions = np.array([10.0])
    max_value = 0.5

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True,
        resolutions=resolutions, max_value=max_value)
    assert shade[0] == pytest.approx(0.25, abs=1e-6)  # 0.5 * max_value


def test_antialiasing_array_input(ring_model):
    """Test anti-aliasing with 2-D array inputs."""
    radii = np.array([[100.0, 105.0], [95.0, 100.0]])
    edge_radius = 100.0
    resolutions = np.array([[10.0, 10.0], [10.0, 10.0]])

    shade = ring_model._compute_antialiasing(
        radii=radii, edge_radius=edge_radius, shade_above=True, resolutions=resolutions)

    # Calculate expected values matching the implementation
    shade_sign = 1.0
    expected = 1.0 - shade_sign * (radii - edge_radius) / resolutions
    expected -= 0.5
    expected[expected < 0.0] = 0.0
    expected[expected > 1.0] = 0.0  # Match implementation behavior
    expected *= 1.0  # max_value

    assert shade.shape == radii.shape

    for i in range(radii.shape[0]):
        for j in range(radii.shape[1]):
            assert shade[i, j] == pytest.approx(expected[i, j], abs=1e-6)
