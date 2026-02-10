"""Unit tests for ring model feature filtering.

Tests date-based feature filtering and validation of feature data structures.
"""

import numpy as np
from numpy.typing import NDArray
import pytest

from nav.nav_model.nav_model_rings import NavModelRings
from nav.support.time import utc_to_et


class MockObservation:
    """Mock observation for testing."""

    def __init__(self, midtime: float | None = None) -> None:
        self.closest_planet = 'SATURN'
        if midtime is None:
            # Default to 2008-01-01 12:00:00
            self.midtime = utc_to_et('2008-01-01 12:00:00')
        else:
            self.midtime = midtime
        self.extdata_shape_vu = (100, 100)

    def make_extfov_zeros(self) -> NDArray[np.float64]:
        return np.zeros((100, 100), dtype=np.float64)

    def make_extfov_false(self) -> NDArray[np.bool_]:
        return np.zeros((100, 100), dtype=bool)


@pytest.fixture
def ring_model() -> NavModelRings:
    """Create a NavModelRings instance for testing."""
    obs = MockObservation()
    return NavModelRings('test_rings', obs)


def test_load_features_no_date_range(ring_model: NavModelRings) -> None:
    """Test loading features without date ranges (always active)."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 1
    assert features[0]['name'] == 'Test Ringlet'


def test_load_features_within_date_range() -> None:
    """Test loading features when observation is within date range."""
    obs = MockObservation(utc_to_et('2009-01-01 12:00:00'))
    ring_model = NavModelRings('test_rings', obs)

    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'start_date': '2008-01-01 12:00:00',
            'end_date': '2010-01-01 12:00:00',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 1


def test_load_features_before_date_range() -> None:
    """Test loading features when observation is before date range."""
    obs = MockObservation(utc_to_et('2007-01-01 12:00:00'))
    ring_model = NavModelRings('test_rings', obs)

    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'start_date': '2008-01-01 12:00:00',
            'end_date': '2010-01-01 12:00:00',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 0


def test_load_features_after_date_range() -> None:
    """Test loading features when observation is after date range."""
    obs = MockObservation(utc_to_et('2011-01-01 12:00:00'))
    ring_model = NavModelRings('test_rings', obs)

    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'start_date': '2008-01-01 12:00:00',
            'end_date': '2010-01-01 12:00:00',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 0


def test_load_features_only_start_date() -> None:
    """Test loading features with only start_date."""
    obs = MockObservation(utc_to_et('2009-01-01 12:00:00'))
    ring_model = NavModelRings('test_rings', obs)

    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'start_date': '2008-01-01 12:00:00',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 1


def test_load_features_only_end_date() -> None:
    """Test loading features with only end_date."""
    obs = MockObservation(utc_to_et('2009-01-01 12:00:00'))
    ring_model = NavModelRings('test_rings', obs)

    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Test Ringlet',
            'end_date': '2010-01-01 12:00:00',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 1


def test_load_features_invalid_type(ring_model: NavModelRings) -> None:
    """Test loading features with invalid feature_type."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'INVALID',
            'name': 'Test Feature',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 0


def test_load_features_no_edges(ring_model: NavModelRings) -> None:
    """Test loading features with neither inner nor outer data."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {'feature_type': 'RINGLET', 'name': 'Test Ringlet'},
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=0.0, max_radius=1e6
    )
    assert len(features) == 0


def test_validate_mode_data_valid(ring_model: NavModelRings) -> None:
    """Test validation of valid mode data."""
    mode_data = [
        {
            'mode': 1,
            'a': 100000.0,
            'rms': 1.0,
            'ae': 10.0,
            'long_peri': 0.0,
            'rate_peri': 0.0,
        }
    ]

    assert ring_model._validate_mode_data(
        mode_data,
        feature_key='test_feature',
        edge_type='inner',
        min_radius=0.0,
        max_radius=1e6,
    )


def test_validate_mode_data_missing_mode(ring_model: NavModelRings) -> None:
    """Test validation of mode data missing mode number."""
    mode_data = [{'a': 100000.0, 'ae': 10.0}]

    assert not ring_model._validate_mode_data(
        mode_data,
        feature_key='test_feature',
        edge_type='inner',
        min_radius=0.0,
        max_radius=1e6,
    )


def test_validate_mode_data_non_positive_a(ring_model: NavModelRings) -> None:
    """Test validation of mode 1 data with non-positive semi-major axis."""
    mode_data = [{'mode': 1, 'a': -100.0, 'ae': 10.0}]

    assert not ring_model._validate_mode_data(
        mode_data,
        feature_key='test_feature',
        edge_type='inner',
        min_radius=0.0,
        max_radius=1e6,
    )


def test_get_base_radius(ring_model: NavModelRings) -> None:
    """Test getting base radius from mode data."""
    mode_data = [{'mode': 1, 'a': 100000.0, 'ae': 10.0}]

    radius = ring_model._get_base_radius(mode_data)
    assert radius == 100000.0


def test_get_base_radius_no_mode_1(ring_model: NavModelRings) -> None:
    """Test getting base radius when mode 1 is not present."""
    mode_data = [{'mode': 2, 'amplitude': 10.0, 'phase': 0.0, 'pattern_speed': 0.0}]

    radius = ring_model._get_base_radius(mode_data)
    assert radius is None


def test_load_features_radius_filtering(ring_model: NavModelRings) -> None:
    """Test loading features with radius filtering."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature_in_range': {
            'feature_type': 'RINGLET',
            'name': 'Feature In Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_feature_below_range': {
            'feature_type': 'RINGLET',
            'name': 'Feature Below Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 50000.0,  # Below min_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 51000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_feature_above_range': {
            'feature_type': 'RINGLET',
            'name': 'Feature Above Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 200000.0,
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 201000.0,  # Above max_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_feature_at_boundaries': {
            'feature_type': 'RINGLET',
            'name': 'Feature At Boundaries',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 90000.0,  # Exactly at min_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 150000.0,  # Exactly at max_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    # Test with radius range that includes some features
    min_radius = 90000.0
    max_radius = 150000.0

    features = ring_model._load_ring_features(
        planet_config,
        ring_model.obs.midtime,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    # Should include: in_range, at_boundaries (both edges within range)
    # Should exclude: below_range (inner edge below), above_range (outer edge above)
    assert len(features) == 2
    feature_names = {f['name'] for f in features}
    assert 'Feature In Range' in feature_names
    assert 'Feature At Boundaries' in feature_names
    assert 'Feature Below Range' not in feature_names
    assert 'Feature Above Range' not in feature_names


def test_load_features_radius_filtering_inner_edge_out_of_range(
    ring_model: NavModelRings,
) -> None:
    """Test that features are excluded if inner edge is out of range."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Feature With Inner Edge Out of Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 50000.0,  # Below min_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 101000.0,  # Within range
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=90000.0, max_radius=150000.0
    )
    # Should be excluded because inner edge is out of range
    assert len(features) == 0


def test_load_features_radius_filtering_outer_edge_out_of_range(
    ring_model: NavModelRings,
) -> None:
    """Test that features are excluded if outer edge is out of range."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_feature': {
            'feature_type': 'RINGLET',
            'name': 'Feature With Outer Edge Out of Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,  # Within range
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
            'outer_data': [
                {
                    'mode': 1,
                    'a': 200000.0,  # Above max_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=90000.0, max_radius=150000.0
    )
    # Should be excluded because outer edge is out of range
    assert len(features) == 0


def test_load_features_radius_filtering_gap_with_single_edge(
    ring_model: NavModelRings,
) -> None:
    """Test radius filtering for gaps with only one edge."""
    planet_config = {
        'epoch': '2008-01-01 12:00:00',
        'feature_width': 100,
        'test_gap_inner_in_range': {
            'feature_type': 'GAP',
            'name': 'Gap With Inner Edge In Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 100000.0,  # Within range
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_gap_inner_out_of_range': {
            'feature_type': 'GAP',
            'name': 'Gap With Inner Edge Out of Range',
            'inner_data': [
                {
                    'mode': 1,
                    'a': 50000.0,  # Below min_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_gap_outer_in_range': {
            'feature_type': 'GAP',
            'name': 'Gap With Outer Edge In Range',
            'outer_data': [
                {
                    'mode': 1,
                    'a': 100000.0,  # Within range
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
        'test_gap_outer_out_of_range': {
            'feature_type': 'GAP',
            'name': 'Gap With Outer Edge Out of Range',
            'outer_data': [
                {
                    'mode': 1,
                    'a': 200000.0,  # Above max_radius
                    'rms': 1.0,
                    'ae': 10.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }
            ],
        },
    }

    features = ring_model._load_ring_features(
        planet_config, ring_model.obs.midtime, min_radius=90000.0, max_radius=150000.0
    )

    # Should include gaps with edges in range, exclude those with edges out of range
    assert len(features) == 2
    feature_names = {f['name'] for f in features}
    assert 'Gap With Inner Edge In Range' in feature_names
    assert 'Gap With Outer Edge In Range' in feature_names
    assert 'Gap With Inner Edge Out of Range' not in feature_names
    assert 'Gap With Outer Edge Out of Range' not in feature_names
