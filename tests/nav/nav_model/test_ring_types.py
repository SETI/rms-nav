"""Unit tests for nav.nav_model.rings.ring_types.

Tests cover RingFeatureType, RingBaseOrbitMode, RingPerturbationMode, and
RingEdgeData. Written using TDD: tests are defined before implementation.
"""

import math

import numpy as np
import pytest

# These imports will fail until ring_types.py is created
from nav.nav_model.rings.ring_types import (
    RingBaseOrbitMode,
    RingEdgeData,
    RingFeatureType,
    RingPerturbationMode,
)


# ---------------------------------------------------------------------------
# RingFeatureType
# ---------------------------------------------------------------------------


def test_ring_feature_type_values() -> None:
    """Enum has exactly GAP and RINGLET with the expected string values."""
    assert RingFeatureType.GAP.value == 'GAP'
    assert RingFeatureType.RINGLET.value == 'RINGLET'


def test_ring_feature_type_from_value() -> None:
    """Enum can be constructed from its string value."""
    assert RingFeatureType('GAP') is RingFeatureType.GAP
    assert RingFeatureType('RINGLET') is RingFeatureType.RINGLET


def test_ring_feature_type_invalid_value() -> None:
    """Constructing enum from an unrecognised string raises ValueError."""
    with pytest.raises(ValueError, match='INVALID'):
        RingFeatureType('INVALID')


# ---------------------------------------------------------------------------
# RingBaseOrbitMode
# ---------------------------------------------------------------------------


def test_base_orbit_mode_valid() -> None:
    """Valid construction sets all fields correctly."""
    m = RingBaseOrbitMode(a=100_000.0, ae=10.0, long_peri=45.0, rate_peri=1.5, rms=2.0)
    assert m.a == 100_000.0
    assert m.ae == 10.0
    assert m.long_peri == 45.0
    assert m.rate_peri == 1.5
    assert m.rms == 2.0


def test_base_orbit_mode_is_frozen() -> None:
    """RingBaseOrbitMode is frozen: attribute assignment raises FrozenInstanceError."""
    m = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass
        m.a = 200_000.0  # type: ignore[misc]


def test_base_orbit_mode_nonpositive_a_raises() -> None:
    """Construction with a <= 0 raises ValueError."""
    with pytest.raises(ValueError, match='a'):
        RingBaseOrbitMode(a=0.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)


def test_base_orbit_mode_negative_a_raises() -> None:
    """Construction with a < 0 raises ValueError."""
    with pytest.raises(ValueError, match='a'):
        RingBaseOrbitMode(a=-1.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)


def test_base_orbit_mode_negative_rms_raises() -> None:
    """Construction with rms < 0 raises ValueError."""
    with pytest.raises(ValueError, match='rms'):
        RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=-0.1)


def test_base_orbit_mode_zero_rms_valid() -> None:
    """Construction with rms == 0 is valid (no data uncertainty)."""
    m = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)
    assert m.rms == 0.0


def test_base_orbit_mode_zero_ae_valid() -> None:
    """ae == 0 (circular orbit) is valid."""
    m = RingBaseOrbitMode(a=50_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.5)
    assert m.ae == 0.0


# ---------------------------------------------------------------------------
# RingPerturbationMode
# ---------------------------------------------------------------------------


def test_perturbation_mode_valid() -> None:
    """Valid construction sets all fields correctly."""
    m = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=90.0, pattern_speed=2.5)
    assert m.mode_num == 2
    assert m.amplitude == 5.0
    assert m.phase == 90.0
    assert m.pattern_speed == 2.5


def test_perturbation_mode_is_frozen() -> None:
    """RingPerturbationMode is frozen."""
    m = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=0.0, pattern_speed=0.0)
    with pytest.raises(AttributeError):
        m.mode_num = 3  # type: ignore[misc]


def test_perturbation_mode_inclination_false_for_radial() -> None:
    """is_inclination_mode is False for modes <= 90."""
    mode2 = RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    mode90 = RingPerturbationMode(mode_num=90, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    assert not mode2.is_inclination_mode
    assert not mode90.is_inclination_mode


def test_perturbation_mode_inclination_true_for_high_modes() -> None:
    """is_inclination_mode is True for modes > 90."""
    mode91 = RingPerturbationMode(mode_num=91, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    mode100 = RingPerturbationMode(mode_num=100, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    assert mode91.is_inclination_mode
    assert mode100.is_inclination_mode


def test_perturbation_mode_negative_amplitude_valid() -> None:
    """Negative amplitude is allowed (it's a perturbation, can oppose the base)."""
    m = RingPerturbationMode(mode_num=3, amplitude=-2.0, phase=0.0, pattern_speed=0.0)
    assert m.amplitude == -2.0


# ---------------------------------------------------------------------------
# RingEdgeData
# ---------------------------------------------------------------------------


def _make_base_orbit(a: float = 100_000.0) -> RingBaseOrbitMode:
    """Helper: create a valid base orbit."""
    return RingBaseOrbitMode(a=a, ae=10.0, long_peri=30.0, rate_peri=1.0, rms=2.0)


def test_edge_data_valid_no_perturbations() -> None:
    """RingEdgeData constructs with base orbit and no perturbations."""
    base = _make_base_orbit()
    edge = RingEdgeData(base_orbit=base, perturbations=())
    assert edge.base_orbit is base
    assert edge.perturbations == ()


def test_edge_data_valid_with_perturbations() -> None:
    """RingEdgeData constructs with perturbations."""
    base = _make_base_orbit()
    p1 = RingPerturbationMode(mode_num=2, amplitude=3.0, phase=0.0, pattern_speed=0.5)
    p2 = RingPerturbationMode(mode_num=91, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(p1, p2))
    assert len(edge.perturbations) == 2


def test_edge_data_is_frozen() -> None:
    """RingEdgeData is frozen."""
    edge = RingEdgeData(base_orbit=_make_base_orbit(), perturbations=())
    with pytest.raises(AttributeError):
        edge.base_orbit = _make_base_orbit(a=200_000.0)  # type: ignore[misc]


def test_edge_data_base_radius_property() -> None:
    """base_radius returns the semi-major axis from the base orbit."""
    edge = RingEdgeData(base_orbit=_make_base_orbit(a=87_000.0), perturbations=())
    assert edge.base_radius == 87_000.0


def test_edge_data_rms_property() -> None:
    """rms returns the RMS from the base orbit."""
    base = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=3.7)
    edge = RingEdgeData(base_orbit=base, perturbations=())
    assert edge.rms == 3.7


def test_edge_data_radial_perturbations_empty() -> None:
    """radial_perturbations returns empty tuple when there are none."""
    edge = RingEdgeData(base_orbit=_make_base_orbit(), perturbations=())
    assert edge.radial_perturbations() == ()


def test_edge_data_radial_perturbations_excludes_inclination() -> None:
    """radial_perturbations excludes modes with mode_num > 90."""
    base = _make_base_orbit()
    radial = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=0.0, pattern_speed=1.0)
    inclin = RingPerturbationMode(mode_num=91, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(radial, inclin))
    result = edge.radial_perturbations()
    assert len(result) == 1
    assert result[0].mode_num == 2


def test_edge_data_radial_perturbations_all_radial() -> None:
    """radial_perturbations returns all modes when all are radial."""
    base = _make_base_orbit()
    p2 = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=0.0, pattern_speed=1.0)
    p3 = RingPerturbationMode(mode_num=3, amplitude=2.0, phase=30.0, pattern_speed=0.5)
    edge = RingEdgeData(base_orbit=base, perturbations=(p2, p3))
    result = edge.radial_perturbations()
    assert len(result) == 2


# ---------------------------------------------------------------------------
# RingEdgeData.parsed_modes_for_backplane
# ---------------------------------------------------------------------------


def test_parsed_modes_base_only() -> None:
    """parsed_modes_for_backplane returns one mode-1 tuple when no perturbations."""
    base = RingBaseOrbitMode(
        a=100_000.0, ae=10.0, long_peri=45.0, rate_peri=180.0, rms=1.0
    )
    edge = RingEdgeData(base_orbit=base, perturbations=())
    modes = edge.parsed_modes_for_backplane()
    assert len(modes) == 1
    mode_num, a, ae, long_peri_rad, rate_peri_rad_per_sec = modes[0]
    assert mode_num == 1
    assert a == 100_000.0
    assert ae == 10.0
    assert long_peri_rad == pytest.approx(math.radians(45.0))
    assert rate_peri_rad_per_sec == pytest.approx(math.radians(180.0) / 86400.0)


def test_parsed_modes_with_radial_perturbation() -> None:
    """parsed_modes_for_backplane includes radial perturbation modes."""
    base = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    p = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=90.0, pattern_speed=360.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(p,))
    modes = edge.parsed_modes_for_backplane()
    assert len(modes) == 2
    # Second mode: (mode_num, amplitude, phase_rad, speed_rad_per_sec)
    mode_num, amplitude, phase_rad, speed_rad_per_sec = modes[1]
    assert mode_num == 2
    assert amplitude == 5.0
    assert phase_rad == pytest.approx(math.radians(90.0))
    assert speed_rad_per_sec == pytest.approx(math.radians(360.0) / 86400.0)


def test_parsed_modes_excludes_inclination_modes() -> None:
    """parsed_modes_for_backplane excludes inclination modes (mode > 90)."""
    base = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    radial = RingPerturbationMode(mode_num=2, amplitude=5.0, phase=0.0, pattern_speed=1.0)
    inclin = RingPerturbationMode(mode_num=91, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(radial, inclin))
    modes = edge.parsed_modes_for_backplane()
    assert len(modes) == 2  # mode 1 + mode 2; mode 91 is excluded
    mode_nums = [m[0] for m in modes]
    assert 1 in mode_nums
    assert 2 in mode_nums
    assert 91 not in mode_nums


def test_parsed_modes_all_inclination_returns_base_only() -> None:
    """When all perturbations are inclination modes, only the base mode-1 tuple is returned."""
    base = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    inclin = RingPerturbationMode(mode_num=91, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(inclin,))
    modes = edge.parsed_modes_for_backplane()
    assert len(modes) == 1
    assert modes[0][0] == 1


def test_parsed_modes_multiple_perturbations_ordering() -> None:
    """parsed_modes_for_backplane preserves order: base first, then perturbations."""
    base = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    p3 = RingPerturbationMode(mode_num=3, amplitude=2.0, phase=0.0, pattern_speed=0.0)
    p2 = RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    edge = RingEdgeData(base_orbit=base, perturbations=(p3, p2))
    modes = edge.parsed_modes_for_backplane()
    assert len(modes) == 3
    assert modes[0][0] == 1   # base first
    assert modes[1][0] == 3   # then p3
    assert modes[2][0] == 2   # then p2


def test_base_orbit_mode_equality() -> None:
    """Two RingBaseOrbitMode instances with the same values compare equal."""
    m1 = RingBaseOrbitMode(a=100_000.0, ae=10.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    m2 = RingBaseOrbitMode(a=100_000.0, ae=10.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    assert m1 == m2


def test_edge_data_equality() -> None:
    """Two RingEdgeData instances with the same data compare equal."""
    base1 = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    base2 = RingBaseOrbitMode(a=100_000.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=1.0)
    e1 = RingEdgeData(base_orbit=base1, perturbations=())
    e2 = RingEdgeData(base_orbit=base2, perturbations=())
    assert e1 == e2
