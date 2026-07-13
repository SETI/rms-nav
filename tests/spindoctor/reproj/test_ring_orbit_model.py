"""Spec tests for ``spindoctor.reproj.ring_orbit_model``.

Contracts under test (from the module and method docstrings):

- All angles are radians; rates (``dw``, ``mean_motion``) are radians/day; times are
  ephemeris time (TDB seconds).
- ``radius_at_longitude`` implements ``a (1 - e^2) / (1 + e cos(lon - curly_w))`` with a
  precessing pericenter ``curly_w = w0 + dw * et / 86400`` referenced to J2000.
- ``inertial_to_corotating`` / ``corotating_to_inertial`` are mutual inverses; the
  co-rotating frame is anchored at ``epoch_utc`` and rotates at ``mean_motion``; results
  are wrapped to ``[0, 2*pi)``.
- Constructor validation raises the documented ``ValueError`` / ``TypeError``.
- ``get_orbit_model_by_name`` returns registered instances or ``None``.
"""

import dataclasses
import math

import julian
import numpy as np
import pytest

from spindoctor.reproj.ring_orbit_model import (
    BRING_OUTER_EDGE,
    FRING_CORE,
    RingOrbitModel,
    get_orbit_model_by_name,
)

_SECONDS_PER_DAY = 86400.0
_EPOCH_UTC = '2005-06-01T00:00:00'


def _epoch_et(epoch_utc: str = _EPOCH_UTC) -> float:
    """Convert an ISO UTC string to ephemeris time (TDB seconds) as the module does.

    Parameters:
        epoch_utc: ISO UTC time string.

    Returns:
        Ephemeris time in TDB seconds.
    """
    return float(julian.tdb_from_tai(julian.tai_from_iso(epoch_utc)))


def _model(
    *,
    a: float = 140000.0,
    e: float = 0.1,
    w0: float = 0.5,
    dw: float = 0.0,
    mean_motion: float = 0.0,
    epoch_utc: str = _EPOCH_UTC,
) -> RingOrbitModel:
    """Build a RingOrbitModel with convenient defaults for tests.

    Parameters:
        a: Semi-major axis (km).
        e: Eccentricity.
        w0: Longitude of pericenter at J2000 (rad).
        dw: Apsidal precession rate (rad/day).
        mean_motion: Co-rotating frame mean motion (rad/day).
        epoch_utc: ISO UTC epoch anchoring the co-rotating frame.

    Returns:
        A RingOrbitModel built from the given parameters.
    """
    return RingOrbitModel(
        name='unit-test-model',
        a=a,
        e=e,
        w0=w0,
        dw=dw,
        mean_motion=mean_motion,
        epoch_utc=epoch_utc,
    )


class TestConstructorValidation:
    """Documented constructor errors and frozen-dataclass behavior."""

    def test_zero_semi_major_axis_rejected(self) -> None:
        """a == 0 raises ValueError naming the semi-major axis."""
        with pytest.raises(ValueError, match='semi-major axis must be positive'):
            _model(a=0.0)

    def test_negative_semi_major_axis_rejected(self) -> None:
        """a < 0 raises ValueError naming the semi-major axis."""
        with pytest.raises(ValueError, match='semi-major axis must be positive'):
            _model(a=-140000.0)

    def test_eccentricity_of_one_rejected(self) -> None:
        """e == 1 is outside the documented half-open interval [0, 1)."""
        with pytest.raises(ValueError, match=r'eccentricity must be in \[0, 1\)'):
            _model(e=1.0)

    def test_negative_eccentricity_rejected(self) -> None:
        """e < 0 is outside the documented half-open interval [0, 1)."""
        with pytest.raises(ValueError, match=r'eccentricity must be in \[0, 1\)'):
            _model(e=-0.001)

    def test_zero_eccentricity_accepted(self) -> None:
        """e == 0 (circular orbit) is inside the documented interval."""
        assert _model(e=0.0).e == 0.0

    def test_nan_semi_major_axis_rejected(self) -> None:
        """A non-finite a raises ValueError naming the field."""
        with pytest.raises(ValueError, match='a must be finite'):
            _model(a=math.nan)

    def test_infinite_pericenter_rejected(self) -> None:
        """A non-finite w0 raises ValueError naming the field."""
        with pytest.raises(ValueError, match='w0 must be finite'):
            _model(w0=math.inf)

    def test_nan_mean_motion_rejected(self) -> None:
        """A non-finite mean_motion raises ValueError naming the field."""
        with pytest.raises(ValueError, match='mean_motion must be finite'):
            _model(mean_motion=math.nan)

    def test_non_string_epoch_rejected(self) -> None:
        """A non-str epoch_utc raises TypeError."""
        with pytest.raises(TypeError, match='epoch_utc must be str'):
            _model(epoch_utc=12345)  # type: ignore[arg-type]

    def test_instances_are_frozen(self) -> None:
        """Field assignment on the frozen dataclass raises FrozenInstanceError."""
        m = _model()
        with pytest.raises(dataclasses.FrozenInstanceError, match='cannot assign'):
            m.a = 1.0  # type: ignore[misc]

    def test_value_equality_for_distinct_instances(self) -> None:
        """Two separately constructed models with equal fields compare equal."""
        assert _model() == _model()

    def test_inequality_when_a_field_differs(self) -> None:
        """Models differing in any field (here ``a``) compare unequal."""
        assert _model() != _model(a=140001.0)


class TestRadiusAtLongitude:
    """Keplerian radius formula with precessing pericenter."""

    def test_circular_orbit_radius_is_constant_a(self) -> None:
        """With e == 0, radius equals the semi-major axis at every longitude."""
        m = _model(e=0.0, a=123456.0)
        lons = np.linspace(0.0, 2.0 * math.pi, 17)
        radii = m.radius_at_longitude(lons, 0.0)
        np.testing.assert_array_equal(radii, np.full(17, 123456.0))

    def test_radius_at_pericenter_is_a_times_one_minus_e(self) -> None:
        """At longitude == pericenter direction, r == a (1 - e)."""
        m = _model(e=0.1, w0=0.7, dw=0.0)
        radii = m.radius_at_longitude(np.array([0.7]), 0.0)
        assert radii[0] == pytest.approx(140000.0 * 0.9, rel=1e-12)

    def test_radius_at_apocenter_is_a_times_one_plus_e(self) -> None:
        """At longitude == pericenter + pi, r == a (1 + e)."""
        m = _model(e=0.1, w0=0.7, dw=0.0)
        radii = m.radius_at_longitude(np.array([0.7 + math.pi]), 0.0)
        assert radii[0] == pytest.approx(140000.0 * 1.1, rel=1e-12)

    def test_radius_at_quadrature_is_semi_latus_rectum(self) -> None:
        """At longitude == pericenter + pi/2, r == a (1 - e^2)."""
        m = _model(e=0.1, w0=0.7, dw=0.0)
        radii = m.radius_at_longitude(np.array([0.7 + math.pi / 2.0]), 0.0)
        assert radii[0] == pytest.approx(140000.0 * (1.0 - 0.01), rel=1e-12)

    def test_pericenter_precesses_at_dw_radians_per_day(self) -> None:
        """After one day with dw == pi/3, the pericenter has advanced by pi/3 from w0."""
        m = _model(e=0.1, w0=0.7, dw=math.pi / 3.0)
        radii = m.radius_at_longitude(np.array([0.7 + math.pi / 3.0]), _SECONDS_PER_DAY)
        assert radii[0] == pytest.approx(140000.0 * 0.9, rel=1e-12)

    def test_array_shape_is_preserved(self) -> None:
        """The output radius array has the same shape as the input longitude array."""
        m = _model()
        lons = np.zeros((3, 4))
        assert m.radius_at_longitude(lons, 0.0).shape == (3, 4)


class TestLongitudeFrameConversions:
    """Inertial <-> co-rotating longitude conversions."""

    def test_conversions_are_identity_at_epoch(self) -> None:
        """At et == epoch, the co-rotating frame coincides with the inertial frame."""
        m = _model(mean_motion=100.0)
        lons = np.array([0.0, 1.0, 3.0, 6.0])
        np.testing.assert_array_equal(m.inertial_to_corotating(lons, _epoch_et()), lons)

    def test_roundtrip_recovers_inertial_longitudes(self) -> None:
        """corotating_to_inertial inverts inertial_to_corotating (mod 2 pi)."""
        m = _model(mean_motion=37.5)
        et = _epoch_et() + 3.75 * _SECONDS_PER_DAY
        lons = np.array([0.0, 0.5, 2.0, 4.5, 6.2])
        back = m.corotating_to_inertial(m.inertial_to_corotating(lons, et), et)
        np.testing.assert_allclose(back, lons, rtol=0.0, atol=1e-12)

    def test_shift_after_one_day_equals_minus_mean_motion(self) -> None:
        """One day past epoch, corotating == (inertial - mean_motion) mod 2 pi."""
        mm = 0.75
        m = _model(mean_motion=mm)
        et = _epoch_et() + _SECONDS_PER_DAY
        lons = np.array([1.0, 2.0, 3.0])
        expected = (lons - mm) % (2.0 * math.pi)
        np.testing.assert_allclose(m.inertial_to_corotating(lons, et), expected, atol=1e-12)

    def test_inertial_to_corotating_wraps_into_zero_two_pi(self) -> None:
        """Converted longitudes are wrapped into [0, 2 pi)."""
        m = _model(mean_motion=1000.0)
        et = _epoch_et() + 12.3 * _SECONDS_PER_DAY
        out = m.inertial_to_corotating(np.linspace(0.0, 6.28, 20), et)
        assert bool(np.all(out >= 0.0))
        assert bool(np.all(out < 2.0 * math.pi))

    def test_corotating_to_inertial_wraps_into_zero_two_pi(self) -> None:
        """Inverse-converted longitudes are wrapped into [0, 2 pi)."""
        m = _model(mean_motion=1000.0)
        et = _epoch_et() + 12.3 * _SECONDS_PER_DAY
        out = m.corotating_to_inertial(np.linspace(0.0, 6.28, 20), et)
        assert bool(np.all(out >= 0.0))
        assert bool(np.all(out < 2.0 * math.pi))


class TestLongitudeRadius:
    """Full-circle (longitude, radius) sampling."""

    def test_length_is_two_pi_over_step(self) -> None:
        """Arrays have length int(2 pi / step) as documented."""
        m = _model(e=0.0)
        step = math.pi / 180.0
        lons, radii = m.longitude_radius(0.0, step=step)
        assert len(lons) == int(2.0 * math.pi / step)
        assert len(radii) == len(lons)

    def test_longitudes_start_at_zero_with_uniform_step(self) -> None:
        """Longitudes are 0, step, 2 step, ..."""
        m = _model(e=0.0)
        step = 0.1
        lons, _ = m.longitude_radius(0.0, step=step)
        assert lons[0] == 0.0
        np.testing.assert_allclose(np.diff(lons), step, rtol=0.0, atol=1e-15)

    def test_radii_match_radius_at_longitude(self) -> None:
        """For a circular orbit every sampled radius equals a."""
        m = _model(e=0.0, a=99000.0)
        _, radii = m.longitude_radius(0.0, step=0.5)
        np.testing.assert_array_equal(radii, np.full(len(radii), 99000.0))

    @pytest.mark.parametrize('step', [0.0, -0.01, math.inf, math.nan])
    def test_invalid_step_rejected(self, step: float) -> None:
        """Non-positive or non-finite steps raise the documented ValueError."""
        m = _model()
        with pytest.raises(ValueError, match='step must be a finite positive number'):
            m.longitude_radius(0.0, step=step)


class TestPredefinedModelsAndRegistry:
    """Pre-defined instances and get_orbit_model_by_name lookups."""

    def test_fring_core_lookup_by_name(self) -> None:
        """The F ring core model is registered under its own name."""
        assert get_orbit_model_by_name('F-RING-CORE-ALBERS-2007') is FRING_CORE

    def test_bring_outer_edge_lookup_by_name(self) -> None:
        """The B ring outer edge model is registered under its own name."""
        assert get_orbit_model_by_name('BRING-OUTER-EDGE') is BRING_OUTER_EDGE

    def test_fring_core_published_parameters(self) -> None:
        """FRING_CORE carries the Albers et al. 2012 Table 3 Fit #2 values."""
        assert FRING_CORE.a == 140221.3
        assert FRING_CORE.e == 0.00235
        assert FRING_CORE.epoch_utc == '2007-01-01'

    def test_unknown_name_returns_none(self) -> None:
        """An unregistered name returns None rather than raising."""
        assert get_orbit_model_by_name('NOT-A-REAL-MODEL') is None

    def test_non_string_name_rejected(self) -> None:
        """A non-str name raises TypeError."""
        with pytest.raises(TypeError, match='expects str'):
            get_orbit_model_by_name(42)  # type: ignore[arg-type]

    def test_empty_name_rejected(self) -> None:
        """An empty name raises ValueError."""
        with pytest.raises(ValueError, match='non-empty string'):
            get_orbit_model_by_name('')

    def test_whitespace_only_name_rejected(self) -> None:
        """A whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match='non-empty string'):
            get_orbit_model_by_name('   ')
