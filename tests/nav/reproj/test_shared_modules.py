"""Tests for nav.reproj shared modules: BodyMosaicMergeStrategy,
RingMosaicMergeStrategy, PhotometricModel, and RingOrbitModel.
"""

import math

import numpy as np
import pytest

from nav.reproj.bodies import BodyMosaicMergeStrategy
from nav.reproj.photometric_model import (
    LambertModel,
    LommelSeeligerModel,
    MinnaertModel,
)
from nav.reproj.ring_orbit_model import BRING_OUTER_EDGE, FRING_CORE, RingOrbitModel
from nav.reproj.rings import RingMosaicMergeStrategy

# =========================================================================
# BodyMosaicMergeStrategy tests
# =========================================================================


class TestBodyMosaicMergeStrategy:
    """Tests for the BodyMosaicMergeStrategy enum."""

    def test_best_resolution_member_exists(self) -> None:
        """BEST_RESOLUTION is a valid enum member."""
        strategy = BodyMosaicMergeStrategy.BEST_RESOLUTION
        assert strategy == BodyMosaicMergeStrategy.BEST_RESOLUTION

    def test_most_coverage_then_resolution_member_exists(self) -> None:
        """MOST_COVERAGE_THEN_RESOLUTION is a valid enum member."""
        strategy = BodyMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        assert strategy == BodyMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION

    def test_members_are_distinct(self) -> None:
        """The two strategies are not equal to each other."""
        assert (
            BodyMosaicMergeStrategy.BEST_RESOLUTION  # type: ignore[comparison-overlap]
            != BodyMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        )

    def test_value_strings(self) -> None:
        """Enum values are the expected string keys."""
        assert BodyMosaicMergeStrategy.BEST_RESOLUTION.value == 'best_resolution'
        assert (
            BodyMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION.value
            == 'most_coverage_then_resolution'
        )


# =========================================================================
# RingMosaicMergeStrategy tests
# =========================================================================


class TestRingMosaicMergeStrategy:
    """Tests for the RingMosaicMergeStrategy enum."""

    def test_best_resolution_member_exists(self) -> None:
        """BEST_RESOLUTION is a valid enum member."""
        strategy = RingMosaicMergeStrategy.BEST_RESOLUTION
        assert strategy == RingMosaicMergeStrategy.BEST_RESOLUTION

    def test_most_coverage_then_resolution_member_exists(self) -> None:
        """MOST_COVERAGE_THEN_RESOLUTION is a valid enum member."""
        strategy = RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        assert strategy == RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION

    def test_members_are_distinct(self) -> None:
        """The two strategies are not equal to each other."""
        assert (
            RingMosaicMergeStrategy.BEST_RESOLUTION  # type: ignore[comparison-overlap]
            != RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        )

    def test_value_strings(self) -> None:
        """Enum values are the expected string keys."""
        assert RingMosaicMergeStrategy.BEST_RESOLUTION.value == 'best_resolution'
        assert (
            RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION.value
            == 'most_coverage_then_resolution'
        )


# =========================================================================
# PhotometricModel tests
# =========================================================================


def _angles(
    incidence_deg: float, emission_deg: float, phase_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar arrays of the three photometric angles in radians."""
    return (
        np.array([math.radians(incidence_deg)]),
        np.array([math.radians(emission_deg)]),
        np.array([math.radians(phase_deg)]),
    )


class TestLambertModel:
    """Tests for LambertModel photometric correction."""

    def test_name_attribute(self) -> None:
        """The model name is 'lambert'."""
        assert LambertModel().name == 'lambert'

    def test_correct_normal_incidence(self) -> None:
        """At zero incidence, correction factor is 1/cos(0) = 1."""
        model = LambertModel()
        data = np.array([1.0])
        inc, emi, phase = _angles(0.0, 30.0, 30.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        assert pytest.approx(result[0], rel=1e-6) == 1.0

    def test_correct_45deg_incidence(self) -> None:
        """At 45 deg incidence, correction factor is 1/cos(45) = sqrt(2)."""
        model = LambertModel()
        data = np.array([1.0])
        inc, emi, phase = _angles(45.0, 0.0, 45.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        expected = 1.0 / math.cos(math.radians(45.0))
        assert pytest.approx(result[0], rel=1e-5) == expected

    def test_correct_high_incidence_clamped(self) -> None:
        """At incidence > ~89 deg the denominator is clamped to min_cos_incidence."""
        model = LambertModel(min_cos_incidence=0.01)
        data = np.array([1.0])
        inc, emi, phase = _angles(89.9, 0.0, 89.9)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        # cos(89.9 deg) < 0.01, so denominator should be clamped to 0.01
        assert pytest.approx(result[0], rel=1e-5) == 1.0 / 0.01

    def test_correct_scales_data(self) -> None:
        """Correction scales the input data proportionally."""
        model = LambertModel()
        data = np.array([2.0])
        inc, emi, phase = _angles(60.0, 0.0, 60.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        expected = 2.0 / math.cos(math.radians(60.0))
        assert pytest.approx(result[0], rel=1e-5) == expected

    def test_no_emission_term(self) -> None:
        """Lambert correction does not depend on emission angle."""
        model = LambertModel()
        data = np.array([1.0])
        inc = np.array([math.radians(30.0)])
        phase = np.array([math.radians(30.0)])
        emi_a = np.array([math.radians(0.0)])
        emi_b = np.array([math.radians(60.0)])
        result_a = model.correct(data.copy(), incidence=inc, emission=emi_a, phase=phase)
        result_b = model.correct(data.copy(), incidence=inc, emission=emi_b, phase=phase)
        assert pytest.approx(result_a[0], rel=1e-6) == result_b[0]


class TestLommelSeeligerModel:
    """Tests for LommelSeeligerModel photometric correction."""

    def test_name_attribute(self) -> None:
        """The model name is 'lommel_seeliger'."""
        assert LommelSeeligerModel().name == 'lommel_seeliger'

    def test_correct_normal_geometry(self) -> None:
        """At normal incidence/emission the correction formula gives expected value."""
        model = LommelSeeligerModel()
        data = np.array([1.0])
        inc, emi, phase = _angles(30.0, 30.0, 30.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        # Lommel-Seeliger: I/F = data * (cos_i + cos_e) / (2 * cos_i)
        # correction = (cos_i + cos_e) / (2 * cos_i)
        cos_i = math.cos(math.radians(30.0))
        cos_e = math.cos(math.radians(30.0))
        expected = (cos_i + cos_e) / (2.0 * cos_i)
        assert pytest.approx(result[0], rel=1e-5) == expected

    def test_correct_different_angles(self) -> None:
        """Correction varies when incidence and emission angles differ."""
        model = LommelSeeligerModel()
        data = np.array([1.0])
        inc, emi, phase = _angles(45.0, 20.0, 40.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        cos_i = math.cos(math.radians(45.0))
        cos_e = math.cos(math.radians(20.0))
        expected = (cos_i + cos_e) / (2.0 * cos_i)
        assert pytest.approx(result[0], rel=1e-5) == expected


class TestMinnaertModel:
    """Tests for MinnaertModel photometric correction."""

    def test_name_attribute(self) -> None:
        """The model name is 'minnaert'."""
        assert MinnaertModel().name == 'minnaert'

    def test_default_k_is_half(self) -> None:
        """Default Minnaert k parameter is 0.5."""
        assert MinnaertModel().k == 0.5

    def test_correct_k_one(self) -> None:
        """With k=1, Minnaert reduces to Lambert (1/cos_i)."""
        model = MinnaertModel(k=1.0)
        data = np.array([1.0])
        inc, emi, phase = _angles(45.0, 30.0, 45.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        expected = 1.0 / math.cos(math.radians(45.0))
        assert pytest.approx(result[0], rel=1e-5) == expected

    def test_correct_k_half(self) -> None:
        """With k=0.5, Minnaert formula gives expected value."""
        model = MinnaertModel(k=0.5)
        data = np.array([1.0])
        inc, emi, phase = _angles(45.0, 30.0, 45.0)
        result = model.correct(data, incidence=inc, emission=emi, phase=phase)
        cos_i = math.cos(math.radians(45.0))
        cos_e = math.cos(math.radians(30.0))
        # Minnaert: corrected = data / (cos_i^k * cos_e^(k-1))
        expected = 1.0 / (cos_i**0.5 * cos_e ** (-0.5))
        assert pytest.approx(result[0], rel=1e-5) == expected


# =========================================================================
# RingOrbitModel tests
# =========================================================================


class TestRingOrbitModel:
    """Tests for the RingOrbitModel frozen dataclass and its methods."""

    def test_fring_core_predefined_instance(self) -> None:
        """FRING_CORE predefined instance has the expected name."""
        assert FRING_CORE.name == 'FRING-CORE'

    def test_bring_outer_edge_predefined_instance(self) -> None:
        """BRING_OUTER_EDGE predefined instance has the expected name."""
        assert BRING_OUTER_EDGE.name == 'BRING-OUTER-EDGE'

    def test_frozen_dataclass_immutable(self) -> None:
        """RingOrbitModel is frozen (immutable)."""
        with pytest.raises((AttributeError, TypeError)):
            FRING_CORE.a = 1.0  # type: ignore[misc]

    def test_negative_a_raises(self) -> None:
        """Constructing a RingOrbitModel with negative semi-major axis raises ValueError."""
        with pytest.raises(ValueError, match='semi-major axis'):
            RingOrbitModel(
                name='test',
                a=-1.0,
                e=0.0,
                w0=0.0,
                dw=0.0,
                mean_motion=1.0,
                epoch_utc='2007-01-01',
            )

    def test_eccentricity_out_of_range_raises(self) -> None:
        """Eccentricity must be in [0, 1)."""
        with pytest.raises(ValueError, match='eccentricity'):
            RingOrbitModel(
                name='test',
                a=100.0,
                e=1.5,
                w0=0.0,
                dw=0.0,
                mean_motion=1.0,
                epoch_utc='2007-01-01',
            )

    def test_radius_at_longitude_circular(self) -> None:
        """For a circular orbit (e=0), radius is constant at semi-major axis."""
        model = RingOrbitModel(
            name='circular',
            a=100000.0,
            e=0.0,
            w0=0.0,
            dw=0.0,
            mean_motion=1.0,
            epoch_utc='2000-01-01T12:00:00',
        )
        et = 0.0
        for lon_deg in (0.0, 45.0, 90.0, 135.0, 180.0):
            lon = math.radians(lon_deg)
            r = model.radius_at_longitude(np.array([lon]), et)
            assert pytest.approx(float(r[0]), rel=1e-6) == 100000.0

    def test_radius_at_longitude_eccentric(self) -> None:
        """Eccentric orbit: radius at pericenter is a*(1-e), at apocenter a*(1+e)."""
        model = FRING_CORE
        et = 0.0
        # Pericenter is at w0 (longitude of pericenter at epoch, dw*0=0)
        w0 = FRING_CORE.w0
        r_peri = model.radius_at_longitude(np.array([w0]), et)
        r_apo = model.radius_at_longitude(np.array([w0 + math.pi]), et)
        expected_peri = FRING_CORE.a * (1.0 - FRING_CORE.e)
        expected_apo = FRING_CORE.a * (1.0 + FRING_CORE.e)
        assert pytest.approx(float(r_peri[0]), rel=1e-4) == expected_peri
        assert pytest.approx(float(r_apo[0]), rel=1e-4) == expected_apo

    def test_corotating_round_trip(self) -> None:
        """inertial_to_corotating followed by corotating_to_inertial is identity."""
        model = FRING_CORE
        et = 1e8
        longitudes = np.linspace(0.0, 2 * math.pi, 36, endpoint=False)
        co = model.inertial_to_corotating(longitudes, et)
        back = model.corotating_to_inertial(co, et)
        np.testing.assert_allclose(back % (2 * math.pi), longitudes % (2 * math.pi), atol=1e-10)

    def test_longitude_radius_length(self) -> None:
        """longitude_radius returns arrays of the expected length."""
        model = FRING_CORE
        et = 0.0
        step = 0.01 * (math.pi / 180.0)
        lons, radii = model.longitude_radius(et, step=step)
        expected_n = int(2.0 * math.pi / step)
        assert len(lons) == expected_n
        assert len(radii) == expected_n

    def test_longitude_radius_consistency(self) -> None:
        """longitude_radius radii match radius_at_longitude at the same longitudes."""
        model = FRING_CORE
        et = 0.0
        step = 0.1 * (math.pi / 180.0)
        lons, radii = model.longitude_radius(et, step=step)
        expected = model.radius_at_longitude(lons, et)
        np.testing.assert_allclose(radii, expected, rtol=1e-6)
