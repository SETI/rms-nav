"""Shared fixtures for nav_model ring tests.

Provides a unified MockObservation and common ring feature data factories used
across multiple test modules. Centralizing these avoids duplicating nearly-identical
mock setup in every test file.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from spindoctor.support.time import utc_to_et


class MockObservation:
    """Minimal mock for oops.Observation, sufficient for ring model unit tests.

    Covers the attributes used by NavModelRings, NavModelRingsBase, and the
    rings subpackage:
    - extdata_shape_vu / make_extfov_zeros / make_extfov_false for image ops
    - midtime for date-range checks
    - sim_time / sim_epoch for simulated-ring tests
    - extfov_margin_v / extfov_margin_u for annotation coordinate shifts
    - data_shape_v / data_shape_u for center coordinate defaults
    """

    def __init__(
        self,
        *,
        midtime: float | None = None,
        shape: tuple[int, int] = (100, 100),
    ) -> None:
        """Create a mock observation.

        Parameters:
            midtime: Observation time in TDB seconds. Defaults to
                2008-01-01 12:00:00 UTC.
            shape: (rows, cols) of the extended FOV array.
        """
        self.closest_planet = 'SATURN'
        self.midtime: float = midtime if midtime is not None else utc_to_et('2008-01-01 12:00:00')
        self.extdata_shape_vu: tuple[int, int] = shape
        self.extfov_margin_v: int = 0
        self.extfov_margin_u: int = 0
        self.data_shape_v: int = shape[0]
        self.data_shape_u: int = shape[1]
        self.sim_time: float = self.midtime
        self.sim_epoch: float = self.midtime

    def make_extfov_zeros(self) -> npt.NDArray[np.float64]:
        """Return a zero-filled float64 array of the extended FOV shape."""
        return np.zeros(self.extdata_shape_vu, dtype=np.float64)

    def make_extfov_false(self) -> npt.NDArray[np.bool_]:
        """Return a False-filled bool array of the extended FOV shape."""
        return np.zeros(self.extdata_shape_vu, dtype=bool)

    def make_extfov_inf(self) -> npt.NDArray[np.float64]:
        """Return an inf-filled float64 array (used for range arrays)."""
        return np.full(self.extdata_shape_vu, math.inf, dtype=np.float64)


@pytest.fixture
def mock_obs() -> MockObservation:
    """Default MockObservation with standard 100x100 shape."""
    return MockObservation()


@pytest.fixture
def mock_obs_2009() -> MockObservation:
    """MockObservation at 2009-01-01 12:00:00 UTC."""
    return MockObservation(midtime=utc_to_et('2009-01-01 12:00:00'))


# ---------------------------------------------------------------------------
# Common ring feature data factories
# ---------------------------------------------------------------------------


def make_mode1_data(
    a: float = 100_000.0,
    rms: float = 1.0,
    ae: float = 10.0,
    *,
    long_peri: float = 0.0,
    rate_peri: float = 0.0,
) -> list[dict[str, Any]]:
    """Return a single-mode mode-1 list (base orbit only).

    Parameters:
        a: Semi-major axis in km.
        rms: Edge RMS uncertainty in km.
        ae: Eccentricity amplitude in km.
        long_peri: Longitude of periapsis in degrees.
        rate_peri: Periapsis precession rate in degrees per year.

    Returns:
        ``list[dict[str, Any]]`` with one mode-1 entry (``mode``, ``a``, ``rms``,
        ``ae``, ``long_peri``, ``rate_peri`` keys).
    """
    return [
        {'mode': 1, 'a': a, 'rms': rms, 'ae': ae, 'long_peri': long_peri, 'rate_peri': rate_peri}
    ]


def make_mode1_with_perturbation(
    a: float = 100_000.0,
    rms: float = 1.0,
    ae: float = 10.0,
    *,
    long_peri: float = 0.0,
    rate_peri: float = 0.0,
    mode_num: int = 2,
    amplitude: float = 5.0,
    phase: float = 45.0,
    pattern_speed: float = 1.0,
) -> list[dict[str, Any]]:
    """Return mode data with a base orbit plus one perturbation mode.

    Parameters:
        a: Semi-major axis in km.
        rms: Edge RMS uncertainty in km.
        ae: Eccentricity amplitude in km.
        long_peri: Longitude of periapsis in degrees.
        rate_peri: Periapsis precession rate in degrees per year.
        mode_num: Perturbation mode number.
        amplitude: Perturbation amplitude in km.
        phase: Phase in degrees.
        pattern_speed: Pattern speed in degrees per year.

    Returns:
        ``list[dict[str, Any]]``: mode-1 dict followed by one perturbation dict.
    """
    return [
        {'mode': 1, 'a': a, 'rms': rms, 'ae': ae, 'long_peri': long_peri, 'rate_peri': rate_peri},
        {'mode': mode_num, 'amplitude': amplitude, 'phase': phase, 'pattern_speed': pattern_speed},
    ]
