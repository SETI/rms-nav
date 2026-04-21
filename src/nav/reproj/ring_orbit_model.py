"""Ring orbit models for co-rotating frame transformations.

Provides the RingOrbitModel frozen dataclass representing a Keplerian ring
orbit with precession, along with pre-defined instances for the F ring core
and B ring outer edge.

All angular quantities are in radians and all time quantities are in seconds
(ET/TDB) unless otherwise stated. Day-based rates are converted internally.
"""

import math
from dataclasses import dataclass, field

import julian
import numpy as np

from nav.support.types import NDArrayFloatType


def _utc2et(s: str) -> float:
    """Convert a UTC string to ephemeris time (TDB seconds)."""
    return float(julian.tdb_from_tai(julian.tai_from_iso(s)))


@dataclass(frozen=True)
class RingOrbitModel:
    """Keplerian orbit model for a ring feature with apsidal precession.

    All angular parameters are in radians. Rate parameters (dw, mean_motion)
    are in radians per day; they are converted to per-second internally when
    needed.

    Attributes:
        name: Human-readable name identifying this ring/model.
        a: Semi-major axis in km. Must be positive.
        e: Orbital eccentricity. Must be in [0, 1).
        w0: Longitude of pericenter at epoch (rad).
        dw: Apsidal precession rate (rad/day).
        mean_motion: Mean motion (rad/day) used for the co-rotating frame.
        epoch_utc: Epoch for the co-rotating frame, as an ISO UTC string.
    """

    name: str
    a: float
    e: float
    w0: float
    dw: float
    mean_motion: float
    epoch_utc: str
    _epoch_et: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and cache the epoch in ET."""
        if self.a <= 0.0:
            raise ValueError(f'RingOrbitModel semi-major axis must be positive, got {self.a}')
        if not (0.0 <= self.e < 1.0):
            raise ValueError(f'RingOrbitModel eccentricity must be in [0, 1), got {self.e}')
        # frozen=True prevents direct assignment; use object.__setattr__.
        # _utc2et may raise TypeError when julian is mocked (e.g. Sphinx docs build);
        # fall back to 0.0 in that case since the value will never be used.
        try:
            epoch_et: float = _utc2et(self.epoch_utc)
        except TypeError:
            epoch_et = 0.0
        object.__setattr__(self, '_epoch_et', epoch_et)

    def radius_at_longitude(self, longitude: NDArrayFloatType, et: float) -> NDArrayFloatType:
        """Return the ring radius (km) at each inertial longitude and time.

        Uses the standard Keplerian orbit equation with a precessing pericenter.

        Parameters:
            longitude: Inertial (true) longitude array (rad).
            et: Observation time as ephemeris time (TDB seconds).

        Returns:
            Radius in km at each element of longitude.
        """
        curly_w = self.w0 + self.dw * et / 86400.0
        result: NDArrayFloatType = (
            self.a * (1.0 - self.e**2) / (1.0 + self.e * np.cos(longitude - curly_w))
        )
        return result

    def _longitude_shift(self, et: float) -> float:
        """Return the co-rotating frame longitude shift at time et.

        The shift is defined such that adding it to an inertial longitude
        gives the co-rotating longitude.

        Parameters:
            et: Observation time as ephemeris time (TDB seconds).

        Returns:
            Longitude shift in radians (wrapped to [0, 2*pi)).
        """
        # Explicit parentheses to clarify precedence: negate, then mod.
        # TODO: Verify sign convention matches original project intent.
        return (-(self.mean_motion * ((et - self._epoch_et) / 86400.0))) % (2.0 * math.pi)

    def inertial_to_corotating(self, longitude: NDArrayFloatType, et: float) -> NDArrayFloatType:
        """Convert inertial longitude (rad) to co-rotating longitude (rad).

        Parameters:
            longitude: Inertial longitude array (rad).
            et: Observation time as ephemeris time (TDB seconds).

        Returns:
            Co-rotating longitude array (rad), wrapped to [0, 2*pi).
        """
        return (longitude + self._longitude_shift(et)) % (2.0 * math.pi)

    def corotating_to_inertial(self, co_long: NDArrayFloatType, et: float) -> NDArrayFloatType:
        """Convert co-rotating longitude (rad) to inertial longitude (rad).

        Parameters:
            co_long: Co-rotating longitude array (rad).
            et: Observation time as ephemeris time (TDB seconds).

        Returns:
            Inertial longitude array (rad), wrapped to [0, 2*pi).
        """
        return (co_long - self._longitude_shift(et)) % (2.0 * math.pi)

    def longitude_radius(
        self, et: float, *, step: float = 0.01 * math.pi / 180.0
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Return arrays of (longitude, radius) covering the full 0..2pi range.

        Parameters:
            et: Observation time as ephemeris time (TDB seconds).
            step: Longitude step size (rad). Defaults to 0.01 degrees.

        Returns:
            Tuple of (longitudes, radii) arrays, each of length
            int(2*pi / step).
        """
        n = int(2.0 * math.pi / step)
        longitudes: NDArrayFloatType = np.arange(n) * step
        radii = self.radius_at_longitude(longitudes, et)
        return longitudes, radii


# Pre-defined instances for Saturn ring features

FRING_CORE = RingOrbitModel(
    name='FRING-CORE',
    a=140221.3,
    e=0.00235,
    w0=24.2 * math.pi / 180.0,
    dw=2.70025 * math.pi / 180.0,
    mean_motion=581.964 * math.pi / 180.0,
    epoch_utc='2007-01-01',
)

BRING_OUTER_EDGE = RingOrbitModel(
    name='BRING-OUTER-EDGE',
    a=117570.0,
    e=0.0,
    w0=0.0,
    dw=0.0,
    mean_motion=758.768 * math.pi / 180.0,
    epoch_utc='2009-08-11',
)
