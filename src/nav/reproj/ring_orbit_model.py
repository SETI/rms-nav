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

_SECONDS_PER_DAY = 86400.0


def _utc2et(s: str) -> float:
    """Convert a UTC string to ephemeris time (TDB seconds)."""
    return float(julian.tdb_from_tai(julian.tai_from_iso(s)))


@dataclass(frozen=True)
class RingOrbitModel:
    """Keplerian orbit model for a ring feature with apsidal precession.

    All angular parameters are in radians. Rate parameters (``dw``,
    ``mean_motion``) are in radians per day; they are converted to per-second
    internally when needed.

    ``a`` (km) is the semi-major axis (positive). ``e`` is eccentricity in
    ``[0, 1)``. ``w0`` is the longitude of pericenter at J2000 (rad); at
    observation time ``et`` (TDB seconds) the pericenter direction is
    ``w0 + dw * et / 86400``. ``dw`` is apsidal precession (rad/day) from J2000.
    ``mean_motion`` (rad/day) drives the co-rotating frame. ``epoch_utc`` is an
    ISO UTC string anchoring that frame, independent of the J2000 apsidal
    reference.

    This is a :func:`dataclasses.dataclass`; fields are documented on each member
    below.
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
        for fname in ('a', 'e', 'w0', 'dw', 'mean_motion'):
            v = getattr(self, fname)
            if not math.isfinite(v):
                raise ValueError(f'RingOrbitModel {fname} must be finite, got {v!r}')
        if self.a <= 0.0:
            raise ValueError(f'RingOrbitModel semi-major axis must be positive, got {self.a}')
        if not (0.0 <= self.e < 1.0):
            raise ValueError(f'RingOrbitModel eccentricity must be in [0, 1), got {self.e}')
        if not isinstance(self.epoch_utc, str):
            raise TypeError(
                'RingOrbitModel epoch_utc must be str (ISO UTC), got '
                f'{type(self.epoch_utc).__name__}'
            )
        # frozen=True prevents direct assignment; use object.__setattr__.
        try:
            epoch_et = _utc2et(self.epoch_utc)
        except TypeError as exc:
            raise TypeError(
                'RingOrbitModel epoch_utc could not be converted to ephemeris time (TDB seconds); '
                f'check epoch_utc is valid calendar data for julian: {self.epoch_utc!r}'
            ) from exc
        if not math.isfinite(epoch_et):
            raise ValueError(
                'RingOrbitModel epoch_utc produced a non-finite ephemeris time (TDB seconds); '
                f'epoch_utc={self.epoch_utc!r}, et={epoch_et!r}'
            )
        object.__setattr__(self, '_epoch_et', epoch_et)

    def radius_at_longitude(self, longitude: NDArrayFloatType, et: float) -> NDArrayFloatType:
        """Return the ring radius (km) at each inertial longitude and time.

        Uses the standard Keplerian orbit equation with a precessing pericenter.
        The pericenter direction at time ``et`` is ``w0 + dw * et``: ``w0`` is
        the value of curly-pi at J2000 (the standard epoch), and ``dw`` is
        integrated from J2000. The independent ``epoch_utc`` field anchors only
        the co-rotating mean-motion frame (see ``_longitude_shift``).

        Parameters:
            longitude: Inertial (true) longitude array (rad).
            et: Observation time as ephemeris time (TDB seconds).

        Returns:
            Radius in km at each element of longitude.
        """
        curly_w = self.w0 + self.dw * et / _SECONDS_PER_DAY
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
        # Explicit parentheses: negate mean-motion x elapsed days, then mod.
        return (-(self.mean_motion * ((et - self._epoch_et) / _SECONDS_PER_DAY))) % (2.0 * math.pi)

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

        Raises:
            ValueError: If step is not a finite positive number.
        """
        if not (math.isfinite(step) and step > 0):
            raise ValueError(
                f'longitude_radius step must be a finite positive number, got {step!r}'
            )
        n = int(2.0 * math.pi / step)
        longitudes: NDArrayFloatType = np.arange(n) * step
        radii = self.radius_at_longitude(longitudes, et)
        return longitudes, radii


# Pre-defined instances for Saturn ring features

FRING_CORE = RingOrbitModel(
    name='F-RING-CORE-ALBERS-2007',
    a=140221.3,
    e=0.00235,
    w0=24.2 * math.pi / 180.0,
    dw=2.70025 * math.pi / 180.0,
    mean_motion=581.964 * math.pi / 180.0,
    epoch_utc='2007-01-01',
)
"""F ring core orbit model from Albers et al. 2012 Table 3 Fit #2.

``epoch_utc='2007-01-01'`` anchors the co-rotating mean-motion frame; ``w0``
is the longitude of pericenter at J2000 (not at ``epoch_utc``). The ``2007``
in the name refers to the co-rotation epoch.
"""

BRING_OUTER_EDGE = RingOrbitModel(
    name='BRING-OUTER-EDGE',
    a=117570.0,
    e=0.0,
    w0=0.0,
    dw=0.0,
    mean_motion=758.768 * math.pi / 180.0,
    epoch_utc='2009-08-11',
)

_KNOWN_ORBIT_MODELS: dict[str, RingOrbitModel] = {m.name: m for m in (FRING_CORE, BRING_OUTER_EDGE)}


def get_orbit_model_by_name(name: str) -> RingOrbitModel | None:
    """Return a pre-defined :class:`RingOrbitModel` by its name, or ``None``.

    Parameters:
        name: Model name (e.g. ``'F-RING-CORE-ALBERS-2007'``).

    Returns:
        The matching :class:`RingOrbitModel` from :data:`_KNOWN_ORBIT_MODELS`, or
        ``None`` if the name is not found.

    Raises:
        TypeError: If ``name`` is not a :class:`str`.
        ValueError: If ``name`` is empty or whitespace-only.
    """
    if not isinstance(name, str):
        raise TypeError(f'get_orbit_model_by_name expects str, got {type(name).__name__}')
    if not name.strip():
        raise ValueError('get_orbit_model_by_name: name must be a non-empty string')
    return _KNOWN_ORBIT_MODELS.get(name)
