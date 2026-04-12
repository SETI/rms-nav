"""Immutable data types for ring feature orbital parameters.

This module defines the core value objects that represent ring edge orbital
data: ``RingFeatureType``, ``RingBaseOrbitMode``, ``RingPerturbationMode``,
and ``RingEdgeData``. These are frozen dataclasses (immutable) because orbital
parameters from YAML config are physical constants that should never be modified
after loading. Validation occurs at construction time so that downstream
rendering code can trust the data without defensive checks.

Separating these types into their own module keeps them free of rendering
dependencies (oops, numpy), enabling lightweight import for testing and for
the simulated model which uses the types for data validation but does not use
backplane rendering.

Design notes:

- ``RingBaseOrbitMode`` and ``RingPerturbationMode`` are distinct types to
  resolve the ambiguity in the YAML config where ``mode: 1`` can appear with
  either base-orbit fields (``a``, ``ae``, ...) or perturbation fields
  (``amplitude``, ``phase``, ...). Having separate types makes the dispatch
  explicit in ``RingFeature.from_config()``.

- Inclination modes (``mode_num > 90``) are stored in ``RingEdgeData.perturbations``
  because they are valid YAML data. However, ``RingEdgeData.radial_perturbations()``
  and ``parsed_modes_for_backplane()`` exclude them, because the ``oops`` backplane
  ``radial_mode()`` function only handles radial (in-plane) perturbations. Making
  the limitation visible here rather than silently skipping them in rendering code
  surfaces the issue for future implementors.
"""

import enum
import math
import numbers
from dataclasses import dataclass
from typing import Any

# Seconds per day: degrees/day (or similar) to radians/second for backplane tuples.
SECONDS_PER_DAY = 86400.0


class RingFeatureType(enum.Enum):
    """Classification of a ring feature as a gap or ringlet.

    Determines the rendering polarity:
    - RINGLET: image is brightened between the two edges (fill between).
    - GAP: image is darkened between the two edges (clear between).
    Single-edge features of either type use fading instead of solid fill.
    """

    GAP = 'GAP'
    RINGLET = 'RINGLET'


@dataclass(frozen=True)
class RingBaseOrbitMode:
    """Mode-1 orbital parameters defining the base orbit of a ring edge.

    This represents the fundamental circular (or nearly circular) orbit of a
    ring edge before any higher-order perturbation modes are applied. It is
    always present in the data; higher modes are optional perturbations.

    Parameters:
        a: Semi-major axis in km. Must be > 0.
        ae: Eccentricity amplitude in km. Zero means circular.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Precession rate of pericenter in degrees/day.
        rms: RMS residual of the orbit fit in km. Must be >= 0.
            Used as the uncertainty measure for navigation.

    Raises:
        TypeError: If any numeric field is not a finite real number (``bool`` is
            rejected because it is a subclass of ``int``).
        ValueError: If ``a`` <= 0, ``ae`` < 0, or ``rms`` < 0.
    """

    a: float
    ae: float
    long_peri: float
    rate_peri: float
    rms: float

    def __post_init__(self) -> None:
        """Validate numeric types, finiteness, and field ranges at construction time."""
        for field_name in ('a', 'ae', 'long_peri', 'rate_peri', 'rms'):
            value = getattr(self, field_name)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(
                    f'RingBaseOrbitMode.{field_name} must be a real number, '
                    f'got {type(value).__name__}'
                )
            fv = float(value)
            if not math.isfinite(fv):
                raise TypeError(
                    f'RingBaseOrbitMode.{field_name} must be a finite number, got {value!r}'
                )
        if self.a <= 0:
            raise ValueError(f'RingBaseOrbitMode.a must be > 0, got {self.a}')
        if self.ae < 0:
            raise ValueError(f'RingBaseOrbitMode.ae must be >= 0, got {self.ae}')
        if self.rms < 0:
            raise ValueError(f'RingBaseOrbitMode.rms must be >= 0, got {self.rms}')


@dataclass(frozen=True)
class RingPerturbationMode:
    """A single radial or inclination perturbation mode for a ring edge.

    Higher-order perturbations are superimposed on the base orbit defined by
    ``RingBaseOrbitMode``. They represent resonance-driven distortions.

    Modes with ``mode_num > 90`` are inclination (out-of-plane) perturbations.
    These are stored in the data model because they appear in real YAML config
    files (e.g. Cassini Division features). However, inclination modes are not
    supported for radial backplane rendering because ``oops.ext_bp.radial_mode``
    only handles in-plane distortions. Use ``is_inclination_mode`` or
    ``RingEdgeData.radial_perturbations()`` to filter them out.

    Parameters:
        mode_num: Perturbation mode number passed to ``oops.ext_bp.radial_mode``.
            Values > 90 indicate inclination modes. Mission ring tables also use
            non-positive indices (e.g. ``0`` or negative modes in Saturn YAML).
        amplitude: Perturbation amplitude in km.
        phase: Perturbation phase in degrees.
        pattern_speed: Pattern speed in degrees/day.

    Raises:
        ValueError: If ``mode_num`` is not an integer (``bool`` is rejected), if
            ``amplitude`` is not an ``int``/``float`` (``bool`` rejected), is not
            finite, or is negative, or if ``phase`` or ``pattern_speed`` is not a
            finite ``int``/``float``.
    """

    mode_num: int
    amplitude: float
    phase: float
    pattern_speed: float

    def __post_init__(self) -> None:
        """Validate mode number, amplitude, phase, and pattern_speed."""
        if isinstance(self.mode_num, bool) or not isinstance(self.mode_num, int):
            raise ValueError(
                f'RingPerturbationMode.mode_num must be int (not bool), got {self.mode_num!r}'
            )
        if isinstance(self.amplitude, bool) or not isinstance(self.amplitude, (int, float)):
            raise ValueError(
                f'RingPerturbationMode.amplitude must be int or float (not bool), '
                f'got {type(self.amplitude).__name__}'
            )
        fa = float(self.amplitude)
        if not math.isfinite(fa):
            raise ValueError(
                f'RingPerturbationMode.amplitude must be a finite number, got {self.amplitude!r}'
            )
        if fa < 0:
            raise ValueError(f'RingPerturbationMode.amplitude must be >= 0, got {self.amplitude}')

        for field_name in ('phase', 'pattern_speed'):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f'RingPerturbationMode.{field_name} must be int or float (not bool), '
                    f'got {type(value).__name__}'
                )
            fv = float(value)
            if not math.isfinite(fv):
                raise ValueError(
                    f'RingPerturbationMode.{field_name} must be a finite number, got {value!r}'
                )

    @property
    def is_inclination_mode(self) -> bool:
        """Return True if this is an inclination (out-of-plane) mode.

        Inclination modes have ``mode_num > 90``. They represent vertical
        perturbations that require out-of-plane backplane support, which is
        not yet implemented. Callers should exclude these from radial rendering.
        """
        return self.mode_num > 90


@dataclass(frozen=True)
class RingEdgeData:
    """All orbital mode data for one edge of a ring feature.

    Combines the base orbit (``RingBaseOrbitMode``) with zero or more
    higher-order perturbations (``RingPerturbationMode``). This is the
    complete description needed to compute the radius of one ring edge at
    any point in the image backplane.

    Parameters:
        base_orbit: The mode-1 base orbit parameters.
        perturbations: Tuple of higher-order perturbation modes. May be empty.
            Inclination modes (mode_num > 90) are accepted here but excluded
            from backplane computation -- see ``radial_perturbations()``.
            A mutable sequence (e.g. ``list``) is accepted at construction and
            stored as an immutable ``tuple``.

    Raises:
        ValueError: If ``base_orbit`` or ``perturbations`` is ``None``.
        TypeError: If ``base_orbit`` is not a ``RingBaseOrbitMode``, if
            ``perturbations`` is not a non-string sequence, or if any sequence
            element is not a ``RingPerturbationMode``.
    """

    base_orbit: RingBaseOrbitMode
    perturbations: tuple['RingPerturbationMode', ...]

    def __post_init__(self) -> None:
        """Validate types and freeze ``perturbations`` as a ``tuple``."""
        bo = self.base_orbit
        if bo is None:
            raise ValueError('RingEdgeData: base_orbit must not be None')
        if not isinstance(bo, RingBaseOrbitMode):
            raise TypeError(
                'RingEdgeData: base_orbit must be an instance of RingBaseOrbitMode, '
                f'got {type(bo).__name__!r}'
            )

        pert_raw = self.perturbations
        if pert_raw is None:
            raise ValueError('RingEdgeData: perturbations must not be None')
        if isinstance(pert_raw, (str, bytes)):
            raise TypeError(
                'RingEdgeData: perturbations must be a sequence of RingPerturbationMode, '
                f'got {type(pert_raw).__name__!r}'
            )
        try:
            pert_seq = tuple(pert_raw)
        except TypeError as exc:
            raise TypeError(
                'RingEdgeData: perturbations must be a sequence of RingPerturbationMode, '
                f'got {type(pert_raw).__name__!r}'
            ) from exc

        for i, item in enumerate(pert_seq):
            if not isinstance(item, RingPerturbationMode):
                raise TypeError(
                    f'RingEdgeData: perturbations[{i}] must be RingPerturbationMode, '
                    f'got {type(item).__name__!r}'
                )

        object.__setattr__(self, 'perturbations', pert_seq)

    @property
    def base_radius(self) -> float:
        """Semi-major axis of the base orbit in km.

        This is the nominal radius used for spatial filtering and conflict
        detection. It is the mean radius of the edge, not the instantaneous
        (perturbed) radius at any given longitude.
        """
        return self.base_orbit.a

    @property
    def rms(self) -> float:
        """RMS residual of the orbit fit in km.

        Propagated to ``RingFeature.uncertainty`` (max of inner and outer edge
        RMS values), which in turn is stored in ``NavModelResult.uncertainty``.
        """
        return self.base_orbit.rms

    def radial_perturbations(self) -> tuple['RingPerturbationMode', ...]:
        """Return only radial (non-inclination) perturbation modes.

        Inclination modes (mode_num > 90) require out-of-plane backplane
        support that is not yet implemented in oops. This method filters them
        out so callers do not need to check individually.

        Returns:
            Tuple of perturbation modes with mode_num <= 90.
        """
        return tuple(p for p in self.perturbations if not p.is_inclination_mode)

    def parsed_modes_for_backplane(self) -> list[tuple[Any, ...]]:
        """Convert edge data to tuples for oops radial_mode computation.

        Returns a list of tuples in the format expected by
        ``oops.ext_bp.radial_mode()``:

        - Mode 1 (base orbit): ``(1, a, ae, long_peri_rad, rate_peri_rad_per_sec)``
        - Other modes: ``(mode_num, amplitude, phase_rad, speed_rad_per_sec)``

        Inclination modes (mode_num > 90) are excluded because
        ``oops.ext_bp.radial_mode`` only supports in-plane perturbations.
        The base orbit always comes first, followed by perturbations in their
        original order.

        Returns:
            List of mode tuples. Always contains at least the base orbit tuple.
        """
        result: list[tuple[Any, ...]] = []

        # Base orbit: convert degrees and degrees/day to radians and radians/second
        long_peri_rad = math.radians(self.base_orbit.long_peri)
        rate_peri_rad_per_sec = math.radians(self.base_orbit.rate_peri) / SECONDS_PER_DAY
        result.append(
            (1, self.base_orbit.a, self.base_orbit.ae, long_peri_rad, rate_peri_rad_per_sec)
        )

        # Radial perturbation modes
        for p in self.radial_perturbations():
            phase_rad = math.radians(p.phase)
            speed_rad_per_sec = math.radians(p.pattern_speed) / SECONDS_PER_DAY
            result.append((p.mode_num, p.amplitude, phase_rad, speed_rad_per_sec))

        return result
