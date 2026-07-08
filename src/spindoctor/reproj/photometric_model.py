"""Photometric correction models for body and ring reprojections.

All models implement the PhotometricModel protocol and apply a correction to
raw pixel brightness based on the local illumination and viewing geometry.
Default is no correction (pass ``photometric_model=None`` to ``BodyMosaic`` or
``RingMosaic``).

All angles are in radians throughout.
"""

import math
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

# Smallest allowed positive cosine clamp / denominator floor (matches ``min_denom`` scale).
_TINY_POSITIVE: float = 1e-15
_PHOTOMETRIC_REAL = (int, float, np.integer, np.floating)


def _photometric_real_scalar(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, _PHOTOMETRIC_REAL):
        raise TypeError(f'{field} must be a real number, got {type(value).__name__}')
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'{field} must be finite, got {value!r}')
    return x


def _validate_cos_clamp(field: str, value: object) -> None:
    """Cosine floor used with ``np.maximum(cos(angle), value)``; must be in ``(0, 1]``."""
    x = _photometric_real_scalar(field, value)
    if x < _TINY_POSITIVE or x > 1.0:
        raise ValueError(
            f'{field} must be finite with {_TINY_POSITIVE} <= {field} <= 1.0, got {value!r}'
        )


def _validate_positive_floor(field: str, value: object) -> None:
    x = _photometric_real_scalar(field, value)
    if x < _TINY_POSITIVE:
        raise ValueError(f'{field} must be finite and >= {_TINY_POSITIVE}, got {value!r}')


class PhotometricModel(Protocol):
    """Protocol for photometric correction applied during reprojection.

    Implementations must provide a string name and a correct() method.
    The correct() method receives all three angles as keyword-only
    NDArrayFloatType parameters (all in radians) and returns the corrected
    data array.
    """

    name: str

    def correct(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Apply photometric correction to data.

        Parameters:
            data: Raw pixel brightness values to correct.
            incidence: Incidence angle at each pixel (rad).
            emission: Emission angle at each pixel (rad).
            phase: Phase angle at each pixel (rad).

        Returns:
            Corrected pixel brightness values, same shape as data.
        """
        ...


@dataclass
class LambertModel:
    """Lambert (Lambertian) photometric correction.

    Divides the data by cos(incidence) to normalize for the amount of sunlight
    falling on the surface element. The emission angle is not included because
    Lambert's law only describes the illumination term; the viewing geometry
    is independent. Near-grazing incidence is clamped to avoid division by
    near-zero values.

    ``min_cos_incidence`` (default 0.01, ~84°) clamps the Lambert denominator;
    see the member entry below.

    Raises:
        TypeError: If ``min_cos_incidence`` is not a real scalar (``bool`` is rejected).
        ValueError: If ``min_cos_incidence`` is not finite or is not in ``[1e-15, 1]``
            (valid cosine clamp range).
    """

    name: str = 'lambert'
    min_cos_incidence: float = 0.01

    def __post_init__(self) -> None:
        _validate_cos_clamp('min_cos_incidence', self.min_cos_incidence)

    def correct(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Apply Lambert photometric correction.

        Parameters:
            data: Raw pixel brightness values.
            incidence: Incidence angle at each pixel (rad).
            emission: Emission angle at each pixel (rad). Not used.
            phase: Phase angle at each pixel (rad). Not used.

        Returns:
            Data divided by max(cos(incidence), min_cos_incidence).
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        return data / cos_i

    def uncorrect(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Undo :meth:`correct` (multiply by clamped ``cos(incidence)``).

        Parameters:
            data: Corrected radiance / reflectance (same shape as in :meth:`correct`).
            incidence: Incidence angle per pixel (rad); ``cos`` is clamped with
                ``self.min_cos_incidence`` like :meth:`correct`.
            emission: Accepted for API symmetry with :meth:`correct`; not used.
            phase: Accepted for API symmetry; not used.

        Returns:
            Uncorrected values ``data * max(cos(incidence), min_cos_incidence)``,
            ``NDArrayFloatType``, same shape as ``data`` (inverse of division in
            :meth:`correct` using the same clamping).
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        return data * cos_i


@dataclass
class LommelSeeligerModel:
    """Lommel-Seeliger photometric correction.

    Models single-scattering on a surface with a bidirectional reflectance
    proportional to 1/(mu0 + mu), where mu0=cos(incidence) and mu=cos(emission).
    The correction factor applied to the data is (mu0 + mu) / (2 * mu0), which
    normalizes out this scattering model.

    ``min_cos_incidence`` defaults to 0.01; see the member entry below.

    Raises:
        TypeError: If a numeric field is not a real scalar (``bool`` is rejected).
        ValueError: If a field is not finite or is outside its allowed range.
    """

    name: str = 'lommel_seeliger'
    min_cos_incidence: float = 0.01
    min_denom: float = 1e-15

    def __post_init__(self) -> None:
        _validate_cos_clamp('min_cos_incidence', self.min_cos_incidence)
        _validate_positive_floor('min_denom', self.min_denom)

    def _signed_sum_denom(
        self, cos_i: NDArrayFloatType, cos_e: NDArrayFloatType
    ) -> NDArrayFloatType:
        """Return ``cos_i + cos_e`` with magnitude floored to ``min_denom`` but sign preserved.

        When ``raw = cos_i + cos_e`` has ``abs(raw) < min_denom``, replace it with
        ``sign(raw) * min_denom``; when ``raw == 0``, use ``+min_denom`` so the
        value is never an unsigned zero denominator.
        """
        raw_denom = cos_i + cos_e
        too_small = np.abs(raw_denom) < self.min_denom
        sign_mag = np.where(
            raw_denom == 0,
            self.min_denom,
            np.sign(raw_denom) * self.min_denom,
        )
        return cast(NDArrayFloatType, np.where(too_small, sign_mag, raw_denom))

    def correct(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Apply Lommel-Seeliger photometric correction.

        Parameters:
            data: Raw pixel brightness values.
            incidence: Incidence angle at each pixel (rad).
            emission: Emission angle at each pixel (rad).
            phase: Phase angle at each pixel (rad). Not used.

        Returns:
            Data multiplied by ``denom / (2 * cos_i)``, where ``cos_i`` is
            ``max(cos(incidence), min_cos_incidence)``, ``cos_e = cos(emission)``,
            and ``denom`` is ``cos_i + cos_e`` with magnitude floored to
            ``min_denom`` while preserving sign (same rule as :meth:`uncorrect`).
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.cos(emission)
        denom = self._signed_sum_denom(cos_i, cos_e)
        return cast(NDArrayFloatType, data * denom / (2.0 * cos_i))

    def uncorrect(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Undo :meth:`correct` using the same incidence clamping and signed denom floor.

        Parameters:
            data: Values after Lommel-Seeliger :meth:`correct`.
            incidence: Incidence angles (rad); cosine clamped with ``min_cos_incidence``.
            emission: Emission angles (rad); ``cos(emission)`` contributes to the denominator.
            phase: Accepted for API symmetry; not used.

        Returns:
            ``NDArrayFloatType`` of same shape as ``data``, ``data * (2*cos_i) / denom``
            where ``denom`` is ``cos_i + cos_e`` with the same signed floor as :meth:`correct`.
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.cos(emission)
        denom = self._signed_sum_denom(cos_i, cos_e)
        return cast(NDArrayFloatType, data * (2.0 * cos_i) / denom)


@dataclass
class MinnaertModel:
    """Minnaert photometric correction.

    Generalizes the Lambert model with a limb-darkening exponent k. For k=1
    this reduces to the Lambert correction (1/cos_i). For k=0.5 this provides
    a uniform disk appearance across many surfaces.

    The correction divides data by cos(incidence)^k * cos(emission)^(k-1).

    Defaults: ``k`` = 0.5, ``min_cos_incidence`` = ``min_cos_emission`` = 0.01;
    see member entries below.

    Raises:
        TypeError: If a numeric field is not a real scalar (``bool`` is rejected).
        ValueError: If a field is not finite or cosine clamps are outside ``(0, 1]``.
    """

    name: str = 'minnaert'
    k: float = 0.5
    min_cos_incidence: float = 0.01
    min_cos_emission: float = 0.01

    def __post_init__(self) -> None:
        _photometric_real_scalar('k', self.k)
        _validate_cos_clamp('min_cos_incidence', self.min_cos_incidence)
        _validate_cos_clamp('min_cos_emission', self.min_cos_emission)

    def correct(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Apply Minnaert photometric correction.

        Parameters:
            data: Raw pixel brightness values.
            incidence: Incidence angle at each pixel (rad).
            emission: Emission angle at each pixel (rad).
            phase: Phase angle at each pixel (rad). Not used.

        Returns:
            Data divided by cos(incidence)^k * cos(emission)^(k-1).
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.maximum(np.cos(emission), self.min_cos_emission)
        return data / (cos_i**self.k * cos_e ** (self.k - 1.0))

    def uncorrect(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Undo :meth:`correct` using the same ``min_cos_incidence`` / ``min_cos_emission`` clamps.

        Parameters:
            data: Corrected radiance / reflectance (same shape as for :meth:`correct`).
            incidence: Incidence angles (rad); ``cos`` clamped to ``min_cos_incidence``.
            emission: Emission angles (rad); ``cos`` clamped to ``min_cos_emission``.
            phase: Accepted for API symmetry; not used.

        Returns:
            ``NDArrayFloatType``, same shape as ``data``, multiplying by
            ``cos_i**k * cos_e**(k-1)`` with the same clamped cosines as :meth:`correct`.
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.maximum(np.cos(emission), self.min_cos_emission)
        return data * (cos_i**self.k * cos_e ** (self.k - 1.0))


def photometric_model_from_name(
    name: str | None,
) -> LambertModel | LommelSeeligerModel | MinnaertModel | None:
    """Return a :class:`LambertModel`, :class:`LommelSeeligerModel`, or :class:`MinnaertModel`.

    Parameters:
        name: Stored model label from a reprojection/mosaic file, or ``None``.

    Returns:
        A fresh model instance, or ``None`` when ``name`` is ``None`` or normalizes
        to an empty string / ``none`` / ``null`` (explicit “no model”).

    Raises:
        ValueError: If ``name`` is non-empty after normalization but not one of the
            supported aliases (see Notes).

    Notes:
        Accepted aliases (case-insensitive, spaces and hyphens map to underscores):
        ``lambert``; ``lommel_seeliger`` / ``lommelseeliger``; ``minnaert``.
    """
    if name is None:
        return None
    n = str(name).strip().lower().replace('-', '_').replace(' ', '_')
    if n in ('', 'none', 'null'):
        return None
    if n == 'lambert':
        return LambertModel()
    if n in ('lommel_seeliger', 'lommelseeliger'):
        return LommelSeeligerModel()
    if n == 'minnaert':
        return MinnaertModel()
    raise ValueError(
        f'Unknown photometric model name {name!r}; expected None, "", "none", "null", '
        f'or one of lambert, lommel_seeliger, minnaert (see photometric_model_from_name).'
    )
