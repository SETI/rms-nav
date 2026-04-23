"""Photometric correction models for body and ring reprojections.

All models implement the PhotometricModel protocol and apply a correction to
raw pixel brightness based on the local illumination and viewing geometry.
Default is no correction (pass ``photometric_model=None`` to ``BodyMosaic`` or
``RingMosaic``).

All angles are in radians throughout.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from nav.support.types import NDArrayFloatType


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

    Attributes:
        min_cos_incidence: Minimum value of cos(incidence) used as the
            denominator. Pixels where cos(incidence) is below this threshold
            are clamped to this value. Defaults to 0.01 (~84 degrees).
    """

    name: str = 'lambert'
    min_cos_incidence: float = 0.01

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

    Attributes:
        min_cos_incidence: Minimum value of cos(incidence) for clamping.
            Defaults to 0.01.
    """

    name: str = 'lommel_seeliger'
    min_cos_incidence: float = 0.01
    min_denom: float = 1e-15

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
            Data multiplied by (cos_i + cos_e) / (2 * cos_i), where cos_i
            and cos_e are the cosines of incidence and emission respectively.
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.cos(emission)
        result: NDArrayFloatType = data * (cos_i + cos_e) / (2.0 * cos_i)
        return result

    def uncorrect(
        self,
        data: NDArrayFloatType,
        *,
        incidence: NDArrayFloatType,
        emission: NDArrayFloatType,
        phase: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Undo :meth:`correct` using the same incidence clamping and denominator floor.

        Parameters:
            data: Values after Lommel-Seeliger :meth:`correct`.
            incidence: Incidence angles (rad); cosine clamped with ``min_cos_incidence``.
            emission: Emission angles (rad); ``cos(emission)`` contributes to the denominator.
            phase: Accepted for API symmetry; not used.

        Returns:
            ``NDArrayFloatType`` of same shape as ``data``, ``data * (2*cos_i) / denom``
            where ``denom = max(|cos_i+cos_e|, min_denom)`` preserves sign.
        """
        cos_i = np.maximum(np.cos(incidence), self.min_cos_incidence)
        cos_e = np.cos(emission)
        denom = cos_i + cos_e
        denom = np.where(np.abs(denom) < self.min_denom, self.min_denom, denom)
        return cast(NDArrayFloatType, data * (2.0 * cos_i) / denom)


@dataclass
class MinnaertModel:
    """Minnaert photometric correction.

    Generalizes the Lambert model with a limb-darkening exponent k. For k=1
    this reduces to the Lambert correction (1/cos_i). For k=0.5 this provides
    a uniform disk appearance across many surfaces.

    The correction divides data by cos(incidence)^k * cos(emission)^(k-1).

    Attributes:
        k: Minnaert limb-darkening exponent. Default 0.5.
        min_cos_incidence: Minimum cos(incidence) for clamping. Defaults to 0.01.
        min_cos_emission: Minimum cos(emission) for clamping. Defaults to 0.01.
    """

    name: str = 'minnaert'
    k: float = 0.5
    min_cos_incidence: float = 0.01
    min_cos_emission: float = 0.01

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
