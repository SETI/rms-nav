"""Photometric correction models for body and ring reprojections.

All models implement the PhotometricModel protocol and apply a correction to
raw pixel brightness based on the local illumination and viewing geometry.
Default is no correction (pass photometric_model=None to BodyMosaic or
RingMosaic).

All angles are in radians throughout.
"""

from dataclasses import dataclass
from typing import Protocol

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
