from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from psfmodel import GaussianPSF, PSF
from starcat import Star

from nav.config import Config
from nav.support.types import PathLike

if TYPE_CHECKING:
    from nav.obs import Obs


class ObsInst(ABC):
    """Mix-in class for instrument models representing spacecraft cameras."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._inst_config: dict[str, Any] | None = None

    @property
    def inst_config(self) -> dict[str, Any] | None:
        """Returns the instrument configuration."""
        return self._inst_config

    @staticmethod
    @abstractmethod
    def from_file(path: PathLike,
                  *,
                  config: Optional[Config] = None,
                  extfov_margin_vu: Optional[tuple[int, int]] = None,
                  **kwargs: Any) -> 'Obs':
        """Creates an instrument instance from an image file.

        Parameters:
            path: Path to the image file.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            extfov_margin_vu: Optional tuple specifying the extended field of view margins
                in (vertical, horizontal) pixels.
            **kwargs: Additional keyword arguments to pass to the instrument constructor.

        Returns:
            An Obs object containing the image data and metadata.
        """
        ...

    def star_psf(self) -> PSF:
        """Returns the point spread function (PSF) model appropriate for stars observed
        by this instrument.

        This generic implementation uses the "star_psf_sigma" configuration value and
        creates a Gaussian PSF with that sigma.

        Returns:
            A PSF model appropriate for stars observed by this instrument.
        """

        if self._inst_config is None:
            raise ValueError('Instrument configuration not set')

        sigma = self._inst_config['star_psf_sigma']
        return GaussianPSF(sigma=sigma)

    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        This generic implementation uses the "star_psf_sizes" configuration value and
        returns the appropriate value for the star's magnitude.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.
        """

        if self._inst_config is None:
            raise ValueError('Instrument configuration not set')

        star_psf_sizes = self._inst_config['star_psf_sizes']
        for mag in sorted(star_psf_sizes):
            if star.vmag < mag:
                return star_psf_sizes[mag]
        return star_psf_sizes[mag]  # Default to largest

    @abstractmethod
    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Returns:
            The minimum usable magnitude for stars in this observation.
        """

        raise NotImplementedError

    @abstractmethod
    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Returns:
            The maximum usable magnitude for stars in this observation.
        """

        raise NotImplementedError

    @abstractmethod
    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for this instrument."""
        ...
