from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from oops import Observation
from psfmodel import PSF
from starcat import Star

from nav.config import Config
from nav.support.nav_base import NavBase
from nav.support.types import PathLike

if TYPE_CHECKING:
    from nav.obs import Obs


class Inst(ABC, NavBase):
    """Base class for instrument models representing spacecraft cameras."""

    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes an instrument model with observation data.

        Parameters:
            obs: OOPS Observation object containing instrument and pointing information.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(config=config)

        self._obs = obs

    @staticmethod
    @abstractmethod
    def from_file(path: PathLike,
                  *,
                  config: Optional[Config] = None,
                  extfov_margin_vu: Optional[tuple[int, int]] = None,
                  **kwargs: Optional[Any]) -> 'Obs':
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

    @property
    def obs(self) -> Observation:
        """Returns the observation object associated with this instrument."""
        return self._obs

    @abstractmethod
    def star_psf(self) -> PSF:
        """Returns the point spread function (PSF) model appropriate for stars observed
        by this instrument.

        Returns:
            A PSF model appropriate for stars observed by this instrument.
        """
        ...

    @abstractmethod
    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.
        """
        ...

    @abstractmethod
    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for this instrument."""
        ...
