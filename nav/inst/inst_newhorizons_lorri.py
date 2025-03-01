from typing import Any, Optional

from oops import Observation
import oops.hosts.newhorizons.lorri
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstNewHorizonsLORRI(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Initializes a New Horizons LORRI instrument instance.
        
        Parameters:
            obs: The Observation object containing New Horizons LORRI image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(obs, logger_name='InstNewHorizonsLORRI', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None) -> ObsSnapshot:
        """Creates an ObsSnapshot from a New Horizons LORRI image file.
        
        Parameters:
            path: Path to the New Horizons LORRI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple specifying the extended field of view margins
                in (vertical, horizontal) pixels.
                
        Returns:
            An ObsSnapshot object containing the image data and metadata.
        """
        config = config or DEFAULT_CONFIG
        obs = oops.hosts.newhorizons.lorri.from_file(path)
        # TODO Calibrate once oops.hosts is fixed.
        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstNewHorizonsLORRI(new_obs, config=config))

        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function for New Horizons LORRI stars.
        
        Returns:
            A Gaussian PSF object with the appropriate sigma value for New Horizons LORRI.
        """
        return GaussianPSF(sigma=1.)  # TODO
