from typing import Any, Optional

import numpy as np
from oops import Observation
import oops.hosts.cassini.iss
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Initializes a Cassini ISS instrument instance.

        Parameters:
            obs: The Observation object containing Cassini ISS image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(obs, logger_name='InstCassiniISS', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> ObsSnapshot:
        """Creates an ObsSnapshot from a Cassini ISS image file.

        Parameters:
            path: Path to the Cassini ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments:

                - fast_distortion: Whether to use a fast distortion model.

                - return_all_planets: Whether to return all planets.

        Returns:
            An ObsSnapshot object containing the image data and metadata.
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        fast_distortion = kwargs.get('fast_distortion', True)
        return_all_planets = kwargs.get('return_all_planets', True)

        logger.debug(f'Reading Cassini ISS image {path}')
        logger.debug(f'  Fast distortion: {fast_distortion}')
        logger.debug(f'  Return all planets: {return_all_planets}')
        obs = oops.hosts.cassini.iss.from_file(path,
                                               fast_distortion=fast_distortion,
                                               return_all_planets=return_all_planets)

        if extfov_margin_vu is None:
            extfov_margin_vu = config._config_dict['cassini_iss']['extfov_margin_vu'][
                obs.data.shape[0]]
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs,
                              config=config,
                              extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstCassiniISS(new_obs, config=config))
        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function for Cassini ISS stars.

        Returns:
            A Gaussian PSF object with the appropriate sigma value for Cassini ISS.
        """
        return GaussianPSF(sigma=0.54)
# PSF_SIGMA = {"NAC":   0.54,
#              "WAC":   0.77,
#              "LORRI": 0.5} # XXX
