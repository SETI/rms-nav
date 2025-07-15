from typing import Any, Optional

import numpy as np
from oops import Observation
import oops.hosts.voyager.iss
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Initializes a Voyager ISS instrument instance.

        Parameters:
            obs: The Observation object containing Voyager ISS image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(obs, logger_name='InstVoyagerISS', **kwargs)

    @staticmethod
    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> ObsSnapshot:
        """Creates an ObsSnapshot from a Voyager ISS image file.

        Parameters:
            path: Path to the Voyager ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsSnapshot object containing the image data and metadata.
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading Voyager ISS image {path}')
        obs = oops.hosts.voyager.iss.from_file(path)

        label3 = obs.dict['LABEL3'].replace('FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            extfov_margin_vu = config._config_dict['voyager_iss']['extfov_margin_vu'][
                obs.data.shape[0]]
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstVoyagerISS(new_obs, config=config))
        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function for Voyager ISS stars.

        Returns:
            A Gaussian PSF object with the appropriate sigma value for Voyager ISS.
        """
        return GaussianPSF(sigma=1.)  # TODO
