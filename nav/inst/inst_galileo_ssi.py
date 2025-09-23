from typing import Any, Optional

import numpy as np
from oops import Observation
import oops.hosts.galileo.ssi
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstGalileoSSI(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        """Initializes a Galileo SSI instrument instance.

        Parameters:
            obs: The Observation object containing Galileo SSI image data.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(obs, logger_name='InstGalileoSSI', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> ObsSnapshot:
        """Creates an ObsSnapshot from a Galileo SSI image file.

        Parameters:
            path: Path to the Galileo SSI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsSnapshot object containing the image data and metadata.
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading Galileo SSI image {path}')
        obs = oops.hosts.galileo.ssi.from_file(path, full_fov=True)

        if extfov_margin_vu is None:
            if isinstance(config._config_dict['galileo_ssi']['extfov_margin_vu'], dict):
                # TODO Do this a better way
                extfov_margin_vu = config._config_dict['galileo_ssi']['extfov_margin_vu'][
                    obs.data.shape[0]]
            else:
                extfov_margin_vu = config._config_dict['galileo_ssi']['extfov_margin_vu']
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstGalileoSSI(new_obs, config=config))

        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function for Galileo SSI stars.

        Returns:
            A Gaussian PSF object with the appropriate sigma value for Galileo SSI.
        """
        return GaussianPSF(sigma=3.)  # TODO
