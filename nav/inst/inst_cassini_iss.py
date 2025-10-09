from typing import Any, Optional

from filecache import FCPath
import numpy as np
from oops import Observation
import oops.hosts.cassini.iss
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.time import et_to_utc
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
        path = FCPath(path).absolute()
        obs = oops.hosts.cassini.iss.from_file(path,
                                               fast_distortion=fast_distortion,
                                               return_all_planets=return_all_planets)
        obs.abspath = path

        if extfov_margin_vu is None:
            if isinstance(config._config_dict['cassini_iss']['extfov_margin_vu'], dict):
                extfov_margin_vu = config._config_dict['cassini_iss']['extfov_margin_vu'][
                    obs.data.shape[0]]
            else:
                extfov_margin_vu = config._config_dict['cassini_iss']['extfov_margin_vu']
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        # TODO This is slow when debug turned off
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
#              "LORRI": 0.5} # TODO

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Cassini ISS.

        Returns:
            A dictionary containing the public metadata for Cassini ISS.
        """

        obs = self.obs
        scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': str(obs.abspath),
            'image_name': obs.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
            'instrument_lid': f'urn:nasa:pds:context:instrument:iss{obs.detector[0].lower()}a.co',
            'start_time_utc': et_to_utc(obs.time[0]),
            'midtime_utc': et_to_utc(obs.midtime),
            'end_time_utc': et_to_utc(obs.time[1]),
            'start_time_et': obs.time[0],
            'midtime_et': obs.midtime,
            'end_time_et': obs.time[1],
            'start_time_scet': scet_start,
            'midtime_scet': (scet_start + scet_end) / 2,
            'end_time_scet': scet_end,
            'image_shape_xy': obs.data_shape_uv,
            'camera': obs.detector,
            'exposure_time': obs.texp,
            'filters': (obs.filter1, obs.filter2),
            'sampling': obs.sampling,
            'gain_mode': obs.gain_mode,
            'description': obs.dict["DESCRIPTION"],
            'observation_id': obs.dict["OBSERVATION_ID"]
        }
