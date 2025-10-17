from typing import Any, Optional, cast

from filecache import FCPath
import numpy as np
from oops import Observation
import oops.hosts.cassini.iss
from psfmodel import GaussianPSF, PSF
from starcat import Star

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes a Cassini ISS instrument instance.

        Parameters:
            obs: The Observation object containing Cassini ISS image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(obs, config=config)

        self._inst_config: dict[str, Any] | None = None

    @staticmethod
    def from_file(path: PathLike,
                  *,
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

        detector = obs.detector.lower()
        inst_config = config._config_dict['cassini_iss'][detector]

        if extfov_margin_vu is None:
            extfov_margin_vu_entry = inst_config['extfov_margin_vu']
            if isinstance(extfov_margin_vu_entry, dict):
                extfov_margin_vu = extfov_margin_vu_entry[obs.data.shape[0]]
            else:
                extfov_margin_vu = extfov_margin_vu_entry
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        # TODO This is slow when debug turned off
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs,
                              config=config,
                              extfov_margin_vu=extfov_margin_vu)
        inst = InstCassiniISS(new_obs, config=config)
        inst._inst_config = inst_config
        new_obs.set_inst(inst)
        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function (PSF) model appropriate for stars observed
        by this instrument.

        Returns:
            A PSF model appropriate for stars observed by this instrument.
        """

        if self._inst_config is None:
            raise ValueError('InstCassiniISS not initialized')

        sigma = self._inst_config['star_psf_sigma']
        return GaussianPSF(sigma=sigma)

    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.
        """

        if self._inst_config is None:
            raise ValueError('InstCassiniISS not initialized')

        return cast(tuple[int, int], self._inst_config['star_psf_size'])

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
            'filters': [obs.filter1, obs.filter2],
            'sampling': obs.sampling,
            'gain_mode': obs.gain_mode,
            'description': obs.dict.get("DESCRIPTION", None),
            'observation_id': obs.dict.get("OBSERVATION_ID", None)
        }
