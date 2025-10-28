from typing import Any, Optional, cast

from filecache import FCPath
import numpy as np
import oops.hosts.cassini.iss
from starcat import Star

from .obs_inst import ObsInst
from .obs_snapshot import ObsSnapshot

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.support.time import et_to_utc
from nav.support.types import PathLike



class ObsCassiniISS(ObsSnapshot, ObsInst):

    @staticmethod
    def from_file(path: PathLike,
                  *,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> ObsSnapshot:
        """Creates an ObsCassiniISS from a Cassini ISS image file.

        Parameters:
            path: Path to the Cassini ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments:

                - fast_distortion: Whether to use a fast distortion model.

                - return_all_planets: Whether to return all planets.

        Returns:
            An ObsCassiniISS object containing the image data and metadata.
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

        new_obs = ObsCassiniISS(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        return new_obs

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Cassini ISS.

        Returns:
            A dictionary containing the public metadata for Cassini ISS.
        """

        scet_start = float(self.dict["SPACECRAFT_CLOCK_START_COUNT"])
        scet_end = float(self.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': str(self.abspath),
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
            'instrument_lid': f'urn:nasa:pds:context:instrument:iss{self.detector[0].lower()}a.co',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            'start_time_scet': scet_start,
            'midtime_scet': (scet_start + scet_end) / 2,
            'end_time_scet': scet_end,
            'image_shape_xy': self.data_shape_uv,
            'camera': self.detector,
            'exposure_time': self.texp,
            'filters': [self.filter1, self.filter2],
            'sampling': self.sampling,
            'gain_mode': self.gain_mode,
            'description': self.dict.get("DESCRIPTION", None),
            'observation_id': self.dict.get("OBSERVATION_ID", None)
        }
