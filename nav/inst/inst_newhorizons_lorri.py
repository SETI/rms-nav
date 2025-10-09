from typing import Any, Optional

from filecache import FCPath
import numpy as np
from oops import Observation
import oops.hosts.newhorizons.lorri
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.time import et_to_utc
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
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> ObsSnapshot:
        """Creates an ObsSnapshot from a New Horizons LORRI image file.

        Parameters:
            path: Path to the New Horizons LORRI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsSnapshot object containing the image data and metadata.
        """
        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading New Horizons LORRI image {path}')
        # TODO calibration=False is required because the hosts module can't find things like
        # the distance from the Sun to M7. How do we handle this?
        path = FCPath(path).absolute()
        obs = oops.hosts.newhorizons.lorri.from_file(path, calibration=False)
        obs.abspath = path

        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            if isinstance(config._config_dict['newhorizons_lorri']['extfov_margin_vu'], dict):
                extfov_margin_vu = config._config_dict['newhorizons_lorri']['extfov_margin_vu'][
                    obs.data.shape[0]]
            else:
                extfov_margin_vu = config._config_dict['newhorizons_lorri']['extfov_margin_vu']
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstNewHorizonsLORRI(new_obs, config=config))

        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function for New Horizons LORRI stars.

        Returns:
            A Gaussian PSF object with the appropriate sigma value for New Horizons LORRI.
        """
        return GaussianPSF(sigma=1.)  # TODO

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for New Horizons LORRI.

        Returns:
            A dictionary containing the public metadata for New Horizons LORRI.
        """

        obs = self.obs
        print(obs.__dict__.keys())
        # TODO
        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': str(obs.abspath),
            'image_name': obs.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.nh',
            'instrument_lid': 'urn:nasa:pds:context:instrument:nh.lorri',
            'start_time_utc': et_to_utc(obs.time[0]),
            'midtime_utc': et_to_utc(obs.midtime),
            'end_time_utc': et_to_utc(obs.time[1]),
            'start_time_et': obs.time[0],
            'midtime_et': obs.midtime,
            'end_time_et': obs.time[1],
            # 'start_time_scet': scet_start,
            # 'midtime_scet': (scet_start + scet_end) / 2,
            # 'end_time_scet': scet_end,
            'image_shape_xy': obs.data_shape_uv,
            'camera': 'LORRI',
            'exposure_time': obs.texp,
            'filters': None,
        }
