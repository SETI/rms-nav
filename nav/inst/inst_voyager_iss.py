from typing import Any, Optional

from filecache import FCPath
import numpy as np
from oops import Observation
import oops.hosts.voyager.iss
from psfmodel import GaussianPSF, PSF
from starcat import Star

from nav.config import Config, DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes a Voyager ISS instrument instance.

        Parameters:
            obs: The Observation object containing Voyager ISS image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(obs, config=config)

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
        path = FCPath(path).absolute()
        obs = oops.hosts.voyager.iss.from_file(path)
        obs.abspath = path

        label3 = obs.dict['LABEL3'].replace('FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            if isinstance(config._config_dict['voyager_iss']['extfov_margin_vu'], dict):
                extfov_margin_vu = config._config_dict['voyager_iss']['extfov_margin_vu'][
                    obs.data.shape[0]]
            else:
                extfov_margin_vu = config._config_dict['voyager_iss']['extfov_margin_vu']
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstVoyagerISS(new_obs, config=config))
        return new_obs

    def star_psf(self) -> PSF:
        """Returns the point spread function (PSF) model appropriate for stars observed
        by this instrument.

        Returns:
            A PSF model appropriate for stars observed by this instrument.
        """

        return GaussianPSF(sigma=3.)  # TODO

    def star_psf_size(self, star: Star) -> tuple[int, int]:
        """Returns the size of the point spread function (PSF) to use for a star.

        Parameters:
            star: The star to get the PSF size for.

        Returns:
            A tuple of the PSF size (v, u) in pixels.
        """

        return (7, 7)  # TODO

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Voyager ISS.

        Returns:
            A dictionary containing the public metadata for Voyager ISS.
        """

        obs = self.obs
        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        spacecraft = obs.dict['LAB02'][4]

        return {
            'image_path': str(obs.abspath),
            'image_name': obs.abspath.name,
            'instrument_host_lid':
                f'urn:nasa:pds:context:instrument_host:spacecraft.vg{spacecraft}',
            'instrument_lid':
                f'urn:nasa:pds:context:instrument:vg{spacecraft}.iss{obs.detector[0].lower()}',
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
            'camera': obs.detector,
            'exposure_time': obs.texp,
            'filters': [obs.filter],
        }
