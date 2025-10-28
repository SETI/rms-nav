from typing import Any, Optional

from filecache import FCPath
import numpy as np
from oops import Observation
import oops.hosts.galileo.ssi
from psfmodel import GaussianPSF, PSF
from starcat import Star

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .obs_inst import ObsInst


class ObsGalileoSSI(ObsInst):

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **kwargs: Any) -> 'ObsGalileoSSI':
        """Creates an ObsGalileoSSI from a Galileo SSI image file.

        Parameters:
            path: Path to the Galileo SSI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsGalileoSSI object containing the image data and metadata.
        """

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading Galileo SSI image {path}')
        path = FCPath(path).absolute()
        obs = oops.hosts.galileo.ssi.from_file(path, full_fov=True)
        obs.abspath = path

        inst_config = config._config_dict['galileo_ssi']
        if extfov_margin_vu is None:
            if isinstance(inst_config['extfov_margin_vu'], dict):
                # TODO Do this a better way
                extfov_margin_vu = inst_config['extfov_margin_vu'][obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config['extfov_margin_vu']
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsGalileoSSI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
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
        """Returns the public metadata for Galileo SSI.

        Returns:
            A dictionary containing the public metadata for Galileo SSI.
        """
        # TODO
        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': str(self.obs.abspath),
            'image_name': self.obs.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.go',
            'instrument_lid': 'urn:nasa:pds:context:instrument:go.ssi',
            'start_time_utc': et_to_utc(self.obs.time[0]),
            'midtime_utc': et_to_utc(self.obs.midtime),
            'end_time_utc': et_to_utc(self.obs.time[1]),
            'start_time_et': self.obs.time[0],
            'midtime_et': self.obs.midtime,
            'end_time_et': self.obs.time[1],
            # 'start_time_scet': scet_start,
            # 'midtime_scet': (scet_start + scet_end) / 2,
            # 'end_time_scet': scet_end,
            'image_shape_xy': self.obs.data_shape_uv,
            'camera': 'SSI',
            'exposure_time': self.obs.texp,
            'filters': [obs.filter],
        }
