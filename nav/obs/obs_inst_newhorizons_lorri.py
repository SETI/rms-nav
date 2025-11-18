from typing import Any, Optional

from filecache import FCPath
import numpy as np

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst


class ObsNewHorizonsLORRI(ObsSnapshotInst):
    """Implements an observation of a New Horizons LORRI image.

    This class provides specialized functionality for accessing and analyzing New
    Horizons LORRI image data.
    """

    @staticmethod
    def from_file(path: PathLike,
                  *,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None,
                  **_kwargs: Any) -> 'ObsNewHorizonsLORRI':
        """Creates an ObsNewHorizonsLORRI from a New Horizons LORRI image file.

        Parameters:
            path: Path to the New Horizons LORRI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsNewHorizonsLORRI object containing the image data and metadata.
        """

        import oops.hosts.newhorizons.lorri

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading New Horizons LORRI image {path}')
        # TODO calibration=False is required because the hosts module can't find things like
        # the distance from the Sun to M7. How do we handle this?
        path = FCPath(path).absolute()
        obs = oops.hosts.newhorizons.lorri.from_file(path, calibration=False)
        obs.abspath = path

        inst_config = config.category('newhorizons_lorri')
        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                extfov_margin_vu = inst_config.extfov_margin_vu[obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsNewHorizonsLORRI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        return new_obs

    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Returns:
            The minimum usable magnitude for stars in this observation.
        """
        return 0.

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Returns:
            The maximum usable magnitude for stars in this observation.
        """
        return 10  # TODO

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for New Horizons LORRI.

        Returns:
            A dictionary containing the public metadata for New Horizons LORRI.
        """

        # TODO
        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': str(self.abspath),
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.nh',
            'instrument_lid': 'urn:nasa:pds:context:instrument:nh.lorri',
            'start_time_utc': et_to_utc(self.time[0]),
            'midtime_utc': et_to_utc(self.midtime),
            'end_time_utc': et_to_utc(self.time[1]),
            'start_time_et': self.time[0],
            'midtime_et': self.midtime,
            'end_time_et': self.time[1],
            # 'start_time_scet': scet_start,
            # 'midtime_scet': (scet_start + scet_end) / 2,
            # 'end_time_scet': scet_end,
            'image_shape_xy': self.data_shape_uv,
            'camera': 'LORRI',
            'exposure_time': self.texp,
            'filters': [],
        }
