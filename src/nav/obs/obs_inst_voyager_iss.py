from pathlib import Path
from typing import Any, cast

import numpy as np
from filecache import FCPath

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst


class ObsVoyagerISS(ObsSnapshotInst):
    """Implements an observation of a Voyager ISS image.

    This class provides specialized functionality for accessing and analyzing Voyager
    ISS image data.
    """

    @staticmethod
    def from_file(
        path: PathLike,
        *,
        config: Config | None = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **_kwargs: Any,
    ) -> 'ObsVoyagerISS':
        """Creates an ObsVoyagerISS from a Voyager ISS image file.

        Parameters:
            path: Path to the Voyager ISS image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsVoyagerISS object containing the image data and metadata.
        """

        import oops.hosts.voyager.iss

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading Voyager ISS image {path}')
        obs = oops.hosts.voyager.iss.from_file(path)
        fc_path = FCPath(path)
        obs.abspath = cast(Path, fc_path.get_local_path()).absolute()
        obs.image_url = str(fc_path.absolute())

        inst_config = config.category('voyager_iss')

        label3 = obs.dict['LABEL3'].replace('FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        # TODO Calibrate once oops.hosts is fixed.

        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                extfov_margin_vu = inst_config.extfov_margin_vu[obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsVoyagerISS(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs._inst_config = inst_config
        return new_obs

    def star_min_usable_vmag(self) -> float:
        """Returns the minimum usable magnitude for stars in this observation.

        Returns:
            The minimum usable magnitude for stars in this observation.
        """
        return 0.0

    def star_max_usable_vmag(self) -> float:
        """Returns the maximum usable magnitude for stars in this observation.

        Returns:
            The maximum usable magnitude for stars in this observation.
        """
        return 10  # TODO

    def get_public_metadata(self) -> dict[str, Any]:
        """Returns the public metadata for Voyager ISS.

        Returns:
            A dictionary containing the public metadata for Voyager ISS.
        """

        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        spacecraft = self.dict['LAB02'][4]

        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': f'urn:nasa:pds:context:instrument_host:spacecraft.vg{spacecraft}',
            'instrument_lid': f'urn:nasa:pds:context:instrument:vg{spacecraft}.iss{self.detector[0].lower()}',
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
            'camera': self.detector,
            'exposure_time': self.texp,
            'filters': [self.filter],
        }
