from typing import Any, Optional, cast

from filecache import FCPath
import numpy as np
from pathlib import Path

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER, Config
from nav.support.time import et_to_utc
from nav.support.types import PathLike

from .obs_snapshot_inst import ObsSnapshotInst


class ObsGalileoSSI(ObsSnapshotInst):
    """Implements an observation of a Galileo SSI image.

    This class provides specialized functionality for accessing and analyzing Galileo
    SSI image data.
    """

    @staticmethod
    def from_file(
        path: PathLike,
        config: Optional[Config] = None,
        extfov_margin_vu: tuple[int, int] | None = None,
        **_kwargs: Any,
    ) -> 'ObsGalileoSSI':
        """Creates an ObsGalileoSSI from a Galileo SSI image file.

        Parameters:
            path: Path to the Galileo SSI image file.
            config: Configuration object to use. If None, uses the default configuration.
            extfov_margin_vu: Optional tuple that overrides the extended field of view margins
                found in the config.
            **_kwargs: Additional keyword arguments (none for this instrument).

        Returns:
            An ObsGalileoSSI object containing the image data and metadata.
        """

        import oops.hosts.galileo.ssi

        config = config or DEFAULT_CONFIG
        logger = DEFAULT_LOGGER

        logger.debug(f'Reading Galileo SSI image {path}')
        obs = oops.hosts.galileo.ssi.from_file(path, full_fov=True)
        fc_path = FCPath(path)
        obs.abspath = cast(Path, fc_path.get_local_path()).absolute()
        obs.image_url = str(fc_path.absolute())

        inst_config = config.category('galileo_ssi')
        if extfov_margin_vu is None:
            if isinstance(inst_config.extfov_margin_vu, dict):
                # TODO Do this a better way
                extfov_margin_vu = inst_config.extfov_margin_vu[obs.data.shape[0]]
            else:
                extfov_margin_vu = inst_config.extfov_margin_vu
        logger.debug(f'  Data shape: {obs.data.shape}')
        logger.debug(f'  Extfov margin vu: {extfov_margin_vu}')
        logger.debug(f'  Data min: {np.min(obs.data)}, max: {np.max(obs.data)}')

        new_obs = ObsGalileoSSI(obs, config=config, extfov_margin_vu=extfov_margin_vu)
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
        """Returns the public metadata for Galileo SSI.

        Returns:
            A dictionary containing the public metadata for Galileo SSI.
        """

        # TODO
        # scet_start = float(obs.dict["SPACECRAFT_CLOCK_START_COUNT"])
        # scet_end = float(obs.dict["SPACECRAFT_CLOCK_STOP_COUNT"])

        return {
            'image_path': self.image_url,
            'image_name': self.abspath.name,
            'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.go',
            'instrument_lid': 'urn:nasa:pds:context:instrument:go.ssi',
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
            'camera': 'SSI',
            'exposure_time': self.texp,
            'filters': [self.filter],
        }
