from typing import Any, Optional

from filecache import FCPath
from oops import Observation
import oops.hosts.voyager.iss
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self,
                 obs: Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstVoyagerISS', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None,
                  extfov_margin_vu: tuple[int, int] | None = None) -> ObsSnapshot:

        config = config or DEFAULT_CONFIG
        obs = oops.hosts.voyager.iss.from_file(path)
        label3 = obs.dict['LABEL3'].replace('FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        # TODO Calibrate once oops.hosts is fixed.

        new_obs = ObsSnapshot(obs, config=config, extfov_margin_vu=extfov_margin_vu)
        new_obs.set_inst(InstVoyagerISS(new_obs, config=config))
        return new_obs

    def star_psf(self) -> PSF:
        return GaussianPSF(sigma=1.)  # TODO
