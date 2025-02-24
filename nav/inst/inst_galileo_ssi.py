from typing import Any, Optional

from oops import Observation
import oops.hosts.galileo.ssi
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.support.types import PathLike

from .inst import Inst


class InstGalileoSSI(Inst):
    def __init__(self,
                 obs: Observation,
                 config: Config,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstGalileoSSI', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None) -> ObsSnapshot:
        config = config or DEFAULT_CONFIG
        obs = oops.hosts.galileo.ssi.from_file(path)
        new_obs = ObsSnapshot(obs, config=config)
        new_obs.set_inst(InstGalileoSSI(new_obs, config=config))

        return new_obs

    def star_psf(self) -> PSF:
        return GaussianPSF(sigma=1.)
