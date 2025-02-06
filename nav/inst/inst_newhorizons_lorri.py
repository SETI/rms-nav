from typing import Any, Optional

import oops.hosts.newhorizons.lorri
from psfmodel import GaussianPSF, PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.util.types import PathLike

from .inst import Inst


class InstNewHorizonsLORRI(Inst):
    def __init__(self,
                 obs: oops.Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstNewHorizonsLORRI', **kwargs)

    @staticmethod
    def from_file(path: PathLike,
                  config: Optional[Config] = None) -> ObsSnapshot:
        config = config or DEFAULT_CONFIG
        obs = oops.hosts.newhorizons.lorri.from_file(path)
        new_obs = ObsSnapshot(obs, config=config)
        new_obs.set_inst(InstNewHorizonsLORRI(new_obs, config=config))

        return new_obs

    def star_psf(self) -> PSF:
        return GaussianPSF(sigma=1.)
