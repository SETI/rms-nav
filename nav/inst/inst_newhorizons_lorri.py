from typing import Any

import oops.hosts.newhorizons.lorri

from nav.obs import ObsSnapshot
from nav.util.types import PathLike

from .inst import Inst


class InstNewHorizonsLORRI(Inst):
    def __init__(self,
                 obs: oops.Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstNewHorizonsLORRI', **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> ObsSnapshot:
        obs = oops.hosts.newhorizons.lorri.from_file(path)
        new_obs = ObsSnapshot(obs)
        new_obs.set_inst(InstNewHorizonsLORRI(new_obs))

        return new_obs