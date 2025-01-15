from typing import Any

import oops.hosts.galileo.ssi

from nav.obs import ObsSnapshot
from nav.util.types import PathLike

from .inst import Inst


class InstGalileoSSI(Inst):
    def __init__(self,
                 obs: oops.Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstGalileoSSI', **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> ObsSnapshot:
        obs = oops.hosts.galileo.ssi.from_file(path)
        new_obs = ObsSnapshot(obs)
        new_obs.set_inst(InstGalileoSSI(new_obs))

        return new_obs