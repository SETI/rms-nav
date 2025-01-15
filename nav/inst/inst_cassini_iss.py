from typing import Any

import oops
import oops.hosts.cassini.iss

from nav.obs import ObsSnapshot
from nav.util.types import PathLike

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self,
                 obs: oops.Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstCassiniISS', **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> ObsSnapshot:
        obs = oops.hosts.cassini.iss.from_file(path,
                                               fast_distortion=True,
                                               return_all_planets=True)
        # TODO Calibrate
        # obs.data = obs.extended_calib.value_from_dn(obs.data).vals
        new_obs = ObsSnapshot(obs)
        new_obs.set_inst(InstCassiniISS(new_obs))

        return new_obs