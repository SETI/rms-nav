from typing import Any

import oops.hosts.voyager.iss

from nav.obs import ObsSnapshot
from nav.util.types import PathLike

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self,
                 obs: oops.Observation,
                 **kwargs: Any) -> None:
        super().__init__(obs, logger_name='InstVoyagerISS', **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> ObsSnapshot:
        obs = oops.hosts.voyager.iss.from_file(path)
        label3 = obs.dict['LABEL3'].replace(
            'FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        new_obs = ObsSnapshot(obs)
        new_obs.set_inst(InstVoyagerISS(new_obs))

        return new_obs