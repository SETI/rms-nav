from typing import Any

import oops.hosts.voyager.iss as voyager_iss

from nav.util.types import PathLike

from .inst import Inst


class InstVoyagerISS(Inst):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> 'InstVoyagerISS':
        obs = voyager_iss.from_file(path)
        label3 = obs.dict['LABEL3'].replace(
            'FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')
        factor = float(label3)
        obs.data = obs.data * factor / 10000
        # TODO Beware - Voyager 1 @ Saturn requires data to be multiplied by 3.345
        # because the Voyager 1 calibration pipeline always assumes the image
        # was taken at Jupiter.
        return InstVoyagerISS(obs)
