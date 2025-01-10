from typing import Any

import oops.hosts.cassini.iss as cassini_iss

from nav.util.types import PathLike

from .inst import Inst


class InstCassiniISS(Inst):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> 'InstCassiniISS':
        obs = cassini_iss.from_file(path, fast_distortion=True, return_all_planets=True)
        # obs.data = obs.extended_calib.value_from_dn(obs.data).vals
        return InstCassiniISS(obs)
