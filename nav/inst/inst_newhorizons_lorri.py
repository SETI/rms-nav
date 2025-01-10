from typing import Any

import oops.hosts.newhorizons.lorri as nh_lorri

from nav.util.types import PathLike

from .inst import Inst


class InstNewHorizonsLORRI(Inst):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> 'InstNewHorizonsLORRI':
        obs = nh_lorri.from_file(path)
        return InstNewHorizonsLORRI(obs)
