from typing import Any

import oops.hosts.galileo.ssi as galileo_ssi

from nav.util.types import PathLike

from .inst import Inst


class InstGalileoSSI(Inst):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_file(path: PathLike) -> 'InstGalileoSSI':
        obs = galileo_ssi.from_file(path)
        return InstGalileoSSI(obs)
