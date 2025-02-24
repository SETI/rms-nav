from abc import ABC, abstractmethod
from typing import Optional

from oops import Observation
from psfmodel import PSF

from nav.config import Config
from nav.support.nav_base import NavBase
from nav.support.types import PathLike


class Inst(ABC, NavBase):
    def __init__(self,
                 obs: Observation,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        super().__init__(config=config, logger_name=logger_name)

        self._obs = obs

    @staticmethod
    @abstractmethod
    def from_file(path: PathLike) -> 'Inst':
        ...

    @property
    def obs(self) -> Observation:
        return self._obs

    @abstractmethod
    def star_psf(self) -> PSF:
        ...
