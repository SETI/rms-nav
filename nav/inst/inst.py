from abc import ABC, abstractmethod
from typing import Optional

import oops
from pdslogger import PdsLogger
from psfmodel import PSF

from nav.config import Config, DEFAULT_CONFIG
from nav.util.types import PathLike


class Inst(ABC):
    def __init__(self,
                 obs: oops.Observation,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        self._obs = obs
        self._config = config or DEFAULT_CONFIG
        self._logger = self._config.logger

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @staticmethod
    @abstractmethod
    def from_file(path: PathLike) -> 'Inst':
        ...

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> PdsLogger:
        return self._logger

    @property
    def obs(self) -> oops.Observation:
        return self._obs

    @abstractmethod
    def star_psf(self) -> PSF:
        ...
