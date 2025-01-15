from abc import ABC, abstractmethod
import logging
from typing import Optional

import oops
import pdslogger

import nav.config.nav_logger
from nav.util.types import PathLike


class Inst(ABC):
    def __init__(self,
                 obs: oops.Observation,
                 *,
                 logger: Optional[logging.Logger] = None,
                 logger_name: Optional[str] = None) -> None:
        self._obs = obs

        if logger is None:
            self._logger = nav.config.nav_logger.DEFAULT_IMAGE_LOGGER
        else:
            self._logger = pdslogger.PdsLogger.as_pdslogger(logger)

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @staticmethod
    @abstractmethod
    def from_file(path: PathLike) -> 'Inst':
        ...

    @property
    def obs(self) -> oops.Observation:
        return self._obs
