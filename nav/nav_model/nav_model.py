from abc import ABC, abstractmethod
from typing import Optional

import oops
from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG
from nav.inst import Inst
from nav.obs import ObsSnapshot


class NavModel(ABC):
    def __init__(self,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        self._obs = obs
        self._config = config or DEFAULT_CONFIG
        self._logger = self._config.logger
        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> PdsLogger:
        return self._logger

    @property
    def obs(self) -> ObsSnapshot:
        return self._obs

    @property
    def inst(self) -> Inst:
        return self.obs.inst
