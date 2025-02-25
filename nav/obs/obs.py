from abc import ABC
from typing import Optional, cast

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG
from nav.inst import Inst


class Obs(ABC):
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._logger = self._config.logger
        self._inst: Optional[Inst] = None

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def logger(self) -> PdsLogger:
        return self._logger

    def set_inst(self, inst: 'Inst') -> None:
        self._inst = inst

    @property
    def inst(self) -> 'Inst':
        return cast(Inst, self._inst)
