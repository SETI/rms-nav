from abc import ABC
import logging
from typing import Optional, cast

import pdslogger

import nav.config.nav_logger
from nav.inst import Inst


class Obs(ABC):
    def __init__(self,
                 *,
                 logger: Optional[logging.Logger] = None,
                 logger_name: Optional[str] = None) -> None:

        self._inst: Optional[Inst] = None

        if logger is None:
            self._logger = nav.config.nav_logger.DEFAULT_IMAGE_LOGGER
        else:
            self._logger = pdslogger.PdsLogger.as_pdslogger(logger)

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)

    def set_inst(self, inst: Inst) -> None:
        self._inst = inst

    @property
    def inst(self) -> Inst:
        return cast(Inst, self._inst)
