from abc import ABC
import logging
from typing import Optional

import pdslogger

import nav.config.nav_logger as nav_logger


class NavTechnique(ABC):
    def __init__(self,
                 *,
                 logger: Optional[logging.Logger] = None,
                 logger_name: Optional[str] = None) -> None:
        if logger is None:
            self._logger = nav_logger.DEFAULT_IMAGE_LOGGER
        else:
            self._logger = pdslogger.PdsLogger.as_pdslogger(logger)

        if logger_name is not None:
            self._logger = self._logger.get_logger(logger_name)
