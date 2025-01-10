from abc import ABC, abstractmethod
import logging
from typing import Optional

import pdslogger

import nav.config.nav_logger


class NavModel(ABC):
    def __init__(self,
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = nav.config.nav_logger.DEFAULT_IMAGE_LOGGER
        else:
            self._logger = pdslogger.PdsLogger.as_pdslogger(logger)
