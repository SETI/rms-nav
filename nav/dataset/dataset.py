from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import pdslogger

import nav.config.nav_logger


class DataSet(ABC):
    def __init__(self,
                 logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = nav.config.nav_logger.DEFAULT_IMAGE_LOGGER
        else:
            self._logger = pdslogger.PdsLogger.as_pdslogger(logger)

    @staticmethod
    @abstractmethod
    def image_name_valid(name: str) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(*args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def yield_image_filenames_index(*args: Any, **kwargs: Any) -> Iterator[Path]:
        ...
