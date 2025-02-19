from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG


class DataSet(ABC):
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
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

    @staticmethod
    @abstractmethod
    def image_name_valid(name: str) -> bool:
        ...

    @staticmethod
    @abstractmethod
    def add_selection_arguments(*args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def yield_image_filenames_from_arguments(*args: Any, **kwargs: Any) -> Iterator[Path]:
        ...

    @abstractmethod
    def yield_image_filenames_index(*args: Any, **kwargs: Any) -> Iterator[Path]:
        ...
