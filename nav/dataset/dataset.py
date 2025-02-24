from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional

from pdslogger import PdsLogger

from nav.config import Config, DEFAULT_CONFIG
from nav.support.nav_base import NavBase

class DataSet(ABC, NavBase):
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:

        super().__init__(config=config, logger_name=logger_name)

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
