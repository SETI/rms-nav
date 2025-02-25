from abc import ABC, abstractmethod
from typing import Optional

from nav.config import Config
from nav.support.nav_base import NavBase


class NavTechnique(ABC, NavBase):
    def __init__(self,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        super().__init__(config=config, logger_name=logger_name)

        self._offset: tuple[float, float] | None = None

    @property
    def offset(self) -> tuple[float, float] | None:
        return self._offset

    @abstractmethod
    def navigate(self) -> None:
        ...
