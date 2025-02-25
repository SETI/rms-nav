from abc import ABC, abstractmethod
from typing import Optional

from nav.config import Config
from nav.support.nav_base import NavBase


class NavTechnique(ABC, NavBase):
    def __init__(self,
                 nav_master: 'NavMaster',
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        super().__init__(config=config, logger_name=logger_name)

        self._nav_master = nav_master
        self._offset: tuple[float, float] | None = None

    @property
    def nav_master(self) -> 'NavMaster':
        return self._nav_master

    @property
    def offset(self) -> tuple[float, float] | None:
        return self._offset

    @abstractmethod
    def navigate(self) -> None:
        ...
