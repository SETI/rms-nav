from abc import ABC, abstractmethod
from typing import Optional

from nav.config import Config
from nav.support.nav_base import NavBase
from nav import nav_master as nav_master_module


class NavTechnique(ABC, NavBase):
    """Base class for navigation techniques.

        Parameters:
            nav_master: The navigation master instance
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            logger_name: Name for the logger. If None, uses class name.
    """

    def __init__(self,
                 nav_master: 'nav_master_module.NavMaster',
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        super().__init__(config=config, logger_name=logger_name)

        self._nav_master = nav_master
        self._offset: tuple[float, float] | None = None
        self._confidence: float | None = None

    @property
    def nav_master(self) -> 'nav_master_module.NavMaster':
        """Returns the navigation master instance."""
        return self._nav_master

    @property
    def offset(self) -> tuple[float, float] | None:
        """Returns the computed offset as a tuple of (x, y) or None if not calculated."""
        return self._offset

    @property
    def confidence(self) -> float | None:
        """Returns the confidence in the computed offset as a float or None if not calculated."""
        return self._confidence

    @abstractmethod
    def navigate(self) -> None:
        """Performs the navigation process.

        This abstract method must be implemented by all navigation technique subclasses.
        """
        ...
