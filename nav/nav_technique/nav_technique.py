from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from nav.config import Config
from nav.support.nav_base import NavBase

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechnique(ABC, NavBase):
    """Base class for navigation techniques.

        Parameters:
            nav_master: The navigation master instance
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
    """

    def __init__(self,
                 nav_master: 'NavMaster',
                 *,
                 config: Optional[Config] = None) -> None:
        super().__init__(config=config)

        self._nav_master = nav_master
        self._offset: tuple[float, float] | None = None
        self._confidence: float | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def nav_master(self) -> 'NavMaster':
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

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary."""
        return self._metadata

    @abstractmethod
    def navigate(self) -> None:
        """Performs the navigation process.

        This abstract method must be implemented by all navigation technique subclasses.
        """
        ...
