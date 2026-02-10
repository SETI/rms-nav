from abc import ABC, abstractmethod
import fnmatch
from typing import Any, Optional, TYPE_CHECKING

from nav.nav_model import NavModel, NavModelCombined

from nav.config import Config
from nav.support.nav_base import NavBase

if TYPE_CHECKING:
    from nav.nav_master import NavMaster


class NavTechnique(ABC, NavBase):
    """Base class for navigation techniques."""

    def __init__(
        self, nav_master: 'NavMaster', *, config: Optional[Config] = None
    ) -> None:
        """Initializes a navigation technique.

        Parameters:
            nav_master: The navigation master instance.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(config=config)

        self._nav_master = nav_master
        self._offset: tuple[float, float] | None = None
        self._uncertainty: tuple[float, float] | None = None
        self._confidence: float | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def nav_master(self) -> 'NavMaster':
        """Returns the navigation master instance."""
        return self._nav_master

    @property
    def offset(self) -> tuple[float, float] | None:
        """Returns the computed offset as a tuple of (v, u) or None if not calculated.

        The floating point offset is of form (dv, du). If an object is predicted by the SPICE
        kernels to be at location (v, u) in the image, then the actual location of the
        object is (v+dv, u+du). This means a positive offset is equivalent to shifting a model
        up and to the right (positive v and u).
        """
        return self._offset

    @property
    def uncertainty(self) -> tuple[float, float] | None:
        """Returns the computed uncertainty as a tuple of (v, u) or None if not calculated."""
        return self._uncertainty

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

    def _filter_models(self, model_names: list[str]) -> list['NavModel']:
        """Filters the available models using glob patterns."""
        models = [
            x
            for x in self.nav_master.all_models
            if any(
                fnmatch.fnmatch(x.name.upper(), pattern.upper())
                for pattern in model_names
            )
        ]
        return models

    def _combine_models(self, model_names: list[str]) -> 'NavModelCombined | None':
        """Returns a combined model from the available models matching patterns."""
        models = self._filter_models(model_names)
        if len(models) == 0:
            return None
        return NavModelCombined('combined', self.nav_master.obs, models)
