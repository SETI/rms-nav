from abc import ABC, abstractmethod
from typing import Any, Optional

from nav.config import Config
from nav.obs import ObsSnapshot
from nav.support.nav_base import NavBase

from .nav_model_result import NavModelResult


class NavModel(ABC, NavBase):
    """Base class for navigation models used to generate synthetic images."""

    def __init__(
        self, name: str, obs: ObsSnapshot, *, config: Optional[Config] = None
    ) -> None:
        """Initializes a navigation model with observation data.

        Parameters:
            name: The name of the model.
            obs: Observation snapshot containing image and metadata.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(config=config)

        # The name of the model
        self._name = name

        # The snapshot describing the observation the model is based on
        self._obs = obs

        # Metadata describing the model and how it was computed
        self._metadata: dict[str, Any] = {}

        # List of results produced by this model (one or more NavModelResult instances)
        self._models: list[NavModelResult] = []

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self._name

    @property
    def obs(self) -> ObsSnapshot:
        """Returns the observation snapshot associated with this model."""
        return self._obs

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary associated with this model."""
        return self._metadata

    @property
    def models(self) -> list[NavModelResult]:
        """Returns the list of results produced by this model."""
        return self._models

    @abstractmethod
    def create_model(
        self,
        *,
        always_create_model: bool = False,
        never_create_model: bool = False,
        create_annotations: bool = True,
    ) -> None:
        """Creates the internal model representation for a navigation model.

        Parameters:
            always_create_model: If True, creates a model even if won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model or
                annotations.
            create_annotations: If True, creates text annotations for the model.
        """
        ...
