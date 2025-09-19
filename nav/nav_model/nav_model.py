from abc import ABC, abstractmethod
from typing import Any, Optional

from nav.annotation import Annotations
from nav.config import Config
from nav.inst import Inst
from nav.obs import ObsSnapshot
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint8Type


class NavModel(ABC, NavBase):
    """Base class for navigation models used to generate synthetic images."""

    def __init__(self,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None,
                 logger_name: Optional[str] = None) -> None:
        """Initializes a navigation model with observation data.

        Parameters:
            obs: Observation snapshot containing image and metadata.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
            logger_name: Name for the logger. If None, uses class name.
        """

        super().__init__(config=config, logger_name=logger_name)

        # The snapshot describing the observation the model is based on
        self._obs = obs

        # The actual model
        self._model_img: NDArrayFloatType | None = None

        # A boolean array indicating which pixels are part of the model (True)
        self._model_mask: NDArrayBoolType | None = None

        # Metadata describing the model
        self._metadata: dict[str, Any] = {}

        # The range from the observer to each point in the model in km; inf if infinitely
        # far away
        self._range: NDArrayFloatType | None = None

        # The uncertainty in the model in pixels
        self._uncertainty: float = 0.

        # An optional list of regions indicated by packed masks. The sum of all of these
        # regions should be equal to the model mask. Each region will be contrast-stretched
        # separately during overlay creation. For example, this allows each star to be
        # contrast-stretched separately.
        self._stretch_regions: list[NDArrayUint8Type] | None = None

        # An optional list of text annotations for the model
        self._annotations: Annotations | None = None

    @property
    def obs(self) -> ObsSnapshot:
        """Returns the observation snapshot associated with this model."""
        return self._obs

    @property
    def inst(self) -> Inst:
        """Returns the instrument associated with the observation."""
        return self.obs.inst

    @property
    def model_img(self) -> NDArrayFloatType | None:
        """Returns the model image array if available, or None if not generated yet."""
        return self._model_img

    @property
    def model_mask(self) -> NDArrayBoolType | None:
        """Returns the model mask array if available, or None if not generated yet."""
        return self._model_mask

    @property
    def metadata(self) -> dict[str, Any]:
        """Returns the metadata dictionary associated with this model."""
        return self._metadata

    @property
    def range(self) -> NDArrayFloatType | float | None:
        """Returns the range to the target body, either as a single value or per-pixel array."""
        return self._range

    @property
    def uncertainty(self) -> float:
        """Returns the uncertainty in the model."""
        return self._uncertainty

    @property
    def stretch_regions(self) -> list[NDArrayUint8Type] | None:
        """Returns the stretch regions for the model if available."""
        return self._stretch_regions

    @property
    def annotations(self) -> Annotations | None:
        """Returns the annotations object associated with this model if available."""
        return self._annotations

    @abstractmethod
    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_annotations: bool = True) -> None:
        """Creates the internal model representation for a navigation model.

        Parameters:
            always_create_model: If True, creates a model even if won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model or
                annotations.
            create_annotations: If True, creates text annotations for the model.
        """
        ...
