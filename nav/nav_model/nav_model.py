from abc import ABC, abstractmethod
from typing import Any, Optional

from nav.annotation import Annotations
from nav.config import Config
from nav.obs import ObsSnapshot
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint8Type


class NavModel(ABC, NavBase):
    """Base class for navigation models used to generate synthetic images."""

    def __init__(self,
                 name: str,
                 obs: ObsSnapshot,
                 *,
                 config: Optional[Config] = None) -> None:
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

        # A model consists of three arrays, each the same size as the extended
        # observation: model image, model mask, and range

        # The actual model in non-normalized floating point pixel values
        self._model_img: NDArrayFloatType | None = None

        # A boolean array indicating which pixels are part of the model (True)
        self._model_mask: NDArrayBoolType | None = None

        # A weighted mask where each pixel is the confidence of the model at that pixel
        self._weighted_mask: NDArrayFloatType | None = None

        # The range from the observer to each point in the model in km; inf if infinitely
        # far away like stars
        self._range: NDArrayFloatType | float | None = None

        # Optional amount to blur the model.
        # This can be a 2x2 covariance matrix or a single value if the blur amount
        # is the same in all directions.
        self._blur_amount: NDArrayFloatType | float | None = None

        # The uncertainty in the model in pixels.
        # This can be a 2x2 covariance matrix or a single value if the uncertainty
        # is the same in all directions.
        self._uncertainty: NDArrayFloatType | float | None = None

        # The confidence in the model
        self._confidence: float | None = None

        # An optional list of regions indicated by packed masks. The sum of all of these
        # regions should be equal to the model mask. Each region will be contrast-stretched
        # separately during overlay creation. For example, this allows each star to be
        # contrast-stretched separately.
        # Note that packbits pads the array with zeros to the nearest multiple of 8
        # in each dimension, so when unpacking the array we have to clip the array
        self._stretch_regions: list[NDArrayUint8Type] | None = None

        # An optional list of text annotations for the model
        self._annotations: Annotations | None = None

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
    def model_img(self) -> NDArrayFloatType | None:
        """Returns the model image array if available, or None if not generated yet."""
        return self._model_img

    @property
    def model_mask(self) -> NDArrayBoolType | None:
        """Returns the model mask array if available, or None if not generated yet."""
        return self._model_mask

    @property
    def range(self) -> NDArrayFloatType | float | None:
        """Returns the range to the target body, either as a single value or per-pixel array."""
        return self._range

    @property
    def uncertainty(self) -> NDArrayFloatType | float | None:
        """Returns the uncertainty in the model."""
        return self._uncertainty

    @property
    def blur_amount(self) -> NDArrayFloatType | float | None:
        """Returns the blur amount for the model."""
        return self._blur_amount

    @property
    def confidence(self) -> float | None:
        """Returns the confidence in the model."""
        return self._confidence

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
