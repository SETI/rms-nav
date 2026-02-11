"""Dataclass encapsulating the result of a navigation model computation."""

from dataclasses import dataclass

from nav.annotation import Annotations
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayUint8Type


@dataclass
class NavModelResult:
    """Encapsulates a single result from a navigation model.

    A model may produce one or more results (e.g., one per body or one combined).
    Each result holds the model image, mask, range, and related per-result data.
    """

    # The actual model in non-normalized floating point pixel values
    model_img: NDArrayFloatType | None = None

    # A boolean array indicating which pixels are part of the model (True)
    model_mask: NDArrayBoolType | None = None

    # A weighted mask where each pixel is the confidence of the model at that pixel
    weighted_mask: NDArrayFloatType | None = None

    # The range from the observer to each point in the model in km; inf if infinitely far
    range: NDArrayFloatType | float | None = None

    # Optional amount to blur the model (2x2 covariance or single value)
    blur_amount: NDArrayFloatType | float | None = None

    # The uncertainty in the model in pixels (2x2 covariance or single value)
    uncertainty: NDArrayFloatType | float | None = None

    # The confidence in the model
    confidence: float | None = None

    # Optional list of packed masks for per-region contrast stretching (e.g., per star)
    stretch_regions: list[NDArrayUint8Type] | None = None

    # Optional text annotations for the model
    annotations: Annotations | None = None
