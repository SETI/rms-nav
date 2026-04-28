"""Global image-quality helpers used during orchestrator preflight.

Three image-level checks happen before any technique runs.  This module owns
the implementations:

- ``saturation_mask``: pixels at or above the per-instrument full-well DN.
- ``cosmic_ray_mask``: single-pixel spikes above 5 sigma over a median-3x3
  neighborhood.

Each helper operates on the entire image and produces a boolean mask
consumed by feature extractors and the gradient computation.  Image-side
operations are global by design — no helper here crops to a predicted
position.
"""

import logging

import numpy as np
from scipy.ndimage import median_filter

from nav.support.types import NDArrayBoolType, NDArrayFloatType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'cosmic_ray_mask',
    'saturation_mask',
]


def saturation_mask(image: NDArrayFloatType, *, full_well_dn: float) -> NDArrayBoolType:
    """Boolean mask with True where pixels are at or above the saturation DN.

    Parameters:
        image: 2-D float input array.
        full_well_dn: Per-instrument saturation DN.  Pixels at or above this
            value are flagged as saturated.

    Returns:
        Boolean mask of the same shape as ``image``.

    Raises:
        TypeError: if ``image`` is not 2-D.
    """
    if image.ndim != 2:
        raise TypeError(f'saturation_mask requires a 2-D image; got ndim={image.ndim}')
    return image >= full_well_dn


def cosmic_ray_mask(
    image: NDArrayFloatType,
    *,
    image_noise_sigma: float,
    k_sigma: float = 5.0,
) -> NDArrayBoolType:
    """Boolean mask flagging single-pixel spikes above a sigma threshold.

    Compares each pixel against a 3x3 median around it (reflect boundary
    handling); pixels whose excess over the local median exceeds
    ``k_sigma * image_noise_sigma`` are flagged.  Suitable as a cheap
    pre-detection step that prevents single hot pixels from masquerading
    as star detections in pattern matching.

    Parameters:
        image: 2-D float input array.
        image_noise_sigma: Global MAD-based noise sigma (DN units); typically
            from ``estimate_image_noise_sigma``.
        k_sigma: Multiplier on ``image_noise_sigma`` (default 5).

    Returns:
        Boolean mask of the same shape as ``image``.

    Raises:
        TypeError: if ``image`` is not 2-D.
        ValueError: if ``image_noise_sigma`` is non-positive.
    """
    if image.ndim != 2:
        raise TypeError(f'cosmic_ray_mask requires a 2-D image; got ndim={image.ndim}')
    if image_noise_sigma <= 0.0:
        raise ValueError(f'image_noise_sigma must be positive; got {image_noise_sigma}')
    a = image.astype(np.float64)
    med = median_filter(a, size=3, mode='reflect')
    threshold = k_sigma * image_noise_sigma
    mask: NDArrayBoolType = (a - med) > threshold
    return mask
