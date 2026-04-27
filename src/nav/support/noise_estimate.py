"""Robust per-image noise estimate over the sensor area.

The autonomous-navigation orchestrator computes ``image_noise_sigma`` once per
image and stores it on ``NavContext`` so every extractor and technique uses
the same value.  This module owns the implementation: a single MAD-based
robust standard-deviation over the sensor pixels.

The estimate is global — it does not require knowledge of where any feature
lives in the image, and is therefore not biased by a wrong SPICE pointing
prediction.
"""

import logging

import numpy as np

from nav.support.misc import mad_std
from nav.support.types import NDArrayBoolType, NDArrayFloatType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'estimate_image_noise_sigma',
]


def estimate_image_noise_sigma(
    image: NDArrayFloatType, sensor_mask: NDArrayBoolType | None = None
) -> float:
    """Return a robust noise sigma over the sensor pixels.

    Uses the MAD-based estimator ``1.4826 * median(abs(x - median(x)))``
    which is insensitive to bright body / star content because the median
    is dominated by background pixels (the majority of the image).  The
    estimate is therefore stable across scene type.

    Parameters:
        image: 2-D float input array.
        sensor_mask: Optional boolean mask with ``True`` for sensor pixels and
            ``False`` for extfov padding.  If ``None``, every pixel of
            ``image`` is treated as sensor data.

    Returns:
        Robust noise sigma in the same DN units as ``image``.

    Raises:
        TypeError: if ``image`` is not 2-D.
        ValueError: if no sensor pixels are available.
    """
    if image.ndim != 2:
        raise TypeError(f'estimate_image_noise_sigma requires a 2-D image; got ndim={image.ndim}')
    if sensor_mask is None:
        sensor = image
    else:
        if sensor_mask.shape != image.shape:
            raise ValueError(
                f'sensor_mask shape {sensor_mask.shape} differs from image shape {image.shape}'
            )
        sensor_pixels = image[sensor_mask]
        if sensor_pixels.size == 0:
            raise ValueError('sensor_mask selects no pixels')
        sensor = sensor_pixels
    return mad_std(np.asarray(sensor, np.float64).ravel())
