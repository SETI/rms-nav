"""Robust per-image noise estimate over the sensor area.

The autonomous-navigation orchestrator computes ``image_noise_sigma`` once per
image and stores it on ``NavContext`` so every extractor and technique uses
the same value.  This module owns the implementation.

The estimate is global -- it does not require knowledge of where any feature
lives in the image, and is therefore not biased by a wrong SPICE pointing
prediction.  It is computed from a 3x3 second-difference (Laplacian) response
so that smooth scene structure (ring brightness ramps, limb shading, a bright
extended disc) cancels and only pixel-to-pixel noise survives; the MAD over
that response further rejects the minority of pixels sitting on sharp edges or
cosmic rays.  A plain MAD of the raw intensities is *not* used: when the scene
is dominated by structure (for example rings filling the frame) it measures the
bright-to-dark spread rather than the noise and overestimates sigma by orders
of magnitude, which in turn pushes the edge-detection threshold above every
real gradient and empties the distance transform.
"""

import numpy as np

from spindoctor.support.misc import mad_std
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'estimate_image_noise_sigma',
]

# The 3x3 second-difference (Laplacian) kernel [[1, -2, 1], [-2, 4, -2],
# [1, -2, 1]] has coefficients whose sum of squares is 36, so for white noise
# of standard deviation ``s`` the response variance is ``36 * s**2``; dividing
# the robust response standard deviation by 6 recovers ``s``.  The double
# difference cancels any locally linear brightness ramp, so smooth scene
# content does not inflate the estimate.
_LAPLACIAN_NORM = 6.0


def estimate_image_noise_sigma(
    image: NDArrayFloatType, sensor_mask: NDArrayBoolType | None = None
) -> float:
    """Return a robust noise sigma over the sensor pixels.

    The estimate is the MAD-based standard deviation of a 3x3 second-difference
    (Laplacian) response, divided by 6.  The second difference cancels smooth
    brightness structure so the value reflects pixel-to-pixel noise rather than
    scene content, and the MAD rejects the minority of pixels on sharp edges or
    cosmic rays.

    Parameters:
        image: 2-D float input array.
        sensor_mask: Optional boolean mask with ``True`` for sensor pixels and
            ``False`` for extfov padding.  If ``None``, every pixel of
            ``image`` is treated as sensor data.  Only response pixels whose
            full 3x3 neighbourhood lies inside the mask contribute.

    Returns:
        Robust noise sigma in the same DN units as ``image``.

    Raises:
        TypeError: if ``image`` is not 2-D.
        ValueError: if no sensor pixels are available.
    """
    if image.ndim != 2:
        raise TypeError(f'estimate_image_noise_sigma requires a 2-D image; got ndim={image.ndim}')
    if sensor_mask is not None:
        if sensor_mask.shape != image.shape:
            raise ValueError(
                f'sensor_mask shape {sensor_mask.shape} differs from image shape {image.shape}'
            )
        if not bool(sensor_mask.any()):
            raise ValueError('sensor_mask selects no pixels')

    img = np.asarray(image, np.float64)

    # Images smaller than the 3x3 kernel cannot form a second difference; fall
    # back to a masked global MAD so a value is still returned.
    if img.shape[0] < 3 or img.shape[1] < 3:
        sensor = img if sensor_mask is None else img[sensor_mask]
        return float(mad_std(sensor.ravel()))

    # Laplacian response over every interior pixel (corners +1, edges -2,
    # centre +4), aligned to image pixels [1:-1, 1:-1].
    response = (
        img[:-2, :-2]
        + img[:-2, 2:]
        + img[2:, :-2]
        + img[2:, 2:]
        - 2.0 * (img[:-2, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:] + img[2:, 1:-1])
        + 4.0 * img[1:-1, 1:-1]
    )

    if sensor_mask is None:
        samples = response.ravel()
        fallback = img.ravel()
    else:
        m = sensor_mask
        interior_valid = (
            m[:-2, :-2]
            & m[:-2, 1:-1]
            & m[:-2, 2:]
            & m[1:-1, :-2]
            & m[1:-1, 1:-1]
            & m[1:-1, 2:]
            & m[2:, :-2]
            & m[2:, 1:-1]
            & m[2:, 2:]
        )
        samples = response[interior_valid]
        fallback = img[sensor_mask].ravel()

    # A second difference touching a NaN missing-data marker is NaN, so drop
    # non-finite responses.  If none survive (no fully-finite 3x3 neighbourhood,
    # e.g. an image that is almost entirely markers), fall back to a global MAD
    # over the finite sensor pixels rather than failing.
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float(mad_std(fallback))

    return float(mad_std(finite) / _LAPLACIAN_NORM)
