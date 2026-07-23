"""Low-order background-brightness gradient score for an image.

A scattered-light (veiling) frame carries a smooth, large-scale brightness
ramp across the sensor that no navigable feature produces.  When such a ramp
is present, the intensity-based body techniques
(``BodyDiscCorrelateNav`` and ``BodyLimbNav``) both measure it rather than
the body, so their pointing errors correlate and must not be fused as
independent witnesses (the ensemble uses this score to detect that case).

The score fits an affine plane to a block-median downsample of the image
(medians are robust to stars and cosmic rays) and reports the plane's
peak-to-peak amplitude divided by the MAD-sigma of the fit residuals.  A flat
star field scores near zero; a frame with a real veiling gradient scores well
above five.  This is the same measure the cohort-curation prescan uses to
select scattered-light candidates, lifted into the library so the navigation
runtime and the curation tooling share one definition.
"""

import warnings

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'BLOCK_SIZE',
    'background_gradient_score',
]

#: Side length in pixels of the square blocks the image is median-downsampled
#: into before the affine-plane fit.  The image must span at least four blocks
#: on each axis for the score to be defined.
BLOCK_SIZE = 16


def background_gradient_score(
    image: NDArrayFloatType, sensor_mask: NDArrayBoolType | None = None
) -> float | None:
    """Return the low-order brightness-gradient score of an image.

    The score is the peak-to-peak amplitude of an affine plane fitted to the
    ``BLOCK_SIZE`` x ``BLOCK_SIZE`` block-median downsample of ``image``,
    divided by the MAD-sigma of the fit residuals.  It is dimensionless and
    scale-free: multiplying the image by a constant leaves it unchanged.

    Parameters:
        image: 2-D float image array in any physical units.  Non-finite
            pixels (for example calibrated-I/F ``NaN`` markers) are treated
            as missing and excluded from every block median.
        sensor_mask: Optional boolean mask, same shape as ``image``, selecting
            true sensor pixels.  When given, non-sensor pixels (extfov padding,
            which the orchestrator fills with zeros) are excluded from the
            block medians, so the padded border does not bias the plane fit.

    Returns:
        The gradient score, or ``None`` when the image spans fewer than four
        blocks on either axis, when no block has a finite median, or when the
        residual sigma is zero (a perfectly planar or constant downsample,
        where the ratio is undefined).
    """
    h, w = image.shape
    b = BLOCK_SIZE
    if h < 4 * b or w < 4 * b:
        return None
    if sensor_mask is not None:
        # Exclude non-sensor (extfov padding) pixels by making them NaN; the
        # NaN-aware block median below then ignores them.
        image = np.where(sensor_mask, image, np.nan)
    hh, ww = h // b * b, w // b * b
    blocks = image[:hh, :ww].reshape(hh // b, b, ww // b, b)
    # Median over each block; NaN-aware so a partially-missing block still
    # contributes its finite pixels and an all-missing block becomes NaN.  An
    # all-missing block legitimately yields NaN (the ``finite`` mask below
    # excludes it), so silence numpy's "All-NaN slice" warning rather than let
    # it surface as an error under the test suite's warnings filter.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(blocks, axis=(1, 3))
    ny, nx = med.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    finite = np.isfinite(med)
    if finite.sum() < 3:
        # Fewer than three finite block medians cannot constrain the three
        # affine-plane coefficients.
        return None
    design = np.column_stack([np.ones(med.size), xx.ravel(), yy.ravel()])[finite.ravel()]
    values = med.ravel()[finite.ravel()]
    if float(values.max()) == float(values.min()):
        # A perfectly constant downsample has no brightness gradient; the
        # affine fit would report only numerical rounding, whose amp/sigma
        # ratio is meaningless.  Real (noisy) frames never hit this branch.
        return None
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    plane = design @ coef
    resid = values - plane
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    amp = float(plane.max() - plane.min())
    if sigma <= 0:
        return None
    return amp / sigma
