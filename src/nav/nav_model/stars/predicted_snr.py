"""Per-star predicted-SNR computation for the star NavModel.

Reliability of a STAR feature comes from how detectable the star is at
its predicted pixel position.  The estimate uses three inputs:

- ``obs.star_psf()`` — the per-camera-per-filter PSF.
- ``NavContext.image_noise_sigma`` — robust MAD-based noise estimate over
  the sensor area.
- ``star.dn`` — the integrated DN expected for the star in the
  instrument's bandpass, computed from its catalog V magnitude via the
  ``2.512 ** -(vmag - 4)`` flux-to-DN scaling.  ``star.dn`` is the
  total signal across the PSF support; the per-pixel signal is that
  total spread over a circular Gaussian-PSF aperture.

The integrated SNR follows the form in Part 1's "Position covariance
per feature type" section:

::

    SNR = total_signal / sqrt(total_signal + read_noise**2 * N_aperture)

with ``total_signal`` in DN and ``read_noise**2 * N_aperture`` standing
in for the variance contribution from background and read noise.
``image_noise_sigma`` is treated as a Gaussian read-noise proxy because
the MAD estimator is dominated by background pixels and combines shot,
read, and dark contributions into a single per-pixel sigma.

``SCLASS_TO_B_MINUS_V`` is re-exported here so callers that need the
spectral-class colour mapping can pull it from the same module that
owns the predicted-SNR formula.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from starcat import SCLASS_TO_B_MINUS_V

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

    from nav.support.types import MutableStar

__all__ = [
    'SCLASS_TO_B_MINUS_V',
    'integrated_signal_dn',
    'predicted_snr',
    'psf_aperture_pixels',
    'psf_sigma_px',
]


def psf_sigma_px(psf: PSF) -> float:
    """Return the Gaussian-equivalent sigma of ``psf`` in pixels.

    Treats every PSF as a 2-D Gaussian for SNR / CRLB purposes.  The
    pipeline ships ``GaussianPSF`` instances populated from
    ``star_psf_sigma`` in ``config_NN_inst_*.yaml``, so the ``sigma``
    attribute is the natural source.  When the field is absent (a
    third-party PSF subclass), we fall back to ``fwhm() / 2.3548``.

    Parameters:
        psf: PSF instance from ``obs.star_psf()``.

    Returns:
        Gaussian sigma in pixels.
    """
    if hasattr(psf, 'sigma'):
        return float(cast(float, psf.sigma))
    fwhm_method = getattr(psf, 'fwhm', None)
    if callable(fwhm_method):
        return float(fwhm_method()) / 2.3548200450309493
    raise AttributeError(f'PSF {type(psf).__name__} exposes neither sigma nor fwhm()')


def psf_aperture_pixels(sigma_px_value: float) -> float:
    """Return the effective number of pixels in the PSF support.

    Uses the standard "noise-equivalent area" of a 2-D Gaussian,
    ``4 * pi * sigma**2``, which is the right scale for converting an
    integrated DN signal into a per-pixel matched-filter SNR.

    Parameters:
        sigma_px_value: Per-pixel PSF sigma in pixels.

    Returns:
        Effective aperture area in pixels (always > 0 for sigma > 0).
    """
    if sigma_px_value <= 0.0:
        raise ValueError(f'sigma_px_value must be > 0; got {sigma_px_value!r}')
    return 4.0 * math.pi * sigma_px_value * sigma_px_value


def integrated_signal_dn(star: MutableStar, mag_offset: float) -> float:
    """Return the predicted in-band DN for a catalog star.

    Applies a per-camera-per-filter ``mag_offset`` to convert the
    catalog V-band magnitude into the instrument's bandpass, then uses
    the standard ``2.512 ** -(vmag - 4)`` flux-to-DN scaling.  Stars
    without a catalog magnitude (vmag is None) are not detectable and
    return 0.0.

    Parameters:
        star: Star record carrying ``vmag``.
        mag_offset: Per-instrument-per-filter magnitude offset
            (mag_in_band - mag_v).  Positive values mean the instrument
            sees the star fainter than the catalog magnitude.

    Returns:
        Predicted integrated DN in the instrument's bandpass.
    """
    if star.vmag is None:
        return 0.0
    in_band_mag = float(star.vmag) + mag_offset
    return float(2.512 ** -(in_band_mag - 4.0))


def predicted_snr(
    star: MutableStar,
    *,
    psf: PSF,
    image_noise_sigma: float,
    mag_offset: float = 0.0,
) -> float:
    """Predicted integrated SNR for a star at its predicted pixel.

    Treats ``image_noise_sigma`` as the per-pixel Gaussian noise from
    background + read noise + dark current; ``star.dn`` is the
    integrated signal across the PSF support.  Implements the formula
    from the design's STAR section:

    ::

        SNR = total_signal / sqrt(total_signal + sigma**2 * N_aperture)

    with ``N_aperture = 4 * pi * sigma_PSF**2``.

    Parameters:
        star: Star record.
        psf: PSF (typically from ``obs.star_psf()``).
        image_noise_sigma: Robust per-pixel noise sigma in DN.
        mag_offset: Catalog-to-instrument magnitude offset (default 0).

    Returns:
        Predicted SNR (dimensionless, >= 0).
    """
    if image_noise_sigma <= 0.0:
        raise ValueError(f'image_noise_sigma must be > 0; got {image_noise_sigma!r}')
    sig = integrated_signal_dn(star, mag_offset)
    if sig <= 0.0:
        return 0.0
    sigma_px_value = psf_sigma_px(psf)
    aperture = psf_aperture_pixels(sigma_px_value)
    variance = sig + image_noise_sigma * image_noise_sigma * aperture
    return sig / math.sqrt(variance)
