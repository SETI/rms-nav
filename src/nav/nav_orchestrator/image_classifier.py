"""NavImageClassifier — quick-fail classifier for incoming images.

Operates on the entire sensor area and assigns an image to one of a small
set of classes.  Most "bad" classes never invoke an extractor —
corrupted images fail in milliseconds with a clear reason.

The classifier is global: no predicted feature positions are used.  Three
cheap statistics drive the decision:

    saturation_frac = fraction of pixels at saturation_threshold_dn or above
    missing_frac    = fraction of pixels equal to missing_data_marker_dn
    noise_sigma     = MAD-based image noise sigma

Per-instrument thresholds live in ``config_4N0_inst_*.yaml``; this module
takes them as constructor parameters so it stays pure-Python and unit-
testable without loading config.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from nav.nav_orchestrator.image_classifier_result import (
    ImageFlag,
    NavImageClassifierResult,
)
from nav.support.noise_estimate import estimate_image_noise_sigma
from nav.support.types import NDArrayBoolType, NDArrayFloatType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'ImageQualityThresholds',
    'NavImageClassifier',
]


@dataclass(frozen=True)
class ImageQualityThresholds:
    """Per-instrument configuration for the image-quality classifier.

    Parameters:
        saturation_threshold_dn: Pixels at or above this DN are flagged
            saturated.
        missing_data_marker_dn: Pixels exactly equal to this value are
            treated as missing data (instrument-specific dropout marker).
        max_saturation_frac_clean: Above this fraction of saturated pixels
            the image is ``fully_overexposed``.
        max_missing_frac_clean: Above this fraction of missing pixels the
            image is ``mostly_missing_data``.
        partial_dropout_min_frac: Below this missing fraction, no
            ``partial_dropout`` flag is raised.  At or above this fraction
            (and below ``max_missing_frac_clean``) the ``partial_dropout``
            advisory flag is set on the result.
        blank_max_dn: If the image's max DN is below this, the image is
            ``blank``.
        noisy_threshold: Above this MAD-noise sigma, the ``noisy`` flag is
            raised (image stays ``clean``).
    """

    saturation_threshold_dn: float = 4095.0
    missing_data_marker_dn: float = 0.0
    max_saturation_frac_clean: float = 0.80
    max_missing_frac_clean: float = 0.30
    partial_dropout_min_frac: float = 0.05
    blank_max_dn: float = 5.0
    noisy_threshold: float = 10.0


@dataclass
class NavImageClassifier:
    """Quick-fail image classifier consumed by the orchestrator.

    Parameters:
        thresholds: Per-instrument thresholds (see ``ImageQualityThresholds``).
    """

    thresholds: ImageQualityThresholds = field(default_factory=ImageQualityThresholds)

    def classify(
        self,
        image: NDArrayFloatType,
        sensor_mask: NDArrayBoolType | None = None,
    ) -> NavImageClassifierResult:
        """Run the classifier and return its verdict.

        Parameters:
            image: 2-D float image array (sensor + extfov padding).
            sensor_mask: Optional boolean mask selecting sensor pixels;
                if ``None``, every pixel is treated as sensor data.

        Returns:
            NavImageClassifierResult.

        Raises:
            TypeError: if ``image`` is not 2-D.
        """
        if image.ndim != 2:
            raise TypeError(f'NavImageClassifier requires a 2-D image; got ndim={image.ndim}')
        if sensor_mask is None:
            sensor = image
        else:
            if not isinstance(sensor_mask, np.ndarray):
                raise TypeError(
                    f'sensor_mask must be a numpy ndarray; got {type(sensor_mask).__name__}'
                )
            if sensor_mask.dtype != np.bool_:
                raise TypeError(f'sensor_mask must have boolean dtype; got {sensor_mask.dtype}')
            if sensor_mask.shape != image.shape:
                raise ValueError(
                    f'sensor_mask shape {sensor_mask.shape} differs from image shape {image.shape}'
                )
            if sensor_mask.size == 0:
                raise ValueError('sensor_mask must not be empty')
            if not sensor_mask.any():
                raise ValueError('sensor_mask must select at least one sensor pixel')
            sensor = image[sensor_mask]
        # Compute statistics on the sensor pixels only.
        sat_mask = sensor >= self.thresholds.saturation_threshold_dn
        miss_mask = sensor == self.thresholds.missing_data_marker_dn
        n_total = max(sensor.size, 1)
        saturation_frac = float(sat_mask.sum()) / float(n_total)
        missing_frac = float(miss_mask.sum()) / float(n_total)
        noise_sigma = estimate_image_noise_sigma(image, sensor_mask)
        max_dn = float(np.max(sensor)) if sensor.size > 0 else 0.0
        flags: list[ImageFlag] = []
        # Outcome decision: blank check runs first so a near-zero image with
        # missing-data marker == 0 isn't mis-classified as "mostly_missing".
        if max_dn < self.thresholds.blank_max_dn:
            return NavImageClassifierResult(
                image_class='blank',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                flags=flags,
            )
        if saturation_frac > self.thresholds.max_saturation_frac_clean:
            return NavImageClassifierResult(
                image_class='fully_overexposed',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                flags=flags,
            )
        if missing_frac > self.thresholds.max_missing_frac_clean:
            return NavImageClassifierResult(
                image_class='mostly_missing_data',
                saturation_frac=saturation_frac,
                missing_frac=missing_frac,
                noise_sigma=noise_sigma,
                max_dn=max_dn,
                flags=flags,
            )
        # Otherwise the image is clean (with optional advisory flags).
        if missing_frac > self.thresholds.partial_dropout_min_frac:
            flags.append('partial_dropout')
        if noise_sigma > self.thresholds.noisy_threshold:
            flags.append('noisy')
        return NavImageClassifierResult(
            image_class='clean',
            saturation_frac=saturation_frac,
            missing_frac=missing_frac,
            noise_sigma=noise_sigma,
            max_dn=max_dn,
            flags=flags,
        )
