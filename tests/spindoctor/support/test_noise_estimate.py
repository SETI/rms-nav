"""Tests for ``spindoctor.support.noise_estimate.estimate_image_noise_sigma``."""

import numpy as np
import pytest

from spindoctor.support.noise_estimate import estimate_image_noise_sigma


def test_estimate_image_noise_sigma_unit_normal() -> None:
    """MAD-based sigma on N(0, 1) data is close to 1."""
    rng = np.random.default_rng(seed=42)
    image = rng.standard_normal(size=(256, 256))
    sigma = estimate_image_noise_sigma(image)
    assert abs(sigma - 1.0) < 0.05


def test_estimate_image_noise_sigma_robust_to_bright_features() -> None:
    """A few bright outliers do not change the estimate appreciably."""
    rng = np.random.default_rng(seed=43)
    image = rng.standard_normal(size=(256, 256))
    image[100:110, 100:110] = 10000.0  # outlier block
    sigma = estimate_image_noise_sigma(image)
    assert abs(sigma - 1.0) < 0.1


def test_estimate_image_noise_sigma_uses_sensor_mask() -> None:
    """Pixels outside the sensor mask are excluded from the estimate."""
    rng = np.random.default_rng(seed=44)
    image = np.zeros((128, 128), np.float64)
    image[:, :64] = rng.standard_normal(size=(128, 64))
    image[:, 64:] = 1e6  # bogus values outside the sensor
    mask = np.zeros((128, 128), bool)
    mask[:, :64] = True
    sigma = estimate_image_noise_sigma(image, sensor_mask=mask)
    assert abs(sigma - 1.0) < 0.1


def test_estimate_image_noise_sigma_rejects_empty_mask() -> None:
    """An all-False sensor mask raises ValueError."""
    image = np.zeros((4, 4), np.float64)
    mask = np.zeros((4, 4), bool)
    with pytest.raises(ValueError, match='no pixels'):
        estimate_image_noise_sigma(image, sensor_mask=mask)


def test_estimate_image_noise_sigma_shape_mismatch() -> None:
    """Mismatched sensor mask shape raises ValueError."""
    image = np.zeros((4, 4), np.float64)
    mask = np.ones((4, 5), bool)
    with pytest.raises(ValueError, match='shape'):
        estimate_image_noise_sigma(image, sensor_mask=mask)


def test_estimate_image_noise_sigma_rejects_non_2d() -> None:
    """Non-2D input is rejected at the entry point."""
    image = np.zeros((4, 4, 4), np.float64)
    with pytest.raises(TypeError, match='2-D'):
        estimate_image_noise_sigma(image)
