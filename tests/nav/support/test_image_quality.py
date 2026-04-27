"""Tests for ``nav.support.image_quality`` helpers."""

import numpy as np
import pytest

from nav.support.image_quality import cosmic_ray_mask, saturation_mask


def test_saturation_mask_threshold() -> None:
    """Pixels at or above full_well_dn are flagged."""
    image = np.array([[0.0, 50.0], [4095.0, 4096.0]], np.float64)
    mask = saturation_mask(image, full_well_dn=4095.0)
    assert mask[1, 0]
    assert mask[1, 1]
    assert not mask[0, 0]
    assert not mask[0, 1]


def test_saturation_mask_rejects_non_2d() -> None:
    """3-D input raises TypeError."""
    image = np.zeros((4, 4, 4), np.float64)
    with pytest.raises(TypeError, match='2-D'):
        saturation_mask(image, full_well_dn=4095.0)


def test_cosmic_ray_mask_finds_planted_spike() -> None:
    """A single bright pixel above 5 sigma is flagged."""
    rng = np.random.default_rng(seed=12345)
    image = rng.standard_normal(size=(64, 64))
    image[20, 30] = 50.0  # 50-sigma spike
    mask = cosmic_ray_mask(image, image_noise_sigma=1.0)
    assert mask[20, 30]
    # Background pixels should not be flagged.
    assert not mask[0, 0]


def test_cosmic_ray_mask_below_threshold_not_flagged() -> None:
    """A spike below 5 sigma is not flagged."""
    image = np.zeros((16, 16), np.float64)
    image[5, 5] = 4.0  # below 5-sigma at sigma=1
    mask = cosmic_ray_mask(image, image_noise_sigma=1.0)
    assert not mask[5, 5]


def test_cosmic_ray_mask_rejects_non_positive_sigma() -> None:
    """Non-positive image_noise_sigma raises ValueError."""
    image = np.zeros((4, 4), np.float64)
    with pytest.raises(ValueError, match='positive'):
        cosmic_ray_mask(image, image_noise_sigma=0.0)


def test_cosmic_ray_mask_rejects_non_2d() -> None:
    """3-D input raises TypeError."""
    image = np.zeros((4, 4, 4), np.float64)
    with pytest.raises(TypeError, match='2-D'):
        cosmic_ray_mask(image, image_noise_sigma=1.0)
