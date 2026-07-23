"""Tests for the low-order background-gradient score."""

import numpy as np
import pytest

from spindoctor.support.background_gradient import (
    BLOCK_SIZE,
    SATURATED_GRADIENT_SCORE,
    background_gradient_score,
)


def test_flat_field_scores_near_zero() -> None:
    """A constant image has no plane amplitude, so it returns None or ~0."""
    image = np.full((128, 128), 500.0, dtype=np.float64)
    # A perfectly flat field has zero residual sigma -> undefined ratio.
    assert background_gradient_score(image) is None


def test_noise_only_field_scores_low() -> None:
    """A flat field plus noise (no ramp) scores well below the scatter cutoff."""
    rng = np.random.default_rng(1)
    image = 500.0 + rng.normal(0.0, 5.0, size=(128, 128))
    score = background_gradient_score(image)
    assert score is not None
    assert score < 5.0


def test_strong_ramp_scores_high() -> None:
    """A clean linear ramp scores far above the scatter cutoff."""
    rng = np.random.default_rng(2)
    _yy, xx = np.mgrid[0:256, 0:256]
    image = 100.0 + 3.0 * xx + rng.normal(0.0, 2.0, size=(256, 256))
    score = background_gradient_score(image)
    assert score is not None
    assert score > 5.0


def test_score_is_scale_invariant() -> None:
    """Multiplying the image by a constant leaves the score unchanged."""
    rng = np.random.default_rng(3)
    yy, _xx = np.mgrid[0:128, 0:128]
    image = 50.0 + 2.0 * yy + rng.normal(0.0, 1.0, size=(128, 128))
    base = background_gradient_score(image)
    scaled = background_gradient_score(image * 7.0)
    assert base is not None
    assert scaled is not None
    assert abs(scaled - base) < 1e-9


def test_too_small_image_returns_none() -> None:
    """An image spanning fewer than four blocks per axis is undefined."""
    small = np.ones((2 * BLOCK_SIZE, 8 * BLOCK_SIZE), dtype=np.float64)
    assert background_gradient_score(small) is None


def test_nan_pixels_are_ignored() -> None:
    """Non-finite pixels do not poison the score; a ramp still reads high."""
    rng = np.random.default_rng(4)
    _yy, xx = np.mgrid[0:256, 0:256]
    image = 100.0 + 3.0 * xx + rng.normal(0.0, 2.0, size=(256, 256))
    image[0:16, 0:16] = np.nan  # one fully-missing block
    image[100, 100] = np.nan  # a scattered dropout
    score = background_gradient_score(image)
    assert score is not None
    assert score > 5.0


def test_all_nan_image_returns_none() -> None:
    """An all-missing image yields too few finite block medians to fit."""
    image = np.full((128, 128), np.nan, dtype=np.float64)
    assert background_gradient_score(image) is None


def test_exact_plane_ramp_saturates_not_none() -> None:
    """A noiseless ramp (zero residual sigma) returns the saturated sentinel.

    The image is an exact integer affine ramp with only three non-collinear
    blocks left finite.  Three block medians define the fitted affine plane
    exactly, so the fit residual -- and hence the MAD-sigma -- is exactly zero.
    A perfectly clean gradient is present, not absent, so the score must be the
    finite saturated sentinel rather than ``None``.
    """
    _yy, xx = np.mgrid[0 : 6 * BLOCK_SIZE, 0 : 6 * BLOCK_SIZE]
    image = (3 * xx + 2 * _yy).astype(np.float64)
    finite = np.zeros(image.shape, dtype=bool)
    for i, j in [(0, 0), (0, 1), (1, 0)]:
        finite[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE, j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE] = True
    image[~finite] = np.nan
    assert background_gradient_score(image) == SATURATED_GRADIENT_SCORE


def test_constant_field_still_returns_none() -> None:
    """A perfectly constant field has no gradient and returns None, not saturated."""
    image = np.full((128, 128), 42.0, dtype=np.float64)
    assert background_gradient_score(image) is None


def test_non_2d_image_raises_type_error() -> None:
    """A non-2-D image is rejected before any masking or downsampling."""
    image = np.zeros((64,), dtype=np.float64)
    with pytest.raises(TypeError, match='image must be 2-D'):
        background_gradient_score(image)


def test_non_boolean_mask_raises_type_error() -> None:
    """A non-boolean sensor mask is rejected rather than silently coerced."""
    image = np.zeros((64, 64), dtype=np.float64)
    mask = np.ones((64, 64), dtype=np.float64)
    with pytest.raises(TypeError, match='sensor_mask must be a boolean ndarray'):
        background_gradient_score(image, mask)  # type: ignore[arg-type]  # wrong dtype on purpose


def test_mismatched_mask_shape_raises_value_error() -> None:
    """A mask whose shape differs from the image is rejected (no broadcasting)."""
    image = np.zeros((64, 64), dtype=np.float64)
    mask = np.ones((1, 64), dtype=bool)
    with pytest.raises(ValueError, match='sensor_mask shape'):
        background_gradient_score(image, mask)


def test_sensor_mask_makes_score_independent_of_padding() -> None:
    """With a sensor mask, the padding value cannot affect the score.

    The orchestrator feeds the extfov-padded frame; the padding value must not
    leak into the gradient score.  The same sensor content under two different
    padding values yields the same masked score, and (in general) different
    unmasked scores.
    """
    rng = np.random.default_rng(9)
    interior = 100.0 + 3.0 * np.mgrid[0:128, 0:128][1] + rng.normal(0.0, 2.0, size=(128, 128))
    mask = np.zeros((160, 160), dtype=bool)
    mask[16:144, 16:144] = True
    img_pad0 = np.zeros((160, 160), dtype=np.float64)
    img_pad0[16:144, 16:144] = interior
    img_pad1 = np.full((160, 160), 1000.0, dtype=np.float64)
    img_pad1[16:144, 16:144] = interior
    masked0 = background_gradient_score(img_pad0, mask)
    masked1 = background_gradient_score(img_pad1, mask)
    assert masked0 is not None
    assert masked1 is not None
    # Padding excluded: the score depends only on the sensor content.
    assert abs(masked0 - masked1) < 1e-9
    # Without the mask, the padding value does leak into the score.
    assert background_gradient_score(img_pad0) != background_gradient_score(img_pad1)
