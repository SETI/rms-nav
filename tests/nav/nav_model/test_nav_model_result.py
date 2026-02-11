"""Unit tests for NavModelResult dataclass."""

import numpy as np

from nav.nav_model.nav_model_result import NavModelResult


def test_nav_model_result_defaults() -> None:
    """NavModelResult with no arguments has all fields None."""
    result = NavModelResult()
    assert result.model_img is None
    assert result.model_mask is None
    assert result.weighted_mask is None
    assert result.range is None
    assert result.blur_amount is None
    assert result.uncertainty is None
    assert result.confidence is None
    assert result.stretch_regions is None
    assert result.annotations is None


def test_nav_model_result_with_values() -> None:
    """NavModelResult accepts and stores provided values."""
    img = np.zeros((10, 10), dtype=np.float64)
    mask = np.ones((10, 10), dtype=bool)
    result = NavModelResult(
        model_img=img,
        model_mask=mask,
        weighted_mask=None,
        range=1000.0,
        blur_amount=None,
        uncertainty=0.5,
        confidence=1.0,
        stretch_regions=None,
        annotations=None,
    )
    assert result.model_img is img
    assert result.model_mask is mask
    assert result.weighted_mask is None
    assert result.range == 1000.0
    assert result.blur_amount is None
    assert result.uncertainty == 0.5
    assert result.confidence == 1.0
    assert result.stretch_regions is None
    assert result.annotations is None


def test_nav_model_result_range_array() -> None:
    """NavModelResult accepts range as an array."""
    rng = np.full((5, 5), 500.0, dtype=np.float64)
    result = NavModelResult(range=rng)
    assert result.range is rng
    assert np.all(result.range == 500.0)
