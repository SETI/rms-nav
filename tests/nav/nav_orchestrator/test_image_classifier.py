"""Tests for ``nav.nav_orchestrator.image_classifier.NavImageClassifier``."""

import numpy as np
import pytest

from nav.nav_orchestrator.image_classifier import (
    ImageQualityThresholds,
    NavImageClassifier,
)


def test_classifier_clean_image_no_flags() -> None:
    """A normal image returns class 'clean' with empty flags."""
    rng = np.random.default_rng(seed=1)
    image = rng.standard_normal(size=(64, 64)) + 100.0
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert result.flags == []


def test_classifier_blank_image() -> None:
    """A near-zero image is classified as blank."""
    image = np.zeros((64, 64), np.float64)
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'blank'


def test_classifier_fully_overexposed() -> None:
    """An image saturated above the threshold yields fully_overexposed."""
    image = np.full((64, 64), 4095.0, np.float64)
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'fully_overexposed'
    assert result.saturation_frac == 1.0


def test_classifier_mostly_missing_data() -> None:
    """An image dominated by the missing-data marker yields mostly_missing_data."""
    image = np.full((64, 64), 0.0, np.float64)
    image[:48, :] = 100.0  # Some non-zero data, but most is zero (missing marker).
    classifier = NavImageClassifier(
        thresholds=ImageQualityThresholds(
            missing_data_marker_dn=0.0,
            max_missing_frac_clean=0.10,
        )
    )
    result = classifier.classify(image)
    assert result.image_class == 'mostly_missing_data'


def test_classifier_partial_dropout_flag() -> None:
    """A small fraction of missing data raises the partial_dropout flag."""
    rng = np.random.default_rng(seed=2)
    image = rng.standard_normal(size=(64, 64)) + 100.0
    image[:5, :] = 0.0  # ~7.8% missing
    classifier = NavImageClassifier()
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert 'partial_dropout' in result.flags


def test_classifier_noisy_flag() -> None:
    """High image noise raises the noisy flag."""
    rng = np.random.default_rng(seed=3)
    image = rng.standard_normal(size=(64, 64)) * 50.0 + 100.0
    classifier = NavImageClassifier(thresholds=ImageQualityThresholds(noisy_threshold=10.0))
    result = classifier.classify(image)
    assert result.image_class == 'clean'
    assert 'noisy' in result.flags


def test_classifier_rejects_non_2d() -> None:
    """3-D input raises TypeError."""
    image = np.zeros((4, 4, 4), np.float64)
    classifier = NavImageClassifier()
    with pytest.raises(TypeError, match='2-D'):
        classifier.classify(image)


def test_classifier_uses_sensor_mask() -> None:
    """Pixels outside the sensor mask are excluded from the classification."""
    image = np.zeros((64, 64), np.float64)
    image[:, :32] = 100.0
    image[:, 32:] = 4095.0  # extfov padding has bogus saturation values
    mask = np.zeros((64, 64), bool)
    mask[:, :32] = True  # sensor is the left half
    classifier = NavImageClassifier()
    result = classifier.classify(image, sensor_mask=mask)
    # Saturation fraction is computed on the sensor area only (zero saturated).
    assert result.saturation_frac == 0.0
    assert result.image_class == 'clean'
