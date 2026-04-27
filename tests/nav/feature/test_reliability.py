"""Tests for ``nav.feature.reliability.FeatureReliabilityGate``."""

import numpy as np

from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import StarFlags
from nav.feature.geometry import StarGeometry
from nav.feature.reliability import (
    DEFAULT_RELIABILITY_THRESHOLDS,
    FeatureReliabilityGate,
)
from nav.support.filters import NavFilterKind, NavFilterSpec


def _make_star(reliability: float, feature_id: str = 'star:1') -> NavFeature:
    """Build a STAR feature with the given reliability score."""
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        geometry=StarGeometry(
            predicted_vu=(0.0, 0.0),
            catalog_vu=(0.0, 0.0),
            bbox_extfov_vu=(0, 0, 4, 4),
        ),
        subject_range_km=1.0e10,
        position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0),
        usable_types=frozenset({NavFeatureType.STAR}),
        flags=StarFlags(),
    )


def test_gate_keeps_features_above_threshold() -> None:
    """Features above the per-type threshold are kept."""
    gate = FeatureReliabilityGate()
    features = [_make_star(0.8)]
    kept, gated = gate.apply(features)
    assert kept == features
    assert gated == []


def test_gate_drops_features_below_threshold() -> None:
    """Features below the per-type threshold are gated."""
    gate = FeatureReliabilityGate()
    features = [_make_star(0.05)]
    kept, gated = gate.apply(features)
    assert kept == []
    assert len(gated) == 1
    assert 'below_threshold' in gated[0].reason


def test_gate_records_per_feature_reason() -> None:
    """The gate records the reliability and threshold values in the reason."""
    gate = FeatureReliabilityGate()
    feat = _make_star(0.05)
    _, gated = gate.apply([feat])
    assert '0.050' in gated[0].reason
    assert '0.200' in gated[0].reason


def test_gate_uses_per_type_threshold() -> None:
    """The threshold lookup is by feature_type, not a global value."""
    custom_thresholds = dict(DEFAULT_RELIABILITY_THRESHOLDS)
    custom_thresholds[NavFeatureType.STAR] = 0.99
    gate = FeatureReliabilityGate(thresholds=custom_thresholds)
    kept, gated = gate.apply([_make_star(0.9)])
    assert kept == []
    assert len(gated) == 1


def test_gate_unknown_feature_type_passes() -> None:
    """A feature whose type has no threshold falls through with no gate."""
    gate = FeatureReliabilityGate(thresholds={})
    kept, gated = gate.apply([_make_star(0.001)])
    assert len(kept) == 1
    assert gated == []
