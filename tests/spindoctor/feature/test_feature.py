"""Tests for ``spindoctor.feature.feature.NavFeature`` and ``NavReliabilityBreakdown``."""

import numpy as np
import pytest

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import StarFlags
from spindoctor.feature.geometry import StarGeometry
from spindoctor.support.filters import NavFilterKind, NavFilterSpec


def _make_star_feature(**overrides: object) -> NavFeature:
    """Construct a minimal STAR NavFeature, accepting per-test overrides."""
    base: dict[str, object] = {
        'feature_id': 'star:UCAC4:144787700',
        'feature_type': NavFeatureType.STAR,
        'source_model': 'stars',
        'geometry': StarGeometry(
            predicted_vu=(100.0, 200.0),
            catalog_vu=(100.0, 200.0),
            bbox_extfov_vu=(95, 195, 105, 205),
        ),
        'subject_range_km': 1.0e10,
        'position_cov_px': np.eye(2, dtype=np.float64) * 0.25,
        'intensity_sigma_rel': 0.05,
        'preferred_filter': NavFilterSpec(kind=NavFilterKind.NONE),
        'reliability': 0.7,
        'reliability_reasons': NavReliabilityBreakdown(predicted_snr=5.0),
        'usable_types': frozenset({NavFeatureType.STAR}),
        'flags': StarFlags(saturated=False),
    }
    base.update(overrides)
    return NavFeature(**base)  # type: ignore[arg-type]


def test_navfeature_constructs() -> None:
    """A minimal STAR feature is constructed cleanly."""
    feat = _make_star_feature()
    assert feat.feature_type is NavFeatureType.STAR
    assert feat.feature_id.startswith('star:')


def test_navfeature_rejects_empty_id() -> None:
    """An empty feature_id raises ValueError with a clear message."""
    with pytest.raises(ValueError, match='feature_id'):
        _make_star_feature(feature_id='')


def test_navfeature_rejects_reliability_above_one() -> None:
    """Reliability outside [0, 1] is rejected."""
    with pytest.raises(ValueError, match='reliability'):
        _make_star_feature(reliability=1.5)


def test_navfeature_rejects_reliability_below_zero() -> None:
    """Negative reliability is rejected."""
    with pytest.raises(ValueError, match='reliability'):
        _make_star_feature(reliability=-0.1)


def test_navfeature_rejects_usable_types_missing_self() -> None:
    """``usable_types`` must contain the feature's own type."""
    with pytest.raises(ValueError, match='usable_types'):
        _make_star_feature(usable_types=frozenset({NavFeatureType.LIMB_ARC}))


def test_navfeature_rejects_non_2x2_covariance() -> None:
    """A 3x3 covariance matrix is rejected."""
    with pytest.raises(ValueError, match='2x2'):
        _make_star_feature(position_cov_px=np.eye(3, dtype=np.float64))


def test_navfeature_rejects_asymmetric_covariance() -> None:
    """A non-symmetric covariance matrix is rejected."""
    cov = np.array([[1.0, 0.5], [0.0, 1.0]], np.float64)
    with pytest.raises(ValueError, match='symmetric'):
        _make_star_feature(position_cov_px=cov)


def test_navfeature_rejects_indefinite_covariance() -> None:
    """A covariance with negative eigenvalues is rejected."""
    cov = np.array([[1.0, 0.0], [0.0, -1.0]], np.float64)
    with pytest.raises(ValueError, match='positive-semidefinite'):
        _make_star_feature(position_cov_px=cov)


def test_navfeature_freezes_covariance() -> None:
    """The position covariance is made read-only after construction."""
    feat = _make_star_feature()
    assert feat.position_cov_px is not None
    assert not feat.position_cov_px.flags.writeable


def test_navfeature_eq_by_feature_id() -> None:
    """Two features with the same feature_id compare equal."""
    a = _make_star_feature()
    b = _make_star_feature()
    assert a == b


def test_navfeature_neq_different_id() -> None:
    """Two features with different feature_ids compare unequal."""
    a = _make_star_feature()
    b = _make_star_feature(feature_id='star:UCAC4:99999999')
    assert a != b


def test_navfeature_hash_uses_feature_id() -> None:
    """NavFeature hashing groups by feature_id."""
    a = _make_star_feature()
    b = _make_star_feature()
    assert hash(a) == hash(b)


def test_navfeature_template_mismatch_raises() -> None:
    """template_img and template_mask with mismatched shapes are rejected."""
    img = np.zeros((4, 4), np.float64)
    mask = np.zeros((4, 5), bool)
    with pytest.raises(ValueError, match='template_img shape'):
        _make_star_feature(template_img=img, template_mask=mask)


def test_navfeature_template_freezes_arrays() -> None:
    """Template arrays are made read-only on construction."""
    img = np.ones((4, 4), np.float64)
    mask = np.ones((4, 4), bool)
    feat = _make_star_feature(template_img=img, template_mask=mask)
    assert feat.template_img is not None
    assert feat.template_mask is not None
    assert not feat.template_img.flags.writeable
    assert not feat.template_mask.flags.writeable


def test_reliability_breakdown_defaults_none() -> None:
    """All NavReliabilityBreakdown fields default to None."""
    rb = NavReliabilityBreakdown()
    assert rb.predicted_snr is None
    assert rb.in_body_silhouette is None
