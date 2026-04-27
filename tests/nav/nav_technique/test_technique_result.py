"""Tests for ``nav.nav_technique.technique_result.NavTechniqueResult``."""

import numpy as np
import pytest

from nav.nav_technique.diagnostics import BodyLimbDiagnostics
from nav.nav_technique.technique_result import NavTechniqueResult


def _make_result(**overrides: object) -> NavTechniqueResult:
    """Build a minimal NavTechniqueResult, accepting per-test overrides."""
    base: dict[str, object] = {
        'technique_name': 'BodyLimbNav',
        'feature_ids': ['limb_arc:MIMAS'],
        'offset_px': (0.5, -0.5),
        'covariance_px2': np.eye(2, dtype=np.float64) * 0.25,
        'confidence': 0.8,
        'spurious': False,
        'at_edge': False,
        'diagnostics': BodyLimbDiagnostics(),
    }
    base.update(overrides)
    return NavTechniqueResult(**base)  # type: ignore[arg-type]


def test_navtechniqueresult_constructs() -> None:
    """A minimal NavTechniqueResult is constructed cleanly."""
    res = _make_result()
    assert res.technique_name == 'BodyLimbNav'
    assert res.confidence == 0.8


def test_navtechniqueresult_rejects_non_2x2_or_3x3() -> None:
    """A 4x4 covariance raises ValueError."""
    with pytest.raises(ValueError, match='2x2 or 3x3'):
        _make_result(covariance_px2=np.eye(4, dtype=np.float64))


def test_navtechniqueresult_rejects_indefinite_covariance() -> None:
    """A negative-eigenvalue covariance is rejected."""
    cov = np.array([[1.0, 0.0], [0.0, -1.0]], np.float64)
    with pytest.raises(ValueError, match='positive-semidefinite'):
        _make_result(covariance_px2=cov)


def test_navtechniqueresult_rejects_asymmetric_covariance() -> None:
    """An asymmetric covariance is rejected."""
    cov = np.array([[1.0, 0.5], [0.0, 1.0]], np.float64)
    with pytest.raises(ValueError, match='symmetric'):
        _make_result(covariance_px2=cov)


def test_navtechniqueresult_rejects_confidence_above_one() -> None:
    """Confidence > 1 is rejected."""
    with pytest.raises(ValueError, match='confidence'):
        _make_result(confidence=1.5)


def test_navtechniqueresult_freezes_covariance() -> None:
    """Covariance becomes read-only after construction."""
    res = _make_result()
    assert not res.covariance_px2.flags.writeable


def test_navtechniqueresult_3x3_accepted() -> None:
    """A 3x3 covariance for rotation-fitting techniques is accepted."""
    cov = np.eye(3, dtype=np.float64) * 0.1
    res = _make_result(covariance_px2=cov, rotation_rad=0.001)
    assert res.covariance_px2.shape == (3, 3)
    assert res.rotation_rad == 0.001


def test_navtechniqueresult_eq_by_name_and_features() -> None:
    """Two results with the same technique and features compare equal."""
    a = _make_result()
    b = _make_result()
    assert a == b


def test_navtechniqueresult_neq_different_features() -> None:
    """Two results with different feature_ids compare unequal."""
    a = _make_result(feature_ids=('limb_arc:MIMAS',))
    b = _make_result(feature_ids=('limb_arc:RHEA',))
    assert a != b
