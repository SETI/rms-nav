"""Tests for ``nav.nav_orchestrator.nav_result.NavResult``."""

import numpy as np
import pytest

from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.nav_result import NavResult
from nav.nav_orchestrator.provenance import Provenance
from nav.support.status_reason import NavStatusReason


def _classifier() -> NavImageClassifierResult:
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )


def _provenance() -> Provenance:
    return Provenance(
        rms_nav_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )


def test_navresult_failed_constructor_no_offset() -> None:
    """NavResult.failed produces status=failed with no offset."""
    result = NavResult.failed(
        status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.offset_px is None
    assert result.confidence == 0.0
    assert result.confidence_rank == 'failed'


def test_navresult_success_derives_sigma_from_covariance() -> None:
    """NavResult.success computes sigma_px from the covariance diagonal."""
    cov = np.diag([0.04, 0.16])
    result = NavResult.success(
        offset_px=(1.0, 2.0),
        covariance_px2=cov,
        confidence=0.85,
        confidence_rank='high',
        status_reason=NavStatusReason.OK,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'success'
    assert result.offset_px == (1.0, 2.0)
    assert result.sigma_px is not None
    assert np.isclose(result.sigma_px[0], 0.2)
    assert np.isclose(result.sigma_px[1], 0.4)


def test_navresult_conflicted_sets_rank_conflicted() -> None:
    """NavResult.conflicted hardcodes confidence_rank='conflicted'."""
    cov = np.eye(2, dtype=np.float64) * 0.5
    result = NavResult.conflicted(
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.3,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.confidence_rank == 'conflicted'
    assert result.status_reason == NavStatusReason.CONFLICTED_TECHNIQUES


def test_navresult_rejects_failed_with_offset() -> None:
    """status='failed' with non-None offset_px raises ValueError."""
    with pytest.raises(ValueError, match='status=failed'):
        NavResult(
            status='failed',
            offset_px=(0.0, 0.0),
            sigma_px=None,
            sigma_along_unobservable_px=None,
            confidence_rank='failed',
            confidence=0.0,
            status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
            covariance_px2=None,
            per_technique=[],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )


def test_navresult_rejects_success_without_offset() -> None:
    """status='success' with None offset_px raises ValueError."""
    with pytest.raises(ValueError, match='offset_px'):
        NavResult(
            status='success',
            offset_px=None,
            sigma_px=None,
            sigma_along_unobservable_px=None,
            confidence_rank='high',
            confidence=0.9,
            status_reason=NavStatusReason.OK,
            covariance_px2=None,
            per_technique=[],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )


def test_navresult_rejects_failed_rank_with_success_status() -> None:
    """confidence_rank='failed' on status='success' raises ValueError."""
    with pytest.raises(ValueError, match='confidence_rank=failed'):
        NavResult(
            status='success',
            offset_px=(0.0, 0.0),
            sigma_px=(0.0, 0.0),
            sigma_along_unobservable_px=None,
            confidence_rank='failed',
            confidence=0.0,
            status_reason=NavStatusReason.OK,
            covariance_px2=np.eye(2, dtype=np.float64),
            per_technique=[],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )
