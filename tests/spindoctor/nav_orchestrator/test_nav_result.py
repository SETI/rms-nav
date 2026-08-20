"""Tests for ``spindoctor.nav_orchestrator.nav_result.NavResult``."""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.support.status_reason import NavStatusReason


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
        spindoctor_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )


def _success(**overrides: Any) -> NavResult:
    """Build a valid success ``NavResult`` with named fields replaced.

    Parameters:
        overrides: Field values replacing the valid defaults, so each test
            names only the field it is about.

    Returns:
        The constructed result, when construction accepts the overrides.
    """
    fields: dict[str, Any] = {
        'status': 'success',
        'offset_px': (1.0, 2.0),
        'sigma_px': (0.2, 0.4),
        'sigma_along_unobservable_px': None,
        'confidence_rank': 'high',
        'confidence': 0.9,
        'status_reason': NavStatusReason.OK,
        'covariance_px2': np.diag([0.04, 0.16]),
        'per_technique': [],
        'feature_inventory': [],
        'image_classifier': _classifier(),
        'provenance': _provenance(),
    }
    fields.update(overrides)
    return NavResult(**fields)


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


@pytest.mark.parametrize(
    'bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
@pytest.mark.parametrize('axis', [0, 1], ids=['dv', 'du'])
def test_navresult_refuses_a_non_finite_offset(axis: int, bad: float) -> None:
    """An offset is a position, and no position is a NaN or an infinity.

    The reported offset is this code's own arithmetic over per-technique
    offsets each already required to be finite, so a non-finite one is a
    defect in the combine.  It is also the one number the metadata document
    records unrounded, so nothing downstream would map it onto a sentinel.

    Parameters:
        axis: Which component of the ``(dv, du)`` pair carries the value.
        bad: The non-finite value that component carries.
    """
    offset = [1.0, 2.0]
    offset[axis] = bad
    with pytest.raises(ValueError, match='offset_px must be finite'):
        _success(offset_px=(offset[0], offset[1]))


@pytest.mark.parametrize(
    'bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
@pytest.mark.parametrize('axis', [0, 1], ids=['dv', 'du'])
def test_navresult_refuses_a_non_finite_sigma(axis: int, bad: float) -> None:
    """The per-axis sigmas are kept finite on purpose, so one that is not is wrong.

    An unobservable translation axis is reported through
    ``sigma_along_unobservable_px``, precisely so a per-axis sigma stays a
    measurement rather than an inflated stand-in for one.

    Parameters:
        axis: Which component of the ``(dv, du)`` sigma pair carries the value.
        bad: The non-finite value that component carries.
    """
    sigma = [0.2, 0.4]
    sigma[axis] = bad
    with pytest.raises(ValueError, match='sigma_px must be finite'):
        _success(sigma_px=(sigma[0], sigma[1]))


@pytest.mark.parametrize(
    'bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
def test_navresult_refuses_a_non_finite_rotation(bad: float) -> None:
    """The fitted rotation is held to the rule its per-technique inputs already are.

    Parameters:
        bad: The non-finite value the rotation carries.
    """
    with pytest.raises(ValueError, match='rotation_rad must be finite'):
        _success(rotation_rad=bad)


@pytest.mark.parametrize(
    'bad', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
def test_navresult_refuses_a_non_finite_rotation_sigma(bad: float) -> None:
    """The uncertainty on the fitted rotation is held to that same rule.

    Parameters:
        bad: The non-finite value the rotation sigma carries.
    """
    with pytest.raises(ValueError, match='sigma_rotation_rad must be finite'):
        _success(sigma_rotation_rad=bad)


def test_navresult_keeps_an_infinite_unobservable_sigma() -> None:
    """The one number the result reports as infinite on purpose.

    A fused covariance with an unobservable translation direction reports
    that direction here rather than through an inflated per-axis sigma, so
    the finiteness rule the other numbers are held to must not reach it.
    """
    result = _success(sigma_along_unobservable_px=math.inf)
    assert result.sigma_along_unobservable_px == math.inf
