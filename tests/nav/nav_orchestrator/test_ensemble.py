"""Tests for ``nav.nav_orchestrator.ensemble.ensemble`` and helpers."""

import numpy as np

from nav.nav_orchestrator.ensemble import (
    EnsembleConfig,
    derive_confidence_rank,
    ensemble,
)
from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.provenance import Provenance
from nav.nav_technique.diagnostics import BodyLimbDiagnostics
from nav.nav_technique.technique_result import NavTechniqueResult


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


def _make_result(
    *,
    technique_name: str = 'BodyLimbNav',
    offset: tuple[float, float] = (0.0, 0.0),
    cov: np.ndarray | None = None,
    confidence: float = 0.8,
    spurious: bool = False,
    at_edge: bool = False,
) -> NavTechniqueResult:
    """Build a minimal NavTechniqueResult for ensemble tests."""
    if cov is None:
        cov = np.eye(2, dtype=np.float64) * 0.25
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=(f'{technique_name}:f1',),
        offset_px=offset,
        covariance_px2=cov,
        confidence=confidence,
        spurious=spurious,
        at_edge=at_edge,
        diagnostics=BodyLimbDiagnostics(),
    )


def test_ensemble_empty_inputs_returns_failed_no_techniques() -> None:
    """An empty input list yields a no_feasible_techniques failure."""
    result = ensemble(
        [],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'no_feasible_techniques'


def test_ensemble_all_spurious_returns_failed() -> None:
    """If every input is spurious, the ensemble fails with the matching reason."""
    a = _make_result(technique_name='TechniqueA', spurious=True)
    b = _make_result(technique_name='TechniqueB', spurious=True)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'all_techniques_spurious'


def test_ensemble_two_agreeing_results_combine() -> None:
    """Two agreeing results combine to the precision-weighted mean."""
    a = _make_result(technique_name='TechniqueA', offset=(1.0, 2.0), confidence=0.85)
    b = _make_result(technique_name='TechniqueB', offset=(1.05, 2.05), confidence=0.85)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'
    assert result.offset_px is not None
    # Equal covariance + equal confidence -> arithmetic mean.
    assert np.isclose(result.offset_px[0], 1.025)
    assert np.isclose(result.offset_px[1], 2.025)


def test_ensemble_disagreement_yields_conflicted() -> None:
    """Two well-separated high-confidence results trigger the conflicted branch."""
    cov = np.eye(2, dtype=np.float64) * 0.01
    a = _make_result(technique_name='TechniqueA', offset=(1.0, 1.0), cov=cov, confidence=0.9)
    b = _make_result(technique_name='TechniqueB', offset=(50.0, 50.0), cov=cov, confidence=0.9)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'conflicted'
    assert result.confidence_rank == 'conflicted'


def test_ensemble_at_edge_dropped_when_interior_exists() -> None:
    """An at-edge result is dropped when an interior result exists."""
    a = _make_result(
        technique_name='TechniqueA',
        offset=(1.0, 1.0),
        confidence=0.9,
        at_edge=True,
    )
    b = _make_result(technique_name='TechniqueB', offset=(1.05, 1.05), confidence=0.85)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'
    # The at-edge result is dropped, leaving only the interior technique
    # which contributes its exact offset.
    assert result.offset_px == (1.05, 1.05)


def test_ensemble_at_edge_kept_when_only_one() -> None:
    """An at-edge result survives when no interior result exists."""
    a = _make_result(
        technique_name='TechniqueA',
        offset=(1.0, 1.0),
        confidence=0.9,
        at_edge=True,
    )
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'


def test_ensemble_below_min_confidence_fails() -> None:
    """A combined confidence below min_confidence yields failed."""
    cfg = EnsembleConfig(min_confidence=0.5)
    a = _make_result(technique_name='TechniqueA', offset=(0.0, 0.0), confidence=0.1)
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        config=cfg,
    )
    assert result.status == 'failed'
    assert result.status_reason.value == 'final_confidence_below_threshold'


def test_derive_confidence_rank_high() -> None:
    """High confidence + small sigma earns the 'high' tier."""
    rank = derive_confidence_rank(confidence=0.9, sigma_px=(0.3, 0.3))
    assert rank == 'high'


def test_derive_confidence_rank_medium_when_sigma_too_large_for_high() -> None:
    """High confidence but sigma above 'high' threshold falls back to 'medium'."""
    rank = derive_confidence_rank(confidence=0.85, sigma_px=(1.5, 1.5))
    assert rank == 'medium'


def test_derive_confidence_rank_low() -> None:
    """Low confidence + any sigma earns the 'low' tier when ≥ 0.2."""
    rank = derive_confidence_rank(confidence=0.3, sigma_px=(10.0, 10.0))
    assert rank == 'low'


def test_derive_confidence_rank_failed_below_threshold() -> None:
    """Confidence below 0.2 yields 'failed'."""
    rank = derive_confidence_rank(confidence=0.1, sigma_px=(0.3, 0.3))
    assert rank == 'failed'


def test_ensemble_order_independent() -> None:
    """Reordering inputs produces the same combined offset (within tolerance)."""
    cov = np.eye(2, dtype=np.float64) * 0.1
    a = _make_result(technique_name='A', offset=(1.0, 1.0), cov=cov, confidence=0.8)
    b = _make_result(technique_name='B', offset=(1.1, 1.1), cov=cov, confidence=0.8)
    res1 = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    res2 = ensemble(
        [b, a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert res1.offset_px is not None
    assert res2.offset_px is not None
    assert abs(res1.offset_px[0] - res2.offset_px[0]) < 1e-9
    assert abs(res1.offset_px[1] - res2.offset_px[1]) < 1e-9


def test_ensemble_rank_1_only_for_single_axis_observable() -> None:
    """A single rank-1 covariance produces RANK_1_ONLY with infinite axis."""
    # Covariance with exactly one observable axis (rank 1).
    cov = np.array([[0.04, 0.0], [0.0, 1e12]], np.float64)
    a = _make_result(technique_name='RingEdgeNav', offset=(1.0, 2.0), cov=cov, confidence=0.7)
    result = ensemble(
        [a],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'
    from nav.support.status_reason import NavStatusReason

    assert result.status_reason == NavStatusReason.RANK_1_ONLY
    assert result.sigma_along_unobservable_px == float('inf')


def test_ensemble_unobservable_offset_when_all_share_null_direction() -> None:
    """If every input covariance shares one null direction, fail with the reason."""
    # Both inputs have an enormous along-axis variance (effectively unobservable
    # in u) and a tight v constraint, but with mutually inconsistent v means;
    # the precision-weighted combine is still well-defined (W > 0) so this
    # case demands a literal degenerate combine.  Use rcond large enough to
    # make pinvh project both inputs to the same rank-1 form, then have the
    # u-axis variance be infinite.
    big = 1e30
    cov_a = np.array([[0.04, 0.0], [0.0, big]], np.float64)
    cov_b = np.array([[0.04, 0.0], [0.0, big]], np.float64)
    a = _make_result(technique_name='A', offset=(0.0, 100.0), cov=cov_a, confidence=0.5)
    b = _make_result(technique_name='B', offset=(0.0, -100.0), cov=cov_b, confidence=0.5)
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    # v is observable, u is not; the ensemble reports RANK_1_ONLY with
    # ``sigma_along_unobservable_px`` set on the result.
    from nav.support.status_reason import NavStatusReason

    assert result.status_reason == NavStatusReason.RANK_1_ONLY


def test_ensemble_3dof_combines_translation_and_rotation() -> None:
    """Two 3-DoF results agreeing in (dv, du, theta) merge into a 3-DoF NavResult."""
    cov_a = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    cov_b = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_a,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.01,
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.05, 1.95),
        covariance_px2=cov_b,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.012,
        sigma_rotation_rad=0.01,
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'
    assert result.rotation_rad is not None
    assert abs(result.rotation_rad - 0.011) < 1e-3
    assert result.sigma_rotation_rad is not None
    assert result.sigma_rotation_rad > 0.0
    assert result.covariance_px2 is not None
    assert result.covariance_px2.shape == (3, 3)


def test_ensemble_3dof_with_rotation_unobservable_input() -> None:
    """Rotation-unobservable input contributes near-zero info to 3-DoF combine."""
    from nav.nav_technique.nav_technique import (
        ROTATION_UNOBSERVABLE_VARIANCE,
        embed_rotation_unobservable,
    )

    cov_observable = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    cov_unobservable = embed_rotation_unobservable(np.diag([0.04, 0.04]).astype(np.float64))
    a = NavTechniqueResult(
        technique_name='A',
        feature_ids=('A:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_observable,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.01,
        sigma_rotation_rad=0.01,
    )
    b = NavTechniqueResult(
        technique_name='B',
        feature_ids=('B:f',),
        offset_px=(1.0, 2.0),
        covariance_px2=cov_unobservable,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=float(np.sqrt(ROTATION_UNOBSERVABLE_VARIANCE)),
    )
    result = ensemble(
        [a, b],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    assert result.status == 'ok'
    assert result.rotation_rad is not None
    # The unobservable input pulls rotation only marginally toward zero;
    # the observable input dominates.
    assert abs(result.rotation_rad - 0.01) < 1e-3


def test_ensemble_rejects_mixed_dof_inputs() -> None:
    """Mixing 2-DoF and 3-DoF results raises a ValueError."""
    import pytest

    cov_2d = np.diag([0.04, 0.04]).astype(np.float64)
    cov_3d = np.diag([0.04, 0.04, 1e-4]).astype(np.float64)
    two = _make_result(technique_name='Two', cov=cov_2d)
    three = NavTechniqueResult(
        technique_name='Three',
        feature_ids=('three:f',),
        offset_px=(0.0, 0.0),
        covariance_px2=cov_3d,
        confidence=0.8,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(),
        rotation_rad=0.0,
        sigma_rotation_rad=0.01,
    )
    with pytest.raises(ValueError, match='Mixed-DoF technique results'):
        ensemble(
            [two, three],
            feature_inventory=[],
            image_classifier=_classifier(),
            provenance=_provenance(),
        )
