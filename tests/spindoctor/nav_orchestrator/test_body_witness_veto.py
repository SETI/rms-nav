"""Tests for ``spindoctor.nav_orchestrator.body_witness_veto``.

The veto is pure logic over already-computed per-technique results, so it is
tested here directly (no rendering) with synthetic results that reproduce the
two confident-wrong body signatures and the legitimate cases that must NOT
veto.
"""

import numpy as np
import pytest

from spindoctor.nav_orchestrator.body_witness_veto import (
    BodyWitnessVeto,
    evaluate_body_witness_veto,
)
from spindoctor.nav_technique.diagnostics import (
    BodyBlobDiagnostics,
    BodyDiscDiagnostics,
    BodyLimbDiagnostics,
    NavTechniqueDiagnostics,
    StarRefineDiagnostics,
    StarUniqueMatchDiagnostics,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError

_SHAPE_LOCK_MAX_PHASE_DEG = 60.0
_COLLAPSE_MIN_PHASE_DEG = 90.0
_FLOOR_PX = 4.0
_FRAC = 0.03
_AGREEMENT_SIGMA = 2.0
_STAR_AGREEMENT_FLOOR_PX = 5.0
_RCOND = 1.0e-9
_TIGHT_COV = np.eye(2, dtype=np.float64) * 0.25


def _result(
    *,
    technique_name: str,
    offset: tuple[float, float],
    source_bodies: frozenset[str],
    spurious: bool = False,
    cov: np.ndarray | None = None,
    diagnostics: NavTechniqueDiagnostics | None = None,
) -> NavTechniqueResult:
    """Build a minimal NavTechniqueResult for veto tests.

    Parameters:
        technique_name: Name the result reports as its producing technique.
        offset: Predicted ``(dv, du)`` offset in pixels.
        source_bodies: Bodies the result claims to have navigated.
        spurious: Whether the result self-flagged spurious.
        cov: Translation covariance; a tight default is used when None.
        diagnostics: Per-technique diagnostics; a bare ``BodyLimbDiagnostics``
            is used when None.

    Returns:
        The assembled ``NavTechniqueResult``.
    """
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=(f'{technique_name}:f1',),
        offset_px=offset,
        covariance_px2=_TIGHT_COV if cov is None else cov,
        confidence=0.8,
        spurious=spurious,
        at_edge=False,
        diagnostics=diagnostics if diagnostics is not None else BodyLimbDiagnostics(),
        source_bodies=source_bodies,
    )


def _blob(
    *,
    offset: tuple[float, float],
    body: str,
    phase_deg: float = 30.0,
    extent_px: float = 150.0,
    spurious: bool = False,
    cov: np.ndarray | None = None,
) -> NavTechniqueResult:
    """Build a BodyBlobNav witness result.

    Parameters:
        offset: Predicted ``(dv, du)`` blob-centroid offset in pixels.
        body: The single body the blob witnesses.
        phase_deg: Maximum phase angle carried on the blob diagnostics.
        extent_px: Predicted body extent carried on the blob diagnostics.
        spurious: Whether the blob result self-flagged spurious.
        cov: Translation covariance; a tight default is used when None.

    Returns:
        A ``NavTechniqueResult`` carrying ``BodyBlobDiagnostics``.
    """
    return _result(
        technique_name='BodyBlobNav',
        offset=offset,
        source_bodies=frozenset({body}),
        spurious=spurious,
        cov=cov,
        diagnostics=BodyBlobDiagnostics(body_extent_px=extent_px, max_phase_angle_deg=phase_deg),
    )


def _star(
    *,
    technique_name: str,
    offset: tuple[float, float],
    diagnostics: NavTechniqueDiagnostics,
    spurious: bool = False,
) -> NavTechniqueResult:
    """Build a star-technique result carrying its own diagnostics.

    Star techniques leave ``source_bodies`` empty, so the result is a
    body-independent geometric witness.

    Parameters:
        technique_name: The star technique the result reports as its producer.
        offset: Predicted ``(dv, du)`` offset in pixels.
        diagnostics: The star diagnostics that classify the fix as single- or
            multi-star.
        spurious: Whether the star result self-flagged spurious (pinned to the
            origin with zero confidence).

    Returns:
        A ``NavTechniqueResult`` with empty ``source_bodies``.
    """
    return _result(
        technique_name=technique_name,
        offset=offset,
        source_bodies=frozenset(),
        spurious=spurious,
        diagnostics=diagnostics,
    )


def _evaluate(
    best_group: list[NavTechniqueResult],
    results: list[NavTechniqueResult],
    fused_offset: tuple[float, float],
    fused_cov: np.ndarray | None = None,
) -> BodyWitnessVeto:
    """Call the veto with the module's default thresholds.

    Parameters:
        best_group: The winning consensus group of technique results.
        results: All per-technique results available to the veto.
        fused_offset: The ensemble's fused ``(dv, du)`` offset in pixels.
        fused_cov: The fused translation covariance; a tight default is used
            when None.

    Returns:
        The ``BodyWitnessVeto`` verdict for the scene.
    """
    return evaluate_body_witness_veto(
        best_group,
        results,
        fused_offset,
        _TIGHT_COV if fused_cov is None else fused_cov,
        shape_lock_max_phase_deg=_SHAPE_LOCK_MAX_PHASE_DEG,
        collapse_min_phase_deg=_COLLAPSE_MIN_PHASE_DEG,
        disagreement_floor_px=_FLOOR_PX,
        disagreement_frac=_FRAC,
        agreement_sigma=_AGREEMENT_SIGMA,
        star_agreement_floor_px=_STAR_AGREEMENT_FLOOR_PX,
        rcond=_RCOND,
    )


def test_shape_lock_fires_when_blob_contradicts_geometric_consensus() -> None:
    """A geometric lock the low-phase blob disagrees with reports shape-lock."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(-14.5, -6.5),
        source_bodies=frozenset({'HYPERION'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    limb = _result(
        technique_name='BodyLimbNav', offset=(-14.5, -7.5), source_bodies=frozenset({'HYPERION'})
    )
    blob = _blob(offset=(-3.8, -3.1), body='HYPERION')
    verdict = _evaluate([disc, limb], [disc, limb, blob], (-14.5, -7.0))
    assert verdict is BodyWitnessVeto.SHAPE_LOCK_SUSPECT


def test_shape_lock_suppressed_by_corroborating_star_consensus() -> None:
    """A trusted star fix that confirms the geometry makes the blob the outlier.

    On a body whose albedo dichotomy drags the brightness centroid off the true
    disc center, the low-phase blob disagrees with a correct limb fit; when a
    multi-star fix independently agrees with that geometric offset within the
    grouping floor, the shape-lock verdict is suppressed and the frame commits.
    """
    limb = _result(
        technique_name='BodyLimbNav', offset=(1.46, 12.85), source_bodies=frozenset({'IAPETUS'})
    )
    star = _star(
        technique_name='StarUniqueMatchNav',
        offset=(1.53, 11.33),
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
    )
    blob = _blob(offset=(1.5, 25.0), body='IAPETUS')
    verdict = _evaluate([limb, star], [limb, star, blob], (1.5, 12.0))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_not_suppressed_by_single_star_fix() -> None:
    """A lone-star fix is not a corroborated consensus, so the veto still fires.

    A one-star unique match localizes the offset from a single detection whose
    identification nothing cross-checks; it cannot vouch for the geometry, so a
    shape lock the blob disputes is still reported conflicted.
    """
    limb = _result(
        technique_name='BodyLimbNav', offset=(1.46, 12.85), source_bodies=frozenset({'IAPETUS'})
    )
    star = _star(
        technique_name='StarUniqueMatchNav',
        offset=(1.53, 11.33),
        diagnostics=StarUniqueMatchDiagnostics(mode='one_star'),
    )
    blob = _blob(offset=(1.5, 25.0), body='IAPETUS')
    verdict = _evaluate([limb, star], [limb, star, blob], (1.5, 12.0))
    assert verdict is BodyWitnessVeto.SHAPE_LOCK_SUSPECT


def test_shape_lock_not_suppressed_when_star_fix_disagrees() -> None:
    """A multi-star fix far from the geometric offset does not corroborate it.

    When the only trusted star fix disagrees with the fused offset by more than
    the grouping floor, it does not confirm the geometry, so a shape lock the
    blob disputes still fires.
    """
    limb = _result(
        technique_name='BodyLimbNav', offset=(-14.5, -7.5), source_bodies=frozenset({'HYPERION'})
    )
    star = _star(
        technique_name='StarRefineNav',
        offset=(3.0, 4.0),
        diagnostics=StarRefineDiagnostics(n_stars_used=6),
    )
    blob = _blob(offset=(-3.8, -3.1), body='HYPERION')
    verdict = _evaluate([limb, star], [limb, star, blob], (-14.5, -7.5))
    assert verdict is BodyWitnessVeto.SHAPE_LOCK_SUSPECT


def test_shape_lock_not_suppressed_by_spurious_star_at_origin() -> None:
    """A spurious star pinned to the origin cannot corroborate a wrong lock.

    A spurious star fix carries a multi-star diagnostic yet is pinned to offset
    (0, 0), so on a genuine shape lock sitting near the origin its zero offset
    would fall within the corroboration floor of the geometric consensus.  It is
    a defeated fix, not an independent witness, so it must not suppress the veto:
    the shape lock still reports conflicted.
    """
    limb = _result(
        technique_name='BodyLimbNav', offset=(0.5, 0.5), source_bodies=frozenset({'HYPERION'})
    )
    star = _star(
        technique_name='StarUniqueMatchNav',
        offset=(0.0, 0.0),
        diagnostics=StarUniqueMatchDiagnostics(mode='two_star'),
        spurious=True,
    )
    blob = _blob(offset=(12.0, 0.5), body='HYPERION')
    verdict = _evaluate([limb, star], [limb, star, blob], (0.5, 0.5))
    assert verdict is BodyWitnessVeto.SHAPE_LOCK_SUSPECT


def test_shape_lock_silent_when_blob_agrees() -> None:
    """A well-matched body whose blob agrees is not vetoed."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(1.4, -0.6),
        source_bodies=frozenset({'HYPERION'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(1.4, -0.5), body='HYPERION')
    verdict = _evaluate([disc], [disc, blob], (1.4, -0.6))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_silent_when_blob_sigma_makes_separation_insignificant() -> None:
    """A noisy blob whose large sigma makes the pixel separation statistically
    consistent with agreement is not vetoed, even past the pixel floor."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(-14.5, -6.5),
        source_bodies=frozenset({'HYPERION'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    # A 100 px-variance (10 px sigma) centroid puts the ~11 px separation at
    # ~1.1 sigma -- below the 2.0 Mahalanobis gate.
    blob = _blob(offset=(-3.8, -3.1), body='HYPERION', cov=np.eye(2, dtype=np.float64) * 100.0)
    verdict = _evaluate([disc], [disc, blob], (-14.5, -6.5))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_silent_when_blob_is_high_phase() -> None:
    """Past half phase the blob is not a trustworthy witness, so no veto fires."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(-14.5, -6.5),
        source_bodies=frozenset({'HYPERION'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(-3.8, -3.1), body='HYPERION', phase_deg=155.0)
    verdict = _evaluate([disc], [disc, blob], (-14.5, -6.5))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_silent_on_multi_body_frame() -> None:
    """A companion body corrupts the centroid, so multi-body frames are exempt."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(-14.5, -6.5),
        source_bodies=frozenset({'RHEA'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    blob_a = _blob(offset=(-3.8, -3.1), body='RHEA')
    blob_b = _blob(offset=(0.0, 0.0), body='DIONE')
    verdict = _evaluate([disc], [disc, blob_a, blob_b], (-14.5, -6.5))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_silent_when_witness_is_spurious() -> None:
    """A spurious blob witness carries no disagreement signal."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(-14.5, -6.5),
        source_bodies=frozenset({'HYPERION'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(-3.8, -3.1), body='HYPERION', spurious=True)
    verdict = _evaluate([disc], [disc, blob], (-14.5, -6.5))
    assert verdict is BodyWitnessVeto.NONE


def test_shape_lock_tolerance_scales_with_body_diameter() -> None:
    """A disagreement under the diameter-scaled tolerance does not veto."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(0.0, 0.0),
        source_bodies=frozenset({'BIGMOON'}),
        diagnostics=BodyDiscDiagnostics(),
    )
    # 5 px disagreement clears the 4 px floor but is under 0.03 * 400 = 12 px.
    blob = _blob(offset=(5.0, 0.0), body='BIGMOON', extent_px=400.0)
    verdict = _evaluate([disc], [disc, blob], (0.0, 0.0))
    assert verdict is BodyWitnessVeto.NONE


def test_lone_blob_collapsed_fires_when_high_phase_blob_disagrees() -> None:
    """A haze-dragged high-phase blob far from the spurious disc declines the frame."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(0.5, 0.5),
        source_bodies=frozenset({'TITAN'}),
        spurious=True,
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(3.2, 29.3), body='TITAN', phase_deg=155.0, extent_px=170.0)
    verdict = _evaluate([blob], [disc, blob], (3.2, 29.3))
    assert verdict is BodyWitnessVeto.LONE_BLOB_COLLAPSED_REGIME


def test_blob_result_without_blob_diagnostics_raises_contract_error() -> None:
    """A BodyBlobNav result must carry BodyBlobDiagnostics, not a default.

    Reading a zero phase and zero extent off a malformed blob would make it look
    like a low-phase, zero-size witness and could bypass the collapsed-regime
    veto, so the contract fails loudly instead.
    """
    bad_blob = _result(
        technique_name='BodyBlobNav',
        offset=(3.2, 29.3),
        source_bodies=frozenset({'TITAN'}),
        diagnostics=BodyLimbDiagnostics(),
    )
    with pytest.raises(NavContractError, match='must carry BodyBlobDiagnostics'):
        _evaluate([bad_blob], [bad_blob], (3.2, 29.3))


def test_lone_blob_not_vetoed_when_spurious_disc_agrees() -> None:
    """A small body whose spurious disc agrees with the blob is a legitimate success."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(1.5, -0.5),
        source_bodies=frozenset({'RHEA'}),
        spurious=True,
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(1.5, -0.5), body='RHEA', phase_deg=120.0, extent_px=20.0)
    verdict = _evaluate([blob], [disc, blob], (1.5, -0.5))
    assert verdict is BodyWitnessVeto.NONE


def test_lone_blob_not_vetoed_when_blob_is_low_phase() -> None:
    """Below half phase the centroid is reliable even if it disagrees with a spurious disc."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(0.0, 0.0),
        source_bodies=frozenset({'ENCELADUS'}),
        spurious=True,
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(20.0, 0.0), body='ENCELADUS', phase_deg=35.0, extent_px=8.0)
    verdict = _evaluate([blob], [disc, blob], (20.0, 0.0))
    assert verdict is BodyWitnessVeto.NONE


def test_lone_blob_collapsed_silent_on_multi_body_frame() -> None:
    """A multi-body frame is exempt: a correct joint blob fit is not declined."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(0.5, 0.5),
        source_bodies=frozenset({'TITAN'}),
        spurious=True,
        diagnostics=BodyDiscDiagnostics(),
    )
    blob_a = _blob(offset=(3.2, 29.3), body='TITAN', phase_deg=155.0, extent_px=170.0)
    blob_b = _blob(offset=(3.2, 29.3), body='RHEA', phase_deg=150.0, extent_px=90.0)
    verdict = _evaluate([blob_a, blob_b], [disc, blob_a, blob_b], (3.2, 29.3))
    assert verdict is BodyWitnessVeto.NONE


def test_lone_blob_not_vetoed_without_spurious_sibling() -> None:
    """A legitimate blob-only success (no geometric technique ran) is not vetoed."""
    blob = _blob(offset=(1.0, -0.5), body='PHOEBE', phase_deg=155.0)
    verdict = _evaluate([blob], [blob], (1.0, -0.5))
    assert verdict is BodyWitnessVeto.NONE


def test_lone_blob_not_vetoed_when_spurious_sibling_is_other_body() -> None:
    """A spurious geometric result on a different body does not veto the blob."""
    disc = _result(
        technique_name='BodyDiscCorrelateNav',
        offset=(0.5, 0.5),
        source_bodies=frozenset({'RHEA'}),
        spurious=True,
        diagnostics=BodyDiscDiagnostics(),
    )
    blob = _blob(offset=(1.0, -0.5), body='PHOEBE', phase_deg=155.0)
    verdict = _evaluate([blob], [disc, blob], (1.0, -0.5))
    assert verdict is BodyWitnessVeto.NONE


def test_no_veto_for_bodyless_consensus() -> None:
    """A consensus with no source bodies (ring / star) is never body-vetoed."""
    ring = _result(technique_name='RingEdgeNav', offset=(2.0, 1.0), source_bodies=frozenset())
    verdict = _evaluate([ring], [ring], (2.0, 1.0))
    assert verdict is BodyWitnessVeto.NONE
