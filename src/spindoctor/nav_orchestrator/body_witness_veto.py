"""Body-witness veto -- cross-technique guards on confident-wrong body locks.

Two body failure modes converge cleanly and yet pass the ordinary ensemble
gates, because the offending offset carries a coherent *systematic* bias no
per-technique diagnostic can see:

- **Shape lock.**  On a body whose real silhouette departs from the predicted
  ellipsoid (mesh-level shape mismatch), the full-disc NCC
  (:class:`~spindoctor.nav_technique.nav_technique_body_disc.BodyDiscCorrelateNav`)
  and the limb fit
  (:class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav`) both
  lock onto the mismatched model and agree with each other at a multi-pixel
  wrong offset, fusing to a high confidence.  The pose-free brightness centroid
  (:class:`~spindoctor.nav_technique.nav_technique_body_blob.BodyBlobNav`) does
  not chase the shape, so it disagrees by pixels -- the one signal that the
  geometric lock is wrong.

- **Collapsed regime.**  At extreme phase a forward-scattering haze crescent
  defeats the disc correlation (it self-flags spurious) while dragging the
  brightness centroid tens of pixels off the body center.  The lone surviving
  blob then carries the fused answer at its structural cap, above the
  acceptance gate.  The defeated disc still self-flags spurious *at a
  position*: its coarse silhouette peak sits near the true center, so the
  spurious result's offset is the witness -- when the high-phase blob disagrees
  with it by pixels, the centroid was haze-dragged and the frame is declined.
  A small or faint body whose disc merely fails to lock does not trip this: the
  spurious disc there agrees with the blob (both found the same body), and
  below half phase the centroid is a reliable geometric-center proxy anyway.

Neither mode is visible inside a single technique's diagnostics, so the guard
is cross-technique: it reads the pose-free blob against the geometric
techniques. The blob is kept available by
:attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.runs_as_witness`
even when a primary already covers the body, and is still superseded in the
fuse. The brightness centroid tracks the geometric center at low phase, but its
lit-hemisphere bias grows toward half phase and is re-modeled only once the
coarse acquisition switches to the crescent template past it. The two checks
therefore bracket the centroid's reliable-witness regime by phase: the
shape-lock check trusts the blob as a witness only below a low phase ceiling,
while the collapsed-regime check treats it as the haze-bias suspect only above
a high phase floor. Both apply only to single-body frames, where no companion
body corrupts the centroid and the whole-frame verdict maps to one body. Each
disagreement is a Mahalanobis gate under the combined covariance with a
body-scaled pixel floor as its lower bound, so a low-SNR centroid whose scatter
is statistically consistent with the geometric position is not vetoed.
"""

from __future__ import annotations

import math
from enum import Enum, auto

import numpy as np

from spindoctor.nav_orchestrator.ensemble_observability import mahalanobis_distance
from spindoctor.nav_technique.diagnostics import BodyBlobDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.types import NDArrayFloatType

__all__ = ['BodyWitnessVeto', 'evaluate_body_witness_veto']

_BLOB_TECHNIQUE = 'BodyBlobNav'
_GEOMETRIC_BODY_TECHNIQUES = frozenset({'BodyDiscCorrelateNav', 'BodyLimbNav'})


def _blob_diagnostics(result: NavTechniqueResult) -> BodyBlobDiagnostics:
    """Return the ``BodyBlobDiagnostics`` a ``BodyBlobNav`` result must carry.

    The veto only ever reads diagnostics off results it has already filtered to
    :data:`_BLOB_TECHNIQUE`, so a blob result whose diagnostics are absent or of
    the wrong type is a broken upstream contract, not a case to paper over with
    defaults: silently reading a zero phase and zero extent would make a
    malformed result look like a low-phase, zero-size witness and could bypass
    the collapsed-regime veto or spuriously trip shape-lock detection.

    Parameters:
        result: A result whose ``technique_name`` is already ``BodyBlobNav``.

    Returns:
        The result's ``BodyBlobDiagnostics``.

    Raises:
        NavContractError: if the result carries no ``BodyBlobDiagnostics``.
    """
    diagnostics = result.diagnostics
    if not isinstance(diagnostics, BodyBlobDiagnostics):
        raise NavContractError(
            f'{_BLOB_TECHNIQUE} result must carry BodyBlobDiagnostics, '
            f'got {type(diagnostics).__name__}'
        )
    return diagnostics


def _translation_block(covariance_px2: NDArrayFloatType) -> NDArrayFloatType:
    """Return the ``(2, 2)`` translation block of a 2- or 3-DoF covariance."""
    cov = np.asarray(covariance_px2, np.float64)
    return np.ascontiguousarray(cov[:2, :2])


def _significant_disagreement(
    mu_a: tuple[float, float],
    cov_a: NDArrayFloatType,
    mu_b: tuple[float, float],
    cov_b: NDArrayFloatType,
    *,
    floor_px: float,
    frac: float,
    extent_px: float,
    agreement_sigma: float,
    rcond: float,
) -> bool:
    """Return whether two body offsets disagree beyond noise.

    The gate is two-sided so a noisy estimate cannot force a veto on scatter
    alone: the Euclidean separation must clear the body-scaled pixel floor
    ``max(floor_px, frac * extent_px)`` -- a lower bound that never vetoes a
    sub-pixel disagreement however tight the covariances -- *and* the
    Mahalanobis distance under the combined translation covariance must exceed
    ``agreement_sigma``, so a low-SNR or small-body centroid whose large sigma
    makes the separation statistically consistent with agreement is not vetoed.
    """
    euclidean_px = math.hypot(mu_a[0] - mu_b[0], mu_a[1] - mu_b[1])
    if euclidean_px <= max(floor_px, frac * extent_px):
        return False
    distance = mahalanobis_distance(
        np.asarray(mu_a, np.float64),
        _translation_block(cov_a),
        np.asarray(mu_b, np.float64),
        _translation_block(cov_b),
        rcond=rcond,
    )
    return distance > agreement_sigma


class BodyWitnessVeto(Enum):
    """Verdict of :func:`evaluate_body_witness_veto`.

    - ``NONE``: no body-witness veto fires; the consensus stands.
    - ``SHAPE_LOCK_SUSPECT``: a geometric body consensus is contradicted by the
      pose-free blob witness on the same well-lit body -- the offset is
      reported ``conflicted`` rather than a confident success.
    - ``LONE_BLOB_COLLAPSED_REGIME``: the consensus rests solely on a high-phase
      blob centroid that disagrees with the position of a sibling geometric
      technique on the same body that self-flagged spurious -- the centroid was
      haze-dragged, so the frame is declined (``failed``) rather than reported
      as a lone-blob success.
    """

    NONE = auto()
    SHAPE_LOCK_SUSPECT = auto()
    LONE_BLOB_COLLAPSED_REGIME = auto()


def _consensus_bodies(best_group: list[NavTechniqueResult]) -> frozenset[str]:
    """Return the union of source bodies over the winning consensus group."""
    bodies: set[str] = set()
    for result in best_group:
        bodies.update(result.source_bodies)
    return frozenset(bodies)


def _distinct_body_count(results: list[NavTechniqueResult]) -> int:
    """Return the number of distinct bodies any body technique reported on.

    Ring and star techniques leave ``source_bodies`` empty, so they do not
    count.  A frame with more than one body is a multi-body frame, where a
    companion body can corrupt a brightness centroid and the blob is not a
    trustworthy position witness.
    """
    bodies: set[str] = set()
    for result in results:
        bodies.update(result.source_bodies)
    return len(bodies)


def _lone_blob_collapsed(
    best_group: list[NavTechniqueResult],
    results: list[NavTechniqueResult],
    fused_offset_px: tuple[float, float],
    fused_covariance_px2: NDArrayFloatType,
    consensus_bodies: frozenset[str],
    *,
    min_phase_deg: float,
    disagreement_floor_px: float,
    disagreement_frac: float,
    agreement_sigma: float,
    rcond: float,
) -> bool:
    """Return whether the collapsed-regime lone-blob signature holds.

    The winning consensus rests solely on the blob centroid, at least one
    consensus blob is at the extreme phase (above ``min_phase_deg``) where a
    forward-scattering haze can drag the centroid, and a sibling geometric
    technique on a consensus body self-flagged spurious *at a position* that
    disagrees significantly (see :func:`_significant_disagreement`) with the
    fused blob offset -- the disc found the body while the haze dragged the
    centroid away.  The spurious sibling self-flagged for a fit-consistency
    reason, but its coarse silhouette peak still locates the body; the check
    fails safe, since a wrong witness position only declines a frame rather
    than reporting a bad offset.  A small or faint body whose spurious disc
    agrees with the blob (both located the same body) does not trip this, and
    neither does a moderate crescent whose blob remains accurate.
    """
    if not best_group:
        return False
    if any(result.technique_name != _BLOB_TECHNIQUE for result in best_group):
        return False
    high_phase_extents = [
        _blob_diagnostics(result).body_extent_px
        for result in best_group
        if _blob_diagnostics(result).max_phase_angle_deg > min_phase_deg
    ]
    if not high_phase_extents:
        return False
    extent_px = max(high_phase_extents)
    for result in results:
        if result.technique_name not in _GEOMETRIC_BODY_TECHNIQUES or not result.spurious:
            continue
        if not (result.source_bodies & consensus_bodies):
            continue
        if _significant_disagreement(
            fused_offset_px,
            fused_covariance_px2,
            result.offset_px,
            result.covariance_px2,
            floor_px=disagreement_floor_px,
            frac=disagreement_frac,
            extent_px=extent_px,
            agreement_sigma=agreement_sigma,
            rcond=rcond,
        ):
            return True
    return False


def _shape_lock_suspect(
    best_group: list[NavTechniqueResult],
    results: list[NavTechniqueResult],
    fused_offset_px: tuple[float, float],
    fused_covariance_px2: NDArrayFloatType,
    consensus_bodies: frozenset[str],
    *,
    max_phase_deg: float,
    disagreement_floor_px: float,
    disagreement_frac: float,
    agreement_sigma: float,
    rcond: float,
) -> bool:
    """Return whether a geometric body lock is contradicted by the blob witness.

    Requires a geometric technique in the winning consensus and a non-spurious
    blob witness on a consensus body, well-lit (phase at most ``max_phase_deg``)
    so its centroid is a trustworthy position reference, whose offset disagrees
    significantly (see :func:`_significant_disagreement`) with the fused offset.
    The ceiling sits below half phase because the brightness centroid's
    lit-hemisphere bias grows with phase and would otherwise disagree with a
    correct geometric fit.
    """
    if not any(result.technique_name in _GEOMETRIC_BODY_TECHNIQUES for result in best_group):
        return False
    for result in results:
        if result.technique_name != _BLOB_TECHNIQUE or result.spurious:
            continue
        if not (result.source_bodies & consensus_bodies):
            continue
        diagnostics = _blob_diagnostics(result)
        if diagnostics.max_phase_angle_deg > max_phase_deg:
            continue
        body_extent_px = diagnostics.body_extent_px
        if _significant_disagreement(
            result.offset_px,
            result.covariance_px2,
            fused_offset_px,
            fused_covariance_px2,
            floor_px=disagreement_floor_px,
            frac=disagreement_frac,
            extent_px=body_extent_px,
            agreement_sigma=agreement_sigma,
            rcond=rcond,
        ):
            return True
    return False


def evaluate_body_witness_veto(
    best_group: list[NavTechniqueResult],
    results: list[NavTechniqueResult],
    fused_offset_px: tuple[float, float],
    fused_covariance_px2: NDArrayFloatType,
    *,
    shape_lock_max_phase_deg: float,
    collapse_min_phase_deg: float,
    disagreement_floor_px: float,
    disagreement_frac: float,
    agreement_sigma: float,
    rcond: float,
) -> BodyWitnessVeto:
    """Decide whether a body consensus is a confident-wrong lock to veto.

    The two checks bracket the brightness centroid's reliable-witness regime.
    The centroid tracks the geometric center at low phase, but its
    lit-hemisphere bias grows toward half phase and is only re-modeled once the
    coarse acquisition switches to the crescent template past it, so the
    shape-lock check trusts the blob only below ``shape_lock_max_phase_deg``
    while the collapsed-regime check treats it as the bias-prone suspect only
    above ``collapse_min_phase_deg``.  Both checks apply only to single-body
    frames: a companion body corrupts the brightness centroid, so on a
    multi-body frame the blob is neither a trustworthy witness nor a reliably
    attributable suspect, and the whole-frame verdict would mis-attribute one
    body's outcome to another.

    Parameters:
        best_group: The winning consensus group the ensemble is about to fuse
            and report.
        results: Every per-technique result on the frame, including spurious
            and fuse-superseded ones -- the full picture the veto reads.
        fused_offset_px: The consensus fused offset ``(dv, du)``.
        fused_covariance_px2: The consensus fused covariance; its translation
            block enters the Mahalanobis disagreement gate.
        shape_lock_max_phase_deg: Phase ceiling (degrees) below which the blob
            centroid is a trustworthy position witness for the shape-lock check.
        collapse_min_phase_deg: Phase floor (degrees) above which a lone blob is
            the haze-bias-prone suspect for the collapsed-regime check.
        disagreement_floor_px: Absolute floor of the cross-technique
            disagreement tolerance (a lower bound on the gate).
        disagreement_frac: Fraction of the body diameter added to the floor as
            the cross-technique disagreement lower bound.
        agreement_sigma: Mahalanobis-distance threshold the disagreement must
            exceed under the combined covariance, so a large-sigma centroid
            whose separation is statistically consistent is not vetoed.
        rcond: rcond for the Mahalanobis pseudo-inverse.

    Returns:
        The :class:`BodyWitnessVeto` verdict; ``NONE`` when no guard fires.
    """
    consensus_bodies = _consensus_bodies(best_group)
    if not consensus_bodies:
        return BodyWitnessVeto.NONE
    # The brightness centroid is trustworthy only on a single-body frame, where
    # no companion corrupts it and the whole-frame verdict maps to one body.
    if _distinct_body_count(results) != 1:
        return BodyWitnessVeto.NONE
    if _lone_blob_collapsed(
        best_group,
        results,
        fused_offset_px,
        fused_covariance_px2,
        consensus_bodies,
        min_phase_deg=collapse_min_phase_deg,
        disagreement_floor_px=disagreement_floor_px,
        disagreement_frac=disagreement_frac,
        agreement_sigma=agreement_sigma,
        rcond=rcond,
    ):
        return BodyWitnessVeto.LONE_BLOB_COLLAPSED_REGIME
    if _shape_lock_suspect(
        best_group,
        results,
        fused_offset_px,
        fused_covariance_px2,
        consensus_bodies,
        max_phase_deg=shape_lock_max_phase_deg,
        disagreement_floor_px=disagreement_floor_px,
        disagreement_frac=disagreement_frac,
        agreement_sigma=agreement_sigma,
        rcond=rcond,
    ):
        return BodyWitnessVeto.SHAPE_LOCK_SUSPECT
    return BodyWitnessVeto.NONE
