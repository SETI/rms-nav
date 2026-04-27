"""Ensemble — reconcile per-technique results into a single NavResult.

The ensemble is the single point in the pipeline where multiple per-technique
estimates become one offset.  Every step is honest:

1. Drop ``spurious=True`` results.
2. Drop ``at_edge=True`` results unless dropping them would empty the set.
3. Group remaining results by Mahalanobis-distance agreement (single-link).
4. Pick the highest summed-confidence group.
5. Combine offsets within that group via precision-weighted (Kalman-style)
   merging.
6. Apply optional disagreement / conflict penalties.
7. Emit a NavResult.

The ensemble is tested in isolation against synthetic per-technique results;
correctness here is what makes the rest of the pipeline trustworthy.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.linalg import pinvh
from scipy.sparse.csgraph import connected_components

from nav.annotation import Annotations
from nav.feature.constants import (
    AGREEMENT_FACTOR_CAP,
    COMBINED_CONFIDENCE_CAP,
)
from nav.nav_orchestrator.feature_summary import NavFeatureSummary
from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.nav_result import ConfidenceRank, NavResult
from nav.nav_orchestrator.provenance import Provenance
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.status_reason import NavStatusReason
from nav.support.types import NDArrayFloatType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'EnsembleConfig',
    'derive_confidence_rank',
    'ensemble',
]


# Default constants used by the ensemble.  Configurable via ``EnsembleConfig``.
DEFAULT_AGREEMENT_SIGMA = 2.0
DEFAULT_AGREEMENT_GAP = 0.5
DEFAULT_DISAGREEMENT_PENALTY = 0.7
DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER = 0.3
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_PINVH_RCOND = 1.0e-9
DEFAULT_TIER_THRESHOLDS: dict[str, dict[str, float | None]] = {
    'high': {'min_confidence': 0.8, 'max_sigma_px': 0.5},
    'medium': {'min_confidence': 0.5, 'max_sigma_px': 2.0},
    'low': {'min_confidence': 0.2, 'max_sigma_px': None},
}


@dataclass(frozen=True)
class EnsembleConfig:
    """Tunable parameters of the ensemble combine.

    Defaults match ``config_540_orchestrator.yaml``.

    Parameters:
        agreement_sigma: Mahalanobis-distance threshold for grouping.
        agreement_gap: Minimum summed-confidence gap between best and
            runner-up groups before declaring a conflict.
        disagreement_penalty: Multiplier on combined confidence when more
            than one group existed.
        conflicted_confidence_multiplier: Additional multiplier when the
            conflicted branch fires.
        min_confidence: Final-result threshold below which the ensemble
            returns NavResult.failed instead of NavResult.ok.
        pinvh_rcond: rcond for ``scipy.linalg.pinvh``.
        tier_thresholds: Mapping ``rank -> {min_confidence, max_sigma_px}``;
            see ``derive_confidence_rank``.
    """

    agreement_sigma: float = DEFAULT_AGREEMENT_SIGMA
    agreement_gap: float = DEFAULT_AGREEMENT_GAP
    disagreement_penalty: float = DEFAULT_DISAGREEMENT_PENALTY
    conflicted_confidence_multiplier: float = DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    pinvh_rcond: float = DEFAULT_PINVH_RCOND
    tier_thresholds: dict[str, dict[str, float | None]] = field(
        default_factory=lambda: dict(DEFAULT_TIER_THRESHOLDS)
    )


def _mahalanobis_distance(
    mu_a: NDArrayFloatType,
    cov_a: NDArrayFloatType,
    mu_b: NDArrayFloatType,
    cov_b: NDArrayFloatType,
    *,
    rcond: float,
) -> float:
    """Return the Mahalanobis distance between two estimates.

    Uses ``pinvh(cov_a + cov_b)`` so rank-deficient inputs are handled
    correctly.  Components of ``mu_a - mu_b`` in the null space of the
    summed covariance are treated as infinite distance — estimates cannot
    agree along an unobservable axis.
    """
    delta = mu_a - mu_b
    cov_sum = cov_a + cov_b
    pinv = pinvh(cov_sum, rtol=rcond)
    # Project delta back through pinv * cov_sum; residual lies in the null
    # space.
    null_proj = delta - cov_sum @ pinv @ delta
    if np.linalg.norm(null_proj) > 1e-6:
        return float('inf')
    d_sq = float(delta.T @ pinv @ delta)
    if d_sq < 0:
        # Numerical safety: pinv may yield a tiny negative quadratic form
        # due to floating-point; clamp to zero.
        d_sq = 0.0
    return float(math.sqrt(d_sq))


def _agreement_groups(
    results: list[NavTechniqueResult], *, agreement_sigma: float, rcond: float
) -> list[list[NavTechniqueResult]]:
    """Single-link clustering by Mahalanobis distance.

    Two results are placed in the same group iff their pairwise distance is
    below ``agreement_sigma``; transitive closure builds final groups via
    connected components.

    Parameters:
        results: Non-empty list of per-technique results.
        agreement_sigma: Maximum pairwise Mahalanobis distance for grouping.
        rcond: rcond passed to ``pinvh``.

    Returns:
        List of groups (each group a list of NavTechniqueResult).
    """
    n = len(results)
    if n == 0:
        return []
    if n == 1:
        return [list(results)]
    # Build a sparse adjacency matrix marking pairs within threshold.
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
    for i in range(n):
        mu_i = np.asarray(results[i].offset_px, np.float64)
        cov_i = np.asarray(results[i].covariance_px2, np.float64)
        for j in range(i + 1, n):
            mu_j = np.asarray(results[j].offset_px, np.float64)
            cov_j = np.asarray(results[j].covariance_px2, np.float64)
            dist = _mahalanobis_distance(mu_i, cov_i, mu_j, cov_j, rcond=rcond)
            if dist <= agreement_sigma:
                rows.extend([i, j])
                cols.extend([j, i])
    # Build dense adjacency.  N is small (typically < 10), so this is fine.
    adj = np.zeros((n, n), dtype=bool)
    adj[rows, cols] = True
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    groups: list[list[NavTechniqueResult]] = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        groups[int(label)].append(results[idx])
    return groups


def _combine_precision_weighted(
    group: list[NavTechniqueResult], *, rcond: float
) -> tuple[tuple[float, float], NDArrayFloatType, bool]:
    """Information-form combine of a group of agreeing results.

    Implements the Kalman-style information-form merge:

        Sigma_combined = pinvh( sum_i pinvh(Sigma_i) )
        mu_combined    = Sigma_combined @ sum_i ( pinvh(Sigma_i) @ mu_i )

    Parameters:
        group: Non-empty list of agreeing results.
        rcond: rcond for ``pinvh``.

    Returns:
        Tuple ``(offset_px, covariance_px2, is_rank_deficient)``.

    Raises:
        ValueError: if ``group`` is empty (defensive; the orchestrator must
            ensure non-emptiness before calling).
    """
    if not group:
        raise ValueError('empty group passed to _combine_precision_weighted')
    info_sum: NDArrayFloatType | None = None
    info_mu_sum: NDArrayFloatType | None = None
    for res in group:
        cov = np.asarray(res.covariance_px2, np.float64)
        info = pinvh(cov, rtol=rcond)
        mu = np.asarray(res.offset_px, np.float64)
        if info_sum is None:
            info_sum = info.copy()
            info_mu_sum = info @ mu
        else:
            info_sum = info_sum + info
            assert info_mu_sum is not None  # narrowed for mypy
            info_mu_sum = info_mu_sum + info @ mu
    assert info_sum is not None
    assert info_mu_sum is not None
    cov_combined = pinvh(info_sum, rtol=rcond)
    mu_combined = cov_combined @ info_mu_sum
    # Check rank-deficiency by comparing combined info matrix's smallest
    # eigenvalue to a small tolerance.
    eigvals = np.linalg.eigvalsh(info_sum)
    is_rank_deficient = bool(eigvals.min() < 1.0 / 1e8)
    return (
        (float(mu_combined[0]), float(mu_combined[1])),
        cast(NDArrayFloatType, cov_combined),
        is_rank_deficient,
    )


def _combine_confidence(
    group: list[NavTechniqueResult],
    *,
    rcond: float,
    disagreement_penalty: float,
    apply_disagreement_penalty: bool,
) -> float:
    """Precision-weighted combine of per-result confidences.

    Weights are ``trace(pinvh(Sigma_i))``: tighter covariances contribute
    more to the combined confidence than loose ones.  The boosted combined
    confidence reflects the number of significant contributors, capped per
    ``AGREEMENT_FACTOR_CAP`` and ``COMBINED_CONFIDENCE_CAP``.

    Parameters:
        group: Non-empty list of agreeing results.
        rcond: rcond for ``pinvh``.
        disagreement_penalty: Multiplier applied if other groups existed.
        apply_disagreement_penalty: True if more than one group existed
            before this combine.

    Returns:
        Combined confidence in ``[0, 1]``; never above
        ``COMBINED_CONFIDENCE_CAP``.

    Raises:
        ValueError: if every input covariance shares a null direction
            (W = 0); the orchestrator's caller routes this to
            ``unobservable_offset`` failure.
    """
    weights = []
    for res in group:
        cov = np.asarray(res.covariance_px2, np.float64)
        info = pinvh(cov, rtol=rcond)
        weights.append(float(np.trace(info)))
    w_total = sum(weights)
    if w_total <= 0.0:
        raise ValueError('precision-weighted combine: total weight is zero; offset is unobservable')
    weighted_avg = sum(w * r.confidence for w, r in zip(weights, group, strict=True))
    weighted_avg /= w_total
    significant_threshold = 0.1 * max(weights)
    n_significant = sum(1 for w in weights if w > significant_threshold)
    if n_significant <= 1:
        agreement_factor = 1.0
    else:
        agreement_factor = 1.0 + 0.5 * math.log2(n_significant)
    agreement_factor = min(agreement_factor, AGREEMENT_FACTOR_CAP)
    combined = min(weighted_avg * agreement_factor, COMBINED_CONFIDENCE_CAP)
    if apply_disagreement_penalty:
        combined *= disagreement_penalty
    return combined


def derive_confidence_rank(
    *,
    confidence: float,
    sigma_px: tuple[float, float] | None,
    tier_thresholds: dict[str, dict[str, float | None]] | None = None,
) -> ConfidenceRank:
    """Derive the five-bucket confidence rank from confidence + sigma.

    ``max_sigma_px`` compares ``max(sigma_dv, sigma_du)`` only.
    ``high`` / ``medium`` / ``low`` tiers require both confidence and sigma
    constraints; ``conflicted`` and ``failed`` are status-driven and not
    chosen here.

    Parameters:
        confidence: Combined confidence in ``[0, 1]``.
        sigma_px: Per-axis 1sigma marginal uncertainty (use ``None`` to mean
            "unknown / not applicable").
        tier_thresholds: Mapping ``rank -> {min_confidence, max_sigma_px}``
            with ``max_sigma_px`` allowed to be ``None``.

    Returns:
        ``'high'``, ``'medium'``, or ``'low'`` if any tier matches; else
        ``'failed'``.
    """
    thresholds = tier_thresholds or DEFAULT_TIER_THRESHOLDS
    max_sigma = max(sigma_px) if sigma_px is not None else None
    ranks: tuple[ConfidenceRank, ...] = ('high', 'medium', 'low')
    for rank in ranks:
        spec = thresholds[rank]
        min_conf = spec['min_confidence']
        max_allowed = spec['max_sigma_px']
        assert min_conf is not None  # min_confidence is always set
        if confidence < min_conf:
            continue
        if max_allowed is not None and (max_sigma is None or max_sigma > max_allowed):
            continue
        return rank
    return 'failed'


def ensemble(
    results: list[NavTechniqueResult],
    *,
    feature_inventory: list[NavFeatureSummary],
    image_classifier: NavImageClassifierResult,
    provenance: Provenance,
    config: EnsembleConfig | None = None,
    model_metadata: dict[str, dict[str, Any]] | None = None,
    annotations: Annotations | None = None,
) -> NavResult:
    """Reconcile per-technique results into a single NavResult.

    Parameters:
        results: Per-technique results from one or both passes.
        feature_inventory: Feature inventory (kept + gated entries).
        image_classifier: Image-quality classifier verdict.
        provenance: Reproducibility envelope.
        config: Optional ``EnsembleConfig`` overrides.
        model_metadata: Optional per-NavModel diagnostic dict map.
        annotations: Optional pre-built annotation collection from the
            orchestrator's ``_collect_annotations`` pass.

    Returns:
        A single NavResult — ok / conflicted / failed.
    """
    cfg = config or EnsembleConfig()
    md = model_metadata if model_metadata is not None else {}
    ann = annotations if annotations is not None else Annotations()
    if not results:
        return NavResult.failed(
            status_reason=NavStatusReason.NO_FEASIBLE_TECHNIQUES,
            image_classifier=image_classifier,
            provenance=provenance,
            model_metadata=md,
            annotations=ann,
        )
    viable = [r for r in results if not r.spurious]
    if not viable:
        return NavResult.failed(
            status_reason=NavStatusReason.ALL_TECHNIQUES_SPURIOUS,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    interior = [r for r in viable if not r.at_edge]
    if interior:
        viable = interior
    groups = _agreement_groups(
        viable,
        agreement_sigma=cfg.agreement_sigma,
        rcond=cfg.pinvh_rcond,
    )
    ranked = sorted(
        groups,
        key=lambda g: sum(r.confidence for r in g),
        reverse=True,
    )
    best_group = ranked[0]
    best_summed_conf = sum(r.confidence for r in best_group)
    apply_disagreement_penalty = len(groups) > 1
    try:
        offset, cov, is_rank_deficient = _combine_precision_weighted(
            best_group, rcond=cfg.pinvh_rcond
        )
        combined_confidence = _combine_confidence(
            best_group,
            rcond=cfg.pinvh_rcond,
            disagreement_penalty=cfg.disagreement_penalty,
            apply_disagreement_penalty=apply_disagreement_penalty,
        )
    except ValueError:
        # Total weight zero — offset unobservable in every contributing
        # input.
        return NavResult.failed(
            status_reason=NavStatusReason.UNOBSERVABLE_OFFSET,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    # Conflict check: best-vs-runner-up summed-confidence gap.
    if len(ranked) > 1:
        runner_up_summed_conf = sum(r.confidence for r in ranked[1])
        gap = best_summed_conf - runner_up_summed_conf
        if gap < cfg.agreement_gap:
            conflicted_confidence = combined_confidence * cfg.conflicted_confidence_multiplier
            return NavResult.conflicted(
                offset_px=offset,
                covariance_px2=cov,
                confidence=conflicted_confidence,
                per_technique=results,
                feature_inventory=feature_inventory,
                image_classifier=image_classifier,
                provenance=provenance,
                model_metadata=md,
                annotations=ann,
            )
    if combined_confidence < cfg.min_confidence:
        return NavResult.failed(
            status_reason=NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    sigma_dv = float(math.sqrt(max(cov[0, 0], 0.0)))
    sigma_du = float(math.sqrt(max(cov[1, 1], 0.0)))
    sigma_along_unobservable_px = float('inf') if is_rank_deficient else None
    rank = derive_confidence_rank(
        confidence=combined_confidence,
        sigma_px=(sigma_dv, sigma_du),
        tier_thresholds=cfg.tier_thresholds,
    )
    if rank == 'failed':
        # Confidence + sigma combination doesn't earn any tier.
        return NavResult.failed(
            status_reason=NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    status_reason = NavStatusReason.RANK_1_ONLY if is_rank_deficient else NavStatusReason.OK
    return NavResult.ok(
        offset_px=offset,
        covariance_px2=cov,
        confidence=combined_confidence,
        confidence_rank=rank,
        status_reason=status_reason,
        per_technique=results,
        feature_inventory=feature_inventory,
        image_classifier=image_classifier,
        provenance=provenance,
        sigma_along_unobservable_px=sigma_along_unobservable_px,
        model_metadata=md,
        annotations=ann,
    )
