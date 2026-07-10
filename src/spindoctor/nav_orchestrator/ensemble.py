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

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, cast

import numpy as np
from scipy.linalg import pinvh
from scipy.sparse.csgraph import connected_components

from spindoctor.annotation import Annotations
from spindoctor.config import IMAGE_LOGGER
from spindoctor.feature.constants import (
    AGREEMENT_FACTOR_CAP,
    COMBINED_CONFIDENCE_CAP,
)
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import ConfidenceRank, NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.nav_technique import technique_tier
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.exceptions import NavContractError
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'EnsembleConfig',
    'derive_confidence_rank',
    'ensemble',
]


# Default constants used by the ensemble.  Configurable via ``EnsembleConfig``.
DEFAULT_AGREEMENT_SIGMA = 2.0
DEFAULT_AGREEMENT_PIXEL_FLOOR = 5.0
DEFAULT_AGREEMENT_GAP = 0.5
DEFAULT_DISAGREEMENT_PENALTY = 0.7
DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER = 0.3
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_PINVH_RCOND = 1.0e-9
DEFAULT_MAX_ALLOWED_ROTATION_DEG = 5.0
# Tier confidence boundaries are sim-anchored (WS-5, 2026-07-10, with the
# #210 covariance floors live): the smallest confidence at which each
# tier's sigma-gated subset of the WS-2 campaign's fused results achieves
# a 0.9 success rate against the tier's error budget.  With honest NCC
# sigmas the medium tier's sigma gate carries the discrimination by
# itself, so its confidence boundary rests at the same 0.2 floor as the
# low tier (the tiers are sigma-differentiated).  Mirrored in
# config_540_orchestrator.yaml, which carries the full provenance note.
DEFAULT_TIER_THRESHOLDS: dict[str, dict[str, float | None]] = {
    'high': {'min_confidence': 0.5, 'max_sigma_px': 0.5},
    'medium': {'min_confidence': 0.2, 'max_sigma_px': 2.0},
    'low': {'min_confidence': 0.2, 'max_sigma_px': None},
}


@dataclass(frozen=True)
class EnsembleConfig:
    """Tunable parameters of the ensemble combine.

    Defaults match ``config_540_orchestrator.yaml``.

    Parameters:
        agreement_sigma: Mahalanobis-distance threshold for grouping.
        agreement_pixel_floor: Translation-distance fallback grouping
            threshold in pixels.  Two results are grouped when *either*
            their Mahalanobis distance is at most ``agreement_sigma``
            *or* their Euclidean translation distance is at most this
            many pixels.  The pixel floor exists because per-technique
            covariances are CRLB-tight (FFT subpixel localization for
            ``BodyDiscCorrelateNav``, M-estimator information for the
            DT-fit techniques), well below the actual position
            uncertainty driven by model error and pointing residuals;
            without the floor, results agreeing visually to a few px
            register as hundreds of sigmas apart and never group.
            Set to ``0.0`` to disable the floor.
        agreement_gap: Minimum summed-confidence gap between best and
            runner-up groups before declaring a conflict.
        disagreement_penalty: Multiplier on combined confidence when more
            than one group existed.
        conflicted_confidence_multiplier: Additional multiplier when the
            conflicted branch fires.
        min_confidence: Final-result threshold below which the ensemble
            returns NavResult.failed instead of NavResult.ok.
        pinvh_rcond: rcond for ``scipy.linalg.pinvh``.
        max_allowed_rotation_deg: Maximum magnitude (in degrees) a 3-DoF
            result's rotation may take before the ensemble rejects it.
            The rotation parameter is combined as a small angle (circular
            mean of ``(dv, du, theta)`` triples); this bound enforces the
            small-angle assumption that every contributing technique fits
            against (every DT/star technique clamps its rotation to
            ``+-max_rotation_deg``, default 5 deg).  A 3-DoF result
            arriving with ``abs(rotation_rad)`` at or above this bound is a
            programming error upstream and raises ``NavContractError``.
        tier_thresholds: Mapping ``rank -> {min_confidence, max_sigma_px}``;
            see ``derive_confidence_rank``.
    """

    agreement_sigma: float = DEFAULT_AGREEMENT_SIGMA
    agreement_pixel_floor: float = DEFAULT_AGREEMENT_PIXEL_FLOOR
    agreement_gap: float = DEFAULT_AGREEMENT_GAP
    disagreement_penalty: float = DEFAULT_DISAGREEMENT_PENALTY
    conflicted_confidence_multiplier: float = DEFAULT_CONFLICTED_CONFIDENCE_MULTIPLIER
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    pinvh_rcond: float = DEFAULT_PINVH_RCOND
    max_allowed_rotation_deg: float = DEFAULT_MAX_ALLOWED_ROTATION_DEG
    tier_thresholds: dict[str, dict[str, float | None]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_TIER_THRESHOLDS)
    )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> 'EnsembleConfig':
        """Build an ``EnsembleConfig`` from a YAML-derived mapping.

        Parameters:
            mapping: The ``orchestrator.ensemble`` config section.  ``None``
                or empty returns the code defaults.  Scalar keys override
                their field; a ``tier_thresholds`` block is merged tier-by-
                tier over the defaults, so a partial override (for example
                only ``low.min_confidence``) leaves the other tiers intact.

        Returns:
            An ``EnsembleConfig`` with mapping values overriding the code
            defaults field-by-field.

        Raises:
            ValueError: If the mapping contains an unknown key, an unknown
                tier name, or an unknown tier-threshold key.
        """
        if not mapping:
            return cls()
        data = dict(mapping)
        tiers_raw = data.pop('tier_thresholds', None)
        scalar_fields = {f.name for f in fields(cls)} - {'tier_thresholds'}
        unknown = set(data) - scalar_fields
        if unknown:
            raise ValueError(f'Unknown orchestrator.ensemble config keys: {sorted(unknown)}')
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            try:
                kwargs[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'orchestrator.ensemble.{key} must be a number; got {value!r}'
                ) from exc
        if tiers_raw is not None:
            if not isinstance(tiers_raw, Mapping):
                raise ValueError('orchestrator.ensemble.tier_thresholds must be a mapping')
            tiers = copy.deepcopy(DEFAULT_TIER_THRESHOLDS)
            for tier, spec in tiers_raw.items():
                if tier not in tiers:
                    raise ValueError(
                        f'Unknown tier {tier!r} in tier_thresholds; valid tiers: {sorted(tiers)}'
                    )
                if not isinstance(spec, Mapping):
                    raise ValueError(f'tier_thresholds[{tier!r}] must be a mapping')
                for key, value in spec.items():
                    if key not in ('min_confidence', 'max_sigma_px'):
                        raise ValueError(f'Unknown tier_thresholds[{tier!r}] key {key!r}')
                    try:
                        tiers[tier][key] = None if value is None else float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f'orchestrator.ensemble.tier_thresholds[{tier!r}].{key} '
                            f'must be a number or null; got {value!r}'
                        ) from exc
            kwargs['tier_thresholds'] = tiers
        return cls(**kwargs)


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
    # Scale-invariant null-space disagreement test: the residual that lies in
    # the null space of the summed covariance must be negligible relative to
    # the size of ``delta`` itself, not against a fixed absolute pixel
    # tolerance (which silently flips behavior as image/offset scales change).
    rel_tol = 1.0e-6
    eps = np.finfo(float).eps
    if np.linalg.norm(null_proj) > rel_tol * max(float(np.linalg.norm(delta)), eps):
        return float('inf')
    d_sq = float(delta.T @ pinv @ delta)
    if d_sq < 0:
        # Numerical safety: pinv may yield a tiny negative quadratic form
        # due to floating-point; clamp to zero.
        d_sq = 0.0
    return float(math.sqrt(d_sq))


def _result_param_vector(res: NavTechniqueResult) -> NDArrayFloatType:
    """Return the parameter vector for a per-technique result.

    Two-DoF results emit ``(dv, du)``; 3-DoF results emit
    ``(dv, du, rotation_rad)``.  The vector length always matches the
    covariance shape — a 3x3 covariance with ``rotation_rad=None`` would be
    inconsistent and raises here.
    """
    cov = np.asarray(res.covariance_px2, np.float64)
    if cov.shape == (3, 3):
        if res.rotation_rad is None:
            raise ValueError(
                f'{res.technique_name}: 3x3 covariance requires rotation_rad to be set'
            )
        return cast(
            NDArrayFloatType,
            np.array([res.offset_px[0], res.offset_px[1], res.rotation_rad], np.float64),
        )
    return cast(NDArrayFloatType, np.array([res.offset_px[0], res.offset_px[1]], np.float64))


def _source_bodies(result: NavTechniqueResult) -> frozenset[str]:
    """Return the set of body names a result was computed against.

    Reads the structured ``NavTechniqueResult.source_bodies`` populated by
    the body-feature techniques from each consumed feature's
    ``NavFeature.body_name``.  Ring and star techniques leave it empty, so
    they are naturally excluded from the body-only fallback-supersession
    filter.  This replaces the earlier ``feature_ids`` string-parsing, which
    silently broke if the feature-id format changed.
    """
    return result.source_bodies


def _drop_superseded_fallbacks(
    results: list[NavTechniqueResult],
) -> list[NavTechniqueResult]:
    """Drop fallback-tier results superseded by a non-spurious primary.

    A fallback result is superseded when at least one of its source
    bodies appears in the source-body set of any non-spurious primary
    result.  Fallback results whose source body has no primary
    coverage stay in the set so a scene with only a fallback (e.g.,
    a body too small for limb fitting, where only BodyBlob ran) still
    produces an offset.

    Returns a new list preserving input ordering; the input list is
    not mutated.
    """
    primary_success_bodies: set[str] = set()
    for r in results:
        if r.spurious:
            continue
        if technique_tier(r.technique_name) != 'primary':
            continue
        primary_success_bodies.update(_source_bodies(r))
    if not primary_success_bodies:
        return list(results)
    kept: list[NavTechniqueResult] = []
    for r in results:
        if technique_tier(r.technique_name) == 'fallback':
            superseded = bool(_source_bodies(r) & primary_success_bodies)
            if superseded:
                IMAGE_LOGGER.info(
                    'Dropping fallback %s for body %s: superseded by a '
                    'non-spurious primary result on the same body',
                    r.technique_name,
                    ', '.join(sorted(_source_bodies(r))) or '(unknown)',
                )
                continue
        kept.append(r)
    return kept


def _agreement_groups(
    results: list[NavTechniqueResult],
    *,
    agreement_sigma: float,
    agreement_pixel_floor: float,
    rcond: float,
    max_allowed_rotation_deg: float,
) -> list[list[NavTechniqueResult]]:
    """Single-link clustering by Mahalanobis distance with a pixel-floor fallback.

    Two results are placed in the same group iff *either* their pairwise
    Mahalanobis distance is below ``agreement_sigma`` *or* their
    Euclidean translation distance (in pixels, ignoring the rotation
    component for 3-DoF results) is at most ``agreement_pixel_floor``.
    Transitive closure builds final groups via connected components.

    The Mahalanobis distance differences the rotation component of two
    3-DoF results linearly; that is correct here because every input
    rotation is bounded to the small-angle window enforced below, so the
    pairwise angle difference never approaches the ``+-pi`` wrap.

    The pixel floor compensates for per-technique covariances that
    report only a CRLB-style precision (FFT subpixel localization or
    M-estimator information) and so under-estimate the actual position
    uncertainty by orders of magnitude.  See ``EnsembleConfig`` for the
    motivation.

    Parameters:
        results: Non-empty list of per-technique results.
        agreement_sigma: Maximum pairwise Mahalanobis distance for grouping.
        agreement_pixel_floor: Maximum pairwise Euclidean translation
            distance (in pixels) for grouping; ``0.0`` disables.
        rcond: rcond passed to ``pinvh``.
        max_allowed_rotation_deg: Small-angle bound; each 3-DoF input's
            ``rotation_rad`` magnitude must stay strictly below this many
            degrees for the linear rotation differencing to be valid.

    Returns:
        List of groups (each group a list of NavTechniqueResult).

    Raises:
        NavContractError: if a 3-DoF input's rotation magnitude is at or
            above the small-angle bound (an upstream programming error).
    """
    n = len(results)
    if n == 0:
        return []
    max_rotation_rad = math.radians(max_allowed_rotation_deg)
    for res in results:
        cov = np.asarray(res.covariance_px2, np.float64)
        # Small-angle assumption: a 3-DoF rotation outside +-max_rotation_deg
        # is an upstream error (every technique clamps its rotation fit to
        # this bound), and would break the linear rotation differencing.
        # Raised as NavContractError (not a bare assert) so the check
        # survives python -O and is never swallowed as an ordinary
        # technique failure.
        if (
            cov.shape == (3, 3)
            and res.rotation_rad is not None
            and abs(res.rotation_rad) >= max_rotation_rad
        ):
            raise NavContractError(
                f'{res.technique_name}: rotation '
                f'{math.degrees(res.rotation_rad):.3f} deg violates small-angle '
                f'bound +-{max_allowed_rotation_deg:.3f} deg'
            )
    if n == 1:
        return [list(results)]
    # Build a sparse adjacency matrix marking pairs within threshold.
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
    for i in range(n):
        mu_i = _result_param_vector(results[i])
        cov_i = np.asarray(results[i].covariance_px2, np.float64)
        for j in range(i + 1, n):
            mu_j = _result_param_vector(results[j])
            cov_j = np.asarray(results[j].covariance_px2, np.float64)
            if mu_i.shape != mu_j.shape:
                # 2-DoF and 3-DoF results never coexist within one image.
                # If the orchestrator routed mismatched shapes through the
                # ensemble, fail loudly rather than coerce.
                raise ValueError(
                    'Mixed-DoF technique results in one ensemble: '
                    f'{results[i].technique_name} produced {mu_i.shape} parameters; '
                    f'{results[j].technique_name} produced {mu_j.shape}.'
                )
            dist = _mahalanobis_distance(mu_i, cov_i, mu_j, cov_j, rcond=rcond)
            mahal_match = dist <= agreement_sigma
            translation_delta = mu_i[:2] - mu_j[:2]
            pixel_dist = float(math.hypot(translation_delta[0], translation_delta[1]))
            pixel_match = agreement_pixel_floor > 0.0 and pixel_dist <= agreement_pixel_floor
            if mahal_match or pixel_match:
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


@dataclass(frozen=True)
class _CombinedEstimate:
    """Output of :func:`_combine_precision_weighted`.

    Parameters:
        offset_px: Combined ``(dv, du)`` translation.
        rotation_rad: Combined rotation in radians; ``None`` when every
            input was 2-DoF.
        covariance_px2: Combined covariance, ``(2, 2)`` or ``(3, 3)``
            matching the input parameter dimensionality.
        is_rank_deficient: True when the summed information matrix has a
            near-zero eigenvalue relative to its largest.
    """

    offset_px: tuple[float, float]
    rotation_rad: float | None
    covariance_px2: NDArrayFloatType
    is_rank_deficient: bool


def _combine_precision_weighted(
    group: list[NavTechniqueResult], *, rcond: float, max_allowed_rotation_deg: float
) -> _CombinedEstimate:
    """Information-form combine of a group of agreeing results.

    Implements the Kalman-style information-form merge for the translation
    components:

        Sigma_combined = pinvh( sum_i pinvh(Sigma_i) )
        mu_combined    = Sigma_combined @ sum_i ( pinvh(Sigma_i) @ mu_i )

    Parameter vectors carry rotation as a third component when every input
    is 3-DoF; the resulting combined estimate then carries a non-``None``
    ``rotation_rad`` field.  The rotation parameter is *not* averaged as a
    plain Euclidean coordinate (which would wrap incorrectly near
    ``+-pi``); instead it is combined on the circle as the precision-weighted
    circular mean ``atan2(sum_i w_i sin theta_i, sum_i w_i cos theta_i)``,
    with ``w_i`` the rotation-component information ``pinvh(Sigma_i)[2, 2]``.
    The translation components and the full combined covariance are produced
    exactly as before by the information-form merge.

    Parameters:
        group: Non-empty list of agreeing results.  Every member must
            share the same parameter dimensionality (all 2-DoF or all
            3-DoF — :func:`_agreement_groups` already enforces this).
        rcond: rcond for ``pinvh``.
        max_allowed_rotation_deg: Small-angle bound; each 3-DoF input's
            ``rotation_rad`` magnitude must stay strictly below this many
            degrees (in radians) for the circular-mean combine to be valid.

    Returns:
        :class:`_CombinedEstimate`.

    Raises:
        ValueError: if ``group`` is empty (defensive; the orchestrator must
            ensure non-emptiness before calling), if total weight is zero,
            or if the group contains a mix of 2-DoF and 3-DoF results.
        NavContractError: if a 3-DoF input's rotation magnitude is at or
            above the small-angle bound (an upstream programming error).
    """
    if not group:
        raise ValueError('empty group passed to _combine_precision_weighted')
    info_sum: NDArrayFloatType | None = None
    info_mu_sum: NDArrayFloatType | None = None
    n_params: int | None = None
    max_rotation_rad = math.radians(max_allowed_rotation_deg)
    rot_w_sin = 0.0
    rot_w_cos = 0.0
    for res in group:
        cov = np.asarray(res.covariance_px2, np.float64)
        if n_params is None:
            n_params = cov.shape[0]
        elif cov.shape[0] != n_params:
            raise ValueError(
                f'mixed-DoF group passed to _combine_precision_weighted: '
                f'expected {n_params}-DoF, got {cov.shape[0]} from {res.technique_name}'
            )
        info = pinvh(cov, rtol=rcond)
        mu = _result_param_vector(res)
        if cov.shape[0] == 3:
            theta = float(mu[2])
            # Small-angle assumption: every contributing technique clamps its
            # rotation fit to +-max_rotation_deg, so a result arriving outside
            # that bound is an upstream programming error, not data the
            # circular-mean combine should silently absorb.  Raised as
            # NavContractError (not a bare assert) so the check survives
            # python -O and is never swallowed as an ordinary technique
            # failure.
            if abs(theta) >= max_rotation_rad:
                raise NavContractError(
                    f'{res.technique_name}: rotation {math.degrees(theta):.3f} deg '
                    f'violates small-angle bound +-{max_allowed_rotation_deg:.3f} deg'
                )
            # Rotation weight: the scalar rotation-information element
            # ``info[2, 2]`` only.  The circular mean needs one scalar weight
            # per angle, so the rotation-translation cross-information
            # ``info[:2, 2]`` (which the information-form translation merge
            # below does retain) cannot enter this average.  Dropping it
            # approximates each input's rotation as independent of its
            # translation, i.e. it uses the conditional rotation information
            # instead of the marginal ``1 / Sigma[2, 2]``; under the
            # small-angle clamp enforced above the rotations differ by at
            # most a few degrees, so the reweighting error this introduces
            # in the combined angle is far below the reported rotation
            # sigma.  The combined covariance itself is exact: it comes from
            # the full information-form merge, cross-terms included.
            w_theta = float(info[2, 2])
            rot_w_sin += w_theta * math.sin(theta)
            rot_w_cos += w_theta * math.cos(theta)
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
    # Scale-independent rank-deficiency check: the smallest eigenvalue is
    # tiny relative to the largest.  The eps guards divide-by-zero on a
    # zero info matrix (which itself is degenerate).
    rel_tol = 1.0e-8
    eps = np.finfo(np.float64).eps
    is_rank_deficient = bool(eigvals.min() / max(abs(eigvals.max()), eps) < rel_tol)
    rotation: float | None = None
    if n_params == 3:
        # Combine the rotation on the circle so angles near +-pi do not cancel
        # to a spurious ~0; fall back to the information-form estimate when the
        # rotation weight is degenerate (every input rotation-unobservable).
        if rot_w_sin == 0.0 and rot_w_cos == 0.0:
            rotation = float(mu_combined[2])
        else:
            rotation = float(math.atan2(rot_w_sin, rot_w_cos))
    return _CombinedEstimate(
        offset_px=(float(mu_combined[0]), float(mu_combined[1])),
        rotation_rad=rotation,
        covariance_px2=cast(NDArrayFloatType, cov_combined),
        is_rank_deficient=is_rank_deficient,
    )


def _combine_confidence(
    group: list[NavTechniqueResult],
    *,
    rcond: float,
    disagreement_penalty: float,
    apply_disagreement_penalty: bool,
) -> float:
    """Precision-weighted combine of per-result confidences.

    Weights are ``trace(pinvh(Sigma_i[:2, :2]))`` -- the positional (v, u)
    precision in ``px^-2``, with any rotation axis marginalised out so the
    weight is not skewed by the unrelated ``rad^-2`` rotation precision of a
    3-DoF result.  Tighter covariances contribute more to the combined
    confidence than loose ones.  The boosted combined
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
        # Weight by the *positional* precision only.  Taking trace(pinvh(cov))
        # over a full 3-DoF covariance would add the v, u precisions (px^-2) to
        # the rotation precision (rad^-2), so a star-field result whose rotation
        # is tightly pinned would dominate the confidence average on an
        # arbitrary, unit-mixed scale.  Marginalising rotation out and tracing
        # the 2x2 translation block keeps every weight in px^-2; the additive
        # trace (rather than det(info)^(1/p)) stays well-defined for the rank-1
        # ring-edge covariances whose unobservable axis carries zero precision.
        # The 2-DoF path is unchanged: cov[:2, :2] is then the whole matrix.
        info_xy = pinvh(np.ascontiguousarray(cov[:2, :2]), rtol=rcond)
        weights.append(float(np.trace(info_xy)))
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
    if max_sigma is None:
        # No covariance was supplied, so every sigma-constrained tier
        # (high / medium) is unreachable and the result can only earn the
        # best sigma-free tier (low).  Surface this: a missing covariance is
        # almost always an upstream technique failing to populate it, not a
        # legitimately unconstrained fit, and would otherwise cap the rank
        # silently.
        IMAGE_LOGGER.warning(
            'derive_confidence_rank: sigma_px is None (no covariance); '
            'sigma-constrained tiers are unreachable and the rank caps at the '
            'best sigma-free tier'
        )
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
        IMAGE_LOGGER.info(
            'All %d technique result(s) returned spurious=True: %s',
            len(results),
            ', '.join(r.technique_name for r in results),
        )
        return NavResult.failed(
            status_reason=NavStatusReason.ALL_TECHNIQUES_SPURIOUS,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    # Drop fallback-tier results superseded by a non-spurious primary
    # for the same body (e.g., BodyTerminatorNav / BodyBlobNav when
    # BodyLimbNav or BodyDiscCorrelateNav succeeded on the same body).
    # The full ``results`` list is preserved on the NavResult for
    # diagnostics; only the ensemble math sees the filtered set.
    viable = _drop_superseded_fallbacks(viable)
    interior = [r for r in viable if not r.at_edge]
    if interior:
        viable = interior
    groups = _agreement_groups(
        viable,
        agreement_sigma=cfg.agreement_sigma,
        agreement_pixel_floor=cfg.agreement_pixel_floor,
        rcond=cfg.pinvh_rcond,
        max_allowed_rotation_deg=cfg.max_allowed_rotation_deg,
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
        combined = _combine_precision_weighted(
            best_group,
            rcond=cfg.pinvh_rcond,
            max_allowed_rotation_deg=cfg.max_allowed_rotation_deg,
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
        IMAGE_LOGGER.info(
            'Combined precision-weighted offset is unobservable: every input '
            'covariance shares one null direction (%d input(s))',
            len(best_group),
        )
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
            IMAGE_LOGGER.info(
                'Conflicted: best-vs-runner-up summed-confidence gap %.3f is '
                'below the agreement_gap threshold %.3f '
                '(best %.3f, runner-up %.3f); conflicted confidence = %.3f '
                '(combined %.3f x conflicted_multiplier %.3f)',
                gap,
                cfg.agreement_gap,
                best_summed_conf,
                runner_up_summed_conf,
                conflicted_confidence,
                combined_confidence,
                cfg.conflicted_confidence_multiplier,
            )
            return NavResult.conflicted(
                offset_px=combined.offset_px,
                covariance_px2=combined.covariance_px2,
                confidence=conflicted_confidence,
                per_technique=results,
                feature_inventory=feature_inventory,
                image_classifier=image_classifier,
                provenance=provenance,
                model_metadata=md,
                annotations=ann,
            )
    if combined_confidence < cfg.min_confidence:
        IMAGE_LOGGER.info(
            'Combined confidence %.3f is below the min_confidence threshold %.3f',
            combined_confidence,
            cfg.min_confidence,
        )
        return NavResult.failed(
            status_reason=NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    cov = combined.covariance_px2
    sigma_dv = float(math.sqrt(max(cov[0, 0], 0.0)))
    sigma_du = float(math.sqrt(max(cov[1, 1], 0.0)))
    sigma_along_unobservable_px = float('inf') if combined.is_rank_deficient else None
    rank = derive_confidence_rank(
        confidence=combined_confidence,
        sigma_px=(sigma_dv, sigma_du),
        tier_thresholds=cfg.tier_thresholds,
    )
    if rank == 'failed':
        # Confidence + sigma combination doesn't earn any tier.  Distinguish the
        # two causes: if the combined confidence cleared the *lowest* tier's
        # ``min_confidence`` yet still earned no tier, the offset was confident
        # but too imprecise (sigma exceeded every tier's ``max_sigma_px``);
        # otherwise the confidence itself was below threshold.
        lowest_min_conf = min(
            float(spec['min_confidence'] or 0.0) for spec in cfg.tier_thresholds.values()
        )
        if combined_confidence >= lowest_min_conf:
            failed_reason = NavStatusReason.FINAL_SIGMA_ABOVE_THRESHOLD
        else:
            failed_reason = NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD
        IMAGE_LOGGER.info(
            'No tier earned (%s): combined confidence %.3f, sigma (dv, du) = '
            '(%.3f, %.3f) px (max %.3f); tier thresholds = %s',
            failed_reason.value,
            combined_confidence,
            sigma_dv,
            sigma_du,
            max(sigma_dv, sigma_du),
            cfg.tier_thresholds,
        )
        return NavResult.failed(
            status_reason=failed_reason,
            image_classifier=image_classifier,
            provenance=provenance,
            per_technique=results,
            feature_inventory=feature_inventory,
            model_metadata=md,
            annotations=ann,
        )
    status_reason = (
        NavStatusReason.RANK_1_ONLY if combined.is_rank_deficient else NavStatusReason.OK
    )
    sigma_rotation_rad: float | None = None
    if combined.rotation_rad is not None and cov.shape == (3, 3):
        sigma_rotation_rad = float(math.sqrt(max(cov[2, 2], 0.0)))
    return NavResult.success(
        offset_px=combined.offset_px,
        covariance_px2=combined.covariance_px2,
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
        rotation_rad=combined.rotation_rad,
        sigma_rotation_rad=sigma_rotation_rad,
    )
