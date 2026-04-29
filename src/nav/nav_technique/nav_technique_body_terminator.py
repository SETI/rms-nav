"""``BodyTerminatorNav`` — translation fit from body terminator polylines.

Mirrors :class:`BodyLimbNav` with three terminator-specific differences:

1. The accepted feature type is ``TERMINATOR_ARC`` instead of ``LIMB_ARC``.
2. Per-body uniform weighting: every vertex of a given body shares one
   inverse-variance weight derived from the body's mean
   ``sigma_normal_per_vertex_px``.  Cross-body weighting reflects albedo
   variation (low-albedo bodies provide tighter terminators than
   high-albedo ones).
3. The confidence spec includes additional terms for the per-body
   visible-terminator-arc fraction and the phase-angle factor flag,
   capturing the design's albedo / phase-geometry sensitivity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import TerminatorArcFlags
from nav.feature.geometry import TerminatorPolyline
from nav.nav_technique.confidence import (
    ConfidenceSpec,
    ConfidenceTerm,
    evaluate_sigmoid_combination,
)
from nav.nav_technique.diagnostics import BodyTerminatorDiagnostics
from nav.nav_technique.dt_fitting import (
    coarse_ncc_search,
    lm_subpixel_refine,
)
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import NavTechnique, log_confidence_breakdown
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyTerminatorNav']


TERMINATOR_MIN_ARC_PX: float = 30.0
"""Minimum surviving polyline length per TERMINATOR_ARC feature for feasibility."""


SPURIOUS_DT_RMS_FACTOR: float = 5.0
"""Final DT residual exceeding this many limb-sigmas marks the result spurious."""


SPURIOUS_DT_FLOOR_PX: float = 4.0
"""Floor for the spurious-detection threshold (terminators are softer than limbs)."""


SPURIOUS_MIN_INLIERS: int = 6
"""Below this Tukey-inlier count the final fit is flagged spurious."""


_AT_EDGE_TOLERANCE_PX: float = 1.0
"""Pixels of slack around the search-window axis bounds for at-edge detection.

A converged offset whose absolute distance from any axis bound (``+/-margin_v``,
``+/-margin_u``) falls within this tolerance is flagged ``at_edge=True`` and
forced to zero confidence by the technique's ``hard_zero_if`` gate.  One pixel
matches the bilinear DT half-cell width: any closer to the boundary and the
LM gradient information is unreliable.
"""


_BODY_TERMINATOR_CONFIDENCE_SPEC = ConfidenceSpec(
    alpha0=-1.0,
    terms=(
        ConfidenceTerm(feature='visible_terminator_arc_fraction', alpha=2.0),
        ConfidenceTerm(feature='dt_fit_rms_px', alpha=-1.0),
        ConfidenceTerm(
            feature='visible_arc_px',
            alpha=0.4,
            divisor=100.0,
            cap_at=1.0,
        ),
        ConfidenceTerm(feature='mean_phase_angle_factor', alpha=1.0),
        ConfidenceTerm(feature='mean_albedo_penalty', alpha=-1.5),
    ),
    hard_zero_if={'at_edge': True},
)
"""Default confidence spec for the body-terminator technique.

Terminator ceiling sits below the limb's ceiling because albedo variation
softens the photometric edge; the phase-angle-factor term boosts
crescent-illumination geometries (where the terminator is geometrically
sharp) and the albedo-penalty term suppresses scenes where surface
mottling would otherwise dominate.
"""


def _build_polyline_mask(
    vertices_vu: NDArrayFloatType, shape_vu: tuple[int, int]
) -> NDArrayBoolType:
    """Render polyline vertices into a boolean image mask aligned to shape_vu."""
    vs = np.rint(vertices_vu[:, 0]).astype(np.int64)
    us = np.rint(vertices_vu[:, 1]).astype(np.int64)
    valid = (vs >= 0) & (vs < shape_vu[0]) & (us >= 0) & (us < shape_vu[1])
    mask = np.zeros(shape_vu, dtype=bool)
    if valid.any():
        mask[vs[valid], us[valid]] = True
    return mask


def _aggregate_terminator_features(
    features: list[NavFeature],
) -> tuple[
    NDArrayFloatType,
    NDArrayFloatType,
    NDArrayFloatType,
    list[str],
    list[float],
    list[float],
]:
    """Concatenate per-body vertices and uniform-per-body sigmas.

    The design dictates that every vertex of a given body shares
    ``1 / sigma_normal_per_vertex_px**2`` derived from the body's mean
    sigma — that captures cross-body albedo variation while smoothing
    out per-vertex sigma noise.  The ``polarity_normals`` are the
    geometric outward normal negated, identical to the limb pipeline,
    because the terminator's image gradient also points from the dark
    side toward the lit side.

    Returns:
        ``(vertices, polarity_normals, sigmas_uniform_per_body,
        feature_ids, phase_angle_factors, albedo_penalties)``.
    """
    vert_chunks: list[NDArrayFloatType] = []
    normal_chunks: list[NDArrayFloatType] = []
    sigma_chunks: list[NDArrayFloatType] = []
    ids: list[str] = []
    phase_factors: list[float] = []
    albedo_penalties: list[float] = []
    for feat in features:
        if not isinstance(feat.geometry, TerminatorPolyline):
            continue
        if feat.geometry.vertices_vu.shape[0] == 0:
            continue
        vertices = feat.geometry.vertices_vu.astype(np.float64)
        outward = feat.geometry.normals_vu.astype(np.float64)
        per_body_sigma = float(feat.geometry.sigma_normal_per_vertex_px.astype(np.float64).mean())
        per_body_sigma_arr = np.full(vertices.shape[0], per_body_sigma, dtype=np.float64)
        vert_chunks.append(vertices)
        normal_chunks.append(-outward)
        sigma_chunks.append(per_body_sigma_arr)
        ids.append(feat.feature_id)
        flags = feat.flags
        if isinstance(flags, TerminatorArcFlags):
            phase_factors.append(float(flags.phase_angle_factor))
        else:
            phase_factors.append(1.0)
        albedo_penalty = feat.reliability_reasons.albedo_penalty
        albedo_penalties.append(float(albedo_penalty) if albedo_penalty is not None else 0.0)
    empty_2 = np.empty((0, 2), np.float64)
    empty_1 = np.empty(0, np.float64)
    vertices_out = np.concatenate(vert_chunks, axis=0) if vert_chunks else empty_2
    normals_out = np.concatenate(normal_chunks, axis=0) if normal_chunks else empty_2
    sigmas_out = np.concatenate(sigma_chunks, axis=0) if sigma_chunks else empty_1
    return vertices_out, normals_out, sigmas_out, ids, phase_factors, albedo_penalties


class BodyTerminatorNav(NavTechnique):
    """Body-terminator DT-based translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({TERMINATOR_ARC})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'BodyTerminatorNav'
    accepts_feature_types = frozenset({NavFeatureType.TERMINATOR_ARC})
    requires_prior = False
    confidence_spec = _BODY_TERMINATOR_CONFIDENCE_SPEC
    confidence_attributes = frozenset(
        {
            'at_edge',
            'visible_terminator_arc_fraction',
            'visible_arc_px',
            'dt_fit_rms_px',
            'lm_iterations',
            'tukey_inlier_count',
            'mean_phase_angle_factor',
            'mean_albedo_penalty',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries a usable terminator arc.

        Reads only the polyline vertex count per feature.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one TERMINATOR_ARC has at least :data:`TERMINATOR_MIN_ARC_PX`
            surviving vertices.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, TerminatorPolyline)
            and f.geometry.vertices_vu.shape[0] >= TERMINATOR_MIN_ARC_PX
        ]
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_terminator_arc_features_with_sufficient_visible_arc',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input terminator polylines.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Polylines with fewer than
                :data:`TERMINATOR_MIN_ARC_PX` vertices are dropped before
                fitting.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`BodyTerminatorDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            if context.image_edge_dt_ext is None or context.image_gradient_vu_ext is None:
                raise RuntimeError(
                    'BodyTerminatorNav requires NavContext.image_edge_dt_ext and '
                    'NavContext.image_gradient_vu_ext to be populated by the orchestrator'
                )
            eligible_features = [
                f
                for f in features
                if isinstance(f.geometry, TerminatorPolyline)
                and f.geometry.vertices_vu.shape[0] >= TERMINATOR_MIN_ARC_PX
            ]
            self.logger.info(
                'Consuming %d TERMINATOR_ARC features (out of %d offered)',
                len(eligible_features),
                len(features),
            )
            (
                vertices,
                polarity_normals,
                sigmas,
                feature_ids,
                phase_factors,
                albedo_penalties,
            ) = _aggregate_terminator_features(eligible_features)
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            polyline_mask = _build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = _search_window_for_obs(context)
            self.logger.debug(
                'Aggregated %d terminator vertices, sigma_normal range [%.3f, %.3f] px, '
                'search window (v, u) = (%d, %d) px',
                int(vertices.shape[0]),
                float(sigmas.min()) if sigmas.size else 0.0,
                float(sigmas.max()) if sigmas.size else 0.0,
                margin_v,
                margin_u,
            )
            coarse_dv, coarse_du = coarse_ncc_search(
                edge_mask,
                polyline_mask,
                (margin_v, margin_u),
            )
            self.logger.debug('Coarse NCC offset: (%d, %d)', coarse_dv, coarse_du)
            result = lm_subpixel_refine(
                vertices_vu=vertices,
                normals_vu=polarity_normals,
                sigma_normal_per_vertex_px=sigmas,
                image_edge_dt=edge_dt,
                image_gradient_vu=gradient_vu,
                initial_offset_vu=(float(coarse_dv), float(coarse_du)),
                use_polarity=True,
            )
            dv_final, du_final = result.offset_vu
            at_edge = (
                abs(dv_final - margin_v) <= _AT_EDGE_TOLERANCE_PX
                or abs(dv_final + margin_v) <= _AT_EDGE_TOLERANCE_PX
                or abs(du_final - margin_u) <= _AT_EDGE_TOLERANCE_PX
                or abs(du_final + margin_u) <= _AT_EDGE_TOLERANCE_PX
            )
            sigma_min_px = float(sigmas.min()) if sigmas.size else 1.0
            spurious = (
                result.rms_px > max(SPURIOUS_DT_FLOOR_PX, SPURIOUS_DT_RMS_FACTOR * sigma_min_px)
                or result.inlier_count < SPURIOUS_MIN_INLIERS
            )
            visible_terminator_arc_fraction = _aggregate_visible_arc_fraction(eligible_features)
            mean_phase = float(np.mean(phase_factors)) if phase_factors else 0.0
            mean_albedo = float(np.mean(albedo_penalties)) if albedo_penalties else 0.0
            diagnostics = BodyTerminatorDiagnostics(
                visible_terminator_arc_fraction=visible_terminator_arc_fraction,
                visible_arc_px=float(vertices.shape[0]),
                dt_fit_rms_px=float(result.rms_px),
                lm_iterations=int(result.iterations),
                tukey_inlier_count=int(result.inlier_count),
            )
            confidence_context = _TerminatorConfidenceContext(
                at_edge=at_edge,
                diagnostics=diagnostics,
                mean_phase_angle_factor=mean_phase,
                mean_albedo_penalty=mean_albedo,
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                confidence_context,
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, RMS %.4f px, inliers %d / %d, '
                'confidence %.4f',
                dv_final,
                du_final,
                result.rms_px,
                result.inlier_count,
                int(vertices.shape[0]),
                float(confidence),
            )
            if spurious or at_edge:
                self.logger.info('Diagnostic flags: spurious=%s, at_edge=%s', spurious, at_edge)
            self.logger.debug(
                'LM iterations = %d, sigma_min = %.3f px, '
                'visible_terminator_arc_fraction = %.3f, mean_phase_factor = %.3f, '
                'mean_albedo_penalty = %.3f',
                result.iterations,
                sigma_min_px,
                visible_terminator_arc_fraction,
                mean_phase,
                mean_albedo,
            )
            covariance = result.covariance
            if covariance.shape != (2, 2):
                covariance = covariance[:2, :2]
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(feature_ids),
                offset_px=(float(dv_final), float(du_final)),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=bool(spurious),
                at_edge=bool(at_edge),
                diagnostics=diagnostics,
            )


class _TerminatorConfidenceContext:
    """Adapter exposing terminator confidence terms in a single attribute set."""

    def __init__(
        self,
        *,
        at_edge: bool,
        diagnostics: BodyTerminatorDiagnostics,
        mean_phase_angle_factor: float,
        mean_albedo_penalty: float,
    ) -> None:
        self.at_edge = at_edge
        self.visible_terminator_arc_fraction = diagnostics.visible_terminator_arc_fraction
        self.visible_arc_px = diagnostics.visible_arc_px
        self.dt_fit_rms_px = diagnostics.dt_fit_rms_px
        self.lm_iterations = diagnostics.lm_iterations
        self.tukey_inlier_count = diagnostics.tukey_inlier_count
        self.mean_phase_angle_factor = mean_phase_angle_factor
        self.mean_albedo_penalty = mean_albedo_penalty


def _aggregate_visible_arc_fraction(features: list[NavFeature]) -> float:
    """Return the per-feature ``visible_arc_fraction`` weighted by vertex count."""
    total_weighted = 0.0
    total_count = 0.0
    for feat in features:
        fraction = feat.reliability_reasons.visible_arc_fraction
        if fraction is None:
            continue
        if not isinstance(feat.geometry, TerminatorPolyline):
            continue
        n = float(feat.geometry.vertices_vu.shape[0])
        total_weighted += float(fraction) * n
        total_count += n
    if total_count == 0.0:
        return 0.0
    return total_weighted / total_count


def _search_window_for_obs(context: NavContext) -> tuple[int, int]:
    """Return ``(margin_v, margin_u)`` for the coarse search."""
    obs = context.obs
    margin = getattr(obs, 'extfov_margin_vu', None)
    if margin is None:
        return (32, 32)
    return (int(margin[0]), int(margin[1]))
