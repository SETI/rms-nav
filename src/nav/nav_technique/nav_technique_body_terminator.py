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

import math
from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import TerminatorArcFlags
from nav.feature.geometry import TerminatorPolyline
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import BodyTerminatorDiagnostics
from nav.nav_technique.dt_fitting import (
    coarse_ncc_search,
    lm_subpixel_refine,
)
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import (
    NavTechnique,
    log_confidence_breakdown,
    rotation_pivot_distance_px,
    search_window_for_obs,
)
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyTerminatorNav']

# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.BodyTerminatorNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


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
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
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
        self.config.read_config()  # ensure cls.tuning is populated
        self._min_arc_px = float(self.tuning['min_arc_px'])
        self._spurious_dt_rms_factor = float(self.tuning['spurious_dt_rms_factor'])
        self._spurious_dt_floor_px = float(self.tuning['spurious_dt_floor_px'])
        self._spurious_min_inliers = int(self.tuning['spurious_min_inliers'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries a usable terminator arc.

        Reads only the polyline vertex count per feature.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one TERMINATOR_ARC has at least the configured ``min_arc_px``
            surviving vertices.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, TerminatorPolyline)
            and f.geometry.vertices_vu.shape[0] >= self._min_arc_px
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
                types.  Polylines with fewer than the configured
                ``min_arc_px`` vertices are dropped before fitting.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``
                — plus ``fit_camera_rotation`` and ``max_rotation_deg``.

        Returns:
            A :class:`NavTechniqueResult` with the recovered offset,
            calibrated confidence, and a populated
            :class:`BodyTerminatorDiagnostics`.  Per
            :class:`BodyLimbNav.navigate`:

            - When ``context.fit_camera_rotation`` is False the result
              carries a ``(2, 2)`` covariance and ``rotation_rad`` /
              ``sigma_rotation_rad`` are ``None`` (an unexpected
              non-(2, 2) covariance from LM is logged at WARNING and
              truncated).
            - When ``context.fit_camera_rotation`` is True the result
              carries a ``(3, 3)`` covariance with the LM-fit rotation
              diagonal and populated ``rotation_rad`` /
              ``sigma_rotation_rad``.  An unexpected covariance shape
              from LM raises ``RuntimeError``.
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
                and f.geometry.vertices_vu.shape[0] >= self._min_arc_px
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
            if vertices.shape[0] == 0:
                raise RuntimeError(
                    'BodyTerminatorNav.navigate received zero usable TERMINATOR_ARC vertices '
                    'despite is_feasible reporting feasibility; aborting fit'
                )
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            polyline_mask = _build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = search_window_for_obs(context)
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
            fit_rotation = bool(context.fit_camera_rotation)
            pivot_vu = (float(vertices[:, 0].mean()), float(vertices[:, 1].mean()))
            pivot_distance = (
                rotation_pivot_distance_px(pivot_vu, edge_dt.shape[:2]) if fit_rotation else 0.0
            )
            result = lm_subpixel_refine(
                vertices_vu=vertices,
                normals_vu=polarity_normals,
                sigma_normal_per_vertex_px=sigmas,
                image_edge_dt=edge_dt,
                image_gradient_vu=gradient_vu,
                initial_offset_vu=(float(coarse_dv), float(coarse_du)),
                use_polarity=True,
                fit_rotation=fit_rotation,
                pivot_vu=pivot_vu if fit_rotation else None,
                pivot_distance_px=pivot_distance,
            )
            dv_final, du_final = result.offset_vu
            max_rotation_rad = math.radians(context.max_rotation_deg)
            rotation_at_edge = fit_rotation and (
                abs(result.rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
            )
            covariance = result.covariance
            rotation_rad: float | None
            sigma_rotation_rad: float | None
            if fit_rotation:
                if covariance.shape != (3, 3):
                    raise RuntimeError(
                        f'BodyTerminatorNav expected 3x3 covariance with fit_rotation; '
                        f'got {covariance.shape}'
                    )
                rotation_rad = float(result.rotation_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(covariance[2, 2]), 0.0)))
            else:
                if covariance.shape != (2, 2):
                    self.logger.warning(
                        'BodyTerminatorNav: lm_subpixel_refine returned %s covariance with '
                        'fit_rotation=False; truncating to (2, 2)',
                        covariance.shape,
                    )
                    covariance = covariance[:2, :2]
                rotation_rad = None
                sigma_rotation_rad = None
            at_edge = (
                abs(dv_final - margin_v) <= self._at_edge_tolerance_px
                or abs(dv_final + margin_v) <= self._at_edge_tolerance_px
                or abs(du_final - margin_u) <= self._at_edge_tolerance_px
                or abs(du_final + margin_u) <= self._at_edge_tolerance_px
                or rotation_at_edge
            )
            sigma_min_px = float(sigmas.min()) if sigmas.size else 1.0
            spurious = (
                result.rms_px
                > max(
                    self._spurious_dt_floor_px,
                    self._spurious_dt_rms_factor * sigma_min_px,
                )
                or result.inlier_count < self._spurious_min_inliers
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
                spurious=bool(spurious),
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
            if fit_rotation and sigma_rotation_rad is not None and rotation_rad is not None:
                self.logger.info(
                    'Rotation = %+.4f deg (sigma %.4f deg)%s',
                    math.degrees(rotation_rad),
                    math.degrees(sigma_rotation_rad),
                    ', AT_EDGE' if rotation_at_edge else '',
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
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(feature_ids),
                offset_px=(float(dv_final), float(du_final)),
                covariance_px2=covariance,
                confidence=float(confidence),
                spurious=bool(spurious),
                at_edge=bool(at_edge),
                diagnostics=diagnostics,
                rotation_rad=rotation_rad,
                sigma_rotation_rad=sigma_rotation_rad,
            )


class _TerminatorConfidenceContext:
    """Adapter exposing terminator confidence terms in a single attribute set."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: BodyTerminatorDiagnostics,
        mean_phase_angle_factor: float,
        mean_albedo_penalty: float,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
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
