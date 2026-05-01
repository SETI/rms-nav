"""``BodyLimbNav`` — translation fit from body limb polylines.

Consumes every ``LIMB_ARC`` feature in the input set, concatenates their
per-vertex positions, weights them by ``1 / sigma_normal_per_vertex_px**2``,
and runs the shared distance-transform fitter to recover a single
translation that minimises the joint cost across all bodies.  Multi-body
inputs improve the fit by ``sqrt(N_bodies)`` when SPICE relative geometry
is correct; the joint-translation parameterisation cannot represent
"swap two moons" mistakes by construction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import LimbPolyline
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import BodyLimbDiagnostics
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

__all__ = ['BodyLimbNav']

# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.BodyLimbNav.tuning``.  No Python-level fallback;
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


def _aggregate_limb_features(
    features: list[NavFeature],
) -> tuple[
    NDArrayFloatType,
    NDArrayFloatType,
    NDArrayFloatType,
    list[str],
]:
    """Concatenate vertices, normals, and per-vertex sigmas from LIMB_ARC features.

    The model normals are negated relative to the geometric outward normal
    so that the ``polarity_filter`` ``dot > 0`` rule corresponds to "image
    gradient points into the body silhouette" — which is the typical
    bright-body convention used by the body NavModel.

    Returns:
        ``(vertices, polarity_normals, sigmas, feature_ids)``.
    """
    vert_chunks: list[NDArrayFloatType] = []
    normal_chunks: list[NDArrayFloatType] = []
    sigma_chunks: list[NDArrayFloatType] = []
    ids: list[str] = []
    for feat in features:
        if not isinstance(feat.geometry, LimbPolyline):
            continue
        if feat.geometry.vertices_vu.shape[0] == 0:
            continue
        vert_chunks.append(feat.geometry.vertices_vu.astype(np.float64))
        # Negate to obtain "polarity normals": the direction the image
        # gradient is expected to point at a bright-body limb.
        normal_chunks.append(-feat.geometry.normals_vu.astype(np.float64))
        sigma_chunks.append(feat.geometry.sigma_normal_per_vertex_px.astype(np.float64))
        ids.append(feat.feature_id)
    empty_2 = np.empty((0, 2), np.float64)
    empty_1 = np.empty(0, np.float64)
    vertices = np.concatenate(vert_chunks, axis=0) if vert_chunks else empty_2
    normals = np.concatenate(normal_chunks, axis=0) if normal_chunks else empty_2
    sigmas = np.concatenate(sigma_chunks, axis=0) if sigma_chunks else empty_1
    return vertices, normals, sigmas, ids


class BodyLimbNav(NavTechnique):
    """Body-limb DT-based translation fit.

    Consumes every ``LIMB_ARC`` feature whose visible arc length meets the
    feasibility threshold and produces one combined translation offset by
    minimising the summed weighted squared distance from the model
    polylines to the image edge distance transform.  Per-vertex weights
    follow the prior precision ``1 / sigma_normal_per_vertex_px**2``;
    Tukey biweight reweighting handles the per-image outliers.

    Class attributes:
        accepts_feature_types: ``frozenset({LIMB_ARC})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'BodyLimbNav'
    accepts_feature_types = frozenset({NavFeatureType.LIMB_ARC})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'visible_limb_arc_fraction',
            'visible_arc_px',
            'dt_fit_rms_px',
            'lm_iterations',
            'tukey_inlier_count',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._min_arc_px = float(self.tuning['min_arc_px'])
        self._spurious_dt_rms_factor = float(self.tuning['spurious_dt_rms_factor'])
        self._spurious_dt_floor_px = float(self.tuning['spurious_dt_floor_px'])
        self._spurious_min_inliers = int(self.tuning['spurious_min_inliers'])
        self._spurious_min_inlier_fraction = float(self.tuning['spurious_min_inlier_fraction'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable limb arc.

        Reads only the polyline vertex count per feature — never any
        pixels — so the report is cheap to obtain even on large feature
        sets.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one LIMB_ARC has at least the configured ``min_arc_px``
            surviving vertices.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, LimbPolyline)
            and f.geometry.vertices_vu.shape[0] >= self._min_arc_px
        ]
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_limb_arc_features_with_sufficient_visible_arc',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from the input limb polylines.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Polylines with fewer than the configured
                ``min_arc_px`` vertices are dropped before fitting.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`BodyLimbDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            if context.image_edge_dt_ext is None or context.image_gradient_vu_ext is None:
                raise RuntimeError(
                    'BodyLimbNav requires NavContext.image_edge_dt_ext and '
                    'NavContext.image_gradient_vu_ext to be populated by the orchestrator'
                )
            eligible_features = [
                f
                for f in features
                if isinstance(f.geometry, LimbPolyline)
                and f.geometry.vertices_vu.shape[0] >= self._min_arc_px
            ]
            self.logger.info(
                'Consuming %d LIMB_ARC features (out of %d offered)',
                len(eligible_features),
                len(features),
            )
            vertices, polarity_normals, sigmas, feature_ids = _aggregate_limb_features(
                eligible_features
            )
            if vertices.shape[0] == 0:
                raise RuntimeError(
                    'BodyLimbNav.navigate received zero usable LIMB_ARC vertices despite '
                    'is_feasible reporting feasibility; aborting fit'
                )
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            polyline_mask = _build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = search_window_for_obs(context)
            self.logger.debug(
                'Aggregated %d limb vertices, sigma_normal range [%.3f, %.3f] px, '
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
                        f'BodyLimbNav expected 3x3 covariance with fit_rotation; '
                        f'got {covariance.shape}'
                    )
                rotation_rad = float(result.rotation_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(covariance[2, 2]), 0.0)))
            else:
                if covariance.shape != (2, 2):
                    self.logger.warning(
                        'BodyLimbNav: lm_subpixel_refine returned %s covariance with '
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
            n_vertices = int(vertices.shape[0])
            inlier_fraction = (
                float(result.inlier_count) / float(n_vertices) if n_vertices > 0 else 0.0
            )
            spurious = (
                result.rms_px
                > max(
                    self._spurious_dt_floor_px,
                    self._spurious_dt_rms_factor * sigma_min_px,
                )
                or result.inlier_count < self._spurious_min_inliers
                or inlier_fraction < self._spurious_min_inlier_fraction
            )
            visible_limb_arc_fraction = _aggregate_visible_arc_fraction(eligible_features)
            diagnostics = BodyLimbDiagnostics(
                visible_limb_arc_fraction=visible_limb_arc_fraction,
                visible_arc_px=float(vertices.shape[0]),
                dt_fit_rms_px=float(result.rms_px),
                lm_iterations=int(result.iterations),
                tukey_inlier_count=int(result.inlier_count),
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _LimbConfidenceContext(
                    at_edge=at_edge, spurious=bool(spurious), diagnostics=diagnostics
                ),
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
                'LM iterations = %d, sigma_min = %.3f px, visible_limb_arc_fraction = %.3f',
                result.iterations,
                sigma_min_px,
                visible_limb_arc_fraction,
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


class _LimbConfidenceContext:
    """Adapter binding ``BodyLimbDiagnostics`` plus ``at_edge`` / ``spurious``.

    The shared :func:`evaluate_sigmoid_combination` helper accepts any
    object whose attributes match the spec's term names.  ``at_edge``
    and ``spurious`` are not part of ``BodyLimbDiagnostics`` (they live
    on ``NavTechniqueResult``) so this small adapter exposes them
    alongside the diagnostic fields the spec consumes.
    """

    def __init__(self, *, at_edge: bool, spurious: bool, diagnostics: BodyLimbDiagnostics) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.visible_limb_arc_fraction = diagnostics.visible_limb_arc_fraction
        self.visible_arc_px = diagnostics.visible_arc_px
        self.dt_fit_rms_px = diagnostics.dt_fit_rms_px
        self.lm_iterations = diagnostics.lm_iterations
        self.tukey_inlier_count = diagnostics.tukey_inlier_count


def _aggregate_visible_arc_fraction(features: list[NavFeature]) -> float:
    """Return the per-feature ``visible_arc_fraction`` weighted by vertex count.

    The reliability breakdown carries each feature's visible-arc fraction
    on the predicted polyline; the diagnostic on the navigation result
    summarises across all consumed features by weighting each fraction by
    the count of surviving vertices.  The result is in ``[0, 1]``.
    """
    total_weighted = 0.0
    total_count = 0.0
    for feat in features:
        fraction = feat.reliability_reasons.visible_arc_fraction
        if fraction is None:
            continue
        if not isinstance(feat.geometry, LimbPolyline):
            continue
        n = float(feat.geometry.vertices_vu.shape[0])
        total_weighted += float(fraction) * n
        total_count += n
    if total_count == 0.0:
        return 0.0
    return total_weighted / total_count
