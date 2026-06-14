"""``RingEdgeNav`` — translation fit from ring-edge polylines.

Consumes every ``RING_EDGE`` feature in the input set and produces a
single combined translation by minimising the joint distance-transform
cost.  When every input ring edge is flagged ``is_straight_line`` the
combined Jacobian is rank-deficient — all parallel ring edges share a
single ring-plane normal, so the along-edge axis is unobservable.  The
returned covariance is honestly rank-1 in that case; the ensemble
combine fuses it with any orthogonal-axis result (a star, body limb,
body blob) before declaring a final answer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import RingEdgePolyline
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import RingEdgeDiagnostics
from nav.nav_technique.dt_fitting import (
    build_polyline_mask,
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
from nav.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['RingEdgeNav']

# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.RingEdgeNav.tuning``.  No Python-level fallback;
# missing-key access in ``__init__`` is a KeyError so a config typo
# fails fast at process startup.


_RANK1_NULL_RELATIVE_THRESHOLD: float = 1.0e-8
"""Eigenvalue ratio below which a 2x2 covariance is treated as rank-1.

Matches the ensemble's scale-independent rank-deficiency test so the
two paths agree on whether a result is rank-deficient.
"""


def _aggregate_ring_edges(
    features: list[NavFeature],
) -> tuple[
    NDArrayFloatType,
    NDArrayFloatType,
    NDArrayFloatType,
    list[str],
    bool,
]:
    """Concatenate vertices, polarity normals, and per-vertex sigmas for ring edges.

    Returns:
        ``(vertices, polarity_normals, sigmas, feature_ids,
        every_edge_is_straight)``.  ``every_edge_is_straight`` is True
        when every input edge has ``is_straight_line=True``; the
        technique uses it to drive the rank-1 covariance path.
    """
    vert_chunks: list[NDArrayFloatType] = []
    normal_chunks: list[NDArrayFloatType] = []
    sigma_chunks: list[NDArrayFloatType] = []
    ids: list[str] = []
    every_straight = True
    seen_any = False
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        if feat.geometry.vertices_vu.shape[0] == 0:
            continue
        seen_any = True
        if not feat.geometry.is_straight_line:
            every_straight = False
        vert_chunks.append(feat.geometry.vertices_vu.astype(np.float64))
        # Negate the radial outward normal so the polarity check
        # (``dot(model, image_gradient) > 0``) accepts edges whose image
        # gradient points across the edge in the same sense as the
        # negated normal.
        normal_chunks.append(-feat.geometry.normals_vu.astype(np.float64))
        sigma_chunks.append(feat.geometry.sigma_radial_per_vertex_px.astype(np.float64))
        ids.append(feat.feature_id)
    empty_2 = np.empty((0, 2), np.float64)
    empty_1 = np.empty(0, np.float64)
    vertices = np.concatenate(vert_chunks, axis=0) if vert_chunks else empty_2
    normals = np.concatenate(normal_chunks, axis=0) if normal_chunks else empty_2
    sigmas = np.concatenate(sigma_chunks, axis=0) if sigma_chunks else empty_1
    return vertices, normals, sigmas, ids, (every_straight if seen_any else False)


class RingEdgeNav(NavTechnique):
    """Ring-edge DT-based translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({RING_EDGE})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'RingEdgeNav'
    accepts_feature_types = frozenset({NavFeatureType.RING_EDGE})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'total_edge_length_px',
            'per_edge_dt_rms_summed',
            'edge_count',
            'is_rank_1',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._spurious_dt_rms_factor = float(self.tuning['spurious_dt_rms_factor'])
        self._spurious_dt_floor_px = float(self.tuning['spurious_dt_floor_px'])
        self._spurious_min_inliers = int(self.tuning['spurious_min_inliers'])
        self._spurious_per_edge_rms_factor = float(self.tuning['spurious_per_edge_rms_factor'])
        self._spurious_max_lm_displacement_px = float(
            self.tuning['spurious_max_lm_displacement_px']
        )
        self._lm_trust_region_px = float(self.tuning['lm_trust_region_px'])
        self._lm_tikhonov_alpha = float(self.tuning['lm_tikhonov_alpha'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable ring edge.

        Reads only the polyline vertex count per feature.  A single
        non-empty ring-edge polyline is sufficient: even an all-flat
        scene produces a useful rank-1 constraint that the ensemble
        will fuse with another feature.

        Parameters:
            features: Feature list filtered to this technique's accepted
                types.

        Returns:
            ``NavFeasibilityReport`` with ``feasible=True`` iff at least
            one RING_EDGE has a non-empty polyline.
        """
        eligible = [
            f
            for f in features
            if isinstance(f.geometry, RingEdgePolyline) and f.geometry.vertices_vu.shape[0] > 0
        ]
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_ring_edge_features',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint-translation offset from ring-edge polylines.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.
            context: Per-image NavContext.  Must carry
                ``image_edge_dt_ext`` and ``image_gradient_vu_ext`` —
                both populated by the orchestrator's ``_make_context``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance (rank-1 when every input edge is straight),
            calibrated confidence, and a populated
            :class:`RingEdgeDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            if context.image_edge_dt_ext is None or context.image_gradient_vu_ext is None:
                raise RuntimeError(
                    'RingEdgeNav requires NavContext.image_edge_dt_ext and '
                    'NavContext.image_gradient_vu_ext to be populated by the orchestrator'
                )
            (
                vertices,
                polarity_normals,
                sigmas,
                feature_ids,
                every_straight,
            ) = _aggregate_ring_edges(features)
            self.logger.info(
                'Consuming %d RING_EDGE features (out of %d offered); every_straight=%s',
                len(feature_ids),
                len(features),
                every_straight,
            )
            edge_dt = context.image_edge_dt_ext
            gradient_vu = context.image_gradient_vu_ext
            edge_mask = edge_dt <= 0.5
            polyline_mask = build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = search_window_for_obs(context)
            self.logger.debug(
                'Aggregated %d ring-edge vertices, sigma_radial range [%.3f, %.3f] px, '
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
            # Ring-edge polarity prediction depends on lighting / gap-vs-ringlet
            # context the catalog does not encode today; skip polarity until
            # the polarity-predictable flag is wired (deferred work).
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
                use_polarity=False,
                fit_rotation=fit_rotation,
                pivot_vu=pivot_vu if fit_rotation else None,
                pivot_distance_px=pivot_distance,
                trust_region_px=self._lm_trust_region_px,
                tikhonov_alpha=self._lm_tikhonov_alpha,
            )
            dv_final, du_final = result.offset_vu
            max_rotation_rad = math.radians(context.max_rotation_deg)
            rotation_at_edge = fit_rotation and (
                abs(result.rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
            )
            # See BodyLimbNav: ``>=`` covers both at-edge and over-edge cases
            # so an LM that walked past the coarse-NCC search window does
            # not silently report an offset outside the extfov margin.
            at_edge = (
                abs(dv_final) >= margin_v - self._at_edge_tolerance_px
                or abs(du_final) >= margin_u - self._at_edge_tolerance_px
                or rotation_at_edge
            )
            sigma_min_px = float(sigmas.min()) if sigmas.size else 1.0
            covariance = result.covariance
            total_edge_length_px = float(vertices.shape[0])
            per_edge_rms_summed = _per_edge_rms_summed(features, result.residuals_px)
            edge_count = len(feature_ids)
            # ``result.rms_px`` is the *Tukey-weighted* residual RMS; when
            # the LM converges to a local minimum where one edge fits
            # cleanly and the rest are wholly mis-aligned, Tukey rejects
            # the bad-edge vertices and ``rms_px`` collapses to near
            # zero — a textbook mis-convergence that the existing
            # ``rms_px > floor`` threshold cannot detect.  The raw
            # ``per_edge_dt_rms_summed`` does not have outlier
            # rejection, so it surfaces the bad edges; flag spurious
            # when the per-edge average exceeds the same DT residual
            # threshold the Tukey-weighted check uses.  The check
            # protects the downstream ensemble combine from a
            # confidence-zero result whose offset is far from the
            # true fit (Cassini Tethys N1572471790 is the calibration
            # case; the LM converged on the wrong ring of three).
            per_edge_rms_threshold = max(
                self._spurious_dt_floor_px,
                self._spurious_per_edge_rms_factor * sigma_min_px,
            )
            lm_displacement_px = float(
                math.hypot(dv_final - float(coarse_dv), du_final - float(coarse_du))
            )
            spurious = (
                result.degenerate
                or result.rms_px
                > max(
                    self._spurious_dt_floor_px,
                    self._spurious_dt_rms_factor * sigma_min_px,
                )
                or result.inlier_count < self._spurious_min_inliers
                or (
                    edge_count > 0
                    and per_edge_rms_summed / float(edge_count) > per_edge_rms_threshold
                )
                or lm_displacement_px > self._spurious_max_lm_displacement_px
            )
            rotation_rad: float | None
            sigma_rotation_rad: float | None
            if fit_rotation:
                if covariance.shape != (3, 3):
                    raise RuntimeError(
                        f'RingEdgeNav expected 3x3 covariance with fit_rotation; '
                        f'got {covariance.shape}'
                    )
                rotation_rad = float(result.rotation_rad)
                sigma_rotation_rad = float(np.sqrt(max(float(covariance[2, 2]), 0.0)))
            else:
                if covariance.shape != (2, 2):
                    self.logger.warning(
                        'RingEdgeNav: lm_subpixel_refine returned %s covariance with '
                        'fit_rotation=False; truncating to (2, 2)',
                        covariance.shape,
                    )
                    covariance = covariance[:2, :2]
                rotation_rad = None
                sigma_rotation_rad = None
            covariance = np.asarray(covariance, np.float64)
            is_rank_1 = _is_rank_1(covariance) or every_straight
            diagnostics = RingEdgeDiagnostics(
                total_edge_length_px=total_edge_length_px,
                per_edge_dt_rms_summed=per_edge_rms_summed,
                edge_count=edge_count,
                is_rank_1=bool(is_rank_1),
            )
            assert self.confidence_spec is not None  # set as class attribute
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _RingEdgeConfidenceContext(at_edge=at_edge, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, RMS %.4f px, inliers %d / %d, '
                'rank_1=%s, confidence %.4f',
                dv_final,
                du_final,
                result.rms_px,
                result.inlier_count,
                int(vertices.shape[0]),
                is_rank_1,
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
                'per_edge_rms_summed = %.3f, total_edge_length = %.1f px',
                result.iterations,
                sigma_min_px,
                per_edge_rms_summed,
                total_edge_length_px,
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


def _is_rank_1(covariance: NDArrayFloatType) -> bool:
    """Return True when the covariance is rank-deficient in its translation block.

    For both 2x2 (translation-only) and 3x3 (translation + rotation) inputs
    the test runs on the top-left 2x2 block — flat-ring rank-deficiency is
    a property of the translation parameterisation, regardless of whether
    rotation is fit.  Uses the same scale-independent test as the ensemble
    combine: the ratio of the smallest absolute eigenvalue to the largest
    must fall below :data:`_RANK1_NULL_RELATIVE_THRESHOLD`.
    """
    if covariance.shape not in ((2, 2), (3, 3)):
        return False
    block = covariance[:2, :2]
    eigvals = np.linalg.eigvalsh(block)
    largest = float(np.abs(eigvals).max())
    smallest = float(np.abs(eigvals).min())
    if largest == 0.0:
        return True
    return smallest / largest < _RANK1_NULL_RELATIVE_THRESHOLD


def _per_edge_rms_summed(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Sum the per-edge weighted RMS DT residual across all consumed edges."""
    total = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        if slice_residuals.size == 0:
            continue
        rms = float(np.sqrt(np.mean(slice_residuals * slice_residuals)))
        total += rms
    return total


class _RingEdgeConfidenceContext:
    """Adapter exposing ring-edge confidence terms in a single attribute set."""

    def __init__(self, *, at_edge: bool, diagnostics: RingEdgeDiagnostics) -> None:
        self.at_edge = at_edge
        self.total_edge_length_px = diagnostics.total_edge_length_px
        self.per_edge_dt_rms_summed = diagnostics.per_edge_dt_rms_summed
        self.edge_count = diagnostics.edge_count
        self.is_rank_1 = diagnostics.is_rank_1
