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

from typing import TYPE_CHECKING

import numpy as np

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import RingEdgePolyline
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import RingEdgeDiagnostics
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
            polyline_mask = _build_polyline_mask(vertices, edge_dt.shape[:2])
            margin_v, margin_u = _search_window_for_obs(context)
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
            result = lm_subpixel_refine(
                vertices_vu=vertices,
                normals_vu=polarity_normals,
                sigma_normal_per_vertex_px=sigmas,
                image_edge_dt=edge_dt,
                image_gradient_vu=gradient_vu,
                initial_offset_vu=(float(coarse_dv), float(coarse_du)),
                use_polarity=False,
            )
            dv_final, du_final = result.offset_vu
            at_edge = (
                abs(dv_final - margin_v) <= self._at_edge_tolerance_px
                or abs(dv_final + margin_v) <= self._at_edge_tolerance_px
                or abs(du_final - margin_u) <= self._at_edge_tolerance_px
                or abs(du_final + margin_u) <= self._at_edge_tolerance_px
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
            covariance = result.covariance
            if covariance.shape != (2, 2):
                covariance = covariance[:2, :2]
            covariance = np.asarray(covariance, np.float64)
            is_rank_1 = _is_rank_1(covariance) or every_straight
            total_edge_length_px = float(vertices.shape[0])
            per_edge_rms_summed = _per_edge_rms_summed(features, result.residuals_px)
            diagnostics = RingEdgeDiagnostics(
                total_edge_length_px=total_edge_length_px,
                per_edge_dt_rms_summed=per_edge_rms_summed,
                edge_count=len(feature_ids),
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
            )


def _is_rank_1(covariance: NDArrayFloatType) -> bool:
    """Return True when the 2x2 covariance is rank-deficient.

    Uses the same scale-independent test as the ensemble combine: the
    ratio of the smallest absolute eigenvalue to the largest must fall
    below :data:`_RANK1_NULL_RELATIVE_THRESHOLD`.
    """
    if covariance.shape != (2, 2):
        return False
    eigvals = np.linalg.eigvalsh(covariance)
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


def _search_window_for_obs(context: NavContext) -> tuple[int, int]:
    """Return ``(margin_v, margin_u)`` for the coarse search."""
    obs = context.obs
    margin = getattr(obs, 'extfov_margin_vu', None)
    if margin is None:
        return (32, 32)
    return (int(margin[0]), int(margin[1]))
