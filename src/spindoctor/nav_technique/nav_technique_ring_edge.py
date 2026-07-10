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

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_technique.confidence import evaluate_sigmoid_combination
from spindoctor.nav_technique.diagnostics import RingEdgeDiagnostics
from spindoctor.nav_technique.dt_fitting import (
    build_polyline_mask,
    coarse_ncc_search,
    lm_subpixel_refine,
)
from spindoctor.nav_technique.feasibility import NavFeasibilityReport
from spindoctor.nav_technique.nav_technique import (
    NavTechnique,
    log_confidence_breakdown,
    rotation_pivot_distance_px,
    search_window_for_obs,
)
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['RingEdgeNav', 'aggregate_edge_normal_angle_deg']

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
            'per_edge_dt_rms_mean',
            'per_edge_dt_median_max',
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
        self._spurious_min_inlier_fraction = float(self.tuning['spurious_min_inlier_fraction'])
        self._spurious_max_lm_displacement_px = float(
            self.tuning['spurious_max_lm_displacement_px']
        )
        self._lm_trust_region_px = float(self.tuning['lm_trust_region_px'])
        self._lm_tikhonov_alpha = float(self.tuning['lm_tikhonov_alpha'])
        self._gradient_ridge_refine = bool(self.tuning['gradient_ridge_refine'])
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
                final_gradient_ridge=self._gradient_ridge_refine,
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
            per_edge_median_max = _per_edge_median_max(features, result.residuals_px)
            edge_count = len(feature_ids)
            # ``result.rms_px`` is the *Tukey-weighted* residual RMS; when
            # the LM converges to a local minimum where one edge fits
            # cleanly and the rest are wholly mis-aligned, Tukey rejects
            # the bad-edge vertices and ``rms_px`` collapses to near
            # zero — a textbook mis-convergence the ``rms_px > floor``
            # threshold cannot detect (Cassini Tethys N1572471790 is the
            # calibration case; the LM converged on the wrong ring of
            # three, anchoring the fit on a third of the model vertices).
            # No residual statistic separates that from a correct fit
            # whose faintest edge simply is not detectable in the image —
            # in both, one edge's residuals sit at the ringlet spacing
            # (Keeler-gap ansa frames run 9-30 px; issue #203).  What does
            # separate them is how much of the model the fit explains:
            # the Tukey inlier fraction is ~1/3 for the wrong-ring lock
            # and >0.8 for a correct flat fit with one undetected edge,
            # so the gate is a minimum inlier fraction over the aggregated
            # vertices.
            total_vertices = int(vertices.shape[0])
            inlier_fraction = (
                float(result.inlier_count) / float(total_vertices) if total_vertices else 0.0
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
                or inlier_fraction < self._spurious_min_inlier_fraction
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
            if is_rank_1:
                # Enforce the rank-1 contract on the reported covariance.
                # On an all-straight scene the along-edge (tangent) offset
                # component is physically unobservable, but the LM
                # covariance often comes back numerically tight along the
                # tangent anyway: the model polyline's *ends* anchor the
                # along-edge coordinate against wherever the detected edge
                # happens to enter and leave the frame — a false
                # constraint.  Rebuild the translation covariance as the
                # LM's marginal variance along the aggregate edge normal
                # plus an effectively infinite tangent variance, so the
                # ensemble's rank-deficiency test fires and the result
                # surfaces as ``rank_1_only`` instead of a confidently
                # full-rank fix (issue #203).
                covariance = _rank1_projected_covariance(covariance, polarity_normals)
            per_edge_rms_mean = per_edge_rms_summed / float(max(edge_count, 1))
            diagnostics = RingEdgeDiagnostics(
                total_edge_length_px=total_edge_length_px,
                per_edge_dt_rms_summed=per_edge_rms_summed,
                per_edge_dt_rms_mean=per_edge_rms_mean,
                per_edge_dt_median_max=per_edge_median_max,
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
                'per_edge_rms_summed = %.3f, per_edge_median_max = %.3f, '
                'total_edge_length = %.1f px',
                result.iterations,
                sigma_min_px,
                per_edge_rms_summed,
                per_edge_median_max,
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


def aggregate_edge_normal_angle_deg(features: list[NavFeature]) -> float | None:
    """Return the dominant edge-normal orientation of an all-straight scene.

    The angle is in degrees in the ``(v, u)`` frame, measured from ``+v``
    toward ``+u`` (the unit normal is ``(cos, sin)``) — the same convention
    as the sidecar schema's ``ground_truth.constraint.normal_angle_deg``.
    The orientation is the dominant eigenvector of the per-vertex normals'
    outer-product sum, so it is independent of each edge's polarity sign.

    Parameters:
        features: Any feature list; only ``RING_EDGE`` polylines are read.

    Returns:
        The orientation in degrees, normalised to ``(-90, 90]``, or ``None``
        unless at least one ring-edge polyline is present and every one is
        straight — the constraint direction is only meaningful for a rank-1
        scene.
    """
    _vertices, normals, _sigmas, ids, every_straight = _aggregate_ring_edges(features)
    if not ids or not every_straight:
        return None
    outer_sum = normals.T @ normals
    _eigvals, eigvecs = np.linalg.eigh(outer_sum)
    n_hat = eigvecs[:, -1]
    angle = math.degrees(math.atan2(float(n_hat[1]), float(n_hat[0])))
    # An orientation, not a direction: fold to (-90, 90].
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    return float(angle)


def _rank1_projected_covariance(
    covariance: NDArrayFloatType, polarity_normals: NDArrayFloatType
) -> NDArrayFloatType:
    """Project a rank-1 fit's covariance onto its observable (edge-normal) axis.

    The translation block becomes the exactly singular ``sigma_n^2 n n^T``,
    where ``n`` is the aggregate edge-normal orientation and ``sigma_n^2``
    the input covariance's marginal variance along it.  Exact singularity is
    the representation the ensemble is built around: ``pinvh`` drops the
    exact null space when forming the information matrix (keeping the
    normal-axis measurement), the combine's rank-deficiency test fires, the
    fused offset becomes the minimum-norm representative along the edge
    (sliding along a straight edge is a symmetry of the scene), and the
    unobservable axis is reported through the
    ``sigma_along_unobservable_px = inf`` sentinel rather than an inflated
    per-axis sigma that would poison the tier check.

    The normal orientation is the dominant eigenvector of the per-vertex
    normals' outer-product sum, which is polarity-sign-independent — the
    aggregated edges' normals point in opposite senses (a gap's inner and
    outer edges), so a plain mean could cancel.

    For a 3x3 (rotation-fitting) covariance the rotation variance is kept
    and the translation-rotation cross-covariances are zeroed: a
    cross-covariance into an unobservable translation direction carries no
    usable information.

    Parameters:
        covariance: The LM's ``(2, 2)`` or ``(3, 3)`` covariance.
        polarity_normals: ``(N, 2)`` per-vertex edge normals (either sign).

    Returns:
        A new covariance of the input shape whose translation block is
        exactly rank-1.
    """
    outer_sum = polarity_normals.T @ polarity_normals
    _eigvals, eigvecs = np.linalg.eigh(outer_sum)
    n_hat = eigvecs[:, -1]
    sigma_n_sq = float(n_hat @ covariance[:2, :2] @ n_hat)
    projected = np.zeros_like(covariance)
    projected[:2, :2] = sigma_n_sq * np.outer(n_hat, n_hat)
    if covariance.shape == (3, 3):
        projected[2, 2] = covariance[2, 2]
    return projected


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


def _per_edge_median_max(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Return the largest per-edge median absolute DT residual across edges.

    A mis-convergence *diagnostic* (surfaced through
    :class:`RingEdgeDiagnostics`, not a spurious gate): a wholly misaligned
    or undetected edge puts most of its vertices roughly a ringlet spacing
    from the nearest image edge, driving its median to that spacing, while
    a well-matched edge's median sits at the fit residual.  Taking the max
    over edges keeps one bad edge visible among any number of clean ones.
    Residuals alone cannot separate "misaligned" from "undetectable in the
    image", which is why the spurious decision uses the inlier fraction
    instead.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex signed DT residuals from the fit, in the same
            concatenation order.

    Returns:
        ``max_e median(|residuals_e|)`` over the consumed edges, or ``0.0``
        when no edge has vertices.
    """
    worst = 0.0
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
        worst = max(worst, float(np.median(np.abs(slice_residuals))))
    return worst


class _RingEdgeConfidenceContext:
    """Adapter exposing ring-edge confidence terms in a single attribute set."""

    def __init__(self, *, at_edge: bool, diagnostics: RingEdgeDiagnostics) -> None:
        self.at_edge = at_edge
        self.total_edge_length_px = diagnostics.total_edge_length_px
        self.per_edge_dt_rms_summed = diagnostics.per_edge_dt_rms_summed
        self.per_edge_dt_rms_mean = diagnostics.per_edge_dt_rms_mean
        self.per_edge_dt_median_max = diagnostics.per_edge_dt_median_max
        self.edge_count = diagnostics.edge_count
        self.is_rank_1 = diagnostics.is_rank_1
