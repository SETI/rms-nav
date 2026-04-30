"""``BodyBlobNav`` — joint-translation fit from body brightness centroids.

Consumes every ``BODY_BLOB`` feature in the input set, computes a
brightness-weighted-moment centroid for each body inside its predicted
bounding box, and recovers a single 2-D translation that maps the
predicted centroids onto the observed centroids in least-squares.  With
``N >= 2`` blobs the fit is over-determined, which makes the technique
robust to centroid errors on any single body.

The technique reports a confidence intrinsically capped at 0.4: a
brightness-weighted centroid is much weaker than a limb fit, so even an
ideal blob match cannot dominate the ensemble.  Per-blob centroid
uncertainty follows the standard CRLB scaling for a uniform-brightness
disc: ``sigma ~ predicted_diameter_px / (2 * sqrt(N_lit) * SNR)``; the
joint fit inherits that scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pdslogger import PdsLogger

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.feature.geometry import BodyBlobGeometry
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import BodyBlobDiagnostics
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import (
    NavTechnique,
    log_confidence_breakdown,
    search_window_for_obs,
)
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['BodyBlobNav']


@dataclass(frozen=True)
class _BlobResiduals:
    """Per-blob residual + statistics arrays for the joint fit.

    ``offsets_v`` / ``offsets_u`` are the per-blob ``observed - predicted``
    centroid vectors, ``weights`` are the per-blob inverse-variance
    weights, ``snrs`` and ``extents`` are diagnostic samples that feed
    the confidence formula via :class:`BodyBlobDiagnostics`.
    """

    consumed: list[NavFeature]
    offsets_v: NDArrayFloatType
    offsets_u: NDArrayFloatType
    weights: NDArrayFloatType
    snrs: list[float]
    extents: list[float]


@dataclass(frozen=True)
class _JointFit:
    """Joint translation result derived from per-blob residuals."""

    dv: float
    du: float
    covariance: NDArrayFloatType
    residual_rms: float


def _filter_blob_features(features: list[NavFeature]) -> list[NavFeature]:
    """Return the subset that carries a ``BODY_BLOB`` geometry payload."""
    return [
        f
        for f in features
        if f.feature_type is NavFeatureType.BODY_BLOB and isinstance(f.geometry, BodyBlobGeometry)
    ]


def _clamp_bbox(
    bbox_extfov_vu: tuple[int, int, int, int],
    extfov_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Clamp ``(v_min, u_min, v_max, u_max)`` to lie inside ``extfov_shape``."""
    v_min, u_min, v_max, u_max = bbox_extfov_vu
    h, w = extfov_shape
    return (
        max(0, int(v_min)),
        max(0, int(u_min)),
        min(int(h), int(v_max)),
        min(int(w), int(u_max)),
    )


def _brightness_weighted_centroid(
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    geometry: BodyBlobGeometry,
) -> tuple[tuple[float, float] | None, float, int, tuple[int, int, int, int]]:
    """Return the brightness-weighted centroid + signal stats for one blob.

    The centroid is computed over every above-noise pixel inside the
    feature's predicted bounding box (the bbox includes per-body slop so
    the actual body silhouette stays inside it under moderate SPICE
    pointing error).  Above-noise pixels are those exceeding
    ``3 * image_noise_sigma``; background DN never biases the moment.
    A blob whose bbox carries no above-noise pixels returns
    ``(None, 0.0, 0, clamped_bbox)`` so the caller can drop it from the
    joint fit and still log its bbox in the per-blob rejection line.

    Returns:
        ``(centroid_vu, mean_signal_above_noise, n_lit_pixels,
        clamped_bbox)``.  ``centroid_vu`` is ``None`` when the blob has
        no usable signal.
    """
    extfov_shape = (image_ext.shape[0], image_ext.shape[1])
    clamped_bbox = _clamp_bbox(geometry.bbox_extfov_vu, extfov_shape)
    v_min, u_min, v_max, u_max = clamped_bbox
    if v_max <= v_min or u_max <= u_min:
        return None, 0.0, 0, clamped_bbox
    patch = image_ext[v_min:v_max, u_min:u_max]
    noise_threshold = 3.0 * max(image_noise_sigma, 1e-9)
    signal_mask: NDArrayBoolType = patch > noise_threshold
    n_lit = int(signal_mask.sum())
    if n_lit == 0:
        return None, 0.0, 0, clamped_bbox
    weights = np.where(signal_mask, patch, 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return None, 0.0, 0, clamped_bbox
    vs = np.arange(v_min, v_max, dtype=np.float64)
    us = np.arange(u_min, u_max, dtype=np.float64)
    centroid_v = float(np.sum(weights * vs[:, None]) / total_weight)
    centroid_u = float(np.sum(weights * us[None, :]) / total_weight)
    mean_signal = float(weights[signal_mask].mean())
    return (centroid_v, centroid_u), mean_signal, n_lit, clamped_bbox


def _collect_per_blob_residuals(
    features: list[NavFeature],
    image_ext: NDArrayFloatType,
    image_noise_sigma: float,
    logger: PdsLogger,
) -> _BlobResiduals:
    """Extract the per-blob ``observed - predicted`` residuals + weights.

    Iterates the input features in order and computes a
    brightness-weighted-moment centroid inside each predicted bbox.
    Blobs with no above-noise signal in their bbox are dropped (and
    logged at DEBUG with the bbox bounds and noise threshold so the
    operator can tell why).  The remaining blobs contribute to the
    joint fit with weight ``N_lit * SNR^2 / radius_px^2`` per the
    BODY_BLOB centroid CRLB.
    """
    consumed: list[NavFeature] = []
    offsets_v: list[float] = []
    offsets_u: list[float] = []
    weights: list[float] = []
    snrs: list[float] = []
    extents: list[float] = []
    noise_threshold = 3.0 * max(image_noise_sigma, 1e-9)
    for feature in features:
        assert isinstance(feature.geometry, BodyBlobGeometry)
        centroid, mean_signal, n_lit, clamped_bbox = _brightness_weighted_centroid(
            image_ext, image_noise_sigma, feature.geometry
        )
        if centroid is None:
            logger.debug(
                'Blob %s has no above-noise signal in predicted bbox %s '
                '(noise threshold = %.4f DN); dropping',
                feature.feature_id,
                clamped_bbox,
                noise_threshold,
            )
            continue
        pred_v, pred_u = feature.geometry.predicted_center_vu
        obs_v, obs_u = centroid
        dv = obs_v - pred_v
        du = obs_u - pred_u
        snr = mean_signal / max(image_noise_sigma, 1e-9)
        # Per-blob weight is the inverse of the centroid CRLB
        # variance: weight ~ N_lit * SNR^2 / R^2.  The upstream
        # emission gate guarantees ``predicted_diameter_px >= 8``,
        # so the radius_px denominator is bounded away from zero
        # and no floor is needed.
        radius_px = feature.geometry.predicted_diameter_px / 2.0
        radius_sq = radius_px * radius_px
        weight = float(n_lit) * snr * snr / radius_sq
        consumed.append(feature)
        offsets_v.append(dv)
        offsets_u.append(du)
        weights.append(max(weight, 1e-9))
        snrs.append(snr)
        extents.append(float(feature.geometry.predicted_diameter_px))
        logger.debug(
            'Blob %s: predicted (%.2f, %.2f), observed (%.2f, %.2f), SNR %.2f, '
            'N_lit %d, weight %.3g',
            feature.feature_id,
            pred_v,
            pred_u,
            obs_v,
            obs_u,
            snr,
            n_lit,
            weight,
        )
    return _BlobResiduals(
        consumed=consumed,
        offsets_v=np.asarray(offsets_v, np.float64),
        offsets_u=np.asarray(offsets_u, np.float64),
        weights=np.asarray(weights, np.float64),
        snrs=snrs,
        extents=extents,
    )


def _joint_offset_from_residuals(residuals: _BlobResiduals) -> _JointFit:
    """Solve the precision-weighted joint translation across the per-blob residuals."""
    offsets_v = residuals.offsets_v
    offsets_u = residuals.offsets_u
    weights = residuals.weights
    total_weight = float(weights.sum())
    dv = float(np.sum(weights * offsets_v) / total_weight)
    du = float(np.sum(weights * offsets_u) / total_weight)
    res_norms = np.hypot(offsets_v - dv, offsets_u - du)
    rms = float(np.sqrt(float(np.mean(res_norms * res_norms))))
    cov = _joint_covariance(
        offsets_v=offsets_v,
        offsets_u=offsets_u,
        weights=weights,
        dv=dv,
        du=du,
    )
    return _JointFit(dv=dv, du=du, covariance=cov, residual_rms=rms)


def _joint_covariance(
    *,
    offsets_v: NDArrayFloatType,
    offsets_u: NDArrayFloatType,
    weights: NDArrayFloatType,
    dv: float,
    du: float,
) -> NDArrayFloatType:
    """Return the per-axis precision-weighted covariance of the joint fit.

    The covariance is diagonal: independent per-axis weighted-mean
    variances ``1 / sum(weights)`` inflated by the residual scatter
    when more than one blob participates.  A single-blob fit reports
    the inverse-precision floor; a multi-blob fit reports the actual
    disagreement.

    The cross-term ``cov(v, u)`` is intentionally zero — per-axis
    residuals are independent under the BODY_BLOB CRLB derivation,
    and the precision-weighted ensemble combine downstream consumes
    diagonals correctly.  Future readers tempted to add
    ``cov_vu = sum(w * res_v * res_u) / sum(w)``: that term has no
    physical interpretation here because the per-axis errors come
    from independent moment integrals along orthogonal axes.
    """
    total_weight = float(weights.sum())
    floor = 1.0 / max(total_weight, 1e-12)
    if offsets_v.size <= 1:
        return float(floor) * np.eye(2, dtype=np.float64)
    residuals_v = offsets_v - dv
    residuals_u = offsets_u - du
    total_weight_sq = max(total_weight * total_weight, 1e-24)
    var_v = max(float(np.sum(weights * residuals_v * residuals_v) / total_weight_sq), floor)
    var_u = max(float(np.sum(weights * residuals_u * residuals_u) / total_weight_sq), floor)
    return np.diag([var_v, var_u]).astype(np.float64)


class BodyBlobNav(NavTechnique):
    """Body-blob brightness-weighted centroid translation fit.

    Class attributes:
        accepts_feature_types: ``frozenset({BODY_BLOB})``.
        requires_prior: ``False`` — the technique runs in pass 1.
    """

    name = 'BodyBlobNav'
    accepts_feature_types = frozenset({NavFeatureType.BODY_BLOB})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'body_snr_inside_predicted_bbox',
            'body_extent_px',
            'blob_count',
            'residual_px',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Return whether the input set carries any usable BODY_BLOB feature.

        Reads only feature metadata; never any pixels.  The technique
        requires at least one ``BODY_BLOB`` with a non-zero predicted
        diameter — otherwise the centroid moment is degenerate.
        """
        eligible = _eligible_blobs(features)
        if not eligible:
            return NavFeasibilityReport(
                feasible=False,
                reason='no_body_blob_features_with_predicted_diameter',
            )
        return NavFeasibilityReport(
            feasible=True,
            reason='ok',
            consumed_feature_count=len(eligible),
        )

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Compute the joint translation that maps predicted to observed centroids.

        Parameters:
            features: Feature list filtered to the technique's accepted
                types.  Blobs that fall outside the extfov or have no
                above-noise signal in their predicted bbox are dropped.
            context: Per-image NavContext.  Reads ``image_ext``,
                ``image_noise_sigma``, and ``obs.extfov_margin_vu``.

        Returns:
            A ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`BodyBlobDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            eligible = _eligible_blobs(features)
            self.logger.info(
                'Consuming %d BODY_BLOB features (out of %d offered)',
                len(eligible),
                len(features),
            )
            margin_v, margin_u = search_window_for_obs(context)
            self.logger.debug('Search window (v, u) = (%d, %d) px', margin_v, margin_u)
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            residuals = _collect_per_blob_residuals(eligible, image_ext, noise_sigma, self.logger)
            if not residuals.consumed:
                return self._fail_no_signal(features=eligible, noise_sigma=noise_sigma)
            fit = _joint_offset_from_residuals(residuals)
            at_edge = (
                abs(fit.dv) >= margin_v - self._at_edge_tolerance_px
                or abs(fit.du) >= margin_u - self._at_edge_tolerance_px
            )
            mean_snr = float(np.mean(residuals.snrs))
            mean_extent = float(np.mean(residuals.extents))
            diagnostics = BodyBlobDiagnostics(
                body_snr_inside_predicted_bbox=mean_snr,
                body_extent_px=mean_extent,
                blob_count=len(residuals.consumed),
                residual_px=fit.residual_rms,
            )
            assert self.confidence_spec is not None
            confidence, breakdown = evaluate_sigmoid_combination(
                self.confidence_spec,
                _BlobConfidenceContext(at_edge=at_edge, diagnostics=diagnostics),
                technique_name=self.name,
                return_breakdown=True,
            )
            log_confidence_breakdown(self.logger, breakdown)
            self.logger.info(
                'Converged at offset (%.4f, %.4f) px, residual RMS %.4f px, mean SNR %.2f, '
                'mean extent %.2f px, blobs %d, confidence %.4f',
                fit.dv,
                fit.du,
                fit.residual_rms,
                mean_snr,
                mean_extent,
                len(residuals.consumed),
                float(confidence),
            )
            if at_edge:
                self.logger.info('Diagnostic flags: at_edge=%s', at_edge)
            return NavTechniqueResult(
                technique_name=self.name,
                feature_ids=tuple(f.feature_id for f in residuals.consumed),
                offset_px=(fit.dv, fit.du),
                covariance_px2=fit.covariance,
                confidence=float(confidence),
                spurious=False,
                at_edge=at_edge,
                diagnostics=diagnostics,
            )

    def _fail_no_signal(
        self, *, features: list[NavFeature], noise_sigma: float
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result when no blob carries signal."""
        diagnostics = BodyBlobDiagnostics(
            body_snr_inside_predicted_bbox=0.0,
            body_extent_px=0.0,
            blob_count=0,
            residual_px=0.0,
        )
        self.logger.info(
            'No BODY_BLOB feature carried above-noise signal in its predicted bbox '
            '(noise threshold = %.4f DN, %d candidate blob(s)); reporting spurious result',
            3.0 * noise_sigma,
            len(features),
        )
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(0.0, 0.0),
            covariance_px2=1e6 * np.eye(2, dtype=np.float64),
            confidence=0.0,
            spurious=True,
            at_edge=False,
            diagnostics=diagnostics,
        )


def _eligible_blobs(features: list[NavFeature]) -> list[NavFeature]:
    """Filter the input set to BODY_BLOB features with non-zero diameter."""
    blob_features = _filter_blob_features(features)
    return [
        f
        for f in blob_features
        if isinstance(f.geometry, BodyBlobGeometry) and f.geometry.predicted_diameter_px > 0.0
    ]


class _BlobConfidenceContext:
    """Adapter binding ``BodyBlobDiagnostics`` plus ``at_edge`` for confidence eval."""

    def __init__(self, *, at_edge: bool, diagnostics: BodyBlobDiagnostics) -> None:
        self.at_edge = at_edge
        self.body_snr_inside_predicted_bbox = diagnostics.body_snr_inside_predicted_bbox
        self.body_extent_px = diagnostics.body_extent_px
        self.blob_count = float(diagnostics.blob_count)
        self.residual_px = diagnostics.residual_px
