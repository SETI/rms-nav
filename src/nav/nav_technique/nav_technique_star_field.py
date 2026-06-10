"""``StarFieldFromCatalogNav`` — multi-star RANSAC pattern matcher.

The technique is the long-pole prior-free star fit: it makes no assumption
about which detection corresponds to which catalog star, and recovers a
2-D translation purely from the *geometry* of bright sources in the image
matched against the predicted catalog field.

Algorithm — four sub-pieces:

1. **Source detection.**  The image is matched-filtered against a
   small Gaussian PSF kernel; pixels that are local maxima above
   ``detection_sigma * image_noise_sigma`` are accepted; a brightness-
   weighted centroid pins each peak's sub-pixel position.  The
   brightest ``max_sources`` survivors (default 30) feed the matcher.
2. **Triplet hashing.**  For every unordered triplet of detected
   sources ``{A, B, C}`` with ``A`` brightest, the hash
   ``(d_AB / d_AC, d_BC / d_AC, ∠BAC)`` is computed.  The same hash is
   computed for catalog triplets (``A`` = brightest by predicted SNR).
   The hash is similarity-invariant — translation, rotation, and
   uniform scale all leave it unchanged — so the matcher recovers
   correspondences without already knowing the offset.
3. **RANSAC.**  Each (detection-triplet, catalog-triplet) candidate is
   scored by counting detection-to-catalog inliers under the
   translation it implies.  Candidates are iterated in deterministic
   order — sorted first by hash distance ascending, then by sorted
   ``(idx_A, idx_B, idx_C)`` ascending — so the matcher's choice of
   winner is fully reproducible (Cardinal Principle 3, Part 3
   §"Determinism in RANSAC").
4. **Verification.**  With the best transform's inlier set, the
   technique refits the translation by Tukey-biweight-reweighted
   least squares; the per-axis variance of the surviving residuals
   becomes the reported covariance.

Phase 8 ships translation-only fitting; the
``fit_camera_rotation`` rotation upgrade arrives in Phase 9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import maximum_filter

from nav.config import Config
from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType
from nav.nav_model.stars.detection import matched_filter_image
from nav.nav_technique._star_helpers import (
    SimilarityFit,
    predicted_snr,
    predicted_vu,
    similarity_transform_fit,
    usable_stars,
)
from nav.nav_technique.confidence import evaluate_sigmoid_combination
from nav.nav_technique.diagnostics import StarFieldDiagnostics
from nav.nav_technique.dt_fitting import DEFAULT_TUKEY_C, tukey_biweight_weights
from nav.nav_technique.feasibility import NavFeasibilityReport
from nav.nav_technique.nav_technique import (
    ROTATION_UNOBSERVABLE_VARIANCE,
    NavTechnique,
    embed_rotation_unobservable,
    log_confidence_breakdown,
    rotation_unobservable_sigma_rad,
    search_window_for_obs,
)
from nav.nav_technique.technique_result import NavTechniqueResult
from nav.support.types import NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['StarFieldFromCatalogNav']


_COLLINEAR_REL_EPS: float = 1.0e-6
"""Relative tolerance for the collinear-triplet rejection in ``_triplet_hash``.

A triplet is treated as collinear (and rejected) when the magnitude of
its 2-D cross product falls below ``_COLLINEAR_REL_EPS * d_ab * d_ac``.
The dimensionless ratio survives uniform scaling of the input, so the
floor expresses "the smaller-angle deviation from a perfect line" in
the same way a relative-error tolerance does for floating-point
comparisons.  ``1e-6`` admits numerical jitter on near-degenerate
real-detection triplets while still catching the geometric
degeneracy.
"""


# All numeric tunables for this technique live in
# ``config_files/config_510_techniques.yaml`` under
# ``techniques.StarFieldFromCatalogNav.tuning``.  Missing-key access in
# ``__init__`` is a KeyError so a config typo fails fast at process
# startup.


@dataclass(frozen=True)
class _DetectedSource:
    """Internal detection record produced by ``_detect_image_sources``.

    The technique uses only ``(v, u, peak_dn)``; richer DAOPHOT
    statistics (sharpness / roundness, saturation tag) are not needed
    for triplet matching and are therefore not surfaced.
    """

    v: float
    u: float
    peak_dn: float


def _gaussian_kernel(sigma_px: float, *, size: int) -> NDArrayFloatType:
    """Return a normalised 2-D isotropic Gaussian matched-filter kernel.

    Parameters:
        sigma_px: PSF sigma in pixels.  Must be strictly positive.
        size: Side length of the (odd) square kernel.  Coerced up to
            the next odd integer when even.

    Returns:
        ``(size, size)`` float64 array, sum-to-unity.
    """
    if sigma_px <= 0.0:
        raise ValueError(f'sigma_px must be > 0; got {sigma_px!r}')
    if size % 2 == 0:
        size += 1
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    vv, uu = np.meshgrid(coords, coords, indexing='ij')
    kernel = np.exp(-(vv * vv + uu * uu) / (2.0 * sigma_px * sigma_px))
    kernel /= float(kernel.sum())
    return kernel


def _detect_image_sources(
    image: NDArrayFloatType,
    *,
    image_noise_sigma: float,
    sigma_px: float,
    detection_sigma: float,
    centroid_box_half_px: int,
    max_sources: int,
) -> list[_DetectedSource]:
    """Detect bright local-max sources in ``image`` and return the brightest set.

    Three-stage pipeline:

    1. Matched-filter the image against a Gaussian kernel sized by
       ``sigma_px``; the response peak amplitude at each pixel is the
       maximum-likelihood signal estimate at that location.
    2. Find pixels that are local maxima within a
       ``(2 * centroid_box_half_px + 1)`` window AND clear
       ``detection_sigma * image_noise_sigma``.
    3. For each surviving peak, fit a brightness-weighted moment in a
       small box around the peak to pull a sub-pixel centroid.

    Detections are ranked by ``(peak_dn descending, v ascending,
    u ascending)`` and the first ``max_sources`` are returned — the
    triplet matcher is M^3 in the input count, so 30 is the typical
    cap (Part 3 §"Determinism in RANSAC").

    Parameters:
        image: 2-D float input.
        image_noise_sigma: Robust per-pixel noise sigma in DN.
        sigma_px: PSF sigma in pixels for the matched-filter kernel.
        detection_sigma: Threshold multiplier on
            ``image_noise_sigma``.
        centroid_box_half_px: Half-width of the centroid box (and of
            the local-max window).
        max_sources: Maximum number of detections to return.

    Returns:
        List of ``_DetectedSource`` records, ordered by brightness
        descending.  May be empty when no peak clears the threshold.
    """
    kernel_size = 2 * centroid_box_half_px + 1
    kernel = _gaussian_kernel(sigma_px, size=kernel_size)
    response = matched_filter_image(image, kernel=kernel)
    threshold = detection_sigma * image_noise_sigma
    local_max = maximum_filter(response, size=kernel_size, mode='reflect')
    candidate_mask = (response == local_max) & (response > threshold)
    h, w = image.shape
    out: list[_DetectedSource] = []
    for v_idx, u_idx in np.argwhere(candidate_mask):
        v_i = int(v_idx)
        u_i = int(u_idx)
        v_lo = max(0, v_i - centroid_box_half_px)
        u_lo = max(0, u_i - centroid_box_half_px)
        v_hi = min(h, v_i + centroid_box_half_px + 1)
        u_hi = min(w, u_i + centroid_box_half_px + 1)
        box = image[v_lo:v_hi, u_lo:u_hi].astype(np.float64)
        bg = float(np.median(box))
        weights = np.clip(box - bg, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            continue
        vs = np.arange(v_lo, v_hi, dtype=np.float64)
        us = np.arange(u_lo, u_hi, dtype=np.float64)
        cv = float(np.sum(vs[:, None] * weights) / total)
        cu = float(np.sum(us[None, :] * weights) / total)
        out.append(_DetectedSource(v=cv, u=cu, peak_dn=float(response[v_i, u_i])))
    out.sort(key=lambda s: (-s.peak_dn, s.v, s.u))
    return out[:max_sources]


def _triplet_hash(
    pa: tuple[float, float],
    pb: tuple[float, float],
    pc: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Return the similarity-invariant ``(d_AB / d_AC, d_BC / d_AC, ∠BAC)`` triple.

    Per Part 3 §"Algorithm specifics" the hash is computed with ``A``
    canonicalised as the brightest of the triplet (the caller is
    responsible for that canonicalisation); both ratios survive
    arbitrary rotation, translation, and uniform scale, which is what
    lets the matcher recover correspondences without knowing the
    transform.

    Parameters:
        pa: Brightest vertex of the triplet.
        pb, pc: The other two vertices, in caller-canonical
            (sorted-index) order so each unordered triplet hashes
            once.

    Returns:
        ``(r1, r2, theta)`` tuple, or ``None`` for degenerate
        configurations (collinear vertices, coincident points) which
        are silently dropped.
    """
    av, au = pa
    bv, bu = pb
    cv, cu = pc
    ab_v = bv - av
    ab_u = bu - au
    ac_v = cv - av
    ac_u = cu - au
    bc_v = cv - bv
    bc_u = cu - bu
    d_ab = math.hypot(ab_v, ab_u)
    d_ac = math.hypot(ac_v, ac_u)
    d_bc = math.hypot(bc_v, bc_u)
    if d_ab <= 0.0 or d_ac <= 0.0:
        return None
    # Collinear-but-distinct triplets produce ``theta = 0`` or ``pi`` —
    # a hash that can match any other 0-or-pi hash regardless of scale,
    # which is a false-match trap for the RANSAC matcher.  Reject when
    # the 2-D cross product magnitude (``d_ab * d_ac * |sin(theta)|``)
    # falls below a small fraction of the side-length product, which
    # is dimensionless and survives uniform scaling of the input.
    cross = ab_v * ac_u - ab_u * ac_v
    if abs(cross) <= _COLLINEAR_REL_EPS * d_ab * d_ac:
        return None
    cos_theta = (ab_v * ac_v + ab_u * ac_u) / (d_ab * d_ac)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return (d_ab / d_ac, d_bc / d_ac, math.acos(cos_theta))


@dataclass(frozen=True)
class _Triplet:
    """One enumerated triplet, hashed in canonical form.

    ``idx_a / idx_b / idx_c`` reference back into the source point
    list; ``a`` is the brightest of the three and ``(b, c)`` are
    sorted by index ascending so each unordered triplet appears once.
    """

    idx_a: int
    idx_b: int
    idx_c: int
    hash_v: tuple[float, float, float]


def _enumerate_triplets(
    points: list[tuple[float, float]],
    brightness_rank: list[int],
) -> list[_Triplet]:
    """Return every canonical ``_Triplet`` over ``points``.

    ``brightness_rank[i]`` is the brightness rank of point ``i``
    (``0`` = brightest); it picks the ``a`` vertex within each
    unordered ``(i, j, k)`` triple.  Degenerate triplets (collinear or
    coincident) are dropped silently.
    """
    n = len(points)
    out: list[_Triplet] = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triple = (i, j, k)
                ranks = (brightness_rank[i], brightness_rank[j], brightness_rank[k])
                a_pos = int(np.argmin(np.asarray(ranks)))
                idx_a = triple[a_pos]
                rest = sorted(idx for idx in triple if idx != idx_a)
                idx_b, idx_c = rest
                hashed = _triplet_hash(points[idx_a], points[idx_b], points[idx_c])
                if hashed is None:
                    continue
                out.append(_Triplet(idx_a=idx_a, idx_b=idx_b, idx_c=idx_c, hash_v=hashed))
    return out


def _hash_distance_sq(
    h1: tuple[float, float, float],
    h2: tuple[float, float, float],
    *,
    ratio_weight: float,
    angle_weight: float,
) -> float:
    """Return the weighted squared Euclidean distance between two triplet hashes.

    ``ratio_weight`` scales the two ratio components and
    ``angle_weight`` scales the angle component.  Both default to 1.0
    in the YAML — radians and dimensionless ratios end up roughly
    comparable in magnitude over the typical triplet shapes.
    """
    dr1 = h1[0] - h2[0]
    dr2 = h1[1] - h2[1]
    da = h1[2] - h2[2]
    return ratio_weight * (dr1 * dr1 + dr2 * dr2) + angle_weight * da * da


def _greedy_inlier_count(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    offset_vu: tuple[float, float],
    *,
    tolerance_px: float,
) -> tuple[int, list[tuple[int, int]]]:
    """Greedy nearest-neighbour matching under the proposed translation.

    Each detection is paired with its closest catalog star (after
    applying the offset to catalog positions) within
    ``tolerance_px``.  Each catalog star can match at most one
    detection — once consumed, it is removed from the candidate pool.

    Parameters:
        detection_pts: ``(N_det, 2)`` array of detection (v, u).
        catalog_pts: ``(N_cat, 2)`` array of catalog (v, u).
        offset_vu: Translation applied to catalog positions before
            matching.
        tolerance_px: Maximum residual distance for a correspondence.

    Returns:
        ``(n_inliers, correspondences)`` where ``correspondences`` is a
        list of ``(det_idx, cat_idx)`` tuples in detection-index
        ascending order.  ``n_inliers`` equals
        ``len(correspondences)``.
    """
    if detection_pts.size == 0 or catalog_pts.size == 0:
        return 0, []
    shifted_catalog = catalog_pts + np.asarray(offset_vu, np.float64)[None, :]
    n_det = detection_pts.shape[0]
    n_cat = shifted_catalog.shape[0]
    available = np.ones(n_cat, dtype=bool)
    pairs: list[tuple[int, int]] = []
    tol_sq = tolerance_px * tolerance_px
    for d_idx in range(n_det):
        d = detection_pts[d_idx]
        if not available.any():
            break
        diffs = shifted_catalog - d[None, :]
        dist_sq = np.sum(diffs * diffs, axis=1)
        dist_sq[~available] = np.inf
        c_idx = int(np.argmin(dist_sq))
        if dist_sq[c_idx] <= tol_sq:
            pairs.append((d_idx, c_idx))
            available[c_idx] = False
    return len(pairs), pairs


def _triplet_centroid_offset(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    *,
    det_indices: tuple[int, int, int],
    cat_indices: tuple[int, int, int],
) -> tuple[float, float]:
    """Return the translation aligning two triplet centroids.

    Centroids are the unweighted mean of three vertex positions.
    Numerically equivalent to ``mean(detection - catalog)`` over the
    triplet but expressed centroid-style for clarity.
    """
    det_v = (
        detection_pts[det_indices[0], 0]
        + detection_pts[det_indices[1], 0]
        + detection_pts[det_indices[2], 0]
    ) / 3.0
    det_u = (
        detection_pts[det_indices[0], 1]
        + detection_pts[det_indices[1], 1]
        + detection_pts[det_indices[2], 1]
    ) / 3.0
    cat_v = (
        catalog_pts[cat_indices[0], 0]
        + catalog_pts[cat_indices[1], 0]
        + catalog_pts[cat_indices[2], 0]
    ) / 3.0
    cat_u = (
        catalog_pts[cat_indices[0], 1]
        + catalog_pts[cat_indices[1], 1]
        + catalog_pts[cat_indices[2], 1]
    ) / 3.0
    return float(det_v - cat_v), float(det_u - cat_u)


def _solve_translation(
    detection_pts: NDArrayFloatType,
    catalog_pts: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> tuple[float, float]:
    """Return the weighted-mean translation ``mean(d - c)``.

    Returns ``(0.0, 0.0)`` when the total weight is non-positive.
    """
    diffs = detection_pts - catalog_pts
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0, 0.0
    dv = float(np.sum(weights * diffs[:, 0]) / total)
    du = float(np.sum(weights * diffs[:, 1]) / total)
    return dv, du


def _seed_from_image_et(image_et: float) -> int:
    """Return a deterministic 64-bit RANSAC seed from the obs midtime ET.

    Per Part 3 §"Determinism in RANSAC" the matcher iterates triplets
    in sorted order (no random sampling), so the seed is informational
    rather than load-bearing.  It is logged so that any future
    tie-breaking RNG can pull from the same value and stay
    reproducible across two back-to-back invocations on the same obs.
    """
    if not math.isfinite(image_et):
        return 0
    # Convert seconds-past-J2000 (TDB) to integer nanoseconds, then
    # mask into the unsigned 64-bit range; ``& 0xFFFFFFFFFFFFFFFF``
    # makes negative ETs (pre-J2000 obs, e.g. Voyager) wrap cleanly
    # rather than overflow ``np.random.default_rng``.
    return round(image_et * 1.0e9) & 0xFFFFFFFFFFFFFFFF


class _StarFieldConfidenceContext:
    """Adapter binding ``StarFieldDiagnostics`` plus side flags."""

    def __init__(
        self,
        *,
        at_edge: bool,
        spurious: bool,
        diagnostics: StarFieldDiagnostics,
    ) -> None:
        self.at_edge = at_edge
        self.spurious = spurious
        self.n_inliers = float(diagnostics.n_inliers)
        self.median_residual_px = diagnostics.median_residual_px
        self.n_detected_sources = float(diagnostics.n_detected_sources)
        self.n_catalog_predicted = float(diagnostics.n_catalog_predicted)


class StarFieldFromCatalogNav(NavTechnique):
    """Multi-star pattern-match translation fit (≥ 3 catalog + image stars).

    Class attributes:
        accepts_feature_types: ``frozenset({STAR})``.
        requires_prior: ``False`` — runs in pass 1.
    """

    name = 'StarFieldFromCatalogNav'
    accepts_feature_types = frozenset({NavFeatureType.STAR})
    requires_prior = False
    confidence_attributes = frozenset(
        {
            'at_edge',
            'spurious',
            'n_inliers',
            'median_residual_px',
            'n_detected_sources',
            'n_catalog_predicted',
        }
    )

    def __init__(self, *, config: Config | None = None) -> None:
        super().__init__(config=config)
        self.config.read_config()  # ensure cls.tuning is populated
        self._max_sources = int(self.tuning['max_sources'])
        self._detection_sigma = float(self.tuning['detection_sigma'])
        self._psf_sigma_px = float(self.tuning['psf_sigma_px'])
        self._centroid_box_half_px = int(self.tuning['centroid_box_half_px'])
        self._hash_match_tolerance = float(self.tuning['hash_match_tolerance'])
        self._hash_ratio_weight = float(self.tuning['hash_ratio_weight'])
        self._hash_angle_weight = float(self.tuning['hash_angle_weight'])
        self._inlier_tolerance_px = float(self.tuning['inlier_tolerance_px'])
        self._min_inliers = int(self.tuning['pattern_match_min_inliers'])
        self._at_edge_tolerance_px = float(self.tuning['at_edge_tolerance_px'])
        self._rotation_at_edge_fraction = float(self.tuning['rotation_at_edge_fraction'])
        # Uncalibrated model-error variance floor (px); added in quadrature to
        # the reported covariance diagonal.  Default 0.0 -> no-op.  See
        # ORCH-001 / config_510_techniques.yaml.
        self._model_error_floor_px = float(self.tuning.get('model_error_floor_px', 0.0))
        if self._min_inliers < 3:
            raise ValueError(
                f'pattern_match_min_inliers must be >= 3 (the matcher needs at '
                f'least one triplet per side); got {self._min_inliers}'
            )
        if self._max_sources < 3:
            raise ValueError(f'max_sources must be >= 3; got {self._max_sources}')

    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport:
        """Report feasibility based on the predictable-star cohort.

        Reads only feature metadata; never any pixels.  Feasible when
        at least three usable STAR features are in the input set —
        below that the matcher cannot form a single triplet.
        """
        usable = usable_stars(features)
        if len(usable) < 3:
            return NavFeasibilityReport(
                feasible=False,
                reason=f'fewer_than_3_predicted_stars (got {len(usable)})',
            )
        return NavFeasibilityReport(feasible=True, reason='ok', consumed_feature_count=len(usable))

    def navigate(self, features: list[NavFeature], context: NavContext) -> NavTechniqueResult:
        """Recover translation from triplet pattern matching against the catalog.

        Parameters:
            features: STAR features (the orchestrator pre-filters by
                type).  At least three must survive
                ``usable_stars``.
            context: Per-image NavContext.

        Returns:
            ``NavTechniqueResult`` with the recovered offset, 2x2
            covariance, calibrated confidence, and a populated
            :class:`StarFieldDiagnostics`.
        """
        with self.logger.open(f'TECHNIQUE: {self.name}'):
            usable = usable_stars(features)
            self.logger.info(
                'Consuming %d usable STAR feature(s) (out of %d offered)',
                len(usable),
                len(features),
            )
            seed = _seed_from_image_et(float(context.provenance.image_et))
            self.logger.debug('RANSAC seed (from obs.midtime ns): %d', seed)
            ranked_catalog = sorted(usable, key=predicted_snr, reverse=True)[: self._max_sources]
            n_catalog_predicted = len(ranked_catalog)
            if n_catalog_predicted < 3:
                return self._fail(
                    features=features,
                    reason=f'fewer_than_3_predicted_stars (got {n_catalog_predicted})',
                    diagnostics=StarFieldDiagnostics(
                        n_inliers=0,
                        median_residual_px=0.0,
                        n_detected_sources=0,
                        n_catalog_predicted=n_catalog_predicted,
                        n_triplets_evaluated=0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            image_ext = np.asarray(context.image_ext, np.float64)
            noise_sigma = float(max(context.image_noise_sigma, 1e-9))
            detected = _detect_image_sources(
                image_ext,
                image_noise_sigma=noise_sigma,
                sigma_px=self._psf_sigma_px,
                detection_sigma=self._detection_sigma,
                centroid_box_half_px=self._centroid_box_half_px,
                max_sources=self._max_sources,
            )
            n_detected_sources = len(detected)
            self.logger.info(
                'Detected %d source(s); using up to %d catalog stars',
                n_detected_sources,
                n_catalog_predicted,
            )
            if n_detected_sources < 3:
                return self._fail(
                    features=features,
                    reason=f'fewer_than_3_detected_sources (got {n_detected_sources})',
                    diagnostics=StarFieldDiagnostics(
                        n_inliers=0,
                        median_residual_px=0.0,
                        n_detected_sources=n_detected_sources,
                        n_catalog_predicted=n_catalog_predicted,
                        n_triplets_evaluated=0,
                    ),
                    fit_rotation=bool(context.fit_camera_rotation),
                )
            return self._match_and_fit(
                features=features,
                detected=detected,
                ranked_catalog=ranked_catalog,
                context=context,
                n_detected_sources=n_detected_sources,
                n_catalog_predicted=n_catalog_predicted,
            )

    def _match_and_fit(
        self,
        *,
        features: list[NavFeature],
        detected: list[_DetectedSource],
        ranked_catalog: list[NavFeature],
        context: NavContext,
        n_detected_sources: int,
        n_catalog_predicted: int,
    ) -> NavTechniqueResult:
        """RANSAC + Tukey-biweight verification on the prepared cohorts."""
        det_points = [(s.v, s.u) for s in detected]
        det_points_arr = np.asarray(det_points, np.float64)
        det_brightness_rank = list(range(n_detected_sources))  # already sorted
        cat_points = [predicted_vu(f) for f in ranked_catalog]
        cat_points_arr = np.asarray(cat_points, np.float64)
        cat_brightness_rank = list(range(n_catalog_predicted))  # already sorted by SNR
        det_triplets = _enumerate_triplets(det_points, det_brightness_rank)
        cat_triplets = _enumerate_triplets(cat_points, cat_brightness_rank)
        if not det_triplets or not cat_triplets:
            return self._fail(
                features=features,
                reason='no_valid_triplets_after_canonicalisation',
                diagnostics=StarFieldDiagnostics(
                    n_inliers=0,
                    median_residual_px=0.0,
                    n_detected_sources=n_detected_sources,
                    n_catalog_predicted=n_catalog_predicted,
                    n_triplets_evaluated=0,
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        candidates = self._enumerate_candidates(det_triplets, cat_triplets)
        n_triplets_evaluated = len(candidates)
        self.logger.debug(
            '%d detection triplet(s) and %d catalog triplet(s) -> %d hash-distance candidates',
            len(det_triplets),
            len(cat_triplets),
            n_triplets_evaluated,
        )
        best = self._score_candidates(
            candidates=candidates,
            cat_triplets=cat_triplets,
            det_points_arr=det_points_arr,
            cat_points_arr=cat_points_arr,
        )
        if best is None or best[0] < self._min_inliers:
            n_inliers = 0 if best is None else best[0]
            return self._fail(
                features=features,
                reason=(f'too_few_inliers ({n_inliers} < min {self._min_inliers})'),
                diagnostics=StarFieldDiagnostics(
                    n_inliers=n_inliers,
                    median_residual_px=0.0,
                    n_detected_sources=n_detected_sources,
                    n_catalog_predicted=n_catalog_predicted,
                    n_triplets_evaluated=n_triplets_evaluated,
                ),
                fit_rotation=bool(context.fit_camera_rotation),
            )
        n_inliers, correspondences, _coarse_offset = best
        det_inliers = det_points_arr[[d for d, _ in correspondences]]
        cat_inliers = cat_points_arr[[c for _, c in correspondences]]
        fit_rotation = bool(context.fit_camera_rotation)
        if fit_rotation:
            sim_fit, weights = self._similarity_refit(det_inliers, cat_inliers)
            offset_vu = sim_fit.translation_vu
            residuals = sim_fit.residuals_vu
            rotation_rad: float | None = float(sim_fit.rotation_rad)
        else:
            offset_vu, weights, residuals = self._tukey_refit(det_inliers, cat_inliers)
            rotation_rad = None
        residual_distances = np.hypot(residuals[:, 0], residuals[:, 1])
        median_residual_px = float(np.median(residual_distances))
        if fit_rotation:
            cov = self._build_covariance_3dof(
                weights=weights,
                residuals=residuals,
                cat_inliers=cat_inliers,
            )
        else:
            cov = self._build_covariance(weights=weights, residuals=residuals)
        diagnostics = StarFieldDiagnostics(
            n_inliers=n_inliers,
            median_residual_px=median_residual_px,
            n_detected_sources=n_detected_sources,
            n_catalog_predicted=n_catalog_predicted,
            n_triplets_evaluated=n_triplets_evaluated,
        )
        margin_v, margin_u = search_window_for_obs(context)
        max_rotation_rad = math.radians(context.max_rotation_deg)
        rotation_at_edge = fit_rotation and (
            rotation_rad is not None
            and abs(rotation_rad) >= self._rotation_at_edge_fraction * max_rotation_rad
        )
        at_edge = (
            abs(offset_vu[0]) >= margin_v - self._at_edge_tolerance_px
            or abs(offset_vu[1]) >= margin_u - self._at_edge_tolerance_px
            or rotation_at_edge
        )
        confidence = self._evaluate_confidence(
            diagnostics=diagnostics, at_edge=at_edge, spurious=False
        )
        consumed_ids = tuple(ranked_catalog[c].feature_id for _, c in correspondences)
        self.logger.info(
            'Pattern-match offset (%.4f, %.4f) px; %d inlier(s); median residual %.4f px; '
            'confidence %.4f',
            offset_vu[0],
            offset_vu[1],
            n_inliers,
            median_residual_px,
            confidence,
        )
        sigma_rotation_rad: float | None
        if fit_rotation:
            sigma_rotation_rad = float(np.sqrt(max(float(cov[2, 2]), 0.0)))
            self.logger.info(
                'Rotation = %+.4f deg (sigma %.4f deg)%s',
                math.degrees(rotation_rad if rotation_rad is not None else 0.0),
                math.degrees(sigma_rotation_rad),
                ', AT_EDGE' if (rotation_at_edge or at_edge) else '',
            )
        else:
            sigma_rotation_rad = None
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=consumed_ids,
            offset_px=(offset_vu[0], offset_vu[1]),
            covariance_px2=cov,
            confidence=confidence,
            spurious=False,
            at_edge=at_edge,
            diagnostics=diagnostics,
            rotation_rad=rotation_rad,
            sigma_rotation_rad=sigma_rotation_rad,
        )

    def _enumerate_candidates(
        self,
        det_triplets: list[_Triplet],
        cat_triplets: list[_Triplet],
    ) -> list[tuple[float, int, int, int, int, int, int, int]]:
        """Return surviving ``(dist_sq, sorted_det_idx..., a, b, c, cat_idx)`` candidates.

        Each detection triplet is matched against every catalog triplet
        whose hash-distance falls within ``hash_match_tolerance``.  The
        candidate list is sorted by ``(hash_dist_sq, sorted
        detection-source indices ascending, catalog-triplet index)`` —
        per AUTONAV_PLAN.md §33 the sorted-ascending detection tuple
        ``(min, mid, max)`` is the canonical tie-breaker, not the
        canonical (a=brightest, b<c) order, because the
        sorted-ascending key is invariant under brightness re-ranking
        and yields bit-identical iteration across two back-to-back
        invocations on the same obs (Cardinal Principle 3).
        """
        tol_sq = self._hash_match_tolerance * self._hash_match_tolerance
        out: list[tuple[float, int, int, int, int, int, int, int]] = []
        for det in det_triplets:
            sorted_indices = sorted([det.idx_a, det.idx_b, det.idx_c])
            sort_lo, sort_mid, sort_hi = sorted_indices[0], sorted_indices[1], sorted_indices[2]
            for ci, cat in enumerate(cat_triplets):
                dist_sq = _hash_distance_sq(
                    det.hash_v,
                    cat.hash_v,
                    ratio_weight=self._hash_ratio_weight,
                    angle_weight=self._hash_angle_weight,
                )
                if dist_sq <= tol_sq:
                    out.append(
                        (
                            dist_sq,
                            sort_lo,
                            sort_mid,
                            sort_hi,
                            det.idx_a,
                            det.idx_b,
                            det.idx_c,
                            ci,
                        )
                    )
        out.sort()
        return out

    def _score_candidates(
        self,
        *,
        candidates: list[tuple[float, int, int, int, int, int, int, int]],
        cat_triplets: list[_Triplet],
        det_points_arr: NDArrayFloatType,
        cat_points_arr: NDArrayFloatType,
    ) -> tuple[int, list[tuple[int, int]], tuple[float, float]] | None:
        """Score every candidate; return the (n_inliers, correspondences, offset) winner.

        Each (det_triplet, cat_triplet) candidate proposes the
        translation that maps the catalog-triplet centroid onto the
        detection-triplet centroid.  Inliers are counted by greedy
        nearest-neighbour matching under that translation.  The first
        candidate (in sorted order) that ties the best inlier count
        wins, which matches the deterministic-iteration contract.
        """
        best: tuple[int, list[tuple[int, int]], tuple[float, float]] | None = None
        for _dist_sq, _lo, _mid, _hi, det_a, det_b, det_c, cat_pos in candidates:
            cat = cat_triplets[cat_pos]
            offset = _triplet_centroid_offset(
                det_points_arr,
                cat_points_arr,
                det_indices=(det_a, det_b, det_c),
                cat_indices=(cat.idx_a, cat.idx_b, cat.idx_c),
            )
            n_inliers, pairs = _greedy_inlier_count(
                det_points_arr,
                cat_points_arr,
                offset_vu=offset,
                tolerance_px=self._inlier_tolerance_px,
            )
            if best is None or n_inliers > best[0]:
                best = (n_inliers, pairs, offset)
        return best

    def _tukey_refit(
        self,
        det_inliers: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> tuple[tuple[float, float], NDArrayFloatType, NDArrayFloatType]:
        """Refit translation by Tukey-biweight-reweighted least squares.

        Two-pass: (i) unweighted mean to seed residual scale; (ii) one
        biweight-reweighted mean at the conventional 4.685 breakdown
        constant.  Returns the final ``(offset, weights, residuals)``
        triple where ``residuals`` is the per-correspondence
        ``(d - (c + offset))`` array.
        """
        n = det_inliers.shape[0]
        weights0 = np.ones(n, dtype=np.float64)
        offset0 = _solve_translation(det_inliers, cat_inliers, weights0)
        residuals0 = det_inliers - cat_inliers - np.asarray(offset0, np.float64)[None, :]
        distances0 = np.hypot(residuals0[:, 0], residuals0[:, 1])
        scale = float(np.median(distances0)) if distances0.size > 0 else 0.0
        if scale <= 0.0:
            # Perfect fit — no scatter — biweight weights are unity by
            # construction (every residual is zero).
            return offset0, weights0, residuals0
        scaled = distances0 / scale
        weights = tukey_biweight_weights(scaled, c=DEFAULT_TUKEY_C)
        if float(weights.sum()) <= 0.0:
            return offset0, weights0, residuals0
        offset = _solve_translation(det_inliers, cat_inliers, weights)
        residuals = det_inliers - cat_inliers - np.asarray(offset, np.float64)[None, :]
        return offset, weights, residuals

    def _similarity_refit(
        self,
        det_inliers: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> tuple[SimilarityFit, NDArrayFloatType]:
        """Run a Tukey-biweight-reweighted Procrustes / similarity fit.

        Two-pass: an unweighted Kabsch fit seeds the residual scale; a
        biweight-reweighted Kabsch fit at the conventional 4.685
        breakdown constant produces the final ``(rotation, translation)``
        pair.  Falls back to the unweighted fit when every residual
        survives at full weight (a perfect fit) or when the biweight
        weights total zero.

        Returns:
            ``(SimilarityFit, weights)`` — the converged fit plus the
            per-correspondence weights used in the final iteration.
        """
        n = det_inliers.shape[0]
        weights0 = np.ones(n, dtype=np.float64)
        seed = similarity_transform_fit(det_inliers, cat_inliers, weights0)
        distances0 = np.hypot(seed.residuals_vu[:, 0], seed.residuals_vu[:, 1])
        scale = float(np.median(distances0)) if distances0.size > 0 else 0.0
        if scale <= 0.0:
            return seed, weights0
        scaled = distances0 / scale
        weights = tukey_biweight_weights(scaled, c=DEFAULT_TUKEY_C)
        if float(weights.sum()) <= 0.0:
            return seed, weights0
        refined = similarity_transform_fit(det_inliers, cat_inliers, weights)
        return refined, weights

    def _build_covariance(
        self, *, weights: NDArrayFloatType, residuals: NDArrayFloatType, n_params: int = 2
    ) -> NDArrayFloatType:
        """Return the per-axis covariance of the translation estimate.

        Reduced-chi-square weighted-mean form.  With ``N`` weighted
        residuals and ``p = n_params`` fitted parameters (2 for the
        translation-only fit, 3 when an outer rotation is co-fitted),
        the per-axis reduced chi-square is

        ::

            chi2_nu_axis = sum_i w_i * r_axis_i**2 / max(N - p, 1)

        and the weighted-mean variance is ``chi2_nu_axis / sum(w_i)``.
        The ``max(N - p, 1)`` degrees-of-freedom factor inflates the
        variance when fewer points than parameters constrain the fit
        (an under-determined geometry cannot be reported as confident).
        A positive-definite floor of ``1 / sum(w_i)`` (pure
        inverse-precision) keeps the covariance non-degenerate in the
        noise-free case where every residual is zero.  Finally the
        uncalibrated ``model_error_floor_px**2`` is added to the
        diagonal (a no-op at the default 0.0).
        """
        total = float(weights.sum())
        if total <= 0.0:
            return np.eye(2, dtype=np.float64)
        n = residuals.shape[0]
        dof = max(n - n_params, 1)
        chi2_nu_v = float(np.sum(weights * residuals[:, 0] ** 2)) / dof
        chi2_nu_u = float(np.sum(weights * residuals[:, 1] ** 2)) / dof
        floor = 1.0 / total
        model_error = self._model_error_floor_px * self._model_error_floor_px
        cov_v = max(chi2_nu_v / total, floor) + model_error
        cov_u = max(chi2_nu_u / total, floor) + model_error
        return np.diag([cov_v, cov_u]).astype(np.float64)

    def _build_covariance_3dof(
        self,
        *,
        weights: NDArrayFloatType,
        residuals: NDArrayFloatType,
        cat_inliers: NDArrayFloatType,
    ) -> NDArrayFloatType:
        """Return the 3x3 covariance for the similarity-transform fit.

        Translation block follows :meth:`_build_covariance` with ``p =
        3`` (three parameters are co-fitted: dv, du, theta); the
        rotation diagonal uses the reduced-chi-square lever-arm formula

        ::

            sigma_theta**2 = chi2_nu_residual / sum_i w_i * |cat_i - cc|**2

        where ``cc`` is the (weighted) catalog centroid, ``|cat_i - cc|``
        is each point's lever-arm distance from the rotation centre, and

        ::

            chi2_nu_residual = 0.5 * (sum_i w_i r_v_i**2 + sum_i w_i r_u_i**2)
                               / max(N - 3, 1)

        is the pooled per-axis reduced chi-square of the fit residuals.
        Cross-terms are zero — under independent-isotropic-residual
        assumptions the translation and rotation parameters of a
        Procrustes fit are uncorrelated.  Whenever the catalog spread is
        too small to constrain rotation (numerically zero or negative
        spread, e.g. coincident inliers) the rotation diagonal collapses
        to the rotation-unobservable sentinel so ``pinvh`` cleanly drops
        the rotation contribution.
        """
        cov_2x2 = self._build_covariance(weights=weights, residuals=residuals, n_params=3)
        total = float(weights.sum())
        if total <= 0.0:
            return embed_rotation_unobservable(cov_2x2)
        cat_c_v = float(np.sum(weights * cat_inliers[:, 0]) / total)
        cat_c_u = float(np.sum(weights * cat_inliers[:, 1]) / total)
        dv = cat_inliers[:, 0] - cat_c_v
        du = cat_inliers[:, 1] - cat_c_u
        spread = float(np.sum(weights * (dv * dv + du * du)))
        n = residuals.shape[0]
        dof = max(n - 3, 1)
        chi2_v = float(np.sum(weights * residuals[:, 0] ** 2))
        chi2_u = float(np.sum(weights * residuals[:, 1] ** 2))
        chi2_nu_residual = 0.5 * (chi2_v + chi2_u) / dof
        if spread <= 0.0:
            sigma_theta_sq = ROTATION_UNOBSERVABLE_VARIANCE
        else:
            sigma_theta_sq = max(chi2_nu_residual / spread, 1.0 / spread)
        cov = np.zeros((3, 3), dtype=np.float64)
        cov[:2, :2] = cov_2x2
        cov[2, 2] = sigma_theta_sq
        return cov

    def _evaluate_confidence(
        self,
        *,
        diagnostics: StarFieldDiagnostics,
        at_edge: bool,
        spurious: bool,
    ) -> float:
        """Run the YAML confidence formula and emit a per-term breakdown."""
        assert self.confidence_spec is not None
        confidence, breakdown = evaluate_sigmoid_combination(
            self.confidence_spec,
            _StarFieldConfidenceContext(
                at_edge=at_edge, spurious=spurious, diagnostics=diagnostics
            ),
            technique_name=self.name,
            return_breakdown=True,
        )
        log_confidence_breakdown(self.logger, breakdown)
        return float(confidence)

    def _fail(
        self,
        *,
        features: list[NavFeature],
        reason: str,
        diagnostics: StarFieldDiagnostics,
        fit_rotation: bool = False,
    ) -> NavTechniqueResult:
        """Return a zero-confidence spurious result with the supplied reason."""
        self.logger.info('Reporting spurious result: %s', reason)
        cov_2x2 = 1e6 * np.eye(2, dtype=np.float64)
        cov = embed_rotation_unobservable(cov_2x2) if fit_rotation else cov_2x2
        return NavTechniqueResult(
            technique_name=self.name,
            feature_ids=tuple(f.feature_id for f in features),
            offset_px=(0.0, 0.0),
            covariance_px2=cov,
            confidence=0.0,
            spurious=True,
            at_edge=False,
            diagnostics=diagnostics,
            rotation_rad=0.0 if fit_rotation else None,
            sigma_rotation_rad=(rotation_unobservable_sigma_rad() if fit_rotation else None),
        )
