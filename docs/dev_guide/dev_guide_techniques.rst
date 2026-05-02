=====================
Navigation Techniques
=====================

A ``NavTechnique`` consumes a subset of the per-image ``NavFeature`` set
plus a per-image ``NavContext`` and produces one
``NavTechniqueResult`` carrying a calibrated translation offset, a
``2x2`` covariance, a ``[0, 1]`` confidence, and a typed diagnostics
dataclass.  The orchestrator's ensemble combine reconciles every
technique's result into a single ``NavResult``.

This page walks the *distance-transform* pipeline that three techniques
share: ``BodyLimbNav``, ``BodyTerminatorNav``, and ``RingEdgeNav``.

DT-based fitting
================

The three DT-based techniques follow the same five-step pipeline:

.. code-block:: text

    +---------------------------+
    |  source image (extfov)    |
    +---------------------------+
                |
                | Gaussian smooth + Sobel
                v
    +---------------------------+
    |  gradient magnitude       |
    |  + gradient (g_v, g_u)    |     <-- once per image, on NavContext
    +---------------------------+
                |
                | threshold > k * noise_sigma
                | + Canny-style directional NMS
                v
    +---------------------------+
    |  thin edge mask           |
    +---------------------------+
                |
                | distance_transform_edt
                v
    +---------------------------+
    |  image_edge_dt_ext        |     <-- once per image, on NavContext
    +---------------------------+
                |
    +-----------|---------------+
    |  per-technique:           |
    |    - render polyline mask |
    |    - coarse_ncc_search    |
    |    - polarity_filter      |
    |    - lm_subpixel_refine   |
    |      (Tukey-reweighted)   |
    |    - information matrix   |
    |      -> covariance        |
    +---------------------------+

Image-side derivatives
----------------------

``nav.nav_orchestrator.image_derivatives.build_image_edge_dt`` is the
helper the orchestrator calls in ``_make_context``.  It Gaussian-smooths
the source image at ``image_gradient_sigma_px`` (default 1.2 px),
computes the Sobel-of-Gaussian gradient vector, takes the magnitude,
thresholds at ``edge_threshold_k_sigma * image_noise_sigma`` (default
``k = 4``), thins via Canny-style directional non-maximum suppression
(comparing each candidate against its two neighbours along the local
gradient direction), and finally takes the truncated distance
transform.  ``compute_image_gradient_vu`` returns the per-pixel
``(g_v, g_u)`` gradient vector image consumed by the polarity filter.

The orchestrator caches all three products on the per-image
``NavContext`` (``image_gradient_ext``, ``image_gradient_vu_ext``,
``image_edge_dt_ext``) so every DT technique reads them rather than
recomputing.

Coarse search and LM refinement
-------------------------------

``nav.nav_technique.dt_fitting`` exposes the five shared helpers:

``coarse_ncc_search``
    Cross-correlates the polyline edge mask against the image edge mask
    over a bounded integer ``(dv, du)`` search window and returns the
    integer-pixel argmax.  Both inputs are binary, so this reduces to
    overlap counting and is exact under the per-shift NCC normaliser.

``polarity_filter``
    Per-vertex gradient-direction agreement test.  At each vertex's
    current position, samples the image gradient vector and dots it with
    the model's polarity normal; vertices with ``dot <= 0`` are rejected
    and assigned an effectively-infinite residual so the Tukey biweight
    zeroes their contribution.  Strict inequality, per the design.

``tukey_biweight_weights``
    Holland-Welsch redescender.  ``w_i = (1 - (r_i / c)^2)^2`` for
    ``|r_i| <= c``, else 0.  ``c = 4.685`` gives 95 % asymptotic
    efficiency on Gaussian residuals when the residuals are scaled by
    their robust scale.

``lm_subpixel_refine``
    Levenberg-Marquardt minimisation of
    ``sum_i w_i * DT(R(theta) (vert_i - pivot) + pivot + (dv, du))^2``
    with iteratively-reweighted Tukey weights.  Damping ``lambda=1e-3``,
    multiplicative update on accept / reject, max 30 iterations,
    termination when the combined step norm
    ``sqrt(d_dv^2 + d_du^2 + (d_theta * pivot_dist)^2)`` drops below
    ``1e-3`` px.  Translation-only by default; rotation enabled with
    ``fit_rotation=True`` and a documented ``pivot_distance_px``.

``information_matrix_to_covariance``
    Returns ``pinvh(J^T diag(w) J, rtol=1e-9)`` so rank-deficient
    Jacobians (from flat ring polylines or other unobservable axes)
    propagate honestly into the per-technique covariance.  Same
    ``rtol`` as the orchestrator's ensemble combine.

Per-technique specifics
=======================

BodyLimbNav
-----------

Accepts ``LIMB_ARC`` features.  ``is_feasible`` returns True when at
least one limb arc has ``visible_arc_px >= LIMB_MIN_ARC_PX`` (30).
Per-vertex weight is ``1 / sigma_normal_per_vertex_px**2``.  Polarity
filtering is enabled — the model's geometric outward normals are
negated before being passed in (the typical bright-body case has the
image gradient pointing into the silhouette, opposite the geometric
outward).  Spurious detection fires when the converged DT RMS exceeds
the larger of ``SPURIOUS_DT_FLOOR_PX = 3`` px and
``5 * min(sigma_normal_per_vertex_px)`` or the Tukey inlier count
drops below ``SPURIOUS_MIN_INLIERS = 6``.

Confidence spec coefficients (placeholders, calibrated against the
image library):

- ``alpha0 = -1``
- ``alpha(visible_limb_arc_fraction) = 3``
- ``alpha(dt_fit_rms_px) = -1.5``
- ``alpha(visible_arc_px / 100, capped at 1) = 0.4``
- hard zero when ``at_edge`` is True

BodyTerminatorNav
-----------------

Accepts ``TERMINATOR_ARC`` features.  Same algorithm as ``BodyLimbNav``
with two terminator-specific changes: each body's per-vertex sigmas are
collapsed to a single per-body scalar (the body's mean), and the
confidence spec carries ``mean_phase_angle_factor`` and
``mean_albedo_penalty`` terms so high-albedo / low-phase scenes
collapse below the rank threshold.

RingEdgeNav
-----------

Accepts ``RING_EDGE`` features.  Polarity is intentionally disabled
because the ring catalog does not yet distinguish bright-ring vs
dark-gap edges (deferred polarity work).  When every input edge has
``is_straight_line=True`` the combined Jacobian is rank-deficient — the
along-edge axis is unobservable — and the technique reports
``is_rank_1=True`` on its diagnostics.  The returned 2x2 covariance is
the M-estimator pseudoinverse, honestly rank-deficient on flat-only
scenes; the ensemble combine fuses it with any orthogonal feature
(star, body limb, body blob) before declaring a final answer.

Body-disc and body-blob techniques
==================================

Two body-side techniques do not fit the DT-fitting shape: full-disc NCC
correlation (``BodyDiscCorrelateNav``) and brightness-weighted-moment
centroid fitting (``BodyBlobNav``).  Both consume a single feature type
and produce one combined translation, so multi-body inputs constrain the
fit jointly without per-body offset ambiguity.

BodyDiscCorrelateNav
--------------------

Accepts ``BODY_DISC`` features.  Each ``BODY_DISC`` carries a
postage-stamp Lambert-shaded disc template (``template_img``) plus a
boolean ``template_mask`` and a ``bbox_extfov_vu`` placement.  The
technique:

1. Fuses the per-body templates into a single extfov-shaped composite
   via :func:`nav.feature.composition.compose_template_features`.  The
   helper sorts features by ``subject_range_km`` ascending so closer
   bodies overwrite farther bodies on overlap (Z-buffer paint per
   Part 0 §2 of the autonav design).  The combined mask is the OR of
   per-body masks.
2. Runs :func:`nav.support.correlate.navigate_with_pyramid_kpeaks`
   against the composite with ``use_gradient='auto'``.  Auto mode tries
   both raw-intensity and gradient-magnitude NCC and keeps the better
   result by ``non-spurious > not-at-edge > higher-quality`` ordering.
3. Reads the pyramid wrapper's ``offset``, ``cov``, ``quality``,
   ``consistency``, ``spurious``, ``at_edge``, and ``used_gradient``
   fields directly into ``BodyDiscDiagnostics``.

``is_feasible`` returns True iff at least one input ``BODY_DISC``
feature carries a template payload.  The technique reports
``spurious=True`` when the pyramid's quality metric falls below
``quality_thresh`` or pyramid consistency drifts past ``consistency_tol``;
``at_edge=True`` when the converged peak lies within 2 px of the search
window edge.

Confidence spec coefficients (placeholders pending calibration
against the operator-curated image library):

- ``alpha0 = -2``
- ``alpha(ncc_peak / 6, capped at 1) = 1.5``
- ``alpha(consistency_px / 2) = -1``
- ``alpha(body_count / 3, capped at 1) = 0.4``
- ``alpha(peak_to_runner_up_ratio / 2, capped at 1) = 0.0`` — wired in
  but disabled until calibration tunes the alpha
- hard zero when ``at_edge`` or ``spurious``

The coefficients live in
``src/nav/config_files/config_510_techniques.yaml`` under
``techniques.BodyDiscCorrelateNav`` and are loaded into
``ConfidenceSpec`` at config-load time by
:func:`nav.nav_technique.confidence_config.load_confidence_spec`.
Re-tune by editing the YAML; no code change required.

Diagnostics fields:

- ``ncc_peak``: pyramid-wrapper quality metric (PSR by default).
- ``peak_to_runner_up_ratio``: ratio of the winning peak's quality to
  the runner-up's, derived from the pyramid wrapper's
  ``top_k_peaks`` field (sorted by quality descending).  Returns
  ``1.0`` when only one peak survives non-maximum suppression — the
  unambiguous-peak case.
- ``consistency_px``: maximum Euclidean drift across pyramid levels —
  ``np.max(np.linalg.norm(level_shifts - final_prior, axis=1))`` over
  the coarse-to-fine cascade in
  :func:`nav.support.correlate.navigate_with_pyramid_kpeaks`.
- ``used_gradient``: ``True`` when auto-mode picked the gradient pass.
- ``body_count``: number of fused BODY_DISC features.

Infeasibility cases:

- No input feature carries a template.

BodyBlobNav
-----------

Accepts ``BODY_BLOB`` features.  Each blob carries only a predicted
centroid and a predicted bounding box (no template — the body is
either irregular or under-resolved, so a Lambert template cannot be
rendered usefully).  The technique:

1. For each blob, computes a brightness-weighted-moment centroid over
   every above-noise pixel inside the predicted bbox.  Above-noise
   means ``image_DN > 3 * image_noise_sigma``; background DN never
   biases the moment.
2. Computes the per-blob residual ``observed_centroid - predicted_center``
   and forms a precision-weighted joint translation (the simple
   weighted mean across all surviving blobs).
3. Per-blob weight ``w_i = N_lit_i * SNR_i^2 / radius_i^2`` from the
   BODY_BLOB centroid CRLB derivation; the joint covariance is
   diagonal with per-axis variance ``1 / sum(w)`` floored to the
   inverse-precision and inflated by residual scatter when ``N >= 2``.

``is_feasible`` returns True iff at least one input ``BODY_BLOB``
carries a non-zero ``predicted_diameter_px``.  The technique flags
``spurious=True`` when no blob has any above-noise signal in its
predicted bbox; ``at_edge=True`` when the converged offset lies within
1 px of the search-window axis bounds.

The confidence formula intrinsically caps at ``0.4`` (the BODY_BLOB
reliability ceiling).  A brightness-weighted centroid is weaker than
a limb fit by design; the cap ensures the technique cannot dominate
the ensemble even when every term saturates.  Coefficient
placeholders pending calibration against the operator-curated image
library:

- ``alpha0 = -1``
- ``alpha(body_snr_inside_predicted_bbox / 4, capped at 1) = 0.5``
- ``alpha((body_extent_px - 8) / 8, capped at 1) = 1``
- ``alpha(blob_count / 3, capped at 1) = 0.4``
- hard zero when ``at_edge``
- hard cap ``0.4`` after the sigmoid

The coefficients live in
``src/nav/config_files/config_510_techniques.yaml`` under
``techniques.BodyBlobNav`` and are loaded into ``ConfidenceSpec`` at
config-load time by
:func:`nav.nav_technique.confidence_config.load_confidence_spec`.

Sample confidence breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running the BodyBlobNav unit tests with DEBUG logging on a 3-blob
synthetic scene (mean-extent 16 px, mean-SNR 8) produces a per-term
trace of the form:

.. code-block:: text

   Confidence breakdown: alpha0=-1.000, sigmoid_arg=0.900 -> confidence=0.4000 (hard_cap applied)
     term 'body_snr_inside_predicted_bbox': raw=8.00, normalized=1.000, alpha=+0.500 -> contribution=+0.500
     term 'body_extent_px':                  raw=16.00, normalized=1.000, alpha=+1.000 -> contribution=+1.000
     term 'blob_count':                      raw=3.00, normalized=1.000, alpha=+0.400 -> contribution=+0.400

The sigmoid argument before clamping is ``-1.0 + 0.5 + 1.0 + 0.4 =
0.9`` (matching the three terms in ``_BODY_BLOB_CONFIDENCE_SPEC``),
the sigmoid evaluates to ``0.711``, and the ``hard_cap = 0.4``
post-sigmoid clamp drops the headline confidence to the BODY_BLOB
ceiling.

Diagnostics fields:

- ``body_snr_inside_predicted_bbox``: mean SNR of above-noise pixels
  inside each predicted bbox, averaged across consumed blobs.
- ``body_extent_px``: mean predicted disc diameter in pixels across
  consumed blobs.
- ``blob_count``: number of blobs that contributed (after dropping
  blobs with no above-noise signal).
- ``residual_px``: RMS scatter of per-blob ``observed - predicted``
  vectors around the joint mean.

Infeasibility cases:

- No input feature carries a non-zero predicted diameter.

Ring-annulus technique
======================

When the rings model emits a ``RING_ANNULUS`` feature instead of
per-edge ``RING_EDGE`` polylines (because adjacent ring edges compress
within the per-planet
``feature_emission.ring_annulus.max_radial_px`` threshold in
``config_510_techniques.yaml``, or because the system-level km/px
threshold fires on a low-resolution ring scene), the per-planet
multi-ring composite template lives on ``NavFeature.template_img``.
The ring-annulus technique mirrors ``BodyDiscCorrelateNav`` for that
template payload.

RingAnnulusNav
--------------

Accepts ``RING_ANNULUS`` features.  Each ``RING_ANNULUS`` carries a
multi-ring composite template (``template_img``) plus a boolean
``template_mask`` and a ``bbox_extfov_vu`` placement.  The technique:

1. Fuses the per-planet annulus templates into a single extfov-shaped
   composite via :func:`nav.feature.composition.compose_template_features`.
   The helper sorts features by ``subject_range_km`` ascending so closer
   ring systems overwrite farther ones on overlap (Z-buffer paint).  The
   combined mask is the OR of per-planet masks.
2. Runs :func:`nav.support.correlate.navigate_with_pyramid_kpeaks`
   against the composite with ``use_gradient='auto'``.  Auto mode tries
   both raw-intensity and gradient-magnitude NCC and keeps the better
   result by ``non-spurious > not-at-edge > higher-quality`` ordering;
   raw wins on broad-brightness-gradient ring geometries (low-resolution
   Saturn rings where the C-ring is uniformly dim) and gradient wins
   when sharp ringlet edges dominate.
3. Reads the pyramid wrapper's ``offset``, ``cov``, ``quality``,
   ``spurious``, ``at_edge``, and ``used_gradient`` fields directly into
   ``RingAnnulusDiagnostics``.

``is_feasible`` returns True iff at least one input ``RING_ANNULUS``
feature carries a template payload.  The technique handles
``len(features) > 1`` for the rare multi-planet case (one
``RING_ANNULUS`` per detectable ring system).  It reports
``spurious=True`` when the pyramid's quality metric falls below
``quality_thresh`` and ``at_edge=True`` when the converged peak lies
within 2 px of the search window edge.

Confidence spec coefficients (placeholders pending calibration against
the operator-curated image library):

- ``alpha0 = -2``
- ``alpha(ncc_peak / 6, capped at 1) = 1.5``
- ``alpha(annulus_count / 2, capped at 1) = 0.4``
- ``alpha(peak_to_runner_up_ratio / 2, capped at 1) = 0.0`` — wired in
  but disabled until calibration tunes the alpha
- hard zero when ``at_edge`` or ``spurious``

The coefficients live in
``src/nav/config_files/config_510_techniques.yaml`` under
``techniques.RingAnnulusNav`` and are loaded into ``ConfidenceSpec`` at
config-load time by
:func:`nav.nav_technique.confidence_config.load_confidence_spec`.

Diagnostics fields:

- ``ncc_peak``: pyramid-wrapper quality metric (PSR by default).
- ``peak_to_runner_up_ratio``: ratio of the winning peak's quality to
  the runner-up's, derived from the pyramid wrapper's ``top_k_peaks``
  field (sorted by quality descending).  Returns ``1.0`` when only one
  peak survives non-maximum suppression — the unambiguous-peak case.
- ``annulus_count``: number of fused ``RING_ANNULUS`` features (one
  per detectable ring system).
- ``used_gradient``: ``True`` when auto-mode picked the gradient pass.

Infeasibility cases:

- No input feature carries a template.

Star techniques
===============

Three techniques consume STAR features.  Each runs against a different
star-count regime, and the orchestrator picks per scene by feasibility
gates.

StarUniqueMatchNav
------------------

Accepts ``STAR`` features.  Runs in pass 1 (no prior required).  Two
paths share one technique:

- **One-star path.** When the catalog reduction yields one star whose
  predicted SNR is at least
  ``brightness_margin_to_next_catalog_star_mag`` (default 1.5 mag)
  brighter than the next-brightest predictable star, the brightest
  detection inside its search window is unambiguously its match.  The
  offset is the centroid minus the prediction; confidence is capped
  at ``one_star_confidence_cap`` (default 0.7).
- **Two-star path.**  With two predictable stars, the technique tries
  both detection-to-prediction assignments and picks the one whose
  joint residual is smaller.  Confidence is capped at
  ``two_star_confidence_cap`` (default 0.8).

``is_feasible`` returns True when at least one usable STAR feature is
present (occluded / cosmic-ray-masked stars are filtered out
upstream).  The technique uses a localised brightness-peak +
brightness-weighted-moment centroid inside a per-prediction window; no
global ``detect_sources`` call is needed.

StarRefineNav
-------------

Accepts ``STAR`` features.  Runs in pass 2 (``requires_prior=True``).
For each predicted catalog star, the technique:

1. Shifts the prediction by the prior offset.
2. Looks for a brightness peak in a small refine window around the
   shifted prediction.
3. Fits a brightness-weighted moment for the sub-pixel centroid.
4. Computes per-star residuals and averages them in inverse-variance
   fashion.

Stars whose detection sits more than ``max_per_star_residual_px`` from
the shifted prediction are dropped before the joint average.  The
refined offset is reported as ``delta + prior_offset`` (the ensemble
combine sees the absolute offset).  The covariance reflects the
residual scatter across surviving stars; with two or more inliers the
technique reports the actual scatter, with one inlier the per-feature
CRLB floor.

StarFieldFromCatalogNav
-----------------------

Accepts ``STAR`` features.  Runs in pass 1 (no prior required).
Requires at least three usable STAR features.  Algorithm:

1. **Source detection.** Matched-filter the image against a Gaussian
   PSF kernel sized by ``psf_sigma_px`` /
   ``centroid_box_half_px``; pixels that are local maxima above
   ``detection_sigma * image_noise_sigma`` clear the gate.  Each
   surviving peak contributes a brightness-weighted moment centroid.
   The brightest ``max_sources`` (default 30) survivors feed the
   matcher; the cap keeps the M^3 triplet enumeration bounded.
2. **Triplet hashing.**  For each unordered triplet of detected
   sources ``{A, B, C}`` with ``A`` brightest, the hash
   ``(d_AB / d_AC, d_BC / d_AC, ∠BAC)`` is computed.  The same
   canonicalisation runs on catalog triplets, ranked by predicted
   SNR.  All three hash components are similarity-invariant
   (translation, rotation, uniform scale), so the matcher recovers
   correspondences without already knowing the transform.
3. **RANSAC.**  Each (det_triplet, cat_triplet) candidate within
   ``hash_match_tolerance`` (weighted Euclidean in the hash space)
   proposes the translation
   ``mean(detection_vertices) - mean(catalog_vertices)``.  Candidates
   are evaluated in deterministic sorted order — ``(hash_distance
   ascending, sorted detection-source indices ascending,
   catalog-triplet index ascending)`` — so the matcher's choice of
   winner is bit-identical across two back-to-back invocations on
   the same obs (Cardinal Principle 3, Part 3 §"Determinism in
   RANSAC").  Inlier count is the score; the winner needs at least
   ``pattern_match_min_inliers`` (default 6).
4. **Verification.**  With the winning inlier correspondences, a
   Tukey-biweight-reweighted weighted least-squares mean refits the
   translation; the per-axis variance of the surviving residuals is
   the reported covariance.

``is_feasible`` returns True when ≥ 3 usable STAR features are
present (below that the matcher cannot form a triplet).
``StarUniqueMatchNav`` and ``StarFieldFromCatalogNav`` are mutually
exclusive at feasibility time: 1–2 predictable stars favour the
unique-match path, ≥ 3 favour the triplet matcher.

Phase 8 ships translation-only fitting; the rotation-enabled
3-DoF Procrustes path lights up in Phase 9 when
``fit_camera_rotation`` is enabled per instrument.

Confidence spec coefficients (placeholders pending Phase 10
calibration against the operator-curated image library):

- ``alpha0 = -2``
- ``alpha((n_inliers - 6) / 6, capped at 1) = 1``
- ``alpha(median_residual_px) = -1``
- ``alpha(n_detected_sources / 30, capped at 1) = 0`` — wired in but
  disabled until calibration tunes the alpha
- ``alpha(n_catalog_predicted / 30, capped at 1) = 0`` — wired in but
  disabled until calibration tunes the alpha
- hard zero when ``at_edge`` or ``spurious``

The coefficients live in
``src/nav/config_files/config_510_techniques.yaml`` under
``techniques.StarFieldFromCatalogNav`` and are loaded into
``ConfidenceSpec`` at config-load time by
:func:`nav.nav_technique.confidence_config.load_confidence_spec`.

Diagnostics fields:

- ``n_inliers``: surviving detection-to-catalog correspondences after
  the Tukey refit.
- ``median_residual_px``: median Euclidean residual on the inliers.
- ``n_detected_sources``: bright peaks the detector kept (capped at
  ``max_sources``).
- ``n_catalog_predicted``: catalog stars the matcher considered (also
  capped at ``max_sources``).
- ``n_triplets_evaluated``: candidate (det_triplet, cat_triplet)
  pairs whose hash distance fell within the match tolerance.

Infeasibility cases:

- Fewer than three usable STAR features (the matcher cannot form a
  triplet).

Body-extractor emission gate
============================

The body NavModel picks which feature types to emit per-image,
per-body:

1. If ``limb_uncertainty_px <= LIMB_ARC_MAX_UNCERTAINTY_PX`` (default
   3 px) and the limb sampler has surviving vertices: emit
   ``LIMB_ARC``.
2. Else if ``predicted_diameter_px >= max(BODY_BLOB_MIN_DIAMETER_PX,
   shape.min_blob_diameter_px)`` (default 8 px, with per-body
   overrides for highly-irregular satellites and gas giants): emit
   ``BODY_BLOB``.
3. Else: emit no body feature for the image.

``limb_uncertainty_px = ellipsoid_residual_km / km_per_px_at_limb`` —
per-image, per-body.  The same body becomes a usable limb arc at one
distance and a blob at another (worked example: an irregular
ring-shepherd moon at 100 km/px has ``limb_uncertainty_px = 0.08``
and emits ``LIMB_ARC``; the same moon at 1 km/px has
``limb_uncertainty_px = 8`` and emits ``BODY_BLOB`` instead).

Logging
=======

Each technique's ``navigate`` body opens a pdslogger section
(``with self.logger.open(f'TECHNIQUE: {self.name}'):``) so the
per-image log file delimits each technique's contribution clearly.
The section logs the consumed feature count, the coarse-search
integer offset, and the converged offset / RMS / inlier count /
confidence on a single INFO line.  The pdslogger
``IMAGE_LOGGER`` is the only logger this codebase uses; the standard
library ``logging`` module is not imported.

See also
========

- :doc:`dev_guide_uncertainty` — derivation of the M-estimator
  information-matrix to covariance step that turns the LM Jacobian at
  convergence into the per-technique 2x2 (or 3x3) covariance reported
  on every ``NavTechniqueResult``.  ``BodyDiscCorrelateNav``'s
  covariance is the Fisher / CRLB covariance produced by
  :func:`nav.support.correlate.fisher_covariance` inside
  :func:`nav.support.correlate.evaluate_candidate` and forwarded
  through ``navigate_with_pyramid_kpeaks``; both ``BodyLimbNav`` and
  ``BodyTerminatorNav`` derive theirs from the Tukey-reweighted
  information matrix; ``BodyBlobNav`` derives a diagonal
  precision-weighted-mean covariance from the per-blob CRLB weights.
- :func:`nav.feature.composition.compose_template_features` — the
  Z-buffer paint helper that ``BodyDiscCorrelateNav`` uses to fuse
  per-body templates into a single composite for the NCC.
- :func:`nav.support.correlate.navigate_with_pyramid_kpeaks` — the
  shared pyramid-NCC entry point.  Returns ``offset``, ``cov``,
  ``quality``, ``consistency``, ``spurious``, ``at_edge``,
  ``used_gradient``, and ``top_k_peaks`` (per-peak telemetry the
  ``BodyDiscCorrelateNav`` peak-to-runner-up diagnostic reads).
