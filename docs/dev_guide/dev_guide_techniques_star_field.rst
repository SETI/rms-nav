==========================================================
Star Field Pattern Match (StarFieldFromCatalogNav)
==========================================================

Overview
========

:class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` is the pass-1
multi-star pattern matcher. It hashes catalog and detection triplets into a translation- and
rotation-invariant feature space, finds correspondences via a KD-tree match in that space,
and runs a RANSAC similarity-transform fit to recover the translation (plus optional in-plane
rotation) that maps the predicted catalog cohort onto the observed detections. The technique
runs without a prior — it is the ensemble's primary star path on scenes where the SPICE
pointing error is too large for
:class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`'s search
window to contain the brightest predictable star.

Feasibility passes when the predictable-star cohort has at least three usable
:data:`~nav.feature.feature_type.NavFeatureType.STAR` features (the matcher cannot form a
single triplet below that count); feasibility fails otherwise.

Theory
======

The technique solves the global star-pattern-matching problem from scratch — without a prior
offset — by reducing translation- and rotation-invariant pattern matching to a high-dimensional
nearest-neighbour search.

Triplet hashing
---------------

For every triplet of three predictable catalog stars the matcher computes a translation- and
rotation-invariant hash: the two ratios of side lengths (sorted longest-to-shortest), plus
the included angle at the longest-side vertex. The same hash is computed for every triplet
of three detected sources in the image. Triplets with matching hashes (within a small
tolerance) are candidate correspondences.

Source detection
----------------

Detections come from a matched-filter search over the extfov image: the image is correlated
with a Gaussian PSF kernel of the configured ``psf_sigma_px``, peaks above
``detection_sigma * image_noise_sigma`` are kept, and a brightness-weighted moment around
each peak gives the sub-pixel centroid. Up to ``max_sources`` brightest detections feed the
hash-space search; the catalog side is similarly capped.

KD-tree match
-------------

The catalog and detection triplet hashes are loaded into a KD-tree in
``(ratio_short_to_long, ratio_mid_to_long, angle_radians)`` space, weighted per axis
(``hash_ratio_weight``, ``hash_angle_weight``); a query of every catalog hash against the
detection KD-tree returns the nearest detection triplet within ``hash_match_tolerance``.
Each match implies a per-star correspondence between three catalog vertices and three
detection vertices.

RANSAC inlier validation
------------------------

Each candidate correspondence proposes a similarity transform (rotation, translation, and an
implied unit scale). The matcher applies the proposed transform to every catalog star and
counts how many predicted positions land within ``inlier_tolerance_px`` of a detection.
The transform with the most inliers wins; below ``pattern_match_min_inliers`` the technique
reports spurious.

Similarity-transform refit
--------------------------

Once the inlier set is known, the matcher runs a weighted Procrustes / Kabsch fit on the
inlier correspondences (using the per-star inverse-trace covariance weights) to refine the
similarity transform. The translation read off the fit is the reported offset; the
rotation, when present, is the converged angle about the catalog-side weighted centroid.

Per-axis covariance
-------------------

The reported translation covariance is derived from the inlier residual scatter and the
catalog-side centroid spread, per the same precision-weighted-mean form
:class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` uses; see
:doc:`dev_guide_techniques_star_refine` for the algebra. When the per-instrument
camera-rotation flag is on and the inlier count supports it, a 3x3 covariance with the
rotation diagonal is reported; otherwise the 2x2 translation block is reported on its own
or embedded in the rank-deficient 3x3.

Restrictions and assumptions
----------------------------

- The matcher needs at least three usable stars to form one triplet. Below that count the
  technique reports infeasible without running.
- The hash-space tolerance is empirically tuned to a typical centroid sigma of ~0.1 px;
  significantly noisier centroids may cause the KD-tree match to miss correct
  correspondences.
- The matcher is translation- and rotation-invariant by construction but not scale-invariant;
  a Procrustes refit explicitly forces unit scale, which is correct for navigation but means
  the fit will not produce a useful result when the per-instrument plate scale is wrong.
- Triplet hashing scales as :math:`O(M^{3})` per side where :math:`M` is the per-side cap
  (``max_sources``). At the default ``max_sources = 30`` the cost is ~27,000 triplets per
  side.

Sources of uncertainty
----------------------

The reported covariance is the precision-weighted-mean form derived from inlier residual
scatter; it does not capture systematic biases (a per-star centroid offset that affects every
detection equally moves the global solution by the same offset and is invisible to the
inlier scatter), and it does not capture catalog-side errors (a star with a wrong predicted
position bumps the inlier count by zero or one and the bulk of the fit is unaffected). When
the converged offset sits within the at-edge tolerance of any axis bound, or when the
rotation parameter is at the configured fraction of its cap, the result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. When the inlier count falls below
``pattern_match_min_inliers`` the result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.StarFieldFromCatalogNav.tuning``
in ``src/nav/config_files/config_510_techniques.yaml``.

- ``max_sources`` — int, default ``30`` (count). Maximum number of brightest detected
  sources / brightest catalog stars per side feeding the matcher. Triplet count is
  :math:`O(M^{3})`, so 30 keeps the candidate list bounded at ~27,000 triplets per side.
- ``detection_sigma`` — float, default ``4.0`` (dimensionless). Detection-threshold
  multiplier on the per-image noise sigma; matched-filter peaks above this threshold are
  candidate detections.
- ``psf_sigma_px`` — float, default ``1.0`` px. Gaussian PSF sigma for the matched-filter
  kernel. Matches the typical ungroomed star PSF; per-instrument overrides may tighten or
  loosen this once the integration library exercises the full instrument set.
- ``centroid_box_half_px`` — int, default ``3`` px. Half-width of the brightness-weighted
  moment box around each accepted peak.
- ``hash_match_tolerance`` — float, default ``0.05`` (dimensionless). KD-tree match radius
  in the (ratio, ratio, angle_rad) hash space; weighted Euclidean. Empirical default for
  centroid sigma ~0.1 px against typical triplet scales.
- ``hash_ratio_weight`` — float, default ``1.0`` (dimensionless). Per-axis weight on the
  ratio dimensions in the hash distance metric.
- ``hash_angle_weight`` — float, default ``1.0`` (dimensionless). Per-axis weight on the
  angle dimension in the hash distance metric.
- ``inlier_tolerance_px`` — float, default ``2.0`` px. Maximum residual for a
  detection-to-catalog correspondence to count as an inlier under a candidate similarity
  transform.
- ``pattern_match_min_inliers`` — int, default ``6`` (count). Minimum inlier count for the
  matcher to accept a transform; below this the technique returns spurious. Must be at
  least 3 (the matcher needs at least one triplet per side).
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses
  this fraction of the per-image
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/nav/config_files/config_4N0_inst_*.yaml`` do not
override any of these knobs.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. Spec is
``techniques.StarFieldFromCatalogNav``; consumes attributes off
:class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics` plus
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers` — alpha = 1.0,
  offset = 6.0, divisor = 6.0, cap at 1.0. Number of detection-to-catalog inliers after
  RANSAC. Saturates at 12 inliers (offset = 6 plus full cap).
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px` —
  alpha = -1.0, offset = 0.0, divisor = 1.0, no cap. Median position residual on inliers.
  Larger residuals pull confidence down.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources` —
  alpha = 0.0, offset = 0.0, divisor = 30.0, cap at 1.0. Number of bright sources detected
  in the image. Carries no weight in the current confidence formula; the wiring is in place so a downstream
  recalibration can tune the alpha.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted` —
  alpha = 0.0, offset = 0.0, divisor = 30.0, cap at 1.0. Number of catalog stars in the
  extfov. Same posture as the detected-sources term.

Hard-zero gate: :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = -2.0`. No post-sigmoid ``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_star_field.py`` —
  :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` and the
  hashing / RANSAC / fit helpers.
- ``src/nav/nav_technique/_star_helpers.py`` — package-private helpers
  ``usable_stars`` (filter), ``local_centroid`` (per-source centroid), and
  ``similarity_transform_fit`` (Kabsch / Procrustes solve).
- ``src/nav/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/nav/nav_technique/diagnostics.py`` —
  :class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class
:class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`, base
:class:`~nav.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique.NavTechnique.name` —
  ``'StarFieldFromCatalogNav'``.
- :attr:`~nav.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset({STAR})``.
- :attr:`~nav.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- :attr:`~nav.nav_technique.nav_technique.NavTechnique.confidence_attributes` —
  ``{'at_edge', 'spurious', 'n_inliers', 'median_residual_px', 'n_detected_sources',
  'n_catalog_predicted'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.navigate`.

Diagnostics
-----------

:class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics`:

- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers` — number of
  detection-to-catalog inliers. Consumed by the confidence formula and the spurious-
  detection gate.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px` — median
  position residual on inliers. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources` — number
  of bright sources detected in the image.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted` — number
  of catalog stars in the extfov.
- :attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_triplets_evaluated` —
  number of triplet candidates considered by RANSAC.

Call path
---------

Call path traced through
:meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.navigate`:

1. Open a logged section. Filter the offered features down to ``usable_stars``
   (predictable, not in a body silhouette, not in a saturation / cosmic-ray mask) and pull
   the predicted catalog positions / SNR off the per-feature
   :attr:`~nav.feature.feature.NavFeature.geometry` and
   :attr:`~nav.feature.feature.NavFeature.flags`.
2. Run the matched-filter detection over
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_ext` to find the brightest
   sources. Cap the catalog and detection cohorts at ``max_sources`` each.
3. Compute every catalog and detection triplet's (ratio, ratio, angle) hash. Load both
   sets into a KD-tree and find correspondences within ``hash_match_tolerance``.
4. RANSAC: for each candidate correspondence triplet, propose a similarity transform and
   count inlier matches across the full catalog cohort using ``inlier_tolerance_px``.
   Keep the transform with the most inliers.

   - **Below min_inliers.**  Return a spurious zero-confidence result with the diagnostic
     fields populated for the JSON sidecar.

5. Run a weighted Procrustes / Kabsch fit on the inlier correspondences via
   ``similarity_transform_fit``. The fit returns the rotation and translation; the
   translation is the reported offset.
6. Result-shape branches on
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` and the inlier
   count:

   - **Translation only** (``fit_camera_rotation`` false). The (2, 2) covariance comes
     from the inlier residual scatter scaled by the catalog-side centroid spread.
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit, two or more inliers.**  The (3, 3) covariance has the per-axis
     translation variances and the rotation variance derived from the inlier residual
     scatter against the catalog-side spread (same algebra as
     :class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` documented at
     :doc:`dev_guide_techniques_star_refine`).
   - **Rotation fit, one inlier.**  The (2, 2) translation block is embedded in a
     rank-deficient (3, 3) via
     :func:`~nav.nav_technique.nav_technique.embed_rotation_unobservable`;
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is ``0.0``
     and the sigma is the rotation-unobservable sentinel.

7. Apply the at-edge tests against the search-window axis bounds and the rotation cap.
8. Build a :class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics`, evaluate the
   confidence spec via :func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the breakdown via
   :func:`~nav.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~nav.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each per-star contribution at audit time.

Examples
========

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV. The stars model emits one ``STAR`` feature per
    predictable catalog star;
    :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` runs the
    triplet hash against the detected sources, lands on a similarity transform with several
    inliers, and reports a translation against the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px. The pass-2
    :class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` consumes this prior
    and polishes the offset using the full predictable cohort; see
    :doc:`dev_guide_techniques_star_refine` for that walk-through.

``stars_plus_body`` (Cassini long-exposure background-stars scene class)
    One body and at least three usable catalog stars in the same FOV. The stars model emits
    one ``STAR`` feature per predictable catalog star whose predicted position lies outside
    the body silhouette and outside any saturation / cosmic-ray mask; the body model emits
    a ``LIMB_ARC`` (or ``BODY_BLOB``) for the body. On pass 1, the
    :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` consumes the body's
    feature first and the orchestrator's ensemble combine populates the per-image prior
    from the limb-derived offset.
    :class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` runs in
    parallel on the star cohort: the triplet hash matches the predicted catalog stars
    against detected sources, the RANSAC inlier set lands on the same translation as the
    body fit, and the ensemble combine tightens the per-image covariance. On pass 2 the
    :class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav` consumes the
    cohort and polishes the offset further.

``faint_stars`` (Galileo SSI / Voyager outer-leg scene class)
    Every catalog star in the FOV has predicted SNR below the per-instrument detection
    floor. The stars model emits no ``STAR`` features whose predicted SNR clears the
    reliability gate. The technique's
    :meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.is_feasible`
    fails with reason ``no_usable_stars`` and the technique skips its navigate pass
    entirely. The orchestrator falls back to whichever body- or ring-derived technique is
    feasible on the scene and surfaces the per-technique infeasibility on the per-image
    :class:`~nav.nav_orchestrator.nav_result.NavResult` so the curator records which gate
    fired.
