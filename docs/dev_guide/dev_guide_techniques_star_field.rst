==========================================================
Star Field Pattern Match (StarFieldFromCatalogNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` is the pass-1
multi-star pattern matcher. It hashes catalog and detection triplets into a translation- and
rotation-invariant feature space, finds correspondences by an exhaustive hash-distance scan
in that space, and runs a RANSAC fit to recover the translation (plus optional in-plane
rotation) that maps the predicted catalog cohort onto the observed detections. The technique
runs without a prior — it is the ensemble's primary star path on scenes where the SPICE
pointing error is too large for
:class:`~spindoctor.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`'s search
window to contain the brightest predictable star.

Feasibility passes when the predictable-star cohort has at least three usable
:data:`~spindoctor.feature.feature_type.NavFeatureType.STAR` features (the matcher cannot form a
single triplet below that count); feasibility fails otherwise.

Theory
======

The technique solves the global star-pattern-matching problem from scratch — without a prior
offset — by reducing translation- and rotation-invariant pattern matching to a high-dimensional
nearest-neighbour search.

Triplet hashing
---------------

For every triplet of three predictable catalog stars the matcher computes a translation- and
rotation-invariant hash ``(d_AB / d_AC, d_BC / d_AC, angle BAC)``, where vertex ``A`` is
canonicalised as the brightest star of the triplet and the other two follow in sorted-index
order. The same hash is computed for every triplet of three detected sources in the image.
Triplets with matching hashes (within a small tolerance) are candidate correspondences.

Source detection
----------------

Detections come from a matched-filter search over the extfov image: the image is correlated
with a Gaussian PSF kernel of the configured ``psf_sigma_px``, peaks above
``detection_sigma * image_noise_sigma`` are kept, and a brightness-weighted moment around
each peak gives the sub-pixel centroid. Up to ``max_sources`` brightest detections feed the
hash-space search; the catalog side is similarly capped.

Hash-space match
----------------

Every detection triplet's hash is compared against every catalog triplet's hash by a
brute-force scan in ``(ratio, ratio, angle_radians)`` space, using a weighted Euclidean
distance (``hash_ratio_weight``, ``hash_angle_weight``). ALL pairs whose hash distance falls
within ``hash_match_tolerance`` survive as candidates, sorted by ``(hash_dist_sq, sorted
detection-source indices ascending, catalog-triplet index)`` so iteration is deterministic
across repeated invocations. Each candidate implies a per-star correspondence between three
catalog vertices and three detection vertices.

RANSAC inlier validation
------------------------

Each candidate correspondence proposes a similarity transform (rotation, translation, and an
implied unit scale). The matcher applies the proposed transform to every catalog star and
counts how many predicted positions land within ``inlier_tolerance_px`` of a detection. The
detection-to-catalog inliers within the tolerance ball are the maximum-cardinality one-to-one
assignment (solved with :func:`scipy.optimize.linear_sum_assignment`), so the count is exact
and independent of detection ordering rather than a greedy nearest-available match that could
strand a detection when two compete for one catalog star. The transform with the most inliers
wins; below ``pattern_match_min_inliers`` the technique reports spurious.

PSF-fit inlier refinement
-------------------------

The detection centroid that pins each source is a brightness-weighted moment: unbiased, but
only noise-limited, so on a faint field its per-star scatter dominates the field offset.
Once the inlier correspondences are fixed, each matched inlier is re-centroided to drive that
scatter down. Two estimators are available per star and they trade off with brightness:

- The **brightness-weighted moment** is unbiased; its error falls roughly as
  :math:`1/\mathrm{SNR}` as the star brightens.
- A **maximum-likelihood PSF fit** (``obs.star_psf().find_position`` against the instrument's
  modelled point-spread function) reaches the minimum variance and so wins decisively when
  the star is faint. An undersampled PSF, however, carries a fixed sub-pixel-phase bias floor
  (~0.08 px for the COISS NAC star PSF, sigma ~0.54 px) that does not improve with brightness.

The two error curves cross near an integrated SNR of ~30 at the field level (the per-star
crossover is a little higher, ~40, but the field fit averages the moment's noise down faster
than the PSF fit's partly-correlated bias). The technique therefore refines a matched inlier
with the PSF fit only while its box SNR is below the configurable ceiling
``psf_refine_snr_max`` (default 30), and keeps the moment above it. The box SNR is
:math:`\sum (\text{box} - \text{median}) / \sqrt{\text{signal} + n_{\text{pix}}\,\sigma^{2}}`
over the fit box. The PSF fit reports its position in the ``eval_rect`` convention (offset
measured from a pixel's lower edge), so the technique subtracts the half-pixel to land in the
pixel-centre convention shared by the detection moment and the catalog prediction. Any inlier
whose fit fails (too close to the image edge, too few good pixels, no convergence) silently
falls back to its moment centroid. The whole step is gated by ``psf_refine_enabled`` and the
obs supplying a ``star_psf()``; without either it is a no-op and the moment centroids stand.

The observation that drives this is brightness-dependent: a dim field (vmag 3-4) improves its
recovered-offset median from ~0.052 px (moment-only) to ~0.023 px, while a bright field
(vmag 0-0.8) recovers to ~0.005 px on the moment alone -- where forcing the PSF fit would
instead *raise* the error to ~0.056 px by exposing the bias floor. See the simulator
performance report's star-field centroiding section for the dim-vs-bright sweep.

Inlier refit
------------

Once the inlier set is known, the matcher refines the transform with a Tukey-biweight-
reweighted fit. With camera rotation off (the default) the refit is a reweighted mean
translation: an unweighted pass seeds the residual scale, then one biweight-reweighted pass
at the conventional 4.685 breakdown constant produces the final offset. When
:attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is on, the
same two-pass Tukey reweighting drives a Procrustes / Kabsch similarity fit instead. In both
branches the weights are Tukey biweight values seeded from the unweighted pass — never
per-star inverse-trace covariance weights. The translation read off the fit is the reported
offset; the rotation, when fitted, is the converged angle about the catalog-side weighted
centroid.

Per-axis covariance
-------------------

The reported translation covariance is derived from the Tukey-weighted inlier residual
scatter alone: the per-axis reduced chi-square divided by the total weight, floored by the
pure inverse-precision ``1 / sum(w)``, with the squared ``model_error_floor_px`` added to
the diagonal. When the per-instrument camera-rotation flag is on, a 3x3 covariance is
reported whose translation block follows the same algebra (with three fitted parameters)
and whose rotation diagonal is described next; the catalog-side centroid spread enters only
through that rotation term. Otherwise the 2x2 translation block is reported on its own.

The rotation variance is the inverse of the rotation Fisher information computed directly
from the per-vertex tangential lever-arm Jacobian ``(-du_i, dv_i)`` about the weighted
catalog centroid, weighted by the per-axis residual precision. This is exact for
anisotropic residuals — unlike a pooled isotropic form that averages the two per-axis
residual variances and can be off by up to a factor of two when the inlier lever arms are
distributed unevenly across the axes. (``StarRefineNav`` keeps its own pooled rotation form;
the two are not shared.)

Restrictions and assumptions
----------------------------

- The matcher needs at least three usable stars to form one triplet. Below that count the
  technique reports infeasible without running.
- The hash-space tolerance is empirically tuned to a typical centroid sigma of ~0.1 px;
  significantly noisier centroids may cause the hash match to miss correct
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
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and the hard-zero
gate forces confidence to zero. When the inlier count falls below
``pattern_match_min_inliers`` the result is flagged
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.StarFieldFromCatalogNav.tuning``
in ``src/spindoctor/config_files/config_510_techniques.yaml``.

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
- ``hash_match_tolerance`` — float, default ``0.05`` (dimensionless). Match radius in the
  (ratio, ratio, angle_rad) hash space; weighted Euclidean. Empirical default for centroid
  sigma ~0.1 px against typical triplet scales.
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
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses
  this fraction of the per-image
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.
- ``rotation_separability_floor_deg`` — float, default ``0.75`` deg. Roll/translation
  separability floor: the camera-roll analysis in the simulator performance report shows
  the matcher cannot separate a roll below ~0.75 deg from a pure translation, so a fitted
  rotation magnitude below this floor is reported with the rotation-unobservable sentinel
  variance and the
  :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.rotation_below_separability_floor`
  diagnostics flag instead of a confident near-zero value.
- ``model_error_floor_px`` — float, default ``0.0`` px. Uncalibrated model-error floor,
  added in quadrature to the reported covariance diagonal
  (``reported_cov = CRLB-form (+) model_error_floor_px**2``). Captures the SPICE pointing
  residual and body-shape error that the pattern-match residual scatter does not. The
  default 0.0 is a no-op pending an integration-library calibration sweep.
- ``psf_refine_enabled`` — int flag, default ``1``. ``1`` enables the PSF-fit re-centroiding
  of matched inliers; ``0`` keeps the brightness-weighted moment centroid everywhere.
- ``psf_refine_box_px`` — int, default ``11`` px (odd). Square box side for the PSF fit and
  the integrated-SNR estimate around each inlier.
- ``psf_refine_search_limit_px`` — float, default ``2.0`` px. Maximum search distance from
  the moment centroid for the PSF fit; the moment is already within a pixel of truth.
- ``psf_refine_snr_max`` — float, default ``30.0`` (dimensionless). Integrated-SNR ceiling
  for the moment/PSF crossover: an inlier whose box SNR exceeds this keeps its moment
  centroid (its noise has already fallen below the PSF fit's sub-pixel-phase bias floor);
  fainter inliers are refined with the PSF fit. The default is tuned to the field-level
  crossover on a nominal background; elevated read noise or a stray-light gradient pulls the
  optimum down toward ~16-21 (see the simulator performance report).

Per-instrument overrides
------------------------

Per-instrument YAML files in ``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not
override any of these knobs.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. Spec is
``techniques.StarFieldFromCatalogNav``; consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious`.

- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers` — alpha = 0.906,
  offset = 6.0, divisor = 6.0, cap at 1.0. Number of detection-to-catalog inliers after
  RANSAC. Saturates at 12 inliers (offset = 6 plus full cap).
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px` —
  alpha = 0.0, offset = 0.0, divisor = 1.0, no cap. Median position residual on inliers.
  Sign-constrained to zero by the calibration fit: with no failure class in the sim
  campaign the residual carried no signal. Kept wired for a future real-anchored
  calibration.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources` —
  alpha = 0.0, offset = 0.0, divisor = 30.0, cap at 1.0. Number of bright sources detected
  in the image. Constant in the calibration campaign (the detector always fills its
  ``max_sources`` budget, noise peaks included), so it carries no information as wired;
  kept at zero weight.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted` —
  alpha = 0.777, offset = 0.0, divisor = 30.0, cap at 1.0. Number of catalog stars in the
  extfov.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = 2.339`. A post-sigmoid ``hard_cap`` of 0.95 clamps the result: the sim
calibration campaign was single-class (every non-spurious pattern match recovered planted
truth), and single-class evidence cannot support confidence above ~0.95 on real frames the
sim cannot represent (distortion, smear, saturated bloom); the ceiling is conservative
pending a real-anchored calibration.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_star_field.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` and the
  hashing / RANSAC / fit helpers.
- ``src/spindoctor/nav_technique/_star_helpers.py`` — package-private helpers
  ``usable_stars`` (filter), ``local_centroid`` (per-source centroid), and
  ``similarity_transform_fit`` (Kabsch / Procrustes solve).
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class
:class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.name` —
  ``'StarFieldFromCatalogNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.accepts_feature_types` —
  ``frozenset({STAR})``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.requires_prior` — ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique.NavTechnique.confidence_attributes` —
  ``{'at_edge', 'spurious', 'n_inliers', 'median_residual_px', 'n_detected_sources',
  'n_catalog_predicted'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers` — number of
  detection-to-catalog inliers. Consumed by the confidence formula and the spurious-
  detection gate.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px` — median
  position residual on inliers. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources` — number
  of bright sources detected in the image.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted` — number
  of catalog stars in the extfov.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.n_triplets_evaluated` —
  number of triplet candidates considered by RANSAC.
- :attr:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics.rotation_below_separability_floor`
  — true when a co-fitted rotation's magnitude fell below the roll/translation
  separability floor (``rotation_separability_floor_deg``, default 0.75 deg per the
  camera-roll separability analysis in the simulator performance report) and the
  rotation was therefore reported with the rotation-unobservable sentinel variance
  instead of a confident near-zero value.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.navigate`:

1. Open a logged section. Filter the offered features down to ``usable_stars``
   (predictable and not occluded by a body silhouette or ring annulus) and pull
   the predicted catalog positions / SNR off the per-feature
   :attr:`~spindoctor.feature.feature.NavFeature.geometry` and
   :attr:`~spindoctor.feature.feature.NavFeature.flags`.
2. Run the matched-filter detection over
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_ext` to find the brightest
   sources. Cap the catalog and detection cohorts at ``max_sources`` each.
3. Compute every catalog and detection triplet's (ratio, ratio, angle) hash. Scan every
   detection-triplet / catalog-triplet pair and keep all candidates whose weighted hash
   distance falls within ``hash_match_tolerance``, sorted for deterministic iteration.
4. RANSAC: for each candidate correspondence triplet, propose a similarity transform and
   count inlier matches across the full catalog cohort using ``inlier_tolerance_px``.
   Keep the transform with the most inliers.

   - **Below min_inliers.**  Return a spurious zero-confidence result with the diagnostic
     fields populated for the JSON sidecar.

5. Refine the inlier detection positions: for each matched inlier below the
   ``psf_refine_snr_max`` integrated-SNR ceiling, replace its moment centroid with a
   maximum-likelihood PSF fit (``obs.star_psf().find_position``, half-pixel-corrected);
   brighter inliers and failed fits keep the moment. Skipped entirely when
   ``psf_refine_enabled`` is ``0`` or the obs exposes no ``star_psf()``.
6. Refit on the (refined) inlier correspondences: with ``fit_camera_rotation`` false a
   Tukey-biweight-reweighted mean translation (an unweighted pass seeds the residual scale,
   then one biweight-reweighted pass); with it true, the same two-pass Tukey reweighting
   drives a Procrustes / Kabsch fit via ``similarity_transform_fit``. The translation is
   the reported offset.
7. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **Translation only** (``fit_camera_rotation`` false). The (2, 2) covariance comes
     from the Tukey-weighted inlier residual scatter (per-axis reduced chi-square over
     the total weight, floored at the pure inverse-precision, plus the squared
     ``model_error_floor_px``).
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**  The (3, 3) covariance has the per-axis translation variances
     (same algebra as above with three fitted parameters) and the rotation variance from
     the inverse rotation Fisher information built on the per-vertex lever arms about the
     weighted catalog centroid. When the fitted rotation magnitude falls below the
     ``rotation_separability_floor_deg`` floor,
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.rotation_rad` keeps
     the fitted value but the rotation variance is replaced by the rotation-unobservable
     sentinel. (A one-inlier rotation fit cannot occur: ``pattern_match_min_inliers`` is
     validated to be at least 3 at load time.)

8. Apply the at-edge tests against the search-window axis bounds and the rotation cap.
9. Build a :class:`~spindoctor.nav_technique.diagnostics.StarFieldDiagnostics`, evaluate the
   confidence spec via :func:`~spindoctor.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the breakdown via
   :func:`~spindoctor.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~spindoctor.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each per-star contribution at audit time.

Examples
========

``star_dominated`` (Cassini ISS WAC, image ``W1580760393_1``)
    Dense star field with no body in FOV. The stars model emits one ``STAR`` feature per
    predictable catalog star;
    :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` runs the
    triplet hash against the detected sources, lands on a similarity transform with several
    inliers, and reports a translation against the operator-verified offset
    :math:`(\Delta v, \Delta u) = (-2.68, -3.68)` px. The pass-2
    :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` consumes this prior
    and polishes the offset using the full predictable cohort; see
    :doc:`dev_guide_techniques_star_refine` for that walk-through.

``stars_plus_body`` (Cassini long-exposure background-stars scene class)
    One body and at least three usable catalog stars in the same FOV. The stars model emits
    one ``STAR`` feature per predictable catalog star whose predicted position lies outside
    the body silhouette and any ring annulus; the body model emits
    a ``LIMB_ARC`` (or ``BODY_BLOB``) for the body. On pass 1, the
    :class:`~spindoctor.nav_technique.nav_technique_body_limb.BodyLimbNav` consumes the body's
    feature first and the orchestrator's ensemble combine populates the per-image prior
    from the limb-derived offset.
    :class:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav` runs in
    parallel on the star cohort: the triplet hash matches the predicted catalog stars
    against detected sources, the RANSAC inlier set lands on the same translation as the
    body fit, and the ensemble combine tightens the per-image covariance. On pass 2 the
    :class:`~spindoctor.nav_technique.nav_technique_star_refine.StarRefineNav` consumes the
    cohort and polishes the offset further.

``faint_stars`` (Galileo SSI / Voyager outer-leg scene class)
    Every catalog star in the FOV is fainter than the per-observation limiting magnitude
    ``obs.star_max_usable_vmag()``. The stars model emits no ``STAR`` features that clear
    the magnitude gate. The technique's
    :meth:`~spindoctor.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.is_feasible`
    fails with reason ``fewer_than_3_predicted_stars (got 0)`` and the technique skips its
    navigate pass entirely. The orchestrator falls back to whichever body- or ring-derived technique is
    feasible on the scene and surfaces the per-technique infeasibility on the per-image
    :class:`~spindoctor.nav_orchestrator.nav_result.NavResult` so the curator records which gate
    fired.
