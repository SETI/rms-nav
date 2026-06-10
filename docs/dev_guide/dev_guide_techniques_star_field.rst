=====================
Star Field Navigation
=====================

Overview
========

This technique exploits the geometric pattern of bright point sources in a star
field.  It makes no prior assumption about which image detection corresponds to
which catalog star: it detects bright sources, predicts the catalog star
positions, and recovers a translation purely from the relative geometry of the
two point sets matched against each other.  Feasibility passes when at least
three usable catalog stars are predicted in the extended field of view -- the
matcher needs at least one triplet per side -- and fails when fewer than three
are available.

Theory
======

The technique recovers a translation from two unlabelled point sets -- detected
image sources and predicted catalog stars -- without knowing the
correspondence.  It does so through a similarity-invariant geometric hash.

Sources are detected by matched-filtering the image against a small Gaussian
point-spread kernel; pixels that are local maxima of the response and exceed a
threshold of a configured multiple of the per-pixel noise are accepted, and a
brightness-weighted moment around each peak yields a sub-pixel centroid.  The
brightest survivors feed the matcher.

For every unordered triplet of points, with the brightest vertex canonicalised
as the apex, a similarity-invariant hash is formed from two side-length ratios
and the apex angle:

.. math::

   h(A, B, C) = \left( \frac{d_{AB}}{d_{AC}},\;
       \frac{d_{BC}}{d_{AC}},\; \angle BAC \right).

Both ratios and the angle are unchanged by translation, rotation, and uniform
scale, so a detection triplet and a catalog triplet that describe the same three
stars produce nearly identical hashes regardless of the unknown offset.
Collinear triplets are rejected because their apex angle degenerates to zero or
:math:`\pi`, a value that would falsely match any other near-collinear triplet.

Candidate triplet pairings are formed by matching every detection hash against
every catalog hash within a weighted Euclidean tolerance, then scored by a
RANSAC-style consensus: each pairing implies the translation that maps the
catalog-triplet centroid onto the detection-triplet centroid, and that
translation is scored by counting detection-to-catalog correspondences within a
pixel tolerance under greedy nearest-neighbour matching, each catalog star
consumed at most once.  Candidates are iterated in a fully deterministic order
-- sorted by hash distance, then by ascending source indices -- so the winning
transform is reproducible across repeated runs on the same observation.

The winning inlier set is refit by Tukey-biweight-reweighted least squares to
recover the final translation and a covariance from the surviving residual
scatter:

.. math::

   \hat{t} = \frac{\sum_i w_i (d_i - c_i)}{\sum_i w_i}, \qquad
   \Sigma_{\text{axis}} = \frac{1}{\sum_i w_i}
       \max\!\left( \frac{\sum_i w_i r_{i,\text{axis}}^2}{N - p},\;
       \frac{1}{\sum_i w_i} \right),

where :math:`d_i` and :math:`c_i` are matched detection and catalog positions,
:math:`w_i` are the biweight weights, :math:`r_i` are the residuals, and
:math:`N - p` are the degrees of freedom (two fitted parameters for translation,
three when a rotation is co-fitted).  When a camera rotation is requested the
refit is an orthogonal Procrustes (Kabsch) fit and the rotation variance follows
a lever-arm formula scaling the pooled residual variance by the inverse of the
weighted catalog spread about its centroid; when that spread is too small to
constrain rotation the rotation variance collapses to the rotation-unobservable
sentinel.  The reported covariance captures the inlier residual scatter inflated
by the degrees-of-freedom factor; it does not model SPICE pointing systematics
beyond an optional uncalibrated model-error floor.  The technique fails -- and
reports a deliberately huge covariance with zero confidence -- when fewer than
three sources are detected, when no valid triplets survive canonicalisation, or
when the best transform musters fewer than the configured minimum inlier count.

Configuration
=============

Tunables live under ``techniques.StarFieldFromCatalogNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``max_sources`` -- int, default ``30`` (count).  Maximum number of brightest
  detected sources and brightest catalog stars fed to the matcher; the triplet
  count grows as the cube of this value, so it bounds the candidate list.
- ``detection_sigma`` -- float, default ``4.0`` (dimensionless).  Detection
  threshold as a multiple of the per-pixel noise sigma; higher rejects fainter
  sources.
- ``psf_sigma_px`` -- float, default ``1.0`` px.  Gaussian point-spread sigma for
  the matched-filter detection kernel; matches the typical ungroomed star PSF.
- ``centroid_box_half_px`` -- int, default ``3`` px.  Half-width of the box used for
  both the local-maximum window and the brightness-weighted centroid; larger
  averages over more pixels.
- ``hash_match_tolerance`` -- float, default ``0.05`` (dimensionless).  Match radius
  in the weighted ratio-ratio-angle hash space; larger admits more candidate
  pairings at the cost of more false matches.
- ``hash_ratio_weight`` -- float, default ``1.0`` (dimensionless).  Weight on the two
  ratio components of the hash distance metric.
- ``hash_angle_weight`` -- float, default ``1.0`` (dimensionless).  Weight on the
  angle component of the hash distance metric.
- ``inlier_tolerance_px`` -- float, default ``2.0`` px.  Maximum residual distance
  for a detection-to-catalog correspondence to count as an inlier under a
  candidate transform; larger loosens the consensus.
- ``pattern_match_min_inliers`` -- int, default ``6`` (count).  Minimum inlier count
  for the matcher to accept a transform; below it the technique returns spurious.
  Must be at least three.
- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  Slack around the
  search-window axis bounds for the at-edge check.
- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  When
  rotation is fit, a converged rotation magnitude past this fraction of the
  per-image rotation cap trips at-edge.
- ``model_error_floor_px`` -- float, default ``0.0`` px.  Uncalibrated model-error
  floor added in quadrature to the reported covariance diagonal; the default is a
  no-op.

Confidence formula
-------------------

The confidence coefficients live in the ``techniques.StarFieldFromCatalogNav``
stanza of ``config_510_techniques.yaml``.  The sigmoid baseline is
``alpha0 = -2.0`` and hard-zero gates force confidence to zero when ``at_edge`` or
``spurious`` is true.  See :doc:`dev_guide_techniques_confidence` for the sigmoid
mathematics.

- ``n_inliers`` -- alpha = 1.0, offset = 6.0, divisor = 6.0, cap at 1.0.  Number of
  matched detection-to-catalog inliers after RANSAC; more matched stars is a
  stronger constraint.
- ``median_residual_px`` -- alpha = -1.0, offset = 0, divisor = 1.0, no cap.  Median
  position residual over the inliers; larger residuals pull confidence down.
- ``n_detected_sources`` -- alpha = 0.0, offset = 0, divisor = 30.0, cap at 1.0.
  Number of bright sources detected in the image; wired with zero weight pending
  calibration.
- ``n_catalog_predicted`` -- alpha = 0.0, offset = 0, divisor = 30.0, cap at 1.0.
  Number of catalog stars predicted in the extended FOV; wired with zero weight
  pending calibration.

Implementation
==============

Source files: ``src/nav/nav_technique/nav_technique_star_field.py`` and the
shared star helpers in ``nav.nav_technique._star_helpers``.  The public class is
:py:class:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav`,
a subclass of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  Its
``accepts_feature_types`` is the single ``STAR`` feature type, its
``requires_prior`` is ``False`` (it runs in pass 1), and its
``confidence_attributes`` set names ``at_edge``, ``spurious``, ``n_inliers``,
``median_residual_px``, ``n_detected_sources``, and ``n_catalog_predicted``.

:py:meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.is_feasible`
reads only feature metadata and returns feasible when at least three usable
``STAR`` features are present.

:py:meth:`~nav.nav_technique.nav_technique_star_field.StarFieldFromCatalogNav.navigate`
derives a deterministic seed from the observation midtime, ranks the catalog
cohort by predicted signal-to-noise, and detects image sources by
matched-filtering against a Gaussian kernel built with the module-private
``_gaussian_kernel`` and ``_detect_image_sources`` helpers (the latter calls
:py:func:`scipy.ndimage.maximum_filter` for the local-maximum test).  When the
catalog cohort or the detection set has fewer than three members it returns
through the private ``_fail`` path; otherwise it delegates to ``_match_and_fit``.

``_match_and_fit`` enumerates canonical triplets per side (module-private
``_enumerate_triplets`` and ``_triplet_hash``), forms hash-distance candidates
through ``_enumerate_candidates``, scores them through ``_score_candidates``, and
returns ``_fail`` when no valid triplets survive or the winner falls below the
minimum inlier count.  The surviving inliers are refit by ``_tukey_refit`` (or
``_similarity_refit`` when rotation is fit), and the covariance is built by
``_build_covariance`` for translation-only or ``_build_covariance_3dof`` when
rotation is co-fitted.  ``_evaluate_confidence`` runs the YAML formula via
``evaluate_sigmoid_combination`` wrapped by a per-technique context adapter and
logs the breakdown through ``log_confidence_breakdown`` (see
:doc:`dev_guide_techniques_confidence`); the Tukey biweight machinery is shared
with the DT techniques (see :doc:`dev_guide_techniques_dt_fitting`).

The result shape branches on whether rotation is fit.  Without rotation the
``covariance_px2`` is ``(2, 2)`` and both ``rotation_rad`` and
``sigma_rotation_rad`` are ``None``.  With rotation the covariance is ``(3, 3)``
with the lever-arm rotation diagonal, ``rotation_rad`` is the Procrustes angle,
and ``sigma_rotation_rad`` is the square root of its diagonal.  Every failure
branch returns a zero-confidence spurious result whose covariance is the
deliberately huge identity (promoted to the rank-deficient ``(3, 3)`` form when
rotation is requested).

Every field of
:py:class:`~nav.nav_technique.diagnostics.StarFieldDiagnostics` is populated:
:py:attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_inliers`,
:py:attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.median_residual_px`,
:py:attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_detected_sources`,
and
:py:attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_catalog_predicted`
feed the confidence formula above, and
:py:attr:`~nav.nav_technique.diagnostics.StarFieldDiagnostics.n_triplets_evaluated`
records the number of hash-distance candidates considered.  The return value is a
:py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

**star_dominated (W1580760393_1_CALIB).** A dense star field with no body or ring
in the field, taken through a clear filter -- the canonical multi-star pattern
case.  Feasibility passes because well more than three catalog stars are
predicted, and the sidecar records ``primary_technique: StarFieldFromCatalogNav``
with ``techniques_must_run`` listing this technique.  The sidecar pins
``status: failed`` with ``confidence_tier: failed``: the pattern matcher runs but
the confidence formula, dominated by ``n_inliers`` and ``median_residual_px``,
returns a value below the orchestrator's acceptance floor on this frame.

**two_bright_stars_no_body (corpus class).** This class has exactly two
unambiguous catalog stars and therefore cannot form the triplet this technique
requires; feasibility fails (fewer than three predicted stars) and the scene is
handled by the unique-match technique instead.
