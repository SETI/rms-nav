============================
Star Unique Match Navigation
============================

Overview
========

This technique exploits the simplest star geometry: one or two catalog stars
bright enough and isolated enough that the brightest image peak inside a small
search window around each prediction is unambiguously the matched detection.  It
needs no global star-detection pass and no triplet pattern matching, so it works
on sparse fields where a multi-star matcher would fail.  Feasibility passes when
at least one usable catalog star is predicted in the extended field of view; it
fails when no usable star is present.  The navigate path then decides between a
one-star and a two-star branch.

Theory
======

The search window around each predicted star position is sized to the
per-instrument SPICE pointing-error envelope, so the brightest peak inside the
window is, by construction, the matched detection.  A brightness-weighted
centroid pins its sub-pixel position, and the offset is the centroid minus the
prediction.

In the one-star branch a single match cannot cross-check itself, so the
technique guards it with a brightness-uniqueness gate: the brightest catalog
star must be at least a configured magnitude margin brighter than the
next-brightest predictable star, otherwise the match is ambiguous and rejected.
The recovered offset is

.. math::

   (\Delta v, \Delta u) = (c_v - p_v,\; c_u - p_u),

the detection centroid minus the prediction, and its confidence is capped below
one because a lone match carries no internal consistency check.

In the two-star branch the two detections are matched to the two predictions
under both possible assignments, and the assignment with the smaller joint
residual is chosen:

.. math::

   r = \min_{\text{assignment}}
       \left\| (d - p) - \overline{(d - p)} \right\|,

where the residual is the centroid-relative scatter that a correct assignment
drives to near zero.  That residual is a genuine cross-check, so the two-star
confidence cap is higher than the one-star cap.  Two detections that resolve to
the same image peak (their centroids less than a pixel apart, as happens when
overlapping windows share one bright star) would fabricate a zero residual, so
the technique falls back to the one-star branch in that case so the
brightness-uniqueness gate can reject the ambiguity.

The covariance floors on the per-feature Cramer-Rao position bound carried by
each star and inflates it by the squared residual so a noisy match reports its
real scatter rather than the noise-free lower bound.  When a camera rotation is
requested, a single star cannot constrain rotation and the result is reported as
rotation-unobservable with a sentinel variance; two stars constrain rotation
through their separation, and the rotation variance follows the analytic
small-angle lever-arm

.. math::

   \sigma_\theta^2 = \frac{2\,(\sigma_v^2 + \sigma_u^2)}{L^2},

where :math:`L` is the catalog separation of the pair.  The reported covariance
captures the per-star centroid bound and the match residual; it does not model
SPICE pointing systematics beyond that floor.

Configuration
=============

Tunables live under ``techniques.StarUniqueMatchNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``brightness_margin_to_next_catalog_star_mag`` -- float, default ``1.5`` mag.
  Minimum magnitude difference to the next-brightest predictable star for the
  one-star branch to fire; larger demands a more uniquely bright star.
- ``search_window_px`` -- float, default ``30.0`` px.  Half-width of the search
  window around each prediction; should bracket the per-instrument pointing-error
  envelope.
- ``centroid_box_half_px`` -- int, default ``3`` px.  Half-width of the
  brightness-weighted centroid box around the detected peak.
- ``max_residual_px`` -- float, default ``4.0`` px.  Maximum best-assignment
  residual in the two-star branch before falling back to the one-star branch.
- ``detection_sigma`` -- float, default ``4.0`` (dimensionless).  Detection
  threshold as a multiple of the per-pixel noise sigma; below it the brightest
  pixel in the window is treated as noise.
- ``one_star_confidence_cap`` -- float, default ``0.7`` (dimensionless).  Post-sigmoid
  confidence cap for the one-star branch; a lone match cannot self-check.  Must
  lie in ``[0, 1]``.
- ``two_star_confidence_cap`` -- float, default ``0.8`` (dimensionless).  Post-sigmoid
  confidence cap for the two-star branch, higher because the residual cross-checks
  the assignment.  Must lie in ``[0, 1]``.
- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  Slack around the
  search-window axis bounds for the at-edge check.
- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  When the
  two-star Procrustes path fits rotation, a converged rotation magnitude past this
  fraction of the per-image rotation cap trips at-edge; the one-star path always
  reports rotation as unobservable so it is unaffected.

Confidence formula
-------------------

The confidence coefficients live in the ``techniques.StarUniqueMatchNav`` stanza
of ``config_510_techniques.yaml``.  The sigmoid baseline is ``alpha0 = -1.0`` and
hard-zero gates force confidence to zero when ``at_edge`` or ``spurious`` is true;
the per-mode caps above are applied after the sigmoid.  See
:doc:`dev_guide_techniques_confidence` for the sigmoid mathematics.

- ``predicted_snr`` -- alpha = 1.0, offset = 0, divisor = 20.0, cap at 1.0.  Predicted
  signal-to-noise of the brightest catalog star; higher SNR tightens the match.
- ``brightness_margin_mag`` -- alpha = 1.0, offset = 1.5, divisor = 1.5, cap at 1.0.
  Magnitude margin to the next-brightest predictable star; additional margin above
  the floor earns confidence.
- ``residual_px`` -- alpha = -1.0, offset = 0, divisor = 2.0, no cap.  Detection-to-
  prediction residual; larger residuals pull confidence down.

Implementation
==============

Source files: ``src/nav/nav_technique/nav_technique_star_unique_match.py`` and the
shared star helpers in ``nav.nav_technique._star_helpers``.  The public class is
:py:class:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav`,
a subclass of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  Its
``accepts_feature_types`` is the single ``STAR`` feature type, its
``requires_prior`` is ``False`` (it runs in pass 1), and its
``confidence_attributes`` set names ``at_edge``, ``spurious``, ``predicted_snr``,
``brightness_margin_mag``, and ``residual_px``.

:py:meth:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.is_feasible`
reads only feature metadata and returns feasible when at least one usable ``STAR``
feature is present, consuming up to two of them.

:py:meth:`~nav.nav_technique.nav_technique_star_unique_match.StarUniqueMatchNav.navigate`
ranks the usable stars by predicted signal-to-noise.  When two or more are
available it attempts the two-star branch through the private ``_try_two_star``;
that branch returns ``None`` -- falling through to the one-star branch -- when a
detection is missing, when both predictions resolve to the same peak, or when the
best-assignment residual exceeds the maximum.  Otherwise it runs the one-star
branch through ``_try_one_star``.

The result shape branches on the chosen path and on whether rotation is fit:

- One-star, no rotation: ``(2, 2)`` covariance from the per-feature bound;
  ``rotation_rad`` and ``sigma_rotation_rad`` are ``None``.
- One-star, with rotation: rank-deficient ``(3, 3)`` covariance via
  ``embed_rotation_unobservable`` (a single match cannot constrain rotation),
  ``rotation_rad`` fixed at ``0.0`` and ``sigma_rotation_rad`` the
  rotation-unobservable sentinel from ``rotation_unobservable_sigma_rad``.
- Two-star, no rotation: ``(2, 2)`` covariance averaging the two stars' bounds;
  rotation fields ``None``.
- Two-star, with rotation: full ``(3, 3)`` covariance with the analytic
  lever-arm rotation diagonal built by ``_build_two_star_covariance_3dof``,
  ``rotation_rad`` the Procrustes angle from ``_similarity_fit_assignment``, and
  ``sigma_rotation_rad`` the square root of the rotation diagonal.

``_evaluate_confidence`` runs the YAML formula via ``evaluate_sigmoid_combination``
wrapped by a per-technique context adapter, logs the breakdown through
``log_confidence_breakdown`` (see :doc:`dev_guide_techniques_confidence`), and
applies the per-mode cap.  Every failure branch returns through the private
``_fail`` path with a zero-confidence spurious result.

Every field of
:py:class:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics` is
populated:
:py:attr:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics.predicted_snr`,
:py:attr:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics.brightness_margin_mag`,
and
:py:attr:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics.residual_px`
feed the confidence formula above, and
:py:attr:`~nav.nav_technique.diagnostics.StarUniqueMatchDiagnostics.mode` records
which branch produced the result (a string, so it is not surfaced to the
formula).  The return value is a
:py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

**one_bright_star_no_body (W1449079117_1_CALIB).** A single bright star (Vega) in
a wide-angle frame through a red filter, with no body or ring.  Feasibility
passes on the one usable star and the one-star branch fires: the brightness-margin
gate is satisfied (no comparably bright competitor), the brightest peak in the
search window is matched, and the offset is recovered against the operator ground
truth of ``(3.06, -0.02)`` px.  The sidecar records this technique under
``techniques_must_run`` with ``confidence_tier: low`` -- the one-star confidence
cap of 0.7 keeps a single unchecked match modest.

**two_bright_stars_no_body (corpus class).** This class has exactly two
unambiguous catalog stars, each at least 1.5 magnitudes brighter than any
competitor, with no body or ring.  Feasibility passes and the two-star branch
fires: both detections are matched, the best-assignment residual cross-checks the
fit, and confidence is capped at the higher two-star limit of 0.8.
