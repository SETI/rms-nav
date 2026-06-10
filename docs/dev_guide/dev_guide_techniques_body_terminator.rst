==========================
Body Terminator Navigation
==========================

Overview
========

The body-terminator technique exploits the brightness edge at the day-night boundary on a
resolved body: along the terminator the surface transitions from sunlit to shadowed, the
image gradient rises, and the predicted geometric terminator of an ellipsoidal body is a curve
whose shape is known from SPICE.  Like the limb technique it concatenates every predicted
terminator-arc polyline, weights the vertices, and runs the shared distance-transform fitter
to recover one translation (and, on rotation-fitted cameras, one rotation) that slides the
predicted terminator onto the observed boundary.  It differs from the limb technique in three
ways: it consumes terminator arcs rather than limb arcs, it assigns every vertex of a given
body one uniform inverse-variance weight derived from that body's mean normal sigma so that
cross-body albedo variation is reflected, and its confidence formula adds terms for the
visible terminator fraction, the phase-angle factor, and an albedo penalty.  Feasibility
passes when at least one predicted terminator arc survives with enough visible vertices to
constrain a translation; feasibility fails when no terminator arc reaches that minimum visible
length.

Theory
======

On a body lit obliquely, the boundary between the sunlit and shadowed hemispheres is a smooth
curve whose projected shape follows from the body pose and the solar direction, both known
from spacecraft ephemeris.  The unknown is again a rigid pointing error that displaces the
whole predicted curve, so a single translation explains every terminator vertex jointly.  The
terminator is a softer, more photometrically variable edge than the sunlit limb: its sharpness
depends on phase angle and on local surface albedo, and crater shadows or albedo boundaries
near the terminator can present competing edges.

Let the predicted terminator be sampled at vertices :math:`\mathbf{p}_i`.  An image-side
distance transform assigns each pixel its distance to the nearest detected edge, and for a
candidate offset :math:`\boldsymbol{\delta}` the residual at vertex :math:`i` is the
distance-transform value at the shifted vertex,

.. math::

   r_i(\boldsymbol{\delta}) = D\!\left(\mathbf{p}_i + \boldsymbol{\delta}\right).

The fit minimises the robust, precision-weighted cost

.. math::

   C(\boldsymbol{\delta}) = \sum_b \frac{1}{\sigma_b^{2}} \sum_{i \in b} \rho\!\left(r_i\right),

where the inner sum runs over the vertices of body :math:`b`, every vertex of that body shares
the single inverse-variance weight :math:`1 / \sigma_b^{2}` formed from the body's mean normal
sigma, and :math:`\rho` is the Tukey biweight loss.  Per-body uniform weighting captures the
albedo-driven difference in terminator quality between bodies -- a dark body yields a tighter
terminator than a bright one -- while smoothing the per-vertex sigma noise.  Vertices whose
gradient polarity disagrees with the expected dark-to-light transition are dropped before the
fit.

The optimisation is the same two-stage scheme used for the limb: a coarse integer-pixel mask
overlap to seed a basin, then a Levenberg-Marquardt stage with Tukey reweighting confined to a
trust region, converging when the combined step norm falls below a small tolerance.  The
information matrix supplies the covariance, which grows to three parameters when a rotation
about the terminator centroid is fitted.

Because the terminator is a photometric feature, its failure modes are harsher than the
limb's: the fitter can lock onto a crater-shadow edge or an albedo boundary that happens to
align with the predicted curve, producing a clean-looking residual on a wrong offset.  No
per-vertex signal distinguishes that mis-convergence, so the technique is treated as a
fallback: when a non-spurious limb or disc fit is available for the same body the ensemble
drops the terminator result rather than risk it overriding a geometric technique.  The
reported covariance captures only the residual scatter of the surviving inliers; it does not
capture SPICE bias, albedo-boundary contamination, or a wholesale lock onto the wrong edge.
Those are caught by the spurious gates: an elevated unweighted residual, too few inliers, too
small an inlier fraction, or a refinement that walked far from the coarse seed.

Configuration
=============

Runtime tunables live under ``techniques.BodyTerminatorNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``min_arc_px`` -- float, default ``30.0`` px.  Minimum surviving polyline vertex count per
  TERMINATOR_ARC for feasibility; raising it rejects shorter arcs that under-constrain the
  translation.
- ``spurious_dt_rms_factor`` -- float, default ``5.0`` (dimensionless).  Multiplier on the
  smallest terminator sigma setting the DT residual ceiling; lowering it flags marginal fits
  spurious sooner.
- ``spurious_dt_floor_px`` -- float, default ``4.0`` px.  Lower bound on the DT residual
  ceiling, one pixel wider than the limb floor because terminators are softer edges.
- ``spurious_min_inliers`` -- int, default ``6`` (count).  Minimum Tukey inlier count below
  which the M-estimator covariance is no longer trusted and the result is flagged spurious.
- ``spurious_min_inlier_fraction`` -- float, default ``0.20`` (dimensionless).  Minimum
  fraction of vertices that must survive Tukey reweighting; below it the fit has almost
  certainly locked onto crater shadows or surface boundaries.
- ``spurious_max_lm_displacement_px`` -- float, default ``4.0`` px.  Maximum distance the
  refinement may move from the integer coarse seed before the result is flagged spurious.
- ``lm_trust_region_px`` -- float, default ``1.0`` px.  Radius of the trust region around the
  coarse seed; tightening it denies the refinement the runway to reach a distant spurious
  minimum at the cost of sub-pixel headroom.
- ``lm_tikhonov_alpha`` -- float, default ``0.0`` (dimensionless).  Strength of the Tikhonov
  anchor pulling the solution toward the coarse seed; larger values resist multi-pixel walks
  but also suppress legitimate sub-pixel refinement.
- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  Slack around the search-window axis
  bounds for the at-edge check; a converged offset within this distance of a bound is flagged
  at-edge.
- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  Fraction of the
  maximum rotation at which the fitted rotation trips the at-edge flag; lower values surface a
  rotation pegged against its cap earlier.

Confidence formula
-------------------

The confidence coefficients live alongside ``tuning`` in the same
``techniques.BodyTerminatorNav`` stanza.  The sigmoid argument starts from ``alpha0`` of
``-1.0`` and adds the linear terms below; the sigmoid mathematics is documented in
:doc:`dev_guide_techniques_confidence`.  The gate ``hard_zero_if`` forces confidence to zero
when ``at_edge`` or ``spurious`` is true.

- ``visible_terminator_arc_fraction`` -- alpha = 2.0, offset = 0, divisor = 1, no cap.
  Fraction of the predicted terminator that was visible and consumed; the dominant positive
  term.
- ``dt_fit_rms_px`` -- alpha = -1.0, offset = 0, divisor = 1, no cap.  Final root-mean-square
  DT residual; larger residuals pull confidence down.
- ``visible_arc_px`` -- alpha = 0.4, offset = 0, divisor = 100.0, cap at 1.0.  Total surviving
  arc length, normalised so a long terminator saturates the term.
- ``mean_phase_angle_factor`` -- alpha = 1.0, offset = 0, divisor = 1, no cap.  Mean per-body
  phase-angle factor; higher phase sharpens the terminator and raises confidence.
- ``mean_albedo_penalty`` -- alpha = -1.5, offset = 0, divisor = 1, no cap.  Mean per-body
  albedo penalty; a brighter, softer terminator pulls confidence down.

Implementation
==============

The technique lives in ``src/nav/nav_technique/nav_technique_body_terminator.py``;
:py:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` subclasses
:py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  It declares
``accepts_feature_types`` of ``frozenset({NavFeatureType.TERMINATOR_ARC})``,
``requires_prior`` of ``False`` (it runs in pass 1), a ``tier`` of ``'fallback'``, and a
``confidence_attributes`` set of ``at_edge``, ``spurious``,
``visible_terminator_arc_fraction``, ``visible_arc_px``, ``dt_fit_rms_px``, ``lm_iterations``,
``tukey_inlier_count``, ``mean_phase_angle_factor``, and ``mean_albedo_penalty``.

:py:meth:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.is_feasible`
reads only the polyline vertex count per feature and returns a feasibility report (see
:doc:`dev_guide_techniques_feasibility`) that is feasible when at least one terminator arc
reaches ``min_arc_px`` vertices.

:py:meth:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.navigate` opens a
logged technique section, validates that the context carries the image edge distance transform
and gradient vectors, and drops arcs shorter than ``min_arc_px``.  The private helper
``_aggregate_terminator_features`` concatenates the surviving vertices, negates the geometric
outward normals into polarity normals, assigns each body's vertices one uniform sigma from the
body's mean, and gathers the per-body phase-angle factors and albedo penalties.  The
module-level ``_build_polyline_mask`` rasterises the vertices for the coarse search.  The
per-image edge distance transform and gradient vectors are sampled from the context, built as
described in :doc:`dev_guide_techniques_image_derivatives`.  The search half-window comes from
``search_window_for_obs``, the coarse integer seed from ``coarse_ncc_search`` and the
sub-pixel solution from ``lm_subpixel_refine``, both in the shared
:doc:`dev_guide_techniques_dt_fitting` machinery, and the rotation pivot distance from
``rotation_pivot_distance_px``.

The result shape branches on ``context.fit_camera_rotation`` exactly as in
:doc:`dev_guide_techniques_body_limb`.  When the flag is false the covariance is two by two
and ``rotation_rad`` / ``sigma_rotation_rad`` are ``None`` (an unexpected covariance shape is
logged at warning and truncated); when it is true the covariance is the three-by-three
translation-plus-rotation information matrix with ``rotation_rad`` and ``sigma_rotation_rad``
populated, and an unexpected covariance shape raises :py:exc:`RuntimeError`.  The at-edge flag
fires when a translation axis reaches the search-window bound within ``at_edge_tolerance_px``
or the fitted rotation exceeds ``rotation_at_edge_fraction`` of the maximum.  The spurious
flag is the disjunction of a degenerate fit, a weighted or unweighted DT residual above the
``spurious_dt_*`` threshold, an inlier count below ``spurious_min_inliers``, an inlier
fraction below ``spurious_min_inlier_fraction``, or a refinement displacement above
``spurious_max_lm_displacement_px``.

The diagnostics object is a
:py:class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics` with fields
``visible_terminator_arc_fraction`` (vertex-count-weighted across the consumed features by the
private ``_aggregate_visible_arc_fraction``), ``visible_arc_px`` (the total vertex count),
``dt_fit_rms_px`` (the converged residual), ``lm_iterations`` (the refinement iteration
count), and ``tukey_inlier_count`` (the surviving inlier count).  The mean phase-angle factor
and mean albedo penalty are not stored on the dataclass; they are computed in ``navigate`` and
passed, alongside the diagnostics fields and the ``at_edge`` / ``spurious`` flags, into the
internal adapter consumed by ``evaluate_sigmoid_combination``.  The calibration is documented
in :doc:`dev_guide_techniques_confidence`, the per-term breakdown is logged through
``log_confidence_breakdown``, and the shared diagnostics dataclass is described in
:doc:`dev_guide_techniques_diagnostics`.

Examples
========

In the ``high_phase_terminator`` scene (Cassini NAC ``N1597846115_2``), a single high-phase
terminator arc fills part of the field with no other features.  Feasibility passes and the
terminator technique runs alongside the limb technique; the limb fit is selected as the
primary, and the terminator fit corroborates it.  Confidence on this scene is driven by the
high ``visible_terminator_arc_fraction`` and a favourable ``mean_phase_angle_factor`` at high
phase, with a small ``dt_fit_rms_px``.

In the ``multi_body`` scene (Cassini NAC ``N1487595731_1``), Dione and Rhea overlap at about
ninety degrees phase.  Feasibility passes, but the combined coarse search over the two bodies'
concatenated terminator polylines finds a wrong global maximum on the crescent geometry, and
the fit converges to about ``(11.58, 12.64)`` px -- roughly thirty pixels off in the column
axis from the operator ground truth ``(7.03, -18.42)`` px.  Despite the wrong offset the fit
reports sub-pixel ``dt_fit_rms_px`` with about seventy-six percent inliers, so neither the
``at_edge`` nor the ``spurious`` gate fires and the technique scores a high confidence of
about ``0.744`` on the wrong answer.  This is why the terminator technique carries the
``'fallback'`` tier; the ensemble's conflict handling, not the per-technique confidence,
prevents this result from being adopted.
