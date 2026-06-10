====================
Body Limb Navigation
====================

Overview
========

The body-limb technique exploits the sharp brightness edge at the sunlit horizon of a
resolved body: where the body silhouette ends and empty sky begins, the image gradient
spikes, and the predicted geometric limb of an ellipsoidal body is a clean curve whose shape
is known from SPICE.  The technique concatenates every predicted limb-arc polyline in the
input feature set, weights each vertex by its prior normal-direction precision, and runs the
shared distance-transform fitter to recover the single translation (and, on cameras whose
rotation is fitted, the single rotation) that best slides the predicted limb onto the
observed edge.  Feasibility passes when at least one predicted limb arc survives with enough
visible vertices to constrain a two-dimensional translation.  Feasibility fails when no
limb-arc feature reaches that minimum visible length -- a body too small to resolve, a body
whose limb is entirely off-screen, or a body whose limb the reliability gate has already
dropped (for example a fully-lit disc whose limb saturates the incidence penalty).

Theory
======

A resolved body presents a silhouette edge where the lit surface meets the background.  For
an ellipsoidal body whose pose is known from spacecraft ephemeris, the projected limb is a
smooth planar curve; the only unknown is the small pointing error that displaces the whole
predicted curve from where the edge actually lies in the image.  Because the error is a rigid
displacement of the camera boresight, every vertex of every limb arc shares the same offset,
so a single translation explains all of them jointly.

Let the predicted limb be sampled at vertices :math:`\mathbf{p}_i` with outward surface
normals :math:`\mathbf{n}_i`.  An image-side distance transform assigns to every pixel its
Euclidean distance to the nearest detected edge.  For a candidate offset
:math:`\boldsymbol{\delta} = (\delta v, \delta u)` the signed residual at vertex :math:`i` is
the distance-transform value sampled at the shifted vertex:

.. math::

   r_i(\boldsymbol{\delta}) = D\!\left(\mathbf{p}_i + \boldsymbol{\delta}\right),

where :math:`D` is the bilinearly interpolated edge distance transform.  The fit minimises the
robust, precision-weighted sum of squared residuals

.. math::

   C(\boldsymbol{\delta}) = \sum_i \frac{\rho\!\left(r_i\right)}{\sigma_i^{2}},

where :math:`\sigma_i` is the prior one-sigma uncertainty of vertex :math:`i` along its
normal direction and :math:`\rho` is the Tukey biweight loss, which redescends to zero for
residuals beyond a tuned multiple of the robust residual scale.  Vertices whose gradient
direction disagrees with the expected limb polarity -- the image gradient should point into
the bright silhouette -- are dropped before the fit so that an interior crater rim cannot be
mistaken for the limb.

The optimisation proceeds in two stages.  A coarse integer-pixel stage scans the
translation-search window by mask overlap to seed a basin; a Levenberg-Marquardt stage with
Tukey reweighting then refines to sub-pixel precision inside a trust region centred on that
seed.  Convergence is declared when the combined step norm drops below a small tolerance.  The
M-estimator's information matrix yields the reported covariance; with rotation enabled the
parameter vector gains a rotation angle about the limb centroid and the covariance grows to
three by three.

The technique is unobservable when no limb arc is long enough to pin both translation axes:
a short arc seen edge-on constrains its normal direction but leaves the tangent direction
free, so a single tiny limb fragment yields a rank-deficient fit.  Multi-body scenes sharpen
the solution -- joint constraints from several bodies reduce the translation uncertainty
roughly as the square root of the body count when the relative SPICE geometry is correct --
and the single-translation parameterisation cannot represent a "swap two moons" mistake by
construction.  The reported covariance captures only the residual scatter of the surviving
inliers about the converged limb; it does not capture SPICE pointing bias, limb-shape error
from topography on a non-ellipsoidal body, or a wholesale mis-convergence onto an interior
edge.  Those failure modes are caught instead by separate spurious gates: an elevated
unweighted residual, too few surviving inliers, too small an inlier fraction, or a refinement
step that walked far from the coarse seed.

Configuration
=============

Runtime tunables live under ``techniques.BodyLimbNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``min_arc_px`` -- float, default ``30.0`` px.  Minimum surviving polyline vertex count per
  LIMB_ARC for feasibility; raising it rejects shorter arcs that under-constrain the
  translation.
- ``spurious_dt_rms_factor`` -- float, default ``5.0`` (dimensionless).  Multiplier on the
  smallest limb sigma that sets the DT residual ceiling; lowering it flags marginal fits
  spurious sooner.
- ``spurious_dt_floor_px`` -- float, default ``3.0`` px.  Lower bound on the DT residual
  ceiling so tight-sigma limbs are not held to an unreasonably small threshold.
- ``spurious_min_inliers`` -- int, default ``6`` (count).  Minimum Tukey inlier count below
  which the M-estimator covariance is no longer trusted and the result is flagged spurious.
- ``spurious_min_inlier_fraction`` -- float, default ``0.20`` (dimensionless).  Minimum
  fraction of vertices that must survive Tukey reweighting; below it the fit has almost
  certainly walked off the limb onto interior features.
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
``techniques.BodyLimbNav`` stanza.  The sigmoid argument starts from ``alpha0`` of
``-1.0`` and adds the linear terms below; the sigmoid mathematics is documented in
:doc:`dev_guide_techniques_confidence`.  The gate ``hard_zero_if`` forces confidence to zero
when ``at_edge`` or ``spurious`` is true.

- ``visible_limb_arc_fraction`` -- alpha = 3.0, offset = 0, divisor = 1, no cap.  Fraction of
  the predicted limb that was visible and consumed; the dominant positive term.
- ``dt_fit_rms_px`` -- alpha = -1.5, offset = 0, divisor = 1, no cap.  Final root-mean-square
  DT residual; larger residuals pull confidence down.
- ``visible_arc_px`` -- alpha = 0.4, offset = 0, divisor = 100.0, cap at 1.0.  Total surviving
  arc length, normalised so a long limb saturates the term.

Implementation
==============

The technique lives in ``src/nav/nav_technique/nav_technique_body_limb.py``;
:py:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` subclasses
:py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  It declares
``accepts_feature_types`` of ``frozenset({NavFeatureType.LIMB_ARC})``, ``requires_prior`` of
``False`` (it runs in the prior-free pass 1), and a ``confidence_attributes`` set of
``at_edge``, ``spurious``, ``visible_limb_arc_fraction``, ``visible_arc_px``,
``dt_fit_rms_px``, ``lm_iterations``, and ``tukey_inlier_count``.

:py:meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.is_feasible` reads only the
polyline vertex count per feature and returns a feasibility report, described under
:doc:`dev_guide_techniques_feasibility`, that is feasible when at least one limb arc reaches
``min_arc_px`` vertices.

:py:meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate` opens a logged
technique section, validates that the context carries the image edge distance transform and
gradient vectors, and drops arcs shorter than ``min_arc_px``.  The private helper
``_aggregate_limb_features`` concatenates the surviving vertices, negates the geometric
outward normals into polarity normals, and gathers the per-vertex sigmas.  The module-level
``_build_polyline_mask`` rasterises the vertices into a boolean mask for the coarse search.
The per-image edge distance transform and gradient vectors are sampled directly from the
context; their construction is documented in :doc:`dev_guide_techniques_image_derivatives`.
The translation-search half-window comes from ``search_window_for_obs``.  The coarse integer
seed comes from ``coarse_ncc_search`` and the sub-pixel solution from ``lm_subpixel_refine``,
both in the shared :doc:`dev_guide_techniques_dt_fitting` machinery; the rotation pivot
distance, used only when rotation is fitted, comes from ``rotation_pivot_distance_px``.

The result shape branches on ``context.fit_camera_rotation``.  When it is false (the Cassini
and New Horizons LORRI posture) the covariance is two by two and ``rotation_rad`` /
``sigma_rotation_rad`` are ``None``; a non-two-by-two covariance returned by the fitter is
logged at warning and truncated to the translation block.  When it is true (Voyager ISS and
Galileo SSI) the covariance is the three-by-three translation-plus-rotation information
matrix, ``rotation_rad`` is the converged angle and ``sigma_rotation_rad`` is the square root
of its diagonal; an unexpected covariance shape raises :py:exc:`RuntimeError`.  The at-edge
flag fires when either translation axis reaches the search-window bound within
``at_edge_tolerance_px`` or when the fitted rotation exceeds ``rotation_at_edge_fraction`` of
the maximum.  The spurious flag is the disjunction of a degenerate fit, a weighted or
unweighted DT residual above the ``spurious_dt_*`` threshold, an inlier count below
``spurious_min_inliers``, an inlier fraction below ``spurious_min_inlier_fraction``, or a
refinement displacement above ``spurious_max_lm_displacement_px``.

The diagnostics object is a
:py:class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics` with fields
``visible_limb_arc_fraction`` (vertex-count-weighted across the consumed features by the
private ``_aggregate_visible_arc_fraction``), ``visible_arc_px`` (the total vertex count),
``dt_fit_rms_px`` (the converged residual), ``lm_iterations`` (the refinement iteration
count), and ``tukey_inlier_count`` (the surviving inlier count).  Confidence is evaluated by
``evaluate_sigmoid_combination`` against an internal adapter that exposes those diagnostics
alongside the ``at_edge`` and ``spurious`` flags; the calibration is documented in
:doc:`dev_guide_techniques_confidence`, and the per-term breakdown is logged through
``log_confidence_breakdown``.  The shared diagnostics dataclass is described in
:doc:`dev_guide_techniques_diagnostics`.

Examples
========

In the ``high_phase_terminator`` scene (Cassini NAC ``N1597846115_2``), a single high-phase
limb-and-terminator body fills part of the field with no other features.  Feasibility passes:
the limb arc clears ``min_arc_px``.  The fit converges to roughly ``(6.09, 1.19)`` px, within
about one pixel of the operator ground truth ``(5.19, 1.30)`` px, and the limb result is
selected as the primary technique.  Confidence is driven by a high
``visible_limb_arc_fraction`` and a small ``dt_fit_rms_px``; neither the ``at_edge`` nor the
``spurious`` gate fires.

In the ``multi_body`` scene (Cassini NAC ``N1487595731_1``), Dione and Rhea overlap at about
ninety degrees phase.  Feasibility passes and the limb fit converges to ``(7.00, -18.00)``
px, within about one pixel of the operator ground truth ``(7.03, -18.42)`` px, with a
confidence of about ``0.239``.  The limb offset agrees closely with the disc technique on the
same scene; the conflicted overall verdict on this scene comes from a separate terminator
mis-convergence, not from the limb fit.
