======================
Star Refine Navigation
======================

Overview
========

This technique is the pass-2 star refiner.  It consumes the prior offset produced
by the pass-1 ensemble and polishes it by re-centroiding every predicted catalog
star around its prior-shifted position, then averaging the per-star residuals into
a small correction.  Because it runs only after a prior offset exists, it is
registered as prior-required and the orchestrator invokes it only on the second
pass.  Feasibility passes when at least one usable catalog star is predicted in
the extended field of view; it fails when no usable star is present.

Theory
======

A pass-1 fit lands within a few pixels of the true offset but may carry residual
centroiding error.  Given a prior offset, the predicted position of each catalog
star is shifted by that prior, and a brightness peak is sought in a small
refinement window around the shifted prediction.  Where a peak is found, a
brightness-weighted centroid yields its sub-pixel position and the per-star
residual is the centroid minus the shifted prediction.  Stars whose nearest peak
sits beyond a maximum residual -- a wrong peak -- are dropped before the fit.

The surviving residuals are averaged in inverse-variance fashion, each star
weighted by the inverse trace of its Cramer-Rao position covariance so that
high-SNR stars count for more:

.. math::

   \Delta = \frac{\sum_i w_i\, r_i}{\sum_i w_i}, \qquad
   w_i = \frac{1}{\operatorname{tr}\,\Sigma_i}.

The correction :math:`\Delta` is reported as a delta from the prior, and the
ensemble adds it back.  The covariance reflects the per-axis scatter of the
surviving residuals; with two or more inliers that scatter is the reported
variance, while with a single inlier there is no scatter to measure and the
per-feature Cramer-Rao floor is reported instead.

A single-inlier refinement carries no independent cross-check beyond the prior it
was handed -- it is the same single observation that drove the pass-1 fit, merely
polished -- so its confidence is capped below the one-star unique-match ceiling so
the ensemble cannot promote a lone refine over the lone match that produced its
prior.  With two or more inliers the per-star residual scatter cross-checks the
joint fit and no such cap applies.

When a camera rotation is requested and two or more inliers survive, the refit is
an orthogonal Procrustes (similarity) fit reconstructing each star's detection
and prediction from the residuals, and the rotation variance follows the same
lever-arm formula the multi-star pattern matcher uses: the pooled residual
variance divided by the weighted catalog spread about its centroid, collapsing to
the rotation-unobservable sentinel when that spread is degenerate.  A
single-inlier refine under a rotation request reports rotation as unobservable
with the sentinel variance.  The reported covariance captures the surviving-star
residual scatter or the Cramer-Rao floor; it does not model SPICE pointing
systematics.

Configuration
=============

Tunables live under ``techniques.StarRefineNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``refine_window_px`` -- float, default ``6.0`` px.  Half-width of the per-star
  refinement window; tighter than the unique-match search window because the prior
  already places the prediction within a few pixels of the star.
- ``centroid_box_half_px`` -- int, default ``3`` px.  Half-width of the
  brightness-weighted centroid box around the detected peak.
- ``max_per_star_residual_px`` -- float, default ``4.0`` px.  Drop a star whose
  detection sits more than this far from the prior-shifted prediction; almost
  certainly a wrong peak.
- ``detection_sigma`` -- float, default ``4.0`` (dimensionless).  Detection threshold
  as a multiple of the per-pixel noise sigma.
- ``min_inliers`` -- int, default ``1`` (count).  Below this many surviving inliers
  the technique reports spurious.  Must be at least one.
- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  Slack around the
  search-window axis bounds for the at-edge check.
- ``single_inlier_confidence_cap`` -- float, default ``0.5`` (dimensionless).  Post-
  sigmoid confidence cap applied when only one inlier survives, set below the
  one-star unique-match ceiling so a lone refine cannot outrank the lone match that
  produced its prior.  Must lie in ``[0, 1]``.
- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  When the
  multi-inlier Procrustes path fits rotation, a converged rotation magnitude past
  this fraction of the per-image rotation cap trips at-edge; a single-inlier refine
  always reports rotation as unobservable so it is unaffected.

Confidence formula
-------------------

The confidence coefficients live in the ``techniques.StarRefineNav`` stanza of
``config_510_techniques.yaml``.  The sigmoid baseline is ``alpha0 = -1.0`` and
hard-zero gates force confidence to zero when ``at_edge`` or ``spurious`` is true;
the single-inlier cap above is applied after the sigmoid.  See
:doc:`dev_guide_techniques_confidence` for the sigmoid mathematics.

- ``n_stars_used`` -- alpha = 1.0, offset = 0, divisor = 5.0, cap at 1.0.  Number of
  stars that survived the per-star quality gates; more inliers add independent
  constraints.
- ``median_pos_err_px`` -- alpha = -1.0, offset = 0, divisor = 1.0, no cap.  Median
  per-star refinement positional error; a lower error means a tighter refinement.
- ``residual_scatter_px`` -- alpha = -1.0, offset = 0, divisor = 1.0, no cap.  Per-axis
  RMS scatter of the per-star residuals; lower scatter is a more internally
  consistent fit.

Implementation
==============

Source files: ``src/nav/nav_technique/nav_technique_star_refine.py`` and the shared
star helpers in ``nav.nav_technique._star_helpers``.  The public class is
:py:class:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav`, a subclass
of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  Its
``accepts_feature_types`` is the single ``STAR`` feature type, its
``requires_prior`` is ``True`` (it runs in pass 2), and its
``confidence_attributes`` set names ``at_edge``, ``spurious``, ``n_stars_used``,
``median_pos_err_px``, and ``residual_scatter_px``.

:py:meth:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav.is_feasible`
reads only feature metadata and returns feasible when at least one usable ``STAR``
feature is present.

:py:meth:`~nav.nav_technique.nav_technique_star_refine.StarRefineNav.navigate`
reads the prior offset from the context; because the technique is prior-required,
the orchestrator only calls it on pass 2, but if the prior is somehow absent the
method returns through the private ``_fail`` path.  Given a prior it collects
per-star residuals through the private ``_collect_residuals`` (each star
re-centroided around its prior-shifted prediction, dropped when no peak is found
or the residual exceeds the maximum), returns ``_fail`` when fewer than the
minimum inliers survive, and otherwise computes the inverse-variance-weighted
delta, the median positional error, and the per-axis residual scatter.

The result shape branches on the inlier count and on whether rotation is fit:

- No rotation: the ``(2, 2)`` covariance from ``_build_covariance`` (the Cramer-Rao
  floor with one inlier, the residual scatter with two or more); ``rotation_rad``
  and ``sigma_rotation_rad`` are ``None``; the reported offset is the prior plus
  the delta.
- Rotation requested with two or more inliers: the private ``_fit_rotation_3dof``
  runs a Procrustes refit, returning the rotation angle, its sigma, a full
  ``(3, 3)`` covariance, and the absolute offset; the at-edge flag is recomputed
  against that absolute offset and the rotation-at-edge test.
- Rotation requested with a single inlier: the ``(2, 2)`` covariance is promoted to
  the rank-deficient ``(3, 3)`` form via ``embed_rotation_unobservable``,
  ``rotation_rad`` is fixed at ``0.0`` and ``sigma_rotation_rad`` is the
  rotation-unobservable sentinel from ``rotation_unobservable_sigma_rad``.

A single-inlier result additionally has its post-sigmoid confidence clamped to the
single-inlier cap.  The YAML formula is evaluated through
``evaluate_sigmoid_combination`` and logged through ``log_confidence_breakdown``
(see :doc:`dev_guide_techniques_confidence`); the inverse-variance weighting reuses
the per-feature Cramer-Rao trace through the module-private
``_trace_inverse_variance`` helper.

Every field of
:py:class:`~nav.nav_technique.diagnostics.StarRefineDiagnostics` is populated and
feeds the confidence formula above:
:py:attr:`~nav.nav_technique.diagnostics.StarRefineDiagnostics.n_stars_used`,
:py:attr:`~nav.nav_technique.diagnostics.StarRefineDiagnostics.median_pos_err_px`,
and
:py:attr:`~nav.nav_technique.diagnostics.StarRefineDiagnostics.residual_scatter_px`.
The return value is a
:py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

**one_bright_star_no_body (W1449079117_1_CALIB).** A single bright star (Vega) in
a wide-angle frame.  On pass 1 the unique-match technique produces a prior; on
pass 2 this technique refines it.  Because only one star is predictable, the
refine has a single inlier, so its confidence is capped at the single-inlier limit
of 0.5 and the per-feature Cramer-Rao floor is reported as the covariance.  The
sidecar records ``primary_technique: StarRefineNav`` with ``status: success`` and
``confidence_tier: low`` -- the refined offset is accepted but at modest
confidence, against the operator ground truth of ``(3.06, -0.02)`` px.  This scene
exercises the prior-required pass-2 hand-off and the single-inlier branch.

**star_dominated (W1580760393_1_CALIB).** A dense star field would, given a
successful pass-1 prior, give this technique many inliers, so the residual scatter
would cross-check the joint fit and the single-inlier cap would not apply.  On the
corpus frame the pass-1 confidence falls below the orchestrator floor (the sidecar
pins ``status: failed``), so no prior is promoted and the refiner is not the
primary technique there; the multi-inlier branch is the one it would take whenever
a prior is available.
