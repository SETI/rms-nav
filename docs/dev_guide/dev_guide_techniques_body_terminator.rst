================================================
Body Terminator Fit (BodyTerminatorNav)
================================================

Overview
========

:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` recovers a single
translation from one or more body terminator polylines by aligning each polyline against the
image's edge-distance-transform. The technique consumes every
:data:`~nav.feature.feature_type.NavFeatureType.TERMINATOR_ARC` feature offered by the
orchestrator, weights every vertex of a given body uniformly by an inverse-variance derived
from that body's mean per-vertex normal sigma, and runs the same coarse-NCC plus
Tukey-reweighted Levenberg-Marquardt refinement that
:doc:`dev_guide_techniques_dt_fitting` describes for the limb fit. The output is the joint
translation that minimises the summed weighted squared distance from the model polylines to
the image edges, plus a covariance derived from the M-estimator information matrix at
convergence.

Feasibility passes when at least one offered ``TERMINATOR_ARC`` carries enough surviving vertices
to constrain a 2-D translation; feasibility fails when every offered ``TERMINATOR_ARC`` has fewer
than the minimum-arc-length floor (a body whose phase angle is too low for a usable
terminator, or whose terminator is mostly hidden by the FOV boundary).

Theory
======

The technique belongs to the distance-transform family of polyline fitters: predicted
polylines are shifted as a rigid body until their vertices lie as close as possible to the
nearest image edge, where "as close as possible" is measured against a precomputed image-edge
distance transform. The shared algorithmic infrastructure handles the heavy lifting; this
section describes the cost function and conventions specific to the terminator fit.

Cost function
-------------

The technique minimises

.. math::

    C(\Delta v, \Delta u, \theta) = \sum_{i} w_{i}(\Delta v, \Delta u, \theta) \,
        \mathrm{DT}\bigl[\,R(\theta)\,(x_{i} - x_{p}) + x_{p} + (\Delta v, \Delta u)\,\bigr]^{2}

where :math:`x_{i}` are the input vertices concatenated across every consumed ``TERMINATOR_ARC``,
:math:`x_{p}` is the rotation pivot (the centroid of the concatenated vertices),
:math:`R(\theta)` is the in-plane rotation matrix, :math:`\mathrm{DT}` is the bilinearly
sampled image-edge distance transform, and the per-vertex weight :math:`w_{i}` is the product
of a per-body uniform precision (the inverse mean variance across the body's terminator
polyline) and a Tukey biweight evaluated at the scaled DT residual. When the per-instrument
camera-rotation flag is off the parameter vector collapses to :math:`(\Delta v, \Delta u)`
and the rotation term is dropped.

Per-body uniform weighting
--------------------------

Unlike the limb fit, every vertex of a given body's terminator shares one inverse-variance
weight derived from that body's mean per-vertex normal sigma. Cross-body weighting reflects
the structural fact that low-albedo bodies provide tighter terminators than high-albedo ones
(an albedo gradient blurs the apparent terminator more than a smooth surface would), so the
mean per-vertex sigma already captures the per-body terminator quality and a single per-body
weight avoids over-weighting a long terminator from a low-confidence body.

Search strategy
---------------

The fit proceeds in the same two stages as the limb fit: a coarse integer cross-correlation
between the rendered terminator polyline mask and the thresholded image edge mask, then a
sub-pixel Levenberg-Marquardt refinement against the bilinearly sampled DT. See
:doc:`dev_guide_techniques_dt_fitting` for the per-iteration mechanics.

Robustness
----------

The Tukey biweight redescender drops vertices whose scaled residuals exceed the Holland-Welsch
cutoff. Polarity filtering — comparing the model normal at each vertex against the local
image gradient direction — is enabled with the model normals oriented so the gradient should
point from the dark side toward the lit side at the terminator. Polarity-rejected vertices
contribute a synthetic near-infinite residual on every iteration so the Tukey biweight zeroes
their weight on the first reweighting; this keeps the terminator fit from latching onto the
opposite-polarity edge of a neighbouring body's limb.

Restrictions and assumptions
----------------------------

- The orchestrator must supply both an image-edge distance transform and a per-pixel gradient
  vector image on the per-image
  :class:`~nav.nav_orchestrator.nav_context.NavContext`; in their absence the navigation
  aborts with a :exc:`RuntimeError`.
- The terminator is assumed to be photometrically distinguishable from the limb. At phase
  angles below approximately 3 degrees the terminator is too close to the limb to be
  usefully separated; the model-side
  :data:`~nav.nav_model.nav_model_body.TERMINATOR_MIN_PHASE_FACTOR` gate suppresses emission
  in that regime so the technique never sees those features.
- The albedo penalty per body is supplied by the model side via the per-body
  :class:`~nav.nav_model.body_shape.BodyShape`; the technique reads it off the feature flags
  but does not re-derive it.
- Multi-body inputs are fused into a single translation by concatenating their per-vertex
  arrays. The joint-translation parameterisation cannot represent disagreement between
  bodies about the offset; if SPICE relative geometry is wrong the joint fit walks toward the
  higher-vertex-count body and the lower-vertex body's residuals appear as outliers the
  Tukey weight zeroes out.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence, scaled by the per-vertex Tukey weights. It does not capture
systematic biases (an under-modelled per-body albedo gradient propagates straight into the
covariance) and it does not capture model-side uncertainty in the SPICE prediction itself.
When the converged offset sits within a small tolerance of any axis bound of the search
window, or when the rotation parameter is at the configured fraction of its cap, the result
is flagged :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
confidence formula's hard-zero gate forces confidence to zero. The spurious tests gate on
the Tukey-weighted DT residual RMS, the unweighted (raw) DT residual RMS against the same
threshold (so a fit where Tukey rejects a wholly mis-aligned arc cannot pass on its
collapsed weighted RMS), the degenerate flag, the inlier count and fraction, and the LM
displacement from the coarse seed; when any of these fails, the result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` and similarly forced
to zero.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyTerminatorNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``min_arc_px`` — float, default ``30.0`` px. Minimum surviving vertex count per
  ``TERMINATOR_ARC`` for feasibility. Shorter terminators do not constrain a 2-D translation
  enough to be worth the LM iteration.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many terminator-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``4.0`` px. Floor of the spurious-detection
  threshold; terminators are softer than limbs so the floor sits one pixel wider than the
  limb fit's.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the
  bilinear-DT half-cell width.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses this
  fraction of the per-image
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

The six keys above are global; the per-instrument YAML files in
``src/nav/config_files/config_4N0_inst_*.yaml`` do not override any of them. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` rather than from this
block.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence` for the per-term arithmetic.
The formula spec is ``techniques.BodyTerminatorNav`` in the same YAML file and consumes
attributes off :class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics` plus the
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` flags.

- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_terminator_arc_fraction`
  — alpha = 2.0, offset = 0.0, divisor = 1.0, no cap. Vertex-weighted average of the
  per-feature visible-arc fraction across consumed ``TERMINATOR_ARC`` features.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.dt_fit_rms_px` —
  alpha = -1.0, offset = 0.0, divisor = 1.0, no cap. Final root-mean-square DT residual
  after LM convergence; smaller is sharper.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_arc_px` —
  alpha = 0.4, offset = 0.0, divisor = 100.0, cap at 1.0. Total surviving polyline length in
  pixels, capped after normalisation. More polyline earns confidence up to a 100-pixel
  saturation point.
- ``mean_phase_angle_factor`` — alpha = 1.0, offset = 0.0, divisor = 1.0, no cap. Mean of
  :math:`\sin(\phi)` across consumed terminators (read off the per-feature
  :class:`~nav.feature.flags.TerminatorArcFlags`). High-phase scenes earn confidence; low
  phase pulls confidence down.
- ``mean_albedo_penalty`` — alpha = -1.5, offset = 0.0, divisor = 1.0, no cap. Mean of the
  per-body albedo penalty (read off the per-feature reliability breakdown). High-albedo
  bodies pull confidence down; uniform low-albedo bodies leave it unchanged.

Hard-zero gate: :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces
confidence to zero before the sigmoid evaluates. The constant baseline is
:math:`\alpha_{0} = -1.0`. No post-sigmoid ``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_body_terminator.py`` —
  :class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` and its private
  aggregation / polyline-mask helpers.
- ``src/nav/nav_technique/dt_fitting.py`` — the shared coarse-NCC and LM-refinement helpers
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/nav/nav_orchestrator/image_derivatives.py`` — the per-image gradient / DT derivatives
  attached to :class:`~nav.nav_orchestrator.nav_context.NavContext`; documented at
  :doc:`dev_guide_techniques_image_derivatives`.
- ``src/nav/nav_technique/confidence.py`` — the shared sigmoid-combination formula evaluator;
  documented at :doc:`dev_guide_techniques_confidence`.
- ``src/nav/nav_technique/diagnostics.py`` —
  :class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class
:class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav`, base
:class:`~nav.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so the orchestrator's ``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.name` —
  ``'BodyTerminatorNav'``.
- :attr:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.accepts_feature_types`
  — ``frozenset({TERMINATOR_ARC})``.
- :attr:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.requires_prior` —
  ``False``. Runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.confidence_attributes`
  — ``{'at_edge', 'spurious', 'visible_terminator_arc_fraction', 'visible_arc_px',
  'dt_fit_rms_px', 'lm_iterations', 'tukey_inlier_count', 'mean_phase_angle_factor',
  'mean_albedo_penalty'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.navigate`.

Diagnostics
-----------

:class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics`:

- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_terminator_arc_fraction`
  — vertex-weighted average of the per-feature visible-arc fraction across consumed
  ``TERMINATOR_ARC`` features. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.visible_arc_px` — total
  surviving polyline arc length in pixels. Consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.dt_fit_rms_px` — weighted
  RMS DT residual at the converged pose. Consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.lm_iterations` — number of
  LM iterations actually performed.
- :attr:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics.tukey_inlier_count` —
  number of vertices that retained a strictly positive Tukey weight at the final estimate.
  Consumed by the spurious-detection gate.

Call path traced through
:meth:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav.navigate`:

1. Open a logged section. Fail fast (:exc:`RuntimeError`) if either
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing.
2. Filter the offered features down to ``TERMINATOR_ARC`` polylines whose surviving vertex count
   is at least ``min_arc_px``, then concatenate the per-feature vertex arrays via the
   private aggregation helper. The helper also computes per-body sigma scalars (mean of the
   per-vertex normal sigmas) and the per-body phase / albedo flags.
3. Build a binary polyline mask and pull the search-window margin off the observation via
   :func:`~nav.nav_technique.nav_technique.search_window_for_obs`. Run
   :func:`~nav.nav_technique.dt_fitting.coarse_ncc_search` on the polyline mask and the
   thresholded edge mask to obtain an integer seed offset.
4. Decide whether to fit camera rotation by reading
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`. When rotation
   is fit, the rotation pivot is set to the centroid of the concatenated vertices and the
   pivot-to-image-centre distance is computed via
   :func:`~nav.nav_technique.nav_technique.rotation_pivot_distance_px`.
5. Call :func:`~nav.nav_technique.dt_fitting.lm_subpixel_refine` with the polyline,
   per-vertex sigmas, the edge DT, the gradient image, the integer seed, and the rotation
   options.
6. Result-shape branches on
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block. Any non-(2, 2) covariance returned by the refiner is logged
     at WARNING and truncated.
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` and
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` are
     ``None``.
   - **Rotation fit.**
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (3, 3) translation + rotation information matrix.
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.rotation_rad` is the
     converged angle and
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` is the
     square root of its diagonal. An unexpected covariance shape raises
     :exc:`RuntimeError`.

7. Apply the at-edge tests against translation axis bounds and the rotation cap, plus the
   spurious tests against the final RMS, the inlier count, and the per-feature
   sigma-floor multiple.
8. Build a :class:`~nav.nav_technique.diagnostics.BodyTerminatorDiagnostics`, evaluate the
   confidence spec via :func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination`,
   log the per-term breakdown via
   :func:`~nav.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.feature_ids` field
preserves every consumed
:attr:`~nav.feature.feature.NavFeature.feature_id` so the orchestrator's curator can
attribute each contribution at audit time.

Examples
========

``high_phase_terminator`` (Cassini ISS NAC, image ``N1597846115_2``)
    A high-phase terminator arc with no other features in the FOV.
    :class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` consumes the
    single ``TERMINATOR_ARC`` feature offered by the body model and converges within ~1 px of
    the operator-verified offset :math:`(\Delta v, \Delta u) = (5.19, 1.30)` px.

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right with about 22 % of the disc off-frame. The body model
    emits a ``TERMINATOR_ARC`` feature alongside ``LIMB_ARC`` and ``BODY_DISC``; on this scene the
    terminator fit reports zero Tukey inliers (the LM does not iterate) and the spurious
    gate forces the confidence to zero. The
    :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` fit is the technique that
    actually drives the orchestrator's primary on this image; see
    :doc:`dev_guide_techniques_body_limb` for that walk-through.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees.
    Two ``TERMINATOR_ARC`` features are offered;
    :class:`~nav.nav_technique.nav_technique_body_terminator.BodyTerminatorNav` fuses them
    into a joint translation. On this scene the multi-body crescent geometry produces a
    coarse-NCC seed at a wrong local minimum (~31 px off-axis); the LM converges near that
    seed and reports sub-pixel RMS with most vertices as inliers, scoring high confidence on
    a wrong answer. The orchestrator's ensemble combine, combined with the limb and disc
    techniques' agreement around the operator-verified offset
    :math:`(\Delta v, \Delta u) = (7.03, -18.42)` px, refuses to commit and reports
    ``status=conflicted``.
