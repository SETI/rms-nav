===========================
Body Limb Fit (BodyLimbNav)
===========================

Overview
========

:class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` recovers a single translation
from one or more body limb polylines by aligning
each polyline against the image's edge-distance-transform. The technique consumes every
:data:`~nav.feature.feature_type.NavFeatureType.LIMB_ARC` feature offered by the orchestrator,
weights each vertex by its prior-precision sigma, and runs a coarse normalised-cross-correlation
search followed by a Tukey-reweighted Levenberg-Marquardt refinement. The output is the
joint translation that minimises the summed weighted squared distance from the model polylines
to the image edges, plus a covariance derived from the M-estimator information matrix at
convergence.

Feasibility passes when at least one offered ``LIMB_ARC`` carries enough surviving vertices to
constrain a 2-D translation; feasibility fails when every offered ``LIMB_ARC`` has fewer than the
minimum-arc-length floor (a body whose limb is mostly hidden by the FOV boundary, occluded by
another body, or lost to shadow).

Theory
======

The technique belongs to the distance-transform family of polyline fitters: predicted polylines
are shifted as a rigid body until their vertices lie as close as possible to the nearest image
edge, where "as close as possible" is measured against a precomputed image-edge distance
transform. Shared algorithmic infrastructure handles the heavy lifting; this section describes
the cost function and conventions specific to the limb fit.

Cost function
-------------

The technique minimises

.. math::

    C(\Delta v, \Delta u, \theta) = \sum_{i} w_{i}(\Delta v, \Delta u, \theta) \,
        \mathrm{DT}\bigl[\,R(\theta)\,(x_{i} - x_{p}) + x_{p} + (\Delta v, \Delta u)\,\bigr]^{2}

where :math:`x_{i}` are the input vertices concatenated across every consumed ``LIMB_ARC``,
:math:`x_{p}` is the rotation pivot (the centroid of the concatenated vertices),
:math:`R(\theta)` is the in-plane rotation matrix, :math:`\mathrm{DT}` is the bilinearly
sampled image-edge distance transform, and the per-vertex weight :math:`w_{i}` is the product
of the prior precision :math:`1 / \sigma_{i}^{2}` (with :math:`\sigma_{i}` the per-vertex normal
sigma supplied by the body model) and a Tukey biweight evaluated at the scaled DT residual
:math:`\mathrm{DT}_{i} / \sigma_{i}`. When the per-instrument camera-rotation flag is off the
parameter vector collapses to :math:`(\Delta v, \Delta u)` and the rotation term is dropped.

Search strategy
---------------

The fit proceeds in two stages:

1. **Coarse integer search.**  The model polyline is rendered into a binary mask, the image
   edges are thresholded into a binary mask of their own (the truncated DT thresholded at
   half a pixel), and an integer-pixel cross-correlation is evaluated over a search window
   bracketing the per-instrument SPICE pointing-error envelope. The argmax of the correlation
   is the seed translation.
2. **Sub-pixel Levenberg-Marquardt refinement.**  Starting from the integer seed, the refiner
   evaluates the cost above, its parameter Jacobian (central differences against the bilinear
   DT), and an LM-damped normal-equation step. After each accepted step the Tukey weights are
   recomputed against the new residuals (iteratively reweighted least squares), so vertices
   that drifted onto an unrelated edge during refinement progressively lose weight.

Robustness
----------

The Tukey biweight is the redescender used by the LM reweighting; its asymptotic 95 % efficiency
constant is the documented default. Vertices whose model normal disagrees with the local image
gradient direction (a *polarity* mismatch — for a bright body on a dark background the gradient
points outward, into the silhouette's exterior) are assigned a near-infinite synthetic residual
on every iteration so the Tukey biweight zeroes their weight on the first reweighting; this
keeps the limb fit from latching onto the body's interior crater rims.

Restrictions and assumptions
----------------------------

- The orchestrator must supply both an image-edge distance transform and a per-pixel gradient
  vector image on the per-image
  :class:`~nav.nav_orchestrator.nav_context.NavContext`; in their absence the navigation aborts
  with a runtime error.
- The vertices and per-vertex normal sigmas must be physically meaningful — vertices with zero
  or negative sigma are rejected by the LM refiner.
- The fit assumes the body is bright against a dark background. The polarity rule is hard-coded
  by inverting the geometric outward normal so that the sign of the test matches the
  bright-on-dark gradient direction; this is correct for every supported instrument's body
  scenes.
- Multi-body inputs are fused into a single translation by concatenating their per-vertex
  arrays. The joint-translation parameterisation cannot represent disagreement between bodies
  about the offset; if SPICE relative geometry is wrong (a body misidentification, a stale SPK)
  the joint fit walks toward the higher-vertex-count body and the lower-vertex body's residuals
  appear as outliers the Tukey weight zeroes out.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence, scaled by the per-vertex Tukey weights. The covariance therefore
reflects the *shape* of the cost surface near the minimum and the surviving inlier population;
it does not capture systematic biases (e.g. an inflation of the per-vertex sigma due to
unmodelled crater roughness) and it does not capture model-side uncertainty in the SPICE
prediction itself (the search-window margin is what bounds that).

One such systematic is a **sub-pixel-phase bias floor of ~0.13-0.15 px** in the recovered
offset. It is *independent of SNR* (it persists on a clean, high-signal frame), so it is not
a noise effect: it arises because the image-edge distance transform is built from a binary
thresholded-and-NMS'd edge mask and is then integer-quantized before bilinear sampling, so the
distance field the Levenberg-Marquardt fit minimizes against is itself quantized at the
sub-pixel scale (see :doc:`dev_guide_techniques_dt_fitting`). Unlike the disc correlator's
sharp-edge S-curve -- which a band-limit in the cross-power refinement removes -- this floor
lives in the distance-transform construction, so a correlator-side fix does not touch it. The
intended remedy is to fit the final sub-pixel offset against the *continuous* gradient field
rather than the quantized DT (the DT/NCC stage still does coarse acquisition); that work is a
separate follow-up. Until then the limb fit is the least precise of the point-feature
techniques on a well-resolved body, though it stays well inside the navigability bound. When the converged offset
sits within a small tolerance of any axis bound of the search window, or when the rotation
parameter is at the configured fraction of its cap, the result is flagged :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and the
confidence formula's hard-zero gate forces confidence to zero. The spurious tests gate on
the Tukey-weighted DT residual RMS, the unweighted (raw) DT residual RMS against the same
threshold (so a fit where Tukey rejects a wholly mis-aligned arc cannot pass on its
collapsed weighted RMS), the degenerate flag, the inlier count and fraction, and the LM
displacement from the coarse seed; when any of these fails, the result is flagged
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` and similarly
forced to zero.

Configuration
=============

All numeric tunables for this technique live in ``techniques.BodyLimbNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``min_arc_px`` — float, default ``30.0`` px. Minimum surviving vertex count per ``LIMB_ARC``
  for feasibility. Shorter limbs do not constrain a 2-D translation enough to be worth the
  LM iteration.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many limb-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``3.0`` px. Floor of the spurious-detection
  threshold; the threshold is the larger of the floor and the per-feature sigma multiple.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``spurious_min_inlier_fraction`` — float, default ``0.05`` (dimensionless). Below this
  inlier fraction the LM has almost certainly walked off the true limb onto internal-body
  features (crater rims, terminator); the result is flagged spurious.
- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the bilinear-DT half-cell width.
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the converged rotation magnitude trips :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` once it
  crosses this fraction of the per-image :attr:`~nav.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

The seven keys above are global; the per-instrument YAML files in
``src/nav/config_files/config_4N0_inst_*.yaml`` do not override any of them. The
search-window margin used by the at-edge test comes from the per-instrument
:class:`~nav.nav_orchestrator.instrument_config.InstrumentSettings` rather than from this block.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared sigmoid
combination, see :doc:`dev_guide_techniques_dt_fitting` for the per-term arithmetic and
:doc:`dev_guide_techniques` for the family-level overview of confidence. The formula spec is
``techniques.BodyLimbNav`` in the same YAML file and consumes attributes off
:class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics` plus the :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and
:attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` flags carried on the result.

- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction` —
  alpha = 3.0, offset = 0.0, divisor = 1.0, no cap.
  Fraction of the polyline (weighted by surviving vertex count across consumed ``LIMB_ARC``
  features) whose vertices were not pre-rejected by the model-side shadow / FOV gates. One
  means every offered vertex is usable.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px` — alpha = -1.5,
  offset = 0.0, divisor = 1.0, no cap. Final root-mean-square DT residual after LM
  convergence; smaller is sharper.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px` — alpha = 0.4,
  offset = 0.0, divisor = 100.0, cap at 1.0. Total surviving polyline length in pixels,
  capped after normalisation. More polyline earns confidence up to a 100-pixel saturation
  point.

Hard-zero gate: :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` and :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` either firing forces the confidence to zero before
the sigmoid is evaluated. The constant baseline is :math:`\alpha_{0} = -1.0`. No post-sigmoid
``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/nav/nav_technique/nav_technique_body_limb.py`` —
  :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` and its private aggregation /
  polyline-mask helpers.
- ``src/nav/nav_technique/dt_fitting.py`` — the shared coarse-NCC and LM-refinement helpers
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/nav/nav_orchestrator/image_derivatives.py`` — the per-image gradient / DT derivatives
  attached to :class:`~nav.nav_orchestrator.nav_context.NavContext`.
- ``src/nav/nav_technique/confidence.py`` — the shared sigmoid-combination formula evaluator.
- ``src/nav/nav_technique/diagnostics.py`` —
  :class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics`.

Public class :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav`, base
:class:`~nav.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__`` so the orchestrator's
``NavTechnique._registry`` discovers it.

Class attributes:

- :attr:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.name` — ``'BodyLimbNav'``.
- :attr:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.accepts_feature_types` —
  ``frozenset({LIMB_ARC})``.
- :attr:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.requires_prior` — ``False``.
  The technique runs in pass 1 of the orchestrator's two-pass pipeline.
- :attr:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.confidence_attributes` — the
  names of every attribute the spec is allowed to read, validated at config-load time:
  ``{'at_edge', 'spurious', 'visible_limb_arc_fraction', 'visible_arc_px', 'dt_fit_rms_px',
  'lm_iterations', 'tukey_inlier_count'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.is_feasible` and
:meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate`.

Diagnostics
-----------

:class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics` is the typed dataclass attached to
the result. Every field is named in the call path or in the confidence formula above:

- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.visible_limb_arc_fraction` —
  vertex-weighted average of the per-feature visible-arc fraction; consumed by the confidence
  formula.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.visible_arc_px` — total
  surviving polyline arc length in pixels; consumed by the confidence formula.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.dt_fit_rms_px` — weighted RMS DT
  residual at the converged pose; consumed by the confidence formula and by the
  spurious-detection gate.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.lm_iterations` — number of LM
  iterations actually performed.
- :attr:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics.tukey_inlier_count` — number of
  vertices that retained a strictly positive Tukey weight at the final estimate; consumed by
  the spurious-detection gate.

Call path traced through
:meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate`:

1. Open a logged section. Fail fast if either
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing from
   the context — the orchestrator's per-image setup
   is responsible for populating both via
   :func:`~nav.nav_orchestrator.image_derivatives.compute_all_image_derivatives`; see
   :doc:`dev_guide_techniques_dt_fitting` for the surface those products expose.
2. Filter the offered features down to ``LIMB_ARC`` polylines whose surviving vertex count is at
   least ``min_arc_px``, then concatenate the per-feature vertex / normal / sigma arrays via
   the private ``_aggregate_limb_features`` helper. The geometric outward normals are negated
   in this step so that the polarity test in the LM refiner expects the image gradient to
   point *into* the body silhouette.
3. Build a binary polyline mask and pull the search-window margin off the observation via
   :func:`~nav.nav_technique.nav_technique.search_window_for_obs`. Run
   :func:`~nav.nav_technique.dt_fitting.coarse_ncc_search` on the polyline mask and the
   thresholded edge mask to obtain an integer seed offset.
4. Decide whether to fit camera rotation by reading
   :attr:`~nav.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`. When rotation
   is fit, the rotation pivot is set to the
   centroid of the concatenated vertices and the pivot-to-image-centre distance is computed
   via :func:`~nav.nav_technique.nav_technique.rotation_pivot_distance_px` for the convergence
   test.
5. Call :func:`~nav.nav_technique.dt_fitting.lm_subpixel_refine` with the polyline,
   per-vertex sigmas, the edge DT, the gradient image, the integer seed, and the rotation
   options. The refiner returns a converged
   :class:`~nav.nav_technique.dt_fitting.LMRefineResult`.
6. Compute the result-shape branches:

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
     :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.sigma_rotation_rad` is
     the square root of its diagonal. An unexpected covariance shape raises
     :exc:`RuntimeError` — a programmer error in the refiner contract is not silently
     absorbed.

7. Apply the at-edge tests against both the translation axis bounds and the rotation cap, and
   the spurious tests against the final RMS, the inlier count, and the inlier fraction.
8. Build a :class:`~nav.nav_technique.diagnostics.BodyLimbDiagnostics`, evaluate the
   confidence spec via :func:`~nav.nav_technique.confidence.evaluate_sigmoid_combination`, log
   the per-term breakdown via
   :func:`~nav.nav_technique.nav_technique.log_confidence_breakdown`, and assemble the
   :class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

The :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.feature_ids` field on the
result preserves every consumed
:attr:`~nav.feature.feature.NavFeature.feature_id` so the orchestrator's curator can attribute
each contribution at audit time.

Examples
========

``body_partial_overflow`` (Cassini ISS NAC, image ``N1484593951_2``)
    Rhea visible in the upper right, about 22 % of the disc off-frame. The body model emits a
    ``LIMB_ARC`` feature; the technique consumes it and converges to
    :math:`(\Delta v, \Delta u) = (12.06, 30.53)` px against an operator-verified ground truth
    of ``(11.0, 29.5)`` px. Feasibility passes (one ``LIMB_ARC``, surviving vertex count well
    above ``min_arc_px``), neither :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.at_edge` nor :attr:`~nav.nav_technique.technique_result.NavTechniqueResult.spurious` fires, and the technique
    becomes the orchestrator's primary on this image.

``multi_body`` (Cassini ISS NAC, image ``N1487595731_1``)
    Dione and Rhea both visible and overlapping at phase angle approximately 90 degrees. Two
    ``LIMB_ARC`` features are offered;
    :class:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav` fuses them into a joint
    translation and
    converges to :math:`(\Delta v, \Delta u) = (7.00, -18.00)` px against an operator-verified
    ground truth of ``(7.03, -18.42)`` px. The technique reports a confidence near 0.24 — the
    sigmoid is drawn down by the modest visible-arc length of each limb on its own — but the
    fit is geometrically correct.

``body_full_fov`` (Cassini ISS NAC, image ``N1572105349_1``)
    Dione fills the FOV, predicted disc diameter approximately 155 px. The body model emits a
    ``LIMB_ARC`` feature, but the upstream feature-reliability gate drops it before
    :meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.is_feasible` is consulted
    (the textbook full-disc, fully-lit limb saturates
    the model-side reliability formula's incidence-factor penalty). The technique therefore
    reports zero consumed features and skips
    :meth:`~nav.nav_technique.nav_technique_body_limb.BodyLimbNav.navigate` for this scene.
