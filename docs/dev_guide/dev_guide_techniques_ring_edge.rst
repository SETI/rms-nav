==========================================================
Ring Edge Fit (RingEdgeNav)
==========================================================

Overview
========

:class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` recovers a single translation
from one or more ring-edge polylines by aligning each polyline against the image's
edge-distance-transform. The technique consumes every offered
:data:`~spindoctor.feature.feature_type.NavFeatureType.RING_EDGE` feature, weights its vertices by
the prior precision derived from the per-edge sigma, and runs the same coarse-NCC plus
Tukey-reweighted Levenberg-Marquardt refinement that
:doc:`dev_guide_techniques_dt_fitting` describes.

When every consumed ring edge is flagged straight-line the combined Jacobian is rank-
deficient — all parallel ring edges share a single ring-plane normal so the along-edge axis
is unobservable. The technique projects the returned covariance to be *exactly* rank-1
(see below); the orchestrator's ensemble combine fuses it with any orthogonal-axis result
(a star, body limb, body blob) before declaring a final answer, and when no orthogonal
information exists the fused result carries
:attr:`~spindoctor.support.status_reason.NavStatusReason.RANK_1_ONLY` with the offset
reported as the minimum-norm representative along the edge and the unobservable axis
surfaced through
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`.

Feasibility passes when at least one offered ``RING_EDGE`` has a non-empty polyline. A single
non-empty edge is sufficient — even an all-flat scene produces a useful rank-1 constraint.
Feasibility fails when every offered ``RING_EDGE`` is empty (a ring system entirely outside the
extended FOV or below the per-pixel resolution threshold).

Theory
======

The technique fits a per-image translation by minimising the weighted squared distance from
the model ring-edge polylines to the image edges, exactly as the limb fit does — see
:doc:`dev_guide_techniques_dt_fitting` for the cost function, the LM mechanics, the polarity
filter, and the Tukey biweight.

Rank-deficient covariance
-------------------------

Ring edges differ from limbs in that each ring edge is only locally observable along its
*radial* direction (orthogonal to the edge tangent). Motion along the tangent of a single
ring edge produces no DT cost change *in the interior*; the finite polyline's end vertices
do respond to tangential motion (they track wherever the detected edge enters and leaves
the frame), so the raw LM covariance of an all-straight scene often comes back numerically
tight along the tangent — a false constraint. The technique therefore enforces the rank-1
contract explicitly: when every consumed edge is straight (or the raw covariance already
fails the relative-eigenvalue rank test), the translation covariance is rebuilt as the
exactly singular ``sigma_n^2 n n^T``, where ``n`` is the dominant eigenvector of the
per-vertex normals' outer-product sum (polarity-sign-independent) and ``sigma_n^2`` the raw
covariance's marginal variance along it. Exact singularity is the representation the
ensemble is built around: ``pinvh`` keeps the normal-axis measurement when forming the
information matrix, the combine's rank-deficiency test fires, and the unobservable axis is
reported through the
:attr:`~spindoctor.nav_orchestrator.nav_result.NavResult.sigma_along_unobservable_px`
sentinel rather than an inflated per-axis sigma.

Multi-edge inputs at different orbital radii share the same ring-plane normal but sample
different points around the projected ring; the joint information matrix becomes full-rank
when at least two non-parallel edges contribute. The technique reports an
:attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag that the curator
surfaces; downstream consumers (the ensemble combine, the operator's per-image log) can
distinguish a rank-1 ring-edge result from a full-rank one.

Restrictions and assumptions
----------------------------

- The orchestrator must populate
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` and
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext`; in their
  absence the technique cannot evaluate the cost.
- Per-vertex normals supplied by the upstream rings model are oriented so that the image
  gradient should point from outside the ring system inward at a bright ring edge against a
  dark background. Polarity-rejected vertices contribute a near-infinite synthetic
  residual so the Tukey biweight zeroes their weight on the first reweighting.
- Ring edges that have collapsed to a sub-pixel radial extent are emitted by the upstream
  model as ``RING_ANNULUS`` templates instead, so the technique never sees them. See
  :doc:`dev_guide_techniques_ring_annulus`.
- The fit assumes the per-image SPICE pose is good enough that the integer NCC seed lands
  in the basin of attraction of the correct ring edge. When the SPICE pointing error
  exceeds the per-instrument search-window margin the seed lands on a wrong edge and the
  LM converges to a wrong local minimum.

Sources of uncertainty
----------------------

The reported covariance is the Moore-Penrose pseudoinverse of the M-estimator information
matrix at convergence; the rank-1 case is projected to an exactly singular covariance whose
only non-null axis is the aggregate edge normal (see above). When the converged offset sits
within the at-edge tolerance of any axis bound,
or when the rotation parameter is at the configured fraction of its cap, the result is
flagged :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

Configuration
=============

All numeric tunables for this technique live in ``techniques.RingEdgeNav.tuning`` in
``src/spindoctor/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` — float, default ``1.0`` px. A converged offset whose absolute
  distance from any search-window axis bound falls within this tolerance is flagged
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`. Matches the
  bilinear-DT half-cell width.
- ``spurious_dt_rms_factor`` — float, default ``5.0`` (dimensionless). Final DT residual
  exceeding this many radial-sigmas marks the result spurious.
- ``spurious_dt_floor_px`` — float, default ``3.0`` px. Floor of the spurious-detection
  threshold; the threshold is the larger of the floor and the per-feature sigma multiple.
- ``spurious_min_inliers`` — int, default ``6`` (count). Below this Tukey-inlier count the
  M-estimator covariance is uninformative; the result is flagged spurious.
- ``spurious_min_inlier_fraction`` — float, default ``0.5`` (dimensionless). A Tukey inlier
  fraction (inliers over aggregated model vertices) below this marks the result spurious.
  This is the wrong-ringlet mis-convergence detector: a fit locked onto the wrong ring
  anchors a minority of the model vertices, while a correct fit whose faintest edge is
  simply undetectable in the image still anchors a large majority.
- **Absent-edge waiver.** In a multi-edge fusion, a low aggregate inlier fraction can be
  fully explained by an edge that is *absent* from the image (a faint edge nothing in the
  frame can match) rather than *misaligned* (a wrong-ring lock). The two are separable by
  the per-edge median DT residual: an absent edge sits far from every detected image edge
  (a large median), while a wrong-lock leaves its rejected vertices lying *on* a detected
  edge they disagree with (a near-zero median). The gate is therefore waived — the fit is
  kept, not flagged spurious — only when all of: at least ``spurious_waiver_min_well_fit_edges``
  edges each independently clear ``spurious_min_inlier_fraction`` and ``spurious_min_inliers``
  on their own vertices (so the surviving edges genuinely constrain the offset); every
  non-well-fit edge has a per-edge median DT residual of at least
  ``spurious_waiver_absent_median_px`` (absent, not misaligned); the translation covariance
  is full-rank; and at least two edges were consumed (a single-edge fit is never waived).
  A waived fit receives a sigma floor added in quadrature so it lands at the ``'low'`` tier
  and cannot outweigh a full-support result.
- ``spurious_waiver_min_well_fit_edges`` — int, default ``1`` (count). Minimum number of
  edges that must each independently clear the inlier-fraction and inlier-count gates for
  the absent-edge waiver to apply.
- ``spurious_waiver_absent_median_px`` — float, default ``5.0`` px. A non-well-fit edge
  whose per-edge median DT residual is at least this large counts as *absent* (waivable)
  rather than *misaligned* (a genuine mis-convergence).
- ``rotation_at_edge_fraction`` — float, default ``0.95`` (dimensionless). When
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` is true, the
  converged rotation magnitude trips
  :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge` once it crosses
  this fraction of the per-image
  :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.max_rotation_deg` cap.

Per-instrument overrides
------------------------

The keys above are global; per-instrument YAML files in
``src/spindoctor/config_files/config_4N0_inst_*.yaml`` do not override any of them.

Confidence formula
------------------

The technique reports a calibrated confidence in :math:`[0, 1]` produced by the shared
sigmoid combination; see :doc:`dev_guide_techniques_confidence`. The formula spec is
``techniques.RingEdgeNav`` and consumes attributes off
:class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics` plus
:attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`.

- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px` —
  alpha = 1.0, offset = 0.0, divisor = 200.0, cap at 1.0. Cumulative pixel length of all
  surviving ring-edge polylines. More polyline earns confidence up to a 200-pixel
  saturation point.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_summed` —
  alpha = -2.0, offset = 0.0, divisor = 1.0, no cap. Sum of per-edge final DT RMS values.
  Larger residuals pull confidence down.

Hard-zero gate: :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.at_edge`
firing forces confidence to zero. The constant baseline is :math:`\alpha_{0} = -1.0`. No
post-sigmoid ``hard_cap`` is applied.

Implementation
==============

Source files:

- ``src/spindoctor/nav_technique/nav_technique_ring_edge.py`` —
  :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`.
- ``src/spindoctor/nav_technique/dt_fitting.py`` — shared coarse-NCC and LM-refinement helpers;
  documented at :doc:`dev_guide_techniques_dt_fitting`.
- ``src/spindoctor/nav_orchestrator/image_derivatives.py`` — image-side derivatives;
  documented at :doc:`dev_guide_techniques_image_derivatives`.
- ``src/spindoctor/nav_technique/confidence.py`` — sigmoid-combination evaluator; documented at
  :doc:`dev_guide_techniques_confidence`.
- ``src/spindoctor/nav_technique/diagnostics.py`` —
  :class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`; documented at
  :doc:`dev_guide_techniques_diagnostics`.

Public class :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav`, base
:class:`~spindoctor.nav_technique.nav_technique.NavTechnique`. Self-registers via
``__init_subclass__``.

Class attributes:

- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.name` — ``'RingEdgeNav'``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.accepts_feature_types` —
  ``frozenset({RING_EDGE})``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.requires_prior` — ``False``.
- :attr:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.confidence_attributes` —
  ``{'at_edge', 'total_edge_length_px', 'per_edge_dt_rms_summed', 'edge_count',
  'is_rank_1'}``.

Public methods (autodocumented at :doc:`/api_reference/api_nav_technique`):
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.is_feasible` and
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.navigate`.

Diagnostics
-----------

:class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`:

- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px` —
  cumulative pixel length of all surviving ring-edge polylines. Consumed by the confidence
  formula.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_summed` — sum
  of per-edge final DT RMS values. Consumed by the confidence formula.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.edge_count` — number of
  ``RING_EDGE`` features fused.
- :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` — True when every
  consumed ring-edge feature was straight-line and the combined covariance is rank-1.

Call path
---------

Call path traced through
:meth:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav.navigate`:

1. Open a logged section. Fail fast (:exc:`RuntimeError`) if either
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_edge_dt_ext` or
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.image_gradient_vu_ext` is missing.
2. Filter the offered features to ``RING_EDGE`` entries with non-empty polylines and
   concatenate the per-feature vertex / normal / sigma arrays.
3. Build the binary polyline mask and pull the search-window margin off the observation via
   :func:`~spindoctor.nav_technique.nav_technique.search_window_for_obs`. Run
   :func:`~spindoctor.nav_technique.dt_fitting.coarse_ncc_search` for the integer seed.
4. Read :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation` and set the
   rotation pivot to the centroid of the concatenated vertices when rotation is fit.
5. Call :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` and capture the converged
   :class:`~spindoctor.nav_technique.dt_fitting.LMRefineResult`.
6. Detect the rank-1 condition: every consumed ``RING_EDGE`` flagged
   ``is_straight_line`` collapses the joint Jacobian to rank-1. The flag is recorded in
   :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` so downstream
   consumers can branch on it.
7. Result-shape branches on
   :attr:`~spindoctor.nav_orchestrator.nav_context.NavContext.fit_camera_rotation`:

   - **No rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (2, 2) translation block returned by
     :func:`~spindoctor.nav_technique.dt_fitting.lm_subpixel_refine` (rank-1 when every edge is
     straight; full-rank otherwise).
   - **Rotation fit.**
     :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.covariance_px2` is the
     (3, 3) translation + rotation information matrix.

8. Apply the at-edge tests and the spurious tests on RMS / inlier-count.
9. Build a :class:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics`, evaluate the
   confidence spec, log the breakdown, and assemble the
   :class:`~spindoctor.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

``ring_only_curved`` (Cassini ISS NAC, image ``N1447064164_1``)
    A curved Saturn-ring scene with no body in FOV. The rings model emits multiple
    ``RING_EDGE`` polylines (the F ring, the A-ring outer edge, etc.) at different orbital
    radii; the curvature gives the joint Jacobian non-degenerate column rank.
    :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` converges to a 2-D
    translation against the operator-verified offset
    :math:`(\Delta v, \Delta u) = (5.85, 3.55)` px. The
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag is False on
    this scene (the curvature lifts the rank-deficiency) and the orchestrator's ensemble
    combine fuses the result without needing an orthogonal-axis cross-check.

``ring_only_curved`` (Cassini ISS NAC, image ``N1492091163_1``)
    A high-curvature single-edge scene with no other ring features in the FOV. The rings
    model emits one ``RING_EDGE`` polyline whose curvature lifts the rank-1 degeneracy.
    :class:`~spindoctor.nav_technique.nav_technique_ring_edge.RingEdgeNav` converges to
    :math:`(\Delta v, \Delta u) \approx (5.00, -25.00)` px against an operator-verified
    ground truth of ``(4.92, -24.32)`` px — sub-pixel agreement on a single curved edge.
    The technique reports a sub-pixel final DT residual; the
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` flag is False
    because the per-edge curvature exceeds
    :data:`~spindoctor.nav_model.nav_model_rings.FLAT_CURVATURE_THRESHOLD_PX`.

``ring_only_flat`` (Saturn ansa scene class, no body in FOV)
    An edge-on or extreme-grazing ring view in which every surviving polyline is flagged
    :attr:`~spindoctor.feature.flags.RingEdgeFlags.is_straight_line`. The joint-Jacobian column
    rank collapses to 1 along the shared radial direction; the technique returns the
    rank-1 pseudo-inverse covariance described in the Theory section, sets
    :attr:`~spindoctor.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` True, and reports
    a calibrated confidence drawn down by the missing along-edge constraint. The
    orchestrator's ensemble combine treats the result as a one-axis observation and
    relies on a cross-feature (a body limb arc, a star prediction) to constrain the
    orthogonal axis. When no cross-feature is available the orchestrator surfaces the
    rank-1 result with
    :attr:`~spindoctor.nav_technique.technique_result.NavTechniqueResult.confidence` capped by
    the rank-1 confidence formula.
