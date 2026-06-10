====================
Ring Edge Navigation
====================

Overview
========

This technique exploits the bright, sharp edges of a planetary ring system: the
inner and outer boundaries of ring gaps, ringlets, and the main-ring edges that
appear in the image as thin high-gradient curves.  Each such edge is predicted
from SPICE as a polyline of ``(v, u)`` vertices, and the technique recovers the
single translation (and, when configured, rotation) that best aligns the
predicted polylines onto the image's detected edges by minimising a joint
distance-transform cost.  Feasibility passes when at least one predicted ring
edge contributes a non-empty polyline; it fails only when no usable ring edge
exists in the feature set.  A scene whose every predicted edge is a straight
line is still feasible: the fit is rank-deficient along the edge direction but
the across-edge constraint is genuine, and the ensemble fuses the rank-1 result
with an orthogonal-axis measurement from another feature.

Theory
======

A ring edge projects into the image as a curve whose local normal points across
the edge in the radial direction.  Where the image gradient is strong and the
predicted edge is correctly registered, the predicted vertices sit on top of the
image's bright edge pixels.  Misregistration by a translation
:math:`(\Delta v, \Delta u)` displaces every predicted vertex off the edge by the
projection of that translation onto the local normal.

The image is preprocessed into an edge distance-transform field
:math:`D(v, u)`, the Euclidean distance from each pixel to the nearest detected
edge pixel.  A perfectly registered polyline lies in the zero-valued trough of
:math:`D`; a misregistered one samples positive distances.  For a candidate
offset, each predicted vertex :math:`p_i` is shifted and the cost is the robust
sum of its sampled distances:

.. math::

   C(\Delta v, \Delta u) = \sum_i \rho\!\left(
       \frac{D(p_i + (\Delta v, \Delta u))}{\sigma_i} \right),

where :math:`\sigma_i` is the per-vertex radial position uncertainty and
:math:`\rho` is the Tukey biweight loss that rejects outlier vertices.

The search proceeds in two stages.  A coarse integer-pixel stage renders the
polyline into a binary mask and scans the normalised cross-correlation of that
mask against the image edge mask over the search window, returning the integer
offset of maximum overlap.  A Levenberg-Marquardt stage then refines to
sub-pixel accuracy: at each iteration the residual is the distance-transform
value at each shifted vertex, the Jacobian is the spatial gradient of the
distance-transform field projected onto the per-vertex normal, and an
iteratively reweighted Tukey biweight scheme down-weights vertices whose
residuals are large.  Convergence is declared when the step length falls below a
tolerance.  The refinement is confined to a trust region around the coarse seed
so the optimiser cannot walk out of the coarse basin onto an unrelated edge.

The reported covariance is the inverse of the Tukey-weighted information matrix
accumulated from the per-vertex normals.  When every predicted edge is a
straight line, all per-vertex normals are parallel: the information matrix is
rank-1, the along-edge direction is unconstrained, and the covariance is
honestly singular in that direction.  This is the fundamental geometric limit of
the technique on a flat edge -- a single straight edge cannot fix position along
its own length.  The covariance captures the radial (across-edge) measurement
scatter and the rank-deficient structure; it does not model SPICE pointing error
or systematic photometric edge-location bias.  When the optimiser locks onto the
wrong ring in a multi-ring scene, the Tukey-weighted residual can collapse to
near zero on the one well-fit edge while the others are grossly misaligned, so
the technique additionally inspects the raw per-edge residual average -- which
has no outlier rejection -- to detect that mode of mis-convergence.

Configuration
=============

Tunables live under ``techniques.RingEdgeNav.tuning`` in
``src/nav/config_files/config_510_techniques.yaml``.

- ``at_edge_tolerance_px`` -- float, default ``1.0`` px.  A converged offset whose
  absolute distance from any search-window axis bound falls within this tolerance
  is flagged at-edge; one pixel matches the bilinear distance-transform half-cell
  width.  Larger values flag more offsets as edge-pinned.
- ``spurious_dt_rms_factor`` -- float, default ``5.0`` (dimensionless).  A final
  Tukey-weighted DT residual exceeding this many radial sigmas marks the result
  spurious; lower tightens the rejection.
- ``spurious_dt_floor_px`` -- float, default ``3.0`` px.  Absolute floor for the
  spurious DT-residual threshold so a tiny per-vertex sigma cannot drive the
  threshold below pixel scale.
- ``spurious_min_inliers`` -- int, default ``6`` (count).  Below this Tukey-inlier
  count the M-estimator covariance is uninformative and the result is flagged
  spurious; raising it demands more surviving vertices.
- ``spurious_per_edge_rms_factor`` -- float, default ``5.0`` (dimensionless).  The
  raw, un-Tukey-weighted per-edge average DT RMS exceeding this many radial sigmas
  marks the result spurious; this catches a lock onto one ring of several that the
  Tukey-weighted RMS hides.
- ``spurious_max_lm_displacement_px`` -- float, default ``4.0`` px.  If the
  refinement walks more than this from the integer coarse seed the result is
  flagged spurious; a defensive backstop to the trust region.
- ``lm_trust_region_px`` -- float, default ``1.0`` px.  Maximum refinement
  displacement from the coarse seed; trial steps outside this radius are rejected,
  keeping the optimiser in the coarse basin.
- ``lm_tikhonov_alpha`` -- float, default ``0.0`` (dimensionless).  Tikhonov anchor
  strength toward the coarse seed; the default leaves the trust region as the sole
  bound so legitimate sub-pixel refinement is not damped.
- ``rotation_at_edge_fraction`` -- float, default ``0.95`` (dimensionless).  When
  rotation is fit, a converged rotation magnitude past this fraction of the
  per-image rotation cap trips at-edge; lower surfaces rotation pinning earlier.

Confidence formula
-------------------

The confidence coefficients live alongside ``tuning`` in the same
``techniques.RingEdgeNav`` stanza.  The sigmoid baseline is ``alpha0 = -1.0`` and a
hard-zero gate forces confidence to zero when ``at_edge`` is true.  See
:doc:`dev_guide_techniques_confidence` for the sigmoid-of-linear-combination
mathematics.

- ``total_edge_length_px`` -- alpha = 1.0, offset = 0, divisor = 200.0, cap at 1.0.
  Cumulative pixel length of all surviving ring-edge polylines; more edge length
  means a better-determined fit.
- ``per_edge_dt_rms_summed`` -- alpha = -2.0, offset = 0, divisor = 1.0, no cap.
  Sum of per-edge final DT RMS values; larger residuals pull confidence down.

Implementation
==============

Source file: ``src/nav/nav_technique/nav_technique_ring_edge.py``.  The public
class is :py:class:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav`, a
subclass of :py:class:`~nav.nav_technique.nav_technique.NavTechnique`.  Its
``accepts_feature_types`` is the single ``RING_EDGE`` feature type, its
``requires_prior`` is ``False`` (it runs in pass 1), and its
``confidence_attributes`` set names ``at_edge``, ``total_edge_length_px``,
``per_edge_dt_rms_summed``, ``edge_count``, and ``is_rank_1``.

:py:meth:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav.is_feasible`
reads only the per-feature polyline vertex count and returns feasible when at
least one ``RING_EDGE`` carries a non-empty polyline.

:py:meth:`~nav.nav_technique.nav_technique_ring_edge.RingEdgeNav.navigate`
requires the per-image distance-transform and gradient-vector fields the
orchestrator places on the context (see
:doc:`dev_guide_techniques_image_derivatives`); it raises :py:exc:`RuntimeError`
when they are absent.  It then:

1. Aggregates every ``RING_EDGE`` polyline into one concatenated vertex array
   together with per-vertex radial sigmas and negated outward normals, and tracks
   whether every consumed edge is a straight line.
2. Builds a binary edge mask from the image distance transform and a binary mask
   from the rendered polyline vertices, sizes the search window from the
   observation's extended-FOV margin via ``search_window_for_obs``, and runs the
   coarse integer cross-correlation via ``coarse_ncc_search`` (see
   :doc:`dev_guide_techniques_dt_fitting`).
3. Refines to sub-pixel accuracy with ``lm_subpixel_refine`` from the shared DT
   fitting machinery, passing the trust region and Tikhonov anchor; when the
   observation requests a camera rotation the refinement co-fits rotation about
   the polyline centroid and ``rotation_pivot_distance_px`` sizes the rotation
   lever arm.
4. Computes the at-edge flag against the search-window bounds (including the
   rotation-at-edge test when rotation is fit), and the spurious flag from the
   degenerate-fit indicator, the Tukey-weighted RMS threshold, the inlier count,
   the raw per-edge RMS average, and the coarse-to-refined displacement.
5. Evaluates the YAML confidence formula via ``evaluate_sigmoid_combination``
   wrapped by a small per-technique context adapter and logs the per-term
   breakdown through ``log_confidence_breakdown`` (see
   :doc:`dev_guide_techniques_confidence`).

The result shape branches on whether rotation is fit.  Without rotation the
``covariance_px2`` is the ``(2, 2)`` translation block and both ``rotation_rad``
and ``sigma_rotation_rad`` are ``None``.  With rotation the covariance is
``(3, 3)``, ``rotation_rad`` is the converged angle, and ``sigma_rotation_rad`` is
the square root of its diagonal.  Orthogonally, the translation block is flagged
rank-1 (via the module-private ``_is_rank_1`` test) whenever every consumed edge
is straight or the eigenvalue ratio of the translation block falls below the
rank-deficiency threshold; this rank-1 flag is reported in
:py:class:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics` and consumed by the
ensemble.

Every field of :py:class:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics` is
populated:
:py:attr:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics.total_edge_length_px`
and
:py:attr:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics.per_edge_dt_rms_summed`
feed the confidence formula above,
:py:attr:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics.edge_count` records
how many edges were fused, and
:py:attr:`~nav.nav_technique.diagnostics.RingEdgeDiagnostics.is_rank_1` records the
rank-deficiency state.  The return value is a
:py:class:`~nav.nav_technique.technique_result.NavTechniqueResult`.

Examples
========

**ring_only_curved (N1492091163_1_CALIB).** A single high-curvature Saturn ring
edge at the right edge of the field, with no body and no other ring features.
The curvature makes the per-vertex normals non-parallel, so the fit is full-rank:
feasibility passes and the DT refinement converges to a roughly ``(5, -25)`` px
offset close to the operator ground truth of ``(4.92, -24.32)`` px.  The dominant
confidence driver is ``per_edge_dt_rms_summed``: with the current divisor of
``1.0`` against a long summed-residual value, the sigmoid argument collapses and
the calibrated confidence is zero, so the sidecar pins ``status: failed`` and the
primary technique ``RingEdgeNav`` even though the recovered geometry is correct.
This scene illustrates that a correct DT fit and a high confidence are
independent properties of the result.

**ring_only_flat (rank-1 fallback).** The companion flat-edge class -- an edge-on
or long-range Saturn ansa frame -- has every predicted edge within half a pixel of
a straight line, so ``is_rank_1`` is true and the covariance is singular along the
edge.  Feasibility still passes because the across-edge constraint is real, and
the ensemble fuses the rank-1 result with an orthogonal-axis measurement from
another feature rather than treating the flat edge as a failure.
