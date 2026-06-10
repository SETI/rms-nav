==========
DT Fitting
==========

Overview
========

The distance-transform (DT) fitting machinery is the shared alignment kernel used by every
polyline-based navigation technique: body-limb, body-terminator, ring-edge, and ring-annulus.
Each of those techniques renders a model curve (a limb arc, a terminator arc, or a ring edge) as
an ordered list of image-plane vertices and asks "what rigid shift, and optionally what small
rotation, makes this curve land on top of the matching image edge?" The answer is found in two
stages: a coarse integer-pixel cross-correlation that lands the curve in the right basin, followed
by a sub-pixel Levenberg-Marquardt (LM) refinement against a precomputed image distance transform
with robust outlier rejection. The module is a set of pure numpy functions; the per-technique
classes assemble vertices, outward normals, and per-vertex prior sigmas and call into them.

Theory
======

A model curve is sampled as :math:`N` ordered vertices :math:`x_i = (v_i, u_i)` in image-plane
pixel coordinates, each carrying an outward unit normal :math:`n_i` (the direction across the edge,
from dark to bright) and a prior standard deviation :math:`\sigma_i` describing how confident the
geometry is in that vertex's normal-direction position.

Coarse integer search
----------------------

The first stage finds an integer offset that maximises the overlap of two binary masks: the model
curve rasterised onto the image grid, and a binarised image edge map. For every integer shift
:math:`(\Delta v, \Delta u)` inside a bounded search window, the score is the cross-correlation

.. math::
    f(\Delta v, \Delta u) = \sum_{v, u} m(v, u)\, e(v + \Delta v, u + \Delta u),

where :math:`m` is the curve mask and :math:`e` is the edge mask. Because both masks are binary,
the score is simply the count of curve pixels that, after shifting, fall on an edge pixel. The
per-shift normaliser of a true normalized cross-correlation is constant in the curve mask and
varies only mildly with the edge mask over a small window, so the argmax of the raw correlation
coincides with the NCC peak. Ties are broken in favour of the shift nearest the origin in
Manhattan distance, so a perfectly flat score surface returns the zero shift.

Sub-pixel refinement
--------------------

The second stage minimises a robust, weighted sum of squared distances. Let :math:`D` be the image
distance transform, whose value at any pixel is the Euclidean distance to the nearest edge pixel
(bilinearly interpolated at sub-pixel positions and truncated at a maximum so empty regions
contribute a bounded cost). The parameter vector is the translation :math:`(\Delta v, \Delta u)`,
optionally augmented by an in-plane rotation :math:`\theta` about a pivot :math:`x_p`. The cost is

.. math::
    C(p) = \sum_i w_i \left[ D\!\left( R(\theta)\,(x_i - x_p) + x_p + (\Delta v, \Delta u) \right) \right]^2,

where :math:`R(\theta)` is the rotation matrix and the per-vertex weight is the product of the
prior precision and a robustness weight:

.. math::
    w_i = \frac{1}{\sigma_i^2} \cdot \psi\!\left( \frac{r_i}{\sigma_i} \right).

Here :math:`r_i` is the DT value sampled at vertex :math:`i` (its residual distance to the nearest
edge) and :math:`\psi` is the Tukey biweight (Holland-Welsch) redescending weight

.. math::
    \psi(s) = \begin{cases}
        \left(1 - (s / c)^2\right)^2 & |s| \le c \\
        0 & |s| > c.
    \end{cases}

The constant :math:`c = 4.685` gives 95% asymptotic efficiency on Gaussian residuals when the
residuals are pre-scaled by the residual scale. Vertices whose scaled residual exceeds :math:`c`
receive zero weight and drop out of the fit entirely, which is what makes the estimator robust to a
minority of vertices that sit on the wrong edge.

The minimisation is iteratively-reweighted least squares wrapped in Levenberg-Marquardt damping.
At each iteration the residuals and a finite-difference Jacobian of the residual vector with
respect to the parameters are evaluated; the damped normal equations
:math:`(H + \lambda\,\mathrm{diag}(H))\,\delta = -g` are solved for a trial step; the cost is
re-evaluated at the trial point using the same weights computed at the start of the iteration
(freezing the weights inside a single Gauss-Newton step is the standard IRLS / LM separation and
prevents the step from chasing a different inlier set). A step that lowers the cost is accepted and
the damping is halved; a step that raises it is rejected and the damping is doubled. The Tukey
weights are recomputed between accepted iterations. Convergence is declared when the combined step
norm, with the rotation step converted to a pixel displacement at the pivot's typical distance,
falls below a small tolerance, or when an iteration cap is reached.

Two optional regularisers contain a failure mode of the joint LM + IRLS loop, in which reweighting
can drag the curve off the coarse seed onto an unrelated edge (a crater rim, a terminator, a
surface boundary). A hard trust region rejects any trial offset that walks farther than a radius
from the integer seed. A soft Tikhonov anchor adds a data-scaled penalty pulling the translation
back toward the seed inside that bound. Rotation is never penalised.

A polarity filter screens vertices before the fit: the image gradient vector sampled at each vertex
is dotted with the model outward normal, and a vertex whose dot product is not strictly positive
(the image edge runs the wrong way) is forced to an effectively infinite residual so the biweight
zeroes it on the first reweighting step.

After convergence the parameter covariance is the Moore-Penrose pseudoinverse of the M-estimator
information matrix :math:`J^\top \mathrm{diag}(w)\, J`, evaluated at the converged pose. The
pseudoinverse handles rank-deficient inputs gracefully: a straight (locally flat) curve constrains
only the across-curve direction, producing a rank-1 information matrix whose covariance has
unbounded variance along the unobservable along-curve direction. The reported covariance is
data-only and deliberately excludes any Tikhonov anchor contribution, because that anchor is a
fitting aid rather than measured information and must not shrink the reported uncertainty. The
covariance therefore captures the geometric leverage of the surviving inliers but not systematic
model error or the risk that the coarse stage seeded the wrong basin.

Restrictions and assumptions
----------------------------

The technique requires an image edge that actually corresponds to the model curve; on a textured
or low-contrast scene the coarse stage may seed the wrong basin and the trust region only limits,
rather than eliminates, the resulting error. A curve that is locally straight is rank-deficient in
the along-curve direction and yields a one-dimensional constraint. A fit in which every vertex is
rejected by the polarity filter or the biweight is degenerate: it reports an infinite residual RMS
and an all-infinite covariance so downstream gates treat it as spurious rather than as a perfect
fit.

Configuration
=============

This module is shared algorithmic infrastructure rather than a directly-configurable component, so
its public surface exposes only the default constants that the calling techniques pass through;
there is no dedicated YAML stanza, and per-technique overrides arrive as ordinary function
arguments. The module-level defaults are:

- ``DEFAULT_TUKEY_C`` — float, default ``4.685`` (dimensionless). Holland-Welsch redescender
  cutoff; vertices whose sigma-scaled residual exceeds this value receive zero weight, so lowering
  it rejects more vertices.
- ``DEFAULT_LM_DAMPING`` — float, default ``0.001`` (dimensionless). Initial Levenberg-Marquardt
  damping; larger values start closer to gradient descent and trust the quadratic model less.
- ``DEFAULT_LM_MAX_ITERATIONS`` — int, default ``30`` (count). Iteration cap before the refinement
  bails out; raising it allows more iterations on pathological inputs.
- ``DEFAULT_LM_STEP_TOLERANCE`` — float, default ``0.001`` px. Combined step-norm threshold below
  which the iteration stops; smaller values demand tighter convergence.
- ``DEFAULT_PINVH_RCOND`` — float, default ``1e-09`` (dimensionless). Relative eigenvalue cutoff
  for the Hermitian pseudoinverse used to form the covariance; eigenvalues below it are treated as
  null, so larger values discard more near-singular directions.

Implementation
==============

Source file: ``src/nav/nav_technique/dt_fitting.py``. It depends on
:py:func:`scipy.linalg.pinvh` for the pseudoinverse and on
:py:func:`nav.support.distance_transform.sample_dt_bilinear` for sub-pixel DT sampling. The module
exposes no class hierarchy; its public surface is a small set of pure functions plus one result
dataclass.

The coarse stage is :py:func:`~nav.nav_technique.dt_fitting.coarse_ncc_search`, which takes the
binary edge mask, the binary polyline mask, and a ``(margin_v, margin_u)`` search window and
returns the integer ``(dv, du)`` at the correlation peak. It fetches the polyline support indices
once, then scans every integer shift in the window, summing the edge mask over the shifted (and
in-bounds) polyline pixels and tracking the nearest-to-origin tie-break key.

:py:func:`~nav.nav_technique.dt_fitting.polarity_filter` returns the per-vertex boolean acceptance
mask by sampling the gradient-vector image (produced by
:py:func:`nav.nav_orchestrator.image_derivatives.compute_image_gradient_vu`, documented on
:doc:`dev_guide_techniques_image_derivatives`) at each shifted vertex and testing whether the dot
product with the model normal is strictly positive.

:py:func:`~nav.nav_technique.dt_fitting.tukey_biweight_weights` returns the redescending weights for
a vector of already-scaled residuals, and
:py:func:`~nav.nav_technique.dt_fitting.information_matrix_to_covariance` forms
:math:`J^\top \mathrm{diag}(w)\, J` and pseudoinverts it via ``pinvh``, symmetrising the result.

The refinement entry point is :py:func:`~nav.nav_technique.dt_fitting.lm_subpixel_refine`. It
validates the vertices, normals, and prior sigmas; runs the polarity filter once at the initial
offset when polarity gating is enabled; defaults the rotation pivot to the vertex centroid; then
iterates the damped IRLS loop. Each iteration calls the private helpers
``_compute_residuals_and_jacobian`` (which rotates and shifts the vertices, samples the DT, and
central-differences the parameter Jacobian), ``_weighted_normal_equations``, ``_weighted_cost``,
and ``_step_norm_px``; rotation is handled by ``_rotate_vertices`` and translation by
``_shift_vertices``. When ``fit_rotation`` is True the parameter vector is ``(dv, du, dtheta)`` and
``pivot_distance_px`` is required for the convergence test; otherwise it is ``(dv, du)``. The
optional ``trust_region_px`` rejects out-of-bound trial offsets without committing them, and
``tikhonov_alpha`` adds the soft seed anchor to the translation block only. After the loop the
function computes the weighted ``rms_px``, the unweighted ``raw_rms_px``, the inlier count, and the
covariance (an all-infinite sentinel when the fit is degenerate), and returns a
:py:class:`~nav.nav_technique.dt_fitting.LMRefineResult`.

:py:class:`~nav.nav_technique.dt_fitting.LMRefineResult` is a frozen dataclass whose
``__post_init__`` freezes its numpy arrays write-protected. Its fields are ``offset_vu``,
``rotation_rad``, ``covariance``, ``residuals_px``, ``weights``, ``rms_px``, ``raw_rms_px``,
``iterations``, ``converged``, ``inlier_count``, and ``degenerate``. The weighted ``rms_px`` is
``+inf`` on a fully-rejected fit; the separate ``raw_rms_px`` retains every outlier so a
mis-converged fit whose bad arc was down-weighted to near-zero still surfaces a large value, which
the techniques' spurious gates read.

Examples
========

Coarse-search cost on a worked scene. Consider a body limb arc on a Cassini NAC frame, whose
extended-FOV grid is roughly :math:`1024^2`. Suppose the limb model rasterises to about
:math:`N = 200` vertex pixels, and the technique uses a square search window with
``margin_v = margin_u = 30``. The integer-NCC stage scans every shift in
:math:`[-30, +30] \times [-30, +30]`, that is :math:`61 \times 61 = 3721` candidate offsets, and at
each one it touches only the 200 sparse polyline pixels rather than the full image, so the dominant
work is on the order of :math:`3721 \times 200 \approx 7.4 \times 10^5` edge-mask lookups — far
cheaper than an FFT cross-correlation over the dense :math:`1024^2` grid for windows of this size.

Refinement on the ``body_partial_overflow`` scene (Cassini NAC ``N1484593951_2_CALIB``, a large
Rhea partially off the upper-right edge with a good limb). Here the DT machinery seeds on the
catalog limb, and :py:func:`~nav.nav_technique.dt_fitting.lm_subpixel_refine` converges the body-
limb technique to an offset of about ``(12.06, 30.53)`` px, within the operator ground truth's
``(11.0, 29.5)`` px and its 1.0 px uncertainty. The same image's terminator arc is a degenerate
fit: every one of the 895 terminator vertices is rejected, the LM does not iterate, the result is
flagged ``degenerate`` with ``inlier_count = 0`` and ``rms_px = +inf``, and the downstream
spurious gate fires — exactly the all-rejected behaviour the infinite-RMS sentinel is designed to
surface.
