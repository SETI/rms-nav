"""Tunable defaults shared by the distance-transform fitting stages.

Each constant carries the rationale for its value; the fitting modules
import them rather than re-declaring per-stage defaults.
"""

__all__ = [
    'DEFAULT_COARSE_MIN_SUPPORT_FRACTION',
    'DEFAULT_LM_DAMPING',
    'DEFAULT_LM_MAX_ITERATIONS',
    'DEFAULT_LM_STEP_TOLERANCE',
    'DEFAULT_PINVH_RCOND',
    'DEFAULT_RIDGE_HALF_WIDTH_PX',
    'DEFAULT_RIDGE_MAX_ITERATIONS',
    'DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX',
    'DEFAULT_RIDGE_SAMPLE_STEP_PX',
    'DEFAULT_TUKEY_C',
]

DEFAULT_COARSE_MIN_SUPPORT_FRACTION: float = 0.5
"""Minimum in-bounds vertex fraction for a coarse-search candidate shift.

:func:`coarse_ncc_search` scores each candidate shift by the fraction of
its *in-bounds* polyline vertices that land on edge pixels.  That ratio
is only meaningful when enough vertices survive the shift: a shift that
clips nearly the whole polyline off-frame can score a perfect 1.0 from a
handful of vertices that happen to land on edge pixels, beating a dense
well-supported match (e.g. 450 of 500 vertices on edges).  A candidate
whose in-bounds vertex fraction falls below this threshold is therefore
ineligible.

The 0.5 value is safe for the intended geometry: the model polyline is
rendered in-frame, and the search window margins (tens of pixels on a
1024 x 1024 image) are small relative to the polyline extent, so the true
pointing offset never clips half the vertices.  A shift that does remove
half the polyline cannot be the correct correction, while 0.5 still
tolerates bodies and ring arcs that legitimately hang partly off-frame.
The zero shift keeps every vertex in bounds (the mask is built from
in-frame vertices only), so at least one candidate is always eligible.
"""


DEFAULT_TUKEY_C: float = 4.685
"""Holland-Welsch redescender constant.

The 4.685 value gives 95 % asymptotic efficiency on Gaussian residuals
when the residuals are pre-scaled by an estimate of the residual scale.
The biweight has zero weight outside ``[-c, c]`` so vertices whose
scaled residuals exceed the constant are rejected entirely.
"""


DEFAULT_LM_DAMPING: float = 1.0e-3
"""Default Levenberg-Marquardt damping ``lambda``.

Mixes Gauss-Newton and gradient-descent: small values trust the
quadratic model, large values fall back on gradient descent.  The
``1e-3`` start value matches the design's prescription and is updated
multiplicatively after each accepted / rejected step.
"""


DEFAULT_LM_MAX_ITERATIONS: int = 30
"""Maximum number of LM iterations before bailing out.

The convergence criterion (combined step norm below
:data:`DEFAULT_LM_STEP_TOLERANCE`) almost always fires within a dozen
iterations; the cap is a safety net for pathological inputs.
"""


DEFAULT_LM_STEP_TOLERANCE: float = 1.0e-3
"""Termination threshold on the combined step norm (pixels).

The combined step norm is ``sqrt(d_dv**2 + d_du**2 + (d_theta * pivot_dist)**2)``;
when rotation is disabled the rotation term is zero.  Once the norm drops
below this threshold the LM iteration stops.
"""


DEFAULT_PINVH_RCOND: float = 1.0e-9
"""Default cutoff for the Hermitian pseudoinverse used in covariance.

Matches the same value the orchestrator's ensemble combine uses; a
single project-wide cutoff keeps rank-deficiency handling consistent.
"""


DEFAULT_RIDGE_HALF_WIDTH_PX: float = 3.0
"""Half-width (pixels) of the gradient-ridge sub-pixel search window.

After the DT Levenberg-Marquardt converges, the final continuous
gradient-ridge stage samples the gradient magnitude along each vertex's
normal across ``[-half_width, +half_width]`` and locates the sub-pixel
peak.  Three pixels covers the residual a clean DT-LM convergence leaves
(the integer-quantized DT zero-set snaps within ~1 px) with margin to
spare, while staying narrow enough that the sampled profile contains a
single edge ridge rather than two adjacent edges.
"""


DEFAULT_RIDGE_SAMPLE_STEP_PX: float = 0.5
"""Spacing (pixels) of the gradient-ridge sample points along each normal.

The peak is located by a three-point parabola fit around the discrete
argmax, so the spacing trades convergence robustness against
sensitivity: a half-pixel step gives a stable parabola on the
Gaussian-smoothed gradient profile (image_gradient_sigma_px ~ 1.2).  The
*converged* offset is unbiased regardless of the step because at the
Gauss-Newton fixed point the vertex sits on the ridge peak, so the
parabola is evaluated symmetrically about the true maximum where its
discretization bias vanishes.
"""


DEFAULT_RIDGE_MAX_ITERATIONS: int = 10
"""Maximum gradient-ridge Gauss-Newton iterations.

The ridge stage starts from the DT-LM optimum (sub-pixel residual), so
the near-linear Gauss-Newton step converges in a handful of iterations;
the cap is a safety net.
"""


DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX: float = 1.5
"""Cap on how far the gradient-ridge stage may move the DT-LM offset.

The ridge refinement is a *sub-pixel* polish of an already-converged
fit; a converged DT-LM optimum is within ~1 px of the true edge, so the
ridge should never walk more than about a pixel.  If the cumulative
displacement from the DT-LM offset exceeds this bound the ridge result
is discarded and the DT-LM offset is kept -- a defensive guard against a
pathological ridge walk onto an unrelated gradient feature.
"""


_INFINITY_DT_PENALTY_PX: float = 1.0e6
"""Effective ``+inf`` residual recorded for polarity-rejected vertices.

A polarity-rejected vertex is excluded from the fit by zeroing its
*weight* directly (the Tukey weight is multiplied by the polarity mask),
so its exclusion is independent of its per-vertex sigma and never relies
on this magnitude.  The penalty residual is recorded in ``raw_residuals``
only so those arrays stay numerically well-defined; because the
corresponding weight is zero it contributes nothing to the cost or the
normal equations.  The ``raw_rms_px`` diagnostic deliberately excludes
polarity-rejected vertices (it averages over the polarity-accepted
residuals only), so this sentinel does not feed the limb / terminator
spurious gate -- excessive polarity rejection is governed by the
inlier-fraction gate instead.  The value is a large-but-finite number
(not literal ``inf``) so downstream array arithmetic stays defined.
"""
