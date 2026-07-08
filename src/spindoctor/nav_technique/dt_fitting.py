"""Shared distance-transform fitting machinery for polyline-based techniques.

The body-limb, body-terminator, and ring-edge techniques all follow the
same algorithm: render the model polyline as a binary mask, take a coarse
2-D NCC against the image edge mask to get an integer offset, then refine
to sub-pixel precision by Levenberg-Marquardt minimisation against the
image distance transform with Tukey-biweight outlier rejection.  After
convergence the M-estimator information matrix is inverted to produce a
covariance estimate.

Each helper here is a pure function over numpy arrays; the per-technique
classes simply assemble vertices / normals / weights and call into them.
The interface is:

* :func:`coarse_ncc_search` — integer-pixel offset from binary masks.
* :func:`polarity_filter` — per-vertex acceptance from gradient direction.
* :func:`tukey_biweight_weights` — Holland-Welsch redescender weights.
* :func:`lm_subpixel_refine` — translation (or translation + rotation) LM
  refinement with Tukey reweighting against a precomputed DT.
* :func:`information_matrix_to_covariance` — Hessian → covariance via
  ``pinvh`` so rank-deficient inputs are handled.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.support.distance_transform import sample_dt_bilinear
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DEFAULT_LM_DAMPING',
    'DEFAULT_LM_MAX_ITERATIONS',
    'DEFAULT_LM_STEP_TOLERANCE',
    'DEFAULT_PINVH_RCOND',
    'DEFAULT_RIDGE_HALF_WIDTH_PX',
    'DEFAULT_RIDGE_MAX_ITERATIONS',
    'DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX',
    'DEFAULT_RIDGE_SAMPLE_STEP_PX',
    'DEFAULT_TUKEY_C',
    'LMRefineResult',
    'RidgeRefineResult',
    'build_polyline_mask',
    'coarse_ncc_search',
    'gradient_ridge_refine',
    'information_matrix_to_covariance',
    'lm_subpixel_refine',
    'polarity_filter',
    'tukey_biweight_weights',
]


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
on this magnitude.  The penalty residual is still recorded in
``raw_residuals`` so the unweighted ``raw_rms_px`` diagnostic reflects
the wrong-polarity vertices; because the corresponding weight is zero it
contributes nothing to the cost or the normal equations.  The value is a
large-but-finite number (not literal ``inf``) only so those arrays stay
numerically well-defined; it must not be enlarged without care, since
``raw_rms_px`` -- and hence the limb / terminator spurious gate -- reads
it directly.
"""


def coarse_ncc_search(
    edge_mask: NDArrayBoolType,
    polyline_mask: NDArrayBoolType,
    search_window_vu: tuple[int, int],
) -> tuple[int, int]:
    """Return the integer offset that maximises the normalised overlap of two masks.

    For each integer shift ``(dv, du)`` in
    ``[-margin_v, +margin_v] x [-margin_u, +margin_u]`` the score is the
    fraction of in-bounds polyline vertices that land on an edge pixel,
    ``f(dv, du) = (sum_{v, u} polyline_mask[v, u] * edge_mask[v + dv, u + du])
    / N_in_bounds(dv, du)``, where ``N_in_bounds`` is the number of polyline
    vertices that remain inside the image after the shift.  Dividing the raw
    overlap count by the in-bounds vertex count gives the per-vertex match
    fraction, whose argmax coincides with the binary normalised
    cross-correlation peak (the NCC equals the square root of this fraction).
    It removes the bias of a raw overlap count toward shifts that simply keep
    more vertices in bounds or place the polyline over a denser local edge
    region.

    Ties are broken by the smaller ``(|dv| + |du|, |dv|, dv, du)`` tuple,
    so the nearest-to-origin shift (by Manhattan distance) wins on
    perfectly flat inputs.

    Parameters:
        edge_mask: ``(H, W)`` boolean image edge map.
        polyline_mask: ``(H, W)`` boolean model polyline mask aligned to
            ``edge_mask``.  Must have the same shape as ``edge_mask``.
        search_window_vu: ``(margin_v, margin_u)`` non-negative integers
            bounding the search range in v and u.

    Returns:
        ``(dv, du)`` integer offset pair at the peak.

    Raises:
        TypeError: if either entry of ``search_window_vu`` is not an int.
        ValueError: if shapes disagree, masks are not 2-D, or
            ``search_window_vu`` is not a length-2 sequence of
            non-negative ints.
    """
    if edge_mask.ndim != 2 or polyline_mask.ndim != 2:
        raise ValueError(
            'edge_mask and polyline_mask must be 2-D; got '
            f'ndims {edge_mask.ndim}, {polyline_mask.ndim}'
        )
    if edge_mask.shape != polyline_mask.shape:
        raise ValueError(
            f'shape mismatch: edge_mask {edge_mask.shape} vs polyline_mask {polyline_mask.shape}'
        )
    # Validate explicitly rather than relying on int() coercion, which would
    # silently truncate floats and raise an unhelpful IndexError on
    # wrong-length sequences.
    if not isinstance(search_window_vu, tuple | list) or len(search_window_vu) != 2:
        raise ValueError(
            f'search_window_vu must be a length-2 sequence of ints; got {search_window_vu!r}'
        )
    margin_v_raw, margin_u_raw = search_window_vu[0], search_window_vu[1]
    if not isinstance(margin_v_raw, int) or isinstance(margin_v_raw, bool):
        raise TypeError(f'search_window_vu[0] must be int; got {type(margin_v_raw).__name__}')
    if not isinstance(margin_u_raw, int) or isinstance(margin_u_raw, bool):
        raise TypeError(f'search_window_vu[1] must be int; got {type(margin_u_raw).__name__}')
    if margin_v_raw < 0 or margin_u_raw < 0:
        raise ValueError(f'search_window_vu must be non-negative; got {search_window_vu!r}')
    margin_v, margin_u = margin_v_raw, margin_u_raw
    height, width = edge_mask.shape
    edge_f = edge_mask.astype(np.float64, copy=False)
    # Scan the bounded window directly: brute-force over O(margin_v * margin_u)
    # offsets is faster than FFT for the typical (50, 50) margins on a
    # 1024 x 1024 image because the cross-correlation involves only the
    # sparse polyline support.  Pre-fetching the polyline indices avoids
    # repeat numpy re-broadcasts.
    poly_vs, poly_us = np.where(polyline_mask)
    if poly_vs.size == 0:
        return (0, 0)
    best_dv = 0
    best_du = 0
    best_score = -1.0
    best_key = (math.inf, math.inf, math.inf, math.inf)
    for dv in range(-margin_v, margin_v + 1):
        shifted_v = poly_vs + dv
        valid_v = (shifted_v >= 0) & (shifted_v < height)
        if not valid_v.any():
            continue
        for du in range(-margin_u, margin_u + 1):
            shifted_u = poly_us + du
            valid = valid_v & (shifted_u >= 0) & (shifted_u < width)
            if not valid.any():
                continue
            sv = shifted_v[valid]
            su = shifted_u[valid]
            # Score is the fraction of in-bounds polyline points (after the
            # shift) that fall on edge pixels: the raw overlap count divided
            # by the in-bounds vertex count.  Normalising by ``sv.size``
            # (== valid.sum(), guaranteed >= 1 by the ``valid.any()`` check
            # above) makes the argmax the binary NCC argmax, so a shift does
            # not win merely by keeping more vertices in bounds or covering a
            # denser edge region.
            score = float(edge_f[sv, su].sum()) / float(sv.size)
            key = (abs(dv) + abs(du), abs(dv), dv, du)
            if score > best_score or (score == best_score and key < best_key):
                best_score = score
                best_key = key
                best_dv = dv
                best_du = du
    return best_dv, best_du


def build_polyline_mask(
    vertices_vu: NDArrayFloatType, shape_vu: tuple[int, int]
) -> NDArrayBoolType:
    """Render polyline vertices into a boolean image mask aligned to ``shape_vu``.

    Each vertex is rounded to the nearest integer pixel; vertices that fall
    outside ``shape_vu`` are silently dropped.  The integer rasterization is
    deliberate -- the mask feeds :func:`coarse_ncc_search`, which scores
    integer-pixel shifts of the binary mask against the edge DT.

    Shared by the limb / terminator / ring-edge techniques (previously a
    verbatim per-module copy).

    Parameters:
        vertices_vu: ``(N, 2)`` polyline vertex positions in ``(v, u)``.
        shape_vu: ``(H, W)`` target mask shape.

    Returns:
        ``(H, W)`` boolean mask, True at each in-bounds rounded vertex.
    """
    vs = np.rint(vertices_vu[:, 0]).astype(np.int64)
    us = np.rint(vertices_vu[:, 1]).astype(np.int64)
    valid = (vs >= 0) & (vs < shape_vu[0]) & (us >= 0) & (us < shape_vu[1])
    mask = np.zeros(shape_vu, dtype=bool)
    if valid.any():
        mask[vs[valid], us[valid]] = True
    return mask


def polarity_filter(
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    image_gradient_vu: NDArrayFloatType,
    *,
    offset_vu: tuple[float, float] = (0.0, 0.0),
) -> NDArrayBoolType:
    """Return per-vertex polarity acceptance against an image gradient.

    For each vertex the helper samples the image gradient vector at the
    vertex's current shifted position and compares the gradient to the
    model's outward normal.  The polarity test is *strictly* greater than
    zero: orthogonal hits (dot product exactly zero, vanishingly rare in
    floating-point arithmetic) are rejected, never silently kept.

    Out-of-bounds vertices are rejected explicitly: a vertex whose
    rounded shifted position falls outside the image is never accepted,
    regardless of the gradient at the clamped boundary pixel.  (Sampling
    the clamped pixel and "letting it reject on its own merits" is unsafe
    because a strong frame-edge gradient -- common after zero-padding into
    the extended FOV -- can align with an outward normal and spuriously
    accept an off-image vertex.)  The clamp below only keeps the gather
    indices valid; the gathered value is discarded for such vertices.

    Parameters:
        vertices_vu: ``(N, 2)`` model vertex positions.
        normals_vu: ``(N, 2)`` model outward normal at each vertex.
        image_gradient_vu: ``(H, W, 2)`` per-pixel gradient image as
            produced by
            :func:`spindoctor.nav_orchestrator.image_derivatives.compute_image_gradient_vu`.
        offset_vu: ``(dv, du)`` shift applied to each vertex before
            sampling.  Defaults to no shift.

    Returns:
        ``(N,)`` boolean mask True where the vertex's polarity agrees
        with the image's local edge direction.

    Raises:
        ValueError: if shape requirements are violated.
    """
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f'vertices_vu must have shape (N, 2); got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    if image_gradient_vu.ndim != 3 or image_gradient_vu.shape[2] != 2:
        raise ValueError(
            f'image_gradient_vu must have shape (H, W, 2); got {image_gradient_vu.shape}'
        )
    height, width, _ = image_gradient_vu.shape
    dv, du = float(offset_vu[0]), float(offset_vu[1])
    rint_v = np.rint(verts[:, 0] + dv).astype(np.int64)
    rint_u = np.rint(verts[:, 1] + du).astype(np.int64)
    in_bounds = (rint_v >= 0) & (rint_v < height) & (rint_u >= 0) & (rint_u < width)
    sample_v = np.clip(rint_v, 0, height - 1)
    sample_u = np.clip(rint_u, 0, width - 1)
    gv = image_gradient_vu[sample_v, sample_u, 0]
    gu = image_gradient_vu[sample_v, sample_u, 1]
    dot = norms[:, 0] * gv + norms[:, 1] * gu
    return cast(NDArrayBoolType, in_bounds & (dot > 0.0))


def tukey_biweight_weights(
    residuals: NDArrayFloatType,
    *,
    c: float = DEFAULT_TUKEY_C,
) -> NDArrayFloatType:
    """Return Tukey biweight weights for a vector of residuals.

    The Holland-Welsch biweight is

    .. math::
        w_i = \\begin{cases}
            \\bigl(1 - (r_i / c)^2\\bigr)^2 & |r_i| \\le c \\\\
            0 & \\text{otherwise}
        \\end{cases}

    so residuals beyond ``c`` are dropped completely.  Callers are
    responsible for scaling residuals to the desired robust scale before
    invoking — typically dividing by an estimate of the residual standard
    deviation so that ``c = 4.685`` corresponds to the conventional 95 %
    asymptotic efficiency under Gaussian errors.

    Parameters:
        residuals: ``(N,)`` array of (already-scaled) residuals.  May
            contain negative entries; only the magnitude is consulted.
        c: Strictly positive cutoff in residual units.  Must be finite.

    Returns:
        ``(N,)`` float64 array of weights in ``[0, 1]``.

    Raises:
        ValueError: if ``c`` is not strictly positive or
            ``residuals`` is not 1-D.
    """
    if not (c > 0.0) or not math.isfinite(c):
        raise ValueError(f'c must be a positive finite number; got {c!r}')
    arr = np.asarray(residuals, np.float64)
    if arr.ndim != 1:
        raise ValueError(f'residuals must be 1-D; got ndim={arr.ndim}')
    scaled = arr / c
    inside = np.abs(scaled) <= 1.0
    weights = np.zeros_like(arr)
    inside_vals = scaled[inside]
    weights[inside] = (1.0 - inside_vals * inside_vals) ** 2
    return cast(NDArrayFloatType, weights)


def information_matrix_to_covariance(
    jacobian: NDArrayFloatType,
    weights: NDArrayFloatType,
    *,
    rcond: float = DEFAULT_PINVH_RCOND,
) -> NDArrayFloatType:
    """Return the parameter covariance from a weighted Jacobian.

    The M-estimator information matrix is ``J^T diag(w) J``; the
    parameter covariance is its Moore-Penrose pseudoinverse via
    :func:`scipy.linalg.pinvh`, which gracefully handles rank-deficient
    inputs (a flat polyline produces a rank-1 information matrix; the
    returned covariance has unbounded variance along the unobservable
    null direction).

    The ``weights`` argument is the per-residual weight (Tukey biweight
    times any prior precision) and is multiplied into ``J`` before the
    matrix product, which both incorporates the IRLS reweighting and
    keeps the result symmetric within numerical tolerance.

    Parameters:
        jacobian: ``(N, P)`` Jacobian of the residual vector with respect
            to the parameter vector.
        weights: ``(N,)`` non-negative residual weights.
        rcond: Pseudoinverse cutoff; eigenvalues smaller than this
            relative to the largest are treated as null.

    Returns:
        ``(P, P)`` covariance matrix.  Symmetric; positive semidefinite
        within ``rcond``.

    Raises:
        ValueError: if shapes disagree, ``weights`` contains negative
            entries, or ``jacobian`` is not 2-D.
    """
    if jacobian.ndim != 2:
        raise ValueError(f'jacobian must be 2-D; got ndim={jacobian.ndim}')
    if weights.ndim != 1 or weights.shape[0] != jacobian.shape[0]:
        raise ValueError(
            'weights must be a 1-D vector matching the Jacobian rows; '
            f'got weights {weights.shape}, jacobian {jacobian.shape}'
        )
    if (weights < 0).any():
        raise ValueError('weights must be non-negative')
    sqrt_w = np.sqrt(weights)
    weighted_j = sqrt_w[:, None] * jacobian
    info = weighted_j.T @ weighted_j
    cov = pinvh(info, rtol=rcond)
    cov = 0.5 * (cov + cov.T)
    return cast(NDArrayFloatType, cov)


@dataclass(frozen=True)
class LMRefineResult:
    """Structured output of :func:`lm_subpixel_refine`.

    Parameters:
        offset_vu: ``(dv, du)`` converged translation in pixels.
        rotation_rad: Converged rotation in radians; ``0.0`` when
            ``fit_rotation`` was False.
        covariance: ``(2, 2)`` or ``(3, 3)`` parameter covariance derived
            from the M-estimator information matrix ``J^T diag(w) J`` at
            the converged point.  This is the DATA-ONLY covariance: it
            deliberately EXCLUDES the Tikhonov anchor diagonal that
            ``tikhonov_alpha`` adds to the iteration Hessian, because that
            anchor is a fitting aid that biases the step rather than
            measured information and so must not shrink the reported
            uncertainty.
        residuals_px: ``(N,)`` per-vertex DT residuals at the final
            estimate (raw, unweighted).
        weights: ``(N,)`` per-vertex weights at the final estimate
            (prior precision times Tukey biweight).
        rms_px: Weighted root-mean-square of the residuals, computed as
            ``sqrt(sum(w * r**2) / sum(w))`` over surviving vertices;
            ``float('inf')`` when every vertex was rejected (the
            degenerate case), so the downstream spurious gates' ``rms_px
            > floor`` test fires instead of reading a zero RMS as a good
            fit.
        raw_rms_px: Unweighted root-mean-square of the residuals,
            computed as ``sqrt(mean(r**2))`` over ALL vertices with no
            weighting.  This is well-defined even in the degenerate
            (all-weights-zero) case, where the residuals still exist; a
            fully-rejected fit therefore yields a large ``raw_rms_px``.
            Because the Tukey reweighting can down-weight a wholly
            mis-aligned arc to ~0, the weighted ``rms_px`` collapses to
            near zero on exactly such a mis-convergence and slips past a
            ``rms_px > floor`` gate; the raw RMS retains those outliers
            and surfaces the bad fit.
        iterations: Number of LM iterations actually performed.
        converged: True if the step-norm tolerance was met before the
            iteration cap.
        inlier_count: Number of vertices that retained a strictly
            positive Tukey weight at the final estimate.
        degenerate: True when no vertex survived reweighting
            (``inlier_count == 0`` or the surviving weights sum to zero).
            In this case ``rms_px`` is ``+inf`` and ``covariance`` is
            all-``inf``; consumers treat it as a spurious fit.
    """

    offset_vu: tuple[float, float]
    rotation_rad: float
    covariance: NDArrayFloatType
    residuals_px: NDArrayFloatType
    weights: NDArrayFloatType
    rms_px: float
    raw_rms_px: float
    iterations: int
    converged: bool
    inlier_count: int
    degenerate: bool

    def __post_init__(self) -> None:
        """Freeze the numpy arrays and validate shapes."""
        for name in ('covariance', 'residuals_px', 'weights'):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f'LMRefineResult.{name} must be an ndarray')
            arr.setflags(write=False)


@dataclass
class _LMState:
    """Mutable LM iteration state.  Internal to :func:`lm_subpixel_refine`."""

    dv: float
    du: float
    dtheta: float
    polarity_mask: NDArrayBoolType
    iteration: int = 0
    converged: bool = False
    raw_residuals: NDArrayFloatType = field(default_factory=lambda: np.zeros(0, np.float64))
    weights: NDArrayFloatType = field(default_factory=lambda: np.zeros(0, np.float64))
    jacobian: NDArrayFloatType = field(default_factory=lambda: np.zeros((0, 2), np.float64))


def _rotate_vertices(
    vertices_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    theta: float,
) -> NDArrayFloatType:
    """Rotate ``vertices_vu`` about ``pivot_vu`` by ``theta`` radians."""
    if theta == 0.0:
        return vertices_vu
    pv, pu = pivot_vu
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    centred_v = vertices_vu[:, 0] - pv
    centred_u = vertices_vu[:, 1] - pu
    new_v = pv + cos_t * centred_v - sin_t * centred_u
    new_u = pu + sin_t * centred_v + cos_t * centred_u
    return cast(NDArrayFloatType, np.stack([new_v, new_u], axis=-1))


def _rotate_directions(
    directions_vu: NDArrayFloatType,
    theta: float,
) -> NDArrayFloatType:
    """Rotate direction vectors (normals) by ``theta`` radians about the origin.

    Unlike :func:`_rotate_vertices` there is no pivot: a normal is a free
    vector, so only the in-plane rotation applies.  Used by the
    gradient-ridge stage to keep each vertex's outward normal aligned with
    the body after a rotation step.
    """
    if theta == 0.0:
        return directions_vu
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    nv = directions_vu[:, 0]
    nu = directions_vu[:, 1]
    rot_v = cos_t * nv - sin_t * nu
    rot_u = sin_t * nv + cos_t * nu
    return cast(NDArrayFloatType, np.stack([rot_v, rot_u], axis=-1))


def _shift_vertices(
    vertices_vu: NDArrayFloatType,
    dv: float,
    du: float,
) -> NDArrayFloatType:
    """Return a copy of ``vertices_vu`` shifted by ``(dv, du)``."""
    out = vertices_vu.copy()
    out[:, 0] += dv
    out[:, 1] += du
    return out


def _compute_residuals_and_jacobian(
    *,
    vertices_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    image_dt: NDArrayFloatType,
    dv: float,
    du: float,
    dtheta: float,
    fit_rotation: bool,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Sample the DT and finite-difference its parameter Jacobian.

    Bilinear DT samples are differentiable almost everywhere, so a
    central-difference Jacobian with a small step gives a good local
    linearisation.  The step is fixed at 0.25 pixels — large enough to
    cross several DT bilinear cells (the DT grows by 1 every full
    pixel), small enough to remain within the bilinear interpolant's
    local quadratic regime.
    """
    base_pos = _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta), dv, du)
    residuals = sample_dt_bilinear(image_dt, base_pos)
    # Central differences with a sub-pixel step.  A 0.25 px step is large
    # enough to cross several DT bilinear cells (the DT grows by 1 every
    # full pixel) but small enough that the linearisation stays inside
    # the local quadratic regime around the converged offset.
    eps = 0.25
    pdv = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, eps, 0.0))
    mdv = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, -eps, 0.0))
    pdu = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, 0.0, eps))
    mdu = sample_dt_bilinear(image_dt, _shift_vertices(base_pos, 0.0, -eps))
    drdv = (pdv - mdv) / (2.0 * eps)
    drdu = (pdu - mdu) / (2.0 * eps)
    if not fit_rotation:
        jacobian = np.stack([drdv, drdu], axis=-1)
        return residuals, cast(NDArrayFloatType, jacobian)
    # Numerical derivative w.r.t. dtheta
    eps_t = 1.0e-3
    plus = sample_dt_bilinear(
        image_dt,
        _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta + eps_t), dv, du),
    )
    minus = sample_dt_bilinear(
        image_dt,
        _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta - eps_t), dv, du),
    )
    drdth = (plus - minus) / (2.0 * eps_t)
    jacobian = np.stack([drdv, drdu, drdth], axis=-1)
    return residuals, cast(NDArrayFloatType, jacobian)


def _weighted_normal_equations(
    jacobian: NDArrayFloatType,
    residuals: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return ``(J^T W J, J^T W r)`` weighted by ``weights``."""
    sqrt_w = np.sqrt(weights)
    weighted_j = sqrt_w[:, None] * jacobian
    weighted_r = sqrt_w * residuals
    hessian = weighted_j.T @ weighted_j
    rhs = weighted_j.T @ weighted_r
    return cast(NDArrayFloatType, hessian), cast(NDArrayFloatType, rhs)


def _weighted_cost(weights: NDArrayFloatType, residuals: NDArrayFloatType) -> float:
    """Sum of ``w_i * r_i**2`` (the quantity LM minimises)."""
    return float(np.sum(weights * residuals * residuals))


def _step_norm_px(
    step: NDArrayFloatType,
    *,
    fit_rotation: bool,
    pivot_distance_px: float,
) -> float:
    """Return the LM convergence step norm in pixel-equivalent units.

    Translation steps contribute their Euclidean magnitude directly; the
    rotation step (when present) is multiplied by ``pivot_distance_px``
    to convert radians into a pixel displacement at the pivot's typical
    distance.  Combined as a Euclidean norm so the same tolerance
    threshold applies to translation-only and translation-plus-rotation
    fits.
    """
    if not fit_rotation:
        return float(math.hypot(step[0], step[1]))
    rotation_step_px = float(step[2]) * pivot_distance_px
    return float(math.sqrt(step[0] ** 2 + step[1] ** 2 + rotation_step_px**2))


def _ridge_normal_distances(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    pivot_vu: tuple[float, float],
    gradient_magnitude: NDArrayFloatType,
    dv: float,
    du: float,
    dtheta: float,
    t_samples: NDArrayFloatType,
    sample_step_px: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Locate the sub-pixel gradient-magnitude ridge along each vertex normal.

    For each vertex at its current (rotated + shifted) position the
    continuous gradient magnitude is bilinearly sampled at
    ``position + t * normal`` for every ``t`` in ``t_samples``; the signed
    normal distance ``t*`` to the ridge is the parabola-interpolated
    location of the sampled maximum.  ``t*`` is the residual the
    gradient-ridge Gauss-Newton stage drives to zero (the vertex then sits
    on the continuous edge ridge).

    Parameters:
        vertices_vu: ``(N, 2)`` base (unshifted) model vertices.
        normals_vu: ``(N, 2)`` base model normals (unit length).
        pivot_vu: rotation pivot.
        gradient_magnitude: ``(H, W)`` continuous gradient magnitude image.
        dv, du: current translation.
        dtheta: current rotation (radians).
        t_samples: ``(K,)`` monotone normal offsets to sample.
        sample_step_px: spacing of ``t_samples`` (pixels).

    Returns:
        ``(t_star, normals_current, valid)``:

        - ``t_star`` ``(N,)`` signed normal distance to the ridge peak.
        - ``normals_current`` ``(N, 2)`` normals rotated by ``dtheta`` (the
          directions the translation Jacobian rows ``-n`` use).
        - ``valid`` ``(N,)`` boolean: True where the sampled maximum is a
          strict interior peak (a usable sub-pixel residual).  A maximum at
          either sampling boundary means the true ridge lies outside the
          search window, so that vertex carries no usable ridge residual.
    """
    norms_current = _rotate_directions(normals_vu, dtheta)
    pos = _shift_vertices(_rotate_vertices(vertices_vu, pivot_vu, dtheta), dv, du)
    n = pos.shape[0]
    k = t_samples.shape[0]
    # Sample points along each normal: (N, K, 2).
    sample_pts = pos[:, None, :] + t_samples[None, :, None] * norms_current[:, None, :]
    # ``sample_dt_bilinear`` is a clamped bilinear interpolator over any 2-D
    # field, so it samples the continuous gradient magnitude here just as it
    # samples the DT elsewhere.
    mag = sample_dt_bilinear(gradient_magnitude, sample_pts.reshape(-1, 2)).reshape(n, k)
    kmax = np.argmax(mag, axis=1)
    rows = np.arange(n)
    interior = (kmax > 0) & (kmax < k - 1)
    # Clamp the peak index so the three-point gather stays in bounds; the
    # boundary peaks are masked out via ``valid`` regardless.
    kc = np.clip(kmax, 1, k - 2)
    y_minus = mag[rows, kc - 1]
    y_zero = mag[rows, kc]
    y_plus = mag[rows, kc + 1]
    denom = y_minus - 2.0 * y_zero + y_plus
    # A genuine peak is concave-down (denom < 0); a non-concave triple has no
    # interior vertex, so fall back to the sampled location (delta = 0).
    concave = denom < -1.0e-12
    delta = np.where(concave, 0.5 * (y_minus - y_plus) / np.where(concave, denom, -1.0), 0.0)
    # Keep the parabola vertex inside the central cell so a noisy triple cannot
    # throw ``t*`` past the neighbouring samples.
    delta = np.clip(delta, -0.5, 0.5)
    t_star = t_samples[kc] + delta * sample_step_px
    return cast(NDArrayFloatType, t_star), norms_current, cast(NDArrayFloatType, interior)


@dataclass(frozen=True)
class RidgeRefineResult:
    """Structured output of :func:`gradient_ridge_refine`.

    Parameters:
        offset_vu: ``(dv, du)`` refined translation in pixels.
        rotation_rad: refined rotation in radians (unchanged from the input
            when ``fit_rotation`` was False).
        iterations: Gauss-Newton iterations performed.
        converged: True if the step-norm tolerance was met before the cap.
        applied: True if the refined pose was accepted.  False when the
            stage found too few valid ridge residuals to fit, or the
            cumulative displacement from the input offset exceeded the
            displacement cap -- in both cases ``offset_vu`` / ``rotation_rad``
            equal the inputs unchanged and the caller keeps the DT-LM pose.
    """

    offset_vu: tuple[float, float]
    rotation_rad: float
    iterations: int
    converged: bool
    applied: bool


def gradient_ridge_refine(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    sigma_normal_per_vertex_px: NDArrayFloatType,
    gradient_magnitude: NDArrayFloatType,
    initial_offset_vu: tuple[float, float],
    weight_mask: NDArrayFloatType | None = None,
    initial_rotation_rad: float = 0.0,
    fit_rotation: bool = False,
    pivot_vu: tuple[float, float] | None = None,
    pivot_distance_px: float = 0.0,
    max_iterations: int = DEFAULT_RIDGE_MAX_ITERATIONS,
    step_tolerance_px: float = DEFAULT_LM_STEP_TOLERANCE,
    tukey_c: float = DEFAULT_TUKEY_C,
    half_width_px: float = DEFAULT_RIDGE_HALF_WIDTH_PX,
    sample_step_px: float = DEFAULT_RIDGE_SAMPLE_STEP_PX,
    max_total_displacement_px: float = DEFAULT_RIDGE_MAX_TOTAL_DISPLACEMENT_PX,
) -> RidgeRefineResult:
    """Polish a polyline alignment against the continuous gradient-ridge field.

    This is the final, sub-pixel stage after the coarse-NCC + DT
    Levenberg-Marquardt acquisition.  The DT-LM minimises distance to an
    integer-quantized edge mask, whose zero-set snaps the recovered edge to
    integer pixels and leaves an SNR-independent sub-pixel-phase bias floor.
    This stage removes that floor by fitting directly to the *continuous*
    (un-quantized) gradient magnitude: for each vertex it finds the
    sub-pixel signed normal distance ``t*`` to the gradient ridge and runs
    Gauss-Newton with Tukey-biweight reweighting to drive every ``t*`` to
    zero.

    The residual for vertex ``i`` is ``r_i = t*_i`` (signed distance from
    the vertex to the ridge along its outward normal ``n_i``).  Moving the
    vertex by a translation ``delta`` changes the residual by ``-(n_i .
    delta)``, so the translation Jacobian rows are ``[-n_v, -n_u]``; the
    rotation column (when fitted) is central-differenced.  The normal
    equations and the Tukey reweighting reuse the same machinery as the
    DT-LM stage.

    Parameters:
        vertices_vu: ``(N, 2)`` base (unshifted) model vertices.
        normals_vu: ``(N, 2)`` model normals; need not be unit length but
            should be (the DT techniques pass unit normals).
        sigma_normal_per_vertex_px: ``(N,)`` strictly-positive prior sigma;
            ``1 / sigma**2`` is the prior precision weight, identical to the
            DT-LM stage.
        gradient_magnitude: ``(H, W)`` continuous gradient magnitude image
            (``hypot`` of the gradient-vector components).
        initial_offset_vu: ``(dv, du)`` starting translation (the DT-LM
            optimum).
        weight_mask: optional ``(N,)`` multiplicative mask (e.g. the DT-LM
            polarity acceptance) applied to every vertex weight.  ``None``
            keeps all vertices.
        initial_rotation_rad: starting rotation (the DT-LM optimum).
        fit_rotation: when True the parameter vector is ``(dv, du, dtheta)``.
        pivot_vu: rotation pivot; defaults to the centroid of
            ``vertices_vu``.
        pivot_distance_px: pivot-to-image-centre distance for the rotation
            step-norm conversion.  Required when ``fit_rotation`` is True.
        max_iterations: Gauss-Newton iteration cap.
        step_tolerance_px: step-norm threshold for convergence.
        tukey_c: Holland-Welsch Tukey constant.
        half_width_px: half-width of the normal search window.
        sample_step_px: spacing of the normal sample points.
        max_total_displacement_px: cap on cumulative displacement from
            ``initial_offset_vu``; exceeding it discards the refinement.

    Returns:
        :class:`RidgeRefineResult`.

    Raises:
        ValueError: if shape requirements are violated, the prior sigmas
            are not strictly positive, or ``fit_rotation`` is True without a
            positive ``pivot_distance_px``.
    """
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    sigmas = np.asarray(sigma_normal_per_vertex_px, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] == 0:
        raise ValueError(f'vertices_vu must have shape (N, 2) with N > 0; got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    if sigmas.ndim != 1 or sigmas.shape[0] != verts.shape[0]:
        raise ValueError(
            'sigma_normal_per_vertex_px must be a 1-D vector matching '
            f'vertices_vu; got {sigmas.shape}'
        )
    if (sigmas <= 0.0).any() or not np.isfinite(sigmas).all():
        raise ValueError('sigma_normal_per_vertex_px entries must be finite and > 0')
    if gradient_magnitude.ndim != 2:
        raise ValueError(f'gradient_magnitude must be 2-D; got ndim={gradient_magnitude.ndim}')
    if fit_rotation and not (pivot_distance_px > 0.0):
        raise ValueError(
            'fit_rotation=True requires pivot_distance_px > 0 for the convergence test'
        )
    base_mask = (
        np.ones(verts.shape[0], np.float64)
        if weight_mask is None
        else np.asarray(weight_mask, np.float64)
    )
    pivot = (
        (float(verts[:, 0].mean()), float(verts[:, 1].mean()))
        if pivot_vu is None
        else (float(pivot_vu[0]), float(pivot_vu[1]))
    )
    inv_sigma_sq = 1.0 / (sigmas * sigmas)
    half = float(half_width_px)
    step = float(sample_step_px)
    # Symmetric, monotone sample offsets including 0; the central sample lets
    # a converged (t* -> 0) vertex report a zero residual exactly.
    n_side = round(half / step)
    t_samples = np.arange(-n_side, n_side + 1, dtype=np.float64) * step
    dv0, du0 = float(initial_offset_vu[0]), float(initial_offset_vu[1])
    dv, du, dtheta = dv0, du0, float(initial_rotation_rad)
    eps_t = 1.0e-3
    iterations = 0
    converged = False
    for _ in range(max_iterations):
        t_star, norms_current, valid = _ridge_normal_distances(
            vertices_vu=verts,
            normals_vu=norms,
            pivot_vu=pivot,
            gradient_magnitude=gradient_magnitude,
            dv=dv,
            du=du,
            dtheta=dtheta,
            t_samples=t_samples,
            sample_step_px=step,
        )
        scaled = t_star / sigmas
        tukey_w = tukey_biweight_weights(scaled, c=tukey_c)
        weights = inv_sigma_sq * tukey_w * valid * base_mask
        if not np.any(weights > 0):
            # No usable ridge evidence; keep whatever pose we have.
            break
        if fit_rotation:
            t_plus, _, _ = _ridge_normal_distances(
                vertices_vu=verts,
                normals_vu=norms,
                pivot_vu=pivot,
                gradient_magnitude=gradient_magnitude,
                dv=dv,
                du=du,
                dtheta=dtheta + eps_t,
                t_samples=t_samples,
                sample_step_px=step,
            )
            t_minus, _, _ = _ridge_normal_distances(
                vertices_vu=verts,
                normals_vu=norms,
                pivot_vu=pivot,
                gradient_magnitude=gradient_magnitude,
                dv=dv,
                du=du,
                dtheta=dtheta - eps_t,
                t_samples=t_samples,
                sample_step_px=step,
            )
            drdth = (t_plus - t_minus) / (2.0 * eps_t)
            jacobian = np.stack([-norms_current[:, 0], -norms_current[:, 1], drdth], axis=-1)
        else:
            jacobian = np.stack([-norms_current[:, 0], -norms_current[:, 1]], axis=-1)
        hessian, rhs = _weighted_normal_equations(jacobian, t_star, weights)
        try:
            gn_step = -np.linalg.solve(hessian, rhs)
        except np.linalg.LinAlgError:
            gn_step = -pinvh(hessian, rtol=DEFAULT_PINVH_RCOND) @ rhs
        dv += float(gn_step[0])
        du += float(gn_step[1])
        if fit_rotation:
            dtheta += float(gn_step[2])
        iterations += 1
        step_norm = _step_norm_px(
            gn_step,
            fit_rotation=fit_rotation,
            pivot_distance_px=pivot_distance_px,
        )
        if step_norm < step_tolerance_px:
            converged = True
            break
    displacement = math.hypot(dv - dv0, du - du0)
    if iterations == 0 or displacement > max_total_displacement_px:
        # Either no Gauss-Newton step ran (no valid ridge evidence) or the
        # stage walked too far to trust; keep the DT-LM pose.
        return RidgeRefineResult(
            offset_vu=(dv0, du0),
            rotation_rad=float(initial_rotation_rad),
            iterations=iterations,
            converged=converged,
            applied=False,
        )
    return RidgeRefineResult(
        offset_vu=(dv, du),
        rotation_rad=dtheta,
        iterations=iterations,
        converged=converged,
        applied=True,
    )


def lm_subpixel_refine(
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    sigma_normal_per_vertex_px: NDArrayFloatType,
    image_edge_dt: NDArrayFloatType,
    image_gradient_vu: NDArrayFloatType | None = None,
    initial_offset_vu: tuple[float, float] = (0.0, 0.0),
    initial_rotation_rad: float = 0.0,
    fit_rotation: bool = False,
    pivot_vu: tuple[float, float] | None = None,
    pivot_distance_px: float = 0.0,
    use_polarity: bool = True,
    max_iterations: int = DEFAULT_LM_MAX_ITERATIONS,
    damping: float = DEFAULT_LM_DAMPING,
    step_tolerance_px: float = DEFAULT_LM_STEP_TOLERANCE,
    tukey_c: float = DEFAULT_TUKEY_C,
    pinvh_rcond: float = DEFAULT_PINVH_RCOND,
    trust_region_px: float | None = None,
    tikhonov_alpha: float = 0.0,
    final_gradient_ridge: bool = False,
) -> LMRefineResult:
    """Refine a polyline-vs-image alignment by Levenberg-Marquardt.

    The cost function is

    .. math::
        C(p) = \\sum_i w_i \\, \\bigl[\\mathrm{DT}\\bigl(R(\\theta)\\,(x_i - x_p)
                                              + x_p + (\\Delta v, \\Delta u)\\bigr)\\bigr]^2

    where :math:`x_i` are the input vertices, :math:`x_p` is the rotation
    pivot, :math:`R(\\theta)` is the in-plane rotation, ``DT`` is the
    bilinearly-sampled image distance transform, and the per-vertex
    weight :math:`w_i` is the product of the prior precision
    :math:`1 / \\sigma_i^2` and the Tukey biweight evaluated at the
    scaled residual :math:`r_i / \\sigma_i`.  The Tukey weights are
    recomputed after each accepted LM step (iteratively-reweighted
    least squares).

    Parameters:
        vertices_vu: ``(N, 2)`` model vertex positions.
        normals_vu: ``(N, 2)`` model outward normals (only used if
            ``use_polarity`` is True).
        sigma_normal_per_vertex_px: ``(N,)`` strictly-positive prior
            sigma in pixels.  ``1 / sigma**2`` is the prior precision
            weight.
        image_edge_dt: ``(H, W)`` precomputed image distance transform.
        image_gradient_vu: ``(H, W, 2)`` gradient vector image; required
            when ``use_polarity`` is True, ignored otherwise.
        initial_offset_vu: ``(dv0, du0)`` starting translation.
        initial_rotation_rad: Starting rotation in radians.
        fit_rotation: When True the parameter vector is
            ``(dv, du, dtheta)``; otherwise ``(dv, du)``.
        pivot_vu: ``(v_p, u_p)`` rotation pivot.  Defaults to the
            centroid of ``vertices_vu``.
        pivot_distance_px: Approximate pivot-to-image-centre distance in
            pixels; used to convert rotation steps into pixel-equivalent
            increments for the convergence test.  Required when
            ``fit_rotation`` is True; ignored otherwise.
        use_polarity: When True, polarity-rejected vertices are assigned
            an effectively-infinite residual so the Tukey biweight zeroes
            their contribution.
        max_iterations: Iteration cap.
        damping: Initial Levenberg-Marquardt damping ``lambda``.
        step_tolerance_px: Step-norm threshold below which the iteration
            terminates with ``converged=True``.
        tukey_c: Holland-Welsch Tukey biweight constant.
        pinvh_rcond: Pseudoinverse cutoff for the final covariance.
        trust_region_px: Optional radius (pixels) around
            ``initial_offset_vu`` outside which a trial step is rejected
            without committing.  ``None`` (default) leaves the LM
            unconstrained — the legacy behaviour.  When set, every
            trial offset is checked against
            ``hypot(trial_dv - dv0, trial_du - du0) <= trust_region_px``;
            a violation marks the step as rejected (lambda doubled,
            iteration counter advanced) without updating ``dv`` / ``du``.
            This contains the joint LM + IRLS instability whereby
            Tukey reweighting can drag the polyline off the integer
            coarse-NCC seed onto an unrelated DT minimum (a crater
            rim, terminator edge, or surface boundary).
        tikhonov_alpha: Strength of a soft Tikhonov anchor pulling the
            translation back toward ``initial_offset_vu``.  ``0`` (default)
            disables the term — the legacy behaviour.  When positive,
            the cost adds a per-iteration penalty
            ``alpha * sum(weights) * ||(dv, du) - (dv0, du0)||^2`` which
            scales with the data so the LM trades off raw DT
            improvement against displacement.  The penalty is applied
            only to the translation degrees of freedom; rotation is
            never penalized.  The trust region is the hard outer
            bound; Tikhonov pulls the LM toward the seed *inside*
            that bound when the DT cost surface has a deeper but
            wrong minimum on the way (crater rims, terminator edges).
            The anchor biases the step only; it is excluded from the
            reported data-only covariance (see ``LMRefineResult``).
        final_gradient_ridge: When True, after the DT Levenberg-Marquardt
            converges, a final :func:`gradient_ridge_refine` stage polishes
            the offset against the *continuous* gradient-magnitude ridge,
            removing the sub-pixel-phase bias floor the integer-quantized DT
            zero-set leaves.  ``image_gradient_vu`` must be supplied (it is
            already required when ``use_polarity`` is True).  The reported
            ``residuals_px`` / ``weights`` / ``rms_px`` / ``covariance`` are
            recomputed against the DT at the ridge-refined pose, so the
            spurious gates and the reported uncertainty stay on the same DT
            footing as without the stage.  ``False`` (default) leaves the
            DT-LM optimum unchanged.
    Returns:
        :class:`LMRefineResult`.

    Raises:
        ValueError: if any shape requirement is violated, the prior
            sigmas are not strictly positive, or ``fit_rotation`` is
            True without a positive ``pivot_distance_px``.
    """
    verts = np.asarray(vertices_vu, np.float64)
    norms = np.asarray(normals_vu, np.float64)
    sigmas = np.asarray(sigma_normal_per_vertex_px, np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] == 0:
        raise ValueError(f'vertices_vu must have shape (N, 2) with N > 0; got {verts.shape}')
    if norms.shape != verts.shape:
        raise ValueError(f'normals_vu must match vertices_vu shape; got {norms.shape}')
    if sigmas.ndim != 1 or sigmas.shape[0] != verts.shape[0]:
        raise ValueError(
            'sigma_normal_per_vertex_px must be a 1-D vector matching '
            f'vertices_vu; got {sigmas.shape}'
        )
    if (sigmas <= 0.0).any() or not np.isfinite(sigmas).all():
        raise ValueError('sigma_normal_per_vertex_px entries must be finite and > 0')
    if image_edge_dt.ndim != 2:
        raise ValueError(f'image_edge_dt must be 2-D; got ndim={image_edge_dt.ndim}')
    if fit_rotation and not (pivot_distance_px > 0.0):
        raise ValueError(
            'fit_rotation=True requires pivot_distance_px > 0 for the convergence test'
        )
    if final_gradient_ridge and image_gradient_vu is None:
        raise ValueError('final_gradient_ridge=True requires image_gradient_vu')
    if use_polarity:
        if image_gradient_vu is None:
            raise ValueError('use_polarity=True requires image_gradient_vu')
        polarity_mask = polarity_filter(
            verts,
            norms,
            image_gradient_vu,
            offset_vu=initial_offset_vu,
        )
    else:
        polarity_mask = np.ones(verts.shape[0], dtype=bool)
    pivot = (
        (float(verts[:, 0].mean()), float(verts[:, 1].mean()))
        if pivot_vu is None
        else (float(pivot_vu[0]), float(pivot_vu[1]))
    )
    inv_sigma_sq = 1.0 / (sigmas * sigmas)
    state = _LMState(
        dv=float(initial_offset_vu[0]),
        du=float(initial_offset_vu[1]),
        dtheta=float(initial_rotation_rad),
        polarity_mask=polarity_mask,
    )
    lambda_ = float(damping)
    best_cost = math.inf
    n_params = 3 if fit_rotation else 2
    while state.iteration < max_iterations:
        residuals, jacobian = _compute_residuals_and_jacobian(
            vertices_vu=verts,
            pivot_vu=pivot,
            image_dt=image_edge_dt,
            dv=state.dv,
            du=state.du,
            dtheta=state.dtheta,
            fit_rotation=fit_rotation,
        )
        # Polarity-reject vertices contribute the constant penalty that
        # the Tukey biweight will zero on the next reweighting step.
        residuals = np.where(state.polarity_mask, residuals, _INFINITY_DT_PENALTY_PX)
        # Compute Tukey biweight weights from the (sigma-scaled) residuals.
        scaled = residuals / sigmas
        tukey_w = tukey_biweight_weights(scaled, c=tukey_c)
        # Zero polarity-rejected vertices directly via the mask rather than
        # relying on the large-penalty residual being driven past the Tukey
        # cutoff: that chain breaks for an enormous per-vertex sigma
        # (scaled = penalty / sigma <= c), and ``navigate`` may not raise.
        # The explicit mask makes rejection independent of sigma.
        weights = inv_sigma_sq * tukey_w * state.polarity_mask
        cost_before = _weighted_cost(weights, residuals)
        state.raw_residuals = residuals
        state.weights = weights
        state.jacobian = jacobian
        if not np.any(weights > 0):
            break
        hessian, rhs = _weighted_normal_equations(jacobian, residuals, weights)
        # Tikhonov anchor toward the initial seed on translation only.
        # Scaled by ``sum(weights)`` so the penalty tracks the data
        # size: ``alpha`` is a dimensionless ratio (penalty per
        # weighted-residual-equivalent at displacement = 1 px).
        # ``rotation`` is never penalised — only the translation
        # block of the (2 or 3)-DoF Hessian / RHS receives the term.
        if tikhonov_alpha > 0.0:
            tikhonov_lambda = float(tikhonov_alpha) * float(weights.sum())
            displacement_v = state.dv - float(initial_offset_vu[0])
            displacement_u = state.du - float(initial_offset_vu[1])
            hessian = hessian.copy()
            rhs = rhs.copy()
            hessian[0, 0] += tikhonov_lambda
            hessian[1, 1] += tikhonov_lambda
            rhs[0] += tikhonov_lambda * displacement_v
            rhs[1] += tikhonov_lambda * displacement_u
            cost_before = cost_before + tikhonov_lambda * (
                displacement_v * displacement_v + displacement_u * displacement_u
            )
        # LM dampening: H_lm = H + lambda * diag(H).
        diag = np.diag(np.diag(hessian))
        hessian_lm = hessian + lambda_ * diag
        try:
            step = -np.linalg.solve(hessian_lm, rhs)
        except np.linalg.LinAlgError:
            step = -pinvh(hessian_lm, rtol=pinvh_rcond) @ rhs
        trial_dv = state.dv + float(step[0])
        trial_du = state.du + float(step[1])
        trial_dtheta = state.dtheta + float(step[2]) if fit_rotation else state.dtheta
        # Trust-region rejection: refuse trial offsets that have walked
        # too far from the integer-precision coarse seed.  The IRLS-LM
        # combo can otherwise drift the polyline onto unrelated DT
        # minima (crater rims, terminator edges) when Tukey reweighting
        # at the trial offset finds a different inlier set.
        if trust_region_px is not None:
            disp = math.hypot(
                trial_dv - float(initial_offset_vu[0]),
                trial_du - float(initial_offset_vu[1]),
            )
            if disp > trust_region_px:
                lambda_ = min(lambda_ * 2.0, 1.0e6)
                state.iteration += 1
                if lambda_ >= 1.0e6:
                    break
                continue
        # Evaluate cost at the trial point.
        trial_residuals, _ = _compute_residuals_and_jacobian(
            vertices_vu=verts,
            pivot_vu=pivot,
            image_dt=image_edge_dt,
            dv=trial_dv,
            du=trial_du,
            dtheta=trial_dtheta,
            fit_rotation=fit_rotation,
        )
        trial_residuals = np.where(state.polarity_mask, trial_residuals, _INFINITY_DT_PENALTY_PX)
        # Compare ``trial_cost`` against ``cost_before`` using the SAME
        # weights computed at the start of this iteration.  Recomputing
        # Tukey biweights at the trial offset (the legacy behaviour)
        # parameterises the cost function by the offset itself, so an
        # "improvement" can mean "the trial offset's reweighting found
        # a different inlier set whose sum-of-squares is lower" rather
        # than "the trial offset has smaller residuals at the current
        # inlier set".  Freezing the weights for the inner LM step is
        # the standard IRLS / LM separation that keeps the cost
        # function fixed during a single Gauss-Newton step; IRLS
        # reweighting still happens between iterations of the outer
        # loop.  Without this, the LM can drift onto unrelated DT
        # minima (crater rims, terminator edges) on textured bodies
        # where multiple basins are reachable from the seed.
        trial_cost = _weighted_cost(weights, trial_residuals)
        if tikhonov_alpha > 0.0:
            trial_disp_v = trial_dv - float(initial_offset_vu[0])
            trial_disp_u = trial_du - float(initial_offset_vu[1])
            trial_cost = trial_cost + tikhonov_lambda * (
                trial_disp_v * trial_disp_v + trial_disp_u * trial_disp_u
            )
        if trial_cost < cost_before:
            state.dv = trial_dv
            state.du = trial_du
            state.dtheta = trial_dtheta
            lambda_ = max(lambda_ * 0.5, 1.0e-12)
            best_cost = trial_cost
            step_norm = _step_norm_px(
                step,
                fit_rotation=fit_rotation,
                pivot_distance_px=pivot_distance_px,
            )
            state.iteration += 1
            # Recompute residuals / weights / Jacobian at the accepted pose
            # immediately so diagnostics (rms_px, inlier_count, covariance)
            # reflect the committed parameters even if the loop exits via
            # max_iterations on the next check.  Without this, an accepted
            # step in the final iteration would leave state.raw_residuals /
            # state.weights / state.jacobian reflecting the pre-step pose.
            residuals_final, jacobian_final = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
            residuals_final = np.where(
                state.polarity_mask, residuals_final, _INFINITY_DT_PENALTY_PX
            )
            final_scaled = residuals_final / sigmas
            final_tukey = tukey_biweight_weights(final_scaled, c=tukey_c)
            state.raw_residuals = residuals_final
            state.weights = inv_sigma_sq * final_tukey * state.polarity_mask
            state.jacobian = jacobian_final
            if step_norm < step_tolerance_px:
                state.converged = True
                break
        else:
            lambda_ = min(lambda_ * 2.0, 1.0e6)
            state.iteration += 1
            if lambda_ >= 1.0e6:
                break
    if best_cost == math.inf:
        # Iteration exited without finding any improvement: fall back to
        # the latest residual / weight set already cached on ``state``.
        residuals_final = state.raw_residuals
        if residuals_final.size == 0:
            residuals_final = sample_dt_bilinear(
                image_edge_dt,
                _shift_vertices(_rotate_vertices(verts, pivot, state.dtheta), state.dv, state.du),
            )
            residuals_final = np.where(
                state.polarity_mask, residuals_final, _INFINITY_DT_PENALTY_PX
            )
            final_scaled = residuals_final / sigmas
            final_tukey = tukey_biweight_weights(final_scaled, c=tukey_c)
            state.raw_residuals = residuals_final
            state.weights = inv_sigma_sq * final_tukey * state.polarity_mask
            _, state.jacobian = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
    # Final continuous gradient-ridge stage.  The DT-LM above minimised
    # distance to an integer-quantized edge mask, whose zero-set snaps the
    # recovered edge to integer pixels and leaves a sub-pixel-phase bias
    # floor; this stage polishes the offset against the un-quantized
    # gradient magnitude.  Only run when there is surviving DT evidence to
    # anchor it (a degenerate DT-LM has no inlier set to start from); the
    # ridge stage itself keeps the DT-LM pose if it finds no usable ridge
    # residual or walks too far.  The DT residuals / weights / Jacobian are
    # then recomputed at the refined pose so the reported rms / covariance /
    # spurious gates stay on the same DT footing as the no-ridge path.
    if final_gradient_ridge and image_gradient_vu is not None and np.any(state.weights > 0):
        gradient_magnitude = np.hypot(image_gradient_vu[..., 0], image_gradient_vu[..., 1]).astype(
            np.float64
        )
        ridge = gradient_ridge_refine(
            vertices_vu=verts,
            normals_vu=norms,
            sigma_normal_per_vertex_px=sigmas,
            gradient_magnitude=gradient_magnitude,
            initial_offset_vu=(state.dv, state.du),
            weight_mask=state.polarity_mask.astype(np.float64),
            initial_rotation_rad=state.dtheta,
            fit_rotation=fit_rotation,
            pivot_vu=pivot,
            pivot_distance_px=pivot_distance_px,
            tukey_c=tukey_c,
        )
        if ridge.applied:
            state.dv, state.du = ridge.offset_vu
            state.dtheta = ridge.rotation_rad
            residuals_ridge, jacobian_ridge = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
            residuals_ridge = np.where(
                state.polarity_mask, residuals_ridge, _INFINITY_DT_PENALTY_PX
            )
            ridge_scaled = residuals_ridge / sigmas
            ridge_tukey = tukey_biweight_weights(ridge_scaled, c=tukey_c)
            state.raw_residuals = residuals_ridge
            state.weights = inv_sigma_sq * ridge_tukey * state.polarity_mask
            state.jacobian = jacobian_ridge
    final_weights = state.weights
    final_residuals = state.raw_residuals
    inlier_count = int(np.sum(final_weights > 0))
    degenerate = inlier_count == 0 or final_weights.sum() == 0.0
    if not degenerate:
        rms_px = float(math.sqrt(np.sum(final_weights * final_residuals**2) / final_weights.sum()))
    else:
        # Every vertex was rejected: there is no surviving evidence to
        # constrain the fit.  Report +inf (not 0.0) so the DT techniques'
        # ``result.rms_px > floor`` spurious test fires; a zero RMS would
        # otherwise be read downstream as a perfect fit.
        rms_px = float('inf')
    # Unweighted RMS over ALL vertices.  Unlike the Tukey-weighted
    # ``rms_px``, this does not down-weight outliers, so a mis-converged
    # fit whose bad arc was rejected to ~0 weight still reports a large
    # raw RMS.  Well-defined even in the degenerate (all-weights-zero)
    # case because the residuals exist regardless of the weights.
    if final_residuals.size:
        raw_rms_px = float(math.sqrt(float(np.mean(final_residuals**2))))
    else:
        raw_rms_px = float('inf')
    covariance: NDArrayFloatType
    # The reported covariance is DATA-ONLY: it is the pseudoinverse of the
    # M-estimator information matrix ``J^T diag(w) J`` evaluated at the
    # converged pose and deliberately EXCLUDES the Tikhonov anchor
    # contribution (``tikhonov_alpha`` adds ``alpha * sum(w)`` to the
    # translation diagonal of the *iteration* Hessian to bias the step,
    # but that prior is a fitting aid, not measured information, so it
    # must not shrink the reported uncertainty).
    #
    # Guard against the "no information" cases that would otherwise let
    # information_matrix_to_covariance produce a misleading zero-covariance
    # answer (pinvh of the zero information matrix is zero — which would
    # falsely advertise perfect certainty about the fit).  These conditions
    # mean there is no inlier evidence to constrain the parameters; the inf
    # sentinel correctly signals "fully unconstrained" and stays consistent
    # with the ``rms_px = +inf`` degenerate result above.
    if state.jacobian.size == 0 or state.jacobian.shape[1] != n_params or degenerate:
        covariance = cast(NDArrayFloatType, np.full((n_params, n_params), np.inf, dtype=np.float64))
    else:
        covariance = information_matrix_to_covariance(
            state.jacobian, final_weights, rcond=pinvh_rcond
        )
    return LMRefineResult(
        offset_vu=(state.dv, state.du),
        rotation_rad=state.dtheta,
        covariance=covariance,
        residuals_px=final_residuals,
        weights=final_weights,
        rms_px=rms_px,
        raw_rms_px=raw_rms_px,
        iterations=state.iteration,
        converged=state.converged,
        inlier_count=inlier_count,
        degenerate=degenerate,
    )
