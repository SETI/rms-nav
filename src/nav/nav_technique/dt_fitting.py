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

from nav.support.distance_transform import sample_dt_bilinear
from nav.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'DEFAULT_LM_DAMPING',
    'DEFAULT_LM_MAX_ITERATIONS',
    'DEFAULT_LM_STEP_TOLERANCE',
    'DEFAULT_PINVH_RCOND',
    'DEFAULT_TUKEY_C',
    'LMRefineResult',
    'coarse_ncc_search',
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


_INFINITY_DT_PENALTY_PX: float = 1.0e6
"""Effective ``+inf`` penalty assigned to polarity-rejected vertices.

The polarity filter forces a vertex to contribute an unbounded cost; we
encode ``+inf`` as a very large finite number so the LM linear system
remains numerically well-defined.  Such a vertex's residual is so large
the Tukey biweight zeroes its weight on the first reweighting step, so
the value's exact magnitude does not influence the converged estimate.
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

    Out-of-bounds vertices have their gradient sampled at the nearest
    in-bounds pixel via clamping; that pixel's gradient is rarely a real
    edge, so the dot product is dominated by background noise and the
    vertex is rejected on its own merits.

    Parameters:
        vertices_vu: ``(N, 2)`` model vertex positions.
        normals_vu: ``(N, 2)`` model outward normal at each vertex.
        image_gradient_vu: ``(H, W, 2)`` per-pixel gradient image as
            produced by
            :func:`nav.nav_orchestrator.image_derivatives.compute_image_gradient_vu`.
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
    sample_v = np.clip(np.rint(verts[:, 0] + dv).astype(np.int64), 0, height - 1)
    sample_u = np.clip(np.rint(verts[:, 1] + du).astype(np.int64), 0, width - 1)
    gv = image_gradient_vu[sample_v, sample_u, 0]
    gu = image_gradient_vu[sample_v, sample_u, 1]
    dot = norms[:, 0] * gv + norms[:, 1] * gu
    return cast(NDArrayBoolType, dot > 0.0)


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
            from the M-estimator information matrix at the converged
            point.
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
        weights = inv_sigma_sq * tukey_w
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
            state.weights = inv_sigma_sq * final_tukey
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
            state.weights = inv_sigma_sq * final_tukey
            _, state.jacobian = _compute_residuals_and_jacobian(
                vertices_vu=verts,
                pivot_vu=pivot,
                image_dt=image_edge_dt,
                dv=state.dv,
                du=state.du,
                dtheta=state.dtheta,
                fit_rotation=fit_rotation,
            )
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
