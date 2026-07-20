"""Robust weighting and covariance for the DT fit.

The Holland-Welsch biweight the IRLS loop reweights with, and the
information-matrix inversion that turns a converged fit into a covariance.
"""

import math
from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_PINVH_RCOND,
    DEFAULT_TUKEY_C,
)
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'information_matrix_to_covariance',
    'tukey_biweight_weights',
]


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
