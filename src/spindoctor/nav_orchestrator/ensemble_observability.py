"""Observability linear algebra shared by the ensemble combine.

Pure-numpy helpers that reason about which parameter directions a
per-technique result actually constrains.  A result reports an
unobservable direction in one of two ways: an exactly-zero variance (the
Moore-Penrose null-space convention the rank-1 ring-edge path enforces)
or a huge axis-aligned sentinel variance (``ROTATION_UNOBSERVABLE_VARIANCE``
= 1e15 and the 1e12-scale translation sentinels).  These functions turn
that convention into observable-subspace bases, a sentinel-aware
pseudo-inverse, and a Mahalanobis distance measured only where two
results genuinely overlap.  They live apart from ``ensemble`` so that
module stays under the 1000-line limit; ``ensemble`` imports them.
"""

import math
from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'mahalanobis_distance',
    'mixed_scale_pinvh',
    'observable_basis',
    'observable_intersection_basis',
]

# The floor separates genuine uncertainty from sentinels: 1e10 px^2 is a
# 1e5 px sigma, far beyond any real constraint on a <= 4096 px image and far
# below the sentinels.  The null threshold matches the rank-1
# relative-eigenvalue test used by the ring-edge technique.
UNOBSERVABLE_VARIANCE_FLOOR = 1.0e10
NULL_RELATIVE_THRESHOLD = 1.0e-8
# Two projectors' summed eigenvalue must reach 2 (within tolerance) for a
# direction to lie in the intersection of their subspaces.
INTERSECTION_EIGENVALUE_TOL = 1.0e-9


def mixed_scale_pinvh(mat: NDArrayFloatType, *, rcond: float) -> NDArrayFloatType:
    """Moore-Penrose ``pinvh`` with unobservable-sentinel axes quarantined.

    ``pinvh``'s eigenvalue cutoff is relative to the *largest* eigenvalue,
    so a huge unobservable-axis sentinel (``ROTATION_UNOBSERVABLE_VARIANCE``
    or a 1e12-scale translation sentinel) raises the cutoff past every
    genuine variance: a raw ``pinvh`` on ``diag(0.04, 0.04, 1e15)``
    truncates the well-constrained 0.04 px^2 axes and the fused offset
    silently collapses to zero along them.  Sentinel axes are axis-aligned
    with zero cross-terms by construction, so they are detected by row
    norm — at or above ``UNOBSERVABLE_VARIANCE_FLOOR`` (covariance side)
    or positive and at or below its reciprocal (information side) — and
    inverted independently as ``1 / diag``; the remaining block gets an
    ordinary Moore-Penrose ``pinvh``, which the information-form merge's
    minimum-norm / orthogonal-projection semantics rely on (a merely
    generalized inverse would project the fused offset obliquely).

    Parameters:
        mat: Symmetric PSD matrix (covariance or information form).
        rcond: Relative eigenvalue cutoff for the non-sentinel block.

    Returns:
        The pseudo-inverse, same shape as ``mat``.
    """
    row_norm = np.linalg.norm(mat, axis=1)
    sentinel = (row_norm >= UNOBSERVABLE_VARIANCE_FLOOR) | (
        (row_norm > 0.0) & (row_norm <= 1.0 / UNOBSERVABLE_VARIANCE_FLOOR)
    )
    if not bool(sentinel.any()):
        return cast(NDArrayFloatType, pinvh(mat, rtol=rcond))
    out = np.zeros_like(mat)
    for i in np.flatnonzero(sentinel):
        out[i, i] = 1.0 / mat[i, i]
    regular = np.flatnonzero(~sentinel)
    if regular.size:
        block = np.ascontiguousarray(mat[np.ix_(regular, regular)])
        out[np.ix_(regular, regular)] = pinvh(block, rtol=rcond)
    return out


def observable_basis(cov: NDArrayFloatType) -> NDArrayFloatType:
    """Orthonormal basis (as columns) of a covariance's observable subspace.

    A direction is unobservable when it is either a sentinel axis (a
    diagonal variance at or above ``UNOBSERVABLE_VARIANCE_FLOOR`` — the
    sentinels are axis-aligned by construction) or a Moore-Penrose null
    of the remaining block (a relative eigenvalue of the
    correlation-form matrix below ``NULL_RELATIVE_THRESHOLD``, so the
    test is not distorted by mixed px/rad units).

    Parameters:
        cov: Symmetric PSD covariance, ``(n, n)``.

    Returns:
        ``(n, k)`` array with orthonormal columns; ``k`` may be zero.
    """
    n = cov.shape[0]
    axes = np.flatnonzero(np.diag(cov) < UNOBSERVABLE_VARIANCE_FLOOR)
    if axes.size == 0:
        return cast(NDArrayFloatType, np.zeros((n, 0), np.float64))
    sub = cov[np.ix_(axes, axes)]
    d = np.sqrt(np.clip(np.diag(sub), 0.0, None))
    d_safe = np.where(d > 0.0, d, 1.0)
    corr = sub / np.outer(d_safe, d_safe)
    w, v = np.linalg.eigh(corr)
    kept = v[:, w > NULL_RELATIVE_THRESHOLD]
    if kept.shape[1] == 0:
        return cast(NDArrayFloatType, np.zeros((n, 0), np.float64))
    # Map the correlation-space directions back to parameter space
    # (range(sub) = D @ range(corr)) and re-orthonormalize via SVD.
    embedded = np.zeros((n, kept.shape[1]), np.float64)
    embedded[axes, :] = d[:, np.newaxis] * kept
    u, s, _ = np.linalg.svd(embedded, full_matrices=False)
    return cast(NDArrayFloatType, u[:, s > NULL_RELATIVE_THRESHOLD * float(s.max())])


def observable_intersection_basis(
    cov_a: NDArrayFloatType, cov_b: NDArrayFloatType
) -> NDArrayFloatType:
    """Orthonormal basis of the intersection of two observable subspaces.

    Parameters:
        cov_a: First covariance, ``(n, n)``.
        cov_b: Second covariance, same shape.

    Returns:
        ``(n, k)`` array with orthonormal columns; ``k`` may be zero.
    """
    basis_a = observable_basis(cov_a)
    basis_b = observable_basis(cov_b)
    n = cov_a.shape[0]
    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return cast(NDArrayFloatType, np.zeros((n, 0), np.float64))
    # Eigenvalues of the summed projectors lie in [0, 2]; a direction lies
    # in both subspaces iff its eigenvalue is 2.
    proj_sum = basis_a @ basis_a.T + basis_b @ basis_b.T
    w, v = np.linalg.eigh(proj_sum)
    return cast(NDArrayFloatType, v[:, w > 2.0 - INTERSECTION_EIGENVALUE_TOL])


def mahalanobis_distance(
    mu_a: NDArrayFloatType,
    cov_a: NDArrayFloatType,
    mu_b: NDArrayFloatType,
    cov_b: NDArrayFloatType,
    *,
    rcond: float,
) -> float:
    """Return the Mahalanobis distance between two estimates.

    Measured only in the intersection of the two estimates' observable
    subspaces.  A direction one estimate cannot observe
    carries junk in its parameter vector — the LM/centroid machinery
    reports *something* there — so differencing it against an estimate
    that does observe it manufactures disagreement out of nothing (the
    flat-ring rank-1 result against a full-rank blob), and differencing
    it against another junk value is meaningless either way.  Estimates
    with no shared observable direction return 0.0: there is nothing to
    disagree on, and grouping them lets the information-form combine
    merge their complementary constraints.
    """
    delta = mu_a - mu_b
    basis = observable_intersection_basis(cov_a, cov_b)
    if basis.shape[1] == 0:
        return 0.0
    delta_r = basis.T @ delta
    cov_r = basis.T @ (cov_a + cov_b) @ basis
    pinv_r = mixed_scale_pinvh(cov_r, rcond=rcond)
    d_sq = float(delta_r.T @ pinv_r @ delta_r)
    if d_sq < 0:
        # Numerical safety: the generalized inverse may yield a tiny
        # negative quadratic form due to floating-point; clamp to zero.
        d_sq = 0.0
    return float(math.sqrt(d_sq))
