"""Covariance geometry for the ring-edge fit.

The rank-1 projection, the aggregate edge-normal orientation, and the radial
orbit-uncertainty channel (the absorbed-translation sensitivity and the
covariance inflation built from it).  Split out of
:mod:`spindoctor.nav_technique.nav_technique_ring_edge` to keep both modules
under the size cap; the technique imports every helper from here.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.linalg import pinvh

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.nav_technique.dt_fitting import DEFAULT_PINVH_RCOND
from spindoctor.support.types import NDArrayFloatType

__all__: list[str] = []


_RANK1_NULL_RELATIVE_THRESHOLD: float = 1.0e-8
"""Eigenvalue ratio below which a 2x2 covariance is treated as rank-1.

Matches the ensemble's scale-independent rank-deficiency test so the
two paths agree on whether a result is rank-deficient.
"""


def _is_rank_1(covariance: NDArrayFloatType) -> bool:
    """Return True when the covariance is rank-deficient in its translation block.

    For both 2x2 (translation-only) and 3x3 (translation + rotation) inputs
    the test runs on the top-left 2x2 block — flat-ring rank-deficiency is
    a property of the translation parameterisation, regardless of whether
    rotation is fit.  Uses the same scale-independent test as the ensemble
    combine: the ratio of the smallest absolute eigenvalue to the largest
    must fall below :data:`_RANK1_NULL_RELATIVE_THRESHOLD`.
    """
    if covariance.shape not in ((2, 2), (3, 3)):
        return False
    block = covariance[:2, :2]
    eigvals = np.linalg.eigvalsh(block)
    largest = float(np.abs(eigvals).max())
    smallest = float(np.abs(eigvals).min())
    if largest == 0.0:
        return True
    return smallest / largest < _RANK1_NULL_RELATIVE_THRESHOLD


def _aggregate_normal_orientation(polarity_normals: NDArrayFloatType) -> NDArrayFloatType:
    """Return the dominant unit-normal orientation of the aggregated edges.

    The dominant eigenvector of the per-vertex normals' outer-product sum;
    polarity-sign-independent (a gap's inner and outer edges carry opposite
    normal senses, so a plain mean could cancel).

    Well-conditioned only when the normals concentrate around one axis, which
    is the rank-1 (all-straight) case this serves; on a well-covered curved
    arc the two eigenvalues converge and the returned axis is arbitrary.  The
    orbit-uncertainty channel therefore uses
    :func:`_absorbed_orbit_sensitivity` instead everywhere except the rank-1
    path, which reuses this exact axis so the projected covariance stays
    exactly singular along the tangent.

    Parameters:
        polarity_normals: ``(N, 2)`` per-vertex edge normals (either sign).

    Returns:
        Unit 2-vector in ``(v, u)`` order.
    """
    outer_sum = polarity_normals.T @ polarity_normals
    _eigvals, eigvecs = np.linalg.eigh(outer_sum)
    return cast(NDArrayFloatType, eigvecs[:, -1])


def _absorbed_orbit_sensitivity(
    polarity_normals: NDArrayFloatType, weights: NDArrayFloatType
) -> NDArrayFloatType:
    """Return how much of a coherent radial displacement the fit absorbs as translation.

    A catalog-orbit error displaces every vertex along its own outward radial
    direction by the same amount ``d``.  The DT fit measures along-normal
    residuals, so the translation it converges to is the weighted
    least-squares minimiser of ``sum_i w_i (n_i . t - d)**2``, i.e. the
    solution of ``M t = d b`` with

    .. math::
        M = \\sum_i w_i \\, n_i n_i^{T}, \\qquad b = \\sum_i w_i \\, n_i .

    The absorbed translation is therefore ``t = d * g`` with ``g = M^{+} b``,
    and a radial uncertainty ``sigma`` contributes ``sigma**2 g g^{T}`` to the
    reported covariance.  Returning ``g`` (not a unit direction) is what makes
    the geometry honest:

    - A short arc has nearly parallel normals: ``g`` is a unit vector along
      them, and the full variance lands on the radial axis.
    - A rank-1 straight edge gives ``M = W n n^{T}`` and ``b = W n``, so
      ``g = n`` exactly -- the same rank-1 term the projected covariance uses.
    - A full annulus has ``b ~ 0``: a uniform radial error is a DILATION, not
      a translation, so the linearized fit absorbs almost none of it and no
      radial axis is singled out.  The inflation does NOT vanish there -- the
      caller turns a small ``||g||`` into an isotropic term (see
      :func:`_orbit_inflated_covariance`), because the nonlinear acquisition
      can still lock onto a translation.  The previous dominant-eigenvector
      construction instead returned a numerically arbitrary axis in this
      regime (the two eigenvalues converge) and widened one arbitrary axis by
      the full variance while leaving the perpendicular axis untouched.

    ``b`` uses the normals with their signs INTACT, because both emitting
    models document ``RingEdgePolyline.normals_vu`` as radially outward per
    vertex (the technique's aggregation applies one global flip, which
    ``g g^T`` is invariant to).  Preserving the relative senses is what makes
    the dilation cancellation above work, and it is also what makes opposite
    radial sides of the planet in a wide field cancel correctly instead of
    being fabricated into a common translation.  The failure mode of a
    mis-signed edge is therefore an UNDER-inflation (a spurious cancellation),
    not an over-inflation.

    Parameters:
        polarity_normals: ``(N, 2)`` per-vertex edge normals (either sign).
        weights: ``(N,)`` non-negative per-vertex final fit weights.

    A small ``||g||`` says the LINEARIZED fit absorbs little, NOT that the
    recovered offset is safe: the acquisition is nonlinear, and the coarse
    integer search can still select a basin whose translation aligns a long
    arc of a radially misplaced ring (the simulated closed-ringlet scene lands
    ~2.2 px from truth on a 2.5 px planted radial error, with every vertex an
    inlier).  The caller therefore blends this direction with an isotropic
    term as ``||g||`` falls -- see :func:`_orbit_inflated_covariance` -- so a
    near-isotropic geometry reports the bound in every direction rather than
    reporting nothing.

    ``||g||`` is NOT capped.  It legitimately exceeds 1 over a wide band of
    arc coverage (about 1.27 at 180 degrees): to explain "every vertex moved
    outward by d" a single translation overshoots the middle of the arc to
    reduce the error at its ends, so the fit really does absorb more than the
    displacement itself.  An earlier revision clamped it at 1.0 on the
    argument that a displacement of ``sigma`` cannot move the answer by more
    than ``sigma``; that clamp contradicted the derivation it was applied to
    and fired in essentially every non-degenerate case, making the shipped
    behavior the clamp's rather than the model's.

    Returns:
        The ``(2,)`` sensitivity vector ``g``.  Magnitude is bounded in
        practice by the pseudoinverse cutoff, which drops directions the fit
        does not constrain (whose variance the fit's own covariance already
        reports as large).
    """
    w = np.asarray(weights, np.float64)
    if w.size == 0 or not bool(np.any(w > 0.0)):
        w = np.ones(polarity_normals.shape[0], np.float64)
    info = (polarity_normals * w[:, None]).T @ polarity_normals
    info = 0.5 * (info + info.T)
    b = w @ polarity_normals
    g = pinvh(info, rtol=DEFAULT_PINVH_RCOND) @ b
    return cast(NDArrayFloatType, g)


def _effective_orbit_sigma_px(features: list[NavFeature], weights: NDArrayFloatType) -> float:
    """Effective fully-correlated radial orbit sigma over the consumed edges.

    Each consumed ``RING_EDGE`` feature carries a declared radial
    orbit-uncertainty sigma (``RingEdgePolyline.sigma_orbit_radial_px``);
    the fit's translation absorbs a weighted mix of the features' coherent
    radial displacements, so the effective sigma is the mean of the
    per-feature sigmas weighted by each feature's share of the final LM
    weight.  The per-feature orbit errors are deliberately treated as FULLY
    CORRELATED (a weighted mean of sigmas, not an independent-error
    quadrature combine): the common multi-edge case is the inner and outer
    edge of the same feature, whose orbit error IS shared.

    That combine is conservative only for SAME-SENSE geometry -- features
    whose outward radial directions point the same way in image space, so a
    common radial error displaces them together.  For features on opposite
    radial sides of the planet in a wide field a common error is a dilation
    that the fit largely does not absorb as translation, and pairing a mean
    of sigmas with one representative direction over-inflates rather than
    erring conservative.  The geometry partially self-limits: wide-spread
    normals shrink the absorbed-sensitivity vector
    (:func:`_absorbed_orbit_sensitivity`), so the inflation shrinks with
    them.  A per-feature sensitivity decomposition is what would model the
    opposite-sense case properly.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        weights: Per-vertex final weights from the fit, in the same
            concatenation order.

    Returns:
        The effective sigma in pixels; ``0.0`` when no feature declares an
        orbit uncertainty.  When every vertex weight is zero (a degenerate
        fit) the maximum declared sigma is returned -- the conservative
        bound, though a degenerate fit is spurious anyway.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    max_sigma = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        w_f = float(np.sum(weights[cursor : cursor + n]))
        cursor += n
        s_f = float(feat.geometry.sigma_orbit_radial_px)
        max_sigma = max(max_sigma, s_f)
        weighted_sum += w_f * s_f
        weight_total += w_f
    if max_sigma <= 0.0:
        return 0.0
    if weight_total <= 0.0:
        return max_sigma
    return weighted_sum / weight_total


def _orbit_inflated_covariance(
    covariance: NDArrayFloatType,
    sensitivity_g: NDArrayFloatType,
    sigma_orbit_px: float,
) -> NDArrayFloatType:
    """Add the orbit-uncertainty variance to the covariance's translation block.

    The added term interpolates between a purely directional and a purely
    isotropic inflation on the absorbed-sensitivity magnitude ``||g||``:

    .. math::
        \\Sigma \\mathrel{+}= \\sigma^{2}
            \\bigl[\\, g g^{T} + (1 - ||g||^{2})\\, I \\,\\bigr]

    - ``||g|| = 1`` (a short arc, or a rank-1 straight edge where ``g`` is
      exactly the projection's own axis): the term is exactly
      ``sigma**2 g g^T``.  Only the radial axis widens, and an exactly
      singular rank-1 covariance stays exactly singular along its tangent.
    - ``||g|| -> 0`` (a closed annulus, or features whose radial directions
      cancel): the term becomes ``sigma**2 I``.  The linearized fit absorbs
      almost nothing there, but the nonlinear acquisition still can -- the
      closed-ringlet regression scene lands ~2.2 px from truth on a 2.5 px
      planted radial error -- and which direction it locks in is exactly what
      cannot be predicted, so the bound is reported on every axis instead of
      on an axis chosen by rounding.

    The interpolation is smooth (no threshold to tune), every term is positive
    semidefinite so the result stays a valid covariance, and the rotation
    block of a 3x3 input is untouched.

    What this actually changes, stated plainly: for ``||g|| <= 1`` the added
    term's MAJOR eigenvalue is exactly ``sigma**2`` whatever direction ``g``
    points (``|g|**2 + (1 - |g|**2) = 1``), and only its minor eigenvalue,
    ``sigma**2 (1 - ||g||**2)``, depends on the geometry.  Since the tier gate
    reads ``max(sigma_dv, sigma_du)``, within that regime the derived
    direction cannot by itself change a tier outcome -- the behavioral change
    against the earlier dominant-eigenvector construction is the ISOTROPIC
    FLOOR it puts under the minor axis, which is what stops a demotable frame
    slipping through on an un-widened perpendicular axis.  The derived
    magnitude does move the major axis once ``||g||`` exceeds 1 (a partially
    covered arc, where the fit overshoots), and there the blend is purely
    directional because the isotropic complement is clamped at zero.

    Parameters:
        covariance: ``(2, 2)`` or ``(3, 3)`` reported covariance.
        sensitivity_g: The ``(2,)`` absorbed-translation sensitivity from
            :func:`_absorbed_orbit_sensitivity` (or a unit radial axis on the
            rank-1 path), in ``(v, u)`` order with ``||g|| <= 1``.
        sigma_orbit_px: The effective orbit sigma in pixels.

    Returns:
        The inflated covariance (a new array).
    """
    out = np.array(covariance, dtype=np.float64, copy=True)
    g = np.asarray(sensitivity_g, np.float64)[:2]
    g_sq = min(float(g @ g), 1.0)
    variance = sigma_orbit_px * sigma_orbit_px
    out[:2, :2] += variance * (np.outer(g, g) + (1.0 - g_sq) * np.eye(2))
    return cast(NDArrayFloatType, out)


def _rank1_projected_covariance(
    covariance: NDArrayFloatType, polarity_normals: NDArrayFloatType
) -> NDArrayFloatType:
    """Project a rank-1 fit's covariance onto its observable (edge-normal) axis.

    The translation block becomes the exactly singular ``sigma_n^2 n n^T``,
    where ``n`` is the aggregate edge-normal orientation and ``sigma_n^2``
    the input covariance's marginal variance along it.  Exact singularity is
    the representation the ensemble is built around: ``pinvh`` drops the
    exact null space when forming the information matrix (keeping the
    normal-axis measurement), the combine's rank-deficiency test fires, the
    fused offset becomes the minimum-norm representative along the edge
    (sliding along a straight edge is a symmetry of the scene), and the
    unobservable axis is reported through the
    ``sigma_along_unobservable_px = inf`` sentinel rather than an inflated
    per-axis sigma that would poison the tier check.

    The normal orientation is the dominant eigenvector of the per-vertex
    normals' outer-product sum, which is polarity-sign-independent — the
    aggregated edges' normals point in opposite senses (a gap's inner and
    outer edges), so a plain mean could cancel.

    For a 3x3 (rotation-fitting) covariance the rotation variance is kept
    and the translation-rotation cross-covariances are zeroed: a
    cross-covariance into an unobservable translation direction carries no
    usable information.

    Parameters:
        covariance: The LM's ``(2, 2)`` or ``(3, 3)`` covariance.
        polarity_normals: ``(N, 2)`` per-vertex edge normals (either sign).

    Returns:
        A new covariance of the input shape whose translation block is
        exactly rank-1.
    """
    n_hat = _aggregate_normal_orientation(polarity_normals)
    sigma_n_sq = float(n_hat @ covariance[:2, :2] @ n_hat)
    projected = np.zeros_like(covariance)
    projected[:2, :2] = sigma_n_sq * np.outer(n_hat, n_hat)
    if covariance.shape == (3, 3):
        projected[2, 2] = covariance[2, 2]
    return projected
