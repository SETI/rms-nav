"""Per-edge fit statistics for the ring-edge technique.

The residual and inlier summaries the spurious gates and the absent-edge
waiver consume, split out of
:mod:`spindoctor.nav_technique.nav_technique_ring_edge` to keep both modules
under the size cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.support.types import NDArrayFloatType

__all__: list[str] = []


def _per_edge_rms_summed(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Sum each consumed edge's UNWEIGHTED RMS DT residual.

    Each edge contributes ``sqrt(mean(r**2))`` over its own slice of the
    per-vertex residual array, and the per-edge values are summed.  No robust
    weighting is applied, which is the point: the LM's own ``rms_px`` is
    Tukey-WEIGHTED and collapses to near zero on exactly the mis-convergence
    this statistic exists to surface (the fit locks onto one edge, Tukey
    rejects the wholly mis-aligned others, and the weighted RMS reports a
    clean fit).  Passing the raw ``LMRefineResult.residuals_px`` keeps those
    rejected vertices in the number.

    The sum grows with the number of fused edges, so the confidence formula
    consumes the edge-count-independent mean derived from it rather than this
    value directly.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex raw DT residuals from the fit, in the same
            concatenation order.

    Returns:
        The summed per-edge RMS in pixels; ``0.0`` when no consumed edge has
        vertices.
    """
    total = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        if slice_residuals.size == 0:
            continue
        rms = float(np.sqrt(np.mean(slice_residuals * slice_residuals)))
        total += rms
    return total


@dataclass(frozen=True)
class _EdgeFitStat:
    """Per-edge fit statistics for the spurious-veto waiver.

    Parameters:
        inlier_count: Vertices of this edge with a strictly positive final
            weight (prior precision times Tukey biweight) -- the same
            criterion ``LMRefineResult.inlier_count`` applies to the
            aggregated vertex set.
        vertex_count: Total vertices of this edge.
        median_abs_residual_px: Median absolute DT residual of this edge's
            vertices at the converged offset.  Small when the edge lies
            along a detected image edge (whether or not the robust fit
            kept it); tens of pixels when no image edge exists near it.
        is_straight: The edge's ``is_straight_line`` flag; a straight edge
            constrains only its normal axis (rank-1).
    """

    inlier_count: int
    vertex_count: int
    median_abs_residual_px: float
    is_straight: bool

    @property
    def inlier_fraction(self) -> float:
        """Fraction of this edge's vertices retained as Tukey inliers."""
        return float(self.inlier_count) / float(self.vertex_count)


def _per_edge_fit_stats(
    features: list[NavFeature],
    *,
    residuals: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> list[_EdgeFitStat]:
    """Return per-edge fit statistics from the final LM residuals and weights.

    Splitting the aggregate fit per edge lets the spurious gate
    distinguish a fusion whose vertices are rejected because some edges
    are undetectable in the image (large per-edge median residuals --
    nothing is there) from a wrong-ring lock that leaves a rejected edge
    sitting on a detected image edge it disagrees with.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex raw DT residuals from the fit, in the same
            concatenation order.
        weights: Per-vertex final weights from the fit, in the same
            concatenation order.

    Returns:
        One :class:`_EdgeFitStat` per consumed non-empty edge, in
        concatenation order.
    """
    stats: list[_EdgeFitStat] = []
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_weights = weights[cursor : cursor + n]
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        stats.append(
            _EdgeFitStat(
                inlier_count=int(np.count_nonzero(slice_weights > 0.0)),
                vertex_count=n,
                median_abs_residual_px=float(np.median(np.abs(slice_residuals))),
                is_straight=bool(feat.geometry.is_straight_line),
            )
        )
    return stats


def _per_edge_median_max(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Return the largest per-edge median absolute DT residual across edges.

    A mis-convergence *diagnostic* (surfaced through
    :class:`RingEdgeDiagnostics`, not a spurious gate): a wholly misaligned
    or undetected edge puts most of its vertices roughly a ringlet spacing
    from the nearest image edge, driving its median to that spacing, while
    a well-matched edge's median sits at the fit residual.  Taking the max
    over edges keeps one bad edge visible among any number of clean ones.
    Residuals alone cannot separate "misaligned" from "undetectable in the
    image", which is why the spurious decision uses the inlier fraction
    instead.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex signed DT residuals from the fit, in the same
            concatenation order.

    Returns:
        ``max_e median(|residuals_e|)`` over the consumed edges, or ``0.0``
        when no edge has vertices.
    """
    worst = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        if slice_residuals.size == 0:
            continue
        worst = max(worst, float(np.median(np.abs(slice_residuals))))
    return worst
