"""Basin second-opinion: scan for a competing DT-cost minimum.

A converged fit whose cost is rivaled by a distant shift is not unimodal;
the consuming technique treats that as a spurious fit.
"""

import math
from dataclasses import dataclass

import numpy as np

from spindoctor.nav_technique.dt_fitting.constants import (
    DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
)
from spindoctor.support.distance_transform import sample_dt_bilinear
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'SecondaryBasin',
    'find_secondary_dt_minimum',
]


@dataclass(frozen=True)
class SecondaryBasin:
    """A competing DT-cost minimum found away from the converged offset.

    Parameters:
        offset_vu: Integer ``(dv, du)`` shift of the competing minimum.
        distance_px: Euclidean distance from the converged offset.
        cost_px: Mean per-vertex DT value at the competing shift.
        converged_cost_px: Mean per-vertex DT value at the converged offset,
            measured with the same sampler so the two costs are comparable.
    """

    offset_vu: tuple[int, int]
    distance_px: float
    cost_px: float
    converged_cost_px: float


def _mean_dt_cost(
    image_edge_dt: NDArrayFloatType,
    vertices_vu: NDArrayFloatType,
    offset_vu: tuple[float, float],
    *,
    min_support: float,
) -> float | None:
    """Mean DT value over the in-bounds vertices at ``offset_vu``.

    Parameters:
        image_edge_dt: 2-D edge distance transform.
        vertices_vu: ``(N, 2)`` polyline vertex positions in ``(v, u)``.
        offset_vu: ``(dv, du)`` shift applied to every vertex.
        min_support: Minimum number of in-bounds vertices for the cost to
            be trustworthy.

    Returns:
        The mean DT value, or None when fewer than ``min_support`` vertices
        remain in bounds.
    """
    height, width = image_edge_dt.shape
    shifted = vertices_vu + np.asarray(offset_vu, np.float64)
    in_bounds = (
        (shifted[:, 0] >= 0.0)
        & (shifted[:, 0] <= height - 1.0)
        & (shifted[:, 1] >= 0.0)
        & (shifted[:, 1] <= width - 1.0)
    )
    if float(np.count_nonzero(in_bounds)) < min_support:
        return None
    values = sample_dt_bilinear(image_edge_dt, shifted[in_bounds])
    return float(values.mean())


def find_secondary_dt_minimum(
    image_edge_dt: NDArrayFloatType,
    vertices_vu: NDArrayFloatType,
    *,
    converged_offset_vu: tuple[float, float],
    search_window_vu: tuple[int, int],
    exclude_radius_px: float,
    min_support_fraction: float = DEFAULT_COARSE_MIN_SUPPORT_FRACTION,
) -> SecondaryBasin | None:
    """Search the offset window for a competing DT-cost basin.

    Samples the mean per-vertex DT cost of the polyline at every integer
    shift in the search window and returns the best-scoring shift farther
    than ``exclude_radius_px`` from the converged offset.  A converged LM
    fit whose cost is rivaled by a distant shift is not unimodal: the DT
    surface offers a second plausible alignment (a crater shadow, an albedo
    boundary, a second body's edge), and the fit deserves no confidence in
    having picked the right one.

    Parameters:
        image_edge_dt: 2-D edge distance transform the LM fit ran against.
        vertices_vu: ``(N, 2)`` polyline vertex positions in ``(v, u)``,
            unshifted. A caller that fitted rotation must pass the vertices
            already rotated by the converged angle (the geometry the LM
            evaluated), or the converged cost is inflated relative to the
            actual fit.
        converged_offset_vu: The LM fit's converged ``(dv, du)``.
        search_window_vu: ``(margin_v, margin_u)`` non-negative integers
            bounding the scan, normally the technique's coarse search
            window.
        exclude_radius_px: Shifts within this Euclidean distance of the
            converged offset belong to the converged basin and are not
            candidates.
        min_support_fraction: Minimum fraction of vertices that must stay
            in bounds for a shift to be scored.

    Returns:
        The best competing basin, or None when the polyline is empty, the
        converged cost is unmeasurable, or no eligible shift exists.
    """
    n_vertices = int(vertices_vu.shape[0])
    if n_vertices == 0:
        return None
    min_support = min_support_fraction * float(n_vertices)
    converged_cost = _mean_dt_cost(
        image_edge_dt, vertices_vu, converged_offset_vu, min_support=min_support
    )
    if converged_cost is None:
        return None
    margin_v, margin_u = int(search_window_vu[0]), int(search_window_vu[1])
    conv_dv, conv_du = float(converged_offset_vu[0]), float(converged_offset_vu[1])
    best: SecondaryBasin | None = None
    for dv in range(-margin_v, margin_v + 1):
        for du in range(-margin_u, margin_u + 1):
            distance = math.hypot(dv - conv_dv, du - conv_du)
            if distance <= exclude_radius_px:
                continue
            cost = _mean_dt_cost(
                image_edge_dt, vertices_vu, (float(dv), float(du)), min_support=min_support
            )
            if cost is None:
                continue
            # Ties break toward the nearest rival so the reported distance is
            # deterministic on cost plateaus (a DT constant along one axis).
            if best is None or (cost, distance) < (best.cost_px, best.distance_px):
                best = SecondaryBasin(
                    offset_vu=(dv, du),
                    distance_px=distance,
                    cost_px=cost,
                    converged_cost_px=converged_cost,
                )
    return best
