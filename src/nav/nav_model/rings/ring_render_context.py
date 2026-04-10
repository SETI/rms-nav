"""Immutable rendering context for ring feature backplane rendering.

This module defines ``RingsRenderContext``, a frozen dataclass that bundles
all dependencies needed to render one ring feature. Passing a single context
object instead of many individual parameters achieves two goals:

1. **Clean method signatures**: ``RingFeature.render(context)`` takes one
   argument instead of six. Adding a new rendering parameter only requires
   updating ``RingsRenderContext``, not every call site.

2. **Immutability contract**: Because the context is frozen, each call to
   ``render()`` receives the same data. Features cannot accidentally modify
   shared rendering state.

``RingsRenderContext`` carries ``all_edge_radii`` -- the sorted sequence of
(radius, label) pairs for all features that survived filtering. This is needed
at render time by ``compute_edge_fade`` to reduce the fade width when a
neighboring edge falls within the fade zone (halving the fade at the conflict
boundary). The filter has already handled *exclusion* (edges whose adjusted
fade would be too narrow); ``compute_edge_fade`` handles *reduction* (edges
whose adjusted fade is still acceptable but narrower than the requested
``fade_width_pix``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nav.support.types import NDArrayFloatType

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RingsRenderContext:
    """Immutable context for backplane-based ring feature rendering.

    Constructed by the orchestrator (``NavModelRings``) once per observation
    and passed unchanged to every ``RingFeature.render()`` call. Contains
    observation data, computed backplane arrays, fade configuration, and the
    sorted list of all surviving edge radii for conflict-based fade reduction.

    The ``all_edge_radii`` tuple is built from features that survived all four
    filter passes. It is used by ``compute_edge_fade`` to reduce fade width
    when a neighboring feature's edge is within the fade zone, preserving the
    current behavior of halving the fade extent at a conflict boundary rather
    than rendering with full width. This is a *width reduction*, not exclusion
    -- exclusion is handled by ``RingFeatureFilter`` before rendering.

    Parameters:
        obs: The observation object (``oops.Observation``). Provides access to
            all backplane computation methods.
        ring_target: Ring target string used for backplane calls, e.g.
            ``'saturn:ring'``.
        epoch: TDB epoch time in seconds used as the reference time for
            multi-mode orbital perturbation calculations.
        resolutions: 2-D array of per-pixel radial resolution in km/pixel.
            Shape matches the extended FOV. Used to compute per-pixel fade
            widths: ``fade_width_km = fade_width_pix * resolutions``.
        fade_width_pix: Fade extent in pixels as configured in the YAML
            (``fade_width_pix`` key). A scalar; per-pixel km extent is computed
            at render time from this value and ``resolutions``.
        all_edge_radii: Sorted tuple of ``(radius_km, edge_label)`` pairs for
            all edges of all features that survived filtering. Used by
            ``compute_edge_fade`` for conflict detection and width reduction.
    """

    obs: Any  # oops.Observation; typed as Any to avoid oops import at module level
    ring_target: str
    epoch: float
    resolutions: NDArrayFloatType
    fade_width_pix: float
    all_edge_radii: tuple[tuple[float, str], ...]
