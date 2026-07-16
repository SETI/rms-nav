"""Image-side ring rendering: annulus/gap features between mode-1 edges.

A ring feature is a solid annulus (or a gap) between mode-1 eccentric
precessing edges with edge-fade shading; the optical-depth ring-system
renderer (projection, tau photometry, transparency compositing) is
deliberately not implemented.  Edge placement math
is shared with the navigator's predicted edges through
:mod:`spindoctor.sim.ring_geometry`.
"""

from typing import Any, cast

import numpy as np

from spindoctor.sim.ring_geometry import (
    compute_antialiasing_shade,
    compute_edge_radii_array,
    compute_fade_factor,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['composite_ring', 'render_ring']


def render_ring(
    img: NDArrayFloatType,
    ring_params: dict[str, Any],
    offset_v: float,
    *,
    offset_u: float,
    time: float = 0.0,
    epoch: float = 0.0,
    shade_solid: bool = False,
    resolution: float = 1.0,
) -> None:
    """Render a single ring or gap into the image.

    Parameters:
        img: Image array to modify in-place.
        ring_params: Dictionary containing ring parameters:
            - name: str, ring name
            - feature_type: str, 'RINGLET' or 'GAP'
            - center_v: float, V coordinate of ring center
            - center_u: float, U coordinate of ring center
            - shading_distance: float, distance in pixels for edge fading
            - inner_data: list[dict], mode data for inner edge (mode 1 required)
            - outer_data: list[dict], mode data for outer edge (mode 1 required)
        offset_v: V offset to apply.
        offset_u: U offset to apply.
        time: Current time in TDB seconds (default 0.0).
        epoch: Epoch time in TDB seconds (default 0.0).
        shade_solid: If True, solid rings (with both edges) are shaded on both sides
            as if they were two rings (one with inner edge only, one with outer edge only).
        resolution: Edge anti-aliasing width in pixels of the grid being rendered
            (the oversampling factor when rendering on an oversampled grid), so a
            ring edge spans one detector pixel of transition regardless of the
            render resolution.

    Raises:
        ValueError: If neither an inner nor an outer edge is specified.
    """
    size_v, size_u = img.shape
    feature_type = ring_params.get('feature_type', 'RINGLET')
    center_v = float(ring_params.get('center_v', size_v / 2.0)) + offset_v
    center_u = float(ring_params.get('center_u', size_u / 2.0)) + offset_u

    # Extract mode 1 data for inner and outer edges
    inner_data = ring_params.get('inner_data', [])
    outer_data = ring_params.get('outer_data', [])

    inner_mode1 = next((m for m in inner_data if m.get('mode') == 1), None)
    outer_mode1 = next((m for m in outer_data if m.get('mode') == 1), None)

    # At least one edge must be specified
    if inner_mode1 is None and outer_mode1 is None:
        raise ValueError('At least one edge (inner or outer) must be specified')

    # Create coordinate grids at pixel centers (0.5 offset from integer coordinates)
    v_coords = np.arange(size_v, dtype=np.float64) + 0.5
    u_coords = np.arange(size_u, dtype=np.float64) + 0.5
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')

    # Compute distances from center at pixel centers
    dv = v_grid - center_v
    du = u_grid - center_u
    distances = np.sqrt(dv * dv + du * du)

    # Compute angles
    angles = np.arctan2(dv, du)

    # Get shading distance parameter (default 20.0 pixels)
    shading_distance = float(ring_params.get('shading_distance', 20.0))

    # Initialize model array for this ring
    ring_model = np.zeros((size_v, size_u), dtype=np.float64)

    inner_radii: NDArrayFloatType | None = None
    if inner_mode1 is not None:
        inner_radii = compute_edge_radii_array(
            angles,
            a=float(inner_mode1.get('a', 0.0)),
            ae=float(inner_mode1.get('ae', 0.0)),
            long_peri=float(inner_mode1.get('long_peri', 0.0)),
            rate_peri=float(inner_mode1.get('rate_peri', 0.0)),
            epoch=epoch,
            time=time,
        )

    outer_radii: NDArrayFloatType | None = None
    if outer_mode1 is not None:
        outer_radii = compute_edge_radii_array(
            angles,
            a=float(outer_mode1.get('a', 0.0)),
            ae=float(outer_mode1.get('ae', 0.0)),
            long_peri=float(outer_mode1.get('long_peri', 0.0)),
            rate_peri=float(outer_mode1.get('rate_peri', 0.0)),
            epoch=epoch,
            time=time,
        )

    # Apply anti-aliasing and shading based on edge configuration and feature type
    # Anti-aliasing formula matches base class:
    #   shade = 0.5 + sign * (edge_radius - radii) / resolution
    # When pixel center is at edge (radii == edge_radius), shade = 0.5
    if feature_type == 'RINGLET':
        # For ringlets: fill region between edges (if both), or shade from single edge
        if inner_radii is not None and outer_radii is not None:
            if shade_solid:
                # Both edges with shade_solid: shade on both sides as if two rings
                inner_edge_dist = distances - inner_radii
                inner_shade = compute_antialiasing_shade(inner_edge_dist, resolution)
                inner_fade = compute_fade_factor(inner_edge_dist, shading_distance)

                outer_edge_dist = outer_radii - distances
                outer_shade = compute_antialiasing_shade(outer_edge_dist, resolution)
                outer_fade = compute_fade_factor(outer_edge_dist, shading_distance)
                ring_model = np.maximum(inner_shade * inner_fade, outer_shade * outer_fade)
            else:
                # Both edges: no shading, just fill the entire region with anti-aliasing
                inner_edge_dist = distances - inner_radii
                inner_shade = compute_antialiasing_shade(inner_edge_dist, resolution)
                outer_edge_dist = outer_radii - distances
                outer_shade = compute_antialiasing_shade(outer_edge_dist, resolution)
                # Coverage is minimum (must be inside both edges)
                ring_model = np.minimum(inner_shade, outer_shade)
        elif inner_radii is not None:
            # Only inner edge: shade outward from inner edge
            inner_edge_dist = distances - inner_radii
            inner_shade = compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = compute_fade_factor(inner_edge_dist, shading_distance)
            ring_model = inner_shade * inner_fade
        else:  # outer_radii is not None
            assert outer_radii is not None
            # Only outer edge: shade inward from outer edge
            outer_edge_dist = outer_radii - distances
            outer_shade = compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = compute_fade_factor(outer_edge_dist, shading_distance)
            ring_model = outer_shade * outer_fade
        # Apply ringlet: add brightness where ring exists
        img[:] = np.clip(img + ring_model, 0.0, 1.0)
    else:  # GAP
        # For gaps: shading extends beyond the defined ring area
        gap_model = cast(NDArrayFloatType, np.zeros((size_v, size_u), dtype=np.float64))
        if inner_radii is not None and outer_radii is not None:
            # Both edges: shade inward from inner edge AND outward from outer edge
            inner_edge_dist = inner_radii - distances
            inner_shade = compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = 1 - compute_fade_factor(inner_edge_dist, shading_distance)

            outer_edge_dist = distances - outer_radii
            outer_shade = compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = 1 - compute_fade_factor(outer_edge_dist, shading_distance)
            gap_model = np.maximum(inner_shade * inner_fade, outer_shade * outer_fade)
        elif inner_radii is not None:
            # Only inner edge: shade inward from inner edge (beyond the edge)
            inner_edge_dist = inner_radii - distances
            inner_shade = compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = 1 - compute_fade_factor(inner_edge_dist, shading_distance)
            gap_model = inner_shade * inner_fade
        else:  # outer_radii is not None
            assert outer_radii is not None
            # Only outer edge: shade outward from outer edge (beyond the edge)
            outer_edge_dist = distances - outer_radii
            outer_shade = compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = 1 - compute_fade_factor(outer_edge_dist, shading_distance)
            gap_model = outer_shade * outer_fade
        # Apply gap: subtract brightness where gap shading exists
        img[:] = np.clip(img - gap_model, 0.0, 1.0)


def composite_ring(
    img: NDArrayFloatType,
    ring_params: dict[str, Any],
    offset_v: float,
    *,
    offset_u: float,
    time: float,
    epoch: float,
    shade_solid: bool,
    resolution: float = 1.0,
) -> NDArrayBoolType:
    """Composite one ring feature into the scene image at its z-order slot.

    A RINGLET is rendered into an empty frame to isolate its coverage, then
    painted over the scene (nearer objects painted later overwrite it); a
    GAP is rendered against a saturated background to isolate what it
    erases, then that erasure is painted over the scene.

    Parameters:
        img: Scene signal image, modified in place.
        ring_params: One ring feature's parameter mapping.
        offset_v: V offset to apply.
        offset_u: U offset to apply.
        time: Current time in TDB seconds.
        epoch: Ring epoch in TDB seconds.
        shade_solid: Whether solid ringlets shade from both edges.
        resolution: Edge anti-aliasing width in pixels of the render grid (the
            oversampling factor on an oversampled grid).

    Returns:
        Boolean mask of the pixels this feature painted.
    """
    size_v, size_u = img.shape
    feature_type = ring_params.get('feature_type', 'RINGLET')
    if feature_type == 'RINGLET':
        # For RINGLET: render into empty image to get ring_coverage
        ring_img = np.zeros((size_v, size_u), dtype=np.float64)
        render_ring(
            ring_img,
            ring_params,
            offset_v,
            offset_u=offset_u,
            time=time,
            epoch=epoch,
            shade_solid=shade_solid,
            resolution=resolution,
        )
        # ring_img now contains just the ring_coverage (since 0 + coverage = coverage)
        ring_mask: NDArrayBoolType = ring_img > 0.0
        # Add ring to main image (proper range-based: lower range overwrites)
        img[ring_mask] = ring_img[ring_mask]
        return ring_mask
    # For GAP: render into image with known background to extract coverage
    # Use 1.0 as background so we can see what was subtracted
    temp_bg = np.ones((size_v, size_u), dtype=np.float64)
    render_ring(
        temp_bg,
        ring_params,
        offset_v,
        offset_u=offset_u,
        time=time,
        epoch=epoch,
        shade_solid=shade_solid,
        resolution=resolution,
    )
    # gap_coverage is what was subtracted: 1.0 - result
    gap_mask: NDArrayBoolType = temp_bg < 1.0
    # Subtract gap from main image (proper range-based: lower range overwrites)
    img[gap_mask] = temp_bg[gap_mask]
    return gap_mask
