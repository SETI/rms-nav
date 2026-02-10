"""Simulated ring rendering for navigation testing.

This module provides functions to render planetary rings in simulated images
for navigation testing. Rings are rendered as circular or elliptical features
with anti-aliased edges.
"""

import math
from typing import Any, cast

import numpy as np

from nav.support.types import NDArrayBoolType, NDArrayFloatType


def compute_edge_radius_mode1(
    center_v: float,
    center_u: float,
    pixel_v: float,
    pixel_u: float,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> float:
    """Compute edge radius from mode 1 parameters at a specific pixel.

    Parameters:
        center_v: V coordinate of ring center.
        center_u: U coordinate of ring center.
        pixel_v: V coordinate of pixel.
        pixel_u: U coordinate of pixel.
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Edge radius in pixels at the given pixel position.
    """
    # Compute angle from center to pixel
    dv = pixel_v - center_v
    du = pixel_u - center_u
    angle = math.atan2(dv, du)

    # Delegate to compute_edge_radius_at_angle which contains the shared ellipse formula
    return compute_edge_radius_at_angle(
        angle=angle,
        a=a,
        ae=ae,
        long_peri=long_peri,
        rate_peri=rate_peri,
        epoch=epoch,
        time=time,
    )


def compute_edge_radius_at_angle(
    angle: float,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> float:
    """Compute edge radius at a specific angle using mode 1 parameters.

    Parameters:
        angle: Angle in radians from center.
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Edge radius in pixels at the given angle.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angle - current_long_peri

    # Compute radius using elliptical orbit equation
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        e = 0.99  # Clamp eccentricity to valid range
    r = a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))

    return r


def _compute_edge_radii_array(
    angles: NDArrayFloatType,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> NDArrayFloatType:
    """Compute edge radii array for all angles using mode 1 parameters.

    Parameters:
        angles: Array of angles in radians from center.
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Array of edge radii in pixels at the given angles.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angles - current_long_peri

    # Compute radius using elliptical orbit equation: r = a(1 - e^2) / (1 + e*cos(ν))
    # where e = ae / a
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        e = 0.99  # Clamp eccentricity to valid range
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(true_anomaly))

    return cast(NDArrayFloatType, r)


def compute_border_atop_simulated(
    size_v: int,
    size_u: int,
    center_v: float,
    center_u: float,
    *,
    a: float,
    ae: float,
    long_peri: float,
    rate_peri: float,
    epoch: float,
    time: float,
) -> NDArrayBoolType:
    """Compute border_atop mask for simulated ring edge.

    This simulates the border_atop backplane function for simulated rings by
    finding pixels where the distance from center transitions across the edge
    radius computed from mode 1 parameters.

    Parameters:
        size_v: Image height in pixels.
        size_u: Image width in pixels.
        center_v: V coordinate of ring center.
        center_u: U coordinate of ring center.
        a: Semi-major axis in pixels (mode 1 'a' value).
        ae: Eccentricity times semi-major axis in pixels.
        long_peri: Longitude of pericenter in degrees.
        rate_peri: Rate of precession in degrees/day.
        epoch: Epoch time (TDB seconds).
        time: Current time (TDB seconds).

    Returns:
        Boolean array where True indicates pixels at the edge.
    """
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

    # Compute edge radius at each angle using elliptical orbit equation
    edge_radii = _compute_edge_radii_array(
        angles,
        a=a,
        ae=ae,
        long_peri=long_peri,
        rate_peri=rate_peri,
        epoch=epoch,
        time=time,
    )

    # Compute difference from target edge radius
    # Use the computed edge radius at each angle, not the constant edge_radius
    # For border_atop, we want pixels where distance transitions across edge_radii
    diff = distances - edge_radii
    sign = np.sign(diff)
    abs_diff = np.abs(diff)

    # Initialize border mask (pixels exactly at edge)
    border = abs_diff == 0.0

    # Find transitions: pixels where sign changes between neighbors
    # Check vertical neighbors
    sign_v = sign[:-1, :]
    sign_v_next = sign[1:, :]
    abs_diff_v = abs_diff[:-1, :]
    abs_diff_v_next = abs_diff[1:, :]

    # Pixels where sign flips and current pixel is closer to edge
    border[:-1, :] |= (sign_v == -sign_v_next) & (abs_diff_v <= abs_diff_v_next)
    border[1:, :] |= (sign_v_next == -sign_v) & (abs_diff_v_next <= abs_diff_v)

    # Check horizontal neighbors
    sign_u = sign[:, :-1]
    sign_u_next = sign[:, 1:]
    abs_diff_u = abs_diff[:, :-1]
    abs_diff_u_next = abs_diff[:, 1:]

    # Pixels where sign flips and current pixel is closer to edge
    border[:, :-1] |= (sign_u == -sign_u_next) & (abs_diff_u <= abs_diff_u_next)
    border[:, 1:] |= (sign_u_next == -sign_u) & (abs_diff_u_next <= abs_diff_u)

    return cast(NDArrayBoolType, border)


def _compute_antialiasing_shade(edge_dist: NDArrayFloatType, resolution: float) -> NDArrayFloatType:
    """Compute anti-aliasing shade from edge distance.

    Parameters:
        edge_dist: Distance from pixel center to edge (positive = outside, negative = inside).
        resolution: Pixel resolution for anti-aliasing.

    Returns:
        Anti-aliasing shade value [0, 1] where 0.5 means pixel center is at edge.
    """
    shade = 0.5 + edge_dist / resolution
    shade[shade < 0.0] = 0.0
    shade[shade > 1.0] = 1.0
    return shade


def _compute_fade_factor(edge_dist: NDArrayFloatType, shading_distance: float) -> NDArrayFloatType:
    """Compute fade factor for edge shading.

    Parameters:
        edge_dist: Distance from pixel center to edge (positive = outside, negative = inside).
        shading_distance: Distance in pixels for edge fading.

    Returns:
        Fade factor [0, 1] where 1.0 is at the edge and 0.0 is at shading_distance away.
    """
    fade_dist = np.maximum(0.0, edge_dist)
    if shading_distance <= 0.0:
        # Step-function fade: 1.0 for edge_dist <= 0, else 0.0
        return cast(NDArrayFloatType, (edge_dist <= 0.0).astype(np.float64))
    return cast(NDArrayFloatType, np.clip(1.0 - fade_dist / shading_distance, 0.0, 1.0))


def render_ring(
    img: NDArrayFloatType,
    ring_params: dict[str, Any],
    offset_v: float,
    offset_u: float,
    *,
    time: float = 0.0,
    epoch: float = 0.0,
    shade_solid: bool = False,
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

    # Extract mode 1 parameters (use defaults if not present)
    inner_a = float(inner_mode1.get('a', 0.0)) if inner_mode1 is not None else 0.0
    inner_ae = float(inner_mode1.get('ae', 0.0)) if inner_mode1 is not None else 0.0
    inner_long_peri = float(inner_mode1.get('long_peri', 0.0)) if inner_mode1 is not None else 0.0
    inner_rate_peri = float(inner_mode1.get('rate_peri', 0.0)) if inner_mode1 is not None else 0.0

    outer_a = float(outer_mode1.get('a', 0.0)) if outer_mode1 is not None else 0.0
    outer_ae = float(outer_mode1.get('ae', 0.0)) if outer_mode1 is not None else 0.0
    outer_long_peri = float(outer_mode1.get('long_peri', 0.0)) if outer_mode1 is not None else 0.0
    outer_rate_peri = float(outer_mode1.get('rate_peri', 0.0)) if outer_mode1 is not None else 0.0

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

    # Compute edge radii at each angle
    resolution = 1.0  # Pixel resolution for anti-aliasing

    # Get shading distance parameter (default 20.0 pixels)
    shading_distance = float(ring_params.get('shading_distance', 20.0))

    # Initialize model array for this ring
    ring_model = np.zeros((size_v, size_u), dtype=np.float64)

    # Compute inner edge radii if inner edge is specified
    if inner_mode1 is not None:
        inner_radii = _compute_edge_radii_array(
            angles,
            a=inner_a,
            ae=inner_ae,
            long_peri=inner_long_peri,
            rate_peri=inner_rate_peri,
            epoch=epoch,
            time=time,
        )
    else:
        inner_radii = None

    # Compute outer edge radii if outer edge is specified
    if outer_mode1 is not None:
        outer_radii = _compute_edge_radii_array(
            angles,
            a=outer_a,
            ae=outer_ae,
            long_peri=outer_long_peri,
            rate_peri=outer_rate_peri,
            epoch=epoch,
            time=time,
        )
    else:
        outer_radii = None

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
                inner_shade = _compute_antialiasing_shade(inner_edge_dist, resolution)
                inner_fade = _compute_fade_factor(inner_edge_dist, shading_distance)

                outer_edge_dist = outer_radii - distances
                outer_shade = _compute_antialiasing_shade(outer_edge_dist, resolution)
                outer_fade = _compute_fade_factor(outer_edge_dist, shading_distance)
                ring_model = np.maximum(inner_shade * inner_fade, outer_shade * outer_fade)
            else:
                # Both edges: no shading, just fill the entire region with anti-aliasing
                inner_edge_dist = distances - inner_radii
                inner_shade = _compute_antialiasing_shade(inner_edge_dist, resolution)
                outer_edge_dist = outer_radii - distances
                outer_shade = _compute_antialiasing_shade(outer_edge_dist, resolution)
                # Coverage is minimum (must be inside both edges)
                ring_model = np.minimum(inner_shade, outer_shade)
        elif inner_radii is not None:
            # Only inner edge: shade outward from inner edge
            inner_edge_dist = distances - inner_radii
            inner_shade = _compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = _compute_fade_factor(inner_edge_dist, shading_distance)
            ring_model = inner_shade * inner_fade
        else:  # outer_radii is not None
            # Only outer edge: shade inward from outer edge
            outer_edge_dist = outer_radii - distances
            outer_shade = _compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = _compute_fade_factor(outer_edge_dist, shading_distance)
            ring_model = outer_shade * outer_fade
        # Apply ringlet: add brightness where ring exists
        img[:] = np.clip(img + ring_model, 0.0, 1.0)
    else:  # GAP
        # For gaps: shading extends beyond the defined ring area
        gap_model = cast(NDArrayFloatType, np.zeros((size_v, size_u), dtype=np.float64))
        if inner_radii is not None and outer_radii is not None:
            # Both edges: shade inward from inner edge AND outward from outer edge
            inner_edge_dist = inner_radii - distances
            inner_shade = _compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = 1 - _compute_fade_factor(inner_edge_dist, shading_distance)

            outer_edge_dist = distances - outer_radii
            outer_shade = _compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = 1 - _compute_fade_factor(outer_edge_dist, shading_distance)
            gap_model = np.maximum(inner_shade * inner_fade, outer_shade * outer_fade)
        elif inner_radii is not None:
            # Only inner edge: shade inward from inner edge (beyond the edge)
            inner_edge_dist = inner_radii - distances
            inner_shade = _compute_antialiasing_shade(inner_edge_dist, resolution)
            inner_fade = 1 - _compute_fade_factor(inner_edge_dist, shading_distance)
            gap_model = inner_shade * inner_fade
        else:  # outer_radii is not None
            # Only outer edge: shade outward from outer edge (beyond the edge)
            outer_edge_dist = distances - outer_radii
            outer_shade = _compute_antialiasing_shade(outer_edge_dist, resolution)
            outer_fade = 1 - _compute_fade_factor(outer_edge_dist, shading_distance)
            gap_model = outer_shade * outer_fade
        # Apply gap: subtract brightness where gap shading exists
        img[:] = np.clip(img - gap_model, 0.0, 1.0)
