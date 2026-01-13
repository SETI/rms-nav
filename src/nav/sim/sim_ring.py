"""Simulated ring rendering for navigation testing.

This module provides functions to render planetary rings in simulated images
for navigation testing. Rings are rendered as circular or elliptical features
with anti-aliased edges.
"""

import math
from typing import Any

import numpy as np


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

    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angle - current_long_peri

    # Compute radius using elliptical orbit equation: r = a(1 - e^2) / (1 + e*cos(ν))
    # where e = ae / a
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        e = 0.99  # Clamp eccentricity to valid range
    r = a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))

    return r


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

    # Compute radius using elliptical orbit equation: r = a(1 - e^2) / (1 + e*cos(ν))
    # where e = ae / a
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        e = 0.99  # Clamp eccentricity to valid range
    r = a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))

    return r


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
) -> np.ndarray:
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
    # Create coordinate grids
    v_coords = np.arange(size_v, dtype=np.float64)
    u_coords = np.arange(size_u, dtype=np.float64)
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')

    # Compute distances from center
    dv = v_grid - center_v
    du = u_grid - center_u
    distances = np.sqrt(dv * dv + du * du)

    # Compute angles
    angles = np.arctan2(dv, du)

    # Compute edge radius at each angle using elliptical orbit equation
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        e = 0.99

    days_since_epoch = (time - epoch) / 86400.0
    long_peri_rad = math.radians(long_peri + rate_peri * days_since_epoch)
    true_anomaly = angles - long_peri_rad
    edge_radii = a * (1.0 - e * e) / (1.0 + e * np.cos(true_anomaly))

    # Compute difference from target edge radius
    # Use the computed edge radius at each angle, not the constant edge_radius
    # For border_atop, we want pixels where distance transitions across edge_radii
    diff = distances - edge_radii
    sign = np.sign(diff)
    abs_diff = np.abs(diff)

    # Initialize border mask (pixels exactly at edge)
    border = (abs_diff == 0.0)

    # Find transitions: pixels where sign changes between neighbors
    # Check vertical neighbors
    sign_v = sign[:-1, :]
    sign_v_next = sign[1:, :]
    abs_diff_v = abs_diff[:-1, :]
    abs_diff_v_next = abs_diff[1:, :]

    # Pixels where sign flips and current pixel is closer to edge
    border[:-1, :] |= ((sign_v == -sign_v_next) & (abs_diff_v <= abs_diff_v_next))
    border[1:, :] |= ((sign_v_next == -sign_v) & (abs_diff_v_next <= abs_diff_v))

    # Check horizontal neighbors
    sign_u = sign[:, :-1]
    sign_u_next = sign[:, 1:]
    abs_diff_u = abs_diff[:, :-1]
    abs_diff_u_next = abs_diff[:, 1:]

    # Pixels where sign flips and current pixel is closer to edge
    border[:, :-1] |= ((sign_u == -sign_u_next) & (abs_diff_u <= abs_diff_u_next))
    border[:, 1:] |= ((sign_u_next == -sign_u) & (abs_diff_u_next <= abs_diff_u))

    return border


def render_ring(
    img: np.ndarray,
    ring_params: dict[str, Any],
    offset_v: float,
    offset_u: float,
    *,
    time: float = 0.0,
    epoch: float = 0.0,
) -> None:
    """Render a single ring or gap into the image.

    Parameters:
        img: Image array to modify in-place.
        ring_params: Dictionary containing ring parameters:
            - name: str, ring name
            - feature_type: str, 'RINGLET' or 'GAP'
            - center_v: float, V coordinate of ring center
            - center_u: float, U coordinate of ring center
            - inner_data: list[dict], mode data for inner edge (mode 1 required)
            - outer_data: list[dict], mode data for outer edge (mode 1 required)
        offset_v: V offset to apply.
        offset_u: U offset to apply.
        time: Current time in TDB seconds (default 0.0).
        epoch: Epoch time in TDB seconds (default 0.0).
    """
    size_v, size_u = img.shape
    feature_type = ring_params.get('feature_type', 'RINGLET')
    center_v = float(ring_params.get('center_v', size_v / 2.0)) + offset_v
    center_u = float(ring_params.get('center_u', size_u / 2.0)) + offset_u

    # Extract mode 1 data for inner and outer edges
    inner_data = ring_params.get('inner_data', [])
    outer_data = ring_params.get('outer_data', [])

    # Find mode 1 for inner edge
    inner_mode1 = None
    for mode in inner_data:
        if mode.get('mode') == 1:
            inner_mode1 = mode
            break

    # Find mode 1 for outer edge
    outer_mode1 = None
    for mode in outer_data:
        if mode.get('mode') == 1:
            outer_mode1 = mode
            break

    if inner_mode1 is None or outer_mode1 is None:
        return  # Skip if mode 1 not found

    # Extract mode 1 parameters
    inner_a = float(inner_mode1.get('a', 0.0))
    inner_ae = float(inner_mode1.get('ae', 0.0))
    inner_long_peri = float(inner_mode1.get('long_peri', 0.0))
    inner_rate_peri = float(inner_mode1.get('rate_peri', 0.0))

    outer_a = float(outer_mode1.get('a', 0.0))
    outer_ae = float(outer_mode1.get('ae', 0.0))
    outer_long_peri = float(outer_mode1.get('long_peri', 0.0))
    outer_rate_peri = float(outer_mode1.get('rate_peri', 0.0))

    # Create coordinate grids
    v_coords = np.arange(size_v, dtype=np.float64)
    u_coords = np.arange(size_u, dtype=np.float64)
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')

    # Compute distances from center
    dv = v_grid - center_v
    du = u_grid - center_u
    distances = np.sqrt(dv * dv + du * du)

    # Compute angles
    angles = np.arctan2(dv, du)

    # Compute inner and outer edge radii at each angle
    # For simplicity, use circular approximation (ae=0) or basic elliptical
    inner_e = inner_ae / inner_a if inner_a > 0 else 0.0
    if inner_e >= 1.0:
        inner_e = 0.99
    outer_e = outer_ae / outer_a if outer_a > 0 else 0.0
    if outer_e >= 1.0:
        outer_e = 0.99

    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    inner_long_peri_rad = math.radians(inner_long_peri + inner_rate_peri * days_since_epoch)
    outer_long_peri_rad = math.radians(outer_long_peri + outer_rate_peri * days_since_epoch)

    # Compute true anomaly for each pixel
    inner_true_anomaly = angles - inner_long_peri_rad
    outer_true_anomaly = angles - outer_long_peri_rad

    # Compute edge radii using elliptical orbit equation
    inner_radii = (inner_a * (1.0 - inner_e * inner_e) /
                   (1.0 + inner_e * np.cos(inner_true_anomaly)))
    outer_radii = (outer_a * (1.0 - outer_e * outer_e) /
                   (1.0 + outer_e * np.cos(outer_true_anomaly)))

    # Apply anti-aliasing at edges
    # Compute fractional coverage for pixels near edges
    inner_edge_dist = distances - inner_radii
    outer_edge_dist = distances - outer_radii

    # Use 1 pixel width for anti-aliasing
    aa_width = 1.0

    # For inner edge: fade from 0 (outside, distance < inner_radius) to 1 (inside)
    # Pixel is fully inside if distance >= inner_radius + aa_width
    inner_fade = np.clip((inner_edge_dist + aa_width) / aa_width, 0.0, 1.0)

    # For outer edge: fade from 1 (inside, distance < outer_radius) to 0 (outside)
    # Pixel is fully outside if distance >= outer_radius + aa_width
    outer_fade = np.clip(1.0 - outer_edge_dist / aa_width, 0.0, 1.0)

    # Ring coverage is minimum of the two fades (must be inside both edges)
    ring_coverage = np.minimum(inner_fade, outer_fade)

    # Apply ring or gap
    if feature_type == 'RINGLET':
        # Ringlet: add brightness where ring exists
        img[:] = np.clip(img + ring_coverage, 0.0, 1.0)
    else:  # GAP
        # Gap: subtract brightness where gap exists
        img[:] = np.clip(img - ring_coverage, 0.0, 1.0)
