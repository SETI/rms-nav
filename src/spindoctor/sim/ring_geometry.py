"""Shared ring-edge geometry for the simulator's two sides.

The image-side forward ring renderer (``spindoctor.sim.forward.ring``) and
the navigator-side predicted-ring renderer (``spindoctor.nav_model.sim_ring``
plus ``spindoctor.nav_model.nav_model_rings_simulated``) must place a mode-1
elliptical edge at byte-identical pixel positions: the same true-anomaly
convention, the same pericenter precession, and the same pixel-center
rasterisation.  With shared conventions the planted scene error is the only
error in a recovery measurement.

Every function takes explicit geometry arguments; none reads a scene
parameter mapping, so no truth-side information can cross the information
boundary here (see ``spindoctor.sim.scene``).
"""

import math
from typing import cast

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'compute_antialiasing_shade',
    'compute_border_atop_simulated',
    'compute_edge_radii_array',
    'compute_edge_radius_at_angle',
    'compute_edge_radius_mode1',
    'compute_fade_factor',
]


def _edge_eccentricity(*, a: float, ae: float) -> float:
    """Eccentricity of a mode 1 ring edge, validated against the ellipse limit.

    Parameters:
        a: Semi-major axis in pixels.
        ae: Eccentricity times semi-major axis in pixels.

    Returns:
        The eccentricity ``ae / a`` (0.0 when ``a`` is not positive).

    Raises:
        ValueError: If the eccentricity is 1 or more, which does not describe
            a closed elliptical edge.
    """
    e = ae / a if a > 0 else 0.0
    if e >= 1.0:
        raise ValueError(
            f'Ring edge eccentricity e = ae/a = {ae}/{a} = {e} is physically '
            f'impossible; a closed elliptical edge requires e < 1'
        )
    return e


def compute_edge_radius_mode1(
    center_v: float,
    center_u: float,
    pixel_v: float,
    *,
    pixel_u: float,
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
        angle,
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

    Raises:
        ValueError: If ``ae / a`` is an eccentricity of 1 or more, which does
            not describe a closed elliptical edge.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angle - current_long_peri

    # Compute radius using elliptical orbit equation
    e = _edge_eccentricity(a=a, ae=ae)
    r = a * (1.0 - e * e) / (1.0 + e * math.cos(true_anomaly))

    return r


def compute_edge_radii_array(
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

    Raises:
        ValueError: If ``ae / a`` is an eccentricity of 1 or more, which does
            not describe a closed elliptical edge.
    """
    # Compute current longitude of pericenter
    days_since_epoch = (time - epoch) / 86400.0
    current_long_peri = math.radians(long_peri + rate_peri * days_since_epoch)

    # Compute true anomaly (angle relative to pericenter)
    true_anomaly = angles - current_long_peri

    # Compute radius using elliptical orbit equation: r = a(1 - e^2) / (1 + e*cos(v))
    # where e = ae / a
    e = _edge_eccentricity(a=a, ae=ae)
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(true_anomaly))

    return cast(NDArrayFloatType, r)


def compute_border_atop_simulated(
    size_v: int,
    size_u: int,
    center_v: float,
    *,
    center_u: float,
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
    edge_radii = compute_edge_radii_array(
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


def compute_antialiasing_shade(edge_dist: NDArrayFloatType, resolution: float) -> NDArrayFloatType:
    """Compute anti-aliasing shade from edge distance.

    Parameters:
        edge_dist: Distance from pixel center to edge (positive = outside,
            negative = inside).
        resolution: Pixel resolution for anti-aliasing.

    Returns:
        Anti-aliasing shade value [0, 1] where 0.5 means pixel center is at edge.
    """
    shade = 0.5 + edge_dist / resolution
    shade[shade < 0.0] = 0.0
    shade[shade > 1.0] = 1.0
    return shade


def compute_fade_factor(edge_dist: NDArrayFloatType, shading_distance: float) -> NDArrayFloatType:
    """Compute fade factor for edge shading.

    Parameters:
        edge_dist: Distance from pixel center to edge (positive = outside,
            negative = inside).
        shading_distance: Distance in pixels for edge fading.

    Returns:
        Fade factor [0, 1] where 1.0 is at the edge and 0.0 is at
        shading_distance away.
    """
    fade_dist = np.maximum(0.0, edge_dist)
    if shading_distance <= 0.0:
        # Step-function fade: 1.0 for edge_dist <= 0, else 0.0
        return cast(NDArrayFloatType, (edge_dist <= 0.0).astype(np.float64))
    return cast(NDArrayFloatType, np.clip(1.0 - fade_dist / shading_distance, 0.0, 1.0))
