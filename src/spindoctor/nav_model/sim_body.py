"""The navigator's predicted-body renderer for simulated scenes.

Renders the smooth Lambert ellipsoid ``NavModelBodySimulated`` predicts
from a scene's idealized body geometry.  This is deliberately the
navigator's *best model*, not the image: surface texture (craters, relief)
is truth-side information rendered only by the image-side twin
(``spindoctor.sim.forward.body``), and the difference between the two is
exactly the model error a scene plants.  Shading conventions are shared
with the image side through :mod:`spindoctor.sim.ellipsoid_geometry`, so
that planted difference is the only difference.
"""

from typing import cast

import numpy as np

from spindoctor.sim.ellipsoid_geometry import ellipsoid_image_normals, lambert_from_normals
from spindoctor.support.types import NDArrayFloatType

__all__ = ['create_simulated_body']


def create_simulated_body(
    size: tuple[int, int],
    center: tuple[float, float],
    axis1: float,
    *,
    axis2: float,
    axis3: float,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    anti_aliasing: float = 0.0,
) -> NDArrayFloatType:
    """Render the predicted body: a smooth ellipsoid with Lambertian shading.

    The body is modeled as a 3D ellipsoid projected onto 2D and illuminated
    using Lambertian shading (cos(incidence)) based on the illumination
    direction and phase angle.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point pixels.
            (0.0, 0.0) is the top-left corner of pixel (0,0), (0.5, 0.5) is the
            center of pixel (0,0).
        axis1: The full width of axis 1 (a) of the ellipsoid in pixels.
        axis2: The full width of axis 2 (b) of the ellipsoid in pixels.
        axis3: The full width of axis 3 (c) of the ellipsoid in pixels (depth).
        rotation_z: Rotation angle around the viewing axis (z-axis) in radians (0 to 2pi).
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
            Controls how much the ellipsoid is tilted toward/away from the viewer.
        illumination_angle: Direction of illumination in the image plane in radians (0 to 2pi).
            0 radians is at the top of the image, pi/2 is to the right.
        phase_angle: Phase angle in radians (0 to pi).
            0 = head-on illumination (fully illuminated),
            pi/2 = side illumination (half illuminated),
            pi = back illumination (no visible illumination).
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.

    Returns:
        A 2D numpy array of shape (size_v, size_u) with float values from 0.0 to 1.0,
        where 0.0 is black and 1.0 is full white.
    """
    size_v, size_u = size

    # Convert full-width axes to half-widths for ellipsoid math
    semi_major_axis = axis1 / 2.0
    semi_minor_axis = axis2 / 2.0
    semi_c_axis = axis3 / 2.0

    # Determine anti-aliasing scale factor (only for limb smoothing)
    if anti_aliasing > 0:
        # Scale factor: 1 (no AA) to 4 (max AA)
        aa_scale = int(1 + 3 * anti_aliasing)
    else:
        aa_scale = 1

    # Work at higher resolution for anti-aliasing at limb
    work_v = size_v * aa_scale
    work_u = size_u * aa_scale
    work_center_v = center[0] * aa_scale
    work_center_u = center[1] * aa_scale
    work_semi_major = semi_major_axis * aa_scale
    work_semi_minor = semi_minor_axis * aa_scale
    work_semi_c = semi_c_axis * aa_scale

    # Create coordinate grids at pixel centers
    # This preserves subpixel alignment such that (0.5, 0.5) refers to the center
    # of pixel (0,0) regardless of supersampling scale.
    v_coords, u_coords = np.mgrid[0:work_v, 0:work_u].astype(float)
    v_coords += 0.5
    u_coords += 0.5
    v_coords -= work_center_v
    u_coords -= work_center_u

    # Apply rotation_z (in-plane rotation around z-axis, clockwise)
    cos_rz = np.cos(rotation_z)
    sin_rz = np.sin(rotation_z)
    v_rot1 = v_coords * cos_rz - u_coords * sin_rz
    u_rot1 = v_coords * sin_rz + u_coords * cos_rz

    # Apply rotation_tilt (rotation around u-axis, tilting toward/away from viewer)
    # This affects the apparent shape and the z-coordinate
    cos_rt = np.cos(rotation_tilt)

    # After tilt, the v coordinate is affected
    # v_rot = v_rot1 * cos_rt (compressed by tilt)
    # z coordinate appears: z = v_rot1 * sin_rt (tilted depth)
    v_rot = v_rot1 * cos_rt
    u_rot = u_rot1
    # z will be computed from ellipsoid equation

    # Compute distance from ellipse center in local coordinates (2D projection)
    # For the visible ellipse: (v_rot/a)^2 + (u_rot/b)^2 <= 1
    ellipse_dist_sq = (v_rot / work_semi_major) ** 2 + (u_rot / work_semi_minor) ** 2
    ellipse_dist = np.sqrt(ellipse_dist_sq)

    # Compute z coordinate for 3D ellipsoid
    # Ellipsoid equation: (v_rot/a)^2 + (u_rot/b)^2 + (z/c)^2 = 1
    # For visible hemisphere: z = c * sqrt(1 - (v_rot/a)^2 - (u_rot/b)^2)
    # Only compute for points inside the ellipse
    z_coords = np.zeros_like(v_rot)
    inside_mask = ellipse_dist_sq <= 1.0
    z_sq = np.maximum(0.0, 1.0 - ellipse_dist_sq[inside_mask])
    z_coords[inside_mask] = work_semi_c * np.sqrt(z_sq)

    # Create base ellipse mask (1.0 inside, 0.0 outside)
    # Anti-aliasing only applied at the limb (edge)
    if anti_aliasing > 0:
        # Smooth transition zone: about 1 pixel at work resolution, only at edge
        edge_width = 3.0
        ellipse_mask = np.clip(1.0 - np.maximum(0, ellipse_dist - 1.0) / edge_width, 0.0, 1.0)
    else:
        ellipse_mask = (ellipse_dist <= 1.0).astype(float)

    normal_v, normal_u, normal_z = ellipsoid_image_normals(
        ellipse_mask,
        v_rot,
        u_rot,
        z_coords=z_coords,
        work_semi_major=work_semi_major,
        work_semi_minor=work_semi_minor,
        work_semi_c=work_semi_c,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
    )
    intensity = (
        lambert_from_normals(
            normal_v,
            normal_u,
            normal_z,
            illumination_angle=illumination_angle,
            phase_angle=phase_angle,
        )
        * ellipse_mask
    )

    # Downsample if anti-aliasing was used
    if aa_scale > 1:
        # Simple box filter downsampling
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))

    # Ensure values are in [0, 1] range
    return cast(NDArrayFloatType, np.clip(intensity, 0.0, 1.0))
