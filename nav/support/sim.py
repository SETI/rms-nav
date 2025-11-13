"""Simulation utilities for creating synthetic images for testing and model correlation.

This module provides functions to create simulated planetary bodies and other
astronomical objects for testing navigation algorithms without requiring real
image files and SPICE kernels.
"""

import numpy as np
from numpy.typing import NDArray

from nav.support.types import NDArrayFloatType


def create_simulated_body(
    size: tuple[int, int],
    semi_major_axis: float,
    semi_minor_axis: float,
    semi_c_axis: float,
    center: tuple[float, float],
    rotation_z: float,
    rotation_tilt: float,
    illumination_angle: float,
    phase_angle: float,
    rough: tuple[float, float],
    craters: float,
    anti_aliasing: float,
) -> NDArrayFloatType:
    """Create a simulated planetary body as an ellipsoid with shading and surface features.

    The body is modeled as a 3D ellipsoid projected onto 2D. The ellipsoid
    can have rough edges (simulating craters on the limb) and internal craters of
    varying sizes. The body is illuminated using Lambertian shading (cos(incidence))
    based on the illumination direction and phase angle.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        semi_major_axis: The semi-major axis (a) of the ellipsoid in pixels.
        semi_minor_axis: The semi-minor axis (b) of the ellipsoid in pixels.
        semi_c_axis: The third semi-axis (c) of the ellipsoid in pixels (depth).
        center: Tuple of (v, u) giving the center position in floating-point pixels.
            (0.0, 0.0) is the top-left corner of pixel (0,0), (0.5, 0.5) is the
            center of pixel (0,0).
        rotation_z: Rotation angle around the viewing axis (z-axis) in radians (0 to 2π).
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to π/2).
            Controls how much the ellipsoid is tilted toward/away from the viewer.
        illumination_angle: Direction of illumination in the image plane in radians (0 to 2π).
            0 radians is at the top of the image, π/2 is to the right.
        phase_angle: Phase angle in radians (0 to π).
            0 = head-on illumination (fully illuminated),
            π/2 = side illumination (half illuminated),
            π = back illumination (no visible illumination).
        rough: Tuple of (mean, std_dev) giving the roughness parameters in pixels.
            mean: Average height deviation from the ellipse edge.
            std_dev: Standard deviation of the roughness distribution.
        craters: Float between 0 and 1 controlling crater density inside the ellipse.
            0 = no craters (even shading), 1 = maximum crater density.
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.

    Returns:
        A 2D numpy array of shape (size_v, size_u) with float values from 0.0 to 1.0,
        where 0.0 is black and 1.0 is full white.
    """
    size_v, size_u = size
    rough_mean, rough_std = rough

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
    sin_rt = np.sin(rotation_tilt)

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

    # Apply edge roughness to the limb
    if rough_mean > 0 or rough_std > 0:
        # Compute angle around ellipse for each point
        angle = np.arctan2(v_rot / work_semi_major, u_rot / work_semi_minor)

        # Generate roughness pattern using multiple frequencies for realistic appearance
        n_freqs = 8
        roughness = np.zeros_like(angle)
        # Use a hash of position for reproducibility but variation
        rng = np.random.RandomState(42)
        for i in range(n_freqs):
            freq = 2 ** i
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = rough_mean / (freq ** 0.7)  # Higher frequencies have less amplitude
            roughness += amplitude * np.sin(freq * angle + phase)

        # Add random component with specified std_dev
        roughness += rng.normal(0, rough_std, size=angle.shape)

        # Apply roughness as radial perturbation at the edge
        # Points near the edge (limb) get more perturbation
        # Use a sharper falloff so roughness is visible
        edge_factor = np.exp(-np.maximum(0, ellipse_dist - 0.95) ** 2 / 0.05)
        roughness *= edge_factor

        # Perturb the ellipse distance
        ellipse_dist_rough = ellipse_dist + roughness / work_semi_major
        ellipse_dist_rough = np.maximum(0.0, ellipse_dist_rough)  # Don't go negative

        # Recompute mask with roughness (only affects edge)
        if anti_aliasing > 0:
            ellipse_mask = np.clip(1.0 - np.maximum(0, ellipse_dist_rough - 1.0) / edge_width, 0.0, 1.0)
        else:
            ellipse_mask = (ellipse_dist_rough <= 1.0).astype(float)

        # Update z coordinates for points affected by roughness
        inside_rough_mask = ellipse_dist_rough <= 1.0
        ellipse_dist_sq_rough = ellipse_dist_rough ** 2
        z_sq_rough = np.maximum(0.0, 1.0 - ellipse_dist_sq_rough[inside_rough_mask])
        z_coords[inside_rough_mask] = work_semi_c * np.sqrt(z_sq_rough)

    # Apply Lambertian shading for 3D ellipsoid
    # For a 3D ellipsoid, the surface normal at point (v, u, z) is:
    # n = (v/a², u/b², z/c²) normalized

    # Compute 3D surface normal in local coordinates
    normal_v_local = np.zeros_like(v_rot)
    normal_u_local = np.zeros_like(u_rot)
    normal_z_local = np.zeros_like(z_coords)

    # Only compute normals for points inside the ellipsoid
    inside_mask = ellipse_mask > 0
    normal_v_local[inside_mask] = v_rot[inside_mask] / (work_semi_major ** 2)
    normal_u_local[inside_mask] = u_rot[inside_mask] / (work_semi_minor ** 2)
    normal_z_local[inside_mask] = z_coords[inside_mask] / (work_semi_c ** 2)

    # Normalize the normal vectors
    normal_mag = np.sqrt(normal_v_local ** 2 + normal_u_local ** 2 + normal_z_local ** 2)
    normal_mag = np.maximum(normal_mag, 1e-10)  # Avoid division by zero
    normal_v_local /= normal_mag
    normal_u_local /= normal_mag
    normal_z_local /= normal_mag

    # Rotate normal back to image coordinates (only v and u components)
    # The z component stays in the depth direction
    # Use inverse rotation (negate sin) to match the coordinate transformation
    normal_v = normal_v_local * cos_rz + normal_u_local * sin_rz
    normal_u = -normal_v_local * sin_rz + normal_u_local * cos_rz
    normal_z = normal_z_local  # z is perpendicular to image plane

    # Illumination direction in 3D space
    # illumination_angle: 0 = top, π/2 = right, π = bottom, 3π/2 = left
    # This is the direction in the image plane
    illum_v_2d = -np.cos(illumination_angle)  # Negative because v increases downward
    illum_u_2d = np.sin(illumination_angle)

    # Phase angle: angle between observer-body-sun
    # phase_angle = 0: full moon (observer and sun on same side, visible face fully lit)
    # phase_angle = π/2: quarter moon (observer and sun perpendicular)
    # phase_angle = π: new moon (observer and sun opposite, visible face dark/backlit)
    #
    # The illumination vector points from body toward sun.
    # For phase_angle = 0: sun is behind observer, so illumination comes from direction
    #   that lights the visible face (positive dot product with visible normals).
    # For phase_angle = π: sun is behind body, so illumination comes from direction
    #   that doesn't light the visible face (negative dot product with visible normals).
    #
    # The z-component of illumination: when phase_angle = π, we want backlit (dark),
    # so z should be negative (illumination away from observer) to make dot product negative.
    # When phase_angle = 0, we want fully lit, so z should be positive to make dot product positive.
    # So: z = cos(phase_angle) gives phase_angle=0 -> z=+1, phase_angle=π -> z=-1
    illum_z = np.cos(phase_angle)  # phase_angle=0 -> z=+1 (lit), phase_angle=π -> z=-1 (backlit/dark)

    # The in-plane component magnitude
    illum_scale_2d = np.sin(phase_angle)
    illum_v_3d = illum_v_2d * illum_scale_2d
    illum_u_3d = illum_u_2d * illum_scale_2d

    # Normalize the 3D illumination direction
    illum_mag = np.sqrt(illum_v_3d ** 2 + illum_u_3d ** 2 + illum_z ** 2)
    if illum_mag > 1e-10:
        illum_v_3d /= illum_mag
        illum_u_3d /= illum_mag
        illum_z_norm = illum_z / illum_mag
    else:
        illum_z_norm = 1.0  # Directly toward observer

    # Only illuminate points on the visible hemisphere (facing toward observer)
    # The z-component of the normal should be positive (pointing toward observer)
    visible_hemisphere = normal_z > 0

    # Compute cosine of incidence angle (Lambertian shading)
    # cos(incidence) = dot(normal, illumination_direction)
    cos_incidence = (normal_v * illum_v_3d +
                     normal_u * illum_u_3d +
                     normal_z * illum_z_norm)

    # Lambertian shading: I = I₀ * max(0, cos(incidence))
    # Only apply to visible hemisphere and clip to [0, 1] range
    illum_strength = np.where(visible_hemisphere, np.clip(cos_incidence, 0.0, 1.0), 0.0)

    # Base intensity from Lambertian shading (only inside the ellipsoid and visible hemisphere)
    intensity = illum_strength * ellipse_mask

    # Add internal craters
    if craters > 0:
        # Generate random craters with better visibility
        rng = np.random.RandomState(43)  # For reproducibility
        n_craters = int(200 * craters)  # More craters for better visibility

        # Initialize crater image to ones, but only apply where ellipse exists
        crater_image = np.ones_like(intensity)

        for _ in range(n_craters):
            # Random position inside ellipse (in local coordinates)
            # Use rejection sampling to ensure craters are inside ellipse
            max_attempts = 100
            v_local = 0.0
            u_local = 0.0
            for _ in range(max_attempts):
                v_local = rng.uniform(-work_semi_major * 0.8, work_semi_major * 0.8)
                u_local = rng.uniform(-work_semi_minor * 0.8, work_semi_minor * 0.8)
                if (v_local / work_semi_major) ** 2 + (u_local / work_semi_minor) ** 2 < 0.7:
                    break

            # Rotate to image coordinates (inverse rotation to match coordinate transform)
            v_crater = v_local * cos_rz + u_local * sin_rz + work_center_v
            u_crater = -v_local * sin_rz + u_local * cos_rz + work_center_u

            # Random crater size (logarithmic distribution for realism)
            crater_radius = rng.lognormal(np.log(3.0), 0.9) * aa_scale
            crater_radius = np.clip(crater_radius, 2.0, work_semi_major * 0.25)

            # Random crater depth (darker = deeper) - make more visible
            # Increase depth range for better visibility
            crater_depth = rng.uniform(0.5, 1.0) * craters

            # Create crater as a circular depression
            v_dist = v_coords - v_crater
            u_dist = u_coords - u_crater
            crater_dist = np.sqrt(v_dist ** 2 + u_dist ** 2)

            # Crater shape: deeper in center, shallower at edges
            # Use a sharper falloff for better visibility
            crater_mask = crater_dist < crater_radius
            # Only apply craters where the ellipse exists
            crater_mask = crater_mask & (ellipse_mask > 0)
            if np.any(crater_mask):
                # Gaussian-like profile with sharper edges
                sigma = crater_radius / 2.5
                crater_profile = 1.0 - crater_depth * np.exp(-crater_dist ** 2 / (2 * sigma ** 2))
                # Apply crater by darkening (multiply, but ensure it only affects where mask is true)
                crater_image[crater_mask] = np.minimum(crater_image[crater_mask], crater_profile[crater_mask])

        # Apply craters to intensity (multiply to darken) - only where ellipse exists
        intensity = intensity * crater_image

    # Downsample if anti-aliasing was used
    if aa_scale > 1:
        # Simple box filter downsampling
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))

    # Ensure values are in [0, 1] range
    intensity = np.clip(intensity, 0.0, 1.0)

    return intensity
