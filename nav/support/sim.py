"""Simulation utilities for creating synthetic images for testing and model correlation.

This module provides functions to create simulated planetary bodies and other
astronomical objects for testing navigation algorithms without requiring real
image files and SPICE kernels.
"""

from typing import Optional, cast

import numpy as np

from nav.support.types import NDArrayFloatType, NDArrayIntType


def create_simulated_body(
    size: tuple[int, int],
    center: tuple[float, float],
    semi_major_axis: float,
    semi_minor_axis: float,
    semi_c_axis: float,
    *,
    rotation_z: float = 0.0,
    rotation_tilt: float = 0.0,
    illumination_angle: float = 0.0,
    phase_angle: float = 0.0,
    crater_fill: float = 0.0,
    crater_min_radius: float = 0.05,
    crater_max_radius: float = 0.25,
    crater_power_law_exponent: float = 3.0,
    crater_relief_scale: float = 0.6,
    anti_aliasing: float = 0.0,
) -> NDArrayFloatType:
    """Create a simulated planetary body as an ellipsoid with shading and surface features.

    The body is modeled as a 3D ellipsoid projected onto 2D. The ellipsoid can have internal
    craters of varying sizes and depths.
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
        crater_fill: Approximate fraction of the ellipse to fill with craters.
        crater_min_radius: Minimum radius of a crater as a fraction of the semi-major axis.
        crater_max_radius: Maximum radius of a crater as a fraction of the semi-major axis.
        crater_power_law_exponent: Power law exponent for the crater radius distribution.
        crater_relief_scale: Scale factor for the crater depth.
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.

    Returns:
        A 2D numpy array of shape (size_v, size_u) with float values from 0.0 to 1.0,
        where 0.0 is black and 1.0 is full white.
    """
    size_v, size_u = size

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

    if crater_fill > 0:
        total_area = np.sum(inside_mask)
        avg_crater_area = (crater_max_radius * semi_major_axis) ** 2
        n_craters = np.clip(int(crater_fill * total_area / avg_crater_area), 0, 1000)

        rng = np.random.RandomState(int(semi_major_axis))  # For reproducibility
        # Choose crater centers strictly inside the ellipse (exclude AA rim and exterior)
        ellipse_mask_nz = ellipse_dist_sq < 1.0
        nz = np.argwhere(ellipse_mask_nz)

        intensity = _add_craters_and_shading(
            ellipse_mask_nz,
            v_coords,
            u_coords,
            nz,                 # indices of non-zero ellipse pixels (list/array of (v,u))
            rng,
            n_craters,
            crater_min_radius * semi_major_axis,
            crater_max_radius * semi_major_axis,
            crater_power_law_exponent,
            crater_relief_scale,
            work_center_v,
            work_center_u,
            aa_scale,
            illumination_angle,     # 0 = from top, pi/2 = from right
            phase_angle,        # 0 = from front, pi/2 = from side, pi = from back
            ellipse_mask=ellipse_mask,
            z_coords=z_coords,
        )

    else:
        intensity = _lambertian_shading(
            ellipse_mask,
            v_rot,
            u_rot,
            z_coords,
            work_semi_major,
            work_semi_minor,
            work_semi_c,
            illumination_angle,
            phase_angle,
            cos_rz,
            sin_rz,
        )

    # Downsample if anti-aliasing was used
    if aa_scale > 1:
        # Simple box filter downsampling
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))

    # Ensure values are in [0, 1] range
    intensity = np.clip(intensity, 0.0, 1.0)

    return intensity


def _lambertian_shading(ellipse_mask: NDArrayFloatType,
                        v_rot: NDArrayFloatType,
                        u_rot: NDArrayFloatType,
                        z_coords: NDArrayFloatType,
                        work_semi_major: float,
                        work_semi_minor: float,
                        work_semi_c: float,
                        illumination_angle: float,
                        phase_angle: float,
                        cos_rz: float,
                        sin_rz: float) -> NDArrayFloatType:
    """Add Lambertian shading to the intensity."""
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
    illum_z = np.cos(phase_angle)

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
    dark_side_illum_strength = 0.01
    light_side_illum_gamma = 0.7
    illum_strength = np.where(visible_hemisphere,
                              np.clip(cos_incidence, dark_side_illum_strength, 1.0), 0.0)
    illum_strength **= light_side_illum_gamma

    # Base intensity from Lambertian shading (only inside the ellipsoid and visible hemisphere)
    intensity = illum_strength * ellipse_mask

    return intensity


def _add_craters_and_shading(ellipse_mask_nz: NDArrayFloatType,
                             v_coords: NDArrayFloatType,
                             u_coords: NDArrayFloatType,
                             nz: NDArrayIntType,
                             rng: np.random.RandomState,
                             n_craters: int,
                             R_min: float,
                             R_max: float,
                             crater_power_law_exponent: float,
                             crater_relief_scale: float,
                             work_center_v: float,
                             work_center_u: float,
                             aa_scale: int,
                             lighting_angle: float,
                             phase_angle: float,
                             *,
                             ellipse_mask: NDArrayFloatType,
                             z_coords: NDArrayFloatType
                             ) -> NDArrayFloatType:
    """
    Returns new intensity with craters + lighting applied.
    """

    # ------------------------------------------------------------------
    # 0. Heightmap we will add craters to
    # ------------------------------------------------------------------
    height = np.zeros_like(ellipse_mask, dtype=float)

    # ------------------------------------------------------------------
    # 1. Radius distribution: power law in [R_min, R_max]
    # ------------------------------------------------------------------
    def power_law_radius(rng: np.random.RandomState,
                         R_min: float,
                         R_max: float,
                         alpha: float,
                         size: Optional[int] = None) -> NDArrayFloatType:
        """Sample R from p(R) ∝ R^(-alpha) on [R_min, R_max], alpha > 1."""
        if alpha <= 1:
            raise ValueError("alpha must be > 1 for a proper power law.")
        a = 1.0 - alpha
        R_min_a = R_min ** a
        R_max_a = R_max ** a
        u = rng.uniform(0.0, 1.0, size=size)
        R = (R_min_a + u * (R_max_a - R_min_a)) ** (1.0 / a)
        return cast(NDArrayFloatType, R)

    # ------------------------------------------------------------------
    # 2. Crater placement + geometry
    # ------------------------------------------------------------------
    for _ in range(n_craters):
        # Random crater center in non-zero ellipse area
        v_crater, u_crater = nz[rng.randint(len(nz))]

        # Radius from power-law distribution
        crater_radius = power_law_radius(rng, R_min * aa_scale, R_max * aa_scale,
                                         crater_power_law_exponent)

        # Compute distances from crater center.
        # v_coords/u_coords are in centered pixel coordinates: (index + 0.5 - work_center_*).
        # Convert crater center (array indices) to the same coordinate system.
        # Compute distances in absolute pixel-index coordinates to avoid frame mismatches:
        # v_coords/u_coords are in centered coords: (idx + 0.5 - work_center_*).
        # Convert them back to absolute index coords by adding work_center_*,
        # then subtract the crater center at (idx_crater + 0.5).
        v_abs = v_coords + work_center_v
        u_abs = u_coords + work_center_u
        center_v_abs = float(v_crater) + 0.5
        center_u_abs = float(u_crater) + 0.5
        v_dist = v_abs - center_v_abs
        u_dist = u_abs - center_u_abs
        crater_dist = np.sqrt(v_dist**2 + u_dist**2)

        # Mask where crater affects height (slightly beyond radius for rim)
        # TODO make config parameter
        crater_mask = (crater_dist < crater_radius * 1.1) & ellipse_mask_nz
        if not np.any(crater_mask):
            continue

        r = crater_dist[crater_mask]
        D = 2.0 * crater_radius  # diameter

        # ------------------------------------------------------------------
        # 2a. Depth as function of size (small craters deeper, large shallower)
        # ------------------------------------------------------------------
        # Choose size range where we blend d/D ratio
        D_small = 10.0 * aa_scale
        D_large = R_max * 2.0  # roughly largest diameters

        # Blend factor in [0,1]
        t = np.clip((D - D_small) / (D_large - D_small + 1e-9), 0.0, 1.0)
        # d/D: small → 0.20, large → 0.07
        d_over_D_mean = (1.0 - t) * 0.20 + t * 0.07

        # Multiplicative noise around mean
        noise = np.exp(rng.normal(loc=0.0, scale=0.25))  # adjust scatter
        d_over_D = d_over_D_mean * noise

        crater_depth = crater_relief_scale * d_over_D * D

        # ------------------------------------------------------------------
        # 2b. Geometric profile: bowl + walls + raised rim
        # ------------------------------------------------------------------
        # TODO make config parameters
        R_floor = 0.6 * crater_radius     # flat-ish floor
        R_rim = crater_radius             # inner rim radius
        R_outer = 1.3 * crater_radius     # where rim merges back to surface

        local_profile = np.zeros_like(r)

        # Central bowl / floor (parabolic-ish)
        inside_floor = (r <= R_floor)
        local_profile[inside_floor] = -crater_depth * (
            1.0 - (r[inside_floor] / R_floor) ** 2
        )

        # Wall up to rim
        wall = (r > R_floor) & (r <= R_rim)
        t_wall = (r[wall] - R_floor) / (R_rim - R_floor + 1e-9)
        local_profile[wall] = -crater_depth * (1.0 - t_wall)

        # Raised rim outside
        rim = (r > R_rim) & (r <= R_outer)
        t_rim = (r[rim] - R_rim) / (R_outer - R_rim + 1e-9)
        rim_height = crater_depth * 0.25
        local_profile[rim] = rim_height * (1.0 - t_rim)

        # Add crater relief to global heightmap
        height[crater_mask] += local_profile

    # import matplotlib.pyplot as plt
    # plt.imshow(height)
    # plt.show()

    # ----------------------------------------------------------------------
    # 3. Lambertian shading using perturbed surface normals from z + height
    # ----------------------------------------------------------------------
    z_with_craters = z_coords + height

    # Approximate surface normal from height field: n ∝ (-dz/du, -dz/dv, 1)
    dz_dv, dz_du = np.gradient(z_with_craters)
    nx = -dz_du
    ny = -dz_dv
    nz_ = np.ones_like(z_with_craters)

    norm = np.sqrt(nx**2 + ny**2 + nz_**2)
    norm = np.maximum(norm, 1e-9)
    nx /= norm
    ny /= norm
    nz_ /= norm

    # Illumination direction in image coordinates
    lx_2d = np.sin(lighting_angle)   # +u to the right
    ly_2d = -np.cos(lighting_angle)  # -v is up (v increases downward)
    lx = lx_2d * np.sin(phase_angle)
    ly = ly_2d * np.sin(phase_angle)
    lz = np.cos(phase_angle)
    Lnorm = np.sqrt(lx**2 + ly**2 + lz**2)
    if Lnorm > 1e-12:
        lx, ly, lz = lx / Lnorm, ly / Lnorm, lz / Lnorm

    # Only illuminate points on the visible hemisphere (facing observer)
    visible = nz_ > 0
    cos_incidence = nx * lx + ny * ly + nz_ * lz
    dark_side_illum_strength = 0.01  # TODO make config parameter
    light_side_illum_gamma = 1  # TODO make config parameter
    lambert = np.where(visible, np.clip(cos_incidence, dark_side_illum_strength, 1.0), 0.0)
    lambert **= light_side_illum_gamma

    # Apply ellipse mask (with AA edge)
    intensity_out = lambert * ellipse_mask
    # Ensure area strictly outside the ellipse is zeroed, to avoid any bleed with AA
    intensity_out[~ellipse_mask_nz] = 0.0
    return intensity_out
