"""Simulation utilities for creating synthetic images for testing and model correlation.

This module provides functions to create simulated planetary bodies and other
astronomical objects for testing navigation algorithms without requiring real
image files and SPICE kernels.
"""

from typing import cast

import numpy as np

from spindoctor.sim.seeds import stable_param_seed
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType


def create_simulated_body(
    size: tuple[int, int],
    center: tuple[float, float],
    axis1: float,
    axis2: float,
    axis3: float,
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
    seed: int | None = None,
) -> NDArrayFloatType:
    """Create a simulated planetary body as an ellipsoid with shading and surface features.

    The body is modeled as a 3D ellipsoid projected onto 2D. The ellipsoid can have internal
    craters of varying sizes and depths. The body is illuminated using Lambertian shading
    (cos(incidence)) based on the illumination direction and phase angle.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        axis1: The full width of axis 1 (a) of the ellipsoid in pixels.
        axis2: The full width of axis 2 (b) of the ellipsoid in pixels.
        axis3: The full width of axis 3 (c) of the ellipsoid in pixels (depth).
        center: Tuple of (v, u) giving the center position in floating-point pixels.
            (0.0, 0.0) is the top-left corner of pixel (0,0), (0.5, 0.5) is the
            center of pixel (0,0).
        rotation_z: Rotation angle around the viewing axis (z-axis) in radians (0 to 2pi).
        rotation_tilt: Tilt angle of the ellipsoid in radians (0 to pi/2).
            Controls how much the ellipsoid is tilted toward/away from the viewer.
        illumination_angle: Direction of illumination in the image plane in radians (0 to 2pi).
            0 radians is at the top of the image, pi/2 is to the right.
        phase_angle: Phase angle in radians (0 to pi).
            0 = head-on illumination (fully illuminated),
            pi/2 = side illumination (half illuminated),
            pi = back illumination (no visible illumination).
        crater_fill: Approximate fraction of the ellipse to fill with craters.
        crater_min_radius: Minimum radius of a crater as a fraction of axis1.
        crater_max_radius: Maximum radius of a crater as a fraction of axis1.
        crater_power_law_exponent: Power law exponent for the crater radius distribution.
        crater_relief_scale: Scale factor for the crater depth.
        anti_aliasing: Float between 0 and 1 controlling anti-aliasing amount at the limb.
            0 = no anti-aliasing, 1 = maximum anti-aliasing. Only affects the edge.
        seed: Random seed for crater generation. If None, uses a hash-based seed.

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

    if crater_fill > 0:
        total_area = np.sum(inside_mask)
        avg_crater_area = (crater_max_radius * semi_major_axis * aa_scale) ** 2
        n_craters = np.clip(int(crater_fill * total_area / avg_crater_area), 0, 1000)

        # Use the provided seed, or fall back to a process-stable seed derived
        # from the body geometry.  The built-in hash is salted per process for
        # some types, so it cannot be used where determinism is required.
        if seed is not None:
            seed_value = int(seed) & 0x7FFFFFFF
        else:
            seed_value = stable_param_seed(axis1, axis2, axis3, center) & 0x7FFFFFFF
        rng = np.random.RandomState(seed_value)
        # Choose crater centers strictly inside the ellipse (exclude AA rim and exterior)
        ellipse_mask_nz = ellipse_dist_sq < 1.0
        nz = np.argwhere(ellipse_mask_nz)

        intensity = _add_craters_and_shading(
            ellipse_mask_nz,
            v_coords,
            u_coords,
            nz,  # indices of non-zero ellipse pixels (list/array of (v,u))
            rng,
            n_craters,
            crater_min_radius * semi_major_axis,
            crater_max_radius * semi_major_axis,
            crater_power_law_exponent,
            crater_relief_scale,
            work_center_v,
            work_center_u,
            aa_scale,
            illumination_angle,  # 0 = from top, pi/2 = from right
            phase_angle,  # 0 = from front, pi/2 = from side, pi = from back
            ellipse_mask=ellipse_mask,
            z_coords=z_coords,
            v_rot=v_rot,
            u_rot=u_rot,
            work_semi_major=work_semi_major,
            work_semi_minor=work_semi_minor,
            work_semi_c=work_semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
        )

    else:
        normal_v, normal_u, normal_z = _ellipsoid_image_normals(
            ellipse_mask,
            v_rot,
            u_rot,
            z_coords,
            work_semi_major=work_semi_major,
            work_semi_minor=work_semi_minor,
            work_semi_c=work_semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
        )
        intensity = (
            _lambert_from_normals(normal_v, normal_u, normal_z, illumination_angle, phase_angle)
            * ellipse_mask
        )

    # Downsample if anti-aliasing was used
    if aa_scale > 1:
        # Simple box filter downsampling
        intensity = intensity.reshape(size_v, aa_scale, size_u, aa_scale).mean(axis=(1, 3))

    # Ensure values are in [0, 1] range
    intensity = np.clip(intensity, 0.0, 1.0)

    return intensity


def _ellipsoid_image_normals(
    ellipse_mask: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    z_coords: NDArrayFloatType,
    *,
    work_semi_major: float,
    work_semi_minor: float,
    work_semi_c: float,
    cos_rz: float,
    sin_rz: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Unit surface normals of the base ellipsoid in image coordinates.

    For a 3D ellipsoid, the surface normal at body-frame point (v, u, z) is
    (v/a^2, u/b^2, z/c^2) normalized.  The in-plane components are then rotated
    back from the ellipsoid's rotated frame to image coordinates through the
    inverse of the rotation_z coordinate transformation; the z component is
    perpendicular to the image plane and unaffected.  Both the smooth and the
    cratered shading paths derive their base normals here, so the two paths
    share a single illumination convention.

    Parameters:
        ellipse_mask: Ellipse coverage mask; normals are computed where > 0.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        z_coords: Depth of the visible ellipsoid surface at each pixel.
        work_semi_major: Semi-major axis (a) at working resolution.
        work_semi_minor: Semi-minor axis (b) at working resolution.
        work_semi_c: Depth semi-axis (c) at working resolution.
        cos_rz: Cosine of the rotation_z angle.
        sin_rz: Sine of the rotation_z angle.

    Returns:
        Tuple of (normal_v, normal_u, normal_z) unit-normal component arrays in
        image coordinates.
    """
    normal_v_local = np.zeros_like(v_rot)
    normal_u_local = np.zeros_like(u_rot)
    normal_z_local = np.zeros_like(z_coords)

    # Only compute normals for points inside the ellipsoid
    inside_mask = ellipse_mask > 0
    normal_v_local[inside_mask] = v_rot[inside_mask] / (work_semi_major**2)
    normal_u_local[inside_mask] = u_rot[inside_mask] / (work_semi_minor**2)
    normal_z_local[inside_mask] = z_coords[inside_mask] / (work_semi_c**2)

    # Normalize the normal vectors
    normal_mag = np.sqrt(normal_v_local**2 + normal_u_local**2 + normal_z_local**2)
    normal_mag = np.maximum(normal_mag, 1e-10)  # Avoid division by zero
    normal_v_local /= normal_mag
    normal_u_local /= normal_mag
    normal_z_local /= normal_mag

    # Rotate normal back to image coordinates (only v and u components)
    # The z component stays in the depth direction
    # Use inverse rotation (negate sin) to match the coordinate transformation
    normal_v = normal_v_local * cos_rz + normal_u_local * sin_rz
    normal_u = -normal_v_local * sin_rz + normal_u_local * cos_rz
    return normal_v, normal_u, normal_z_local


def _lambert_from_normals(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Lambertian illumination strength for image-frame unit surface normals.

    The normals must already be unit length (or zero outside the body).  Both
    the smooth and the cratered shading paths use this single implementation
    so their illumination conventions cannot diverge.

    Parameters:
        normal_v: V component of the unit surface normal in image coordinates.
        normal_u: U component of the unit surface normal in image coordinates.
        normal_z: Z (toward-observer) component of the unit surface normal.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Illumination strength array in [0, 1]; 0 on the far hemisphere.
    """
    # Illumination direction in 3D space
    # illumination_angle: 0 = top, pi/2 = right, pi = bottom, 3pi/2 = left
    # This is the direction in the image plane
    illum_v_2d = -np.cos(illumination_angle)  # Negative because v increases downward
    illum_u_2d = np.sin(illumination_angle)

    # Phase angle: angle between observer-body-sun
    # phase_angle = 0: full moon (observer and sun on same side, visible face fully lit)
    # phase_angle = pi/2: quarter moon (observer and sun perpendicular)
    # phase_angle = pi: new moon (observer and sun opposite, visible face dark/backlit)
    #
    # The illumination vector points from body toward sun.
    # For phase_angle = 0: sun is behind observer, so illumination comes from direction
    #   that lights the visible face (positive dot product with visible normals).
    # For phase_angle = pi: sun is behind body, so illumination comes from direction
    #   that doesn't light the visible face (negative dot product with visible normals).
    #
    # The z-component of illumination: when phase_angle = pi, we want backlit (dark),
    # so z should be negative (illumination away from observer) to make dot product negative.
    # When phase_angle = 0, we want fully lit, so z should be positive to make dot product positive.
    # So: z = cos(phase_angle) gives phase_angle=0 -> z=+1, phase_angle=pi -> z=-1
    illum_z = np.cos(phase_angle)

    # The in-plane component magnitude
    illum_scale_2d = np.sin(phase_angle)
    illum_v_3d = illum_v_2d * illum_scale_2d
    illum_u_3d = illum_u_2d * illum_scale_2d

    # Normalize the 3D illumination direction
    illum_mag = np.sqrt(illum_v_3d**2 + illum_u_3d**2 + illum_z**2)
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
    cos_incidence = normal_v * illum_v_3d + normal_u * illum_u_3d + normal_z * illum_z_norm

    # Lambertian shading: I = I_0 * max(0, cos(incidence))
    # Only apply to visible hemisphere and clip to [0, 1] range
    dark_side_illum_strength = 0.01  # TODO make config parameter
    light_side_illum_gamma = 1  # TODO make config parameter
    illum_strength = np.where(
        visible_hemisphere, np.clip(cos_incidence, dark_side_illum_strength, 1.0), 0.0
    )
    illum_strength **= light_side_illum_gamma

    return cast(NDArrayFloatType, illum_strength)


def _add_craters_and_shading(
    ellipse_mask_nz: NDArrayBoolType,
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
    z_coords: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    work_semi_major: float,
    work_semi_minor: float,
    work_semi_c: float,
    cos_rz: float,
    sin_rz: float,
) -> NDArrayFloatType:
    """Carve craters into the ellipsoid surface and apply Lambertian shading.

    The base-surface normals and the Lambertian illumination model are shared
    with the smooth (no-crater) path via ``_ellipsoid_image_normals`` and
    ``_lambert_from_normals``, so a cratered body and a smooth body with the
    same pose and illumination shade identically outside the craters.

    Parameters:
        ellipse_mask_nz: Boolean mask of pixels strictly inside the ellipse.
        v_coords: Centered v pixel coordinates at working resolution.
        u_coords: Centered u pixel coordinates at working resolution.
        nz: Array of (v, u) indices of pixels inside the ellipse.
        rng: Seeded random stream for crater placement and sizes.
        n_craters: Number of craters to place.
        R_min: Minimum crater radius in pixels (before aa scaling).
        R_max: Maximum crater radius in pixels (before aa scaling).
        crater_power_law_exponent: Power law exponent for crater radii.
        crater_relief_scale: Scale factor for crater depth.
        work_center_v: Body center v at working resolution.
        work_center_u: Body center u at working resolution.
        aa_scale: Anti-aliasing supersampling factor.
        lighting_angle: In-plane light direction in radians (0 = from top).
        phase_angle: Phase angle in radians (0 = fully lit, pi = backlit).
        ellipse_mask: Ellipse coverage mask including the soft AA rim.
        z_coords: Depth of the visible ellipsoid surface at each pixel.
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        work_semi_major: Semi-major axis (a) at working resolution.
        work_semi_minor: Semi-minor axis (b) at working resolution.
        work_semi_c: Depth semi-axis (c) at working resolution.
        cos_rz: Cosine of the rotation_z angle.
        sin_rz: Sine of the rotation_z angle.

    Returns:
        New intensity array with craters and lighting applied.
    """

    # ------------------------------------------------------------------
    # 0. Heightmap we will add craters to
    # ------------------------------------------------------------------
    height = np.zeros_like(ellipse_mask, dtype=float)

    # ------------------------------------------------------------------
    # 1. Radius distribution: power law in [R_min, R_max]
    # ------------------------------------------------------------------
    def power_law_radius(
        rng: np.random.RandomState,
        R_min: float,
        R_max: float,
        alpha: float,
        size: int | None = None,
    ) -> NDArrayFloatType:
        """Sample R from p(R) ~ R^(-alpha) on [R_min, R_max], alpha > 1."""
        if alpha <= 1:
            raise ValueError('alpha must be > 1 for a proper power law.')
        a = 1.0 - alpha
        R_min_a = R_min**a
        R_max_a = R_max**a
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
        crater_radius = power_law_radius(
            rng, R_min * aa_scale, R_max * aa_scale, crater_power_law_exponent
        )

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
        # d/D: small -> 0.20, large -> 0.07
        d_over_D_mean = (1.0 - t) * 0.20 + t * 0.07

        # Multiplicative noise around mean
        noise = np.exp(rng.normal(loc=0.0, scale=0.25))  # adjust scatter
        d_over_D = d_over_D_mean * noise

        crater_depth = crater_relief_scale * d_over_D * D

        # ------------------------------------------------------------------
        # 2b. Geometric profile: bowl + walls + raised rim
        # ------------------------------------------------------------------
        # TODO make config parameters
        R_floor = 0.6 * crater_radius  # flat-ish floor
        R_rim = crater_radius  # inner rim radius
        R_outer = 1.3 * crater_radius  # where rim merges back to surface

        local_profile = np.zeros_like(r)

        # Central bowl / floor (parabolic-ish)
        inside_floor = r <= R_floor
        local_profile[inside_floor] = -crater_depth * (1.0 - (r[inside_floor] / R_floor) ** 2)

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
    # 3. Lambertian shading: base-ellipsoid normals perturbed by crater relief
    # ----------------------------------------------------------------------
    # The base-surface normal comes from the same analytic formula (and the
    # same rotation_z back-rotation to image coordinates) as the smooth,
    # no-crater path, so both shading paths share one illumination convention.
    normal_v, normal_u, normal_z = _ellipsoid_image_normals(
        ellipse_mask,
        v_rot,
        u_rot,
        z_coords,
        work_semi_major=work_semi_major,
        work_semi_minor=work_semi_minor,
        work_semi_c=work_semi_c,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
    )

    # The crater height field is carved along the viewing axis, so the exact
    # normal of the combined surface (base + height) is proportional to
    # (-d(z + height)/dv, -d(z + height)/du, 1).  Scaling that vector by the
    # base normal's z component rewrites it as the base normal plus a
    # height-gradient term, which stays finite at the limb where the base
    # surface gradient diverges.
    dheight_dv, dheight_du = np.gradient(height)
    perturbed_v = normal_v - normal_z * dheight_dv
    perturbed_u = normal_u - normal_z * dheight_du
    perturbed_mag = np.sqrt(perturbed_v**2 + perturbed_u**2 + normal_z**2)
    perturbed_mag = np.maximum(perturbed_mag, 1e-10)  # Avoid division by zero
    perturbed_v /= perturbed_mag
    perturbed_u /= perturbed_mag
    perturbed_z = normal_z / perturbed_mag

    lambert = _lambert_from_normals(
        perturbed_v, perturbed_u, perturbed_z, lighting_angle, phase_angle
    )

    # Apply ellipse mask (with AA edge)
    intensity_out = lambert * ellipse_mask
    # Ensure area strictly outside the ellipse is zeroed, to avoid any bleed with AA
    intensity_out[~ellipse_mask_nz] = 0.0
    return intensity_out
