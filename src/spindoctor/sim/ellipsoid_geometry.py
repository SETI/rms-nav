"""Shared ellipsoid shading geometry for the simulator's two sides.

The image-side forward renderer (``spindoctor.sim.forward.body``) and the
navigator-side predicted-body renderer (``spindoctor.nav_model.sim_body``)
must shade an ellipsoid with byte-identical conventions: the same surface
normals, the same light direction, and the same Lambert clamp.  With shared
conventions the planted scene error is the only error in a recovery
measurement; independent implementations would each carry their own
conventions and any delta between them would contaminate the measurement as
an unknown systematic.

These helpers take explicit geometry arguments only.  They never read a
scene parameter mapping, so they cannot carry truth-side information across
the information boundary (see ``spindoctor.sim.scene``).
"""

from typing import cast

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['ellipsoid_image_normals', 'illumination_vector', 'lambert_from_normals']


def ellipsoid_image_normals(
    ellipse_mask: NDArrayFloatType,
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    *,
    z_coords: NDArrayFloatType,
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


def illumination_vector(
    *, illumination_angle: float, phase_angle: float
) -> tuple[float, float, float]:
    """The unit body-to-sun direction in image coordinates.

    Both simulator sides derive the light direction here, so their
    illumination conventions cannot diverge.

    The in-plane direction comes from ``illumination_angle`` (0 = from the
    top of the image, pi/2 = from the right; the v component is negated
    because v increases downward).  The out-of-plane component encodes the
    phase angle -- the observer-body-sun angle: ``z = cos(phase_angle)`` so
    phase 0 (full) lights the visible face head-on and phase pi (new) lights
    it from behind, while the in-plane magnitude is ``sin(phase_angle)``.

    Parameters:
        illumination_angle: In-plane light direction in radians; 0 is from
            the top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Tuple of (v, u, z) components of the unit illumination direction;
        z points toward the observer.
    """
    illum_v_2d = -np.cos(illumination_angle)  # Negative because v increases downward
    illum_u_2d = np.sin(illumination_angle)

    illum_z = np.cos(phase_angle)
    illum_scale_2d = np.sin(phase_angle)
    illum_v_3d = illum_v_2d * illum_scale_2d
    illum_u_3d = illum_u_2d * illum_scale_2d

    # Normalize the 3D illumination direction (already unit up to rounding;
    # the guard covers a degenerate zero vector only).
    illum_mag = np.sqrt(illum_v_3d**2 + illum_u_3d**2 + illum_z**2)
    if illum_mag > 1e-10:
        illum_v_3d /= illum_mag
        illum_u_3d /= illum_mag
        illum_z_norm = illum_z / illum_mag
    else:
        illum_z_norm = 1.0  # Directly toward observer
    return float(illum_v_3d), float(illum_u_3d), float(illum_z_norm)


def lambert_from_normals(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    *,
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
    illum_v_3d, illum_u_3d, illum_z_norm = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )

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
