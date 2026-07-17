"""Image-side atmosphere rendering for haze-limb (Titan-class) bodies.

A body carrying an ``atmosphere`` block gains an exponential haze layer above
its surface, composited onto the body radiance by :func:`apply_atmosphere`
after the disc is shaded.  The haze is a truth key the navigator never sees:
its predicted-body model keeps a hard limb at the reference radius, so the
soft rendered limb is a designed mismatch (the substrate for the Titan
altitude-versus-phase problem).

**Tangent optical depth.**  A line of sight grazing the body at tangent
altitude ``h`` (pixels above the reference radius) accumulates a slant
optical depth

    ``tau(h) = tau_ref * exp(-(h - ref_altitude_px) / scale_height_px)``

so ``tau_ref`` is the tangent optical depth at ``ref_altitude_px``.  An
optional detached haze shell adds a Gaussian bump in ``tau`` centred at
``detached_px`` above the surface.

**Single scattering.**  The emergent haze brightness is a source term times
an opacity term.  The source is a single-scattering albedo scaled by a
Henyey-Greenstein phase factor (forward-scattering when ``g`` > 0, which
brightens the limb at high phase and, in the limit, produces the ring of
light past phase 150 deg) and a wrapped illumination weight that stays
positive a scale-height's worth of arc past the terminator (so the
terminator brightens past 90 deg incidence instead of cutting off).  The
opacity is ``1 - exp(-tau)``: above the limb ``tau`` is the tangent optical
depth, so the limb becomes a soft exponential ramp whose apparent radius
grows with the haze brightness (hence with phase); on the disc the slant
optical depth grows toward the limb as ``1 / cos(emission)``, so the haze
concentrates at the limb and stays faint at disc centre.

A body without an ``atmosphere`` block never calls into this module and
renders hard-limbed, byte-for-byte as before.
"""

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from spindoctor.sim.ellipsoid_geometry import illumination_vector
from spindoctor.support.types import NDArrayFloatType

__all__ = ['AtmosphereSpec', 'apply_atmosphere', 'atmosphere_spec_from_params', 'hg_phase_factor']

# Single-scattering brightness scale of a fully lit, optically thick grazing
# haze column (before the phase factor and illumination weight).  The haze
# emergent intensity is clipped into the [0, 1] signal plane, so a strong
# forward-scattering peak saturates by design.
_HAZE_ALBEDO = 0.6

# Amplitude of a detached shell's Gaussian tau bump, as a multiple of
# tau_ref; the shell's radial width is one scale height.
_SHELL_AMPLITUDE = 1.0

# The tangent optical depth below which the above-limb glow is treated as
# black: it bounds the rendered halo (and therefore the cost) to a limb band
# a few scale heights deep rather than the whole frame.
_TAU_EPS = 1e-3

# Floor on cos(emission) for the on-disc slant path, so the limb (where the
# emergent-cosine vanishes) saturates smoothly instead of dividing by zero.
_MU_FLOOR = 1e-3


@dataclass(frozen=True)
class AtmosphereSpec:
    """The exponential haze layer of one atmospheric body.

    All pixel quantities are in units of the render grid the haze is
    composited on (the oversampled grid when the scene oversamples), matching
    the body's already-scaled semi-axes.

    Parameters:
        scale_height_px: Haze e-folding scale height in pixels (> 0).
        tau_ref: Tangent optical depth at ``ref_altitude_px`` (> 0).
        ref_altitude_px: Altitude above the reference radius at which the
            tangent optical depth equals ``tau_ref``, in pixels.
        g: Henyey-Greenstein asymmetry parameter in (-1, 1); positive is
            forward-scattering (bright limb at high phase).
        detached_px: Altitude of an optional detached haze shell above the
            surface, in pixels; None for no shell.
    """

    scale_height_px: float
    tau_ref: float
    ref_altitude_px: float = 0.0
    g: float = 0.0
    detached_px: float | None = None


def atmosphere_spec_from_params(
    body_params: dict[str, object], *, oversample: int
) -> AtmosphereSpec | None:
    """Build an :class:`AtmosphereSpec` from a body's ``atmosphere`` block.

    Returns None when the body carries no atmosphere, so a body without the
    block never enters the haze path.  Pixel lengths are scaled to the render
    grid by ``oversample`` exactly as the body's axes are.

    Parameters:
        body_params: One scene body entry.
        oversample: The render grid's oversampling factor.

    Returns:
        The scaled spec, or None when the body has no ``atmosphere`` block.
    """
    atmosphere = body_params.get('atmosphere')
    if not isinstance(atmosphere, dict):
        return None
    os_factor = max(1, int(oversample))
    detached = atmosphere.get('detached_px')
    return AtmosphereSpec(
        scale_height_px=float(atmosphere['scale_height_px']) * os_factor,
        tau_ref=float(atmosphere['tau_ref']),
        ref_altitude_px=float(atmosphere.get('ref_altitude_px', 0.0)) * os_factor,
        g=float(atmosphere.get('g', 0.0)),
        detached_px=(float(detached) * os_factor if detached is not None else None),
    )


def hg_phase_factor(g: float, phase_angle: float) -> float:
    """The Henyey-Greenstein phase factor at a phase angle (1 at ``g`` = 0).

    Single scattering turns light through a scattering angle
    ``Theta = pi - phase``, so ``cos(Theta) = -cos(phase)``; a positive ``g``
    peaks at ``Theta`` = 0 (phase = pi), which is why forward-scattering haze
    is brightest at high phase.  The factor is normalized to 1 at ``g`` = 0
    for every phase, so it is a pure angular modulation of the haze source.

    Parameters:
        g: Henyey-Greenstein asymmetry parameter in (-1, 1).
        phase_angle: Phase angle in radians.

    Returns:
        The multiplicative phase factor (> 0).
    """
    cos_theta = -math.cos(phase_angle)
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
    return float((1.0 - g * g) / max(denom, 1e-12))


def tangent_optical_depth(h_px: NDArrayFloatType, spec: AtmosphereSpec) -> NDArrayFloatType:
    """Tangent (slant) optical depth of the haze at tangent altitude ``h_px``.

    The exponential column plus, when the spec carries one, a Gaussian
    detached-shell bump one scale height wide centred at ``detached_px``.

    Parameters:
        h_px: Tangent altitude above the reference radius, in pixels.
        spec: The haze spec.

    Returns:
        The tangent optical depth at each altitude.
    """
    scale_height = max(spec.scale_height_px, 1e-6)
    tau = spec.tau_ref * np.exp(-(h_px - spec.ref_altitude_px) / scale_height)
    if spec.detached_px is not None:
        bump = (h_px - spec.detached_px) / scale_height
        tau = tau + spec.tau_ref * _SHELL_AMPLITUDE * np.exp(-0.5 * bump * bump)
    return tau


def _outer_altitude(spec: AtmosphereSpec) -> float:
    """The tangent altitude beyond which the above-limb glow is negligible.

    Where the smooth column has fallen to :data:`_TAU_EPS`, plus a detached
    shell's reach (its centre and three scale heights).

    Parameters:
        spec: The haze spec.

    Returns:
        The bounding tangent altitude in pixels.
    """
    scale_height = max(spec.scale_height_px, 1e-6)
    reach = spec.ref_altitude_px + scale_height * math.log(max(spec.tau_ref / _TAU_EPS, 1.0))
    if spec.detached_px is not None:
        reach = max(reach, spec.detached_px + 3.0 * scale_height)
    return max(reach, scale_height)


def apply_atmosphere(
    body_shape: NDArrayFloatType,
    spec: AtmosphereSpec,
    *,
    center_v: float,
    center_u: float,
    semi_a: float,
    semi_b: float,
    semi_c: float,
    rotation_z: float,
    rotation_tilt: float,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Composite the haze layer onto a reference-centred body radiance.

    The haze is added to the disc radiance over a limb band a few scale
    heights deep, so a large frame costs no more than the body's own
    footprint.  The returned array is a new array; the input is never mutated
    (it may be a shared render cache entry).

    Parameters:
        body_shape: The shaded body radiance at the reference centre, in
            [0, 1], 0 outside the body.
        spec: The haze spec (pixel lengths already on this grid).
        center_v: Body centre v the shape was rendered at, in grid pixels.
        center_u: Body centre u the shape was rendered at, in grid pixels.
        semi_a: Semi-axis a in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Depth semi-axis c in grid pixels.
        rotation_z: In-plane rotation about the viewing axis, radians.
        rotation_tilt: Tilt toward/away from the viewer, radians.
        illumination_angle: In-plane light direction, radians (0 = top).
        phase_angle: Phase angle in radians (0 fully lit, pi backlit).

    Returns:
        The body radiance with the haze composited, clipped to [0, 1].
    """
    out = np.array(body_shape, dtype=np.float64, copy=True)
    semi_a = max(semi_a, 1e-6)
    semi_b = max(semi_b, 1e-6)
    r_mean = 0.5 * (semi_a + semi_b)
    scale_height = max(spec.scale_height_px, 1e-6)

    size_v, size_u = out.shape
    v_ctr, u_ctr = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
    v_ctr += 0.5 - center_v
    u_ctr += 0.5 - center_u

    cos_rz = math.cos(rotation_z)
    sin_rz = math.sin(rotation_z)
    cos_rt = math.cos(rotation_tilt)
    v_rot = (v_ctr * cos_rz - u_ctr * sin_rz) * cos_rt
    u_rot = v_ctr * sin_rz + u_ctr * cos_rz
    e2 = (v_rot / semi_a) ** 2 + (u_rot / semi_b) ** 2

    # Restrict all heavy work to the limb band: the disc plus a halo out to
    # where the tangent glow vanishes.  This bounds the cost to the body's
    # footprint rather than the frame.
    e_outer = 1.0 + _outer_altitude(spec) / r_mean
    band = e2 <= e_outer * e_outer
    if not band.any():
        return out

    e2_b = e2[band]
    e_b = np.sqrt(e2_b)
    h_b = (e_b - 1.0) * r_mean
    v_rot_b = v_rot[band]
    u_rot_b = u_rot[band]
    v_ctr_b = v_ctr[band]
    u_ctr_b = u_ctr[band]
    inside = e2_b < 1.0

    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )

    # Solar elevation (radians) driving the wrapped illumination weight: from
    # the surface incidence on the disc, and from the limb-azimuth incidence
    # in the halo (where there is no surface, only the atmospheric column
    # standing above the limb at that azimuth).
    mu0 = np.zeros_like(e_b)
    if inside.any():
        mu0[inside] = _disc_incidence(
            v_rot_b[inside],
            u_rot_b[inside],
            e2_b[inside],
            semi_a=semi_a,
            semi_b=semi_b,
            semi_c=semi_c,
            cos_rz=cos_rz,
            sin_rz=sin_rz,
            illum_v=illum_v,
            illum_u=illum_u,
            illum_z=illum_z,
        )
    outside = ~inside
    if outside.any():
        rho = np.hypot(v_ctr_b[outside], u_ctr_b[outside])
        rho = np.maximum(rho, 1e-9)
        mu0[outside] = (v_ctr_b[outside] * illum_v + u_ctr_b[outside] * illum_u) / rho

    delta_wrap = max(math.sqrt(2.0 * scale_height / r_mean), 0.05)
    elevation = np.arcsin(np.clip(mu0, -1.0, 1.0))
    illum_weight = 0.5 * (1.0 + np.tanh(elevation / delta_wrap))

    source = _HAZE_ALBEDO * hg_phase_factor(spec.g, phase_angle) * illum_weight

    # Opacity: the tangent glow above the limb, and the slant column on the
    # disc (concentrated at the limb as 1 / cos(emission)).
    opacity = np.zeros_like(e_b)
    if outside.any():
        tau_out = tangent_optical_depth(h_b[outside], spec)
        opacity[outside] = 1.0 - np.exp(-tau_out)
    if inside.any():
        mu_emit = np.sqrt(np.maximum(1.0 - e2_b[inside], 0.0))
        # The tangent optical depth at ref altitude is deeper than the
        # vertical column by the usual grazing factor sqrt(2 pi R / H); invert
        # it so the on-disc column is the physical vertical depth.
        geom = math.sqrt(2.0 * math.pi * r_mean / scale_height)
        tau_vert = spec.tau_ref / max(geom, 1e-6)
        tau_slant = tau_vert / np.maximum(mu_emit, _MU_FLOOR)
        opacity[inside] = 1.0 - np.exp(-tau_slant)

    haze = source * opacity
    out[band] = np.clip(out[band] + haze, 0.0, 1.0)
    return out


def _disc_incidence(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    e2: NDArrayFloatType,
    *,
    semi_a: float,
    semi_b: float,
    semi_c: float,
    cos_rz: float,
    sin_rz: float,
    illum_v: float,
    illum_u: float,
    illum_z: float,
) -> NDArrayFloatType:
    """Cosine of the surface incidence angle at on-disc haze pixels.

    The ellipsoid surface normal in image coordinates (rotated back from the
    body frame) dotted with the illumination direction; shared shading
    conventions are not required here because the haze never crosses the
    information boundary, but the same normal construction keeps the haze
    aligned with the disc shading.

    Parameters:
        v_rot: Rotated-frame v coordinate of each pixel.
        u_rot: Rotated-frame u coordinate of each pixel.
        e2: Squared normalized ellipse radial function at each pixel.
        semi_a: Semi-axis a in grid pixels.
        semi_b: Semi-axis b in grid pixels.
        semi_c: Depth semi-axis c in grid pixels.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        illum_v: V component of the unit illumination direction.
        illum_u: U component of the unit illumination direction.
        illum_z: Z component of the unit illumination direction.

    Returns:
        ``cos(incidence)`` per pixel (negative on the night side).
    """
    z = semi_c * np.sqrt(np.maximum(1.0 - e2, 0.0))
    nv_local = v_rot / (semi_a * semi_a)
    nu_local = u_rot / (semi_b * semi_b)
    nz_local = z / (semi_c * semi_c)
    mag = np.sqrt(nv_local**2 + nu_local**2 + nz_local**2)
    mag = np.maximum(mag, 1e-12)
    nv_local /= mag
    nu_local /= mag
    nz_local /= mag
    normal_v = nv_local * cos_rz + nu_local * sin_rz
    normal_u = -nv_local * sin_rz + nu_local * cos_rz
    return cast(NDArrayFloatType, normal_v * illum_v + normal_u * illum_u + nz_local * illum_z)
