"""Image-side optical-depth ring-system renderer.

Draws the ``ring_system`` scene block: radial tau features (ringlets and
gaps) on mode-1 eccentric precessing orbits, projected through the shared
opening-angle geometry (:mod:`spindoctor.sim.ring_geometry`) and lit by the
single-scattering closed forms.  The output is a set of per-pixel maps --
emitted intensity, transmission ``exp(-tau/mu)``, and line-of-sight depth --
that the radiance stage composites against the body stack as a transmission
screen: ``img = I_ring + exp(-tau/mu) * img_behind``, evaluated far to near
per pixel, so low-tau features reveal the background instead of erasing it
and stars behind the ring attenuate physically.

Photometry (the normative equation set), with ``mu = |sin B_obs|``,
``mu0 = |sin B_sun|``, lit iff ``sign(B_obs) == sign(B_sun)``, and one-term
Henyey-Greenstein ``P(g, alpha)``:

- lit:   ``I = A/4 * P * mu0/(mu0 + mu) * (1 - exp(-tau*(1/mu0 + 1/mu)))``
- unlit: ``I = A/4 * P * mu0/(mu0 - mu) * (exp(-tau/mu0) - exp(-tau/mu))``,
  with the limit ``A/4 * P * (tau/mu) * exp(-tau/mu)`` when
  ``|mu0 - mu| < 1e-6``.

An opening angle of exactly 0 (either side) renders nothing.  The unlit
branch produces the real inversion: moderate-tau features bright from the
dark side, high-tau features nearly black.

The albedo ``A`` and asymmetry ``g`` are per-feature truth keys; where
features overlap radially the composed emission uses the tau-weighted mean
of their ``A/4 * P`` factors over the positive (ringlet) contributions,
which reduces exactly to the closed form for any single feature.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.sim.ring_geometry import (
    compute_antialiasing_shade,
    compute_edge_radii_array,
    ring_los_depth,
    ring_plane_from_sky,
    ring_radial_scale,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'RING_ALBEDO_DEFAULT',
    'RING_PHASE_G_DEFAULT',
    'RingSystemMaps',
    'henyey_greenstein_phase',
    'render_ring_system',
    'ring_reflection_factor',
]

# Single-scattering albedo (times normalization) default; dusty features set
# their own higher values per feature.
RING_ALBEDO_DEFAULT: float = 0.5
# Henyey-Greenstein asymmetry default: main-ring backscatter.  Dusty,
# forward-scattering features set g ~ +0.6 per feature.
RING_PHASE_G_DEFAULT: float = -0.3
# Below this |mu0 - mu| the unlit closed form is numerically indeterminate
# (0/0) and the analytic limit replaces it.
_UNLIT_MU_MATCH_TOL: float = 1e-6


@dataclass(frozen=True)
class RingSystemMaps:
    """Per-pixel maps of a rendered ring system on the (oversampled) grid.

    Parameters:
        intensity: Emitted ring brightness ``I`` per pixel (normalized signal
            units).
        transmission: ``exp(-tau/mu)`` per pixel: the fraction of background
            light (bodies behind the ring, stars, sky) that passes through.
        mask: Pixels where the system carries any optical depth.
        depth_km: Observer distance ``range_km - dlos_km`` per pixel, or None
            when the scene gives the system no ``range_km`` (the system then
            has no depth relation to bodies; overlap is a scene error).
    """

    intensity: NDArrayFloatType
    transmission: NDArrayFloatType
    mask: NDArrayBoolType
    depth_km: NDArrayFloatType | None


def henyey_greenstein_phase(g: float, alpha_deg: float) -> float:
    """One-term Henyey-Greenstein phase function at phase angle ``alpha``.

    ``P = (1 - g**2) / (1 + g**2 + 2*g*cos(alpha))**1.5`` with alpha the
    phase angle (0 at opposition): negative ``g`` backscatters (bright at
    low phase), positive ``g`` forward-scatters (dusty features brighten
    strongly toward alpha = 180).

    Parameters:
        g: Asymmetry parameter, strictly inside (-1, 1).
        alpha_deg: Phase angle in degrees, [0, 180].

    Returns:
        The phase function value (positive).
    """
    cos_alpha = math.cos(math.radians(alpha_deg))
    return float((1.0 - g * g) / (1.0 + g * g + 2.0 * g * cos_alpha) ** 1.5)


def ring_reflection_factor(
    tau: NDArrayFloatType,
    mu: float,
    mu0: float,
    *,
    lit: bool,
) -> NDArrayFloatType:
    """The geometric factor of the single-scattering closed forms.

    Multiplied by ``A/4 * P`` this is the emitted intensity: the lit form
    saturates toward ``mu0/(mu0 + mu)`` at high tau, while the unlit form
    peaks at moderate tau and falls to zero for an opaque ring (nothing
    diffuses through) -- the real dark-side inversion.

    Parameters:
        tau: Normal optical depth per pixel (non-negative).
        mu: ``|sin B_obs|``, nonzero.
        mu0: ``|sin B_sun|``, nonzero.
        lit: Whether the observer sees the lit face
            (``sign(B_obs) == sign(B_sun)``).

    Returns:
        The per-pixel geometric factor.
    """
    if lit:
        return mu0 / (mu0 + mu) * (1.0 - np.exp(-tau * (1.0 / mu0 + 1.0 / mu)))
    if abs(mu0 - mu) < _UNLIT_MU_MATCH_TOL:
        # The mu0 -> mu limit of the unlit form (the closed form is 0/0).
        return (tau / mu) * np.exp(-tau / mu)
    return mu0 / (mu0 - mu) * (np.exp(-tau / mu0) - np.exp(-tau / mu))


def render_ring_system(
    shape: tuple[int, int],
    ring_system: Mapping[str, Any],
    *,
    center_v: float,
    center_u: float,
    node_deg: float,
    time: float = 0.0,
    epoch: float = 0.0,
    oversample: int = 1,
) -> RingSystemMaps:
    """Render a ring system's per-pixel intensity/transmission/depth maps.

    The caller resolves the sky placement (planted offset, spacecraft-
    ephemeris parallax, camera roll -- a roll rotates the projected pattern,
    which is exactly ``node_deg`` plus the roll angle) and passes the final
    center and node; orbit radii and widths are detector-pixel scene values
    scaled to the render grid here.

    Parameters:
        shape: The (oversampled) render-grid shape ``(V*os, U*os)``.
        ring_system: The validated scene ``ring_system`` mapping.
        center_v: Ring-system center v on the render grid (offset applied).
        center_u: Ring-system center u on the render grid (offset applied).
        node_deg: Sky position angle of the ascending node, in degrees
            (camera roll already added).
        time: Scene time in TDB seconds (mode-1 pericenter precession).
        epoch: Ring epoch in TDB seconds.
        oversample: The render-grid oversampling factor; radii, widths, and
            the anti-aliasing window scale by it, and the depth map converts
            back through it so ``depth_km`` is grid-independent.

    Returns:
        The rendered :class:`RingSystemMaps`.

    Raises:
        ValueError: If a feature's kind is not a renderable profile (the
            validator rejects these; this guards direct callers).
    """
    size_v, size_u = shape
    geometry = ring_system.get('geometry') or {}
    b_obs = float(geometry.get('opening_deg_obs', 0.0))
    b_sun = float(geometry.get('opening_deg_sun', 0.0))
    zero_maps = RingSystemMaps(
        intensity=np.zeros(shape, dtype=np.float64),
        transmission=np.ones(shape, dtype=np.float64),
        mask=np.zeros(shape, dtype=np.bool_),
        depth_km=None,
    )
    # An exactly edge-on ring plane (either angle) renders nothing.
    if b_obs == 0.0 or b_sun == 0.0:
        return zero_maps
    features = ring_system.get('features') or []
    if not features:
        return zero_maps

    os = int(oversample)
    mu = abs(math.sin(math.radians(b_obs)))
    mu0 = abs(math.sin(math.radians(b_sun)))
    lit = (b_obs > 0.0) == (b_sun > 0.0)
    alpha_deg = float(ring_system.get('phase_deg', 0.0))

    # Pixel-center coordinates relative to the projected ring center, mapped
    # back to the ring plane through the shared inverse projection.
    v_coords = np.arange(size_v, dtype=np.float64) + 0.5
    u_coords = np.arange(size_u, dtype=np.float64) + 0.5
    v_grid, u_grid = np.meshgrid(v_coords, u_coords, indexing='ij')
    dv = v_grid - center_v
    du = u_grid - center_u
    r, lam, x, y = ring_plane_from_sky(dv, du, opening_deg_obs=b_obs, node_deg=node_deg)
    # Ring-plane radial distance per image pixel: dividing by this converts a
    # radial distance to image pixels, so anti-aliased edges span one detector
    # pixel regardless of the foreshortening direction.
    radial_scale = ring_radial_scale(r, x, y, opening_deg_obs=b_obs)

    # Compose the radial tau profile: ringlets add their tau between their
    # edges, gaps subtract (tau suppression), and the sum clips at zero.  The
    # emission's A/4 * P factor is the tau-weighted mean over the positive
    # contributions, so a single feature reproduces its closed form exactly.
    tau_map = np.zeros(shape, dtype=np.float64)
    ap_weighted = np.zeros(shape, dtype=np.float64)
    ap_weight = np.zeros(shape, dtype=np.float64)
    for feature in features:
        kind = str(feature.get('kind'))
        if kind not in ('ringlet', 'gap'):
            raise ValueError(f'ring feature kind {kind!r} is not renderable')
        orbit = feature.get('orbit') or {}
        a = float(orbit.get('a', 0.0)) * os
        ae = float(orbit.get('ae', 0.0)) * os
        long_peri = float(orbit.get('long_peri', 0.0))
        rate_peri = float(orbit.get('rate_peri', 0.0))
        width = float(feature.get('width', 0.0)) * os
        tau = float(feature.get('tau', 0.0))
        # Inner and outer edges share the orbit shape; the outer edge is the
        # same ellipse widened by the radial width.  lam is ring-plane
        # longitude, so long_peri lives in the ring-plane frame (the node
        # angle entered only the sky projection above).
        r_inner = compute_edge_radii_array(
            lam, a=a, ae=ae, long_peri=long_peri, rate_peri=rate_peri, epoch=epoch, time=time
        )
        r_outer = compute_edge_radii_array(
            lam,
            a=a + width,
            ae=ae,
            long_peri=long_peri,
            rate_peri=rate_peri,
            epoch=epoch,
            time=time,
        )
        inner_shade = compute_antialiasing_shade((r - r_inner) / radial_scale, float(os))
        outer_shade = compute_antialiasing_shade((r_outer - r) / radial_scale, float(os))
        coverage = np.minimum(inner_shade, outer_shade)
        contribution = tau * coverage
        if kind == 'gap':
            tau_map -= contribution
        else:
            tau_map += contribution
            albedo = float(feature.get('albedo', RING_ALBEDO_DEFAULT))
            phase_g = float(feature.get('phase_g', RING_PHASE_G_DEFAULT))
            ap = albedo / 4.0 * henyey_greenstein_phase(phase_g, alpha_deg)
            ap_weighted += ap * contribution
            ap_weight += contribution
    np.clip(tau_map, 0.0, None, out=tau_map)

    ap_map = np.divide(
        ap_weighted, ap_weight, out=np.zeros(shape, dtype=np.float64), where=ap_weight > 0.0
    )
    intensity = ap_map * ring_reflection_factor(tau_map, mu, mu0, lit=lit)
    transmission: NDArrayFloatType = np.exp(-tau_map / mu)
    mask: NDArrayBoolType = tau_map > 0.0

    depth_km: NDArrayFloatType | None = None
    range_km = ring_system.get('range_km')
    if range_km is not None:
        # dlos is positive toward the observer on the render grid; divide by
        # the oversampling to return to detector pixels before the km scale.
        km_per_pixel = float(ring_system.get('km_per_pixel', 1.0))
        dlos_km = ring_los_depth(y, opening_deg_obs=b_obs) / float(os) * km_per_pixel
        depth_km = float(range_km) - dlos_km

    return RingSystemMaps(
        intensity=intensity,
        transmission=transmission,
        mask=mask,
        depth_km=depth_km,
    )
