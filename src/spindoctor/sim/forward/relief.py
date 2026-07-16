"""Truth-side body relief: the limb/terminator topography field.

The relief is a 2-D field on the body surface, not a 1-D field on the limb,
so the limb perturbation and the terminator shadowing are slices of one
consistent surface.  ``h(lat, lon)`` is fractional relief (height / local
radius): a periodic 2-D Gaussian random field synthesized by 2-D FFT on a
(lat, lon) grid.  The spectral coefficients are independent complex
Gaussians with variance ``S(k) proportional to exp(-(|k| * corr_rad / 2)**2)``,
where ``|k|`` is the total angular wavenumber and ``corr_rad`` is the
correlation length in radians of surface arc; the band limit is
``kmax = ceil(8 / corr_rad)``, where S has fallen to ~1e-7 of its peak.

Modes with total wavenumber below 3 are zeroed: the degree-1 content of a
radius perturbation is, to first order, a *translation* of the body -- a
planted, untruthed center offset no limb fit could distinguish from the
pointing error -- and degree-2 content aliases ellipsoid shape error, which
is its own scene knob.  After zeroing, the field is rescaled so the *limb
slice's* standard deviation equals the commanded RMS per-realization.

**Coordinate convention.**  The field's poles sit on the observer axis, so
the sub-observer horizon circle (the limb) is the field's equator: latitude
is elevation out of the limb plane toward the observer, longitude is
azimuth around the limb.  On the equator the (lat, lon) grid metric is
exact, so the limb slice -- the statistic the scene commands -- carries the
commanded RMS and correlation length with no map distortion.  The known
cost is the standard lat-lon caveat: high field latitudes (near the disc
center, and the terminator's approach to it at high phase) are
over-corrugated in longitude by ``~cos(lat)``.  At the default 15-degree
correlation length the affected caps are small; the contract fixes the
spectrum and the limb-slice statistics, not the synthesis mesh.

The terminator shadow march (:func:`march_shadows`) works in *absolute*
heights ``H = h * R`` in pixels (R the local body radius in pixels): the
fractional field (~0.01) and surface distances (pixels) are never compared
directly.  A point is shadowed iff some upstream sample at surface distance
``d`` toward the sun satisfies ``H_up - H_pt > d / tan(i_pt)``, with
``i_pt`` the point's incidence angle.  The march is capped at
``d_max = min((H_max - H_min) * tan(i_pt), sqrt(2 * R * H_max))``: the
first term is the longest shadow the terrain can cast, the second the
horizon limit that bounds the tangent's divergence at the terminator
itself, so cost is bounded and the geometry stays physical.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['MarchStats', 'ReliefField', 'march_shadows', 'synthesize_relief_field']

# Spectral modes with total angular wavenumber below this are zeroed
# (degree 1 = body translation, degree 2 = ellipsoid shape aliasing).
_LOW_DEGREE_CUTOFF = 3.0
# Band limit factor: kmax = ceil(8 / corr_rad); S(kmax) is ~1e-7 of peak.
_BAND_LIMIT_FACTOR = 8.0
# Synthesis grid bounds: at least 64 for a usable limb slice, at most 8192
# so a degenerate correlation length cannot demand an unbounded FFT.
_MIN_GRID = 64
_MAX_GRID = 8192


@dataclass(frozen=True)
class ReliefField:
    """One realization of the fractional relief field ``h(lat, lon)``.

    Parameters:
        grid: ``(n, n)`` periodic field over the full (lat, lon) torus; row
            ``i`` is latitude ``-pi + 2*pi*i/n`` (the limb equator is row
            ``n // 2``), column ``j`` is longitude ``2*pi*j/n``.
        rms: The commanded limb-slice standard deviation (h/R, unitless).
        corr_deg: The correlation length in degrees of surface arc.
        h_min: Global field minimum (fractional).
        h_max: Global field maximum (fractional).
    """

    grid: NDArrayFloatType
    rms: float
    corr_deg: float
    h_min: float
    h_max: float

    def sample(self, lat: NDArrayFloatType, lon: NDArrayFloatType) -> NDArrayFloatType:
        """Bilinearly sample the periodic field at (lat, lon) in radians.

        Parameters:
            lat: Latitudes above the limb plane (any shape; wrapped).
            lon: Longitudes around the limb (same shape as ``lat``; wrapped).

        Returns:
            Fractional relief values, same shape as the inputs.
        """
        n = self.grid.shape[0]
        rows = (np.asarray(lat, dtype=np.float64) + np.pi) * (n / (2.0 * np.pi))
        cols = np.asarray(lon, dtype=np.float64) * (n / (2.0 * np.pi))
        sampled = ndimage.map_coordinates(
            self.grid, [rows, cols], order=1, mode='grid-wrap', prefilter=False
        )
        return np.asarray(sampled, dtype=np.float64)

    def limb_slice(self) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """The limb slice ``h(0, phi)`` as interpolation nodes.

        Returns:
            Tuple ``(phi_nodes, values)``: longitudes ``0 .. 2*pi`` inclusive
            (the final node repeats the first, closing the circle) and the
            field values on the equator row, ready for ``np.interp``.
        """
        n = self.grid.shape[0]
        values = np.concatenate([self.grid[n // 2], self.grid[n // 2, :1]])
        phi_nodes = np.linspace(0.0, 2.0 * np.pi, n + 1)
        return phi_nodes, values

    def limb_delta(self, phi: NDArrayFloatType) -> NDArrayFloatType:
        """The limb perturbation ``delta(phi)`` at azimuths ``phi`` (radians).

        Parameters:
            phi: Azimuths around the limb (any real values; wrapped to
                ``[0, 2*pi)``).

        Returns:
            Fractional limb displacement values, same shape as ``phi``.
        """
        phi_nodes, values = self.limb_slice()
        wrapped = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * np.pi)
        return np.interp(wrapped, phi_nodes, values)


@lru_cache(maxsize=8)
def _unit_relief_grid(corr_deg: float, seed: int) -> NDArrayFloatType:
    """A field realization normalized to unit limb-slice standard deviation.

    The RMS knob scales the field linearly, so realizations are cached per
    (correlation length, seed) at unit limb RMS and scaled on request.

    Parameters:
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed (fresh Gaussian coefficient draws per seed).

    Returns:
        The ``(n, n)`` unit-normalized field grid.  All-zero when the band
        ``[3, kmax]`` is empty (a correlation length so long that no mode
        survives the low-degree zeroing).
    """
    corr_rad = math.radians(corr_deg)
    kmax = math.ceil(_BAND_LIMIT_FACTOR / corr_rad)
    n = int(min(_MAX_GRID, max(_MIN_GRID, 2 ** math.ceil(math.log2(max(4 * kmax, 1))))))

    # White Gaussian noise shaped in the spectral domain: the FFT of white
    # noise is a field of independent complex Gaussians (with the Hermitian
    # symmetry a real field requires), so multiplying by sqrt(S(k)) gives
    # coefficients with variance proportional to S(k).
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    spectrum = np.fft.rfft2(white)

    k_lat = np.fft.fftfreq(n, d=1.0 / n)
    k_lon = np.fft.rfftfreq(n, d=1.0 / n)
    k_total = np.hypot(k_lat[:, np.newaxis], k_lon[np.newaxis, :])
    amplitude = np.exp(-((k_total * corr_rad / 2.0) ** 2) / 2.0)
    amplitude[k_total < _LOW_DEGREE_CUTOFF] = 0.0
    amplitude[k_total > kmax] = 0.0

    field = np.fft.irfft2(spectrum * amplitude, s=(n, n))
    limb_std = float(np.std(field[n // 2]))
    if limb_std <= 0.0:
        return np.zeros((n, n), dtype=np.float64)
    return np.asarray(field / limb_std, dtype=np.float64)


def synthesize_relief_field(rms: float, corr_deg: float, seed: int) -> ReliefField:
    """Synthesize one relief-field realization.

    Parameters:
        rms: Commanded limb-slice standard deviation (h/R, unitless);
            0 yields the all-zero field.
        corr_deg: Correlation length in degrees of surface arc.
        seed: The realization seed, derived from the body's seed chain.

    Returns:
        The scaled :class:`ReliefField` realization.
    """
    grid = _unit_relief_grid(float(corr_deg), int(seed)) * float(rms)
    return ReliefField(
        grid=grid,
        rms=float(rms),
        corr_deg=float(corr_deg),
        h_min=float(grid.min()),
        h_max=float(grid.max()),
    )


def clear_relief_caches() -> None:
    """Drop the cached field realizations (test/tooling helper)."""
    _unit_relief_grid.cache_clear()


@dataclass(frozen=True)
class MarchStats:
    """Diagnostics of one terminator shadow march.

    Parameters:
        candidate_count: Number of points the march started from.
        steps_executed: Surface-arc steps actually executed (the loop count).
        max_steps: The global step bound implied by the march cap; the loop
            can never exceed it regardless of the domain's extent.
    """

    candidate_count: int
    steps_executed: int
    max_steps: int


def march_shadows(
    start_v: NDArrayFloatType,
    start_u: NDArrayFloatType,
    *,
    h_point: NDArrayFloatType,
    tan_incidence: NDArrayFloatType,
    radius_px: NDArrayFloatType,
    height_map: NDArrayFloatType,
    arc_per_px_map: NDArrayFloatType,
    domain_mask: NDArrayBoolType,
    h_max: float,
    h_min: float,
    sun_v: float,
    sun_u: float,
) -> tuple[NDArrayBoolType, MarchStats]:
    """March candidate points toward the sun and flag the shadowed ones.

    Each candidate steps toward the sun in units of one pixel of *surface*
    arc (the image-plane step is ``1 / arc_per_px`` at the current sample,
    so foreshortening near the limb is corrected: an image pixel there spans
    a large surface step).  A candidate is shadowed iff an upstream sample
    at surface distance ``d`` satisfies ``H_up - H_pt > d / tan(i_pt)``.
    The per-point cap ``d_max = min((H_max - H_min) * tan(i_pt),
    sqrt(2 * R * H_max))`` bounds the work; ``H_max``/``H_min`` are the
    field's global extremes at the point's local radius.

    Parameters:
        start_v: Candidate image-row positions (work-grid pixels, float).
        start_u: Candidate image-column positions, same shape.
        h_point: Fractional relief at each candidate's surface point.
        tan_incidence: Tangent of each candidate's incidence angle (> 0).
        radius_px: Local body radius at each candidate, in work pixels.
        height_map: Absolute heights ``h * R_local`` (work px) sampled at
            each domain pixel's surface point; zero outside the domain.
        arc_per_px_map: Surface arc (work px) spanned by one image pixel
            along the sun direction, per domain pixel.
        domain_mask: Pixels where the maps are valid; a ray leaving the
            domain stops marching (the caller sizes the domain to cover the
            longest capped march, so no caster can sit beyond it).
        h_max: Global field maximum (fractional).
        h_min: Global field minimum (fractional).
        sun_v: Image-plane unit direction toward the sun, row component.
        sun_u: Image-plane unit direction toward the sun, column component.

    Returns:
        Tuple of (shadow flags per candidate, :class:`MarchStats`).
    """
    n = int(start_v.size)
    shadow = np.zeros(n, dtype=np.bool_)
    if n == 0 or h_max <= h_min:
        return shadow, MarchStats(candidate_count=n, steps_executed=0, max_steps=0)

    size_v, size_u = height_map.shape
    tan_i = np.maximum(np.asarray(tan_incidence, dtype=np.float64), 1e-12)
    radius = np.asarray(radius_px, dtype=np.float64)
    height_pt = np.asarray(h_point, dtype=np.float64) * radius
    d_max = np.minimum(
        (h_max - h_min) * radius * tan_i,
        radius * math.sqrt(max(2.0 * h_max, 0.0)),
    )
    max_steps = math.ceil(float(d_max.max())) if d_max.size else 0

    # Live-set arrays, compressed as candidates resolve.
    index = np.arange(n)
    pos_v = np.asarray(start_v, dtype=np.float64).copy()
    pos_u = np.asarray(start_u, dtype=np.float64).copy()
    cur_v = np.clip(np.floor(pos_v).astype(np.intp), 0, size_v - 1)
    cur_u = np.clip(np.floor(pos_u).astype(np.intp), 0, size_u - 1)
    distance = np.zeros(n, dtype=np.float64)

    steps = 0
    while index.size and steps < max_steps:
        steps += 1
        # One pixel of surface arc; the image step shrinks where an image
        # pixel spans more surface (foreshortening near the limb).
        arc = np.maximum(arc_per_px_map[cur_v, cur_u], 0.1)
        step_px = 1.0 / arc
        pos_v = pos_v + sun_v * step_px
        pos_u = pos_u + sun_u * step_px
        distance = distance + 1.0

        cur_v = np.floor(pos_v).astype(np.intp)
        cur_u = np.floor(pos_u).astype(np.intp)
        inside = (cur_v >= 0) & (cur_v < size_v) & (cur_u >= 0) & (cur_u < size_u)
        cur_v = np.clip(cur_v, 0, size_v - 1)
        cur_u = np.clip(cur_u, 0, size_u - 1)
        inside &= domain_mask[cur_v, cur_u]

        # Samples beyond the point's cap cannot shadow it: the terrain term
        # makes such hits impossible algebraically, and the horizon term
        # excludes terrain the body's curvature hides.
        hit = (
            inside & (distance <= d_max) & (height_map[cur_v, cur_u] - height_pt > distance / tan_i)
        )
        shadow[index[hit]] = True

        keep = inside & ~hit & (distance < d_max)
        if not keep.all():
            index = index[keep]
            pos_v = pos_v[keep]
            pos_u = pos_u[keep]
            cur_v = cur_v[keep]
            cur_u = cur_u[keep]
            distance = distance[keep]
            tan_i = tan_i[keep]
            height_pt = height_pt[keep]
            d_max = d_max[keep]

    return shadow, MarchStats(candidate_count=n, steps_executed=steps, max_steps=max_steps)
