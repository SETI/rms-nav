"""Statistics of the body relief field and the terminator shadow march.

The relief-field contract pins the spectrum and the limb-slice statistics:
the limb slice's standard deviation equals the commanded RMS
per-realization, no spectral power sits below total wavenumber 3 (degree 1
is a body translation, degree 2 aliases ellipsoid shape error) or above the
band limit, and distinct seeds are distinct terrains.  The shadow march is
pinned by its inequality (``H_up - H_pt > d / tan(i)``) and its work cap
``d_max = min((H_max - H_min) * tan(i), sqrt(2 * R * H_max))``.
"""

import math
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.relief import (
    march_shadows,
    synthesize_relief_field,
)

_RMS = 0.01
_CORR_DEG = 15.0


def test_limb_slice_rms_matches_commanded_over_100_seeds() -> None:
    """Every realization's limb-slice standard deviation equals the commanded RMS.

    The field is rescaled per-realization after the low-degree zeroing, so
    each seed's limb slice carries the commanded RMS; 100 seeds bound the
    worst case well inside the 10% acceptance band.
    """
    worst = 0.0
    for seed in range(100):
        field = synthesize_relief_field(_RMS, _CORR_DEG, seed)
        n = field.grid.shape[0]
        limb_std = float(np.std(field.grid[n // 2]))
        worst = max(worst, abs(limb_std - _RMS) / _RMS)
    assert worst < 0.10
    # The per-realization normalization makes the match essentially exact.
    assert worst < 1e-9


def test_low_degree_power_is_zero() -> None:
    """Spectral power at total wavenumber below 3 is exactly zero."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 7)
    n = field.grid.shape[0]
    spectrum = np.fft.rfft2(field.grid)
    k_lat = np.fft.fftfreq(n, d=1.0 / n)
    k_lon = np.fft.rfftfreq(n, d=1.0 / n)
    k_total = np.hypot(k_lat[:, np.newaxis], k_lon[np.newaxis, :])
    low = np.abs(spectrum[k_total < 3.0])
    assert low.size > 0
    assert float(low.max()) == pytest.approx(0.0, abs=1e-9 * float(np.abs(spectrum).max()))


def test_power_beyond_band_limit_is_zero() -> None:
    """Spectral power beyond kmax = ceil(8 / corr_rad) is exactly zero."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 7)
    n = field.grid.shape[0]
    kmax = math.ceil(8.0 / math.radians(_CORR_DEG))
    spectrum = np.fft.rfft2(field.grid)
    k_lat = np.fft.fftfreq(n, d=1.0 / n)
    k_lon = np.fft.rfftfreq(n, d=1.0 / n)
    k_total = np.hypot(k_lat[:, np.newaxis], k_lon[np.newaxis, :])
    high = np.abs(spectrum[k_total > kmax])
    assert high.size > 0
    assert float(high.max()) == pytest.approx(0.0, abs=1e-9 * float(np.abs(spectrum).max()))


def test_band_power_is_present() -> None:
    """The surviving band [3, kmax] actually carries the field's power."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 7)
    assert float(np.std(field.grid)) > 0.0
    assert field.h_max > 0.0
    assert field.h_min < 0.0


def test_limb_correlation_length_tracks_the_knob() -> None:
    """A longer correlation length yields a limb slice with fewer zero crossings."""

    def crossings(corr_deg: float) -> int:
        field = synthesize_relief_field(_RMS, corr_deg, 11)
        _phi, values = field.limb_slice()
        signs = np.sign(values[:-1])
        return int(np.sum(signs[1:] != signs[:-1]))

    assert crossings(30.0) < crossings(5.0)


def test_distinct_seeds_are_distinct_terrains() -> None:
    """Fresh coefficient draws per seed: two seeds never share a realization."""
    first = synthesize_relief_field(_RMS, _CORR_DEG, 1)
    second = synthesize_relief_field(_RMS, _CORR_DEG, 2)
    assert not np.array_equal(first.grid, second.grid)


def test_same_seed_is_deterministic() -> None:
    """The same (rms, corr, seed) triple reproduces the same field exactly."""
    first = synthesize_relief_field(_RMS, _CORR_DEG, 3)
    second = synthesize_relief_field(_RMS, _CORR_DEG, 3)
    assert np.array_equal(first.grid, second.grid)


def test_rms_zero_is_the_zero_field() -> None:
    """RMS 0 (the default) is exactly the flat, relief-free surface."""
    field = synthesize_relief_field(0.0, _CORR_DEG, 5)
    assert float(np.abs(field.grid).max()) == 0.0


def test_limb_delta_interpolates_the_equator_row() -> None:
    """limb_delta at node longitudes returns the equator row values exactly."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 9)
    n = field.grid.shape[0]
    phi = np.arange(n) * (2.0 * np.pi / n)
    np.testing.assert_allclose(field.limb_delta(phi), field.grid[n // 2], atol=1e-12)


def test_sample_wraps_periodically() -> None:
    """Sampling wraps in both coordinates (the field lives on a torus)."""
    field = synthesize_relief_field(_RMS, _CORR_DEG, 9)
    lat = np.array([0.3])
    lon = np.array([1.1])
    np.testing.assert_allclose(
        field.sample(lat, lon), field.sample(lat + 2.0 * np.pi, lon + 2.0 * np.pi), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Shadow march: inequality and cap.
# ---------------------------------------------------------------------------


def _flat_maps(
    size: int, *, ridge_col: int | None, ridge_h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat unit-arc terrain maps with an optional tall ridge column."""
    height = np.zeros((size, size), dtype=np.float64)
    if ridge_col is not None:
        height[:, ridge_col] = ridge_h
    arc = np.ones((size, size), dtype=np.float64)
    domain = np.ones((size, size), dtype=np.bool_)
    return height, arc, domain


def test_march_shadow_inequality_both_sides() -> None:
    """A ridge shadows a downstream point iff it is within (H_up - H_pt) * tan(i).

    The candidate sits at height 0 with the sun along +u; a ridge of height
    H sits ``d`` pixels upstream.  The point is shadowed exactly when
    ``H > d / tan(i)``, i.e. when ``d < H * tan(i)``.
    """
    ridge_h = 5.0
    tan_i = 4.0
    # Shadow reach: H * tan(i) = 20 px.  Candidate at column 30 with the
    # ridge at column 45 is 15 px downstream (shadowed); at column 60 the
    # ridge is 30 px upstream (beyond reach, lit).
    height, arc, domain = _flat_maps(96, ridge_col=45, ridge_h=ridge_h)
    common: dict[str, Any] = {
        'h_point': np.array([0.0, 0.0]),
        'tan_incidence': np.array([tan_i, tan_i]),
        'radius_px': np.array([100.0, 100.0]),
        'height_map': height,
        'arc_per_px_map': arc,
        'domain_mask': domain,
        'h_max': ridge_h / 100.0,
        'h_min': 0.0,
        'sun_v': 0.0,
        'sun_u': 1.0,
    }
    shadow, stats = march_shadows(np.array([48.0, 48.0]), np.array([30.5, 15.5]), **common)
    assert stats.candidate_count == 2
    assert bool(shadow[0])
    assert not bool(shadow[1])


def test_march_cap_bounds_the_work() -> None:
    """The march never exceeds the cap even when the domain invites it.

    A near-terminator candidate (tan(i) enormous) on a huge flat domain
    would march thousands of steps naively; the horizon term
    ``sqrt(2 * R * H_max)`` caps the step count at its predicted bound.
    """
    size = 4096
    height = np.zeros((8, size), dtype=np.float64)
    arc = np.ones((8, size), dtype=np.float64)
    domain = np.ones((8, size), dtype=np.bool_)
    radius = 200.0
    h_max = 0.02
    # Horizon cap: R * sqrt(2 * h_max) = 40 px; the terrain term
    # (h_max - h_min) * R * tan(i) = 4 * 1e6 px would run off the domain.
    shadow, stats = march_shadows(
        np.array([4.0]),
        np.array([2.0]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([1e6]),
        radius_px=np.array([radius]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=h_max,
        h_min=-h_max,
        sun_v=0.0,
        sun_u=1.0,
    )
    predicted_cap = radius * math.sqrt(2.0 * h_max)
    assert stats.max_steps == math.ceil(predicted_cap)
    assert stats.steps_executed <= stats.max_steps
    assert not bool(shadow[0])


def test_march_ignores_terrain_beyond_the_horizon_cap() -> None:
    """A ridge past the horizon cap cannot shadow, however tall the tangent says.

    With tan(i) huge the terrain term alone would let a 60-px-distant ridge
    shadow the point; the horizon term (40 px here) excludes it.
    """
    radius = 200.0
    h_max = 0.02
    height, arc, domain = _flat_maps(128, ridge_col=80, ridge_h=h_max * radius)
    shadow, _stats = march_shadows(
        np.array([64.0]),
        np.array([20.5]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([1e6]),
        radius_px=np.array([radius]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=h_max,
        h_min=0.0,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert not bool(shadow[0])


def test_march_foreshortening_scales_surface_distance() -> None:
    """Where one image px spans two surface px, the shadow reach halves in image px.

    With ``arc_per_px = 2`` everywhere, a ridge 15 image px upstream lies 30
    surface px away, beyond the 20-surface-px shadow reach that shadowed the
    same layout at unit arc.
    """
    ridge_h = 5.0
    tan_i = 4.0
    height, arc, domain = _flat_maps(96, ridge_col=45, ridge_h=ridge_h)
    arc *= 2.0
    shadow, _stats = march_shadows(
        np.array([48.0]),
        np.array([30.5]),
        h_point=np.array([0.0]),
        tan_incidence=np.array([tan_i]),
        radius_px=np.array([100.0]),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=ridge_h / 100.0,
        h_min=0.0,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert not bool(shadow[0])


def test_march_empty_candidates_is_a_noop() -> None:
    """No candidates: no work, no shadow flags."""
    height, arc, domain = _flat_maps(8, ridge_col=None, ridge_h=0.0)
    shadow, stats = march_shadows(
        np.zeros(0),
        np.zeros(0),
        h_point=np.zeros(0),
        tan_incidence=np.zeros(0),
        radius_px=np.zeros(0),
        height_map=height,
        arc_per_px_map=arc,
        domain_mask=domain,
        h_max=0.01,
        h_min=-0.01,
        sun_v=0.0,
        sun_u=1.0,
    )
    assert shadow.size == 0
    assert stats.steps_executed == 0
