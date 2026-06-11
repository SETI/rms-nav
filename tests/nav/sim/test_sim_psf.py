"""Per-instrument star PSF for the simulator (B5).

Sim stars use the selected instrument's ``star_psf_sigma`` so their centroid and
spread match the PSF the navigator fits.  These tests check that the rendered
star spread scales with the PSF sigma and differs between instruments.
"""

from typing import Any

import numpy as np

from nav.config import DEFAULT_CONFIG
from nav.sim.instruments import resolve_sim_inst_config
from nav.sim.render import render_combined_model, render_stars


def _rms_spread(image: np.ndarray) -> float:
    """Intensity-weighted RMS radius of a single star patch."""
    vv, uu = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    total = float(image.sum())
    cv = float((image * vv).sum()) / total
    cu = float((image * uu).sum()) / total
    var = float((image * ((vv - cv) ** 2 + (uu - cu) ** 2)).sum()) / total
    return float(np.sqrt(var))


def _render_one_star(sigma: float) -> np.ndarray:
    """Render a single centered star at the given default PSF sigma."""
    img = np.zeros((41, 41), dtype=np.float64)
    stars = [{'name': 's', 'v': 20.0, 'u': 20.0, 'vmag': 0.0, 'psf_size': (31, 31)}]
    out, _, _ = render_stars(img, stars, 0.0, 0.0, default_psf_sigma=sigma)
    return out


def test_spread_increases_with_sigma() -> None:
    """A larger PSF sigma yields a broader rendered star."""
    narrow = _rms_spread(_render_one_star(0.6))
    wide = _rms_spread(_render_one_star(3.0))
    assert wide > narrow


def test_spread_monotonic_across_sigmas() -> None:
    """Rendered star spread rises monotonically with PSF sigma."""
    spreads = [_rms_spread(_render_one_star(s)) for s in (0.6, 1.0, 2.0, 3.0)]
    assert all(spreads[i + 1] > spreads[i] for i in range(len(spreads) - 1))


def test_coiss_and_gossi_psf_sigmas_differ() -> None:
    """The two instruments declare different star PSF sigmas (test premise)."""
    coiss = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')['star_psf_sigma'])
    gossi = float(resolve_sim_inst_config(DEFAULT_CONFIG, 'gossi')['star_psf_sigma'])
    assert coiss < gossi


def _noiseless_star_scene(instrument: str, *, size: int = 41) -> dict[str, Any]:
    """A single-star scene with detector noise disabled for a clean PSF."""
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': instrument,
        'noise': {'poisson': False, 'read_noise_dn': 0.0},
        'stars': [{'name': 's', 'v': size / 2, 'u': size / 2, 'vmag': 0.0, 'psf_size': (31, 31)}],
    }


def test_render_uses_instrument_psf_sigma() -> None:
    """A coiss frame renders tighter stars than a gossi frame end-to-end."""
    coiss_img, _ = render_combined_model(_noiseless_star_scene('coiss_nac'))
    gossi_img, _ = render_combined_model(_noiseless_star_scene('gossi'))
    assert _rms_spread(coiss_img) < _rms_spread(gossi_img)
