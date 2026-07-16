"""Star deposition under an active whole-scene optics PSF.

With an active optics PSF (an explicit block, the navigator-matched form, or
``instrument_defaults``) the radiance stage deposits each star's total signal
as a sub-pixel point mass, so the scene PSF is the ONLY convolution a star
receives: a floor scene's rendered star sigma equals the navigator's
configured sigma, and an ``instrument_defaults`` star reproduces the catalog
kernel.  Pre-spreading and then convolving would widen every star by sqrt(2).
With no optics PSF, stars keep the Gaussian pre-spread render of the plain
detector-grid path.
"""

from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from spindoctor.sim.forward.artifacts_catalog import PSF_KERNELS
from spindoctor.sim.forward.psf import psf_kernel, psf_truncation_for_instrument
from spindoctor.sim.render import render_combined_model

# Explicit zeroing of every stochastic detector knob, so the star-profile
# measurements below see only the deterministic signal chain.
_QUIET_NOISE: dict[str, Any] = {
    'poisson': False,
    'read_noise_dn': 0.0,
    'bias_dn': 0.0,
    'dark_current_e_per_sec': 0.0,
    'hot_pixel_fraction': 0.0,
    'banding_amplitude_e': 0.0,
    'bias_pedestal_sigma_dn': 0.0,
    'bias_row_gradient_dn': 0.0,
    'bias_col_gradient_dn': 0.0,
}


def _star_scene(instrument: str, **extra: Any) -> dict[str, Any]:
    """A single centered star on a quiet detector, plus extra top-level keys."""
    scene: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'instrument': instrument,
        'noise': dict(_QUIET_NOISE),
        'stars': [{'name': 'S', 'v': 32.0, 'u': 32.0, 'vmag': 0.5}],
    }
    scene.update(extra)
    return scene


def _moment_stats(img: np.ndarray, center: int, radius: int) -> tuple[float, float, float, float]:
    """Centroid and box-corrected moment sigmas of a star cutout.

    The detector image samples the pixel-integrated profile, which inflates
    the raw second moment by the pixel box variance (1/12); subtracting it
    recovers the underlying profile sigma.
    """
    win = img[center - radius : center + radius + 1, center - radius : center + radius + 1]
    vv, uu = np.mgrid[0 : win.shape[0], 0 : win.shape[1]].astype(np.float64)
    total = float(win.sum())
    mv = float((win * vv).sum()) / total
    mu = float((win * uu).sum()) / total
    var_v = float((win * (vv - mv) ** 2).sum()) / total
    var_u = float((win * (uu - mu) ** 2).sum()) / total
    sigma_v = float(np.sqrt(max(var_v - 1.0 / 12.0, 0.0)))
    sigma_u = float(np.sqrt(max(var_u - 1.0 / 12.0, 0.0)))
    return mv + center - radius, mu + center - radius, sigma_v, sigma_u


def test_floor_star_sigma_equals_the_navigator_sigma() -> None:
    """A match_navigator floor star renders at the configured 0.77, not 0.77*sqrt(2)."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    img, _ = render_combined_model(scene)
    _, _, sigma_v, sigma_u = _moment_stats(img, 32, 8)
    assert abs(sigma_v - 0.77) < 0.02 * 0.77
    assert abs(sigma_u - 0.77) < 0.02 * 0.77


def test_floor_star_centroid_lands_at_the_catalog_position() -> None:
    """The point-mass deposit puts the star centroid exactly at its position."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    img, _ = render_combined_model(scene)
    cv, cu, _, _ = _moment_stats(img, 32, 8)
    assert abs(cv - 32.0) < 0.05
    assert abs(cu - 32.0) < 0.05


def test_floor_star_truth_records_the_rendered_sigma() -> None:
    """The star hit-test metadata carries the scene-PSF sigma, not a pre-spread one."""
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    _, meta = render_combined_model(scene)
    assert meta['star_info'][0]['sigma'] == 0.77


def test_instrument_defaults_star_reproduces_the_catalog_kernel() -> None:
    """An instrument_defaults star's profile is the catalog kernel to quantization.

    The reference is built the way the renderer builds it: the star's total
    signal deposited bilinearly on the oversampled grid, convolved with the
    instrument's catalog kernel, and box-downsampled.  A least-squares
    amplitude fit removes the flux scale; the per-pixel residual is then
    bounded by the detector's integer-DN rounding.
    """
    scene = _star_scene(
        'coiss_nac',
        artifacts={'instrument_defaults': True},
        # A no-op distortion block disables the catalog residual so the
        # profile comparison sees only the PSF.
        optics={'distortion': {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 0.0}},
    )
    scene['stars'] = [{'name': 'S', 'v': 32.0, 'u': 32.0, 'vmag': 1.0}]
    img, _ = render_combined_model(scene)
    radius = 10
    cut = img[32 - radius : 32 + radius + 1, 32 - radius : 32 + radius + 1].astype(np.float64)

    os = 4
    k = PSF_KERNELS['coiss_nac']
    kernel = psf_kernel(
        k['sigma_v'],
        k['sigma_u'],
        k['w'],
        k['r0'],
        k['n'],
        truncation_px=psf_truncation_for_instrument('coiss_nac'),
        oversample=os,
    )
    grid = np.zeros((64 * os, 64 * os), dtype=np.float64)
    pos = 32.0 * os + (os - 1) / 2.0
    low = int(np.floor(pos))
    frac = pos - low
    for dv, wv in ((0, 1.0 - frac), (1, frac)):
        for du, wu in ((0, 1.0 - frac), (1, frac)):
            grid[low + dv, low + du] += wv * wu
    det = fftconvolve(grid, kernel, mode='same').reshape(64, os, 64, os).mean(axis=(1, 3))
    ref = det[32 - radius : 32 + radius + 1, 32 - radius : 32 + radius + 1]

    amplitude = float((cut * ref).sum() / (ref * ref).sum())
    residual = np.abs(cut - amplitude * ref)
    assert float(cut.max()) > 100.0
    assert float(residual.max()) <= 0.6


def test_no_optics_star_keeps_the_pre_spread_render() -> None:
    """Without an optics PSF the star is Gaussian pre-spread at its own sigma."""
    img, meta = render_combined_model(_star_scene('coiss_wac'))
    cv, cu, sigma_v, _ = _moment_stats(img, 32, 8)
    assert abs(cv - 32.0) < 0.05
    assert abs(cu - 32.0) < 0.05
    # Pre-spread at the instrument sigma; loose bound (measurement only).
    assert abs(sigma_v - 0.77) < 0.05
    # The pre-spread mode records the star's own sigma.
    assert meta['star_info'][0]['sigma'] == 0.77


def test_background_stars_follow_the_same_deposit_rule() -> None:
    """Background stars sharpen to the scene PSF the same way catalog stars do.

    With the floor PSF active, a pre-spread background star convolved again
    would carry sqrt(2) times the width; the point-mass deposit keeps it at
    the scene sigma.  The single seeded background star of this scene lands
    away from the frame edge, so the local moment window is clean.
    """
    scene = _star_scene('coiss_wac', optics={'psf': {'match_navigator': True}})
    scene['stars'] = []
    scene['background_stars_num'] = 1
    scene['background_stars_psf_sigma'] = 0.77
    img, _ = render_combined_model(scene)
    peak_v, peak_u = np.unravel_index(int(np.argmax(img)), img.shape)
    radius = 6
    win = img[peak_v - radius : peak_v + radius + 1, peak_u - radius : peak_u + radius + 1]
    vv = np.mgrid[0 : win.shape[0], 0 : win.shape[1]][0].astype(np.float64)
    total = float(win.sum())
    mv = float((win * vv).sum()) / total
    var_v = float((win * (vv - mv) ** 2).sum()) / total
    sigma_v = float(np.sqrt(max(var_v - 1.0 / 12.0, 0.0)))
    assert abs(sigma_v - 0.77) < 0.02 * 0.77
