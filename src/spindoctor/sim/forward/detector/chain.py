"""The detector stage: composed signal to digitized detector counts.

This is the normative unit chain of the forward model.  For a CCD the composed
intensive signal is converted to electrons through the exposure, the point-source
electron plane is added, and the frame passes through Poisson shot noise,
electron-domain full-well bloom, read noise, coherent banding, gain to DN, bias
structure, quantization, and the ADC clip.  A camera's physical saturation
therefore emerges from ``full_well_e / gain_e_per_dn`` (below the ADC ceiling for
Cassini), not from the ADC clip.

The Voyager vidicon skips the electron conversion and applies its noise directly
in DN (line-correlated read noise plus a faint coherent component).  A calibrated
(I/F) scene renders through the full DN chain and then inverts the calibration
transform, so calibrated products carry propagated shot/read noise and
quantization texture in I/F units.

The deterministic conversion (signal to DN) always runs -- it is what makes a DN
frame -- while the stochastic and structured sub-effects (shot noise, read noise,
cosmic rays, dark/hot pixels, banding, bias structure) activate only when their
scene block or ``instrument_defaults`` requests them, so a scene with no such
block renders a clean DN frame (the self-consistency floor).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.sim.forward.detector.noise_stages import (
    add_banding,
    add_bias_structure,
    add_cosmic_rays,
    add_dark_current,
    add_hot_pixels,
)
from spindoctor.sim.forward.detector.params import DetectorParams, resolve_detector_params
from spindoctor.sim.forward.feature_loci import dilated_pixels, extract_feature_loci
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['apply_detector', 'apply_saturation', 'quantize_dn']


def _stage_rng(seed: int, effect: str) -> np.random.Generator:
    """A generator for one detector sub-effect, seeded independently by name."""
    return np.random.default_rng(derive_effect_seed(seed, f'detector/{effect}'))


def _hot_pixel_pool(
    frame: SimFrame, dp: DetectorParams
) -> tuple[NDArrayIntType, NDArrayIntType] | None:
    """The adversarial hot-pixel placement pool, or None for uniform placement.

    When the hot_pixels artifact mode requests adversarial placement, the
    population is biased onto the pixels on and beside the navigation features
    (limb / ring-edge arcs and star positions, dilated by a few pixels).
    """
    if not dp.hot_pixel_adversarial:
        return None
    loci = extract_feature_loci(frame.truth, frame.signal.shape)
    return dilated_pixels(loci, radius=3, shape=frame.signal.shape)


def apply_saturation(
    electrons: NDArrayFloatType,
    *,
    full_well_e: float,
    bloom_length: int = 0,
) -> None:
    """Cap the electron image at the full well, optionally blooming along columns.

    Charge above ``full_well_e`` spills along the column (the v axis) up to
    ``bloom_length`` pixels each way, conserving the total excess, before every
    pixel is capped at the full well.  A saturated star therefore blooms into a
    vertical streak the way it does on cameras with column bleed, and the capped
    pixels read ``full_well_e / gain`` DN after conversion (below the ADC clip on
    an antiblooming-free camera such as Cassini's NAC).

    Parameters:
        electrons: The electron image, modified in place.
        full_well_e: The full-well ceiling in electrons.
        bloom_length: Column-bloom half-length in pixels; 0 disables bloom.
    """
    if bloom_length > 0:
        excess = np.maximum(electrons - full_well_e, 0.0)
        if float(excess.max()) > 0.0:
            width = 2 * bloom_length + 1
            spread = ndimage.uniform_filter1d(excess, size=width, axis=0, mode='constant')
            np.minimum(electrons, full_well_e, out=electrons)
            electrons += spread
    np.minimum(electrons, full_well_e, out=electrons)


def quantize_dn(dn: NDArrayFloatType, *, mode: str, saturation_dn: float) -> NDArrayFloatType:
    """Quantize a DN image by the selected ADC sub-mode.

    Parameters:
        dn: The DN image (float).
        mode: 'exact' (round to integer, uniform bins), '8bit' (integer bins
            with a hard 255 code ceiling), 'uneven_12bit' (integer bins with
            histogram spikes at the power-of-two bit boundaries), or
            'sqrt_lut' (square-root companding to 8 bits and back, leaving a
            signal-dependent residual).
        saturation_dn: The ADC ceiling, used to scale the companding LUT.

    Returns:
        The quantized DN image.

    Raises:
        ValueError: If ``mode`` is not a known quantization sub-mode.
    """
    if mode == 'exact':
        return np.rint(dn)
    if mode == '8bit':
        # An 8-bit output word: integer DN clipped at the 255 code ceiling.
        # The ceiling is the word width, not the scene's saturation_dn, so an
        # 8-bit mode on a deeper detector still tops out at 255.
        return np.clip(np.rint(dn), 0.0, 255.0)
    if mode == 'uneven_12bit':
        # Uneven bit weights concentrate codes at the power-of-two carry
        # boundaries: values within one DN of a 2^m boundary snap to it, which
        # is the histogram-spike signature of an ADC with unequal bit weights.
        quantized = np.rint(dn)
        max_bit = int(np.log2(max(saturation_dn, 2.0))) + 1
        for m in range(1, max_bit):
            boundary = float(2**m)
            near = np.abs(quantized - boundary) <= 1.0
            quantized[near] = boundary
        return quantized
    if mode == 'sqrt_lut':
        # Square-root companding: encode to 8 bits through a sqrt LUT and decode
        # back, so the quantization step tracks the photon noise and the inverse
        # leaves a residual that grows with signal.
        ceiling = max(saturation_dn, 1.0)
        clipped = np.clip(dn, 0.0, ceiling)
        code = np.rint(255.0 * np.sqrt(clipped / ceiling))
        return ceiling * (code / 255.0) ** 2
    raise ValueError(f'unknown quantization mode {mode!r}')


def _convert_to_electrons(signal: NDArrayFloatType, dp: DetectorParams) -> NDArrayFloatType:
    """Convert the intensive [0, 1] signal to electrons through the exposure."""
    scale = (
        dp.signal_full_scale_frac
        * dp.full_well_e
        * (dp.exposure_sec / dp.exposure_ref_sec if dp.exposure_ref_sec > 0.0 else 1.0)
    )
    return np.clip(signal, 0.0, 1.0) * scale


def _apply_ccd(frame: SimFrame, dp: DetectorParams, rng: np.random.Generator) -> None:
    """Run the CCD electron unit chain, overwriting the signal plane with DN."""
    electrons = _convert_to_electrons(frame.signal, dp)
    # Point sources are already electrons; add them AFTER the intensive
    # conversion and BEFORE Poisson so stars never pass through the signal scale.
    electrons += frame.point_e
    add_dark_current(
        electrons,
        rate_e_per_sec=dp.dark_current_e_per_sec,
        exposure_sec=dp.exposure_sec,
    )
    if dp.poisson:
        electrons = rng.poisson(np.maximum(electrons, 0.0)).astype(np.float64)
    add_hot_pixels(
        electrons,
        fraction=dp.hot_pixel_fraction,
        amplitude_e=dp.hot_pixel_amplitude_e,
        column_factor=dp.hot_pixel_column_factor,
        rng=_stage_rng(dp.random_seed, 'hot_pixels'),
        candidate_pool=_hot_pixel_pool(frame, dp),
    )
    apply_saturation(electrons, full_well_e=dp.full_well_e, bloom_length=dp.bloom_length)
    # Cosmic rays deposit well above the full well AFTER the bloom cap, so they
    # reach the ADC ceiling and land on the orchestrator's masks.
    add_cosmic_rays(
        electrons,
        rate_per_sec=dp.cosmic_ray_rate_per_sec,
        exposure_sec=dp.exposure_sec,
        pixel_area_cm2=dp.pixel_area_cm2,
        amplitude_e=dp.full_well_e,
        rng=_stage_rng(dp.random_seed, 'cosmic_rays'),
    )
    if dp.read_noise_e > 0.0:
        electrons += rng.normal(0.0, dp.read_noise_e, size=electrons.shape)
    add_banding(
        electrons,
        amplitude_e=dp.banding_amplitude_e,
        period_px=dp.banding_period_px,
        rng=_stage_rng(dp.random_seed, 'banding'),
    )
    dn = electrons / dp.gain_e_per_dn + dp.bias_dn
    add_bias_structure(
        dn,
        pedestal_sigma_dn=dp.bias_pedestal_sigma_dn,
        row_gradient_dn=dp.bias_row_gradient_dn,
        col_gradient_dn=dp.bias_col_gradient_dn,
        rng=_stage_rng(dp.random_seed, 'bias_structure'),
    )
    dn = quantize_dn(dn, mode=dp.quantization, saturation_dn=dp.saturation_dn)
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn


def _apply_vidicon(frame: SimFrame, dp: DetectorParams, rng: np.random.Generator) -> None:
    """Run the vidicon DN chain: no electron conversion, DN-domain noise.

    The vidicon is not photon-noise dominated, so its signal maps straight to
    the 8-bit DN full scale and the noise is applied in DN: a line-correlated
    read-noise term (a per-line offset plus a within-line white component) and a
    faint vertical coherent periodic component, then 8-bit quantization.
    """
    signal_dn = np.clip(frame.signal, 0.0, 1.0) * dp.signal_full_scale_frac * dp.full_well_dn
    size_v, size_u = signal_dn.shape
    vidicon = dp.vidicon
    line_sigma = float(vidicon.get('read_noise_line_dn', 0.0))
    pixel_sigma = float(vidicon.get('read_noise_pixel_dn', 0.0))
    dn = signal_dn
    if line_sigma > 0.0:
        dn = dn + rng.normal(0.0, line_sigma, size=(size_v,))[:, None]
    if pixel_sigma > 0.0:
        dn = dn + rng.normal(0.0, pixel_sigma, size=(size_v, size_u))
    coherent_amp = float(vidicon.get('coherent_amplitude_dn', 0.0))
    coherent_period = float(vidicon.get('coherent_period_px', 0.0))
    if coherent_amp > 0.0 and coherent_period > 0.0:
        phase = float(_stage_rng(dp.random_seed, 'vidicon_coherent').uniform(0.0, 2.0 * np.pi))
        cols = np.arange(size_u, dtype=np.float64)
        coherent = coherent_amp * np.sin(2.0 * np.pi * cols / coherent_period + phase)
        dn = dn + coherent[None, :]
    dn = dn + dp.bias_dn
    dn = quantize_dn(dn, mode=dp.quantization, saturation_dn=dp.saturation_dn)
    np.clip(dn, 0.0, dp.saturation_dn, out=dn)
    frame.signal[:] = dn


def _apply_calibration_inverse(frame: SimFrame, dp: DetectorParams) -> None:
    """Invert the calibration transform: DN to I/F, bias and dark subtracted first.

    Matches the real pipeline: the bias pedestal and dark pedestal are subtracted
    before the exposure divide, so a calibrated frame carries no spurious 1/
    exposure pedestal.  The calibration scale is derived so a noise-free signal
    of 1.0 at the reference exposure round-trips to I/F 1.0.
    """
    exposure = dp.exposure_sec if dp.exposure_sec > 0.0 else 1.0
    denom = dp.calibration_scale_dn_per_s_per_if * exposure
    if denom <= 0.0:
        return
    frame.signal[:] = (frame.signal - dp.bias_dn - dp.dark_dn) / denom


def apply_detector(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Detector stage: convert the composed signal to detector counts in place.

    All signal is composed before this stage runs, so the shot term sees the
    noise-free signal it should grow with.  Feature masks in ``frame.truth`` were
    derived from the noise-free signal and are unaffected.

    Parameters:
        frame: The frame whose signal plane is converted in place.
        params: The full scene mapping; resolved into detector parameters via
            :func:`spindoctor.sim.forward.detector.params.resolve_detector_params`.
        rng: The stage generator, used for the shot and read-noise streams; the
            structured sub-effects derive their own named streams from the scene
            seed.
    """
    dp = resolve_detector_params(params)
    if dp.detector_model == 'vidicon':
        _apply_vidicon(frame, dp, rng)
    else:
        _apply_ccd(frame, dp, rng)
    if dp.data_units != 'raw_dn':
        _apply_calibration_inverse(frame, dp)
