"""Resolved detector parameters for one scene render.

:class:`DetectorParams` collapses the emulated instrument's config block, the
per-instrument catalog defaults (:mod:`spindoctor.sim.forward.artifacts_catalog`),
the scene ``detector`` / ``noise`` blocks, and the ``artifacts.instrument_defaults``
switch into one flat, resolved view the detector stage reads.

Resolution precedence, highest first: an explicit scene key (``detector`` block,
then ``noise`` block), then the catalog value when ``instrument_defaults`` is on,
then the disabled floor (physical-chain artifacts default to zero so an
unconfigured scene renders a clean DN frame, per the stage-activation rule).

The read-noise override keeps the historical ``noise.read_noise_dn`` key working:
it is a DN value, converted to electrons through the resolved gain so a scene
that pinned a DN read-noise level keeps that DN-level behavior under the electron
chain.  ``signal_full_scale_frac`` keeps its meaning (the well fraction a signal
of 1.0 fills at the reference exposure); the image-side DN well is derived
(``full_well_e / gain_e_per_dn``) and the navigator-side ``full_well_dn`` config
key is left untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults
from spindoctor.sim.instruments import resolve_sim_inst_config

__all__ = ['DetectorParams', 'resolve_detector_params']


class DetectorParamError(ValueError):
    """Raised when a scene selects an unavailable detector configuration."""


@dataclass
class DetectorParams:
    """The flat, resolved detector view for one render.

    Parameters:
        detector_model: 'ccd' (electron chain) or 'vidicon' (DN chain).
        data_units: 'raw_dn' or 'calibrated_if'.
        signal_full_scale_frac: Well fraction a signal of 1.0 fills at the
            reference exposure.
        full_well_e: Full well in electrons (CCD path).
        exposure_ref_sec: Exposure the well fraction references.
        exposure_sec: The scene exposure.
        gain_e_per_dn: Resolved gain (electrons per DN) for the selected state.
        read_noise_e: Read-noise sigma in electrons (CCD path).
        bias_dn: Additive DN bias pedestal.
        saturation_dn: ADC clip ceiling in DN.
        full_well_dn: Published ADC-referenced well (vidicon DN full scale).
        quantization: ADC quantization sub-mode.
        poisson: Whether shot noise is applied.
        bloom_length: Electron-domain full-well bloom half-length (0 disables).
        cosmic_ray_rate_per_sec: Cosmic-ray fluence (events / cm^2 / sec).
        pixel_area_cm2: Detector pixel area (scales the cosmic-ray count).
        dark_current_e_per_sec: Dark current (electrons / sec); 0 disables.
        hot_pixel_fraction: Fraction of pixels that are hot; 0 disables.
        hot_pixel_amplitude_e: Hot-pixel amplitude scale in electrons.
        hot_pixel_column_factor: Warm-column fraction bled from a hot pixel.
        banding_amplitude_e: Coherent-banding amplitude in electrons; 0 disables.
        banding_period_px: Coherent-banding spatial period in pixels.
        bias_pedestal_sigma_dn: Per-image bias-pedestal jitter (DN); 0 disables.
        bias_row_gradient_dn: Low-order row bias gradient span (DN).
        bias_col_gradient_dn: Low-order column bias gradient span (DN).
        vidicon: The vidicon DN-noise sub-parameters (vidicon path only).
        calibration_scale_dn_per_s_per_if: Derived I/F calibration scale.
        dark_dn: Dark pedestal in DN subtracted before the I/F divide.
        random_seed: The scene seed for the per-effect sub-streams.
        instrument_defaults: Whether the physical-chain opt-in is on.
    """

    detector_model: str
    data_units: str
    signal_full_scale_frac: float
    full_well_e: float
    exposure_ref_sec: float
    exposure_sec: float
    gain_e_per_dn: float
    read_noise_e: float
    bias_dn: float
    saturation_dn: float
    full_well_dn: float
    quantization: str
    poisson: bool
    bloom_length: int
    cosmic_ray_rate_per_sec: float
    pixel_area_cm2: float
    dark_current_e_per_sec: float
    hot_pixel_fraction: float
    hot_pixel_amplitude_e: float
    hot_pixel_column_factor: float
    banding_amplitude_e: float
    banding_period_px: float
    bias_pedestal_sigma_dn: float
    bias_row_gradient_dn: float
    bias_col_gradient_dn: float
    vidicon: dict[str, float]
    calibration_scale_dn_per_s_per_if: float
    dark_dn: float
    random_seed: int = 42
    instrument_defaults: bool = False


def _default_or_zero(
    scene_noise: Mapping[str, Any],
    catalog: Mapping[str, Any],
    key: str,
    *,
    instrument_defaults: bool,
) -> float:
    """A physical-chain knob: scene override, else catalog (if on), else 0."""
    if key in scene_noise:
        return float(scene_noise[key])
    if instrument_defaults:
        return float(catalog.get(key, 0.0))
    return 0.0


def _resolve_gain(
    catalog: Mapping[str, Any], detector_block: Mapping[str, Any], *, instrument: str | None
) -> float:
    """Select ``gain_e_per_dn`` from the catalog table by the scene gain state.

    Parameters:
        catalog: The instrument's ``DETECTOR_DEFAULTS`` entry.
        detector_block: The scene ``detector`` block (may set ``gain_state``).
        instrument: The sim instrument name, for error messages.

    Returns:
        The electrons-per-DN gain for the selected state.

    Raises:
        DetectorParamError: If the selected state has no catalog entry.
    """
    table = catalog.get('gain_e_per_dn_by_state') or {0: 1.0}
    state = detector_block.get('gain_state')
    if state is None:
        state = int(catalog.get('default_gain_state', 0))
    state = int(state)
    if state not in table:
        raise DetectorParamError(
            f'sim instrument {instrument!r} has no catalogued gain state {state}; '
            f'available states: {sorted(table)}'
        )
    return float(table[state])


def resolve_detector_params(params: Mapping[str, Any]) -> DetectorParams:
    """Collapse the scene, config, and catalog into a resolved detector view.

    Parameters:
        params: The full scene ``sim_params`` mapping.

    Returns:
        The resolved :class:`DetectorParams`.

    Raises:
        DetectorParamError: If the scene selects an unavailable gain state.
    """
    instrument = params.get('instrument')
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, instrument, params.get('instrument_config')
    )
    inst_noise = inst_config.get('noise') or {}
    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    catalog = resolve_detector_defaults(instrument)

    detector_block = params.get('detector') or {}
    scene_noise = params.get('noise') or {}
    artifacts = params.get('artifacts') or {}
    instrument_defaults = bool(artifacts.get('instrument_defaults', False))

    detector_model = str(detector_block.get('detector_model', catalog['detector_model']))
    data_units = str(inst_config.get('data_units', 'raw_dn'))
    exposure_sec = float(params.get('exposure_sec', 1.0))
    exposure_ref_sec = float(
        detector_block.get('exposure_ref_sec', catalog.get('exposure_ref_sec', 1.0))
    )
    quantization = str(detector_block.get('quantization', catalog.get('quantization', 'exact')))

    signal_full_scale_frac = float(
        scene_noise.get(
            'signal_full_scale_frac',
            inst_noise.get('signal_full_scale_frac', sim_noise['signal_full_scale_frac']),
        )
    )
    bias_dn = float(
        scene_noise.get('bias_dn', catalog.get('bias_dn', sim_noise.get('bias_dn', 0.0)))
    )
    saturation_dn = float(inst_noise.get('saturation_dn', sim_noise['saturation_dn']))
    full_well_dn = float(inst_noise.get('full_well_dn', sim_noise['full_well_dn']))

    gain_e_per_dn = _resolve_gain(catalog, detector_block, instrument=instrument)
    full_well_e = float(catalog.get('full_well_e', full_well_dn * gain_e_per_dn))

    # Read noise: a scene DN override wins (converted to electrons via gain);
    # otherwise the catalog electrons value when instrument_defaults is on;
    # otherwise the honest floor (no read noise).
    if 'read_noise_dn' in scene_noise:
        read_noise_e = float(scene_noise['read_noise_dn']) * gain_e_per_dn
    elif instrument_defaults:
        read_noise_e = float(catalog.get('read_noise_e', 0.0))
    else:
        read_noise_e = 0.0

    # Poisson defaults off (the honest floor); a noise block opts in explicitly.
    poisson = bool(scene_noise.get('poisson', False))
    bloom_length = int(scene_noise.get('bloom_length', sim_noise.get('bloom_length', 0)))
    cosmic_ray_rate = float(
        scene_noise.get('cosmic_ray_rate_per_sec', sim_noise.get('cosmic_ray_rate_per_sec', 0.0))
    )
    pixel_area_cm2 = float(scene_noise.get('pixel_area_cm2', sim_noise.get('pixel_area_cm2', 1.0)))

    dark_current = _default_or_zero(
        scene_noise, catalog, 'dark_current_e_per_sec', instrument_defaults=instrument_defaults
    )
    hot_pixel_fraction = _default_or_zero(
        scene_noise, catalog, 'hot_pixel_fraction', instrument_defaults=instrument_defaults
    )
    banding_amplitude = _default_or_zero(
        scene_noise, catalog, 'banding_amplitude_e', instrument_defaults=instrument_defaults
    )
    bias_pedestal_sigma = _default_or_zero(
        scene_noise, catalog, 'bias_pedestal_sigma_dn', instrument_defaults=instrument_defaults
    )
    bias_row_gradient = _default_or_zero(
        scene_noise, catalog, 'bias_row_gradient_dn', instrument_defaults=instrument_defaults
    )
    bias_col_gradient = _default_or_zero(
        scene_noise, catalog, 'bias_col_gradient_dn', instrument_defaults=instrument_defaults
    )
    # Shape parameters (period, amplitude scales) track the catalog; they never
    # activate a stage on their own (the amplitude/fraction gate does that).
    hot_pixel_amplitude = float(
        scene_noise.get('hot_pixel_amplitude_e', catalog.get('hot_pixel_amplitude_e', 0.0))
    )
    hot_pixel_column_factor = float(
        scene_noise.get('hot_pixel_column_factor', catalog.get('hot_pixel_column_factor', 0.0))
    )
    banding_period_px = float(
        scene_noise.get('banding_period_px', catalog.get('banding_period_px', 64.0))
    )

    # The vidicon DN-noise sub-parameters activate on the vidicon path only, and
    # like every physical-chain artifact stay disabled until instrument_defaults
    # is on or the scene sets them explicitly (the stage-activation floor).
    catalog_vidicon = dict(catalog.get('vidicon') or {}) if instrument_defaults else {}
    scene_vidicon = dict(scene_noise.get('vidicon') or {})
    vidicon = {**catalog_vidicon, **{k: float(v) for k, v in scene_vidicon.items()}}

    # Derived I/F calibration scale: a signal of 1.0 at the reference exposure,
    # rendered noise-free, round-trips through the inverse transform to I/F 1.0.
    # Uses the image-side DN well (frac * full_well_e / gain for the CCD path,
    # frac * full_well_dn for the vidicon path).
    if detector_model == 'vidicon':
        image_well_dn = signal_full_scale_frac * full_well_dn
    else:
        image_well_dn = signal_full_scale_frac * full_well_e / gain_e_per_dn
    dark_dn = 0.0 if detector_model == 'vidicon' else dark_current * exposure_sec / gain_e_per_dn
    calibration_scale = (
        image_well_dn / exposure_ref_sec if exposure_ref_sec > 0.0 else image_well_dn
    )

    return DetectorParams(
        detector_model=detector_model,
        data_units=data_units,
        signal_full_scale_frac=signal_full_scale_frac,
        full_well_e=full_well_e,
        exposure_ref_sec=exposure_ref_sec,
        exposure_sec=exposure_sec,
        gain_e_per_dn=gain_e_per_dn,
        read_noise_e=read_noise_e,
        bias_dn=bias_dn,
        saturation_dn=saturation_dn,
        full_well_dn=full_well_dn,
        quantization=quantization,
        poisson=poisson,
        bloom_length=bloom_length,
        cosmic_ray_rate_per_sec=cosmic_ray_rate,
        pixel_area_cm2=pixel_area_cm2,
        dark_current_e_per_sec=dark_current,
        hot_pixel_fraction=hot_pixel_fraction,
        hot_pixel_amplitude_e=hot_pixel_amplitude,
        hot_pixel_column_factor=hot_pixel_column_factor,
        banding_amplitude_e=banding_amplitude,
        banding_period_px=banding_period_px,
        bias_pedestal_sigma_dn=bias_pedestal_sigma,
        bias_row_gradient_dn=bias_row_gradient,
        bias_col_gradient_dn=bias_col_gradient,
        vidicon=vidicon,
        calibration_scale_dn_per_s_per_if=calibration_scale,
        dark_dn=dark_dn,
        random_seed=int(params.get('random_seed', 42)),
        instrument_defaults=instrument_defaults,
    )
