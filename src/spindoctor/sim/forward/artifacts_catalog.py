"""Per-instrument artifact defaults for the forward model.

These tables hold the per-instrument optical parameters an ``instrument_defaults``
scene turns on: the whole-scene PSF kernel and the residual geometric-distortion
amplitude.  Every value here is interim -- sized from published FWHMs and
documented residual-error bounds, pending the per-instrument measurement passes
-- and is provenance-tagged as such in the comments beside it.

Keys are sim instrument names (see ``spindoctor.sim.instruments.SIM_INSTRUMENTS``).
"""

import copy
from collections.abc import Mapping
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, MODE_KEYS

__all__ = [
    'ARTIFACT_MODES',
    'DETECTOR_DEFAULTS',
    'DISTORTION_RESIDUAL_RMS_PX',
    'MODE_KEYS',
    'PSF_KERNELS',
    'resolve_detector_defaults',
    'resolve_mode_with_catalog',
]

# The artifact-mode registry (:mod:`spindoctor.sim.forward.artifact_modes`) is the
# single source of truth for the ``artifacts`` block's per-mode keys, their
# rendering stage, per-instrument availability, and parameter schemas.  It is
# re-exported here so the detector catalog and the mode registry share one
# import point.  Every *loss-mode* incidence defaults to 0 even under
# ``instrument_defaults`` -- naming an instrument selects a signal chain, not a
# set of transmission defects -- so loss modes are planted only by scenes that
# configure them explicitly.

# Whole-scene PSF kernel parameters per instrument, all radii in detector
# pixels.  The core sigma comes from each camera's measured FWHM (sigma =
# FWHM / 2.355); the wing parameters (w, r0, n) are interim, expressed as wing
# energy fractions so the delivered kernels conserve flux.
#
# Provenance (all interim):
# - coiss_nac/coiss_wac sigma from the Cassini ISS measured FWHMs; the
#   core-to-wing dynamic range beyond the truncation window is stray-light
#   scope, so the shipped wing fraction is small.
# - vgiss sigma is an interim estimate: the Voyager references publish no FWHM,
#   and GEOMED resampling broadens whatever the vidicon delivered.
# - gossi sigma is a directly published value, not FWHM / 2.355.
# - nhlorri is elliptical (sigma_v != sigma_u) per the LORRI PSF references.
# The wing parameters are the first quantities the realism-match pass tunes.
PSF_KERNELS: dict[str, dict[str, float]] = {
    'coiss_nac': {'sigma_v': 0.55, 'sigma_u': 0.55, 'w': 2.5e-2, 'r0': 2.0, 'n': 3.0},
    'coiss_wac': {'sigma_v': 0.64, 'sigma_u': 0.64, 'w': 2.5e-2, 'r0': 2.0, 'n': 3.0},
    'vgiss': {'sigma_v': 0.85, 'sigma_u': 0.85, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
    'gossi': {'sigma_v': 0.80, 'sigma_u': 0.80, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
    'nhlorri': {'sigma_v': 1.13, 'sigma_u': 0.87, 'w': 1.2e-2, 'r0': 2.0, 'n': 3.0},
}

# Residual geometric-distortion RMS displacement over the frame, in detector
# pixels: the error remaining after the navigator applies each instrument's
# known distortion model (the only distortion actually present in the frames
# the pipeline consumes).  All values are interim, pending the per-instrument
# star-field residual measurement; the Cassini corrected field is sub-pixel,
# Voyager GEOMED products carry ~1 px internal error, and the mapped-field
# cameras sit in between.
DISTORTION_RESIDUAL_RMS_PX: dict[str, float] = {
    'coiss_nac': 0.1,
    'coiss_wac': 0.1,
    'gossi': 0.05,
    'nhlorri': 0.05,
    'vgiss': 1.0,
}


def _coiss_alias(table: dict[str, Any]) -> None:
    """Point the calibrated Cassini instrument names at their raw entries."""
    for calib, raw in (('coiss_calib_nac', 'coiss_nac'), ('coiss_calib_wac', 'coiss_wac')):
        if raw in table and calib not in table:
            table[calib] = table[raw]


_coiss_alias(PSF_KERNELS)
_coiss_alias(DISTORTION_RESIDUAL_RMS_PX)


# Per-instrument detector electron-chain and noise-model parameters.  Every
# value is interim and provenance-tagged; the electron chain, gain tables, read
# noise, and vidicon numbers come from the instrument calibration references
# (plan Section 5) and are the first quantities the realism-match pass revisits.
#
# Chain (CCD): electrons = signal * signal_full_scale_frac * full_well_e *
# (exposure_sec / exposure_ref_sec); Poisson; electron-domain full-well bloom
# against full_well_e; read noise (read_noise_e); DN = electrons / gain_e_per_dn
# + bias_dn; quantization; clip at saturation_dn.  gain_e_per_dn is selected
# from gain_e_per_dn_by_state[gain_state]; a scene selecting a state absent from
# that table is a validation error (5.2), not a silent guess.  The image-side
# well in DN is DERIVED (full_well_e / gain_e_per_dn); the navigator-side
# full_well_dn config key is a separate published ADC-referenced value.
#
# The 'instrument_defaults' physical-chain knobs (dark_current_e_per_sec,
# hot_pixel_fraction, hot_pixel_amplitude_e, banding_amplitude_e,
# banding_period_px, bias_pedestal_sigma_dn, bias_row_gradient_dn,
# bias_col_gradient_dn, bloom_length, quantization) are interim placeholders
# sized from the 5.2-5.6 descriptions; per-scene overrides ride in the
# truth-side noise block.  instrument_defaults also turns on Poisson shot
# noise (a property of the electron chain itself, not a per-camera number);
# the loss modes (cosmic rays, missing data) are artifact incidences and stay
# at zero.
#
# The vidicon (Voyager) path skips the electron conversion; its noise is applied
# directly in DN (5.3): line-correlated read noise (per-line offset +
# within-line white), a faint coherent periodic component, and 8-bit
# quantization.  A vidicon entry carries a 'vidicon' sub-map instead of the CCD
# electron keys.
DETECTOR_DEFAULTS: dict[str, dict[str, Any]] = {
    'coiss_nac': {
        'detector_model': 'ccd',
        # 5.2 interim: full well ~110k e- (saturates ~3600 DN at gain 2, below
        # the 4095 ADC clip).
        'full_well_e': 110.0e3,
        # 5.2 interim: a typical NAC science exposure, so a converted scene at
        # this exposure keeps today's brightness scale.
        'exposure_ref_sec': 1.0,
        # 5.2 interim gain states ~233/95/30/13 e-/DN; tour-standard state 2.
        'gain_e_per_dn_by_state': {0: 233.0, 1: 95.0, 2: 30.0, 3: 13.0},
        'default_gain_state': 2,
        'read_noise_e': 12.0,  # 5.2 interim
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 5.0,  # 5.2 interim (RBI-dominated dark)
        'hot_pixel_fraction': 2.0e-3,  # 5.2 interim (~0.15-0.28% of pixels)
        'hot_pixel_amplitude_e': 4.0e4,  # 5.2 interim (near full well)
        # 5.2 interim: total-charge fraction bled into the warm column above a
        # hot pixel (the streak integral, not a per-pixel amplitude).
        'hot_pixel_column_factor': 0.3,
        'banding_amplitude_e': 30.0,  # 5.2 interim (~30 e- NAC 2 Hz)
        'banding_period_px': 64.0,  # 5.2 interim (line-readout-rate period)
        'bias_pedestal_sigma_dn': 2.0,  # 5.2 interim (per-image pedestal jitter)
        'bias_row_gradient_dn': 1.0,  # 5.2 interim (readout-direction gradient)
        'bias_col_gradient_dn': 0.5,  # 5.2 interim
        'bloom_length': 4,  # 5.2 interim (no antiblooming; column bleed above the well)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.2).  incidence is never catalogued:
        # a mode activates only when a scene sets it.  Shapes here are the values
        # a scene inherits when it names a mode without spelling every parameter.
        'artifact_modes': {
            # ~30 e- NAC 2 Hz horizontal banding, line-readout-rate period.
            'banding_coherent': {
                'amplitude_e': 30.0,
                'period_px': 64.0,
                'orientation': 'horizontal',
            },
            # Per-image pedestal jitter plus readout-direction gradients.
            'bias_structure': {
                'pedestal_sigma_dn': 2.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            # RBI-dominated dark grows toward the last readout line.
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 150.0},
            'bloom': {'bloom_length': 4},
            # Anti-blooming vertical 2-px pairs in unsummed long exposures.
            'bright_dark_pairs': {'amplitude_e': 4.0e3},
            # No published ISS rate; the interplanetary regime near full-well amp.
            'radiation_transients': {'amplitude_e': 4.0e4},
            # Dust donuts (<1% each) accumulating over the mission.
            'fixed_pattern': {'dust_donut_count': 5},
        },
    },
    'coiss_wac': {
        'detector_model': 'ccd',
        'full_well_e': 95.0e3,  # 5.2 interim
        'exposure_ref_sec': 1.0,
        # 5.2: only state 2 is catalogued for the WAC; selecting another WAC
        # state is a validation error until its full table is sourced.
        'gain_e_per_dn_by_state': {2: 28.0},
        'default_gain_state': 2,
        'read_noise_e': 12.0,  # 5.2 interim
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 5.0,
        'hot_pixel_fraction': 2.0e-3,
        'hot_pixel_amplitude_e': 3.5e4,
        'hot_pixel_column_factor': 0.3,
        'banding_amplitude_e': 6.0,  # 5.2 interim (~6 e- WAC 4 Hz)
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 2.0,
        'bias_row_gradient_dn': 1.0,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 4,  # 5.2 interim (no antiblooming; column bleed above the well)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.2); ~6 e- WAC 4 Hz banding.
        'artifact_modes': {
            'banding_coherent': {
                'amplitude_e': 6.0,
                'period_px': 64.0,
                'orientation': 'horizontal',
            },
            'bias_structure': {
                'pedestal_sigma_dn': 2.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 150.0},
            'bloom': {'bloom_length': 4},
            'bright_dark_pairs': {'amplitude_e': 3.5e3},
            'radiation_transients': {'amplitude_e': 3.5e4},
            'fixed_pattern': {'dust_donut_count': 5},
        },
    },
    'gossi': {
        'detector_model': 'ccd',
        'full_well_e': 108.0e3,  # 5.4 interim
        'exposure_ref_sec': 0.2,  # 5.4 interim (typical science exposure)
        # 5.4 interim gain states ~1822/377/187/39 e-/DN; common science state 2.
        'gain_e_per_dn_by_state': {0: 1822.0, 1: 377.0, 2: 187.0, 3: 39.0},
        'default_gain_state': 2,
        'read_noise_e': 31.0,  # 5.4 (full-res; 44 e- in summation mode)
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 10.0,  # 5.4 interim (RTG-driven dark spikes)
        'hot_pixel_fraction': 3.0e-3,
        'hot_pixel_amplitude_e': 4.0e4,
        # 5.4 interim (early-blooming columns; total-charge fraction).
        'hot_pixel_column_factor': 0.5,
        'banding_amplitude_e': 65.0,  # 5.4 interim (~0.35 DN at gain 2 -> ~65 e-)
        'banding_period_px': 42.0,  # 5.4 (2400 Hz supply-noise comb every 42 px)
        'bias_pedestal_sigma_dn': 1.0,
        'bias_row_gradient_dn': 1.0,  # 5.4 (summation-mode L-R shading ramp)
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 6,  # 5.4 interim (early-blooming columns)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.4).
        'artifact_modes': {
            # 42-px vertical supply-noise comb (~0.35 DN at gain 2 -> ~65 e-),
            # plus a <8 Hz horizontal component in high gain (orientation both).
            'banding_coherent': {'amplitude_e': 65.0, 'period_px': 42.0, 'orientation': 'both'},
            'bias_structure': {
                'pedestal_sigma_dn': 1.0,
                'row_gradient_dn': 1.0,
                'col_gradient_dn': 0.5,
            },
            # Shutter line-dependent exposure offset (~1.5 -> ~1.05 ms, line 1->800).
            'dark_ramp': {'kind': 'exposure_shading', 'top_factor': 1.5, 'bottom_factor': 1.05},
            'bloom': {'bloom_length': 6},
            # Ganymede-distance regime: ~1e4 spikes/frame scale in the readout,
            # amplitudes few DN steeply falling (near full well at the top).
            'radiation_transients': {'amplitude_e': 4.0e4},
            # 33-px photolithography stitch comb plus corner vignetting; the
            # 8-bit ADC contours worst at DN multiples of 8.
            'fixed_pattern': {
                'stitch_period_px': 33,
                'stitch_amplitude_dn': 1.0,
                'vignetting_frac': 0.03,
                'dust_donut_count': 4,
            },
            'contouring_8bit': {'step': 8},
            # HMA/HCA vertical decimation (5.4): only every Nth line carries
            # valid data, so the Galileo periodic-line default is 'keep'.
            'alternating_lines': {'mode': 'keep'},
        },
    },
    'nhlorri': {
        'detector_model': 'ccd',
        # 5.5: ADC-limited full well 4095 DN x 21 e-/DN.
        'full_well_e': 86.0e3,
        'exposure_ref_sec': 0.1,  # 5.5 interim (typical encounter exposure)
        # 5.5: single gain state ~21 e-/DN (1x1); 4x4 binning reads ~19.4.
        'gain_e_per_dn_by_state': {0: 21.0},
        'default_gain_state': 0,
        'read_noise_e': 23.0,  # 5.5 (~1.1 DN)
        'bias_dn': 545.0,  # 5.5 (~545 DN estimated per-image from dark columns)
        'dark_current_e_per_sec': 0.04,  # 5.5 (negligible dark current)
        # 5.5: LORRI has NO hot pixels (PDS maps are zeroes); disabled.
        'hot_pixel_fraction': 0.0,
        'hot_pixel_amplitude_e': 0.0,
        'hot_pixel_column_factor': 0.0,
        'banding_amplitude_e': 17.0,  # 5.5 interim (~0.8 DN vertical banding)
        'banding_period_px': 128.0,  # 5.5 interim
        'bias_pedestal_sigma_dn': 1.0,
        'bias_row_gradient_dn': 0.5,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 2,  # 5.5 interim (short column bleed)
        'quantization': 'exact',
        # Per-mode shape defaults (interim, 5.5).  frame_transfer_smear is the
        # one artifact mode LORRI turns on under instrument_defaults (the defining
        # LORRI artifact, per 15.7): the detector resolver injects it at these
        # nominal scrub/transfer times when instrument_defaults is on and the
        # scene has not overridden it.  Every other mode stays off until a scene
        # sets its incidence.
        'artifact_modes': {
            # ~12 ms pre-exposure scrub, ~11 ms post-exposure transfer.
            'frame_transfer_smear': {'t_scrub_sec': 0.012, 't_transfer_sec': 0.011},
            # Saturated compact sources undershoot up to ~12 DN along readout.
            'serial_tail': {'amplitude_dn': 12.0, 'length_px': 8, 'direction': 'right'},
            # <=1 DN horizontal striping plus ~0.8 DN vertical banding near low
            # columns (~17 e-); orientation both approximates the two families.
            'banding_coherent': {'amplitude_e': 17.0, 'period_px': 128.0, 'orientation': 'both'},
            # ~4% corner vignetting, 0.9% PRNU, <=0.5 DN even/odd jail bars,
            # ~1% dust donuts.
            'fixed_pattern': {
                'vignetting_frac': 0.04,
                'prnu_rms': 0.009,
                'jail_bar_dn': 0.5,
                'dust_donut_count': 3,
            },
            # ~16 hits per readout-dominated short exposure; mostly single px.
            'radiation_transients': {'amplitude_e': 8.6e4},
        },
    },
    'vgiss': {
        'detector_model': 'vidicon',
        # The vidicon skips the electron conversion; signal maps straight to the
        # 8-bit DN full scale, then the DN-domain vidicon noise model applies.
        'exposure_ref_sec': 1.0,
        'bias_dn': 20.0,
        'quantization': '8bit',
        'vidicon': {
            # 5.3 interim: readout-chain noise ~0.3-0.75 DN low gain / 2.2-2.6 DN
            # high gain; a per-line-correlated offset plus a within-line white
            # component (the two summing in quadrature to the quoted RMS).
            'read_noise_line_dn': 1.8,
            'read_noise_pixel_dn': 1.8,
            # 5.3: faint coherent periodic component (2.4 kHz vertical, ~0.5 DN
            # peak-to-peak).
            'coherent_amplitude_dn': 0.25,
            'coherent_period_px': 8.0,
        },
        # Per-mode shape defaults (interim, 5.3), all GEOMED-level: the beam-bend
        # limb bias that survives reseau-anchored correction, the shortened
        # erase-cycle residual image, the readout dark-current line ramp, the
        # reseau-removal scars on the ~46-px lattice, and the GEOMED resample
        # texture (blank border + missing-line interpolation banding).
        'artifact_modes': {
            'beam_bend': {'amplitude_px': 1.0},
            'residual_image': {'amplitude': 0.05, 'prior': 'self_offset', 'offset_px': [5, 5]},
            # 48 x n s readout ramp, nonlinear in wait time.
            'dark_ramp': {'kind': 'dark_gradient', 'amplitude_e': 6.0, 'nonlinear': 1.5},
            'contouring_8bit': {'step': 8},
            'reseau_scars': {'spacing_px': 46, 'patch_radius_px': 4},
            'resample_texture': {
                'warp_amp_px': 0.3,
                'blank_border_px': 2,
                'missing_line_interp': False,
            },
        },
    },
    # The instrument-agnostic 'generic' / 'sim' block: an ideal 12-bit detector
    # whose electron well equals its DN depth at unit gain, so a generic scene's
    # electron chain reproduces the direct signal-to-DN mapping (electrons == DN)
    # rather than imposing a specific camera's radiometry.
    'generic': {
        'detector_model': 'ccd',
        'full_well_e': 4095.0,
        'exposure_ref_sec': 1.0,
        'gain_e_per_dn_by_state': {0: 1.0},
        'default_gain_state': 0,
        'read_noise_e': 1.0,
        'bias_dn': 20.0,
        'dark_current_e_per_sec': 0.0,
        'hot_pixel_fraction': 0.0,
        'hot_pixel_amplitude_e': 0.0,
        'hot_pixel_column_factor': 0.0,
        'banding_amplitude_e': 0.0,
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 0.0,
        'bias_row_gradient_dn': 0.0,
        'bias_col_gradient_dn': 0.0,
        'bloom_length': 0,
        'quantization': 'exact',
    },
}

_coiss_alias(DETECTOR_DEFAULTS)


def resolve_detector_defaults(instrument: str | None) -> dict[str, Any]:
    """Return the detector-parameter defaults for a sim instrument.

    Parameters:
        instrument: The sim instrument name (see
            ``spindoctor.sim.instruments.SIM_INSTRUMENTS``), one of the generic
            aliases, or ``None`` for the instrument-agnostic block.

    Returns:
        A fresh copy of the instrument's ``DETECTOR_DEFAULTS`` entry, falling
        back to the generic block for the generic aliases or an unknown name.
    """
    key = 'generic' if instrument is None or instrument in ('generic', 'sim') else instrument
    entry = DETECTOR_DEFAULTS.get(key, DETECTOR_DEFAULTS['generic'])
    return copy.deepcopy(entry)


def resolve_mode_with_catalog(
    mode_name: str, scene_cfg: Mapping[str, Any], instrument: str | None
) -> dict[str, Any]:
    """Resolve an artifact mode's parameters with the per-instrument catalog.

    The resolution precedence for a mode's shape parameters is scene value, then
    the instrument's catalog default block (``artifact_modes`` in
    ``DETECTOR_DEFAULTS``), then the registry default.  ``incidence`` is never
    read from the catalog: a mode activates only when a scene sets its incidence
    (or, for the one physical-signal-chain member LORRI turns on, when the
    detector resolver injects it under ``instrument_defaults``).

    Parameters:
        mode_name: A registered artifact-mode name.
        scene_cfg: The scene's map for the mode (already validated).
        instrument: The sim instrument name, for the catalog lookup.

    Returns:
        A fresh dict carrying every parameter at its resolved value.
    """
    catalog_modes = resolve_detector_defaults(instrument).get('artifact_modes') or {}
    catalog = catalog_modes.get(mode_name) or {}
    resolved: dict[str, Any] = {}
    for param in ARTIFACT_MODES[mode_name].params:
        name = param.name
        if name in scene_cfg and scene_cfg[name] is not None:
            resolved[name] = scene_cfg[name]
        elif name != 'incidence' and name in catalog:
            resolved[name] = catalog[name]
        else:
            resolved[name] = param.default
    return resolved
