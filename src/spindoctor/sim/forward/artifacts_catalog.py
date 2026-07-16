"""Per-instrument artifact defaults for the forward model.

These tables hold the per-instrument optical parameters an ``instrument_defaults``
scene turns on: the whole-scene PSF kernel and the residual geometric-distortion
amplitude.  Every value here is interim -- sized from published FWHMs and
documented residual-error bounds, pending the per-instrument measurement passes
-- and is provenance-tagged as such in the comments beside it.

Keys are sim instrument names (see ``spindoctor.sim.instruments.SIM_INSTRUMENTS``).
"""

import copy
from typing import Any

from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES, MODE_KEYS

__all__ = [
    'ARTIFACT_MODES',
    'DETECTOR_DEFAULTS',
    'DISTORTION_RESIDUAL_RMS_PX',
    'MODE_KEYS',
    'PSF_KERNELS',
    'resolve_detector_defaults',
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
        # LORRI's 0.011 s frame-transfer smear will hook into
        # instrument_defaults here when the telemetry-artifacts work lands.
        'quantization': 'exact',
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
