"""Per-instrument artifact defaults for the forward model.

These tables hold the per-instrument optical parameters an ``instrument_defaults``
scene turns on: the whole-scene PSF kernel and the residual geometric-distortion
amplitude.  Every value here is interim -- sized from published FWHMs and
documented residual-error bounds, pending the per-instrument measurement passes
-- and is provenance-tagged as such in the comments beside it.

Keys are sim instrument names (see ``spindoctor.sim.instruments.SIM_INSTRUMENTS``).
"""

from typing import Any

__all__ = ['DISTORTION_RESIDUAL_RMS_PX', 'PSF_KERNELS']

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
