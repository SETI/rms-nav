"""Image-side telemetry stage: what transmission loses or mangles.

Present fidelity carries only the per-pixel missing-data markers (a
placeholder loss geometry: real cameras lose whole lines, partial lines,
alternating lines, truncated frame bottoms, or compression blocks).  Phase C
replaces the geometry with the structured loss modes and the per-instrument
artifact catalog defaults.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.instruments import resolve_sim_inst_config

__all__ = ['apply_telemetry']


def apply_telemetry(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Telemetry stage: overwrite lost pixels with the missing-data marker.

    Runs after the detector stage, so markers overwrite readout values the
    way a downlink dropout erases transmitted pixels.  Applies only on the
    raw-DN path (the calibrated path carries no marker convention at present
    fidelity).

    Parameters:
        frame: The frame whose signal plane is modified in place.
        params: The full scene mapping; reads the ``noise`` block's
            ``missing_data_rate`` and the emulated instrument's marker value.
        rng: The stage generator, used for dropout placement.
    """
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    if inst_config.get('data_units', 'raw_dn') != 'raw_dn':
        return
    sim_noise = DEFAULT_CONFIG.category('sim')['noise']
    scene_noise = params.get('noise') or {}
    missing_data_rate = float(
        scene_noise.get('missing_data_rate', sim_noise.get('missing_data_rate', 0.0))
    )
    if missing_data_rate <= 0.0:
        return
    inst_noise = inst_config.get('noise') or {}
    marker_dn = float(inst_noise.get('marker_value', 0))
    missing_mask = rng.random(size=frame.signal.shape) < missing_data_rate
    frame.signal[missing_mask] = marker_dn
