"""Per-instrument coupling for the simulator (B2).

A sim scene may name an instrument so the rendered frame and its obs go through
the same per-instrument units, noise, and saturation settings the navigator
applies to a real frame.  These tests cover the config resolver, the obs-level
``InstrumentSettings``, and the DN-vs-I/F rendering split.
"""

from typing import Any

import pytest

from nav.config import DEFAULT_CONFIG
from nav.nav_orchestrator.instrument_config import instrument_settings_from_obs
from nav.obs.obs_inst_sim import ObsSim
from nav.sim.instruments import resolve_sim_inst_config
from nav.sim.render import render_combined_model


def _obs(instrument: str | None, *, size: int = 64) -> ObsSim:
    """Build an ObsSim for the given instrument from a minimal scene."""
    params: dict[str, Any] = {'size_v': size, 'size_u': size, 'random_seed': 1}
    if instrument is not None:
        params['instrument'] = instrument
    return ObsSim.from_file('/tmp/sim_test.json', sim_params=params)


def test_resolver_maps_coiss_nac_to_raw_dn() -> None:
    """The coiss_nac name resolves to the Cassini NAC raw block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    assert cfg['data_units'] == 'raw_dn'


def test_resolver_maps_coiss_nac_saturation() -> None:
    """The resolved coiss_nac block carries the Cassini full-well DN."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'coiss_nac')
    assert float(cfg['noise']['saturation_dn']) == 4095.0


def test_resolver_maps_vgiss_to_calibrated_if() -> None:
    """The vgiss name resolves to a calibrated-IF block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, 'vgiss')
    assert cfg['data_units'] == 'calibrated_if'


def test_resolver_generic_returns_sim_block() -> None:
    """A None instrument resolves to the standalone sim block."""
    cfg = resolve_sim_inst_config(DEFAULT_CONFIG, None)
    assert cfg['data_units'] == 'raw_dn'
    assert 'signal_full_scale_frac' in cfg['noise']


def test_resolver_rejects_unknown_instrument() -> None:
    """An unrecognised instrument name raises with a clear message."""
    with pytest.raises(ValueError, match='unknown sim instrument'):
        resolve_sim_inst_config(DEFAULT_CONFIG, 'hubble_wfc3')


def test_settings_coiss_nac_saturation() -> None:
    """A coiss_nac obs reports the Cassini saturation DN to the orchestrator."""
    settings = instrument_settings_from_obs(_obs('coiss_nac'))
    assert settings.saturation_dn == 4095.0


def test_settings_gossi_enables_rotation_fit() -> None:
    """A gossi obs carries Galileo's camera-rotation-fit flag."""
    settings = instrument_settings_from_obs(_obs('gossi'))
    assert settings.fit_camera_rotation is True


def test_settings_vgiss_is_calibrated_if() -> None:
    """A vgiss obs reports calibrated-IF units with no saturation gate."""
    settings = instrument_settings_from_obs(_obs('vgiss'))
    assert settings.data_units == 'calibrated_if'
    assert settings.saturation_dn is None


def test_settings_generic_defaults_to_raw_dn() -> None:
    """An instrument-less obs falls back to the generic raw_dn sim block."""
    settings = instrument_settings_from_obs(_obs(None))
    assert settings.data_units == 'raw_dn'


def _lit_body_scene(instrument: str, *, size: int = 64) -> dict[str, Any]:
    """A scene with a single lit body for the given instrument."""
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': instrument,
        'bodies': [
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'axis1': size * 0.6,
                'axis2': size * 0.5,
                'axis3': size * 0.5,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
            }
        ],
    }


def test_raw_dn_render_is_in_dn() -> None:
    """A raw_dn instrument render reaches DN well above the [0, 1] signal range."""
    img, _ = render_combined_model(_lit_body_scene('coiss_nac'))
    assert float(img.max()) > 100.0


def test_gossi_scales_to_8bit_well() -> None:
    """The gossi render scales to its 8-bit well rather than the 12-bit default."""
    img, _ = render_combined_model(_lit_body_scene('gossi'))
    # full_well 255 * default frac 0.5 -> lit signal peaks near ~128 DN, so the
    # frame stays well under the 12-bit default scale.
    assert float(img.max()) < 255.0


def test_calibrated_if_render_stays_in_if_range() -> None:
    """A calibrated_if instrument render leaves the signal in I/F [0, 1]."""
    img, _ = render_combined_model(_lit_body_scene('vgiss'))
    assert float(img.max()) <= 1.0
