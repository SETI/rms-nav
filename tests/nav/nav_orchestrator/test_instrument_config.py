"""Tests for ``nav.nav_orchestrator.instrument_config``."""

from __future__ import annotations

from typing import Any

import pytest

from nav.nav_orchestrator.image_classifier import ImageQualityThresholds
from nav.nav_orchestrator.instrument_config import (
    InstrumentSettings,
    instrument_settings_from_obs,
)


class _FakeObs:
    """Minimal stand-in carrying an ``inst_config`` dict."""

    def __init__(self, inst_config: dict[str, Any] | None) -> None:
        self.inst_config = inst_config


def _raw_dn_block() -> dict[str, Any]:
    return {
        'data_units': 'raw_dn',
        'noise': {
            'saturation_dn': 4095.0,
            'full_well_dn': 4095.0,
            'expected_noise_dn': 4.0,
            'marker_value': 0,
            'read_noise_dn': 4.0,
        },
        'image_quality_thresholds': {
            'blank_max_dn': 5.0,
            'saturation_threshold_dn': 4095.0,
            'noisy_threshold_dn': 10.0,
            'max_missing_frac_clean': 0.30,
            'max_overexposed_frac_clean': 0.80,
        },
        'fit_camera_rotation': False,
        'max_rotation_deg': 5.0,
    }


def test_no_inst_config_returns_legacy_defaults() -> None:
    """An obs without inst_config yields legacy raw_dn defaults and no saturation_dn."""
    settings = instrument_settings_from_obs(_FakeObs(None))
    assert isinstance(settings, InstrumentSettings)
    assert settings.data_units == 'raw_dn'
    assert settings.saturation_dn is None
    assert isinstance(settings.thresholds, ImageQualityThresholds)
    assert settings.fit_camera_rotation is False


def test_raw_dn_block_loads_thresholds() -> None:
    """A populated raw_dn inst_config produces the expected thresholds."""
    obs = _FakeObs(_raw_dn_block())
    settings = instrument_settings_from_obs(obs)
    assert settings.data_units == 'raw_dn'
    assert settings.saturation_dn == 4095.0
    assert settings.thresholds.blank_max_dn == 5.0
    assert settings.thresholds.saturation_threshold_dn == 4095.0
    assert settings.thresholds.noisy_threshold == 10.0
    assert settings.thresholds.max_saturation_frac_clean == 0.80
    assert settings.thresholds.max_missing_frac_clean == 0.30


def test_calibrated_if_block_uses_if_thresholds() -> None:
    """A calibrated_if instrument reads saturation_threshold_if etc."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {'marker_value': 'NaN'},
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'saturation_threshold_if': 10.0,
            'noisy_threshold_if': 0.005,
            'max_missing_frac_clean': 0.05,
            'max_overexposed_frac_clean': 0.80,
        },
        'fit_camera_rotation': False,
        'max_rotation_deg': 5.0,
    }
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert settings.data_units == 'calibrated_if'
    assert settings.saturation_dn is None
    assert settings.thresholds.blank_max_dn == 1.0e-3
    assert settings.thresholds.saturation_threshold_dn == 10.0
    assert settings.thresholds.noisy_threshold == 0.005


def test_missing_data_units_raises() -> None:
    """An inst_config without data_units fails fast."""
    block = _raw_dn_block()
    del block['data_units']
    with pytest.raises(ValueError, match='data_units'):
        instrument_settings_from_obs(_FakeObs(block))


def test_unknown_data_units_raises() -> None:
    """A bogus data_units value fails fast."""
    block = _raw_dn_block()
    block['data_units'] = 'reflectance'
    with pytest.raises(ValueError, match='data_units'):
        instrument_settings_from_obs(_FakeObs(block))


def test_raw_dn_missing_noise_block_raises() -> None:
    """A raw_dn instrument without a noise block fails fast."""
    block = _raw_dn_block()
    del block['noise']
    with pytest.raises(ValueError, match='noise block'):
        instrument_settings_from_obs(_FakeObs(block))


def test_raw_dn_missing_saturation_dn_raises() -> None:
    """A raw_dn instrument without saturation_dn fails fast."""
    block = _raw_dn_block()
    del block['noise']['saturation_dn']
    with pytest.raises(ValueError, match='saturation_dn'):
        instrument_settings_from_obs(_FakeObs(block))


def test_max_rotation_must_be_positive() -> None:
    """A non-positive max_rotation_deg fails fast."""
    block = _raw_dn_block()
    block['max_rotation_deg'] = -1.0
    with pytest.raises(ValueError, match='max_rotation_deg'):
        instrument_settings_from_obs(_FakeObs(block))


def test_calibrated_if_default_marker_is_nan() -> None:
    """A calibrated_if instrument with an unset marker_value defaults to NaN."""
    block = {
        'data_units': 'calibrated_if',
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'saturation_threshold_if': 10.0,
            'noisy_threshold_if': 0.005,
        },
        'noise': None,
    }
    settings = instrument_settings_from_obs(_FakeObs(block))
    import math

    assert math.isnan(settings.marker_value)


def test_shipped_inst_configs_load_cleanly() -> None:
    """Every shipped per-instrument config loads through the helper."""
    from nav.config import Config

    config = Config()
    config.read_config()
    blocks = [
        config.category('cassini_iss').get('nac'),
        config.category('cassini_iss').get('wac'),
        config.category('newhorizons_lorri'),
        config.category('voyager_iss'),
        config.category('galileo_ssi'),
    ]
    for block in blocks:
        settings = instrument_settings_from_obs(_FakeObs(block))
        assert settings.data_units == 'raw_dn'
        assert settings.saturation_dn is not None
        assert settings.saturation_dn > 0.0
