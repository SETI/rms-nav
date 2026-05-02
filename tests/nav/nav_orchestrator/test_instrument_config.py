"""Tests for ``nav.nav_orchestrator.instrument_config``."""

from __future__ import annotations

import math
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
    # raw_dn instruments default to a unity DN-to-image-unit scale.
    assert settings.signal_dn_to_image_unit_scale == 1.0


def test_raw_dn_explicit_unity_signal_scale_accepted() -> None:
    """raw_dn instruments may set signal_dn_to_image_unit_scale=1.0 explicitly."""
    block = _raw_dn_block()
    block['noise']['signal_dn_to_image_unit_scale'] = 1.0
    settings = instrument_settings_from_obs(_FakeObs(block))
    assert settings.signal_dn_to_image_unit_scale == 1.0


def test_raw_dn_non_unity_signal_scale_rejected() -> None:
    """raw_dn instruments must keep the scale at 1.0; any other value fails fast."""
    block = _raw_dn_block()
    block['noise']['signal_dn_to_image_unit_scale'] = 0.5
    with pytest.raises(ValueError, match='signal_dn_to_image_unit_scale'):
        instrument_settings_from_obs(_FakeObs(block))


def test_calibrated_if_block_uses_if_thresholds() -> None:
    """calibrated_if reads I/F thresholds and disables the saturation gate."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {
            'marker_value': 'NaN',
            'signal_dn_to_image_unit_scale': 5.0e-7,
        },
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
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
    # Per Phase 10 §F: the saturation gate is intentionally off for
    # calibrated_if, so the classifier sees an inf threshold and
    # ``saturation_frac`` stays 0.0 regardless of pixel values.
    assert math.isinf(settings.thresholds.saturation_threshold_dn)
    assert settings.thresholds.noisy_threshold == 0.005
    assert settings.signal_dn_to_image_unit_scale == pytest.approx(5.0e-7)


def test_calibrated_if_rejects_explicit_saturation_threshold() -> None:
    """A stale ``saturation_threshold_if`` field fails fast at load time."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {
            'marker_value': 'NaN',
            'signal_dn_to_image_unit_scale': 5.0e-7,
        },
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'saturation_threshold_if': 10.0,
            'noisy_threshold_if': 0.005,
        },
    }
    with pytest.raises(ValueError, match='saturation_threshold_if'):
        instrument_settings_from_obs(_FakeObs(block))


def test_calibrated_if_missing_signal_scale_raises() -> None:
    """calibrated_if without signal_dn_to_image_unit_scale fails fast."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {'marker_value': 'NaN'},
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'noisy_threshold_if': 0.005,
        },
    }
    with pytest.raises(ValueError, match='signal_dn_to_image_unit_scale'):
        instrument_settings_from_obs(_FakeObs(block))


def test_calibrated_if_non_positive_signal_scale_raises() -> None:
    """A zero or negative DN-to-image-unit scale fails fast."""
    block = {
        'data_units': 'calibrated_if',
        'noise': {
            'marker_value': 'NaN',
            'signal_dn_to_image_unit_scale': 0.0,
        },
        'image_quality_thresholds': {
            'blank_max_if': 1.0e-3,
            'noisy_threshold_if': 0.005,
        },
    }
    with pytest.raises(ValueError, match='signal_dn_to_image_unit_scale'):
        instrument_settings_from_obs(_FakeObs(block))


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
            'noisy_threshold_if': 0.005,
        },
        # The signal scale is required for calibrated_if; the marker
        # default check still applies once the rest of the noise block
        # is well-formed.
        'noise': {'signal_dn_to_image_unit_scale': 5.0e-7},
    }
    settings = instrument_settings_from_obs(_FakeObs(block))

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
        assert settings.signal_dn_to_image_unit_scale == 1.0


def test_shipped_calibrated_if_configs_load_cleanly() -> None:
    """Every shipped calibrated-IF block loads with a populated signal scale."""
    from nav.config import Config

    config = Config()
    config.read_config()
    blocks = [
        config.category('cassini_iss_calib').get('nac'),
        config.category('cassini_iss_calib').get('wac'),
    ]
    for block in blocks:
        settings = instrument_settings_from_obs(_FakeObs(block))
        assert settings.data_units == 'calibrated_if'
        assert settings.saturation_dn is None
        assert settings.signal_dn_to_image_unit_scale > 0.0
        assert settings.signal_dn_to_image_unit_scale != 1.0
        # Saturation gate disabled for calibrated_if (Phase 10 §F).
        assert math.isinf(settings.thresholds.saturation_threshold_dn)
