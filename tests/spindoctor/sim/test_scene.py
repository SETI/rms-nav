"""Flat-schema validation, boundary classification, and save/load round-trip."""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.sim.scene import (
    _ALLOWED_KEYS,
    _OBJECT_BLOCKS,
    TOP_LEVEL_IDEALIZED_KEYS,
    TOP_LEVEL_TRUTH_KEYS,
    TRUTH_KEYS,
    SimSceneValidationError,
    load_sim_scene,
    save_sim_scene,
    validate_sim_params,
)


def _sim_params() -> dict[str, Any]:
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'offset_v': 3.0,
        'offset_u': -2.0,
        'bodies': [{'name': 'RHEA', 'center_v': 64.0, 'center_u': 64.0, 'axis1': 80.0}],
        'noise': {'poisson': True, 'read_noise_dn': 4.0},
        'background_stars_num': 12,
    }


def test_save_injects_schema_version_and_scene_name(tmp_path: Path) -> None:
    """A saved scene carries the schema version and the filename stem as its name."""
    path = tmp_path / 'roundtrip.yaml'
    save_sim_scene(_sim_params(), path)
    scene = load_sim_scene(path)
    assert scene['schema_version'] == 2
    assert scene['scene_name'] == 'roundtrip'


def test_loaded_scene_is_the_flat_sim_params(tmp_path: Path) -> None:
    """A loaded scene is the flat sim_params dict the renderer consumes."""
    path = tmp_path / 'rt2.yaml'
    save_sim_scene(_sim_params(), path)
    scene = load_sim_scene(path)
    assert scene['instrument'] == 'coiss_nac'
    assert scene['size_v'] == 128
    assert scene['size_u'] == 128
    assert scene['offset_v'] == 3.0
    assert scene['offset_u'] == -2.0
    assert scene['bodies'][0]['name'] == 'RHEA'
    assert scene['background_stars_num'] == 12


def test_save_then_load_preserves_values(tmp_path: Path) -> None:
    """Saving then loading reproduces every flat field verbatim."""
    path = tmp_path / 'named_scene.yaml'
    params = _sim_params()
    save_sim_scene(params, path)
    scene = load_sim_scene(path)
    for key, value in params.items():
        assert scene[key] == value


def test_load_rejects_bad_instrument(tmp_path: Path) -> None:
    """An unknown instrument fails validation."""
    path = tmp_path / 'bad.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: bad\ninstrument: hubble\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(path)


def test_load_rejects_unknown_key(tmp_path: Path) -> None:
    """An unmodeled top-level key fails validation."""
    path = tmp_path / 'typo.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: typo\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\nwobble: 5\n'
    )
    with pytest.raises(SimSceneValidationError, match='unknown scene keys'):
        load_sim_scene(path)


def test_load_rejects_nonpositive_size(tmp_path: Path) -> None:
    """A non-positive image size fails validation."""
    path = tmp_path / 'small.yaml'
    path.write_text(
        'schema_version: 2\nscene_name: small\ninstrument: coiss_nac\n'
        'size_v: 0\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='size_v'):
        load_sim_scene(path)


def test_load_rejects_schema_version_1(tmp_path: Path) -> None:
    """The loader accepts only the current schema version."""
    path = tmp_path / 'v1.yaml'
    path.write_text(
        'schema_version: 1\nscene_name: v1\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='schema_version must be 2'):
        load_sim_scene(path)


def test_validate_sim_params_accepts_dict_author_scene() -> None:
    """A programmatic scene without schema_version/scene_name validates."""
    params = _sim_params()
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_unknown_body_key() -> None:
    """An unmodeled per-body key fails validation."""
    params = _sim_params()
    params['bodies'][0]['albedo_wobble'] = 0.5
    with pytest.raises(SimSceneValidationError, match=r'bodies\[0\].*albedo_wobble'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_star_key() -> None:
    """An unmodeled per-star key fails validation."""
    params = _sim_params()
    params['stars'] = [{'name': 'S', 'v': 10.0, 'u': 10.0, 'vmag': 5.0, 'colour': 'red'}]
    with pytest.raises(SimSceneValidationError, match=r'stars\[0\].*colour'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_ring_key() -> None:
    """An unmodeled per-ring key fails validation."""
    params = _sim_params()
    params['rings'] = [{'name': 'R', 'feature_type': 'RINGLET', 'tau': 0.5}]
    with pytest.raises(SimSceneValidationError, match=r'rings\[0\].*tau'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_noise_key() -> None:
    """An unmodeled noise key fails validation instead of silently doing nothing."""
    params = _sim_params()
    params['noise'] = {'poisson': True, 'shot_noise': True}
    with pytest.raises(SimSceneValidationError, match=r'noise.*shot_noise'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_the_full_noise_inventory() -> None:
    """Every accepted noise key (including the vidicon sub-map) validates."""
    params = _sim_params()
    params['noise'] = {
        'poisson': True,
        'read_noise_dn': 4.0,
        'bias_dn': 20.0,
        'cosmic_ray_rate_per_sec': 0.001,
        'missing_data_rate': 0.01,
        'signal_full_scale_frac': 0.5,
        'pixel_area_cm2': 1.0,
        'dark_current_e_per_sec': 5.0,
        'hot_pixel_fraction': 0.002,
        'hot_pixel_amplitude_e': 4.0e4,
        'hot_pixel_column_factor': 0.3,
        'banding_amplitude_e': 30.0,
        'banding_period_px': 64.0,
        'bias_pedestal_sigma_dn': 2.0,
        'bias_row_gradient_dn': 1.0,
        'bias_col_gradient_dn': 0.5,
        'bloom_length': 4,
        'vidicon': {
            'read_noise_line_dn': 1.8,
            'read_noise_pixel_dn': 1.8,
            'coherent_amplitude_dn': 0.25,
            'coherent_period_px': 8.0,
        },
    }
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_mistyped_noise_value() -> None:
    """A non-boolean poisson value fails with the offending key named."""
    params = _sim_params()
    params['noise'] = {'poisson': 'yes'}
    with pytest.raises(SimSceneValidationError, match=r'noise\.poisson'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_unknown_vidicon_key() -> None:
    """An unmodeled vidicon sub-key fails validation."""
    params = _sim_params()
    params['noise'] = {'vidicon': {'sweep_rate_dn': 1.0}}
    with pytest.raises(SimSceneValidationError, match=r'vidicon.*sweep_rate_dn'):
        validate_sim_params(params)


def test_validate_sim_params_rejects_uncatalogued_wac_gain_state() -> None:
    """A WAC gain state outside the catalog fails at validation time."""
    params = _sim_params()
    params['instrument'] = 'coiss_wac'
    params['detector'] = {'gain_state': 3}
    with pytest.raises(
        SimSceneValidationError, match=r'gain_state 3 is not catalogued.*coiss_wac.*\[2\]'
    ):
        validate_sim_params(params)


def test_validate_sim_params_accepts_catalogued_gain_state() -> None:
    """The catalogued WAC state 2 validates."""
    params = _sim_params()
    params['instrument'] = 'coiss_wac'
    params['detector'] = {'gain_state': 2}
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_v1_body_range_key() -> None:
    """The v1 per-body 'range' key is gone; 'range_km' replaced it."""
    params = _sim_params()
    params['bodies'][0]['range'] = 500000.0
    with pytest.raises(SimSceneValidationError, match=r'bodies\[0\].*range'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_body_range_km() -> None:
    """The per-body 'range_km' key validates."""
    params = _sim_params()
    params['bodies'][0]['range_km'] = 500000.0
    assert validate_sim_params(params) is params


def test_validate_sim_params_rejects_truth_key_in_nav_override() -> None:
    """nav_override may only carry idealized body keys (believed geometry)."""
    params = _sim_params()
    params['bodies'][0]['nav_override'] = {'crater_fill': 0.0}
    with pytest.raises(SimSceneValidationError, match='nav_override may only override'):
        validate_sim_params(params)


def test_validate_sim_params_accepts_idealized_nav_override() -> None:
    """nav_override with idealized keys (the shape-mismatch fixture) validates."""
    params = _sim_params()
    params['bodies'][0]['nav_override'] = {'mesh_lumpiness': 0.0}
    assert validate_sim_params(params) is params


def test_every_top_level_key_is_classified() -> None:
    """Every allowed top-level key has exactly one boundary classification."""
    classified = TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS
    assert classified == _ALLOWED_KEYS
    assert not TOP_LEVEL_IDEALIZED_KEYS & TOP_LEVEL_TRUTH_KEYS


@pytest.mark.parametrize('block', sorted(_OBJECT_BLOCKS))
def test_every_object_key_is_classified(block: str) -> None:
    """Every allowed per-object key has exactly one boundary classification."""
    allowed, idealized, truth = _OBJECT_BLOCKS[block]
    assert idealized | truth == allowed
    assert not idealized & truth


def test_truth_keys_cover_top_level_and_blocks() -> None:
    """TRUTH_KEYS carries the top-level names plus dotted per-block paths."""
    assert TOP_LEVEL_TRUTH_KEYS <= TRUTH_KEYS
    for block, (_allowed, _idealized, truth) in _OBJECT_BLOCKS.items():
        for key in truth:
            assert f'{block}.{key}' in TRUTH_KEYS
