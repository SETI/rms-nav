"""Flat-schema validation and save/load round-trip (spindoctor.sim.scene)."""

from pathlib import Path
from typing import Any

import pytest

from spindoctor.sim.scene import (
    SimSceneValidationError,
    load_sim_scene,
    save_sim_scene,
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
    assert scene['schema_version'] == 1
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
        'schema_version: 1\nscene_name: bad\ninstrument: hubble\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(path)


def test_load_rejects_unknown_key(tmp_path: Path) -> None:
    """An unmodeled top-level key fails validation."""
    path = tmp_path / 'typo.yaml'
    path.write_text(
        'schema_version: 1\nscene_name: typo\ninstrument: coiss_nac\n'
        'size_v: 64\nsize_u: 64\nrandom_seed: 1\nwobble: 5\n'
    )
    with pytest.raises(SimSceneValidationError, match='unknown scene keys'):
        load_sim_scene(path)


def test_load_rejects_nonpositive_size(tmp_path: Path) -> None:
    """A non-positive image size fails validation."""
    path = tmp_path / 'small.yaml'
    path.write_text(
        'schema_version: 1\nscene_name: small\ninstrument: coiss_nac\n'
        'size_v: 0\nsize_u: 64\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='size_v'):
        load_sim_scene(path)
