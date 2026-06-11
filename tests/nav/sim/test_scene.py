"""Scene schema mapping and save/load round-trip (nav.sim.scene)."""

from pathlib import Path
from typing import Any

import pytest

from nav.sim.scene import (
    SimSceneValidationError,
    load_sim_scene,
    save_sim_scene,
    scene_dict_from_sim_params,
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


def test_scene_dict_maps_image_size() -> None:
    """The sim image size folds into image_size_vu."""
    scene = scene_dict_from_sim_params(_sim_params(), scene_name='s')
    assert scene['image_size_vu'] == [128, 128]


def test_scene_dict_offset_becomes_ground_truth() -> None:
    """A rendered offset folds back into the planted ground truth."""
    scene = scene_dict_from_sim_params(_sim_params(), scene_name='s')
    assert scene['ground_truth']['planted_offset_dv_px'] == 3.0
    assert scene['ground_truth']['planted_offset_du_px'] == -2.0


def test_scene_dict_background_stars_fold_into_block() -> None:
    """The background star count folds into a stars block."""
    scene = scene_dict_from_sim_params(_sim_params(), scene_name='s')
    assert scene['stars']['background_count'] == 12


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    """A saved scene reloads with matching geometry and planted offset."""
    path = tmp_path / 'roundtrip.yaml'
    save_sim_scene(_sim_params(), path)
    scene = load_sim_scene(path)
    assert scene.instrument == 'coiss_nac'
    assert scene.image_size_vu == (128, 128)
    assert scene.bodies[0]['name'] == 'RHEA'
    assert scene.ground_truth.planted_offset_dv_px == 3.0


def test_round_trip_params_match(tmp_path: Path) -> None:
    """to_sim_params after a save reproduces the offset and instrument."""
    path = tmp_path / 'rt2.yaml'
    save_sim_scene(_sim_params(), path)
    params = load_sim_scene(path).to_sim_params()
    assert params['instrument'] == 'coiss_nac'
    assert params['offset_v'] == 3.0
    assert params['background_stars_num'] == 12


def test_saved_scene_name_matches_filename(tmp_path: Path) -> None:
    """The saved scene_name is the filename stem, so it validates."""
    path = tmp_path / 'named_scene.yaml'
    save_sim_scene(_sim_params(), path)
    assert load_sim_scene(path).scene_name == 'named_scene'


def test_load_rejects_bad_instrument(tmp_path: Path) -> None:
    """An unknown instrument fails validation."""
    path = tmp_path / 'bad.yaml'
    path.write_text(
        'schema_version: 1\nscene_name: bad\ninstrument: hubble\n'
        'image_size_vu: [64, 64]\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(path)
