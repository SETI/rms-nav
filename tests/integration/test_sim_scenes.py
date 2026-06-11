"""Structural invariants for the simulator scene catalog (Phase T1).

These tests only parse and render in-process (no holdings or SPICE), so they run
in the default suite.  They assert every scene validates, every directory is a
declared class, names are consistent and unique, and every scene renders.
"""

from pathlib import Path

import pytest

from nav.sim.render import render_combined_model
from tests.integration.sim_scene import (
    DECLARED_SIM_SCENE_CLASSES,
    SimSceneValidationError,
    iter_scene_paths,
    load_sim_scene,
)

_SCENE_PATHS = iter_scene_paths()
_SCENE_IDS = [f'{p.parent.name}/{p.stem}' for p in _SCENE_PATHS]


def test_catalog_is_not_empty() -> None:
    """The catalog contains at least one scene."""
    assert _SCENE_PATHS


def test_subdirectories_are_declared_classes() -> None:
    """Every populated scene-class directory is a declared class."""
    classes = {p.parent.name for p in _SCENE_PATHS}
    unknown = classes - DECLARED_SIM_SCENE_CLASSES
    assert not unknown, f'undeclared scene classes: {sorted(unknown)}'


def test_scene_names_are_unique() -> None:
    """No two scenes share a scene_name across the catalog."""
    names = [p.stem for p in _SCENE_PATHS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_scene_validates(path: Path) -> None:
    """Each scene YAML parses and validates."""
    scene = load_sim_scene(path)
    assert scene.scene_name == path.stem
    assert scene.scene_class in DECLARED_SIM_SCENE_CLASSES


@pytest.mark.parametrize('path', _SCENE_PATHS, ids=_SCENE_IDS)
def test_scene_renders(path: Path) -> None:
    """Each scene's sim params render without error and produce signal."""
    scene = load_sim_scene(path)
    img, _ = render_combined_model(scene.to_sim_params())
    assert img.shape == tuple(scene.image_size_vu)
    assert float(img.max()) > 0.0


def test_planted_offset_becomes_render_offset() -> None:
    """A planted ground-truth offset maps onto the rendered sim offset."""
    scene = load_sim_scene(
        next(p for p in _SCENE_PATHS if p.parent.name == 'algorithmic_invariants')
    )
    params = scene.to_sim_params()
    assert params['offset_v'] == scene.ground_truth.planted_offset_dv_px
    assert params['offset_u'] == scene.ground_truth.planted_offset_du_px


def test_validator_rejects_unknown_instrument(tmp_path: Path) -> None:
    """An unknown instrument name fails validation."""
    bad = tmp_path / 'bad.yaml'
    bad.write_text(
        'schema_version: 1\nscene_name: bad\ninstrument: hubble\n'
        'image_size_vu: [64, 64]\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='instrument'):
        load_sim_scene(bad)


def test_validator_rejects_name_mismatch(tmp_path: Path) -> None:
    """A scene_name that does not match the filename fails validation."""
    bad = tmp_path / 'actual_name.yaml'
    bad.write_text(
        'schema_version: 1\nscene_name: other_name\ninstrument: coiss_nac\n'
        'image_size_vu: [64, 64]\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='must match filename stem'):
        load_sim_scene(bad)


def test_validator_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A non-current schema_version fails validation."""
    bad = tmp_path / 'v2.yaml'
    bad.write_text(
        'schema_version: 2\nscene_name: v2\ninstrument: coiss_nac\n'
        'image_size_vu: [64, 64]\nrandom_seed: 1\n'
    )
    with pytest.raises(SimSceneValidationError, match='schema_version'):
        load_sim_scene(bad)
