"""Planted-offset algorithmic invariants (Phase T4).

Each scene under ``sim_scenes/algorithmic_invariants/`` carries a ground-truth
offset that is applied as the rendered offset.  A navigator predicting the
unshifted geometry must recover it.  These tests render each scene into an
ObsSim, navigate it, and assert success plus recovery within tolerance -- a true
invariant (correct by construction), so unlike a baseline it never needs
re-blessing.  Everything is in-process, so this runs in the default suite.
"""

from pathlib import Path
from typing import Any

import pytest

from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import NavOrchestrator
from nav.obs.obs_inst_sim import ObsSim
from nav.sim.scene import SimScene, iter_scene_paths, load_sim_scene

# Recovery tolerance in pixels.  The disc/correlation techniques converge to a
# few tenths of a pixel on these clean scenes; 1.0 px is a safe invariant bound.
_OFFSET_TOLERANCE_PX = 1.0

_INVARIANTS_DIR = Path(__file__).parent / 'sim_scenes' / 'algorithmic_invariants'
_SCENE_PATHS = iter_scene_paths(_INVARIANTS_DIR.parent)
_INVARIANT_PATHS = [p for p in _SCENE_PATHS if p.parent.name == 'algorithmic_invariants']
_INVARIANT_IDS = [p.stem for p in _INVARIANT_PATHS]


def _navigate(scene: SimScene, *, only_techniques: str = '*') -> Any:
    obs = ObsSim.from_file('/tmp/invariant.json', sim_params=scene.to_sim_params())
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=only_techniques
    )
    return orchestrator.navigate(obs)


def test_there_are_invariant_scenes() -> None:
    """The algorithmic-invariants class is populated."""
    assert _INVARIANT_PATHS


@pytest.mark.parametrize('path', _INVARIANT_PATHS, ids=_INVARIANT_IDS)
def test_invariant_scene_navigates(path: Path) -> None:
    """Each planted-offset scene navigates to a success status."""
    result = _navigate(load_sim_scene(path))
    assert result.status == 'success'


@pytest.mark.parametrize('path', _INVARIANT_PATHS, ids=_INVARIANT_IDS)
def test_invariant_recovers_planted_v(path: Path) -> None:
    """Each scene recovers its planted v offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene.ground_truth.planted_offset_dv_px) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _INVARIANT_PATHS, ids=_INVARIANT_IDS)
def test_invariant_recovers_planted_u(path: Path) -> None:
    """Each scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene.ground_truth.planted_offset_du_px) < _OFFSET_TOLERANCE_PX


# The blob scene is small enough that BodyBlobNav -- consuming the BODY_BLOB
# feature NavModelBodySimulated now emits -- is the load-bearing technique.
# Pinning ``only_techniques`` to it proves the simulated body's blob feature is
# both produced and consumable, independent of the disc correlation.
_BLOB_SCENE_PATH = _INVARIANTS_DIR / 'planted_offset_blob.yaml'


def test_blob_scene_navigates_via_blob_alone() -> None:
    """The small-body scene navigates with BodyBlobNav as the only technique."""
    scene = load_sim_scene(_BLOB_SCENE_PATH)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.status == 'success'


def test_blob_alone_recovers_planted_v() -> None:
    """BodyBlobNav alone recovers the planted v offset within tolerance."""
    scene = load_sim_scene(_BLOB_SCENE_PATH)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene.ground_truth.planted_offset_dv_px) < _OFFSET_TOLERANCE_PX


def test_blob_alone_recovers_planted_u() -> None:
    """BodyBlobNav alone recovers the planted u offset within tolerance."""
    scene = load_sim_scene(_BLOB_SCENE_PATH)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene.ground_truth.planted_offset_du_px) < _OFFSET_TOLERANCE_PX
