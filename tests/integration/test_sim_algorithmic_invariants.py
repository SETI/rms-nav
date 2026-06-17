"""Planted-offset algorithmic invariants (Phase T4).

Each scene under ``sim_scenes/algorithmic_invariants/`` carries a ground-truth
offset that is applied as the rendered offset.  A navigator predicting the
unshifted geometry must recover it.  These tests render each scene into an
ObsSim, navigate it, and assert success plus offset recovery within tolerance -- a
true invariant (correct by construction), so unlike a baseline it never needs
re-blessing.  Everything is in-process, so this runs in the default suite.

Scenes split into two kinds:

* **Disc scenes** navigate by ``BodyDiscCorrelateNav`` and the *full ensemble*
  recovers the planted offset stably, so they exercise the generic full-ensemble
  recovery assertions.
* **Blob scenes** (``planted_offset_blob*``) are designed so ``BodyBlobNav`` --
  consuming the ``BODY_BLOB`` feature ``NavModelBodySimulated`` emits -- is the
  load-bearing technique.  Their recovery is asserted with ``only_techniques``
  pinned to the blob, which proves the simulated body's blob feature is produced
  and consumable and is deterministic.  The full ensemble on these scenes still
  navigates to success (a disc-spurious high-phase crescent falls to the blob),
  but its recovered offset sits near the weak fallback's limit and jitters
  across processes under parallel BLAS load, so the *tolerance* assertion is made
  blob-only rather than on the full ensemble.
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

# Scenes whose stem starts with this are blob-designed: recovery is verified
# blob-only because the full ensemble falls to the weak blob fallback and its
# recovered offset jitters across processes near the technique's limit.
_BLOB_STEM_PREFIX = 'planted_offset_blob'
_DISC_PATHS = [p for p in _INVARIANT_PATHS if not p.stem.startswith(_BLOB_STEM_PREFIX)]
_DISC_IDS = [p.stem for p in _DISC_PATHS]
_BLOB_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_BLOB_STEM_PREFIX)]
_BLOB_IDS = [p.stem for p in _BLOB_PATHS]


def _navigate(scene: SimScene, *, only_techniques: str = '*') -> Any:
    obs = ObsSim.from_file('/tmp/invariant.json', sim_params=scene.to_sim_params())
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=only_techniques
    )
    return orchestrator.navigate(obs)


def test_there_are_invariant_scenes() -> None:
    """The algorithmic-invariants class is populated."""
    assert _INVARIANT_PATHS


def test_there_are_blob_scenes() -> None:
    """At least one blob-designed scene exists for the blob-only coverage."""
    assert _BLOB_PATHS


@pytest.mark.parametrize('path', _INVARIANT_PATHS, ids=_INVARIANT_IDS)
def test_invariant_scene_navigates(path: Path) -> None:
    """Each planted-offset scene navigates to a success status (full ensemble)."""
    result = _navigate(load_sim_scene(path))
    assert result.status == 'success'


@pytest.mark.parametrize('path', _DISC_PATHS, ids=_DISC_IDS)
def test_invariant_recovers_planted_v(path: Path) -> None:
    """Each disc scene recovers its planted v offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene.ground_truth.planted_offset_dv_px) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _DISC_PATHS, ids=_DISC_IDS)
def test_invariant_recovers_planted_u(path: Path) -> None:
    """Each disc scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene.ground_truth.planted_offset_du_px) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_scene_navigates_via_blob_alone(path: Path) -> None:
    """Each blob scene navigates with BodyBlobNav as the only technique."""
    result = _navigate(load_sim_scene(path), only_techniques='BodyBlobNav')
    assert result.status == 'success'


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_alone_recovers_planted_v(path: Path) -> None:
    """BodyBlobNav alone recovers each blob scene's planted v offset.

    Guards the BODY_BLOB feature emission, the background-pedestal subtraction,
    and the sky-noise threshold; without the latter two the thresholded observed
    centroid diverges from the model's continuous predicted centroid by several
    pixels on the high-phase crescent scene.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene.ground_truth.planted_offset_dv_px) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_alone_recovers_planted_u(path: Path) -> None:
    """BodyBlobNav alone recovers each blob scene's planted u offset."""
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene.ground_truth.planted_offset_du_px) < _OFFSET_TOLERANCE_PX
