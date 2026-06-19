"""Planted-offset algorithmic invariants (Phase T4).

Each scene under ``sim_scenes/algorithmic_invariants/`` carries a ground-truth
offset that is applied as the rendered offset.  A navigator predicting the
unshifted geometry must recover it.  These tests render each scene into an
ObsSim, navigate it, and assert success plus offset recovery within tolerance -- a
true invariant (correct by construction), so unlike a baseline it never needs
re-blessing.  Everything is in-process, so this runs in the default suite.

Scenes split into three kinds:

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
* **Star scenes** (``planted_offset_star*``) carry no body or rings, so the
  ``STAR`` features ``NavModelStarsSimulated`` emits are the only signal and the
  star techniques (``StarFieldFromCatalogNav``, ``StarUniqueMatchNav``,
  ``StarRefineNav``) are the load-bearing ones.  The recovery is asserted on the
  full ensemble: with no body there is no disc-spurious fallback to jitter, and
  the recovered offset sits a few hundredths of a pixel from truth, far from the
  tolerance bound.
* **Technique-pinned star scenes** isolate one star technique each.
  ``planted_unique_star*`` is a single star -- too sparse for the field matcher,
  so ``StarUniqueMatchNav``'s one-star path carries it, pinned with
  ``only_techniques``.  ``planted_refine_star*`` is a multi-star field whose
  pass-1 ensemble reaches 'success' and installs a prior, so the pass-2
  ``StarRefineNav`` runs; it is read from the *full*-ensemble per-technique result
  because a prior-requiring pass-2 technique cannot be isolated with
  ``only_techniques``.
"""

import math
from pathlib import Path
from typing import Any

import pytest

from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import NavOrchestrator
from nav.obs.obs_inst_sim import ObsSim
from nav.sim.scene import iter_scene_paths, load_sim_scene

# Recovery tolerance in pixels.  The disc/correlation techniques converge to a
# few tenths of a pixel on these clean scenes; 1.0 px is a safe invariant bound.
_OFFSET_TOLERANCE_PX = 1.0
# Recovery tolerance in degrees for the planted camera roll.  StarField recovers
# the roll to a few hundredths of a degree on a clean field; 0.3 deg is a safe
# invariant bound that absorbs cross-process solver jitter.
_ROTATION_TOLERANCE_DEG = 0.3

_INVARIANTS_DIR = Path(__file__).parent / 'sim_scenes' / 'algorithmic_invariants'
_SCENE_PATHS = iter_scene_paths(_INVARIANTS_DIR.parent)
_INVARIANT_PATHS = [p for p in _SCENE_PATHS if p.parent.name == 'algorithmic_invariants']
_INVARIANT_IDS = [p.stem for p in _INVARIANT_PATHS]

# Scenes whose stem starts with this are blob-designed: recovery is verified
# blob-only because the full ensemble falls to the weak blob fallback and its
# recovered offset jitters across processes near the technique's limit.
_BLOB_STEM_PREFIX = 'planted_offset_blob'
# Star-designed scenes carry no body/rings; the star techniques are load-bearing
# and the full ensemble recovers stably.
_STAR_STEM_PREFIX = 'planted_offset_star'
# Rotation-designed scenes carry a planted camera roll.  Recovery is verified on
# the StarFieldFromCatalogNav per-technique result: that technique recovers the
# roll geometrically (non-spurious), but on a clean sim field its confidence sits
# at the placeholder-alpha floor, so the *fused* status is a stable 'failed'
# rather than 'success' -- which is why the roll is asserted per-technique and the
# scene is held out of the full-ensemble navigate assertion.
_ROTATION_STEM_PREFIX = 'planted_rotation'
# Limb scenes route through a LIMB_ARC the simulated body emits; the full ensemble
# still recovers via the disc correlation (so they double as disc scenes) and the
# limb path is asserted per-technique.
_LIMB_STEM_PREFIX = 'planted_offset_limb'
# Ring scenes route through a RING_EDGE the simulated ring emits; like rotation,
# the technique recovers geometrically but its placeholder-alpha confidence holds
# the fused status at a stable 'failed', so recovery is asserted per-technique and
# the scene is held out of the full-ensemble navigate assertion.
_RING_STEM_PREFIX = 'planted_offset_ring'
# Unique-match scenes are a single unambiguous star: StarFieldFromCatalogNav's
# pattern match is infeasible, so StarUniqueMatchNav's one-star path is the
# load-bearing technique.  Recovery is asserted on its per-technique result
# (navigated with only that technique, which needs no prior).
_UNIQUE_MATCH_STEM_PREFIX = 'planted_unique_star'
# Refine scenes are a pure multi-star field whose pass-1 ensemble reaches
# 'success' and installs a prior, so the pass-2 StarRefineNav runs.  Recovery is
# asserted on its per-technique result read from the full-ensemble navigate
# (only_techniques cannot isolate a prior-requiring pass-2 technique).
_REFINE_STEM_PREFIX = 'planted_refine_star'
_BLOB_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_BLOB_STEM_PREFIX)]
_BLOB_IDS = [p.stem for p in _BLOB_PATHS]
_STAR_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_STAR_STEM_PREFIX)]
_STAR_IDS = [p.stem for p in _STAR_PATHS]
_ROTATION_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_ROTATION_STEM_PREFIX)]
_ROTATION_IDS = [p.stem for p in _ROTATION_PATHS]
_LIMB_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_LIMB_STEM_PREFIX)]
_LIMB_IDS = [p.stem for p in _LIMB_PATHS]
_RING_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_RING_STEM_PREFIX)]
_RING_IDS = [p.stem for p in _RING_PATHS]
_UNIQUE_MATCH_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_UNIQUE_MATCH_STEM_PREFIX)]
_UNIQUE_MATCH_IDS = [p.stem for p in _UNIQUE_MATCH_PATHS]
_REFINE_PATHS = [p for p in _INVARIANT_PATHS if p.stem.startswith(_REFINE_STEM_PREFIX)]
_REFINE_IDS = [p.stem for p in _REFINE_PATHS]
# Disc scenes assert full-ensemble offset recovery.  Blob, star, rotation, ring,
# and the technique-pinned star scenes recover on a pinned technique instead (the
# fused offset is weak, absent, or per-technique), so they are held out; limb
# scenes stay in (they recover via the disc correlation).
_DISC_EXCLUDE = (
    _BLOB_STEM_PREFIX,
    _STAR_STEM_PREFIX,
    _ROTATION_STEM_PREFIX,
    _RING_STEM_PREFIX,
    _UNIQUE_MATCH_STEM_PREFIX,
    _REFINE_STEM_PREFIX,
)
_DISC_PATHS = [p for p in _INVARIANT_PATHS if not p.stem.startswith(_DISC_EXCLUDE)]
_DISC_IDS = [p.stem for p in _DISC_PATHS]
# Scenes whose full ensemble reaches 'success'.  Rotation and ring sit below the
# threshold on a clean field; the unique-match (single star) and refine scenes
# are pinned per-technique, so they are held out of the success assertion too.
_NAVIGATES_EXCLUDE = (
    _ROTATION_STEM_PREFIX,
    _RING_STEM_PREFIX,
    _UNIQUE_MATCH_STEM_PREFIX,
    _REFINE_STEM_PREFIX,
)
_NAVIGATES_SUCCESS_PATHS = [
    p for p in _INVARIANT_PATHS if not p.stem.startswith(_NAVIGATES_EXCLUDE)
]
_NAVIGATES_SUCCESS_IDS = [p.stem for p in _NAVIGATES_SUCCESS_PATHS]


def _navigate(scene: dict[str, Any], *, only_techniques: str = '*') -> Any:
    obs = ObsSim.from_file('/tmp/invariant.yaml', sim_params=scene)
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


def test_there_are_star_scenes() -> None:
    """At least one star-designed scene exists for the star coverage."""
    assert _STAR_PATHS


def test_there_are_rotation_scenes() -> None:
    """At least one planted-roll scene exists for the rotation coverage."""
    assert _ROTATION_PATHS


def test_there_are_limb_scenes() -> None:
    """At least one limb-designed scene exists for the BodyLimbNav coverage."""
    assert _LIMB_PATHS


def test_there_are_ring_scenes() -> None:
    """At least one ring-designed scene exists for the RingEdgeNav coverage."""
    assert _RING_PATHS


def test_there_are_unique_match_scenes() -> None:
    """At least one single-star scene exists for the StarUniqueMatchNav coverage."""
    assert _UNIQUE_MATCH_PATHS


def test_there_are_refine_scenes() -> None:
    """At least one star-field scene exists for the StarRefineNav coverage."""
    assert _REFINE_PATHS


@pytest.mark.parametrize('path', _NAVIGATES_SUCCESS_PATHS, ids=_NAVIGATES_SUCCESS_IDS)
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
    assert abs(result.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _DISC_PATHS, ids=_DISC_IDS)
def test_invariant_recovers_planted_u(path: Path) -> None:
    """Each disc scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


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
    assert abs(result.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _BLOB_PATHS, ids=_BLOB_IDS)
def test_blob_alone_recovers_planted_u(path: Path) -> None:
    """BodyBlobNav alone recovers each blob scene's planted u offset."""
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyBlobNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _STAR_PATHS, ids=_STAR_IDS)
def test_star_scene_recovers_planted_v(path: Path) -> None:
    """Each star scene recovers its planted v offset within tolerance.

    Guards the ``STAR`` feature emission from ``NavModelStarsSimulated`` and the
    half-pixel-correct star rendering: without the eval-offset correction the
    rendered star centroid sits half a pixel from the model's prediction and the
    recovered offset carries a constant bias.  With no body in the frame the
    star techniques are the only contributors, so this is a direct star-path
    invariant on the full ensemble.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _STAR_PATHS, ids=_STAR_IDS)
def test_star_scene_recovers_planted_u(path: Path) -> None:
    """Each star scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    result = _navigate(scene)
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


def _starfield_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with StarFieldFromCatalogNav alone; return its result.

    Reads the per-technique result directly rather than the fused offset because
    the roll is the quantity under test and the technique's confidence (not its
    geometry) is what keeps the fused status below 'success' on a clean field.
    """
    result = _navigate(scene, only_techniques='StarFieldFromCatalogNav')
    matches = [t for t in result.per_technique if t.technique_name == 'StarFieldFromCatalogNav']
    assert matches, 'StarFieldFromCatalogNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _ROTATION_PATHS, ids=_ROTATION_IDS)
def test_rotation_scene_recovers_planted_roll(path: Path) -> None:
    """Each rotation scene recovers its planted camera roll within tolerance.

    Guards the camera-roll rendering (``offset_rotation_deg``) and the
    ``fit_camera_rotation`` scene override: the renderer rotates each star about
    the boresight, and StarFieldFromCatalogNav's similarity fit recovers the
    planted roll.  Asserted on the StarField per-technique result -- the
    technique recovers the roll geometrically (non-spurious) even though its
    placeholder-alpha confidence holds the fused status below 'success'.
    """
    scene = load_sim_scene(path)
    technique = _starfield_result(scene)
    assert not technique.spurious
    assert technique.rotation_rad is not None
    recovered_deg = math.degrees(technique.rotation_rad)
    assert abs(recovered_deg - scene['offset_rotation_deg']) < _ROTATION_TOLERANCE_DEG


@pytest.mark.parametrize('path', _LIMB_PATHS, ids=_LIMB_IDS)
def test_limb_alone_recovers_planted_v(path: Path) -> None:
    """BodyLimbNav alone recovers each limb scene's planted v offset.

    Guards the LIMB_ARC emission from the simulated body: the silhouette boundary
    is sampled into a vertex polyline with outward normals, and BodyLimbNav's
    distance-transform fit aligns it to the image edge.
    """
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyLimbNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _LIMB_PATHS, ids=_LIMB_IDS)
def test_limb_alone_recovers_planted_u(path: Path) -> None:
    """BodyLimbNav alone recovers each limb scene's planted u offset."""
    scene = load_sim_scene(path)
    result = _navigate(scene, only_techniques='BodyLimbNav')
    assert result.offset_px is not None
    assert abs(result.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


def _ring_edge_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with RingEdgeNav alone; return its per-technique result.

    Like the rotation scene, the offset is read from the per-technique result
    because the technique's clean-field confidence (not its geometry) keeps the
    fused status below 'success'.
    """
    result = _navigate(scene, only_techniques='RingEdgeNav')
    matches = [t for t in result.per_technique if t.technique_name == 'RingEdgeNav']
    assert matches, 'RingEdgeNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _RING_PATHS, ids=_RING_IDS)
def test_ring_scene_recovers_planted_v(path: Path) -> None:
    """Each ring scene recovers its planted v offset within tolerance.

    Guards the RING_EDGE emission from the simulated ring: each rendered edge is
    sampled into a radial-normal polyline that RingEdgeNav fits against the
    image-edge distance transform.  Two curved arcs constrain both axes.
    """
    scene = load_sim_scene(path)
    technique = _ring_edge_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _RING_PATHS, ids=_RING_IDS)
def test_ring_scene_recovers_planted_u(path: Path) -> None:
    """Each ring scene recovers its planted u offset within tolerance."""
    scene = load_sim_scene(path)
    technique = _ring_edge_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


def _unique_match_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with StarUniqueMatchNav alone; return its result.

    The technique needs no prior (pass 1), so it can be isolated.  The offset is
    read from the per-technique result because the one-star confidence cap and
    the placeholder-alpha star confidences keep the fused status off a stable
    'success' on a clean single-star field.
    """
    result = _navigate(scene, only_techniques='StarUniqueMatchNav')
    matches = [t for t in result.per_technique if t.technique_name == 'StarUniqueMatchNav']
    assert matches, 'StarUniqueMatchNav produced no technique result'
    return matches[0]


@pytest.mark.parametrize('path', _UNIQUE_MATCH_PATHS, ids=_UNIQUE_MATCH_IDS)
def test_unique_match_scene_recovers_planted_v(path: Path) -> None:
    """Each single-star scene recovers its planted v offset via StarUniqueMatchNav.

    Guards the one-star path: with no companion star the brightness-uniqueness
    gate passes trivially and the brightest detection inside the lone star's
    search window is its unambiguous match, so centroid-minus-prediction is the
    offset.
    """
    scene = load_sim_scene(path)
    technique = _unique_match_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _UNIQUE_MATCH_PATHS, ids=_UNIQUE_MATCH_IDS)
def test_unique_match_scene_recovers_planted_u(path: Path) -> None:
    """Each single-star scene recovers its planted u offset via StarUniqueMatchNav."""
    scene = load_sim_scene(path)
    technique = _unique_match_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX


def _star_refine_result(scene: dict[str, Any]) -> Any:
    """Navigate ``scene`` with the full ensemble; return its StarRefineNav result.

    StarRefineNav requires the pass-1 prior, so it cannot be isolated with
    ``only_techniques`` (that would starve it of a prior and it would fail).  The
    full ensemble runs pass 1 (StarFieldFromCatalogNav + StarUniqueMatchNav) to a
    'success' that installs the prior, then pass 2 runs StarRefineNav -- its
    presence here confirms the prior path fired.
    """
    result = _navigate(scene, only_techniques='*')
    matches = [t for t in result.per_technique if t.technique_name == 'StarRefineNav']
    assert matches, 'StarRefineNav produced no technique result (pass-1 prior did not install)'
    return matches[0]


@pytest.mark.parametrize('path', _REFINE_PATHS, ids=_REFINE_IDS)
def test_refine_scene_recovers_planted_v(path: Path) -> None:
    """Each refine scene recovers its planted v offset via StarRefineNav.

    Guards the pass-2 prior-refinement path: StarRefineNav re-centroids each
    predicted star around the prior-shifted position and reports the
    inverse-variance-weighted refined absolute offset.  Asserting on the
    per-technique result also confirms the prior installed (the technique only
    runs in pass 2 after a 'success' pass-1 ensemble).
    """
    scene = load_sim_scene(path)
    technique = _star_refine_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[0] - scene['offset_v']) < _OFFSET_TOLERANCE_PX


@pytest.mark.parametrize('path', _REFINE_PATHS, ids=_REFINE_IDS)
def test_refine_scene_recovers_planted_u(path: Path) -> None:
    """Each refine scene recovers its planted u offset via StarRefineNav."""
    scene = load_sim_scene(path)
    technique = _star_refine_result(scene)
    assert not technique.spurious
    assert technique.offset_px is not None
    assert abs(technique.offset_px[1] - scene['offset_u']) < _OFFSET_TOLERANCE_PX
