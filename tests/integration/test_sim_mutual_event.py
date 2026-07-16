"""Behavioral checks for the mutual-event scene class.

A grazing two-body overlap must still navigate to its planted offset within
tolerance, and the navigator-side simulated body model must predict the FULL
limb of both bodies -- no masking of the occluded arc exists, so the hidden
arc is genuine model error the robust limb fit has to absorb.  The deeper
overlaps' fused outcomes are pinned by their scenes' ``expected`` blocks
(test_sim_expected) and their baselines.

Everything renders and navigates in-process (no holdings or SPICE), so this
runs in the default suite.
"""

from pathlib import Path
from typing import Any

import numpy as np

from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_model import build_models_for_obs
from spindoctor.nav_orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import load_sim_scene
from tests.shims.context import bare_nav_context

_SCENES_DIR = Path(__file__).parent / 'sim_scenes' / 'mutual_event'


def _load(scene_name: str) -> dict[str, Any]:
    return load_sim_scene(_SCENES_DIR / f'{scene_name}.yaml')


def _navigate(scene: dict[str, Any]) -> NavResult:
    obs = ObsSim.from_file('/tmp/mutual_event.yaml', sim_params=scene)
    orchestrator = NavOrchestrator(build_models_for_obs(obs), only_models='*', only_techniques='*')
    return orchestrator.navigate(obs)


def test_grazing_overlap_navigates_within_tolerance() -> None:
    """The grazing mutual event recovers its planted offset to sub-pixel."""
    scene = _load('mutual_grazing')
    result = _navigate(scene)
    assert result.status == 'success'
    assert result.offset_px is not None
    assert abs(result.offset_px[0] - scene['offset_v']) < 0.5
    assert abs(result.offset_px[1] - scene['offset_u']) < 0.5


def test_navigator_predicts_full_limbs_for_both_bodies() -> None:
    """Both bodies emit complete limb arcs: the occluded arc is model error.

    ``NavModelBodySimulated`` renders each body in isolation, so the far
    body's predicted limb circle is complete even where the near body hides
    it in the image -- there is no occlusion masking on the navigator side.
    The vertices must surround each body's center in all four azimuthal
    quadrants, and the feature must claim a fully visible arc.
    """
    scene = _load('mutual_deep')
    obs = ObsSim.from_file('/tmp/mutual_event.yaml', sim_params=scene)
    context = bare_nav_context(obs)
    limb_features = {}
    for model in build_models_for_obs(obs):
        model.create_model()
        for feature in model.to_features(context):
            if feature.feature_type is NavFeatureType.LIMB_ARC:
                limb_features[feature.feature_id] = feature
    assert set(limb_features) == {'limb_arc:FAR', 'limb_arc:NEAR'}
    for body, center_u in (('FAR', 95.0), ('NEAR', 110.0)):
        feature = limb_features[f'limb_arc:{body}']
        assert feature.flags.visible_arc_fraction == 1.0  # type: ignore[union-attr]
        vertices = feature.geometry.vertices_vu  # type: ignore[union-attr]
        dv = vertices[:, 0] - (128.0 + obs.extfov_margin_v)
        du = vertices[:, 1] - (center_u + obs.extfov_margin_u)
        quadrants = {(bool(a), bool(b)) for a, b in zip(dv >= 0, du >= 0, strict=True)}
        assert quadrants == {(False, False), (False, True), (True, False), (True, True)}, (
            f'{body} limb does not surround its center: {sorted(quadrants)}'
        )
        # The limb is a closed circle: azimuth coverage has no gap larger
        # than a few degrees.
        azimuths = np.sort(np.arctan2(du, dv))
        gaps = np.diff(np.concatenate([azimuths, [azimuths[0] + 2.0 * np.pi]]))
        assert float(np.degrees(gaps.max())) < 10.0
