"""Fast regression tests for specific navigation defects surfaced by the sweeps.

Each test pins the affected technique on one dedicated scene under
``sim_scenes/regression/`` and asserts the correct behaviour, so the defect is
guarded in the normal pytest run without executing the full (slow,
runner-only) sweep suite. These tests are in-process and unmarked, so they run in
the default suite.
"""

import math
from pathlib import Path

from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import NavOrchestrator
from nav.obs.obs_inst_sim import ObsSim
from nav.sim.scene import load_sim_scene

_REGRESSION_DIR = Path(__file__).parent / 'sim_scenes' / 'regression'

# A well-resolved disc must recover a whole-pixel offset to a small fraction of a
# pixel.  The correlator upsamples to 1/128 px and reaches that on raw intensity,
# so 0.1 px is a generous bound that the correct behaviour clears comfortably.
_DISC_SUBPIXEL_TOLERANCE_PX = 0.1


def _disc_offset_error(scene_name: str) -> float:
    """Pin BodyDiscCorrelateNav on a regression scene; return its offset error."""
    scene = load_sim_scene(_REGRESSION_DIR / f'{scene_name}.yaml')
    obs = ObsSim.from_file('/tmp/regression.json', sim_params=scene.to_sim_params())
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques='BodyDiscCorrelateNav'
    ).navigate(obs)
    disc = next(
        (t for t in result.per_technique if t.technique_name == 'BodyDiscCorrelateNav'), None
    )
    assert disc is not None
    assert not disc.spurious
    assert disc.offset_px is not None
    gt = scene.ground_truth
    return math.hypot(
        disc.offset_px[0] - gt.planted_offset_dv_px,
        disc.offset_px[1] - gt.planted_offset_du_px,
    )


def test_disc_recovers_whole_pixel_offset() -> None:
    """The disc correlation recovers a whole-pixel offset to within a fraction of a pixel.

    Guards the gradient-magnitude NCC sub-pixel bias: the integer peak is found on
    the gradient surfaces but the sub-pixel offset is refined on raw intensity, so
    a whole-pixel offset no longer carries the ~0.3 px-per-axis rectification bias.
    """
    assert _disc_offset_error('disc_subpixel_offset') < _DISC_SUBPIXEL_TOLERANCE_PX
