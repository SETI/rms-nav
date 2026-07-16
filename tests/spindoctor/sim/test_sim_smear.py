"""Motion smear: whole-scene and differential per-object-class blur.

These tests cover the line-segment smear kernel (a point becomes a streak of
the right length and direction), the zero-length no-op, and differential smear
(a stars-only entry trails the star field while a body stays sharp).
"""

from typing import Any

import numpy as np

from spindoctor.sim.forward.smear import apply_smear, smear_kernel
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.sim.render import render_combined_model


def test_zero_length_kernel_is_none() -> None:
    """A negligible drift produces no kernel (a no-op smear)."""
    assert smear_kernel(0.0, 0.0) is None


def test_kernel_streaks_along_the_motion_direction() -> None:
    """A horizontal drift spreads a point along u, not v."""
    kernel = smear_kernel(0.0, 8.0)
    assert kernel is not None
    v_extent = int(np.count_nonzero(kernel.sum(axis=1) > 1e-6))
    u_extent = int(np.count_nonzero(kernel.sum(axis=0) > 1e-6))
    assert u_extent > v_extent
    assert abs(float(kernel.sum()) - 1.0) < 1e-9


def test_whole_scene_smear_streaks_a_point() -> None:
    """A single point smears into a streak of about the commanded length."""
    frame = new_sim_frame(40, 40)
    frame.signal[20, 20] = 1.0
    apply_smear(frame, smear=[{'dv_px': 0.0, 'du_px': 8.0, 'object_class': 'all'}], oversample=1)
    lit_u = np.where(frame.signal[20, :] > 1e-3)[0]
    assert int(lit_u.max() - lit_u.min()) >= 6
    # The streak stays on its row (no spread along v).
    assert float(frame.signal[18, 20]) < 1e-6


def test_whole_scene_smear_conserves_signal() -> None:
    """Averaging over the drift conserves the interior signal."""
    frame = new_sim_frame(40, 40)
    frame.signal[20, 20] = 1.0
    apply_smear(frame, smear=[{'dv_px': 4.0, 'du_px': 0.0, 'object_class': 'all'}], oversample=1)
    assert abs(float(frame.signal.sum()) - 1.0) < 1e-6


def _star_and_body_scene(*, smear: list[dict[str, Any]] | None) -> dict[str, Any]:
    """A scene with a centered body and an off-centre star, optional smear."""
    scene: dict[str, Any] = {
        'size_v': 60,
        'size_u': 60,
        'random_seed': 5,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [{'name': 'B', 'center_v': 40.0, 'center_u': 40.0, 'axis1': 16.0, 'axis2': 16.0}],
        'stars': [{'name': 'S', 'v': 14.0, 'u': 14.0, 'vmag': 0.0}],
    }
    if smear is not None:
        scene['optics'] = {'smear': smear}
    return scene


def test_differential_smear_trails_stars_but_not_bodies() -> None:
    """A stars-only smear spreads the star while the body stays sharp."""
    plain, _ = render_combined_model(_star_and_body_scene(smear=None))
    trailed, _ = render_combined_model(
        _star_and_body_scene(smear=[{'dv_px': 8.0, 'du_px': 0.0, 'object_class': 'stars'}])
    )
    # The body region is untouched by a stars-only smear.
    body_plain = plain[32:48, 32:48]
    body_trailed = trailed[32:48, 32:48]
    assert np.allclose(body_plain, body_trailed, atol=1e-9)
    # The star peak drops as its flux spreads into a trail.
    assert float(trailed[14, 14]) < float(plain[14, 14])
    trail = int(np.count_nonzero(trailed[:, 14] > 1e-3))
    assert trail >= 6
