"""Ring/body depth ordering when rings carry a physical range.

A ring that carries ``range_km`` sorts against the bodies' physical
``range_km`` (spk_error scenes require ranges on every object, so they always
order physically); a ring without one falls back to the hint-unit ``range``
key, whose defaults keep hint-only rings behind bodies.
"""

from typing import Any

import numpy as np

from spindoctor.sim.render import render_combined_model


def _scene(ring_range_km: float | None) -> dict[str, Any]:
    """A body at 1e6 km overlapped by a ring annulus at the given range."""
    ring: dict[str, Any] = {
        'name': 'R',
        'feature_type': 'RINGLET',
        'center_v': 30.0,
        'center_u': 30.0,
        'inner_data': [{'mode': 1, 'a': 8.0, 'ae': 0.0}],
        'outer_data': [{'mode': 1, 'a': 16.0, 'ae': 0.0}],
        'shading_distance': 4.0,
    }
    if ring_range_km is not None:
        ring['range_km'] = ring_range_km
    return {
        'size_v': 60,
        'size_u': 60,
        'random_seed': 5,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [
            {
                'name': 'B',
                'center_v': 30.0,
                'center_u': 30.0,
                'axis1': 20.0,
                'axis2': 20.0,
                'axis3': 20.0,
                'phase_angle': 0.0,
                'range_km': 1.0e6,
            }
        ],
        'rings': [ring],
    }


def test_ring_with_larger_range_km_renders_behind_the_body() -> None:
    """A ring at range_km 2e6 sorts behind a body at 1e6: the body occludes it."""
    body_only = dict(_scene(2.0e6))
    body_only = {k: v for k, v in body_only.items() if k != 'rings'}
    body_img, body_meta = render_combined_model(body_only)
    behind_img, behind_meta = render_combined_model(_scene(2.0e6))
    mask = behind_meta['body_masks'][0]
    assert bool(mask.any())
    assert np.array_equal(behind_img[mask], body_img[body_meta['body_masks'][0]])


def test_ring_with_smaller_range_km_renders_in_front_of_the_body() -> None:
    """A ring at range_km 5e5 sorts in front of the same body and overprints it."""
    body_only = {k: v for k, v in _scene(5.0e5).items() if k != 'rings'}
    body_img, _ = render_combined_model(body_only)
    front_img, front_meta = render_combined_model(_scene(5.0e5))
    mask = front_meta['body_masks'][0]
    assert not np.array_equal(front_img[mask], body_img[mask])


def test_ring_without_range_km_keeps_the_hint_default_behind_bodies() -> None:
    """A hint-only ring (default range 1000+) renders behind default-range bodies.

    The hint-unit fallback is only meaningful against bodies at their default
    ranges (index + 1), so this scene leaves the body's range_km unset.
    """
    scene = _scene(None)
    del scene['bodies'][0]['range_km']
    body_only = {k: v for k, v in scene.items() if k != 'rings'}
    body_img, _ = render_combined_model(body_only)
    hint_img, hint_meta = render_combined_model(scene)
    mask = hint_meta['body_masks'][0]
    assert np.array_equal(hint_img[mask], body_img[mask])
