"""ObsSim construction publishes the renderer's output metadata.

``ObsSim.from_file`` renders the scene and stores the renderer's truth-side
output on the snapshot for the image-side consumers (recovery scoring, the
backplane writer).  These tests guard the per-body mask map: a renderer
change that stops populating ``body_mask_map`` degrades those consumers
silently, because nothing on the navigation path reads it.
"""

from typing import Any

import numpy as np

from spindoctor.obs.obs_inst_sim import ObsSim

_SIZE = 96


def _body_scene() -> dict[str, Any]:
    """A minimal scene with one named ellipsoid body."""
    return {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'instrument': 'coiss_nac',
        'random_seed': 7,
        'bodies': [
            {
                'name': 'PEBBLE',
                'center_v': _SIZE / 2.0,
                'center_u': _SIZE / 2.0,
                'axis1': 40.0,
                'axis2': 30.0,
            }
        ],
    }


def test_body_mask_map_is_populated() -> None:
    """A scene with a body yields a non-empty per-body mask map."""
    obs = ObsSim.from_file('/tmp/obs_sim_body.yaml', sim_params=_body_scene())
    assert obs.sim_body_mask_map
    assert 'PEBBLE' in obs.sim_body_mask_map


def test_body_mask_shape_matches_the_render() -> None:
    """Each body mask is a boolean plane the shape of the rendered image."""
    obs = ObsSim.from_file('/tmp/obs_sim_body.yaml', sim_params=_body_scene())
    mask = obs.sim_body_mask_map['PEBBLE']
    assert mask.dtype == np.bool_
    assert mask.shape == np.asarray(obs.data).shape
    assert mask.any()
