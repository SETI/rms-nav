"""The information boundary: no truth key is reachable through nav_params.

This is the independence guarantee of the simulator-realism program: the
navigator-side models consume only ``obs.nav_params``, and this test builds a
scene exercising EVERY entry of the schema's ``TRUTH_KEYS`` inventory and
asserts none is reachable through the filtered view.  The test iterates the
frozenset itself, so a truth key added to the schema without a filter/test
extension fails here, not silently.
"""

from typing import Any

import pytest

from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.scene import (
    TOP_LEVEL_TRUTH_KEYS,
    TRUTH_KEYS,
    build_nav_params,
    validate_sim_params,
)

# One concrete exercising value per truth key.  The completeness check below
# fails if TRUTH_KEYS gains an entry with no sample here, so a later phase
# adding a truth key must extend this scene in the same change.
_TRUTH_SAMPLES: dict[str, Any] = {
    'random_seed': 99,
    'offset_v': 2.5,
    'offset_u': -1.5,
    'offset_rotation_deg': 0.4,
    'shade_solid_rings': True,
    'sky_counts': {'a': -3.1, 'b': 0.34, 'density_factor': 5.0, 'diffuse_e_per_px': 1.0},
    'noise': {'poisson': True, 'read_noise_dn': 4.0, 'bias_dn': 12.0},
    'oversample': 4,
    'optics': {
        'psf': {'sigma_v': 0.6, 'sigma_u': 0.6, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
        'smear': [{'dv_px': 1.5, 'du_px': 0.0, 'object_class': 'all'}],
        'distortion': {'k1': 0.01, 'k2': 0.0, 'center_v': 48.0, 'center_u': 48.0},
        'ghosts': [{'dv_px': 5.0, 'du_px': -3.0, 'amplitude': 0.05, 'defocus_sigma': 2.0}],
        'stray_light': {'amplitude': 0.2, 'direction_deg': 30.0, 'model': 'linear'},
    },
    'detector': {
        'gain_state': 2,
        'detector_model': 'ccd',
        'exposure_ref_sec': 1.0,
        'quantization': 'exact',
    },
    'artifacts': {
        'instrument_defaults': True,
        'adversarial': True,
        'missing_lines': {'incidence': 2.0, 'contiguous_run': True},
        'hot_pixels': {'incidence': 0.001, 'amplitude_e': 40000.0},
    },
    'spk_error': {'dv_px': 0.8, 'du_px': -0.4, 'reference_range_km': 100000.0},
    'bodies.crater_fill': 0.4,
    'bodies.crater_min_radius': 0.06,
    'bodies.crater_max_radius': 0.2,
    'bodies.crater_power_law_exponent': 2.5,
    'bodies.crater_relief_scale': 0.5,
    'bodies.seed': 11,
    'bodies.anti_aliasing': 0.7,
    'bodies.nav_override': {'mesh_lumpiness': 0.0},
    'stars.psf_sigma': 2.5,
}

# The true (rendered) values nav_override hides from the navigator.
_TRUE_MESH_LUMPINESS = 0.4


def _truth_exercising_scene() -> dict[str, Any]:
    """A validated scene carrying every truth key with a non-default value."""
    scene: dict[str, Any] = {
        'schema_version': 2,
        'scene_name': 'boundary_probe',
        'instrument': 'coiss_nac',
        'size_v': 96,
        'size_u': 96,
        'exposure_sec': 1.5,
        'time': 100.0,
        'ring_epoch': 50.0,
        'closest_planet': 'SATURN',
        'bodies': [
            {
                'name': 'LUMPY',
                'shape_model': 'polyhedral_mesh',
                'center_v': 40.0,
                'center_u': 40.0,
                'axis1': 30.0,
                'axis2': 26.0,
                'axis3': 26.0,
                'illumination_angle': 20.0,
                'phase_angle': 30.0,
                'range_km': 500000.0,
                'mesh_lumpiness': _TRUE_MESH_LUMPINESS,
                'mesh_seed': 3,
                'crater_fill': _TRUTH_SAMPLES['bodies.crater_fill'],
                'crater_min_radius': _TRUTH_SAMPLES['bodies.crater_min_radius'],
                'crater_max_radius': _TRUTH_SAMPLES['bodies.crater_max_radius'],
                'crater_power_law_exponent': (_TRUTH_SAMPLES['bodies.crater_power_law_exponent']),
                'crater_relief_scale': _TRUTH_SAMPLES['bodies.crater_relief_scale'],
                'seed': _TRUTH_SAMPLES['bodies.seed'],
                'anti_aliasing': _TRUTH_SAMPLES['bodies.anti_aliasing'],
                'nav_override': dict(_TRUTH_SAMPLES['bodies.nav_override']),
            }
        ],
        'stars': [
            {
                'name': 'S1',
                'v': 70.0,
                'u': 70.0,
                'vmag': 4.0,
                'psf_sigma': _TRUTH_SAMPLES['stars.psf_sigma'],
            }
        ],
        'rings': [
            {
                'name': 'R1',
                'feature_type': 'RINGLET',
                'center_v': 48.0,
                'center_u': 48.0,
                'range_km': 200000.0,
                'inner_data': [{'mode': 1, 'a': 30.0}],
                'outer_data': [{'mode': 1, 'a': 38.0}],
            }
        ],
    }
    for key in TOP_LEVEL_TRUTH_KEYS:
        scene[key] = _TRUTH_SAMPLES[key]
    return validate_sim_params(scene, source='boundary_probe')


def test_every_truth_key_has_an_exercising_sample() -> None:
    """The probe scene covers TRUTH_KEYS exactly (no stale or missing samples)."""
    assert set(_TRUTH_SAMPLES) == set(TRUTH_KEYS)


@pytest.mark.parametrize('truth_key', sorted(TRUTH_KEYS))
def test_truth_key_unreachable_through_nav_params(truth_key: str) -> None:
    """No TRUTH_KEYS entry is reachable through the filtered view."""
    nav = build_nav_params(_truth_exercising_scene())
    if '.' in truth_key:
        block, key = truth_key.split('.', 1)
        for obj in nav.get(block, []):
            assert key not in obj
    else:
        assert truth_key not in nav


def test_probe_scene_actually_carries_every_truth_key() -> None:
    """Guards the parametrized test: the probe exercises what it claims."""
    scene = _truth_exercising_scene()
    for truth_key in TRUTH_KEYS:
        if '.' in truth_key:
            block, key = truth_key.split('.', 1)
            assert any(key in obj for obj in scene[block])
        else:
            assert truth_key in scene


def test_nav_override_is_overlaid_not_exposed() -> None:
    """The navigator sees the believed value; the true value never crosses."""
    nav = build_nav_params(_truth_exercising_scene())
    body = nav['bodies'][0]
    assert body['mesh_lumpiness'] == 0.0
    assert 'nav_override' not in body


def test_idealized_keys_survive_the_filter() -> None:
    """Catalog geometry and epochs the navigator may know pass through."""
    nav = build_nav_params(_truth_exercising_scene())
    assert nav['ring_epoch'] == 50.0
    assert nav['time'] == 100.0
    assert nav['bodies'][0]['range_km'] == 500000.0
    star = nav['stars'][0]
    assert star['vmag'] == 4.0
    assert nav['rings'][0]['inner_data'] == [{'mode': 1, 'a': 30.0}]


def test_obs_sim_exposes_only_the_filtered_view() -> None:
    """ObsSim publishes nav_params and no renderer star records."""
    obs = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=_truth_exercising_scene())
    for truth_key in TRUTH_KEYS:
        if '.' in truth_key:
            block, key = truth_key.split('.', 1)
            for obj in obs.nav_params.get(block, []):
                assert key not in obj
        else:
            assert truth_key not in obs.nav_params
    # The star-list replumb: the renderer's output star records no longer
    # cross to the navigator side under any name.
    assert not hasattr(obs, 'sim_star_list')


def test_star_limit_is_independent_of_scene_noise() -> None:
    """The star detection limit derives from published config, not scene noise.

    Two observations differing only in the truth-side ``noise`` block must
    report the same ``star_max_usable_vmag``: the navigator's limiting
    magnitude comes from the emulated instrument's published detector model,
    so the scene's planted noise cannot leak through this channel.  A scene
    that plants noise different from the published values gets an
    honestly-wrong detection limit by design.
    """
    scene_quiet = _truth_exercising_scene()
    scene_noisy = _truth_exercising_scene()
    scene_noisy['noise'] = {
        'poisson': False,
        'read_noise_dn': 250.0,
        'bias_dn': 0.0,
        'signal_full_scale_frac': 0.01,
    }
    obs_quiet = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=scene_quiet)
    obs_noisy = ObsSim.from_file('/tmp/boundary_probe.yaml', sim_params=scene_noisy)
    assert obs_noisy.star_max_usable_vmag() == obs_quiet.star_max_usable_vmag()


def test_nav_params_values_are_isolated_copies() -> None:
    """Mutating the filtered view cannot reach back into the scene."""
    scene = _truth_exercising_scene()
    nav = build_nav_params(scene)
    nav['bodies'][0]['axis1'] = 999.0
    nav['rings'][0]['inner_data'][0]['a'] = 999.0
    assert scene['bodies'][0]['axis1'] == 30.0
    assert scene['rings'][0]['inner_data'][0]['a'] == 30.0
