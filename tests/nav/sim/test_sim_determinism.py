"""Determinism guarantees for the simulator's randomized effects.

These tests pin the seed-derivation contract (process-stable, per-effect
independent sub-seeds) and assert that rendering the same scene twice produces
byte-identical pixels even after the render caches are cleared.
"""

from typing import Any

import numpy as np

from nav.sim import render
from nav.sim.seeds import derive_effect_seed, stable_param_seed

# Pinned values guard against an accidental switch to a salted hash (e.g. the
# built-in ``hash``): a salted derivation would not reproduce these constants.
_EXPECTED_NOISE_SEED = 2193400513
_EXPECTED_BACKGROUND_STARS_SEED = 2528068292
_EXPECTED_CRATERS_SEED = 4259976418
_EXPECTED_PARAM_SEED = 3733498255

_SEED_CEILING = 2**32


def _clear_render_caches() -> None:
    """Drop every lru_cache in the render module so RNG paths re-run."""
    render._render_combined_model_cached.cache_clear()
    render._render_stars_cached.cache_clear()
    render._render_body_shape_cached.cache_clear()
    render._render_bodies_positioned_cached.cache_clear()
    render._render_background_noise_cached.cache_clear()
    render._render_background_stars_cached.cache_clear()


def _scene() -> dict[str, Any]:
    """A scene exercising noise, background stars, and crater RNG paths."""
    return {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 42,
        'background_noise_intensity': 0.05,
        'background_stars_num': 20,
        'bodies': [
            {
                'name': 'SIM-BODY-1',
                'center_v': 32.0,
                'center_u': 32.0,
                'axis1': 30.0,
                'axis2': 24.0,
                'axis3': 24.0,
                'crater_fill': 0.3,
            }
        ],
    }


def test_derive_effect_seed_is_process_stable() -> None:
    """The noise sub-seed matches its pinned, salt-free value."""
    assert derive_effect_seed(42, 'noise') == _EXPECTED_NOISE_SEED


def test_derive_effect_seed_distinct_per_effect() -> None:
    """Different effect names derive different streams from one scene seed."""
    seeds = {
        derive_effect_seed(42, 'noise'),
        derive_effect_seed(42, 'background_stars'),
        derive_effect_seed(42, 'craters'),
    }
    assert len(seeds) == 3


def test_derive_effect_seed_distinct_per_scene_seed() -> None:
    """The same effect derives different streams for different scene seeds."""
    assert derive_effect_seed(42, 'noise') != derive_effect_seed(43, 'noise')


def test_derive_effect_seed_in_randomstate_range() -> None:
    """Derived seeds stay within the RandomState-accepted range."""
    assert 0 <= derive_effect_seed(42, 'noise') < _SEED_CEILING


def test_pinned_effect_seeds_match() -> None:
    """All three effect sub-seeds match their pinned values."""
    assert derive_effect_seed(42, 'background_stars') == _EXPECTED_BACKGROUND_STARS_SEED
    assert derive_effect_seed(42, 'craters') == _EXPECTED_CRATERS_SEED


def test_stable_param_seed_is_process_stable() -> None:
    """The geometry fallback seed matches its pinned, salt-free value."""
    assert stable_param_seed(1.0, 2.0, 3.0, (4.0, 5.0)) == _EXPECTED_PARAM_SEED


def test_stable_param_seed_in_randomstate_range() -> None:
    """The fallback seed stays within the RandomState-accepted range."""
    assert 0 <= stable_param_seed(1.0, 2.0, 3.0, (4.0, 5.0)) < _SEED_CEILING


def test_render_is_byte_identical_across_cache_clears() -> None:
    """Re-rendering a scene after clearing caches reproduces identical pixels."""
    _clear_render_caches()
    img_a, _ = render.render_combined_model(_scene())
    _clear_render_caches()
    img_b, _ = render.render_combined_model(_scene())
    assert np.array_equal(img_a, img_b)


def test_render_body_index_map_identical_across_cache_clears() -> None:
    """The body index map is also reproduced byte-for-byte."""
    _clear_render_caches()
    _, meta_a = render.render_combined_model(_scene())
    _clear_render_caches()
    _, meta_b = render.render_combined_model(_scene())
    assert np.array_equal(meta_a['body_index_map'], meta_b['body_index_map'])


def test_different_scene_seed_changes_pixels() -> None:
    """Changing only the scene seed changes the rendered noise field."""
    _clear_render_caches()
    img_a, _ = render.render_combined_model(_scene())
    other = _scene()
    other['random_seed'] = 7
    _clear_render_caches()
    img_b, _ = render.render_combined_model(other)
    assert not np.array_equal(img_a, img_b)
