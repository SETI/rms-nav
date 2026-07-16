"""Scene-radiance stage: compose the noise-free signal image.

Composes background stars, catalog stars, and the ring/body stack (far to
near, nearer objects overwriting) into the frame's normalized signal plane,
applying the scene's planted pointing offset and camera roll.  Feature truth
(rendered star records, body masks, inventory, z-order maps) is accumulated
into ``frame.truth`` for the renderer's output metadata.

Present-fidelity placeholders: composition happens directly on the detector
grid (``oversample == 1``) with per-element anti-aliasing instead of an
oversampled radiance image, stars are PSF-spread in signal units rather than
deposited into ``point_e`` as electrons, and occlusion is mask-overwrite
rather than transparency compositing (phases B, D, and F).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.forward.body import render_single_body
from spindoctor.sim.forward.ring import composite_ring
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.forward.star import render_background_stars, render_stars
from spindoctor.sim.instruments import resolve_sim_inst_config
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.types import NDArrayBoolType, NDArrayIntType

__all__ = ['compose_scene_radiance']


def compose_scene_radiance(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Compose the scene's noise-free radiance into the frame in place.

    Parameters:
        frame: The frame whose signal plane is composed in place; its
            ``truth`` dict receives the renderer output metadata (``stars``,
            ``bodies``, ``rings``, ``inventory``, ``star_info``,
            ``body_masks``, ``ring_masks``, ``order_near_to_far``,
            ``body_index_map``, ``body_mask_map``).
        params: The full scene mapping.
        rng: The stage generator.  Unused directly: this stage's randomized
            sub-effects (background stars, craters) run behind parameter-keyed
            caches that need scalar seeds, so they derive named sub-seeds from
            the scene's ``random_seed`` instead of consuming generator state.
    """
    del rng
    img = frame.signal
    size_v, size_u = img.shape
    random_seed = int(params.get('random_seed', 42))
    background_stars_seed = derive_effect_seed(random_seed, 'scene_radiance/background_stars')
    crater_seed = derive_effect_seed(random_seed, 'scene_radiance/craters')

    offset_v = float(params.get('offset_v', 0.0))
    offset_u = float(params.get('offset_u', 0.0))
    # Camera roll about the boresight, part of the planted pointing error the
    # navigator recovers; suppressed in GUI preview mode alongside the offset.
    offset_rotation_deg = float(params.get('offset_rotation_deg', 0.0))

    # Resolve the per-instrument config once; stars use its PSF sigma so their
    # centroid diagnostics match the navigator's PSF.
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
    )
    star_psf_sigma = float(inst_config['star_psf_sigma'])

    # Background stars are part of the sky signal, composed before noise.
    background_stars_num = int(params.get('background_stars_num', 0))
    background_stars_psf_sigma = float(params.get('background_stars_psf_sigma', star_psf_sigma))
    background_stars_distribution_exponent = float(
        params.get('background_stars_distribution_exponent', 2.5)
    )
    render_background_stars(
        img,
        background_stars_num,
        background_stars_seed,
        psf_sigma=background_stars_psf_sigma,
        distribution_exponent=background_stars_distribution_exponent,
    )

    stars_params = params.get('stars', []) or []
    bodies_params = params.get('bodies', []) or []
    rings_params = params.get('rings', []) or []

    img, sim_star_list, star_info = render_stars(
        img,
        stars_params,
        offset_v,
        offset_u=offset_u,
        default_psf_sigma=star_psf_sigma,
        rotation_deg=offset_rotation_deg,
    )

    # Process rings: assign default ranges
    for ring_number, ring_params in enumerate(rings_params):
        if 'range' in ring_params:
            ring_params['range'] = float(ring_params['range'])
        else:
            # Default range: start after bodies (assuming bodies use 1, 2, 3, ...)
            # Use a large starting value to ensure rings are behind bodies by default
            ring_params['range'] = float(ring_number + 1000.0)

    # Process bodies: assign default ranges and apply the camera roll.  A roll
    # rotates each body's centre about the boresight and adds to its line-of-sight
    # pose (rotation_z), so a body moves and turns under the same pointing
    # rotation the stars do; the body NavModel predicts the unrolled geometry.
    roll_cos = float(np.cos(np.radians(offset_rotation_deg)))
    roll_sin = float(np.sin(np.radians(offset_rotation_deg)))
    roll_center_v = size_v / 2.0
    roll_center_u = size_u / 2.0
    bodies_with_ranges = []
    for body_number, body_params in enumerate(bodies_params):
        body_params_copy = dict(body_params)
        if 'range' in body_params_copy:
            body_params_copy['range'] = float(body_params_copy['range'])
        else:
            body_params_copy['range'] = float(body_number + 1)
        if offset_rotation_deg != 0.0:
            cv = float(body_params_copy.get('center_v', roll_center_v)) - roll_center_v
            cu = float(body_params_copy.get('center_u', roll_center_u)) - roll_center_u
            body_params_copy['center_v'] = roll_center_v + roll_cos * cv - roll_sin * cu
            body_params_copy['center_u'] = roll_center_u + roll_sin * cv + roll_cos * cu
            body_params_copy['rotation_z'] = (
                float(body_params_copy.get('rotation_z', 0.0)) + offset_rotation_deg
            )
        bodies_with_ranges.append(body_params_copy)

    # Combine rings and bodies, sort by range (far to near)
    render_items: list[tuple[float, str, Any, int]] = []
    for idx, ring_params in enumerate(rings_params):
        render_items.append((ring_params['range'], 'ring', ring_params, idx))
    for idx, body_params in enumerate(bodies_with_ranges):
        render_items.append((body_params['range'], 'body', body_params, idx))

    # Sort all items by range (far to near)
    render_items.sort(key=lambda x: x[0], reverse=True)

    # Render in range order (far to near)
    time = float(params.get('time', 0.0))
    epoch = float(params.get('ring_epoch', 0.0))
    shade_solid = bool(params.get('shade_solid_rings', False))
    # Track ring masks in original order for click detection
    ring_mask_map: dict[int, NDArrayBoolType] = {}

    # Track body data for final metadata
    body_models_dict: dict[str, dict[str, Any]] = {}
    # Store body masks by original index, not render order
    body_mask_map_by_idx: dict[int, NDArrayBoolType] = {}
    body_mask_map_dict: dict[str, NDArrayBoolType] = {}
    inventory_dict: dict[str, dict[str, float]] = {}
    body_index_map: NDArrayIntType = np.zeros((size_v, size_u), dtype=np.int32)

    ref_center_v = size_v / 2.0
    ref_center_u = size_u / 2.0

    # Build order_near_to_far for bodies (needed for body_index_map)
    sorted_bodies_by_range = sorted(bodies_with_ranges, key=lambda x: x['range'])
    order_near_to_far = [
        bp.get('name', f'SIM-BODY-{i + 1}').upper() for i, bp in enumerate(sorted_bodies_by_range)
    ]

    for _range_val, item_type, item_params, orig_idx in render_items:
        if item_type == 'ring':
            ring_mask_map[orig_idx] = composite_ring(
                img,
                item_params,
                offset_v,
                offset_u=offset_u,
                time=time,
                epoch=epoch,
                shade_solid=shade_solid,
            )
        elif item_type == 'body':
            # Render single body
            body_mask, body_info = render_single_body(
                img,
                item_params,
                offset_v,
                offset_u=offset_u,
                seed=crater_seed,
                body_index=orig_idx,
                ref_center_v=ref_center_v,
                ref_center_u=ref_center_u,
            )
            # Store mask by original index for proper ordering
            body_mask_map_by_idx[orig_idx] = body_mask
            body_mask_map_dict[body_info['name']] = body_mask
            body_models_dict[body_info['name']] = body_info['params']
            inventory_dict[body_info['name']] = body_info['inventory']
            # Index into near-to-far order is 1-based
            near_index = order_near_to_far.index(body_info['name']) + 1
            body_index_map[body_mask] = near_index

    # Build body_masks in original order (matching bodies_params)
    body_masks: list[NDArrayBoolType] = []
    for idx in range(len(bodies_with_ranges)):
        if idx in body_mask_map_by_idx:
            body_masks.append(body_mask_map_by_idx[idx])
        else:
            # Should not happen, but create empty mask if missing
            body_masks.append(np.zeros((size_v, size_u), dtype=np.bool_))

    # Build ring_masks in original order for click detection
    ring_masks: list[NDArrayBoolType] = []
    for idx in range(len(rings_params)):
        if idx in ring_mask_map:
            ring_masks.append(ring_mask_map[idx])
        else:
            # Should not happen, but create empty mask if missing
            ring_masks.append(np.zeros((size_v, size_u), dtype=np.bool_))

    frame.truth.update(
        {
            'stars': sim_star_list,
            'bodies': body_models_dict,
            'rings': rings_params,
            'inventory': inventory_dict,
            'star_info': star_info,
            'body_masks': body_masks,
            'ring_masks': ring_masks,
            'order_near_to_far': order_near_to_far,
            'body_index_map': body_index_map,
            'body_mask_map': body_mask_map_dict,
        }
    )
