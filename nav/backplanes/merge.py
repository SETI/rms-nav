from typing import Any

import cspyce
import numpy as np

from nav.obs import ObsSnapshot


def _union_backplane_types(bodies_result: dict, rings_result: dict) -> list[str]:
    types: set[str] = set()
    # Bodies
    for body_name, entry in bodies_result.get('per_body', {}).items():
        types.update(entry.get('arrays', {}).keys())
    # Rings
    types.update(rings_result.get('arrays', {}).keys())
    return sorted(list(types))


def merge_sources_into_master(
    snapshot: ObsSnapshot,
    *,
    bodies_result: dict,
    rings_result: dict,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Merge bodies and rings per-pixel based on nearest distance.

    Parameters:
        snapshot: The observation snapshot.
        bodies_result: The bodies result.
        rings_result: The rings result.

    Returns:
        A tuple containing the master by type, body id map, and merge info.
    """

    height, width = snapshot.data.shape
    master_by_type: dict[str, np.ndarray] = {}
    body_id_map = np.zeros((height, width), dtype=np.int16)

    # Build source list with per-pixel distance and NAIF IDs
    sources: list[dict[str, Any]] = []

    # Bodies: scalar distances per body, broadcast within mask; inf elsewhere
    for body_name, entry in bodies_result.get('per_body', {}).items():
        distance_scalar = float(entry.get('distance', np.inf))
        # Create per-pixel distance with inf default and scalar where within any mask
        any_mask = None
        for m in entry.get('masks', {}).values():
            any_mask = m if any_mask is None else (any_mask | m)
        if any_mask is None:
            continue
        distance = np.where(any_mask, distance_scalar, np.inf).astype(np.float32)
        try:
            naif_id = int(cspyce.bodn2c(body_name))
        except Exception as e:
            if snapshot.is_simulated:
                # Unknown body name; create a deterministic fake NAIF ID in int16 range
                naif_id = 10000 + (abs(hash(body_name)) % 20000)
            else:
                raise e  # This is a real problem
        sources.append({
            'name': body_name,
            'naif_id': naif_id,
            'distance': distance,
            'arrays': entry.get('arrays', {}),
            'masks': entry.get('masks', {}),
        })

    # Rings: per-pixel distance directly
    if rings_result.get('enabled', False):
        planet = rings_result.get('planet')
        naif_id = int(cspyce.bodn2c(str(planet))) if planet else 0
        sources.append({
            'name': rings_result.get('target_key'),
            'naif_id': naif_id,
            'distance': np.asarray(rings_result.get('distance'), dtype=np.float32),
            'arrays': rings_result.get('arrays', {}),
            'masks': rings_result.get('masks', {}),
        })

    if not sources:
        return master_by_type, body_id_map, {'sources': []}

    # Stack distances to choose nearest per pixel
    dist_stack = np.stack([s['distance'] for s in sources], axis=0)  # [S, H, W]
    nearest_idx = np.argmin(dist_stack, axis=0)  # [H, W]

    # Build master arrays per backplane type across all sources
    all_types = _union_backplane_types(bodies_result, rings_result)
    for bp_type in all_types:
        master = np.zeros((height, width), dtype=np.float32)
        # For each source, place values where it is the nearest and has valid mask for this type
        for idx, src in enumerate(sources):
            arrays: dict = src['arrays']
            masks: dict = src['masks']
            if bp_type not in arrays or bp_type not in masks:
                continue
            src_vals = arrays[bp_type]
            src_mask = masks[bp_type]
            take = (nearest_idx == idx) & src_mask
            master[take] = src_vals[take]
        master_by_type[bp_type] = master

    # Compose body_id_map similarly
    for idx, src in enumerate(sources):
        src_any_mask = None
        for m in src['masks'].values():
            src_any_mask = m if src_any_mask is None else (src_any_mask | m)
        if src_any_mask is None:
            continue
        take = (nearest_idx == idx) & src_any_mask
        body_id_map[take] = np.int16(src['naif_id'])

    merge_info = {
        'sources': [{'name': s['name'], 'naif_id': s['naif_id']} for s in sources],
    }

    return master_by_type, body_id_map, merge_info
