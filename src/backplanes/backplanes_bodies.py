from typing import Any

import numpy as np
from oops.meshgrid import Meshgrid
from oops.backplane import Backplane

from nav.config import Config
from nav.obs import ObsSnapshot


def create_body_backplanes(snapshot: ObsSnapshot,
                           config: Config) -> dict[str, Any]:
    """Create configured body backplanes embedded in full-frame arrays.

    Parameters:
        snapshot: The observation snapshot.
        config: The configuration.

    Returns:
        A dictionary containing the body backplanes.
    """

    # per_body maps body_name -> arrays/masks/distance
    result: dict[str, Any] = {
        'per_body': {},
        'types': [],             # available backplane types for bodies
        'order': [],             # body names sorted by increasing distance
    }

    # Build inventory and candidate body list
    if snapshot.is_simulated:
        inv = snapshot.sim_inventory
        candidate_names = list(inv.keys())
    else:
        closest_planet = snapshot.closest_planet
        if closest_planet is None:
            return result
        body_list = [closest_planet] + config.satellites(closest_planet)
        inv = snapshot.inventory(body_list, return_type='full')
        candidate_names = list(inv.keys())

    # Filter and sort by distance
    bodies_by_range = [(name, inv[name]) for name in candidate_names
                       if name in inv and snapshot.inventory_body_in_fov(inv[name])]
    bodies_by_range.sort(key=lambda x: x[1]['range'])
    result['order'] = [x[0] for x in bodies_by_range]

    # Configured backplanes for bodies
    bodies_cfg = getattr(config.backplanes, 'bodies', None)
    if bodies_cfg is None:
        bodies_cfg = [
            {'name': 'longitude', 'method': 'longitude'},
            {'name': 'latitude', 'method': 'latitude'},
            {'name': 'incidence_angle', 'method': 'incidence_angle'},
            {'name': 'phase_angle', 'method': 'phase_angle'},
            {'name': 'emission_angle', 'method': 'emission_angle'},
            {'name': 'finest_resolution', 'method': 'finest_resolution'},
            {'name': 'coarsest_resolution', 'method': 'coarsest_resolution'},
        ]
    bp_types = [bp['name'] for bp in bodies_cfg]
    result['types'] = bp_types

    # For each body, create a restricted meshgrid and evaluate configured backplanes
    for body_name, inv_info in bodies_by_range:
        u_min = int(inv_info['u_min_unclipped'])
        u_max = int(inv_info['u_max_unclipped'])
        v_min = int(inv_info['v_min_unclipped'])
        v_max = int(inv_info['v_max_unclipped'])
        u0, u1, v0, v1 = snapshot.clip_rect_fov(u_min, u_max, v_min, v_max)
        if u1 < u0 or v1 < v0:
            # Nothing visible
            continue

        # Build restricted meshgrid covering the clipped rectangle (inclusive indices)
        meshgrid = Meshgrid.for_fov(
            snapshot.fov,
            origin=(u0 + 0.5, v0 + 0.5),
            limit=(u1 + 0.5, v1 + 0.5),
            swap=True
        )
        bp = Backplane(snapshot, meshgrid=meshgrid)

        per_type_arrays: dict[str, np.ndarray] = {}
        per_type_masks: dict[str, np.ndarray] = {}

        for bp_cfg in bodies_cfg:
            name = bp_cfg['name']
            method = bp_cfg.get('method', name)

            # Evaluate backplane for this body
            if snapshot.is_simulated:
                full = np.zeros(snapshot.data.shape, dtype=np.float32)
                full_mask = np.zeros(snapshot.data.shape, dtype=bool)
                seed = abs(hash((body_name, name))) % (2**32)
                rng = np.random.default_rng(seed)
                val = float(rng.uniform(1.0, 100.0))
                # Build mask from sim body index map if available
                sub_mask = None
                # Prefer body mask map if available
                mask_map = snapshot.sim_body_mask_map
                mask = mask_map.get(str(body_name).upper(), None)
                if mask is None:
                    # Try direct key
                    mask = mask_map.get(body_name, None)
                if mask is not None:
                    sub_mask = mask[v0:v1 + 1, u0:u1 + 1].astype(bool)
                if sub_mask is None:
                    try:
                        # Build robust name list for matching
                        order_names = [str(n).upper() for n in snapshot.sim_body_order_near_to_far]
                        body_idx = order_names.index(str(body_name).upper()) + 1
                        sim_map = snapshot.sim_body_index_map
                        sub_mask = (sim_map[v0:v1 + 1, u0:u1 + 1] == body_idx)
                    except Exception:
                        # Fallback: fill entire rect (last resort)
                        sub_mask = np.ones(
                            (v1 - v0 + 1, u1 - u0 + 1),
                            dtype=bool
                        )
                full_slice = full[v0:v1 + 1, u0:u1 + 1]
                # Only fill where this body contributed
                full_slice[sub_mask] = val
                full[v0:v1 + 1, u0:u1 + 1] = full_slice
                full_mask[v0:v1 + 1, u0:u1 + 1] = sub_mask
                per_type_arrays[name] = full
                per_type_masks[name] = full_mask
                continue
            # This will raise an exception if the backplane is not available
            func = getattr(bp, method)
            vals = func(body_name)
            # Convert to masked array via .mvals to preserve mask
            mvals = vals.mvals  # masked numpy array
            # Embed into full-frame arrays
            full = np.zeros(snapshot.data.shape, dtype=np.float32)
            full_mask = np.zeros(snapshot.data.shape, dtype=bool)
            full[v0:v1 + 1, u0:u1 + 1] = np.ma.filled(mvals, fill_value=0.0).astype(np.float32)
            mask = ~np.ma.getmaskarray(mvals)  # type: ignore[no-untyped-call]
            full_mask[v0:v1 + 1, u0:u1 + 1] = mask  # True where valid

            per_type_arrays[name] = full
            per_type_masks[name] = full_mask

        if per_type_arrays:
            # Calculate min/max statistics for each backplane type
            body_stats: dict[str, dict[str, float]] = {}
            for bp_type, bp_array in per_type_arrays.items():
                bp_mask = per_type_masks[bp_type]
                # Extract valid values using the mask
                valid_values = bp_array[bp_mask]
                if len(valid_values) > 0:
                    # Check if this backplane type is in radians and needs conversion
                    bp_cfg = next((b for b in bodies_cfg if b['name'] == bp_type), None)
                    units = bp_cfg.get('units', '') if bp_cfg else ''
                    # Convert radians to degrees for angle backplanes
                    if units.lower() == 'rad' or any(
                            k in bp_type.lower() for k in
                            ('longitude', 'latitude', 'incidence', 'emission', 'phase')):
                        valid_values = np.degrees(valid_values)
                    min_val = float(np.nanmin(valid_values))
                    max_val = float(np.nanmax(valid_values))
                    body_stats[bp_type] = {'min': min_val, 'max': max_val}

            result['per_body'][body_name] = {
                'arrays': per_type_arrays,
                'masks': per_type_masks,
                'distance': float(inv_info['range']),
                'statistics': body_stats,
            }

    return result
