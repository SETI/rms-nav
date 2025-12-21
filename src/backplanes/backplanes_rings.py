from typing import Any

import numpy as np

from nav.config import Config
from nav.obs import ObsSnapshot


def create_ring_backplanes(snapshot: ObsSnapshot,
                           config: Config) -> dict[str, Any] | None:
    """Create configured ring backplanes over the full image, if applicable.

    Returns an empty dict if no rings are configured or closest planet is None.

    Parameters:
        snapshot: The observation snapshot.
        config: The configuration.

    Returns:
        A dictionary containing the ring backplanes, or None if there is no closest planet
        or ring backplanes are not configured.

        - planet: The planet name.
        - target_key: The target key.
        - types: The ring backplane types.
        - arrays: The ring backplane arrays.
        - masks: The ring backplane masks.
        - distance: The ring backplane distance.

    Raises:
        Exception: If the ring backplanes are not available.
    """

    result: dict[str, Any] = {
        'planet': None,
        'target_key': None,
        'types': [],
        'arrays': {},
        'masks': {},
        'distance': None,  # per-pixel distance array
    }

    if snapshot.is_simulated:
        return None

    closest_planet = snapshot.closest_planet
    if closest_planet is None:
        return None

    rings_cfg = getattr(config.backplanes, 'rings', [])
    if not rings_cfg:
        return None

    # Use planet name
    target_key = f'{closest_planet}_RING_SYSTEM'
    if closest_planet == 'SATURN':
        target_key = 'SATURN_MAIN_RINGS'

    bp = snapshot.bp

    result['planet'] = closest_planet
    result['target_key'] = target_key

    # Evaluate configured ring backplanes
    ring_types = [bp_cfg['name'] for bp_cfg in rings_cfg]
    result['types'] = ring_types

    for bp_cfg in rings_cfg:
        name = bp_cfg['name']
        method = bp_cfg.get('method', name)
        func = getattr(bp, method)
        vals = func(target_key)

        if method == 'distance':
            # Save as the per-pixel distance field for merge ordering
            result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

        mvals = vals.mvals
        full = np.asarray(np.ma.filled(mvals, fill_value=np.nan), dtype=np.float32)
        mask = ~np.ma.getmaskarray(mvals)  # type: ignore[no-untyped-call]

        if np.any(mask):
            result['arrays'][name] = full
            result['masks'][name] = mask

    # Ensure distance is present
    if result['distance'] is None:
        # If not explicitly configured, compute per-pixel distance for ordering
        vals = bp.distance(target_key, direction='dep')
        result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

    # Calculate min/max statistics for ring backplanes
    ring_stats: dict[str, dict[str, float]] = {}
    for name, arr in result['arrays'].items():
        mask = result['masks'][name]
        valid_values = arr[mask]
        if len(valid_values) > 0:
            # Check if this backplane type is in radians and needs conversion
            bp_cfg = next((r for r in rings_cfg if r['name'] == name), None)
            units = bp_cfg.get('units', '') if bp_cfg else ''
            # Convert radians to degrees for angle backplanes
            if units.lower() == 'rad' or any(
                    k in name.lower() for k in
                    ('longitude', 'latitude', 'incidence', 'emission', 'phase')):
                valid_values = np.degrees(valid_values)
            min_val = float(np.nanmin(valid_values))
            max_val = float(np.nanmax(valid_values))
            ring_stats[name] = {'min': min_val, 'max': max_val}

    result['statistics'] = ring_stats

    return result
