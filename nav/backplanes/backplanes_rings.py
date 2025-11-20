from typing import Any

import numpy as np

from nav.config import Config
from nav.obs import ObsSnapshot


RingsResult = dict[str, Any]


def create_ring_backplanes(snapshot: ObsSnapshot,
                           config: Config) -> RingsResult:
    """Create configured ring backplanes over the full image, if applicable.

    Returns an empty dict if no rings are configured or closest planet is None.

    Parameters:
        snapshot: The observation snapshot.
        config: The configuration.

    Returns:
        A dictionary containing the ring backplanes.

        - enabled: Whether rings are enabled.
        - planet: The planet name.
        - target_key: The target key.
        - types: The ring backplane types.
        - arrays: The ring backplane arrays.
        - masks: The ring backplane masks.
        - distance: The ring backplane distance.

    Raises:
        Exception: If the ring backplanes are not available.
    """

    result: RingsResult = {
        'enabled': False,
        'planet': None,
        'target_key': None,
        'types': [],
        'arrays': {},
        'masks': {},
        'distance': None,  # per-pixel distance array
    }

    if snapshot.is_simulated:
        return result

    closest_planet = snapshot.closest_planet
    if closest_planet is None:
        return result

    rings_cfg = getattr(config.backplanes, 'rings', [])
    if not rings_cfg:
        return result

    # Use planet name
    target_key = f'{closest_planet}:RING'
    bp = snapshot.bp

    result['enabled'] = True
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
        full = np.asarray(np.ma.filled(mvals, fill_value=0.0), dtype=np.float32)
        mask = ~np.ma.getmaskarray(mvals)  # type: ignore[no-untyped-call]

        result['arrays'][name] = full
        result['masks'][name] = mask

    # Ensure distance is present
    if result['distance'] is None:
        # If not explicitly configured, compute per-pixel distance for ordering
        vals = bp.distance(target_key, direction='dep')
        result['distance'] = np.asarray(vals.mvals.filled(np.inf), dtype=np.float32)

    return result
