from typing import Any

import cspyce
import numpy as np
from pdslogger import PdsLogger

from nav.support.types import NDArrayFloatType, NDArrayIntType


def calculate_backplane_statistics(
    body_id_map: NDArrayIntType,
    backplane_arrays: dict[str, NDArrayFloatType],
    logger: PdsLogger,
) -> dict[str, Any]:
    """Calculate min/max statistics for backplanes per body and rings.

    Parameters:
        body_id_map: Array mapping pixel locations to SPICE body IDs (int32).
        backplane_arrays: Dictionary mapping backplane type names to float arrays.
        logger: Logger for diagnostic messages.

    Returns:
        Dictionary with structure:
        {
            "bodies": {
                body_name: {
                    backplane_type: {"min": ..., "max": ...}
                }
            },
            "rings": {
                backplane_type: {"min": ..., "max": ...}
            }
        }
        Body names are used (not SPICE IDs), and results are keyed by body name.
        Rings data is always generated even if there are no bodies.
    """

    result: dict[str, Any] = {
        'bodies': {},
        'rings': {},
    }

    # Find unique body IDs (excluding 0, which means no body)
    unique_ids = np.unique(body_id_map)
    unique_ids = unique_ids[unique_ids > 0]

    # Map body IDs to names and extract statistics per body
    body_id_to_name: dict[int, str] = {}
    for body_id in unique_ids:
        body_name = cspyce.bodc2n(body_id)
        body_id_to_name[body_id] = body_name

    # For each body, extract pixels and calculate min/max for each backplane
    for body_id, body_name in body_id_to_name.items():
        body_mask = (body_id_map == body_id)
        if not np.any(body_mask):
            logger.warning('Body mask is empty for body_id %d (%s) - this should not happen',
                           body_id, body_name)
            continue

        body_stats: dict[str, dict[str, float]] = {}
        for bp_type, bp_array in backplane_arrays.items():
            # Skip ring backplanes - they are processed separately
            if bp_type.upper().startswith('RING_'):
                continue

            if bp_array.shape != body_id_map.shape:
                raise ValueError(
                    f'Shape mismatch for backplane {bp_type}: '
                    f'expected {body_id_map.shape}, got {bp_array.shape}')

            # Extract values for this body
            body_values = bp_array[body_mask]
            # Filter out invalid values
            valid_values = body_values[np.isfinite(body_values)]
            if len(valid_values) == 0:
                logger.debug('No valid values for body %s, backplane %s', body_name, bp_type)
                continue

            min_val = float(np.nanmin(valid_values))
            max_val = float(np.nanmax(valid_values))
            body_stats[bp_type] = {'min': min_val, 'max': max_val}

        if body_stats:
            result['bodies'][body_name] = body_stats

    # For rings, calculate min/max across all pixels (rings don't use body_id_map)
    # Ring backplanes are identified by name prefix "RING_"
    # This is processed even if there are no bodies
    for bp_type, bp_array in backplane_arrays.items():
        if bp_type.upper().startswith('RING_'):
            if bp_array.shape != body_id_map.shape:
                raise ValueError(
                    f'Shape mismatch for ring backplane {bp_type}: '
                    f'expected {body_id_map.shape}, got {bp_array.shape}')

            valid_values = bp_array[np.isfinite(bp_array)]
            if len(valid_values) > 0:
                min_val = float(np.nanmin(valid_values))
                max_val = float(np.nanmax(valid_values))
                result['rings'][bp_type] = {'min': min_val, 'max': max_val}

    return result
