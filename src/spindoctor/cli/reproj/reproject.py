"""Thin wrappers around BodyMosaic.reproject() and RingMosaic.reproject().

These functions translate CLI argument namespaces into keyword arguments
for the underlying reproject() calls.
"""

import argparse
import math
from collections.abc import Sequence
from typing import Any, cast

from spindoctor.cli.reproj.factories import parse_zoom_arg
from spindoctor.obs import ObsSnapshotInst
from spindoctor.reproj.bodies import BodyMosaic, BodyReprojResult
from spindoctor.reproj.rings import RingMosaic, RingReprojResult


def _as_len2_float_range(arg_name: str, value: object) -> tuple[float, float]:
    """Return ``value`` as a pair of floats, validating length and type."""
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f'{arg_name} must be a length-2 sequence of numbers, not str/bytes; '
            f'got {type(value).__name__}'
        )
    if not isinstance(value, Sequence):
        raise ValueError(f'{arg_name} must be a length-2 sequence')
    seq: Sequence[object] = value
    if len(seq) != 2:
        raise ValueError(f'{arg_name} must have length 2, got length {len(seq)}')
    try:
        return float(cast(Any, seq[0])), float(cast(Any, seq[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{arg_name} elements must be convertible to float') from exc


def _deg_to_rad(deg: float) -> float:
    """Convert degrees to radians.

    Parameters:
        deg: Angle in degrees.

    Returns:
        The same angle in radians.
    """
    return math.radians(deg)


def reproject_one_body(
    obs: ObsSnapshotInst,
    mosaic: BodyMosaic,
    *,
    image_name: str = '',
) -> BodyReprojResult:
    """Reproject a single observation using the given BodyMosaic.

    Parameters:
        obs: Observation snapshot (FOV already adjusted if needed).
        mosaic: The BodyMosaic to use for reprojection.
        image_name: Label stored on the result (e.g. source image stem).

    Returns:
        A ``BodyReprojResult`` for the observation.
    """
    return mosaic.reproject(obs, image_name=image_name)


def reproject_one_ring(
    obs: ObsSnapshotInst,
    args: argparse.Namespace,
    mosaic: RingMosaic,
    *,
    image_name: str = '',
) -> RingReprojResult:
    """Reproject a single observation's ring data.

    Parameters:
        obs: Observation snapshot (FOV already adjusted if needed).
        args: Parsed CLI namespace (must include ``add_ring_args`` fields).
        mosaic: The RingMosaic to use for reprojection.
        image_name: Label stored on the result (e.g. source image stem).

    Returns:
        A ``RingReprojResult`` for the observation.
    """
    zoom = parse_zoom_arg(args.zoom)

    longitude_range = None
    if args.longitude_range is not None:
        lo_deg, hi_deg = _as_len2_float_range('args.longitude_range', args.longitude_range)
        longitude_range = (_deg_to_rad(lo_deg), _deg_to_rad(hi_deg))

    radius_range = None
    if args.radius_range is not None:
        r0, r1 = _as_len2_float_range('args.radius_range', args.radius_range)
        radius_range = (r0, r1)

    return mosaic.reproject(
        obs,
        longitude_range=longitude_range,
        radius_range=radius_range,
        margin=int(args.margin),
        zoom_amt=zoom,
        omit_shadow=args.omit_shadow,
        image_name=image_name,
    )
