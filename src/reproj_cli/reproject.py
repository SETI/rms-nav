"""Thin wrappers around BodyMosaic.reproject() and RingMosaic.reproject().

These functions translate CLI argument namespaces into keyword arguments
for the underlying reproject() calls.
"""

import argparse
import math

from nav.reproj.bodies import BodyMosaic, BodyReprojResult
from nav.reproj.rings import RingMosaic, RingReprojResult
from reproj_cli.factories import parse_zoom_arg


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def reproject_one_body(
    obs: object,
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
    obs: object,
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
        longitude_range = (
            _deg_to_rad(float(args.longitude_range[0])),
            _deg_to_rad(float(args.longitude_range[1])),
        )

    radius_range = None
    if args.radius_range is not None:
        radius_range = (float(args.radius_range[0]), float(args.radius_range[1]))

    return mosaic.reproject(
        obs,
        longitude_range=longitude_range,
        radius_range=radius_range,
        margin=int(args.margin),
        zoom_amt=zoom,
        omit_shadow=args.omit_shadow,
        image_name=image_name,
    )
