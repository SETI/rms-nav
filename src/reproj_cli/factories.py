"""Factory functions that build BodyMosaic / RingMosaic from parsed CLI args."""

import argparse
import math

import numpy as np

from nav.reproj.bodies import BodyMosaic, BodyMosaicMergeStrategy
from nav.reproj.photometric_model import (
    LambertModel,
    LommelSeeligerModel,
    MinnaertModel,
    PhotometricModel,
)
from nav.reproj.ring_orbit_model import BRING_OUTER_EDGE, FRING_CORE
from nav.reproj.rings import RingMosaic, RingMosaicMergeStrategy


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def build_body_mosaic(args: argparse.Namespace) -> BodyMosaic:
    """Construct a BodyMosaic from parsed CLI arguments.

    Parameters:
        args: Namespace produced by a parser that includes ``add_body_args``.

    Returns:
        A freshly-initialised ``BodyMosaic``.
    """
    photometric_model_name: str = args.photometric_model
    phot_model: PhotometricModel | None
    if photometric_model_name == 'none':
        phot_model = None
    elif photometric_model_name == 'lambert':
        phot_model = LambertModel()
    elif photometric_model_name == 'lommel-seeliger':
        phot_model = LommelSeeligerModel()
    elif photometric_model_name == 'minnaert':
        phot_model = MinnaertModel()
    else:
        raise ValueError(f'Unknown photometric model: {photometric_model_name!r}')

    lat_range = None
    if args.lat_range is not None:
        lat_range = (
            _deg_to_rad(float(args.lat_range[0])),
            _deg_to_rad(float(args.lat_range[1])),
        )

    lon_range = None
    if args.lon_range is not None:
        lon_range = (
            _deg_to_rad(float(args.lon_range[0])),
            _deg_to_rad(float(args.lon_range[1])),
        )

    return BodyMosaic(
        body_name=args.body_name.upper(),
        lat_resolution=_deg_to_rad(float(args.lat_resolution)),
        lon_resolution=_deg_to_rad(float(args.lon_resolution)),
        lat_range=lat_range,
        lon_range=lon_range,
        dynamic=args.dynamic,
        max_incidence=_deg_to_rad(float(args.max_incidence)),
        max_emission=_deg_to_rad(float(args.max_emission)),
        max_resolution=float(args.max_resolution) if args.max_resolution is not None else None,
        edge_margin=int(args.edge_margin),
        zoom=int(args.zoom),
        latlon_type=args.latlon_type,
        lon_direction=args.lon_direction,
        photometric_model=phot_model,
        merge_strategy=BodyMosaicMergeStrategy.BEST_RESOLUTION,
        image_dtype=np.dtype(args.image_dtype),
        metadata_dtype=np.dtype(args.metadata_dtype),
    )


def build_ring_mosaic(args: argparse.Namespace) -> RingMosaic:
    """Construct a RingMosaic from parsed CLI arguments.

    Parameters:
        args: Namespace produced by a parser that includes ``add_ring_args``.

    Returns:
        A freshly-initialised ``RingMosaic``.
    """
    orbit_model_name: str = args.orbit_model
    if orbit_model_name == 'none':
        orbit_model = None
    elif orbit_model_name == 'fring_core':
        orbit_model = FRING_CORE
    elif orbit_model_name == 'bring_outer_edge':
        orbit_model = BRING_OUTER_EDGE
    else:
        raise ValueError(f'Unknown orbit model: {orbit_model_name!r}')

    merge_strategy_name: str = args.merge_strategy
    if merge_strategy_name == 'best_resolution':
        merge_strategy = RingMosaicMergeStrategy.BEST_RESOLUTION
    elif merge_strategy_name == 'most_coverage_then_resolution':
        merge_strategy = RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
    else:
        raise ValueError(f'Unknown merge strategy: {merge_strategy_name!r}')

    photometric_model_name: str = args.photometric_model
    phot_model: PhotometricModel | None
    if photometric_model_name == 'none':
        phot_model = None
    elif photometric_model_name == 'lambert':
        phot_model = LambertModel()
    elif photometric_model_name == 'lommel-seeliger':
        phot_model = LommelSeeligerModel()
    elif photometric_model_name == 'minnaert':
        phot_model = MinnaertModel()
    else:
        raise ValueError(f'Unknown photometric model: {photometric_model_name!r}')

    return RingMosaic(
        body_name=args.planet.upper(),
        radius_inner=float(args.radius_inner),
        radius_outer=float(args.radius_outer),
        longitude_resolution=_deg_to_rad(float(args.longitude_resolution)),
        radius_resolution=float(args.radius_resolution),
        merge_strategy=merge_strategy,
        orbit_model=orbit_model,
        image_dtype=np.dtype(args.image_dtype),
        metadata_dtype=np.dtype(args.metadata_dtype),
        photometric_model=phot_model,
    )


def parse_zoom_arg(zoom_str: str) -> int | tuple[int, int]:
    """Parse the ``--zoom`` argument, which may be ``"N"`` or ``"R,L"``.

    Parameters:
        zoom_str: String from the CLI ``--zoom`` argument.

    Returns:
        An integer, or a ``(radial, longitudinal)`` tuple of integers.

    Raises:
        ValueError: If the string cannot be parsed as a valid zoom specification.
    """
    zoom_str = zoom_str.strip()
    if ',' in zoom_str:
        parts = zoom_str.split(',')
        if len(parts) != 2:
            raise ValueError(f'--zoom: expected "N" or "R,L", got {zoom_str!r}')
        return (int(parts[0].strip()), int(parts[1].strip()))
    return int(zoom_str)
