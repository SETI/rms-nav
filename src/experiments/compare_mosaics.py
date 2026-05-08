#!/usr/bin/env python3
"""compare_mosaics -- Compare two ring or body mosaic/reprojection files.

Usage::

    python compare_mosaics.py [--photometry MODE]
                              [--output-ratio-mosaic PATH]
                              [--output-statistics-file PATH]
                              FILE1 FILE2

Applies an optional photometric adjustment to both files (undo the stored
correction first, then apply the requested model), checks that the two files
are compatible (same body and same grid resolutions), aligns them on the
shared physical extent (lat/lon overlap for body files, longitude_antimask
intersection for ring files), prints per-image statistics side-by-side,
then prints statistics on FILE1 / FILE2. Also prints statistics on the
per-pixel absolute delta of the metadata angles (phase, emission, incidence,
and -- for body files -- sub-solar and sub-observer lat/lon). Longitude
deltas use the shortest angular distance.

Reprojection results (RingReprojResult, BodyReprojResult) and mosaic data
(RingMosaicData, BodyMosaicData) are treated uniformly: both ends of the
comparison are cropped to the shared region regardless of which kind they
are or whether the kinds match.

If --output-ratio-mosaic is given, saves the ratio as a new file of the same
type and format as FILE1 so it can be viewed with nav_mosaic_display.

If --output-statistics-file is given, dumps every computed statistic to the
given path. The format is selected from the file extension:
  .json        -> JSON
  .yaml / .yml -> YAML
  .csv         -> CSV (flattened table with columns: section,key,statistic,value)

Supported file types (auto-detected from file contents):
  RingReprojResult, RingMosaicData, BodyReprojResult, BodyMosaicData

Photometric modes:
  as_saved        -- use stored pixel values unchanged (default)
  uncorrected     -- alias for intrinsic; undo any stored correction
  intrinsic       -- undo any stored correction
  lambert         -- undo stored, apply Lambert law
  lommel_seeliger -- undo stored, apply Lommel-Seeliger law
  minnaert        -- undo stored, apply Minnaert law
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import numpy.ma as ma
from ruamel.yaml import YAML

from nav.reproj.bodies import BodyMosaicData, BodyReprojResult
from nav.reproj.rings import RingMosaicData, RingReprojResult
from nav.support.file import clean_obj
from nav.ui.mosaic_viewer.photometric_display import (
    compute_body_display_image,
    compute_ring_display_image,
)

_PHOTOMETRY_CHOICES = [
    'as_saved',
    'uncorrected',
    'intrinsic',
    'lambert',
    'lommel_seeliger',
    'minnaert',
]

_RAD2DEG = 180.0 / math.pi
_TWO_PI = 2.0 * math.pi

# Longitudes are circular; report shortest-arc distance rather than naive |a-b|.
_LONGITUDE_ANGLE_NAMES = frozenset({'sub_solar_lon', 'sub_observer_lon'})

# Order used when iterating over angle deltas for both display and serialization.
_BODY_ANGLE_ORDER = (
    'phase',
    'emission',
    'incidence',
    'sub_solar_lon',
    'sub_solar_lat',
    'sub_observer_lon',
    'sub_observer_lat',
)
_RING_ANGLE_ORDER = ('phase', 'emission', 'incidence')


def _load_file(path):
    """Load a mosaic or reprojection file; auto-detect type from contents.

    Returns:
        (obj, kind) where kind is one of: 'ring_reproj', 'ring_mosaic',
        'body_reproj', 'body_mosaic'.
    """
    for cls, kind in [
        (RingReprojResult, 'ring_reproj'),
        (RingMosaicData, 'ring_mosaic'),
        (BodyReprojResult, 'body_reproj'),
        (BodyMosaicData, 'body_mosaic'),
    ]:
        try:
            return cls.load(path), kind
        except (ValueError, KeyError):
            continue
    raise ValueError(f'Cannot determine file type for: {path!r}')


def _apply_photometry(obj, kind, mode):
    """Return image after applying the requested photometric mode.

    Converts stored radians to degrees, then delegates to the display
    functions which handle undo of any stored correction and application
    of the new model.
    """
    if mode == 'uncorrected':
        mode = 'intrinsic'

    if kind.startswith('ring'):
        # ma.asarray preserves the mask when the input is a MaskedArray
        # (RingMosaicData) and produces a no-mask array for plain ndarrays
        # (RingReprojResult).
        phase_deg = np.rad2deg(ma.asarray(obj.mean_phase, dtype=np.float64))
        emission_deg = np.rad2deg(ma.asarray(obj.mean_emission, dtype=np.float64))
        incidence_deg = np.rad2deg(
            float(obj.incidence if kind == 'ring_reproj' else obj.mean_incidence)
        )
        return compute_ring_display_image(
            mode=mode,
            image_ma=obj.img,
            photometric_model_name=obj.photometric_model_name,
            mean_phase_deg=phase_deg,
            mean_emission_deg=emission_deg,
            mean_incidence_deg=incidence_deg,
        )
    else:
        # .astype() on a MaskedArray preserves the mask; np.asarray would strip it.
        phase_deg = np.rad2deg(obj.phase.astype(np.float64))
        emission_deg = np.rad2deg(obj.emission.astype(np.float64))
        incidence_deg = np.rad2deg(obj.incidence.astype(np.float64))
        return compute_body_display_image(
            mode=mode,
            image_ma=obj.img,
            photometric_model_name=obj.photometric_model_name,
            phase_deg=phase_deg,
            emission_deg=emission_deg,
            incidence_deg=incidence_deg,
        )


def _compatibility_errors(obj1, kind1, path1, obj2, kind2, path2):
    """Return a list of compatibility problem strings (empty = fully compatible).

    Shape mismatches are NOT flagged here, even for ring files: body files
    share a global lat/lon grid anchored at -pi/2 / 0, and ring files share a
    full-longitude grid via ``longitude_antimask``. The caller is expected to
    crop both ends to their physical overlap with :func:`_align_pair`. Only
    differences that prevent any meaningful alignment (different category,
    body, or grid resolution) are reported here.
    """
    cat1 = 'ring' if kind1.startswith('ring') else 'body'
    cat2 = 'ring' if kind2.startswith('ring') else 'body'
    if cat1 != cat2:
        return [
            f'different categories: {os.path.basename(path1)!r} is {cat1}, '
            f'{os.path.basename(path2)!r} is {cat2}'
        ]

    errors = []

    if obj1.body_name != obj2.body_name:
        errors.append(f'body_name: {obj1.body_name!r} vs {obj2.body_name!r}')

    if cat1 == 'ring':
        if not math.isclose(obj1.longitude_resolution, obj2.longitude_resolution, rel_tol=1e-6):
            errors.append(
                f'longitude_resolution: {obj1.longitude_resolution} vs {obj2.longitude_resolution}'
            )
        if not math.isclose(obj1.radius_resolution, obj2.radius_resolution, rel_tol=1e-6):
            errors.append(
                f'radius_resolution: {obj1.radius_resolution} vs {obj2.radius_resolution}'
            )
        if not math.isclose(obj1.radius_inner, obj2.radius_inner, rel_tol=1e-6):
            errors.append(f'radius_inner: {obj1.radius_inner} vs {obj2.radius_inner}')
        if not math.isclose(obj1.radius_outer, obj2.radius_outer, rel_tol=1e-6):
            errors.append(f'radius_outer: {obj1.radius_outer} vs {obj2.radius_outer}')
        if obj1.longitude_antimask.shape != obj2.longitude_antimask.shape:
            errors.append(
                f'longitude_antimask shape: {obj1.longitude_antimask.shape} vs '
                f'{obj2.longitude_antimask.shape}'
            )
    else:
        if not math.isclose(obj1.lat_resolution, obj2.lat_resolution, rel_tol=1e-6):
            errors.append(f'lat_resolution: {obj1.lat_resolution} vs {obj2.lat_resolution}')
        if not math.isclose(obj1.lon_resolution, obj2.lon_resolution, rel_tol=1e-6):
            errors.append(f'lon_resolution: {obj1.lon_resolution} vs {obj2.lon_resolution}')
        if obj1.latlon_type != obj2.latlon_type:
            errors.append(f'latlon_type: {obj1.latlon_type!r} vs {obj2.latlon_type!r}')
        if obj1.lon_direction != obj2.lon_direction:
            errors.append(f'lon_direction: {obj1.lon_direction!r} vs {obj2.lon_direction!r}')

    return errors


# Origin of the body lat/lon grid: row 0 is at lat = -pi/2, col 0 at lon = 0.
# Both BodyReprojResult and BodyMosaicData use this same origin.
_BODY_LAT_ORIGIN = -math.pi / 2.0
_BODY_LON_ORIGIN = 0.0


def _body_global_indices(obj, kind):
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` global pixel indices.

    Indices are inclusive on both ends and refer to the (-pi/2, 0)-anchored
    global grid that both reproj and mosaic body files share. ``BodyMosaicData``
    stores its physical extent in ``lat_range`` / ``lon_range`` and the
    indices are derived from those; ``BodyReprojResult`` already records its
    integer ``lat_idx_range`` / ``lon_idx_range``.
    """
    n_rows, n_cols = obj.img.shape
    lat_res = obj.lat_resolution
    lon_res = obj.lon_resolution
    if kind == 'body_mosaic':
        lat_min = round((obj.lat_range[0] - _BODY_LAT_ORIGIN) / lat_res)
        lon_min = round((obj.lon_range[0] - _BODY_LON_ORIGIN) / lon_res)
    else:
        lat_min = int(obj.lat_idx_range[0])
        lon_min = int(obj.lon_idx_range[0])
    return lat_min, lat_min + n_rows - 1, lon_min, lon_min + n_cols - 1


def _crop_body_file(obj, kind, target_lat_min, target_lat_max,
                    target_lon_min, target_lon_max):
    """Return ``obj`` cropped to a global lat/lon index window.

    The crop window is given as inclusive global pixel indices on the shared
    body grid. The 2D fields (``img``, ``resolution``, ``eff_resolution``,
    ``phase``, ``emission``, ``incidence``) are sliced in place. For
    ``body_mosaic`` we also slice ``time``, ``image_number`` and update
    ``lat_range`` / ``lon_range``; for ``body_reproj`` we update
    ``lat_idx_range`` / ``lon_idx_range``. Per-image arrays (sub_solar_*,
    sub_observer_*, contributing_image_names, etc.) are carried through
    unchanged because ``image_number`` continues to index them correctly.
    """
    lat_min_src, _, lon_min_src, _ = _body_global_indices(obj, kind)
    n_rows, n_cols = obj.img.shape
    r0 = max(0, target_lat_min - lat_min_src)
    r1 = min(n_rows, target_lat_max - lat_min_src + 1)
    c0 = max(0, target_lon_min - lon_min_src)
    c1 = min(n_cols, target_lon_max - lon_min_src + 1)
    sl = (slice(r0, r1), slice(c0, c1))

    common = {
        'img': obj.img[sl],
        'resolution': obj.resolution[sl],
        'eff_resolution': obj.eff_resolution[sl],
        'phase': obj.phase[sl],
        'emission': obj.emission[sl],
        'incidence': obj.incidence[sl],
    }
    if kind == 'body_mosaic':
        new_lat_range = (
            _BODY_LAT_ORIGIN + (lat_min_src + r0) * obj.lat_resolution,
            _BODY_LAT_ORIGIN + (lat_min_src + r1 - 1) * obj.lat_resolution,
        )
        new_lon_range = (
            _BODY_LON_ORIGIN + (lon_min_src + c0) * obj.lon_resolution,
            _BODY_LON_ORIGIN + (lon_min_src + c1 - 1) * obj.lon_resolution,
        )
        return dataclasses.replace(
            obj,
            **common,
            time=obj.time[sl],
            image_number=obj.image_number[sl],
            lat_range=new_lat_range,
            lon_range=new_lon_range,
        )
    return dataclasses.replace(
        obj,
        **common,
        lat_idx_range=(lat_min_src + r0, lat_min_src + r1 - 1),
        lon_idx_range=(lon_min_src + c0, lon_min_src + c1 - 1),
    )


def _align_body_pair(obj1, kind1, obj2, kind2):
    """Crop two body files to their shared lat/lon window on the global grid.

    Works for any combination of ``body_reproj`` / ``body_mosaic``. Returns
    the two files with 2D arrays sliced to the same shape; the kind of each
    is preserved. No-op when the global-index windows already coincide.

    Raises:
        ValueError: If the two files do not overlap in lat/lon.
    """
    a_lat0, a_lat1, a_lon0, a_lon1 = _body_global_indices(obj1, kind1)
    b_lat0, b_lat1, b_lon0, b_lon1 = _body_global_indices(obj2, kind2)
    lat_min = max(a_lat0, b_lat0)
    lat_max = min(a_lat1, b_lat1)
    lon_min = max(a_lon0, b_lon0)
    lon_max = min(a_lon1, b_lon1)
    if lat_max < lat_min or lon_max < lon_min:
        raise ValueError(
            'Body files do not overlap on the shared lat/lon grid: '
            f'file1 lat_idx=[{a_lat0},{a_lat1}] lon_idx=[{a_lon0},{a_lon1}], '
            f'file2 lat_idx=[{b_lat0},{b_lat1}] lon_idx=[{b_lon0},{b_lon1}]'
        )

    if (a_lat0, a_lat1, a_lon0, a_lon1) == (lat_min, lat_max, lon_min, lon_max) and \
       (b_lat0, b_lat1, b_lon0, b_lon1) == (lat_min, lat_max, lon_min, lon_max):
        return obj1, obj2

    return (
        _crop_body_file(obj1, kind1, lat_min, lat_max, lon_min, lon_max),
        _crop_body_file(obj2, kind2, lat_min, lat_max, lon_min, lon_max),
    )


def _crop_ring_file(obj, kind, intersect_antimask):
    """Return ``obj`` restricted to the longitude bins where ``intersect_antimask`` is True.

    ``intersect_antimask`` has the same length as ``obj.longitude_antimask``
    and selects the global longitude bins that survive. The sparse columns of
    the 2D arrays are sliced to keep only those whose full-grid bin is
    selected; the 1D per-longitude arrays (``mean_phase``, ``mean_emission``,
    ``mean_radial_resolution``, ``mean_angular_resolution``) are sliced the
    same way, and ``longitude_antimask`` is replaced by the intersection.

    For ``ring_mosaic`` the 1D ``time`` and ``image_number`` arrays are also
    sliced. ``ring_reproj``'s scalar ``time`` / ``incidence`` and the shared
    ``orbit_model`` / ``orbit_model_name`` carry through unchanged.
    """
    bin_indices = np.where(np.asarray(obj.longitude_antimask, dtype=np.bool_))[0]
    keep_cols = np.asarray(intersect_antimask, dtype=np.bool_)[bin_indices]
    common = {
        'longitude_antimask': np.asarray(intersect_antimask, dtype=np.bool_).copy(),
        'img': obj.img[:, keep_cols],
        'mean_radial_resolution': obj.mean_radial_resolution[keep_cols],
        'mean_angular_resolution': obj.mean_angular_resolution[keep_cols],
        'mean_phase': obj.mean_phase[keep_cols],
        'mean_emission': obj.mean_emission[keep_cols],
    }
    if kind == 'ring_mosaic':
        return dataclasses.replace(
            obj,
            **common,
            image_number=obj.image_number[keep_cols],
            time=obj.time[keep_cols],
            longitude_range=None,
        )
    return dataclasses.replace(obj, **common)


def _align_ring_pair(obj1, kind1, obj2, kind2):
    """Crop two ring files to the longitudes that exist in both.

    Works for any combination of ``ring_reproj`` / ``ring_mosaic`` whose
    ``longitude_antimask`` arrays are the same length (i.e. compatible
    longitude resolution). The radius axis is already implicitly aligned by
    the equal ``radius_inner`` / ``radius_outer`` / ``radius_resolution``
    enforced in :func:`_compatibility_errors`. No-op when the antimasks
    already coincide.

    Raises:
        ValueError: If the two files have no longitudes in common.
    """
    mask1 = np.asarray(obj1.longitude_antimask, dtype=np.bool_)
    mask2 = np.asarray(obj2.longitude_antimask, dtype=np.bool_)
    intersect = mask1 & mask2
    if not intersect.any():
        raise ValueError(
            'Ring files have no longitudes in common: '
            f'file1 has {int(mask1.sum())} bins, '
            f'file2 has {int(mask2.sum())} bins, intersection is empty.'
        )
    if np.array_equal(mask1, intersect) and np.array_equal(mask2, intersect):
        return obj1, obj2
    return (
        _crop_ring_file(obj1, kind1, intersect),
        _crop_ring_file(obj2, kind2, intersect),
    )


def _align_pair(obj1, kind1, obj2, kind2):
    """Dispatch to the body- or ring-specific alignment based on the kinds.

    Returns the two files cropped to their shared physical extent. Both
    files come back with ``img`` arrays of the same shape regardless of
    which kind they are. Raises ``ValueError`` if the files do not overlap.
    """
    cat = 'ring' if kind1.startswith('ring') else 'body'
    if cat == 'ring':
        return _align_ring_pair(obj1, kind1, obj2, kind2)
    return _align_body_pair(obj1, kind1, obj2, kind2)


def _compute_stats(arr, extra_mask=None):
    """Return a dict of statistics for unmasked pixels, optionally with an extra exclusion mask."""
    combined = ma.masked_array(
        arr,
        mask=ma.getmaskarray(arr) | (extra_mask if extra_mask is not None else False),
    )
    valid = np.asarray(combined.compressed(), dtype=np.float64)
    n = len(valid)
    if n == 0:
        nan = float('nan')
        return {k: nan for k in ('n', 'min', 'p10', 'p25', 'p50', 'mean', 'std', 'p75', 'p90', 'max')}
    return {
        'n': float(n),
        'min': float(np.min(valid)),
        'p10': float(np.percentile(valid, 10)),
        'p25': float(np.percentile(valid, 25)),
        'p50': float(np.percentile(valid, 50)),
        'mean': float(np.mean(valid)),
        'std': float(np.std(valid)),
        'p75': float(np.percentile(valid, 75)),
        'p90': float(np.percentile(valid, 90)),
        'max': float(np.max(valid)),
    }


def _fmt(value, fmt):
    return fmt.format(value) if math.isfinite(value) else 'N/A'


def _ratio_of_stats(stats1, stats2):
    """Return ``{stat_name: stats1[name] / stats2[name]}`` for every statistic.

    For each statistic key (``min``, ``p10``, ..., ``max``) the value is
    ``stats1[key] / stats2[key]`` as a float, or ``nan`` if either side is
    non-finite or ``stats2[key]`` is zero. The ``n`` key reports the per-file
    sample count ratio (almost always 1.0 once the inputs share an overlap
    region).
    """
    out = {}
    for key in ('n', 'min', 'p10', 'p25', 'p50', 'mean', 'std', 'p75', 'p90', 'max'):
        v1, v2 = stats1[key], stats2[key]
        if not (math.isfinite(v1) and math.isfinite(v2)) or v2 == 0.0:
            out[key] = float('nan')
        else:
            out[key] = v1 / v2
    return out


def _print_stats_table(label1, label2, stats1, stats2,
                       ratio_of_stats_dict, ratio_of_pixels):
    """Print a four-column statistics table for the I/F image comparison.

    Columns: file 1 stats, file 2 stats, ratio-of-stats (``stats1[k]/stats2[k]``
    per row), and ratio-of-pixels (statistics of the per-pixel ratio image
    ``img1 / img2`` over pixels valid in both files).

    Either ratio dict may be ``None``; the corresponding column is then
    rendered as ``N/A`` for every row. The same two ratio dicts are written
    to the statistics file (``--output-statistics-file``) under
    ``ratio_of_stats`` and ``ratio_of_pixels`` so the console and the file
    agree exactly.
    """
    rows = [
        ('n',   'N valid',  '{:.0f}'),
        ('min', 'Min',      '{:.6f}'),
        ('p10', 'P10',      '{:.6f}'),
        ('p25', 'P25',      '{:.6f}'),
        ('p50', 'Median',   '{:.6f}'),
        ('mean','Mean',     '{:.6f}'),
        ('std', 'Std dev',  '{:.6f}'),
        ('p75', 'P75',      '{:.6f}'),
        ('p90', 'P90',      '{:.6f}'),
        ('max', 'Max',      '{:.6f}'),
    ]

    lw = 10   # label column width
    cw = 18   # data column width

    sep = '-' * (lw + 2 + cw + 2 + cw + 2 + cw + 2 + cw)
    print()
    print(
        f'{"Statistic":<{lw}}  {label1[:cw]:>{cw}}  {label2[:cw]:>{cw}}  '
        f'{"Ratio of stats":>{cw}}  {"Ratio of pixels":>{cw}}'
    )
    print(sep)
    for key, name, fmt in rows:
        v1, v2 = stats1[key], stats2[key]
        s1 = _fmt(v1, fmt)
        s2 = _fmt(v2, fmt)
        sr1 = (
            'N/A' if ratio_of_stats_dict is None else _fmt(ratio_of_stats_dict[key], fmt)
        )
        sr2 = (
            'N/A' if ratio_of_pixels is None else _fmt(ratio_of_pixels[key], fmt)
        )
        print(f'{name:<{lw}}  {s1:>{cw}}  {s2:>{cw}}  {sr1:>{cw}}  {sr2:>{cw}}')
    print()


def _is_numeric_scalar(value):
    """Check whether ``value`` is a Python or NumPy real scalar (no arrays)."""
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _expand_per_image_to_pixels(per_image, image_number):
    """Expand a per-image 1D array to a per-pixel 2D masked array via ``image_number``.

    ``image_number`` is a MaskedArray of indices into ``per_image``. Returns a
    MaskedArray with the same shape as ``image_number`` whose mask matches
    ``image_number.mask``. Out-of-range indices (which can happen if a mosaic
    was loaded with stale per-image arrays) are masked off rather than
    raised.
    """
    mask = ma.getmaskarray(image_number)
    if per_image.size == 0:
        return ma.masked_array(
            np.zeros(image_number.shape, dtype=np.float64),
            mask=np.ones(image_number.shape, dtype=bool),
        )
    indices = np.where(mask, 0, np.asarray(image_number, dtype=np.int64))
    out_of_range = (indices < 0) | (indices >= per_image.size)
    indices_clipped = np.clip(indices, 0, per_image.size - 1)
    values = np.asarray(per_image, dtype=np.float64)[indices_clipped]
    return ma.masked_array(values, mask=mask | out_of_range)


def _angle_arrays_radians(obj, kind):
    """Return ``{name: scalar_or_array_in_radians}`` for the metadata angles.

    Body files contribute phase, emission, incidence, sub_solar_lon,
    sub_solar_lat, sub_observer_lon, sub_observer_lat. For ``body_reproj``
    the four sub-solar/observer values are stored as scalars; for
    ``body_mosaic`` they are reconstructed per-pixel by indexing the
    per-image arrays with the pixel ``image_number``.

    Ring files contribute phase, emission, incidence. For both
    ``ring_reproj`` and ``ring_mosaic`` the incidence is a single scalar
    (the latter named ``mean_incidence`` in the source dataclass).
    """
    out = {}
    if kind in ('body_reproj', 'body_mosaic'):
        out['phase'] = obj.phase
        out['emission'] = obj.emission
        out['incidence'] = obj.incidence
        if kind == 'body_reproj':
            out['sub_solar_lon'] = float(obj.sub_solar_lon)
            out['sub_solar_lat'] = float(obj.sub_solar_lat)
            out['sub_observer_lon'] = float(obj.sub_observer_lon)
            out['sub_observer_lat'] = float(obj.sub_observer_lat)
        else:
            out['sub_solar_lon'] = _expand_per_image_to_pixels(
                obj.sub_solar_lon_per_image, obj.image_number
            )
            out['sub_solar_lat'] = _expand_per_image_to_pixels(
                obj.sub_solar_lat_per_image, obj.image_number
            )
            out['sub_observer_lon'] = _expand_per_image_to_pixels(
                obj.sub_observer_lon_per_image, obj.image_number
            )
            out['sub_observer_lat'] = _expand_per_image_to_pixels(
                obj.sub_observer_lat_per_image, obj.image_number
            )
    else:
        out['phase'] = obj.mean_phase
        out['emission'] = obj.mean_emission
        out['incidence'] = (
            float(obj.incidence) if kind == 'ring_reproj' else float(obj.mean_incidence)
        )
    return out


def _broadcast_extra_mask(extra_mask, target_shape):
    """Reduce a 2-D image-overlap mask to ``target_shape``, or return None.

    Body angle arrays already have the same 2-D shape as the overlap mask,
    so the mask is returned unchanged. Ring angle arrays are 1-D per
    longitude column with length ``overlap_mask.shape[1]``; in that case a
    column is treated as masked (i.e. excluded from the delta stats) when
    every row at that column is masked-in-either, mirroring how the image
    overlap is defined. Any other shape combination returns ``None``, which
    means the per-array mask is the only filter.
    """
    if extra_mask is None:
        return None
    if extra_mask.shape == target_shape:
        return extra_mask
    if (
        extra_mask.ndim == 2
        and len(target_shape) == 1
        and extra_mask.shape[1] == target_shape[0]
    ):
        return extra_mask.all(axis=0)
    return None


def _absolute_angle_delta_deg(a1, a2, *, wraparound):
    """Return |a1 - a2| in degrees. Inputs are radians.

    If ``wraparound`` is true, use the shortest angular distance modulo 2*pi
    rather than naive |a1 - a2|. Inputs may be Python scalars, NumPy
    scalars, ndarrays, or MaskedArrays. The return type matches the most
    structured input (MaskedArray > ndarray > scalar).
    """
    a1m = ma.asarray(a1, dtype=np.float64)
    a2m = ma.asarray(a2, dtype=np.float64)
    diff = ma.abs(a1m - a2m)
    if wraparound:
        diff = ma.minimum(diff, _TWO_PI - diff)
    return diff * _RAD2DEG


def _angle_delta_records(angles1, angles2, *, overlap_mask):
    """Compute one delta record per metadata angle, in display order.

    Returns a list of dicts, each with at minimum::

        {'name': str, 'kind': 'scalar' | 'pixel', 'wraparound': bool,
         'units': 'deg'}

    Scalar records add ``value_file1_deg``, ``value_file2_deg``, and
    ``delta_deg``. Pixel records add ``stats`` (dict of n/min/p10/.../max
    over the overlap mask), plus ``value_file1_deg`` / ``value_file2_deg``
    when one side is a scalar (broadcast against the array side).
    """
    records = []
    for name in angles1:
        if name not in angles2:
            continue
        a1, a2 = angles1[name], angles2[name]
        wrap = name in _LONGITUDE_ANGLE_NAMES
        a1_scalar = _is_numeric_scalar(a1)
        a2_scalar = _is_numeric_scalar(a2)

        if a1_scalar and a2_scalar:
            delta_val = _absolute_angle_delta_deg(a1, a2, wraparound=wrap)
            records.append(
                {
                    'name': name,
                    'kind': 'scalar',
                    'wraparound': wrap,
                    'units': 'deg',
                    'value_file1_deg': float(a1) * _RAD2DEG,
                    'value_file2_deg': float(a2) * _RAD2DEG,
                    'delta_deg': float(np.asarray(delta_val)),
                }
            )
            continue

        delta_arr = ma.asarray(
            _absolute_angle_delta_deg(a1, a2, wraparound=wrap), dtype=np.float64
        )
        # The image-level overlap mask is 2-D; ring angle arrays are 1-D
        # per longitude column, so reduce the 2-D mask to a column mask
        # (a column counts as overlapping iff at least one row is valid in
        # both files). The delta array's own mask is preserved either way.
        extra = _broadcast_extra_mask(overlap_mask, delta_arr.shape)
        stats = _compute_stats(delta_arr, extra_mask=extra)
        rec = {
            'name': name,
            'kind': 'pixel',
            'wraparound': wrap,
            'units': 'deg',
            'stats': stats,
        }
        if a1_scalar:
            rec['value_file1_deg'] = float(a1) * _RAD2DEG
        if a2_scalar:
            rec['value_file2_deg'] = float(a2) * _RAD2DEG
        records.append(rec)
    return records


def _print_angle_delta_table(records):
    """Print a multi-column table of per-pixel delta statistics in degrees.

    Pixel records become columns in the main table; scalar records are
    printed below as one-liners showing both file values and the delta.
    """
    pixel_records = [r for r in records if r['kind'] == 'pixel']
    scalar_records = [r for r in records if r['kind'] == 'scalar']

    if pixel_records:
        rows = [
            ('n',    'N valid',  '{:.0f}'),
            ('min',  'Min',      '{:.4f}'),
            ('p10',  'P10',      '{:.4f}'),
            ('p25',  'P25',      '{:.4f}'),
            ('p50',  'Median',   '{:.4f}'),
            ('mean', 'Mean',     '{:.4f}'),
            ('std',  'Std dev',  '{:.4f}'),
            ('p75',  'P75',      '{:.4f}'),
            ('p90',  'P90',      '{:.4f}'),
            ('max',  'Max',      '{:.4f}'),
        ]
        lw = 10
        cw = 14
        print('Per-pixel |angle1 - angle2| (deg):')
        header = f'{"Statistic":<{lw}}  ' + '  '.join(
            f'{r["name"][:cw]:>{cw}}' for r in pixel_records
        )
        print(header)
        print('-' * len(header))
        for key, name, fmt in rows:
            cells = '  '.join(
                f'{_fmt(r["stats"][key], fmt):>{cw}}' for r in pixel_records
            )
            print(f'{name:<{lw}}  {cells}')
        print()

    if scalar_records:
        print('Scalar angle deltas (deg):')
        for rec in scalar_records:
            tag = ' (shortest arc)' if rec['wraparound'] else ''
            print(
                f'  {rec["name"]:<20s}'
                f' file1={rec["value_file1_deg"]:>10.4f}'
                f' file2={rec["value_file2_deg"]:>10.4f}'
                f' |delta|={rec["delta_deg"]:>10.4f}{tag}'
            )
        print()


def _save_ratio(ratio_img, obj1, kind1, output_path):
    """Save ratio image in the same type and format as obj1."""
    ratio_f = ma.masked_array(
        np.asarray(ratio_img, dtype=np.float32),
        mask=ma.getmaskarray(ratio_img),
    )
    new_dtype = np.dtype(np.float32)
    out = dataclasses.replace(
        obj1, img=ratio_f, photometric_model_name=None, image_dtype=new_dtype
    )
    out.save(output_path)
    print(f'Ratio saved to: {output_path}')


def _statistics_format(path):
    """Return one of 'json', 'yaml', 'csv' from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        return 'json'
    if ext in ('.yaml', '.yml'):
        return 'yaml'
    if ext == '.csv':
        return 'csv'
    raise ValueError(
        f'Cannot infer statistics format from extension {ext!r}; '
        'use .json, .yaml, .yml, or .csv'
    )


def _flatten_for_csv(payload):
    """Flatten the statistics payload into ``(section, key, statistic, value)`` rows.

    Sections come straight from the top-level keys of ``payload``. For dicts
    of stats (``image``, ``ratio``, etc.) the statistic name fills the third
    column. For metadata and overlap, ``statistic`` is empty. For angle
    deltas, the ``angle_deltas`` section holds one row per (angle, stat).
    """
    rows = []

    metadata = payload.get('metadata', {})
    for key, value in metadata.items():
        rows.append(('metadata', key, '', value))

    overlap = payload.get('overlap', {})
    for key, value in overlap.items():
        rows.append(('overlap', key, '', value))

    for section in ('image_file1', 'image_file2', 'ratio_of_stats', 'ratio_of_pixels'):
        block = payload.get(section)
        if not block:
            continue
        for stat_name, value in block.items():
            rows.append((section, '', stat_name, value))

    for rec in payload.get('angle_deltas', []):
        name = rec['name']
        rows.append(('angle_deltas', name, 'kind', rec['kind']))
        rows.append(('angle_deltas', name, 'units', rec['units']))
        rows.append(('angle_deltas', name, 'wraparound', rec['wraparound']))
        if rec['kind'] == 'scalar':
            rows.append(('angle_deltas', name, 'value_file1_deg', rec['value_file1_deg']))
            rows.append(('angle_deltas', name, 'value_file2_deg', rec['value_file2_deg']))
            rows.append(('angle_deltas', name, 'delta_deg', rec['delta_deg']))
        else:
            for stat_name, value in rec['stats'].items():
                rows.append(('angle_deltas', name, stat_name, value))
            if 'value_file1_deg' in rec:
                rows.append(('angle_deltas', name, 'value_file1_deg', rec['value_file1_deg']))
            if 'value_file2_deg' in rec:
                rows.append(('angle_deltas', name, 'value_file2_deg', rec['value_file2_deg']))
    return rows


def _save_statistics(payload, output_path):
    """Write the statistics payload to ``output_path`` in JSON, YAML, or CSV.

    The format is inferred from the extension. NumPy scalar/array types are
    converted to native Python types via ``nav.support.file.clean_obj``
    before serialization.
    """
    fmt = _statistics_format(output_path)
    cleaned = clean_obj(payload)
    if fmt == 'json':
        with open(output_path, 'w') as f:
            json.dump(cleaned, f, indent=2)
    elif fmt == 'yaml':
        yaml = YAML()
        yaml.default_flow_style = False
        with open(output_path, 'w') as f:
            yaml.dump(cleaned, f)
    else:  # csv
        rows = _flatten_for_csv(cleaned)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('section', 'key', 'statistic', 'value'))
            writer.writerows(rows)
    print(f'Statistics saved to: {output_path}')


def _build_statistics_payload(*, args, kind1, kind2, obj1, obj2, same_shape,
                              n_overlap, n_only1, n_only2,
                              stats_label, stats1, stats2,
                              ratio_of_stats_dict, ratio_of_pixels,
                              angle_records):
    """Assemble the statistics payload that gets serialized.

    Holds two distinct ratio summaries: ``ratio_of_stats`` is the per-statistic
    quotient ``stats1[k] / stats2[k]`` (handy for spotting a constant
    brightness offset between the two files), and ``ratio_of_pixels`` is the
    full statistics block of the per-pixel ratio image ``img1 / img2`` over
    pixels valid in both files (the more meaningful pixel-level quantity).
    Both blocks are also rendered as columns in the console table so the
    file and the printout match. All numeric values are kept as
    Python/NumPy scalars; ``clean_obj`` handles the final type conversion
    right before serialization.
    """
    metadata = {
        'file1': args.file1,
        'file2': args.file2,
        'photometry': args.photometry,
        'kind1': kind1,
        'kind2': kind2,
        'body_name': obj1.body_name,
        'shape_file1': list(obj1.img.shape),
        'shape_file2': list(obj2.img.shape),
        'analyze_all_pixels': bool(args.analyze_all_pixels),
        'stats_region': stats_label,
    }
    payload = {
        'metadata': metadata,
        'overlap': {
            'same_shape': bool(same_shape),
            'n_overlap': int(n_overlap) if n_overlap is not None else None,
            'n_only_file1': int(n_only1) if n_only1 is not None else None,
            'n_only_file2': int(n_only2) if n_only2 is not None else None,
        },
        'image_file1': stats1,
        'image_file2': stats2,
        'angle_deltas': angle_records,
    }
    if ratio_of_stats_dict is not None:
        payload['ratio_of_stats'] = ratio_of_stats_dict
    if ratio_of_pixels is not None:
        payload['ratio_of_pixels'] = ratio_of_pixels
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(__doc__.split('\n\n')[1:]),
    )
    parser.add_argument('file1', metavar='FILE1', help='First mosaic or reprojection file.')
    parser.add_argument('file2', metavar='FILE2', help='Second mosaic or reprojection file.')
    parser.add_argument(
        '--photometry',
        default='as_saved',
        choices=_PHOTOMETRY_CHOICES,
        metavar='MODE',
        help=(
            'Photometric mode: as_saved (default), uncorrected/intrinsic, '
            'lambert, lommel_seeliger, minnaert. '
            'Undoes the stored correction before applying the new model.'
        ),
    )
    parser.add_argument(
        '--output-ratio-mosaic',
        default=None,
        metavar='PATH',
        help='If given, save FILE1/FILE2 ratio as a mosaic to this path.',
    )
    parser.add_argument(
        '--output-statistics-file',
        default=None,
        metavar='PATH',
        help=(
            'If given, dump every computed statistic to this path. The '
            'format is inferred from the extension: .json, .yaml/.yml, or '
            '.csv.'
        ),
    )
    parser.add_argument(
        '--analyze-all-pixels',
        action='store_true',
        help=(
            'Compute per-file statistics over all valid pixels in each image '
            'independently, rather than restricting to pixels valid in both. '
            'The ratio is still computed only where both images have valid data.'
        ),
    )
    args = parser.parse_args()

    try:
        # Validate the statistics output extension before doing real work so
        # we fail fast on a typo'd path.
        if args.output_statistics_file:
            _statistics_format(args.output_statistics_file)

        print(f'Loading {args.file1} ...')
        obj1, kind1 = _load_file(args.file1)
        print(f'  Type: {kind1}, shape: {obj1.img.shape}, body: {obj1.body_name}')

        print(f'Loading {args.file2} ...')
        obj2, kind2 = _load_file(args.file2)
        print(f'  Type: {kind2}, shape: {obj2.img.shape}, body: {obj2.body_name}')

        compat_errors = _compatibility_errors(obj1, kind1, args.file1, obj2, kind2, args.file2)
        if compat_errors:
            msg = 'Files are not fully compatible:\n' + '\n'.join(f'  {e}' for e in compat_errors)
            if args.analyze_all_pixels:
                print(f'\nWARNING: {msg}\n', file=sys.stderr)
            else:
                raise ValueError(msg)
        else:
            print('Compatibility: OK')

        # Reproj results and mosaic data hold the same kind of geometry over
        # the same shared grid (a global lat/lon grid for body files, a
        # full-longitude grid via ``longitude_antimask`` for ring files), so
        # we crop both ends to their physical overlap regardless of whether
        # they are reproj or mosaic. After alignment both ``img`` arrays have
        # the same shape; everything downstream goes through the
        # overlap-aware code path.
        try:
            obj1, obj2 = _align_pair(obj1, kind1, obj2, kind2)
            aligned = True
        except ValueError as align_err:
            if args.analyze_all_pixels:
                print(f'\nWARNING: {align_err}\n', file=sys.stderr)
                aligned = False
            else:
                raise
        if aligned and obj1.img.shape != obj2.img.shape:
            # Defensive: alignment should always produce same-shape outputs.
            raise RuntimeError(
                f'Internal error: alignment produced different shapes '
                f'{obj1.img.shape} vs {obj2.img.shape}'
            )
        if aligned:
            print(f'Aligned shape: {obj1.img.shape}')

        print(f"Applying photometric mode: '{args.photometry}' ...")
        img1 = _apply_photometry(obj1, kind1, args.photometry)
        img2 = _apply_photometry(obj2, kind2, args.photometry)

        same_shape = img1.shape == img2.shape

        ratio_img = None
        ratio_of_pixels = None
        overlap_mask = None
        n_only1 = n_only2 = n_overlap = None
        if same_shape:
            img1_f = np.asarray(ma.filled(img1, np.nan), dtype=np.float64)
            img2_f = np.asarray(ma.filled(img2, np.nan), dtype=np.float64)
            mask1 = ma.getmaskarray(img1)
            mask2 = ma.getmaskarray(img2)
            overlap_mask = mask1 | mask2
            zero2 = img2_f == 0.0
            ratio_mask = overlap_mask | zero2
            ratio_data = np.where(
                ratio_mask,
                np.nan,
                img1_f / np.where(zero2, 1.0, img2_f),
            )
            ratio_img = ma.masked_array(ratio_data, mask=ratio_mask)
            ratio_of_pixels = _compute_stats(ratio_img)

            n_only1 = int(np.sum(~mask1 & mask2))
            n_only2 = int(np.sum(mask1 & ~mask2))
            n_overlap = int(np.sum(~overlap_mask))
            print(f'Overlap: {n_overlap} valid pixels in both images '
                  f'({n_only1} only in file1, {n_only2} only in file2)')
        else:
            print('Overlap: N/A (alignment skipped, different shapes)')

        if args.analyze_all_pixels or not same_shape:
            stats1 = _compute_stats(img1)
            stats2 = _compute_stats(img2)
            stats_region = 'all_pixels'
            label1, label2 = 'File 1 (all px)', 'File 2 (all px)'
        else:
            stats1 = _compute_stats(img1, extra_mask=overlap_mask)
            stats2 = _compute_stats(img2, extra_mask=overlap_mask)
            stats_region = 'overlap'
            label1, label2 = 'File 1 (overlap)', 'File 2 (overlap)'
        ratio_of_stats_dict = _ratio_of_stats(stats1, stats2)
        _print_stats_table(
            label1, label2, stats1, stats2, ratio_of_stats_dict, ratio_of_pixels
        )

        # Angle deltas always use the same-shape overlap region produced by
        # alignment. If alignment was bypassed via --analyze-all-pixels we
        # have nothing meaningful to compare per pixel and skip the table.
        if same_shape:
            angles1 = _angle_arrays_radians(obj1, kind1)
            angles2 = _angle_arrays_radians(obj2, kind2)
            angle_records = _angle_delta_records(
                angles1,
                angles2,
                overlap_mask=overlap_mask,
            )
            _print_angle_delta_table(angle_records)
        else:
            angle_records = []
            print(
                '\nAngle delta statistics: skipped (alignment was bypassed).\n',
                file=sys.stderr,
            )

        if args.output_ratio_mosaic:
            if not same_shape:
                print(
                    '\nWARNING: --output-ratio-mosaic ignored: alignment was bypassed\n',
                    file=sys.stderr,
                )
            else:
                _save_ratio(ratio_img, obj1, kind1, args.output_ratio_mosaic)

        if args.output_statistics_file:
            payload = _build_statistics_payload(
                args=args,
                kind1=kind1,
                kind2=kind2,
                obj1=obj1,
                obj2=obj2,
                same_shape=same_shape,
                n_overlap=n_overlap,
                n_only1=n_only1,
                n_only2=n_only2,
                stats_label=stats_region,
                stats1=stats1,
                stats2=stats2,
                ratio_of_stats_dict=ratio_of_stats_dict,
                ratio_of_pixels=ratio_of_pixels,
                angle_records=angle_records,
            )
            _save_statistics(payload, args.output_statistics_file)

    except KeyboardInterrupt:
        print('', file=sys.stderr)
        sys.exit(130)
    except (ValueError, TypeError) as e:
        print(f'\nERROR: {e}\n', file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f'\nERROR: {e.filename}: {e.strerror}\n', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
