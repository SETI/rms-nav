#!/usr/bin/env python3
"""compare_mosaics -- Compare two ring or body mosaic/reprojection files.

Usage::

    python compare_mosaics.py [--photometry MODE] [--output-ratio-file PATH] FILE1 FILE2

Applies an optional photometric adjustment to both files (undo the stored
correction first, then apply the requested model), checks that the two files
are compatible (same body, same grid dimensions and resolutions), prints
per-image statistics side-by-side, then prints statistics on FILE1 / FILE2.
If --output-ratio-file is given, saves the ratio as a new file of the same
type and format as FILE1 so it can be viewed with nav_mosaic_display.

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
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import numpy.ma as ma

from nav.reproj.bodies import BodyMosaicData, BodyReprojResult
from nav.reproj.rings import RingMosaicData, RingReprojResult
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

    For body mosaics, a shape mismatch is NOT flagged here because both files
    share a global lat/lon grid anchored at -pi/2 / 0; the caller is expected
    to crop both to their physical overlap via :func:`_align_body_mosaics`.
    A shape mismatch on a ring mosaic is still a hard incompatibility because
    the ring-radius sampling and the longitude_antimask are file-specific.
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
        if obj1.img.shape != obj2.img.shape:
            errors.append(f'image shape: {obj1.img.shape} vs {obj2.img.shape}')
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


def _body_overlap_range(obj1, obj2):
    """Return the lat/lon intersection of two body mosaics, or None if disjoint.

    Both mosaics are anchored to the same global grid (-pi/2 step
    ``lat_resolution`` for latitude, 0 step ``lon_resolution`` for longitude),
    so the lat_range / lon_range bounds are integer multiples of the
    resolution and a simple per-axis ``max(min, ...)`` / ``min(max, ...)``
    intersection is well-defined.
    """
    lat_min = max(obj1.lat_range[0], obj2.lat_range[0])
    lat_max = min(obj1.lat_range[1], obj2.lat_range[1])
    lon_min = max(obj1.lon_range[0], obj2.lon_range[0])
    lon_max = min(obj1.lon_range[1], obj2.lon_range[1])

    # Each cell covers one resolution-step; allow a half-cell tolerance before
    # declaring "no overlap" to keep round-off-induced gaps from looking empty.
    lat_tol = 0.5 * obj1.lat_resolution
    lon_tol = 0.5 * obj1.lon_resolution
    if lat_max < lat_min - lat_tol or lon_max < lon_min - lon_tol:
        return None
    return (lat_min, lat_max), (lon_min, lon_max)


def _crop_body_mosaic(obj, lat_range, lon_range):
    """Return a new ``BodyMosaicData`` cropped to the given lat/lon range.

    The 2D fields (``img``, ``resolution``, ``eff_resolution``, ``phase``,
    ``emission``, ``incidence``, ``time``, ``image_number``) are sliced in
    place; the ``lat_range`` / ``lon_range`` metadata is updated to reflect
    the actual extent of the surviving rows/columns. All other fields
    (per-image arrays, ``contributing_image_names``, dtypes, etc.) are
    carried through unchanged.
    """
    lat_res = obj.lat_resolution
    lon_res = obj.lon_resolution

    r0 = int(round((lat_range[0] - obj.lat_range[0]) / lat_res))
    r1 = int(round((lat_range[1] - obj.lat_range[0]) / lat_res)) + 1
    c0 = int(round((lon_range[0] - obj.lon_range[0]) / lon_res))
    c1 = int(round((lon_range[1] - obj.lon_range[0]) / lon_res)) + 1

    n_rows, n_cols = obj.img.shape
    r0 = max(0, min(r0, n_rows))
    r1 = max(r0, min(r1, n_rows))
    c0 = max(0, min(c0, n_cols))
    c1 = max(c0, min(c1, n_cols))

    sl = (slice(r0, r1), slice(c0, c1))
    new_lat_range = (
        obj.lat_range[0] + r0 * lat_res,
        obj.lat_range[0] + (r1 - 1) * lat_res,
    )
    new_lon_range = (
        obj.lon_range[0] + c0 * lon_res,
        obj.lon_range[0] + (c1 - 1) * lon_res,
    )
    return dataclasses.replace(
        obj,
        img=obj.img[sl],
        resolution=obj.resolution[sl],
        eff_resolution=obj.eff_resolution[sl],
        phase=obj.phase[sl],
        emission=obj.emission[sl],
        incidence=obj.incidence[sl],
        time=obj.time[sl],
        image_number=obj.image_number[sl],
        lat_range=new_lat_range,
        lon_range=new_lon_range,
    )


def _align_body_mosaics(obj1, obj2):
    """Crop two body mosaics to their physical lat/lon overlap.

    No-op when the inputs already share a shape and lat/lon range. Returns a
    pair of ``BodyMosaicData`` instances (possibly the originals) whose 2D
    arrays line up index-for-index in lat/lon space.

    Raises:
        ValueError: If the two mosaics do not overlap in lat/lon.
    """
    if (
        obj1.img.shape == obj2.img.shape
        and obj1.lat_range == obj2.lat_range
        and obj1.lon_range == obj2.lon_range
    ):
        return obj1, obj2

    overlap = _body_overlap_range(obj1, obj2)
    if overlap is None:
        raise ValueError(
            f'Body mosaics do not overlap: '
            f'lat {obj1.lat_range} vs {obj2.lat_range}, '
            f'lon {obj1.lon_range} vs {obj2.lon_range}'
        )
    lat_range, lon_range = overlap
    return _crop_body_mosaic(obj1, lat_range, lon_range), _crop_body_mosaic(
        obj2, lat_range, lon_range
    )


def _compute_stats(arr, extra_mask=None):
    """Return a dict of statistics for unmasked pixels, optionally with an extra exclusion mask."""
    combined = ma.masked_array(arr, mask=ma.getmaskarray(arr) | (extra_mask if extra_mask is not None else False))
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


def _stat_ratio(v1, v2):
    """Return v1/v2 as float, or nan if not computable."""
    if not (math.isfinite(v1) and math.isfinite(v2)) or v2 == 0.0:
        return float('nan')
    return v1 / v2


def _print_stats_table(label1, label2, stats1, stats2):
    """Print a three-column statistics table; ratio column is always stats1/stats2."""
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

    sep = '-' * (lw + 2 + cw + 2 + cw + 2 + cw)
    print()
    print(f'{"Statistic":<{lw}}  {label1[:cw]:>{cw}}  {label2[:cw]:>{cw}}  {"Ratio (1/2)":>{cw}}')
    print(sep)
    for key, name, fmt in rows:
        v1, v2 = stats1[key], stats2[key]
        s1 = _fmt(v1, fmt)
        s2 = _fmt(v2, fmt)
        sr = _fmt(_stat_ratio(v1, v2), fmt)
        print(f'{name:<{lw}}  {s1:>{cw}}  {s2:>{cw}}  {sr:>{cw}}')
    print()


def _save_ratio(ratio_img, obj1, kind1, output_path):
    """Save ratio image in the same type and format as obj1."""
    ratio_f = ma.masked_array(
        np.asarray(ratio_img, dtype=np.float32),
        mask=ma.getmaskarray(ratio_img),
    )
    new_dtype = np.dtype(np.float32)
    out = dataclasses.replace(obj1, img=ratio_f, photometric_model_name=None, image_dtype=new_dtype)
    out.save(output_path)
    print(f'Ratio saved to: {output_path}')


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
        '--output-ratio-file',
        default=None,
        metavar='PATH',
        help='If given, save FILE1/FILE2 ratio as a mosaic to this path.',
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

        # Body mosaics are routinely cropped to their per-file data extent on
        # save (BodyMosaic.to_bounded), so two mosaics of the same body at the
        # same resolution can have different shapes. They share a global grid,
        # so align them on the physical lat/lon overlap before comparing.
        if (
            kind1 == 'body_mosaic'
            and kind2 == 'body_mosaic'
            and (obj1.img.shape != obj2.img.shape or obj1.lat_range != obj2.lat_range
                 or obj1.lon_range != obj2.lon_range)
        ):
            obj1, obj2 = _align_body_mosaics(obj1, obj2)
            print(
                f'Aligned to lat/lon overlap: '
                f'lat=[{math.degrees(obj1.lat_range[0]):.3f},'
                f'{math.degrees(obj1.lat_range[1]):.3f}] deg, '
                f'lon=[{math.degrees(obj1.lon_range[0]):.3f},'
                f'{math.degrees(obj1.lon_range[1]):.3f}] deg, '
                f'shape: {obj1.img.shape}'
            )

        print(f"Applying photometric mode: '{args.photometry}' ...")
        img1 = _apply_photometry(obj1, kind1, args.photometry)
        img2 = _apply_photometry(obj2, kind2, args.photometry)

        same_shape = img1.shape == img2.shape

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

            n_only1 = int(np.sum(~mask1 & mask2))
            n_only2 = int(np.sum(mask1 & ~mask2))
            n_overlap = int(np.sum(~overlap_mask))
            print(f'Overlap: {n_overlap} valid pixels in both images '
                  f'({n_only1} only in file1, {n_only2} only in file2)')
        else:
            print('Overlap: N/A (different shapes)')

        if args.analyze_all_pixels or not same_shape:
            stats1 = _compute_stats(img1)
            stats2 = _compute_stats(img2)
            _print_stats_table('File 1 (all px)', 'File 2 (all px)', stats1, stats2)
        else:
            stats1 = _compute_stats(img1, extra_mask=overlap_mask)
            stats2 = _compute_stats(img2, extra_mask=overlap_mask)
            _print_stats_table('File 1 (overlap)', 'File 2 (overlap)', stats1, stats2)

        if args.output_ratio_file:
            if not same_shape:
                print(
                    '\nWARNING: --output-ratio-file ignored: images have different shapes\n',
                    file=sys.stderr,
                )
            else:
                _save_ratio(ratio_img, obj1, kind1, args.output_ratio_file)

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
