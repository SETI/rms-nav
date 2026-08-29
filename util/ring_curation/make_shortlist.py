"""Build an operator-curation shortlist of Cassini ISS ring images.

Stratifies the RMS Node ring-summary metadata (one row per image, no SPICE
needed) across the axes that drive ring navigation behavior:

  - radial resolution band (km/px)
  - lit vs unlit ring face   (sign of solar vs observer ring opening angle)
  - ring opening regime      (|observer ring opening angle|)

and, across the whole list, coverage of the catalog's ring REGIONS, so the
set exercises different feature types (gap edges, ringlet edges, the broad
A/B edges) and the full spread of catalog orbit uncertainties (0.08 km to
10.18 km, a 124x range).

Geometry alone cannot tell whether ring features are actually VISIBLE in a
frame: the summary reports which radii intersect the FOV, but not occlusion
by Saturn, foreshortening, exposure, or smear.  A first geometry-only
shortlist was mostly unusable for exactly those reasons, so selection is
two-stage:

  1. ``pool``   -- write ``pool.csv``: the top candidates per grid cell,
     several deep, stratified across face and opening within each cell.
  2. (outside this script) every pooled image's holdings preview is
     inspected visually; verdicts land in ``screen.csv`` with columns
     ``image_id,verdict,reason`` and verdict one of good / marginal / bad.
     Usable frames also get a scene-background call in ``background.csv``
     (``image_id,background,reason``, background one of sky / mixed /
     planet): sky means the rings fill the frame or sit on dark sky,
     planet means Saturn's disk dominates and the rings are mostly seen
     against it, mixed is substantially both.
  3. ``select`` -- fill the grid only from visually-screened candidates:
     primaries and spares from ``good``, geometry back-fill from ``good``
     then ``marginal``.  A cell with no visually usable frame stays empty
     and is reported, rather than being filled with an unusable one.
     Planet-dominated frames are chosen only when a cell has no sky or
     mixed alternative -- optically thin rings seen from the unlit side
     often show ONLY in transmission against the disk, so those cells
     keep their frame, tagged, rather than going empty.  Mixed frames
     are kept as a deliberate axis (rings against the planet are a real
     scene class the navigator must handle), and every shortlist row
     carries its ``background`` so the composition is explicit.

Edge-on frames (either opening angle within 0.3 deg of zero) are excluded
outright: a ring seen edge-on is a line, with no radial features to
navigate against.

Candidates the operator then navigates manually
(sd_offset <dataset> <image> --manual, "Save as Library Entry...").

Truth policy per the 2026-08-28 audit revision: bundle-published pointing is
NOT used for anything here; membership in the spokes bundle is only recorded
as a column, and non-bundle images are preferred for variety.
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

META = Path('/mnt/ganymede/PDS/holdings/metadata/COISS_2xxx/COISS_2999')
REPO = Path(__file__).resolve().parents[2]
AUDIT = Path('/seti/newnav/spokes-bundle-audit')
OUT = Path(__file__).resolve().parent

RES_BANDS = [(0, 5), (5, 25), (25, 100), (100, 300), (300, 1000), (1000, 3000)]
OPEN_BINS = [('near-plane', 0.0, 8.0), ('mid', 8.0, 20.0), ('open', 20.0, 90.0)]
POOL_PER_PRIMARY_CELL = 10  # region x band cells; screened visually before selection
POOL_PER_GEO_CELL = 4  # band x face x opening back-fill cells


def label_columns(lbl_path: Path) -> dict[str, int]:
    """COLUMN_NUMBER -> NAME mapping from the PDS3 label, as name -> 0-based index."""
    text = lbl_path.read_text()
    cols = {}
    for m in re.finditer(r'COLUMN_NUMBER\s*=\s*(\d+)\s+NAME\s*=\s*"?([A-Z0-9_]+)"?', text):
        cols[m.group(2)] = int(m.group(1)) - 1
    return cols


def catalog_regions() -> tuple[list[tuple[str, float, float]], list[tuple[float, float]]]:
    """Region intervals spanned by catalog edges, plus every (radius, rms) edge."""
    with open(REPO / 'src/spindoctor/config_files/config_310_saturn_rings.yaml') as fp:
        d = yaml.safe_load(fp)
    feats = d['rings']['ring_features']['SATURN']['features']
    edges = []
    for f in feats.values():
        for side in ('inner_data', 'outer_data'):
            data = f.get(side) or []
            if data:
                m1 = next((e for e in data if e.get('mode') == 1), data[0])
                edges.append((float(m1['a']), float(m1.get('rms', 0.0))))
    regions = [
        ('C_inner', 74615, 80000),
        ('C_outer', 84000, 90614),
        ('B_inner_feats', 100024, 104083),
        ('B_outer_huygens', 117569, 117931),
        ('CassiniDiv_mid', 118283, 120316),
        ('A_inner', 122050, 122578),
        ('Encke', 133423, 133745),
        ('Keeler_A_outer', 136485, 136773),
    ]
    return regions, sorted(edges)


def build_rows() -> tuple[list[dict], list[str], list[str]]:
    """All geometry-eligible candidates, plus the region and band name lists."""
    cols = label_columns(META / 'COISS_2999_ring_summary.lbl')
    need = [
        'VOLUME_ID',
        'FILE_SPECIFICATION_NAME',
        'MINIMUM_RING_RADIUS',
        'MAXIMUM_RING_RADIUS',
        'FINEST_RADIAL_RESOLUTION',
        'MINIMUM_RING_PHASE_ANGLE',
        'MAXIMUM_RING_PHASE_ANGLE',
        'SOLAR_RING_OPENING_ANGLE',
        'OBSERVER_RING_OPENING_ANGLE',
    ]
    idx = {n: cols[n] for n in need}
    regions, edges = catalog_regions()

    bundle = set()
    ball = AUDIT / 'lists/bundle_all_6781_names.txt'
    if ball.exists():
        for n in ball.read_text().split():
            bundle.add(('N' if n.endswith('n') else 'W') + n[:10])

    rows = []
    with open(META / 'COISS_2999_ring_summary.tab') as fp:
        for rec in csv.reader(fp):
            try:
                rmin = float(rec[idx['MINIMUM_RING_RADIUS']])
                rmax = float(rec[idx['MAXIMUM_RING_RADIUS']])
                res = float(rec[idx['FINEST_RADIAL_RESOLUTION']])
                sopen = float(rec[idx['SOLAR_RING_OPENING_ANGLE']])
                oopen = float(rec[idx['OBSERVER_RING_OPENING_ANGLE']])
                pmin = float(rec[idx['MINIMUM_RING_PHASE_ANGLE']])
                pmax = float(rec[idx['MAXIMUM_RING_PHASE_ANGLE']])
            except (ValueError, IndexError):
                continue
            if not (0.05 <= res < 3000) or rmax <= rmin:
                continue
            if abs(oopen) < 0.3 or abs(sopen) < 0.3:
                continue  # edge-on: no radial features to navigate against
            covered = [n for n, lo, hi in regions if rmin < hi and rmax > lo]
            if not covered:
                continue
            in_range = [(a, r) for a, r in edges if rmin <= a <= rmax]
            if not in_range:
                continue
            spec = rec[idx['FILE_SPECIFICATION_NAME']].strip()
            image_id = spec.split('/')[-1].replace('.LBL', '')
            pds3 = image_id.split('_')[0]
            face = 'lit' if (sopen > 0) == (oopen > 0) else 'unlit'
            band = next((f'{lo}-{hi}' for lo, hi in RES_BANDS if lo <= res < hi), None)
            if band is None:
                continue
            obin = next((n for n, lo, hi in OPEN_BINS if lo <= abs(oopen) < hi), 'open')
            rms_vals = [r for _, r in in_range]
            rows.append(
                {
                    'image_id': image_id,
                    'camera': pds3[0],
                    'volume': rec[idx['VOLUME_ID']].strip(),
                    'file_spec': spec,
                    'res_kmpx': round(res, 3),
                    'res_band': band,
                    'face': face,
                    'opening_bin': obin,
                    'obs_opening_deg': round(oopen, 2),
                    'solar_opening_deg': round(sopen, 2),
                    'phase_min_deg': round(pmin, 1),
                    'phase_max_deg': round(pmax, 1),
                    'ring_rmin_km': round(rmin),
                    'ring_rmax_km': round(rmax),
                    'regions': '+'.join(covered),
                    'n_regions': len(covered),
                    'n_catalog_edges': len(in_range),
                    'edge_rms_min_km': round(min(rms_vals), 2),
                    'edge_rms_max_km': round(max(rms_vals), 2),
                    'in_spokes_bundle': pds3 in bundle,
                }
            )
    region_names = [n for n, _, _ in regions]
    band_names = [f'{lo}-{hi}' for lo, hi in RES_BANDS]
    return rows, region_names, band_names


def static_prefer(r: dict) -> tuple:
    """Cell-local ranking: targeted frames first, then fresh (non-bundle) imagery."""
    return (r['n_regions'], r['in_spokes_bundle'], r['res_kmpx'])


def cmd_pool() -> int:
    """Write pool.csv: the visual-screening candidate pool, several deep per cell."""
    rows, region_names, band_names = build_rows()
    pooled: dict[str, dict] = {}

    def add(r: dict, cell: str, rank: int) -> None:
        if r['image_id'] in pooled:
            return
        r = dict(r)
        r['cell'] = cell
        r['pool_rank'] = rank
        pooled[r['image_id']] = r

    # Primary grid: region x band, round-robin across (face, opening) subgroups
    # inside each cell so the screened pool keeps its geometry diversity.
    for region in region_names:
        for band in band_names:
            cell_rows = [
                r for r in rows if region in r['regions'].split('+') and r['res_band'] == band
            ]
            groups: dict[tuple[str, str], list[dict]] = {}
            for r in cell_rows:
                groups.setdefault((r['face'], r['opening_bin']), []).append(r)
            for g in groups.values():
                g.sort(key=static_prefer)
            order = sorted(groups)
            taken = 0
            depth = 0
            while taken < POOL_PER_PRIMARY_CELL:
                any_left = False
                for key in order:
                    g = groups[key]
                    if depth < len(g):
                        any_left = True
                        add(g[depth], f'{region}|{band}', taken)
                        taken += 1
                        if taken >= POOL_PER_PRIMARY_CELL:
                            break
                if not any_left:
                    break
                depth += 1

    # Geometry back-fill pool: every (band, face, opening) cell keeps a few
    # candidates so the lit/unlit and opening axes can be covered even where
    # the primary grid's picks fail the visual screen.
    geo_cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        geo_cells.setdefault((r['res_band'], r['face'], r['opening_bin']), []).append(r)
    for key in sorted(geo_cells):
        pool = sorted(geo_cells[key], key=static_prefer)
        for rank, r in enumerate(pool[:POOL_PER_GEO_CELL]):
            add(r, 'geo|' + '|'.join(key), rank)

    out = OUT / 'pool.csv'
    plist = sorted(pooled.values(), key=lambda r: (r['cell'], r['pool_rank']))
    with out.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(plist[0].keys()))
        w.writeheader()
        w.writerows(plist)
    print(f'{len(rows)} candidates survived the geometry filters')
    print(f'pooled {len(plist)} for visual screening -> {out}')
    return 0


def load_screen(path: Path) -> dict[str, str]:
    """image_id -> verdict from the visual-screening CSV."""
    verdicts = {}
    with path.open() as fp:
        for rec in csv.DictReader(fp):
            v = rec['verdict'].strip().lower()
            if v not in ('good', 'marginal', 'bad'):
                raise ValueError(f'unknown verdict {v!r} for {rec["image_id"]}')
            verdicts[rec['image_id']] = v
    return verdicts


def load_background(path: Path) -> dict[str, str]:
    """image_id -> scene background (sky / mixed / planet) from background.csv."""
    backgrounds = {}
    with path.open() as fp:
        for rec in csv.DictReader(fp):
            b = rec['background'].strip().lower()
            if b not in ('sky', 'mixed', 'planet'):
                raise ValueError(f'unknown background {b!r} for {rec["image_id"]}')
            backgrounds[rec['image_id']] = b
    return backgrounds


def cmd_select(screen_path: Path, background_path: Path) -> int:
    """Fill the grid from visually screened candidates and write shortlist.csv."""
    rows, region_names, band_names = build_rows()
    verdicts = load_screen(screen_path)
    backgrounds = load_background(background_path)
    good = [r for r in rows if verdicts.get(r['image_id']) == 'good']
    marginal = [r for r in rows if verdicts.get(r['image_id']) == 'marginal']
    unclassified = [r['image_id'] for r in good + marginal if r['image_id'] not in backgrounds]
    if unclassified:
        raise ValueError(
            f'{len(unclassified)} usable frames lack a background call, e.g. {unclassified[:3]}'
        )
    for r in good + marginal:
        r['background'] = backgrounds[r['image_id']]
    print(f'screen: {len(verdicts)} verdicts; usable good={len(good)} marginal={len(marginal)}')

    geo_seen: dict[tuple[str, str, str], int] = {}
    cam_seen = {'N': 0, 'W': 0}
    picked: list[dict] = []
    picked_ids: set[str] = set()

    def take(r: dict, role: str) -> None:
        r = dict(r)
        r['role'] = role
        picked.append(r)
        picked_ids.add(r['image_id'])
        key = (r['res_band'], r['face'], r['opening_bin'])
        geo_seen[key] = geo_seen.get(key, 0) + 1
        cam_seen[r['camera']] += 1

    def prefer(r: dict) -> tuple:
        geo = geo_seen.get((r['res_band'], r['face'], r['opening_bin']), 0)
        return (
            r['background'] == 'planet',  # a planet-dominated frame only as last resort
            r['n_regions'],  # targeted beats panoramic
            geo,  # unseen geometry first
            r['in_spokes_bundle'],  # fresh imagery first
            cam_seen[r['camera']],  # balance the cameras
        )

    empty_cells: list[str] = []
    for role in ('primary', 'spare'):
        for region in region_names:
            for band in band_names:
                pool = [
                    r
                    for r in good
                    if region in r['regions'].split('+')
                    and r['res_band'] == band
                    and r['image_id'] not in picked_ids
                ]
                if not pool:
                    if role == 'primary':
                        empty_cells.append(f'{region} x {band}')
                    continue
                pool.sort(key=prefer)
                take(pool[0], f'{role}:{region}')

    # Geometry back-fill: every (band, face, opening) cell with a screened
    # candidate gets at least one pick; marginal frames are allowed here.
    geo_cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in good + marginal:
        geo_cells.setdefault((r['res_band'], r['face'], r['opening_bin']), []).append(r)
    for key in sorted(geo_cells):
        if geo_seen.get(key, 0) == 0:
            pool = [r for r in geo_cells[key] if r['image_id'] not in picked_ids]
            if pool:
                pool.sort(key=prefer)
                take(pool[0], 'geometry-fill')

    out_csv = OUT / 'shortlist.csv'
    with out_csv.open('w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=list(picked[0].keys()))
        w.writeheader()
        w.writerows(picked)

    prim = [p for p in picked if p['role'].startswith('primary')]
    print(f'picked {len(picked)} ({len(prim)} primaries, {len(picked) - len(prim)} spares/fills)')
    import statistics

    print(f'regions in view per primary: median {statistics.median(p["n_regions"] for p in prim)}')
    print('\nprimary grid — region x resolution band:')
    print(f'  {"":16s}' + ''.join(f'{b:>11s}' for b in band_names))
    for region in region_names:
        line = f'  {region:16s}'
        for band in band_names:
            n = sum(1 for p in prim if p['role'] == f'primary:{region}' and p['res_band'] == band)
            line += f'{n:>11d}'
        print(line)
    if empty_cells:
        print('\nprimary cells with NO visually usable candidate (left empty):')
        for c in empty_cells:
            print(f'  {c}')
    print('\nface / opening coverage over the whole list:')
    for k, v in sorted(Counter((p['face'], p['opening_bin']) for p in picked).items()):
        print(f'  {k[0]:8s} {k[1]:10s} {v:3d}')
    print('\nscene background over the whole list:')
    for k, v in sorted(Counter(p['background'] for p in picked).items()):
        print(f'  {k:8s} {v:3d}')
    print(
        '  cameras: '
        + ', '.join(f'{k}={v}' for k, v in sorted(Counter(p['camera'] for p in picked).items()))
    )
    print(f'  in spokes bundle: {sum(1 for p in picked if p["in_spokes_bundle"])} of {len(picked)}')
    print(f'\nwrote {out_csv}')
    return 0


def main() -> int:
    """Dispatch the pool and select subcommands."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('pool', help='write pool.csv, the visual-screening candidate pool')
    sel = sub.add_parser('select', help='write shortlist.csv from screened candidates')
    sel.add_argument(
        '--screen',
        type=Path,
        default=OUT / 'screen.csv',
        help='visual-screening verdicts (image_id,verdict,reason)',
    )
    sel.add_argument(
        '--background',
        type=Path,
        default=OUT / 'background.csv',
        help='scene-background calls (image_id,background,reason)',
    )
    args = ap.parse_args()
    if args.cmd == 'pool':
        return cmd_pool()
    return cmd_select(args.screen, args.background)


if __name__ == '__main__':
    sys.exit(main())
