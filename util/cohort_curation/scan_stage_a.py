"""Stage A cohort scan (plans/COHORT_CURATION_PLAN.md): candidate manifest
for the eight empty scene classes of the Phase-10 49-image budget:

    body_irregular, faint_stars, negative_cases, ring_only_flat,
    ring_plus_body, scattered_light, stars_plus_body,
    two_bright_stars_no_body

Scans the local PDS geometry metadata only (no image access).  Star counts
use the UCAC4 catalog directly with the per-camera limiting-magnitude
formulas mirrored from spindoctor.obs.obs_inst_* (Pogson form:
anchor + ln(texp)/ln(2.512)).

Output (under _work/cohort_curation/, gitignored): candidates_batch001.yaml
(stratified, seeded sample) plus scan_counts.txt with raw per-class hit
counts.

Run:  venv/bin/python util/cohort_curation/scan_stage_a.py
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import yaml

import pdsmeta as pm

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'   # generated outputs (gitignored)

SEED = 20260708
QUERY_NAME = 'stage_a_batch001'

RADII = json.loads((HERE / 'body_radii.json').read_text())['radius_km']

# Usable epochs per mission (see spice_coverage.json _provenance); frames
# outside these windows fail navigation with missing-SPICE errors, so they
# are excluded from candidacy for every scene class.
SPICE_COVERAGE = json.loads(
    (HERE / 'spice_coverage.json').read_text())['windows']


def epoch_covered(mission: str, image_time: str) -> bool:
    """True if the ISO image time falls inside a usable SPICE window.

    Parameters:
        mission: Mission key ('COISS', 'VGISS', 'GOSSI', 'NHLORRI').
        image_time: ISO date/time string from the index table; an empty
            or unknown value is treated as covered (Stage B triage
            catches any survivor).
    """
    if not image_time:
        return True
    windows = SPICE_COVERAGE.get(mission)
    if not windows:
        return True
    day = image_time[:10]
    return any(lo <= day <= hi for lo, hi in windows)

# Images already in the library (curated or uncurated) -- never re-offer.
EXISTING_IDS = {
    p.stem.replace('_CALIB', '')
    for p in (REPO / 'tests/integration/image_library').rglob('*.yaml')
}

IRREGULARS = {'PHOEBE', 'HYPERION', 'JANUS', 'EPIMETHEUS', 'PROMETHEUS',
              'PANDORA', 'ATLAS', 'PAN', 'TELESTO', 'CALYPSO', 'HELENE'}

FOV_DEG = {('COISS', 'NAC'): 0.35, ('COISS', 'WAC'): 3.5,
           ('VGISS', 'NA'): 0.424, ('VGISS', 'WA'): 3.169,
           ('GOSSI', 'SSI'): 0.46, ('NHLORRI', 'LORRI'): 0.29}

FRAME_PX = 1024.0
FLAT_SAGITTA_PX = 0.5          # PHASE10 rank-1 curvature threshold
FLAT_MIN_APPARENT_R_PX = FRAME_PX * FRAME_PX / (8.0 * FLAT_SAGITTA_PX)

LN_POGSON = math.log(2.512)


def maglim(mission: str, camera: str, texp_s: float) -> float:
    """Limiting vmag; mirrors spindoctor.obs.obs_inst_* star_max_usable_vmag."""
    anchors = {('COISS', 'NAC'): 10.5, ('COISS', 'WAC'): 10.7,
               ('GOSSI', 'SSI'): 10.3, ('NHLORRI', 'LORRI'): 11.7,
               ('VGISS', 'NA'): 8.3, ('VGISS', 'WA'): 5.9}
    anchor = anchors[(mission, camera)]
    if texp_s <= 0:
        return anchor
    if (mission, camera) == ('COISS', 'WAC'):
        return anchor + math.log(texp_s / 26.0) / LN_POGSON
    return anchor + math.log(texp_s) / LN_POGSON


# ---------------------------------------------------------------- ring edges

def load_saturn_ring_edges() -> dict[str, float]:
    """{feature_edge_name: a_km} for every mode-1 edge in config_310."""
    cfg = yaml.safe_load(
        (REPO / 'src/spindoctor/config_files/config_310_saturn_rings.yaml')
        .read_text())
    edges: dict[str, float] = {}
    feats = cfg['rings']['ring_features']['SATURN']['features']
    for fname, feat in feats.items():
        for side in ('inner_data', 'outer_data'):
            for mode in feat.get(side) or []:
                if mode.get('mode') == 1 and 'a' in mode:
                    edges[f'{fname}.{side.split("_")[0]}'] = mode['a']
    return edges


RING_EDGES = load_saturn_ring_edges()


def edges_in_frame(rmin: float, rmax: float) -> list[str]:
    margin = 0.02 * (rmax - rmin)
    return [name for name, a in RING_EDGES.items()
            if rmin + margin <= a <= rmax - margin]


# ---------------------------------------------------------------- star counts

_UCAC4 = None


def star_vmags(ra_deg: float, dec_deg: float, fov_deg: float,
               vmag_max: float) -> list[float]:
    """Sorted vmags of UCAC4 stars in a box around the pointing."""
    global _UCAC4
    if _UCAC4 is None:
        from starcat import UCAC4StarCatalog
        _UCAC4 = UCAC4StarCatalog('/data/external-data/star-catalogs/UCAC4')
    half = math.radians(0.75 * fov_deg)      # frame + pointing-error margin
    dec = math.radians(dec_deg)
    ra = math.radians(ra_deg % 360.0)
    dec_min = max(dec - half, -math.pi / 2)
    dec_max = min(dec + half, math.pi / 2)
    cosd = max(math.cos(dec), 0.05)
    ra_half = half / cosd
    vmags: list[float] = []
    if ra - ra_half < 0 or ra + ra_half > 2 * math.pi:
        boxes = [(0.0, (ra + ra_half) % (2 * math.pi)),
                 ((ra - ra_half) % (2 * math.pi), 2 * math.pi)]
    else:
        boxes = [(ra - ra_half, ra + ra_half)]
    for ra0, ra1 in boxes:
        for star in _UCAC4.find_stars(ra_min=ra0, ra_max=ra1,
                                      dec_min=dec_min, dec_max=dec_max,
                                      vmag_max=vmag_max):
            v = getattr(star, 'vmag', None)
            if v is not None:
                vmags.append(float(v))
    return sorted(vmags)


# ---------------------------------------------------------------- helpers

def app_diam_px(target: str, center_res: float | None) -> float | None:
    r = RADII.get(target)
    if r is None or center_res is None or center_res <= 0:
        return None
    return 2.0 * r / center_res


def resolved(t: pm.SummaryTable, row: list[str]) -> bool:
    return (t.num(row, 'MINIMUM_PLANETOCENTRIC_LATITUDE') is not None
            and t.num(row, 'MINIMUM_IAU_LONGITUDE') is not None)


def phase_bin(phase: float | None) -> str:
    if phase is None:
        return 'unknown'
    for hi, name in ((30, '<30'), (60, '30-60'), (90, '60-90'),
                     (120, '90-120')):
        if phase < hi:
            return name
    return '>120'


def res_decade(res: float | None) -> str:
    if res is None:
        return 'unknown'
    if res < 1:
        return '<1km'
    if res < 10:
        return '1-10km'
    return '>10km'


def cand(scene_class: str, volset: str, volume: str, filespec: str,
         mission: str, camera: str, dataset: str, strata: tuple,
         selection: dict, *, needs_visual: bool = False) -> dict:
    stem = Path(filespec.strip()).stem
    img_name = stem.split('_')[0] if mission != 'GOSSI' else stem
    return {
        'scene_class': scene_class,
        'image_name': img_name,
        'filespec': filespec.strip(),
        'volset': volset,
        'volume': volume,
        'mission': mission,
        'camera': camera,
        'dataset': dataset,
        'strata': ' | '.join(str(s) for s in strata),
        'needs_visual': needs_visual,
        'selection': selection,
    }


# ---------------------------------------------------------------- COISS scan

def scan_coiss() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    star_frame_pool: list[dict] = []      # cheap-pass no-body frames
    body_star_pool: list[dict] = []       # cheap-pass one-body frames
    tiny_pool: list[dict] = []            # negative: tiny body frames

    # COISS_1xxx (Jupiter cruise) feeds the star/negative pools only: the
    # ring-edge catalog in config_310 is Saturn's, so ring classes are
    # restricted to COISS_2xxx.  main_ring_span is the radius window that
    # decides whether visible main rings are in frame.
    volsets = [
        dict(volset='COISS_2xxx', dataset='coiss_saturn',
             planet_kind='saturn_summary', ring_classes=True,
             main_ring_span=(70000.0, 145000.0)),
        dict(volset='COISS_1xxx', dataset='coiss_cruise',
             planet_kind='jupiter_summary', ring_classes=False,
             main_ring_span=(100000.0, 132000.0)),
    ]
    for vs in volsets:
        for volume in pm.volumes(vs['volset']):
            _scan_coiss_volume(vs, volume, out, star_frame_pool,
                               body_star_pool, tiny_pool)
    _coiss_star_passes(out, star_frame_pool, body_star_pool, tiny_pool)
    return out


def _scan_coiss_volume(vs: dict, volume: str, out: dict[str, list[dict]],
                       star_frame_pool: list[dict],
                       body_star_pool: list[dict],
                       tiny_pool: list[dict]) -> None:
    volset, dataset = vs['volset'], vs['dataset']
    ring_classes = vs['ring_classes']
    span_lo, span_hi = vs['main_ring_span']

    moon = pm.load_table(volset, volume, 'moon_summary')
    ring = pm.load_table(volset, volume, 'ring_summary')
    sat = pm.load_table(volset, volume, vs['planet_kind'])
    idx = pm.load_table(volset, volume, 'index')
    if not (moon and idx):
        return

    # The index uses .IMG filespecs; the summary tables use .LBL.
    # Join on the extension-stripped path.
    def key(fs: str | None) -> str:
        return (fs or '').rsplit('.', 1)[0]

    moon_rows: dict[str, list[list[str]]] = defaultdict(list)
    for row in moon.rows:
        moon_rows[key(moon.get(row, 'FILE_SPECIFICATION_NAME'))].append(row)
    ring_rows = ({key(ring.get(r, 'FILE_SPECIFICATION_NAME')): r
                  for r in ring.rows} if ring else {})
    sat_rows = ({key(sat.get(r, 'FILE_SPECIFICATION_NAME')): r
                 for r in sat.rows} if sat else {})

    for irow in idx.rows:
        filespec = idx.get(irow, 'FILE_SPECIFICATION_NAME')
        if not filespec:
            continue
        filespec = key(filespec) + '.LBL'
        stem = Path(filespec).stem
        if stem.split('_')[0] in EXISTING_IDS or stem in EXISTING_IDS:
            continue
        camera = 'NAC' if stem.startswith('N') else 'WAC'
        texp_ms = idx.num(irow, 'EXPOSURE_DURATION')
        texp = texp_ms / 1000.0 if texp_ms else None
        filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
        time_full = idx.get(irow, 'IMAGE_TIME') or ''
        if not epoch_covered('COISS', time_full):
            continue
        time = time_full[:4]
        ra = idx.num(irow, 'RIGHT_ASCENSION')
        dec = idx.num(irow, 'DECLINATION')

        mrows = moon_rows.get(key(filespec), [])
        rrow = ring_rows.get(key(filespec))
        srow = sat_rows.get(key(filespec))
        planet_disc = srow is not None and resolved(sat, srow)

        res_moons = []      # (target, diam_px, row)
        all_diams = []
        for mr in mrows:
            tgt = (moon.get(mr, 'TARGET_NAME') or '').strip()
            d = app_diam_px(tgt, moon.num(mr, 'CENTER_RESOLUTION'))
            if d is not None:
                all_diams.append((tgt, d))
            if resolved(moon, mr) and d is not None:
                res_moons.append((tgt, d, mr))

        redges = []
        rmin = rmax = bobs = rres = None
        if rrow is not None:
            rmin = ring.num(rrow, 'MINIMUM_RING_RADIUS')
            rmax = ring.num(rrow, 'MAXIMUM_RING_RADIUS')
            bobs = ring.num(rrow, 'OBSERVER_RING_OPENING_ANGLE')
            rres = ring.num(rrow, 'FINEST_RING_INTERCEPT_RESOLUTION')
            if ring_classes and rmin is not None and rmax is not None:
                redges = edges_in_frame(rmin, rmax)
        # A ring_summary row exists for nearly every frame (the ring
        # PLANE crosses the FOV); visible main rings require the radius
        # range to overlap the main-ring span.
        rings_visible = (rmin is not None and rmax is not None
                         and rmin < span_hi and rmax > span_lo)

        # ---- body_irregular
        for tgt, d, mr in res_moons:
            if tgt in IRREGULARS and 150 <= d <= 800:
                ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
                out['body_irregular'].append(cand(
                    'body_irregular', volset, volume, filespec, 'COISS',
                    camera, dataset, (tgt, phase_bin(ph)),
                    {'target': tgt, 'apparent_diameter_px': round(d, 1),
                     'center_phase_deg': ph, 'filter': filt,
                     'year': time}))

        # ---- ring_plus_body (skip near-edge-on rings; PHASE10 gotcha)
        if (redges and res_moons and not planet_disc
                and bobs is not None and abs(bobs) >= 1.0):
            tgt, d, mr = max(res_moons, key=lambda x: x[1])
            if d >= 50:
                ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
                out['ring_plus_body'].append(cand(
                    'ring_plus_body', volset, volume, filespec, 'COISS',
                    camera, dataset,
                    (tgt, phase_bin(ph), round(bobs or 0, -1)),
                    {'target': tgt, 'apparent_diameter_px': round(d, 1),
                     'ring_edges_in_frame': redges[:6],
                     'ring_opening_deg': bobs,
                     'center_phase_deg': ph, 'filter': filt,
                     'year': time}))

        # ---- ring_only_flat
        if (redges and not res_moons and not planet_disc
                and bobs is not None and abs(bobs) >= 1.0
                and rres is not None and rres > 0):
            best = None
            for name in redges:
                a = RING_EDGES[name]
                r_app = a * abs(math.sin(math.radians(bobs))) / rres
                if best is None or r_app > best[1]:
                    best = (name, r_app)
            assert best is not None
            is_flat = best[1] >= FLAT_MIN_APPARENT_R_PX
            closeup = rres < 0.3
            if is_flat or closeup:
                out['ring_only_flat'].append(cand(
                    'ring_only_flat', volset, volume, filespec, 'COISS',
                    camera, dataset,
                    (round(bobs, -1), res_decade(rres)),
                    {'ring_edges_in_frame': redges[:6],
                     'flattest_edge': best[0],
                     'apparent_curvature_radius_px': round(best[1]),
                     'criterion': 'sagitta' if is_flat else 'closeup',
                     'ring_opening_deg': bobs,
                     'finest_ring_intercept_res_km': rres,
                     'filter': filt, 'year': time}))

        # ---- pools for star-count classes (catalog query deferred)
        # No exposure floor: bright-pair star-cal frames are typically
        # SHORT exposures (low maglim leaves only the pair detectable).
        no_body = not mrows or all(not resolved(moon, mr) for mr in mrows)
        if (no_body and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and texp > 0
                and camera == 'NAC'):
            star_frame_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time))

        if (len(res_moons) == 1 and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and texp >= 0.5
                and camera == 'NAC'
                and 30 <= res_moons[0][1] <= 700):
            tgt, d, mr = res_moons[0]
            ph = moon.num(mr, 'CENTER_PHASE_ANGLE')
            body_star_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time,
                     target=tgt, diam=d, phase=ph))

        # ---- negative: tiny distant body, no rings
        if (mrows and not rings_visible and not planet_disc
                and ra is not None and dec is not None
                and texp is not None and all_diams
                and all(d < 6 for _, d in all_diams)
                and not res_moons and camera == 'NAC'):
            tiny_pool.append(
                dict(filespec=filespec, volset=volset, volume=volume,
                     dataset=dataset, camera=camera,
                     texp=texp, ra=ra, dec=dec, filt=filt, year=time,
                     bodies={t: round(d, 2) for t, d in all_diams}))


def _coiss_star_passes(out: dict[str, list[dict]],
                       star_frame_pool: list[dict],
                       body_star_pool: list[dict],
                       tiny_pool: list[dict]) -> None:
    rng = random.Random(SEED)

    def sample(pool: list[dict], n: int) -> list[dict]:
        pool = sorted(pool, key=lambda c: c['filespec'])
        rng.shuffle(pool)
        return pool[:n]

    for fr in sample(star_frame_pool, len(star_frame_pool)):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim + 2.0)
        n_bright = sum(1 for v in vm if v <= lim)
        if n_bright == 2 and (len(vm) < 3 or vm[2] >= vm[1] + 1.5):
            out['two_bright_stars_no_body'].append(cand(
                'two_bright_stars_no_body', fr['volset'], fr['volume'],
                fr['filespec'], 'COISS', fr['camera'], fr['dataset'],
                (fr['year'],),
                {'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'star_vmags': [round(v, 2) for v in vm[:5]],
                 'n_detectable': n_bright, 'filter': fr['filt'],
                 'year': fr['year']}))
        elif n_bright == 0 and fr['texp'] <= 3.0 and len(vm) == 0:
            out['negative_cases'].append(cand(
                'negative_cases', fr['volset'], fr['volume'],
                fr['filespec'], 'COISS', fr['camera'], fr['dataset'],
                ('COISS', 'empty_sky'),
                {'type': 'empty_sky', 'texp_s': fr['texp'],
                 'maglim': round(lim, 2), 'n_stars_within_2mag': len(vm),
                 'filter': fr['filt'], 'year': fr['year']}))

    for fr in sample(body_star_pool, 400):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim)
        if len(vm) >= 3:
            out['stars_plus_body'].append(cand(
                'stars_plus_body', fr['volset'], fr['volume'],
                fr['filespec'], 'COISS', fr['camera'], fr['dataset'],
                (fr['target'], phase_bin(fr['phase'])),
                {'target': fr['target'],
                 'apparent_diameter_px': round(fr['diam'], 1),
                 'n_detectable_stars': len(vm),
                 'star_vmags': [round(v, 2) for v in vm[:8]],
                 'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'center_phase_deg': fr['phase'], 'filter': fr['filt'],
                 'year': fr['year']}))

    for fr in sample(tiny_pool, 200):
        lim = maglim('COISS', fr['camera'], fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('COISS', fr['camera'])],
                        lim)
        if not vm:
            out['negative_cases'].append(cand(
                'negative_cases', fr['volset'], fr['volume'],
                fr['filespec'], 'COISS', fr['camera'], fr['dataset'],
                ('COISS', 'tiny_body'),
                {'type': 'tiny_body_no_stars', 'bodies_px': fr['bodies'],
                 'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'n_detectable_stars': 0, 'filter': fr['filt'],
                 'year': fr['year']}))



# ---------------------------------------------------------------- GO scan

def scan_go() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    volset, dataset = 'GO_0xxx', 'gossi'
    rng = random.Random(SEED + 1)
    pool: list[dict] = []
    stray_pool: list[dict] = []

    for volume in pm.volumes(volset):
        body = pm.load_table(volset, volume, 'body_summary')
        sky = pm.load_table(volset, volume, 'sky_summary')
        ring = pm.load_table(volset, volume, 'ring_summary')
        idx = pm.load_table(volset, volume, 'index')
        if not (idx and sky):
            continue
        inv = pm.load_inventory(volset, volume)

        body_rows: dict[str, list[list[str]]] = defaultdict(list)
        if body:
            for row in body.rows:
                body_rows[body.get(row, 'FILE_SPECIFICATION_NAME')].append(row)
        ring_specs = ({ring.get(r, 'FILE_SPECIFICATION_NAME')
                       for r in ring.rows} if ring else set())
        sky_rows = {sky.get(r, 'FILE_SPECIFICATION_NAME'): r
                    for r in sky.rows}

        for irow in idx.rows:
            filespec = (idx.get(irow, 'FILE_SPECIFICATION_NAME') or '').strip()
            if not filespec:
                continue
            stem = Path(filespec).stem
            if stem in EXISTING_IDS:
                continue
            texp_ms = idx.num(irow, 'EXPOSURE_DURATION')
            texp = texp_ms / 1000.0 if texp_ms else None
            filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
            tgt = (idx.get(irow, 'TARGET_NAME') or '').strip()
            time_full = idx.get(irow, 'IMAGE_TIME') or ''
            if not epoch_covered('GOSSI', time_full):
                continue
            year = time_full[:4]

            brows = body_rows.get(filespec, [])
            res_bodies = [b for b in brows if resolved(body, b)]
            skyrow = sky_rows.get(filespec)

            no_body = not res_bodies
            if (no_body and filespec not in ring_specs and skyrow is not None
                    and texp is not None and texp > 0):
                ra0 = sky.num(skyrow, 'MINIMUM_RIGHT_ASCENSION')
                ra1 = sky.num(skyrow, 'MAXIMUM_RIGHT_ASCENSION')
                de0 = sky.num(skyrow, 'MINIMUM_DECLINATION')
                de1 = sky.num(skyrow, 'MAXIMUM_DECLINATION')
                if None not in (ra0, ra1, de0, de1):
                    pool.append(dict(
                        filespec=filespec, volume=volume, texp=texp,
                        ra=(ra0 + ra1) / 2 if ra0 <= ra1 else ra1,
                        dec=(de0 + de1) / 2, filt=filt, year=year, tgt=tgt))

            # stray-light candidates: Earth/Moon/Venus in inventory but
            # not resolved on disc (bright body near/behind the frame)
            if (tgt in ('EARTH', 'MOON', 'VENUS')
                    and brows and not res_bodies
                    and texp is not None and texp >= 0.05):
                stray_pool.append(dict(
                    filespec=filespec, volume=volume, texp=texp,
                    filt=filt, year=year, tgt=tgt))

    pool = sorted(pool, key=lambda c: c['filespec'])
    rng.shuffle(pool)
    for fr in pool[:400]:
        lim = maglim('GOSSI', 'SSI', fr['texp'])
        vm = star_vmags(fr['ra'], fr['dec'], FOV_DEG[('GOSSI', 'SSI')],
                        lim + 2.0)
        n_bright = sum(1 for v in vm if v <= lim - 0.3)
        n_faint = sum(1 for v in vm if lim - 0.3 < v <= lim + 2.0)
        if n_bright == 0 and n_faint >= 3:
            out['faint_stars'].append(cand(
                'faint_stars', volset, fr['volume'], fr['filespec'],
                'GOSSI', 'SSI', dataset, (fr['year'],),
                {'texp_s': fr['texp'], 'maglim': round(lim, 2),
                 'n_bright': 0, 'n_faint_within_2mag': n_faint,
                 'faintest_usable_vmags':
                     [round(v, 2) for v in vm[:6]],
                 'target': fr['tgt'], 'filter': fr['filt'],
                 'year': fr['year']}))
        elif n_bright == 0 and n_faint == 0 and fr['texp'] <= 0.5:
            out['negative_cases'].append(cand(
                'negative_cases', volset, fr['volume'], fr['filespec'],
                'GOSSI', 'SSI', dataset, ('GOSSI', 'empty_sky'),
                {'type': 'empty_sky', 'texp_s': fr['texp'],
                 'maglim': round(lim, 2), 'n_stars_within_2mag': 0,
                 'target': fr['tgt'], 'filter': fr['filt'],
                 'year': fr['year']}))

    stray_pool = sorted(stray_pool, key=lambda c: c['filespec'])
    rng.shuffle(stray_pool)
    for fr in stray_pool[:20]:
        out['scattered_light'].append(cand(
            'scattered_light', volset, fr['volume'], fr['filespec'],
            'GOSSI', 'SSI', dataset, ('GOSSI', fr['year']),
            {'surrogate': 'bright body in inventory, unresolved on disc',
             'target': fr['tgt'], 'texp_s': fr['texp'],
             'filter': fr['filt'], 'year': fr['year']},
            needs_visual=True))

    return out


# ---------------------------------------------------------------- VGISS scan

def scan_vgiss() -> dict[str, list[dict]]:
    """Voyager: scattered-light surrogates (Saturn-encounter frames with
    nothing resolved) and short-exposure empty negatives.  The VGISS index
    has no RA/dec so star-count screens are not possible here."""
    out: dict[str, list[dict]] = defaultdict(list)
    rng = random.Random(SEED + 2)
    stray_pool: list[dict] = []
    neg_pool: list[dict] = []

    for volset in ('VGISS_6xxx',):
        for volume in pm.volumes(volset):
            moon = pm.load_table(volset, volume, 'moon_summary')
            sat = pm.load_table(volset, volume, 'saturn_summary')
            ring = pm.load_table(volset, volume, 'ring_summary')
            idx = pm.load_table(volset, volume, 'index')
            if not idx:
                continue

            moon_specs_resolved = set()
            if moon:
                for row in moon.rows:
                    if resolved(moon, row):
                        moon_specs_resolved.add(
                            moon.get(row, 'FILE_SPECIFICATION_NAME'))
            sat_specs_resolved = set()
            if sat:
                for row in sat.rows:
                    if resolved(sat, row):
                        sat_specs_resolved.add(
                            sat.get(row, 'FILE_SPECIFICATION_NAME'))
            ring_specs = ({ring.get(r, 'FILE_SPECIFICATION_NAME')
                           for r in ring.rows} if ring else set())

            for irow in idx.rows:
                product = (idx.get(irow, 'PRODUCT_TYPE') or '').strip()
                if product != 'CALIBRATED_IMAGE':
                    continue
                filespec = (idx.get(irow, 'FILE_SPECIFICATION_NAME')
                            or '').strip()
                # geometry tables reference the _RAW variant
                rawspec = filespec.replace('_CALIB', '_RAW')
                stem = Path(filespec).stem.split('_')[0]
                if stem in EXISTING_IDS:
                    continue
                texp_s = idx.num(irow, 'EXPOSURE_DURATION')
                filt = (idx.get(irow, 'FILTER_NAME') or '').strip()
                tgt = (idx.get(irow, 'TARGET_NAME') or '').strip()
                time_full = idx.get(irow, 'IMAGE_TIME') or ''
                if not epoch_covered('VGISS', time_full):
                    continue
                year = time_full[:4]
                inst = (idx.get(irow, 'INSTRUMENT_NAME') or '')
                camera = 'WA' if 'WIDE' in inst else 'NA'

                nothing_resolved = (rawspec not in moon_specs_resolved
                                    and rawspec not in sat_specs_resolved
                                    and rawspec not in ring_specs)
                if not nothing_resolved or texp_s is None:
                    continue
                rec = dict(filespec=filespec, volset=volset, volume=volume,
                           texp=texp_s, filt=filt, year=year, tgt=tgt,
                           camera=camera)
                if tgt == 'SATURN' and texp_s >= 1.0:
                    stray_pool.append(rec)
                elif texp_s <= 0.5 and tgt in ('DARK', 'SKY', 'STAR',
                                               'SATURN'):
                    neg_pool.append(rec)

    stray_pool = sorted(stray_pool, key=lambda c: c['filespec'])
    rng.shuffle(stray_pool)
    for fr in stray_pool[:20]:
        out['scattered_light'].append(cand(
            'scattered_light', fr['volset'], fr['volume'], fr['filespec'],
            'VGISS', fr['camera'], 'vgiss', ('VGISS', fr['year']),
            {'surrogate': 'Saturn-target frame, nothing resolved in FOV',
             'target': fr['tgt'], 'texp_s': fr['texp'],
             'filter': fr['filt'], 'year': fr['year']},
            needs_visual=True))

    neg_pool = sorted(neg_pool, key=lambda c: c['filespec'])
    rng.shuffle(neg_pool)
    for fr in neg_pool[:10]:
        out['negative_cases'].append(cand(
            'negative_cases', fr['volset'], fr['volume'], fr['filespec'],
            'VGISS', fr['camera'], 'vgiss', ('VGISS', 'empty_short_exp'),
            {'type': 'empty_short_exposure', 'target': fr['tgt'],
             'texp_s': fr['texp'], 'filter': fr['filt'],
             'year': fr['year']},
            needs_visual=True))

    return out


# ---------------------------------------------------------------- sampling

QUOTAS = {
    'body_irregular': 20,
    'ring_only_flat': 20,
    'ring_plus_body': 20,
    'stars_plus_body': 20,
    'two_bright_stars_no_body': 12,
    'faint_stars': 12,
    'scattered_light': 16,
    'negative_cases': 16,
}


def stratified_sample(cands: list[dict], n: int, rng: random.Random) -> list[dict]:
    strata: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        strata[c['strata']].append(c)
    for group in strata.values():
        group.sort(key=lambda c: c['filespec'])
        rng.shuffle(group)
    picked: list[dict] = []
    keys = sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def main() -> None:
    results: dict[str, list[dict]] = defaultdict(list)
    for scan in (scan_coiss, scan_go, scan_vgiss):
        for cls, cands in scan().items():
            results[cls].extend(cands)

    # a frame can satisfy only one class: scarce-first priority
    priority = ['two_bright_stars_no_body', 'faint_stars', 'scattered_light',
                'ring_only_flat', 'stars_plus_body', 'ring_plus_body',
                'body_irregular', 'negative_cases']
    seen: set[str] = set()
    for cls in priority:
        keep = []
        for c in results.get(cls, []):
            if c['filespec'] not in seen:
                seen.add(c['filespec'])
                keep.append(c)
        results[cls] = keep

    rng = random.Random(SEED + 3)
    manifest: dict = {
        'query': QUERY_NAME,
        'seed': SEED,
        'generated': '2026-07-08',
        'notes': 'Stage A candidates for the 8 empty scene classes '
                 '(plans/COHORT_CURATION_PLAN.md section 5 step 1). '
                 'Selection criteria recorded per candidate under '
                 '"selection".',
        'classes': {},
    }
    counts_lines = []
    for cls in sorted(QUOTAS):
        cands = results.get(cls, [])
        sample = stratified_sample(cands, QUOTAS[cls], rng)
        manifest['classes'][cls] = sample
        counts_lines.append(f'{cls}: {len(cands)} hits -> {len(sample)} sampled')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'candidates_batch001.yaml'
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=100))
    counts = '\n'.join(counts_lines)
    (OUT_DIR / 'scan_counts.txt').write_text(counts + '\n')
    print(counts)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
