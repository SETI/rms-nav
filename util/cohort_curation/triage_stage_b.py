"""Stage B automated triage (plans/COHORT_CURATION_PLAN.md section 4).

Runs the autonomous pipeline (sd_offset) on every Stage A candidate and
decides, without operator input, which frames are promoted to the operator
review batch:

- auto-drop: pipeline crash, majority-missing frames, saturated-bloom
  frames, and navigable-class frames the feasibility gates reject
- promote: frames with a proposed offset that is self-consistent by machine
  checks (technique agreement, finite sigma)
- negative_cases: promoted when the pipeline FAILS cleanly (failure is the
  desired behavior; the PNG lets the operator confirm unnavigability)
- needs_visual frames (scattered_light surrogates): promoted whenever the
  pipeline produced a summary PNG at all
- star-class frames that fail are promoted flagged=manual_nav_queue (the
  calibrated-CISS star gate is known-uncalibrated; PHASE10 workflow B)

Writes triage_report.yaml next to the manifest.

Run:  venv/bin/python util/cohort_curation/triage_stage_b.py [--limit N]

Already-triaged frames (a *_metadata.json exists under triage_results/) are
reused without re-running the pipeline; pass --force to re-run them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'   # generated outputs (gitignored)
RESULTS_ROOT = OUT_DIR / 'triage_results'
VENV_BIN = REPO / 'venv/bin'

ENV = {
    'PDS3_HOLDINGS_DIR': '/mnt/ganymede/PDS/holdings',
    'OOPS_RESOURCES': '/home/rfrench/DS/Shared/OOPS-Resources',
    'UCAC4_PATH': '/data/external-data/star-catalogs/UCAC4',
    'YBSC_PATH': '/data/external-data/star-catalogs/YBSC',
    'PATH': '/usr/bin:/bin',
    'HOME': str(Path.home()),
}

STAR_CLASSES = {'faint_stars', 'two_bright_stars_no_body', 'stars_plus_body'}

TIMEOUT_S = 1200        # Galileo/Voyager camera-rotation fits are slow


def image_name_for(candidate: dict) -> str:
    stem = Path(candidate['filespec']).stem
    if candidate['mission'] in ('VGISS', 'COISS'):
        # The dataset index carries names without the product/version
        # suffix (C2783018, N1828132857); startswith-matching needs the
        # bare name.
        return stem.split('_')[0]
    if candidate['mission'] == 'NHLORRI':
        # lor_0003103486_0x630_sci -> LOR_0003103486
        return stem.upper()[:14]
    return stem


def run_one(candidate: dict) -> dict:
    name = image_name_for(candidate)
    cmd = [str(VENV_BIN / 'sd_offset'), candidate['dataset'], name,
           '--nav-results-root', str(RESULTS_ROOT)]
    rec: dict = {'image_name': name, **{k: candidate[k] for k in
                 ('scene_class', 'filespec', 'volume', 'mission', 'camera',
                  'dataset', 'strata', 'needs_visual', 'selection')}}
    if candidate.get('_skip'):
        rec.update(exit_code=None,
                   log_tail=['(skipped via --skip-names)'],
                   triage='dropped',
                   triage_reason='skipped: navigation exhausts memory '
                                 '(OOM killer); needs pipeline investigation')
        return rec
    if not candidate.get('_force') and _result_files(name, '_metadata.json'):
        rec['exit_code'] = 0
        rec['log_tail'] = ['(reused existing triage result)']
        return _evaluate(candidate, rec)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_S, env=ENV, cwd=REPO)
        rec['exit_code'] = proc.returncode
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
        rec['log_tail'] = tail
    except subprocess.TimeoutExpired:
        rec.update(exit_code=None, log_tail=['TIMEOUT'],
                   triage='dropped', triage_reason='timeout')
        return rec

    return _evaluate(candidate, rec)


def _result_files(name: str, suffix: str) -> list:
    files = sorted(RESULTS_ROOT.rglob(f'{name}*{suffix}'))
    if not files and name != name.lower():
        files = sorted(RESULTS_ROOT.rglob(f'{name.lower()}*{suffix}'))
    return files


def _evaluate(candidate: dict, rec: dict) -> dict:
    name = rec['image_name']
    meta_files = _result_files(name, '_metadata.json')
    png_files = _result_files(name, '_summary.png')
    if not meta_files:
        rec.update(triage='dropped',
                   triage_reason='no metadata written (pipeline error)')
        return rec
    meta = json.loads(meta_files[0].read_text())
    rec['metadata_path'] = str(meta_files[0])
    rec['summary_png'] = str(png_files[0]) if png_files else None

    nav = meta.get('navigation_result') or {}
    classifier = nav.get('image_classifier') or {}
    status = meta.get('status')
    rec['status'] = status
    rec['status_reason'] = nav.get('status_reason')
    rec['offset_px'] = nav.get('offset_px')
    rec['sigma_px'] = nav.get('sigma_px')
    rec['confidence'] = nav.get('confidence')
    rec['confidence_rank'] = nav.get('confidence_rank')
    rec['techniques_used'] = nav.get('techniques_used')
    per_tech = nav.get('per_technique') or []
    rec['per_technique_offsets'] = {
        t.get('technique_name'): t.get('offset_px') for t in per_tech}

    # --- machine checks -----------------------------------------------
    warnings: list[str] = []
    missing = classifier.get('missing_frac') or 0.0
    saturated = classifier.get('saturation_frac') or 0.0
    if missing > 0.5:
        rec.update(triage='dropped',
                   triage_reason=f'majority missing data ({missing:.0%})')
        return rec
    if saturated > 0.3:
        rec.update(triage='dropped',
                   triage_reason=f'saturated bloom ({saturated:.0%})')
        return rec

    # Infrastructure errors (missing SPICE coverage, bad file, ...) are
    # not navigation outcomes: the frame proves nothing as a negative
    # case and cannot be manually navigated (no model to align), so it
    # is unusable for any class.
    if status == 'error':
        rec['status_error'] = meta.get('status_error')
        rec.update(triage='dropped',
                   triage_reason='infrastructure error, not a navigation '
                                 f'outcome: {rec["status_error"]}')
        return rec

    cls = candidate['scene_class']
    ok = status == 'success' and rec['offset_px'] is not None

    if cls == 'negative_cases':
        if not ok:
            rec.update(triage='promoted',
                       triage_reason='pipeline failed as desired for a '
                                     'negative case')
        else:
            rec.update(triage='promoted',
                       triage_reason='UNEXPECTED success on negative '
                                     'candidate; operator decides',)
            warnings.append('negative candidate navigated: '
                            f'offset={rec["offset_px"]}')
        rec['triage_warnings'] = warnings
        return rec

    if not ok:
        if candidate['needs_visual'] or cls in STAR_CLASSES:
            rec.update(triage='promoted',
                       triage_reason='failed but class is visual/star '
                                     '(gate uncalibrated); flagged for '
                                     'manual nav')
            warnings.append('manual_nav_queue')
        else:
            rec.update(triage='dropped',
                       triage_reason=f'not navigable: {status} / '
                                     f'{rec["status_reason"]}')
        rec['triage_warnings'] = warnings
        return rec

    # technique agreement: max pairwise offset spread
    offsets = [o for o in rec['per_technique_offsets'].values() if o]
    if len(offsets) >= 2:
        dvs = [o[0] for o in offsets]
        dus = [o[1] for o in offsets]
        spread = max(max(dvs) - min(dvs), max(dus) - min(dus))
        rec['technique_spread_px'] = round(spread, 2)
        if spread > 10.0:
            warnings.append(f'technique disagreement {spread:.1f} px')

    if classifier.get('flags'):
        warnings.append(f'classifier flags: {classifier["flags"]}')

    rec.update(triage='promoted', triage_reason='proposed offset produced',
               triage_warnings=warnings)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='triage only the first N candidates (debug)')
    ap.add_argument('--classes', default='',
                    help='comma-separated subset of scene classes')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--force', action='store_true',
                    help='re-run frames that already have triage results')
    ap.add_argument('--skip-names', default='',
                    help='comma-separated image names to mark as dropped '
                         'without running (e.g. known memory-exhausting '
                         'frames)')
    ap.add_argument('--batch', type=int, default=1,
                    help='batch number (selects manifest and report names)')
    args = ap.parse_args()

    manifest = yaml.safe_load(
        (OUT_DIR / f'candidates_batch{args.batch:03d}.yaml').read_text())
    cands = [c for cls, group in manifest['classes'].items() for c in group]
    if args.classes:
        wanted = set(args.classes.split(','))
        cands = [c for c in cands if c['scene_class'] in wanted]
    if args.limit:
        cands = cands[:args.limit]
    skip_names = {s for s in args.skip_names.split(',') if s}
    for c in cands:
        c['_force'] = args.force
        c['_skip'] = image_name_for(c) in skip_names

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f'triaging {len(cands)} candidates with {args.workers} workers',
          flush=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_one, cands), 1):
            results.append(rec)
            print(f'[{i}/{len(cands)}] {rec["image_name"]} '
                  f'({rec["scene_class"]}): {rec.get("triage")} - '
                  f'{rec.get("triage_reason")}', flush=True)

    promoted = [r for r in results if r.get('triage') == 'promoted']
    report = {
        'manifest': f'candidates_batch{args.batch:03d}.yaml',
        'n_triaged': len(results),
        'n_promoted': len(promoted),
        'results': results,
    }
    report_name = ('triage_report.yaml' if args.batch == 1
                   else f'triage_report_batch{args.batch:03d}.yaml')
    (OUT_DIR / report_name).write_text(
        yaml.safe_dump(report, sort_keys=False, width=100))
    print(f'\npromoted {len(promoted)}/{len(results)}; '
          f'wrote {report_name}')


if __name__ == '__main__':
    main()
