"""Plausibility cross-check: calibrated pipeline vs operator library tiers.

Runs the autonomous pipeline over every operator-curated sidecar in
``tests/integration/image_library/images/`` (same plumbing as
``test_autonomous_nav``) and reports each comparison INDEPENDENTLY per
image -- status match, tier match, offset agreement within the sidecar's
slack, primary technique, must-run / must-skip -- instead of stopping at
the first failed assertion the way the test does.  The operator tiers are
plausibility cross-checks on the calibration, never fit targets; a
wholesale mismatch means either the labels or the calibration needs
review.

Needs the local-holdings environment (``source /seti/newnav/setup.sh``).

Run:

    venv/bin/python util/calibration/library_crosscheck.py \
        --workers 6 --out _work/calibration/library_crosscheck.md
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))


def _check_one(sidecar_path_str: str) -> dict[str, Any]:
    """Worker: navigate one library image and compare against its sidecar."""
    import os

    from filecache import FCPath
    from tests.integration.sidecar import load_sidecar

    from spindoctor.dataset.dataset import ImageFile, ImageFiles
    from spindoctor.navigate_image_files import navigate_image_files
    from spindoctor.obs import (
        ObsCassiniISS,
        ObsGalileoSSI,
        ObsNewHorizonsLORRI,
        ObsVoyagerISS,
    )

    mission_to_obs = {
        'COISS': ObsCassiniISS,
        'VGISS': ObsVoyagerISS,
        'GOSSI': ObsGalileoSSI,
        'NHLORRI': ObsNewHorizonsLORRI,
    }
    sidecar = load_sidecar(Path(sidecar_path_str))
    row: dict[str, Any] = {
        'image_id': sidecar.image_id,
        'scene_class': sidecar.scene_tags[0] if sidecar.scene_tags else '?',
        'expected_status': sidecar.expected.status,
        'expected_tier': sidecar.expected.confidence_tier,
    }
    url = sidecar.image_url
    if url.startswith('pds3://'):
        holdings_root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
        url = f'{holdings_root}/{url[len("pds3://") :]}'
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(url),
                label_file_url=FCPath(url),
                results_path_stub=sidecar.image_id,
            )
        ]
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _success, metadata = navigate_image_files(
                mission_to_obs[sidecar.mission],
                image_files,
                FCPath(tmp),
                write_output_files=False,
            )
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row
    nav_meta = metadata.get('navigation_result') or {}
    row['actual_status'] = nav_meta.get('status') or metadata.get('status')
    row['actual_tier'] = nav_meta.get('confidence_rank')
    row['confidence'] = nav_meta.get('confidence')
    row['status_ok'] = row['actual_status'] == sidecar.expected.status
    row['tier_ok'] = row['actual_tier'] == sidecar.expected.confidence_tier
    offset = metadata.get('offset')
    if sidecar.expected.status == 'success' and offset is not None:
        slack = sidecar.ground_truth.offset_uncertainty_px + 0.5
        dv_err = abs(float(offset[0]) - sidecar.ground_truth.offset_dv_px)
        du_err = abs(float(offset[1]) - sidecar.ground_truth.offset_du_px)
        row['offset_err_px'] = round(max(dv_err, du_err), 3)
        row['offset_ok'] = dv_err <= slack and du_err <= slack
    elif sidecar.expected.status == 'success':
        row['offset_err_px'] = None
        row['offset_ok'] = False
    else:
        row['offset_err_px'] = None
        row['offset_ok'] = None
    per_technique = nav_meta.get('per_technique', [])
    names = [entry.get('technique_name') for entry in per_technique]
    # The primary is the highest-confidence NON-spurious technique: a
    # spurious result is excluded from the ensemble, so it cannot be the
    # primary even when its raw confidence is high (e.g. a rank-1 RingEdgeNav
    # flagged spurious on a ring_plus_body frame).
    non_spurious = [e for e in per_technique if not e.get('spurious', False)]
    if sidecar.expected.status == 'success' and non_spurious:
        ordered = sorted(
            non_spurious,
            key=lambda entry: (
                -float(entry.get('confidence', 0.0)),
                str(entry.get('technique_name')),
            ),
        )
        row['primary_ok'] = ordered[0].get('technique_name') == sidecar.expected.primary_technique
        row['actual_primary'] = ordered[0].get('technique_name')
    else:
        row['primary_ok'] = None
        row['actual_primary'] = None
    row['must_run_ok'] = all(n in names for n in sidecar.expected.techniques_must_run)
    row['must_skip_ok'] = all(n not in names for n in sidecar.expected.techniques_must_skip)
    return row


def _init_worker() -> None:
    """Pin threads and silence loggers (same rationale as collect.py)."""
    import os

    for var in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[var] = '1'
    import pdslogger

    from spindoctor.config.logger import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)


def main(argv: list[str] | None = None) -> int:
    """Cross-check every sidecar and write the per-image comparison table."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument(
        '--out', type=Path, default=REPO / '_work/calibration/library_crosscheck.md'
    )
    args = parser.parse_args(argv)

    # Workers expand pds3:// sidecar URLs against this root; failing here
    # gives one clear startup error instead of a KeyError in every worker.
    if not os.environ.get('PDS3_HOLDINGS_DIR'):
        parser.error('PDS3_HOLDINGS_DIR must be set (root of the PDS3 holdings tree)')

    from tests.integration.sidecar import LibraryRoot

    paths = [str(p) for p in LibraryRoot().discover_sidecar_paths()]
    print(f'{len(paths)} sidecars')
    rows: list[dict[str, Any]] = []
    with multiprocessing.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for row in pool.imap_unordered(_check_one, paths):
            rows.append(row)
            print(f'{len(rows)}/{len(paths)} {row["image_id"]}', flush=True)
    rows.sort(key=lambda r: (r['scene_class'], r['image_id']))

    def frac(key: str) -> str:
        relevant = [r for r in rows if r.get(key) is not None and 'error' not in r]
        if not relevant:
            return 'n/a'
        return f'{sum(1 for r in relevant if r[key])}/{len(relevant)}'

    lines = [
        '# Library cross-check (calibrated pipeline vs operator sidecars)',
        '',
        f'{len(rows)} sidecars.  Independent per-check agreement:',
        '',
        f'- status: {frac("status_ok")}',
        f'- confidence tier: {frac("tier_ok")}',
        f'- offset within slack: {frac("offset_ok")}',
        f'- primary technique: {frac("primary_ok")}',
        f'- must_run: {frac("must_run_ok")}   must_skip: {frac("must_skip_ok")}',
        f'- pipeline exceptions: {sum(1 for r in rows if "error" in r)}',
        '',
        '| image | class | exp status/tier | act status/tier | conf | off err | off ok |'
        ' primary ok |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        if 'error' in r:
            lines.append(
                f'| {r["image_id"]} | {r["scene_class"]} | ERROR: {r["error"][:60]} | | | | | |'
            )
            continue
        conf = f'{r["confidence"]:.2f}' if r.get('confidence') is not None else '-'
        lines.append(
            f'| {r["image_id"]} | {r["scene_class"]} '
            f'| {r["expected_status"]}/{r["expected_tier"]} '
            f'| {r["actual_status"]}/{r["actual_tier"]} | {conf} '
            f'| {r["offset_err_px"] if r["offset_err_px"] is not None else "-"} '
            f'| {_mark(r["offset_ok"])} | {_mark(r["primary_ok"])} |'
        )
    lines.append('')
    tier_pairs = Counter(
        (r['expected_tier'], r.get('actual_tier')) for r in rows if 'error' not in r
    )
    lines += ['## Tier confusion (expected -> actual)', '']
    for (exp, act), n in sorted(tier_pairs.items(), key=lambda kv: -kv[1]):
        lines.append(f'- {exp} -> {act}: {n}')
    lines.append('')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines))
    print(f'Wrote {args.out}')
    return 0


def _mark(value: bool | None) -> str:
    """Render a tri-state check as a table cell."""
    if value is None:
        return '-'
    return 'yes' if value else 'NO'


if __name__ == '__main__':
    raise SystemExit(main())
