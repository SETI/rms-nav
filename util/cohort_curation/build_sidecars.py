"""Stage D: image-library sidecars from operator review votes.

Consumes a voted votes.yaml (Stage C) plus the Stage B triage report and
per-image navigation metadata, and emits one sidecar YAML + companion
overlay PNG per y-voted frame under
tests/integration/image_library/images/<class>/.

Vote handling:

- ``y`` on a navigable class: sidecar with the pipeline's proposed
  offset as operator-verified ground truth (the operator confirmed the
  overlay alignment during review).
- ``y`` on negative_cases: expected.status=failed sidecar; the
  ground-truth offset is a documented placeholder (the regression test
  only compares offsets for expected.status=success).
- ``y`` on ring_only_flat: deferred (rank-1 ground truth unsupported;
  see issues #203 / #204) and listed in the follow-ups file.
- ``m`` votes and reclassify comments: collected into
  _work/cohort_curation/batch_NNN_followups.yaml (manual-nav queue).

Run:  venv/bin/python util/cohort_curation/build_sidecars.py --batch 1
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import yaml
from vicar import VicarImage

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'
LIBRARY = REPO / 'tests/integration/image_library/images'
HOLDINGS_PREFIX = '/mnt/ganymede/PDS/holdings/'

DEFERRED_CLASSES = {'ring_only_flat'}   # rank-1 GT unsupported (#203/#204)

# ground_truth.offset_uncertainty_px per class (PHASE10 rubric: 1.0 for
# sharp limbs / bright stars, 2.0 for soft features or star-poor).
UNCERTAINTY = {
    'body_irregular': 2.0,
    'ring_plus_body': 1.0,
    'stars_plus_body': 1.0,
}

# expected.confidence_tier the *calibrated* pipeline ought to reach
# ("if you'd be unsurprised either way, write medium").
TIER = {
    'body_irregular': 'medium',
    'ring_plus_body': 'medium',
    'stars_plus_body': 'medium',
}

# Single-mode classes take the rubric's quick-map technique; ensemble
# classes pick the highest-confidence non-spurious technique instead.
PRIMARY_MAP = {'body_irregular': 'BodyBlobNav'}
ENSEMBLE_CLASSES = {'ring_plus_body', 'stars_plus_body'}


def str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    if '\n' in data:
        return dumper.represent_scalar(
            'tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(str, str_representer)


def filter_combo(image_path: str) -> str:
    """Canonical filter combo (sorted, '+'-joined) from the image label.

    Parameters:
        image_path: Path to the VICAR image whose label carries
            FILTER_NAME (a string or a list of strings).
    """
    lab = VicarImage.from_file(image_path, strict=False).label
    try:
        raw = lab['FILTER_NAME']
    except KeyError:
        return 'UNKNOWN'
    filters = list(raw) if isinstance(raw, (list, tuple)) else [str(raw)]
    return '+'.join(sorted(str(f).strip() for f in filters if str(f).strip()))


def primary_technique(rec: dict, meta: dict) -> str:
    """Expected pass-1 winner for one frame per the PHASE10 rubric.

    Parameters:
        rec: Triage-report record (scene_class).
        meta: Navigation metadata JSON for the frame.
    """
    cls = rec['scene_class']
    if cls in PRIMARY_MAP:
        return PRIMARY_MAP[cls]
    per_tech = (meta.get('navigation_result') or {}).get('per_technique') or []
    live = [t for t in per_tech
            if not t.get('spurious') and (t.get('confidence') or 0) > 0]
    if live:
        # Rubric tie-break: (-confidence, technique_name) ascending.
        live.sort(key=lambda t: (-(t.get('confidence') or 0),
                                 t.get('technique_name') or ''))
        return str(live[0]['technique_name'])
    if cls == 'stars_plus_body':
        return 'StarFieldFromCatalogNav'
    return 'RingEdgeNav'


def scene_tag_secondary(rec: dict, meta: dict) -> str | None:
    """Body / morphology tag to accompany the class tag."""
    target = ((rec.get('selection') or {}).get('target')
              or (rec.get('selection') or {}).get('tgt') or '')
    if target and target not in ('DARK', 'SKY'):
        return str(target).lower()
    sel_type = (rec.get('selection') or {}).get('type') or ''
    if sel_type:
        return str(sel_type)
    return None


def build_one(entry: dict, rec: dict, ui_version: str,
              verified_date: datetime.date) -> Path:
    """Write the sidecar + companion PNG for one y-voted frame.

    Parameters:
        entry: votes.yaml entry (seq, image_name, vote, comment).
        rec: Triage-report record for the same frame.
        ui_version: ground_truth.ui_version string.
        verified_date: Date of the operator's review pass.
    """
    cls = rec['scene_class']
    meta = json.loads(Path(rec['metadata_path']).read_text())
    image_path = meta['observation']['image_path']
    image_id = Path(image_path).stem
    negative = cls == 'negative_cases'

    url = image_path
    if url.startswith(HOLDINGS_PREFIX):
        url = 'pds3://' + url[len(HOLDINGS_PREFIX):]

    tags = [cls]
    sec = scene_tag_secondary(rec, meta)
    if sec:
        tags.append(sec)

    comment = (entry.get('comment') or '').strip()
    streaked = 'streak' in comment.lower()

    note_lines = [
        f'{cls} exemplar from cohort-curation review batch '
        f'{entry["_batch"]:03d} (seq {entry["seq"]:03d}).',
        f'Selection: {rec.get("strata")}; '
        f'{json.dumps(rec.get("selection") or {}, sort_keys=True)}.',
    ]
    if negative:
        note_lines += [
            'Operator confirmed nothing navigable at review; the pipeline '
            f'failed as required (status={rec.get("status")}, '
            f'reason={rec.get("status_reason")}).',
            'Ground-truth offset is a placeholder: the scene is '
            'unnavigable by design, so no offset is measurable; the '
            'regression assertion for this sidecar is expected.status='
            'failed (offsets are only compared for status=success).',
        ]
    else:
        note_lines += [
            'Operator verified the pipeline overlay alignment at the '
            'proposed offset during batch review (summary PNG plus, for '
            'star classes, a hard-stretch star-check PNG with circled '
            'catalog positions).',
            f'Pipeline: status={rec.get("status")}, '
            f'confidence={rec.get("confidence")}, '
            f'rank={rec.get("confidence_rank")}, '
            f'techniques={rec.get("techniques_used")}, '
            f'per-technique offsets='
            f'{json.dumps(rec.get("per_technique_offsets") or {})}.',
        ]
    if comment:
        note_lines.append(f'Operator comment: {comment}')

    if negative:
        gt_offset = (0.0, 0.0)
        gt_unc = 99.0
    else:
        gt_offset = tuple(rec['offset_px'])
        gt_unc = UNCERTAINTY.get(cls, 2.0)
        if streaked:
            gt_unc = max(gt_unc, 2.0)

    if negative:
        expected = {
            'status': 'failed',
            'confidence_tier': 'failed',
            'primary_technique': primary_technique(rec, meta),
            'techniques_must_run': [],
            'techniques_must_skip': [],
        }
    else:
        prim = primary_technique(rec, meta)
        expected = {
            'status': 'success',
            'confidence_tier': TIER.get(cls, 'medium'),
            'primary_technique': prim,
            'techniques_must_run': [prim],
            'techniques_must_skip': [],
        }

    sidecar = {
        'schema_version': 1,
        'image_id': image_id,
        'mission': rec['mission'],
        'camera': rec['camera'],
        'filter_combo': filter_combo(image_path),
        'image_url': url,
        'scene_tags': tags,
        'ground_truth': {
            'offset_dv_px': round(float(gt_offset[0]), 4),
            'offset_du_px': round(float(gt_offset[1]), 4),
            'offset_uncertainty_px': gt_unc,
            'source': 'operator_verified',
            'operator': 'rfrench',
            'verified_date': verified_date,
            'ui_version': ui_version,
            'notes': '\n'.join(note_lines) + '\n',
        },
        'expected': expected,
    }

    out_dir = LIBRARY / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{image_id}.yaml'
    out_path.write_text(yaml.dump(sidecar, sort_keys=False, width=78))
    png = rec.get('summary_png')
    if png and Path(png).exists():
        shutil.copyfile(png, out_dir / f'{image_id}.png')
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, required=True)
    args = ap.parse_args()

    batch_dir = REPO / '_work/cohort_review' / f'batch_{args.batch:03d}'
    votes = yaml.safe_load((batch_dir / 'votes.yaml').read_text())
    report = yaml.safe_load((OUT_DIR / 'triage_report.yaml').read_text())
    byname = {r['image_name']: r for r in report['results']}

    try:
        ui_version = f'spindoctor {version("rms-spindoctor")}'
    except Exception:
        ui_version = 'spindoctor (dev)'
    ui_version += f' (cohort review batch {args.batch:03d})'
    verified_date = datetime.date(2026, 7, 9)

    written: list[str] = []
    deferred: list[dict] = []
    manual_queue: list[dict] = []
    reclassify: list[dict] = []

    for entry in votes['images']:
        entry['_batch'] = args.batch
        rec = byname.get(entry['image_name'])
        vote = entry.get('vote')
        comment = (entry.get('comment') or '').strip()
        if comment.lower().startswith('reclassify:'):
            reclassify.append({'image_name': entry['image_name'],
                               'from': entry['scene_class'],
                               'to': comment.split(':', 1)[1].strip(),
                               'seq': entry['seq']})
        if vote == 'm':
            manual_queue.append({'image_name': entry['image_name'],
                                 'scene_class': entry['scene_class'],
                                 'seq': entry['seq'],
                                 'comment': comment or None})
            continue
        if vote != 'y' or not rec:
            continue
        if entry['scene_class'] in DEFERRED_CLASSES:
            deferred.append({'image_name': entry['image_name'],
                             'scene_class': entry['scene_class'],
                             'seq': entry['seq'],
                             'reason': 'rank-1 ground truth unsupported '
                                       '(#203/#204)'})
            continue
        out = build_one(entry, rec, ui_version, verified_date)
        written.append(str(out.relative_to(REPO)))
        print(f'wrote {out.relative_to(REPO)}')

    followups = {
        'batch': args.batch,
        'manual_nav_queue': manual_queue,
        'deferred_rank1': deferred,
        'reclassify': reclassify,
    }
    fu_path = OUT_DIR / f'batch_{args.batch:03d}_followups.yaml'
    fu_path.write_text(yaml.safe_dump(followups, sort_keys=False))
    print(f'\n{len(written)} sidecars; followups -> {fu_path}')
    print(f'manual queue {len(manual_queue)}, deferred {len(deferred)}, '
          f'reclassify {len(reclassify)}')


if __name__ == '__main__':
    main()
