"""Stage C review-batch builder (plans/COHORT_CURATION_PLAN.md section 4).

Takes the promoted frames from triage_report.yaml and produces
_work/cohort_review/batch_NNN/ with, per image:

- NNN_<image_name>.png -- the pipeline summary PNG (model overlay at the
  proposed offset) with a text margin burned in: proposed (dv, du), scene
  class, confidence, and any triage warnings
- votes.yaml -- pre-filled with vote: null / comment: '' per image

The operator's entire job: look at each PNG, set vote y/n (+ optional
comment), hand the file back.

Run:  venv/bin/python util/cohort_curation/build_review_batch.py --batch 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT_DIR = REPO / '_work/cohort_curation'  # generated outputs (gitignored)

VOTE_INSTRUCTIONS = (
    'Set vote to y, m, or n per image (optional comment). '
    'y = good example of the class AND the overlay is aligned at the '
    'proposed offset (for negative_cases: good example that correctly '
    'has nothing to navigate). '
    'm = good example of the class, but the proposed offset is missing '
    'or misaligned; frame is kept and routed to the manual-nav queue or '
    'curated as an expected-failure. '
    'n = not a good example of the class, or unusable image; discard. '
    'ring_only_flat: vote on class membership only (y for a clean '
    'straight edge); a straight edge is rank-1 and cannot be fully '
    'navigated, so no alignment judgment is expected.'
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A monospace font at the given size, falling back to PIL's default.

    Parameters:
        size: Point size for the TrueType candidates.
    """
    for name in ('DejaVuSansMono.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = load_font(16)
FONT_SMALL = load_font(13)


def compose(rec: dict, out_path: Path, seq: int) -> None:
    """Write one review PNG: summary image plus a burned-in text margin.

    Parameters:
        rec: Triage record (summary_png, offset, confidence, warnings).
        out_path: Destination PNG path.
        seq: 1-based sequence number within the review batch.
    """
    src = rec.get('summary_png')
    if src and Path(src).exists():
        img = Image.open(src).convert('RGB')
    else:
        img = Image.new('RGB', (1024, 1024), (30, 30, 30))
        d = ImageDraw.Draw(img)
        d.text((40, 480), 'NO SUMMARY PNG (pipeline failed)', fill=(255, 80, 80), font=FONT)

    margin = 96
    canvas = Image.new('RGB', (img.width, img.height + margin), (12, 12, 12))
    canvas.paste(img, (0, margin))
    d = ImageDraw.Draw(canvas)

    off = rec.get('offset_px')
    off_txt = (
        f'proposed (dv, du) = ({off[0]:+.2f}, {off[1]:+.2f}) px' if off else 'NO PROPOSED OFFSET'
    )
    conf = rec.get('confidence')
    conf_txt = f'conf {conf:.2f} ({rec.get("confidence_rank")})' if conf is not None else 'conf n/a'
    line1 = (
        f'#{seq:03d}  {rec["image_name"]}  [{rec["scene_class"]}]  {rec["mission"]}/{rec["camera"]}'
    )
    line2 = f'{off_txt}   {conf_txt}   status={rec.get("status")}'
    warn = '; '.join(rec.get('triage_warnings') or [])
    if rec.get('needs_visual'):
        warn = (
            'VISUAL CHECK (class needs eyeball confirmation); ' + warn
            if warn
            else 'VISUAL CHECK (class needs eyeball confirmation)'
        )
    d.text((10, 8), line1, fill=(255, 255, 255), font=FONT)
    d.text((10, 34), line2, fill=(200, 200, 200), font=FONT)
    if warn:
        d.text((10, 60), f'! {warn}'[:130], fill=(255, 200, 80), font=FONT_SMALL)
    canvas.save(out_path)


# Review-batch size cap is 100 (operator, 2026-07-08); per-class caps keep
# every class represented with headroom over the Phase-10 minima.
CLASS_CAPS = {
    'body_irregular': 15,
    'faint_stars': 10,
    'negative_cases': 10,
    'ring_only_flat': 12,
    'ring_plus_body': 15,
    'scattered_light': 12,
    'stars_plus_body': 15,
    'two_bright_stars_no_body': 10,
    'body_full_fov': 8,
    'body_partial_overflow': 8,
    'body_mostly_offscreen': 6,
    'multi_body': 8,
    'high_phase_terminator': 6,
    'below_resolution_body': 8,
    'ring_only_curved': 6,
    'star_dominated': 8,
    'one_bright_star_no_body': 8,
}

# Classes whose triage-dropped frames are rescued into the manual-nav queue
# when the scene itself is right but the pipeline failed wholesale (the
# operator's ground truth does not depend on the pipeline).
RESCUE_CLASSES = {'ring_only_flat'}


def select(promoted: list[dict]) -> list[dict]:
    """Per-class stratified pick: prefer frames with offsets and low
    technique spread, then round-robin across strata."""
    rng = random.Random(20260708)
    picked: list[dict] = []
    by_class: dict[str, list[dict]] = {}
    for r in promoted:
        by_class.setdefault(r['scene_class'], []).append(r)
    for cls, group in sorted(by_class.items()):
        cap = CLASS_CAPS.get(cls, 4)
        strata: dict[str, list[dict]] = {}
        for r in group:
            strata.setdefault(r.get('strata', ''), []).append(r)
        for s in strata.values():
            s.sort(key=lambda r: (r.get('offset_px') is None, r.get('technique_spread_px') or 0.0))
        chosen: list[dict] = []
        keys = sorted(strata)
        rng.shuffle(keys)
        while len(chosen) < cap and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(chosen) < cap:
                    chosen.append(strata[k].pop(0))
        picked.extend(chosen)
    return picked


def main() -> None:
    """Build the review batch directory (PNGs + votes.yaml) for one batch."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=1)
    args = ap.parse_args()

    report_name = (
        'triage_report.yaml' if args.batch == 1 else f'triage_report_batch{args.batch:03d}.yaml'
    )
    report = yaml.safe_load((OUT_DIR / report_name).read_text())
    promoted = [r for r in report['results'] if r.get('triage') == 'promoted']
    for r in report['results']:
        if (
            r.get('triage') == 'dropped'
            and r['scene_class'] in RESCUE_CLASSES
            and r.get('summary_png')
        ):
            r['triage_warnings'] = (r.get('triage_warnings') or []) + [
                'manual_nav_queue',
                f'rescued: pipeline said {r.get("status_reason")}',
            ]
            promoted.append(r)
    promoted = select(promoted)
    promoted.sort(key=lambda r: (r['scene_class'], r['image_name']))

    batch_dir = REPO / '_work/cohort_review' / f'batch_{args.batch:03d}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    votes = []
    for seq, rec in enumerate(promoted, 1):
        png_name = f'{seq:03d}_{rec["image_name"]}.png'
        compose(rec, batch_dir / png_name, seq)
        votes.append(
            {
                'seq': seq,
                'image_name': rec['image_name'],
                'scene_class': rec['scene_class'],
                'png': png_name,
                'proposed_offset_dv_du_px': rec.get('offset_px'),
                'warnings': rec.get('triage_warnings') or [],
                'vote': None,
                'comment': '',
            }
        )

    (batch_dir / 'votes.yaml').write_text(
        yaml.safe_dump(
            {'batch': args.batch, 'instructions': VOTE_INSTRUCTIONS, 'images': votes},
            sort_keys=False,
            width=100,
        )
    )
    print(f'wrote {batch_dir}/votes.yaml with {len(votes)} images')


if __name__ == '__main__':
    main()
