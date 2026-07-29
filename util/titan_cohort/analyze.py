"""Report what a cohort run proved, against the validation plan's four bounds.

Consumes the ``rows.jsonl`` written by ``collect.py`` and prints:

- the outcome of every frame, grouped by the flags read off its legacy
  annotation, so a refusal can be read against what the annotation warned of;
- **evidence tier (a)**, star-anchored agreement: on a frame where a
  prior-free star technique also locked, the star offset and the haze offset
  are two independent measurements of one scene-wide translation, and the
  per-axis 2-sigma test between them is the strongest per-frame truth
  available without an operator eyeball;
- **evidence tier (b)**, within-sequence consistency: clean frames of the
  same target within 30 minutes;
- **evidence tier (c)**, cross-filter consistency: near-simultaneous frames
  (10 minutes) through different filters, which is the direct test of the
  method's filter-independence claim;
- a **companion-body witness** channel, reported separately and NOT counted
  in the acceptance fractions: a body technique locking on another moon in
  the same frame is the same physics as the star anchor, but the validation
  plan states its bound over star anchors, so the two are kept apart.

Run::

    python util/titan_cohort/analyze.py _work/titan_cohort/run1/rows.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from cohort import ADVERSE_FLAGS  # noqa: E402

TECHNIQUE = 'TitanHazeNav'

# Prior-free techniques whose offset is an independent witness of the same
# translation.  StarRefineNav is excluded on purpose: it is seeded by the
# pass-1 prior, which on a Titan frame is usually this technique's own answer.
STAR_WITNESSES = ('StarFieldFromCatalogNav', 'StarUniqueMatchNav')
BODY_WITNESSES = (
    'BodyLimbNav',
    'BodyDiscCorrelateNav',
    'BodyTerminatorNav',
    'BodyBlobNav',
)

SEQUENCE_WINDOW_S = 30 * 60.0
CROSS_FILTER_WINDOW_S = 10 * 60.0

# Acceptance bounds from the validation plan, restated here so a run reports
# against them without the reader holding the plan open.
CLEAN_ACCEPT_FRACTION = 0.70
PAIR_PASS_FRACTION = 0.90


def _parse_time(stamp: str | None) -> float | None:
    """Seconds since the epoch for a PDS3 ``IMAGE_TIME``, or None."""
    if not stamp:
        return None
    try:
        return datetime.strptime(stamp, '%Y-%jT%H:%M:%S.%f').timestamp()
    except ValueError:
        return None


def _sigma(covariance: Any, axis: int) -> float | None:
    """One-sigma on image axis ``axis`` (0 = v, 1 = u) from a covariance."""
    try:
        variance = float(covariance[axis][axis])
    except (TypeError, IndexError, ValueError):
        return None
    if not math.isfinite(variance) or variance < 0.0:
        return None
    return math.sqrt(variance)


def _titan(row: dict[str, Any]) -> dict[str, Any] | None:
    """The technique's entry on a row, or None when it never ran."""
    return row.get('techniques', {}).get(TECHNIQUE)


def _committed(row: dict[str, Any]) -> dict[str, Any] | None:
    """The technique's entry when it produced a usable offset, else None."""
    entry = _titan(row)
    if entry is None or entry.get('spurious') or entry.get('offset_px') is None:
        return None
    return entry


def _titan_feature(row: dict[str, Any]) -> dict[str, Any] | None:
    """The emitted ``TITAN_LIMB`` feature record, or None."""
    for feature in row.get('features', []):
        if feature.get('feature_type') == 'TITAN_LIMB':
            return feature
    return None


def _outcome(row: dict[str, Any]) -> tuple[str, str]:
    """Classify one frame as ``(outcome, attribution)``.

    Outcomes: ``committed``, ``gated`` (the feature never reached the
    technique), ``spurious`` (a named technique gate refused it), or
    ``no_feature`` (the model emitted nothing at all).
    """
    entry = _titan(row)
    if entry is None:
        feature = _titan_feature(row)
        if feature is None:
            return 'no_feature', str(row.get('status_reason') or row.get('error') or 'unknown')
        return 'gated', str(feature.get('gate_reason') or 'gated')
    if entry.get('spurious'):
        gate = (entry.get('diagnostics') or {}).get('gate_failed')
        return 'spurious', str(gate or 'unnamed')
    return 'committed', ''


def _pair_test(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, list[float], list[float]]:
    """Per-axis 2-sigma agreement between two technique entries.

    Returns:
        ``(passed, [dv, du] differences, [tol_v, tol_u] tolerances)``.
    """
    diffs = [float(a['offset_px'][i]) - float(b['offset_px'][i]) for i in range(2)]
    tolerances = []
    for axis in range(2):
        sa = _sigma(a.get('covariance_px2'), axis)
        sb = _sigma(b.get('covariance_px2'), axis)
        if sa is None or sb is None:
            tolerances.append(float('nan'))
        else:
            tolerances.append(2.0 * math.hypot(sa, sb))
    passed = all(math.isfinite(tolerances[i]) and abs(diffs[i]) <= tolerances[i] for i in range(2))
    return passed, diffs, tolerances


def _star_anchors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tier (a): every (haze, prior-free star) pair on one frame."""
    pairs = []
    for row in rows:
        titan = _committed(row)
        if titan is None:
            continue
        for name in STAR_WITNESSES:
            witness = row.get('techniques', {}).get(name)
            if witness is None or witness.get('spurious'):
                continue
            passed, diffs, tolerances = _pair_test(titan, witness)
            pairs.append(
                {
                    'image_id': row['image_id'],
                    'flags': row['flags'],
                    'witness': name,
                    'label': f'{row["image_id"]} vs {name}',
                    'titan_offset': titan['offset_px'],
                    'witness_offset': witness['offset_px'],
                    'diffs': diffs,
                    'axis_diffs': _axis_split(diffs, _sun_angle_deg(titan)),
                    'tolerances': tolerances,
                    'passed': passed,
                }
            )
    return pairs


def _body_witnesses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supplementary: every (haze, companion-body technique) pair."""
    pairs = []
    for row in rows:
        titan = _committed(row)
        if titan is None:
            continue
        for name in BODY_WITNESSES:
            witness = row.get('techniques', {}).get(name)
            if witness is None or witness.get('spurious'):
                continue
            passed, diffs, tolerances = _pair_test(titan, witness)
            pairs.append(
                {
                    'image_id': row['image_id'],
                    'flags': row['flags'],
                    'witness': name,
                    'label': f'{row["image_id"]} vs {name}',
                    'diffs': diffs,
                    'axis_diffs': _axis_split(diffs, _sun_angle_deg(titan)),
                    'tolerances': tolerances,
                    'passed': passed,
                }
            )
    return pairs


def _frame_pairs(
    rows: list[dict[str, Any]], *, window_s: float, same_filter: bool | None, clean_only: bool
) -> list[dict[str, Any]]:
    """Pairs of committed frames close in time, optionally filter-matched.

    Parameters:
        rows: Cohort rows.
        window_s: Maximum separation in seconds.
        same_filter: True to require identical filters, False to require
            different ones, None to accept either.
        clean_only: Restrict to frames whose annotation named no defect.

    Returns:
        One entry per qualifying pair with its 2-sigma verdict.
    """
    usable = []
    for row in rows:
        titan = _committed(row)
        if titan is None:
            continue
        if clean_only and 'clean' not in row['flags']:
            continue
        stamp = _parse_time(row.get('image_time'))
        if stamp is None:
            continue
        usable.append((stamp, row, titan))
    usable.sort(key=lambda item: item[0])
    pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            dt = usable[j][0] - usable[i][0]
            if dt > window_s:
                break
            row_a, row_b = usable[i][1], usable[j][1]
            if row_a.get('target') != row_b.get('target'):
                continue
            filters_equal = row_a.get('filters') == row_b.get('filters')
            if same_filter is True and not filters_equal:
                continue
            if same_filter is False and filters_equal:
                continue
            passed, diffs, tolerances = _pair_test(usable[i][2], usable[j][2])
            entry_a, entry_b = usable[i][2], usable[j][2]
            mean_sun = 0.5 * (_sun_angle_deg(entry_a) + _sun_angle_deg(entry_b))
            pairs.append(
                {
                    'a': row_a['image_id'],
                    'b': row_b['image_id'],
                    'label': f'{row_a["image_id"]} vs {row_b["image_id"]}',
                    'dt_s': round(dt, 1),
                    'filters_a': row_a.get('filters'),
                    'filters_b': row_b.get('filters'),
                    'diffs': diffs,
                    'axis_diffs': _axis_split(diffs, mean_sun),
                    'fitted_radius_km': (
                        float((entry_a.get('diagnostics') or {}).get('fitted_haze_radius_km', 0.0)),
                        float((entry_b.get('diagnostics') or {}).get('fitted_haze_radius_km', 0.0)),
                    ),
                    'tolerances': tolerances,
                    'passed': passed,
                }
            )
    return pairs


def _fmt(values: list[float]) -> str:
    """Render a numeric pair compactly."""
    return '(' + ', '.join(f'{v:.2f}' for v in values) + ')'


def _sun_angle_deg(entry: dict[str, Any]) -> float:
    """The symmetry-axis angle the technique used, in degrees."""
    return float((entry.get('diagnostics') or {}).get('sun_angle_deg', 0.0))


def _axis_split(diffs: list[float], sun_angle_deg: float) -> tuple[float, float]:
    """Resolve an image-axis ``(dv, du)`` difference onto ``(cross, along)``.

    The two axes carry different error mechanisms and different bounds -- the
    mirror-correlation axis against the limb-arc axis -- so a disagreement
    reported only in image coordinates says nothing about which one moved.

    Parameters:
        diffs: The ``(dv, du)`` difference in pixels.
        sun_angle_deg: The symmetry-axis angle.

    Returns:
        ``(cross_track, along_track)`` components in pixels.
    """
    theta = math.radians(sun_angle_deg)
    c_hat = (math.cos(theta), -math.sin(theta))
    a_hat = (math.sin(theta), math.cos(theta))
    return (
        diffs[0] * c_hat[0] + diffs[1] * c_hat[1],
        diffs[0] * a_hat[0] + diffs[1] * a_hat[1],
    )


def _axis_summary(title: str, pairs: list[dict[str, Any]]) -> list[str]:
    """Render the per-axis disagreement distribution of a pair channel."""
    lines = [f'### {title}', '']
    if not pairs:
        return [*lines, '(no pairs)', '']
    lines += ['| pair | cross | along | fitted R (km) |', '|---|---|---|---|']
    cross_values, along_values = [], []
    for pair in pairs:
        cross, along = pair['axis_diffs']
        cross_values.append(abs(cross))
        along_values.append(abs(along))
        radii = pair.get('fitted_radius_km')
        radius_text = (
            f'{radii[0]:.0f} / {radii[1]:.0f} (d {radii[0] - radii[1]:+.0f})' if radii else '-'
        )
        lines.append(f'| {pair["label"]} | {cross:+.2f} | {along:+.2f} | {radius_text} |')
    rms_cross = math.sqrt(sum(v * v for v in cross_values) / len(cross_values))
    rms_along = math.sqrt(sum(v * v for v in along_values) / len(along_values))
    lines += [
        '',
        f'|cross| max {max(cross_values):.2f} px, rms {rms_cross:.2f} px; '
        f'|along| max {max(along_values):.2f} px, rms {rms_along:.2f} px.  '
        f'A pair difference is two frames in quadrature, so the implied '
        f'per-frame scale is rms/sqrt(2): cross {rms_cross / math.sqrt(2):.2f} px, '
        f'along {rms_along / math.sqrt(2):.2f} px.',
        '',
    ]
    return lines


def report(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    """Build the full campaign report."""
    lines: list[str] = ['# Titan cohort run', '']
    config = manifest.get('titan_config', {})
    lines += [
        f'{len(rows)} frames.  Gate config: '
        f'max_residual_rms_px={config.get("arc", {}).get("max_residual_rms_px")}, '
        f'min_gradient_snr={config.get("arc", {}).get("min_gradient_snr")}, '
        f'max_second_peak_ratio='
        f'{config.get("symmetry", {}).get("max_second_peak_ratio")}, '
        f'sigma floors cross='
        f'{config.get("symmetry", {}).get("sigma_floor_cross_px")} '
        f'along={config.get("arc", {}).get("sigma_floor_along_px")}',
        '',
        '## Per-frame outcome',
        '',
        '| image | flags | outcome | attribution | offset (dv, du) | conf | resid |',
        '|---|---|---|---|---|---|---|',
    ]
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in sorted(rows, key=lambda r: (';'.join(r['flags']), r['image_id'])):
        outcome, attribution = _outcome(row)
        entry = _titan(row)
        diagnostics = (entry or {}).get('diagnostics') or {}
        offset = (entry or {}).get('offset_px')
        offset_text = _fmt(list(offset)) if offset and outcome == 'committed' else '-'
        confidence = (entry or {}).get('confidence')
        resid = diagnostics.get('arc_residual_rms_px')
        lines.append(
            f'| {row["image_id"]} | {";".join(row["flags"])} | {outcome} | {attribution} '
            f'| {offset_text} | {confidence if confidence is not None else "-"} '
            f'| {resid if resid is not None else "-"} |'
        )
        for flag in row['flags']:
            by_class[flag][outcome] += 1
    lines += ['', '## Outcome by flag', '', '| flag | committed | spurious | gated | no feature |']
    lines.append('|---|---|---|---|---|')
    for flag in sorted(by_class):
        counts = by_class[flag]
        lines.append(
            f'| {flag} | {counts["committed"]} | {counts["spurious"]} '
            f'| {counts["gated"]} | {counts["no_feature"]} |'
        )

    clean = [r for r in rows if 'clean' in r['flags']]
    clean_committed = [r for r in clean if _committed(r) is not None]
    fraction = len(clean_committed) / len(clean) if clean else 0.0
    lines += [
        '',
        '## Acceptance 1 - clean-frame accept rate',
        '',
        f'{len(clean_committed)}/{len(clean)} = {fraction:.1%} '
        f'(bound {CLEAN_ACCEPT_FRACTION:.0%}) '
        f'{"PASS" if fraction >= CLEAN_ACCEPT_FRACTION else "FAIL"}',
        '',
        'Every clean-frame non-accept, with its attribution:',
        '',
    ]
    for row in sorted(clean, key=lambda r: r['image_id']):
        if _committed(row) is not None:
            continue
        outcome, attribution = _outcome(row)
        feature = _titan_feature(row) or {}
        reasons = feature.get('reliability_reasons') or {}
        lines.append(
            f'- `{row["image_id"]}` {outcome}: {attribution} '
            f'(envelope {reasons.get("titan_envelope_diameter_px", "?")} px, '
            f'occluded {reasons.get("titan_occluded_fraction", "?")})'
        )

    anchors = _star_anchors(rows)
    sequence = _frame_pairs(rows, window_s=SEQUENCE_WINDOW_S, same_filter=True, clean_only=True)
    cross = _frame_pairs(rows, window_s=CROSS_FILTER_WINDOW_S, same_filter=False, clean_only=False)
    body = _body_witnesses(rows)

    def pair_block(title: str, pairs: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
        passed = sum(1 for p in pairs if p['passed'])
        lines.extend(
            ['', f'## {title}', '', f'{passed}/{len(pairs)} pass the per-axis 2-sigma test', '']
        )
        if not pairs:
            return
        lines.append('| ' + ' | '.join(columns) + ' | d(dv, du) | 2-sigma | verdict |')
        lines.append('|' + '---|' * (len(columns) + 3))
        for pair in pairs:
            cells = [str(pair[c]) for c in columns]
            lines.append(
                '| '
                + ' | '.join(cells)
                + f' | {_fmt(pair["diffs"])} | {_fmt(pair["tolerances"])} '
                + f'| {"pass" if pair["passed"] else "FAIL"} |'
            )

    pair_block('Evidence tier (a) - star-anchored', anchors, ('image_id', 'witness'))
    pair_block(
        'Evidence tier (b) - within-sequence (clean, <= 30 min)',
        sequence,
        ('a', 'b', 'dt_s'),
    )
    pair_block(
        'Evidence tier (c) - cross-filter (<= 10 min)',
        cross,
        ('a', 'b', 'dt_s', 'filters_a', 'filters_b'),
    )
    pair_block('Supplementary - companion-body witnesses', body, ('image_id', 'witness'))

    lines += ['', '## Axis-resolved consistency', '']
    lines += _axis_summary('Star-anchored (tier a)', anchors)
    lines += _axis_summary('Within-sequence (tier b)', sequence)
    lines += _axis_summary('Cross-filter (tier c)', cross)

    ab = anchors + sequence
    ab_passed = sum(1 for p in ab if p['passed'])
    ab_fraction = ab_passed / len(ab) if ab else 0.0
    lines += [
        '',
        '## Acceptance 2 - (a) + (b) pair agreement',
        '',
        f'{ab_passed}/{len(ab)} = {ab_fraction:.1%} (bound {PAIR_PASS_FRACTION:.0%}) '
        f'{"PASS" if ab_fraction >= PAIR_PASS_FRACTION else "FAIL"}',
    ]

    lines += [
        '',
        '## Acceptance 4 - adverse frames produce no confident-wrong lock',
        '',
        '| image | flags | outcome | attribution | offset | conf |',
        '|---|---|---|---|---|---|',
    ]
    for row in sorted(rows, key=lambda r: r['image_id']):
        if not (ADVERSE_FLAGS & set(row['flags'])):
            continue
        outcome, attribution = _outcome(row)
        entry = _titan(row)
        offset = (entry or {}).get('offset_px')
        lines.append(
            f'| {row["image_id"]} | {";".join(row["flags"])} | {outcome} | {attribution} '
            f'| {_fmt(list(offset)) if offset and outcome == "committed" else "-"} '
            f'| {(entry or {}).get("confidence", "-")} |'
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one row per frame: the run's outcome in a committable form.

    The campaign directory itself is large and stays out of the repository,
    so this is what makes a past run's per-frame result readable without
    re-running it.

    Parameters:
        rows: Cohort rows.
        path: Destination CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                'image_id',
                'flags',
                'camera',
                'filters',
                'phase_deg',
                'outcome',
                'attribution',
                'haze_dv_px',
                'haze_du_px',
                'haze_confidence',
                'envelope_diameter_px',
                'arc_residual_rms_px',
                'fitted_haze_radius_km',
                'ensemble_status',
                'ensemble_status_reason',
            ]
        )
        for row in sorted(rows, key=lambda r: r['image_id']):
            outcome, attribution = _outcome(row)
            entry = _titan(row) or {}
            diagnostics = entry.get('diagnostics') or {}
            offset = entry.get('offset_px') if outcome == 'committed' else None
            writer.writerow(
                [
                    row['image_id'],
                    ';'.join(row['flags']),
                    row.get('camera'),
                    row.get('filters'),
                    diagnostics.get('phase_deg'),
                    outcome,
                    attribution,
                    offset[0] if offset else '',
                    offset[1] if offset else '',
                    entry.get('confidence', ''),
                    diagnostics.get('envelope_diameter_px', ''),
                    diagnostics.get('arc_residual_rms_px', ''),
                    diagnostics.get('fitted_haze_radius_km', ''),
                    row.get('nav_status'),
                    row.get('status_reason'),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    """Load a run's rows and print (or write) its report."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--summary-csv', type=Path, default=None)
    args = parser.parse_args(argv)

    manifest: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    with args.rows.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('manifest'):
                manifest = record
            else:
                rows.append(record)
    if args.summary_csv:
        write_summary_csv(rows, args.summary_csv)
        print(f'Wrote {args.summary_csv}')
    text = report(rows, manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f'Wrote {args.out}')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
