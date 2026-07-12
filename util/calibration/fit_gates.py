"""Derive orchestrator-level gates from fused sim results.

The per-technique alphas (``fit.py``) change every fused confidence, so the
ensemble-level acceptance parameters in ``config_540_orchestrator.yaml``
(tier ``min_confidence`` boundaries and the final ``min_confidence`` gate)
must be derived from a collection run AFTER the fitted alphas are written
into ``config_510_techniques.yaml``.

Method ("tier boundaries map to stated error percentiles"):

- Tier sigma limits (``max_sigma_px`` 0.5 / 2.0) are the tier *definitions*
  (high ~ sub-half-pixel, medium ~ couple-pixel) and stay fixed; this
  script fits the confidence boundary of each tier as the smallest
  threshold at which the tier's subset achieves the target success rate
  (default 0.9) against the tier's own error budget.
- The final ``min_confidence`` gate is set where the empirical success
  probability of accepting a fused result crosses 0.5 against the
  ``--err-fit-px`` "still clearly a fit" budget.

Also reports the per-technique sigma coverage check (does the actual error
fall within k * reported sigma at the expected rate), for the report and
for future ``model_error_floor_px`` decisions.

Run:

    venv/bin/python util/calibration/fit_gates.py _work/calibration/rows_v2.jsonl \
        --out-report _work/calibration/gates_v2.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / 'src'))

TIER_BUDGETS = {
    'high': {'max_sigma_px': 0.5, 'err_budget_px': 0.5},
    'medium': {'max_sigma_px': 2.0, 'err_budget_px': 2.0},
}


def _fused_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract fused confidence / sigma / error arrays from scene rows."""
    conf, sigma, err = [], [], []
    for row in rows:
        fused = row.get('fused')
        if not fused or fused['offset_error_px'] is None:
            continue
        conf.append(fused['confidence'])
        sigma.append(fused['sigma_max_px'] if fused['sigma_max_px'] is not None else np.inf)
        err.append(fused['offset_error_px'])
    return {
        'confidence': np.array(conf),
        'sigma': np.array(sigma),
        'error': np.array(err),
    }


def _tier_boundary(
    arrays: dict[str, np.ndarray],
    *,
    max_sigma_px: float,
    err_budget_px: float,
    target_rate: float,
) -> dict[str, Any]:
    """Smallest confidence threshold meeting the tier's success target."""
    mask = arrays['sigma'] <= max_sigma_px
    conf = arrays['confidence'][mask]
    err = arrays['error'][mask]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    chosen = None
    curve = []
    for c in grid:
        sel = conf >= c
        n = int(sel.sum())
        if n < 20:
            continue
        rate = float((err[sel] <= err_budget_px).mean())
        curve.append({'min_confidence': float(c), 'n': n, 'success_rate': round(rate, 3)})
        if chosen is None and rate >= target_rate:
            chosen = float(c)
    return {'boundary': chosen, 'curve': curve, 'n_in_sigma': int(mask.sum())}


def _min_confidence_gate(arrays: dict[str, np.ndarray], *, err_fit_px: float) -> dict[str, Any]:
    """Confidence below which accepted fused results are mostly not fits."""
    conf = arrays['confidence']
    err = arrays['error']
    grid = np.round(np.arange(0.05, 0.71, 0.05), 2)
    curve = []
    chosen = None
    for c in grid:
        band = (conf >= c - 0.049) & (conf <= c + 0.049)
        n = int(band.sum())
        if n < 15:
            continue
        rate = float((err[band] <= err_fit_px).mean())
        curve.append({'confidence_band': float(c), 'n': n, 'fit_rate': round(rate, 3)})
        if chosen is None and rate >= 0.5:
            chosen = float(c)
    return {'gate': chosen, 'curve': curve}


def _sigma_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-technique: fraction of usable rows with error <= k * reported sigma."""
    per: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        for t in row.get('techniques', []):
            if t['spurious'] or t['at_edge'] or t['offset_error_px'] is None:
                continue
            if t['sigma_max_px'] is None or not np.isfinite(t['sigma_max_px']):
                continue
            per.setdefault(t['name'], []).append((t['offset_error_px'], t['sigma_max_px']))
    out = []
    for name, pairs in sorted(per.items()):
        err = np.array([p[0] for p in pairs])
        sig = np.array([p[1] for p in pairs])
        entry = {'technique': name, 'n': len(pairs)}
        for k in (1.0, 2.0, 3.0):
            entry[f'coverage_{k:g}sigma'] = round(float((err <= k * sig).mean()), 3)
        # 2D Gaussian reference: P(r <= k*sigma) = 1 - exp(-k^2/2)
        out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    """Compute tier boundaries, the min-confidence gate, and sigma coverage."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    parser.add_argument('--target-rate', type=float, default=0.9)
    parser.add_argument('--err-fit-px', type=float, default=3.0)
    parser.add_argument('--out-report', type=Path, required=True)
    args = parser.parse_args(argv)

    rows = []
    with args.rows.open() as fh:
        for line in fh:
            record = json.loads(line)
            if not record.get('manifest'):
                rows.append(record)
    arrays = _fused_arrays(rows)
    print(f'{len(rows)} scene rows, {len(arrays["confidence"])} fused results')

    lines = [
        '# Orchestrator gate derivation (sim-anchored)',
        '',
        f'{len(arrays["confidence"])} fused results with an offset.',
        f'Tier target success rate: {args.target_rate}; '
        f'"still a fit" budget: {args.err_fit_px} px.',
        '',
    ]
    for tier, budget in TIER_BUDGETS.items():
        result = _tier_boundary(
            arrays,
            max_sigma_px=budget['max_sigma_px'],
            err_budget_px=budget['err_budget_px'],
            target_rate=args.target_rate,
        )
        print(f'{tier}: boundary {result["boundary"]} (n_in_sigma {result["n_in_sigma"]})')
        lines += [
            f'## Tier {tier} (sigma <= {budget["max_sigma_px"]} px, '
            f'err budget {budget["err_budget_px"]} px)',
            '',
            f'Proposed min_confidence: **{result["boundary"]}**',
            '',
            '| min_confidence | n | success rate |',
            '|---|---|---|',
        ]
        lines += [
            f'| {c["min_confidence"]} | {c["n"]} | {c["success_rate"]} |' for c in result['curve']
        ]
        lines.append('')

    gate = _min_confidence_gate(arrays, err_fit_px=args.err_fit_px)
    print(f'final min_confidence gate: {gate["gate"]}')
    lines += [
        '## Final min_confidence gate',
        '',
        f'Proposed: **{gate["gate"]}** (confidence band where fit rate crosses 0.5)',
        '',
        '| confidence band | n | fit rate |',
        '|---|---|---|',
    ]
    lines += [f'| {c["confidence_band"]} | {c["n"]} | {c["fit_rate"]} |' for c in gate['curve']]
    lines.append('')

    lines += [
        '## Per-technique sigma coverage (2D Gaussian reference: 39% / 86% / 99% at 1/2/3 sigma)',
        '',
        '| technique | n | <=1 sigma | <=2 sigma | <=3 sigma |',
        '|---|---|---|---|---|',
    ]
    for entry in _sigma_coverage(rows):
        lines.append(
            f'| {entry["technique"]} | {entry["n"]} | {entry["coverage_1sigma"]} | '
            f'{entry["coverage_2sigma"]} | {entry["coverage_3sigma"]} |'
        )
    lines.append('')

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.out_report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
