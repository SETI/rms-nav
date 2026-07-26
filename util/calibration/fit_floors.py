"""Calibrate model-error covariance floors for the NCC techniques (#210).

The NCC peak-curvature covariance measures photon statistics only, so the
reported positional sigma of BodyDiscCorrelateNav / RingAnnulusNav /
BodyBlobNav sits orders of magnitude below the actual recovery error
against planted truth (1-sigma coverage ~0.01 vs the 0.39 2D-Gaussian
reference).  Each technique carries a ``model_error_floor_px`` tuning knob
added in quadrature to the covariance diagonal; this script solves, per
technique, for the floor that brings the 2-sigma coverage of
``sqrt(sigma_reported^2 + floor^2)`` to the 2D-Gaussian expectation
(``1 - exp(-2) = 0.865``) on the campaign's usable rows.

Run after a collection pass:

    venv/bin/python util/calibration/fit_floors.py _work/calibration/rows_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TECHNIQUES = (
    'BodyDiscCorrelateNav',
    'RingAnnulusNav',
    'BodyBlobNav',
    'BodyLimbNav',
    'BodyTerminatorNav',
    'StarRefineNav',
    'TitanHazeNav',
)
TARGET_2SIGMA = 1.0 - math.exp(-2.0)


def _coverage(errors: np.ndarray, sigmas: np.ndarray, floor: float, k: float = 2.0) -> float:
    """Fraction of rows with error within k * floored sigma.

    Parameters:
        errors: Per-row absolute offset errors (px).
        sigmas: Per-row reported max positional sigmas (px).
        floor: Candidate floor added in quadrature to every sigma (px).
        k: Coverage multiple (2.0 = the 2-sigma target).

    Returns:
        Fraction of rows whose error falls within ``k * sqrt(sigma^2 + floor^2)``.
    """
    floored = np.sqrt(sigmas**2 + floor**2)
    return float((errors <= k * floored).mean())


def solve_floor(errors: np.ndarray, sigmas: np.ndarray) -> float:
    """Bisect the floor bringing 2-sigma coverage to the Gaussian target.

    Parameters:
        errors: Per-row absolute offset errors (px).
        sigmas: Per-row reported max positional sigmas (px).

    Returns:
        The smallest floor (px, within the [0, 50] bracket at 60-step
        bisection precision) whose 2-sigma coverage reaches
        ``TARGET_2SIGMA``; ``0.0`` when the unfloored sigmas already
        cover.
    """
    lo, hi = 0.0, 50.0
    if _coverage(errors, sigmas, lo) >= TARGET_2SIGMA:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _coverage(errors, sigmas, mid) < TARGET_2SIGMA:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(argv: list[str] | None = None) -> int:
    """Solve and report the per-technique floors.

    Parameters:
        argv: Argument list; None uses ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('rows', type=Path)
    args = parser.parse_args(argv)

    per: dict[str, list[tuple[float, float]]] = {name: [] for name in TECHNIQUES}
    with args.rows.open() as fh:
        for line in fh:
            record: dict[str, Any] = json.loads(line)
            if record.get('manifest'):
                continue
            for t in record.get('techniques', []):
                if t['name'] not in per or t['spurious'] or t['at_edge']:
                    continue
                if t['offset_error_px'] is None or t['sigma_max_px'] is None:
                    continue
                if not math.isfinite(t['sigma_max_px']):
                    continue
                per[t['name']].append((t['offset_error_px'], t['sigma_max_px']))

    for name, pairs in per.items():
        if len(pairs) < 50:
            print(f'{name}: only {len(pairs)} usable rows; skipped')
            continue
        errors = np.array([p[0] for p in pairs])
        sigmas = np.array([p[1] for p in pairs])
        floor = solve_floor(errors, sigmas)
        cov_before = [round(_coverage(errors, sigmas, 0.0, k), 3) for k in (1.0, 2.0, 3.0)]
        cov_after = [round(_coverage(errors, sigmas, floor, k), 3) for k in (1.0, 2.0, 3.0)]
        print(
            f'{name}: n={len(pairs)} floor={floor:.3f} px  '
            f'coverage 1/2/3 sigma: {cov_before} -> {cov_after}  '
            f'(2D Gaussian reference 0.393 / 0.865 / 0.989)'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
