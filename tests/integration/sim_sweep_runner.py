"""Run the sim parameter sweeps and emit per-step JSON (Phase T3).

Invoked as ``python -m tests.integration.sim_sweep_runner`` from a clean
checkout.  For every spec under ``tests/integration/sim_sweeps/`` it navigates
each sweep step and writes ``tests/integration/sim_sweeps/results/<name>.json``
containing one row per value: ``value``, ``status``, ``offset_error_px``,
``confidence``, and ``primary_technique``.  The JSON is a human-readable
diagnostic artifact (response curves), not a re-blessed regression baseline --
``test_sim_sweeps`` asserts the invariants directly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from tests.integration.sim_sweep import iter_sweep_paths, load_sweep, run_sweep

_SWEEPS_ROOT = Path(__file__).parent / 'sim_sweeps'
_RESULTS_ROOT = _SWEEPS_ROOT / 'results'


def main() -> int:
    """Render + navigate every sweep and write its per-step JSON."""
    sweep_paths = iter_sweep_paths(_SWEEPS_ROOT)
    if not sweep_paths:
        print('No sweeps found; nothing to do.', file=sys.stderr)
        return 1
    _RESULTS_ROOT.mkdir(exist_ok=True)
    for sweep_path in sweep_paths:
        spec = load_sweep(sweep_path)
        rows = run_sweep(spec)
        document = {
            'sweep_name': spec.sweep_name,
            'base_scene': str(spec.base_scene.relative_to(_SWEEPS_ROOT.parent / 'sim_scenes')),
            'parameters': list(spec.parameters),
            'rows': [asdict(row) for row in rows],
        }
        out_path = _RESULTS_ROOT / f'{spec.sweep_name}.json'
        out_path.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
        techniques = [row.primary_technique for row in rows]
        print(f'{spec.sweep_name}: {len(spec.values)} steps -> {techniques}')
    print(f'Wrote {len(sweep_paths)} sweep result(s) to {_RESULTS_ROOT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
