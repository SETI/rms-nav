"""Collect calibration rows: navigate randomized sim scenes, record truth.

For every generated scene (see ``scene_gen``) this runs the full autonomous
ensemble in-process (the sim needs no external holdings) and writes one JSON
line per frame to the output file:

- the planted (offset_v, offset_u) ground truth
- the fused result: status, confidence, tier rank, offset error, sigma
- every per-technique result: confidence, spurious/at_edge flags, the
  technique's own offset error and reported sigma, and its full typed
  diagnostics dict -- the (x_i, error) pairs the sigmoid fit consumes

Run (from an activated project venv; ``source /seti/newnav/setup.sh``):

    venv/bin/python util/calibration/collect.py \
        --per-family 400 --workers 8 --out _work/calibration/rows.jsonl

The campaign seed defaults to 20260709; pass ``--seed`` to draw a different
campaign. Rows are written incrementally so a partial run is still usable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from scene_gen import FAMILIES, generate_scenes  # noqa: E402

DEFAULT_SEED = 20260709
DEFAULT_OUT = REPO / '_work/calibration/rows.jsonl'


def _json_safe(value: Any) -> Any:
    """Coerce a diagnostics value to something json.dumps accepts."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'__nonfinite__': repr(value)}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _sigma_max_px(covariance: Any) -> float | None:
    """Max positional 1-sigma (px) from a technique covariance, or None."""
    import numpy as np

    try:
        cov = np.asarray(covariance, dtype=float)
        diag = np.diag(cov)[:2]
        if np.any(diag < 0):
            return None
        return float(np.sqrt(diag.max()))
    except (TypeError, ValueError):
        # Malformed covariance payload (ragged / non-numeric / wrong rank);
        # anything else is a genuine bug and must surface.
        return None


def _navigate_one(task: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    """Worker: navigate one scene and return its calibration row."""
    scene_id, family, sim_params = task
    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator import NavOrchestrator
    from spindoctor.obs.obs_inst_sim import ObsSim

    planted_v = float(sim_params.get('offset_v', 0.0))
    planted_u = float(sim_params.get('offset_u', 0.0))
    row: dict[str, Any] = {
        'scene_id': scene_id,
        'family': family,
        'planted': {'dv': planted_v, 'du': planted_u},
        'sim_params': sim_params,
    }
    try:
        # The path is a label only (sim_params overrides the file read), but
        # FCPath resolves it, so it must sit under a real directory.
        obs = ObsSim.from_file(f'/tmp/{scene_id}.json', sim_params=sim_params)
        orchestrator = NavOrchestrator(
            build_models_for_obs(obs), only_models='*', only_techniques='*'
        )
        result = orchestrator.navigate(obs)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        return row

    if result.offset_px is None:
        fused_err = None
    else:
        fused_err = math.hypot(result.offset_px[0] - planted_v, result.offset_px[1] - planted_u)
    row['fused'] = {
        'status': str(result.status),
        'status_reason': str(result.status_reason),
        'confidence': float(result.confidence),
        'confidence_rank': str(result.confidence_rank),
        'offset_error_px': fused_err,
        'sigma_max_px': (None if result.sigma_px is None else float(max(result.sigma_px))),
    }
    techniques = []
    for tr in result.per_technique:
        err = math.hypot(tr.offset_px[0] - planted_v, tr.offset_px[1] - planted_u)
        diagnostics = {
            key: _json_safe(value) for key, value in dataclasses.asdict(tr.diagnostics).items()
        }
        techniques.append(
            {
                'name': tr.technique_name,
                'confidence': float(tr.confidence),
                'spurious': bool(tr.spurious),
                'at_edge': bool(tr.at_edge),
                'offset_error_px': err,
                'sigma_max_px': _sigma_max_px(tr.covariance_px2),
                'diagnostics': diagnostics,
            }
        )
    row['techniques'] = techniques
    return row


def _init_worker() -> None:
    """Pin per-worker threading and silence the per-image logger.

    One BLAS/OpenMP thread per worker: the pool already saturates the
    cores, and letting every worker spin a full thread team oversubscribes
    the machine badly (measured ~10x wall-clock inflation).  The env vars
    must be set before the worker's first numpy import, which is why this
    runs in the pool initializer rather than at module import.
    """
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

    null_handler = pdslogger.NULL_HANDLER
    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(null_handler)


def main(argv: list[str] | None = None) -> int:
    """Generate scenes, navigate them in a worker pool, write JSONL rows."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--per-family', type=int, default=400)
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    families = [f.strip() for f in args.families.split(',') if f.strip()]
    tasks: list[tuple[str, str, dict[str, Any]]] = []
    for family in families:
        for scene_id, sim_params in generate_scenes(
            family, args.per_family, campaign_seed=args.seed
        ):
            tasks.append((scene_id, family, sim_params))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    done = 0
    failed = 0
    with args.out.open('w') as out:
        out.write(
            json.dumps(
                {
                    'manifest': True,
                    'campaign_seed': args.seed,
                    'per_family': args.per_family,
                    'families': families,
                    'n_scenes': len(tasks),
                }
            )
            + '\n'
        )
        with multiprocessing.Pool(
            processes=args.workers, initializer=_init_worker, maxtasksperchild=200
        ) as pool:
            for row in pool.imap_unordered(_navigate_one, tasks, chunksize=4):
                out.write(json.dumps(row, sort_keys=True) + '\n')
                done += 1
                if 'error' in row:
                    failed += 1
                if done % 100 == 0:
                    elapsed = time.monotonic() - start
                    rate = done / elapsed
                    remaining = (len(tasks) - done) / rate if rate > 0 else 0
                    print(
                        f'{done}/{len(tasks)} rows ({failed} errors), '
                        f'{rate:.1f}/s, ~{remaining / 60:.1f} min left',
                        flush=True,
                    )
    elapsed = time.monotonic() - start
    print(f'Wrote {done} rows ({failed} errors) to {args.out} in {elapsed / 60:.1f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
