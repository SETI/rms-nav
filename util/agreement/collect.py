"""Collect agreement rows: navigate composition scenes, record per-technique offsets.

For every generated scene (see ``scene_gen``) this runs the autonomous
pipeline in-process and writes one JSON line per scene: the planted
(offset_v, offset_u) truth, the scene's geometry angles, and every
per-technique result from each configured run.  Single-body compositions
navigate once with every model; the multi_body composition navigates once
per body (``only_models='body_sim:<NAME>'``) so each body contributes its
own limb/disc estimates; ring-bearing compositions add a blob-only run
(``only_techniques=['BodyBlobNav']``) to supply an extra estimator that
never touches the shared gradient/DT products.

Shared-bias injection (the bias-independence stage) is harness-level only:
``--injection dt_shift`` monkeypatches the orchestrator module's
``compute_all_image_derivatives`` so every DT technique sees the shared
gradient / edge-distance-transform products translated by a per-scene
random bias, and ``--injection noise_scale`` monkeypatches
``estimate_image_noise_sigma`` to scale the single shared noise estimate.
Nothing under ``src/`` changes; the injected values are recorded per row.

Run (from an activated project venv; ``source /seti/newnav/setup.sh``):

    venv/bin/python util/agreement/collect.py \
        --per-family 400 --workers 8 --out _work/agreement/rows.jsonl

The campaign seed defaults to 20260719; pass ``--seed`` to draw a different
campaign.  Rows are written incrementally so a partial run is still usable.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'src'))

from scene_gen import FAMILIES, generate_scenes  # noqa: E402

DEFAULT_SEED = 20260719
DEFAULT_OUT = REPO / '_work/agreement/rows.jsonl'

INJECTION_KINDS = ('none', 'dt_shift', 'noise_scale')


def _runs_for_family(family: str) -> list[dict[str, Any]]:
    """Return the orchestrator run configurations for a scene family.

    Parameters:
        family: Scene family name.

    Returns:
        List of run dicts with ``key``, ``only_models``, ``only_techniques``.
    """
    if family == 'multi_body':
        return [
            {'key': 'body:RHEA', 'only_models': 'body_sim:RHEA', 'only_techniques': '*'},
            {'key': 'body:DIONE', 'only_models': 'body_sim:DIONE', 'only_techniques': '*'},
        ]
    runs: list[dict[str, Any]] = [{'key': 'full', 'only_models': '*', 'only_techniques': '*'}]
    if 'ring' in family:
        # BodyBlobNav is a fallback the orchestrator skips whenever a primary
        # body technique produced a result, so the shared-layer-independent
        # blob estimate needs its own restricted run.
        runs.append({'key': 'blob', 'only_models': '*', 'only_techniques': ['BodyBlobNav']})
    return runs


def _draw_injection(kind: str, scene_id: str, seed: int, sigma_px: float) -> dict[str, Any]:
    """Draw the per-scene injection values (recorded in the row).

    Parameters:
        kind: One of :data:`INJECTION_KINDS`.
        scene_id: Scene identifier (part of the draw key).
        seed: Campaign seed (part of the draw key).
        sigma_px: Per-axis standard deviation of the dt_shift bias draw.

    Returns:
        Dict describing the injection (``{'kind': 'none'}`` when disabled).
    """
    if kind == 'none':
        return {'kind': 'none'}
    rng = random.Random(f'{seed}/{scene_id}/inject')
    if kind == 'dt_shift':
        return {
            'kind': 'dt_shift',
            'bias_v': rng.gauss(0.0, sigma_px),
            'bias_u': rng.gauss(0.0, sigma_px),
        }
    return {
        'kind': 'noise_scale',
        'factor': math.exp(rng.uniform(math.log(2.0), math.log(8.0))),
    }


def _apply_injection(injection: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    """Monkeypatch the shared preprocessing layer per the injection spec.

    The patch targets the orchestrator module's imported references, so
    every technique downstream of the shared per-image products sees the
    injected version while techniques that read the image directly are
    untouched.

    Parameters:
        injection: Draw from :func:`_draw_injection`.

    Returns:
        List of ``(module, attribute, original)`` entries for restoration.
    """
    import numpy as np

    import spindoctor.nav_orchestrator.orchestrator as orch_mod

    patched: list[tuple[Any, str, Any]] = []
    if injection['kind'] == 'dt_shift':
        from scipy.ndimage import shift as nd_shift

        bias = (float(injection['bias_v']), float(injection['bias_u']))
        real_compute = orch_mod.compute_all_image_derivatives  # type: ignore[attr-defined]

        def injected_compute(
            image_ext: Any, image_noise_sigma: float, *, config: Any = None
        ) -> Any:
            gradient, edge_dt, gradient_vu = real_compute(
                image_ext, image_noise_sigma, config=config
            )
            kw = {'order': 1, 'mode': 'nearest'}
            gradient = nd_shift(gradient, bias, **kw)
            edge_dt = nd_shift(edge_dt, bias, **kw)
            gradient_vu = np.stack(
                [
                    nd_shift(gradient_vu[..., 0], bias, **kw),
                    nd_shift(gradient_vu[..., 1], bias, **kw),
                ],
                axis=-1,
            )
            return gradient, edge_dt, gradient_vu

        patched.append((orch_mod, 'compute_all_image_derivatives', real_compute))
        orch_mod.compute_all_image_derivatives = injected_compute  # type: ignore[attr-defined]
    elif injection['kind'] == 'noise_scale':
        factor = float(injection['factor'])
        real_noise = orch_mod.estimate_image_noise_sigma  # type: ignore[attr-defined]

        def injected_noise(image: Any, sensor_mask: Any) -> float:
            return factor * float(real_noise(image, sensor_mask))

        patched.append((orch_mod, 'estimate_image_noise_sigma', real_noise))
        orch_mod.estimate_image_noise_sigma = injected_noise  # type: ignore[attr-defined,assignment]
    return patched


def _restore(patched: list[tuple[Any, str, Any]]) -> None:
    """Undo :func:`_apply_injection`."""
    for module, attribute, original in patched:
        setattr(module, attribute, original)


def _navigate_one(
    task: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Worker: navigate one scene through its configured runs."""
    scene_id, family, sim_params, geometry, injection = task
    import numpy as np

    from spindoctor.nav_model import build_models_for_obs
    from spindoctor.nav_orchestrator import NavOrchestrator
    from spindoctor.obs.obs_inst_sim import ObsSim

    row: dict[str, Any] = {
        'scene_id': scene_id,
        'family': family,
        'planted': {
            'dv': float(sim_params.get('offset_v', 0.0)),
            'du': float(sim_params.get('offset_u', 0.0)),
        },
        'geometry': geometry,
        'injection': injection,
        'runs': {},
    }
    patched = _apply_injection(injection)
    try:
        # The path is a label only (sim_params overrides the file read), but
        # FCPath resolves it, so it must sit under a real directory.
        obs = ObsSim.from_file(f'/tmp/{scene_id}.json', sim_params=sim_params)
        for run in _runs_for_family(family):
            orchestrator = NavOrchestrator(
                build_models_for_obs(obs),
                only_models=run['only_models'],
                only_techniques=run['only_techniques'],
            )
            result = orchestrator.navigate(obs)
            techniques = []
            for tr in result.per_technique:
                cov = np.asarray(tr.covariance_px2, dtype=float)[:2, :2]
                techniques.append(
                    {
                        'name': tr.technique_name,
                        'offset_v': float(tr.offset_px[0]),
                        'offset_u': float(tr.offset_px[1]),
                        'covariance_px2': [float(x) for x in cov.ravel()],
                        'confidence': float(tr.confidence),
                        'spurious': bool(tr.spurious),
                        'at_edge': bool(tr.at_edge),
                    }
                )
            row['runs'][run['key']] = techniques
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        _restore(patched)
    return row


def _init_worker() -> None:
    """Pin per-worker threading and silence the per-image logger.

    One BLAS/OpenMP thread per worker: the pool already saturates the
    cores.  The env vars must be set before the worker's first numpy
    import, which is why this runs in the pool initializer.
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
    parser.add_argument('--injection', choices=INJECTION_KINDS, default='none')
    parser.add_argument(
        '--injection-sigma-px',
        type=float,
        default=0.7,
        help='per-axis sigma of the dt_shift bias draw',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    families = [f.strip() for f in args.families.split(',') if f.strip()]
    tasks = []
    for family in families:
        for scene_id, sim_params, geometry in generate_scenes(
            family, args.per_family, campaign_seed=args.seed
        ):
            injection = _draw_injection(
                args.injection, scene_id, args.seed, args.injection_sigma_px
            )
            tasks.append((scene_id, family, sim_params, geometry, injection))

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
                    'injection': args.injection,
                    'injection_sigma_px': args.injection_sigma_px,
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
