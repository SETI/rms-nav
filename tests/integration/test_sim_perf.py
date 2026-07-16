"""Single-core cold-render performance budget for the forward model.

A 512x512 star-field scene with a whole-scene PSF plus the full detector stack
at oversample 4 must render in under 2 s single-core, and a 1024x1024
Cassini-class star-field scene in under 8 s.  The budgets bound the optics +
detector stack; a scene with a large lit body at oversample 4 currently exceeds
them because the body renderer's per-subsample shading dominates, a cost
outside these budgets pending the body-renderer replacement.  The budget is a
*cold-render* budget: the render caches are cleared so the timed render pays the
kernel-build and compile costs a first render pays.

The harness enforces single-core itself: it pins the process to one CPU with an
affinity mask and sets the BLAS/OpenMP thread-count environment variables, so an
unpinned numpy FFT cannot silently multithread and fake the budget.  This is
integration-marked so it runs under ``pytest -m ''`` (and the deliberate
integration layer) but not the fast unit suite.

The assertion reads the pinned single-core wall time.  Under heavy machine load
(for example, other agents sharing the host) the timed render can exceed the
budget purely from contention; the budget is not raised for that -- a failure is
reported and investigated, not blessed.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from spindoctor.sim import render

pytestmark = pytest.mark.integration

_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
)


@contextmanager
def _single_core() -> Iterator[None]:
    """Pin the process to one CPU and cap BLAS/OpenMP threads for the duration."""
    saved_env = {name: os.environ.get(name) for name in _THREAD_ENV_VARS}
    for name in _THREAD_ENV_VARS:
        os.environ[name] = '1'
    saved_affinity: set[int] | None = None
    if hasattr(os, 'sched_getaffinity'):
        saved_affinity = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {min(saved_affinity)})
    try:
        yield
    finally:
        if saved_affinity is not None:
            os.sched_setaffinity(0, saved_affinity)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _psf_detector_scene(size: int) -> dict[str, Any]:
    """A size x size Cassini-class star field with a PSF and the full detector stack.

    A body-free star-navigation frame at oversample 4 exercises exactly the
    cost the budget bounds: the whole-scene PSF convolution on the oversampled
    grid and the electron detector stack (shot + read noise, dark, hot pixels,
    banding, bias structure, cosmic rays, bloom, quantization).  A large lit
    body would dominate the timing with the body renderer's oversampled
    shading cost, which is outside the optics + detector budget.
    """
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'artifacts': {'instrument_defaults': True},
        'optics': {'psf': {'sigma_v': 0.55, 'sigma_u': 0.55, 'w': 0.025, 'r0': 2.0, 'n': 3.0}},
        'noise': {
            'poisson': True,
            'read_noise_dn': 4.0,
            'cosmic_ray_rate_per_sec': 0.001,
            'bloom_length': 2,
        },
        'sky_counts': {'density_factor': 200.0},
    }


def _cold_render_seconds(scene: dict[str, Any]) -> float:
    """Time one render from cold caches, single-core."""
    with _single_core():
        render.clear_render_caches()
        start = time.perf_counter()
        render.render_combined_model(scene)
        return time.perf_counter() - start


def test_512_psf_detector_render_under_2s() -> None:
    """A 512x512 PSF + detector scene renders in under 2 s single-core (cold)."""
    elapsed = _cold_render_seconds(_psf_detector_scene(512))
    assert elapsed < 2.0, f'512x512 cold render took {elapsed:.2f}s (budget 2.0s)'


def test_1024_cassini_render_under_8s() -> None:
    """A 1024x1024 Cassini-class scene renders in under 8 s single-core (cold)."""
    elapsed = _cold_render_seconds(_psf_detector_scene(1024))
    assert elapsed < 8.0, f'1024x1024 cold render took {elapsed:.2f}s (budget 8.0s)'
