"""Single-core cold-render performance budget for the forward model.

A 512x512 scene with a whole-scene PSF plus the full detector stack at
oversample 4 must render in under 2 s single-core, and a 1024x1024
Cassini-class scene in under 8 s -- both for a star-field frame (the
optics + detector stack) and for a frame dominated by a large lit body
with limb relief (the topographic body renderer's split-resolution path
plus the terminator shadow march).  The budget is a *cold-render* budget:
the render caches are cleared so the timed render pays the kernel-build
and compile costs a first render pays.

The harness enforces single-core itself: it pins the process to one CPU with an
affinity mask and sets the BLAS/OpenMP thread-count environment variables, so an
unpinned numpy FFT cannot silently multithread and fake the budget.  This is
integration-marked so it runs under ``pytest -m ''`` (and the deliberate
integration layer) but not the fast unit suite.

The assertion reads the render's process CPU time on the pinned core, not wall
time.  With the affinity mask and single-threaded BLAS the two are equal on an
idle machine, but wall time also charges the render for time slices consumed by
unrelated processes sharing the host (a parallel test battery, other agents),
which is contention, not render cost.  CPU time measures exactly what the
budget bounds -- the work one core must do -- and is immune to load.  Wall time
is still measured and reported in the failure message for context.
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

    A body-free star-navigation frame at oversample 4 exercises the whole-scene
    PSF convolution on the oversampled grid and the electron detector stack
    (shot + read noise, dark, hot pixels, banding, bias structure, cosmic rays,
    bloom, quantization).
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


def _body_psf_detector_scene(size: int) -> dict[str, Any]:
    """A size x size body-dominated frame with relief, a PSF, and the detector stack.

    A large lit body with limb relief at oversample 4 exercises the topographic
    body renderer under the same budgets as the star field: detector-resolution
    shading upsampled to the oversampled grid, the relief-perturbed silhouette,
    and the capped terminator shadow march, followed by the PSF convolution and
    the detector stack.  The high phase angle puts the terminator across the
    disc so the march band is fully active.
    """
    scene = _psf_detector_scene(size)
    del scene['sky_counts']
    scene['bodies'] = [
        {
            'name': 'BUDGET-BODY',
            'center_v': size / 2.0,
            'center_u': size / 2.0,
            'axis1': size * 0.7,
            'axis2': size * 0.7,
            'axis3': size * 0.7,
            'illumination_angle': 30.0,
            'phase_angle': 70.0,
            'limb_relief_rms': 0.01,
            'limb_relief_corr_deg': 15.0,
        }
    ]
    return scene


def _cold_render_seconds(scene: dict[str, Any]) -> tuple[float, float]:
    """Time one render from cold caches, single-core.

    Returns:
        (cpu_seconds, wall_seconds) for the render. The budget assertion
        reads the CPU time (contention-immune on the pinned core); the wall
        time is reported for context.
    """
    with _single_core():
        render.clear_render_caches()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        render.render_combined_model(scene)
        return time.process_time() - cpu_start, time.perf_counter() - wall_start


def test_512_psf_detector_render_under_2s() -> None:
    """A 512x512 PSF + detector scene renders in under 2 s single-core (cold)."""
    cpu, wall = _cold_render_seconds(_psf_detector_scene(512))
    assert cpu < 2.0, f'512x512 cold render took {cpu:.2f}s CPU (budget 2.0s; wall {wall:.2f}s)'


def test_1024_cassini_render_under_8s() -> None:
    """A 1024x1024 Cassini-class scene renders in under 8 s single-core (cold)."""
    cpu, wall = _cold_render_seconds(_psf_detector_scene(1024))
    assert cpu < 8.0, f'1024x1024 cold render took {cpu:.2f}s CPU (budget 8.0s; wall {wall:.2f}s)'


# Body-bearing budget scenes (#290): the previous body renderer's
# per-subsample shading overran these budgets ~4x at oversample 4; the
# topographic renderer's split-resolution path must hold them.


def test_512_body_render_under_2s() -> None:
    """A 512x512 lit-body scene with relief renders in under 2 s single-core."""
    cpu, wall = _cold_render_seconds(_body_psf_detector_scene(512))
    assert cpu < 2.0, (
        f'512x512 body cold render took {cpu:.2f}s CPU (budget 2.0s; wall {wall:.2f}s)'
    )


def test_1024_body_render_under_8s() -> None:
    """A 1024x1024 lit-body scene with relief renders in under 8 s single-core."""
    cpu, wall = _cold_render_seconds(_body_psf_detector_scene(1024))
    assert cpu < 8.0, (
        f'1024x1024 body cold render took {cpu:.2f}s CPU (budget 8.0s; wall {wall:.2f}s)'
    )
