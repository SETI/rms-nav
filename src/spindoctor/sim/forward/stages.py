"""The stage interface of the forward-model rendering pipeline.

A render is a fixed-order sequence of stages, each a callable matching the
:class:`Stage` protocol, mutating a :class:`SimFrame` in place.  The order
(scene radiance, optics, downsample, detector, telemetry) is physical: light
is composed, passes through the camera optics, lands on the detector grid,
is read out, and is then transmitted.

Each stage receives its own ``numpy.random.Generator`` seeded from the
scene's single ``random_seed`` via
``derive_effect_seed(random_seed, '<stage-name>')``, so one stage's noise
realization is independent of which other stages are enabled.  A stage name
is therefore part of its scenes' noise realization: renaming a stage reseeds
it and regenerates the affected baselines.

Placeholders at present fidelity (phase B and D fill these in):

- ``oversample`` is always 1: the radiance stage composes directly on the
  detector grid and :func:`downsample_to_detector` is a no-op.
- ``point_e`` is allocated but unused: stars are drawn PSF-spread in
  normalized signal units by the radiance stage rather than deposited as
  point-mass electrons for a whole-scene optics PSF.
- ``signal`` carries normalized [0, ~1] scene units through the optics
  stage and is converted to DN in place by the detector stage; the
  electrons/gain unit chain is not implemented yet.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['SimFrame', 'Stage', 'downsample_to_detector', 'new_sim_frame']


@dataclass
class SimFrame:
    """The mutable image state threaded through the rendering stages.

    Parameters:
        signal: ``(V*os, U*os)`` float64 image of intensive scene signal
            (I/F-like normalized units): bodies, rings, and diffuse
            backgrounds.  The detector stage converts it to DN in place.
        point_e: ``(V*os, U*os)`` float64 image reserved for point sources
            (stars, moonlets) in electrons, kept separate because one array
            cannot carry two unit systems through the detector stage's
            signal-to-electron conversion.  Unused at present fidelity (see
            the module docstring).
        oversample: Oversampling factor ``os >= 1``; the detector grid is
            ``(V, U)``.  Always 1 at present fidelity.
        truth: Feature truth accumulated by the radiance stage (the rendered
            star records, body masks, inventory, and z-order maps).  This is
            renderer output metadata; none of it crosses the information
            boundary to the navigator side.
    """

    signal: NDArrayFloatType
    point_e: NDArrayFloatType
    oversample: int = 1
    truth: dict[str, Any] = field(default_factory=dict)


class Stage(Protocol):
    """One rendering stage: a pure in-place transform of a :class:`SimFrame`.

    Parameters:
        frame: The frame to mutate.
        params: The full validated scene ``sim_params`` mapping.  A stage
            whose scene block is absent is disabled and contributes nothing
            (per-stage parameter blocks land with their phases).
        rng: The stage's own seeded random generator.
    """

    def __call__(
        self,
        frame: SimFrame,
        *,
        params: Mapping[str, Any],
        rng: np.random.Generator,
    ) -> None: ...


def new_sim_frame(size_v: int, size_u: int, *, oversample: int = 1) -> SimFrame:
    """Allocate a zeroed :class:`SimFrame` for a ``(size_v, size_u)`` detector.

    Parameters:
        size_v: Detector-grid height in pixels.
        size_u: Detector-grid width in pixels.
        oversample: Oversampling factor (1 at present fidelity).

    Returns:
        A frame with zeroed ``signal`` and ``point_e`` planes.
    """
    shape = (size_v * oversample, size_u * oversample)
    return SimFrame(
        signal=np.zeros(shape, dtype=np.float64),
        point_e=np.zeros(shape, dtype=np.float64),
        oversample=oversample,
    )


def downsample_to_detector(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Box-downsample the oversampled planes to the detector grid.

    The box filter is a mean over the ``os**2`` subsamples, so the intensive
    ``signal`` passes through unchanged in level.  At present fidelity the
    pipeline runs at ``oversample == 1`` and this stage is a no-op; it holds
    the pipeline slot so the phase-B oversampled optics stage can land
    without reordering.

    Parameters:
        frame: The frame to downsample in place.
        params: The scene mapping (unused; downsampling has no scene knobs).
        rng: The stage generator (unused; downsampling is deterministic).
    """
    del params, rng
    os = frame.oversample
    if os == 1:
        return
    size_v = frame.signal.shape[0] // os
    size_u = frame.signal.shape[1] // os
    frame.signal = frame.signal.reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    # The point-source plane carries extensive electron weights scaled by
    # os**2 at deposition, so the mean conserves the per-star electron sum.
    frame.point_e = frame.point_e.reshape(size_v, os, size_u, os).mean(axis=(1, 3))
    frame.oversample = 1
