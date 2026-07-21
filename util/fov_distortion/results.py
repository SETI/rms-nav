"""Per-instrument aggregation of per-frame measurements.

Combines the per-frame twist and residual-distortion measurements from
:mod:`measure` into an instrument-level summary: the twist-consistency verdict,
the rotation-fitting recommendation, and an aggregate radial distortion model
fitted to the pooled per-star residuals of every frame.  One summary covers one
instrument and camera, so every frame shares an image shape and the pooled
residuals can be fitted in a single common frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from util.fov_distortion.aggregate import (
    RotationRecommendation,
    TwistConsistency,
    recommend_rotation_fitting,
    twist_consistency,
)
from util.fov_distortion.decompose import RadialModel, fit_radial_distortion
from util.fov_distortion.measure import FrameMeasurement

__all__ = ['InstrumentSummary', 'PooledRadial', 'summarize_instrument']

FloatArray = NDArray[np.float64]

# Floor on a per-frame twist sigma (degrees) so a noiseless synthetic frame or
# a degenerate lever arm cannot drive the inverse-variance weighting to
# infinity.  Well below any real per-frame measurement uncertainty.
_SIGMA_FLOOR_DEG = 1e-4


@dataclass(frozen=True)
class PooledRadial:
    """Per-star radial residuals pooled over an instrument's frames.

    Parameters:
        rho_n: Normalized field radius per star (``|p - center| / rho_ref``).
        radial_px: Radial residual component per star (pixels).
        nonradial_px: Tangential residual component per star (pixels).
        model: Aggregate radial distortion model fitted to the pool.
        residual_after_fit_px: Radial residual after the aggregate model is
            subtracted, per star.
    """

    rho_n: FloatArray
    radial_px: FloatArray
    nonradial_px: FloatArray
    model: RadialModel
    residual_after_fit_px: FloatArray


@dataclass(frozen=True)
class InstrumentSummary:
    """Instrument-and-camera-level summary.

    Parameters:
        inst_id: Instrument id.
        label: Human-readable instrument / camera label.
        n_frames_total: Frames attempted.
        n_frames_ok: Frames that yielded a decomposition.
        consistency: Twist-consistency statistics, or ``None`` if no frame
            succeeded.
        recommendation: Rotation-fitting recommendation, or ``None``.
        pooled_radial: Pooled radial residuals + aggregate model, or ``None``.
        median_floor_px: Median per-frame post-radial residual RMS -- the
            centroid + astrometric noise floor.
        ok_frames: The frames that produced a decomposition.
    """

    inst_id: str
    label: str
    n_frames_total: int
    n_frames_ok: int
    consistency: TwistConsistency | None
    recommendation: RotationRecommendation | None
    pooled_radial: PooledRadial | None
    median_floor_px: float
    ok_frames: list[FrameMeasurement]


def summarize_instrument(
    inst_id: str,
    label: str,
    frames: list[FrameMeasurement],
    *,
    radial_powers: tuple[int, ...] = (3, 5),
    scatter_corner_threshold_px: float = 0.15,
) -> InstrumentSummary:
    """Aggregate one instrument-and-camera's frames into a summary.

    Parameters:
        inst_id: Instrument id.
        label: Human-readable label for the instrument / camera.
        frames: All attempted frame measurements (ok and failed).
        radial_powers: Powers for the aggregate radial fit.
        scatter_corner_threshold_px: Corner-displacement threshold below which
            the frame-to-frame twist scatter counts as a single common twist.

    Returns:
        An :class:`InstrumentSummary`.
    """
    ok = [f for f in frames if f.decomposition is not None]
    if not ok:
        return InstrumentSummary(
            inst_id=inst_id,
            label=label,
            n_frames_total=len(frames),
            n_frames_ok=0,
            consistency=None,
            recommendation=None,
            pooled_radial=None,
            median_floor_px=float('nan'),
            ok_frames=[],
        )

    twists: list[float] = []
    sigmas: list[float] = []
    for frame in ok:
        assert frame.decomposition is not None
        twist = frame.decomposition.twist
        if not np.isfinite(twist.sigma_rotation_deg):
            continue
        twists.append(twist.rotation_deg)
        sigmas.append(max(twist.sigma_rotation_deg, _SIGMA_FLOOR_DEG))
    consistency = twist_consistency(
        np.asarray(twists, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        ok[0].rho_ref_px,
        scatter_corner_threshold_px=scatter_corner_threshold_px,
    )
    recommendation = recommend_rotation_fitting(consistency)

    pooled = _pool_radial(ok, radial_powers)
    floors = [f.decomposition.rms_after_radial_px for f in ok if f.decomposition is not None]
    median_floor = float(np.median(floors)) if floors else float('nan')

    return InstrumentSummary(
        inst_id=inst_id,
        label=label,
        n_frames_total=len(frames),
        n_frames_ok=len(ok),
        consistency=consistency,
        recommendation=recommendation,
        pooled_radial=pooled,
        median_floor_px=median_floor,
        ok_frames=ok,
    )


def _pool_radial(frames: list[FrameMeasurement], powers: tuple[int, ...]) -> PooledRadial | None:
    """Pool per-star radial / tangential residuals and fit an aggregate model."""
    rho_n_parts: list[FloatArray] = []
    radial_parts: list[FloatArray] = []
    nonradial_parts: list[FloatArray] = []
    pred_parts: list[FloatArray] = []
    resid_parts: list[FloatArray] = []
    rho_ref = 0.0
    center = (0.0, 0.0)
    for frame in frames:
        assert frame.decomposition is not None
        pred = np.array([m.predicted_vu for m in frame.stars], dtype=np.float64)
        resid = frame.decomposition.twist.residuals_vu
        if pred.shape[0] != resid.shape[0] or pred.shape[0] == 0:
            continue
        cv, cu = frame.center_vu
        offset = pred - np.asarray([cv, cu])
        rho = np.hypot(offset[:, 0], offset[:, 1])
        safe = rho > 0.0
        rhat = np.zeros_like(offset)
        rhat[safe] = offset[safe] / rho[safe, None]
        that = np.column_stack([-rhat[:, 1], rhat[:, 0]])
        rho_n_parts.append(rho / frame.rho_ref_px)
        radial_parts.append(np.sum(resid * rhat, axis=1))
        nonradial_parts.append(np.sum(resid * that, axis=1))
        pred_parts.append(pred)
        resid_parts.append(resid)
        rho_ref = frame.rho_ref_px
        center = frame.center_vu
    if not pred_parts:
        return None

    pred_all = np.concatenate(pred_parts, axis=0)
    resid_all = np.concatenate(resid_parts, axis=0)
    model = fit_radial_distortion(pred_all, resid_all, center, rho_ref, powers=powers)

    rho_n = np.concatenate(rho_n_parts)
    radial = np.concatenate(radial_parts)
    nonradial = np.concatenate(nonradial_parts)
    after_fit = radial - model.radial_displacement_px(rho_n * rho_ref)
    return PooledRadial(
        rho_n=rho_n,
        radial_px=radial,
        nonradial_px=nonradial,
        model=model,
        residual_after_fit_px=after_fit,
    )
