"""Synthetic frame builders shared by the aggregation and plotting tests."""

from __future__ import annotations

import math

import numpy as np
from util.fov_distortion.decompose import decompose_frame
from util.fov_distortion.measure import FrameMeasurement, StarMeasurement


def make_frame(
    *,
    twist_deg: float,
    k1: float,
    n_side: int = 9,
    shape: tuple[int, int] = (1024, 1024),
    noise_px: float = 0.03,
    seed: int = 0,
    name: str = 'SYN',
) -> FrameMeasurement:
    """Build a synthetic measured frame with a planted twist and radial term."""
    rng = np.random.default_rng(seed)
    center = ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)
    rho_ref = 0.5 * math.hypot(shape[0], shape[1])
    coords = np.linspace(60.0, shape[0] - 60.0, n_side)
    vv, uu = np.meshgrid(coords, coords)
    pred = np.column_stack([vv.ravel(), uu.ravel()]).astype(np.float64)
    off = pred - np.asarray(center)
    rho = np.hypot(off[:, 0], off[:, 1])
    keep = rho > 0.0
    pred = pred[keep]
    rho = rho[keep]
    theta = math.radians(twist_deg)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    rotated = (pred - np.asarray(center)) @ rot.T + np.asarray(center)
    rhat = (pred - np.asarray(center)) / rho[:, None]
    rho_n = rho / rho_ref
    radial = rho_ref * k1 * rho_n**3
    det = rotated + radial[:, None] * rhat + rng.normal(0, noise_px, pred.shape)

    stars = [
        StarMeasurement(
            predicted_vu=(float(p[0]), float(p[1])),
            detected_vu=(float(d[0]), float(d[1])),
            vmag=9.0,
            peak_dn=100.0,
        )
        for p, d in zip(pred, det, strict=True)
    ]
    decomp = decompose_frame(pred, det, center, rho_ref, powers=(3, 5))
    return FrameMeasurement(
        image_name=name,
        url=f'mem://{name}',
        inst_id='syn',
        image_shape=shape,
        offset_vu=(0.0, 0.0),
        center_vu=center,
        rho_ref_px=rho_ref,
        stars=stars,
        decomposition=decomp,
    )
