"""Independent inverse helpers shared by the C-matrix tests.

The hermetic and the real-frame C-matrix tests both check the recorded
attitude by recovering the planted offset back out of it.  The recovery is
written from the offset model itself, using only oops FOV primitives, and
imports nothing from the forward path under test: that independence is what
lets it catch a sign, conjugation or composition error.  A helper that
called the implementation would agree with it whatever it did.
"""

from __future__ import annotations

import numpy as np
import oops

from spindoctor.obs import ObsSnapshotInst


def offset_from_correction(fov: oops.fov.FOV, correction: np.ndarray) -> tuple[float, float]:
    """Recover the ``(dv, du)`` offset an oops-frame correction rotation encodes.

    Maps the corrected boresight direction back through the rotation, reads
    off the tangent-plane point that direction came from, and converts the
    implied ``xy_offset`` back into pixels.

    Parameters:
        fov: The observation's unmodified oops FOV.
        correction: The 3x3 correction rotation in oops observation frame
            coordinates.

    Returns:
        The ``(dv, du)`` offset in pixels that produces this correction.
    """
    xy_los = fov.xy_from_uv(fov.uv_los)
    corrected = np.asarray(fov.los_from_xy(xy_los).unit().vals, np.float64)
    uncorrected = np.asarray(correction, np.float64).T @ corrected
    xy_uncorrected = oops.Pair((uncorrected[0] / uncorrected[2], uncorrected[1] / uncorrected[2]))
    uv = fov.uv_from_xy(xy_los - xy_uncorrected)
    return (
        float(uv.vals[1] - fov.uv_los.vals[1]),
        float(uv.vals[0] - fov.uv_los.vals[0]),
    )


def observation_attitude(obs: ObsSnapshotInst, et: float) -> np.ndarray:
    """Read the oops observation frame's J2000-to-frame rotation at one epoch.

    Parameters:
        obs: The observation whose frame is evaluated.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation in the oops observation frame convention.
    """
    transform = obs.frame.wrt(oops.frame.Frame.J2000).transform_at_time(et)
    return np.asarray(transform.matrix.vals, np.float64).reshape(3, 3)
