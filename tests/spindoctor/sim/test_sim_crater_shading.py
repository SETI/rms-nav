"""Crater and no-crater shading share one illumination convention (SIM-3).

The cratered shading path derives its base-surface normals from the same
analytic formula (including the rotation_z back-rotation to image coordinates)
as the smooth path, so a cratered body's lit hemisphere and terminator match
the same body rendered without craters at any pose.  Before the fix the crater
path shaded against the raw depth-field gradient, whose frame diverges from
the smooth path's for rotated poses.
"""

import numpy as np

from spindoctor.sim.sim_body import create_simulated_body

_SIZE = (101, 101)
_CENTER = (50.5, 50.5)
_AXES = (70.0, 30.0, 30.0)
_ILLUMINATION_ANGLE = np.pi / 4
_PHASE_ANGLE = np.pi / 2
_CENTROID_TOLERANCE = 0.5


def _render(*, rotation_z: float, rotation_tilt: float, crater_fill: float) -> np.ndarray:
    """Render the reference body at the given pose and crater fill."""
    return create_simulated_body(
        _SIZE,
        _CENTER,
        *_AXES,
        rotation_z=rotation_z,
        rotation_tilt=rotation_tilt,
        illumination_angle=_ILLUMINATION_ANGLE,
        phase_angle=_PHASE_ANGLE,
        crater_fill=crater_fill,
        anti_aliasing=1.0,
        seed=5,
    )


def _bright_centroid(img: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid of the bright (lit) side of the body."""
    bright = img * (img > 0.5 * img.max())
    vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    total = float(bright.sum())
    return float((bright * vv).sum() / total), float((bright * uu).sum() / total)


def test_rotated_cratered_body_matches_smooth_bright_centroid_v() -> None:
    """At rotation_z = pi/2 the cratered lit hemisphere matches the smooth one (v)."""
    smooth = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.0)
    cratered = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.3)
    assert abs(_bright_centroid(cratered)[0] - _bright_centroid(smooth)[0]) < _CENTROID_TOLERANCE


def test_rotated_cratered_body_matches_smooth_bright_centroid_u() -> None:
    """At rotation_z = pi/2 the cratered lit hemisphere matches the smooth one (u).

    Before the shared-convention fix this centroid was displaced by roughly 9
    pixels for this pose, so the tolerance is a strong regression guard.
    """
    smooth = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.0)
    cratered = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.3)
    assert abs(_bright_centroid(cratered)[1] - _bright_centroid(smooth)[1]) < _CENTROID_TOLERANCE


def test_rotated_cratered_body_actually_has_craters() -> None:
    """The rotated cratered render differs from the smooth one (test premise)."""
    smooth = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.0)
    cratered = _render(rotation_z=np.pi / 2, rotation_tilt=1.0, crater_fill=0.3)
    assert not np.array_equal(cratered, smooth)


def test_unrotated_cratered_body_matches_smooth_bright_centroid_v() -> None:
    """At rotation_z = 0 the crater and smooth paths agree as before (v)."""
    smooth = _render(rotation_z=0.0, rotation_tilt=0.0, crater_fill=0.0)
    cratered = _render(rotation_z=0.0, rotation_tilt=0.0, crater_fill=0.3)
    assert abs(_bright_centroid(cratered)[0] - _bright_centroid(smooth)[0]) < _CENTROID_TOLERANCE


def test_unrotated_cratered_body_matches_smooth_bright_centroid_u() -> None:
    """At rotation_z = 0 the crater and smooth paths agree as before (u)."""
    smooth = _render(rotation_z=0.0, rotation_tilt=0.0, crater_fill=0.0)
    cratered = _render(rotation_z=0.0, rotation_tilt=0.0, crater_fill=0.3)
    assert abs(_bright_centroid(cratered)[1] - _bright_centroid(smooth)[1]) < _CENTROID_TOLERANCE
