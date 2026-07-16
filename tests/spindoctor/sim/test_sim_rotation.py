"""Camera-roll rendering and the per-scene ``fit_camera_rotation`` override.

A planted camera roll rotates the rendered scene about the boresight before the
translation offset; the simulated star NavModel predicts the unrolled geometry,
so a star technique recovers the roll.  These tests cover the renderer geometry
(a star lands at its analytically rotated position) and the scene-level
``fit_camera_rotation`` override that lets a scene exercise the 3-DoF path on any
emulated camera, independent of that camera's real rotation-fitting flag.
"""

import numpy as np

from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model

_SIZE = 128
_CENTER = _SIZE / 2.0


def _noiseless_params(**overrides: object) -> dict[str, object]:
    """coiss_nac sim params with detector noise suppressed for exact geometry."""
    params: dict[str, object] = {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'bodies': [],
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
    }
    params.update(overrides)
    return params


def _centroid(img: np.ndarray) -> tuple[float, float]:
    """Brightness-weighted centroid of a single-source image."""
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    weights = np.clip(img - np.median(img), 0.0, None)
    total = float(weights.sum())
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


def test_roll_rotates_star_about_boresight() -> None:
    """A 90 deg roll lands a star at its analytically rotated position.

    The star at ``(40, 90)`` is ``(-24, 26)`` from the centre; a +90 deg roll
    (matrix ``[[0, -1], [1, 0]]`` in ``(v, u)``) maps that to ``(-26, -24)``, so
    the rendered centroid must land at ``(38, 40)``.
    """
    params = _noiseless_params(
        offset_rotation_deg=90.0,
        stars=[{'name': 'S', 'v': 40.0, 'u': 90.0, 'vmag': 2.0, 'psf_sigma': 2.0}],
    )
    img, _meta = render_combined_model(params)
    centroid_v, centroid_u = _centroid(img)
    assert abs(centroid_v - 38.0) < 0.05
    assert abs(centroid_u - 40.0) < 0.05


def test_zero_roll_leaves_star_in_place() -> None:
    """A zero roll renders the star at its catalog position (no displacement)."""
    params = _noiseless_params(
        offset_rotation_deg=0.0,
        stars=[{'name': 'S', 'v': 40.0, 'u': 90.0, 'vmag': 2.0, 'psf_sigma': 2.0}],
    )
    img, _meta = render_combined_model(params)
    centroid_v, centroid_u = _centroid(img)
    assert abs(centroid_v - 40.0) < 0.05
    assert abs(centroid_u - 90.0) < 0.05


def test_star_record_keeps_unrolled_position() -> None:
    """The emitted star record carries the unrolled catalog ``(v, u)``.

    The roll is applied to the rendered image only; the NavModel must predict the
    unrolled geometry so a technique recovers the roll rather than cancelling it.
    """
    params = _noiseless_params(
        offset_rotation_deg=30.0,
        stars=[{'name': 'S', 'v': 40.0, 'u': 90.0, 'vmag': 2.0}],
    )
    _img, meta = render_combined_model(params)
    star = meta['stars'][0]
    assert star.v == 40.0
    assert star.u == 90.0


def test_fit_camera_rotation_override_enables_3dof() -> None:
    """The scene override turns on rotation fitting for an emulated coiss_nac."""
    obs = ObsSim.from_file(
        '/tmp/rot_override.json',
        sim_params=_noiseless_params(fit_camera_rotation=True),
    )
    assert instrument_settings_from_obs(obs).fit_camera_rotation is True


def test_fit_camera_rotation_defaults_to_instrument() -> None:
    """Without the override the obs uses the instrument's own flag (coiss: off)."""
    obs = ObsSim.from_file('/tmp/rot_default.json', sim_params=_noiseless_params())
    assert instrument_settings_from_obs(obs).fit_camera_rotation is False
