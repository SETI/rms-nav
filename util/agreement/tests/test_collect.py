"""Tests for the agreement collector's injection construction.

Covers the per-scene injection draws and the PSF-layer sim_params rewrite
without rendering a scene (rendering is exercised by the campaign itself).
Requires the spindoctor package on the path; run from the repo checkout with
``PYTHONPATH=src``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / 'src'))

from collect import _apply_injection, _draw_injection, _inject_sim_params  # noqa: E402
from scene_gen import generate_scenes  # noqa: E402

_SEED = 20260719


def _psf_scene() -> dict[str, object]:
    """One validated limb_disc_psf scene's sim_params."""
    _, params, _ = generate_scenes('limb_disc_psf', 1, campaign_seed=_SEED)[0]
    return params


def test_draw_none_is_disabled() -> None:
    """The 'none' draw carries no bias."""
    assert _draw_injection('none', 'scene', _SEED, 0.7) == {'kind': 'none'}


def test_draw_psf_broaden_is_deterministic() -> None:
    """The same scene/seed redraws the same broadening factor."""
    a = _draw_injection('psf_broaden', 'limb_disc_psf_00000', _SEED, 0.7)
    b = _draw_injection('psf_broaden', 'limb_disc_psf_00000', _SEED, 0.7)
    assert a == b


def test_draw_psf_broaden_factor_in_range() -> None:
    """The broadening factor stays inside its declared log-uniform range."""
    draw = _draw_injection('psf_broaden', 'limb_disc_psf_00003', _SEED, 0.7)
    assert draw['kind'] == 'psf_broaden'
    assert 1.4 <= draw['factor'] <= 3.0


def test_draw_psf_aniso_carries_an_axis() -> None:
    """The anisotropic draw names the broadened axis."""
    draw = _draw_injection('psf_aniso', 'limb_disc_psf_00001', _SEED, 0.7)
    assert draw['kind'] == 'psf_aniso'
    assert draw['axis'] in ('v', 'u')
    assert 1.6 <= draw['factor'] <= 3.5


def test_inject_non_psf_passes_through() -> None:
    """A DT-layer injection leaves the render params (and applied) untouched."""
    params = _psf_scene()
    draw = {'kind': 'dt_shift', 'bias_v': 0.1, 'bias_u': -0.2}
    render_params, applied = _inject_sim_params(params, draw)
    assert render_params is params
    assert applied == {}


def test_inject_psf_broaden_is_isotropic() -> None:
    """psf_broaden scales both core sigmas by the factor above the matched sigma."""
    params = _psf_scene()
    draw = {'kind': 'psf_broaden', 'factor': 2.0}
    render_params, applied = _inject_sim_params(params, draw)
    psf = render_params['optics']['psf']
    assert psf['sigma_v'] == pytest.approx(applied['base_sigma_px'] * 2.0)
    assert psf['sigma_u'] == pytest.approx(applied['base_sigma_px'] * 2.0)
    assert psf['w'] == 0.0


def test_inject_psf_broaden_exceeds_matched_sigma() -> None:
    """The rendered sigma is broader than the navigator's matched sigma."""
    params = _psf_scene()
    _, applied = _inject_sim_params(params, {'kind': 'psf_broaden', 'factor': 1.5})
    assert applied['sigma_v_px'] > applied['base_sigma_px']


def test_inject_psf_aniso_broadens_one_axis() -> None:
    """psf_aniso broadens only the named axis (a zero-mean elliptical kernel)."""
    params = _psf_scene()
    render_params, applied = _inject_sim_params(
        params, {'kind': 'psf_aniso', 'factor': 2.5, 'axis': 'v'}
    )
    psf = render_params['optics']['psf']
    assert psf['sigma_v'] == pytest.approx(applied['base_sigma_px'] * 2.5)
    assert psf['sigma_u'] == pytest.approx(applied['base_sigma_px'])


def test_inject_psf_does_not_mutate_input() -> None:
    """The injection deep-copies: the source scene keeps its matched-PSF block."""
    params = _psf_scene()
    _inject_sim_params(params, {'kind': 'psf_broaden', 'factor': 2.0})
    assert params['optics']['psf'] == {'match_navigator': True}


def test_inject_psf_base_is_the_navigator_nac_sigma() -> None:
    """The broadening base is the emulated Cassini NAC's configured star_psf_sigma.

    Pinned against the config value directly (not the code's own report) so a
    wrong base -- e.g. the WAC 0.77 px or the sim-default 1.0 px instead of the
    navigator's 0.54 px NAC belief -- would fail here rather than pass silently.
    """
    params = _psf_scene()
    _, applied = _inject_sim_params(params, {'kind': 'psf_broaden', 'factor': 2.0})
    assert applied['base_sigma_px'] == pytest.approx(0.54)


def test_apply_injection_leaves_nav_side_unpatched_for_psf() -> None:
    """A PSF injection acts on the render only; it adds no orchestrator patch."""
    assert _apply_injection({'kind': 'psf_broaden', 'factor': 2.0}) == []


def test_apply_injection_leaves_nav_side_unpatched_for_aniso() -> None:
    """The anisotropic PSF injection also touches only the render side."""
    assert _apply_injection({'kind': 'psf_aniso', 'factor': 2.0, 'axis': 'v'}) == []
