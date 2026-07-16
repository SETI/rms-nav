"""Stray-light gradient for the simulator (B6).

A scene may add a smooth scattered-light field that the navigator's BANDPASS_DOG
source-image filter is meant to suppress.  These tests cover the field shapes
and confirm the DOG bandpass removes most of the gradient.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.optics import apply_stray_light
from spindoctor.sim.render import render_combined_model
from spindoctor.support.filters import NavFilterKind, NavFilterSpec, apply_filter


def test_linear_creates_a_gradient() -> None:
    """A linear field brightens one edge relative to the opposite edge."""
    img = np.zeros((40, 40), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, direction_deg=0.0, model='linear')
    assert float(img[-1, :].mean()) > float(img[0, :].mean())


def test_linear_direction_rotates_the_ramp() -> None:
    """A 90-degree direction ramps along u instead of v."""
    img = np.zeros((40, 40), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, direction_deg=90.0, model='linear')
    assert float(img[:, -1].mean()) > float(img[:, 0].mean())


def test_radial_peaks_at_center() -> None:
    """A radial field is brightest at its centre and dim at the corners."""
    img = np.zeros((41, 41), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, model='radial')
    assert float(img[20, 20]) > float(img[0, 0])


def test_zero_amplitude_is_a_noop() -> None:
    """Zero amplitude leaves the image untouched."""
    img = np.full((16, 16), 0.2, dtype=np.float64)
    apply_stray_light(img, amplitude=0.0, model='linear')
    assert np.all(img == 0.2)


def test_unknown_model_raises() -> None:
    """An unrecognised model name raises with a clear message."""
    img = np.zeros((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="model must be 'linear' or 'radial'"):
        apply_stray_light(img, amplitude=0.5, model='spiral')


def _half_gradient(image: np.ndarray) -> float:
    """Mean-brightness difference between the bottom and top halves."""
    half = image.shape[0] // 2
    return abs(float(image[half:, :].mean()) - float(image[:half, :].mean()))


def test_bandpass_dog_suppresses_stray_light() -> None:
    """The DOG bandpass removes most of a linear stray-light gradient."""
    img = np.zeros((64, 64), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, direction_deg=0.0, model='linear')
    spec = NavFilterSpec(kind=NavFilterKind.BANDPASS_DOG, bandpass_cutoffs_px=(5.0, 0.7))
    filtered = apply_filter(img, spec)
    assert _half_gradient(filtered) < 0.1 * _half_gradient(img)


def _scene(*, stray: dict[str, Any] | None) -> dict[str, Any]:
    """A coiss scene, optionally with a stray-light block."""
    params: dict[str, Any] = {
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0},
    }
    if stray is not None:
        params['optics'] = {'stray_light': stray}
    return params


def test_render_stray_light_raises_background() -> None:
    """A scene with stray light renders brighter than one without."""
    plain, _ = render_combined_model(_scene(stray=None))
    lit, _ = render_combined_model(
        _scene(stray={'amplitude': 0.3, 'direction_deg': 0.0, 'model': 'linear'})
    )
    assert float(lit.mean()) > float(plain.mean())
