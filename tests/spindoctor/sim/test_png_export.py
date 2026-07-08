"""Tests for the simulated-image PNG export helpers."""

from pathlib import Path

import numpy as np
from PIL import Image

from spindoctor.sim.png_export import save_png, stretch_to_uint8


def test_stretch_returns_uint8_same_shape() -> None:
    """The stretch returns a uint8 array of the input shape."""
    out = stretch_to_uint8(np.linspace(0.0, 10.0, 100).reshape(10, 10))
    assert out.dtype == np.uint8
    assert out.shape == (10, 10)


def test_stretch_spans_full_range() -> None:
    """A smooth gradient stretches to span the 8-bit range."""
    out = stretch_to_uint8(
        np.linspace(0.0, 1.0, 256).reshape(16, 16), low_percentile=0.0, high_percentile=100.0
    )
    assert int(out.min()) == 0
    assert int(out.max()) == 255


def test_stretch_ignores_hot_pixel() -> None:
    """A single hot pixel does not crush a body with contrast to black."""
    img = np.linspace(50.0, 150.0, 400).reshape(20, 20)
    img[0, 0] = 1.0e6
    out = stretch_to_uint8(img)
    # The hot pixel is clipped to white, and the bright end of the body (just
    # below the 99.5th percentile) is still near white rather than crushed.
    assert int(out[19, 19]) > 200


def test_stretch_handles_nan() -> None:
    """Non-finite pixels map to the black point without raising."""
    img = np.full((8, 8), 50.0)
    img[0, 0] = np.nan
    out = stretch_to_uint8(img)
    assert int(out[0, 0]) == 0


def test_stretch_constant_image_is_black() -> None:
    """A constant image has no contrast and returns all zeros."""
    out = stretch_to_uint8(np.full((8, 8), 7.0))
    assert int(out.max()) == 0


def test_gamma_brightens_midtones() -> None:
    """A gamma above 1 raises a mid-level pixel's output."""
    img = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    plain = stretch_to_uint8(img, low_percentile=0.0, high_percentile=100.0, gamma=1.0)
    lifted = stretch_to_uint8(img, low_percentile=0.0, high_percentile=100.0, gamma=2.0)
    assert int(lifted[8, 0]) > int(plain[8, 0])


def test_save_png_writes_upscaled_grayscale(tmp_path: Path) -> None:
    """save_png writes a grayscale PNG at the requested magnification."""
    out = save_png(np.full((10, 12), 5.0), tmp_path / 'out.png', upscale=3)
    assert out.is_file()
    with Image.open(out) as im:
        assert im.mode == 'L'
        assert im.size == (36, 30)
