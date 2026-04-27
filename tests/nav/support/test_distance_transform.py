"""Tests for ``nav.support.distance_transform`` helpers."""

import numpy as np
import pytest

from nav.support.distance_transform import apply_translation, sample_dt_bilinear


def test_apply_translation_shifts_vertices() -> None:
    """apply_translation moves every vertex by (dv, du)."""
    verts = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = apply_translation(verts, 1.5, -2.0)
    expected = np.array([[11.5, 18.0], [31.5, 38.0]])
    assert np.allclose(out, expected)


def test_apply_translation_preserves_input() -> None:
    """apply_translation does not modify its input array."""
    verts = np.array([[10.0, 20.0]])
    original = verts.copy()
    apply_translation(verts, 5.0, 5.0)
    assert np.allclose(verts, original)


def test_apply_translation_rejects_wrong_shape() -> None:
    """A non-(N, 2) input raises ValueError."""
    verts = np.zeros((4, 3))
    with pytest.raises(ValueError, match=r'\(N, 2\)'):
        apply_translation(verts, 0.0, 0.0)


def test_sample_dt_bilinear_at_integer_position() -> None:
    """Sampling at an integer vertex returns the exact DT value."""
    dt = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    verts = np.array([[1.0, 1.0]])
    out = sample_dt_bilinear(dt, verts)
    assert np.isclose(out[0], 2.0)


def test_sample_dt_bilinear_interpolates_subpixel() -> None:
    """Sampling at a half-pixel offset returns the bilinear average."""
    dt = np.array([[0.0, 0.0], [4.0, 4.0]])
    verts = np.array([[0.5, 0.5]])
    out = sample_dt_bilinear(dt, verts)
    assert np.isclose(out[0], 2.0)


def test_sample_dt_bilinear_clamps_out_of_bounds() -> None:
    """Out-of-bounds vertices are clamped to the closest in-bounds pixel."""
    dt = np.array([[0.0, 0.0], [0.0, 0.0]])
    verts = np.array([[-5.0, -5.0]])
    out = sample_dt_bilinear(dt, verts)
    assert out[0] == 0.0


def test_sample_dt_bilinear_rejects_non_2d_dt() -> None:
    """A non-2D DT array raises ValueError."""
    dt = np.zeros((4, 4, 4))
    verts = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError, match='2-D'):
        sample_dt_bilinear(dt, verts)


def test_sample_dt_bilinear_rejects_wrong_vertex_shape() -> None:
    """A non-(N, 2) vertex array raises ValueError."""
    dt = np.zeros((4, 4))
    verts = np.zeros((4, 3))
    with pytest.raises(ValueError, match=r'\(N, 2\)'):
        sample_dt_bilinear(dt, verts)
