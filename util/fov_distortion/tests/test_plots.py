"""Smoke tests for the figure writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from util.fov_distortion.plots import (
    plot_frame_decomposition,
    plot_instrument_distortion_map,
    plot_instrument_radial,
    plot_instrument_twist,
)
from util.fov_distortion.results import summarize_instrument
from util.fov_distortion.tests._synthetic import make_frame


def test_plot_instrument_twist_writes_png(tmp_path: Path) -> None:
    frames = [make_frame(twist_deg=0.05, k1=0.002, seed=i, name=f'F{i}') for i in range(6)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    out = tmp_path / 'twist.png'
    plot_instrument_twist(summary, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_instrument_radial_writes_png(tmp_path: Path) -> None:
    frames = [make_frame(twist_deg=0.05, k1=0.01, seed=i, name=f'F{i}') for i in range(6)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    out = tmp_path / 'radial.png'
    plot_instrument_radial(summary, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_instrument_distortion_map_writes_png(tmp_path: Path) -> None:
    frames = [make_frame(twist_deg=0.05, k1=0.01, seed=i, name=f'F{i}') for i in range(6)]
    summary = summarize_instrument('syn', 'Synthetic', frames)
    out = tmp_path / 'distortion_map.png'
    plot_instrument_distortion_map(summary, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_frame_decomposition_writes_png(tmp_path: Path) -> None:
    frame = make_frame(twist_deg=0.1, k1=0.01, seed=1)
    image = np.zeros(frame.image_shape, dtype=np.float64)
    out = tmp_path / 'sample.png'
    plot_frame_decomposition(frame, image, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_frame_decomposition_requires_decomposition(tmp_path: Path) -> None:
    frame = make_frame(twist_deg=0.1, k1=0.01, seed=1)
    from dataclasses import replace

    bare = replace(frame, decomposition=None)
    with pytest.raises(ValueError, match='no decomposition'):
        plot_frame_decomposition(bare, np.zeros(frame.image_shape), str(tmp_path / 'x.png'))
