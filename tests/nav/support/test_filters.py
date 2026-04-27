"""Tests for ``nav.support.filters`` (NavFilterSpec, apply_filter)."""

import numpy as np
import pytest

from nav.support.filters import NavFilterKind, NavFilterSpec, apply_filter


def _delta(shape: tuple[int, int]) -> np.ndarray:
    """Return a 2-D delta array (zeros with one center pixel set to 1)."""
    arr = np.zeros(shape, np.float64)
    arr[shape[0] // 2, shape[1] // 2] = 1.0
    return arr


def test_apply_filter_none_kind_returns_input() -> None:
    """NONE kind short-circuits to the input array (identity)."""
    arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    spec = NavFilterSpec(kind=NavFilterKind.NONE)
    out = apply_filter(arr, spec)
    assert out is arr


def test_apply_filter_subnull_sigma_returns_input() -> None:
    """Specs whose sigma is below the null threshold short-circuit to identity."""
    arr = np.arange(16, dtype=np.float64).reshape(4, 4)
    spec = NavFilterSpec(kind=NavFilterKind.ISOTROPIC_GAUSSIAN, sigma_xy=(0.1, 0.1))
    out = apply_filter(arr, spec)
    assert out is arr


def test_apply_filter_isotropic_gaussian_blurs_delta() -> None:
    """An isotropic Gaussian smears a delta input into a positive blob."""
    arr = _delta((11, 11))
    spec = NavFilterSpec(kind=NavFilterKind.ISOTROPIC_GAUSSIAN, sigma_xy=(1.0, 1.0))
    out = apply_filter(arr, spec)
    assert out.shape == arr.shape
    assert out[5, 5] < 1.0
    assert out[5, 5] > 0.0
    # Energy is preserved by gaussian_filter (within floating tolerance).
    assert np.isclose(out.sum(), arr.sum(), rtol=1e-6)


def test_apply_filter_anisotropic_gaussian_axis_aligned() -> None:
    """Axis-aligned anisotropic Gaussian blurs more along the wider axis."""
    arr = _delta((21, 21))
    cov = np.array([[1.0, 0.0], [0.0, 9.0]], np.float64)
    spec = NavFilterSpec(kind=NavFilterKind.ANISOTROPIC_GAUSSIAN, covariance_px2=cov)
    out = apply_filter(arr, spec)
    # Profile along v-axis (small sigma) is narrower than along u-axis.
    v_profile = out[:, 10]
    u_profile = out[10, :]
    assert v_profile[10] > u_profile[10] / 2  # peak narrower in v


def test_apply_filter_anisotropic_gaussian_requires_covariance() -> None:
    """Missing covariance_px2 raises a clear ValueError."""
    arr = _delta((11, 11))
    # null_filter_threshold_sigma=0 forces the dispatch past the null-threshold
    # short-circuit and into the operation branch where the missing covariance
    # is detected.
    spec = NavFilterSpec(
        kind=NavFilterKind.ANISOTROPIC_GAUSSIAN,
        null_filter_threshold_sigma=0.0,
    )
    with pytest.raises(ValueError, match='covariance_px2'):
        apply_filter(arr, spec)


def test_apply_filter_bandpass_dog_zero_mean() -> None:
    """DoG bandpass on a uniform image yields ~zero output."""
    arr = np.ones((16, 16), np.float64) * 100.0
    spec = NavFilterSpec(kind=NavFilterKind.BANDPASS_DOG, bandpass_cutoffs_px=(8.0, 1.0))
    out = apply_filter(arr, spec)
    # DoG of uniform background is exactly zero everywhere.
    assert np.max(np.abs(out)) < 1e-9


def test_apply_filter_bandpass_dog_validates_cutoffs() -> None:
    """BANDPASS_DOG with lo <= hi raises with a helpful message."""
    arr = np.zeros((4, 4), np.float64)
    spec = NavFilterSpec(
        kind=NavFilterKind.BANDPASS_DOG,
        bandpass_cutoffs_px=(1.0, 1.0),
        null_filter_threshold_sigma=0.0,
    )
    with pytest.raises(ValueError, match='lo > hi > 0'):
        apply_filter(arr, spec)


def test_apply_filter_gradient_of_gaussian_finds_step() -> None:
    """Gradient-of-Gaussian peaks at the location of a step edge."""
    # Use a strip of constant value with the same value continuing all the way
    # to one boundary, so the only intra-image edge is the central step.
    arr = np.ones((21, 41), np.float64)
    arr[:, :20] = 0.0
    spec = NavFilterSpec(kind=NavFilterKind.GRADIENT_OF_GAUSSIAN, sigma_xy=(1.0, 1.0))
    out = apply_filter(arr, spec)
    # Restrict to a window around the step (avoiding boundary effects).
    peak_u = int(np.argmax(out[10, 5:35])) + 5
    assert peak_u in {19, 20}


def test_apply_filter_distance_transform_known_distance() -> None:
    """DT with an isolated edge pixel yields Euclidean distances."""
    arr = np.zeros((9, 9), np.float64)
    arr[4, 4] = 1.0
    spec = NavFilterSpec(kind=NavFilterKind.DISTANCE_TRANSFORM, dt_half_width_px=10.0)
    out = apply_filter(arr, spec)
    # Edge pixel itself has distance 0.
    assert out[4, 4] == 0.0
    # Distance from corner to center is sqrt((4)**2 + (4)**2) ≈ 5.6568.
    assert np.isclose(out[0, 0], np.sqrt(32.0), atol=1e-6)


def test_apply_filter_distance_transform_clips_to_half_width() -> None:
    """DT clips values to ``dt_half_width_px``."""
    arr = np.zeros((9, 9), np.float64)
    arr[4, 4] = 1.0
    spec = NavFilterSpec(kind=NavFilterKind.DISTANCE_TRANSFORM, dt_half_width_px=2.0)
    out = apply_filter(arr, spec)
    assert out.max() <= 2.0


def test_apply_filter_distance_transform_no_edge_pixels() -> None:
    """DT on an all-zero array returns the saturated half-width everywhere."""
    arr = np.zeros((4, 4), np.float64)
    spec = NavFilterSpec(kind=NavFilterKind.DISTANCE_TRANSFORM, dt_half_width_px=3.5)
    out = apply_filter(arr, spec)
    assert np.all(out == 3.5)


def test_apply_filter_morph_dilate_grows_structuring_element() -> None:
    """Dilation with sigma_xy=2 expands an isolated 1 to a 5x5 region."""
    arr = np.zeros((9, 9), np.float64)
    arr[4, 4] = 1.0
    spec = NavFilterSpec(kind=NavFilterKind.MORPH_DILATE, sigma_xy=(2.0, 2.0))
    out = apply_filter(arr, spec)
    # 5x5 centered on (4, 4) is set to 1.0.
    assert np.allclose(out[2:7, 2:7], 1.0)


def test_apply_filter_rejects_non_2d_input() -> None:
    """Non-2D input raises TypeError with a clear message."""
    arr = np.zeros((3, 3, 3), np.float64)
    spec = NavFilterSpec(kind=NavFilterKind.NONE)
    with pytest.raises(TypeError, match='2-D'):
        apply_filter(arr, spec)
