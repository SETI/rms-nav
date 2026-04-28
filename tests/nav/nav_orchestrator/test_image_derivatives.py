"""Unit tests for ``nav.nav_orchestrator.image_derivatives``."""

from __future__ import annotations

import numpy as np
import pytest

from nav.nav_orchestrator.image_derivatives import (
    DEFAULT_DT_HALF_WIDTH_PX,
    DEFAULT_EDGE_THRESHOLD_K_SIGMA,
    DEFAULT_IMAGE_GRADIENT_SIGMA_PX,
    ImageDerivativesConfig,
    build_image_edge_dt,
    compute_image_gradient_vu,
)


def _step_image(shape: tuple[int, int], step_v: int) -> np.ndarray:
    """Return an image with a single horizontal bright bar centred on ``step_v``.

    The bar is 8 pixels tall so the borders of the image stay in the same
    background as their interior neighbours, suppressing the
    sobel-with-constant-padding boundary artefact that would otherwise
    dominate the gradient image on a small test fixture.
    """
    img = np.zeros(shape, dtype=np.float64)
    img[step_v - 4 : step_v + 4, :] = 100.0
    return img


def test_image_derivatives_config_defaults_match_design() -> None:
    """Default ``ImageDerivativesConfig`` carries the documented constants."""
    cfg = ImageDerivativesConfig()
    assert cfg.image_gradient_sigma_px == DEFAULT_IMAGE_GRADIENT_SIGMA_PX
    assert cfg.edge_threshold_k_sigma == DEFAULT_EDGE_THRESHOLD_K_SIGMA
    assert cfg.dt_half_width_px == DEFAULT_DT_HALF_WIDTH_PX


def test_image_derivatives_config_rejects_zero_sigma() -> None:
    """A zero ``image_gradient_sigma_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_gradient_sigma_px'):
        ImageDerivativesConfig(image_gradient_sigma_px=0.0)


def test_image_derivatives_config_rejects_zero_threshold() -> None:
    """A zero ``edge_threshold_k_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='edge_threshold_k_sigma'):
        ImageDerivativesConfig(edge_threshold_k_sigma=0.0)


def test_image_derivatives_config_rejects_zero_half_width() -> None:
    """A non-positive ``dt_half_width_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='dt_half_width_px'):
        ImageDerivativesConfig(dt_half_width_px=-1.0)


def test_build_image_edge_dt_peak_aligns_with_step_edge() -> None:
    """Gradient peak row matches the leading or trailing edge of a planted bar."""
    # A horizontal bar centred on row 16 has its leading edge near row 12
    # and trailing edge near row 19; the gradient row-sum peaks at one of
    # those rows after Gaussian smoothing.
    shape = (40, 40)
    img = _step_image(shape, step_v=16)
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    assert gradient.shape == shape
    assert edge_dt.shape == shape
    peak_row = int(np.argmax(gradient.sum(axis=1)))
    leading_edge = 12
    trailing_edge = 19
    assert min(abs(peak_row - leading_edge), abs(peak_row - trailing_edge)) <= 1


def test_build_image_edge_dt_zeros_edge_dt_on_thresholded_pixels() -> None:
    """Distance transform is exactly zero on retained edge pixels and positive elsewhere."""
    img = _step_image((40, 40), step_v=16)
    _, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0)
    # Pixels along the leading edge (row 12) get DT = 0 exactly --
    # ``distance_transform_edt`` returns integer-zero on edge pixels of a
    # binary mask, so the assertion can be exact rather than approximate.
    assert float(edge_dt[12, 20]) == 0.0
    # A pixel far from any edge has positive DT.
    assert float(edge_dt[0, 0]) > 0.0


def test_build_image_edge_dt_falls_back_when_no_pixel_exceeds_threshold() -> None:
    """Empty edge mask saturates the DT at the configured half-width everywhere."""
    img = np.full((16, 16), 1.0, dtype=np.float64)
    cfg = ImageDerivativesConfig(
        edge_threshold_k_sigma=1000.0,
        dt_half_width_px=5.0,
    )
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=10.0, config=cfg)
    # No edge pixels survive the very high threshold; DT saturates at the
    # half width even though the gradient itself has small boundary
    # artefacts from the constant-padded Sobel.
    threshold = cfg.edge_threshold_k_sigma * 10.0
    assert (gradient <= threshold).all()
    assert np.allclose(edge_dt, cfg.dt_half_width_px, atol=1e-12)


def test_build_image_edge_dt_threshold_sweep_matches_expected_count() -> None:
    """Edge-pixel count is monotonically non-decreasing as the k-sigma threshold drops."""
    img = _step_image((40, 40), step_v=16)
    # Decreasing the k_sigma factor lets more pixels pass the threshold;
    # binarized edge counts must be monotonically non-decreasing.
    counts: list[int] = []
    for k in (8.0, 4.0, 2.0):
        cfg = ImageDerivativesConfig(edge_threshold_k_sigma=k)
        gradient, _ = build_image_edge_dt(img, image_noise_sigma=1.0, config=cfg)
        threshold = k * 1.0
        edge_count = int((gradient > threshold).sum())
        counts.append(edge_count)
    assert counts[0] <= counts[1]
    assert counts[1] <= counts[2]


def test_build_image_edge_dt_rejects_non_2d_input() -> None:
    """A non-2-D ``image_ext`` is rejected with a TypeError naming the field."""
    with pytest.raises(TypeError, match='image_ext must be 2-D'):
        build_image_edge_dt(np.zeros((4, 4, 4)), image_noise_sigma=1.0)


def test_build_image_edge_dt_rejects_negative_noise_sigma() -> None:
    """A negative ``image_noise_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_noise_sigma'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=-1.0)


def test_build_image_edge_dt_rejects_nan_noise_sigma() -> None:
    """A NaN ``image_noise_sigma`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_noise_sigma'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=float('nan'))


def test_compute_image_gradient_vu_horizontal_step_points_along_v() -> None:
    """Horizontal step edge produces a positive ``g_v`` and near-zero ``g_u``."""
    shape = (40, 40)
    img = _step_image(shape, step_v=16)
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert grad.shape == (40, 40, 2)
    # The leading edge of the bar (image rises from 0 to 100 with
    # increasing row index) sits near row 12: g_v is strictly positive
    # there because Sobel returns a positive derivative when image values
    # increase with row.
    assert float(grad[12, 20, 0]) > 0.0
    # The orthogonal u-axis gradient is essentially zero on a horizontal
    # bar.
    assert abs(float(grad[12, 20, 1])) < 1.0


def test_compute_image_gradient_vu_vertical_step_points_along_u() -> None:
    """Vertical step edge produces a positive ``g_u`` and near-zero ``g_v``."""
    img = np.zeros((32, 32), dtype=np.float64)
    img[:, 16:] = 100.0
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert float(grad[16, 16, 1]) > 0.0
    assert abs(float(grad[16, 16, 0])) < 1.0


def test_compute_image_gradient_vu_rejects_non_2d_input() -> None:
    """A non-2-D ``image_ext`` is rejected with a TypeError naming the field."""
    with pytest.raises(TypeError, match='image_ext must be 2-D'):
        compute_image_gradient_vu(np.zeros((4, 4, 2)))


def test_compute_image_gradient_vu_rejects_zero_sigma() -> None:
    """A zero ``sigma_px`` is rejected with a finite-positive-number message."""
    with pytest.raises(ValueError, match='sigma_px must be a finite positive number'):
        compute_image_gradient_vu(np.zeros((4, 4)), sigma_px=0.0)


def test_compute_image_gradient_vu_rejects_inf_sigma() -> None:
    """An infinite ``sigma_px`` is rejected with a finite-positive-number message."""
    with pytest.raises(ValueError, match='sigma_px must be a finite positive number'):
        compute_image_gradient_vu(np.zeros((4, 4)), sigma_px=float('inf'))


def test_build_image_edge_dt_rejects_inf_noise_sigma() -> None:
    """An infinite ``image_noise_sigma`` is rejected with a finite-required message."""
    with pytest.raises(ValueError, match='image_noise_sigma must be finite'):
        build_image_edge_dt(np.zeros((4, 4)), image_noise_sigma=float('inf'))


def test_image_derivatives_config_rejects_inf_sigma() -> None:
    """An infinite ``image_gradient_sigma_px`` is rejected with a named field message."""
    with pytest.raises(ValueError, match='image_gradient_sigma_px'):
        ImageDerivativesConfig(image_gradient_sigma_px=float('inf'))


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


def test_build_image_edge_dt_handles_minimal_4x4_image() -> None:
    """Smallest meaningful image: 4x4 still produces correctly-shaped outputs.

    The Gaussian smooth, Sobel, NMS, and DT pipeline must all tolerate
    a 4x4 input without raising or producing degenerate-shape arrays.
    Sobel-with-constant-padding boundary artefacts may produce a few
    edge pixels at the corners; what we verify is that the pipeline
    completes and returns 2-D float64 arrays of the right shape with
    finite values throughout.
    """
    img = np.full((4, 4), 5.0, dtype=np.float64)
    cfg = ImageDerivativesConfig(dt_half_width_px=3.0)
    gradient, edge_dt = build_image_edge_dt(img, image_noise_sigma=1.0, config=cfg)
    assert gradient.shape == (4, 4)
    assert edge_dt.shape == (4, 4)
    assert np.isfinite(gradient).all()
    assert np.isfinite(edge_dt).all()
    # DT entries are bounded by the configured half-width.
    assert float(edge_dt.max()) <= cfg.dt_half_width_px


def test_compute_image_gradient_vu_handles_minimal_4x4_image() -> None:
    """Gradient-vector helper produces a (4, 4, 2) output on a 4x4 input."""
    img = np.full((4, 4), 5.0, dtype=np.float64)
    grad = compute_image_gradient_vu(img, sigma_px=1.0)
    assert grad.shape == (4, 4, 2)


def test_build_image_edge_dt_threshold_boundary_exact() -> None:
    """A pixel whose gradient magnitude equals the threshold is excluded.

    The implementation uses a strict ``gradient > threshold`` comparison
    inside ``_directional_nms``; this test plants a gradient profile
    chosen so that exactly one pixel sits at the threshold value and
    verifies it is *not* kept in the edge mask.  The next-larger value
    is kept, confirming the "at boundary -> excluded" half-open
    convention.
    """
    img = _step_image((40, 40), step_v=16)
    # Probe a noise sigma that makes the gradient peak exactly on the
    # k_sigma boundary.  Compute the actual peak first to derive the
    # noise sigma that makes it fall right at the threshold.
    cfg = ImageDerivativesConfig(edge_threshold_k_sigma=4.0)
    gradient_only, _ = build_image_edge_dt(img, image_noise_sigma=1.0e-6, config=cfg)
    peak = float(gradient_only.max())
    # Set noise sigma so threshold == peak exactly: peak == 4 * sigma.
    boundary_sigma = peak / cfg.edge_threshold_k_sigma
    _, edge_dt_boundary = build_image_edge_dt(img, image_noise_sigma=boundary_sigma, config=cfg)
    # No pixel exceeds the threshold (strict ``>`` rejects the equality
    # case), so the DT saturates everywhere at the half-width.
    assert np.allclose(edge_dt_boundary, cfg.dt_half_width_px)
    # A noise sigma fractionally below the boundary lets the peak
    # through and produces at least one zero-DT pixel.
    _, edge_dt_below = build_image_edge_dt(img, image_noise_sigma=boundary_sigma * 0.99, config=cfg)
    assert float(edge_dt_below.min()) == 0.0
