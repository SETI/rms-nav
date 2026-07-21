"""Unit tests for the measurement helpers that do not need SPICE.

The full :func:`measure_frame` path requires holdings, kernels, and star
catalogs and is exercised by the campaign driver rather than here; these tests
cover the pieces that run on a synthetic image.
"""

from __future__ import annotations

import numpy as np
import pytest
from psfmodel import GaussianPSF
from util.fov_distortion.measure import MeasureParams, _centroid_star, _odd_box


def _gaussian_star(
    shape: tuple[int, int], center_vu: tuple[float, float], sigma: float
) -> np.ndarray:
    """A single Gaussian star on a flat low background."""
    vv, uu = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    v0, u0 = center_vu
    star = np.exp(-((vv - v0) ** 2 + (uu - u0) ** 2) / (2.0 * sigma**2))
    image: np.ndarray = 5.0 + 1000.0 * star
    return image


def test_odd_box_is_odd() -> None:
    box = _odd_box(7)
    assert box == (15, 15)


def test_centroid_recovers_relative_star_positions() -> None:
    # The analysis relies on the centroider being accurate in the *differences*
    # between star positions (a constant pixel-convention offset is absorbed by
    # the rigid-fit translation).  Inject two stars a known vector apart and
    # check the detected separation matches to well under a tenth of a pixel.
    shape = (96, 96)
    truth_a = (40.0, 40.0)
    truth_b = (61.3, 68.6)
    psf = GaussianPSF(sigma=1.5)
    params = MeasureParams(box_half_px=7, search_limit_px=2.5)
    det_a = _centroid_star(_gaussian_star(shape, truth_a, 1.5), psf, (40.0, 40.0), params)
    det_b = _centroid_star(_gaussian_star(shape, truth_b, 1.5), psf, (61.0, 69.0), params)
    assert det_a is not None
    assert det_b is not None
    (av, au), _ = det_a
    (bv, bu), _ = det_b
    assert (bv - av) == pytest.approx(truth_b[0] - truth_a[0], abs=0.05)
    assert (bu - au) == pytest.approx(truth_b[1] - truth_a[1], abs=0.05)


def test_centroid_rejects_blank_region() -> None:
    shape = (64, 64)
    image = np.full(shape, 5.0, dtype=np.float64)
    psf = GaussianPSF(sigma=1.5)
    params = MeasureParams(box_half_px=7, min_peak_over_background=1.3)
    result = _centroid_star(image, psf, (32.0, 32.0), params)
    assert result is None


def test_centroid_rejects_out_of_bounds() -> None:
    shape = (64, 64)
    image = _gaussian_star(shape, (3.0, 3.0), sigma=1.5)
    psf = GaussianPSF(sigma=1.5)
    params = MeasureParams(box_half_px=7)
    result = _centroid_star(image, psf, (2.0, 2.0), params)
    assert result is None
