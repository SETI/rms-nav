"""Tests for nav.support.correlate, focused on masked NCC and pyramid navigation."""

import numpy as np
import pytest

from nav.support.correlate import (
    masked_ncc,
    navigate_single_scale_kpeaks,
    navigate_with_pyramid_kpeaks,
)
from nav.support.image import pad_top_left

# =========================================================================
# Helpers
# =========================================================================


def _gaussian_patch(
    shape: tuple[int, int],
    sigma: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Create a 2-D Gaussian patch with optional subpixel offset."""
    v_size, u_size = shape
    cv = (v_size - 1) / 2.0
    cu = (u_size - 1) / 2.0
    vv, uu = np.meshgrid(np.arange(v_size), np.arange(u_size), indexing='ij')
    dv = vv - (cv + offset[0])
    du = uu - (cu + offset[1])
    return np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))


def _make_single_star(
    *,
    image_size: tuple[int, int] = (64, 64),
    model_size: tuple[int, int] = (64, 64),
    star_sigma: float = 2.0,
    mask_half: int = 15,
    image_offset: tuple[float, float] = (1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic single-star scene with known offset."""
    ih, iw = image_size
    mh, mw = model_size
    psf_size = (2 * mask_half + 1, 2 * mask_half + 1)

    image = np.zeros(image_size, dtype=np.float64)
    icv, icu = ih // 2, iw // 2
    image[
        icv - mask_half : icv + mask_half + 1,
        icu - mask_half : icu + mask_half + 1,
    ] = _gaussian_patch(psf_size, star_sigma, offset=image_offset)

    model = np.zeros(model_size, dtype=np.float64)
    mask = np.zeros(model_size, dtype=bool)
    mcv, mcu = mh // 2, mw // 2
    model[
        mcv - mask_half : mcv + mask_half + 1,
        mcu - mask_half : mcu + mask_half + 1,
    ] = _gaussian_patch(psf_size, star_sigma, offset=(0.0, 0.0))
    mask[
        mcv - mask_half : mcv + mask_half + 1,
        mcu - mask_half : mcu + mask_half + 1,
    ] = True

    return image, model, mask


def _make_multi_star(
    *,
    image_size: tuple[int, int] = (64, 64),
    star_sigma: float = 2.0,
    mask_half: int = 5,
    image_offset: tuple[float, float] = (1.0, 0.0),
    positions: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic multi-star scene with all stars sharing the same offset."""
    if positions is None:
        positions = [(20, 20), (32, 40), (45, 15)]

    psf_size = (2 * mask_half + 1, 2 * mask_half + 1)
    image = np.zeros(image_size, dtype=np.float64)
    model = np.zeros(image_size, dtype=np.float64)
    mask = np.zeros(image_size, dtype=bool)

    for sv, su in positions:
        image[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] += _gaussian_patch(psf_size, star_sigma, offset=image_offset)
        model[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] += _gaussian_patch(psf_size, star_sigma, offset=(0.0, 0.0))
        mask[
            sv - mask_half : sv + mask_half + 1,
            su - mask_half : su + mask_half + 1,
        ] = True

    return image, model, mask


# =========================================================================
# masked_ncc unit tests
# =========================================================================


class TestMaskedNcc:
    """Tests for the masked_ncc function."""

    def test_ncc_perfect_match_peaks_at_one(self) -> None:
        """NCC at zero shift should be 1.0 when image == model."""
        size = (32, 32)
        model = np.zeros(size)
        mask = np.zeros(size, dtype=bool)
        model[10:20, 10:20] = _gaussian_patch((10, 10), 2.0)
        mask[10:20, 10:20] = True
        image = model.copy()

        ph, pw = size[0] * 2, size[1] * 2
        ip = pad_top_left(image, ph, pw)
        mp = pad_top_left(model, ph, pw)
        wp = pad_top_left(mask, ph, pw)
        ncc, _ = masked_ncc(ip, mp, wp)

        assert ncc[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_ncc_bounded(self) -> None:
        """NCC values must lie in [-1, 1] for a single-star scene."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc, _ = masked_ncc(ip, mp, wp)

        assert np.all(ncc >= -1.0 - 1e-6)
        assert np.all(ncc <= 1.0 + 1e-6)

    def test_ncc_peak_at_correct_offset(self) -> None:
        """Peak of NCC surface coincides with known single-star offset."""
        image, model, mask = _make_single_star(image_offset=(2.0, 0.0))
        mh, mw = model.shape
        ih, iw = image.shape
        ph, pw = ih + mh, iw + mw
        ip = pad_top_left(image, ph, pw)
        mp = pad_top_left(model, ph, pw)
        wp = pad_top_left(mask, ph, pw)
        ncc, _ = masked_ncc(ip, mp, wp)

        peak_idx = np.unravel_index(np.argmax(ncc), ncc.shape)
        assert int(peak_idx[0]) == 2
        assert int(peak_idx[1]) == 0

    def test_ncc_real_valued(self) -> None:
        """NCC output must be a real-valued (non-complex) array."""
        image, model, mask = _make_single_star()
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc, num = masked_ncc(ip, mp, wp)

        assert not np.iscomplexobj(ncc)
        assert not np.iscomplexobj(num)

    def test_numerator_peaks_at_correct_offset_sparse_template(self) -> None:
        """NCC numerator peaks correctly even when template is sparse in mask.

        When the PSF covers only a small fraction of the mask, the NCC
        itself can plateau near 1.0 at many shifts.  The numerator
        must still peak at the correct offset because it scales with
        the image variance under the mask.
        """
        image, model, mask = _make_single_star(
            image_size=(128, 128),
            model_size=(128, 128),
            star_sigma=2.0,
            mask_half=30,
            image_offset=(3.0, -2.0),
        )
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        _ncc, num = masked_ncc(ip, mp, wp)

        num_peak = np.unravel_index(np.argmax(num), num.shape)
        assert int(num_peak[0]) == 3
        assert int(num_peak[1]) == ip.shape[1] - 2

    def test_ncc_mask_excludes_padding(self) -> None:
        """Changing model values outside mask must not alter the NCC."""
        image, model, mask = _make_single_star()
        mh, mw = model.shape
        ih, iw = image.shape
        ip = pad_top_left(image, ih + mh, iw + mw)
        mp = pad_top_left(model, ih + mh, iw + mw)
        wp = pad_top_left(mask, ih + mh, iw + mw)
        ncc1, num1 = masked_ncc(ip, mp, wp)

        model2 = model.copy()
        model2[~mask] = 999.0
        mp2 = pad_top_left(model2, ih + mh, iw + mw)
        ncc2, num2 = masked_ncc(ip, mp2, wp)

        np.testing.assert_allclose(ncc1, ncc2, atol=1e-10)
        np.testing.assert_allclose(num1, num2, atol=1e-10)


# =========================================================================
# Single-scale correlation tests
# =========================================================================


class TestSingleScale:
    """Tests for navigate_single_scale_kpeaks."""

    def test_single_star_integer_offset(self) -> None:
        """Single-star with integer offset converges to the correct shift."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            logger=None,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_single_star_subpixel_offset(self) -> None:
        """Single-star with subpixel offset converges within tolerance."""
        image, model, mask = _make_single_star(image_offset=(0.3, -0.7))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=64,
            metric='psr',
            logger=None,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(0.3, abs=0.05)
        assert dx == pytest.approx(-0.7, abs=0.05)

    def test_single_star_quality_above_threshold(self) -> None:
        """PSR quality for a clean single-star must be well above 6.0."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_single_scale_kpeaks(
            image=image,
            model=model,
            mask=mask,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            logger=None,
        )
        assert result['quality'] > 6.0


# =========================================================================
# Pyramid correlation tests
# =========================================================================


class TestPyramid:
    """Tests for navigate_with_pyramid_kpeaks."""

    def test_single_star_not_spurious(self) -> None:
        """Pyramid must not flag a clean single-star as spurious."""
        image, model, mask = _make_single_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_single_star_subpixel(self) -> None:
        """Pyramid converges for a single-star with subpixel offset."""
        image, model, mask = _make_single_star(image_offset=(0.5, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=64,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(0.5, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)

    def test_multi_star_converges(self) -> None:
        """Multi-star scene converges and is not flagged as spurious."""
        image, model, mask = _make_multi_star(image_offset=(1.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=3,
            max_peaks=5,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        assert not result['spurious']
        dy, dx = result['offset']
        assert dy == pytest.approx(1.0, abs=0.1)
        assert dx == pytest.approx(0.0, abs=0.1)

    def test_zero_offset(self) -> None:
        """No offset between image and model returns approximately (0, 0)."""
        image, model, mask = _make_single_star(image_offset=(0.0, 0.0))
        result = navigate_with_pyramid_kpeaks(
            image,
            model,
            mask,
            pyramid_levels=2,
            max_peaks=3,
            upsample_factor=16,
            metric='psr',
            quality_thresh=6.0,
            consistency_tol=2.0,
        )
        dy, dx = result['offset']
        assert dy == pytest.approx(0.0, abs=0.05)
        assert dx == pytest.approx(0.0, abs=0.05)
