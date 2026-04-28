"""Tests for ``nav.nav_model.stars.predicted_snr``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from starcat import SCLASS_TO_B_MINUS_V as CANONICAL_SCLASS_TO_B_MINUS_V

from nav.nav_model.stars.predicted_snr import (
    SCLASS_TO_B_MINUS_V,
    integrated_signal_dn,
    predicted_snr,
    psf_aperture_pixels,
    psf_sigma_px,
)


@dataclass
class _FakeStar:
    """Minimal star stand-in for predicted_snr tests."""

    vmag: float | None


class _FakeGaussianPSF:
    """Minimal PSF stand-in exposing only a ``sigma`` attribute."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma


class _FakeFwhmPSF:
    """PSF stand-in that exposes ``fwhm()`` but not ``sigma``."""

    def __init__(self, fwhm: float) -> None:
        self._fwhm = fwhm

    def fwhm(self) -> float:
        """Return the constant FWHM passed at construction."""
        return self._fwhm


def test_psf_sigma_px_reads_sigma_attribute() -> None:
    """``psf_sigma_px`` returns the PSF's ``sigma`` attribute when present."""
    assert psf_sigma_px(_FakeGaussianPSF(sigma=1.25)) == 1.25  # type: ignore[arg-type]


def test_psf_sigma_px_falls_back_to_fwhm() -> None:
    """When ``sigma`` is absent, the helper converts FWHM via the Gaussian factor."""
    psf = _FakeFwhmPSF(fwhm=2.3548200450309493)
    assert math.isclose(psf_sigma_px(psf), 1.0, rel_tol=1e-12)  # type: ignore[arg-type]


def test_psf_sigma_px_raises_when_neither_sigma_nor_fwhm_present() -> None:
    """The helper raises ``AttributeError`` for an unsupported PSF subclass."""

    class _NoApiPSF:
        pass

    with pytest.raises(AttributeError, match='exposes neither sigma nor fwhm'):
        psf_sigma_px(_NoApiPSF())  # type: ignore[arg-type]


def test_psf_aperture_pixels_returns_4_pi_sigma_squared() -> None:
    """The aperture matches the Gaussian noise-equivalent area formula."""
    expected = 4.0 * math.pi * 0.7 * 0.7
    assert math.isclose(psf_aperture_pixels(0.7), expected, rel_tol=1e-12)


def test_psf_aperture_pixels_rejects_non_positive_sigma() -> None:
    """A zero or negative sigma raises ``ValueError`` with the offending value."""
    with pytest.raises(ValueError, match='sigma_px_value must be > 0'):
        psf_aperture_pixels(0.0)


def test_integrated_signal_dn_uses_canonical_anchor() -> None:
    """The flux-to-DN scaling matches ``2.512 ** -(vmag - 4)`` at vmag=4."""
    star: Any = _FakeStar(vmag=4.0)
    assert math.isclose(integrated_signal_dn(star, mag_offset=0.0), 1.0, rel_tol=1e-12)


def test_integrated_signal_dn_applies_mag_offset() -> None:
    """The mag offset shifts the catalog magnitude before the flux conversion."""
    star: Any = _FakeStar(vmag=4.0)
    expected = 2.512 ** -(4.0 + 0.5 - 4.0)
    assert math.isclose(integrated_signal_dn(star, mag_offset=0.5), expected, rel_tol=1e-12)


def test_integrated_signal_dn_returns_zero_when_vmag_missing() -> None:
    """Stars without a catalog vmag produce zero predicted DN."""
    star: Any = _FakeStar(vmag=None)
    assert integrated_signal_dn(star, mag_offset=0.0) == 0.0


def test_predicted_snr_matches_design_formula() -> None:
    """Predicted SNR matches the design's matched-filter form for known inputs."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeGaussianPSF(sigma=1.0)
    image_noise_sigma = 0.5
    sig = 1.0
    aperture = 4.0 * math.pi
    expected = sig / math.sqrt(sig + image_noise_sigma**2 * aperture)
    out = predicted_snr(star, psf=psf, image_noise_sigma=image_noise_sigma)
    assert math.isclose(out, expected, rel_tol=1e-12)


def test_predicted_snr_returns_zero_for_missing_vmag() -> None:
    """A star with no vmag produces zero SNR (no signal)."""
    star: Any = _FakeStar(vmag=None)
    psf: Any = _FakeGaussianPSF(sigma=1.0)
    assert predicted_snr(star, psf=psf, image_noise_sigma=0.5) == 0.0


def test_predicted_snr_raises_for_non_positive_noise_sigma() -> None:
    """A non-positive noise sigma raises ``ValueError`` naming the bad value."""
    star: Any = _FakeStar(vmag=4.0)
    psf: Any = _FakeGaussianPSF(sigma=1.0)
    with pytest.raises(ValueError, match='image_noise_sigma must be > 0'):
        predicted_snr(star, psf=psf, image_noise_sigma=0.0)


def test_sclass_to_b_minus_v_re_export_matches_starcat() -> None:
    """The re-exported lookup table is the same dict as ``starcat``'s."""
    assert SCLASS_TO_B_MINUS_V is CANONICAL_SCLASS_TO_B_MINUS_V
