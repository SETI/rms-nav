import numpy as np
from numpy.fft import fft2

from nav.support.correlate import upsampled_dft
from nav.support.types import NDArrayFloatType


def _gaussian(shape: tuple[int, int],
              sigma: float,
              offset: tuple[float, float]) -> NDArrayFloatType:
    v, u = shape
    ov, ou = offset
    cv = (v - 1) / 2.0
    cu = (u - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(v), np.arange(u), indexing='ij')
    dv = yy - (cv + ov)
    du = xx - (cu + ou)
    return np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))


def _estimate_dy(usfac: int, region: int) -> float:
    shape = (64, 64)
    A = _gaussian(shape, sigma=2.0, offset=(0.5, 0.0))
    B = _gaussian(shape, sigma=2.0, offset=(0.0, 0.0))
    X = fft2(A) * np.conj(fft2(B))
    dy_i = 0
    oy = region // 2
    Up = upsampled_dft(X, usfac, (region, region), (oy - dy_i*usfac, oy))
    upy, _ = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (upy - oy) / usfac
    return float(dy)


def test_upsampled_dft_region_too_small_biases_result() -> None:
    # A 3x3 region misses the true subpixel maximum when upsample_factor is large
    for usfac in [4, 8, 16, 32]:
        dy = _estimate_dy(usfac, 3)
        assert not np.isclose(dy, 0.5, atol=1e-3)


def test_upsampled_dft_region_scales_with_factor_is_correct() -> None:
    # Region size must be >= upsample_factor+1 to include the peak ~0.5*usfac away
    for usfac in [2, 4, 8, 16, 32]:
        region = usfac + 1
        dy = _estimate_dy(usfac, region)
        assert np.isclose(dy, 0.5, atol=1e-3)
