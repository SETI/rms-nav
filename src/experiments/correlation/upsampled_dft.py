import numpy as np
from numpy.fft import fft2

from nav.support.correlate import upsampled_dft


def gaussian_patch(shape, sigma, offset):
    v_size, u_size = shape
    ov, ou = offset
    cv = (v_size - 1) / 2.0
    cu = (u_size - 1) / 2.0
    vv, uu = np.meshgrid(np.arange(v_size), np.arange(u_size), indexing='ij')
    dv = vv - (cv + ov)
    du = uu - (cu + ou)
    g = np.exp(-(dv**2 + du**2) / (2.0 * sigma**2))
    return g


def estimate_subpixel_shift(usfac: int, frac: float) -> float:
    shape = (64, 64)
    true_shift = (frac, 0.0)
    A = gaussian_patch(shape, sigma=2.0, offset=true_shift)
    B = gaussian_patch(shape, sigma=2.0, offset=(0.0, 0.0))
    X = fft2(A) * np.conj(fft2(B))

    # Integer peak index nearest the signed shift: 0 if frac<=0.5 else -1
    # Use region that scales with upsample factor
    region = usfac + 1
    dy_i = 0 if frac <= 0.5 else -1
    oy = region // 2
    Up = upsampled_dft(X, usfac, (region, region), (oy - dy_i * usfac, oy))
    upy, _ = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (upy - oy) / usfac
    return float(dy)


if __name__ == '__main__':
    usfacs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    # Include grid-aligned fractions and additional non-aligned fractions
    fracs = [
        0.00,
        0.125,
        0.25,
        0.375,
        0.50,
        0.625,
        0.75,
        0.875,
        0.07,
        0.11,
        0.19,
        0.23,
        0.31,
        0.44,
        0.58,
        0.67,
        0.73,
        0.86,
        0.99,
    ]
    print('usfac, frac, est_dy, gt_signed, abs_err')
    for usfac in usfacs:
        for frac in fracs:
            gt_signed = frac if frac <= 0.5 else frac - 1.0
            dy = estimate_subpixel_shift(usfac, frac)
            err = abs(dy - gt_signed)
            print(f'{usfac:>5d}, {frac:5.3f}, {dy:8.4f}, {gt_signed:8.4f}, {err:8.5f}')
