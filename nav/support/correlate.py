# mypy: ignore-errors

from typing import Any

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, ifftshift

from nav.support.image import (crop_center,
                               normalize_array,
                               pad_top_left)
from nav.support.misc import mad_std
from nav.support.types import NDArrayFloatType, NDArrayBoolType

# ==============================================================
# Small utilities
# ==============================================================

def int_to_signed(idx, size):
    """Map [0..size-1] argmax index to signed displacement coordinate."""
    return idx if idx < size//2 else idx - size

# ==============================================================
# Fourier helpers
# ==============================================================

def fourier_shift(img: NDArrayFloatType, dy: float, dx: float) -> NDArrayFloatType:
    """Subpixel shift via Fourier shift theorem (positive = down/right)."""
    img_v, img_u = img.shape
    fy = fftfreq(img_v)[:, None]
    fx = fftfreq(img_u)[None, :]
    phase = np.exp(-2j * np.pi * (dy*fy + dx*fx))
    return np.real(ifft2(fft2(img) * phase))

def upsampled_dft(X: NDArrayFloatType,
                  up_factor: int,
                  region_sz: tuple[int, int],
                  offsets: tuple[int, int]) -> NDArrayFloatType:
    """Localized upsampled DFT (Guizar–Sicairos)."""
    X_v, X_u = X.shape
    region_v, region_u = region_sz
    oy, ox = offsets
    ky = ifftshift(np.arange(X_v))
    kx = ifftshift(np.arange(X_u))
    a = np.arange(region_v) - oy
    b = np.arange(region_u) - ox
    j2pi = 1j * 2.0 * np.pi
    Er = np.exp((-j2pi / (X_v*up_factor)) * (a[:, None] @ ky[None, :]))
    Ec = np.exp((-j2pi / (X_u*up_factor)) * (kx[:, None] @ b[None, :]))
    return Er @ X @ Ec

# ==============================================================
# Masked NCC (linear correlation via padding)
# ==============================================================

def masked_ncc(image: NDArrayFloatType,
               model: NDArrayFloatType,
               mask: NDArrayBoolType) -> NDArrayFloatType:
    """
    Masked normalized cross-correlation surface between image and model
    with mask.
    All must be same padded shape.
    """
    image_fft = fft2(image)
    model_fft = fft2(model * mask)
    mask_fft = fft2(mask)

    # numerator
    num = np.real(ifft2(image_fft * np.conj(model_fft)))

    # local stats for I over mask support
    sumI = ifft2(image_fft * np.conj(mask_fft))
    sumI2 = ifft2(fft2(image**2) * np.conj(mask_fft))

    sumW = np.sum(mask)
    meanM = np.sum(model*mask) / (sumW + 1e-12)
    varM = np.sum(((model*mask) - meanM)**2) / (sumW + 1e-12)

    meanI = sumI / (sumW + 1e-12)
    varI = (sumI2/(sumW + 1e-12)) - meanI**2
    varI[varI < 0] = 0.0

    denom = np.sqrt(varI * varM + 1e-12)
    return (num - np.real(meanI)*np.sum(model*mask)) / (denom + 1e-12)

"""CodeRabbit says:

Fix masked NCC math; current normalization is incorrect

The numerator misses the symmetric mean terms and the denominator uses a scalar varM but omits shift‑wise varI properly. Use standard masked NCC with shift‑wise sums via FFT.

-def masked_ncc(I, M, W):
+def masked_ncc(I, M, W):
@@
-    FI = fft2(I)
-    FM = fft2(M * W)
-    FW = fft2(W)
-
-    # numerator
-    num = np.real(ifft2(FI * np.conj(FM)))
-
-    # local stats for I over mask support
-    sumI = ifft2(FI * np.conj(FW))
-    sumI2 = ifft2(fft2(I**2) * np.conj(FW))
-
-    sumW = np.sum(W)
-    meanM = np.sum(M*W) / (sumW + 1e-12)
-    varM = np.sum(((M*W) - meanM)**2) / (sumW + 1e-12)
-
-    meanI = sumI / (sumW + 1e-12)
-    varI = (sumI2/(sumW + 1e-12)) - meanI**2
-    varI[varI < 0] = 0.0
-
-    denom = np.sqrt(varI * varM + 1e-12)
-    return (num - np.real(meanI)*np.sum(M*W)) / (denom + 1e-12)
+    FI = fft2(I)
+    FW = fft2(W)
+    FMW = fft2(M * W)
+
+    # Sums over shifting mask support
+    sumW = np.sum(W) + 1e-12
+    sumIW = ifft2(FI * np.conj(FW))
+    sumI2W = ifft2(fft2(I**2) * np.conj(FW))
+
+    # Model stats (constant over shifts)
+    sumMW = np.sum(M * W)
+    meanM = sumMW / sumW
+    varMW = np.sum(((M - meanM) * W)**2) + 1e-12
+
+    # Cross and means
+    sumIMW = ifft2(FI * np.conj(FMW))
+    meanI = sumIW / sumW
+
+    # Numerator of NCC
+    num = sumIMW - meanI * sumMW - meanM * sumIW + meanI * meanM * sumW
+
+    # Denominator: sqrt( var_I(s) * var_M ), with var_I(s) under mask W
+    varI = sumI2W - 2.0 * meanI * sumIW + (meanI**2) * sumW
+    varI[varI < 0] = 0.0
+    denom = np.sqrt(varI * varMW) + 1e-12
+    return np.real(num) / denom
"""

# ==============================================================
# Peak metrics & selection
# ==============================================================

def psr_metric(corr: NDArrayFloatType,
               rc: tuple[int, int],
               guard: int = 5) -> float:
    corr_v, corr_u = corr.shape
    row, col = rc
    peak = corr[row, col]
    y, x = np.ogrid[:corr_v, :corr_u]
    mask = (y-row)**2 + (x-col)**2 > guard**2
    bg = corr[mask]
    return (peak - bg.mean()) / (bg.std() + 1e-12)

def pmr_metric(corr: NDArrayFloatType, peak_val: float) -> float:
    return peak_val / (corr.mean() + 1e-12)

def per_metric(corr: NDArrayFloatType, peak_val: float) -> float:
    return peak_val / (np.sqrt(np.sum(corr**2)) + 1e-12)

def nms_topk(corr: NDArrayFloatType,
             k: int = 5,
             radius: int = 5) -> list[tuple[int, int, float]]:
    """Non-maximum suppression to get top-k peaks."""
    corr_v, corr_u = corr.shape
    work = corr.copy()
    peaks = []
    for _ in range(k):
        idx = np.argmax(work)
        v = work.flat[idx]
        if not np.isfinite(v):
            break
        row, col = np.unravel_index(idx, work.shape)
        peaks.append((row, col, v))
        y, x = np.ogrid[:corr_v, :corr_u]
        work[(y-row)**2 + (x-col)**2 <= radius**2] = -np.inf
    return peaks

# ==============================================================
# Fisher / CRLB
# ==============================================================

def fisher_covariance(model_aligned: NDArrayFloatType,
                      sigma_n: float) -> NDArrayFloatType:
    sy = np.gradient(model_aligned, axis=0)
    sx = np.gradient(model_aligned, axis=1)
    Sxx = np.sum(sx * sx)
    Syy = np.sum(sy * sy)
    Sxy = np.sum(sx * sy)
    F = (1.0/(sigma_n**2 + 1e-18)) * np.array([[Sxx, Sxy],[Sxy, Syy]])
    return np.linalg.pinv(F + 1e-12*np.eye(2))

# ==============================================================
# Single-scale, K-peak evaluation (with optional prior)
# ==============================================================

def evaluate_candidate(image_pad: NDArrayFloatType,
                       model_pad: NDArrayFloatType,
                       mask_pad: NDArrayBoolType,
                       corr: NDArrayFloatType,
                       rc: tuple[int, int],
                       upsample_factor: int,
                       model_shape: tuple[int, int],
                       image_shape: tuple[int, int],
                       prior_shift=None,
                       prior_weight=0.0,
                       metric='psr'):
    """
    Evaluate a candidate for the navigation.

    Parameters:
        image_pad: The padded image.
        model_pad: The padded model.
        mask_pad: The padded mask.
        corr: The correlation matrix.
        rc: The row and column of the candidate.
        upsample_factor: The upsample factor.
        model_shape: The shape of the model.
        image_shape: The shape of the image.
        prior_shift: The prior shift.
        prior_weight: The prior weight.
        metric: The metric to use for the navigation.

    Returns:
        A dictionary containing the navigation result.
    """

    corr_v, corr_h = corr.shape
    p, q = rc
    dy_i, dx_i = int_to_signed(p, corr_v), int_to_signed(q, corr_h)

    # Subpixel refinement: local upsampled DFT of correlation spectrum numerator
    spec = fft2(image_pad) * np.conj(fft2(model_pad * mask_pad))
    oy = int(np.floor(upsample_factor*0.5))
    ox = int(np.floor(upsample_factor*0.5))
    Up = upsampled_dft(spec, upsample_factor, (3,3),
                       [oy - dy_i*upsample_factor, ox - dx_i*upsample_factor])
    upy, upx = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (upy - oy) / upsample_factor
    dx = dx_i + (upx - ox) / upsample_factor

    # Align combined model and compute residual stats
    model_h, model_w = model_shape
    image_h, image_w = image_shape
    model_shift = fourier_shift(model_pad[:model_h,:model_w], dy, dx)
    model_crop = crop_center(model_shift, (image_h, image_w))
    image_crop = image_pad[:image_h, :image_w]
    resid = normalize_array(image_crop) - normalize_array(model_crop)
    sigma_n = mad_std(resid)
    if sigma_n <= 1e-12:
        sigma_n = max(resid.std(), 1e-6)
    cov = fisher_covariance(model_crop, sigma_n)

    # Quality metric
    peak_val = corr[p, q]
    if metric == 'psr':
        quality = psr_metric(corr, (p, q))
    elif metric == 'pmr':
        quality = pmr_metric(corr, peak_val)
    elif metric == 'per':
        quality = per_metric(corr, peak_val)
    else:
        raise ValueError(f"metric must be 'psr', 'pmr', or 'per', not '{metric}'")

    # Prior penalty (encourage pyramid consistency or external priors)
    if prior_shift is not None and prior_weight > 0.0:
        dist = np.hypot(dy - prior_shift[0], dx - prior_shift[1])
        quality -= prior_weight * dist

    return {
        "offset": (float(dy), float(dx)),
        "cov": cov,
        "sigma_xy": (float(np.sqrt(cov[0,0])), float(np.sqrt(cov[1,1]))),
        "quality": float(quality),
        "peak_val": float(peak_val),
        "rc": (int(p), int(q))
    }

def navigate_single_scale_kpeaks(image: NDArrayFloatType,
                                 model: NDArrayFloatType,
                                 mask: NDArrayBoolType,
                                 max_peaks: int = 5,
                                 upsample_factor: int = 16,
                                 metric: str = 'psr',
                                 prior_shift: tuple[float, float] | None = None,
                                 prior_weight: float = 0.0,
                                 nms_radius: int = 5) -> dict[str, Any]:
    """
    One-scale masked NCC + top-K candidate evaluation.

    Parameters:
        image: The image to navigate.
        model: The model to navigate.
        mask: The mask to use for the navigation.
        max_peaks: The number of peaks to use for the navigation.
        upsample_factor: The upsample factor to use for the navigation.
        metric: The metric to use for the navigation.
        prior_shift: The prior shift to use for the navigation.
        prior_weight: The prior weight to use for the navigation.
        nms_radius: The radius to use for the non-maximum suppression.

    Returns:
        A dictionary containing the navigation result.
    """

    image_norm = normalize_array(image)
    model_h, model_w = model.shape
    image_h, image_w = image_norm.shape
    padded_h, padded_w = image_h + model_h, image_w + model_w

    image_pad = pad_top_left(image_norm, padded_h, padded_w)
    model_pad = pad_top_left(model, padded_h, padded_w)
    mask_pad = pad_top_left(mask, padded_h, padded_w)

    corr = masked_ncc(image_pad, model_pad, mask_pad)
    peaks = nms_topk(corr, k=max_peaks, radius=nms_radius)

    candidates = []
    for p, q, _ in peaks:
        candidates.append(
            evaluate_candidate(image_pad, model_pad, mask_pad, corr, (p,q),
                               upsample_factor, (model_h,model_w), (image_h,image_w),
                               prior_shift=prior_shift, prior_weight=prior_weight,
                               metric=metric)
        )
    if not candidates:
        return {
            'offset': (0.0, 0.0),
            'cov': np.diag([1e6, 1e6]),
            'sigma_xy': (1e3, 1e3),
            'quality': -np.inf,
        }
    return max(candidates, key=lambda r: r["quality"])

# ==============================================================
# Pyramid wrapper with K-peak final selection
# ==============================================================

def navigate_with_pyramid_kpeaks(image: NDArrayFloatType,
                                 model: NDArrayFloatType,
                                 mask: NDArrayBoolType,
                                 pyramid_levels: int = 3,
                                 max_peaks: int = 5,
                                 upsample_factor: int = 16,
                                 metric: str = 'psr',
                                 quality_thresh: float = 6.0,
                                 consistency_tol: float = 2.0,
                                 nms_radius: int = 5,
                                 prior_weight_final: float = 0.25) -> dict[str, Any]:
    """TODO Clean this up
    Build class-aware effective model + mask, run coarse->fine, then evaluate K peaks at final scale.
    Returns dict with shift, covariance, sigma_xy, quality, consistency, spurious flag.

    Parameters:
        image: The source image to navigate.
        model: The model to navigate.
        mask: The mask indicating which pixels in the model are valid.
        pyramid_levels: The number of pyramid levels to use.
        max_peaks: The number of peaks to use for the navigation.
        upsample_factor: The upsample factor to use for the navigation.
        metric: The metric to use for the navigation.
        quality_thresh: The quality threshold to use for the navigation.
        consistency_tol: The consistency tolerance to use for the navigation.
        nms_radius: The radius to use for the non-maximum suppression.
        prior_weight_final: The prior weight to use for the final navigation.

    Returns:
        A dictionary containing the navigation result.
    """

    # Coarse-to-fine prior sequence
    level_shifts = []
    for lvl in range(pyramid_levels, 0, -1):
        s = 2**(lvl-1)
        image_downsampled = image[::s, ::s]

        # Downsample model & mask by simple stride (keeps alignment to top-left)
        # TODO This is a bad way to downsample - should take mean of blocks
        model_downsampled = model[::s, ::s]
        mask_downsampled = mask[::s, ::s]

        res_lvl = navigate_single_scale_kpeaks(
            image_downsampled, model_downsampled, mask_downsampled,
            max_peaks=1, upsample_factor=upsample_factor,
            metric=metric, prior_shift=None, prior_weight=0.0,
            nms_radius=nms_radius
        )
        # Scale shift back to full res
        level_shifts.append((res_lvl['offset'][0]*s, res_lvl['offset'][1]*s))

    # Consistency: max deviation to last level’s shift
    shifts_arr = np.array(level_shifts, dtype=np.float64)
    final_prior = shifts_arr[-1]
    consistency = float(np.max(np.linalg.norm(shifts_arr - final_prior, axis=1)))

    # Final level: K-peak evaluation with prior penalty
    result = navigate_single_scale_kpeaks(
        image, model, mask,
        max_peaks=max_peaks, upsample_factor=upsample_factor,
        metric=metric, prior_shift=final_prior, prior_weight=prior_weight_final,
        nms_radius=nms_radius
    )

    spurious = (result["quality"] < quality_thresh) or (consistency > consistency_tol)

    return {
        "offset": result["offset"],
        "cov": result["cov"],
        "sigma_xy": result["sigma_xy"],
        "quality": result["quality"],
        "metric": metric,
        "consistency": consistency,
        "spurious": bool(spurious)
    }
