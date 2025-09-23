import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, ifftshift

# ==============================================================
# Data structures (lightweight containers)
# ==============================================================

class StarComponent:
    """
    Stars layer: model array is a star field (often sparse).
    Parameters
    ----------
    array : 2D float64
        Model star image on the model grid (larger than I).
    psf_cov : 2x2 float64
        PSF covariance used to blur the model (pixels^2).
    confidence : float in [0,1]
        Weight of this component.
    W : 2D float64 (0/1)
        Valid-pixel mask (1 over star footprint or 'known' area; 0 elsewhere).
    """
    def __init__(self, array, psf_cov, confidence, W):
        self.array = np.array(array, np.float64)
        self.psf_cov = np.array(psf_cov, np.float64).reshape(2,2)
        self.confidence = float(confidence)
        self.W = np.array(W, np.float64)

class BodyComponent:
    """
    Body (moon/planet) layer: model is a smooth ellipsoid/silhouette.
    We register primarily on the limb (edge). We derive a limb annulus mask
    and a limb-gradient model; the annulus width is set by:
        width_px ~= base + k * (crater_height / resolution)
    Parameters
    ----------
    array : 2D float64
        Smooth body rendering on the model grid.
    resolution : float
        Effective linear resolution (meters/pixel, km/pixel, etc). Only ratios matter.
    crater_height : float
        Scale of vertical relief (same units as 'resolution' numerator; only ratio used).
    confidence : float in [0,1]
        Weight of this component.
    U : 2x2 float64
        Optional extra blur covariance for limb uncertainty (pixels^2).
    """
    def __init__(self, array, resolution, crater_height, confidence, U=None):
        self.array = np.array(array, np.float64)
        self.resolution = float(resolution)
        self.crater_height = float(crater_height)
        self.confidence = float(confidence)
        self.U = np.eye(2) if U is None else np.array(U, np.float64).reshape(2,2)

class RingComponent:
    """
    Ring edges: model contains ONLY edge pixels for specific rings (not texture).
    Each edge has a radial-position uncertainty captured by an anisotropic covariance U.
    Parameters
    ----------
    array : 2D float64
        Ring-edge intensity template (edges bright, background ~0).
    U : 2x2 float64
        Anisotropic blur covariance (align U's principal axis radially in preprocessing;
        here we assume 'array' already oriented in model coordinates).
    confidence : float in [0,1]
        Weight for this ring.
    W : 2D float64 (0/1)
        Valid mask — should be 1 only on modeled edge pixels (or thin band).
    """
    def __init__(self, array, U, confidence, W):
        self.array = np.array(array, np.float64)
        self.U = np.array(U, np.float64).reshape(2,2)
        self.confidence = float(confidence)
        self.W = np.array(W, np.float64)

class TitanMask:
    """
    Titan-like diffuse blob to be IGNORED for correlation (mask only).
    Parameters
    ----------
    W : 2D float64 (0/1)
        Mask (1 where valid, 0 where Titan must be ignored).
        Usually this is a mask that is 0 over Titan + halo and 1 elsewhere.
    """
    def __init__(self, W):
        self.W = np.array(W, np.float64)

# ==============================================================
# Small utilities
# ==============================================================

def normalize(a, eps=1e-12):
    """Zero-mean, unit-std normalization (safe if nearly constant)."""
    a = np.asarray(a, np.float64)
    m = a.mean()
    s = a.std()
    if s < eps:
        return np.zeros_like(a)
    return (a - m) / s

def pad_top_left(a, H, W):
    """Place array 'a' at (0,0) inside zeros(H,W)."""
    out = np.zeros((H, W), np.float64)
    h, w = a.shape
    out[:h, :w] = a
    return out

def int_to_signed(idx, size):
    """Map [0..size-1] argmax index to signed displacement coordinate."""
    return idx if idx < size//2 else idx - size

def crop_center(img, out_shape):
    """Center crop to out_shape (h,w)."""
    H, W = img.shape
    h, w = out_shape
    sy = (H - h)//2
    sx = (W - w)//2
    return img[sy:sy+h, sx:sx+w]

def mad_std(a):
    m = np.median(a)
    return 1.4826 * np.median(np.abs(a - m))

# ==============================================================
# Fourier helpers
# ==============================================================

def gaussian_blur_cov(img, Sigma):
    """Blur by anisotropic Gaussian with covariance Sigma in frequency domain."""
    h, w = img.shape
    fy = fftfreq(h)[:, None]
    fx = fftfreq(w)[None, :]
    Syy, Syx = Sigma[0,0], Sigma[0,1]
    Sxy, Sxx = Sigma[1,0], Sigma[1,1]
    q = Syy*(fy*fy) + (Syx+Sxy)*(fy*fx) + Sxx*(fx*fx)
    H = np.exp(-2.0 * (np.pi**2) * q)
    return np.real(ifft2(fft2(img) * H))

def gradient_magnitude(img):
    """Simple isotropic gradient magnitude."""
    gy = np.gradient(img, axis=0)
    gx = np.gradient(img, axis=1)
    return np.hypot(gy, gx)

def fourier_shift(img, dy, dx):
    """Subpixel shift via Fourier shift theorem (positive = down/right)."""
    H, W = img.shape
    fy = fftfreq(H)[:, None]
    fx = fftfreq(W)[None, :]
    phase = np.exp(-2j*np.pi*(dy*fy + dx*fx))
    return np.real(ifft2(fft2(img) * phase))

def upsampled_dft(X, up_factor, region_sz, offsets):
    """Localized upsampled DFT (Guizar–Sicairos)."""
    nr, nc = X.shape
    ry, rx = region_sz
    oy, ox = offsets
    ky = ifftshift(np.arange(nr))
    kx = ifftshift(np.arange(nc))
    a = np.arange(ry) - oy
    b = np.arange(rx) - ox
    j2pi = 1j * 2.0 * np.pi
    Er = np.exp((-j2pi/(nr*up_factor)) * (a[:,None] @ ky[None,:]))
    Ec = np.exp((-j2pi/(nc*up_factor)) * (kx[:,None] @ b[None,:]))
    return Er @ X @ Ec

# ==============================================================
# Masked NCC (linear correlation via padding)
# ==============================================================

def masked_ncc(I, M, W):
    """
    Masked normalized cross-correlation surface between image I and model M with mask W.
    All must be same padded shape.
    """
    FI = fft2(I)
    FM = fft2(M * W)
    FW = fft2(W)

    # numerator
    num = np.real(ifft2(FI * np.conj(FM)))

    # local stats for I over mask support
    sumI = ifft2(FI * np.conj(FW))
    sumI2 = ifft2(fft2(I**2) * np.conj(FW))

    sumW = np.sum(W)
    meanM = np.sum(M*W) / (sumW + 1e-12)
    varM = np.sum(((M*W) - meanM)**2) / (sumW + 1e-12)

    meanI = sumI / (sumW + 1e-12)
    varI = (sumI2/(sumW + 1e-12)) - meanI**2
    varI[varI < 0] = 0.0

    denom = np.sqrt(varI * varM + 1e-12)
    return (num - np.real(meanI)*np.sum(M*W)) / (denom + 1e-12)

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

def psr_metric(corr, rc, guard=5):
    H, W = corr.shape
    p, q = rc
    peak = corr[p,q]
    y, x = np.ogrid[:H, :W]
    mask = (y-p)**2 + (x-q)**2 > guard**2
    bg = corr[mask]
    return (peak - bg.mean()) / (bg.std() + 1e-12)

def pmr_metric(corr, peak_val):
    return peak_val / (corr.mean() + 1e-12)

def per_metric(corr, peak_val):
    return peak_val / (np.sqrt(np.sum(corr**2)) + 1e-12)

def nms_topk(corr, k=5, radius=5):
    """Non-maximum suppression to get top-k peaks."""
    H, W = corr.shape
    work = corr.copy()
    peaks = []
    for _ in range(k):
        idx = np.argmax(work)
        v = work.flat[idx]
        if not np.isfinite(v):
            break
        p, q = np.unravel_index(idx, work.shape)
        peaks.append((p, q, v))
        y, x = np.ogrid[:H, :W]
        work[(y-p)**2 + (x-q)**2 <= radius**2] = -np.inf
    return peaks

# ==============================================================
# Fisher / CRLB
# ==============================================================

def fisher_covariance(model_aligned, sigma_n):
    sy = np.gradient(model_aligned, axis=0)
    sx = np.gradient(model_aligned, axis=1)
    Sxx = np.sum(sx*sx)
    Syy = np.sum(sy*sy)
    Sxy = np.sum(sx*sy)
    F = (1.0/(sigma_n**2 + 1e-18)) * np.array([[Sxx, Sxy],[Sxy, Syy]])
    return np.linalg.pinv(F + 1e-12*np.eye(2))

# ==============================================================
# Class-aware effective model builder
# ==============================================================

def build_effective_model(star_components, body_components, ring_components, titan_masks):
    """
    Build the effective model M_eff and mask W_eff combining all classes:
      - Stars: PSF-blur, normalize, weight by confidence.
      - Bodies: derive limb annulus mask & limb gradient model, blur by U, weight.
      - Rings: anisotropic blur by U, mask is edge band, weight.
      - Titan: multiply W_eff by these masks (zeros inside Titan region).
    Returns M_eff_norm, W_eff (both same shape as model arrays).
    """
    # Infer model canvas from the first available component
    for src in (star_components or []) + (body_components or []) + (ring_components or []):
        Hm, Wm = src.array.shape
        break
    else:
        raise ValueError("No components given.")

    M_eff = np.zeros((Hm, Wm), np.float64)
    W_eff = np.zeros((Hm, Wm), np.float64)
    total_w = 0.0

    # Stars
    for s in star_components or []:
        A = normalize(s.array)
        A = gaussian_blur_cov(A, s.psf_cov)
        M_eff += s.confidence * A
        W_eff += s.confidence * s.W
        total_w += s.confidence

    # Bodies: limb annulus extraction + gradient model
    # Set annulus half-width in pixels from crater_height/resolution
    base = 1.5  # pixels
    k_bumpy = 2.0  # scale factor
    for b in body_components or []:
        B = normalize(b.array)
        # Limb emphasis: gradient magnitude of a slightly blurred silhouette
        Bg = gaussian_blur_cov(B, b.U)  # optional extra uncertainty blur
        G = gradient_magnitude(Bg)
        # limb mask = thresholded gradient band, dilated by width ~ base + k*H/R
        thr = 0.5 * np.max(G) if np.max(G) > 0 else 0.0
        limb = (G >= thr).astype(np.float64)
        width = base + k_bumpy * (b.crater_height / max(b.resolution, 1e-12))
        # approximate dilation by Gaussian blur of the binary mask then clip
        limb_band = gaussian_blur_cov(limb, np.diag([width**2, width**2]))
        limb_band = (limb_band > 0.1).astype(np.float64)

        # model contribution = gradient magnitude within limb band
        M_eff += b.confidence * (G * limb_band)
        W_eff += b.confidence * limb_band
        total_w += b.confidence

    # Rings
    for r in ring_components or []:
        R = normalize(r.array)
        Rb = gaussian_blur_cov(R, r.U)  # anisotropic radial blur encoded in U
        M_eff += r.confidence * Rb
        W_eff += r.confidence * r.W
        total_w += r.confidence

    if total_w > 0:
        M_eff /= total_w
        W_eff /= total_w

    # Titan masks (multiply: 0 inside Titan -> ignore)
    for t in titan_masks or []:
        W_eff *= t.W

    M_eff = normalize(M_eff)
    # Normalize mask to [0,1]
    W_eff = np.clip(W_eff, 0.0, 1.0)
    return M_eff, W_eff

# ==============================================================
# Single-scale, K-peak evaluation (with optional prior)
# ==============================================================

def evaluate_candidate(I_pad, M_pad, W_pad, corr, rc, upsample_factor, M_shape, I_shape,
                       prior_shift=None, prior_weight=0.0, metric="psr"):
    H, W = corr.shape
    p, q = rc
    dy_i, dx_i = int_to_signed(p, H), int_to_signed(q, W)

    # Subpixel refinement: local upsampled DFT of correlation spectrum numerator
    spec = fft2(I_pad) * np.conj(fft2(M_pad * W_pad))
    oy = int(np.floor(upsample_factor*0.5))
    ox = int(np.floor(upsample_factor*0.5))
    Up = upsampled_dft(spec, upsample_factor, (3,3),
                       [oy - dy_i*upsample_factor, ox - dx_i*upsample_factor])
    upy, upx = np.unravel_index(np.argmax(np.abs(Up)), Up.shape)
    dy = dy_i + (upy - oy)/upsample_factor
    dx = dx_i + (upx - ox)/upsample_factor

    # Align combined model and compute residual stats
    Mh, Mw = M_shape
    Hi, Wi = I_shape
    M_shift = fourier_shift(M_pad[:Mh,:Mw], dy, dx)
    M_crop = crop_center(M_shift, (Hi, Wi))
    I_crop = I_pad[:Hi, :Wi]
    resid = normalize(I_crop) - normalize(M_crop)
    sigma_n = mad_std(resid)
    if sigma_n <= 1e-12:
        sigma_n = max(resid.std(), 1e-6)
    cov = fisher_covariance(M_crop, sigma_n)

    # Quality metric
    peak_val = corr[p,q]
    if metric == "psr":
        quality = psr_metric(corr, (p,q))
    elif metric == "pmr":
        quality = pmr_metric(corr, peak_val)
    elif metric == "per":
        quality = per_metric(corr, peak_val)
    else:
        raise ValueError("metric must be 'psr', 'pmr', or 'per'")

    # Prior penalty (encourage pyramid consistency or external priors)
    if prior_shift is not None and prior_weight > 0.0:
        dist = np.hypot(dy - prior_shift[0], dx - prior_shift[1])
        quality -= prior_weight * dist

    return {
        "shift": (float(dy), float(dx)),
        "cov": cov,
        "sigma_xy": (float(np.sqrt(cov[0,0])), float(np.sqrt(cov[1,1]))),
        "quality": float(quality),
        "peak_val": float(peak_val),
        "rc": (int(p), int(q))
    }

def register_single_scale_kpeaks(I, M_eff, W_eff, max_peaks=5, upsample_factor=16,
                                 metric="psr", prior_shift=None, prior_weight=0.0,
                                 nms_radius=5):
    """
    One-scale masked NCC + top-K candidate evaluation.
    """
    I_n = normalize(I)
    Mh, Mw = M_eff.shape
    Hi, Wi = I_n.shape
    H, W = Hi + Mh, Wi + Mw

    I_pad = pad_top_left(I_n, H, W)
    M_pad = pad_top_left(M_eff, H, W)
    W_pad = pad_top_left(W_eff, H, W)

    corr = masked_ncc(I_pad, M_pad, W_pad)
    peaks = nms_topk(corr, k=max_peaks, radius=nms_radius)

    candidates = []
    for p, q, _ in peaks:
        candidates.append(
            evaluate_candidate(I_pad, M_pad, W_pad, corr, (p,q),
                               upsample_factor, (Mh,Mw), (Hi,Wi),
                               prior_shift=prior_shift, prior_weight=prior_weight,
                               metric=metric)
        )
    if not candidates:
        return {
            "shift": (0.0,0.0),
            "cov": np.diag([1e6,1e6]),
            "sigma_xy": (1e3,1e3),
            "quality": -np.inf,
        }
    return max(candidates, key=lambda r: r["quality"])

# ==============================================================
# Pyramid wrapper with K-peak final selection
# ==============================================================

def register_with_pyramid_kpeaks(
    I,
    star_components=None,
    body_components=None,
    ring_components=None,
    titan_masks=None,
    pyramid_levels=3,
    max_peaks=5,
    upsample_factor=16,
    metric="psr",
    quality_thresh=6.0,
    consistency_tol=2.0,
    nms_radius=5,
    prior_weight_final=0.25
):
    """
    Build class-aware effective model + mask, run coarse->fine, then evaluate K peaks at final scale.
    Returns dict with shift, covariance, sigma_xy, quality, consistency, spurious flag.
    """
    star_components = star_components or []
    body_components = body_components or []
    ring_components = ring_components or []
    titan_masks = titan_masks or []

    # Build base effective model once (components are already on model grid)
    M_eff_full, W_eff_full = build_effective_model(
        star_components, body_components, ring_components, titan_masks
    )
    Mh, Mw = M_eff_full.shape

    # Coarse-to-fine prior sequence
    level_shifts = []
    for lvl in range(pyramid_levels, 0, -1):
        s = 2**(lvl-1)
        I_ds = I[::s, ::s]

        # Downsample model & mask by simple stride (keeps alignment to top-left)
        # TODO This is a bad way to downsample - should take mean of blocks
        M_ds = M_eff_full[::s, ::s]
        W_ds = W_eff_full[::s, ::s]

        res_lvl = register_single_scale_kpeaks(
            I_ds, M_ds, W_ds,
            max_peaks=1, upsample_factor=upsample_factor,
            metric=metric, prior_shift=None, prior_weight=0.0,
            nms_radius=nms_radius
        )
        # Scale shift back to full res
        level_shifts.append((res_lvl["shift"][0]*s, res_lvl["shift"][1]*s))

    # Consistency: max deviation to last level’s shift
    shifts_arr = np.array(level_shifts, np.float64)
    final_prior = shifts_arr[-1]
    consistency = float(np.max(np.linalg.norm(shifts_arr - final_prior, axis=1)))

    # Final level: K-peak evaluation with prior penalty
    result = register_single_scale_kpeaks(
        I, M_eff_full, W_eff_full,
        max_peaks=max_peaks, upsample_factor=upsample_factor,
        metric=metric, prior_shift=final_prior, prior_weight=prior_weight_final,
        nms_radius=nms_radius
    )

    spurious = (result["quality"] < quality_thresh) or (consistency > consistency_tol)

    return {
        "shift": result["shift"],
        "cov": result["cov"],
        "sigma_xy": result["sigma_xy"],
        "quality": result["quality"],
        "metric": metric,
        "consistency": consistency,
        "spurious": bool(spurious),
        "model_shape": (Mh, Mw)
    }

# ==============================================================
# Example (small synthetic; replace with real data)
# ==============================================================

if __name__ == "__main__":
    # Image (observed)
    I = np.zeros((256,256))
    I[120:136, 100:116] = 1.0  # bright "star-like" block

    # Model canvas ~2x linear area (here slightly larger)
    Mh, Mw = 272, 280

    # --- Stars ---
    star_model = np.zeros((Mh,Mw))
    cy, cx = Mh//2, Mw//2
    star_model[cy-8:cy+8, cx-8:cx+8] = 1.0
    star_mask = (star_model > 0).astype(np.float64)
    stars = [StarComponent(star_model, psf_cov=np.diag([1.0,1.0]), confidence=1.0, W=star_mask)]

    # --- Bodies (one body) ---
    body_model = np.zeros((Mh,Mw))
    rr = 24
    yy, xx = np.ogrid[:Mh,:Mw]
    body_model[(yy-cy)**2 + (xx-cx-30)**2 <= rr**2] = 1.0  # smooth disk
    bodies = [BodyComponent(body_model, resolution=5.0, crater_height=0.5, confidence=0.7, U=np.diag([0.5,0.5]))]

    # --- Rings (one thin edge) ---
    ring_model = np.zeros((Mh,Mw))
    ring_r = 60
    edge = np.abs(np.sqrt((yy-cy)**2 + (xx-cx)**2) - ring_r) <= 1.0
    ring_model[edge] = 1.0
    ring_mask = (ring_model > 0).astype(np.float64)
    rings = [RingComponent(ring_model, U=np.diag([1.5,0.2]), confidence=0.5, W=ring_mask)]

    # --- Titan (mask out a blob region; here none) ---
    titan = []  # e.g., TitanMask(W_titan) with zeros inside Titan

    res = register_with_pyramid_kpeaks(
        I,
        star_components=stars,
        body_components=bodies,
        ring_components=rings,
        titan_masks=titan,
        pyramid_levels=3,  # TODO Remove hard-coded values
        max_peaks=5,
        upsample_factor=16,
        metric="psr",           # 'psr' | 'pmr' | 'per'
        quality_thresh=6.0,
        consistency_tol=2.0,
        nms_radius=5,
        prior_weight_final=0.25
    )

    print("Estimated shift (dy, dx):", res["shift"])
    print("Uncertainty (σy, σx):    ", res["sigma_xy"])
    print("Quality ({}):            ".format(res["metric"]), res["quality"])
    print("Consistency (px):        ", res["consistency"])
    print("Spurious?:               ", res["spurious"])


def make_ring_edge_model(shape, center, edges, angular_sampling=720):
    """
    Create a model array and mask for a set of ring/gap edges.

    Parameters
    ----------
    shape : (H, W)
        Output array shape (model grid).
    center : (cy, cx)
        Ring center (pixels).
    edges : list of dict
        Each entry should have:
          - radius_px (float): expected orbital radius in pixels
          - uncertainty_px (float): 1σ radial uncertainty in pixels
          - confidence (float, optional): weight (default=1.0)
    angular_sampling : int
        Number of azimuth samples to approximate circular edges.

    Returns
    -------
    model_array : 2D float64
        Array with thin bright lines at edge positions, blurred radially by σ.
    mask_array : 2D float64
        Mask (1 near edges, 0 elsewhere).
    """
    H, W = shape
    cy, cx = center
    yy, xx = np.indices(shape)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    model = np.zeros(shape, np.float64)
    mask = np.zeros(shape, np.float64)

    for e in edges:
        r0 = e["radius_px"]
        sigma = max(e["uncertainty_px"], 1e-3)
        w = e.get("confidence", 1.0)

        # Gaussian radial profile around r0
        prof = np.exp(-0.5 * ((r - r0)/sigma)**2)

        model += w * prof
        mask += (prof > np.exp(-0.5*(2.0)**2)).astype(np.float64) * w  # keep ~±2σ band

    # Normalize
    if mask.max() > 0:
        mask /= mask.max()
    if model.max() > 0:
        model /= model.max()

    return model, mask

# Example: Saturn-like rings
edges = [
    {"radius_px": 50.0, "uncertainty_px": 1.5, "confidence": 1.0},  # inner edge
    {"radius_px": 80.0, "uncertainty_px": 2.0, "confidence": 0.8},  # outer edge
    {"radius_px": 120.0, "uncertainty_px": 3.5, "confidence": 0.5}, # faint outer ring
]

model_array, mask_array = make_ring_edge_model(
    shape=(256,256),
    center=(128,128),
    edges=edges
)
