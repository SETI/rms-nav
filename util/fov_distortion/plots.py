"""Figures for the FOV distortion and twist analysis.

Three figure kinds:

- :func:`plot_frame_decomposition` -- a per-frame sample image with the star
  residual field drawn as magnified arrows, before and after the twist is
  removed, plus the radial residual against field radius.  This is the visual
  that shows, for one representative frame per instrument, what the residual
  errors and the twist look like.
- :func:`plot_instrument_twist` -- per-frame twist with error bars and the
  instrument's weighted-mean band, showing whether the twist is one common
  value or scatters.
- :func:`plot_instrument_radial` -- the pooled radial residual against
  normalized field radius with the aggregate model, and the post-fit residual
  showing the centroid-noise floor rising toward the edge.

All figures are written with ``dpi=110`` to match the other committed report
assets.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from util.fov_distortion.measure import FrameMeasurement
from util.fov_distortion.results import InstrumentSummary

__all__ = [
    'plot_frame_decomposition',
    'plot_instrument_distortion_map',
    'plot_instrument_radial',
    'plot_instrument_twist',
]

FloatArray = NDArray[np.float64]
_DPI = 110


def _stretch(image: FloatArray) -> tuple[float, float]:
    """Return robust display limits for a star-field image."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 25.0))
    hi = float(np.percentile(finite, 99.8))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _arrow_scale(residuals: FloatArray, image_shape: tuple[int, int]) -> float:
    """Choose an arrow magnification so a typical residual is clearly visible."""
    mag = np.hypot(residuals[:, 0], residuals[:, 1])
    typical = float(np.median(mag[mag > 0.0])) if np.any(mag > 0.0) else 0.0
    if typical <= 0.0:
        return 1.0
    target = 0.06 * max(image_shape)
    return target / typical


def _draw_field(
    ax: Axes,
    image: FloatArray,
    detected: FloatArray,
    residuals: FloatArray,
    scale: float,
    title: str,
) -> None:
    """Draw the image with per-star residual arrows at a fixed magnification."""
    lo, hi = _stretch(image)
    ax.imshow(image, origin='upper', cmap='gray', vmin=lo, vmax=hi, aspect='equal')
    ax.quiver(
        detected[:, 1],
        detected[:, 0],
        residuals[:, 1] * scale,
        residuals[:, 0] * scale,
        angles='xy',
        scale_units='xy',
        scale=1.0,
        color='#ff6060',
        width=0.004,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_frame_decomposition(frame: FrameMeasurement, image: FloatArray, path: str) -> None:
    """Write the per-frame sample figure: residual field and radial profile.

    The left two panels draw the per-star residual arrows over the frame's
    pixels, before and after the twist is removed; the right panel plots the
    radial residual against field radius with the fitted model.

    Parameters:
        frame: A measured frame with a populated decomposition.
        image: The frame's pixel array (same shape as ``frame.image_shape``).
        path: Output PNG path.

    Raises:
        ValueError: if the frame has no decomposition.
    """
    if frame.decomposition is None:
        raise ValueError('frame has no decomposition to plot')
    decomp = frame.decomposition
    pred = np.array([m.predicted_vu for m in frame.stars], dtype=np.float64)
    det = np.array([m.detected_vu for m in frame.stars], dtype=np.float64)
    offset = np.asarray(frame.offset_vu, dtype=np.float64)
    # Remove the median translation so the panel shows the twist and distortion
    # pattern rather than a uniform pointing / centroid-convention offset.
    raw_residual = det - (pred + offset)
    field_residual = raw_residual - np.median(raw_residual, axis=0)
    twist_residual = decomp.twist.residuals_vu

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    scale = _arrow_scale(field_residual, frame.image_shape)
    _draw_field(
        axes[0],
        image,
        det,
        field_residual,
        scale,
        f'Residual field, translation removed (arrows x{scale:.0f})\n'
        f'twist {decomp.twist.rotation_deg:+.4f} deg, '
        f'raw RMS {decomp.rms_raw_px:.3f} px',
    )
    _draw_field(
        axes[1],
        image,
        det,
        twist_residual,
        scale,
        f'After twist removed (arrows x{scale:.0f})\n'
        f'RMS {decomp.rms_after_twist_px:.3f} px, radial k1 {decomp.radial.k_sim[0]:+.2e}',
    )

    center = np.asarray(frame.center_vu)
    rho = np.hypot(pred[:, 0] - center[0], pred[:, 1] - center[1])
    rhat = (pred - center) / np.where(rho[:, None] > 0, rho[:, None], 1.0)
    radial_comp = np.sum(twist_residual * rhat, axis=1)
    rho_n = rho / frame.rho_ref_px
    order = np.argsort(rho)
    axes[2].scatter(rho_n, radial_comp, s=14, color='#3070c0', alpha=0.7, label='per star')
    axes[2].plot(
        rho_n[order],
        decomp.radial.radial_displacement_px(rho[order]),
        color='#d02020',
        lw=1.8,
        label='fitted radial model',
    )
    axes[2].axhline(0.0, color='#909090', lw=0.8)
    axes[2].set_xlabel('normalized field radius')
    axes[2].set_ylabel('radial residual (px)')
    axes[2].set_title(f'{frame.image_name}: radial distortion\n{decomp.n_stars} stars', fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_instrument_twist(summary: InstrumentSummary, path: str) -> None:
    """Write the per-instrument twist-scatter figure.

    Parameters:
        summary: An instrument summary with a consistency verdict.
        path: Output PNG path.

    Raises:
        ValueError: if the summary carries no consistency statistics.
    """
    if summary.consistency is None:
        raise ValueError('summary has no consistency statistics to plot')
    twists = np.array(
        [f.decomposition.twist.rotation_deg for f in summary.ok_frames if f.decomposition],
        dtype=np.float64,
    )
    sigmas = np.array(
        [f.decomposition.twist.sigma_rotation_deg for f in summary.ok_frames if f.decomposition],
        dtype=np.float64,
    )
    idx = np.arange(twists.size)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        idx,
        twists,
        yerr=sigmas,
        fmt='o',
        ms=4,
        color='#3070c0',
        ecolor='#a0a0a0',
        capsize=2,
        label='per-frame twist',
    )
    mean = summary.consistency.weighted_mean_deg
    sigma_mean = summary.consistency.sigma_mean_deg
    ax.axhline(mean, color='#d02020', lw=1.6, label=f'weighted mean {mean:+.3f} deg')
    ax.axhspan(mean - sigma_mean, mean + sigma_mean, color='#d02020', alpha=0.15)
    ax.axhline(0.0, color='#909090', lw=0.8)
    verdict = 'consistent' if summary.consistency.consistent else 'inconsistent'
    ax.set_xlabel('frame index')
    ax.set_ylabel('twist (deg)')
    ax.set_title(
        f'{summary.label}: per-frame twist\n'
        f'{verdict}: scatter {summary.consistency.scatter_corner_px:.2f} px at corner '
        f'(threshold {summary.consistency.scatter_corner_threshold_px:.2f} px), '
        f'{summary.consistency.n_frames} frames',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def _bin_field(
    positions_vu: FloatArray,
    vectors_vu: FloatArray,
    image_shape: tuple[int, int],
    n_cells: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Average per-star vectors into an ``n_cells`` x ``n_cells`` grid.

    Returns the cell-center ``(v, u)`` and mean ``(dv, du)`` for cells with at
    least one star, plus the per-cell mean magnitude for a background image.
    """
    h, w = image_shape
    v_edges = np.linspace(0.0, h, n_cells + 1)
    u_edges = np.linspace(0.0, w, n_cells + 1)
    v_idx = np.clip(np.digitize(positions_vu[:, 0], v_edges) - 1, 0, n_cells - 1)
    u_idx = np.clip(np.digitize(positions_vu[:, 1], u_edges) - 1, 0, n_cells - 1)
    cell_v: list[float] = []
    cell_u: list[float] = []
    mean_dv: list[float] = []
    mean_du: list[float] = []
    magnitude = np.full((n_cells, n_cells), np.nan)
    for vi in range(n_cells):
        for ui in range(n_cells):
            sel = (v_idx == vi) & (u_idx == ui)
            if not np.any(sel):
                continue
            dv = float(np.mean(vectors_vu[sel, 0]))
            du = float(np.mean(vectors_vu[sel, 1]))
            cell_v.append(0.5 * (v_edges[vi] + v_edges[vi + 1]))
            cell_u.append(0.5 * (u_edges[ui] + u_edges[ui + 1]))
            mean_dv.append(dv)
            mean_du.append(du)
            magnitude[vi, ui] = math.hypot(dv, du)
    return (
        np.array([cell_v, cell_u], dtype=np.float64).T,
        np.array([mean_dv, mean_du], dtype=np.float64).T,
        magnitude,
        np.array([v_edges[0], v_edges[-1], u_edges[0], u_edges[-1]], dtype=np.float64),
    )


def plot_instrument_distortion_map(
    summary: InstrumentSummary, path: str, *, n_cells: int = 8
) -> None:
    """Write the per-instrument 2-D non-rotational distortion map.

    Left: the full post-twist residual field (translation and twist removed)
    averaged into a grid over the field of view -- the non-rotational
    distortion, radial and non-radial together. Right: the non-radial component
    alone (each residual with its radial projection removed), which isolates
    tangential and decentering distortion that a purely radial model does not
    capture.

    Parameters:
        summary: An instrument summary with pooled residuals.
        path: Output PNG path.
        n_cells: Grid resolution per axis.

    Raises:
        ValueError: if the summary carries no pooled residual data.
    """
    pooled = summary.pooled_radial
    if pooled is None:
        raise ValueError('summary has no pooled residual data to plot')

    pred = pooled.predicted_vu
    resid = pooled.residual_vu
    center = np.asarray(pooled.center_vu, dtype=np.float64)
    offset = pred - center
    rho = np.hypot(offset[:, 0], offset[:, 1])
    rhat = np.zeros_like(offset)
    safe = rho > 0.0
    rhat[safe] = offset[safe] / rho[safe, None]
    radial_comp = np.sum(resid * rhat, axis=1)
    nonradial_vec = resid - radial_comp[:, None] * rhat

    full_rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    nonradial_rms = float(np.sqrt(np.mean(np.sum(nonradial_vec**2, axis=1))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, vectors, title in (
        (axes[0], resid, f'Full non-rotational distortion\nRMS {full_rms:.3f} px'),
        (axes[1], nonradial_vec, f'Non-radial component only\nRMS {nonradial_rms:.3f} px'),
    ):
        centers, means, magnitude, extent = _bin_field(pred, vectors, pooled.image_shape, n_cells)
        ax.imshow(
            magnitude,
            origin='upper',
            extent=(extent[2], extent[3], extent[1], extent[0]),
            cmap='viridis',
            alpha=0.6,
            aspect='equal',
        )
        typical = float(np.median(np.hypot(means[:, 0], means[:, 1]))) if means.size else 0.0
        scale = (0.09 * max(pooled.image_shape) / typical) if typical > 0 else 1.0
        ax.quiver(
            centers[:, 1],
            centers[:, 0],
            means[:, 1] * scale,
            means[:, 0] * scale,
            angles='xy',
            scale_units='xy',
            scale=1.0,
            color='#ffffff',
            width=0.005,
        )
        ax.set_title(f'{title} (arrows x{scale:.0f})', fontsize=10)
        ax.set_xlabel('u (px)')
        ax.set_ylabel('v (px)')
    fig.suptitle(f'{summary.label}: non-rotational distortion field', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)


def plot_instrument_radial(summary: InstrumentSummary, path: str) -> None:
    """Write the per-instrument pooled radial distortion figure.

    Parameters:
        summary: An instrument summary with pooled radial residuals.
        path: Output PNG path.

    Raises:
        ValueError: if the summary carries no pooled radial data.
    """
    pooled = summary.pooled_radial
    if pooled is None:
        raise ValueError('summary has no pooled radial data to plot')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(pooled.rho_n)
    rho_n_sorted = pooled.rho_n[order]
    axes[0].scatter(
        pooled.rho_n, pooled.radial_px, s=8, color='#3070c0', alpha=0.4, label='per star'
    )
    model_curve = pooled.model.radial_displacement_px(rho_n_sorted * pooled.model.rho_ref_px)
    k1 = pooled.model.k_sim[0]
    k2 = pooled.model.k_sim[1] if len(pooled.model.k_sim) > 1 else 0.0
    axes[0].plot(
        rho_n_sorted,
        model_curve,
        color='#d02020',
        lw=2.0,
        label=f'aggregate model k1 {k1:+.2e}, k2 {k2:+.2e}',
    )
    axes[0].axhline(0.0, color='#909090', lw=0.8)
    axes[0].set_xlabel('normalized field radius')
    axes[0].set_ylabel('radial residual (px)')
    axes[0].set_title(f'{summary.label}: pooled radial distortion', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Binned RMS of the post-fit residual vs radius: the centroid-noise floor,
    # which rises toward the edge where PSF-edge centroiding degrades.
    bins = np.linspace(0.0, float(pooled.rho_n.max()) + 1e-6, 9)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rms = np.full(centers.size, np.nan)
    for i in range(centers.size):
        sel = (pooled.rho_n >= bins[i]) & (pooled.rho_n < bins[i + 1])
        if np.any(sel):
            rms[i] = math.sqrt(float(np.mean(pooled.residual_after_fit_px[sel] ** 2)))
    axes[1].plot(centers, rms, 'o-', color='#208040')
    axes[1].set_xlabel('normalized field radius')
    axes[1].set_ylabel('post-fit radial residual RMS (px)')
    axes[1].set_title('Residual floor vs field radius\n(centroid + astrometric noise)', fontsize=11)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
