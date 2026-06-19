"""Annotated-summary-PNG rendering shared by the autonomous and manual paths.

The autonomous pipeline (``nav.navigate_image_files._write_summary_png``)
and the manual-navigation dialog both produce a labelled overlay PNG of
the source image with each NavModel's annotation drawn on top.  The
rendering logic lives here so both code paths produce visually
identical PNGs from the same ``(obs, annotations, offset_px)`` triple.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nav.support.image import apply_linear_gamma_stretch
from nav.support.types import NDArrayFloatType, NDArrayUint8Type

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from nav.annotation import Annotations
    from nav.obs import ObsSnapshot

__all__ = [
    'grayscale_to_rgb_with_quantile_stretch',
    'render_annotated_summary_rgb',
]


def render_annotated_summary_rgb(
    obs: ObsSnapshot,
    annotations: Annotations,
    offset_px: tuple[float, float] = (0.0, 0.0),
) -> NDArrayUint8Type:
    """Composite ``obs.data`` with ``annotations.combine`` at ``offset_px``.

    Builds a quantile-stretched grayscale background from the FOV image,
    asks the annotations layer for its FOV-shaped RGB overlay at the
    requested offset, and replaces every pixel where the overlay carries
    any non-zero color channel.  When the annotations collection is
    empty, the returned RGB is the source-image grayscale alone — so the
    result is always a faithful record of what the navigator saw.

    Parameters:
        obs: Observation snapshot supplying the background image.
        annotations: Merged ``Annotations`` collection from every
            NavModel that contributed.
        offset_px: ``(dv, du)`` offset that shifts the overlay onto the
            best-fit pose.  The convention matches every other offset
            in the pipeline: predicted + offset = actual.

    Returns:
        ``(H, W, 3)`` uint8 RGB array in FOV coordinates.
    """
    image_fov = np.asarray(obs.data, dtype=np.float64)
    rgb = grayscale_to_rgb_with_quantile_stretch(image_fov)
    overlay = annotations.combine(offset=offset_px)
    if overlay is not None:
        mask = overlay.any(axis=-1)
        rgb[mask] = overlay[mask]
    return rgb


def grayscale_to_rgb_with_quantile_stretch(image: NDArrayFloatType) -> NDArrayUint8Type:
    """Build a uint8 RGB grayscale background from a float image.

    The black point is fixed at the 0.001 quantile.  The white point
    adapts to the number of "bright" pixels in the image: the default
    0.999 quantile clips the top 0.1 % of pixels, but on an image with
    only a handful of bright outliers (a sparse star field over dark
    sky, a distant body against empty sky) that fixed clip count
    saturates every bright pixel to 255 even though the brightest is
    much brighter than the rest.

    The fix counts the bright outliers via a robust median + 15 * MAD
    threshold and clips at most half of them — so the brightest few
    are saturated but the remaining bright pixels keep their relative
    brightness ordering.  When the image carries many bright pixels
    (a body filling the FOV, a busy ring scene) the original 0.1 %
    behavior dominates and nothing about the existing visualization
    changes.
    """
    finite = np.isfinite(image)
    if not finite.any():
        clean = np.zeros_like(image)
        black = 0.0
        white = 1.0
    else:
        clean = np.where(finite, image, 0.0)
        finite_values = image[finite]
        n_finite = int(finite_values.size)
        black = float(np.quantile(finite_values, 0.001))

        default_clip_count = max(1, round(n_finite * 0.001))
        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)))
        if mad > 0.0:
            # 15 * MAD ~ 10 * sigma for gaussian noise (MAD = 0.6745 *
            # sigma).  Even on a 1 M-pixel detector a 10-sigma threshold
            # catches no noise pixels (P > 10 sigma ~ 1.5e-23) so the
            # bright-pixel count reflects real outliers (stars, body
            # limbs, ring edges) without polluting the count with the
            # gaussian-noise tail.
            bright_threshold = median + 15.0 * mad
            n_bright = int(np.sum(finite_values > bright_threshold))
        else:
            n_bright = 0
        if n_bright == 0:
            clip_count = default_clip_count
        else:
            # Clip only the brightest 5 % of outliers — the remaining
            # 95 % stretch across the visible 0..255 range and preserve
            # their relative brightness ordering.  Half-clipping
            # (n_bright // 2) was too aggressive for "few bright
            # pixels" scenes where the user wants to see the gradient
            # within the bright region (a sparse star field, a small
            # body against dark sky, a thin ring against empty sky):
            # half the brights still saturate to 255 and the visual is
            # over-exposed.  5 % keeps that count small (1 of 20)
            # while still saturating the very brightest pixel so the
            # overall stretch is anchored.
            clip_count = min(default_clip_count, max(1, n_bright // 20))

        clip_quantile = 1.0 - clip_count / n_finite
        white = float(np.quantile(finite_values, clip_quantile))
        if white <= black:
            white = float(np.nextafter(black, np.inf))

    stretched = apply_linear_gamma_stretch(clean, black=black, white=white, gamma=1.0)
    gray = (stretched * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)
