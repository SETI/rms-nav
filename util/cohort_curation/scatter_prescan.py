"""Pixel-level gradient prescan for scattered_light candidates (Stage A).

Geometry metadata cannot predict whether a frame actually shows a
stray-light gradient (batch-1 lesson: 0/10 metadata-selected candidates
survived operator review).  This module reads the candidate images and
scores the amplitude of the low-order brightness surface against the
residual noise; only frames with a strong, well-resolved gradient are
kept as candidates.

Score: fit an affine plane to the 16x16 block-median image (medians are
robust to stars and cosmic rays) and report the plane's peak-to-peak
amplitude divided by the MAD-sigma of the residuals.  A flat star field
scores ~0-3; a frame with a real veiling gradient scores well above 5.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from vicar import VicarImage

# Derived from the same environment variable the pipeline uses (set by
# /seti/newnav/setup.sh); the literal fallback matches the operator host.
HOLDINGS_VOLUMES = (
    Path(os.environ.get('PDS3_HOLDINGS_DIR', '/mnt/ganymede/PDS/holdings'))
    / 'volumes'
)

MIN_SCORE = 5.0


def image_path_for(cand: dict) -> Path:
    """Local holdings path of a candidate's image file.

    Parameters:
        cand: Stage A candidate dict (volset, volume, filespec with a
            .LBL extension).

    Returns:
        The .IMG path under the local holdings volumes tree (existence
        is not checked here; the caller handles missing files).
    """
    spec = cand['filespec'].rsplit('.', 1)[0] + '.IMG'
    return HOLDINGS_VOLUMES / cand['volset'] / cand['volume'] / spec


def gradient_score(data: np.ndarray) -> tuple[float, float] | None:
    """(score, amplitude) of the low-order brightness surface.

    Parameters:
        data: 2-D image array in any physical units.

    Returns:
        ``(score, amplitude)`` where score is the plane peak-to-peak
        divided by the MAD-sigma of the residuals, or None when the
        image is too small or the residual sigma is zero.
    """
    h, w = data.shape
    b = 16
    if h < 4 * b or w < 4 * b:
        return None
    hh, ww = h // b * b, w // b * b
    blocks = data[:hh, :ww].reshape(hh // b, b, ww // b, b)
    med = np.median(blocks, axis=(1, 3))

    ny, nx = med.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.column_stack([np.ones(med.size), xx.ravel(), yy.ravel()])
    coef, *_ = np.linalg.lstsq(a, med.ravel(), rcond=None)
    plane = (a @ coef).reshape(med.shape)
    resid = med - plane
    sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    amp = float(plane.max() - plane.min())
    if sigma <= 0:
        return None
    return amp / sigma, amp


def prescan(cands: list[dict], *, keep: int) -> list[dict]:
    """Return the strongest-gradient candidates, best first.

    Parameters:
        cands: scattered_light candidates from the metadata scan.
        keep: Maximum number of candidates to return.

    Returns:
        Up to ``keep`` candidates with ``gradient_score`` /
        ``gradient_amplitude`` added to their selection dicts, sorted
        strongest-gradient first; unreadable or low-scoring frames are
        dropped.
    """
    scored: list[tuple[float, dict]] = []
    n_read = n_missing = 0
    for c in cands:
        path = image_path_for(c)
        if not path.exists():
            n_missing += 1
            continue
        try:
            data = VicarImage.from_file(
                str(path), strict=False).data_2d.astype(np.float64)
        except Exception:
            n_missing += 1
            continue
        n_read += 1
        result = gradient_score(data)
        if result is None:
            continue
        score, amp = result
        if score < MIN_SCORE:
            continue
        c['selection']['gradient_score'] = round(score, 1)
        c['selection']['gradient_amplitude'] = round(amp, 5)
        scored.append((score, c))
    scored.sort(key=lambda t: (-t[0], t[1]['filespec']))
    print(f'scatter prescan: {len(cands)} candidates, {n_read} read, '
          f'{n_missing} unreadable, {len(scored)} above score '
          f'{MIN_SCORE}, keeping {min(keep, len(scored))}')
    return [c for _, c in scored[:keep]]
