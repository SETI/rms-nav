"""Render the sweep response curves as figures for the performance report.

Reads the per-sweep JSON written by :mod:`sim_sweep_runner` and writes PNG charts
under ``docs/simulator_report/_figures/``: the sub-pixel and wide-range offset
accuracy of every technique, and the camera-roll recovery.  Run as part of
``python -m tests.integration.sim_sweep_runner`` (or standalone) -- not part of
pytest.  Requires ``matplotlib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

_RESULTS_ROOT = Path(__file__).parent / 'sim_sweeps' / 'results'
_FIGURES_ROOT = Path(__file__).parent.parent.parent / 'docs' / 'simulator_report' / '_figures'

# Technique -> (fine sweep, wide sweep, marker, label) for the offset figures.
_OFFSET_TECHNIQUES = [
    ('disc', 'disc_offset_fine', 'disc_offset_wide', 'o', 'BodyDiscCorrelateNav'),
    ('blob', 'blob_offset_fine', 'blob_offset_wide', 's', 'BodyBlobNav'),
    ('limb', 'limb_offset_fine', 'limb_offset_wide', '^', 'BodyLimbNav'),
    ('ring', 'ring_offset_fine', 'ring_offset_wide', 'D', 'RingEdgeNav'),
    ('star', 'star_offset_fine', 'star_offset_wide', 'v', 'StarFieldFromCatalogNav'),
]


def _load(name: str) -> list[dict[str, Any]]:
    """Return the rows of one sweep result, or an empty list if absent."""
    path = _RESULTS_ROOT / f'{name}.json'
    if not path.is_file():
        return []
    return list(json.loads(path.read_text())['rows'])


def _xy(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Return (value, key) pairs for rows where ``key`` is populated."""
    xs = [r['value'] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


def _offset_figure(which: str, title: str, out_name: str) -> None:
    """Plot offset error vs planted offset for every technique's ``which`` sweep."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for _key, fine, wide, marker, label in _OFFSET_TECHNIQUES:
        name = fine if which == 'fine' else wide
        xs, ys = _xy(_load(name), 'offset_error_px')
        if xs:
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('planted offset (px)')
    ax.set_ylabel('recovered-offset error (px, log scale)')
    ax.set_title(title)
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / out_name, dpi=110)
    plt.close(fig)


def _star_regime_figure() -> None:
    """Overlay the dim- and bright-star field sweeps to show the SNR crossover.

    The two sweeps share geometry and planted offset; only the stars' brightness
    differs.  The dim field rides the PSF-refined error floor; the bright field
    keeps the moment centroid (above the configured SNR ceiling) and reaches a
    far smaller error -- the visible payoff of the per-star moment/PSF choice.
    """
    dim = _load('star_offset_fine')
    bright = _load('star_offset_fine_bright')
    if not dim and not bright:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for rows, marker, label in [
        (dim, 'v', 'dim field (PSF-refined, vmag 3-4)'),
        (bright, 'o', 'bright field (moment, vmag 0-0.8)'),
    ]:
        xs, ys = _xy(rows, 'offset_error_px')
        if xs:
            ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, label=label)
    ax.set_yscale('log')
    ax.set_xlabel('planted offset (px)')
    ax.set_ylabel('recovered-offset error (px, log scale)')
    ax.set_title('Star-field sub-pixel accuracy: dim vs bright (PSF-refine crossover)')
    ax.grid(True, which='both', ls=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'star_regime_accuracy.png', dpi=110)
    plt.close(fig)


def _rotation_figure() -> None:
    """Plot recovered-roll error vs planted roll for the star-field sweep."""
    rows = _load('star_rotation')
    xs, ys = _xy(rows, 'rotation_error_deg')
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(xs, ys, marker='o', ms=5, lw=1.2, color='tab:purple')
    ax.set_xlabel('planted camera roll (deg)')
    ax.set_ylabel('recovered-roll error (deg)')
    ax.set_title('Star-field camera-roll recovery')
    ax.grid(True, ls=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(_FIGURES_ROOT / 'rotation_accuracy.png', dpi=110)
    plt.close(fig)


def generate_plots() -> list[Path]:
    """Render all report figures from the sweep results; return their paths."""
    _FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    _offset_figure(
        'fine',
        'Sub-pixel offset accuracy by technique (dense fractional sweep)',
        'offset_accuracy_fine.png',
    )
    _offset_figure(
        'wide',
        'Wide-range offset accuracy by technique (to the navigable ceiling)',
        'offset_accuracy_wide.png',
    )
    _star_regime_figure()
    _rotation_figure()
    return sorted(_FIGURES_ROOT.glob('*.png'))


if __name__ == '__main__':
    for path in generate_plots():
        print(f'wrote {path}')
