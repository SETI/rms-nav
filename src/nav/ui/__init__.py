"""PyQt6 and matplotlib UI for navigation (mosaics, manual offset, shared widgets).

This directory is a regular package so ``nav.ui`` and ``nav.ui.mosaic_viewer`` share
the same import layout.

Modules:

    ``common``
        Shared UI helpers (e.g. stretch controls, zoom/pan) used across dialogs and
        the mosaic viewer.
    ``manual_nav_dialog``
        Matplotlib-based manual navigation dialog.

Subpackages:

    ``mosaic_viewer``
        Ring and body mosaic windows, projections, graticule, and tiled image display.
"""

__all__ = []
