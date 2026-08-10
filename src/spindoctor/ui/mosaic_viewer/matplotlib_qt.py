"""Matplotlib QtAgg helpers (third-party constructors lack usable stubs under mypy)."""

# Annotations must stay strings: the docs build mocks the Qt backend, and an
# evaluated ``FigureCanvasQTAgg | object`` union on a mocked class raises at
# import time, which would knock this module out of the API reference.
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def new_figure_canvas_qtagg(fig: Figure) -> FigureCanvasQTAgg:
    """Construct a Qt Agg canvas for ``fig``.

    Parameters:
        fig: Matplotlib figure to embed.

    Returns:
        A ``FigureCanvasQTAgg`` wrapping ``fig``.
    """
    if not isinstance(fig, Figure):
        raise TypeError(f'fig must be a matplotlib.figure.Figure, got {type(fig).__name__}')
    return FigureCanvasQTAgg(fig)  # type: ignore[no-untyped-call]


def canvas_draw_idle(canvas: FigureCanvasQTAgg | object) -> None:
    """Schedule a deferred redraw on a Qt Agg canvas.

    Parameters:
        canvas: A :class:`~matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`, or any
            object that exposes a callable ``draw_idle`` like
            :meth:`FigureCanvasQTAgg.draw_idle` (defers painting until the Qt event loop
            runs). Callers should narrow with ``isinstance(..., FigureCanvasQTAgg)`` before
            using canvas-specific APIs beyond ``draw_idle``.
    """
    if isinstance(canvas, FigureCanvasQTAgg):
        canvas.draw_idle()  # type: ignore[no-untyped-call]
        return
    draw_idle = getattr(canvas, 'draw_idle', None)
    if callable(draw_idle):
        draw_idle()
        return
    raise TypeError(
        f'canvas must be a FigureCanvasQTAgg (or expose draw_idle), got {type(canvas).__name__}'
    )
