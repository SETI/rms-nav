"""Matplotlib QtAgg helpers (third-party constructors lack usable stubs under mypy)."""

from typing import Any

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


def canvas_draw_idle(canvas: FigureCanvasQTAgg | Any) -> None:
    """Schedule a deferred redraw on a Qt Agg canvas.

    Parameters:
        canvas: ``FigureCanvasQTAgg``, or any object with a callable ``draw_idle`` method.

    Returns:
        ``None``.
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
