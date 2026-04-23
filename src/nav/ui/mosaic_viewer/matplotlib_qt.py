"""Matplotlib QtAgg helpers (third-party constructors lack usable stubs under mypy)."""

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def new_figure_canvas_qtagg(fig: Figure) -> Any:
    """Return a ``FigureCanvasQTAgg`` for *fig*."""
    return FigureCanvasQTAgg(fig)  # type: ignore[no-untyped-call]


def canvas_draw_idle(canvas: Any) -> None:
    """Schedule a matplotlib canvas redraw."""
    canvas.draw_idle()
