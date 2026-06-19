"""Histogram-based stretch control for the mosaic viewer.

Provides :class:`HistogramStretchWidget`: a custom-painted Qt widget that shows
the pixel-value distribution of the displayed image with two draggable
indicators (black and white). Each indicator sits below the histogram, draws a
vertical line through the plot, and emits its new value through a caller-supplied
callback.

Designed to replace the linear black/white sliders in
``nav_mosaic_display_rings`` and ``nav_mosaic_display_body``; the gamma slider
is kept alongside it. Histogram counts are shown on a logarithmic vertical
axis so small populations remain visible against a dominant peak.
"""

from collections.abc import Callable

import numpy as np
import numpy.ma as ma
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygon,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

_HISTOGRAM_BINS = 256
_INDICATOR_STRIP_HEIGHT_PX = 18
_INDICATOR_HALF_WIDTH_PX = 6
_INDICATOR_HEIGHT_PX = 10
_PICK_THRESHOLD_PX = 12
_LABEL_AREA_PX = 16
_PLOT_MARGIN_PX = 4
_DEFAULT_MIN_HEIGHT_PX = 140

_BG_COLOR = QColor(245, 245, 245)
_PLOT_BG_COLOR = QColor(255, 255, 255)
_PLOT_BORDER_COLOR = QColor(180, 180, 180)
_BAR_COLOR = QColor(80, 110, 180)
_BLACK_MARKER_COLOR = QColor(20, 20, 20)
_WHITE_MARKER_COLOR = QColor(220, 80, 80)
_LABEL_COLOR = QColor(60, 60, 60)


class HistogramStretchWidget(QWidget):
    """Histogram with draggable black/white indicators driving a stretch.

    The widget owns the black and white values; callbacks are invoked when the
    user drags an indicator. Histogram counts and x-axis range are derived from
    image data passed to :meth:`set_data`.
    """

    def __init__(
        self,
        *,
        on_black_changed: Callable[[float], None],
        on_white_changed: Callable[[float], None],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the widget.

        Parameters:
            on_black_changed: Called with the new black level (data units) when
                the black indicator is dragged.
            on_white_changed: Called with the new white level (data units) when
                the white indicator is dragged.
            parent: Optional Qt parent.
        """
        super().__init__(parent)
        self._on_black_changed = on_black_changed
        self._on_white_changed = on_white_changed

        self._vmin: float = 0.0
        self._vmax: float = 1.0
        self._black: float = 0.0
        self._white: float = 1.0
        self._counts: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._max_log_count: float = 0.0
        self._drag: str | None = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(_DEFAULT_MIN_HEIGHT_PX)

    def set_data(self, image_ma: ma.MaskedArray | None) -> None:
        """Recompute the histogram from valid pixels in ``image_ma``.

        Auto-scales the x-axis to the data ``[min, max]`` and the y-axis to the
        maximum bin count (via ``log1p`` for visibility). Existing indicator
        positions are clamped into the new range.

        Parameters:
            image_ma: 2-D masked array, or ``None`` to clear the histogram.
        """
        if image_ma is None:
            self._counts = None
            self._edges = None
            self._max_log_count = 0.0
            self.update()
            return
        valid = np.asarray(image_ma.compressed(), dtype=np.float64)
        if valid.size > 0:
            valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            self._vmin, self._vmax = 0.0, 1.0
            self._counts = None
            self._edges = None
            self._max_log_count = 0.0
            self.update()
            return
        vmin = float(np.min(valid))
        vmax = float(np.max(valid))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self._vmin = vmin
        self._vmax = vmax
        counts, edges = np.histogram(valid, bins=_HISTOGRAM_BINS, range=(vmin, vmax))
        log_counts = np.log1p(counts.astype(np.float64))
        self._counts = log_counts
        self._edges = edges
        self._max_log_count = float(log_counts.max()) if log_counts.size > 0 else 0.0
        self._black = float(np.clip(self._black, vmin, vmax))
        self._white = float(np.clip(self._white, vmin, vmax))
        self.update()

    def set_range(self, vmin: float, vmax: float) -> None:
        """Set the x-axis range without touching the histogram bin data.

        Useful when the caller wants to sync the indicator range with a known
        data range before histogram data has been pushed.
        """
        if vmax <= vmin:
            vmax = vmin + 1.0
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self._black = float(np.clip(self._black, self._vmin, self._vmax))
        self._white = float(np.clip(self._white, self._vmin, self._vmax))
        self.update()

    def set_values(self, black: float, white: float) -> None:
        """Move both indicators without firing callbacks."""
        self._black = float(black)
        self._white = float(white)
        self.update()

    def get_values(self) -> tuple[float, float]:
        """Return ``(black, white)`` in data units."""
        return self._black, self._white

    def _plot_rect(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` of the histogram plot area in widget pixels."""
        px = _PLOT_MARGIN_PX
        py = _PLOT_MARGIN_PX
        pw = max(0, self.width() - 2 * _PLOT_MARGIN_PX)
        ph = max(
            0,
            self.height() - _PLOT_MARGIN_PX - _INDICATOR_STRIP_HEIGHT_PX - _LABEL_AREA_PX,
        )
        return px, py, pw, ph

    def _value_to_x(self, value: float) -> int:
        px, _py, pw, _ph = self._plot_rect()
        if self._vmax <= self._vmin or pw <= 0:
            return px
        frac = (value - self._vmin) / (self._vmax - self._vmin)
        frac = max(0.0, min(1.0, frac))
        return round(px + frac * pw)

    def _x_to_value(self, x: int) -> float:
        px, _py, pw, _ph = self._plot_rect()
        if pw <= 0:
            return self._vmin
        frac = (x - px) / pw
        frac = max(0.0, min(1.0, frac))
        return self._vmin + frac * (self._vmax - self._vmin)

    def paintEvent(self, event: QPaintEvent | None) -> None:
        """Render histogram, indicator lines, triangle markers, and value labels."""
        del event
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), _BG_COLOR)
            plot_x, plot_y, plot_w, plot_h = self._plot_rect()
            if plot_w <= 0 or plot_h <= 0:
                return
            painter.fillRect(plot_x, plot_y, plot_w, plot_h, _PLOT_BG_COLOR)
            painter.setPen(QPen(_PLOT_BORDER_COLOR, 1))
            painter.drawRect(plot_x, plot_y, plot_w, plot_h)
            self._paint_bars(painter, plot_y, plot_h)
            self._paint_indicator_lines(painter, plot_y, plot_h)
            self._paint_indicator_triangles(painter, plot_y + plot_h)
            self._paint_labels(painter, plot_x, plot_y + plot_h)
        finally:
            painter.end()

    def _paint_bars(self, painter: QPainter, py: int, ph: int) -> None:
        if (
            self._counts is None
            or self._edges is None
            or self._max_log_count <= 0
            or self._counts.size == 0
        ):
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_BAR_COLOR))
        n_bins = int(self._counts.size)
        base_y = py + ph
        scale = ph / self._max_log_count
        edges = self._edges
        counts = self._counts
        for i in range(n_bins):
            x0 = self._value_to_x(float(edges[i]))
            x1 = self._value_to_x(float(edges[i + 1]))
            bar_w = max(1, x1 - x0)
            bar_h = round(float(counts[i]) * scale)
            if bar_h <= 0:
                continue
            painter.drawRect(x0, base_y - bar_h, bar_w, bar_h)

    def _paint_indicator_lines(self, painter: QPainter, py: int, ph: int) -> None:
        black_x = self._value_to_x(self._black)
        white_x = self._value_to_x(self._white)
        painter.setPen(QPen(_BLACK_MARKER_COLOR, 1, Qt.PenStyle.SolidLine))
        painter.drawLine(black_x, py, black_x, py + ph)
        painter.setPen(QPen(_WHITE_MARKER_COLOR, 1, Qt.PenStyle.SolidLine))
        painter.drawLine(white_x, py, white_x, py + ph)

    def _paint_indicator_triangles(self, painter: QPainter, top_y: int) -> None:
        black_x = self._value_to_x(self._black)
        white_x = self._value_to_x(self._white)
        _draw_triangle(painter, black_x, top_y, _BLACK_MARKER_COLOR)
        _draw_triangle(painter, white_x, top_y, _WHITE_MARKER_COLOR)

    def _paint_labels(self, painter: QPainter, left_x: int, strip_top: int) -> None:
        baseline_y = strip_top + _INDICATOR_STRIP_HEIGHT_PX + _LABEL_AREA_PX - 4
        if baseline_y > self.height():
            return
        painter.setPen(QPen(_LABEL_COLOR, 1))
        font = painter.font()
        size = font.pointSizeF()
        if size > 0:
            font.setPointSizeF(max(7.0, size - 1.0))
        painter.setFont(font)
        text = f'B = {self._black:.4g}    W = {self._white:.4g}'
        painter.drawText(left_x, baseline_y, text)

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        """Begin dragging whichever indicator the click is closest to."""
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        x = int(event.position().x())
        black_x = self._value_to_x(self._black)
        white_x = self._value_to_x(self._white)
        d_black = abs(x - black_x)
        d_white = abs(x - white_x)
        if min(d_black, d_white) > _PICK_THRESHOLD_PX:
            return
        if d_black < d_white:
            self._drag = 'black'
        elif d_white < d_black:
            self._drag = 'white'
        else:
            # Coincident / equidistant indicators: tie-break by which side of
            # the marker was clicked so the white indicator stays reachable when
            # it sits on top of black (clicking to its right grabs white, else
            # black) -- otherwise white could never be separated from black.
            self._drag = 'white' if x > black_x else 'black'
        self._handle_drag_to(x)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        """Update the active indicator while the mouse is dragged."""
        if event is None or self._drag is None:
            return
        self._handle_drag_to(int(event.position().x()))

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        """Release any active drag on left-button up."""
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = None

    def _handle_drag_to(self, x: int) -> None:
        epsilon = max(1e-12, (self._vmax - self._vmin) * 1e-6)
        new_val = self._x_to_value(x)
        if self._drag == 'black':
            new_val = max(self._vmin, min(new_val, self._white - epsilon))
            self._black = new_val
            self.update()
            self._on_black_changed(self._black)
        elif self._drag == 'white':
            new_val = min(self._vmax, max(new_val, self._black + epsilon))
            self._white = new_val
            self.update()
            self._on_white_changed(self._white)


def _draw_triangle(painter: QPainter, cx: int, top_y: int, color: QColor) -> None:
    poly = QPolygon(
        [
            QPoint(cx - _INDICATOR_HALF_WIDTH_PX, top_y + _INDICATOR_HEIGHT_PX),
            QPoint(cx + _INDICATOR_HALF_WIDTH_PX, top_y + _INDICATOR_HEIGHT_PX),
            QPoint(cx, top_y + 1),
        ]
    )
    painter.setBrush(QBrush(color))
    painter.setPen(QPen(color, 1))
    painter.drawPolygon(poly)
