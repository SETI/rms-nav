from collections.abc import Callable
from typing import Optional, Any

import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QFormLayout,
    QHBoxLayout,
    QSlider,
    QWidget,
)


class ZoomPanController:
    """
    Reusable zoom/pan handler using the logic from create_simulated_body_model.
    Plug into an existing UI by providing:
      - label: QLabel showing the image (mouse events are routed by the UI)
      - scroll_area: QScrollArea containing the label
      - get_zoom(): float — returns current zoom factor
      - set_zoom(z: float): None — sets zoom factor and updates any labels
      - update_display(): None — applies current zoom/pan to the UI
      - set_zoom_label_text(label: str): Optional[Callable[[str], None]] — optional
        callback that receives the fully formatted zoom label string to display
    The controller adjusts scrollbars for panning and zoom anchoring.
    """

    def __init__(
        self,
        *,
        label: QLabel,
        scroll_area: QScrollArea,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
        update_display: Callable[[], None],
        set_zoom_label_text: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._label = label
        self._scroll = scroll_area
        self._get_zoom = get_zoom
        self._set_zoom = set_zoom
        self._update_display = update_display
        self._set_zoom_label_text = set_zoom_label_text

        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_scroll_xy: Optional[tuple[int, int]] = None

    # Event handlers (UI should call these from its own handlers)
    def on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
            sh = self._scroll.horizontalScrollBar()
            sv = self._scroll.verticalScrollBar()
            self._drag_start_scroll_xy = (
                sh.value() if sh is not None else 0,
                sv.value() if sv is not None else 0,
            )

    def on_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_start_pos is not None and self._drag_start_scroll_xy is not None:
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._drag_start_pos
            sh = self._scroll.horizontalScrollBar()
            sv = self._scroll.verticalScrollBar()
            if sh is not None:
                new_h = int(
                    max(0, min(sh.maximum(), self._drag_start_scroll_xy[0] - delta.x()))
                )
                sh.setValue(new_h)
            if sv is not None:
                new_v = int(
                    max(0, min(sv.maximum(), self._drag_start_scroll_xy[1] - delta.y()))
                )
                sv.setValue(new_v)

    def on_mouse_release(self, _event: QMouseEvent) -> None:
        self._drag_start_pos = None
        self._drag_start_scroll_xy = None

    def on_wheel(self, event: QWheelEvent) -> None:
        label_pos = event.position().toPoint()
        viewport = self._scroll.viewport()
        if viewport is None:
            return
        viewport_pos = self._label.mapTo(viewport, label_pos)
        vx = viewport_pos.x()
        vy = viewport_pos.y()
        sh = self._scroll.horizontalScrollBar()
        sv = self._scroll.verticalScrollBar()
        if sh is None or sv is None:
            return
        scaled_x = vx + sh.value()
        scaled_y = vy + sv.value()
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        self._zoom_at_point(factor, vx, vy, scaled_x, scaled_y)

    def zoom_in_center(self) -> None:
        viewport = self._scroll.viewport()
        if viewport is None:
            return
        cx = viewport.width() // 2
        cy = viewport.height() // 2
        sh = self._scroll.horizontalScrollBar()
        sv = self._scroll.verticalScrollBar()
        if sh is None or sv is None:
            return
        scaled_x = cx + sh.value()
        scaled_y = cy + sv.value()
        self._zoom_at_point(1.2, cx, cy, scaled_x, scaled_y)

    def zoom_out_center(self) -> None:
        viewport = self._scroll.viewport()
        if viewport is None:
            return
        cx = viewport.width() // 2
        cy = viewport.height() // 2
        sh = self._scroll.horizontalScrollBar()
        sv = self._scroll.verticalScrollBar()
        if sh is None or sv is None:
            return
        scaled_x = cx + sh.value()
        scaled_y = cy + sv.value()
        self._zoom_at_point(1.0 / 1.2, cx, cy, scaled_x, scaled_y)

    def reset_view(self) -> None:
        self._set_zoom(1.0)
        if self._set_zoom_label_text:
            self._set_zoom_label_text('zoom: 1.00x')
        # Let the UI handle any additional state
        self._update_display()

    # Public zoom-at-point API
    def zoom_at_point(
        self,
        factor: float,
        viewport_x: int,
        viewport_y: int,
        scaled_x: float,
        scaled_y: float,
    ) -> None:
        """
        Zoom by a factor anchored at the given viewport coordinates, where
        scaled_x/y are the corresponding coordinates in the scaled image space.
        """
        self._zoom_at_point(factor, viewport_x, viewport_y, scaled_x, scaled_y)

    # Internal
    def _zoom_at_point(
        self, factor: float, vx: int, vy: int, scaled_x: float, scaled_y: float
    ) -> None:
        old_zoom = self._get_zoom()
        new_zoom = float(np.clip(old_zoom * factor, 0.1, 50.0))
        if new_zoom == old_zoom:
            return
        img_x = scaled_x / old_zoom
        img_y = scaled_y / old_zoom
        new_scroll_x = img_x * new_zoom - vx
        new_scroll_y = img_y * new_zoom - vy
        self._set_zoom(new_zoom)
        if self._set_zoom_label_text:
            self._set_zoom_label_text(f'zoom: {new_zoom:.2f}x')
        # Set scroll positions
        sh = self._scroll.horizontalScrollBar()
        sv = self._scroll.verticalScrollBar()
        if sh is not None:
            sh.setValue(int(max(0, min(sh.maximum(), new_scroll_x))))
        if sv is not None:
            sv.setValue(int(max(0, min(sv.maximum(), new_scroll_y))))
        self._update_display()


def build_stretch_controls(
    form: QFormLayout,
    *,
    img_min: float,
    img_max: float,
    black_init: float,
    white_init: float,
    gamma_init: float,
    on_black_changed: Callable[[float], None],
    on_white_changed: Callable[[float], None],
    on_gamma_changed: Callable[[float], None],
) -> dict[str, Any]:
    """
    Construct black/white/gamma controls matching manual_nav_dialog behavior with shared formatting.
    Returns dict with widgets and mappers:
      slider_black, label_black, slider_white, label_white, slider_gamma, label_gamma,
      to_slider(val)->int, from_slider(pos)->float, set_values(black, white, gamma)
    """
    # Sliders
    slider_black = QSlider(Qt.Orientation.Horizontal)
    slider_black.setRange(0, 1000)
    slider_white = QSlider(Qt.Orientation.Horizontal)
    slider_white.setRange(0, 1000)
    slider_gamma = QSlider(Qt.Orientation.Horizontal)
    slider_gamma.setRange(10, 500)  # 0.10..5.00
    # Real-time updates during drags
    slider_black.setTracking(True)
    slider_white.setTracking(True)

    # Labels (fixed width, 5 digits after decimal)
    label_black = QLabel(f'{black_init:.5f}')
    label_white = QLabel(f'{white_init:.5f}')
    label_gamma = QLabel(f'{gamma_init:.5f}')
    for lbl in (label_black, label_white, label_gamma):
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setMinimumWidth(80)

    # Mapping helpers
    lo = img_min
    hi = img_max if img_max > img_min else (img_min + 1.0)

    def to_slider(val: float) -> int:
        return round(1000.0 * (val - lo) / (hi - lo))

    def from_slider(pos: int) -> float:
        return lo + (hi - lo) * (pos / 1000.0)

    # Initial positions
    slider_black.setValue(to_slider(black_init))
    slider_white.setValue(to_slider(white_init))
    slider_gamma.setValue(round(gamma_init * 100))

    # Wire up signals
    def _black_slot(v: int) -> None:
        val = from_slider(v)
        label_black.setText(f'{val:.5f}')
        on_black_changed(val)

    def _white_slot(v: int) -> None:
        val = from_slider(v)
        label_white.setText(f'{val:.5f}')
        on_white_changed(val)

    def _gamma_slot(v: int) -> None:
        val = max(0.10, v / 100.0)
        label_gamma.setText(f'{val:.5f}')
        on_gamma_changed(val)

    slider_black.valueChanged.connect(_black_slot)
    slider_white.valueChanged.connect(_white_slot)
    slider_gamma.valueChanged.connect(_gamma_slot)

    # Rows in form
    row_b = QHBoxLayout()
    row_b.addWidget(slider_black)
    row_b.addWidget(label_black)
    row_w = QHBoxLayout()
    row_w.addWidget(slider_white)
    row_w.addWidget(label_white)
    row_g = QHBoxLayout()
    row_g.addWidget(slider_gamma)
    row_g.addWidget(label_gamma)

    # Workaround: QFormLayout requires QWidget; build holders
    holder_b = QWidget()
    holder_b.setLayout(row_b)
    form.addRow('Black point:', holder_b)
    holder_w = QWidget()
    holder_w.setLayout(row_w)
    form.addRow('White point:', holder_w)
    holder_g = QWidget()
    holder_g.setLayout(row_g)
    form.addRow('Gamma:', holder_g)

    def set_values(black: float, white: float, gamma: float) -> None:
        # Block signals to avoid recursion
        slider_black.blockSignals(True)
        slider_white.blockSignals(True)
        slider_gamma.blockSignals(True)
        slider_black.setValue(to_slider(black))
        slider_white.setValue(to_slider(white))
        slider_gamma.setValue(round(gamma * 100))
        slider_black.blockSignals(False)
        slider_white.blockSignals(False)
        slider_gamma.blockSignals(False)
        label_black.setText(f'{black:.5f}')
        label_white.setText(f'{white:.5f}')
        label_gamma.setText(f'{gamma:.5f}')

    def set_range(new_img_min: float, new_img_max: float) -> None:
        nonlocal lo, hi
        lo = float(new_img_min)
        hi = (
            float(new_img_max)
            if new_img_max > new_img_min
            else (float(new_img_min) + 1.0)
        )

    return {
        'slider_black': slider_black,
        'label_black': label_black,
        'slider_white': slider_white,
        'label_white': label_white,
        'slider_gamma': slider_gamma,
        'label_gamma': label_gamma,
        'to_slider': to_slider,
        'from_slider': from_slider,
        'set_values': set_values,
        'set_range': set_range,
    }
