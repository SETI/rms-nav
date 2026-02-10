import math
from typing import Any, cast

import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from nav.config import Config
from nav.nav_model import NavModelCombined
from nav.obs import ObsSnapshot
from nav.support.correlate import masked_ncc, navigate_with_pyramid_kpeaks
from nav.support.types import NDArrayFloatType, NDArrayUint8Type
from nav.ui.common import ZoomPanController, build_stretch_controls


def _apply_stretch_gamma(
    image: NDArrayFloatType, black: float, white: float, gamma: float
) -> NDArrayUint8Type:
    """Apply black/white/gamma to a float image and return uint8 mono."""
    if white <= black:
        white = black + 1e-6
    scaled = np.clip((image - black) / (white - black), 0.0, 1.0)
    if gamma <= 0:
        gamma = 1.0
    scaled = np.power(scaled, 1.0 / gamma)
    return cast(NDArrayUint8Type, (scaled * 255.0).astype(np.uint8))


def _bilinear_sample_periodic(arr: NDArrayFloatType, y: float, x: float) -> float:
    """Periodic bilinear sample on 2D array arr at float indices (y, x)."""
    h, w = arr.shape
    # Wrap
    x = x % w
    y = y % h
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    dx = x - x0
    dy = y - y0
    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    return float(
        v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) + v10 * (1 - dx) * dy + v11 * dx * dy
    )


class _ImageLabel(QLabel):
    """Image label that forwards input events to the dialog handlers."""

    def __init__(self, owner_dialog: 'ManualNavDialog') -> None:
        super().__init__()
        self._owner = owner_dialog

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_press(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._owner._on_mouse_release(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            self._owner._on_wheel(event)


class ManualNavDialog(QDialog):
    """Manual navigation dialog for overlaying image and combined model."""

    def __init__(
        self,
        *,
        obs: ObsSnapshot,
        combined_model: NavModelCombined,
        config: Config | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Manual Navigation')
        self.setMinimumSize(1200, 800)

        self._obs = obs
        self._model = combined_model
        self._config = config

        self.setWindowTitle(f'Manual Navigation - {obs.abspath.name}')

        # Image and model arrays
        self._img_fov = obs.data  # V x U, float64
        self._img_ext = obs.extdata  # for correlation
        if (
            len(self._model.models) == 0
            or self._model.models[0].model_img is None
            or self._model.models[0].model_mask is None
        ):
            raise ValueError('Combined model is missing image or mask')
        self._model_img_ext = self._model.models[0].model_img
        self._model_mask_ext = self._model.models[0].model_mask

        # Stretch/gamma parameters
        self._black = float(np.quantile(self._img_fov, 0.001))
        self._white = float(np.quantile(self._img_fov, 0.999))
        if self._black >= self._white:
            self._white = self._black + 0.01
        self._gamma = 1.0
        self._alpha = 0.5  # transparency of model overlay
        # For slider mapping
        self._stretch_min = float(np.min(self._img_fov))
        self._stretch_max = float(np.max(self._img_fov))

        # Offsets (dv, du)
        self._dv = 0.0
        self._du = 0.0

        # Zoom/pan state
        self._zoom = 1.0
        self._drag_start_pos: QPoint | None = None
        self._drag_mode: str | None = None  # 'offset' (right)
        self._drag_start_offset: tuple[float, float] | None = None
        # Zoom rendering mode
        self._zoom_sharp = True

        # Precompute correlation surface once for status bar display
        self._precompute_correlation_surface()

        # Build UI
        self._build_ui()
        self._refresh_overlay()

    # ---- Correlation helpers ----

    def _precompute_correlation_surface(self) -> None:
        """Compute masked NCC surface on padded arrays; reuse for sampling."""
        image = np.asarray(self._img_ext, dtype=np.float64)
        model = np.asarray(self._model_img_ext, dtype=np.float64)
        mask = np.asarray(self._model_mask_ext, dtype=bool)
        # Pad to correlation convention used elsewhere (masked_ncc handles padding-independent math)
        # Here we directly compute the full NCC surface.
        self._corr_surface = masked_ncc(image, model, mask)
        self._corr_h, self._corr_w = self._corr_surface.shape

    def _offset_to_corr_indices(self, dv: float, du: float) -> tuple[float, float]:
        """Map signed (dv, du) to correlation surface indices for sampling."""
        # Same mapping as int_to_signed inverse: idx = s if s >= 0 else s + size
        y = dv if dv >= 0 else dv + self._corr_h
        x = du if du >= 0 else du + self._corr_w
        return y, x

    def _current_corr_value(self) -> float:
        y, x = self._offset_to_corr_indices(self._dv, self._du)
        return _bilinear_sample_periodic(self._corr_surface, y, x)

    # ---- UI construction ----

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left: image viewport with zoom controls
        left = QVBoxLayout()
        zoom_row = QHBoxLayout()
        self._btn_zoom_out = QPushButton('Zoom -')
        self._btn_zoom_in = QPushButton('Zoom +')
        self._btn_reset = QPushButton('Reset View')
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._btn_zoom_out.clicked.connect(self._zoom_out_center)
        self._btn_zoom_in.clicked.connect(self._zoom_in_center)
        self._btn_reset.clicked.connect(self._reset_view)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        # Prevent Enter from triggering zoom buttons; keep them out of focus chain
        for btn in (self._btn_zoom_out, self._btn_zoom_in, self._btn_reset):
            try:
                btn.setAutoDefault(False)
                btn.setDefault(False)
            except Exception:
                pass
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        zoom_row.addStretch()
        zoom_row.addWidget(self._btn_zoom_out)
        zoom_row.addWidget(self._btn_zoom_in)
        zoom_row.addWidget(self._zoom_sharp_check)
        zoom_row.addWidget(self._btn_reset)
        zoom_row.addStretch()
        left.addLayout(zoom_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setMinimumSize(700, 700)
        self._scroll.setStyleSheet('background-color: black;')
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = _ImageLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet('background-color: black;')
        self._label.setMouseTracking(True)
        self._scroll.setWidget(self._label)

        left.addWidget(self._scroll)
        layout.addLayout(left, stretch=2)

        # Status bar (within dialog)
        status = QStatusBar()
        self._status_label = QLabel('V, U: --, --  Value: --  Correlation: --')
        self._zoom_label = QLabel('Zoom: 1.00x')
        status.addWidget(self._status_label)
        status.addPermanentWidget(self._zoom_label)
        left.addWidget(status)

        # Right: controls
        right = QVBoxLayout()
        # Stretch group (common controls)
        stretch_group = QGroupBox('Image Stretch')
        stretch_form = QFormLayout()
        controls = build_stretch_controls(
            stretch_form,
            img_min=self._stretch_min,
            img_max=self._stretch_max,
            black_init=self._black,
            white_init=self._white,
            gamma_init=self._gamma,
            on_black_changed=self._on_black_changed,
            on_white_changed=self._on_white_changed,
            on_gamma_changed=self._on_gamma_changed,
        )
        # Keep attribute names for downstream code
        self._slider_black = controls['slider_black']
        self._slider_white = controls['slider_white']
        self._slider_gamma = controls['slider_gamma']
        self._lbl_black = controls['label_black']
        self._lbl_white = controls['label_white']
        self._lbl_gamma = controls['label_gamma']
        self._stretch_controls = controls
        # Reset stretch button
        self._btn_reset_stretch = QPushButton('Reset Stretch')
        self._btn_reset_stretch.clicked.connect(self._on_reset_stretch)
        stretch_form.addRow('', self._btn_reset_stretch)
        stretch_group.setLayout(stretch_form)
        right.addWidget(stretch_group)

        # Overlay group
        overlay_group = QGroupBox('Overlay')
        overlay_form = QFormLayout()
        # Transparency slider 0..100 -> alpha 0..1
        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 100)
        self._slider_alpha.setValue(round(self._alpha * 100))
        self._lbl_alpha = QLabel(f'{self._alpha:.2f}')
        self._slider_alpha.valueChanged.connect(lambda v: self._on_alpha_changed(v / 100.0))
        row_a = QHBoxLayout()
        row_a.addWidget(self._slider_alpha)
        row_a.addWidget(self._lbl_alpha)
        overlay_form.addRow('Model transparency:', row_a)
        overlay_group.setLayout(overlay_form)
        right.addWidget(overlay_group)

        # Offsets
        offset_group = QGroupBox('Offset (pixels)')
        offset_form = QFormLayout()
        # V and U with 0.001 precision
        self._spin_dv = QDoubleSpinBox()
        self._spin_du = QDoubleSpinBox()
        self._spin_dv.setDecimals(3)
        self._spin_du.setDecimals(3)
        # Bounds based on extfov margins
        self._spin_dv.setRange(-self._obs.extfov_margin_v, self._obs.extfov_margin_v)
        self._spin_du.setRange(-self._obs.extfov_margin_u, self._obs.extfov_margin_u)
        self._spin_dv.setSingleStep(0.1)
        self._spin_du.setSingleStep(0.1)
        self._spin_dv.setValue(self._dv)
        self._spin_du.setValue(self._du)
        self._spin_dv.valueChanged.connect(self._on_spin_dv)
        self._spin_du.valueChanged.connect(self._on_spin_du)
        offset_form.addRow('dV (rows):', self._spin_dv)
        offset_form.addRow('dU (cols):', self._spin_du)
        offset_group.setLayout(offset_form)
        right.addWidget(offset_group)

        # Buttons: Auto, OK/Cancel
        btn_row = QHBoxLayout()
        self._btn_auto = QPushButton('Auto')
        self._btn_ok = QPushButton('OK')
        self._btn_cancel = QPushButton('Cancel')
        self._btn_auto.clicked.connect(self._on_auto)
        self._btn_ok.clicked.connect(self.accept)
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_auto)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_cancel)
        right.addLayout(btn_row)
        right.addStretch(1)

        layout.addLayout(right, stretch=1)
        # Initialize zoom/pan controller for left-pan and wheel zoom
        self._zoom_ctl = ZoomPanController(
            label=self._label,
            scroll_area=self._scroll,
            get_zoom=lambda: self._zoom,
            set_zoom=lambda z: setattr(self, '_zoom', float(z)),
            update_display=self._update_display_only,
            set_zoom_label_text=lambda s: self._zoom_label.setText(s),
        )

    # ---- Event handlers ----

    def _on_black_changed(self, val: float) -> None:
        self._black = float(val)
        self._lbl_black.setText(f'{self._black:.5f}')
        self._refresh_overlay()

    def _on_white_changed(self, val: float) -> None:
        self._white = float(val)
        self._lbl_white.setText(f'{self._white:.5f}')
        self._refresh_overlay()

    def _on_gamma_changed(self, val: float) -> None:
        self._gamma = float(val)
        self._lbl_gamma.setText(f'{self._gamma:.5f}')
        self._refresh_overlay()

    def _on_alpha_changed(self, val: float) -> None:
        self._alpha = float(np.clip(val, 0.0, 1.0))
        self._lbl_alpha.setText(f'{self._alpha:.2f}')
        self._refresh_overlay()

    def _stretch_to_slider(self, val: float) -> int:
        denom = (
            self._stretch_max - self._stretch_min
            if (self._stretch_max > self._stretch_min)
            else 1.0
        )
        return round(1000.0 * (val - self._stretch_min) / denom)

    def _on_reset_stretch(self) -> None:
        # Recompute defaults from current image
        self._black = float(np.quantile(self._img_fov, 0.001))
        self._white = float(np.quantile(self._img_fov, 0.999))
        if self._black >= self._white:
            self._white = self._black + 0.01
        self._gamma = 1.0
        # Update UI via common helper
        self._stretch_controls['set_values'](self._black, self._white, self._gamma)
        # Redraw
        self._refresh_overlay()

    def _on_spin_dv(self, val: float) -> None:
        self._dv = float(val)
        self._refresh_overlay()

    def _on_spin_du(self, val: float) -> None:
        self._du = float(val)
        self._refresh_overlay()

    def _on_auto(self) -> None:
        # Call the same KPeaks correlation used by correlate_all
        up_factor = (
            getattr(self._config.offset, 'correlation_fft_upsample_factor', 128)
            if self._config
            else 128
        )
        res = navigate_with_pyramid_kpeaks(
            image=self._img_ext,
            model=self._model_img_ext,
            mask=self._model_mask_ext,
            upsample_factor=up_factor,
            logger=None,
        )
        dv, du = float(res['offset'][0]), float(res['offset'][1])
        # Clamp to extfov bounds
        dv = float(np.clip(dv, -self._obs.extfov_margin_v + 1, self._obs.extfov_margin_v - 1))
        du = float(np.clip(du, -self._obs.extfov_margin_u + 1, self._obs.extfov_margin_u - 1))
        self._dv, self._du = dv, du
        self._spin_dv.blockSignals(True)
        self._spin_du.blockSignals(True)
        self._spin_dv.setValue(self._dv)
        self._spin_du.setValue(self._du)
        self._spin_dv.blockSignals(False)
        self._spin_du.blockSignals(False)
        self._refresh_overlay()

    # Mouse handling
    def _on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Use common zoom/pan controller for left-button pan
            self._zoom_ctl.on_mouse_press(event)
            self._label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_mode = 'offset'
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_offset = (self._dv, self._du)
            self._label.setCursor(Qt.CursorShape.SizeAllCursor)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_mode == 'offset' and self._drag_start_pos is not None:
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._drag_start_pos
            if self._drag_start_offset is not None:
                # Convert label-pixel delta to image pixels via zoom
                du = self._drag_start_offset[1] + (delta.x() / max(self._zoom, 1e-6))
                dv = self._drag_start_offset[0] + (delta.y() / max(self._zoom, 1e-6))
                # Clamp within extfov bounds (minus 1 to keep slices valid after rounding)
                dv = float(
                    np.clip(
                        dv,
                        -self._obs.extfov_margin_v + 1,
                        self._obs.extfov_margin_v - 1,
                    )
                )
                du = float(
                    np.clip(
                        du,
                        -self._obs.extfov_margin_u + 1,
                        self._obs.extfov_margin_u - 1,
                    )
                )
                self._dv, self._du = dv, du
                # Update spin boxes without feedback loop
                self._spin_dv.blockSignals(True)
                self._spin_du.blockSignals(True)
                self._spin_dv.setValue(self._dv)
                self._spin_du.setValue(self._du)
                self._spin_dv.blockSignals(False)
                self._spin_du.blockSignals(False)
                self._refresh_overlay()
        else:
            # Delegate to common controller for hover/move and update status
            self._zoom_ctl.on_mouse_move(event)
            self._update_status_from_mouse(event)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_release(event)
            self._label.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start_pos = None
            self._drag_start_offset = None
            self._drag_mode = None
            self._label.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_wheel(self, event: QWheelEvent) -> None:
        # Ignore wheel-zoom if focus is on editable controls
        fw = self.focusWidget()
        if isinstance(fw, (QDoubleSpinBox, QSlider)):
            event.ignore()
            return
        # Delegate wheel zoom to common controller
        self._zoom_ctl.on_wheel(event)

    # ---- Zoom/pan helpers (parity with sim_body_gui) ----

    def _zoom_in_center(self) -> None:
        viewport = cast(QWidget, self._scroll.viewport())
        cx = viewport.width() // 2
        cy = viewport.height() // 2
        sh = cast(QScrollBar, self._scroll.horizontalScrollBar())
        sv = cast(QScrollBar, self._scroll.verticalScrollBar())
        scaled_x = cx + sh.value()
        scaled_y = cy + sv.value()
        self._zoom_at_point(1.2, cx, cy, scaled_x, scaled_y)

    def _zoom_out_center(self) -> None:
        viewport = cast(QWidget, self._scroll.viewport())
        cx = viewport.width() // 2
        cy = viewport.height() // 2
        sh = cast(QScrollBar, self._scroll.horizontalScrollBar())
        sv = cast(QScrollBar, self._scroll.verticalScrollBar())
        scaled_x = cx + sh.value()
        scaled_y = cy + sv.value()
        self._zoom_at_point(1.0 / 1.2, cx, cy, scaled_x, scaled_y)

    def _zoom_at_point(
        self, factor: float, vx: int, vy: int, scaled_x: float, scaled_y: float
    ) -> None:
        if self._pixmap_base is None:
            return
        old_zoom = self._zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.1, 50.0))
        if new_zoom == old_zoom:
            return
        # Use controller public API to maintain pan correctly
        self._zoom_ctl.zoom_at_point(factor, vx, vy, scaled_x, scaled_y)

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._zoom_label.setText(f'zoom: {self._zoom:.2f}x')
        self._update_display_only()

    # ---- Rendering ----

    def _compose_overlay_pixmap(self) -> None:
        """Compose the RGB overlay pixmap based on current stretch/offset/alpha."""
        # Primary image (FOV) -> red channel (mono repeated into RGB then tinted)
        img_u8 = _apply_stretch_gamma(self._img_fov, self._black, self._white, self._gamma)
        h, w = img_u8.shape
        # Extract model slice at current (dv, du)
        # Note: extract_offset_array will round inside; for display that's acceptable
        model_slice = self._obs.extract_offset_array(self._model_img_ext, (self._dv, self._du))
        model_u8 = (np.clip(model_slice, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Build RGB with red for image, green for model
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = img_u8
        # Alpha composite model into green channel
        # composite = (1-A)*image + A*model
        green = (1.0 - self._alpha) * img_u8.astype(np.float32) + self._alpha * model_u8.astype(
            np.float32
        )
        rgb[:, :, 1] = np.clip(green, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = img_u8

        # Create QImage/QPixmap
        qimage = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap(w, h)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)
        painter.end()
        self._pixmap_base = pixmap

    def _update_display_only(self) -> None:
        """Update label to show scaled/panned image."""
        if self._pixmap_base is None:
            return
        scaled_w = int(self._pixmap_base.width() * self._zoom)
        scaled_h = int(self._pixmap_base.height() * self._zoom)
        transform_mode = (
            Qt.TransformationMode.FastTransformation
            if self._zoom_sharp
            else Qt.TransformationMode.SmoothTransformation
        )
        scaled = self._pixmap_base.scaled(
            scaled_w, scaled_h, Qt.AspectRatioMode.KeepAspectRatio, transform_mode
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled_w, scaled_h)
        # Update status bar with latest corr (no mouse move)
        self._status_label.setText(
            f'V, U: --, --  Value: --  Correlation: {self._current_corr_value():.6f}'
        )

    def _refresh_overlay(self) -> None:
        """Rebuild overlay pixmap and update view."""
        self._compose_overlay_pixmap()
        self._update_display_only()

    def _update_status_from_mouse(self, event: QMouseEvent) -> None:
        # Position in label coordinates == scaled image coordinates
        scaled_x = float(event.position().x())
        scaled_y = float(event.position().y())
        # Convert to original image coords
        img_u = scaled_x / max(self._zoom, 1e-6)
        img_v = scaled_y / max(self._zoom, 1e-6)
        h, w = self._img_fov.shape
        if 0 <= img_v < h and 0 <= img_u < w:
            v0 = int(img_v)
            u0 = int(img_u)
            v1 = min(v0 + 1, h - 1)
            u1 = min(u0 + 1, w - 1)
            dv = img_v - v0
            du = img_u - u0
            val = (
                self._img_fov[v0, u0] * (1 - du) * (1 - dv)
                + self._img_fov[v0, u1] * du * (1 - dv)
                + self._img_fov[v1, u0] * (1 - du) * dv
                + self._img_fov[v1, u1] * du * dv
            )
            corr_val = self._current_corr_value()
            self._status_label.setText(
                f'V, U: {img_v:.2f}, {img_u:.2f}  Value: {val:.6f}  Correlation: {corr_val:.6f}'
            )
        else:
            self._status_label.setText(
                f'V, U: --, --  Value: --  Correlation: {self._current_corr_value():.6f}'
            )

    # ---- Dialog control ----

    def run_modal(self) -> tuple[bool, tuple[float, float] | None, float | None]:
        """Run the dialog modally, creating a QApplication if necessary."""
        app_created = False
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
            app_created = True
        result = self.exec()
        accepted = result == QDialog.DialogCode.Accepted
        chosen = (self._dv, self._du) if accepted else None
        corr = self._current_corr_value() if accepted else None
        if app_created:
            # Do not quit an existing app
            app.quit()
        return accepted, chosen, corr

    # ---- Zoom options ----
    def _toggle_zoom_sharp(self, state: Any) -> None:
        self._zoom_sharp = state == int(cast(int, Qt.CheckState.Checked.value))
        self._update_display_only()

    # Internal buffers
    _pixmap_base: QPixmap | None = None
