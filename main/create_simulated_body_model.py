#!/usr/bin/env python3
import json
import sys
from typing import Any, Optional, cast

import numpy as np
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QPushButton,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QCheckBox,
)

from nav.sim.render import render_combined_model


class ImageLabel(QLabel):
    def __init__(
        self,
        parent: Optional[QWidget],
        on_press: Any,
        on_move: Any,
        on_release: Any,
        on_wheel: Any,
    ) -> None:
        super().__init__(parent)
        self._on_press = on_press
        self._on_move = on_move
        self._on_release = on_release
        self._on_wheel = on_wheel

    def mousePressEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_press(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_move(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:
        if event is not None:
            self._on_release(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:
        if event is not None:
            self._on_wheel(event)


class ParameterUpdater(QObject):
    update_requested = pyqtSignal()

    def __init__(self, delay_ms: int = 120) -> None:
        super().__init__()
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.update_requested)
        self._delay_ms = delay_ms

    def request_update(self) -> None:
        self._timer.stop()
        self._timer.start(self._delay_ms)

    def immediate_update(self) -> None:
        self._timer.stop()
        self.update_requested.emit()


class CreateSimulatedBodyModel(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Create Simulated Body Model')
        self.setMinimumSize(1300, 850)

        # Data model mirrors JSON schema
        self.sim_params: dict[str, Any] = {
            'size_v': 512,
            'size_u': 512,
            'offset_v': 0.0,
            'offset_u': 0.0,
            'stars': [],
            'bodies': [],
        }

        # Render cache/meta
        self._current_image: Optional[np.ndarray] = None
        self._last_meta: dict[str, Any] = {}
        self._base_pixmap: Optional[QPixmap] = None

        # View state (copied math from existing GUI)
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start_pos: Optional[QPoint] = None
        self._drag_start_pan: Optional[tuple[float, float]] = None
        self._right_drag_active = False
        self._selected_model_key: Optional[tuple[str, int]] = None  # ('body' or 'star', index)
        self._last_drag_img_vu: Optional[tuple[float, float]] = None

        self._show_visual_aids = True
        self._zoom_sharp = True

        self._updater = ParameterUpdater(140)
        self._updater.update_requested.connect(self._update_image)

        self._setup_ui()
        self._update_image()

    # ---- UI setup ----
    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left image with zoom/pan (copied logic)
        left = QVBoxLayout()
        zoom_row = QHBoxLayout()
        zoom_row.addStretch()
        self._zoom_out_btn = QPushButton('Zoom -')
        self._zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_row.addWidget(self._zoom_out_btn)
        self._zoom_in_btn = QPushButton('Zoom +')
        self._zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_row.addWidget(self._zoom_in_btn)
        self._reset_view_btn = QPushButton('Reset View')
        self._reset_view_btn.clicked.connect(self._reset_view)
        zoom_row.addWidget(self._reset_view_btn)
        zoom_row.addStretch()
        left.addLayout(zoom_row)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setMinimumSize(700, 700)
        self._scroll_area.setStyleSheet('background-color: black;')
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = ImageLabel(
            self,
            self._on_mouse_press,
            self._on_mouse_move,
            self._on_mouse_release,
            self._on_wheel,
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet('background-color: black;')
        self._image_label.setMouseTracking(True)
        self._scroll_area.setWidget(self._image_label)

        left.addWidget(self._scroll_area)
        main_layout.addLayout(left, stretch=2)

        # Status bar
        status_bar = QStatusBar()
        self._status_label = QLabel('v, u: --, --  value: --')
        status_bar.addWidget(self._status_label)
        self._zoom_label = QLabel('zoom: 1.00x')
        status_bar.addPermanentWidget(self._zoom_label)
        self.setStatusBar(status_bar)

        # Right tabs panel
        right = QVBoxLayout()

        # Warning label for range duplicates
        self._warning_label = QLabel('')
        self._warning_label.setStyleSheet('color: orange;')
        right.addWidget(self._warning_label)

        self._tabs = QTabWidget()
        self._tabs.setMovable(True)
        right.addWidget(self._tabs, stretch=1)

        # General tab (always first)
        self._general_tab = QWidget()
        gen_layout = QFormLayout(self._general_tab)
        self._size_v_spin = QSpinBox()
        self._size_v_spin.setRange(64, 4096)
        self._size_v_spin.setValue(self.sim_params['size_v'])
        self._size_v_spin.valueChanged.connect(self._on_size_v)
        gen_layout.addRow('Size V (height):', self._size_v_spin)

        self._size_u_spin = QSpinBox()
        self._size_u_spin.setRange(64, 4096)
        self._size_u_spin.setValue(self.sim_params['size_u'])
        self._size_u_spin.valueChanged.connect(self._on_size_u)
        gen_layout.addRow('Size U (width):', self._size_u_spin)

        self._offset_v_spin = QDoubleSpinBox()
        self._offset_v_spin.setRange(-10000.0, 10000.0)
        self._offset_v_spin.setDecimals(3)
        self._offset_v_spin.setValue(self.sim_params['offset_v'])
        self._offset_v_spin.valueChanged.connect(self._on_offset_v)
        gen_layout.addRow('Offset V:', self._offset_v_spin)

        self._offset_u_spin = QDoubleSpinBox()
        self._offset_u_spin.setRange(-10000.0, 10000.0)
        self._offset_u_spin.setDecimals(3)
        self._offset_u_spin.setValue(self.sim_params['offset_u'])
        self._offset_u_spin.valueChanged.connect(self._on_offset_u)
        gen_layout.addRow('Offset U:', self._offset_u_spin)

        self._tabs.addTab(self._general_tab, 'general')

        # Buttons row
        btns = QHBoxLayout()
        self._add_tab_btn = QPushButton('Add Tab')
        self._add_tab_btn.clicked.connect(self._add_tab_dialog)
        btns.addWidget(self._add_tab_btn)
        self._del_tab_btn = QPushButton('Delete Tab')
        self._del_tab_btn.clicked.connect(self._delete_current_tab)
        btns.addWidget(self._del_tab_btn)
        btns.addStretch()

        self._save_img_btn = QPushButton('Save Image (PNG)')
        self._save_img_btn.clicked.connect(self._save_image)
        btns.addWidget(self._save_img_btn)

        self._save_json_btn = QPushButton('Save Parameters (JSON)')
        self._save_json_btn.clicked.connect(self._save_parameters)
        btns.addWidget(self._save_json_btn)

        self._load_json_btn = QPushButton('Load Parameters (JSON)')
        self._load_json_btn.clicked.connect(self._load_parameters)
        btns.addWidget(self._load_json_btn)

        right.addLayout(btns)

        # Visual options
        vis_row = QHBoxLayout()
        self._visual_aids_check = QCheckBox('Show Visual Aids')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._visual_aids_check.stateChanged.connect(self._toggle_visual_aids)
        vis_row.addWidget(self._visual_aids_check)
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        vis_row.addWidget(self._zoom_sharp_check)
        right.addLayout(vis_row)

        main_layout.addLayout(right, stretch=1)

    # ---- Event handlers: pan/zoom copied logic ----
    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self._current_image is not None:
            self._display_image()

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_start_pan = (self._pan_x, self._pan_y)
            self._image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            # Select model at cursor
            img_v, img_u = self._label_pos_to_image_vu(event.position().toPoint())
            self._select_model_at(img_v, img_u)
            self._right_drag_active = True
            self._last_drag_img_vu = (img_v, img_u)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_start_pos is not None and self._drag_start_pan is not None:
            current_pos = event.globalPosition().toPoint()
            delta = current_pos - self._drag_start_pos
            self._pan_x = self._drag_start_pan[0] - delta.x()
            self._pan_y = self._drag_start_pan[1] - delta.y()
            self._update_display()
        else:
            # status
            self._update_status_bar(event.position().toPoint())

        # Right-drag to move selected model
        if self._right_drag_active and self._selected_model_key is not None:
            img_v, img_u = self._label_pos_to_image_vu(event.position().toPoint())
            self._move_selected_by(img_v, img_u)

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = None
            self._drag_start_pan = None
            self._image_label.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_drag_active = False
            self._last_drag_img_vu = None

    def _on_wheel(self, event: QWheelEvent) -> None:
        label_pos = event.position().toPoint()
        viewport = self._scroll_area.viewport()
        if viewport is None:
            return
        viewport_pos = self._image_label.mapTo(viewport, label_pos)
        viewport_x = viewport_pos.x()
        viewport_y = viewport_pos.y()
        scrollbar_h = self._scroll_area.horizontalScrollBar()
        scrollbar_v = self._scroll_area.verticalScrollBar()
        if scrollbar_h is None or scrollbar_v is None:
            return
        scaled_image_x = viewport_x + scrollbar_h.value()
        scaled_image_y = viewport_y + scrollbar_v.value()

        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_at_point(1.2, viewport_x, viewport_y, scaled_image_x, scaled_image_y)
        else:
            self._zoom_at_point(1.0/1.2, viewport_x, viewport_y, scaled_image_x, scaled_image_y)

    def _zoom_in(self) -> None:
        if self._base_pixmap is not None:
            viewport = self._scroll_area.viewport()
            if viewport is None:
                return
            center_x = viewport.width() // 2
            center_y = viewport.height() // 2
            scrollbar_h = self._scroll_area.horizontalScrollBar()
            scrollbar_v = self._scroll_area.verticalScrollBar()
            if scrollbar_h is None or scrollbar_v is None:
                return
            scaled_x = center_x + scrollbar_h.value()
            scaled_y = center_y + scrollbar_v.value()
            self._zoom_at_point(1.2, center_x, center_y, scaled_x, scaled_y)

    def _zoom_out(self) -> None:
        if self._base_pixmap is not None:
            viewport = self._scroll_area.viewport()
            if viewport is None:
                return
            center_x = viewport.width() // 2
            center_y = viewport.height() // 2
            scrollbar_h = self._scroll_area.horizontalScrollBar()
            scrollbar_v = self._scroll_area.verticalScrollBar()
            if scrollbar_h is None or scrollbar_v is None:
                return
            scaled_x = center_x + scrollbar_h.value()
            scaled_y = center_y + scrollbar_v.value()
            self._zoom_at_point(1.0/1.2, center_x, center_y, scaled_x, scaled_y)

    def _zoom_at_point(
        self,
        factor: float,
        viewport_x: int,
        viewport_y: int,
        scaled_x: float,
        scaled_y: float,
    ) -> None:
        if self._base_pixmap is None:
            return
        old_zoom = self._zoom_factor
        new_zoom = max(0.1, min(50.0, old_zoom * factor))
        if new_zoom == old_zoom:
            return
        img_x = scaled_x / old_zoom
        img_y = scaled_y / old_zoom
        new_scroll_x = img_x * new_zoom - viewport_x
        new_scroll_y = img_y * new_zoom - viewport_y
        self._pan_x = new_scroll_x
        self._pan_y = new_scroll_y
        self._zoom_factor = new_zoom
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _label_pos_to_image_vu(self, label_pos: QPoint) -> tuple[float, float]:
        scaled_x = float(label_pos.x())
        scaled_y = float(label_pos.y())
        img_u = scaled_x / self._zoom_factor
        img_v = scaled_y / self._zoom_factor
        return img_v, img_u

    def _update_status_bar(self, label_pos: QPoint) -> None:
        self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')
        if self._current_image is None:
            self._status_label.setText('v, u: --, --  value: --')
            return
        img_v, img_u = self._label_pos_to_image_vu(label_pos)
        height, width = self._current_image.shape
        if 0 <= img_v < height and 0 <= img_u < width:
            v0 = int(img_v)
            u0 = int(img_u)
            v1 = min(v0 + 1, height - 1)
            u1 = min(u0 + 1, width - 1)
            dv = img_v - v0
            du = img_u - u0
            val00 = self._current_image[v0, u0]
            val01 = self._current_image[v0, u1]
            val10 = self._current_image[v1, u0]
            val11 = self._current_image[v1, u1]
            val = (val00 * (1 - du) * (1 - dv) +
                   val01 * du * (1 - dv) +
                   val10 * (1 - du) * dv +
                   val11 * du * dv)
            self._status_label.setText(f'v, u: {img_v:.2f}, {img_u:.2f}  value: {val:.6f}')
        else:
            self._status_label.setText('v, u: --, --  value: --')

    # ---- Sim param handlers ----
    def _on_size_v(self, value: int) -> None:
        self.sim_params['size_v'] = value
        self._updater.request_update()

    def _on_size_u(self, value: int) -> None:
        self.sim_params['size_u'] = value
        self._updater.request_update()

    def _on_offset_v(self, value: float) -> None:
        self.sim_params['offset_v'] = value
        self._updater.request_update()

    def _on_offset_u(self, value: float) -> None:
        self.sim_params['offset_u'] = value
        self._updater.request_update()

    # ---- Tab management ----
    def _add_tab_dialog(self) -> None:
        msg = QMessageBox(self)
        msg.setWindowTitle('Add Tab')
        msg.setText('Add a body or a star?')
        body_btn = msg.addButton('body', QMessageBox.ButtonRole.AcceptRole)
        star_btn = msg.addButton('star', QMessageBox.ButtonRole.AcceptRole)
        msg.addButton('cancel', QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == body_btn:
            self._add_body_tab()
        elif clicked == star_btn:
            self._add_star_tab()
        else:
            return

    def _add_body_tab(self, params: Optional[dict[str, Any]] = None) -> None:
        p = params or {
            'name': f'body{len(self.sim_params["bodies"])+1}',
            'center_v': self.sim_params['size_v'] / 2.0,
            'center_u': self.sim_params['size_u'] / 2.0,
            'semi_major_axis': 100.0,
            'semi_minor_axis': 80.0,
            'semi_c_axis': 80.0,
            'rotation_z': 0.0,
            'rotation_tilt': 0.0,
            'illumination_angle': 0.0,
            'phase_angle': 0.0,
            'crater_fill': 0.0,
            'crater_min_radius': 0.05,
            'crater_max_radius': 0.25,
            'crater_power_law_exponent': 3.0,
            'crater_relief_scale': 0.6,
            'anti_aliasing': 0.5,
            'range': len(self.sim_params['bodies']) + 1,
        }
        idx = len(self.sim_params['bodies'])
        self.sim_params['bodies'].append(p)
        tab = self._build_body_tab(idx)
        self._tabs.addTab(tab, p.get('name', f'body{idx+1}'))
        self._tabs.setCurrentWidget(tab)
        self._validate_ranges()
        self._updater.request_update()

    def _add_star_tab(self, params: Optional[dict[str, Any]] = None) -> None:
        p = params or {
            'name': f'star{len(self.sim_params["stars"])+1}',
            'v': self.sim_params['size_v'] / 2.0,
            'u': self.sim_params['size_u'] / 2.0,
            'vmag': 3.0,
            'spectral_class': 'G2',
            'psf_sigma': 3.0,
        }
        idx = len(self.sim_params['stars'])
        self.sim_params['stars'].append(p)
        tab = self._build_star_tab(idx)
        self._tabs.addTab(tab, p.get('name', f'star{idx+1}'))
        self._tabs.setCurrentWidget(tab)
        self._updater.request_update()

    def _delete_current_tab(self) -> None:
        idx = self._tabs.currentIndex()
        # Do not delete General (index 0)
        if idx <= 0:
            return
        widget = self._tabs.widget(idx)
        if widget is None:
            return
        # Determine if it's body or star by stored property
        kind = widget.property('kind')
        data_index = widget.property('data_index')
        if kind == 'body':
            if 0 <= data_index < len(self.sim_params['bodies']):
                del self.sim_params['bodies'][data_index]
        elif kind == 'star':
            if 0 <= data_index < len(self.sim_params['stars']):
                del self.sim_params['stars'][data_index]
        self._tabs.removeTab(idx)
        # Rebuild tabs indices to align with lists
        self._rebuild_dynamic_tabs()
        self._validate_ranges()
        self._updater.request_update()

    def _rebuild_dynamic_tabs(self) -> None:
        # Remove all non-general tabs and recreate
        while self._tabs.count() > 1:
            self._tabs.removeTab(1)
        for i, _ in enumerate(self.sim_params['bodies']):
            tab = self._build_body_tab(i)
            self._tabs.addTab(tab, self.sim_params['bodies'][i].get('name', f'body{i+1}'))
        for i, _ in enumerate(self.sim_params['stars']):
            tab = self._build_star_tab(i)
            self._tabs.addTab(tab, self.sim_params['stars'][i].get('name', f'star{i+1}'))

    # ---- Build body tab ----
    def _build_body_tab(self, idx: int) -> QWidget:
        p = self.sim_params['bodies'][idx]
        w = QWidget()
        w.setProperty('kind', 'body')
        w.setProperty('data_index', idx)
        fl = QFormLayout(w)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_body_name(i, t))
        fl.addRow('Name:', name_edit)

        center_v = QDoubleSpinBox()
        center_v.setRange(0.0, 10000.0)
        center_v.setDecimals(1)
        center_v.setValue(p.get('center_v', 0.0))
        center_v.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'center_v', v)
        )
        fl.addRow('Center V:', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(0.0, 10000.0)
        center_u.setDecimals(1)
        center_u.setValue(p.get('center_u', 0.0))
        center_u.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'center_u', v)
        )
        fl.addRow('Center U:', center_u)

        smaj = QDoubleSpinBox()
        smaj.setRange(1.0, 5000.0)
        smaj.setDecimals(1)
        smaj.setValue(p.get('semi_major_axis', 0.0))
        smaj.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'semi_major_axis', v)
        )
        fl.addRow('Semi-major axis:', smaj)
        smin = QDoubleSpinBox()
        smin.setRange(1.0, 5000.0)
        smin.setDecimals(1)
        smin.setValue(p.get('semi_minor_axis', 0.0))
        smin.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'semi_minor_axis', v)
        )
        fl.addRow('Semi-minor axis:', smin)
        sc = QDoubleSpinBox()
        sc.setRange(1.0, 5000.0)
        sc.setDecimals(1)
        sc.setValue(p.get('semi_c_axis', 0.0))
        sc.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'semi_c_axis', v)
        )
        fl.addRow('Semi-c axis (depth):', sc)

        rz = QDoubleSpinBox()
        rz.setRange(0.0, 360.0)
        rz.setDecimals(1)
        rz.setSuffix('°')
        rz.setValue(p.get('rotation_z', 0.0))
        rz.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'rotation_z', v)
        )
        fl.addRow('Rotation Z:', rz)
        rt = QDoubleSpinBox()
        rt.setRange(0.0, 90.0)
        rt.setDecimals(1)
        rt.setSuffix('°')
        rt.setValue(p.get('rotation_tilt', 0.0))
        rt.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'rotation_tilt', v)
        )
        fl.addRow('Rotation Tilt:', rt)

        illum = QDoubleSpinBox()
        illum.setRange(0.0, 360.0)
        illum.setDecimals(1)
        illum.setSuffix('°')
        illum.setValue(p.get('illumination_angle', 0.0))
        illum.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'illumination_angle', v)
        )
        fl.addRow('Illumination angle:', illum)
        phase = QDoubleSpinBox()
        phase.setRange(0.0, 180.0)
        phase.setDecimals(1)
        phase.setSuffix('°')
        phase.setValue(p.get('phase_angle', 0.0))
        phase.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'phase_angle', v)
        )
        fl.addRow('Phase angle:', phase)

        cf = QDoubleSpinBox()
        cf.setRange(0.0, 10.0)
        cf.setDecimals(3)
        cf.setSingleStep(0.01)
        cf.setValue(p.get('crater_fill', 0.0))
        cf.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_fill', v)
        )
        fl.addRow('Crater fill (0-10):', cf)
        cmin = QDoubleSpinBox()
        cmin.setRange(0.01, 0.25)
        cmin.setDecimals(3)
        cmin.setSingleStep(0.005)
        cmin.setValue(p.get('crater_min_radius', 0.05))
        cmin.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_min_radius', v)
        )
        fl.addRow('Crater min radius:', cmin)
        cmax = QDoubleSpinBox()
        cmax.setRange(0.01, 0.25)
        cmax.setDecimals(3)
        cmax.setSingleStep(0.005)
        cmax.setValue(p.get('crater_max_radius', 0.25))
        cmax.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_max_radius', v)
        )
        fl.addRow('Crater max radius:', cmax)
        cexp = QDoubleSpinBox()
        cexp.setRange(1.1, 5.0)
        cexp.setDecimals(2)
        cexp.setSingleStep(0.05)
        cexp.setValue(p.get('crater_power_law_exponent', 3.0))
        cexp.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_power_law_exponent', v)
        )
        fl.addRow('Crater power-law exponent:', cexp)
        crs = QDoubleSpinBox()
        crs.setRange(0.0, 3.0)
        crs.setDecimals(3)
        crs.setSingleStep(0.01)
        crs.setValue(p.get('crater_relief_scale', 0.6))
        crs.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'crater_relief_scale', v)
        )
        fl.addRow('Crater relief scale:', crs)

        aa = QDoubleSpinBox()
        aa.setRange(0.0, 1.0)
        aa.setDecimals(3)
        aa.setSingleStep(0.01)
        aa.setValue(p.get('anti_aliasing', 0.5))
        aa.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'anti_aliasing', v)
        )
        fl.addRow('Anti-aliasing:', aa)

        rng = QDoubleSpinBox()
        rng.setRange(-1e9, 1e9)
        rng.setDecimals(3)
        rng.setValue(p.get('range', idx+1))
        rng.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'range', v, trigger_validate=True)
        )
        fl.addRow('Range:', rng)

        return w

    # ---- Build star tab ----
    def _build_star_tab(self, idx: int) -> QWidget:
        p = self.sim_params['stars'][idx]
        w = QWidget()
        w.setProperty('kind', 'star')
        w.setProperty('data_index', idx)
        fl = QFormLayout(w)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_star_name(i, t))
        fl.addRow('Name:', name_edit)

        v_spin = QDoubleSpinBox()
        v_spin.setRange(0.0, 10000.0)
        v_spin.setDecimals(1)
        v_spin.setValue(p.get('v', 0.0))
        v_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'v', v)
        )
        fl.addRow('V:', v_spin)
        u_spin = QDoubleSpinBox()
        u_spin.setRange(0.0, 10000.0)
        u_spin.setDecimals(1)
        u_spin.setValue(p.get('u', 0.0))
        u_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'u', v)
        )
        fl.addRow('U:', u_spin)

        vmag = QDoubleSpinBox()
        vmag.setRange(-10.0, 30.0)
        vmag.setDecimals(2)
        vmag.setValue(p.get('vmag', 3.0))
        vmag.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'vmag', v)
        )
        fl.addRow('Magnitude (V):', vmag)
        sclass = QLineEdit(p.get('spectral_class', 'G2'))
        sclass.textChanged.connect(lambda t, i=idx: self._on_star_field(i, 'spectral_class', t))
        fl.addRow('Spectral class:', sclass)
        psf = QDoubleSpinBox()
        psf.setRange(0.1, 20.0)
        psf.setDecimals(2)
        psf.setValue(p.get('psf_sigma', 3.0))
        psf.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'psf_sigma', v)
        )
        fl.addRow('PSF sigma:', psf)

        return w

    # ---- Field handlers ----
    def _on_body_field(self,
                       idx: int,
                       key: str,
                       value: Any,
                       *,
                       trigger_validate: bool = False) -> None:
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()
            if trigger_validate and key == 'range':
                self._validate_ranges()

    def _on_star_field(self, idx: int, key: str, value: Any) -> None:
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_body_name(self, idx: int, text: str) -> None:
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['name'] = text
            # update tab title
            self._update_tab_titles()
            self._updater.request_update()

    def _on_star_name(self, idx: int, text: str) -> None:
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _update_tab_titles(self) -> None:
        # General tab remains 'general'
        for i, p in enumerate(self.sim_params['bodies']):
            tab_idx = 1 + i
            if tab_idx < self._tabs.count():
                self._tabs.setTabText(tab_idx, p.get('name', f'body{i+1}'))
        base = 1 + len(self.sim_params['bodies'])
        for j, p in enumerate(self.sim_params['stars']):
            tab_idx = base + j
            if tab_idx < self._tabs.count():
                self._tabs.setTabText(tab_idx, p.get('name', f'star{j+1}'))

    def _validate_ranges(self) -> None:
        ranges = [
            str(self.sim_params['bodies'][i].get('range'))
            for i in range(len(self.sim_params['bodies']))
        ]
        duplicates = len(ranges) != len(set(ranges))
        self._warning_label.setText(
            'Warning: duplicate body ranges' if duplicates else ''
        )

    # ---- Rendering ----
    def _update_image(self) -> None:
        try:
            img, meta = render_combined_model(self.sim_params)
            self._current_image = img
            self._last_meta = meta
            self._display_image()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to render image:\n{str(e)}')

    def _display_image(self) -> None:
        if self._current_image is None:
            return
        img_uint8 = (np.clip(self._current_image, 0.0, 1.0) * 255).astype(np.uint8)
        height, width = img_uint8.shape
        img_uint8 = np.ascontiguousarray(img_uint8.copy())
        qimage = QImage(
            img_uint8.tobytes(),
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)
        if self._show_visual_aids:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            # Draw centers for bodies
            for b in self.sim_params.get('bodies', []):
                center_x = int(b.get('center_u', 0))
                center_y = int(b.get('center_v', 0))
                painter.drawEllipse(center_x - 4, center_y - 4, 8, 8)
            # Draw stars as small crosses
            pen = QPen(QColor(255, 255, 0), 1)
            painter.setPen(pen)
            for s in self.sim_params.get('stars', []):
                u = int(s.get('u', 0))
                v = int(s.get('v', 0))
                painter.drawLine(u - 4, v, u + 4, v)
                painter.drawLine(u, v - 4, u, v + 4)
        painter.end()
        self._base_pixmap = pixmap
        self._update_display()
        self._image_label.repaint()
        viewport = self._scroll_area.viewport()
        if viewport is not None:
            viewport.repaint()

    def _update_display(self) -> None:
        if self._base_pixmap is None:
            return
        scaled_width = int(self._base_pixmap.width() * self._zoom_factor)
        scaled_height = int(self._base_pixmap.height() * self._zoom_factor)
        transform_mode = (Qt.TransformationMode.FastTransformation
                          if self._zoom_sharp else Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap = self._base_pixmap.scaled(
            scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, transform_mode
        )
        self._image_label.setPixmap(scaled_pixmap)
        self._image_label.resize(scaled_width, scaled_height)
        scrollbar_h = self._scroll_area.horizontalScrollBar()
        scrollbar_v = self._scroll_area.verticalScrollBar()
        viewport = self._scroll_area.viewport()
        if scrollbar_h is None or scrollbar_v is None or viewport is None:
            return
        scrollbar_h.setRange(0, max(0, scaled_width - viewport.width()))
        scrollbar_v.setRange(0, max(0, scaled_height - viewport.height()))
        scroll_pos_h = int(max(0, min(scrollbar_h.maximum(), self._pan_x)))
        scroll_pos_v = int(max(0, min(scrollbar_v.maximum(), self._pan_y)))
        scrollbar_h.setValue(scroll_pos_h)
        scrollbar_v.setValue(scroll_pos_v)

    # ---- Selection / drag-move ----
    def _select_model_at(self, img_v: float, img_u: float) -> None:
        # Check bodies first: order_near_to_far; pick first whose mask contains pixel
        body_masks = self._last_meta.get('body_masks', [])
        order_near_to_far = self._last_meta.get('order_near_to_far', [])
        bodies = self.sim_params.get('bodies', [])
        if body_masks and bodies:
            # Map names to index in original list by matching 'name' upper
            name_to_index = {b.get('name', '').upper(): i for i, b in enumerate(bodies)}
            height = int(self.sim_params['size_v'])
            width = int(self.sim_params['size_u'])
            v_i = int(round(img_v))
            u_i = int(round(img_u))
            if 0 <= v_i < height and 0 <= u_i < width:
                # We don't have masks in near-to-far order; rebuild mapping: body_masks
                # is in far-to-near composition order per render_bodies loop. For hit
                # test, reconstruct mapping by sorting inventory ranges.
                inv = self._last_meta.get('inventory', {})
                name_to_mask: dict[str, Any] = {}
                sorted_items = sorted(
                    inv.items(),
                    key=lambda kv: kv[1]['range'],
                    reverse=True,
                )
                body_names_far_to_near = [k for k, _ in sorted_items]
                for nm, m in zip(body_names_far_to_near, body_masks):
                    name_to_mask[nm] = m
                for nm in order_near_to_far:
                    m = name_to_mask.get(nm)
                    if m is not None and bool(m[v_i, u_i]):
                        idx = name_to_index.get(nm, None)
                        if idx is not None:
                            self._selected_model_key = ('body', idx)
                            self._tabs.setCurrentIndex(1 + idx)
                            return
        # Stars: evaluate PSF contribution approx via Gaussian envelope
        star_info = self._last_meta.get('star_info', [])
        if star_info:
            # Stars are behind bodies; if no body hit, check stars
            for j, info in enumerate(star_info):
                cv = info['center_v']
                cu = info['center_u']
                sigma = info['sigma']
                dv = img_v - cv
                du = img_u - cu
                r2 = dv*dv + du*du
                # Gaussian threshold ~ 3 sigma circle
                if r2 <= (3.0 * sigma) ** 2:
                    self._selected_model_key = ('star', j)
                    # Switch to star tab: index = 1 + num_bodies + j
                    self._tabs.setCurrentIndex(1 + len(self.sim_params['bodies']) + j)
                    return
        self._selected_model_key = None

    def _move_selected_by(self, img_v: float, img_u: float) -> None:
        if self._last_drag_img_vu is None or self._selected_model_key is None:
            self._last_drag_img_vu = (img_v, img_u)
            return
        prev_v, prev_u = self._last_drag_img_vu
        dv = img_v - prev_v
        du = img_u - prev_u
        kind, idx = self._selected_model_key
        if kind == 'body' and 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['center_v'] = float(
                self.sim_params['bodies'][idx].get('center_v', 0.0) + dv
            )
            self.sim_params['bodies'][idx]['center_u'] = float(
                self.sim_params['bodies'][idx].get('center_u', 0.0) + du
            )
            self._updater.immediate_update()
        elif kind == 'star' and 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx]['v'] = float(
                self.sim_params['stars'][idx].get('v', 0.0) + dv
            )
            self.sim_params['stars'][idx]['u'] = float(
                self.sim_params['stars'][idx].get('u', 0.0) + du
            )
            self._updater.immediate_update()
        self._last_drag_img_vu = (img_v, img_u)

    # ---- Save/Load ----
    def _save_image(self) -> None:
        if self._current_image is None:
            QMessageBox.warning(self, 'No Image', 'No image to save.')
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Image',
            'simulated_model.png',
            'PNG Images (*.png)',
        )
        if filename:
            try:
                from PIL import Image
                img_uint8 = (
                    np.clip(self._current_image, 0.0, 1.0) * 255
                ).astype(np.uint8)
                Image.fromarray(img_uint8, mode='L').save(filename)
                QMessageBox.information(self, 'Saved', f'Image saved to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image:\n{str(e)}')

    def _save_parameters(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Parameters',
            'simulated_body_params.json',
            'JSON Files (*.json)',
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.sim_params, f, indent=2)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save parameters:\n{str(e)}')

    def _load_parameters(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Parameters',
            '',
            'JSON Files (*.json)',
        )
        if filename:
            try:
                with open(filename) as f:
                    params = json.load(f)
                # Shallow validation
                self.sim_params = {
                    'size_v': int(params.get('size_v', 512)),
                    'size_u': int(params.get('size_u', 512)),
                    'offset_v': float(params.get('offset_v', 0.0)),
                    'offset_u': float(params.get('offset_u', 0.0)),
                    'bodies': list(params.get('bodies', [])),
                    'stars': list(params.get('stars', [])),
                }
                # Update general UI
                self._size_v_spin.setValue(self.sim_params['size_v'])
                self._size_u_spin.setValue(self.sim_params['size_u'])
                self._offset_v_spin.setValue(self.sim_params['offset_v'])
                self._offset_u_spin.setValue(self.sim_params['offset_u'])
                # Rebuild tabs
                self._rebuild_dynamic_tabs()
                self._update_tab_titles()
                self._validate_ranges()
                self._updater.immediate_update()
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load parameters:\n{str(e)}')

    # ---- Visual toggles ----
    def _toggle_visual_aids(self, state: Any) -> None:
        if isinstance(state, Qt.CheckState):
            self._show_visual_aids = (state is Qt.CheckState.Checked)
        elif isinstance(state, int):
            self._show_visual_aids = (state == cast(int, Qt.CheckState.Checked.value))
        else:
            self._show_visual_aids = False
        if self._current_image is not None:
            self._base_pixmap = None
            self._display_image()

    def _toggle_zoom_sharp(self, state: Any) -> None:
        if isinstance(state, Qt.CheckState):
            self._zoom_sharp = (state is Qt.CheckState.Checked)
        elif isinstance(state, int):
            self._zoom_sharp = (state == cast(int, Qt.CheckState.Checked.value))
        else:
            self._zoom_sharp = False
        self._update_display()


def main() -> None:
    app = QApplication(sys.argv)
    window = CreateSimulatedBodyModel()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
