#!/usr/bin/env python3
from collections.abc import Callable
import json
import os
import re
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
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from nav.sim.render import render_combined_model
from nav.ui.common import ZoomPanController


class ImageLabel(QLabel):
    def __init__(
        self,
        parent: Optional[QWidget],
        on_press: Callable[[QMouseEvent], None],
        on_move: Callable[[QMouseEvent], None],
        on_release: Callable[[QMouseEvent], None],
        on_wheel: Callable[[QWheelEvent], None],
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

    def __init__(self, delay_ms: int) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setInterval(delay_ms)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._emit_update)

    def request_update(self) -> None:
        self._timer.start()

    def immediate_update(self) -> None:
        self._timer.stop()
        self.update_requested.emit()

    def _emit_update(self) -> None:
        self.update_requested.emit()


class CreateSimulatedImageModel(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Create Simulated Model')
        self.setMinimumSize(1300, 850)

        # Data model mirrors JSON schema
        self.sim_params: dict[str, Any] = {
            'size_v': 512,
            'size_u': 512,
            'offset_v': 0.0,
            'offset_u': 0.0,
            'random_seed': 42,
            'closest_planet': 'SATURN',
            'time': 0.0,
            'ring_epoch': 0.0,
            'background_noise_intensity': 0.0,
            'background_stars_num': 0,
            'background_stars_psf_sigma': 0.9,
            'background_stars_distribution_exponent': 2.5,
            'shade_solid_rings': False,
            'stars': [],
            'bodies': [],
            'rings': [],
        }

        # Render cache/meta
        self._current_image: Optional[np.ndarray] = None
        self._last_meta: dict[str, Any] = {}
        self._base_pixmap: Optional[QPixmap] = None

        # View state
        self._zoom_factor = 1.0
        self._right_drag_active = False
        # ('body'/'star'/'ring', index)
        self._selected_model_key: Optional[tuple[str, int]] = None
        self._last_drag_img_vu: Optional[tuple[float, float]] = None
        # Track last valid (non-"+") tab for cancel behavior
        self._last_valid_tab_index = 0  # Start with General tab

        self._show_visual_aids = True
        self._zoom_sharp = True

        self._updater = ParameterUpdater(140)
        self._updater.update_requested.connect(self._update_image)

        self._setup_ui()
        self._update_image()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left image with zoom/pan
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
        self._scroll_area.setStyleSheet('background-color: #303000;')
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = ImageLabel(
            self,
            self._on_press,
            self._on_move,
            self._on_release,
            self._on_wheel,
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet('background-color: #800000;')
        self._image_label.setMouseTracking(True)
        self._scroll_area.setWidget(self._image_label)

        left.addWidget(self._scroll_area)
        main_layout.addLayout(left, stretch=2)

        # Status bar
        status_bar = QStatusBar()
        self._status_label = QLabel('V, U: --------, --------  Value: --')
        status_bar.addWidget(self._status_label)
        self._zoom_label = QLabel('Zoom: 1.00x')
        status_bar.addPermanentWidget(self._zoom_label)
        self.setStatusBar(status_bar)

        # Right tabs panel
        right = QVBoxLayout()

        # Warning label for range duplicates
        self._warning_label = QLabel('')
        self._warning_label.setStyleSheet('color: orange;')
        right.addWidget(self._warning_label)

        self._tabs = QTabWidget()
        self._tabs.setMovable(False)  # Prevent manual reordering
        # Connect tab bar click to detect clicks on "+" tab
        self._tabs.tabBarClicked.connect(self._on_tab_bar_clicked)
        # Track current tab changes to remember last valid tab
        self._tabs.currentChanged.connect(self._on_tab_changed)
        right.addWidget(self._tabs, stretch=1)

        # General tab (always first)
        self._general_tab = QWidget()
        gen_layout = QFormLayout(self._general_tab)
        self._size_v_spin = QSpinBox()
        self._size_v_spin.setRange(10, 9999)
        self._size_v_spin.setValue(self.sim_params['size_v'])
        self._size_v_spin.valueChanged.connect(self._on_size_v)
        gen_layout.addRow('Size V (height):', self._size_v_spin)

        self._size_u_spin = QSpinBox()
        self._size_u_spin.setRange(10, 9999)
        self._size_u_spin.setValue(self.sim_params['size_u'])
        self._size_u_spin.valueChanged.connect(self._on_size_u)
        gen_layout.addRow('Size U (width):', self._size_u_spin)

        self._offset_v_spin = QDoubleSpinBox()
        self._offset_v_spin.setRange(-10000.0, 10000.0)
        self._offset_v_spin.setDecimals(3)
        self._offset_v_spin.setValue(self.sim_params['offset_v'])
        self._offset_v_spin.setToolTip(
            'Offsets are saved in the model but not shown in the preview.')
        self._offset_v_spin.valueChanged.connect(self._on_offset_v)
        gen_layout.addRow('Offset V:', self._offset_v_spin)

        self._offset_u_spin = QDoubleSpinBox()
        self._offset_u_spin.setRange(-10000.0, 10000.0)
        self._offset_u_spin.setDecimals(3)
        self._offset_u_spin.setValue(self.sim_params['offset_u'])
        self._offset_u_spin.setToolTip(
            'Offsets are saved in the model but not shown in the preview.')
        self._offset_u_spin.valueChanged.connect(self._on_offset_u)
        gen_layout.addRow('Offset U:', self._offset_u_spin)

        # Random seed
        self._random_seed_spin = QSpinBox()
        self._random_seed_spin.setRange(0, 2147483647)
        self._random_seed_spin.setValue(self.sim_params['random_seed'])
        self._random_seed_spin.valueChanged.connect(self._on_random_seed)
        gen_layout.addRow('Random seed:', self._random_seed_spin)

        # Closest planet (for ring models)
        self._closest_planet_combo = QComboBox()
        self._closest_planet_combo.setEditable(True)
        self._closest_planet_combo.addItems(['MERCURY', 'VENUS', 'EARTH', 'MARS', 'JUPITER',
                                             'SATURN', 'URANUS', 'NEPTUNE', 'PLUTO'])
        closest_planet = self.sim_params.get('closest_planet', 'SATURN')
        if closest_planet:
            index = self._closest_planet_combo.findText(closest_planet)
            if index >= 0:
                self._closest_planet_combo.setCurrentIndex(index)
            else:
                self._closest_planet_combo.setCurrentText(closest_planet)
        self._closest_planet_combo.currentTextChanged.connect(self._on_closest_planet)
        gen_layout.addRow('Closest planet:', self._closest_planet_combo)

        # Time (TDB seconds)
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(-1e10, 1e10)
        self._time_spin.setDecimals(1)
        self._time_spin.setValue(self.sim_params.get('time', 0.0))
        self._time_spin.setToolTip('Current time in TDB seconds for ring calculations')
        self._time_spin.valueChanged.connect(self._on_time)
        gen_layout.addRow('Time (TDB sec):', self._time_spin)

        # Ring epoch (TDB seconds)
        self._epoch_spin = QDoubleSpinBox()
        self._epoch_spin.setRange(-1e10, 1e10)
        self._epoch_spin.setDecimals(1)
        self._epoch_spin.setValue(self.sim_params.get('ring_epoch', 0.0))
        self._epoch_spin.setToolTip('Ring epoch time in TDB seconds for ring mode calculations')
        self._epoch_spin.valueChanged.connect(self._on_epoch)
        gen_layout.addRow('Ring Epoch (TDB sec):', self._epoch_spin)

        # Background noise slider with min/max labels and spinbox
        noise_row = QHBoxLayout()
        noise_row.setSpacing(4)
        noise_row.setContentsMargins(0, 0, 0, 0)
        noise_min_label = QLabel('0.0')
        noise_min_label.setFixedWidth(35)
        noise_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        noise_max_label = QLabel('1.0')
        noise_max_label.setFixedWidth(35)
        noise_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self._background_noise_slider.setRange(0, 1000)
        noise_slider_init_val = int(
            self.sim_params['background_noise_intensity'] * 1000)
        self._background_noise_slider.setValue(noise_slider_init_val)
        self._background_noise_slider.valueChanged.connect(self._on_background_noise_slider)
        self._background_noise_spin = QDoubleSpinBox()
        self._background_noise_spin.setRange(0.0, 1.0)
        self._background_noise_spin.setDecimals(3)
        self._background_noise_spin.setSingleStep(0.001)
        self._background_noise_spin.setValue(self.sim_params['background_noise_intensity'])
        self._background_noise_spin.valueChanged.connect(self._on_background_noise_spin)
        noise_row.addWidget(noise_min_label)
        noise_row.addWidget(self._background_noise_slider, stretch=1)
        noise_row.addWidget(noise_max_label)
        noise_row.addWidget(self._background_noise_spin)
        noise_holder = QWidget()
        noise_holder.setLayout(noise_row)
        gen_layout.addRow('Background noise intensity:', noise_holder)

        # Background stars slider with min/max labels and spinbox
        stars_row = QHBoxLayout()
        stars_row.setSpacing(4)
        stars_row.setContentsMargins(0, 0, 0, 0)
        stars_min_label = QLabel('0')
        stars_min_label.setFixedWidth(35)
        stars_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        stars_max_label = QLabel('1000')
        stars_max_label.setFixedWidth(40)
        stars_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_stars_slider = QSlider(Qt.Orientation.Horizontal)
        self._background_stars_slider.setRange(0, 1000)
        self._background_stars_slider.setValue(self.sim_params['background_stars_num'])
        self._background_stars_slider.valueChanged.connect(self._on_background_stars_slider)
        self._background_stars_spin = QSpinBox()
        self._background_stars_spin.setRange(0, 1000)
        self._background_stars_spin.setValue(self.sim_params['background_stars_num'])
        self._background_stars_spin.valueChanged.connect(self._on_background_stars_spin)
        stars_row.addWidget(stars_min_label)
        stars_row.addWidget(self._background_stars_slider, stretch=1)
        stars_row.addWidget(stars_max_label)
        stars_row.addWidget(self._background_stars_spin)
        stars_holder = QWidget()
        stars_holder.setLayout(stars_row)
        gen_layout.addRow('Background stars num:', stars_holder)

        # Background stars PSF sigma slider with min/max labels and spinbox
        psf_sigma_row = QHBoxLayout()
        psf_sigma_row.setSpacing(4)
        psf_sigma_row.setContentsMargins(0, 0, 0, 0)
        psf_sigma_min_label = QLabel('0.1')
        psf_sigma_min_label.setFixedWidth(35)
        psf_sigma_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        psf_sigma_max_label = QLabel('3.0')
        psf_sigma_max_label.setFixedWidth(40)
        psf_sigma_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_stars_psf_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        # 0.1 to 3.0 with 0.01 steps
        self._background_stars_psf_sigma_slider.setRange(1, 300)
        psf_sigma_slider_val = int(
            self.sim_params['background_stars_psf_sigma'] * 100)
        self._background_stars_psf_sigma_slider.setValue(psf_sigma_slider_val)
        self._background_stars_psf_sigma_slider.valueChanged.connect(
            self._on_background_stars_psf_sigma_slider)
        self._background_stars_psf_sigma_spin = QDoubleSpinBox()
        self._background_stars_psf_sigma_spin.setRange(0.1, 3.0)
        self._background_stars_psf_sigma_spin.setDecimals(2)
        self._background_stars_psf_sigma_spin.setSingleStep(0.1)
        psf_sigma_spin_val = self.sim_params['background_stars_psf_sigma']
        self._background_stars_psf_sigma_spin.setValue(psf_sigma_spin_val)
        self._background_stars_psf_sigma_spin.valueChanged.connect(
            self._on_background_stars_psf_sigma_spin)
        psf_sigma_row.addWidget(psf_sigma_min_label)
        psf_sigma_row.addWidget(self._background_stars_psf_sigma_slider, stretch=1)
        psf_sigma_row.addWidget(psf_sigma_max_label)
        psf_sigma_row.addWidget(self._background_stars_psf_sigma_spin)
        psf_sigma_holder = QWidget()
        psf_sigma_holder.setLayout(psf_sigma_row)
        gen_layout.addRow('Background stars PSF sigma:', psf_sigma_holder)

        # Background stars distribution exponent slider with min/max labels and spinbox
        dist_exp_row = QHBoxLayout()
        dist_exp_row.setSpacing(4)
        dist_exp_row.setContentsMargins(0, 0, 0, 0)
        dist_exp_min_label = QLabel('1.0')
        dist_exp_min_label.setFixedWidth(35)
        dist_exp_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dist_exp_max_label = QLabel('4.0')
        dist_exp_max_label.setFixedWidth(40)
        dist_exp_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_stars_dist_exp_slider = QSlider(Qt.Orientation.Horizontal)
        # 1.0 to 4.0 with 0.01 steps
        self._background_stars_dist_exp_slider.setRange(100, 400)
        dist_exp_slider_val = int(
            self.sim_params['background_stars_distribution_exponent'] * 100)
        self._background_stars_dist_exp_slider.setValue(dist_exp_slider_val)
        self._background_stars_dist_exp_slider.valueChanged.connect(
            self._on_background_stars_dist_exp_slider)
        self._background_stars_dist_exp_spin = QDoubleSpinBox()
        self._background_stars_dist_exp_spin.setRange(1.0, 4.0)
        self._background_stars_dist_exp_spin.setDecimals(2)
        self._background_stars_dist_exp_spin.setSingleStep(0.1)
        dist_exp_spin_val = self.sim_params['background_stars_distribution_exponent']
        self._background_stars_dist_exp_spin.setValue(dist_exp_spin_val)
        self._background_stars_dist_exp_spin.valueChanged.connect(
            self._on_background_stars_dist_exp_spin)
        dist_exp_row.addWidget(dist_exp_min_label)
        dist_exp_row.addWidget(self._background_stars_dist_exp_slider, stretch=1)
        dist_exp_row.addWidget(dist_exp_max_label)
        dist_exp_row.addWidget(self._background_stars_dist_exp_spin)
        dist_exp_holder = QWidget()
        dist_exp_holder.setLayout(dist_exp_row)
        gen_layout.addRow('Background stars distribution exponent:', dist_exp_holder)

        # Add General tab first
        self._tabs.addTab(self._general_tab, 'General')

        # Add "+" tab for adding new objects (fake tab - just header, no content, always last)
        self._add_tab_widget = QWidget()
        self._tabs.addTab(self._add_tab_widget, '+')

        # Ensure correct tab order
        self._ensure_tab_order()

        # Buttons row (no Add/Delete buttons - handled by tabs)
        btns = QHBoxLayout()
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

        # Visual options with Exit button on same line
        vis_row = QHBoxLayout()
        self._visual_aids_check = QCheckBox('Show Visual Aids')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._visual_aids_check.stateChanged.connect(self._toggle_visual_aids)
        vis_row.addWidget(self._visual_aids_check)
        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._zoom_sharp_check.stateChanged.connect(self._toggle_zoom_sharp)
        vis_row.addWidget(self._zoom_sharp_check)
        self._shade_solid_rings_check = QCheckBox('Shade solid rings')
        self._shade_solid_rings_check.setChecked(self.sim_params.get('shade_solid_rings', False))
        self._shade_solid_rings_check.stateChanged.connect(self._toggle_shade_solid_rings)
        vis_row.addWidget(self._shade_solid_rings_check)
        vis_row.addStretch()
        exit_btn = QPushButton('Exit')
        exit_btn.clicked.connect(self.close)
        vis_row.addWidget(exit_btn)
        right.addLayout(vis_row)

        main_layout.addLayout(right, stretch=1)
        # Initialize common zoom/pan controller for left-button pan and wheel zoom
        self._zoom_ctl = ZoomPanController(
            label=self._image_label,
            scroll_area=self._scroll_area,
            get_zoom=lambda: self._zoom_factor,
            set_zoom=lambda z: setattr(self, '_zoom_factor', float(z)),
            update_display=self._update_display,
            set_zoom_label_text=lambda s: self._zoom_label.setText(s),
        )

    # ---- Event handlers: pan/zoom ----
    def _on_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_press(event)
            self._image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            # Select model at cursor
            img_v, img_u = self._label_pos_to_image_vu(event.position().toPoint())
            self._select_model_at(img_v, img_u)
            self._right_drag_active = True
            self._last_drag_img_vu = (img_v, img_u)

    def _on_move(self, event: QMouseEvent) -> None:
        self._zoom_ctl.on_mouse_move(event)
        # status
        self._update_status_bar(event.position().toPoint())

        # Right-drag to move selected model
        if self._right_drag_active and self._selected_model_key is not None:
            img_v, img_u = self._label_pos_to_image_vu(event.position().toPoint())
            self._move_selected_by(img_v, img_u)

    def _on_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_ctl.on_mouse_release(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self._right_drag_active = False
            self._last_drag_img_vu = None

    def _on_wheel(self, event: QWheelEvent) -> None:
        self._zoom_ctl.on_wheel(event)

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

    def _zoom_at_point(self,
                       factor: float,
                       viewport_x: int,
                       viewport_y: int,
                       scaled_x: float,
                       scaled_y: float,
                       ) -> None:
        if self._base_pixmap is None:
            return
        old_zoom = self._zoom_factor
        new_zoom = float(np.clip(old_zoom * factor, 0.1, 50.0))
        if new_zoom == old_zoom:
            return
        # Delegate to the controller to perform zoom and maintain pan
        self._zoom_ctl.zoom_at_point(factor, viewport_x, viewport_y, scaled_x, scaled_y)

    def _reset_view(self) -> None:
        self._zoom_factor = 1.0
        self._zoom_label.setText(f'Zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _label_pos_to_image_vu(self, label_pos: QPoint) -> tuple[float, float]:
        scaled_x = float(label_pos.x())
        scaled_y = float(label_pos.y())
        img_u = scaled_x / self._zoom_factor
        img_v = scaled_y / self._zoom_factor
        return img_v, img_u

    def _update_status_bar(self, label_pos: QPoint) -> None:
        self._zoom_label.setText(f'Zoom: {self._zoom_factor:.2f}x')
        if self._current_image is None:
            self._status_label.setText('V, U: --------, --------  Value: --')
            return
        img_v, img_u = self._label_pos_to_image_vu(label_pos)
        height, width = self._current_image.shape
        if 0 <= img_v < height and 0 <= img_u < width:
            v0 = int(img_v)
            u0 = int(img_u)
            val = self._current_image[v0, u0]
            self._status_label.setText(f'V, U: {img_v:8.2f}, {img_u:8.2f}  Value: {val:9.6f}')
        else:
            self._status_label.setText('V, U: --------, --------  Value: --')

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

    def _on_random_seed(self, value: int) -> None:
        self.sim_params['random_seed'] = value
        self._updater.request_update()

    def _on_closest_planet(self, text: str) -> None:
        # Store as None if empty, otherwise store the text (uppercase)
        if text.strip():
            self.sim_params['closest_planet'] = text.strip().upper()
        else:
            self.sim_params['closest_planet'] = None
        self._updater.request_update()

    def _on_time(self, value: float) -> None:
        self.sim_params['time'] = value
        self._updater.request_update()

    def _on_epoch(self, value: float) -> None:
        self.sim_params['ring_epoch'] = value
        self._updater.request_update()

    def _on_background_noise_slider(self, value: int) -> None:
        noise_val = value / 1000.0
        self._background_noise_spin.blockSignals(True)
        self._background_noise_spin.setValue(noise_val)
        self._background_noise_spin.blockSignals(False)
        self.sim_params['background_noise_intensity'] = noise_val
        self._updater.request_update()

    def _on_background_noise_spin(self, value: float) -> None:
        slider_val = int(value * 1000)
        self._background_noise_slider.blockSignals(True)
        self._background_noise_slider.setValue(slider_val)
        self._background_noise_slider.blockSignals(False)
        self.sim_params['background_noise_intensity'] = value
        self._updater.request_update()

    def _on_background_stars_slider(self, value: int) -> None:
        self._background_stars_spin.blockSignals(True)
        self._background_stars_spin.setValue(value)
        self._background_stars_spin.blockSignals(False)
        self.sim_params['background_stars_num'] = value
        self._updater.request_update()

    def _on_background_stars_spin(self, value: int) -> None:
        self._background_stars_slider.blockSignals(True)
        self._background_stars_slider.setValue(value)
        self._background_stars_slider.blockSignals(False)
        self.sim_params['background_stars_num'] = value
        self._updater.request_update()

    def _on_background_stars_psf_sigma_slider(self, value: int) -> None:
        psf_sigma_val = value / 100.0
        self._background_stars_psf_sigma_spin.blockSignals(True)
        self._background_stars_psf_sigma_spin.setValue(psf_sigma_val)
        self._background_stars_psf_sigma_spin.blockSignals(False)
        self.sim_params['background_stars_psf_sigma'] = psf_sigma_val
        self._updater.request_update()

    def _on_background_stars_psf_sigma_spin(self, value: float) -> None:
        slider_val = int(value * 100)
        self._background_stars_psf_sigma_slider.blockSignals(True)
        self._background_stars_psf_sigma_slider.setValue(slider_val)
        self._background_stars_psf_sigma_slider.blockSignals(False)
        self.sim_params['background_stars_psf_sigma'] = value
        self._updater.request_update()

    def _on_background_stars_dist_exp_slider(self, value: int) -> None:
        dist_exp_val = value / 100.0
        self._background_stars_dist_exp_spin.blockSignals(True)
        self._background_stars_dist_exp_spin.setValue(dist_exp_val)
        self._background_stars_dist_exp_spin.blockSignals(False)
        self.sim_params['background_stars_distribution_exponent'] = dist_exp_val
        self._updater.request_update()

    def _on_background_stars_dist_exp_spin(self, value: float) -> None:
        slider_val = int(value * 100)
        self._background_stars_dist_exp_slider.blockSignals(True)
        self._background_stars_dist_exp_slider.setValue(slider_val)
        self._background_stars_dist_exp_slider.blockSignals(False)
        self.sim_params['background_stars_distribution_exponent'] = value
        self._updater.request_update()

    # ---- Tab management ----
    def _ensure_tab_order(self) -> None:
        """Ensure General is first and '+' is last."""
        # Block signals to prevent tab change events during reordering
        self._tabs.blockSignals(True)

        general_idx = -1
        plus_idx = -1
        for i in range(self._tabs.count()):
            text = self._tabs.tabText(i)
            if text == 'General':
                general_idx = i
            elif text == '+':
                plus_idx = i

        # Remember current tab before reordering
        current_idx = self._tabs.currentIndex()
        current_widget = None
        if current_idx >= 0 and current_idx < self._tabs.count():
            current_widget = self._tabs.widget(current_idx)

        # Move General to first if needed
        if general_idx >= 0 and general_idx != 0:
            general_widget = self._tabs.widget(general_idx)
            self._tabs.removeTab(general_idx)
            self._tabs.insertTab(0, general_widget, 'General')
            # Recalculate plus_idx after removal
            for i in range(self._tabs.count()):
                if self._tabs.tabText(i) == '+':
                    plus_idx = i
                    break

        # Move "+" to last if needed
        if plus_idx >= 0 and plus_idx != self._tabs.count() - 1:
            plus_widget = self._tabs.widget(plus_idx)
            self._tabs.removeTab(plus_idx)
            self._tabs.addTab(plus_widget, '+')

        # Restore current tab if it still exists
        if current_widget is not None:
            for i in range(self._tabs.count()):
                if self._tabs.widget(i) == current_widget:
                    self._tabs.setCurrentIndex(i)
                    break

        # Unblock signals
        self._tabs.blockSignals(False)

    def _on_tab_changed(self, index: int) -> None:
        """Track tab changes and intercept switches to the '+' tab."""
        # Ignore invalid indices (can happen during tab rebuilding)
        if index < 0 or index >= self._tabs.count():
            return

        # If signals are blocked, we're in the middle of a programmatic change - don't intercept
        if self._tabs.signalsBlocked():
            # Still track valid tabs for future reference
            tab_text = self._tabs.tabText(index)
            if tab_text != '+':
                self._last_valid_tab_index = index
            return

        tab_text = self._tabs.tabText(index)

        # If switching to the "+" tab, intercept it
        if tab_text == '+':
            # Get the last valid tab index
            prev_tab = self._last_valid_tab_index
            # Ensure it's valid
            if (prev_tab < 0 or prev_tab >= self._tabs.count() or
                    self._tabs.tabText(prev_tab) == '+'):
                # Fallback: find the last non-"+", non-General tab, or use General
                prev_tab = 0  # Default to General
                # Start from second-to-last, skip General
                for i in range(self._tabs.count() - 2, 0, -1):
                    if self._tabs.tabText(i) != '+':
                        prev_tab = i
                        break

            # Block signals to prevent recursion
            self._tabs.blockSignals(True)
            # Switch back to the previous tab immediately (before showing dialog)
            self._tabs.setCurrentIndex(prev_tab)
            self._tabs.blockSignals(False)

            # Now show the dialog
            result = self._add_tab_dialog()
            # If canceled, we've already switched back, so we're done
            # If successful, the new tab will be created and automatically selected
            if not result:
                # Make sure we're still on the previous tab (should already be, but be explicit)
                if (prev_tab >= 0 and prev_tab < self._tabs.count() and
                        self._tabs.tabText(prev_tab) != '+'):
                    self._tabs.blockSignals(True)
                    self._tabs.setCurrentIndex(prev_tab)
                    self._tabs.blockSignals(False)
        else:
            # This is a valid tab, remember it
            self._last_valid_tab_index = index

    def _on_tab_bar_clicked(self, index: int) -> None:
        # This is just for tracking - the actual interception happens in _on_tab_changed
        pass

    def _add_tab_dialog(self) -> bool:
        """Show dialog to add object. Returns True if object was added, False if canceled."""
        msg = QMessageBox(self)
        msg.setWindowTitle('Add object')
        msg.setText('Add what type of model?')
        body_btn = msg.addButton('Body', QMessageBox.ButtonRole.AcceptRole)
        ring_btn = msg.addButton('Ring', QMessageBox.ButtonRole.AcceptRole)
        star_btn = msg.addButton('Star', QMessageBox.ButtonRole.AcceptRole)
        msg.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == body_btn:
            self._add_body_tab()
            return True
        elif clicked == star_btn:
            self._add_star_tab()
            return True
        elif clicked == ring_btn:
            self._add_ring_tab()
            return True
        else:
            return False

    def _find_unique_name(self, base_name: str) -> str:
        """Find a unique name by incrementing the number suffix if needed.

        Checks bodies, stars, and rings to ensure the name is unique.
        """
        # Collect all existing names (case-insensitive)
        existing_names = set()
        for body in self.sim_params.get('bodies', []):
            existing_names.add(body.get('name', '').lower())
        for star in self.sim_params.get('stars', []):
            existing_names.add(star.get('name', '').lower())
        for ring in self.sim_params.get('rings', []):
            existing_names.add(ring.get('name', '').lower())

        # Try the base name first
        if base_name.lower() not in existing_names:
            return base_name

        # Extract base prefix and number if present
        match = re.match(r'^(.+?)(\d+)$', base_name)
        if match:
            prefix = match.group(1)
            start_num = int(match.group(2))
        else:
            # No number suffix, add one
            prefix = base_name
            start_num = 1

        # Increment until we find a unique name
        num = start_num + 1
        while True:
            candidate = f'{prefix}{num}'
            if candidate.lower() not in existing_names:
                return candidate
            num += 1

    def _add_body_tab(self, params: Optional[dict[str, Any]] = None) -> None:
        if params is None:
            default_name = f'Body{len(self.sim_params["bodies"])+1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                'center_v': self.sim_params['size_v'] // 2 + 0.5,
                'center_u': self.sim_params['size_u'] // 2 + 0.5,
                'range': self._find_unique_range(),
                'axis1': 100.0,
                'axis2': 80.0,
                'axis3': 80.0,
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
            }
        else:
            p = params
        idx = len(self.sim_params['bodies'])
        self.sim_params['bodies'].append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._validate_ranges()
        self._updater.request_update()

    def _add_ring_tab(self, params: Optional[dict[str, Any]] = None) -> None:
        if params is None:
            default_name = f'Ring{len(self.sim_params["rings"])+1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                'feature_type': 'RINGLET',
                'center_v': self.sim_params['size_v'] // 2 + 0.5,
                'center_u': self.sim_params['size_u'] // 2 + 0.5,
                'range': self._find_unique_range(),
                'shading_distance': 20.0,
                'inner_data': [{
                    'mode': 1,
                    'a': 100.0,
                    'rms': 1.0,
                    'ae': 0.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }],
                'outer_data': [{
                    'mode': 1,
                    'a': 120.0,
                    'rms': 1.0,
                    'ae': 0.0,
                    'long_peri': 0.0,
                    'rate_peri': 0.0,
                }],
            }
        else:
            p = params
        idx = len(self.sim_params['rings'])
        self.sim_params['rings'].append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('ring', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._updater.request_update()

    def _add_star_tab(self, params: Optional[dict[str, Any]] = None) -> None:
        if params is None:
            default_name = f'Star{len(self.sim_params["stars"])+1}'
            unique_name = self._find_unique_name(default_name)
            p = {
                'name': unique_name,
                'v': self.sim_params['size_v'] // 2 + 0.5,
                'u': self.sim_params['size_u'] // 2 + 0.5,
                'vmag': 3.0,
                'spectral_class': 'G2',
                'psf_sigma': 1.0,
                'psf_size': (11, 11),
            }
        else:
            p = params
        idx = len(self.sim_params['stars'])
        self.sim_params['stars'].append(p)
        # Rebuild tabs to ensure consistency and proper ordering
        self._rebuild_dynamic_tabs()
        # Find and select the newly added tab
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            self._tabs.setCurrentIndex(tab_idx)
        self._updater.request_update()

    def _find_tab_by_properties(self, kind: str, data_index: int) -> Optional[int]:
        """Find tab index by kind and data_index properties.

        Returns the tab index if found, None otherwise.
        """
        for tab_idx in range(self._tabs.count()):
            widget = self._tabs.widget(tab_idx)
            if widget is None:
                continue
            widget_kind = widget.property('kind')
            widget_data_index = widget.property('data_index')
            if (widget_kind == kind and
                    widget_data_index is not None and
                    widget_data_index == data_index):
                return tab_idx
        return None

    def _delete_current_tab(self) -> None:
        """Delete the currently selected tab by looking up its data_index."""
        tab_idx = self._tabs.currentIndex()
        widget = self._tabs.widget(tab_idx)
        if widget is None:
            # No widget at this tab index, nothing to delete
            return
        data_index = widget.property('data_index')
        widget_kind = widget.property('kind')
        if data_index is None or widget_kind is None:
            # Widget doesn't have required properties (e.g., General or "+" tab)
            return
        self._delete_tab_by_index(widget_kind, data_index)

    def _delete_tab_by_index(self, kind: str, data_index: int) -> None:
        """Delete a tab by its kind ('body' or 'star') and data_index."""
        # Use the helper function to find the correct tab
        tab_idx = self._find_tab_by_properties(kind, data_index)
        if tab_idx is None:
            # Tab not found, nothing to delete
            return

        # Verify the widget matches what we expect
        widget = self._tabs.widget(tab_idx)
        if widget is None:
            return
        widget_kind = widget.property('kind')
        widget_data_index = widget.property('data_index')
        if widget_kind != kind or widget_data_index != data_index:
            # Safety check: widget doesn't match what we're looking for
            return

        # Delete from the correct list
        if kind == 'body':
            if 0 <= data_index < len(self.sim_params['bodies']):
                del self.sim_params['bodies'][data_index]
        elif kind == 'ring':
            if 0 <= data_index < len(self.sim_params['rings']):
                del self.sim_params['rings'][data_index]
        elif kind == 'star':
            if 0 <= data_index < len(self.sim_params['stars']):
                del self.sim_params['stars'][data_index]
        else:
            raise AssertionError(f'Unknown kind: {kind}')

        # Block signals before removing tab to prevent unwanted tab change events
        self._tabs.blockSignals(True)
        self._tabs.removeTab(tab_idx)
        self._tabs.blockSignals(False)

        # Rebuild tabs indices to align with lists
        self._rebuild_dynamic_tabs()
        self._ensure_tab_order()  # Ensure order is correct
        self._validate_ranges()
        self._updater.request_update()

    def _rebuild_dynamic_tabs(self) -> None:
        # Save General and "+" tab widgets
        general_widget = None
        plus_widget = None
        for i in range(self._tabs.count()):
            text = self._tabs.tabText(i)
            if text == 'General':
                general_widget = self._tabs.widget(i)
            elif text == '+':
                plus_widget = self._tabs.widget(i)

        # Remember current tab before rebuilding (if it's a valid tab)
        current_idx = self._tabs.currentIndex()
        target_tab_name = None
        if current_idx >= 0 and current_idx < self._tabs.count():
            current_text = self._tabs.tabText(current_idx)
            if current_text not in ('General', '+'):
                # Try to identify which body/star this was
                widget = self._tabs.widget(current_idx)
                if widget is not None:
                    widget_kind = widget.property('kind')
                    widget_data_index = widget.property('data_index')
                    if widget_kind == 'body' and widget_data_index is not None:
                        if 0 <= widget_data_index < len(self.sim_params['bodies']):
                            body_name = self.sim_params['bodies'][widget_data_index].get(
                                'name', f'Body{widget_data_index+1}')
                            target_tab_name = body_name
                    elif widget_kind == 'ring' and widget_data_index is not None:
                        if 0 <= widget_data_index < len(self.sim_params['rings']):
                            ring_name = self.sim_params['rings'][widget_data_index].get(
                                'name', f'Ring{widget_data_index+1}')
                            target_tab_name = ring_name
                    elif widget_kind == 'star' and widget_data_index is not None:
                        if 0 <= widget_data_index < len(self.sim_params['stars']):
                            star_name = self.sim_params['stars'][widget_data_index].get(
                                'name', f'Star{widget_data_index+1}')
                            target_tab_name = star_name
                    else:
                        raise AssertionError(f'Unknown kind: {widget_kind}')

        # Block signals during rebuild to prevent tab change handler from firing
        self._tabs.blockSignals(True)

        # Remove all tabs
        while self._tabs.count() > 0:
            self._tabs.removeTab(0)

        # Re-add in correct order: General first, then bodies (sorted by range),
        # then rings (sorted by name), then stars (sorted by name), then "+"
        if general_widget is not None:
            self._tabs.addTab(general_widget, 'General')

        # Add body tabs (sorted by range)
        body_indices = list(range(len(self.sim_params['bodies'])))
        body_indices.sort(
            key=lambda i: (
                self.sim_params['bodies'][i].get('range', float('inf')),
                self.sim_params['bodies'][i].get('name', f'Body{i+1}').lower()
            )
        )
        for i in body_indices:
            tab = self._build_body_tab(i)
            tab_name = self.sim_params['bodies'][i].get('name', f'Body{i+1}')
            self._tabs.addTab(tab, tab_name)

        # Add ring tabs (sorted by name)
        ring_indices = list(range(len(self.sim_params['rings'])))
        ring_indices.sort(
            key=lambda i: self.sim_params['rings'][i].get('name', f'Ring{i+1}').lower()
        )
        for i in ring_indices:
            tab = self._build_ring_tab(i)
            tab_name = self.sim_params['rings'][i].get('name', f'Ring{i+1}')
            self._tabs.addTab(tab, tab_name)

        # Add star tabs (sorted by name)
        star_indices = list(range(len(self.sim_params['stars'])))
        star_indices.sort(
            key=lambda i: self.sim_params['stars'][i].get('name', f'Star{i+1}').lower()
        )
        for i in star_indices:
            tab = self._build_star_tab(i)
            tab_name = self.sim_params['stars'][i].get('name', f'Star{i+1}')
            self._tabs.addTab(tab, tab_name)

        # Add "+" tab last
        if plus_widget is not None:
            self._tabs.addTab(plus_widget, '+')
        else:
            # Create if it doesn't exist
            self._add_tab_widget = QWidget()
            self._tabs.addTab(self._add_tab_widget, '+')

        # Restore the previously selected tab if it still exists
        if target_tab_name is not None:
            found = False
            for i in range(self._tabs.count()):
                if self._tabs.tabText(i) == target_tab_name:
                    self._tabs.setCurrentIndex(i)
                    self._last_valid_tab_index = i
                    found = True
                    break
            if not found:
                # Tab was deleted, default to General
                self._tabs.setCurrentIndex(0)
                self._last_valid_tab_index = 0
        else:
            # Default to General tab (index 0)
            self._tabs.setCurrentIndex(0)
            self._last_valid_tab_index = 0

        # Ensure we're on a valid tab (not "+") before unblocking signals
        current_idx = self._tabs.currentIndex()
        if current_idx >= 0 and current_idx < self._tabs.count():
            if self._tabs.tabText(current_idx) == '+':
                # Shouldn't happen, but be safe
                self._tabs.setCurrentIndex(0)
                self._last_valid_tab_index = 0

        # Unblock signals - this might emit currentChanged, but we're on General so it's safe
        self._tabs.blockSignals(False)

    # ---- Build body tab ----
    def _build_body_tab(self, idx: int) -> QWidget:
        p = self.sim_params['bodies'][idx]
        w = QWidget()
        w.setProperty('kind', 'body')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_body_name(i, t))
        fl.addRow('Name:', name_edit)

        center_v = QDoubleSpinBox()
        center_v.setRange(-10000.0, 20000.0)
        center_v.setDecimals(1)
        center_v.setValue(p.get('center_v', 0.0))
        center_v.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'center_v', v)
        )
        fl.addRow('Center V:', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-10000.0, 20000.0)
        center_u.setDecimals(1)
        center_u.setValue(p.get('center_u', 0.0))
        center_u.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'center_u', v)
        )
        fl.addRow('Center U:', center_u)
        # Keep references so drag updates can sync the UI
        w.center_v_spin = center_v  # type: ignore
        w.center_u_spin = center_u  # type: ignore

        # Range field (for layering/ordering)
        rng = QDoubleSpinBox()
        rng.setRange(-1e9, 1e9)
        rng.setDecimals(3)
        rng.setValue(p.get('range', idx+1))
        rng.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'range', v, trigger_validate=True)
        )
        fl.addRow('Range:', rng)

        smaj = QDoubleSpinBox()
        smaj.setRange(1.0, 5000.0)
        smaj.setDecimals(1)
        smaj.setValue(p.get('axis1', 0.0))
        smaj.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'axis1', v)
        )
        fl.addRow('Axis 1:', smaj)
        smin = QDoubleSpinBox()
        smin.setRange(1.0, 5000.0)
        smin.setDecimals(1)
        smin.setValue(p.get('axis2', 0.0))
        smin.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'axis2', v)
        )
        fl.addRow('Axis 2:', smin)
        sc = QDoubleSpinBox()
        sc.setRange(1.0, 5000.0)
        sc.setDecimals(1)
        sc.setValue(p.get('axis3', 0.0))
        sc.valueChanged.connect(
            lambda v, i=idx: self._on_body_field(i, 'axis3', v)
        )
        fl.addRow('Axis 3:', sc)

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

        # Crater fill slider with min/max labels and spinbox
        cf_row = QHBoxLayout()
        cf_row.setSpacing(4)
        cf_row.setContentsMargins(0, 0, 0, 0)
        cf_min_label = QLabel('0.0')
        cf_min_label.setFixedWidth(35)
        cf_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cf_max_label = QLabel('10.0')
        cf_max_label.setFixedWidth(40)
        cf_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        cf_slider = QSlider(Qt.Orientation.Horizontal)
        cf_slider.setRange(0, 10000)  # 0.0 to 10.0 with 0.001 steps
        cf_slider.setValue(int(p.get('crater_fill', 0.0) * 1000))
        cf_slider.valueChanged.connect(
            lambda v, i=idx: self._on_body_crater_fill_slider(i, v)
        )
        cf_spin = QDoubleSpinBox()
        cf_spin.setRange(0.0, 10.0)
        cf_spin.setDecimals(3)
        cf_spin.setSingleStep(0.01)
        cf_spin.setValue(p.get('crater_fill', 0.0))
        cf_spin.valueChanged.connect(
            lambda v, i=idx: self._on_body_crater_fill_spin(i, v)
        )
        cf_row.addWidget(cf_min_label)
        cf_row.addWidget(cf_slider, stretch=1)
        cf_row.addWidget(cf_max_label)
        cf_row.addWidget(cf_spin)
        cf_holder = QWidget()
        cf_holder.setLayout(cf_row)
        fl.addRow('Crater fill (0-10):', cf_holder)
        # Store references for sync
        w.crater_fill_slider = cf_slider  # type: ignore
        w.crater_fill_spin = cf_spin  # type: ignore
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

        # Anti-aliasing slider with min/max labels and spinbox
        aa_row = QHBoxLayout()
        aa_row.setSpacing(4)
        aa_row.setContentsMargins(0, 0, 0, 0)
        aa_min_label = QLabel('0.0')
        aa_min_label.setFixedWidth(35)
        aa_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        aa_max_label = QLabel('1.0')
        aa_max_label.setFixedWidth(35)
        aa_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        aa_slider = QSlider(Qt.Orientation.Horizontal)
        aa_slider.setRange(0, 1000)  # 0.0 to 1.0 with 0.001 steps
        aa_slider.setValue(int(p.get('anti_aliasing', 0.5) * 1000))
        aa_slider.valueChanged.connect(
            lambda v, i=idx: self._on_body_anti_aliasing_slider(i, v)
        )
        aa_spin = QDoubleSpinBox()
        aa_spin.setRange(0.0, 1.0)
        aa_spin.setDecimals(3)
        aa_spin.setSingleStep(0.01)
        aa_spin.setValue(p.get('anti_aliasing', 0.5))
        aa_spin.valueChanged.connect(
            lambda v, i=idx: self._on_body_anti_aliasing_spin(i, v)
        )
        aa_row.addWidget(aa_min_label)
        aa_row.addWidget(aa_slider, stretch=1)
        aa_row.addWidget(aa_max_label)
        aa_row.addWidget(aa_spin)
        aa_holder = QWidget()
        aa_holder.setLayout(aa_row)
        fl.addRow('Anti-aliasing:', aa_holder)
        # Store references for sync
        w.anti_aliasing_slider = aa_slider  # type: ignore
        w.anti_aliasing_spin = aa_spin  # type: ignore

        # Delete button at bottom
        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('body', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

        return w

    # ---- Build ring tab ----
    def _build_ring_tab(self, idx: int) -> QWidget:
        p = self.sim_params['rings'][idx]
        w = QWidget()
        w.setProperty('kind', 'ring')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_ring_name(i, t))
        fl.addRow('Name:', name_edit)

        # Feature type (RINGLET or GAP)
        feature_type_combo = QComboBox()
        feature_type_combo.addItems(['RINGLET', 'GAP'])
        feature_type = p.get('feature_type', 'RINGLET')
        feature_type_combo.setCurrentText(feature_type)
        feature_type_combo.currentTextChanged.connect(
            lambda t, i=idx: self._on_ring_field(i, 'feature_type', t)
        )
        fl.addRow('Feature type:', feature_type_combo)

        center_v = QDoubleSpinBox()
        center_v.setRange(-10000.0, 20000.0)
        center_v.setDecimals(1)
        center_v.setValue(p.get('center_v', 0.0))
        center_v.valueChanged.connect(
            lambda v, i=idx: self._on_ring_field(i, 'center_v', v)
        )
        fl.addRow('Center V:', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-10000.0, 20000.0)
        center_u.setDecimals(1)
        center_u.setValue(p.get('center_u', 0.0))
        center_u.valueChanged.connect(
            lambda v, i=idx: self._on_ring_field(i, 'center_u', v)
        )
        fl.addRow('Center U:', center_u)
        # Keep references so drag updates can sync the UI
        w.center_v_spin = center_v  # type: ignore
        w.center_u_spin = center_u  # type: ignore

        # Range field (for layering/ordering)
        rng = QDoubleSpinBox()
        rng.setRange(-1e9, 1e9)
        rng.setDecimals(3)
        # Default range: start after bodies (assuming bodies use 1, 2, 3, ...)
        default_range = len(self.sim_params['rings']) + len(self.sim_params['bodies']) + 1
        rng.setValue(p.get('range', default_range))
        rng.valueChanged.connect(
            lambda v, i=idx: self._on_ring_field(i, 'range', v)
        )
        fl.addRow('Range:', rng)

        # Shading distance parameter
        shading_distance = QDoubleSpinBox()
        shading_distance.setRange(0.0, 1000.0)
        shading_distance.setDecimals(1)
        shading_distance.setSuffix(' px')
        shading_distance.setValue(p.get('shading_distance', 20.0))
        shading_distance.valueChanged.connect(
            lambda v, i=idx: self._on_ring_field(i, 'shading_distance', v)
        )
        fl.addRow('Shading distance:', shading_distance)

        # Inner edge checkbox and mode 1 parameters
        inner_data = p.get('inner_data', [])
        has_inner = len(inner_data) > 0 and any(m.get('mode') == 1 for m in inner_data)
        inner_mode1: dict[str, Any] = next((m for m in inner_data if m.get('mode') == 1), {})

        inner_checkbox = QCheckBox('Enable Inner Edge')
        inner_checkbox.setChecked(has_inner)
        inner_checkbox.clicked.connect(
            lambda checked, i=idx, cb=inner_checkbox: self._on_ring_inner_enabled(
                i, checked, cb)
        )
        fl.addRow(inner_checkbox, QLabel(''))

        inner_label = QLabel('<b>Inner Edge (Mode 1)</b>')
        fl.addRow(inner_label, QLabel(''))

        inner_a = QDoubleSpinBox()
        inner_a.setRange(1.0, 10000.0)
        inner_a.setDecimals(1)
        inner_a.setValue(inner_mode1.get('a', 100.0))
        inner_a.setEnabled(has_inner)
        inner_a.valueChanged.connect(
            lambda v, i=idx: self._on_ring_inner_mode1(i, 'a', v)
        )
        fl.addRow('Inner a:', inner_a)
        inner_ae = QDoubleSpinBox()
        inner_ae.setRange(0.0, 1000.0)
        inner_ae.setDecimals(2)
        inner_ae.setValue(inner_mode1.get('ae', 0.0))
        inner_ae.setEnabled(has_inner)
        inner_ae.valueChanged.connect(
            lambda v, i=idx: self._on_ring_inner_mode1(i, 'ae', v)
        )
        fl.addRow('Inner ae:', inner_ae)
        inner_long_peri = QDoubleSpinBox()
        inner_long_peri.setRange(0.0, 360.0)
        inner_long_peri.setDecimals(1)
        inner_long_peri.setSuffix('°')
        inner_long_peri.setValue(inner_mode1.get('long_peri', 0.0))
        inner_long_peri.setEnabled(has_inner)
        inner_long_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_inner_mode1(i, 'long_peri', v)
        )
        fl.addRow('Inner long_peri:', inner_long_peri)
        inner_rate_peri = QDoubleSpinBox()
        inner_rate_peri.setRange(-1000.0, 1000.0)
        inner_rate_peri.setDecimals(3)
        inner_rate_peri.setSuffix('°/day')
        inner_rate_peri.setValue(inner_mode1.get('rate_peri', 0.0))
        inner_rate_peri.setEnabled(has_inner)
        inner_rate_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_inner_mode1(i, 'rate_peri', v)
        )
        fl.addRow('Inner rate_peri:', inner_rate_peri)
        inner_rms = QDoubleSpinBox()
        inner_rms.setRange(0.0, 1000.0)
        inner_rms.setDecimals(3)
        inner_rms.setValue(inner_mode1.get('rms', 1.0))
        inner_rms.setEnabled(has_inner)
        inner_rms.valueChanged.connect(
            lambda v, i=idx: self._on_ring_inner_mode1(i, 'rms', v)
        )
        fl.addRow('Inner rms:', inner_rms)

        # Store references for enabling/disabling
        w.inner_checkbox = inner_checkbox  # type: ignore
        w.inner_controls = [inner_label, inner_a, inner_ae,  # type: ignore
                            inner_long_peri, inner_rate_peri, inner_rms]
        # Store individual spinbox references for reading values
        w.inner_a = inner_a  # type: ignore
        w.inner_ae = inner_ae  # type: ignore
        w.inner_long_peri = inner_long_peri  # type: ignore
        w.inner_rate_peri = inner_rate_peri  # type: ignore
        w.inner_rms = inner_rms  # type: ignore

        # Outer edge checkbox and mode 1 parameters
        outer_data = p.get('outer_data', [])
        has_outer = len(outer_data) > 0 and any(m.get('mode') == 1 for m in outer_data)
        outer_mode1: dict[str, Any] = next((m for m in outer_data if m.get('mode') == 1), {})

        outer_checkbox = QCheckBox('Enable Outer Edge')
        outer_checkbox.setChecked(has_outer)
        outer_checkbox.clicked.connect(
            lambda checked, i=idx, cb=outer_checkbox: self._on_ring_outer_enabled(
                i, checked, cb)
        )
        fl.addRow(outer_checkbox, QLabel(''))

        outer_label = QLabel('<b>Outer Edge (Mode 1)</b>')
        fl.addRow(outer_label, QLabel(''))

        outer_a = QDoubleSpinBox()
        outer_a.setRange(1.0, 10000.0)
        outer_a.setDecimals(1)
        outer_a.setValue(outer_mode1.get('a', 120.0))
        outer_a.setEnabled(has_outer)
        outer_a.valueChanged.connect(
            lambda v, i=idx: self._on_ring_outer_mode1(i, 'a', v)
        )
        fl.addRow('Outer a:', outer_a)
        outer_ae = QDoubleSpinBox()
        outer_ae.setRange(0.0, 1000.0)
        outer_ae.setDecimals(2)
        outer_ae.setValue(outer_mode1.get('ae', 0.0))
        outer_ae.setEnabled(has_outer)
        outer_ae.valueChanged.connect(
            lambda v, i=idx: self._on_ring_outer_mode1(i, 'ae', v)
        )
        fl.addRow('Outer ae:', outer_ae)
        outer_long_peri = QDoubleSpinBox()
        outer_long_peri.setRange(0.0, 360.0)
        outer_long_peri.setDecimals(1)
        outer_long_peri.setSuffix('°')
        outer_long_peri.setValue(outer_mode1.get('long_peri', 0.0))
        outer_long_peri.setEnabled(has_outer)
        outer_long_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_outer_mode1(i, 'long_peri', v)
        )
        fl.addRow('Outer long_peri:', outer_long_peri)
        outer_rate_peri = QDoubleSpinBox()
        outer_rate_peri.setRange(-1000.0, 1000.0)
        outer_rate_peri.setDecimals(3)
        outer_rate_peri.setSuffix('°/day')
        outer_rate_peri.setValue(outer_mode1.get('rate_peri', 0.0))
        outer_rate_peri.setEnabled(has_outer)
        outer_rate_peri.valueChanged.connect(
            lambda v, i=idx: self._on_ring_outer_mode1(i, 'rate_peri', v)
        )
        fl.addRow('Outer rate_peri:', outer_rate_peri)
        outer_rms = QDoubleSpinBox()
        outer_rms.setRange(0.0, 1000.0)
        outer_rms.setDecimals(3)
        outer_rms.setValue(outer_mode1.get('rms', 1.0))
        outer_rms.setEnabled(has_outer)
        outer_rms.valueChanged.connect(
            lambda v, i=idx: self._on_ring_outer_mode1(i, 'rms', v)
        )
        fl.addRow('Outer rms:', outer_rms)

        # Store references for enabling/disabling
        w.outer_checkbox = outer_checkbox  # type: ignore[attr-defined]
        w.outer_controls = [outer_label, outer_a, outer_ae,  # type: ignore[attr-defined]
                            outer_long_peri, outer_rate_peri, outer_rms]
        # Store individual spinbox references for reading values
        w.outer_a = outer_a  # type: ignore[attr-defined]
        w.outer_ae = outer_ae  # type: ignore[attr-defined]
        w.outer_long_peri = outer_long_peri  # type: ignore[attr-defined]
        w.outer_rate_peri = outer_rate_peri  # type: ignore[attr-defined]
        w.outer_rms = outer_rms  # type: ignore[attr-defined]

        # Delete button at bottom
        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('ring', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

        return w

    # ---- Build star tab ----
    def _build_star_tab(self, idx: int) -> QWidget:
        p = self.sim_params['stars'][idx]
        w = QWidget()
        w.setProperty('kind', 'star')
        w.setProperty('data_index', idx)
        main_layout = QVBoxLayout(w)
        fl = QFormLayout()
        main_layout.addLayout(fl)

        name_edit = QLineEdit(p.get('name', ''))
        name_edit.textChanged.connect(lambda t, i=idx: self._on_star_name(i, t))
        fl.addRow('Name:', name_edit)

        v_spin = QDoubleSpinBox()
        v_spin.setRange(-10000.0, 20000.0)
        v_spin.setDecimals(1)
        v_spin.setValue(p.get('v', 0.0))
        v_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'v', v)
        )
        fl.addRow('V:', v_spin)
        u_spin = QDoubleSpinBox()
        u_spin.setRange(-10000.0, 20000.0)
        u_spin.setDecimals(1)
        u_spin.setValue(p.get('u', 0.0))
        u_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_field(i, 'u', v)
        )
        fl.addRow('U:', u_spin)
        # Keep references so drag updates can sync the UI
        w.v_spin = v_spin  # type: ignore[attr-defined]
        w.u_spin = u_spin  # type: ignore[attr-defined]

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

        # PSF size V slider with min/max labels and spinbox
        # Map slider positions 0-11 to odd values 1, 3, 5, ..., 23
        psf_size_v_row = QHBoxLayout()
        psf_size_v_row.setSpacing(4)
        psf_size_v_row.setContentsMargins(0, 0, 0, 0)
        psf_size_v_min_label = QLabel('1')
        psf_size_v_min_label.setFixedWidth(35)
        psf_size_v_min_label.setAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
        psf_size_v_max_label = QLabel('23')
        psf_size_v_max_label.setFixedWidth(40)
        psf_size_v_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                          Qt.AlignmentFlag.AlignVCenter)
        psf_size_v_slider = QSlider(Qt.Orientation.Horizontal)
        psf_size_v_slider.setRange(0, 11)  # 12 positions for odd values 1-23
        psf_size_v_default = p.get('psf_size', (11, 11))[0]
        psf_size_v_default = self._ensure_odd_psf_size(psf_size_v_default)
        # Convert odd value to slider position: (value - 1) // 2
        psf_size_v_slider.setValue((psf_size_v_default - 1) // 2)
        psf_size_v_slider.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_v_slider(i, v)
        )
        psf_size_v_spin = QSpinBox()
        psf_size_v_spin.setRange(1, 23)
        psf_size_v_spin.setSingleStep(2)  # Step by 2 to keep odd
        psf_size_v_spin.setValue(psf_size_v_default)
        psf_size_v_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_v_spin(i, v)
        )
        psf_size_v_row.addWidget(psf_size_v_min_label)
        psf_size_v_row.addWidget(psf_size_v_slider, stretch=1)
        psf_size_v_row.addWidget(psf_size_v_max_label)
        psf_size_v_row.addWidget(psf_size_v_spin)
        psf_size_v_holder = QWidget()
        psf_size_v_holder.setLayout(psf_size_v_row)
        fl.addRow('PSF size V:', psf_size_v_holder)
        # Store references for sync
        w.psf_size_v_slider = psf_size_v_slider  # type: ignore[attr-defined]
        w.psf_size_v_spin = psf_size_v_spin  # type: ignore[attr-defined]

        # PSF size U slider with min/max labels and spinbox
        # Map slider positions 0-11 to odd values 1, 3, 5, ..., 23
        psf_size_u_row = QHBoxLayout()
        psf_size_u_row.setSpacing(4)
        psf_size_u_row.setContentsMargins(0, 0, 0, 0)
        psf_size_u_min_label = QLabel('1')
        psf_size_u_min_label.setFixedWidth(35)
        psf_size_u_min_label.setAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
        psf_size_u_max_label = QLabel('23')
        psf_size_u_max_label.setFixedWidth(40)
        psf_size_u_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft |
                                          Qt.AlignmentFlag.AlignVCenter)
        psf_size_u_slider = QSlider(Qt.Orientation.Horizontal)
        psf_size_u_slider.setRange(0, 11)  # 12 positions for odd values 1-23
        psf_size_u_default = p.get('psf_size', (11, 11))[1]
        psf_size_u_default = self._ensure_odd_psf_size(psf_size_u_default)
        # Convert odd value to slider position: (value - 1) // 2
        psf_size_u_slider.setValue((psf_size_u_default - 1) // 2)
        psf_size_u_slider.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_u_slider(i, v)
        )
        psf_size_u_spin = QSpinBox()
        psf_size_u_spin.setRange(1, 23)
        psf_size_u_spin.setSingleStep(2)  # Step by 2 to keep odd
        psf_size_u_spin.setValue(psf_size_u_default)
        psf_size_u_spin.valueChanged.connect(
            lambda v, i=idx: self._on_star_psf_size_u_spin(i, v)
        )
        psf_size_u_row.addWidget(psf_size_u_min_label)
        psf_size_u_row.addWidget(psf_size_u_slider, stretch=1)
        psf_size_u_row.addWidget(psf_size_u_max_label)
        psf_size_u_row.addWidget(psf_size_u_spin)
        psf_size_u_holder = QWidget()
        psf_size_u_holder.setLayout(psf_size_u_row)
        fl.addRow('PSF size U:', psf_size_u_holder)
        # Store references for sync
        w.psf_size_u_slider = psf_size_u_slider  # type: ignore[attr-defined]
        w.psf_size_u_spin = psf_size_u_spin  # type: ignore[attr-defined]

        # Delete button at bottom
        delete_btn = QPushButton('Delete')
        delete_btn.clicked.connect(
            lambda _checked=False, i=idx: self._delete_tab_by_index('star', i)
        )
        main_layout.addStretch()
        main_layout.addWidget(delete_btn)

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

    def _on_body_name(self, idx: int, text: str) -> None:
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['name'] = text
            # update tab title
            self._update_tab_titles()
            self._updater.request_update()

    def _on_body_crater_fill_slider(self, idx: int, value: int) -> None:
        fill_val = value / 1000.0
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin = tab_w.crater_fill_spin  # type: ignore
                spin.blockSignals(True)
                spin.setValue(fill_val)
                spin.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['crater_fill'] = fill_val
            self._updater.request_update()

    def _on_body_crater_fill_spin(self, idx: int, value: float) -> None:
        slider_val = int(value * 1000)
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                slider = tab_w.crater_fill_slider  # type: ignore
                slider.blockSignals(True)
                slider.setValue(slider_val)
                slider.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['crater_fill'] = value
            self._updater.request_update()

    def _on_body_anti_aliasing_slider(self, idx: int, value: int) -> None:
        aa_val = value / 1000.0
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin = tab_w.anti_aliasing_spin  # type: ignore
                spin.blockSignals(True)
                spin.setValue(aa_val)
                spin.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['anti_aliasing'] = aa_val
            self._updater.request_update()

    def _on_body_anti_aliasing_spin(self, idx: int, value: float) -> None:
        slider_val = int(value * 1000)
        tab_idx = self._find_tab_by_properties('body', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                slider = tab_w.anti_aliasing_slider  # type: ignore
                slider.blockSignals(True)
                slider.setValue(slider_val)
                slider.blockSignals(False)
        if 0 <= idx < len(self.sim_params['bodies']):
            self.sim_params['bodies'][idx]['anti_aliasing'] = value
            self._updater.request_update()

    def _on_ring_field(self, idx: int, key: str, value: Any) -> None:
        if 0 <= idx < len(self.sim_params['rings']):
            self.sim_params['rings'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_ring_name(self, idx: int, text: str) -> None:
        if 0 <= idx < len(self.sim_params['rings']):
            self.sim_params['rings'][idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _get_or_create_mode1(self,
                             data_list: list[dict[str, Any]],
                             default_a: float) -> dict[str, Any]:
        """Find or create mode 1 dictionary in the given data list.

        Parameters:
            data_list: List of mode dictionaries (inner_data or outer_data).
            default_a: Default 'a' value to use when creating new mode 1.

        Returns:
            Mode 1 dictionary (existing or newly created).
        """
        mode1 = next((m for m in data_list if m.get('mode') == 1), None)
        if mode1 is None:
            mode1 = {'mode': 1, 'a': default_a, 'rms': 1.0, 'ae': 0.0,
                     'long_peri': 0.0, 'rate_peri': 0.0}
            data_list.append(mode1)
        return mode1

    def _on_ring_inner_mode1(self, idx: int, key: str, value: float) -> None:
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]
            if 'inner_data' not in ring:
                ring['inner_data'] = []
            inner_data = ring['inner_data']
            mode1 = self._get_or_create_mode1(inner_data, 100.0)
            mode1[key] = float(value)
            # Ensure rms is always present
            if 'rms' not in mode1:
                mode1['rms'] = 1.0
            self._updater.request_update()

    def _on_ring_outer_mode1(self, idx: int, key: str, value: float) -> None:
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]
            if 'outer_data' not in ring:
                ring['outer_data'] = []
            outer_data = ring['outer_data']
            mode1 = self._get_or_create_mode1(outer_data, 120.0)
            mode1[key] = float(value)
            # Ensure rms is always present
            if 'rms' not in mode1:
                mode1['rms'] = 1.0
            self._updater.request_update()

    def _on_ring_inner_enabled(self, idx: int, enabled: bool, checkbox: QCheckBox) -> None:
        """Handle inner edge checkbox state change.

        Parameters:
            idx: Ring index.
            enabled: Whether the inner edge checkbox is now enabled.
            checkbox: The inner edge checkbox widget.
        """
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]

            # Get the outer checkbox state directly
            tab_idx = self._find_tab_by_properties('ring', idx)
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    outer_checkbox = tab_widget.outer_checkbox  # type: ignore
                    has_outer_checked = outer_checkbox.isChecked()
                else:
                    # Fallback to data model if widget not found
                    outer_data = ring.get('outer_data', [])
                    has_outer_checked = (len(outer_data) > 0 and
                                         any(m.get('mode') == 1 for m in outer_data))
            else:
                # Fallback to data model if tab not found
                outer_data = ring.get('outer_data', [])
                has_outer_checked = (len(outer_data) > 0 and
                                     any(m.get('mode') == 1 for m in outer_data))

            # Prevent disabling if outer is also disabled
            if not enabled and not has_outer_checked:
                # Re-enable the checkbox - block signals to prevent recursion
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                return

            if enabled:
                # Enable inner edge - ensure mode 1 exists
                if 'inner_data' not in ring:
                    ring['inner_data'] = []
                inner_data = ring['inner_data']
                mode1 = self._get_or_create_mode1(inner_data, 100.0)
                # Read current values from UI controls instead of using defaults
                if tab_idx is not None:
                    tab_widget = self._tabs.widget(tab_idx)
                    if tab_widget is not None:
                        mode1['a'] = float(tab_widget.inner_a.value())  # type: ignore
                        mode1['ae'] = float(tab_widget.inner_ae.value())  # type: ignore
                        mode1['long_peri'] = float(
                            tab_widget.inner_long_peri.value())  # type: ignore
                        mode1['rate_peri'] = float(
                            tab_widget.inner_rate_peri.value())  # type: ignore
                        mode1['rms'] = float(tab_widget.inner_rms.value())  # type: ignore
            else:
                # Disable inner edge - remove inner_data
                ring.pop('inner_data', None)

            # Update UI controls
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    for control in tab_widget.inner_controls:  # type: ignore
                        control.setEnabled(enabled)

            self._updater.request_update()

    def _on_ring_outer_enabled(self, idx: int, enabled: bool, checkbox: QCheckBox) -> None:
        """Handle outer edge checkbox state change.

        Parameters:
            idx: Ring index.
            enabled: Whether the outer edge checkbox is now enabled.
            checkbox: The outer edge checkbox widget.
        """
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]

            # Get the inner checkbox state directly
            tab_idx = self._find_tab_by_properties('ring', idx)
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    inner_checkbox = tab_widget.inner_checkbox  # type: ignore
                    has_inner_checked = inner_checkbox.isChecked()
                else:
                    # Fallback to data model if widget not found
                    inner_data = ring.get('inner_data', [])
                    has_inner_checked = (len(inner_data) > 0 and
                                         any(m.get('mode') == 1 for m in inner_data))
            else:
                # Fallback to data model if tab not found
                inner_data = ring.get('inner_data', [])
                has_inner_checked = (len(inner_data) > 0 and
                                     any(m.get('mode') == 1 for m in inner_data))

            # Prevent disabling if inner is also disabled
            if not enabled and not has_inner_checked:
                # Re-enable the checkbox - block signals to prevent recursion
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                return

            if enabled:
                # Enable outer edge - ensure mode 1 exists
                if 'outer_data' not in ring:
                    ring['outer_data'] = []
                outer_data = ring['outer_data']
                mode1 = self._get_or_create_mode1(outer_data, 120.0)
                # Read current values from UI controls instead of using defaults
                if tab_idx is not None:
                    tab_widget = self._tabs.widget(tab_idx)
                    if tab_widget is not None:
                        mode1['a'] = float(tab_widget.outer_a.value())  # type: ignore
                        mode1['ae'] = float(tab_widget.outer_ae.value())  # type: ignore
                        mode1['long_peri'] = float(
                            tab_widget.outer_long_peri.value())  # type: ignore
                        mode1['rate_peri'] = float(
                            tab_widget.outer_rate_peri.value())  # type: ignore
                        mode1['rms'] = float(tab_widget.outer_rms.value())  # type: ignore
            else:
                # Disable outer edge - remove outer_data
                ring.pop('outer_data', None)

            # Update UI controls
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    for control in tab_widget.outer_controls:  # type: ignore
                        control.setEnabled(enabled)

            self._updater.request_update()

    def _on_star_field(self, idx: int, key: str, value: Any) -> None:
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_star_name(self, idx: int, text: str) -> None:
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _ensure_odd_psf_size(self, value: int) -> int:
        """Ensure PSF size is an odd integer in the range [1, 23].

        Parameters:
            value: The value to normalize.

        Returns:
            Clamped odd integer value in [1, 23].
        """
        value = int(value)
        value = max(1, min(23, value))
        if value % 2 == 0:
            value = max(1, value - 1)
        return value

    def _on_star_psf_size_slider(self, idx: int, dimension: int, value: int) -> None:
        """Handle PSF size slider change for a star.

        Parameters:
            idx: Star index.
            dimension: 0 for V, 1 for U.
            value: Slider value (0-11).
        """
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        # Convert slider position (0-11) to odd value (1, 3, 5, ..., 23)
        odd_value = value * 2 + 1
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                spin_attr = 'psf_size_v_spin' if dimension == 0 else 'psf_size_u_spin'
                spin = getattr(tab_w, spin_attr, None)
                if spin is not None:
                    spin.blockSignals(True)
                    spin.setValue(odd_value)
                    spin.blockSignals(False)
        current_psf_size = self.sim_params['stars'][idx].get('psf_size', (11, 11))
        if dimension == 0:
            self.sim_params['stars'][idx]['psf_size'] = (odd_value, current_psf_size[1])
        else:
            self.sim_params['stars'][idx]['psf_size'] = (current_psf_size[0], odd_value)
        self._updater.request_update()

    def _on_star_psf_size_spin(self, idx: int, dimension: int, value: int) -> None:
        """Handle PSF size spinbox change for a star.

        Parameters:
            idx: Star index.
            dimension: 0 for V, 1 for U.
            value: Spinbox value.
        """
        if not (0 <= idx < len(self.sim_params['stars'])):
            return
        # Coerce to nearest odd in [1, 23] and clamp
        odd_value = self._ensure_odd_psf_size(value)
        tab_idx = self._find_tab_by_properties('star', idx)
        if tab_idx is not None:
            tab_w = self._tabs.widget(tab_idx)
            if tab_w is not None:
                # Update spinbox if value was adjusted
                if odd_value != value:
                    spin_attr = 'psf_size_v_spin' if dimension == 0 else 'psf_size_u_spin'
                    spin = getattr(tab_w, spin_attr, None)
                    if spin is not None:
                        spin.blockSignals(True)
                        spin.setValue(odd_value)
                        spin.blockSignals(False)
                # Convert odd value to slider position: (value - 1) // 2
                slider_attr = 'psf_size_v_slider' if dimension == 0 else 'psf_size_u_slider'
                slider = getattr(tab_w, slider_attr, None)
                if slider is not None:
                    slider.blockSignals(True)
                    slider.setValue((odd_value - 1) // 2)
                    slider.blockSignals(False)
        current_psf_size = self.sim_params['stars'][idx].get('psf_size', (11, 11))
        if dimension == 0:
            self.sim_params['stars'][idx]['psf_size'] = (odd_value, current_psf_size[1])
        else:
            self.sim_params['stars'][idx]['psf_size'] = (current_psf_size[0], odd_value)
        self._updater.request_update()

    def _on_star_psf_size_v_slider(self, idx: int, value: int) -> None:
        """Handle PSF size V slider change for a star."""
        self._on_star_psf_size_slider(idx, 0, value)

    def _on_star_psf_size_v_spin(self, idx: int, value: int) -> None:
        """Handle PSF size V spinbox change for a star."""
        self._on_star_psf_size_spin(idx, 0, value)

    def _on_star_psf_size_u_slider(self, idx: int, value: int) -> None:
        """Handle PSF size U slider change for a star."""
        self._on_star_psf_size_slider(idx, 1, value)

    def _on_star_psf_size_u_spin(self, idx: int, value: int) -> None:
        """Handle PSF size U spinbox change for a star."""
        self._on_star_psf_size_spin(idx, 1, value)

    def _update_tab_titles(self) -> None:
        # Rebuild tabs to maintain sorted order when names change
        # This ensures bodies and stars are always sorted by name
        self._rebuild_dynamic_tabs()

    def _find_unique_range(self) -> float:
        """Find a unique range value by incrementing from 1 until one doesn't exist."""
        existing_ranges = set()
        for body in self.sim_params.get('bodies', []):
            range_val = body.get('range')
            if range_val is not None:
                existing_ranges.add(float(range_val))
        for ring in self.sim_params.get('rings', []):
            range_val = ring.get('range')
            if range_val is not None:
                existing_ranges.add(float(range_val))

        # Start from 1 and increment until we find a unique range
        candidate = 1.0
        while candidate in existing_ranges:
            candidate += 1.0
        return candidate

    def _validate_ranges(self) -> None:
        """Check for duplicate body ranges and display a warning if found."""
        ranges = []
        for i in range(len(self.sim_params['bodies'])):
            range_val = self.sim_params['bodies'][i].get('range')
            if range_val is not None:
                ranges.append(float(range_val))
        duplicates = len(ranges) != len(set(ranges))
        self._warning_label.setText(
            'Warning: duplicate body ranges' if duplicates else ''
        )

    # ---- Rendering ----
    def _update_image(self) -> None:
        try:
            # Render image (caching is handled in render.py)
            img, meta = render_combined_model(self.sim_params, ignore_offset=True)
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
            # Draw ring centers as small circles
            pen = QPen(QColor(0, 255, 255), 2)
            painter.setPen(pen)
            for r in self.sim_params.get('rings', []):
                center_u = int(r.get('center_u', 0))
                center_v = int(r.get('center_v', 0))
                painter.drawEllipse(center_u - 4, center_v - 4, 8, 8)
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

    # ---- Selection / drag-move ----
    def _select_model_at(self, img_v: float, img_u: float) -> None:
        """Select the topmost model at the given image coordinates based on range ordering.

        Checks bodies and rings together, sorted by range (near to far), and selects
        the first match (topmost object).
        """
        height = int(self.sim_params['size_v'])
        width = int(self.sim_params['size_u'])
        v_i = int(round(img_v))
        u_i = int(round(img_u))
        if not (0 <= v_i < height and 0 <= u_i < width):
            self._selected_model_key = None
            return

        # Collect all objects (bodies and rings) with their ranges and masks
        objects: list[tuple[float, str, int, Any]] = []  # (range, kind, index, mask)

        # Add bodies
        body_masks = self._last_meta.get('body_masks', [])
        bodies = self.sim_params.get('bodies', [])
        inv = self._last_meta.get('inventory', {})
        if body_masks and bodies:
            # body_masks is in the original order of bodies_params, so match by index
            for idx, body in enumerate(bodies):
                if idx < len(body_masks):
                    body_name = body.get('name', '').upper()
                    range_val = inv.get(body_name, {}).get('range', float('inf'))
                    objects.append((range_val, 'body', idx, body_masks[idx]))

        # Add rings
        ring_masks = self._last_meta.get('ring_masks', [])
        rings = self.sim_params.get('rings', [])
        if ring_masks and rings:
            for j, ring_mask in enumerate(ring_masks):
                if j < len(rings):
                    range_val = float(rings[j].get('range', 1000.0 + j))
                    objects.append((range_val, 'ring', j, ring_mask))

        # Sort by range (near to far = ascending range)
        objects.sort(key=lambda x: x[0])

        # Check objects in order (near to far), select first match
        for _, kind, idx, mask in objects:
            if mask is not None and bool(mask[v_i, u_i]):
                self._selected_model_key = (kind, idx)
                tab_idx = self._find_tab_by_properties(kind, idx)
                if tab_idx is not None:
                    self._tabs.setCurrentIndex(tab_idx)
                return

        # Stars: evaluate PSF contribution approx via Gaussian envelope
        # Stars are always behind bodies and rings
        star_info = self._last_meta.get('star_info', [])
        if star_info:
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
                    # Switch to star tab by finding it by properties
                    tab_idx = self._find_tab_by_properties('star', j)
                    if tab_idx is not None:
                        self._tabs.setCurrentIndex(tab_idx)
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
            # Sync the tab spin boxes for this body
            tab_idx = self._find_tab_by_properties('body', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    cv_spin = tab_w.center_v_spin  # type: ignore
                    cu_spin = tab_w.center_u_spin  # type: ignore
                    cv_spin.setValue(self.sim_params['bodies'][idx]['center_v'])
                    cu_spin.setValue(self.sim_params['bodies'][idx]['center_u'])
            self._updater.immediate_update()
        elif kind == 'ring' and 0 <= idx < len(self.sim_params['rings']):
            self.sim_params['rings'][idx]['center_v'] = float(
                self.sim_params['rings'][idx].get('center_v', 0.0) + dv
            )
            self.sim_params['rings'][idx]['center_u'] = float(
                self.sim_params['rings'][idx].get('center_u', 0.0) + du
            )
            # Sync the tab spin boxes for this ring
            tab_idx = self._find_tab_by_properties('ring', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    cv_spin = tab_w.center_v_spin  # type: ignore
                    cu_spin = tab_w.center_u_spin  # type: ignore
                    cv_spin.setValue(self.sim_params['rings'][idx]['center_v'])
                    cu_spin.setValue(self.sim_params['rings'][idx]['center_u'])
            self._updater.immediate_update()
        elif kind == 'star' and 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx]['v'] = float(
                self.sim_params['stars'][idx].get('v', 0.0) + dv
            )
            self.sim_params['stars'][idx]['u'] = float(
                self.sim_params['stars'][idx].get('u', 0.0) + du
            )
            # Sync the tab spin boxes for this star
            tab_idx = self._find_tab_by_properties('star', idx)
            if tab_idx is not None:
                tab_w = self._tabs.widget(tab_idx)
                if tab_w is not None:
                    v_spin = tab_w.v_spin  # type: ignore
                    u_spin = tab_w.u_spin  # type: ignore
                    v_spin.setValue(self.sim_params['stars'][idx]['v'])
                    u_spin.setValue(self.sim_params['stars'][idx]['u'])
            self._updater.immediate_update()
        else:
            raise AssertionError(f'Unknown kind: {kind}')
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
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image:\n{str(e)}')

    def _save_parameters(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Parameters',
            'simulated_image_params.json',
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
                background_noise_val = params.get('background_noise_intensity', 0.0)
                background_stars_val = params.get('background_stars_num', 0)
                self.sim_params = {
                    'size_v': int(params.get('size_v', 512)),
                    'size_u': int(params.get('size_u', 512)),
                    'offset_v': float(params.get('offset_v', 0.0)),
                    'offset_u': float(params.get('offset_u', 0.0)),
                    'random_seed': int(params.get('random_seed', 42)),
                    'background_noise_intensity': float(background_noise_val),
                    'background_stars_num': int(background_stars_val),
                    'background_stars_psf_sigma': float(
                        params.get('background_stars_psf_sigma', 0.9)),
                    'background_stars_distribution_exponent': float(
                        params.get('background_stars_distribution_exponent', 2.5)),
                    'time': float(params.get('time', 0.0)),
                    'ring_epoch': float(params.get('ring_epoch', 0.0)),
                    'closest_planet': params.get('closest_planet'),
                    'bodies': list(params.get('bodies', [])),
                    'stars': list(params.get('stars', [])),
                    'rings': list(params.get('rings', [])),
                }
                # Update general UI
                self._size_v_spin.setValue(self.sim_params['size_v'])
                self._size_u_spin.setValue(self.sim_params['size_u'])
                self._offset_v_spin.setValue(self.sim_params['offset_v'])
                self._offset_u_spin.setValue(self.sim_params['offset_u'])
                self._random_seed_spin.setValue(self.sim_params['random_seed'])
                # Update time and epoch
                self._time_spin.setValue(self.sim_params.get('time', 0.0))
                self._epoch_spin.setValue(self.sim_params.get('ring_epoch', 0.0))
                # Update closest planet
                closest_planet = self.sim_params.get('closest_planet', 'SATURN')
                index = self._closest_planet_combo.findText(closest_planet)
                if index >= 0:
                    self._closest_planet_combo.setCurrentIndex(index)
                else:
                    self._closest_planet_combo.setCurrentText(closest_planet)
                # Update background noise controls
                self._background_noise_slider.blockSignals(True)
                noise_slider_val = int(self.sim_params['background_noise_intensity'] * 1000)
                self._background_noise_slider.setValue(noise_slider_val)
                self._background_noise_slider.blockSignals(False)
                self._background_noise_spin.blockSignals(True)
                self._background_noise_spin.setValue(self.sim_params['background_noise_intensity'])
                self._background_noise_spin.blockSignals(False)
                # Update background stars controls
                self._background_stars_slider.blockSignals(True)
                self._background_stars_slider.setValue(self.sim_params['background_stars_num'])
                self._background_stars_slider.blockSignals(False)
                self._background_stars_spin.blockSignals(True)
                self._background_stars_spin.setValue(self.sim_params['background_stars_num'])
                self._background_stars_spin.blockSignals(False)
                # Update background stars PSF sigma controls
                self._background_stars_psf_sigma_slider.blockSignals(True)
                psf_sigma_val = int(self.sim_params['background_stars_psf_sigma'] * 100)
                self._background_stars_psf_sigma_slider.setValue(psf_sigma_val)
                self._background_stars_psf_sigma_slider.blockSignals(False)
                self._background_stars_psf_sigma_spin.blockSignals(True)
                psf_sigma_spin_val = self.sim_params['background_stars_psf_sigma']
                self._background_stars_psf_sigma_spin.setValue(psf_sigma_spin_val)
                self._background_stars_psf_sigma_spin.blockSignals(False)
                # Update background stars distribution exponent controls
                self._background_stars_dist_exp_slider.blockSignals(True)
                dist_exp_slider_val = int(
                    self.sim_params['background_stars_distribution_exponent'] * 100)
                self._background_stars_dist_exp_slider.setValue(dist_exp_slider_val)
                self._background_stars_dist_exp_slider.blockSignals(False)
                self._background_stars_dist_exp_spin.blockSignals(True)
                dist_exp_spin_val = self.sim_params['background_stars_distribution_exponent']
                self._background_stars_dist_exp_spin.setValue(dist_exp_spin_val)
                self._background_stars_dist_exp_spin.blockSignals(False)
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
        self._zoom_sharp = (state == int(cast(int, Qt.CheckState.Checked.value)))
        self._update_display()

    def _toggle_shade_solid_rings(self, state: Any) -> None:
        checked = (state == int(cast(int, Qt.CheckState.Checked.value)))
        self.sim_params['shade_solid_rings'] = checked
        self._updater.request_update()


def main() -> None:
    app = QApplication(sys.argv)
    window = CreateSimulatedImageModel()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
