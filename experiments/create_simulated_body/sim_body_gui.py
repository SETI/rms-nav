#!/usr/bin/env python3
"""GUI application for experimenting with create_simulated_body function.

This application provides an interactive interface to test and visualize
the create_simulated_body function with real-time parameter adjustment.
"""
import json
import math
import sys
from typing import Any

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
    QWheelEvent
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QFormLayout,
    QStatusBar,
)

from nav.support.sim import create_simulated_body


class ParameterUpdater(QObject):
    """Helper class to handle debounced parameter updates."""

    update_requested = pyqtSignal()

    def __init__(self, delay_ms: int = 100) -> None:
        super().__init__()
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.update_requested)
        self._delay_ms = delay_ms

    def request_update(self) -> None:
        """Request an update, resetting the timer if already running."""
        self._timer.stop()
        self._timer.start(self._delay_ms)

    def immediate_update(self) -> None:
        """Trigger an immediate update."""
        self._timer.stop()
        self.update_requested.emit()


class SimulatedBodyGUI(QMainWindow):
    """Main GUI window for simulated body creation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Simulated Body Generator')
        self.setMinimumSize(1200, 800)

        self._set_defaults()

        # Current image data
        self._current_image: np.ndarray | None = None
        self._base_pixmap: QPixmap | None = None  # Original pixmap with visual aids

        # Zoom and pan state
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start_pos: QPoint | None = None
        self._drag_start_pan: tuple[float, float] | None = None

        # Parameter updater for debouncing
        self._updater = ParameterUpdater(delay_ms=150)
        self._updater.update_requested.connect(self._update_image)

        # Setup UI
        self._setup_ui()

        # Initial image generation
        self._update_image()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle window resize events."""
        super().resizeEvent(event)
        # Redisplay image with current zoom and pan
        if self._current_image is not None:
            self._display_image()

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Store mouse position in global coordinates to avoid coordinate system issues
            self._drag_start_pos = event.globalPosition().toPoint()
            # Store current pan position
            self._drag_start_pan = (self._pan_x, self._pan_y)
            self._image_label.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_mouse_move(self, event: QMouseEvent) -> None:
        """Handle mouse move for panning and status bar update."""
        if self._drag_start_pos is not None and self._drag_start_pan is not None:
            # Get current mouse position in global coordinates
            current_pos = event.globalPosition().toPoint()
            # Calculate delta in global coordinates (device-independent)
            delta = current_pos - self._drag_start_pos
            # Image moves with mouse: drag right = image moves right (scroll left, so pan decreases)
            self._pan_x = self._drag_start_pan[0] - delta.x()
            self._pan_y = self._drag_start_pan[1] - delta.y()
            self._update_display()
        else:
            # Update status bar with v,u coordinates
            self._update_status_bar(event.position().toPoint())

    def _on_mouse_release(self, event: QMouseEvent) -> None:
        """Handle mouse release for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = None
            self._drag_start_pan = None
            self._image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_wheel(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        # Get mouse position relative to the label widget
        label_pos = event.position().toPoint()
        # Map label position into scroll-area viewport coordinates
        viewport_pos = self._image_label.mapTo(self._scroll_area.viewport(), label_pos)
        viewport_x = viewport_pos.x()
        viewport_y = viewport_pos.y()
        # Position in scaled image = viewport position + scroll offset
        scrollbar_h = self._scroll_area.horizontalScrollBar()
        scrollbar_v = self._scroll_area.verticalScrollBar()
        scaled_image_x = viewport_x + scrollbar_h.value()
        scaled_image_y = viewport_y + scrollbar_v.value()

        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_in_at_point(viewport_x, viewport_y, scaled_image_x, scaled_image_y)
        else:
            self._zoom_out_at_point(viewport_x, viewport_y, scaled_image_x, scaled_image_y)

    def _zoom_in(self) -> None:
        """Zoom in at center."""
        if self._base_pixmap is not None:
            # Use viewport center as the reference point
            viewport = self._scroll_area.viewport()
            center_x = viewport.width() // 2
            center_y = viewport.height() // 2
            # Scaled image coordinates under viewport center
            scrollbar_h = self._scroll_area.horizontalScrollBar()
            scrollbar_v = self._scroll_area.verticalScrollBar()
            scaled_x = center_x + scrollbar_h.value()
            scaled_y = center_y + scrollbar_v.value()
            self._zoom_at_point(1.2, center_x, center_y, scaled_x, scaled_y)

    def _zoom_out(self) -> None:
        """Zoom out at center."""
        if self._base_pixmap is not None:
            # Use viewport center as the reference point
            viewport = self._scroll_area.viewport()
            center_x = viewport.width() // 2
            center_y = viewport.height() // 2
            # Scaled image coordinates under viewport center
            scrollbar_h = self._scroll_area.horizontalScrollBar()
            scrollbar_v = self._scroll_area.verticalScrollBar()
            scaled_x = center_x + scrollbar_h.value()
            scaled_y = center_y + scrollbar_v.value()
            self._zoom_at_point(1.0 / 1.2, center_x, center_y, scaled_x, scaled_y)

    def _zoom_in_at_point(self, viewport_x: int, viewport_y: int, scaled_x: float,
                          scaled_y: float) -> None:
        """Zoom in at a specific point."""
        self._zoom_at_point(1.2, viewport_x, viewport_y, scaled_x, scaled_y)

    def _zoom_out_at_point(self, viewport_x: int, viewport_y: int, scaled_x: float,
                           scaled_y: float) -> None:
        """Zoom out at a specific point."""
        self._zoom_at_point(1.0 / 1.2, viewport_x, viewport_y, scaled_x, scaled_y)

    def _zoom_at_point(self, factor: float, viewport_x: int, viewport_y: int, scaled_x: float,
                       scaled_y: float) -> None:
        """Zoom at a specific point, maintaining that point's position in image coordinates."""
        if self._base_pixmap is None:
            return

        old_zoom = self._zoom_factor
        new_zoom = old_zoom * factor
        new_zoom = max(0.1, min(50.0, new_zoom))  # Limit zoom range (increased max)

        if new_zoom == old_zoom:
            return

        # scaled_x, scaled_y are in scaled image coordinates (position in the current scaled image)
        # Convert to original image coordinates
        img_x = scaled_x / old_zoom
        img_y = scaled_y / old_zoom

        # After zoom, we want the same image point to be at the same viewport position
        # The viewport position is where the mouse is (viewport_x, viewport_y)
        # After zoom, the same image point will be at: new_scaled_pos = img_x * new_zoom
        # We want: viewport_pos = new_scaled_pos - new_pan
        # Therefore: new_pan = new_scaled_pos - viewport_pos
        # new_pan = img_x * new_zoom - viewport_x
        new_scroll_x = img_x * new_zoom - viewport_x
        new_scroll_y = img_y * new_zoom - viewport_y

        # Update pan to match the new scroll position
        self._pan_x = new_scroll_x
        self._pan_y = new_scroll_y

        self._zoom_factor = new_zoom
        # Update zoom label in status bar
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _reset_view(self) -> None:
        """Reset zoom and pan to default."""
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        # Update zoom label in status bar
        if hasattr(self, '_zoom_label'):
            self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')
        self._update_display()

    def _update_status_bar(self, label_pos: QPoint) -> None:
        """Update status bar with v,u coordinates and pixel value."""
        # Update zoom scale
        self._zoom_label.setText(f'zoom: {self._zoom_factor:.2f}x')

        if self._current_image is None:
            self._status_label.setText('v, u: --, --  value: --')
            return

        # label_pos is in label coordinates; since label pixels map 1:1 to scaled image,
        # the position in the scaled image is simply label_pos.
        scaled_x = float(label_pos.x())
        scaled_y = float(label_pos.y())

        # Convert to original image coordinates (accounting for zoom)
        img_u = scaled_x / self._zoom_factor
        img_v = scaled_y / self._zoom_factor

        # Check if within image bounds
        height, width = self._current_image.shape
        if 0 <= img_v < height and 0 <= img_u < width:
            # Get pixel value using bilinear interpolation
            v0 = int(img_v)
            u0 = int(img_u)
            v1 = min(v0 + 1, height - 1)
            u1 = min(u0 + 1, width - 1)

            # Bilinear interpolation weights
            dv = img_v - v0
            du = img_u - u0

            # Sample the four surrounding pixels
            val00 = self._current_image[v0, u0]
            val01 = self._current_image[v0, u1]
            val10 = self._current_image[v1, u0]
            val11 = self._current_image[v1, u1]

            # Bilinear interpolation
            val = (val00 * (1 - du) * (1 - dv) +
                   val01 * du * (1 - dv) +
                   val10 * (1 - du) * dv +
                   val11 * du * dv)

            self._status_label.setText(f'v, u: {img_v:.2f}, {img_u:.2f}  value: {val:.6f}')
        else:
            self._status_label.setText('v, u: --, --  value: --')

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left side: Image display with pan and zoom
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(5)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addStretch()
        self._zoom_out_button = QPushButton('Zoom -')
        self._zoom_out_button.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self._zoom_out_button)
        self._zoom_in_button = QPushButton('Zoom +')
        self._zoom_in_button.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self._zoom_in_button)
        self._zoom_reset_button = QPushButton('Reset View')
        self._zoom_reset_button.clicked.connect(self._reset_view)
        zoom_layout.addWidget(self._zoom_reset_button)
        zoom_layout.addStretch()
        image_layout.addLayout(zoom_layout)

        # Scroll area for panning
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setMinimumSize(600, 600)
        self._scroll_area.setStyleSheet('background-color: black;')
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image label inside scroll area
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet('background-color: black;')
        self._image_label.setMouseTracking(True)
        self._image_label.mousePressEvent = self._on_mouse_press
        self._image_label.mouseMoveEvent = self._on_mouse_move
        self._image_label.mouseReleaseEvent = self._on_mouse_release
        self._image_label.wheelEvent = self._on_wheel

        self._scroll_area.setWidget(self._image_label)
        image_layout.addWidget(self._scroll_area)

        main_layout.addWidget(image_container, stretch=2)

        # Status bar
        status_bar = QStatusBar()
        self._status_label = QLabel('v, u: --, --  value: --')
        status_bar.addWidget(self._status_label)
        self._zoom_label = QLabel('zoom: 1.00x')
        status_bar.addPermanentWidget(self._zoom_label)
        self.setStatusBar(status_bar)

        # Right side: Control panel
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(10)

        # Image Size group
        size_group = QGroupBox('Image Size')
        size_layout = QFormLayout()

        self._size_v_spin = QSpinBox()
        self._size_v_spin.setRange(64, 2048)
        self._size_v_spin.setValue(self._size_v)
        self._size_v_spin.valueChanged.connect(self._on_size_v_changed)
        size_layout.addRow('Size V (height):', self._size_v_spin)

        self._size_u_spin = QSpinBox()
        self._size_u_spin.setRange(64, 2048)
        self._size_u_spin.setValue(self._size_u)
        self._size_u_spin.valueChanged.connect(self._on_size_u_changed)
        size_layout.addRow('Size U (width):', self._size_u_spin)

        size_group.setLayout(size_layout)
        control_layout.addWidget(size_group)

        # Ellipse Geometry group
        ellipse_group = QGroupBox('Ellipse Geometry')
        ellipse_layout = QFormLayout()

        self._center_v_spin = QDoubleSpinBox()
        self._center_v_spin.setRange(0.0, 10000.0)
        self._center_v_spin.setDecimals(1)
        self._center_v_spin.setValue(self._center_v)
        self._center_v_spin.valueChanged.connect(self._on_center_v_changed)
        ellipse_layout.addRow('Center V:', self._center_v_spin)

        self._center_u_spin = QDoubleSpinBox()
        self._center_u_spin.setRange(0.0, 10000.0)
        self._center_u_spin.setDecimals(1)
        self._center_u_spin.setValue(self._center_u)
        self._center_u_spin.valueChanged.connect(self._on_center_u_changed)
        ellipse_layout.addRow('Center U:', self._center_u_spin)

        self._semi_major_spin = QDoubleSpinBox()
        self._semi_major_spin.setRange(1.0, 1000.0)
        self._semi_major_spin.setDecimals(1)
        self._semi_major_spin.setValue(self._semi_major_axis)
        self._semi_major_spin.valueChanged.connect(self._on_semi_major_changed)
        ellipse_layout.addRow('Semi-major axis:', self._semi_major_spin)

        self._semi_minor_spin = QDoubleSpinBox()
        self._semi_minor_spin.setRange(1.0, 1000.0)
        self._semi_minor_spin.setDecimals(1)
        self._semi_minor_spin.setValue(self._semi_minor_axis)
        self._semi_minor_spin.valueChanged.connect(self._on_semi_minor_changed)
        ellipse_layout.addRow('Semi-minor axis:', self._semi_minor_spin)

        self._semi_c_spin = QDoubleSpinBox()
        self._semi_c_spin.setRange(1.0, 1000.0)
        self._semi_c_spin.setDecimals(1)
        self._semi_c_spin.setValue(self._semi_c_axis)
        self._semi_c_spin.valueChanged.connect(self._on_semi_c_changed)
        ellipse_layout.addRow('Semi-c axis (depth):', self._semi_c_spin)

        self._rotation_z_spin = QDoubleSpinBox()
        self._rotation_z_spin.setRange(0.0, 360.0)
        self._rotation_z_spin.setDecimals(1)
        self._rotation_z_spin.setSuffix('°')
        self._rotation_z_spin.setValue(self._rotation_z)
        self._rotation_z_spin.valueChanged.connect(self._on_rotation_z_changed)
        ellipse_layout.addRow('Rotation Z:', self._rotation_z_spin)

        self._rotation_tilt_spin = QDoubleSpinBox()
        self._rotation_tilt_spin.setRange(0.0, 90.0)
        self._rotation_tilt_spin.setDecimals(1)
        self._rotation_tilt_spin.setSuffix('°')
        self._rotation_tilt_spin.setValue(self._rotation_tilt)
        self._rotation_tilt_spin.valueChanged.connect(self._on_rotation_tilt_changed)
        ellipse_layout.addRow('Rotation Tilt:', self._rotation_tilt_spin)

        ellipse_group.setLayout(ellipse_layout)
        control_layout.addWidget(ellipse_group)

        # Illumination group
        illum_group = QGroupBox('Illumination')
        illum_layout = QFormLayout()

        self._illum_angle_spin = QDoubleSpinBox()
        self._illum_angle_spin.setRange(0.0, 360.0)
        self._illum_angle_spin.setDecimals(1)
        self._illum_angle_spin.setSuffix('°')
        self._illum_angle_spin.setValue(self._illumination_angle)
        self._illum_angle_spin.valueChanged.connect(self._on_illum_angle_changed)
        illum_layout.addRow('Illumination angle:', self._illum_angle_spin)

        self._phase_angle_spin = QDoubleSpinBox()
        self._phase_angle_spin.setRange(0.0, 180.0)
        self._phase_angle_spin.setDecimals(1)
        self._phase_angle_spin.setSuffix('°')
        self._phase_angle_spin.setValue(self._phase_angle)
        self._phase_angle_spin.valueChanged.connect(self._on_phase_angle_changed)
        illum_layout.addRow('Phase angle:', self._phase_angle_spin)

        illum_group.setLayout(illum_layout)
        control_layout.addWidget(illum_group)

        # Crater Parameters group
        crater_group = QGroupBox('Craters')
        crater_layout = QFormLayout()

        self._crater_fill_spin = QDoubleSpinBox()
        self._crater_fill_spin.setRange(0.0, 10.0)
        self._crater_fill_spin.setDecimals(3)
        self._crater_fill_spin.setSingleStep(0.01)
        self._crater_fill_spin.setValue(self._crater_fill)
        self._crater_fill_spin.valueChanged.connect(self._on_crater_fill_changed)
        crater_layout.addRow('Crater fill (0-10):', self._crater_fill_spin)

        self._crater_min_radius_spin = QDoubleSpinBox()
        self._crater_min_radius_spin.setRange(0.01, 0.25)
        self._crater_min_radius_spin.setDecimals(3)
        self._crater_min_radius_spin.setSingleStep(0.005)
        self._crater_min_radius_spin.setValue(self._crater_min_radius)
        self._crater_min_radius_spin.valueChanged.connect(self._on_crater_min_radius_changed)
        crater_layout.addRow('Crater min radius (0.01-0.25):', self._crater_min_radius_spin)

        self._crater_max_radius_spin = QDoubleSpinBox()
        self._crater_max_radius_spin.setRange(0.01, 0.25)
        self._crater_max_radius_spin.setDecimals(3)
        self._crater_max_radius_spin.setSingleStep(0.005)
        self._crater_max_radius_spin.setValue(self._crater_max_radius)
        self._crater_max_radius_spin.valueChanged.connect(self._on_crater_max_radius_changed)
        crater_layout.addRow('Crater max radius (0.01-0.25):', self._crater_max_radius_spin)

        self._crater_power_law_exponent_spin = QDoubleSpinBox()
        self._crater_power_law_exponent_spin.setRange(1.1, 5.0)
        self._crater_power_law_exponent_spin.setDecimals(2)
        self._crater_power_law_exponent_spin.setSingleStep(0.05)
        self._crater_power_law_exponent_spin.setValue(self._crater_power_law_exponent)
        self._crater_power_law_exponent_spin.valueChanged.connect(self._on_crater_power_law_exponent_changed)
        crater_layout.addRow('Crater power-law exponent (1.1-5):', self._crater_power_law_exponent_spin)

        self._crater_relief_scale_spin = QDoubleSpinBox()
        self._crater_relief_scale_spin.setRange(0.0, 3.0)
        self._crater_relief_scale_spin.setDecimals(3)
        self._crater_relief_scale_spin.setSingleStep(0.01)
        self._crater_relief_scale_spin.setValue(self._crater_relief_scale)
        self._crater_relief_scale_spin.valueChanged.connect(self._on_crater_relief_scale_changed)
        crater_layout.addRow('Crater relief scale (0-3):', self._crater_relief_scale_spin)

        crater_group.setLayout(crater_layout)
        control_layout.addWidget(crater_group)

        # Quality group
        quality_group = QGroupBox('Quality')
        quality_layout = QFormLayout()

        self._anti_aliasing_slider = QSlider(Qt.Orientation.Horizontal)
        self._anti_aliasing_slider.setRange(0, 1000)
        self._anti_aliasing_slider.setValue(int(self._anti_aliasing * 1000))
        self._anti_aliasing_slider.valueChanged.connect(self._on_anti_aliasing_changed)
        self._anti_aliasing_label = QLabel(f'{self._anti_aliasing:.3f}')
        aa_layout = QHBoxLayout()
        aa_layout.addWidget(self._anti_aliasing_slider)
        aa_layout.addWidget(self._anti_aliasing_label)
        quality_layout.addRow('Anti-aliasing:', aa_layout)

        quality_group.setLayout(quality_layout)
        control_layout.addWidget(quality_group)

        # Visual Aids and Sharp Zoom
        visual_layout = QHBoxLayout()
        self._visual_aids_check = QCheckBox('Show Visual Aids')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._visual_aids_check.stateChanged.connect(self._on_visual_aids_changed)
        visual_layout.addWidget(self._visual_aids_check)

        self._zoom_sharp_check = QCheckBox('Sharp zoom')
        self._zoom_sharp_check.setChecked(self._zoom_sharp)
        self._zoom_sharp_check.stateChanged.connect(self._on_zoom_sharp_changed)
        visual_layout.addWidget(self._zoom_sharp_check)

        control_layout.addLayout(visual_layout)

        control_layout.addStretch()

        # Buttons
        button_layout = QVBoxLayout()

        self._update_button = QPushButton('Update Now')
        self._update_button.clicked.connect(self._force_update)
        button_layout.addWidget(self._update_button)

        self._save_image_button = QPushButton('Save Image (PNG)')
        self._save_image_button.clicked.connect(self._save_image)
        button_layout.addWidget(self._save_image_button)

        self._save_params_button = QPushButton('Save Parameters (JSON)')
        self._save_params_button.clicked.connect(self._save_parameters)
        button_layout.addWidget(self._save_params_button)

        self._load_params_button = QPushButton('Load Parameters (JSON)')
        self._load_params_button.clicked.connect(self._load_parameters)
        button_layout.addWidget(self._load_params_button)

        self._reset_button = QPushButton('Reset to Defaults')
        self._reset_button.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self._reset_button)

        control_layout.addLayout(button_layout)

        scroll_area.setWidget(control_widget)
        main_layout.addWidget(scroll_area, stretch=1)

    def _on_size_v_changed(self, value: int) -> None:
        """Handle size_v parameter change."""
        self._size_v = value
        self._updater.request_update()

    def _on_size_u_changed(self, value: int) -> None:
        """Handle size_u parameter change."""
        self._size_u = value
        self._updater.request_update()

    def _on_semi_major_changed(self, value: float) -> None:
        """Handle semi_major_axis parameter change."""
        self._semi_major_axis = value
        self._updater.request_update()

    def _on_semi_minor_changed(self, value: float) -> None:
        """Handle semi_minor_axis parameter change."""
        self._semi_minor_axis = value
        self._updater.request_update()

    def _on_semi_c_changed(self, value: float) -> None:
        """Handle semi_c_axis parameter change."""
        self._semi_c_axis = value
        self._updater.request_update()

    def _on_center_v_changed(self, value: float) -> None:
        """Handle center_v parameter change."""
        self._center_v = value
        self._updater.request_update()

    def _on_center_u_changed(self, value: float) -> None:
        """Handle center_u parameter change."""
        self._center_u = value
        self._updater.request_update()

    def _on_rotation_z_changed(self, value: float) -> None:
        """Handle rotation_z parameter change."""
        self._rotation_z = value
        self._updater.request_update()

    def _on_rotation_tilt_changed(self, value: float) -> None:
        """Handle rotation_tilt parameter change."""
        self._rotation_tilt = value
        self._updater.request_update()

    def _on_illum_angle_changed(self, value: float) -> None:
        """Handle illumination_angle parameter change."""
        self._illumination_angle = value
        self._updater.request_update()

    def _on_phase_angle_changed(self, value: float) -> None:
        """Handle phase_angle parameter change."""
        self._phase_angle = value
        self._updater.request_update()

    def _on_crater_fill_changed(self, value: float) -> None:
        """Handle crater_fill parameter change."""
        self._crater_fill = value
        self._updater.request_update()

    def _on_crater_min_radius_changed(self, value: float) -> None:
        """Handle crater_min_radius parameter change."""
        self._crater_min_radius = value
        self._updater.request_update()

    def _on_crater_max_radius_changed(self, value: float) -> None:
        """Handle crater_max_radius parameter change."""
        self._crater_max_radius = value
        self._updater.request_update()

    def _on_crater_power_law_exponent_changed(self, value: float) -> None:
        """Handle crater_power_law_exponent parameter change."""
        self._crater_power_law_exponent = value
        self._updater.request_update()

    def _on_crater_relief_scale_changed(self, value: float) -> None:
        """Handle crater_relief_scale parameter change."""
        self._crater_relief_scale = value
        self._updater.request_update()

    def _on_anti_aliasing_changed(self, value: int) -> None:
        """Handle anti_aliasing parameter change."""
        self._anti_aliasing = value / 1000.0
        self._anti_aliasing_label.setText(f'{self._anti_aliasing:.3f}')
        self._updater.request_update()

    def _on_visual_aids_changed(self, state: int) -> None:
        """Handle visual aids checkbox change."""
        # state is 0 (Unchecked), 1 (PartiallyChecked), or 2 (Checked)
        self._show_visual_aids = (state == Qt.CheckState.Checked.value)
        # Regenerate the base pixmap with or without visual aids
        if self._current_image is not None:
            # Force regeneration by clearing the base pixmap first
            self._base_pixmap = None
            self._display_image()

    def _on_zoom_sharp_changed(self, state: int) -> None:
        """Handle sharp zoom checkbox change."""
        self._zoom_sharp = (state == Qt.CheckState.Checked.value)
        self._update_display()

    def _force_update(self) -> None:
        """Force an immediate update."""
        self._updater.immediate_update()

    def _update_image(self) -> None:
        """Generate and display the simulated body image."""
        try:
            # Convert degrees to radians for angles
            rotation_z_rad = math.radians(self._rotation_z)
            rotation_tilt_rad = math.radians(self._rotation_tilt)
            illumination_angle_rad = math.radians(self._illumination_angle)
            phase_angle_rad = math.radians(self._phase_angle)

            # Generate the image
            image_data = create_simulated_body(
                size=(self._size_v, self._size_u),
                semi_major_axis=self._semi_major_axis,
                semi_minor_axis=self._semi_minor_axis,
                semi_c_axis=self._semi_c_axis,
                center=(self._center_v, self._center_u),
                rotation_z=rotation_z_rad,
                rotation_tilt=rotation_tilt_rad,
                illumination_angle=illumination_angle_rad,
                phase_angle=phase_angle_rad,
                crater_fill=self._crater_fill,
                crater_min_radius=self._crater_min_radius,
                crater_max_radius=self._crater_max_radius,
                crater_power_law_exponent=self._crater_power_law_exponent,
                crater_relief_scale=self._crater_relief_scale,
                anti_aliasing=self._anti_aliasing,
            )

            self._current_image = image_data
            self._display_image()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate image:\n{str(e)}')

    def _display_image(self) -> None:
        """Display the current image with optional visual aids."""
        if self._current_image is None:
            return

        # Convert numpy array to QImage
        # Scale to 0-255 range
        img_uint8 = (self._current_image * 255).astype(np.uint8)
        height, width = img_uint8.shape

        # Ensure array is contiguous and create a copy for QImage
        img_uint8 = np.ascontiguousarray(img_uint8.copy())

        # Create grayscale QImage from the copied data
        qimage = QImage(img_uint8.data, width, height, width, QImage.Format.Format_Grayscale8)
        # Create a proper deep copy of the QImage so we can draw on it
        qimage = qimage.copy()

        # Always create a fresh pixmap explicitly (not from QImage) so we can draw on it
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor(0, 0, 0))  # Fill with black background

        # Draw the image onto the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(0, 0, qimage)

        # Add visual aids if enabled
        if self._show_visual_aids:

            # Draw center point
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            center_x = int(self._center_u)
            center_y = int(self._center_v)
            painter.drawEllipse(center_x - 5, center_y - 5, 10, 10)

            # Draw axes (semi-major and semi-minor)
            # Match the clockwise rotation used in sim.py:
            # v_rot = v * cos - u * sin, u_rot = v * sin + u * cos
            rotation_z_rad = math.radians(self._rotation_z)
            cos_rz = math.cos(rotation_z_rad)
            sin_rz = math.sin(rotation_z_rad)

            # Semi-major axis (longer) - points along v direction in local coords
            #   (u_local=0, v_local=1)
            # In sim.py: ellipse uses (v_rot/semi_major)^2, so v is the major axis direction
            # After rotation in sim.py: v_rot = v*cos - u*sin, so for v_local=1:
            #   v_rot = cos, u_rot = sin
            # In screen coords: x = u = sin, y = -v = -cos
            major_len = self._semi_major_axis
            major_end_x = center_x + major_len * sin_rz
            major_end_y = center_y - major_len * cos_rz
            pen.setColor(QColor(0, 255, 0))
            painter.setPen(pen)
            painter.drawLine(center_x, center_y, int(major_end_x), int(major_end_y))

            # Semi-minor axis (shorter) - points along u direction in local coords
            #   (u_local=1, v_local=0)
            # In sim.py: ellipse uses (u_rot/semi_minor)^2, so u is the minor axis direction
            # After rotation in sim.py: v_rot = v*cos - u*sin, so for u_local=1:
            #   v_rot = -sin, u_rot = cos
            # In screen coords: x = u = cos, y = -v = -(-sin) = sin
            minor_len = self._semi_minor_axis
            minor_end_x = center_x + minor_len * cos_rz
            minor_end_y = center_y + minor_len * sin_rz
            pen.setColor(QColor(0, 0, 255))
            painter.setPen(pen)
            painter.drawLine(center_x, center_y, int(minor_end_x), int(minor_end_y))

            # Draw illumination direction arrow
            illum_rad = math.radians(self._illumination_angle)
            arrow_len = min(self._semi_major_axis, self._semi_minor_axis) * 0.8
            arrow_end_x = center_x + arrow_len * math.sin(illum_rad)
            arrow_end_y = center_y - arrow_len * math.cos(illum_rad)
            pen.setColor(QColor(255, 255, 0))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(center_x, center_y, int(arrow_end_x), int(arrow_end_y))

            # Draw arrowhead
            arrow_angle = illum_rad + math.pi
            arrow_size = 10
            arrow_x1 = arrow_end_x + arrow_size * math.cos(arrow_angle - math.pi / 6)
            arrow_y1 = arrow_end_y - arrow_size * math.sin(arrow_angle - math.pi / 6)
            arrow_x2 = arrow_end_x + arrow_size * math.cos(arrow_angle + math.pi / 6)
            arrow_y2 = arrow_end_y - arrow_size * math.sin(arrow_angle + math.pi / 6)
            painter.drawLine(int(arrow_end_x), int(arrow_end_y), int(arrow_x1), int(arrow_y1))
            painter.drawLine(int(arrow_end_x), int(arrow_end_y), int(arrow_x2), int(arrow_y2))

        # Always end the painter (it was started to draw the image)
        painter.end()

        # Always update the base pixmap and refresh display
        self._base_pixmap = pixmap
        # Force immediate display update
        self._update_display()
        # Ensure widgets are repainted
        self._image_label.repaint()
        self._scroll_area.viewport().repaint()

    def _update_display(self) -> None:
        """Update the displayed image with current zoom and pan."""
        if self._base_pixmap is None:
            return

        # Calculate scaled size
        scaled_width = int(self._base_pixmap.width() * self._zoom_factor)
        scaled_height = int(self._base_pixmap.height() * self._zoom_factor)

        transform_mode = (Qt.TransformationMode.FastTransformation
                          if self._zoom_sharp else Qt.TransformationMode.SmoothTransformation)
        scaled_pixmap = self._base_pixmap.scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            transform_mode
        )

        # Set the label size to accommodate the scaled image
        self._image_label.setPixmap(scaled_pixmap)
        self._image_label.resize(scaled_width, scaled_height)

        # Update scroll area to show the panned position
        # The scrollbars will automatically appear when content is larger than viewport
        scrollbar_h = self._scroll_area.horizontalScrollBar()
        scrollbar_v = self._scroll_area.verticalScrollBar()

        # Update scrollbar ranges first (they may have changed due to zoom)
        scrollbar_h.setRange(0, max(0, scaled_width - self._scroll_area.viewport().width()))
        scrollbar_v.setRange(0, max(0, scaled_height - self._scroll_area.viewport().height()))

        # Set scroll position from pan
        # Pan represents the offset of the image from the top-left of the viewport
        scroll_pos_h = int(max(0, min(scrollbar_h.maximum(), self._pan_x)))
        scroll_pos_v = int(max(0, min(scrollbar_v.maximum(), self._pan_y)))
        scrollbar_h.setValue(scroll_pos_h)
        scrollbar_v.setValue(scroll_pos_v)

    def _save_image(self) -> None:
        """Save the current image as PNG."""
        if self._current_image is None:
            QMessageBox.warning(self, 'No Image', 'No image to save.')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Image',
            'simulated_body.png',
            'PNG Images (*.png)'
        )

        if filename:
            try:
                from PIL import Image
                img_uint8 = (self._current_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_uint8, mode='L')
                pil_image.save(filename)
                QMessageBox.information(self, 'Success', f'Image saved to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save image:\n{str(e)}')

    def _get_parameters_dict(self) -> dict[str, Any]:
        """Get current parameters as a dictionary."""
        return {
            'name': 'simulated_body',
            'size_v': self._size_v,
            'size_u': self._size_u,
            'center_v': self._center_v,
            'center_u': self._center_u,
            'semi_major_axis': self._semi_major_axis,
            'semi_minor_axis': self._semi_minor_axis,
            'semi_c_axis': self._semi_c_axis,
            'rotation_z': self._rotation_z,  # degrees
            'rotation_tilt': self._rotation_tilt,  # degrees
            'illumination_angle': self._illumination_angle,  # degrees
            'phase_angle': self._phase_angle,  # degrees
            'crater_fill': self._crater_fill,
            'crater_min_radius': self._crater_min_radius,
            'crater_max_radius': self._crater_max_radius,
            'crater_power_law_exponent': self._crater_power_law_exponent,
            'crater_relief_scale': self._crater_relief_scale,
            'anti_aliasing': self._anti_aliasing,
            # We intentionally don't save the show_visual_aids and zoom_sharp parameters
            # so that the JSON file is more easily used in a simulated model file,
            # but if the user adds these fields to the JSON file, they will be restored
            # on load.
        }

    def _set_parameters_from_dict(self, params: dict[str, Any]):
        """Set parameters from a dictionary."""
        self._size_v = params.get('size_v', self._size_v)
        self._size_u = params.get('size_u', self._size_u)
        self._semi_major_axis = params.get('semi_major_axis', self._semi_major_axis)
        self._semi_minor_axis = params.get('semi_minor_axis', self._semi_minor_axis)
        self._semi_c_axis = params.get('semi_c_axis', self._semi_c_axis)
        self._center_v = params.get('center_v', self._center_v)
        self._center_u = params.get('center_u', self._center_u)
        self._rotation_z = params.get('rotation_z', self._rotation_z)
        self._rotation_tilt = params.get('rotation_tilt', self._rotation_tilt)
        self._illumination_angle = params.get('illumination_angle', self._illumination_angle)
        self._phase_angle = params.get('phase_angle', self._phase_angle)
        self._crater_fill = params.get('crater_fill', self._crater_fill)
        self._crater_min_radius = params.get('crater_min_radius', self._crater_min_radius)
        self._crater_max_radius = params.get('crater_max_radius', self._crater_max_radius)
        self._crater_power_law_exponent = params.get('crater_power_law_exponent', self._crater_power_law_exponent)
        self._crater_relief_scale = params.get('crater_relief_scale', self._crater_relief_scale)
        self._anti_aliasing = params.get('anti_aliasing', self._anti_aliasing)
        self._show_visual_aids = params.get('show_visual_aids', self._show_visual_aids)
        self._zoom_sharp = params.get('zoom_sharp', self._zoom_sharp)

        # Update UI controls
        self._size_v_spin.setValue(self._size_v)
        self._size_u_spin.setValue(self._size_u)
        self._semi_major_spin.setValue(self._semi_major_axis)
        self._semi_minor_spin.setValue(self._semi_minor_axis)
        self._semi_c_spin.setValue(self._semi_c_axis)
        self._center_v_spin.setValue(self._center_v)
        self._center_u_spin.setValue(self._center_u)
        self._rotation_z_spin.setValue(self._rotation_z)
        self._rotation_tilt_spin.setValue(self._rotation_tilt)
        self._illum_angle_spin.setValue(self._illumination_angle)
        self._phase_angle_spin.setValue(self._phase_angle)
        self._crater_fill_spin.setValue(self._crater_fill)
        self._crater_min_radius_spin.setValue(self._crater_min_radius)
        self._crater_max_radius_spin.setValue(self._crater_max_radius)
        self._crater_power_law_exponent_spin.setValue(self._crater_power_law_exponent)
        self._crater_relief_scale_spin.setValue(self._crater_relief_scale)
        self._anti_aliasing_slider.setValue(int(self._anti_aliasing * 1000))
        self._anti_aliasing_label.setText(f'{self._anti_aliasing:.3f}')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._zoom_sharp_check.setChecked(self._zoom_sharp)

        # Trigger update
        self._updater.immediate_update()

    def _save_parameters(self) -> None:
        """Save current parameters to JSON file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            'Save Parameters',
            'simulated_body_params.json',
            'JSON Files (*.json)'
        )

        if filename:
            try:
                params = self._get_parameters_dict()
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=2)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to save parameters:\n{str(e)}')

    def _load_parameters(self) -> None:
        """Load parameters from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            'Load Parameters',
            '',
            'JSON Files (*.json)'
        )

        if filename:
            try:
                with open(filename) as f:
                    params = json.load(f)
                self._set_parameters_from_dict(params)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to load parameters:\n{str(e)}')

    def _set_defaults(self) -> None:
        """Set all parameters to default values."""
        self._size_v = 512
        self._size_u = 512
        self._semi_major_axis = 100.0
        self._semi_minor_axis = 80.0
        self._semi_c_axis = 80.0
        self._center_v = 256.5
        self._center_u = 256.5
        self._rotation_z = 0.0
        self._rotation_tilt = 0.0
        self._illumination_angle = 0.0
        self._phase_angle = 0.0
        self._crater_fill = 0.0
        self._crater_min_radius = 0.05
        self._crater_max_radius = 0.25
        self._crater_power_law_exponent = 3.0
        self._crater_relief_scale = 0.6
        self._anti_aliasing = 0.5
        self._show_visual_aids = True
        self._zoom_sharp = True

    def _reset_defaults(self) -> None:
        """Reset all parameters to default values."""
        self._set_defaults()

        # Update UI controls
        self._size_v_spin.setValue(self._size_v)
        self._size_u_spin.setValue(self._size_u)
        self._semi_major_spin.setValue(self._semi_major_axis)
        self._semi_minor_spin.setValue(self._semi_minor_axis)
        self._semi_c_spin.setValue(self._semi_c_axis)
        self._center_v_spin.setValue(self._center_v)
        self._center_u_spin.setValue(self._center_u)
        self._rotation_z_spin.setValue(self._rotation_z)
        self._rotation_tilt_spin.setValue(self._rotation_tilt)
        self._illum_angle_spin.setValue(self._illumination_angle)
        self._phase_angle_spin.setValue(self._phase_angle)
        self._crater_fill_spin.setValue(self._crater_fill)
        self._crater_min_radius_spin.setValue(self._crater_min_radius)
        self._crater_max_radius_spin.setValue(self._crater_max_radius)
        self._crater_power_law_exponent_spin.setValue(self._crater_power_law_exponent)
        self._crater_relief_scale_spin.setValue(self._crater_relief_scale)
        self._anti_aliasing_slider.setValue(int(self._anti_aliasing * 1000))
        self._anti_aliasing_label.setText(f'{self._anti_aliasing:.3f}')
        self._visual_aids_check.setChecked(self._show_visual_aids)
        self._zoom_sharp_check.setChecked(self._zoom_sharp)

        # Trigger update
        self._updater.immediate_update()


def main() -> None:
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = SimulatedBodyGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
