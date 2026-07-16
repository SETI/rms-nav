"""Background-sky panel for the General tab.

The background sky is drawn from a cumulative star-count law
``log10 N(<m) = a + b*m`` per square degree, scaled by the frame's field of view
and a local-density multiplier, plus an optional flat diffuse-sky floor.  The
panel exposes the density multiplier as a slider paired with a spin box, and the
count-law intercept / slope and the diffuse floor as spin boxes.  The values are
stored under the scene's ``sky_counts`` block.
"""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase


class BackgroundStarsMixin(SimEditorBase):
    """Builds and handles the background-sky (sky_counts) panel."""

    def _sky_counts(self) -> dict[str, Any]:
        """The scene's ``sky_counts`` block, created with defaults if absent."""
        sky = self.sim_params.get('sky_counts')
        if not isinstance(sky, dict):
            sky = {'a': -3.1, 'b': 0.34, 'density_factor': 0.0, 'diffuse_e_per_px': 0.0}
            self.sim_params['sky_counts'] = sky
        return sky

    def _build_background_stars_panel(self, gen_layout: QFormLayout) -> None:
        """Add the background-sky control rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        sky = self._sky_counts()

        # Density multiplier: a slider (0 - 200, in tenths) paired with a spin box.
        density_row = QHBoxLayout()
        density_row.setSpacing(4)
        density_row.setContentsMargins(0, 0, 0, 0)
        density_min_label = QLabel('0')
        density_min_label.setFixedWidth(35)
        density_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        density_max_label = QLabel('200')
        density_max_label.setFixedWidth(40)
        density_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._sky_density_slider = QSlider(Qt.Orientation.Horizontal)
        self._sky_density_slider.setRange(0, 2000)
        self._sky_density_slider.setValue(int(float(sky['density_factor']) * 10))
        self._sky_density_slider.valueChanged.connect(self._on_sky_density_slider)
        self._sky_density_spin = QDoubleSpinBox()
        self._sky_density_spin.setRange(0.0, 200.0)
        self._sky_density_spin.setDecimals(1)
        self._sky_density_spin.setSingleStep(0.5)
        self._sky_density_spin.setValue(float(sky['density_factor']))
        self._sky_density_spin.valueChanged.connect(self._on_sky_density_spin)
        density_row.addWidget(density_min_label)
        density_row.addWidget(self._sky_density_slider, stretch=1)
        density_row.addWidget(density_max_label)
        density_row.addWidget(self._sky_density_spin)
        density_holder = QWidget()
        density_holder.setLayout(density_row)
        gen_layout.addRow('Sky density factor:', density_holder)

        # Count-law intercept a and slope b, plus the diffuse floor, as spin boxes.
        self._sky_a_spin = QDoubleSpinBox()
        self._sky_a_spin.setRange(-8.0, 2.0)
        self._sky_a_spin.setDecimals(2)
        self._sky_a_spin.setSingleStep(0.1)
        self._sky_a_spin.setValue(float(sky['a']))
        self._sky_a_spin.valueChanged.connect(self._on_sky_a_spin)
        gen_layout.addRow('Sky count law a:', self._sky_a_spin)

        self._sky_b_spin = QDoubleSpinBox()
        self._sky_b_spin.setRange(0.0, 1.0)
        self._sky_b_spin.setDecimals(3)
        self._sky_b_spin.setSingleStep(0.01)
        self._sky_b_spin.setValue(float(sky['b']))
        self._sky_b_spin.valueChanged.connect(self._on_sky_b_spin)
        gen_layout.addRow('Sky count law b:', self._sky_b_spin)

        self._sky_diffuse_spin = QDoubleSpinBox()
        self._sky_diffuse_spin.setRange(0.0, 1000.0)
        self._sky_diffuse_spin.setDecimals(2)
        self._sky_diffuse_spin.setSingleStep(1.0)
        self._sky_diffuse_spin.setValue(float(sky.get('diffuse_e_per_px', 0.0)))
        self._sky_diffuse_spin.valueChanged.connect(self._on_sky_diffuse_spin)
        gen_layout.addRow('Sky diffuse floor (e-/px):', self._sky_diffuse_spin)

    def _on_sky_density_slider(self, value: int) -> None:
        """Sync the density spin box and store the value."""
        density = value / 10.0
        self._sky_density_spin.blockSignals(True)
        self._sky_density_spin.setValue(density)
        self._sky_density_spin.blockSignals(False)
        self._sky_counts()['density_factor'] = density
        self._updater.request_update()

    def _on_sky_density_spin(self, value: float) -> None:
        """Sync the density slider and store the value."""
        self._sky_density_slider.blockSignals(True)
        self._sky_density_slider.setValue(int(value * 10))
        self._sky_density_slider.blockSignals(False)
        self._sky_counts()['density_factor'] = value
        self._updater.request_update()

    def _on_sky_a_spin(self, value: float) -> None:
        """Store the count-law intercept."""
        self._sky_counts()['a'] = value
        self._updater.request_update()

    def _on_sky_b_spin(self, value: float) -> None:
        """Store the count-law slope."""
        self._sky_counts()['b'] = value
        self._updater.request_update()

    def _on_sky_diffuse_spin(self, value: float) -> None:
        """Store the diffuse-sky floor."""
        self._sky_counts()['diffuse_e_per_px'] = value
        self._updater.request_update()
