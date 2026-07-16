"""Background-stars panel for the General tab.

The count, PSF sigma, and magnitude-distribution exponent of the random
background-star field, each exposed as a slider paired with a spin box.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase


class BackgroundStarsMixin(SimEditorBase):
    """Builds and handles the background-stars panel."""

    def _build_background_stars_panel(self, gen_layout: QFormLayout) -> None:
        """Add the background-star slider rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
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
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        psf_sigma_max_label = QLabel('3.0')
        psf_sigma_max_label.setFixedWidth(40)
        psf_sigma_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_stars_psf_sigma_slider = QSlider(Qt.Orientation.Horizontal)
        # 0.1 to 3.0 with 0.01 steps
        self._background_stars_psf_sigma_slider.setRange(1, 300)
        psf_sigma_slider_val = int(self.sim_params['background_stars_psf_sigma'] * 100)
        self._background_stars_psf_sigma_slider.setValue(psf_sigma_slider_val)
        self._background_stars_psf_sigma_slider.valueChanged.connect(
            self._on_background_stars_psf_sigma_slider
        )
        self._background_stars_psf_sigma_spin = QDoubleSpinBox()
        self._background_stars_psf_sigma_spin.setRange(0.1, 3.0)
        self._background_stars_psf_sigma_spin.setDecimals(2)
        self._background_stars_psf_sigma_spin.setSingleStep(0.1)
        psf_sigma_spin_val = self.sim_params['background_stars_psf_sigma']
        self._background_stars_psf_sigma_spin.setValue(psf_sigma_spin_val)
        self._background_stars_psf_sigma_spin.valueChanged.connect(
            self._on_background_stars_psf_sigma_spin
        )
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
        dist_exp_min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dist_exp_max_label = QLabel('4.0')
        dist_exp_max_label.setFixedWidth(40)
        dist_exp_max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._background_stars_dist_exp_slider = QSlider(Qt.Orientation.Horizontal)
        # 1.0 to 4.0 with 0.01 steps
        self._background_stars_dist_exp_slider.setRange(100, 400)
        dist_exp_slider_val = int(self.sim_params['background_stars_distribution_exponent'] * 100)
        self._background_stars_dist_exp_slider.setValue(dist_exp_slider_val)
        self._background_stars_dist_exp_slider.valueChanged.connect(
            self._on_background_stars_dist_exp_slider
        )
        self._background_stars_dist_exp_spin = QDoubleSpinBox()
        self._background_stars_dist_exp_spin.setRange(1.0, 4.0)
        self._background_stars_dist_exp_spin.setDecimals(2)
        self._background_stars_dist_exp_spin.setSingleStep(0.1)
        dist_exp_spin_val = self.sim_params['background_stars_distribution_exponent']
        self._background_stars_dist_exp_spin.setValue(dist_exp_spin_val)
        self._background_stars_dist_exp_spin.valueChanged.connect(
            self._on_background_stars_dist_exp_spin
        )
        dist_exp_row.addWidget(dist_exp_min_label)
        dist_exp_row.addWidget(self._background_stars_dist_exp_slider, stretch=1)
        dist_exp_row.addWidget(dist_exp_max_label)
        dist_exp_row.addWidget(self._background_stars_dist_exp_spin)
        dist_exp_holder = QWidget()
        dist_exp_holder.setLayout(dist_exp_row)
        gen_layout.addRow('Background stars distribution exponent:', dist_exp_holder)

    def _on_background_stars_slider(self, value: int) -> None:
        """Sync the count spin box and update the star count."""
        self._background_stars_spin.blockSignals(True)
        self._background_stars_spin.setValue(value)
        self._background_stars_spin.blockSignals(False)
        self.sim_params['background_stars_num'] = value
        self._updater.request_update()

    def _on_background_stars_spin(self, value: int) -> None:
        """Sync the count slider and update the star count."""
        self._background_stars_slider.blockSignals(True)
        self._background_stars_slider.setValue(value)
        self._background_stars_slider.blockSignals(False)
        self.sim_params['background_stars_num'] = value
        self._updater.request_update()

    def _on_background_stars_psf_sigma_slider(self, value: int) -> None:
        """Sync the PSF-sigma spin box and update the value."""
        psf_sigma_val = value / 100.0
        self._background_stars_psf_sigma_spin.blockSignals(True)
        self._background_stars_psf_sigma_spin.setValue(psf_sigma_val)
        self._background_stars_psf_sigma_spin.blockSignals(False)
        self.sim_params['background_stars_psf_sigma'] = psf_sigma_val
        self._updater.request_update()

    def _on_background_stars_psf_sigma_spin(self, value: float) -> None:
        """Sync the PSF-sigma slider and update the value."""
        slider_val = int(value * 100)
        self._background_stars_psf_sigma_slider.blockSignals(True)
        self._background_stars_psf_sigma_slider.setValue(slider_val)
        self._background_stars_psf_sigma_slider.blockSignals(False)
        self.sim_params['background_stars_psf_sigma'] = value
        self._updater.request_update()

    def _on_background_stars_dist_exp_slider(self, value: int) -> None:
        """Sync the distribution-exponent spin box and update the value."""
        dist_exp_val = value / 100.0
        self._background_stars_dist_exp_spin.blockSignals(True)
        self._background_stars_dist_exp_spin.setValue(dist_exp_val)
        self._background_stars_dist_exp_spin.blockSignals(False)
        self.sim_params['background_stars_distribution_exponent'] = dist_exp_val
        self._updater.request_update()

    def _on_background_stars_dist_exp_spin(self, value: float) -> None:
        """Sync the distribution-exponent slider and update the value."""
        slider_val = int(value * 100)
        self._background_stars_dist_exp_slider.blockSignals(True)
        self._background_stars_dist_exp_slider.setValue(slider_val)
        self._background_stars_dist_exp_slider.blockSignals(False)
        self.sim_params['background_stars_distribution_exponent'] = value
        self._updater.request_update()
