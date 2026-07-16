"""Star tab builder and change handlers.

Builds the per-star editing tab (position, magnitude, spectral class, PSF sigma,
smear vector, catalog name, and the V / U PSF-window sizes) and owns the
handlers that write star fields back into the data model.
"""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase


class StarTabMixin(SimEditorBase):
    """Builds and handles the per-star editing tab."""

    def _build_star_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the star at ``idx``."""
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
        v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'v', v))
        fl.addRow('V:', v_spin)
        u_spin = QDoubleSpinBox()
        u_spin.setRange(-10000.0, 20000.0)
        u_spin.setDecimals(1)
        u_spin.setValue(p.get('u', 0.0))
        u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'u', v))
        fl.addRow('U:', u_spin)
        # Keep references so drag updates can sync the UI
        w.v_spin = v_spin  # type: ignore[attr-defined]
        w.u_spin = u_spin  # type: ignore[attr-defined]

        vmag = QDoubleSpinBox()
        vmag.setRange(-10.0, 30.0)
        vmag.setDecimals(2)
        vmag.setValue(p.get('vmag', 3.0))
        vmag.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'vmag', v))
        fl.addRow('Magnitude (V):', vmag)
        sclass = QLineEdit(p.get('spectral_class', 'G2'))
        sclass.textChanged.connect(lambda t, i=idx: self._on_star_field(i, 'spectral_class', t))
        fl.addRow('Spectral class:', sclass)
        psf = QDoubleSpinBox()
        psf.setRange(0.1, 20.0)
        psf.setDecimals(2)
        psf.setValue(p.get('psf_sigma', 3.0))
        psf.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'psf_sigma', v))
        fl.addRow('PSF sigma:', psf)

        move_v_spin = QDoubleSpinBox()
        move_v_spin.setRange(-200.0, 200.0)
        move_v_spin.setDecimals(2)
        move_v_spin.setValue(float(p.get('move_v', 0.0)))
        move_v_spin.setToolTip('Star smear vector V (px) during the exposure.')
        move_v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'move_v', v))
        fl.addRow('Smear V (px):', move_v_spin)
        move_u_spin = QDoubleSpinBox()
        move_u_spin.setRange(-200.0, 200.0)
        move_u_spin.setDecimals(2)
        move_u_spin.setValue(float(p.get('move_u', 0.0)))
        move_u_spin.setToolTip('Star smear vector U (px) during the exposure.')
        move_u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_field(i, 'move_u', v))
        fl.addRow('Smear U (px):', move_u_spin)
        catalog_edit = QLineEdit(str(p.get('catalog_name', 'SIM')))
        catalog_edit.setToolTip('Source-catalog label carried on the star.')
        catalog_edit.textChanged.connect(lambda t, i=idx: self._on_star_field(i, 'catalog_name', t))
        fl.addRow('Catalog name:', catalog_edit)

        # PSF size V slider with min/max labels and spinbox
        # Map slider positions 0-11 to odd values 1, 3, 5, ..., 23
        psf_size_v_row = QHBoxLayout()
        psf_size_v_row.setSpacing(4)
        psf_size_v_row.setContentsMargins(0, 0, 0, 0)
        psf_size_v_min_label = QLabel('1')
        psf_size_v_min_label.setFixedWidth(35)
        psf_size_v_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_v_max_label = QLabel('23')
        psf_size_v_max_label.setFixedWidth(40)
        psf_size_v_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
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
        psf_size_v_spin.valueChanged.connect(lambda v, i=idx: self._on_star_psf_size_v_spin(i, v))
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
        psf_size_u_min_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        psf_size_u_max_label = QLabel('23')
        psf_size_u_max_label.setFixedWidth(40)
        psf_size_u_max_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
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
        psf_size_u_spin.valueChanged.connect(lambda v, i=idx: self._on_star_psf_size_u_spin(i, v))
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
    def _on_star_field(self, idx: int, key: str, value: Any) -> None:
        """Write a scalar star field into the data model and re-render."""
        if 0 <= idx < len(self.sim_params['stars']):
            self.sim_params['stars'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_star_name(self, idx: int, text: str) -> None:
        """Rename a star and refresh the tab titles."""
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
