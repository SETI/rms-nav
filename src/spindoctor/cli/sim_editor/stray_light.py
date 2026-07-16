"""Stray-light panel for the General tab.

An additive low-frequency gradient the navigator's BANDPASS_DOG filter is meant
to suppress.  Writes the ``sim_params['optics']['stray_light']`` block;
amplitude 0 (default) means off.
"""

from typing import Any

from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout

from spindoctor.cli.sim_editor.base import SimEditorBase


class StrayLightMixin(SimEditorBase):
    """Builds and handles the stray-light panel."""

    def _build_stray_panel(self, gen_layout: QFormLayout) -> None:
        """Add the stray-light rows to the General tab layout.

        Parameters:
            gen_layout: The General tab's form layout.
        """
        self._stray_amplitude_spin = QDoubleSpinBox()
        self._stray_amplitude_spin.setRange(0.0, 1.0)
        self._stray_amplitude_spin.setDecimals(3)
        self._stray_amplitude_spin.setSingleStep(0.01)
        self._stray_amplitude_spin.setValue(float(self._stray_value('amplitude', 0.0)))
        self._stray_amplitude_spin.setToolTip('Stray-light amplitude (0 = off).')
        self._stray_amplitude_spin.valueChanged.connect(self._on_stray_amplitude)
        gen_layout.addRow('Stray light amplitude:', self._stray_amplitude_spin)

        self._stray_direction_spin = QDoubleSpinBox()
        self._stray_direction_spin.setRange(0.0, 360.0)
        self._stray_direction_spin.setDecimals(1)
        self._stray_direction_spin.setWrapping(True)
        self._stray_direction_spin.setValue(float(self._stray_value('direction_deg', 0.0)))
        self._stray_direction_spin.setToolTip('Gradient direction for the linear model, degrees.')
        self._stray_direction_spin.valueChanged.connect(self._on_stray_direction)
        gen_layout.addRow('Stray light direction (deg):', self._stray_direction_spin)

        self._stray_model_combo = QComboBox()
        self._stray_model_combo.addItems(['linear', 'radial'])
        stray_model = str(self._stray_value('model', 'linear'))
        stray_model_index = self._stray_model_combo.findText(stray_model)
        if stray_model_index >= 0:
            self._stray_model_combo.setCurrentIndex(stray_model_index)
        self._stray_model_combo.setToolTip('linear ramp or radial bump.')
        self._stray_model_combo.currentTextChanged.connect(self._on_stray_model)
        gen_layout.addRow('Stray light model:', self._stray_model_combo)

        self._stray_center_v_spin = QDoubleSpinBox()
        self._stray_center_v_spin.setRange(-10000.0, 20000.0)
        self._stray_center_v_spin.setDecimals(1)
        self._stray_center_v_spin.setValue(float(self._stray_value('center_v', 0.0)))
        self._stray_center_v_spin.setToolTip('Radial-model bump centre V (0 = frame centre).')
        self._stray_center_v_spin.valueChanged.connect(self._on_stray_center_v)
        gen_layout.addRow('Stray light center V:', self._stray_center_v_spin)

        self._stray_center_u_spin = QDoubleSpinBox()
        self._stray_center_u_spin.setRange(-10000.0, 20000.0)
        self._stray_center_u_spin.setDecimals(1)
        self._stray_center_u_spin.setValue(float(self._stray_value('center_u', 0.0)))
        self._stray_center_u_spin.setToolTip('Radial-model bump centre U (0 = frame centre).')
        self._stray_center_u_spin.valueChanged.connect(self._on_stray_center_u)
        gen_layout.addRow('Stray light center U:', self._stray_center_u_spin)

    def _stray_value(self, key: str, default: Any) -> Any:
        """Read a value from the optics.stray_light block, or a default."""
        optics = self.sim_params.get('optics')
        stray = optics.get('stray_light') if isinstance(optics, dict) else None
        if isinstance(stray, dict) and key in stray:
            return stray[key]
        return default

    def _set_stray(self, key: str, value: Any) -> None:
        """Write a value into the optics.stray_light block and re-render."""
        if self._syncing:
            return
        optics = self.sim_params.setdefault('optics', {})
        if not isinstance(optics, dict):
            optics = {}
            self.sim_params['optics'] = optics
        stray = optics.setdefault('stray_light', {})
        if not isinstance(stray, dict):
            stray = {}
            optics['stray_light'] = stray
        stray[key] = value
        self._updater.request_update()

    def _on_stray_amplitude(self, value: float) -> None:
        """Set the stray-light amplitude."""
        self._set_stray('amplitude', float(value))

    def _on_stray_direction(self, value: float) -> None:
        """Set the linear-model gradient direction."""
        self._set_stray('direction_deg', float(value))

    def _on_stray_model(self, text: str) -> None:
        """Set the stray-light model (linear or radial)."""
        self._set_stray('model', text or 'linear')

    def _on_stray_center(self, key: str, value: float) -> None:
        """Set or omit a radial-model bump centre coordinate."""
        if self._syncing:
            return
        # 0 means "use the frame centre": omit the key so the renderer defaults it.
        if value == 0.0:
            optics = self.sim_params.get('optics')
            stray = optics.get('stray_light') if isinstance(optics, dict) else None
            if isinstance(stray, dict):
                stray.pop(key, None)
            self._updater.request_update()
        else:
            self._set_stray(key, float(value))

    def _on_stray_center_v(self, value: float) -> None:
        """Set or omit the radial-model bump centre V."""
        self._on_stray_center('center_v', value)

    def _on_stray_center_u(self, value: float) -> None:
        """Set or omit the radial-model bump centre U."""
        self._on_stray_center('center_u', value)
