"""Artifacts tab for the simulated-image editor.

Authors the scene-level ``artifacts`` switch and the ``detector`` override
block.  ``instrument_defaults`` opts the whole scene into the emulated camera's
physical signal chain (its catalog PSF, distortion residual, dark / hot / banding
/ bias-structure noise); leaving it unchecked keeps those keys absent so the
scene renders the self-consistency floor.  The detector group is a checkable
override for the electron-chain gain state, the detector model, and the exposure
the well fraction references; unchecking it removes the ``detector`` key so the
per-instrument catalog defaults apply.  This tab grows in later phases: the
per-mode loss-incidence rows attach as further sections under the same discipline.
"""

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase

_DETECTOR_MODELS: list[str] = ['ccd', 'vidicon']


class ArtifactsTabMixin(SimEditorBase):
    """Builds and handles the Artifacts tab."""

    def _build_artifacts_tab(self) -> QWidget:
        """Build the scrollable Artifacts tab and return its container widget."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(self._build_artifacts_switch_group())
        layout.addWidget(self._build_detector_group())
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self._artifacts_tab = scroll
        return scroll

    # ---- Instrument-defaults switch ----

    def _build_artifacts_switch_group(self) -> QGroupBox:
        """Build the physical-signal-chain opt-in group."""
        group = QGroupBox('Instrument physical chain')
        form = QFormLayout(group)
        self._instrument_defaults_check = QCheckBox()
        self._instrument_defaults_check.setToolTip(
            "Turn on the emulated camera's physical signal chain at catalog "
            'values (PSF, distortion residual, dark / hot / banding / bias noise).'
        )
        self._instrument_defaults_check.setChecked(self._instrument_defaults_on())
        form.addRow('Instrument defaults:', self._instrument_defaults_check)
        self._instrument_defaults_check.toggled.connect(self._on_instrument_defaults)
        return group

    def _instrument_defaults_on(self) -> bool:
        """Whether the scene opts into the instrument physical chain."""
        artifacts = self.sim_params.get('artifacts')
        return isinstance(artifacts, dict) and bool(artifacts.get('instrument_defaults', False))

    def _on_instrument_defaults(self, checked: bool) -> None:
        """Insert or remove the artifacts block."""
        if self._syncing:
            return
        if checked:
            self.sim_params['artifacts'] = {'instrument_defaults': True}
        else:
            self.sim_params.pop('artifacts', None)
        self._updater.request_update()

    # ---- Detector override group ----

    def _build_detector_group(self) -> QGroupBox:
        """Build the detector-override group."""
        group = QGroupBox('Detector override')
        group.setCheckable(True)
        form = QFormLayout(group)
        detector = self.sim_params.get('detector')
        block = detector if isinstance(detector, dict) else {}

        self._detector_gain_state_spin = QSpinBox()
        self._detector_gain_state_spin.setRange(0, 3)
        self._detector_gain_state_spin.setValue(int(block.get('gain_state', 0)))
        self._detector_gain_state_spin.setToolTip(
            'Electron-chain gain state; must be catalogued for the instrument.'
        )
        form.addRow('Gain state:', self._detector_gain_state_spin)

        self._detector_model_combo = QComboBox()
        self._detector_model_combo.addItems(_DETECTOR_MODELS)
        model_index = self._detector_model_combo.findText(str(block.get('detector_model', 'ccd')))
        if model_index >= 0:
            self._detector_model_combo.setCurrentIndex(model_index)
        self._detector_model_combo.setToolTip('ccd electron chain or vidicon DN chain.')
        form.addRow('Detector model:', self._detector_model_combo)

        self._detector_exposure_ref_spin = QDoubleSpinBox()
        self._detector_exposure_ref_spin.setRange(0.001, 10000.0)
        self._detector_exposure_ref_spin.setDecimals(4)
        self._detector_exposure_ref_spin.setSingleStep(0.1)
        self._detector_exposure_ref_spin.setValue(float(block.get('exposure_ref_sec', 1.0)))
        self._detector_exposure_ref_spin.setToolTip(
            'Exposure (sec) the signal full-scale fraction references.'
        )
        form.addRow('Exposure ref (sec):', self._detector_exposure_ref_spin)

        self._detector_group = group
        group.setChecked(isinstance(detector, dict))
        self._detector_gain_state_spin.valueChanged.connect(self._on_detector_value_int)
        self._detector_exposure_ref_spin.valueChanged.connect(self._on_detector_value_float)
        self._detector_model_combo.currentTextChanged.connect(self._on_detector_text)
        group.toggled.connect(self._on_detector_group_toggled)
        return group

    def _detector_block_from_widgets(self) -> dict[str, Any]:
        """Assemble the detector block the widgets describe."""
        return {
            'gain_state': int(self._detector_gain_state_spin.value()),
            'detector_model': self._detector_model_combo.currentText() or 'ccd',
            'exposure_ref_sec': float(self._detector_exposure_ref_spin.value()),
        }

    def _write_detector(self) -> None:
        """Write the detector block when the group is enabled."""
        if self._syncing:
            return
        if self._detector_group.isChecked():
            self.sim_params['detector'] = self._detector_block_from_widgets()
            self._updater.request_update()

    def _on_detector_group_toggled(self, on: bool) -> None:
        """Insert or remove the detector block."""
        if self._syncing:
            return
        if on:
            self.sim_params['detector'] = self._detector_block_from_widgets()
        else:
            self.sim_params.pop('detector', None)
        self._updater.request_update()

    def _on_detector_value_int(self, _value: int) -> None:
        """Rewrite the detector block on an integer spin edit."""
        self._write_detector()

    def _on_detector_value_float(self, _value: float) -> None:
        """Rewrite the detector block on a float spin edit."""
        self._write_detector()

    def _on_detector_text(self, _text: str) -> None:
        """Rewrite the detector block on a model-combo edit."""
        self._write_detector()

    # ---- Scene-load sync ----

    def _sync_artifacts_from_params(self) -> None:
        """Rebuild every Artifacts-tab widget from the current sim_params."""
        self._syncing = True
        try:
            self._instrument_defaults_check.setChecked(self._instrument_defaults_on())
            detector = self.sim_params.get('detector')
            block = detector if isinstance(detector, dict) else {}
            self._detector_gain_state_spin.setValue(int(block.get('gain_state', 0)))
            model_index = self._detector_model_combo.findText(
                str(block.get('detector_model', 'ccd'))
            )
            if model_index >= 0:
                self._detector_model_combo.setCurrentIndex(model_index)
            self._detector_exposure_ref_spin.setValue(float(block.get('exposure_ref_sec', 1.0)))
            self._detector_group.setChecked(isinstance(detector, dict))
        finally:
            self._syncing = False
