"""Ring tab builder and change handlers.

Builds the per-ring editing tab (feature type, centre, physical range,
shading, and the inner / outer mode-1 edge parameters with enable checkboxes)
and owns the handlers that write ring fields back into the data model.
"""

from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from spindoctor.cli.sim_editor.base import SimEditorBase


class RingTabMixin(SimEditorBase):
    """Builds and handles the per-ring editing tab."""

    def _build_ring_tab(self, idx: int) -> QWidget:
        """Build the editing tab widget for the ring at ``idx``."""
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
        center_v.valueChanged.connect(lambda v, i=idx: self._on_ring_field(i, 'center_v', v))
        fl.addRow('Center V:', center_v)
        center_u = QDoubleSpinBox()
        center_u.setRange(-10000.0, 20000.0)
        center_u.setDecimals(1)
        center_u.setValue(p.get('center_u', 0.0))
        center_u.valueChanged.connect(lambda v, i=idx: self._on_ring_field(i, 'center_u', v))
        fl.addRow('Center U:', center_u)
        # Keep references so drag updates can sync the UI
        w.center_v_spin = center_v  # type: ignore[attr-defined]
        w.center_u_spin = center_u  # type: ignore[attr-defined]

        # Physical range (km): optional-key discipline -- absent unless set.
        # Required on every ring of an spk_error scene; range_km is the only
        # depth-ordering key (a ring without one is drawn behind everything,
        # and overlapping a ranged object without one is a render error).
        has_range_km = p.get('range_km') is not None
        range_km_check = QCheckBox('Set physical range (km)')
        range_km_check.setChecked(has_range_km)
        range_km_check.setToolTip(
            'Write a physical range_km on this ring (depth ordering and '
            'spk_error parallax); unchecked leaves the key absent.'
        )
        range_km_spin = QDoubleSpinBox()
        range_km_spin.setRange(0.01, 1.0e12)
        range_km_spin.setDecimals(1)
        range_km_spin.setValue(float(p.get('range_km', 1.0e6)))
        range_km_spin.setEnabled(has_range_km)
        range_km_check.clicked.connect(
            lambda checked, i=idx, spin=range_km_spin: self._on_ring_range_km_enabled(
                i, checked, spin
            )
        )
        range_km_spin.valueChanged.connect(lambda v, i=idx: self._on_ring_range_km_value(i, v))
        fl.addRow(range_km_check, range_km_spin)
        w.range_km_check = range_km_check  # type: ignore[attr-defined]
        w.range_km_spin = range_km_spin  # type: ignore[attr-defined]

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
            lambda checked, i=idx, cb=inner_checkbox: self._on_ring_inner_enabled(i, checked, cb)
        )
        fl.addRow(inner_checkbox, QLabel(''))

        inner_label = QLabel('<b>Inner Edge (Mode 1)</b>')
        fl.addRow(inner_label, QLabel(''))

        inner_a = QDoubleSpinBox()
        inner_a.setRange(1.0, 10000.0)
        inner_a.setDecimals(1)
        inner_a.setValue(inner_mode1.get('a', 100.0))
        inner_a.setEnabled(has_inner)
        inner_a.valueChanged.connect(lambda v, i=idx: self._on_ring_inner_mode1(i, 'a', v))
        fl.addRow('Inner a:', inner_a)
        inner_ae = QDoubleSpinBox()
        inner_ae.setRange(0.0, 1000.0)
        inner_ae.setDecimals(2)
        inner_ae.setValue(inner_mode1.get('ae', 0.0))
        inner_ae.setEnabled(has_inner)
        inner_ae.valueChanged.connect(lambda v, i=idx: self._on_ring_inner_mode1(i, 'ae', v))
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
        inner_rms.valueChanged.connect(lambda v, i=idx: self._on_ring_inner_mode1(i, 'rms', v))
        fl.addRow('Inner rms:', inner_rms)

        # Store references for enabling/disabling
        w.inner_checkbox = inner_checkbox  # type: ignore[attr-defined]
        w.inner_controls = [  # type: ignore[attr-defined]
            inner_label,
            inner_a,
            inner_ae,
            inner_long_peri,
            inner_rate_peri,
            inner_rms,
        ]
        # Store individual spinbox references for reading values
        w.inner_a = inner_a  # type: ignore[attr-defined]
        w.inner_ae = inner_ae  # type: ignore[attr-defined]
        w.inner_long_peri = inner_long_peri  # type: ignore[attr-defined]
        w.inner_rate_peri = inner_rate_peri  # type: ignore[attr-defined]
        w.inner_rms = inner_rms  # type: ignore[attr-defined]

        # Outer edge checkbox and mode 1 parameters
        outer_data = p.get('outer_data', [])
        has_outer = len(outer_data) > 0 and any(m.get('mode') == 1 for m in outer_data)
        outer_mode1: dict[str, Any] = next((m for m in outer_data if m.get('mode') == 1), {})

        outer_checkbox = QCheckBox('Enable Outer Edge')
        outer_checkbox.setChecked(has_outer)
        outer_checkbox.clicked.connect(
            lambda checked, i=idx, cb=outer_checkbox: self._on_ring_outer_enabled(i, checked, cb)
        )
        fl.addRow(outer_checkbox, QLabel(''))

        outer_label = QLabel('<b>Outer Edge (Mode 1)</b>')
        fl.addRow(outer_label, QLabel(''))

        outer_a = QDoubleSpinBox()
        outer_a.setRange(1.0, 10000.0)
        outer_a.setDecimals(1)
        outer_a.setValue(outer_mode1.get('a', 120.0))
        outer_a.setEnabled(has_outer)
        outer_a.valueChanged.connect(lambda v, i=idx: self._on_ring_outer_mode1(i, 'a', v))
        fl.addRow('Outer a:', outer_a)
        outer_ae = QDoubleSpinBox()
        outer_ae.setRange(0.0, 1000.0)
        outer_ae.setDecimals(2)
        outer_ae.setValue(outer_mode1.get('ae', 0.0))
        outer_ae.setEnabled(has_outer)
        outer_ae.valueChanged.connect(lambda v, i=idx: self._on_ring_outer_mode1(i, 'ae', v))
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
        outer_rms.valueChanged.connect(lambda v, i=idx: self._on_ring_outer_mode1(i, 'rms', v))
        fl.addRow('Outer rms:', outer_rms)

        # Store references for enabling/disabling
        w.outer_checkbox = outer_checkbox  # type: ignore[attr-defined]
        w.outer_controls = [  # type: ignore[attr-defined]
            outer_label,
            outer_a,
            outer_ae,
            outer_long_peri,
            outer_rate_peri,
            outer_rms,
        ]
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

    # ---- Field handlers ----
    def _on_ring_field(self, idx: int, key: str, value: Any) -> None:
        """Write a scalar ring field into the data model and re-render."""
        if 0 <= idx < len(self.sim_params['rings']):
            self.sim_params['rings'][idx][key] = (
                float(value) if isinstance(value, (int, float)) else value
            )
            self._updater.request_update()

    def _on_ring_range_km_enabled(self, idx: int, enabled: bool, spin: QDoubleSpinBox) -> None:
        """Insert or remove the ring's optional physical range_km key."""
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]
            if enabled:
                ring['range_km'] = float(spin.value())
            else:
                ring.pop('range_km', None)
            spin.setEnabled(enabled)
            self._updater.request_update()

    def _on_ring_range_km_value(self, idx: int, value: float) -> None:
        """Update the ring's physical range_km when the key is enabled."""
        if 0 <= idx < len(self.sim_params['rings']):
            ring = self.sim_params['rings'][idx]
            if 'range_km' in ring:
                ring['range_km'] = float(value)
                self._updater.request_update()

    def _on_ring_name(self, idx: int, text: str) -> None:
        """Rename a ring and refresh the tab titles."""
        if 0 <= idx < len(self.sim_params['rings']):
            self.sim_params['rings'][idx]['name'] = text
            self._update_tab_titles()
            self._updater.request_update()

    def _get_or_create_mode1(
        self, data_list: list[dict[str, Any]], default_a: float
    ) -> dict[str, Any]:
        """Find or create mode 1 dictionary in the given data list.

        Parameters:
            data_list: List of mode dictionaries (inner_data or outer_data).
            default_a: Default 'a' value to use when creating new mode 1.

        Returns:
            Mode 1 dictionary (existing or newly created).
        """
        mode1 = next((m for m in data_list if m.get('mode') == 1), None)
        if mode1 is None:
            mode1 = {
                'mode': 1,
                'a': default_a,
                'rms': 1.0,
                'ae': 0.0,
                'long_peri': 0.0,
                'rate_peri': 0.0,
            }
            data_list.append(mode1)
        return mode1

    def _on_ring_inner_mode1(self, idx: int, key: str, value: float) -> None:
        """Update a mode-1 inner-edge parameter."""
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
        """Update a mode-1 outer-edge parameter."""
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
                    outer_checkbox = tab_widget.outer_checkbox  # type: ignore[attr-defined]
                    has_outer_checked = outer_checkbox.isChecked()
                else:
                    # Fallback to data model if widget not found
                    outer_data = ring.get('outer_data', [])
                    has_outer_checked = len(outer_data) > 0 and any(
                        m.get('mode') == 1 for m in outer_data
                    )
            else:
                # Fallback to data model if tab not found
                outer_data = ring.get('outer_data', [])
                has_outer_checked = len(outer_data) > 0 and any(
                    m.get('mode') == 1 for m in outer_data
                )

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
                        mode1['a'] = float(tab_widget.inner_a.value())  # type: ignore[attr-defined]
                        mode1['ae'] = float(tab_widget.inner_ae.value())  # type: ignore[attr-defined]
                        mode1['long_peri'] = float(tab_widget.inner_long_peri.value())  # type: ignore[attr-defined]
                        mode1['rate_peri'] = float(tab_widget.inner_rate_peri.value())  # type: ignore[attr-defined]
                        mode1['rms'] = float(tab_widget.inner_rms.value())  # type: ignore[attr-defined]
            else:
                # Disable inner edge - remove inner_data
                ring.pop('inner_data', None)

            # Update UI controls
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    for control in tab_widget.inner_controls:  # type: ignore[attr-defined]
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
                    inner_checkbox = tab_widget.inner_checkbox  # type: ignore[attr-defined]
                    has_inner_checked = inner_checkbox.isChecked()
                else:
                    # Fallback to data model if widget not found
                    inner_data = ring.get('inner_data', [])
                    has_inner_checked = len(inner_data) > 0 and any(
                        m.get('mode') == 1 for m in inner_data
                    )
            else:
                # Fallback to data model if tab not found
                inner_data = ring.get('inner_data', [])
                has_inner_checked = len(inner_data) > 0 and any(
                    m.get('mode') == 1 for m in inner_data
                )

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
                        mode1['a'] = float(tab_widget.outer_a.value())  # type: ignore[attr-defined]
                        mode1['ae'] = float(tab_widget.outer_ae.value())  # type: ignore[attr-defined]
                        mode1['long_peri'] = float(tab_widget.outer_long_peri.value())  # type: ignore[attr-defined]
                        mode1['rate_peri'] = float(tab_widget.outer_rate_peri.value())  # type: ignore[attr-defined]
                        mode1['rms'] = float(tab_widget.outer_rms.value())  # type: ignore[attr-defined]
            else:
                # Disable outer edge - remove outer_data
                ring.pop('outer_data', None)

            # Update UI controls
            if tab_idx is not None:
                tab_widget = self._tabs.widget(tab_idx)
                if tab_widget is not None:
                    for control in tab_widget.outer_controls:  # type: ignore[attr-defined]
                        control.setEnabled(enabled)

            self._updater.request_update()
