"""Per-tab widget-state tests for the ``sd_create_simulated_image`` editor.

These exercise the editor's data model directly through its widgets, with no
file I/O: every optics / detector / artifacts / star control follows the
absent-key discipline (an enable checkbox inserts its block; unchecked leaves
the key absent), edits write only their own key, and GUI-added objects carry
canonical value types.  The save / load / round-trip acceptance tests live in
``test_sim_editor_round_trip``.

Qt runs headless (offscreen platform).
"""

import os
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(
        f'PyQt6 QApplication init failed: {exc}',
        allow_module_level=True,
    )

from spindoctor.cli.sim_editor import CreateSimulatedImageModel


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image editor model."""
    return CreateSimulatedImageModel()


def test_gui_added_star_psf_size_is_list(model: Any) -> None:
    """A GUI-added star stores psf_size as a list, not a tuple."""
    model._add_star_tab()
    psf_size = model.sim_params['stars'][0]['psf_size']
    assert isinstance(psf_size, list)
    assert psf_size == [11, 11]


def test_star_psf_size_edit_stays_a_list(model: Any) -> None:
    """Editing a star's PSF-window size keeps psf_size a list."""
    model._add_star_tab()
    model._on_star_psf_size_v_spin(0, 7)
    psf_size = model.sim_params['stars'][0]['psf_size']
    assert isinstance(psf_size, list)
    assert psf_size[0] == 7


def test_default_scene_has_no_optics_block(model: Any) -> None:
    """A fresh scene carries no optics block (the stage-activation floor)."""
    assert 'optics' not in model.sim_params


def test_psf_enable_inserts_block(model: Any) -> None:
    """Enabling the PSF group inserts an explicit kernel block."""
    model._psf_optics_group.setChecked(True)
    assert 'psf' in model.sim_params['optics']


def test_psf_disable_removes_optics_key(model: Any) -> None:
    """Disabling the only optics sub-block drops the optics key entirely."""
    model._psf_optics_group.setChecked(True)
    model._psf_optics_group.setChecked(False)
    assert 'optics' not in model.sim_params


def test_match_navigator_writes_canonical_form(model: Any) -> None:
    """The match-navigator checkbox writes the exclusive canonical PSF form."""
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}


def test_distortion_center_keys_absent_unless_enabled(model: Any) -> None:
    """The distortion block omits the optical-centre keys until enabled."""
    model._distortion_group.setChecked(True)
    block = model.sim_params['optics']['distortion']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_distortion_center_zero_is_authorable(model: Any) -> None:
    """An explicit 0.0 optical centre survives (no 0.0-to-absent flip)."""
    model._distortion_group.setChecked(True)
    model._distortion_center_check.setChecked(True)
    model._distortion_center_v_spin.setValue(0.0)
    model._distortion_center_u_spin.setValue(0.0)
    block = model.sim_params['optics']['distortion']
    assert block['center_v'] == 0.0
    assert block['center_u'] == 0.0


def test_distortion_center_uncheck_drops_both_keys(model: Any) -> None:
    """Unchecking the optical-centre enable removes both centre keys."""
    model._distortion_group.setChecked(True)
    model._distortion_center_check.setChecked(True)
    model._distortion_center_v_spin.setValue(40.0)
    model._distortion_center_u_spin.setValue(50.0)
    model._distortion_center_check.setChecked(False)
    block = model.sim_params['optics']['distortion']
    assert 'center_v' not in block
    assert 'center_u' not in block


def test_ring_range_km_unchecked_leaves_key_absent(model: Any) -> None:
    """Disabling the ring physical-range control removes the key."""
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    ring_tab = model._tabs.widget(tab_idx)
    ring_tab.range_km_check.click()
    assert 'range_km' in model.sim_params['rings'][0]
    ring_tab.range_km_check.click()
    assert 'range_km' not in model.sim_params['rings'][0]


def test_match_navigator_disables_kernel_spins(model: Any) -> None:
    """Matching the navigator disables the explicit-kernel spins."""
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)
    assert model._psf_sigma_v_spin.isEnabled() is False


def test_smear_row_edit_updates_params(model: Any) -> None:
    """Editing a smear row's drift updates the smear list in sim_params."""
    model._smear_group.setChecked(True)
    model._on_add_smear_clicked()
    model._smear_rows[0].dv_spin.setValue(3.0)
    assert model.sim_params['optics']['smear'][0]['dv_px'] == 3.0


def test_ghost_enable_inserts_list(model: Any) -> None:
    """Enabling the ghost group with a row inserts a ghost list."""
    model._ghosts_group.setChecked(True)
    model._on_add_ghost_clicked()
    assert len(model.sim_params['optics']['ghosts']) == 1


def test_distortion_disable_removes_block(model: Any) -> None:
    """Disabling the distortion group removes its block."""
    model._distortion_group.setChecked(True)
    assert 'distortion' in model.sim_params['optics']
    model._distortion_group.setChecked(False)
    assert 'optics' not in model.sim_params


def test_oversample_checkbox_toggles_key(model: Any) -> None:
    """The oversample checkbox inserts and removes the top-level key."""
    model._oversample_check.setChecked(True)
    model._oversample_spin.setValue(4)
    assert model.sim_params['oversample'] == 4
    model._oversample_check.setChecked(False)
    assert 'oversample' not in model.sim_params


def test_spk_error_toggle_inserts_and_removes(model: Any) -> None:
    """The spk_error group inserts and removes the block with its three keys."""
    model._spk_error_group.setChecked(True)
    assert set(model.sim_params['spk_error']) == {'dv_px', 'du_px', 'reference_range_km'}
    model._spk_error_group.setChecked(False)
    assert 'spk_error' not in model.sim_params


def test_instrument_defaults_toggles_artifacts_key(model: Any) -> None:
    """The instrument-defaults checkbox inserts and removes the artifacts key."""
    model._instrument_defaults_check.setChecked(True)
    assert model.sim_params['artifacts'] == {'instrument_defaults': True}
    model._instrument_defaults_check.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_detector_group_toggles_key(model: Any) -> None:
    """The detector group inserts an empty block and removes it when disabled.

    The block starts empty (per-key discipline): unedited keys stay absent so
    the instrument's catalog defaults keep applying.
    """
    model._detector_group.setChecked(True)
    assert model.sim_params['detector'] == {}
    model._detector_group.setChecked(False)
    assert 'detector' not in model.sim_params


def test_detector_edit_writes_only_the_edited_key(model: Any) -> None:
    """A single spin edit writes its own key and nothing else."""
    model._detector_group.setChecked(True)
    model._detector_gain_state_spin.setValue(3)
    assert model.sim_params['detector'] == {'gain_state': 3}


def test_detector_quantization_is_authorable(model: Any) -> None:
    """The quantization combo writes the detector.quantization key."""
    model._detector_group.setChecked(True)
    model._detector_quantization_combo.setCurrentText('sqrt_lut')
    assert model.sim_params['detector'] == {'quantization': 'sqrt_lut'}


# ---- Registry-driven artifact-mode rows ----


def test_every_registry_mode_has_a_row(model: Any) -> None:
    """The tab generates one row per registered artifact mode."""
    from spindoctor.sim.forward.artifact_modes import ARTIFACT_MODES

    assert set(model._mode_rows) == set(ARTIFACT_MODES)


def test_adversarial_check_toggles_key(model: Any) -> None:
    """The adversarial checkbox inserts and removes the artifacts.adversarial key."""
    model._adversarial_check.setChecked(True)
    assert model.sim_params['artifacts'] == {'adversarial': True}
    model._adversarial_check.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_mode_enable_inserts_empty_map(model: Any) -> None:
    """Enabling a mode row inserts an empty map (absent-key discipline)."""
    model._mode_rows['missing_lines'].group.setChecked(True)
    assert model.sim_params['artifacts'] == {'missing_lines': {}}


def test_mode_param_edit_writes_only_edited_key(model: Any) -> None:
    """Editing one mode parameter writes only that key into the mode map."""
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(3.0)
    assert model.sim_params['artifacts']['missing_lines'] == {'incidence': 3.0}


def test_mode_disable_removes_key_and_prunes_block(model: Any) -> None:
    """Disabling the only enabled mode removes the artifacts block entirely."""
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(3.0)
    row.group.setChecked(False)
    assert 'artifacts' not in model.sim_params


def test_mode_enum_param_writes_native_choice(model: Any) -> None:
    """An enum row writes the choice in its native type (an int period)."""
    row = model._mode_rows['alternating_lines']
    row.group.setChecked(True)
    row._scalar_widgets['period'].setCurrentText('4')
    assert model.sim_params['artifacts']['alternating_lines']['period'] == 4


def test_mode_int_list_param_absent_until_set(model: Any) -> None:
    """A rect/window list key stays absent until its enable box is checked."""
    row = model._mode_rows['cutout_window']
    row.group.setChecked(True)
    assert 'rect' not in model.sim_params['artifacts']['cutout_window']
    row._list_checks['rect'].setChecked(True)
    for index, spin in enumerate(row._list_spins['rect']):
        spin.setValue(10 + index)
    assert model.sim_params['artifacts']['cutout_window']['rect'] == [10, 11, 12, 13]


def test_switches_and_modes_coexist_in_block(model: Any) -> None:
    """Instrument-defaults, adversarial, and a mode share one artifacts block."""
    model._instrument_defaults_check.setChecked(True)
    model._adversarial_check.setChecked(True)
    model._mode_rows['missing_lines'].group.setChecked(True)
    assert model.sim_params['artifacts'] == {
        'instrument_defaults': True,
        'adversarial': True,
        'missing_lines': {},
    }


def test_mode_availability_disables_unavailable_row(model: Any) -> None:
    """A mode unavailable on the instrument is disabled with the registry reason."""
    model.sim_params['instrument'] = 'nhlorri'
    model._refresh_artifact_mode_availability()
    row = model._mode_rows['hot_pixels']
    assert row.group.isEnabled() is False
    assert 'LORRI' in row.group.toolTip()


def test_mode_availability_enables_available_row(model: Any) -> None:
    """An available mode is enabled and carries its incidence semantics tooltip."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._refresh_artifact_mode_availability()
    row = model._mode_rows['hot_pixels']
    assert row.group.isEnabled() is True
    assert 'incidence' in row.group.toolTip()


# ---- Per-star information-asymmetry controls ----


def _star_tab(model: Any) -> Any:
    """Return the widget of the first star's tab."""
    tab_idx = model._find_tab_by_properties('star', 0)
    assert tab_idx is not None
    return model._tabs.widget(tab_idx)


def test_star_default_omits_navigable_key(model: Any) -> None:
    """A GUI-added star leaves the navigable key absent (absent means navigable)."""
    model._add_star_tab()
    assert 'navigable' not in model.sim_params['stars'][0]


def test_star_navigable_uncheck_writes_false(model: Any) -> None:
    """Unchecking navigable writes the present ``navigable: false`` key."""
    model._add_star_tab()
    _star_tab(model).navigable_check.click()
    assert model.sim_params['stars'][0]['navigable'] is False


def test_star_navigable_recheck_removes_key(model: Any) -> None:
    """Re-checking navigable drops the key back to absent."""
    model._add_star_tab()
    check = _star_tab(model).navigable_check
    check.click()
    check.click()
    assert 'navigable' not in model.sim_params['stars'][0]


def test_star_catalog_error_toggle_writes_both_keys(model: Any) -> None:
    """Enabling catalog error inserts both v and u keys; disabling removes them."""
    model._add_star_tab()
    check = _star_tab(model).catalog_error_check
    check.click()
    star = model.sim_params['stars'][0]
    assert 'catalog_error_v' in star
    assert 'catalog_error_u' in star
    check.click()
    assert 'catalog_error_v' not in star
    assert 'catalog_error_u' not in star


def test_star_catalog_error_spin_writes_value(model: Any) -> None:
    """A catalog-error spin edit writes its own key once enabled."""
    model._add_star_tab()
    tab = _star_tab(model)
    tab.catalog_error_check.click()
    tab.catalog_error_v_spin.setValue(1.5)
    assert model.sim_params['stars'][0]['catalog_error_v'] == 1.5


def test_star_delta_mag_toggle_inserts_and_removes(model: Any) -> None:
    """The variable-star enable inserts and removes the delta_mag key."""
    model._add_star_tab()
    check = _star_tab(model).delta_mag_check
    check.click()
    assert 'delta_mag' in model.sim_params['stars'][0]
    check.click()
    assert 'delta_mag' not in model.sim_params['stars'][0]


def test_star_companion_toggle_inserts_and_removes(model: Any) -> None:
    """The companion group inserts the map with its three keys and removes it."""
    model._add_star_tab()
    group = _star_tab(model).companion_group
    group.setChecked(True)
    assert set(model.sim_params['stars'][0]['companion']) == {'sep_px', 'delta_mag', 'angle_deg'}
    group.setChecked(False)
    assert 'companion' not in model.sim_params['stars'][0]


def test_star_companion_spin_writes_sub_key(model: Any) -> None:
    """A companion spin edit updates its own sub-key in the companion map."""
    model._add_star_tab()
    tab = _star_tab(model)
    tab.companion_group.setChecked(True)
    tab.companion_sep_spin.setValue(4.0)
    assert model.sim_params['stars'][0]['companion']['sep_px'] == 4.0


def test_star_scatter_toggle_inserts_and_removes(model: Any) -> None:
    """The star-catalog-scatter checkbox inserts and removes the top-level key."""
    model._star_scatter_check.click()
    model._star_scatter_spin.setValue(0.5)
    assert model.sim_params['star_catalog_scatter_px'] == 0.5
    model._star_scatter_check.click()
    assert 'star_catalog_scatter_px' not in model.sim_params


def test_expected_toggle_inserts_and_removes(model: Any) -> None:
    """The expected group inserts the block with a status and removes it."""
    model._expected_group.setChecked(True)
    assert model.sim_params['expected']['status'] == 'success'
    model._expected_group.setChecked(False)
    assert 'expected' not in model.sim_params


def test_expected_tier_none_writes_null(model: Any) -> None:
    """The confidence-tier (none) choice stores a null tier (assert status only)."""
    model._expected_group.setChecked(True)
    model._expected_tier_combo.setCurrentText('(none)')
    assert model.sim_params['expected']['confidence_tier'] is None


def test_expected_reason_line_edit_sets_and_clears(model: Any) -> None:
    """The status-reason edit writes a non-empty token and drops an empty one."""
    model._expected_group.setChecked(True)
    model._expected_reason_edit.setText('no_signal_in_image')
    assert model.sim_params['expected']['status_reason'] == 'no_signal_in_image'
    model._expected_reason_edit.setText('')
    assert 'status_reason' not in model.sim_params['expected']
