"""GUI smoke tests for ``src/main/nav_create_simulated_image.py``.

These cover the ``_load_parameters`` JSON-load path, specifically that
``shade_solid_rings`` round-trips into both the data model and its checkbox,
and that a missing or null ``closest_planet`` falls back to ``SATURN`` without
raising (``QComboBox.findText(None)`` would otherwise raise ``TypeError``).
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
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

# The driver lives under ``src/main`` and is not an importable package, so load
# it directly from its file path the same way ``python src/main/...`` would.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / 'src' / 'main' / 'nav_create_simulated_image.py'
)
_spec = importlib.util.spec_from_file_location('nav_create_simulated_image', _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
ncsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ncsi)


@pytest.fixture
def qapp() -> QApplication:
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image GUI model."""
    return ncsi.CreateSimulatedImageModel()


def _load_json(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write ``payload`` to a temp file and drive ``_load_parameters`` through it."""
    json_path = tmp_path / 'params.json'
    json_path.write_text(json.dumps(payload), encoding='utf-8')
    monkeypatch.setattr(
        QFileDialog,
        'getOpenFileName',
        staticmethod(lambda *a, **k: (str(json_path), 'JSON')),
    )

    # ``_load_parameters`` swallows exceptions into a critical dialog; surface
    # any such failure as a test error instead so a regression is visible.
    def _fail_on_critical(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'_load_parameters raised an error dialog: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_fail_on_critical))
    model._load_parameters()


def test_load_shade_solid_rings_true_syncs_param(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading ``shade_solid_rings: true`` sets the data-model flag to True."""
    _load_json(monkeypatch, model, tmp_path, {'shade_solid_rings': True})
    assert model.sim_params['shade_solid_rings'] is True


def test_load_shade_solid_rings_true_checks_box(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading ``shade_solid_rings: true`` checks the wired checkbox."""
    _load_json(monkeypatch, model, tmp_path, {'shade_solid_rings': True})
    assert model._shade_solid_rings_check.isChecked() is True


def test_load_shade_solid_rings_false_unchecks_box(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading ``shade_solid_rings: false`` clears the wired checkbox."""
    model._shade_solid_rings_check.setChecked(True)
    _load_json(monkeypatch, model, tmp_path, {'shade_solid_rings': False})
    assert model._shade_solid_rings_check.isChecked() is False


def test_load_missing_closest_planet_defaults_to_saturn(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A JSON file with no ``closest_planet`` key falls back to ``SATURN``."""
    _load_json(monkeypatch, model, tmp_path, {'shade_solid_rings': False})
    assert model.sim_params['closest_planet'] == 'SATURN'


def test_load_null_closest_planet_defaults_to_saturn(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """An explicit ``closest_planet: null`` falls back to ``SATURN`` without raising."""
    _load_json(monkeypatch, model, tmp_path, {'closest_planet': None})
    assert model.sim_params['closest_planet'] == 'SATURN'


def test_load_explicit_closest_planet_is_preserved(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A real ``closest_planet`` value is preserved in the data model."""
    _load_json(monkeypatch, model, tmp_path, {'closest_planet': 'JUPITER'})
    assert model.sim_params['closest_planet'] == 'JUPITER'


def test_default_instrument_is_generic(model: Any) -> None:
    """A fresh model defaults to the generic (instrument-agnostic) frame."""
    assert model.sim_params['instrument'] == 'generic'


def test_instrument_combo_drives_sim_params(model: Any) -> None:
    """Selecting an instrument updates the data model's instrument."""
    model._instrument_combo.setCurrentText('coiss_nac')
    assert model.sim_params['instrument'] == 'coiss_nac'


def test_load_instrument_syncs_combo(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene with an instrument syncs both the model and the combo."""
    _load_json(monkeypatch, model, tmp_path, {'instrument': 'gossi'})
    assert model.sim_params['instrument'] == 'gossi'
    assert model._instrument_combo.currentText() == 'gossi'


def test_load_missing_instrument_defaults_to_generic(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene without an instrument key falls back to generic."""
    _load_json(monkeypatch, model, tmp_path, {'random_seed': 7})
    assert model.sim_params['instrument'] == 'generic'


def test_load_preserves_noise_block(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene round-trips the catalog-only noise block."""
    noise = {'poisson': False, 'read_noise_dn': 12.0}
    _load_json(monkeypatch, model, tmp_path, {'noise': noise})
    assert model.sim_params['noise'] == noise


def test_load_preserves_stray_light_block(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene round-trips the catalog-only stray_light block."""
    stray = {'amplitude': 0.3, 'model': 'radial'}
    _load_json(monkeypatch, model, tmp_path, {'stray_light': stray})
    assert model.sim_params['stray_light'] == stray


def test_default_has_no_dead_background_noise_key(model: Any) -> None:
    """The inert background_noise_intensity key is gone from the defaults."""
    assert 'background_noise_intensity' not in model.sim_params


def test_default_noise_block_present(model: Any) -> None:
    """A fresh model carries a detector-noise block with Poisson on."""
    assert model.sim_params['noise']['poisson'] is True


def test_poisson_toggle_updates_noise(model: Any) -> None:
    """Unchecking Poisson writes through to the noise block."""
    model._poisson_check.setChecked(False)
    assert model.sim_params['noise']['poisson'] is False


def test_read_noise_spin_updates_noise(model: Any) -> None:
    """The read-noise spin writes read_noise_dn into the noise block."""
    model._read_noise_spin.setValue(12.5)
    assert model.sim_params['noise']['read_noise_dn'] == 12.5


def test_cosmic_ray_spin_updates_noise(model: Any) -> None:
    """The cosmic-ray spin writes cosmic_ray_rate_per_sec into the noise block."""
    model._cosmic_ray_spin.setValue(0.002)
    assert model.sim_params['noise']['cosmic_ray_rate_per_sec'] == 0.002


def test_missing_data_spin_updates_noise(model: Any) -> None:
    """The missing-data spin writes missing_data_rate into the noise block."""
    model._missing_data_spin.setValue(0.1)
    assert model.sim_params['noise']['missing_data_rate'] == 0.1


def test_load_noise_block_syncs_widgets(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a noise block syncs the panel widgets."""
    _load_json(monkeypatch, model, tmp_path, {'noise': {'poisson': False, 'read_noise_dn': 9.0}})
    assert model._poisson_check.isChecked() is False
    assert model._read_noise_spin.value() == 9.0


def test_stray_light_defaults_off(model: Any) -> None:
    """A fresh model has no stray-light amplitude set (off)."""
    assert model._stray_amplitude_spin.value() == 0.0


def test_stray_amplitude_updates_block(model: Any) -> None:
    """The amplitude spin writes into the stray_light block."""
    model._stray_amplitude_spin.setValue(0.4)
    assert model.sim_params['stray_light']['amplitude'] == 0.4


def test_stray_direction_updates_block(model: Any) -> None:
    """The direction spin writes direction_deg into the stray_light block."""
    model._stray_direction_spin.setValue(45.0)
    assert model.sim_params['stray_light']['direction_deg'] == 45.0


def test_stray_model_updates_block(model: Any) -> None:
    """The model combo writes the stray_light model."""
    model._stray_model_combo.setCurrentText('radial')
    assert model.sim_params['stray_light']['model'] == 'radial'


def test_load_stray_light_syncs_widgets(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a stray_light block syncs the panel widgets."""
    _load_json(
        monkeypatch,
        model,
        tmp_path,
        {'stray_light': {'amplitude': 0.25, 'direction_deg': 90.0, 'model': 'radial'}},
    )
    assert model._stray_amplitude_spin.value() == 0.25
    assert model._stray_direction_spin.value() == 90.0
    assert model._stray_model_combo.currentText() == 'radial'


def test_saturation_overlay_defaults_off(model: Any) -> None:
    """The saturation overlay starts disabled."""
    assert model._show_saturation_overlay is False


def test_toggle_saturation_overlay_sets_flag(model: Any) -> None:
    """Checking the overlay box flips the display flag."""
    model._saturation_overlay_check.setChecked(True)
    assert model._show_saturation_overlay is True


def test_saturation_dn_for_raw_instrument(model: Any) -> None:
    """A raw-DN instrument reports its saturation DN."""
    model.sim_params['instrument'] = 'coiss_nac'
    assert model._current_saturation_dn() == 4095.0


def test_saturation_dn_none_for_calibrated_instrument(model: Any) -> None:
    """A calibrated-IF instrument has no saturation DN."""
    model.sim_params['instrument'] = 'vgiss'
    assert model._current_saturation_dn() is None


def test_saturation_overlay_updates_status(model: Any) -> None:
    """Enabling the overlay renders and reports a saturation fraction."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._update_image()
    model._saturation_overlay_check.setChecked(True)
    assert 'Saturated' in model._saturation_label.text()


def test_saturation_overlay_off_clears_status(model: Any) -> None:
    """Disabling the overlay clears the saturation status."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._update_image()
    model._saturation_overlay_check.setChecked(True)
    model._saturation_overlay_check.setChecked(False)
    assert model._saturation_label.text() == ''


def test_psf_preview_collapsed_by_default(model: Any) -> None:
    """The PSF preview pane starts collapsed with its inset hidden."""
    assert model._psf_group.isChecked() is False
    assert model._psf_image_label.isVisibleTo(model._psf_group) is False


def test_psf_preview_expands_and_annotates(model: Any) -> None:
    """Expanding the pane shows the PSF inset and its sigma / FWHM."""
    model.sim_params['instrument'] = 'coiss_nac'
    model._psf_group.setChecked(True)
    assert model._psf_image_label.isVisibleTo(model._psf_group) is True
    assert not model._psf_image_label.pixmap().isNull()
    assert 'sigma' in model._psf_info_label.text()


def test_psf_preview_tracks_instrument(model: Any) -> None:
    """Switching instruments updates the PSF sigma shown."""
    model._psf_group.setChecked(True)
    model._instrument_combo.setCurrentText('coiss_nac')
    coiss_text = model._psf_info_label.text()
    model._instrument_combo.setCurrentText('gossi')
    gossi_text = model._psf_info_label.text()
    assert coiss_text != gossi_text


def test_new_body_defaults_to_ellipsoid(model: Any) -> None:
    """A newly added body uses the ellipsoid shape model by default."""
    model._add_body_tab()
    assert model.sim_params['bodies'][0]['shape_model'] == 'ellipsoid'


def test_new_body_has_mesh_fields(model: Any) -> None:
    """A newly added body carries the mesh shape parameters."""
    model._add_body_tab()
    body = model.sim_params['bodies'][0]
    assert body['mesh_lumpiness'] == 0.3
    assert body['pose_euler_deg'] == [0.0, 0.0, 0.0]


def test_body_shape_model_field_updates(model: Any) -> None:
    """Setting a body's shape_model writes through to the data model."""
    model._add_body_tab()
    model._on_body_field(0, 'shape_model', 'polyhedral_mesh')
    assert model.sim_params['bodies'][0]['shape_model'] == 'polyhedral_mesh'


def test_body_pose_handler_updates_axis(model: Any) -> None:
    """The pose handler writes one axis of the body's mesh pose."""
    model._add_body_tab()
    model._on_body_pose(0, 1, 90.0)
    assert model.sim_params['bodies'][0]['pose_euler_deg'] == [0.0, 90.0, 0.0]


def test_body_pose_handler_pads_missing_pose(model: Any) -> None:
    """The pose handler tolerates a body that lacks a pose list."""
    model.sim_params['bodies'].append({'name': 'B', 'center_v': 1.0, 'center_u': 1.0})
    model._on_body_pose(0, 2, 45.0)
    assert model.sim_params['bodies'][0]['pose_euler_deg'] == [0.0, 0.0, 45.0]


def _no_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any error dialog raise so failures surface as test errors."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


def test_save_scene_writes_valid_yaml(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving a scene writes a YAML the schema validates."""
    from nav.sim.scene import load_sim_scene

    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    out = tmp_path / 'myscene.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    scene = load_sim_scene(out)
    assert scene.instrument == 'coiss_nac'
    assert scene.scene_name == 'myscene'
    assert scene.image_size_vu == (128, 128)


def test_load_scene_populates_model(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading a scene YAML populates the data model from it."""
    scene_yaml = tmp_path / 'loadme.yaml'
    scene_yaml.write_text(
        'schema_version: 1\nscene_name: loadme\ninstrument: gossi\n'
        'image_size_vu: [96, 96]\nrandom_seed: 5\n'
    )
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(scene_yaml), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model.sim_params['instrument'] == 'gossi'
    assert model.sim_params['size_v'] == 96


def test_scene_round_trips_through_gui(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene saved then loaded preserves instrument, size, and a body."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 100
    model.sim_params['size_u'] = 100
    model.sim_params['bodies'] = [
        {'name': 'B', 'center_v': 50.0, 'center_u': 50.0, 'axis1': 40.0, 'axis2': 40.0}
    ]
    out = tmp_path / 'rt.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    model.sim_params['instrument'] = 'generic'
    model._load_scene()
    assert model.sim_params['instrument'] == 'coiss_nac'
    assert model.sim_params['size_v'] == 100
    assert model.sim_params['bodies'][0]['name'] == 'B'
