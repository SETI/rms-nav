"""Round-trip acceptance tests for the ``sd_create_simulated_image`` editor.

These prove the editor's data model preserves loss-free scene round-tripping:

* A v2 scene exercising the full current key inventory loads into the editor,
  survives a single edit through the editor's data model, and re-saves with
  every other block byte-for-byte identical in meaning (loaded dicts compared,
  not raw bytes).
* A scene authored entirely through the GUI (added body / ring / star) is
  idempotent under save -> load -> save -> load.
* A GUI-added star carries a list-valued ``psf_size`` (never a tuple), so the
  YAML dumper and the reloaded form agree.

Qt runs headless (offscreen platform).
"""

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

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.sim.scene import _ALLOWED_KEYS, load_sim_scene, save_sim_scene

# A v2 scene exercising the full current key inventory: every top-level key,
# and every per-object idealized and truth key for a body, a ring, and a star.
_FULL_SCENE: dict[str, Any] = {
    'instrument': 'coiss_nac',
    'size_v': 128,
    'size_u': 128,
    'random_seed': 7,
    'exposure_sec': 2.5,
    'offset_v': 1.25,
    'offset_u': -0.75,
    'offset_rotation_deg': 3.5,
    'midtime_utc': '2010-01-01T00:00:00Z',
    'closest_planet': 'SATURN',
    'time': 100.0,
    'ring_epoch': 50.0,
    'shade_solid_rings': True,
    'sky_counts': {'a': -3.0, 'b': 0.35, 'density_factor': 8.0, 'diffuse_e_per_px': 2.5},
    'star_catalog_scatter_px': 0.4,
    'expected': {
        'status': 'success',
        'confidence_tier': 'high',
        'status_reason': 'ok',
    },
    'fit_camera_rotation': True,
    'noise': {
        'poisson': True,
        'read_noise_dn': 6.0,
        'cosmic_ray_rate_per_sec': 0.001,
        'missing_data_rate': 0.01,
        'bias_dn': 18.0,
        'bloom_length': 4,
        'signal_full_scale_frac': 0.6,
        'pixel_area_cm2': 1.5,
    },
    'oversample': 2,
    'optics': {
        'psf': {'sigma_v': 0.6, 'sigma_u': 0.5, 'w': 0.02, 'r0': 2.0, 'n': 3.0},
        'smear': [
            {'dv_px': 1.0, 'du_px': 0.0, 'object_class': 'all'},
            {'dv_px': 0.0, 'du_px': 2.0, 'object_class': 'stars'},
        ],
        'distortion': {
            'k1': 0.01,
            'k2': 0.0,
            'center_v': 40.0,
            'center_u': 50.0,
            'nonradial_rms_px': 0.3,
        },
        'ghosts': [{'dv_px': 10.0, 'du_px': -5.0, 'amplitude': 0.01, 'defocus_sigma': 2.0}],
        'stray_light': {
            'amplitude': 0.3,
            'direction_deg': 35.0,
            'model': 'radial',
            'center_v': 40.0,
            'center_u': 50.0,
        },
    },
    'spk_error': {'dv_px': 0.5, 'du_px': -0.5, 'reference_range_km': 1000.0},
    'detector': {
        'gain_state': 2,
        'detector_model': 'ccd',
        'exposure_ref_sec': 1.0,
        'quantization': 'exact',
    },
    'artifacts': {
        'instrument_defaults': True,
        'adversarial': True,
        # One representative mode per stage family: a telemetry structured loss,
        # a detector-electronics mode, and a routed quantization mode -- all
        # available on coiss_nac -- so the round-trip exercises the generated
        # mode rows, not only the two switches.
        'missing_lines': {'incidence': 2.0, 'contiguous_run': True},
        'banding_coherent': {'incidence': 0.5, 'amplitude_e': 3.0},
        'quantization_lut': {'incidence': 1.0},
    },
    'instrument_config': {'inherit': 'coiss_nac'},
    'bodies': [
        {
            'name': 'Mimas',
            'shape_model': 'polyhedral_mesh',
            'center_v': 64.0,
            'center_u': 64.0,
            'axis1': 90.0,
            'axis2': 70.0,
            'axis3': 60.0,
            'rotation_z': 12.0,
            'rotation_tilt': 8.0,
            'illumination_angle': 45.0,
            'phase_angle': 30.0,
            'range_km': 1000.0,
            'km_per_pixel': 5.0,
            'mesh_lumpiness': 0.4,
            'mesh_n_lat': 20,
            'mesh_n_lon': 40,
            'mesh_seed': 3,
            'pose_euler_deg': [10.0, 35.0, 0.0],
            'crater_fill': 1.5,
            'crater_min_radius': 0.05,
            'crater_max_radius': 0.2,
            'crater_power_law_exponent': 3.0,
            'crater_relief_scale': 0.6,
            'seed': 11,
            'anti_aliasing': 0.5,
            'nav_override': {
                'shape_model': 'ellipsoid',
                'mesh_lumpiness': 0.0,
                'pose_euler_deg': [10.0, 35.0, 0.0],
            },
        }
    ],
    'rings': [
        {
            'name': 'RingA',
            'feature_type': 'RINGLET',
            'center_v': 64.0,
            'center_u': 64.0,
            'shading_distance': 20.0,
            'range': 5.0,
            'range_km': 2000.0,
            'inner_data': [
                {'mode': 1, 'a': 100.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
            ],
            'outer_data': [
                {'mode': 1, 'a': 120.0, 'rms': 1.0, 'ae': 0.0, 'long_peri': 0.0, 'rate_peri': 0.0}
            ],
        }
    ],
    'stars': [
        {
            'name': 'S1',
            'catalog_name': 'UCAC4',
            'v': 30.0,
            'u': 40.0,
            'vmag': 6.0,
            'spectral_class': 'G2',
            'move_v': 1.0,
            'move_u': -2.0,
            'psf_size': [11, 11],
            'psf_sigma': 1.2,
        },
        {
            # A non-navigable confounder star carrying every truth-side star key:
            # a planted catalog-position error, an unresolved companion, and a
            # variable-brightness delta.
            'name': 'S2',
            'catalog_name': 'UCAC4',
            'v': 80.0,
            'u': 90.0,
            'vmag': 8.0,
            'navigable': False,
            'catalog_error_v': 0.6,
            'catalog_error_u': -0.4,
            'delta_mag': 0.5,
            'companion': {'sep_px': 2.5, 'delta_mag': 1.5, 'angle_deg': 30.0},
        },
    ],
}


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


def _no_critical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any error dialog raise so a failed save/load surfaces as a test error."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


def _comparable(scene: dict[str, Any]) -> dict[str, Any]:
    """Drop the file-identity metadata so two saved scenes compare on content."""
    return {k: v for k, v in scene.items() if k not in ('schema_version', 'scene_name')}


def test_full_inventory_round_trip_preserves_other_blocks(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Editing one field leaves every other block identical after a save cycle."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    original = load_sim_scene(src)

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()

    # Edit exactly one field through the editor's data model (drives _on_random_seed).
    model._random_seed_spin.setValue(2024)
    assert model.sim_params['random_seed'] == 2024

    out = tmp_path / 'edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)

    # The edited field changed; nothing else did.
    assert resaved['random_seed'] == 2024
    for key in _ALLOWED_KEYS - {'schema_version', 'scene_name', 'random_seed'}:
        assert resaved.get(key) == original.get(key), f'block {key!r} did not survive'


def test_full_inventory_edit_is_the_only_change(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The nested body/ring/star blocks survive with identical meaning."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    original = load_sim_scene(src)

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._random_seed_spin.setValue(2024)

    out = tmp_path / 'edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)

    assert resaved['bodies'] == original['bodies']
    assert resaved['rings'] == original['rings']
    assert resaved['stars'] == original['stars']


def test_gui_authored_scene_save_load_idempotent(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A GUI-authored scene is stable under save -> load -> save -> load."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_body_tab()
    model._add_ring_tab()
    model._add_star_tab()  # GUI-added star carries a list psf_size

    file1 = tmp_path / 'gui1.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(file1), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded1 = load_sim_scene(file1)

    # Feed the loaded scene back through the editor and re-save.
    model._apply_params_dict(loaded1)
    file2 = tmp_path / 'gui2.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(file2), 'YAML'))
    )
    model._save_scene()
    loaded2 = load_sim_scene(file2)

    assert _comparable(loaded1) == _comparable(loaded2)


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


def test_gui_added_star_scene_saves_without_error(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving a scene with a GUI-added star succeeds and preserves psf_size."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_star_tab()

    out = tmp_path / 'star.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['stars'][0]['psf_size'] == [11, 11]


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


def test_match_navigator_survives_save_and_load(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Saving persists the authored match-navigator form and never mutates it.

    The renderer resolves the navigator-matched PSF only when it builds the
    kernel, so both the live editor state and the file keep the authored form
    and the checkbox is still set after a reload.
    """
    model.sim_params['instrument'] = 'coiss_wac'
    model._psf_optics_group.setChecked(True)
    model._psf_match_nav_check.setChecked(True)

    out = tmp_path / 'floor.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    # Saving does not rewrite the live editor state.
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}
    saved = load_sim_scene(out)
    assert saved['optics']['psf'] == {'match_navigator': True}

    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._load_scene()
    assert model.sim_params['optics']['psf'] == {'match_navigator': True}
    assert model._psf_match_nav_check.isChecked() is True


def test_psf_sigma_u_widget_defaults_to_sigma_v(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A PSF block omitting sigma_u displays the renderer's default (sigma_v)."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'psf': {'sigma_v': 1.3}},
    }
    src = tmp_path / 'psf.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._psf_sigma_u_spin.value() == 1.3


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


def test_partial_distortion_edit_leaves_center_u_absent(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A k1 edit on a partial {k1, center_v} block never backfills center_u."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'distortion': {'k1': 0.01, 'center_v': 40.0}},
    }
    src = tmp_path / 'partial_distortion.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._distortion_k1_spin.setValue(0.02)
    assert 'center_u' not in model.sim_params['optics']['distortion']

    out = tmp_path / 'partial_distortion_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['optics']['distortion'] == {'k1': 0.02, 'center_v': 40.0}


def test_partial_distortion_center_u_displays_frame_center(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The centre-u spin shows the effective default (frame centre) when absent."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'optics': {'distortion': {'k1': 0.01, 'center_v': 40.0}},
    }
    src = tmp_path / 'partial_distortion.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._distortion_center_v_spin.value() == 40.0
    assert model._distortion_center_u_spin.value() == 32.0


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


def test_ring_spk_error_scene_authors_and_validates(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A ring + spk_error scene authored through the editor saves cleanly.

    spk_error requires range_km on every ring; the ring tab's physical-range
    control makes the key authorable (absent unless set).
    """
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    model._add_ring_tab()
    tab_idx = model._find_tab_by_properties('ring', 0)
    assert tab_idx is not None
    ring_tab = model._tabs.widget(tab_idx)
    ring_tab.range_km_check.click()
    ring_tab.range_km_spin.setValue(2.0e6)
    assert model.sim_params['rings'][0]['range_km'] == 2.0e6
    model._spk_error_group.setChecked(True)

    out = tmp_path / 'ring_spk.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['rings'][0]['range_km'] == 2.0e6
    assert loaded['spk_error']['reference_range_km'] > 0.0


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


def test_partial_detector_scene_edit_preserves_authored_keys(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Nudging one detector spin keeps an authored quantization key intact."""
    scene = {
        'instrument': 'coiss_nac',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'detector': {'quantization': 'sqrt_lut'},
    }
    src = tmp_path / 'partial.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._detector_gain_state_spin.setValue(3)

    out = tmp_path / 'partial_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['detector'] == {'quantization': 'sqrt_lut', 'gain_state': 3}


def test_vgiss_scene_stays_vidicon_through_an_edit(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Editing a vgiss detector spin never backfills a ccd detector_model."""
    scene = {
        'instrument': 'vgiss',
        'size_v': 64,
        'size_u': 64,
        'random_seed': 1,
        'detector': {'exposure_ref_sec': 2.0},
    }
    src = tmp_path / 'vgiss.yaml'
    save_sim_scene(scene, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    # The widgets display the vgiss catalog defaults for unauthored keys.
    assert model._detector_model_combo.currentText() == 'vidicon'
    model._detector_exposure_ref_spin.setValue(3.0)

    out = tmp_path / 'vgiss_edited.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    model._save_scene()
    resaved = load_sim_scene(out)
    assert resaved['detector'] == {'exposure_ref_sec': 3.0}

    from spindoctor.sim.forward.detector.params import resolve_detector_params

    assert resolve_detector_params(resaved).detector_model == 'vidicon'


def test_load_full_optics_syncs_group_states(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full-inventory scene checks every optics/artifacts group."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._psf_optics_group.isChecked() is True
    assert len(model._smear_rows) == 2
    assert model._ghosts_group.isChecked() is True
    assert model._spk_error_group.isChecked() is True
    assert model._detector_group.isChecked() is True
    assert model._instrument_defaults_check.isChecked() is True


def test_load_then_disable_optics_clears_blocks(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A loaded sub-block can be disabled back to absence after sync."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    model._detector_group.setChecked(False)
    assert 'detector' not in model.sim_params


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


def test_load_full_scene_syncs_mode_rows(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full-inventory scene checks its adversarial switch and modes."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._adversarial_check.isChecked() is True
    assert model._mode_rows['missing_lines'].group.isChecked() is True
    assert model._mode_rows['banding_coherent'].group.isChecked() is True
    assert model._mode_rows['quantization_lut'].group.isChecked() is True
    incidence = model._mode_rows['missing_lines']._scalar_widgets['incidence'].value()
    assert incidence == 2.0


def test_gui_authored_mode_scene_validates(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """A scene authored through the mode rows saves and reloads cleanly."""
    model.sim_params['instrument'] = 'coiss_nac'
    model.sim_params['size_v'] = 128
    model.sim_params['size_u'] = 128
    row = model._mode_rows['missing_lines']
    row.group.setChecked(True)
    row._scalar_widgets['incidence'].setValue(2.0)
    model._adversarial_check.setChecked(True)

    out = tmp_path / 'modes.yaml'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(out), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._save_scene()
    loaded = load_sim_scene(out)
    assert loaded['artifacts']['missing_lines'] == {'incidence': 2.0}
    assert loaded['artifacts']['adversarial'] is True


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


def test_load_full_scene_syncs_star_asymmetry_controls(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """Loading the full scene populates the scene-level star controls."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    assert model._star_scatter_check.isChecked() is True
    assert model._star_scatter_spin.value() == 0.4
    assert model._expected_group.isChecked() is True
    assert model._expected_status_combo.currentText() == 'success'
    assert model._expected_tier_combo.currentText() == 'high'


def test_load_full_scene_syncs_confounder_star_tab(
    monkeypatch: pytest.MonkeyPatch, model: Any, tmp_path: Path
) -> None:
    """The confounder star's tab reflects its planted truth keys after load."""
    src = tmp_path / 'full.yaml'
    save_sim_scene(_FULL_SCENE, src)
    monkeypatch.setattr(
        QFileDialog, 'getOpenFileName', staticmethod(lambda *a, **k: (str(src), 'YAML'))
    )
    _no_critical(monkeypatch)
    model._load_scene()
    # S2 is the non-navigable confounder; find its tab by name.
    s2_index = next(i for i, s in enumerate(model.sim_params['stars']) if s['name'] == 'S2')
    tab_idx = model._find_tab_by_properties('star', s2_index)
    assert tab_idx is not None
    tab = model._tabs.widget(tab_idx)
    assert tab.navigable_check.isChecked() is False
    assert tab.catalog_error_check.isChecked() is True
    assert tab.companion_group.isChecked() is True
    assert tab.delta_mag_check.isChecked() is True
