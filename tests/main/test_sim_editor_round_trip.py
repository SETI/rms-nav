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
    'background_stars_num': 12,
    'background_stars_psf_sigma': 1.1,
    'background_stars_distribution_exponent': 2.3,
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
    'optics': {
        'stray_light': {
            'amplitude': 0.3,
            'direction_deg': 35.0,
            'model': 'radial',
            'center_v': 40.0,
            'center_u': 50.0,
        }
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
        }
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
