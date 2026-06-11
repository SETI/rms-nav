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
