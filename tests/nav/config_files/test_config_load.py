"""Smoke tests for the shipped YAML configuration files.

Asserts the renumbered ``config_NNN_*.yaml`` set loads cleanly under
``Config.read_config`` and that bootstrap angles are now in degrees.
"""

from __future__ import annotations

from nav.config import Config


def test_default_config_loads_cleanly() -> None:
    """The full bundled config set merges without raising."""
    config = Config()
    config.read_config()
    assert config.is_loaded


def test_bootstrap_angles_are_degrees() -> None:
    """``config_070_bootstrap.yaml`` exposes degree-valued angle fields."""
    config = Config()
    config.read_config()
    bootstrap = config.bootstrap
    # Degree-keyed names plus magnitudes consistent with Cardinal Principle
    # #4 — values are in degrees, not radians.
    assert bootstrap.max_phase_angle_deg == 135.0
    assert bootstrap.max_incidence_angle_deg == 70.0
    assert bootstrap.max_emission_angle_deg == 70.0
    assert bootstrap.lon_resolution_deg == 0.5
    assert bootstrap.lat_resolution_deg == 0.5
    assert bootstrap.max_subsolar_dist_deg == 45.0


def test_per_instrument_required_fields_present() -> None:
    """Every shipped 4N0 instrument block has the required Phase 3 keys."""
    config = Config()
    config.read_config()
    cassini = config.category('cassini_iss')
    for camera in (cassini['nac'], cassini['wac']):
        assert camera['data_units'] == 'raw_dn'
        assert camera['noise']['saturation_dn'] == 4095
        assert camera['fit_camera_rotation'] is False
        assert camera['max_rotation_deg'] == 5.0
        assert camera['mag_offset']['fallback_combo']
    voyager = config.category('voyager_iss')
    assert voyager['data_units'] == 'raw_dn'
    assert voyager['noise']['saturation_dn'] == 255
    galileo = config.category('galileo_ssi')
    assert galileo['data_units'] == 'raw_dn'
    assert galileo['noise']['saturation_dn'] == 255


def test_body_shape_section_loaded() -> None:
    """Body-shape entries are exposed via ``config.body_shape``."""
    config = Config()
    config.read_config()
    body_shape = config.body_shape
    assert 'MIMAS' in body_shape
    mimas = body_shape['MIMAS']
    # _sources keys are stripped at load time.
    assert '_sources' not in mimas
    assert 'shape_class_hint' in mimas
