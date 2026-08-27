"""Smoke tests for the shipped YAML configuration files.

Asserts the renumbered ``config_NNN_*.yaml`` set loads cleanly under
``Config.read_config`` and that bootstrap angles are now in degrees.
"""

from __future__ import annotations

from spindoctor.config import Config


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


def test_no_shipped_instrument_fits_camera_rotation() -> None:
    """No instrument enables rotation fitting, and enabling one needs work first.

    A fitted rotation is not currently a well-defined quantity.  Each technique
    turns about its own centre, and the same physical twist about two different
    pivots differs by the pure translation ``(I - R(theta))(P - Q)``, so
    techniques measuring one rotation report translations that the ensemble
    fuses as though they were comparable.  A fitted rotation also cannot be
    carried into a corrected attitude, so it suppresses the corrected C-matrix
    and omits the frame from every corrected kernel.

    Enabling it for an instrument therefore needs the rotation redesign first
    -- fitting every technique about the FOV centre -- and needs the conflicted
    result path taught to carry a rotation, which today it silently drops.
    This test fails when a configuration enables it, so that arrives as a
    decision rather than as a surprise.
    """
    config = Config()
    config.read_config()
    cassini = config.category('cassini_iss')
    blocks = [
        cassini['nac'],
        cassini['wac'],
        config.category('voyager_iss'),
        config.category('galileo_ssi'),
        config.category('newhorizons_lorri'),
    ]
    assert [b for b in blocks if b['fit_camera_rotation'] is True] == []


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
        assert camera['mag_offset']['fallback_combo'] == 'CL1+CL2'
    voyager = config.category('voyager_iss')
    assert voyager['data_units'] == 'calibrated_if'
    assert voyager['noise']['saturation_dn'] == 255
    assert voyager['fit_camera_rotation'] is False
    assert voyager['max_rotation_deg'] == 5.0
    assert voyager['mag_offset']['fallback_combo'] == 'CL'
    galileo = config.category('galileo_ssi')
    assert galileo['data_units'] == 'raw_dn'
    assert galileo['noise']['saturation_dn'] == 255
    assert galileo['fit_camera_rotation'] is False
    assert galileo['max_rotation_deg'] == 5.0
    assert galileo['mag_offset']['fallback_combo'] == 'CL'
    nhlorri = config.category('newhorizons_lorri')
    assert nhlorri['data_units'] == 'raw_dn'
    assert nhlorri['noise']['saturation_dn'] == 4095
    assert nhlorri['fit_camera_rotation'] is False
    assert nhlorri['max_rotation_deg'] == 5.0
    assert nhlorri['mag_offset']['fallback_combo'] == '1'


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
