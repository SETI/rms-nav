"""Tests for the logging configuration key namespace and its validation."""

import argparse
from pathlib import Path

import pytest

from spindoctor.config.config import Config
from spindoctor.config.config_helper import load_default_and_user_config
from spindoctor.config.logging_keys import (
    CATEGORY_KEYS,
    LOG_LEVEL_NAMES,
    LOGGER_KEYS,
    OTHER_LOG_KEYS,
    log_key_for,
    model_log_keys,
    normalize_level,
    technique_log_keys,
    validate_logging_config,
)
from spindoctor.config.program_names import PROGRAM_NAMES, SD_MOSAIC, SD_OFFSET


def _config_with_logging(tmp_path: Path, body: str) -> Config:
    """Build a Config whose logging section is overridden by ``body``.

    Parameters:
        tmp_path: Directory to write the override file into.
        body: YAML text placed under a ``logging:`` key.

    Returns:
        A loaded Config carrying the shipped defaults plus ``body``.
    """
    override = tmp_path / 'override.yaml'
    override.write_text(f'logging:\n{body}')
    config = Config()
    config.read_config()
    config.update_config(str(override))
    return config


# ---------------------------------------------------------------------------
# log_key_for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('class_name', 'expected'),
    [
        ('TitanHazeNav', 'titan_haze'),
        ('BodyDiscCorrelateNav', 'body_disc_correlate'),
        ('StarFieldFromCatalogNav', 'star_field_from_catalog'),
        ('NavTechniqueManual', 'manual'),
        ('NavModelStars', 'stars'),
        ('NavModelBody', 'body'),
    ],
)
def test_log_key_derived_from_class_name(class_name: str, expected: str) -> None:
    cls = type(class_name, (), {})
    assert log_key_for(cls) == expected


def test_log_key_collapses_simulated_onto_its_sibling() -> None:
    real = type('NavModelRings', (), {})
    simulated = type('NavModelRingsSimulated', (), {})
    assert log_key_for(real) == log_key_for(simulated)


def test_log_key_honors_a_declared_override() -> None:
    cls = type('SomethingAwkwardNav', (), {'log_key': 'chosen_name'})
    assert log_key_for(cls) == 'chosen_name'


def test_log_key_ignores_an_inherited_override() -> None:
    parent = type('ParentNav', (), {'log_key': 'parent_key'})
    child = type('BodyLimbNav', (parent,), {})
    assert log_key_for(child) == 'body_limb'


# ---------------------------------------------------------------------------
# Registry-derived key sets
# ---------------------------------------------------------------------------


def test_technique_keys_include_a_known_technique() -> None:
    assert 'titan_haze' in technique_log_keys()


def test_technique_keys_include_the_interactive_technique() -> None:
    assert 'manual' in technique_log_keys()


def test_technique_keys_are_all_snake_case() -> None:
    offenders = [key for key in technique_log_keys() if key != key.lower()]
    assert offenders == []


def test_model_keys_include_the_four_families() -> None:
    assert model_log_keys() >= frozenset({'stars', 'body', 'rings', 'titan'})


def test_shipped_model_classes_map_to_exactly_four_families() -> None:
    # Scoped to shipped classes because the registry is process-global and test
    # modules elsewhere in the suite define their own NavModel subclasses, which
    # self-register and would otherwise make this assertion order-dependent.
    from spindoctor.nav_model.nav_model import NavModel

    shipped = {
        log_key_for(cls) for cls in NavModel._registry if cls.__module__.startswith('spindoctor.')
    }
    assert shipped == frozenset({'stars', 'body', 'rings', 'titan'})


def test_no_backend_key_leaked_into_other() -> None:
    assert OTHER_LOG_KEYS.isdisjoint({'nav', 'backplane', 'reproj'})


# ---------------------------------------------------------------------------
# Shipped defaults
# ---------------------------------------------------------------------------


def test_shipped_defaults_define_a_logging_section() -> None:
    config = Config()
    config.read_config()
    assert config.logging != {}


def test_shipped_defaults_are_valid() -> None:
    config = Config()
    config.read_config()
    validate_logging_config(config)


def test_shipped_defaults_preserve_the_annotation_level() -> None:
    config = Config()
    config.read_config()
    assert config.logging['other']['annotate'] == 'ERROR'


def test_shipped_defaults_disable_strict_scope() -> None:
    config = Config()
    config.read_config()
    assert config.logging['strict_scope'] is False


@pytest.mark.parametrize('logger_key', sorted(LOGGER_KEYS))
def test_shipped_defaults_define_each_logger_default(logger_key: str) -> None:
    config = Config()
    config.read_config()
    assert config.logging[logger_key] == 'INFO'


@pytest.mark.parametrize('category', sorted(CATEGORY_KEYS))
def test_shipped_defaults_define_each_category(category: str) -> None:
    config = Config()
    config.read_config()
    assert category in config.logging


# ---------------------------------------------------------------------------
# Validation: accepted input
# ---------------------------------------------------------------------------


def test_accepts_a_technique_override(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_haze: DEBUG\n')
    validate_logging_config(config)


def test_accepts_a_model_override(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  models:\n    rings: WARNING\n')
    validate_logging_config(config)


def test_accepts_a_program_block(tmp_path: Path) -> None:
    config = _config_with_logging(
        tmp_path, f'  programs:\n    {SD_MOSAIC}:\n      main: WARNING\n      image: DEBUG\n'
    )
    validate_logging_config(config)


@pytest.mark.parametrize('level', sorted(LOG_LEVEL_NAMES))
def test_accepts_every_level_name(tmp_path: Path, level: str) -> None:
    config = _config_with_logging(tmp_path, f'  main: {level}\n')
    validate_logging_config(config)


def test_accepts_a_lowercase_level_name(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  main: debug\n')
    validate_logging_config(config)


# ---------------------------------------------------------------------------
# Validation: rejected input
# ---------------------------------------------------------------------------


def test_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  wibble: INFO\n')
    with pytest.raises(ValueError, match=r'logging\.wibble'):
        validate_logging_config(config)


def test_rejects_an_unknown_technique_key(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_hazey: DEBUG\n')
    with pytest.raises(ValueError, match='titan_hazey'):
        validate_logging_config(config)


def test_unknown_technique_error_names_the_category(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_hazey: DEBUG\n')
    with pytest.raises(ValueError, match='techniques key'):
        validate_logging_config(config)


def test_rejects_a_backend_named_as_a_module(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  other:\n    reproj: DEBUG\n')
    with pytest.raises(ValueError, match='reproj'):
        validate_logging_config(config)


def test_rejects_an_unknown_level_name(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  main: CHATTY\n')
    with pytest.raises(ValueError, match='CHATTY'):
        validate_logging_config(config)


def test_rejects_a_non_string_level(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  main: 20\n')
    with pytest.raises(ValueError, match='level name string'):
        validate_logging_config(config)


def test_rejects_an_unknown_program(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  programs:\n    sd_nonesuch:\n      main: INFO\n')
    with pytest.raises(ValueError, match='sd_nonesuch'):
        validate_logging_config(config)


def test_rejects_strict_scope_inside_a_program_block(tmp_path: Path) -> None:
    config = _config_with_logging(
        tmp_path, f'  programs:\n    {SD_OFFSET}:\n      strict_scope: true\n'
    )
    with pytest.raises(ValueError, match='global setting'):
        validate_logging_config(config)


def test_rejects_a_non_boolean_strict_scope(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  strict_scope: yes please\n')
    with pytest.raises(ValueError, match='true or false'):
        validate_logging_config(config)


def test_rejects_a_scalar_category(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, '  techniques: DEBUG\n')
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(config)


def test_rejects_a_scalar_program_block(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, f'  programs:\n    {SD_OFFSET}: DEBUG\n')
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(config)


def test_rejects_a_programs_block_nested_in_a_program(tmp_path: Path) -> None:
    config = _config_with_logging(
        tmp_path,
        f'  programs:\n    {SD_OFFSET}:\n      programs:\n        {SD_OFFSET}:\n'
        f'          main: INFO\n',
    )
    with pytest.raises(ValueError, match='does not nest'):
        validate_logging_config(config)


def test_nesting_cannot_smuggle_past_an_unknown_key(tmp_path: Path) -> None:
    # Without the nesting guard the inner block is skipped entirely, so keys
    # rejected one level up would load cleanly here.
    config = _config_with_logging(
        tmp_path,
        f'  programs:\n    {SD_OFFSET}:\n      programs:\n        {SD_OFFSET}:\n'
        f'          wibble: NOT_A_LEVEL\n',
    )
    with pytest.raises(ValueError):
        validate_logging_config(config)


# ---------------------------------------------------------------------------
# Level canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [('debug', 'DEBUG'), ('  info  ', 'INFO'), ('WaRnInG', 'WARNING'), ('NONE', 'NONE')],
)
def test_normalize_level(raw: str, expected: str) -> None:
    assert normalize_level(raw) == expected


def test_normalize_level_matches_what_validation_accepts(tmp_path: Path) -> None:
    config = _config_with_logging(tmp_path, "  main: '  debug  '\n")
    validate_logging_config(config)
    assert normalize_level(config.logging['main']) == 'DEBUG'


# ---------------------------------------------------------------------------
# Other-category keys name real components
# ---------------------------------------------------------------------------


def test_other_keys_name_no_dead_module() -> None:
    assert 'nav_correlate_all' not in OTHER_LOG_KEYS


def test_other_keys_include_the_correlation_module() -> None:
    assert 'correlate' in OTHER_LOG_KEYS


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


def test_startup_load_rejects_a_bad_logging_section(tmp_path: Path) -> None:
    override = tmp_path / 'bad.yaml'
    override.write_text('logging:\n  techniques:\n    no_such_technique: DEBUG\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    with pytest.raises(ValueError, match='no_such_technique'):
        load_default_and_user_config(arguments, Config())


def test_startup_load_accepts_a_good_logging_section(tmp_path: Path) -> None:
    override = tmp_path / 'good.yaml'
    override.write_text('logging:\n  techniques:\n    titan_haze: DEBUG\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    config = Config()
    load_default_and_user_config(arguments, config)
    assert config.logging['techniques']['titan_haze'] == 'DEBUG'


# ---------------------------------------------------------------------------
# Program identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('module_name', 'expected'),
    [
        ('sd_offset', SD_OFFSET),
        ('sd_offset_cloud_tasks', SD_OFFSET),
        ('sd_backplanes', 'sd_backplanes'),
        ('sd_backplanes_cloud_tasks', 'sd_backplanes'),
        ('sd_mosaic', SD_MOSAIC),
        ('sd_mosaic_cloud_tasks', SD_MOSAIC),
        ('sd_create_bundle', 'sd_create_bundle'),
        ('sd_consolidate_metadata', 'sd_consolidate_metadata'),
    ],
)
def test_dispatch_module_declares_its_program_identity(module_name: str, expected: str) -> None:
    module = __import__(f'spindoctor.cli.{module_name}', fromlist=['PROGRAM_NAME'])
    declared = module.PROGRAM_NAME
    assert declared == expected


@pytest.mark.parametrize(
    'module_name',
    [
        'sd_offset',
        'sd_offset_cloud_tasks',
        'sd_backplanes',
        'sd_backplanes_cloud_tasks',
        'sd_mosaic',
        'sd_mosaic_cloud_tasks',
        'sd_create_bundle',
        'sd_consolidate_metadata',
    ],
)
def test_declared_program_identity_is_registered(module_name: str) -> None:
    module = __import__(f'spindoctor.cli.{module_name}', fromlist=['PROGRAM_NAME'])
    assert module.PROGRAM_NAME in PROGRAM_NAMES
