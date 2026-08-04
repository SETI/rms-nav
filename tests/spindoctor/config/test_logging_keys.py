"""Tests for the logging configuration key namespace and its validation."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from spindoctor.config.config import Config
from spindoctor.config.config_helper import load_default_and_user_config
from spindoctor.config.logging_config import BACKEND_NAMES
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
from spindoctor.support.nav_base import NavBase


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
    """A technique or model class derives the documented snake_case key."""
    cls = type(class_name, (), {})
    assert log_key_for(cls) == expected


def test_log_key_of_a_real_model_is_its_family() -> None:
    """A model class derives the bare family name."""
    assert log_key_for(type('NavModelRings', (), {})) == 'rings'


def test_log_key_of_a_simulated_model_is_the_same_family() -> None:
    """A simulated model collapses onto its real sibling's key, not a variant of it."""
    assert log_key_for(type('NavModelRingsSimulated', (), {})) == 'rings'


def test_log_key_strips_both_suffixes_in_either_order() -> None:
    """A name carrying Simulated and Nav reduces fully, whichever order they appear in."""
    assert log_key_for(type('FooSimulatedNav', (), {})) == 'foo'


def test_log_key_honors_a_declared_override() -> None:
    """An explicit log_key class attribute wins over the derived value."""
    cls = type('SomethingAwkwardNav', (), {'log_key': 'chosen_name'})
    assert log_key_for(cls) == 'chosen_name'


def test_log_key_honors_an_inherited_override() -> None:
    """A family sharing one key declares it once on their base."""
    parent = type('ParentNav', (), {'log_key': 'parent_key'})
    child = type('BodyLimbNav', (parent,), {})
    assert log_key_for(child) == 'parent_key'


def test_the_config_key_matches_what_the_component_reads() -> None:
    """The configuration namespace and the runtime lookup cannot disagree.

    A key honored at run time but not here would name a component the
    configuration rejects, while the key it did accept governed nothing.
    """
    parent = type('FamilyBaseNav', (NavBase,), {'log_key': 'family'})
    child = type('MemberOneNav', (parent,), {})
    assert log_key_for(child) == child().resolved_log_key


# ---------------------------------------------------------------------------
# Registry-derived key sets
# ---------------------------------------------------------------------------


def test_technique_keys_include_a_known_technique() -> None:
    """A registered autonomous technique appears in the technique key set."""
    assert 'titan_haze' in technique_log_keys()


def test_technique_keys_include_the_interactive_technique() -> None:
    """The interactive technique gets a key despite being outside the autonomous registry."""
    assert 'manual' in technique_log_keys()


def test_technique_keys_are_all_snake_case() -> None:
    """No derived technique key carries an upper-case character."""
    offenders = [key for key in technique_log_keys() if key != key.lower()]
    assert offenders == []


def test_model_keys_include_the_four_families() -> None:
    """The four shipped model families are present in the model key set."""
    assert model_log_keys() >= frozenset({'stars', 'body', 'rings', 'titan'})


def test_shipped_model_classes_map_to_exactly_four_families() -> None:
    """The shipped model classes collapse onto exactly four family keys."""
    # Scoped to shipped classes because the registry is process-global and test
    # modules elsewhere in the suite define their own NavModel subclasses, which
    # self-register and would otherwise make this assertion order-dependent.
    from spindoctor.nav_model.nav_model import NavModel

    shipped = {
        log_key_for(cls) for cls in NavModel._registry if cls.__module__.startswith('spindoctor.')
    }
    assert shipped == frozenset({'stars', 'body', 'rings', 'titan'})


def test_no_backend_key_leaked_into_other() -> None:
    """A per-image backend is not addressable as an "other" module key."""
    assert OTHER_LOG_KEYS.isdisjoint(BACKEND_NAMES)


# ---------------------------------------------------------------------------
# Shipped defaults
# ---------------------------------------------------------------------------


def test_shipped_defaults_define_a_logging_section() -> None:
    """The bundled defaults ship a non-empty logging section."""
    config = Config()
    config.read_config()
    assert config.logging != {}


def test_shipped_defaults_are_valid() -> None:
    """The bundled defaults pass their own validation."""
    config = Config()
    config.read_config()
    validate_logging_config(config)


def test_shipped_defaults_preserve_the_annotation_level() -> None:
    """The shipped annotation level matches the behavior it replaces."""
    config = Config()
    config.read_config()
    assert config.logging['other']['annotate'] == 'ERROR'


def test_shipped_defaults_disable_strict_scope() -> None:
    """Strict scope is off by default, so production warns rather than raising."""
    config = Config()
    config.read_config()
    assert config.logging['strict_scope'] is False


@pytest.mark.parametrize('logger_key', sorted(LOGGER_KEYS))
def test_shipped_defaults_define_each_logger_default(logger_key: str) -> None:
    """Each logger has a shipped default level."""
    config = Config()
    config.read_config()
    assert config.logging[logger_key] == 'INFO'


@pytest.mark.parametrize('category', sorted(CATEGORY_KEYS))
def test_shipped_defaults_define_each_category(category: str) -> None:
    """Each override category is present in the shipped defaults."""
    config = Config()
    config.read_config()
    assert category in config.logging


# ---------------------------------------------------------------------------
# Validation: accepted input
# ---------------------------------------------------------------------------


def test_accepts_a_technique_override(tmp_path: Path) -> None:
    """A per-technique level override validates."""
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_haze: DEBUG\n')
    validate_logging_config(config)


def test_accepts_a_model_override(tmp_path: Path) -> None:
    """A per-model level override validates."""
    config = _config_with_logging(tmp_path, '  models:\n    rings: WARNING\n')
    validate_logging_config(config)


def test_accepts_a_program_block(tmp_path: Path) -> None:
    """A per-program block setting both logger defaults validates."""
    config = _config_with_logging(
        tmp_path, f'  programs:\n    {SD_MOSAIC}:\n      main: WARNING\n      image: DEBUG\n'
    )
    validate_logging_config(config)


@pytest.mark.parametrize('level', sorted(LOG_LEVEL_NAMES))
def test_accepts_every_level_name(tmp_path: Path, level: str) -> None:
    """Every documented level name is accepted."""
    config = _config_with_logging(tmp_path, f'  main: {level}\n')
    validate_logging_config(config)


def test_accepts_a_lowercase_level_name(tmp_path: Path) -> None:
    """Level names are accepted in lower case."""
    config = _config_with_logging(tmp_path, '  main: debug\n')
    validate_logging_config(config)


# ---------------------------------------------------------------------------
# Validation: rejected input
# ---------------------------------------------------------------------------


def test_rejects_an_unknown_top_level_key(tmp_path: Path) -> None:
    """An unknown key in the logging section is rejected by dotted path."""
    config = _config_with_logging(tmp_path, '  wibble: INFO\n')
    with pytest.raises(ValueError, match=r'logging\.wibble'):
        validate_logging_config(config)


def test_rejects_an_unknown_technique_key(tmp_path: Path) -> None:
    """A misspelled technique key is rejected rather than silently ignored."""
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_hazey: DEBUG\n')
    with pytest.raises(ValueError, match='titan_hazey'):
        validate_logging_config(config)


def test_unknown_technique_error_names_the_category(tmp_path: Path) -> None:
    """The rejection message names the category whose keys were expected."""
    config = _config_with_logging(tmp_path, '  techniques:\n    titan_hazey: DEBUG\n')
    with pytest.raises(ValueError, match='techniques key'):
        validate_logging_config(config)


def test_rejects_a_backend_named_as_a_module(tmp_path: Path) -> None:
    """A backend name is not accepted as an "other" module key."""
    config = _config_with_logging(tmp_path, '  other:\n    reproj: DEBUG\n')
    with pytest.raises(ValueError, match='reproj'):
        validate_logging_config(config)


def test_rejects_an_unknown_level_name(tmp_path: Path) -> None:
    """A level name outside the documented set is rejected."""
    config = _config_with_logging(tmp_path, '  main: CHATTY\n')
    with pytest.raises(ValueError, match='CHATTY'):
        validate_logging_config(config)


def test_rejects_a_non_string_level(tmp_path: Path) -> None:
    """A non-string level value is rejected with a type-specific message."""
    config = _config_with_logging(tmp_path, '  main: 20\n')
    with pytest.raises(ValueError, match='level name string'):
        validate_logging_config(config)


def test_rejects_an_unknown_program(tmp_path: Path) -> None:
    """A program block naming an unregistered program is rejected."""
    config = _config_with_logging(tmp_path, '  programs:\n    sd_nonesuch:\n      main: INFO\n')
    with pytest.raises(ValueError, match='sd_nonesuch'):
        validate_logging_config(config)


def test_rejects_strict_scope_inside_a_program_block(tmp_path: Path) -> None:
    """Strict scope is global and cannot be set for one program."""
    config = _config_with_logging(
        tmp_path, f'  programs:\n    {SD_OFFSET}:\n      strict_scope: true\n'
    )
    with pytest.raises(ValueError, match='global setting'):
        validate_logging_config(config)


def test_rejects_a_non_boolean_strict_scope(tmp_path: Path) -> None:
    """A non-boolean strict_scope is rejected."""
    config = _config_with_logging(tmp_path, '  strict_scope: yes please\n')
    with pytest.raises(ValueError, match='true or false'):
        validate_logging_config(config)


def test_rejects_a_scalar_category(tmp_path: Path) -> None:
    """A category written as a scalar rather than a mapping is rejected."""
    config = _config_with_logging(tmp_path, '  techniques: DEBUG\n')
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(config)


def test_rejects_a_scalar_program_block(tmp_path: Path) -> None:
    """A program block written as a scalar rather than a mapping is rejected."""
    config = _config_with_logging(tmp_path, f'  programs:\n    {SD_OFFSET}: DEBUG\n')
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(config)


def test_rejects_a_non_mapping_logging_section() -> None:
    """A logging section that is not a mapping is rejected rather than treated as absent.

    Loading such a section through ``update_config`` is already refused one
    layer down, so this exercises the guard directly on behalf of callers that
    validate a Config they assembled themselves.
    """
    stub = cast(Config, SimpleNamespace(logging=[]))
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(stub)


def test_non_mapping_logging_section_is_refused_at_load(tmp_path: Path) -> None:
    """A logging section written as a sequence fails when the override is merged."""
    override = tmp_path / 'override.yaml'
    override.write_text('logging: []\n')
    config = Config()
    config.read_config()
    with pytest.raises(ValueError, match='expected a mapping'):
        config.update_config(str(override))


def test_rejects_a_non_mapping_programs_block(tmp_path: Path) -> None:
    """A programs block written as a sequence is rejected, not treated as empty."""
    config = _config_with_logging(tmp_path, '  programs: []\n')
    with pytest.raises(ValueError, match='must be a mapping'):
        validate_logging_config(config)


def test_rejects_a_programs_block_nested_in_a_program(tmp_path: Path) -> None:
    """Per-program overrides do not nest inside one another."""
    config = _config_with_logging(
        tmp_path,
        f'  programs:\n    {SD_OFFSET}:\n      programs:\n        {SD_OFFSET}:\n'
        f'          main: INFO\n',
    )
    with pytest.raises(ValueError, match='does not nest'):
        validate_logging_config(config)


def test_nesting_cannot_smuggle_past_an_unknown_key(tmp_path: Path) -> None:
    """A key rejected at the top level is not accepted by burying it one level deeper."""
    # Without the nesting guard the inner block is skipped entirely, so keys
    # rejected one level up would load cleanly here.
    config = _config_with_logging(
        tmp_path,
        f'  programs:\n    {SD_OFFSET}:\n      programs:\n        {SD_OFFSET}:\n'
        f'          wibble: NOT_A_LEVEL\n',
    )
    with pytest.raises(ValueError, match='programs'):
        validate_logging_config(config)


# ---------------------------------------------------------------------------
# Level canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [('debug', 'DEBUG'), ('  info  ', 'INFO'), ('WaRnInG', 'WARNING'), ('NONE', 'NONE')],
)
def test_normalize_level(raw: str, expected: str) -> None:
    """Level names canonicalize to upper case with surrounding whitespace removed."""
    assert normalize_level(raw) == expected


def test_normalize_level_matches_what_validation_accepts(tmp_path: Path) -> None:
    """A level validation accepts canonicalizes to a documented name."""
    config = _config_with_logging(tmp_path, "  main: '  debug  '\n")
    validate_logging_config(config)
    assert normalize_level(config.logging['main']) == 'DEBUG'


# ---------------------------------------------------------------------------
# Other-category keys name real components
# ---------------------------------------------------------------------------


def test_other_keys_name_no_dead_module() -> None:
    """The other category carries no key naming a module that does not exist."""
    assert 'nav_correlate_all' not in OTHER_LOG_KEYS


def test_other_keys_include_the_annotation_module() -> None:
    """Annotation opens its own section, so it is addressable."""
    assert 'annotate' in OTHER_LOG_KEYS


def test_every_other_key_names_a_component_that_opens_a_section() -> None:
    """A key naming a component that never opens a section would do nothing.

    Guards the invariant directly rather than restating a list: each key must
    appear either in a ``logged_section`` call or as a declared ``log_key``,
    which are the two ways a component gets a section for its level to be
    applied at.
    """
    import spindoctor

    root = Path(spindoctor.__file__).parent
    sources = '\n'.join(f.read_text() for f in root.rglob('*.py'))
    missing = [
        key
        for key in sorted(OTHER_LOG_KEYS)
        if f"logged_section('{key}'" not in sources
        and f"log_key: ClassVar[str] = '{key}'" not in sources
    ]
    assert missing == []


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


def test_startup_load_names_the_section_for_an_empty_override(tmp_path: Path) -> None:
    """A section written with no body is reported by name, not as an AttributeError."""
    override = tmp_path / 'empty_section.yaml'
    override.write_text('general:\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    with pytest.raises(ValueError, match='general'):
        load_default_and_user_config(arguments, Config())


def test_startup_load_tolerates_a_namespace_without_config_file() -> None:
    """A bare Namespace is accepted, since callers legitimately omit config_file."""
    config = Config()
    load_default_and_user_config(argparse.Namespace(), config)
    assert config.logging != {}


def test_startup_load_rejects_a_bad_logging_section(tmp_path: Path) -> None:
    """A bad logging section fails when the configuration is loaded, not later."""
    override = tmp_path / 'bad.yaml'
    override.write_text('logging:\n  techniques:\n    no_such_technique: DEBUG\n')
    arguments = argparse.Namespace(config_file=[str(override)])
    with pytest.raises(ValueError, match='no_such_technique'):
        load_default_and_user_config(arguments, Config())


def test_startup_load_accepts_a_good_logging_section(tmp_path: Path) -> None:
    """A valid override survives the startup load and reaches the merged config."""
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
    """Each dispatch module declares the program identity it shares logs and config with."""
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
    """Every declared identity is one validation will accept."""
    module = __import__(f'spindoctor.cli.{module_name}', fromlist=['PROGRAM_NAME'])
    assert module.PROGRAM_NAME in PROGRAM_NAMES
