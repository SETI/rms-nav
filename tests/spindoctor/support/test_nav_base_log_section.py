"""Tests for per-component log sections and the keys that configure them."""

import tempfile
from collections.abc import Iterator

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.annotation.annotations import Annotations
from spindoctor.config import (
    IMAGE_LOGGER,
    LogLevels,
    LogRole,
    LogSinks,
    build_image_log_handlers,
    image_scope,
    set_log_levels,
)
from spindoctor.nav_model.nav_model_body import NavModelBody
from spindoctor.nav_model.nav_model_rings import NavModelRings
from spindoctor.nav_model.nav_model_rings_simulated import NavModelRingsSimulated
from spindoctor.nav_technique.nav_technique_body_limb import BodyLimbNav
from spindoctor.nav_technique.nav_technique_titan_haze import TitanHazeNav
from spindoctor.support.nav_base import NavBase

_STAMP = '2026-07-29T12-00-00'


@pytest.fixture(autouse=True)
def _restore_levels() -> Iterator[None]:
    """Discard any levels a test installs, so the next one resolves afresh."""
    yield
    set_log_levels(None)


def _emit(levels: LogLevels, component: NavBase, method: str, message: str) -> str:
    """Log one record inside the component's section and return the log text.

    Parameters:
        levels: Levels to install for the run.
        component: The component whose section the record is logged inside.
        method: Name of the level method to call.
        message: Text to log.

    Returns:
        The contents of the per-image log file.
    """
    set_log_levels(levels)
    root = FCPath(tempfile.mkdtemp())
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', LogSinks(log_root=root), levels, timestamp=_STAMP
    )
    logger = pdslogger.PdsLogger.get_logger(f'section_{id(component)}', lognames=False)
    logger.remove_all_handlers()
    with (
        image_scope(logger),
        IMAGE_LOGGER.open('IMAGE', handler=handlers),
        component.log_section('COMPONENT'),
    ):
        getattr(IMAGE_LOGGER, method)(message)
    for handler in handlers:
        handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('component', 'expected'),
    [
        (TitanHazeNav, 'titan_haze'),
        (BodyLimbNav, 'body_limb'),
    ],
)
def test_a_technique_is_keyed_by_its_snake_case_name(
    component: type[NavBase], expected: str
) -> None:
    """A technique is configured under the snake_case form of its class name."""
    assert component().resolved_log_key == expected


def test_a_model_is_keyed_by_its_family() -> None:
    """A model is configured under its family rather than its class."""
    assert NavModelRings.__new__(NavModelRings).resolved_log_key == 'rings'


def test_a_simulated_model_shares_its_siblings_key() -> None:
    """A simulated model is configured with the model it stands in for."""
    real = NavModelRings.__new__(NavModelRings).resolved_log_key
    simulated = NavModelRingsSimulated.__new__(NavModelRingsSimulated).resolved_log_key
    assert real == simulated


def test_a_body_model_is_keyed_by_family() -> None:
    """Every body model shares one key regardless of which body it renders."""
    assert NavModelBody.__new__(NavModelBody).resolved_log_key == 'body'


def test_a_declared_key_wins_over_the_derived_one() -> None:
    """Annotation is configured as "annotate", not as its class name."""
    assert Annotations.__new__(Annotations).resolved_log_key == 'annotate'


def test_a_declared_key_is_inherited() -> None:
    """A family sharing one key declares it once on their base."""

    class _Base(NavBase):
        log_key = 'shared'

    class _Derived(_Base):
        pass

    assert _Derived().resolved_log_key == 'shared'


def test_an_underived_subclass_keys_by_its_own_name() -> None:
    """A subclass with no declared key derives one from its own class name."""

    class _SomeThingNav(NavBase):
        pass

    assert _SomeThingNav().resolved_log_key == 'some_thing'


# ---------------------------------------------------------------------------
# Levels reaching the log
# ---------------------------------------------------------------------------


def test_a_raised_module_writes_below_the_image_level() -> None:
    """A component configured more verbose than the image level actually writes."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit(levels, TitanHazeNav(), 'debug', 'HAZE-DEBUG')
    assert 'HAZE-DEBUG' in text


def test_an_unraised_module_is_unaffected() -> None:
    """Raising one component does not make another verbose."""
    levels = LogLevels(image='INFO', modules={'titan_haze': 'DEBUG'})
    text = _emit(levels, BodyLimbNav(), 'debug', 'LIMB-DEBUG')
    assert 'LIMB-DEBUG' not in text


def test_a_quieted_module_drops_its_lesser_records() -> None:
    """A component configured quieter than the image level is suppressed."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit(levels, BodyLimbNav(), 'info', 'LIMB-INFO')
    assert 'LIMB-INFO' not in text


def test_a_quieted_module_still_reports_errors() -> None:
    """Quieting a component does not lose its errors."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'ERROR'})
    text = _emit(levels, BodyLimbNav(), 'error', 'LIMB-ERROR')
    assert 'LIMB-ERROR' in text


def test_a_module_with_no_override_follows_the_image_level() -> None:
    """A component nobody configured takes the image level."""
    levels = LogLevels(image='DEBUG')
    text = _emit(levels, BodyLimbNav(), 'debug', 'LIMB-DEBUG')
    assert 'LIMB-DEBUG' in text


def test_a_silenced_module_writes_nothing() -> None:
    """A component configured NONE emits nothing, and does not raise doing so."""
    levels = LogLevels(image='INFO', modules={'body_limb': 'NONE'})
    text = _emit(levels, BodyLimbNav(), 'critical', 'LIMB-CRITICAL')
    assert 'LIMB-CRITICAL' not in text


def test_an_explicit_level_overrides_the_configured_one() -> None:
    """A caller passing level= keeps control of its own section."""
    levels = LogLevels(image='ERROR')
    set_log_levels(levels)
    logger = pdslogger.PdsLogger.get_logger('section_explicit', lognames=False)
    logger.remove_all_handlers()
    technique = BodyLimbNav()
    with image_scope(logger), technique.log_section('COMPONENT', level='debug'):
        assert logger.level == 10


# ---------------------------------------------------------------------------
# Role interaction
# ---------------------------------------------------------------------------


def test_a_section_opens_on_the_components_own_logger() -> None:
    """A main-role component sections its own logger, not the image logger."""

    class _RunScoped(NavBase):
        log_role = LogRole.MAIN

    component = _RunScoped()
    with component.log_section('RUN SECTION'):
        pass  # opening it at all is the assertion; a wrong logger would raise
