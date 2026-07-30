"""Tests for image-scope routing, logger roles, and scope enforcement."""

from collections.abc import Iterator

import pdslogger
import pytest

from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER, LogRole, LogScopeError
from spindoctor.config.log_scope import (
    _reset_reported_call_sites,
    image_scope,
    image_scope_is_open,
    set_strict_scope,
    strict_scope,
)
from spindoctor.support.nav_base import NavBase


@pytest.fixture(autouse=True)
def _isolate_scope_state() -> Iterator[None]:
    """Restore the strict-scope switch and warning memory around each test.

    The warning is deduplicated per call site for the life of the process, so
    without this reset a test asserting on the warning would pass or fail
    depending on whether an earlier test had already reported the same line.
    """
    previous = strict_scope()
    _reset_reported_call_sites()
    yield
    set_strict_scope(previous)
    _reset_reported_call_sites()


@pytest.fixture
def recording_logger() -> Iterator[tuple[pdslogger.PdsLogger, list[str]]]:
    """Provide a logger capturing its records into a list.

    Yields:
        Tuple of the logger and the list its records accumulate in.
    """
    import logging

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = pdslogger.PdsLogger.get_logger('test_scope_target', lognames=False)
    logger.remove_all_handlers()
    logger.add_handler(_Capture())
    yield logger, records
    logger.remove_all_handlers()


# ---------------------------------------------------------------------------
# Scope state
# ---------------------------------------------------------------------------


def test_no_scope_is_open_by_default() -> None:
    """Outside any image, no scope is reported open."""
    assert image_scope_is_open() is False


def test_a_scope_reports_itself_open() -> None:
    """Inside an image scope, the scope reports open."""
    with image_scope():
        assert image_scope_is_open() is True


def test_a_scope_closes_on_exit() -> None:
    """Leaving an image scope restores the outside state."""
    with image_scope():
        pass
    assert image_scope_is_open() is False


def test_a_scope_closes_after_an_exception() -> None:
    """An exception inside a scope does not leave it open."""
    with pytest.raises(RuntimeError), image_scope():
        raise RuntimeError('boom')
    assert image_scope_is_open() is False


def test_a_nested_scope_keeps_the_outer_logger(
    recording_logger: tuple[pdslogger.PdsLogger, list[str]],
) -> None:
    """An inner scope does not displace the image the outer one established."""
    logger, _ = recording_logger
    with image_scope(logger) as outer, image_scope() as inner:
        assert inner is outer


def test_leaving_a_nested_scope_leaves_the_outer_open(
    recording_logger: tuple[pdslogger.PdsLogger, list[str]],
) -> None:
    """Closing an inner scope does not end the image."""
    logger, _ = recording_logger
    with image_scope(logger):
        with image_scope():
            pass
        assert image_scope_is_open() is True


# ---------------------------------------------------------------------------
# Routing through the proxy
# ---------------------------------------------------------------------------


def test_a_record_in_scope_reaches_the_scope_logger(
    recording_logger: tuple[pdslogger.PdsLogger, list[str]],
) -> None:
    """Inside a scope, image records go to that scope's logger."""
    logger, records = recording_logger
    with image_scope(logger):
        IMAGE_LOGGER.info('in scope')
    assert any('in scope' in record for record in records)


def test_opening_a_section_on_the_proxy_enters_a_scope() -> None:
    """The outermost open on the image logger is what establishes the scope."""
    with IMAGE_LOGGER.open('IMAGE'):
        assert image_scope_is_open() is True


def test_a_section_opened_on_the_proxy_closes_its_scope() -> None:
    """Leaving the outermost section ends the image scope."""
    with IMAGE_LOGGER.open('IMAGE'):
        pass
    assert image_scope_is_open() is False


def test_nested_sections_stay_in_one_scope(
    recording_logger: tuple[pdslogger.PdsLogger, list[str]],
) -> None:
    """A model or technique section writes into the image already open."""
    logger, records = recording_logger
    with image_scope(logger), IMAGE_LOGGER.open('TECHNIQUE: titan_haze'):
        IMAGE_LOGGER.info('technique detail')
    assert any('technique detail' in record for record in records)


# ---------------------------------------------------------------------------
# Out-of-scope handling
# ---------------------------------------------------------------------------


def test_an_out_of_scope_record_is_not_lost(capsys: pytest.CaptureFixture[str]) -> None:
    """A record logged with no scope open still reaches the main logger."""
    set_strict_scope(False)
    IMAGE_LOGGER.info('orphan record')
    assert 'orphan record' in capsys.readouterr().out


def test_an_out_of_scope_record_warns(capsys: pytest.CaptureFixture[str]) -> None:
    """The warning names the offending call site."""
    set_strict_scope(False)
    IMAGE_LOGGER.info('orphan record')
    assert 'no image scope open' in capsys.readouterr().out


def test_the_warning_names_the_calling_module(capsys: pytest.CaptureFixture[str]) -> None:
    """The report identifies where the stray call came from, not the proxy."""
    set_strict_scope(False)
    IMAGE_LOGGER.info('orphan record')
    assert 'test_log_scope' in capsys.readouterr().out


def test_the_warning_is_reported_once_per_call_site(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A loop logging out of scope does not flood the log with warnings."""
    set_strict_scope(False)
    for _ in range(5):
        IMAGE_LOGGER.info('orphan record')
    assert capsys.readouterr().out.count('no image scope open') == 1


def test_strict_scope_raises_instead(capsys: pytest.CaptureFixture[str]) -> None:
    """With strict scope on, an out-of-scope record is an error."""
    set_strict_scope(True)
    with pytest.raises(LogScopeError, match='no image scope open'):
        IMAGE_LOGGER.info('orphan record')


def test_strict_scope_names_the_offender() -> None:
    """The raised error identifies the call site so it can be fixed."""
    set_strict_scope(True)
    with pytest.raises(LogScopeError, match='test_log_scope'):
        IMAGE_LOGGER.warning('orphan record')


def test_strict_scope_allows_an_in_scope_record(
    recording_logger: tuple[pdslogger.PdsLogger, list[str]],
) -> None:
    """Strict scope does not interfere with correctly scoped logging."""
    logger, records = recording_logger
    set_strict_scope(True)
    with image_scope(logger):
        IMAGE_LOGGER.info('properly scoped')
    assert any('properly scoped' in record for record in records)


# ---------------------------------------------------------------------------
# Role binding
# ---------------------------------------------------------------------------


def test_the_default_role_is_image() -> None:
    """A navigation component logs to the image logger unless it says otherwise."""
    assert NavBase.log_role is LogRole.IMAGE


def test_an_image_role_component_gets_the_proxy() -> None:
    """An image-role component holds the image logger."""
    assert NavBase().logger is IMAGE_LOGGER


def test_a_main_role_component_gets_the_main_logger() -> None:
    """Declaring the main role binds the component to the run's logger."""

    class _RunScoped(NavBase):
        log_role = LogRole.MAIN

    assert _RunScoped().logger is MAIN_LOGGER


def test_the_opt_in_fixture_enables_strict_scope(strict_log_scope: None) -> None:
    """Requesting the fixture makes an out-of-scope image log raise."""
    with pytest.raises(LogScopeError):
        IMAGE_LOGGER.info('orphan record')


def test_the_suite_is_permissive_by_default() -> None:
    """Without the fixture the suite matches production and only warns."""
    assert strict_scope() is False


def test_the_dataset_is_main_role() -> None:
    """Enumeration spans the run, so DataSet logs to the main logger."""
    from spindoctor.dataset.dataset import DataSet

    assert DataSet.log_role is LogRole.MAIN


def test_a_main_role_component_never_trips_scope_enforcement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A main-role component logs outside any image without being reported."""

    class _RunScoped(NavBase):
        log_role = LogRole.MAIN

    set_strict_scope(True)
    _RunScoped().logger.info('run-level record')
    assert 'no image scope open' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Programs that carry no logger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'module_name',
    [
        'spindoctor.cli.stats.ingest',
        'spindoctor.cli.stats.report',
        'spindoctor.cli.sd_backplane_viewer',
        'spindoctor.ui.mosaic_viewer.common',
        'spindoctor.ui.mosaic_viewer.body_window',
        'spindoctor.ui.mosaic_viewer.ring_window',
    ],
)
def test_no_logger_survives_in_a_print_only_module(module_name: str) -> None:
    """The statistics and GUI modules hold no logger to trip scope enforcement."""
    import importlib

    module = importlib.import_module(module_name)
    offenders = [
        name
        for name in ('IMAGE_LOGGER', 'MAIN_LOGGER', 'logger')
        if getattr(module, name, None) is not None
    ]
    assert offenders == []
