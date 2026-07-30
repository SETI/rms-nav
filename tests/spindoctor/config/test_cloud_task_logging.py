"""Tests that a cloud task writes nothing to the terminal its worker owns."""

import argparse
import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER, image_scope
from spindoctor.config.config import Config
from spindoctor.config.logging_config import (
    LogLevels,
    LogSinks,
    RunLogging,
    build_cloud_task_logging,
    build_image_log_handlers,
    isolate_cloud_task_logging,
    log_levels,
    resolve_log_levels,
)
from spindoctor.config.program_names import SD_OFFSET

_STAMP = '2026-07-29T12-00-00'


@pytest.fixture
def root_sink() -> Iterator[io.StringIO]:
    """Stand in for the root handler cloud_tasks installs in each worker.

    ``cloud_tasks`` calls ``logging.basicConfig`` inside every worker
    subprocess, which attaches a handler to the root logger.  Anything that
    propagates there is re-emitted on the worker's terminal, so a test for
    isolation needs somewhere to observe that happening.

    Yields:
        The stream the stand-in root handler writes to.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    saved_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)
        handler.close()


def _config() -> Config:
    """Build a Config carrying the shipped defaults.

    Returns:
        The loaded Config.
    """
    config = Config()
    config.read_config()
    return config


def _args(**kwargs: object) -> argparse.Namespace:
    """Build a Namespace carrying only the given logging arguments.

    Parameters:
        **kwargs: Argument names and values to set.

    Returns:
        The populated Namespace.
    """
    return argparse.Namespace(**kwargs)


def _task_logging(tmp_path: Path, **kwargs: object) -> RunLogging:
    """Resolve one task's logging with the log root named explicitly.

    ``--log-root`` is the most specific source, so naming it here keeps the
    result from depending on whether the machine running the tests happens to
    export ``NAV_RESULTS_ROOT``.

    Parameters:
        tmp_path: Directory to use as the log root.
        **kwargs: Further logging arguments to pass.

    Returns:
        The resolved :class:`RunLogging`.
    """
    return build_cloud_task_logging(SD_OFFSET, _args(log_root=str(tmp_path), **kwargs), _config())


def _log_one_image(run_logging: RunLogging, message: str) -> FCPath | None:
    """Log one record the way a per-image backend does, and return the log path.

    The handlers have to be attached to the section, not merely built: a
    section opened without them leaves the image logger to fall back on
    printing, which would make a test for what reaches the terminal pass for
    the wrong reason.

    Parameters:
        run_logging: The task's resolved logging.
        message: Text to log inside the image's section.

    Returns:
        The path written to, or None when no file sink is enabled.
    """
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', run_logging.sinks, run_logging.levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('IMAGE', handler=handlers):
            IMAGE_LOGGER.info(message)
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    return path


# ---------------------------------------------------------------------------
# The two paths to the terminal
# ---------------------------------------------------------------------------


def test_the_main_logger_keeps_a_sink(tmp_path: Path) -> None:
    """The main logger is never left with no handlers at all.

    A handler-less PdsLogger prints every record to stdout regardless of level,
    so "no main logger" has to mean a null sink rather than no sinks.
    """
    isolate_cloud_task_logging()
    assert MAIN_LOGGER.handlers == [pdslogger.NULL_HANDLER]


def test_a_main_record_reaches_no_terminal(capsys: pytest.CaptureFixture[str]) -> None:
    """A surviving main-logger call site is inert rather than printed."""
    isolate_cloud_task_logging()
    MAIN_LOGGER.info('MAIN-CANARY')
    assert 'MAIN-CANARY' not in capsys.readouterr().out


def test_a_main_record_does_not_propagate(root_sink: io.StringIO) -> None:
    """A main record is not re-emitted by the worker's root handler."""
    isolate_cloud_task_logging()
    MAIN_LOGGER.info('MAIN-CANARY')
    assert 'MAIN-CANARY' not in root_sink.getvalue()


def test_an_image_record_does_not_propagate(tmp_path: Path, root_sink: io.StringIO) -> None:
    """An image record is not re-emitted by the worker's root handler."""
    _log_one_image(_task_logging(tmp_path), 'IMAGE-CANARY')
    assert 'IMAGE-CANARY' not in root_sink.getvalue()


def test_an_image_record_reaches_no_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An image record does not reach stdout either."""
    _log_one_image(_task_logging(tmp_path), 'IMAGE-CANARY')
    assert 'IMAGE-CANARY' not in capsys.readouterr().out


def test_an_image_record_outside_a_section_reaches_no_terminal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Between one image and the next the image logger still has a sink.

    Its section handlers are attached for the duration of a section and gone
    afterwards, and a PdsLogger with none prints rather than going quiet.
    """
    _task_logging(tmp_path)
    with image_scope():
        IMAGE_LOGGER.info('BETWEEN-IMAGES-CANARY')
    assert 'BETWEEN-IMAGES-CANARY' not in capsys.readouterr().out


def test_an_out_of_scope_image_record_reaches_no_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The main logger is silent even for records routed to it from elsewhere.

    An image-scoped component logging with no scope open is redirected to the
    main logger, which would print it if the null sink were not in place.
    """
    isolate_cloud_task_logging()
    IMAGE_LOGGER.warning('OUT-OF-SCOPE-CANARY')
    assert 'OUT-OF-SCOPE-CANARY' not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_the_image_console_is_refused(tmp_path: Path) -> None:
    """--log-image-to-console cannot open a console a cloud task must not use."""
    assert _task_logging(tmp_path, log_image_to_console=True).sinks.image_console is False


def test_the_main_console_is_refused(tmp_path: Path) -> None:
    """--log-main-to-console likewise cannot reach the worker's terminal."""
    assert _task_logging(tmp_path, log_main_to_console=True).sinks.main_console is False


def test_no_main_log_file_is_written(tmp_path: Path) -> None:
    """A task writes no main log, so concurrent workers cannot collide on one."""
    assert _task_logging(tmp_path).main_log_path is None


def test_the_image_file_sink_survives(tmp_path: Path) -> None:
    """Per-image logs are still written; only the terminal is closed off."""
    assert _task_logging(tmp_path).sinks.image_file is True


def test_an_image_log_file_is_written(tmp_path: Path) -> None:
    """The image's log reaches its file rather than being suppressed with the rest."""
    path = _log_one_image(_task_logging(tmp_path), 'IMAGE-CANARY')
    assert path is not None
    with path.open('r') as stream:
        assert 'IMAGE-CANARY' in stream.read()


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


def test_levels_match_the_interactive_driver(tmp_path: Path) -> None:
    """An image's log reads the same whichever driver produced it."""
    arguments = _args(log_root=str(tmp_path), log_level=['DEBUG', 'titan_haze=ERROR'])
    config = _config()
    run_logging = build_cloud_task_logging(SD_OFFSET, arguments, config)
    assert run_logging.levels == resolve_log_levels(SD_OFFSET, arguments, config)


def test_a_module_level_still_applies(tmp_path: Path) -> None:
    """Per-module configuration is not lost to the isolation."""
    run_logging = _task_logging(tmp_path, log_level=['titan_haze=ERROR'])
    assert run_logging.levels.for_module('titan_haze') == 'ERROR'


# ---------------------------------------------------------------------------
# Repeated application
# ---------------------------------------------------------------------------


def test_isolation_is_idempotent() -> None:
    """Applying isolation per task does not accumulate handlers."""
    isolate_cloud_task_logging()
    isolate_cloud_task_logging()
    isolate_cloud_task_logging()
    assert MAIN_LOGGER.handlers == [pdslogger.NULL_HANDLER]


def test_isolation_replaces_an_existing_main_sink(tmp_path: Path) -> None:
    """A main logger left configured by something else is taken over."""
    MAIN_LOGGER.add_handler(pdslogger.file_handler(tmp_path / 'stale.log', level='info'))
    isolate_cloud_task_logging()
    assert MAIN_LOGGER.handlers == [pdslogger.NULL_HANDLER]


def test_an_unresolvable_log_root_reaches_no_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Failing to find a log root is reported into the null sink, not printed.

    The warning is emitted by the resolver itself, so isolation has to be in
    place before resolution rather than applied to its result.
    """
    for name in ('NAV_LOG_ROOT', 'NAV_RESULTS_ROOT'):
        monkeypatch.delenv(name, raising=False)
    build_cloud_task_logging(SD_OFFSET, _args(), _config())
    assert 'No log root' not in capsys.readouterr().out


def test_an_unresolvable_log_root_drops_the_image_file_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nowhere to write, a task runs without log files rather than failing."""
    for name in ('NAV_LOG_ROOT', 'NAV_RESULTS_ROOT'):
        monkeypatch.delenv(name, raising=False)
    run_logging = build_cloud_task_logging(SD_OFFSET, _args(), _config())
    assert run_logging.sinks.image_file is False


def test_the_levels_are_installed_for_components(tmp_path: Path) -> None:
    """Components read the same levels the handlers were built at."""
    _task_logging(tmp_path, log_level=['body_limb=ERROR'])
    assert log_levels().for_module('body_limb') == 'ERROR'


def test_isolation_alone_does_not_disturb_resolved_levels() -> None:
    """Isolation configures sinks; it does not change what any module logs at."""
    isolate_cloud_task_logging()
    assert resolve_log_levels(SD_OFFSET, _args(), _config()) == LogLevels(
        main='INFO', image='INFO', modules={'annotate': 'ERROR'}
    )


def test_the_image_logger_keeps_a_sink() -> None:
    """The image logger is bound to a null sink too, not only the main one."""
    isolate_cloud_task_logging()
    assert IMAGE_LOGGER.handlers == [pdslogger.NULL_HANDLER]


def test_isolating_twice_does_not_double_the_image_sink() -> None:
    """Applying isolation per task does not accumulate image handlers either."""
    isolate_cloud_task_logging()
    isolate_cloud_task_logging()
    assert IMAGE_LOGGER.handlers == [pdslogger.NULL_HANDLER]


# ---------------------------------------------------------------------------
# A logger configured off
# ---------------------------------------------------------------------------


def test_an_image_section_can_be_opened_when_the_image_logger_is_off() -> None:
    """``NONE`` is a level the configuration accepts, so a section must take it.

    pdslogger has no level of that name and rejects it outright, so the value
    passed to ``open`` is the numeric silent level instead.
    """
    logger = pdslogger.PdsLogger.get_logger('cloud_task_none', lognames=False)
    logger.add_handler(pdslogger.NULL_HANDLER)
    with (
        image_scope(logger),
        IMAGE_LOGGER.open('IMAGE', level=LogLevels(image='NONE').image_section_level()),
    ):
        assert True


def test_an_image_logger_configured_off_writes_nothing(tmp_path: Path) -> None:
    """A section opened at the silent level emits none of its records."""
    levels = LogLevels(image='NONE')
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N1', LogSinks(log_root=FCPath(str(tmp_path))), levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('IMAGE', handler=handlers, level=levels.image_section_level()):
            IMAGE_LOGGER.critical('SILENCED-CANARY')
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is None
