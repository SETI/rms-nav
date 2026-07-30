"""Tests for logging level resolution, sink selection, and logger construction."""

import argparse
import logging
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.config.config import Config
from spindoctor.config.logging_config import (
    BACKEND_NAMES,
    DEFAULT_LEVEL,
    LogLevels,
    LogSinks,
    build_image_log_handlers,
    build_main_logger,
    image_log_path,
    main_log_path,
    resolve_log_levels,
    sinks_from_arguments,
)
from spindoctor.config.program_names import SD_MOSAIC, SD_OFFSET

_STAMP = '2026-07-29T12-00-00'


def _config(tmp_path: Path, body: str = '') -> Config:
    """Build a Config carrying the shipped defaults plus a logging override.

    Parameters:
        tmp_path: Directory to write the override file into.
        body: YAML text placed under a ``logging:`` key; empty for defaults only.

    Returns:
        The loaded Config.
    """
    config = Config()
    config.read_config()
    if body:
        override = tmp_path / 'override.yaml'
        override.write_text(f'logging:\n{body}')
        config.update_config(str(override))
    return config


def _args(**kwargs: object) -> argparse.Namespace:
    """Build a Namespace carrying only the given logging arguments.

    Parameters:
        **kwargs: Argument names and values to set.

    Returns:
        The populated Namespace.
    """
    return argparse.Namespace(**kwargs)


def _close(handlers: list[object]) -> None:
    """Close any handlers holding an open file.

    Parameters:
        handlers: Handlers returned by a builder.
    """
    for handler in handlers:
        close = getattr(handler, 'close', None)
        if close is not None:
            close()


def _sinks(tmp_path: Path, **kwargs: bool) -> LogSinks:
    """Build a LogSinks rooted at ``tmp_path``.

    Parameters:
        tmp_path: Directory used as the log root.
        **kwargs: Sink flags to override.

    Returns:
        The constructed LogSinks.
    """
    return LogSinks(log_root=FCPath(str(tmp_path)), **kwargs)


# ---------------------------------------------------------------------------
# Level resolution: defaults and configuration
# ---------------------------------------------------------------------------


def test_defaults_resolve_to_info(tmp_path: Path) -> None:
    """With nothing configured, both loggers resolve to the fallback level."""
    levels = resolve_log_levels(SD_OFFSET, None, _config(tmp_path))
    assert levels.main == DEFAULT_LEVEL


def test_image_default_resolves_to_info(tmp_path: Path) -> None:
    """The image logger shares the fallback level."""
    levels = resolve_log_levels(SD_OFFSET, None, _config(tmp_path))
    assert levels.image == DEFAULT_LEVEL


def test_config_sets_the_main_level(tmp_path: Path) -> None:
    """A configured main level is used."""
    levels = resolve_log_levels(SD_OFFSET, None, _config(tmp_path, '  main: WARNING\n'))
    assert levels.main == 'WARNING'


def test_config_level_is_canonicalized(tmp_path: Path) -> None:
    """A lower-case configured level resolves to its canonical spelling."""
    levels = resolve_log_levels(SD_OFFSET, None, _config(tmp_path, '  main: warning\n'))
    assert levels.main == 'WARNING'


def test_module_override_beats_the_image_default(tmp_path: Path) -> None:
    """A named module takes its own level rather than the image default."""
    config = _config(tmp_path, '  image: WARNING\n  techniques:\n    titan_haze: DEBUG\n')
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_unnamed_module_takes_the_image_default(tmp_path: Path) -> None:
    """A module with no override of its own follows the image level."""
    config = _config(tmp_path, '  image: WARNING\n  techniques:\n    titan_haze: DEBUG\n')
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('body_limb') == 'WARNING'


def test_category_default_applies_to_its_members(tmp_path: Path) -> None:
    """A category default governs every module in that category."""
    config = _config(tmp_path, '  techniques:\n    default: DEBUG\n')
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('body_limb') == 'DEBUG'


def test_category_default_does_not_cross_categories(tmp_path: Path) -> None:
    """A technique category default leaves models alone."""
    config = _config(tmp_path, '  techniques:\n    default: DEBUG\n')
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('rings') == DEFAULT_LEVEL


def test_module_override_beats_its_category_default(tmp_path: Path) -> None:
    """A named module wins over the default of its own category."""
    config = _config(tmp_path, '  techniques:\n    default: DEBUG\n    titan_haze: ERROR\n')
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.for_module('titan_haze') == 'ERROR'


def test_shipped_annotation_level_survives_resolution(tmp_path: Path) -> None:
    """The shipped annotation override reaches the resolved levels."""
    levels = resolve_log_levels(SD_OFFSET, None, _config(tmp_path))
    assert levels.for_module('annotate') == 'ERROR'


# ---------------------------------------------------------------------------
# Level resolution: the program merge
# ---------------------------------------------------------------------------


def test_program_block_overrides_the_global_default(tmp_path: Path) -> None:
    """A program's block wins over the top-level value for that program."""
    config = _config(
        tmp_path, f'  main: INFO\n  programs:\n    {SD_MOSAIC}:\n      main: WARNING\n'
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.main == 'WARNING'


def test_program_block_does_not_affect_another_program(tmp_path: Path) -> None:
    """One program's block leaves every other program untouched."""
    config = _config(
        tmp_path, f'  main: INFO\n  programs:\n    {SD_MOSAIC}:\n      main: WARNING\n'
    )
    levels = resolve_log_levels(SD_OFFSET, None, config)
    assert levels.main == 'INFO'


def test_program_block_inherits_unmentioned_keys(tmp_path: Path) -> None:
    """Overriding one key leaves the rest of the global block in force."""
    config = _config(
        tmp_path, f'  image: WARNING\n  programs:\n    {SD_MOSAIC}:\n      main: ERROR\n'
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.image == 'WARNING'


def test_program_block_merges_within_a_category(tmp_path: Path) -> None:
    """A program overriding one module keeps the global entries for the others."""
    config = _config(
        tmp_path,
        '  techniques:\n    titan_haze: DEBUG\n    body_limb: ERROR\n'
        f'  programs:\n    {SD_MOSAIC}:\n      techniques:\n        titan_haze: WARNING\n',
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.for_module('body_limb') == 'ERROR'


def test_program_block_wins_within_a_category(tmp_path: Path) -> None:
    """The program's value for a module wins over the global one."""
    config = _config(
        tmp_path,
        '  techniques:\n    titan_haze: DEBUG\n'
        f'  programs:\n    {SD_MOSAIC}:\n      techniques:\n        titan_haze: WARNING\n',
    )
    levels = resolve_log_levels(SD_MOSAIC, None, config)
    assert levels.for_module('titan_haze') == 'WARNING'


# ---------------------------------------------------------------------------
# Level resolution: command line
# ---------------------------------------------------------------------------


def test_bare_log_level_sets_both_loggers(tmp_path: Path) -> None:
    """A bare --log-level sets the default for the main logger."""
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), _config(tmp_path))
    assert levels.main == 'DEBUG'


def test_bare_log_level_sets_the_image_logger_too(tmp_path: Path) -> None:
    """A bare --log-level sets the default for the image logger as well."""
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), _config(tmp_path))
    assert levels.image == 'DEBUG'


def test_log_level_main_beats_the_bare_form(tmp_path: Path) -> None:
    """The per-logger flag is more specific than the global one."""
    args = _args(log_level=['DEBUG'], log_level_main='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.main == 'ERROR'


def test_log_level_main_leaves_the_image_logger_alone(tmp_path: Path) -> None:
    """Setting the main level does not change the image level."""
    args = _args(log_level=['DEBUG'], log_level_main='ERROR')
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.image == 'DEBUG'


def test_module_form_sets_one_module(tmp_path: Path) -> None:
    """--log-level MODULE=LEVEL sets that module."""
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_module_form_leaves_other_modules_alone(tmp_path: Path) -> None:
    """A per-module flag affects only the module it names."""
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.for_module('body_limb') == DEFAULT_LEVEL


def test_module_form_beats_the_configuration(tmp_path: Path) -> None:
    """A module named on the command line wins over the configured value."""
    config = _config(tmp_path, '  techniques:\n    titan_haze: ERROR\n')
    args = _args(log_level=['titan_haze=DEBUG'])
    levels = resolve_log_levels(SD_OFFSET, args, config)
    assert levels.for_module('titan_haze') == 'DEBUG'


def test_bare_and_module_forms_combine(tmp_path: Path) -> None:
    """The documented "debug everywhere except one" invocation works."""
    args = _args(log_level=['debug', 'titan_haze=info'])
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.for_module('titan_haze') == 'INFO'


def test_bare_form_still_applies_to_other_modules(tmp_path: Path) -> None:
    """The bare form governs every module the per-module form did not name."""
    args = _args(log_level=['debug', 'titan_haze=info'])
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.for_module('body_limb') == 'DEBUG'


def test_a_later_bare_form_wins(tmp_path: Path) -> None:
    """Repeating the bare form takes the last value rather than combining."""
    args = _args(log_level=['debug', 'error'])
    levels = resolve_log_levels(SD_OFFSET, args, _config(tmp_path))
    assert levels.main == 'ERROR'


def test_command_line_beats_a_program_block(tmp_path: Path) -> None:
    """A per-logger flag wins over a per-program configured value."""
    config = _config(tmp_path, f'  programs:\n    {SD_MOSAIC}:\n      main: WARNING\n')
    levels = resolve_log_levels(SD_MOSAIC, _args(log_level_main='ERROR'), config)
    assert levels.main == 'ERROR'


def test_configured_module_beats_a_bare_command_line_level(tmp_path: Path) -> None:
    """A configured module is more specific than a global command-line default."""
    config = _config(tmp_path, '  techniques:\n    titan_haze: ERROR\n')
    levels = resolve_log_levels(SD_OFFSET, _args(log_level=['DEBUG']), config)
    assert levels.for_module('titan_haze') == 'ERROR'


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


def test_default_sinks_put_main_on_the_console(tmp_path: Path) -> None:
    """The main logger writes to the console by default."""
    assert sinks_from_arguments(None, FCPath(str(tmp_path))).main_console is True


def test_default_sinks_keep_image_off_the_console(tmp_path: Path) -> None:
    """The image logger stays off the console by default."""
    assert sinks_from_arguments(None, FCPath(str(tmp_path))).image_console is False


def test_default_sinks_write_both_log_files(tmp_path: Path) -> None:
    """Both loggers write files by default."""
    sinks = sinks_from_arguments(None, FCPath(str(tmp_path)))
    assert (sinks.main_file, sinks.image_file) == (True, True)


@pytest.mark.parametrize(
    ('flag', 'attribute'),
    [
        ('log_main_to_console', 'main_console'),
        ('log_main_to_file', 'main_file'),
        ('log_image_to_console', 'image_console'),
        ('log_image_to_file', 'image_file'),
    ],
)
def test_each_sink_flag_is_honored(tmp_path: Path, flag: str, attribute: str) -> None:
    """Each sink flag independently controls its own sink."""
    sinks = sinks_from_arguments(_args(**{flag: True}), FCPath(str(tmp_path)))
    assert getattr(sinks, attribute) is True


@pytest.mark.parametrize(
    ('flag', 'attribute'),
    [
        ('log_main_to_console', 'main_console'),
        ('log_main_to_file', 'main_file'),
        ('log_image_to_console', 'image_console'),
        ('log_image_to_file', 'image_file'),
    ],
)
def test_each_sink_flag_can_disable(tmp_path: Path, flag: str, attribute: str) -> None:
    """Each sink flag can turn its sink off."""
    sinks = sinks_from_arguments(_args(**{flag: False}), FCPath(str(tmp_path)))
    assert getattr(sinks, attribute) is False


# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------


def test_main_log_path_is_under_the_program_directory(tmp_path: Path) -> None:
    """A main log lands under a directory named for its program."""
    path = main_log_path(FCPath(str(tmp_path)), SD_OFFSET, timestamp=_STAMP)
    assert path.as_posix().endswith(f'{SD_OFFSET}/main_{_STAMP}.log')


@pytest.mark.parametrize('backend', sorted(BACKEND_NAMES))
def test_image_log_path_is_under_the_backend_directory(tmp_path: Path, backend: str) -> None:
    """An image log lands under a directory named for its backend."""
    path = image_log_path(
        FCPath(str(tmp_path)), backend, 'COISS_2001/data/N123_1', timestamp=_STAMP
    )
    assert path.as_posix().endswith(f'{backend}/COISS_2001/data/N123_1_{_STAMP}.log')


def test_image_log_path_rejects_an_unknown_backend(tmp_path: Path) -> None:
    """A backend that is not one of the three is rejected."""
    with pytest.raises(ValueError, match='Unknown backend'):
        image_log_path(FCPath(str(tmp_path)), 'wibble', 'stub', timestamp=_STAMP)


def test_two_backends_do_not_collide_for_one_image(tmp_path: Path) -> None:
    """The same image logged by two backends lands in two distinct files."""
    root = FCPath(str(tmp_path))
    nav = image_log_path(root, 'nav', 'vol/N123', timestamp=_STAMP)
    reproj = image_log_path(root, 'reproj', 'vol/N123', timestamp=_STAMP)
    assert nav.as_posix() != reproj.as_posix()


# ---------------------------------------------------------------------------
# Logger construction
# ---------------------------------------------------------------------------


def test_main_logger_gets_both_handlers(tmp_path: Path) -> None:
    """With both sinks enabled the main logger carries two handlers."""
    logger = pdslogger.PdsLogger('test_main_both', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    count = len(logger.handlers)
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert count == 2


def test_main_logger_writes_to_the_reported_path(tmp_path: Path) -> None:
    """The returned path is the file the main logger actually writes."""
    logger = pdslogger.PdsLogger('test_main_path', lognames=False)
    path = build_main_logger(
        logger, SD_OFFSET, _sinks(tmp_path, main_console=False), LogLevels(), timestamp=_STAMP
    )
    logger.info('a line')
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert path is not None
    assert 'a line' in Path(path.as_posix()).read_text()


def test_main_logger_reports_no_path_without_a_file_sink(tmp_path: Path) -> None:
    """Disabling the file sink reports no path."""
    logger = pdslogger.PdsLogger('test_main_nofile', lognames=False)
    path = build_main_logger(
        logger, SD_OFFSET, _sinks(tmp_path, main_file=False), LogLevels(), timestamp=_STAMP
    )
    assert path is None


def test_rebuilding_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Building twice replaces the handlers rather than accumulating them."""
    logger = pdslogger.PdsLogger('test_main_twice', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(), timestamp=_STAMP)
    count = len(logger.handlers)
    _close(logger.handlers)
    logger.remove_all_handlers()
    assert count == 2


def test_disabling_every_main_sink_attaches_the_null_handler(tmp_path: Path) -> None:
    """A logger with no sink still has a handler, so it cannot fall back to printing."""
    logger = pdslogger.PdsLogger('test_main_null', lognames=False)
    build_main_logger(
        logger,
        SD_OFFSET,
        _sinks(tmp_path, main_console=False, main_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_a_silenced_main_logger_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With every sink off, logging produces no terminal output at all."""
    logger = pdslogger.PdsLogger('test_main_quiet', lognames=False)
    build_main_logger(
        logger,
        SD_OFFSET,
        _sinks(tmp_path, main_console=False, main_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    logger.info('should not appear')
    logger.error('should not appear either')
    assert capsys.readouterr().out == ''


def test_level_none_silences_the_logger(tmp_path: Path) -> None:
    """A level of NONE attaches the null handler rather than a live sink."""
    logger = pdslogger.PdsLogger('test_main_none', lognames=False)
    build_main_logger(logger, SD_OFFSET, _sinks(tmp_path), LogLevels(main='NONE'), timestamp=_STAMP)
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_image_handlers_are_built_for_the_backend(tmp_path: Path) -> None:
    """Image handlers report the backend-scoped path they will write."""
    handlers, path = build_image_log_handlers(
        'nav', 'vol/N123', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
    )
    _close(handlers)
    assert path is not None
    assert '/nav/' in path.as_posix()


def test_image_handlers_default_to_file_only(tmp_path: Path) -> None:
    """With default sinks the image logger gets exactly one handler."""
    handlers, _ = build_image_log_handlers(
        'nav', 'vol/N123', _sinks(tmp_path), LogLevels(), timestamp=_STAMP
    )
    _close(handlers)
    assert len(handlers) == 1


def test_image_handlers_reject_an_unknown_backend(tmp_path: Path) -> None:
    """An unknown backend is rejected even when no file sink is enabled."""
    with pytest.raises(ValueError, match='Unknown backend'):
        build_image_log_handlers(
            'wibble', 'vol/N123', _sinks(tmp_path, image_file=False), LogLevels(), timestamp=_STAMP
        )


def test_image_handlers_fall_back_to_the_null_handler(tmp_path: Path) -> None:
    """An image logger with no sink still gets a handler."""
    handlers, _ = build_image_log_handlers(
        'nav',
        'vol/N123',
        _sinks(tmp_path, image_console=False, image_file=False),
        LogLevels(),
        timestamp=_STAMP,
    )
    assert isinstance(handlers[0], logging.NullHandler)


def test_no_builder_ever_returns_an_empty_handler_list(tmp_path: Path) -> None:
    """Across every sink combination, a logger is never left able to print by fallback."""
    empty = []
    for console in (True, False):
        for to_file in (True, False):
            handlers, _ = build_image_log_handlers(
                'nav',
                'vol/N123',
                _sinks(tmp_path, image_console=console, image_file=to_file),
                LogLevels(),
                timestamp=_STAMP,
            )
            _close(handlers)
            if not handlers:
                empty.append((console, to_file))
    assert empty == []
