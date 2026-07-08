"""Logger instances and setup for the SpinDoctor pipeline.

Provides two PdsLogger instances:

- ``MAIN_LOGGER`` (``"sd_offset"``) -- top-level program events, written to stdout
  and to a timestamped logfile under ``NAV_RESULTS_ROOT/logs/sd_offset/``.
- ``IMAGE_LOGGER`` (``"nav_image"``) -- per-image processing events; both its stdout
  handler and its per-image logfile handler are attached as local handlers inside each
  ``logger.open()`` context so they are active only while that image is being processed.

Call ``setup_logging()`` from ``main()`` after the nav-results root and CLI
arguments have been resolved. It is safe to call more than once: existing
``MAIN_LOGGER`` handlers are removed before new ones are attached.
"""

import argparse

# The stdlib ``logging`` import below is type-annotation-only: it supplies
# ``logging.Handler`` / ``logging.FileHandler`` for the pdslogger-backed
# handlers in this module.  spindoctor.config is intentionally exempt from the "no
# stdlib logging in core code" rule (which targets spindoctor.feature / nav_model /
# nav_orchestrator / nav_technique / spindoctor.support); all runtime logging here
# goes through pdslogger.
import logging
from typing import TYPE_CHECKING, cast

import pdslogger
from filecache import FCPath

if TYPE_CHECKING:
    from .config import Config

MAIN_LOGGER = pdslogger.PdsLogger('sd_offset', lognames=False, digits=3)
IMAGE_LOGGER = pdslogger.PdsLogger('nav_image', lognames=False, digits=3)

_FALLBACK_LEVEL = 'INFO'
_ALLOWED_LOG_LEVELS = frozenset({'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'})


def _resolve_level(attr_name: str, arguments: argparse.Namespace | None, config: 'Config') -> str:
    """Return the log level for ``attr_name``, checking args then config then fallback.

    Parameters:
        attr_name: Name of the config key and argparse attribute (underscore form).
        arguments: Parsed CLI arguments, or None to skip the args check.
        config: Navigation configuration.

    Returns:
        A log-level string (e.g. ``"INFO"``).

    Raises:
        TypeError: If a non-``None`` value from arguments or config is not a string.
        ValueError: If the string is empty after stripping or is not a known level name.
    """
    level: object | None
    if arguments is not None:
        level = getattr(arguments, attr_name, None)
    else:
        level = None
    if level is None:
        level = getattr(config.general, attr_name, None)
    if level is None:
        return _FALLBACK_LEVEL
    if not isinstance(level, str):
        raise TypeError(f'log level {attr_name!r} must be str or None, got {type(level).__name__}')
    normalized = level.strip().upper()
    if normalized == '':
        raise ValueError(f'log level {attr_name!r} is empty or whitespace only')
    if normalized not in _ALLOWED_LOG_LEVELS:
        raise ValueError(
            f'log level {attr_name!r} must be one of '
            f'{sorted(_ALLOWED_LOG_LEVELS)}, got {normalized!r}'
        )
    return normalized


def setup_logging(
    arguments: argparse.Namespace,
    config: 'Config',
    nav_results_root_str: str,
) -> None:
    """Configure MAIN_LOGGER with stdout and a timestamped file handler.

    Reads each log level from ``arguments`` first, then from ``config.general``,
    then falls back to ``"INFO"``.  The main logfile is written as a timestamped
    file under ``{nav_results_root}/logs/sd_offset/``.

    IMAGE_LOGGER handlers are **not** configured here; both its console and per-image
    file handlers are attached as local handlers inside each ``IMAGE_LOGGER.open()``
    context via ``image_log_handlers()``.  However, the image log levels from
    ``--log-level-image-console`` and ``--log-level-image-file`` are validated here
    so that invalid values are caught at startup (before the batch loop) alongside
    the main-level validation.

    Parameters:
        arguments: Parsed CLI arguments; may carry ``log_level_main_console``,
            ``log_level_main_file``, ``log_level_image_console``, and
            ``log_level_image_file`` attributes.
        config: Navigation configuration providing ``general.*`` fallback values.
        nav_results_root_str: Local filesystem path to the navigation results root.

    Raises:
        TypeError: If a configured log level is not a string or ``None``.
        ValueError: If a configured log level string is empty or not a standard name.
    """
    main_console_level = _resolve_level('log_level_main_console', arguments, config)
    main_file_level = _resolve_level('log_level_main_file', arguments, config)
    # Validate image log levels now so --log-level-image-console and
    # --log-level-image-file errors surface at startup via the same try/except
    # block in main() that guards this call, not on the first image in the batch loop.
    _resolve_level('log_level_image_console', arguments, config)
    _resolve_level('log_level_image_file', arguments, config)

    # Replace any existing handlers so repeated setup does not duplicate log lines.
    for existing in list(MAIN_LOGGER.handlers):
        MAIN_LOGGER.removeHandler(existing)
        existing.close()

    MAIN_LOGGER.add_handler(pdslogger.stream_handler(level=main_console_level))

    log_dir = FCPath(nav_results_root_str) / 'logs' / 'sd_offset'
    log_base = log_dir / 'sd_offset.log'
    MAIN_LOGGER.add_handler(
        pdslogger.file_handler(log_base, level=main_file_level, rotation='ymdhms')
    )


def image_log_handlers(
    image_log_path: FCPath,
    arguments: argparse.Namespace | None,
    config: 'Config',
) -> list[logging.Handler]:
    """Create local handlers for a single image: a stdout handler and a file handler.

    Both levels are resolved from ``arguments``, then ``config.general``, then
    ``"INFO"``.  The returned handlers should be passed to ``IMAGE_LOGGER.open()`` so
    they are active only while that image is being processed and are automatically
    removed when that context exits.

    Parameters:
        image_log_path: Destination path for this image's log file.
        arguments: Parsed CLI arguments, or None to fall back entirely to config.
        config: Navigation configuration.

    Returns:
        A list containing a stdout stream handler and a file handler for the image log.

    Raises:
        TypeError: If a configured log level is not a string or ``None``.
        ValueError: If a configured log level string is empty or not a standard name.
    """
    image_console_level = _resolve_level('log_level_image_console', arguments, config)
    image_file_level = _resolve_level('log_level_image_file', arguments, config)
    return [
        pdslogger.stream_handler(level=image_console_level),
        cast(logging.FileHandler, pdslogger.file_handler(image_log_path, level=image_file_level)),
    ]
