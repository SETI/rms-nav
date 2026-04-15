"""Logger instances and setup for the nav_offset pipeline.

Provides two PdsLogger instances:

- ``MAIN_LOGGER`` (``"nav_offset"``) -- top-level program events, written to stdout
  and to a timestamped logfile under ``NAV_RESULTS_ROOT/logs/nav_offset/``.
- ``IMAGE_LOGGER`` (``"nav_image"``) -- per-image processing events; both its stdout
  handler and its per-image logfile handler are attached as local handlers inside each
  ``logger.open()`` context so they are active only while that image is being processed.
- ``DEFAULT_LOGGER`` -- alias for ``IMAGE_LOGGER`` retained for backward compatibility.

Call ``setup_logging()`` once in ``main()`` after the nav-results root and CLI
arguments have been resolved.
"""

from __future__ import annotations

import argparse
import logging
from filecache import FCPath
from typing import TYPE_CHECKING, cast

import pdslogger

if TYPE_CHECKING:
    from .config import Config

MAIN_LOGGER = pdslogger.PdsLogger('nav_offset', lognames=False, digits=3)
IMAGE_LOGGER = pdslogger.PdsLogger('nav_image', lognames=False, digits=3)
DEFAULT_LOGGER = IMAGE_LOGGER

_FALLBACK_LEVEL = 'INFO'


def _resolve_level(attr_name: str, arguments: argparse.Namespace, config: Config) -> str:
    """Return the log level for ``attr_name``, checking args then config then fallback.

    Parameters:
        attr_name: Name of the config key and argparse attribute (underscore form).
        arguments: Parsed CLI arguments.
        config: Navigation configuration.

    Returns:
        A log-level string (e.g. ``"INFO"``).
    """
    level: str | None = getattr(arguments, attr_name, None)
    if level is None:
        level = getattr(config.general, attr_name, None)
    if level is None:
        level = _FALLBACK_LEVEL
    return level.upper()


def setup_logging(
    arguments: argparse.Namespace,
    config: Config,
    nav_results_root_str: str,
) -> None:
    """Configure MAIN_LOGGER with stdout and a timestamped file handler.

    Reads each log level from ``arguments`` first, then from ``config.general``,
    then falls back to ``"INFO"``.  The main logfile is written as a timestamped
    file under ``{nav_results_root}/logs/nav_offset/``.

    IMAGE_LOGGER handlers are **not** configured here; both its console and per-image
    file handlers are attached as local handlers inside each ``IMAGE_LOGGER.open()``
    context via ``image_log_handlers()``.

    Parameters:
        arguments: Parsed CLI arguments; may carry ``log_level_main_console`` and
            ``log_level_main_file`` attributes.
        config: Navigation configuration providing ``general.*`` fallback values.
        nav_results_root_str: Local filesystem path to the navigation results root.
    """
    main_console_level = _resolve_level('log_level_main_console', arguments, config)
    main_file_level = _resolve_level('log_level_main_file', arguments, config)

    MAIN_LOGGER.add_handler(pdslogger.stream_handler(level=main_console_level))

    log_dir = FCPath(nav_results_root_str) / 'logs' / 'nav_offset'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_base = log_dir / 'nav_offset.log'
    MAIN_LOGGER.add_handler(
        pdslogger.file_handler(log_base, level=main_file_level, rotation='ymdhms')
    )


def image_log_handlers(
    image_log_path: FCPath,
    arguments: argparse.Namespace,
    config: Config,
) -> list[logging.Handler]:
    """Create local handlers for a single image: a stdout handler and a file handler.

    Both levels are resolved from ``arguments``, then ``config.general``, then
    ``"INFO"``.  The returned handlers should be passed to ``IMAGE_LOGGER.open()`` so
    they are active only while that image is being processed and are automatically
    removed when that context exits.

    Parameters:
        image_log_path: Destination path for this image's log file.
        arguments: Parsed CLI arguments.
        config: Navigation configuration.

    Returns:
        A list containing a stdout stream handler and a file handler for the image log.
    """
    image_console_level = _resolve_level('log_level_image_console', arguments, config)
    image_file_level = _resolve_level('log_level_image_file', arguments, config)
    return [
        pdslogger.stream_handler(level=image_console_level),
        cast(logging.FileHandler, pdslogger.file_handler(image_log_path, level=image_file_level)),
    ]
