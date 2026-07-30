"""Resolution and construction of the two SpinDoctor loggers.

Two loggers exist.  The main logger covers one program run and reports what
the program is doing at the top level.  The image logger covers one image
inside one per-image backend and carries that image's processing detail.

Each logger writes to a console sink, a file sink, or both.  Both sinks always
share a level, so one level per module governs whatever sinks are enabled.
That is what lets a module's verbosity be expressed as the ``level=`` argument
of a single ``logger.open(...)`` call: pdslogger applies a section level as a
floor before its handlers, so with both handlers at the same level the floor
is the effective level and the two sinks cannot disagree.

Levels resolve in two steps.  The top-level ``logging`` block is first merged
with ``logging.programs.<program>``, the program block winning key by key;
then, against that merged result, the most specific setting wins:

    --log-level MODULE=LEVEL
      > <category>.<module>
      > <category>.default
      > --log-level-main / --log-level-image
      > --log-level LEVEL
      > main / image
      > INFO

Building a logger attaches exactly the enabled sinks.  When neither is
enabled the builder attaches ``NULL_HANDLER``, because a PdsLogger left with
no handlers does not go quiet -- it falls back to printing every record to
stdout regardless of level.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pdslogger
from filecache import FCPath
from pdslogger import PdsLogger

from .logging_keys import CATEGORY_KEYS, normalize_level

if TYPE_CHECKING:
    from .config import Config

__all__ = [
    'BACKEND_NAMES',
    'DEFAULT_LEVEL',
    'LOG_TIMESTAMP_FORMAT',
    'LogLevels',
    'LogSinks',
    'build_image_log_handlers',
    'build_main_logger',
    'image_log_path',
    'main_log_path',
    'resolve_log_levels',
    'run_timestamp',
    'sinks_from_arguments',
]


DEFAULT_LEVEL = 'INFO'
"""Level used when nothing in the arguments or configuration says otherwise."""

LOG_TIMESTAMP_FORMAT = '%Y-%m-%dT%H-%M-%S'
"""Suffix format distinguishing one run's log file from the next."""

BACKEND_NAMES = frozenset({'nav', 'backplane', 'reproj'})
"""Per-image backends, each owning a subtree of the log root."""

_OFF = 'NONE'
_CATEGORY_DEFAULT_KEY = 'default'
_PROGRAMS_KEY = 'programs'


@dataclass(frozen=True)
class LogLevels:
    """Resolved level for each logger and each module that overrides it.

    Parameters:
        main: Level for the main logger.
        image: Level for the image logger, and for any image-scoped module
            with no override of its own.
        modules: Level per module key, for the modules that override the
            image level.  Keys are the snake_case names validated by
            :mod:`spindoctor.config.logging_keys`.
    """

    main: str = DEFAULT_LEVEL
    image: str = DEFAULT_LEVEL
    modules: dict[str, str] = field(default_factory=dict)

    def for_module(self, log_key: str) -> str:
        """Return the level governing ``log_key``.

        Parameters:
            log_key: A module's snake_case configuration key.

        Returns:
            The module's own level if it has one, else the image level.
        """
        return self.modules.get(log_key, self.image)


@dataclass(frozen=True)
class LogSinks:
    """Which sinks each logger writes to, and where files are written.

    Parameters:
        log_root: Root directory for every log file this run produces.
        main_console: Whether the main logger writes to stdout.
        main_file: Whether the main logger writes to a file.
        image_console: Whether the image logger writes to stdout.
        image_file: Whether the image logger writes to a file.
    """

    log_root: FCPath
    main_console: bool = True
    main_file: bool = True
    image_console: bool = False
    image_file: bool = True


def _level_or_none(value: Any) -> str | None:
    """Return a canonicalized level name, or None when nothing was configured.

    Parameters:
        value: A configured or command-line level, possibly absent.

    Returns:
        The canonical level name, or None if ``value`` is absent.
    """
    if value is None:
        return None
    return normalize_level(str(value))


def _merged_logging_block(program_name: str, config: 'Config') -> dict[str, Any]:
    """Return the logging configuration in force for one program.

    The top-level block is merged with that program's block under
    ``programs``, the program's value winning key by key.  Categories merge a
    level deeper, so a program can override one module without restating the
    rest of the category.

    Parameters:
        program_name: The program's identity, as declared by its dispatch
            module.
        config: The loaded configuration.

    Returns:
        A mapping in the shape of the top-level ``logging`` block, with the
        ``programs`` key removed.
    """
    section = config.logging
    block: dict[str, Any] = {k: v for k, v in dict(section).items() if k != _PROGRAMS_KEY}
    for key in CATEGORY_KEYS:
        if isinstance(block.get(key), dict):
            block[key] = dict(block[key])

    overrides = dict(section).get(_PROGRAMS_KEY) or {}
    program_block = overrides.get(program_name) or {}
    for key, value in dict(program_block).items():
        if key in CATEGORY_KEYS and isinstance(value, dict):
            merged = dict(block.get(key) or {})
            merged.update(dict(value))
            block[key] = merged
        else:
            block[key] = value
    return block


def resolve_log_levels(
    program_name: str,
    arguments: argparse.Namespace | None,
    config: 'Config',
) -> LogLevels:
    """Resolve the level governing each logger and each overriding module.

    Applies the merge and precedence described in the module docstring.  A
    module named on the command line beats one named in the configuration; a
    module named anywhere beats its category default; a category default beats
    the per-logger default.

    Parameters:
        program_name: The program's identity, used to select its block under
            ``logging.programs``.
        arguments: Parsed command-line arguments, or None to resolve from the
            configuration alone.  Recognized attributes are ``log_level``,
            ``log_level_main``, and ``log_level_image``.
        config: The loaded configuration.

    Returns:
        The resolved :class:`LogLevels`.
    """
    block = _merged_logging_block(program_name, config)

    cli_global, cli_modules = _parse_log_level_arguments(arguments)
    cli_main = _level_or_none(getattr(arguments, 'log_level_main', None))
    cli_image = _level_or_none(getattr(arguments, 'log_level_image', None))

    main = cli_main or cli_global or _level_or_none(block.get('main')) or DEFAULT_LEVEL
    image = cli_image or cli_global or _level_or_none(block.get('image')) or DEFAULT_LEVEL

    modules: dict[str, str] = {}
    for category in CATEGORY_KEYS:
        entries = block.get(category)
        if not isinstance(entries, dict):
            continue
        category_default = _level_or_none(entries.get(_CATEGORY_DEFAULT_KEY))
        for key, value in entries.items():
            if key == _CATEGORY_DEFAULT_KEY:
                continue
            level = _level_or_none(value)
            if level is not None:
                modules[key] = level
        if category_default is not None:
            # Applies to every module in the category that did not name itself.
            # Recorded per known key rather than as a category-wide entry so
            # LogLevels stays a flat lookup.
            for key in _category_member_keys(category):
                modules.setdefault(key, category_default)

    modules.update(cli_modules)
    return LogLevels(main=main, image=image, modules=modules)


def _category_member_keys(category: str) -> frozenset[str]:
    """Return every module key belonging to ``category``.

    Parameters:
        category: One of the categories in
            :data:`spindoctor.config.logging_keys.CATEGORY_KEYS`.

    Returns:
        The category's valid module keys.
    """
    from .logging_keys import OTHER_LOG_KEYS, model_log_keys, technique_log_keys

    if category == 'techniques':
        return technique_log_keys()
    if category == 'models':
        return model_log_keys()
    return OTHER_LOG_KEYS


def _parse_log_level_arguments(
    arguments: argparse.Namespace | None,
) -> tuple[str | None, dict[str, str]]:
    """Split the repeatable ``--log-level`` values into global and per-module.

    A bare ``LEVEL`` sets the default for both loggers; a ``MODULE=LEVEL`` pair
    sets one module.  The last bare value wins, so a later flag overrides an
    earlier one rather than silently combining.

    Parameters:
        arguments: Parsed command-line arguments, or None.

    Returns:
        Tuple of the global level (or None) and a mapping of module key to
        level.
    """
    values = getattr(arguments, 'log_level', None) or []
    if isinstance(values, str):
        values = [values]

    global_level: str | None = None
    modules: dict[str, str] = {}
    for value in values:
        text = str(value)
        if '=' in text:
            key, _, level = text.partition('=')
            modules[key.strip()] = normalize_level(level)
        else:
            global_level = normalize_level(text)
    return global_level, modules


def main_log_path(log_root: FCPath, program_name: str, *, timestamp: str) -> FCPath:
    """Return the path of a program run's main log file.

    Parameters:
        log_root: Root directory for this run's logs.
        program_name: The program's identity, which names its directory.
        timestamp: Run timestamp in :data:`LOG_TIMESTAMP_FORMAT`.

    Returns:
        ``{log_root}/{program_name}/main_{timestamp}.log``.
    """
    return log_root / program_name / f'main_{timestamp}.log'


def image_log_path(
    log_root: FCPath, backend: str, results_path_stub: str, *, timestamp: str
) -> FCPath:
    """Return the path of one image's log file for one backend.

    The subtree is keyed by backend rather than by program, so an image's log
    for a given stage is in one place whichever driver produced it.

    Parameters:
        log_root: Root directory for this run's logs.
        backend: One of :data:`BACKEND_NAMES`.
        results_path_stub: The image's results path stub.
        timestamp: Run timestamp in :data:`LOG_TIMESTAMP_FORMAT`.

    Returns:
        ``{log_root}/{backend}/{results_path_stub}_{timestamp}.log``.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    if backend not in BACKEND_NAMES:
        raise ValueError(f'Unknown backend {backend!r}; expected one of {sorted(BACKEND_NAMES)}')
    return log_root / backend / f'{results_path_stub}_{timestamp}.log'


def run_timestamp() -> str:
    """Return the current local time formatted for a log file name.

    Returns:
        The timestamp in :data:`LOG_TIMESTAMP_FORMAT`.
    """
    return datetime.now().strftime(LOG_TIMESTAMP_FORMAT)


def _handlers_for(*, console: bool, to_file: bool, path: FCPath | None, level: str) -> list[Any]:
    """Build the handler list for one logger.

    Parameters:
        console: Whether to attach a stdout handler.
        to_file: Whether to attach a file handler.
        path: Destination for the file handler; required when ``to_file``.
        level: Level shared by both sinks.

    Returns:
        The handlers to attach.  Never empty: with no sink enabled the list is
        ``[NULL_HANDLER]``, because a PdsLogger with no handlers prints every
        record to stdout regardless of level.

    Raises:
        ValueError: If ``to_file`` is set without a ``path``.
    """
    handlers: list[Any] = []
    if level != _OFF:
        if console:
            handlers.append(pdslogger.stream_handler(level=level.lower()))
        if to_file:
            if path is None:
                raise ValueError('A file sink was requested without a destination path')
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(pdslogger.file_handler(path, level=level.lower()))
    if not handlers:
        handlers.append(pdslogger.NULL_HANDLER)
    return handlers


def build_main_logger(
    logger: PdsLogger,
    program_name: str,
    sinks: LogSinks,
    levels: LogLevels,
    *,
    timestamp: str | None = None,
) -> FCPath | None:
    """Attach this run's sinks to the main logger.

    Any handlers already present are removed first, so calling this twice does
    not duplicate output.

    Parameters:
        logger: The main logger to configure.
        program_name: The program's identity, which names its log directory.
        sinks: Which sinks to attach and where files go.
        levels: Resolved levels; the main level applies to both sinks.
        timestamp: Run timestamp for the file name; defaults to now.

    Returns:
        The path written to, or None when no file sink is enabled.
    """
    # remove_all_handlers detaches but does not close, so a rebuild would leak
    # the previous run's open log file.
    for existing in list(logger.handlers):
        existing.close()
    logger.remove_all_handlers()
    stamp = timestamp if timestamp is not None else run_timestamp()
    path = main_log_path(sinks.log_root, program_name, timestamp=stamp) if sinks.main_file else None
    for handler in _handlers_for(
        console=sinks.main_console, to_file=sinks.main_file, path=path, level=levels.main
    ):
        logger.add_handler(handler)
    return path


def build_image_log_handlers(
    backend: str,
    results_path_stub: str,
    sinks: LogSinks,
    levels: LogLevels,
    *,
    timestamp: str | None = None,
) -> tuple[list[Any], FCPath | None]:
    """Build the handlers for one image, to be attached for that image only.

    The handlers are returned rather than attached because the image logger
    scopes them to a single ``logger.open(...)`` section, which removes them
    when the image is done.

    Parameters:
        backend: The per-image backend, one of :data:`BACKEND_NAMES`.
        results_path_stub: The image's results path stub.
        sinks: Which sinks to attach and where files go.
        levels: Resolved levels; the image level applies to both sinks.
        timestamp: Run timestamp for the file name; defaults to now.

    Returns:
        Tuple of the handlers and the path written to, the latter None when no
        file sink is enabled.

    Raises:
        ValueError: If ``backend`` is not a known backend.
    """
    # Validated up front so a typo fails the same way whether or not a file
    # sink happens to be enabled.
    if backend not in BACKEND_NAMES:
        raise ValueError(f'Unknown backend {backend!r}; expected one of {sorted(BACKEND_NAMES)}')
    stamp = timestamp if timestamp is not None else run_timestamp()
    path = (
        image_log_path(sinks.log_root, backend, results_path_stub, timestamp=stamp)
        if sinks.image_file
        else None
    )
    handlers = _handlers_for(
        console=sinks.image_console, to_file=sinks.image_file, path=path, level=levels.image
    )
    return handlers, path


def sinks_from_arguments(arguments: argparse.Namespace | None, log_root: FCPath) -> LogSinks:
    """Build the sink selection from command-line arguments.

    Each sink keeps its default unless the corresponding flag was given.

    Parameters:
        arguments: Parsed command-line arguments, or None for the defaults.
        log_root: Resolved log root for this run.

    Returns:
        The resolved :class:`LogSinks`.
    """
    defaults = LogSinks(log_root=log_root)

    def flag(name: str, default: bool) -> bool:
        value = getattr(arguments, name, None)
        return default if value is None else bool(value)

    return LogSinks(
        log_root=log_root,
        main_console=flag('log_main_to_console', defaults.main_console),
        main_file=flag('log_main_to_file', defaults.main_file),
        image_console=flag('log_image_to_console', defaults.image_console),
        image_file=flag('log_image_to_file', defaults.image_file),
    )
