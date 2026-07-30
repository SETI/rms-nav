"""Routing of log records to the main or the image logger.

Two loggers exist, and every component belongs to exactly one.  Components
whose work spans a run -- enumerating a dataset, tallying totals -- belong to
the main logger.  Components working on one image belong to the image logger.

The image logger is reached through :data:`IMAGE_LOGGER`, a proxy rather than
a logger.  Opening a section on it enters an image scope; while that scope is
open the proxy forwards to the real logger underneath, and every nested
section, model and technique writes into the same per-image log.  Outside any
scope there is nothing sensible to forward to, so the proxy routes the record
to the main logger, warns once about the call site, and -- with strict scope
enabled, as the test suite does -- raises instead.

Logging from an image-scoped component with no image open is a defect rather
than a mode to support.  Every occurrence found while this was written was
either correctly scoped or a component bound to the wrong logger.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

import pdslogger
from pdslogger import PdsLogger

from .logger import MAIN_LOGGER

__all__ = [
    'IMAGE_LOGGER',
    'ImageLoggerProxy',
    'LogRole',
    'LogScopeError',
    'image_scope',
    'image_scope_is_open',
    'set_strict_scope',
    'strict_scope',
]


class LogRole(Enum):
    """Which logger a component's records belong to."""

    MAIN = 'main'
    """Work spanning a whole run: enumeration, totals, per-image progress."""

    IMAGE = 'image'
    """Work on one image: models, techniques, reprojection, annotation."""


class LogScopeError(RuntimeError):
    """An image-scoped component logged with no image scope open."""


_ACTIVE_IMAGE_LOGGER: ContextVar[PdsLogger | None] = ContextVar(
    'spindoctor_active_image_logger', default=None
)

_DEFAULT_IMAGE_LOGGER = pdslogger.PdsLogger('nav_image', lognames=False, digits=3)

_strict_scope = False
_reported_call_sites: set[tuple[str, str, int]] = set()


def set_strict_scope(enabled: bool) -> None:
    """Choose whether an out-of-scope image log raises or only warns.

    Production warns, so one mis-scoped call cannot abort a batch over a log
    line.  The test suite raises, so a newly introduced one fails rather than
    appearing quietly in a main log.

    Parameters:
        enabled: True to raise on an out-of-scope image log.
    """
    global _strict_scope
    _strict_scope = enabled


def strict_scope() -> bool:
    """Whether an out-of-scope image log currently raises.

    Returns:
        True when strict scope is enabled.
    """
    return _strict_scope


def image_scope_is_open() -> bool:
    """Whether an image scope is currently open on this task.

    Returns:
        True when :func:`image_scope` is active.
    """
    return _ACTIVE_IMAGE_LOGGER.get() is not None


def _caller_site() -> tuple[str, str, int]:
    """Identify the first frame outside this module.

    Returns:
        Tuple of module name, function name, and line number of the code that
        logged, used both to name the offender and to deduplicate the warning.
    """
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get('__name__') == __name__:
        frame = frame.f_back  # type: ignore[assignment]
    if frame is None:  # pragma: no cover - a frame outside this module always exists
        return ('<unknown>', '<unknown>', 0)
    return (frame.f_globals.get('__name__', '<unknown>'), frame.f_code.co_name, frame.f_lineno)


def _report_out_of_scope() -> None:
    """Handle an image-scoped record logged with no image scope open.

    Raises:
        LogScopeError: When strict scope is enabled.
    """
    module, function, line = _caller_site()
    where = f'{module}.{function} (line {line})'
    if _strict_scope:
        raise LogScopeError(
            f'{where} logged to the image logger with no image scope open. '
            f'A component scoped to one image must log inside an image scope; '
            f'if its work spans the run, bind it to the main logger instead.'
        )
    site = (module, function, line)
    if site not in _reported_call_sites:
        _reported_call_sites.add(site)
        MAIN_LOGGER.warning(
            'Log routed to the main logger: %s logged to the image logger with no '
            'image scope open. Reported once per call site.',
            where,
        )


def _reset_reported_call_sites() -> None:
    """Forget which call sites have been warned about.

    Exposed for tests, which need each case to warn independently rather than
    inheriting the deduplication state of an earlier one.
    """
    _reported_call_sites.clear()


@contextmanager
def image_scope(logger: PdsLogger | None = None) -> Iterator[PdsLogger]:
    """Make ``logger`` the image logger for the duration of the block.

    Nesting is a no-op: an inner scope keeps the logger the outer one
    established, so a model or technique opening its own section does not
    displace the image it is running under.

    Parameters:
        logger: The real logger to route image records to.  None uses the
            default image logger.

    Yields:
        The logger image records will reach.
    """
    active = _ACTIVE_IMAGE_LOGGER.get()
    if active is not None:
        yield active
        return
    target = logger if logger is not None else _DEFAULT_IMAGE_LOGGER
    token = _ACTIVE_IMAGE_LOGGER.set(target)
    try:
        yield target
    finally:
        _ACTIVE_IMAGE_LOGGER.reset(token)


class ImageLoggerProxy:
    """Forwards image-scoped records to whichever logger the scope selected.

    Presents the part of the ``PdsLogger`` interface SpinDoctor uses, so a
    component can hold this object wherever it would have held a logger.
    Records arriving outside an image scope are routed to the main logger and
    reported; see the module docstring.
    """

    def _target(self) -> PdsLogger:
        """Return the logger this record should reach.

        Returns:
            The scope's logger, or the main logger when none is open.
        """
        active = _ACTIVE_IMAGE_LOGGER.get()
        if active is not None:
            return active
        _report_out_of_scope()
        return MAIN_LOGGER

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at DEBUG. See :meth:`pdslogger.PdsLogger.debug`."""
        self._target().debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at INFO. See :meth:`pdslogger.PdsLogger.info`."""
        self._target().info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at WARNING. See :meth:`pdslogger.PdsLogger.warning`."""
        self._target().warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR. See :meth:`pdslogger.PdsLogger.error`."""
        self._target().error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at CRITICAL. See :meth:`pdslogger.PdsLogger.critical`."""
        self._target().critical(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with its stack trace.

        See :meth:`pdslogger.PdsLogger.exception`.
        """
        self._target().exception(message, *args, **kwargs)

    def log(self, level: Any, message: str, *args: Any, **kwargs: Any) -> None:
        """Log at an explicit level. See :meth:`pdslogger.PdsLogger.log`."""
        self._target().log(level, message, *args, **kwargs)

    def set_level(self, level: Any) -> None:
        """Set the level of the current tier. See ``PdsLogger.set_level``."""
        self._target().set_level(level)

    @property
    def handlers(self) -> Any:
        """The handlers of the logger records currently reach."""
        return self._target().handlers

    @contextmanager
    def open(self, title: str, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """Open a section, entering an image scope if none is open.

        Opening a section on the image logger is what establishes an image
        scope, so the outermost such call selects the logger every nested
        section, model and technique will write to.  A nested call passes
        straight through to the scope's logger.

        Parameters:
            title: Title of the section.
            *args: Passed to ``PdsLogger.open``.
            **kwargs: Passed to ``PdsLogger.open``.

        Yields:
            Whatever ``PdsLogger.open`` yields.
        """
        with image_scope() as target, target.open(title, *args, **kwargs) as section:
            yield section


IMAGE_LOGGER = ImageLoggerProxy()
"""The image logger: a proxy resolving to whichever image scope is open."""
