"""Reaching the command-line surface a program actually builds for itself.

A program's flags are asserted against the parser it runs with, not against a
reconstruction of it: several are added by shared helpers, and a program
acquires or loses a whole group by calling one. An interactive program is asked
for its help text, which is what a user meets; a cloud-task driver builds its
parser inside ``async_main`` and hands it straight to the worker, so the worker
is intercepted and the parser taken from it.
"""

import argparse
import asyncio
import contextlib
import importlib
import io
import sys

__all__ = ['cloud_task_parser', 'program_help_text']


def program_help_text(program: str, argv: list[str]) -> str:
    """Return what ``program --help`` prints.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``, for a program that reads its
            dataset or mode from argv before parsing.

    Returns:
        The help text.
    """
    module = importlib.import_module(f'spindoctor.cli.{program}')
    buffer = io.StringIO()
    saved = sys.argv
    sys.argv = [program, *argv, '--help']
    try:
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            module.main()
    finally:
        sys.argv = saved
    return buffer.getvalue()


def cloud_task_parser(program: str) -> argparse.ArgumentParser:
    """Return the parser a cloud-task driver builds for itself.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.

    Returns:
        The parser the driver would have run with.
    """
    module = importlib.import_module(f'spindoctor.cli.{program}')
    captured: dict[str, argparse.ArgumentParser] = {}

    class _CapturedError(Exception):
        """Raised to stop the driver once its parser has been seen."""

    def _intercept(*args: object, **kwargs: object) -> None:
        parser = kwargs.get('argparser')
        assert isinstance(parser, argparse.ArgumentParser)
        captured['parser'] = parser
        raise _CapturedError

    real_worker = module.Worker
    module.Worker = _intercept  # type: ignore[attr-defined]
    try:
        with contextlib.suppress(_CapturedError):
            asyncio.run(module.async_main())
    finally:
        module.Worker = real_worker  # type: ignore[attr-defined]
    return captured['parser']
