"""Tests that a selection the enumeration refuses is reported, not traced back.

The selection arguments are finally read while images are being enumerated, so
that is where a contradictory pair of them, or a results index that cannot be
opened, cannot be read, or does not cover the results root, is first diagnosed.
Each of those is a misconfigured run rather than a broken one, and each already
carries a message saying what to change.  A traceback buries that message under
six frames, and an index URL can carry a database password into the terminal
with it.

Only those are reported that way, which is why the refusal has a type of its
own.  An enumeration raises a plain ``ValueError`` for a volume name that is not
one, for a number a label would not yield, and for an outright programming
error; a run that ends in one of those has gone wrong rather than been
misconfigured, and its traceback is what says where.
"""

import argparse
from collections.abc import Iterator
from typing import Any

import pytest

from spindoctor.cli import sd_offset
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.dataset.results_filter import SelectionError

REFUSAL = 'sqlite:////tmp/absent.sqlite3: there is no results index at /tmp/absent.sqlite3'
"""A message of the shape the results index refuses an unopenable URL with."""

BUG = 'Unexpected keyword arguments: {}'
"""A message of the shape the enumeration raises when it was called wrongly."""


class _RefusingDataset:
    """A dataset whose enumeration refuses the selection it was given."""

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Refuse the selection, as the real enumeration does.

        Parameters:
            arguments: The parsed command line, unused.

        Yields:
            Nothing; the refusal is raised before the first image.

        Raises:
            SelectionError: Always.
        """
        raise SelectionError(REFUSAL)
        yield  # pragma: no cover -- makes this a generator function


class _BrokenDataset:
    """A dataset whose enumeration fails for a reason nobody configured."""

    def yield_image_files_from_arguments(
        self, arguments: argparse.Namespace
    ) -> Iterator[ImageFiles]:
        """Fail the way a caller error deep in the enumeration fails.

        Parameters:
            arguments: The parsed command line, unused.

        Yields:
            Nothing; the failure happens before the first image.

        Raises:
            ValueError: Always.
        """
        raise ValueError(BUG)
        yield  # pragma: no cover -- makes this a generator function


class _RecordingLogger:
    """A stand-in for the main logger that keeps what it was asked to report."""

    def __init__(self) -> None:
        """Start with nothing reported."""
        self.errors: list[str] = []

    def error(self, message: str, *args: Any) -> None:
        """Record one formatted error.

        Parameters:
            message: The format string.
            args: Its arguments.
        """
        self.errors.append(message % args)


@pytest.fixture
def refusing(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    """Install a refusing dataset and a logger that records the report.

    Parameters:
        monkeypatch: Fixture the two stand-ins are installed through.

    Returns:
        The logger, to read the report from.
    """
    logger = _RecordingLogger()
    monkeypatch.setattr(sd_offset, 'DATASET', _RefusingDataset())
    monkeypatch.setattr(sd_offset, 'MAIN_LOGGER', logger)
    return logger


@pytest.fixture
def breaking(monkeypatch: pytest.MonkeyPatch) -> _RecordingLogger:
    """Install a dataset that fails for an unconfigurable reason, and the same logger.

    Parameters:
        monkeypatch: Fixture the two stand-ins are installed through.

    Returns:
        The logger, to show that nothing was reported through it.
    """
    logger = _RecordingLogger()
    monkeypatch.setattr(sd_offset, 'DATASET', _BrokenDataset())
    monkeypatch.setattr(sd_offset, 'MAIN_LOGGER', logger)
    return logger


def test_a_refused_selection_ends_the_run(refusing: _RecordingLogger) -> None:
    """The run stops; it does not go on to navigate an arbitrary subset."""
    with pytest.raises(SystemExit) as excinfo:
        list(sd_offset._selected_image_files(argparse.Namespace()))
    assert excinfo.value.code == 1


def test_a_refused_selection_is_reported(refusing: _RecordingLogger) -> None:
    """The message that says what to change is what the operator is shown."""
    with pytest.raises(SystemExit):
        list(sd_offset._selected_image_files(argparse.Namespace()))
    assert refusing.errors == [REFUSAL]


def test_a_failure_nobody_configured_keeps_its_traceback(breaking: _RecordingLogger) -> None:
    """A run that went wrong is not a run that was misconfigured.

    Reporting one line and exiting would throw away the stack that says where
    the failure is, for every ValueError a label, a conversion or a caller error
    raises anywhere inside the enumeration.
    """
    with pytest.raises(ValueError, match='Unexpected keyword arguments'):
        list(sd_offset._selected_image_files(argparse.Namespace()))


def test_a_failure_nobody_configured_is_not_reported_as_advice(
    breaking: _RecordingLogger,
) -> None:
    """Printing it as "here is what to change" would be advice nobody can act on."""
    with pytest.raises(ValueError):
        list(sd_offset._selected_image_files(argparse.Namespace()))
    assert breaking.errors == []
