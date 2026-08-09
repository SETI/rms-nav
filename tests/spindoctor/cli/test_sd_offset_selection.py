"""Tests that a selection the enumeration refuses is reported, not traced back.

The selection arguments are finally read while images are being enumerated, so
that is where a contradictory pair of them, or a results index that cannot be
opened or does not cover the results root, is first diagnosed.  Each of those is
a misconfigured run rather than a broken one, and each already carries a message
saying what to change.  A traceback buries that message under six frames, and an
index URL can carry a database password into the terminal with it.
"""

import argparse
from collections.abc import Iterator
from typing import Any

import pytest

from spindoctor.cli import sd_offset
from spindoctor.dataset.dataset import ImageFiles

REFUSAL = 'sqlite:////tmp/absent.sqlite3: there is no results index at /tmp/absent.sqlite3'
"""A message of the shape the results index refuses an unopenable URL with."""


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
            ValueError: Always.
        """
        raise ValueError(REFUSAL)
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
