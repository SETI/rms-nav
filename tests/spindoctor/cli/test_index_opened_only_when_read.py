"""When a resolved results index is opened, and when it deliberately is not.

An index that will not open fails a run rather than letting it read files
instead, because a run that silently changed storage would be a slow and
differently-classified run wearing the same command line.  That refusal belongs
to the modes that read a navigation record.  A mode that reads none -- a dry
run, or a mosaic pass over reprojections already on disk -- must not acquire a
new way to fail, since a machine exporting ``NAV_RESULTS_DB`` would otherwise
break invocations that worked before the variable was set.

Each program is driven through its own ``main``, with only the dataset
enumeration replaced, so what is exercised is the order the program really does
things in rather than a restatement of it.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from spindoctor.cli import sd_backplanes, sd_mosaic

_UNOPENABLE = 'sqlite:///{path}'
"""A URL naming a SQLite file that does not exist, which cannot be opened."""


class _NoImages:
    """A dataset that enumerates nothing and takes no selection arguments."""

    def add_selection_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add nothing: these runs select no images.

        Parameters:
            parser: The program's parser.
        """

    def yield_image_files_from_arguments(self, arguments: argparse.Namespace) -> Iterator[Any]:
        """Yield no images.

        Parameters:
            arguments: The parsed command line.

        Yields:
            Nothing.
        """
        return iter(())


@pytest.fixture
def datasetless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both programs' dataset lookup with one that enumerates nothing.

    Parameters:
        monkeypatch: Patcher, which reverts after the test.
    """
    for module in (sd_backplanes, sd_mosaic):
        monkeypatch.setattr(module, 'dataset_name_to_class', lambda _name: _NoImages)


def _absent_index(tmp_path: Path) -> str:
    """Return a URL naming an index file that was never written.

    Parameters:
        tmp_path: Directory the named file would have been in.

    Returns:
        The URL.
    """
    return _UNOPENABLE.format(path=(tmp_path / 'nowhere' / 'index.sqlite3').as_posix())


def _backplane_argv(tmp_path: Path, *flags: str) -> list[str]:
    """Return a backplane command line naming both roots and an absent index.

    Parameters:
        tmp_path: Directory the roots are placed under.
        flags: Extra flags for the mode under test.

    Returns:
        The arguments, without the program name.
    """
    return [
        'coiss_saturn',
        '--nav-results-root',
        (tmp_path / 'nav').as_posix(),
        '--backplane-results-root',
        (tmp_path / 'backplanes').as_posix(),
        '--results-db',
        _absent_index(tmp_path),
        '--no-log-main-to-file',
        *flags,
    ]


def _mosaic_argv(tmp_path: Path, *flags: str) -> list[str]:
    """Return a ring-mosaic command line naming a root and an absent index.

    Parameters:
        tmp_path: Directory the roots are placed under.
        flags: Extra flags for the mode under test.

    Returns:
        The arguments, without the program name.
    """
    return [
        'rings',
        'coiss_saturn',
        '--nav-results-root',
        (tmp_path / 'nav').as_posix(),
        '--output-dir',
        (tmp_path / 'out').as_posix(),
        '--planet',
        'SATURN',
        '--radius-inner',
        '74000',
        '--radius-outer',
        '140000',
        '--radius-resolution',
        '100',
        '--longitude-resolution',
        '0.1',
        '--results-db',
        _absent_index(tmp_path),
        '--no-log-main-to-file',
        *flags,
    ]


def _run(module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one program's ``main`` with the given command line.

    Parameters:
        module: The dispatch module.
        argv: The arguments, without the program name.
        monkeypatch: Patcher, used for ``sys.argv``.
    """
    monkeypatch.setattr('sys.argv', [module.__name__, *argv])
    module.main()


def test_a_backplane_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run says what it would do and reads no navigation record."""
    _run(sd_backplanes, _backplane_argv(tmp_path, '--dry-run'), monkeypatch)


def test_a_backplane_run_that_does_read_records_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: without the dry run the same index refuses the same run."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        _run(sd_backplanes, _backplane_argv(tmp_path), monkeypatch)


def test_a_mosaic_run_that_skips_reprojection_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mosaic pass builds from reprojections on disk and looks nothing up."""
    _run(sd_mosaic, _mosaic_argv(tmp_path, '--skip-reproject'), monkeypatch)


def test_a_mosaic_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nor does a dry run, which stops before every per-image lookup."""
    _run(sd_mosaic, _mosaic_argv(tmp_path, '--dry-run', '--skip-mosaic'), monkeypatch)


def test_a_mosaic_run_that_does_reproject_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the reprojection pass is the reader, so it refuses the index."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        _run(sd_mosaic, _mosaic_argv(tmp_path, '--skip-mosaic'), monkeypatch)
