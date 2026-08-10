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

from pathlib import Path

import pytest
from tests.spindoctor.cli.conftest import backplane_argv, mosaic_argv, run_program

from spindoctor.cli import sd_backplanes, sd_mosaic

_UNOPENABLE = 'sqlite:///{path}'
"""A URL naming a SQLite file that does not exist, which cannot be opened."""


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
    return backplane_argv(tmp_path, _absent_index(tmp_path), '--no-log-main-to-file', *flags)


def _mosaic_argv(tmp_path: Path, *flags: str) -> list[str]:
    """Return a ring-mosaic command line naming a root and an absent index.

    Parameters:
        tmp_path: Directory the roots are placed under.
        flags: Extra flags for the mode under test.

    Returns:
        The arguments, without the program name.
    """
    return mosaic_argv(tmp_path, _absent_index(tmp_path), '--no-log-main-to-file', *flags)


def test_a_backplane_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run says what it would do and reads no navigation record."""
    run_program(sd_backplanes, _backplane_argv(tmp_path, '--dry-run'), monkeypatch)


def test_a_backplane_run_that_does_read_records_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: without the dry run the same index refuses the same run."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_backplanes, _backplane_argv(tmp_path), monkeypatch)


def test_a_mosaic_run_that_skips_reprojection_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mosaic pass builds from reprojections on disk and looks nothing up."""
    run_program(sd_mosaic, _mosaic_argv(tmp_path, '--skip-reproject'), monkeypatch)


def test_a_mosaic_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nor does a dry run, which stops before every per-image lookup."""
    run_program(sd_mosaic, _mosaic_argv(tmp_path, '--dry-run', '--skip-mosaic'), monkeypatch)


def test_a_mosaic_run_that_does_reproject_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the reprojection pass is the reader, so it refuses the index."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_mosaic, _mosaic_argv(tmp_path, '--skip-mosaic'), monkeypatch)
