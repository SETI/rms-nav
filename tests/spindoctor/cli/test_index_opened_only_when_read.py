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
from typing import Any

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


def _urls_asked_for(module: Any, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Record the index each source the program builds is asked to open.

    Asserting on the recorded list rather than on the run merely completing:
    "no index was opened" and "an index was opened and its failure swallowed"
    both look like a run that finished, and the second is the regression.

    Parameters:
        module: The dispatch module whose source construction is watched.
        monkeypatch: Patcher, which reverts after the test.

    Returns:
        The list the spy appends to, one entry per call, holding the URL that
        call was given.
    """
    urls: list[str | None] = []
    real = module.build_pointing_source

    def spy(nav_results_root: Any, *, results_db_url: str | None = None) -> Any:
        urls.append(results_db_url)
        return real(nav_results_root, results_db_url=results_db_url)

    monkeypatch.setattr(module, 'build_pointing_source', spy)
    return urls


def test_a_backplane_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run says what it would do and reads no navigation record.

    It returns before building a source at all, so nothing is asked of the
    index and no root has to be resolved either.
    """
    urls = _urls_asked_for(sd_backplanes, monkeypatch)
    run_program(sd_backplanes, _backplane_argv(tmp_path, '--dry-run'), monkeypatch)
    assert urls == []


def test_a_backplane_run_that_does_read_records_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: without the dry run the same index refuses the same run."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_backplanes, _backplane_argv(tmp_path), monkeypatch)


def test_a_backplane_run_that_does_read_records_is_given_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And is given the resolved URL, which is what makes the dry run's silence mean something.

    A program that never passed the URL on would satisfy the dry-run assertion
    for the wrong reason.
    """
    urls = _urls_asked_for(sd_backplanes, monkeypatch)
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_backplanes, _backplane_argv(tmp_path), monkeypatch)
    assert urls == [_absent_index(tmp_path)]


def test_a_mosaic_run_that_skips_reprojection_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mosaic pass builds from reprojections on disk and looks nothing up.

    It still builds a source, because the pass is handed one either way; what
    it must not do is name the index to it.
    """
    urls = _urls_asked_for(sd_mosaic, monkeypatch)
    run_program(sd_mosaic, _mosaic_argv(tmp_path, '--skip-reproject'), monkeypatch)
    assert urls == [None]


def test_a_mosaic_dry_run_does_not_open_the_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nor does a dry run, which stops before every per-image lookup."""
    urls = _urls_asked_for(sd_mosaic, monkeypatch)
    run_program(sd_mosaic, _mosaic_argv(tmp_path, '--dry-run', '--skip-mosaic'), monkeypatch)
    assert urls == [None]


def test_a_mosaic_run_that_does_reproject_still_fails_on_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: the reprojection pass is the reader, so it refuses the index."""
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_mosaic, _mosaic_argv(tmp_path, '--skip-mosaic'), monkeypatch)


def test_a_mosaic_run_that_does_reproject_is_given_that_index(
    tmp_path: Path, datasetless: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same control for the two mosaic modes above, which pass on no URL.

    Both would be satisfied by a program that had stopped resolving the option
    at all; only a reading pass that is handed it separates the two.
    """
    urls = _urls_asked_for(sd_mosaic, monkeypatch)
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        run_program(sd_mosaic, _mosaic_argv(tmp_path, '--skip-mosaic'), monkeypatch)
    assert urls == [_absent_index(tmp_path)]
