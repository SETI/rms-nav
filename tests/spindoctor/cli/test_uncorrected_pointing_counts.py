"""What a run counts as an image left without the pointing it asked for.

The count is a shortfall rather than a census: it answers "how many images
wanted a recorded pointing and did not get one", which is the number an
operator acts on.  A run given no navigation results root wanted none, so none
of its images is short of anything, and counting them would report a whole pass
as degraded for doing exactly what it was told.

Both the local driver and the cloud-task worker keep that meaning, and the two
are checked separately because they carry their own copy of the loop.
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import oops
import pytest
from filecache import FCPath

from spindoctor.cli import sd_mosaic
from spindoctor.cli.reproj.pointing_source import FilePointingSource, PointingSource
from spindoctor.config import LogLevels, LogSinks, RunLogging
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.obs import ObsSnapshotInst

_STUB = 'COISS_2001/data/1461994336_1462054659/N1000000000_1_CALIB'


class _StubReprojResult:
    """Stands in for a reprojection, which this count does not need computed."""

    def save(self, path: Any) -> None:
        """Pretend to write the product.

        Parameters:
            path: Ignored.
        """


class _StubObs:
    """Placeholder observation carrying just what the pointing applier touches."""

    def __init__(self) -> None:
        """Give the stub a real FOV to wrap and a no-op cache reset."""
        self.fov: Any = oops.fov.FlatFOV((0.001, 0.001), (4, 4))

    def reset_all(self) -> None:
        """Pretend to clear the cached geometry."""


class _StubObsClass:
    """Observation class whose images always load."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> object:
        """Return a placeholder observation.

        Parameters:
            path: Ignored.
            **kwargs: Ignored.

        Returns:
            A stub observation; only the pointing applier inspects it here.
        """
        return _StubObs()


class _OneImage:
    """A dataset enumerating exactly one image, under a fixed stub."""

    def __init__(self, root: Path) -> None:
        """Name the directory the image pretends to live in.

        Parameters:
            root: Directory the image file URL is built under.
        """
        self._root = root

    def yield_image_files_from_arguments(self, arguments: argparse.Namespace) -> Iterator[Any]:
        """Yield the one image.

        Parameters:
            arguments: Ignored.

        Yields:
            One batch holding one image.
        """
        name = _STUB.rsplit('/', 1)[-1]
        yield ImageFiles(
            image_files=[
                ImageFile(
                    image_file_url=FCPath(self._root / f'{name}.IMG'),
                    label_file_url=FCPath(self._root / f'{name}.LBL'),
                    results_path_stub=_STUB,
                    index_file_row={},
                )
            ]
        )


def _uncorrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pointing_source: PointingSource
) -> int:
    """Run one reprojection pass over one image and return its uncorrected count.

    Parameters:
        tmp_path: Directory the outputs and logs are written under.
        monkeypatch: Patcher, which reverts the dataset after the test.
        pointing_source: Where the pass reads its navigation records.

    Returns:
        ``n_uncorrected`` for the pass.
    """
    monkeypatch.setattr(sd_mosaic, 'DATASET', _OneImage(tmp_path))
    run_logging = RunLogging(
        sinks=LogSinks(log_root=FCPath(tmp_path / 'logs')),
        levels=LogLevels(),
        timestamp='2026-08-09T12-00-00',
        main_log_path=None,
    )
    _n_done, _n_skipped, _n_failed, n_uncorrected = sd_mosaic._run_reproject_pass(
        run_logging,
        args=argparse.Namespace(
            overwrite=True, dry_run=False, no_write_output_files=True, image_name=None
        ),
        pointing_source=pointing_source,
        output_dir=FCPath(tmp_path / 'out'),
        prefix='',
        fmt='fits',
        subject_name='SATURN',
        obs_class=cast(type[ObsSnapshotInst], _StubObsClass),
        reproject_fn=lambda obs, name: cast(Any, _StubReprojResult()),
    )
    return n_uncorrected


def test_a_pass_that_asked_for_no_pointing_counts_nothing_as_uncorrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was asked for, so nothing this pass processed is missing anything.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    assert _uncorrected(tmp_path, monkeypatch, pointing_source=FilePointingSource(None)) == 0


def test_a_pass_that_asked_and_found_nothing_counts_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for it: a pass given a root and no record is short one pointing.

    Without this the fix above would be indistinguishable from a counter that
    stopped counting.

    Parameters:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest monkeypatch fixture.
    """
    source = FilePointingSource(FCPath(tmp_path / 'nav'))
    assert _uncorrected(tmp_path, monkeypatch, pointing_source=source) == 1
