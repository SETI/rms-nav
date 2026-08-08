"""Tests that each cloud-task driver isolates logging inside its task.

The isolation has to be applied per task rather than once at startup, because
workers are spawned and do not inherit it, and it has to be applied by every
driver rather than by most of them.  These check the wiring itself: that
``process_task`` resolves its logging through the cloud-task builder, which is
what withholds the terminal.
"""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import oops
import pytest
from cloud_tasks.worker import WorkerData
from filecache import FCPath

from spindoctor.cli import (
    sd_backplanes_cloud_tasks,
    sd_mosaic_cloud_tasks,
    sd_offset_cloud_tasks,
)
from spindoctor.config.config import Config
from spindoctor.config.logging_config import RunLogging, build_cloud_task_logging
from spindoctor.config.program_names import SD_BACKPLANES, SD_MOSAIC

_DATASET = 'COISS_saturn'


class _StubWorkerData:
    """Stands in for the cloud_tasks worker's data object."""

    def __init__(self, **kwargs: object) -> None:
        """Build worker data carrying only the given CLI arguments.

        Parameters:
            **kwargs: Argument names and values for the parsed namespace.
        """
        self.args = argparse.Namespace(config_file=None, log_root=None, **kwargs)


def _worker_data(**kwargs: object) -> WorkerData:
    """Build the worker data a driver reads its CLI arguments from.

    Parameters:
        **kwargs: Argument names and values for the parsed namespace.

    Returns:
        The stub, typed as the worker data a driver expects.  A driver reads
        only ``args`` from it, so building the real thing would mean standing
        up a worker for no benefit.
    """
    return cast(WorkerData, _StubWorkerData(**kwargs))


class _StubMosaic:
    """Stands in for a mosaic, which a wiring test does not need to build."""

    body_name = 'SATURN'


class _Recorder:
    """Records whether the cloud-task logging builder was called."""

    def __init__(self, log_root: FCPath) -> None:
        """Prepare a recorder writing its logs under ``log_root``.

        Parameters:
            log_root: Directory to use as the log root.
        """
        self.called = False
        self.program_name: str | None = None
        self.fallback_log_root: Any = None
        self._log_root = log_root

    def __call__(self, *args: Any, **kwargs: Any) -> RunLogging:
        """Record what the driver asked for, and resolve with no terminal.

        The arguments are kept rather than discarded: which program identity a
        driver claims decides which ``logging.programs`` block governs it, and
        its fallback log root is where its logs go when nothing else names
        one.  A recorder that only noted the call would pass with both wrong.

        Parameters:
            *args: The driver's positional arguments.
            **kwargs: The driver's keyword arguments.

        Returns:
            Logging resolved against the recorder's directory.
        """
        self.called = True
        self.program_name = args[0] if args else None
        self.fallback_log_root = kwargs.get('fallback_log_root')
        return build_cloud_task_logging(
            'sd_offset', argparse.Namespace(log_root=self._log_root.as_posix()), _config()
        )


def _config() -> Config:
    """Build a Config carrying the shipped defaults.

    Returns:
        The loaded Config.
    """
    config = Config()
    config.read_config()
    return config


def _image_entry(root: FCPath) -> dict[str, Any]:
    """Build one well-formed ``files`` entry for a task.

    Parameters:
        root: Directory the referenced files would live under.

    Returns:
        The task's per-image dict.
    """
    return {
        'image_file_url': (root / 'N1234567890_1.IMG').as_posix(),
        'label_file_url': (root / 'N1234567890_1.LBL').as_posix(),
        'results_path_stub': 'COISS_2001/N1234567890_1',
        'index_file_row': {},
    }


def test_the_offset_driver_isolates_its_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sd_offset_cloud_tasks resolves logging through the cloud-task builder."""
    recorder = _Recorder(FCPath(tmp_path))
    monkeypatch.setattr(sd_offset_cloud_tasks, 'build_cloud_task_logging', recorder)
    sd_offset_cloud_tasks.process_task(
        'task-1', {}, _worker_data(nav_results_root=FCPath(tmp_path).as_posix())
    )
    assert recorder.called


def test_the_backplanes_driver_isolates_its_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sd_backplanes_cloud_tasks resolves logging through the cloud-task builder.

    Its identity and its fallback root are both asserted: the first decides
    which ``logging.programs`` block governs the worker, the second where its
    logs go when nothing else names a root.
    """
    recorder = _Recorder(FCPath(tmp_path))
    monkeypatch.setattr(sd_backplanes_cloud_tasks, 'build_cloud_task_logging', recorder)
    monkeypatch.setattr(
        sd_backplanes_cloud_tasks, 'generate_backplanes_image_files', lambda *a, **k: None
    )
    sd_backplanes_cloud_tasks.process_task(
        'task-1',
        {'dataset_name': _DATASET, 'files': [_image_entry(FCPath(tmp_path))]},
        _worker_data(
            nav_results_root=FCPath(tmp_path).as_posix(),
            backplane_results_root=FCPath(tmp_path).as_posix(),
        ),
    )
    assert recorder.program_name == SD_BACKPLANES


def test_the_mosaic_driver_isolates_its_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sd_mosaic_cloud_tasks resolves logging through the cloud-task builder.

    A reprojection worker is not required to have a navigation results root,
    so it falls back to the task's own output directory.
    """
    recorder = _Recorder(FCPath(tmp_path))
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_cloud_task_logging', recorder)
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_ring_mosaic', lambda *a, **k: _StubMosaic())
    sd_mosaic_cloud_tasks.process_task(
        'task-1',
        {
            'mode': 'rings',
            'dataset_name': _DATASET,
            'files': [],
            'arguments': {'output_dir': FCPath(tmp_path).as_posix()},
        },
        _worker_data(nav_results_root=FCPath(tmp_path).as_posix()),
    )
    assert recorder.program_name == SD_MOSAIC


def test_the_backplanes_driver_falls_back_to_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its logs go under the backplane results root, not the navigation one."""
    recorder = _Recorder(FCPath(tmp_path))
    monkeypatch.setattr(sd_backplanes_cloud_tasks, 'build_cloud_task_logging', recorder)
    monkeypatch.setattr(
        sd_backplanes_cloud_tasks, 'generate_backplanes_image_files', lambda *a, **k: None
    )
    backplane_root = FCPath(tmp_path) / 'bp'
    sd_backplanes_cloud_tasks.process_task(
        'task-1',
        {'dataset_name': _DATASET, 'files': [_image_entry(FCPath(tmp_path))]},
        _worker_data(
            nav_results_root=FCPath(tmp_path).as_posix(),
            backplane_results_root=backplane_root.as_posix(),
        ),
    )
    assert recorder.fallback_log_root == backplane_root / 'logs'


def test_the_mosaic_driver_falls_back_to_the_task_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reprojection worker need not have a navigation results root at all."""
    recorder = _Recorder(FCPath(tmp_path))
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_cloud_task_logging', recorder)
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_ring_mosaic', lambda *a, **k: _StubMosaic())
    output_dir = FCPath(tmp_path) / 'out'
    sd_mosaic_cloud_tasks.process_task(
        'task-1',
        {
            'mode': 'rings',
            'dataset_name': _DATASET,
            'files': [],
            'arguments': {'output_dir': output_dir.as_posix()},
        },
        _worker_data(nav_results_root=FCPath(tmp_path).as_posix()),
    )
    assert recorder.fallback_log_root == output_dir / 'logs'


# ---------------------------------------------------------------------------
# Reprojecting without an offset is counted, not passed over
# ---------------------------------------------------------------------------


class _StubReprojResult:
    """Stands in for a reprojection, which this test does not need to compute."""

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


def _reproject_task_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, nav_root: FCPath | None
) -> dict[str, Any]:
    """Run one reprojection task far enough to reach the offset lookup.

    Parameters:
        tmp_path: Directory used for output and logs.
        monkeypatch: Fixture used to stub the image load and reprojection.
        nav_root: Navigation results root handed to the task, or None.

    Returns:
        The task result.
    """
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_ring_mosaic', lambda *a, **k: _StubMosaic())
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'inst_name_to_obs_class', lambda _: _StubObsClass)
    monkeypatch.setattr(
        sd_mosaic_cloud_tasks, 'reproject_one_ring', lambda *a, **k: _StubReprojResult()
    )
    worker = _worker_data(nav_results_root=FCPath(tmp_path).as_posix())
    worker.nav_results_root_path = nav_root  # type: ignore[attr-defined]
    _, result = sd_mosaic_cloud_tasks.process_task(
        'task-1',
        {
            'mode': 'rings',
            'dataset_name': _DATASET,
            'files': [_image_entry(FCPath(tmp_path))],
            'arguments': {'output_dir': FCPath(tmp_path).as_posix(), 'no_write_output_files': True},
        },
        worker,
    )
    return cast(dict[str, Any], result)


def test_an_image_reprojected_without_an_offset_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task has no run log, so the count is what carries this out.

    A batch registered entirely on uncorrected pointing otherwise looks
    exactly like one that applied every offset it was given.
    """
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=FCPath(tmp_path) / 'nav')
    assert result['n_uncorrected'] == 1


def test_the_count_comes_with_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong results root and a genuinely unnavigated image are not the same."""
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=FCPath(tmp_path) / 'nav')
    assert result['pointing_reasons'] == {'no_metadata': 1}


def test_an_offset_fallback_is_tallied_but_not_uncorrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image that fell back to the offset path did get a correction.

    Its reason lands in the per-reason tally so the batch's pointing story is
    visible, while ``n_uncorrected`` keeps meaning what it says: images with
    no correction at all.
    """
    nav_root = FCPath(tmp_path) / 'nav'
    Path((nav_root / 'COISS_2001').as_posix()).mkdir(parents=True, exist_ok=True)
    (nav_root / 'COISS_2001' / 'N1234567890_1_metadata.json').write_text(
        json.dumps({'status': 'success', 'offset': [1.0, -2.0]})
    )
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=nav_root)
    assert result['pointing_reasons'] == {'no_pointing_block': 1}
    assert result['n_uncorrected'] == 0


def test_a_missing_offset_key_is_counted_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch pass survives one defect-shaped record and counts it.

    The backplane stage raises on the same record class, because a
    single-image task should fail loudly; a batch pass must not lose the
    whole task to one bad record, so here it is a tally entry instead.
    """
    nav_root = FCPath(tmp_path) / 'nav'
    Path((nav_root / 'COISS_2001').as_posix()).mkdir(parents=True, exist_ok=True)
    (nav_root / 'COISS_2001' / 'N1234567890_1_metadata.json').write_text(
        json.dumps({'status': 'success'})
    )
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=nav_root)
    assert result['pointing_reasons'] == {'missing_offset_key': 1}
    assert result['n_done'] == 1


def test_the_image_is_still_reprojected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncorrected pointing produces a product, so it counts as done too."""
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=FCPath(tmp_path) / 'nav')
    assert result['n_done'] == 1


def test_asking_for_no_offsets_still_counts_uncorrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that never asked for offsets is still an uncorrected run.

    The count exists so a batch reprojected entirely on uncorrected pointing
    is visible in the task result; deliberateness lives in the reason tally,
    which stays empty because nothing was asked for and so nothing degraded.
    """
    result = _reproject_task_result(tmp_path, monkeypatch, nav_root=None)
    assert result['n_uncorrected'] == 1
    assert 'pointing_reasons' not in result
