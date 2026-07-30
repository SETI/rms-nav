"""Tests that each cloud-task driver isolates logging inside its task.

The isolation has to be applied per task rather than once at startup, because
workers are spawned and do not inherit it, and it has to be applied by every
driver rather than by most of them.  These check the wiring itself: that
``process_task`` resolves its logging through the cloud-task builder, which is
what withholds the terminal.
"""

import argparse
from pathlib import Path
from typing import Any, cast

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
        self._log_root = log_root

    def __call__(self, *args: Any, **kwargs: Any) -> RunLogging:
        """Record the call and resolve logging with the terminal withheld.

        Parameters:
            *args: Passed through by the driver; ignored.
            **kwargs: Passed through by the driver; ignored.

        Returns:
            Logging resolved against the recorder's directory.
        """
        self.called = True
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
    """sd_backplanes_cloud_tasks resolves logging through the cloud-task builder."""
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
    assert recorder.called


def test_the_mosaic_driver_isolates_its_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sd_mosaic_cloud_tasks resolves logging through the cloud-task builder."""
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
    assert recorder.called
