"""How a cloud-task worker gets the source it reads navigation records through.

The framework runs each task in a process it spawns for that task and hands the
task the worker's shared data by serializing it.  So a source built at worker
startup is a source no task ever receives, and an index-backed one -- which
holds a database engine and a connection pool -- cannot be serialized at all.
Each task therefore builds its own from the worker's command line, which is
what does cross, and closes it when it is done.

The failure that follows from getting this wrong is silent in exactly the way
this phase exists to prevent: a task that could not obtain the source it was
meant to use, and looked nothing up instead, reprojects its whole batch on
uncorrected pointing and reports the same counts as one that applied every
recorded attitude it was given.  So a named index that cannot be opened fails
the task by name rather than degrading it.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from cloud_tasks.worker import WorkerData
from filecache import FCPath

from spindoctor.cli import sd_backplanes_cloud_tasks, sd_mosaic_cloud_tasks
from spindoctor.cli.reproj.pointing_source import FilePointingSource, PointingSource

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
        The stub, typed as the worker data a driver expects.
    """
    return cast(WorkerData, _StubWorkerData(**kwargs))


def _arguments(url: str | None) -> argparse.Namespace:
    """Build the parsed command line a worker resolves its index from.

    Parameters:
        url: The value of ``--results-db``, or None for no index.

    Returns:
        The namespace.
    """
    return argparse.Namespace(results_db=url)


def _absent_index(tmp_path: Path) -> str:
    """Return a URL naming an index file that was never written.

    Parameters:
        tmp_path: Directory the named file would have been in.

    Returns:
        The URL.
    """
    return f'sqlite:///{(tmp_path / "nowhere" / "index.sqlite3").as_posix()}'


class _StubMosaic:
    """Stands in for a mosaic, which these tests do not need to build."""

    body_name = 'SATURN'


class _ClosedWhenTheTaskEnds:
    """A source that records whether whoever took it gave it back."""

    def __init__(self) -> None:
        """Start out open."""
        self.closed = False

    def read_record(self, image_file: Any) -> dict[str, Any]:
        """Answer for no image.

        Parameters:
            image_file: Ignored.

        Returns:
            Nothing; this is never reached.

        Raises:
            FileNotFoundError: Always.
        """
        raise FileNotFoundError('nothing recorded this image')

    def load_pointing(self, image_file: Any) -> Any:
        """Answer for no image.

        Parameters:
            image_file: Ignored.

        Returns:
            Nothing; this is never reached.

        Raises:
            FileNotFoundError: Always.
        """
        raise FileNotFoundError('nothing recorded this image')

    def close(self) -> None:
        """Record that the task released it."""
        self.closed = True


def _mosaic_task(tmp_path: Path, worker: WorkerData) -> tuple[bool, Any]:
    """Run one reprojection task over an empty batch.

    An empty batch reaches the source and nothing else, which is what these
    assert on.

    Parameters:
        tmp_path: Directory the task writes under.
        worker: The worker data the task reads its command line from.

    Returns:
        The driver's ``(retry, result)``.
    """
    return sd_mosaic_cloud_tasks.process_task(
        'task-1',
        {
            'mode': 'rings',
            'dataset_name': _DATASET,
            'files': [],
            'arguments': {'output_dir': FCPath(tmp_path).as_posix()},
        },
        worker,
    )


def _backplane_task(tmp_path: Path, worker: WorkerData) -> tuple[bool, Any]:
    """Run one backplane task over an empty batch.

    Parameters:
        tmp_path: Directory the task writes under.
        worker: The worker data the task reads its command line from.

    Returns:
        The driver's ``(retry, result)``.
    """
    return sd_backplanes_cloud_tasks.process_task(
        'task-1', {'dataset_name': _DATASET, 'files': []}, worker
    )


def test_a_task_with_no_index_reads_documents(tmp_path: Path) -> None:
    """The default is unchanged: no URL means the documents, as it always does."""
    built = sd_backplanes_cloud_tasks._task_pointing_source(_arguments(None), FCPath(tmp_path))
    assert isinstance(built, FilePointingSource)


def test_a_task_builds_its_source_from_the_workers_own_command_line(tmp_path: Path) -> None:
    """Which is the one thing that crosses into the process a task runs in."""
    url = _absent_index(tmp_path)
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        sd_backplanes_cloud_tasks._task_pointing_source(_arguments(url), FCPath(tmp_path))


def _runner(driver: str) -> Callable[[Path, WorkerData], tuple[bool, Any]]:
    """Return the function that runs one task of the named driver.

    Parameters:
        driver: ``'backplanes'`` or ``'mosaic'``.

    Returns:
        The runner for that driver.
    """
    return _backplane_task if driver == 'backplanes' else _mosaic_task


def _refused_index_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str
) -> tuple[bool, Any]:
    """Run one task of the named driver against an index that will not open.

    The whole worker contract is written once here, so a change to what a
    worker is handed is made once for every assertion about the refusal.

    Parameters:
        tmp_path: Directory the task writes under.
        monkeypatch: Fixture used to stub the mosaic factory.
        driver: Which of the two drivers is under test.

    Returns:
        The driver's ``(retry, result)``.
    """
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_ring_mosaic', lambda *a, **k: _StubMosaic())
    worker = _worker_data(
        nav_results_root=FCPath(tmp_path).as_posix(),
        backplane_results_root=FCPath(tmp_path).as_posix(),
        results_db=_absent_index(tmp_path),
    )
    return _runner(driver)(tmp_path, worker)


@pytest.mark.parametrize('driver', ['backplanes', 'mosaic'])
def test_an_index_that_cannot_be_opened_fails_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    """Both drivers refuse rather than degrade, and they refuse the same way.

    Parameters:
        tmp_path: Directory the task writes under.
        monkeypatch: Fixture used to stub the mosaic factory.
        driver: Which of the two drivers is under test.
    """
    _, result = _refused_index_result(tmp_path, monkeypatch, driver)
    assert result['status_error'] == 'unusable_results_db'


@pytest.mark.parametrize('driver', ['backplanes', 'mosaic'])
def test_such_a_task_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    """A URL that will not open will not open on the next attempt either.

    Parameters:
        tmp_path: Directory the task writes under.
        monkeypatch: Fixture used to stub the mosaic factory.
        driver: Which of the two drivers is under test.
    """
    retry, _ = _refused_index_result(tmp_path, monkeypatch, driver)
    assert retry is False


@pytest.mark.parametrize('driver', ['backplanes', 'mosaic'])
def test_the_refusal_names_what_would_have_written_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    """An operator reading the task result has to know what to fix.

    Parameters:
        tmp_path: Directory the task writes under.
        monkeypatch: Fixture used to stub the mosaic factory.
        driver: Which of the two drivers is under test.
    """
    _, result = _refused_index_result(tmp_path, monkeypatch, driver)
    assert 'sd_stats_ingest' in result['status_exception']


@pytest.mark.parametrize('driver', ['backplanes', 'mosaic'])
def test_the_task_closes_the_source_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str
) -> None:
    """Nothing outlives the task to close it, and an index-backed one pools connections.

    Parameters:
        tmp_path: Directory the task writes under.
        monkeypatch: Fixture used to stub the source and the mosaic factory.
        driver: Which of the two drivers is under test.
    """
    monkeypatch.setattr(sd_mosaic_cloud_tasks, 'build_ring_mosaic', lambda *a, **k: _StubMosaic())
    source = _ClosedWhenTheTaskEnds()
    module = sd_backplanes_cloud_tasks if driver == 'backplanes' else sd_mosaic_cloud_tasks
    monkeypatch.setattr(
        module, 'build_pointing_source', lambda *a, **k: cast(PointingSource, source)
    )
    worker = _worker_data(
        nav_results_root=FCPath(tmp_path).as_posix(),
        backplane_results_root=FCPath(tmp_path).as_posix(),
        results_db=None,
    )
    _runner(driver)(tmp_path, worker)
    assert source.closed
