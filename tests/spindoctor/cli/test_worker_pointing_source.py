"""How long a cloud-task worker's navigation-record source lives.

One backplane task is one image, and the URL the source is built from is the
worker's own command line, identical for every task it is handed.  Rebuilding
the source per task would therefore buy nothing and cost, for an index-backed
one, a connection and two bookkeeping queries per image -- in the stage whose
purpose is removing one round trip per image.  So it is built once and kept.

A build that failed is deliberately not kept, because a worker started while
the index was unreachable would otherwise answer nothing for the rest of its
life.
"""

import argparse
from pathlib import Path
from typing import cast

import pytest
from cloud_tasks.worker import WorkerData
from filecache import FCPath

from spindoctor.cli import sd_backplanes_cloud_tasks
from spindoctor.cli.reproj.pointing_source import FilePointingSource


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


def test_two_tasks_share_one_source(tmp_path: Path) -> None:
    """The second task is answered by the source the first one built."""
    worker = _worker_data()
    first = sd_backplanes_cloud_tasks._worker_pointing_source(
        _arguments(None), FCPath(tmp_path), worker
    )
    second = sd_backplanes_cloud_tasks._worker_pointing_source(
        _arguments(None), FCPath(tmp_path), worker
    )
    assert second is first


def test_the_source_outlives_the_task_that_built_it(tmp_path: Path) -> None:
    """It is kept on the worker, which is what lets the next task find it."""
    worker = _worker_data()
    built = sd_backplanes_cloud_tasks._worker_pointing_source(
        _arguments(None), FCPath(tmp_path), worker
    )
    assert getattr(worker, 'pointing_source', None) is built


def test_a_worker_with_no_index_still_reads_documents(tmp_path: Path) -> None:
    """The default is unchanged: no URL means the documents, as it always does."""
    worker = _worker_data()
    built = sd_backplanes_cloud_tasks._worker_pointing_source(
        _arguments(None), FCPath(tmp_path), worker
    )
    assert isinstance(built, FilePointingSource)


def test_a_build_that_failed_is_not_kept(tmp_path: Path) -> None:
    """A worker started while the index was down answers once it is back up.

    Caching the failure would leave the worker refusing every task it is ever
    handed, which for a long-lived worker turns a moment's outage into a lost
    fleet member.
    """
    worker = _worker_data()
    url = f'sqlite:///{(tmp_path / "nowhere" / "index.sqlite3").as_posix()}'
    with pytest.raises(ValueError, match='sd_stats_ingest'):
        sd_backplanes_cloud_tasks._worker_pointing_source(_arguments(url), FCPath(tmp_path), worker)
    assert getattr(worker, 'pointing_source', None) is None
