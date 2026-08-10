"""Each cloud-task driver, across the process boundary its framework runs it over.

``cloud_tasks`` starts one process per task, spawned rather than forked, and
passes the worker's shared data to it by serializing it.  So whatever a driver's
worker startup leaves on that data has to survive being serialized, and whatever
a task needs has to be built where the task runs.  A results index is neither:
it holds a database engine and a connection pool, and serializing one raises in
the *parent*, before a single task starts.

That failure is invisible to every test that calls ``process_task`` in-process,
and it takes out the whole of a driver's index-backed mode rather than one image
of it.  So these drive each driver's own ``async_main`` up to the point the
worker would start, take the shared data it really built, and hand it to a real
spawn context -- which is the operation that fails.  The child then runs a task
and reports what it got, so the boundary is crossed in both directions.

One child runs both drivers' tasks: a spawned interpreter pays for importing
this package from scratch, and every assertion below reads the same run.
"""

import argparse
import asyncio
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from tests.spindoctor.cli.cloud_task_spawn_helpers import run_tasks
from tests.spindoctor.cli.reproj.conftest import (
    FAILED_STUB,
    REASON_TREE,
    UNNAVIGATED_STUB,
    build_tree,
    index_for,
)

from spindoctor.cli import sd_backplanes_cloud_tasks, sd_mosaic_cloud_tasks
from spindoctor.cli.reproj.args import add_common_output_args, add_ring_args

_DATASET = 'COISS_saturn'

_CHILD_TIMEOUT_S = 300.0
"""How long the child is given: it imports this package in a fresh interpreter."""


def _worker_from_startup(module: Any, argv: list[str]) -> Any:
    """Run one driver's startup and return the worker it built.

    The real ``Worker`` is constructed, so the shared data carries everything
    the framework puts there -- the parsed command line and both process events
    -- and only its run loop is replaced.

    Parameters:
        module: The dispatch module.
        argv: The worker's arguments, without the program name.

    Returns:
        The worker, whose ``_data`` is what a task would be handed.
    """
    captured: dict[str, Any] = {}
    real_worker = module.Worker

    class _Captured(real_worker):  # type: ignore[valid-type, misc]
        """The real worker, kept rather than started."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Build the real worker and keep it.

            Parameters:
                *args: The driver's positional arguments.
                **kwargs: The driver's keyword arguments.
            """
            super().__init__(*args, **kwargs)
            captured['worker'] = self

        async def start(self) -> None:
            """Do not run the loop: what is under test is what startup left behind."""

    saved_argv = sys.argv
    module.Worker = _Captured
    sys.argv = [module.__name__, *argv]
    try:
        asyncio.run(module.async_main())
    finally:
        sys.argv = saved_argv
        module.Worker = real_worker
    return captured['worker']


def _ring_task_arguments(output_dir: Path) -> dict[str, Any]:
    """Build the per-task arguments a ring reprojection task carries.

    Taken from the option groups the program declares rather than from a
    hand-written dict, so a task built here is the shape ``sd_mosaic`` really
    enqueues.

    Parameters:
        output_dir: Where the task would write its reprojections.

    Returns:
        The arguments dict.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_common_output_args(parser)
    add_ring_args(parser)
    arguments = parser.parse_args(
        [
            '--no-write-output-files',
            '--output-dir',
            output_dir.as_posix(),
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
        ]
    )
    return dict(vars(arguments))


def _image_entry(stub: str, holdings: Path) -> dict[str, Any]:
    """Build one ``files`` entry naming the stub its record is looked up under.

    The image itself is never written.  Both drivers reach the navigation
    record without one -- the backplane stage decides whether there is work
    from the record before it opens anything -- and an index-backed lookup is
    what these are about.

    Parameters:
        stub: The results path stub.
        holdings: Directory the named image would live in.

    Returns:
        The task's per-image dict.
    """
    name = stub.rsplit('/', 1)[-1]
    return {
        'image_file_url': (holdings / f'{name}.IMG').as_posix(),
        'label_file_url': (holdings / f'{name}.LBL').as_posix(),
        'results_path_stub': stub,
        'index_file_row': {},
    }


@pytest.fixture(scope='module')
def spawned_results(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    """Run one task per driver in a process the spawn context really starts.

    Parameters:
        tmp_path_factory: Fixture used to make the run's directory.

    Returns:
        What each driver's ``process_task`` returned, in driver order:
        reprojection first, backplanes second.
    """
    tmp_path = tmp_path_factory.mktemp('spawn')
    quiet = pdslogger.PdsLogger('cloud_task_spawn_ingest')
    quiet.set_level('ERROR')
    root = tmp_path / 'nav'
    build_tree(root, REASON_TREE)
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet).dispose()
    url = f'sqlite:///{database.as_posix()}'
    task_file = tmp_path / 'tasks.json'
    task_file.write_text('[]', encoding='utf-8')
    holdings = tmp_path / 'holdings'
    holdings.mkdir()

    mosaic_worker = _worker_from_startup(
        sd_mosaic_cloud_tasks,
        [
            '--nav-results-root',
            root.as_posix(),
            '--results-db',
            url,
            '--task-file',
            task_file.as_posix(),
        ],
    )
    backplanes_worker = _worker_from_startup(
        sd_backplanes_cloud_tasks,
        [
            '--nav-results-root',
            root.as_posix(),
            '--backplane-results-root',
            (tmp_path / 'backplanes').as_posix(),
            '--results-db',
            url,
            '--task-file',
            task_file.as_posix(),
        ],
    )

    jobs = [
        (
            'sd_mosaic_cloud_tasks',
            mosaic_worker._data,
            {
                'mode': 'rings',
                'dataset_name': _DATASET,
                'files': [_image_entry(UNNAVIGATED_STUB, holdings)],
                'arguments': _ring_task_arguments(tmp_path / 'out'),
            },
        ),
        (
            'sd_backplanes_cloud_tasks',
            backplanes_worker._data,
            {'dataset_name': _DATASET, 'files': [_image_entry(FAILED_STUB, holdings)]},
        ),
    ]
    results_path = tmp_path / 'results.json'
    context = multiprocessing.get_context('spawn')
    process = context.Process(target=run_tasks, args=(jobs, results_path.as_posix()))
    process.start()
    process.join(timeout=_CHILD_TIMEOUT_S)
    assert process.exitcode == 0
    return list(json.loads(results_path.read_text(encoding='utf-8')))


def test_the_reprojection_task_runs_in_the_process_spawned_for_it(
    spawned_results: list[Any],
) -> None:
    """Its worker data crossed, so the task ran at all.

    Nothing on that data may hold a database engine: serializing one raises in
    the parent, and every task of the run fails before the child starts.

    Parameters:
        spawned_results: What each driver's task returned.
    """
    assert spawned_results[0]['status'] == 'success'


def test_that_task_opened_the_index_where_it_could_use_it(
    spawned_results: list[Any],
) -> None:
    """It built its own source in the child, from the command line that crossed.

    A task that could not open the index fails naming it, so a plain success is
    the evidence that the index opened where the task runs.

    Parameters:
        spawned_results: What each driver's task returned.
    """
    assert 'status_error' not in spawned_results[0]


def test_the_backplane_task_read_a_real_row_in_that_process(
    spawned_results: list[Any],
) -> None:
    """And answered from it: the stub it was given is one the index holds.

    The recorded outcome comes back out of the index rather than out of a
    document, which is the whole of what an index-backed worker does.

    Parameters:
        spawned_results: What each driver's task returned.
    """
    assert spawned_results[1]['nav_status'] == 'error'


def test_the_backplane_task_reports_the_skip_that_row_implies(
    spawned_results: list[Any],
) -> None:
    """The result the enqueuer counts, produced across the boundary.

    Parameters:
        spawned_results: What each driver's task returned.
    """
    assert spawned_results[1]['status'] == 'skipped'
