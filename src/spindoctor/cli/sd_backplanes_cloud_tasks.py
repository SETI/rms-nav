#!/usr/bin/env python3
################################################################################
# sd_backplanes_cloud_tasks.py
#
# Backplanes generator when image batches are provided by cloud_tasks.
################################################################################

import argparse
import asyncio
import os
import sys
from typing import Any, cast

from cloud_tasks.worker import Worker, WorkerData
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.backplanes.backplanes import generate_backplanes_image_files
from spindoctor.cli.reproj.pointing_source import build_pointing_source
from spindoctor.config import (
    DEFAULT_CONFIG,
    build_cloud_task_logging,
    get_backplane_results_root,
    get_nav_results_root,
    get_results_db_url,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_BACKPLANES
from spindoctor.dataset import dataset_name_to_inst_name
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.obs import inst_name_to_obs_class

PROGRAM_NAME = SD_BACKPLANES
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Generate backplanes for a single batch of image files.

    Parameters:
        task_id: The ID of the task.
        task_data: The data for the task, carrying ``dataset_name`` and the
            ``files`` to process.
        worker_data: The data for the worker.

    Returns:
        Tuple of ``(retry, result)``.  ``retry`` is always False.  ``result``
        names the error when the task could not run -- including
        ``unusable_results_db`` when an index was named that cannot be opened or
        has not ingested this root, which fails the task rather than falling
        back to reading files -- and otherwise reports whether the image was
        processed or skipped.
    """

    arguments = cast(argparse.Namespace, worker_data.args)
    load_default_and_user_config(arguments, DEFAULT_CONFIG)

    # Derive roots
    try:
        nav_results_root_str = get_nav_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {'status': 'error', 'status_error': 'no_nav_root'}
    nav_results_root = FileCache(None).new_path(nav_results_root_str)

    try:
        backplane_results_root_str = get_backplane_results_root(arguments, DEFAULT_CONFIG)
    except ValueError:
        return False, {'status': 'error', 'status_error': 'no_backplane_root'}
    backplane_results_root = FileCache(None).new_path(backplane_results_root_str)

    dataset_name = task_data.get('dataset_name')
    if dataset_name is None:
        return False, {'status': 'error', 'status_error': 'no_dataset_name'}
    try:
        inst_name = dataset_name_to_inst_name(dataset_name)
    except KeyError:
        return False, {
            'status': 'error',
            'status_error': 'unknown_dataset',
            'status_exception': f'Unknown dataset "{dataset_name}"',
        }
    obs_class = inst_name_to_obs_class(inst_name)
    files = task_data.get('files')
    if files is None:
        return False, {'status': 'error', 'status_error': 'no_files'}
    image_files = []
    for file in files:
        image_file_url = file.get('image_file_url', None)
        label_file_url = file.get('label_file_url', None)
        results_path_stub = file.get('results_path_stub', None)
        index_file_row = file.get('index_file_row', None)
        if image_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_image_file_url'}
        if label_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_label_file_url'}
        if results_path_stub is None:
            return False, {'status': 'error', 'status_error': 'no_results_path_stub'}
        image_file = ImageFile(
            image_file_url=FCPath(image_file_url),
            label_file_url=FCPath(label_file_url),
            results_path_stub=results_path_stub,
            index_file_row=index_file_row,
        )
        image_files.append(image_file)

    # Resolved the same way the interactive driver does, so the same backend's
    # logs land in one tree whichever driver produced them.
    run_logging = build_cloud_task_logging(
        PROGRAM_NAME,
        arguments,
        DEFAULT_CONFIG,
        fallback_log_root=backplane_results_root / 'logs',
    )

    # Built per task, as every other resource this worker uses is: one task is
    # one image, so the open costs nothing against loading it, and the worker
    # keeps its property of resolving its whole environment from the task it was
    # handed rather than from state left over from startup.
    try:
        pointing_source = build_pointing_source(
            nav_results_root, results_db_url=get_results_db_url(arguments, DEFAULT_CONFIG)
        )
    except ValueError as exc:
        return False, {
            'status': 'error',
            'status_error': 'unusable_results_db',
            'status_exception': str(exc),
        }

    try:
        result = generate_backplanes_image_files(
            obs_class,
            ImageFiles(image_files=image_files),
            pointing_source=pointing_source,
            backplane_results_root=backplane_results_root,
            write_output_files=True,
            run_logging=run_logging,
        )
    finally:
        pointing_source.close()

    # Returned rather than only logged: an image can be skipped because its
    # navigation did not succeed, and a task has no run log to say so.
    return False, result  # No retry under any circumstances


async def async_main() -> None:
    argparser = argparse.ArgumentParser(
        description='Backplanes Main Interface (Cloud Tasks version)'
    )

    environment_group = argparser.add_argument_group('Environment')
    environment_group.add_argument(
        '--config-file',
        action='append',
        default=None,
        help="""The configuration file(s) to use to override default settings;
        may be specified multiple times. If not provided, attempts to load
        ./nav_default_config.yaml if present.""",
    )
    environment_group.add_argument(
        '--backplane-results-root',
        type=str,
        default=None,
        help='Root directory for backplane results; overrides NAV_RESULTS_ROOT or config',
    )
    environment_group.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help='Root directory for prior navigation results (metadata, offsets)',
    )
    environment_group.add_argument(
        '--results-db',
        type=str,
        default=None,
        metavar='URL',
        help=(
            'Connection URL of the results index written by sd_stats_ingest; overrides '
            'NAV_RESULTS_DB and the environment.results_db configuration variable. Each '
            "image's navigation record is then read as one row instead of one file. Pass "
            '"none" to read the files even where an index is configured.'
        ),
    )

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    """Synchronous entry point; runs ``asyncio.run(async_main())``."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
