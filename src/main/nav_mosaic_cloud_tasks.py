#!/usr/bin/env python3
################################################################################
# nav_mosaic_cloud_tasks.py
#
# Reprojection driver when image batches are provided by cloud_tasks. Each task
# reprojects the images it names and writes per-image reprojection files under
# the task's ``output_dir``. The mosaic-combination pass (accumulating
# reprojections into a final mosaic) is not performed here; run ``nav_mosaic
# <mode> --skip-reproject`` after all tasks complete to produce the final
# mosaic.
#
# CLI accepts only ``--config-file`` and ``--nav-results-root``. Every other
# parameter (output directory, format, mosaic geometry, body/planet selection,
# etc.) is read from each task's ``task_data['arguments']`` dict.
################################################################################

import argparse
import asyncio
import os
import sys
import traceback
from datetime import datetime
from typing import Any, cast

from cloud_tasks.worker import Worker, WorkerData
from filecache import FCPath, FileCache

# Make CLI runnable from source tree with
#    python src/package
package_source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, package_source_path)

from nav.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    get_nav_results_root,
    image_log_handlers,
    load_default_and_user_config,
)
from nav.dataset import dataset_name_to_inst_name
from nav.dataset.dataset import ImageFile
from nav.obs import ObsSnapshotInst, inst_name_to_obs_class
from nav.reproj.bodies import BodyMosaic, BodyReprojResult
from nav.reproj.rings import RingMosaic, RingReprojResult
from reproj_cli.factories import build_body_mosaic, build_ring_mosaic
from reproj_cli.offsets import apply_offset_to_obs, load_offset_if_any
from reproj_cli.paths import per_image_output_path
from reproj_cli.reproject import reproject_one_body, reproject_one_ring

# Mode (``'rings'`` or ``'body'``) captured from argv before the Worker is
# started. ``process_task`` reads this to pick the right mosaic factory and
# reprojection function.
_MODE: str = ''


def _log_main_exception(msg: str, *args: object) -> None:
    """Log an exception with full traceback (frames plus final error line)."""
    MAIN_LOGGER.exception(msg, *args, stacktrace=False, more=traceback.format_exc())


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Reproject the images named in a single cloud_tasks task.

    Parameters:
        task_id: The ID of the task.
        task_data: The data for the task. It is expected to contain the following keys:
            - "dataset_name": The name of the dataset.
            - "files": The files to process. Each is a dict with the keys:
                - "image_file_url": The URL of the image file.
                - "label_file_url": The URL of the label file.
                - "results_path_stub": The path stub for the results.
                - "index_file_row": The row from the index file for the image file.
            - "arguments": A dict of every per-task parameter (all of the CLI
              arguments previously accepted by ``nav_mosaic``'s rings/body modes
              except ``--config-file`` and ``--nav-results-root``). Keys use the
              same snake_case names as the argparse destinations; for example
              ``output_dir``, ``prefix``, ``format``, ``overwrite``,
              ``no_write_output_files``, ``image_name``, and the body- or
              ring-specific fields consumed by
              :func:`reproj_cli.factories.build_body_mosaic` /
              :func:`reproj_cli.factories.build_ring_mosaic` and
              :func:`reproj_cli.reproject.reproject_one_ring`.
        worker_data: The data for the worker (parsed CLI namespace in ``args``,
            which holds only ``config_file`` and ``nav_results_root``).

    Returns:
        Tuple of ``(retry, result)``. ``retry`` is always ``False``; ``result`` is
        ``None`` on success or a ``{'status': 'error', ...}`` dict on failure.
    """
    cli_args = cast(argparse.Namespace, worker_data.args)
    load_default_and_user_config(cli_args, DEFAULT_CONFIG)

    nav_results_root_path: FCPath | None = None
    try:
        nav_results_root_str = get_nav_results_root(cli_args, DEFAULT_CONFIG)
    except ValueError:
        nav_results_root_str = None
    if nav_results_root_str:
        nav_results_root_path = FileCache(None).new_path(nav_results_root_str)

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

    task_arguments = task_data.get('arguments')
    if task_arguments is None:
        return False, {'status': 'error', 'status_error': 'no_arguments'}
    task_args = argparse.Namespace(**task_arguments)

    output_dir_str = getattr(task_args, 'output_dir', None)
    if output_dir_str is None:
        return False, {'status': 'error', 'status_error': 'no_output_dir'}

    mosaic: BodyMosaic | RingMosaic
    if _MODE == 'body':
        mosaic = build_body_mosaic(task_args)
    else:
        mosaic = build_ring_mosaic(task_args)

    output_dir = FCPath(output_dir_str)
    prefix: str = getattr(task_args, 'prefix', '')
    fmt: str = getattr(task_args, 'format', 'fits')
    overwrite: bool = getattr(task_args, 'overwrite', False)
    no_write_output_files: bool = getattr(task_args, 'no_write_output_files', False)
    image_name_override: str | None = getattr(task_args, 'image_name', None)

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

        out_path = per_image_output_path(
            output_dir, prefix, image_file, fmt, subject_name=mosaic.body_name
        )

        if not overwrite and out_path.exists():
            MAIN_LOGGER.debug('Skipping (exists): %s', out_path)
            continue

        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        image_log_path = (
            FCPath(output_dir) / 'logs' / (image_file.results_path_stub + '_' + timestamp + '.log')
        )
        image_log_path.parent.mkdir(parents=True, exist_ok=True)
        local_handlers = image_log_handlers(image_log_path, cli_args, DEFAULT_CONFIG)

        with IMAGE_LOGGER.open(
            f'REPROJECT {image_file.image_file_url}',
            handler=local_handlers,
        ):
            try:
                image_path = image_file.image_file_path.absolute()
                obs = obs_class.from_file(image_path, extfov_margin_vu=(0, 0))

                offset = load_offset_if_any(nav_results_root_path, image_file)
                if offset is not None:
                    apply_offset_to_obs(cast(ObsSnapshotInst, obs), offset[0], offset[1])

                img_label = (
                    image_name_override
                    if image_name_override is not None
                    else image_file.image_file_path.stem
                )
                obs_inst = cast(ObsSnapshotInst, obs)
                result: BodyReprojResult | RingReprojResult
                if _MODE == 'body':
                    result = reproject_one_body(
                        obs_inst, cast(BodyMosaic, mosaic), image_name=img_label
                    )
                else:
                    result = reproject_one_ring(
                        obs_inst, task_args, cast(RingMosaic, mosaic), image_name=img_label
                    )

                if not no_write_output_files:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    result.save(out_path)
                    MAIN_LOGGER.info('Saved reproj: %s', out_path)
            except Exception:
                _log_main_exception('Error reprojecting %s', image_file.image_file_url)
            finally:
                MAIN_LOGGER.info('Wrote reprojection log to %s', image_log_path)

    return False, None  # No retry under any circumstances


async def async_main() -> None:
    global _MODE

    argv = sys.argv[1:]
    if not argv or argv[0] not in ('rings', 'body'):
        print(
            'Usage: nav_mosaic_cloud_tasks <rings|body> [options]',
            file=sys.stderr,
        )
        sys.exit(1)
    _MODE = argv[0]
    rest = argv[1:]

    argparser = argparse.ArgumentParser(
        prog=f'nav_mosaic_{_MODE}_cloud_tasks',
        description=(
            'Reproject ring images (Cloud Tasks version).'
            if _MODE == 'rings'
            else 'Reproject body images (Cloud Tasks version).'
        ),
    )
    env = argparser.add_argument_group('Environment')
    env.add_argument(
        '--config-file',
        action='append',
        default=None,
        help=(
            'Config file(s) to override default settings; may be specified multiple '
            'times. If omitted, attempts to load ./nav_default_config.yaml if present.'
        ),
    )
    env.add_argument(
        '--nav-results-root',
        type=str,
        default=None,
        help=(
            'Root directory of nav_offset results. When provided, pre-computed offsets '
            'from _metadata.json files are applied before reprojection.'
        ),
    )

    worker = Worker(process_task, args=rest, argparser=argparser)
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    asyncio.run(async_main())


def rings_main() -> None:
    """Entry point for ``nav_mosaic_rings_cloud_tasks``; prepends ``rings`` to argv."""
    sys.argv = [sys.argv[0], 'rings', *sys.argv[1:]]
    main()


def body_main() -> None:
    """Entry point for ``nav_mosaic_body_cloud_tasks``; prepends ``body`` to argv."""
    sys.argv = [sys.argv[0], 'body', *sys.argv[1:]]
    main()


if __name__ == '__main__':
    main()
