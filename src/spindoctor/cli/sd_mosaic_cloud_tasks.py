#!/usr/bin/env python3
################################################################################
# sd_mosaic_cloud_tasks.py
#
# Reprojection driver when image batches are provided by cloud_tasks. Each task
# reprojects the images it names and writes per-image reprojection files under
# the task's ``output_dir``. The mode (``'rings'`` or ``'body'``) is read from
# each task's ``task_data['mode']`` field, so a single worker process can
# handle mixed ring and body tasks from the same queue.
#
# The mosaic-combination pass (accumulating reprojections into a final mosaic)
# is not performed here; run ``sd_mosaic <mode> --skip-reproject`` after all
# tasks complete to produce the final mosaic.
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
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.reproj.factories import build_body_mosaic, build_ring_mosaic
from spindoctor.cli.reproj.offsets import apply_offset_to_obs, load_offset_if_any
from spindoctor.cli.reproj.paths import per_image_output_path
from spindoctor.cli.reproj.reproject import reproject_one_body, reproject_one_ring
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    get_nav_results_root,
    image_log_handlers,
    load_default_and_user_config,
)
from spindoctor.dataset import dataset_name_to_inst_name
from spindoctor.dataset.dataset import ImageFile
from spindoctor.obs import ObsSnapshotInst, inst_name_to_obs_class
from spindoctor.reproj.bodies import BodyMosaic, BodyReprojResult
from spindoctor.reproj.rings import RingMosaic, RingReprojResult


def _resolve_nav_results_root_fcpath(cli_args: argparse.Namespace) -> FCPath | None:
    """Return the nav results root as an ``FCPath``, or ``None`` if unset."""
    try:
        nav_results_root_str = get_nav_results_root(cli_args, DEFAULT_CONFIG)
    except ValueError:
        return None
    return FileCache(None).new_path(nav_results_root_str)


def _log_main_exception(msg: str, *args: object) -> None:
    """Log an exception with full traceback (frames plus final error line)."""
    MAIN_LOGGER.exception(msg, *args, stacktrace=False, more=traceback.format_exc())


def _safe_stub_for_image_log(results_path_stub: object, *, default: str = 'image') -> str:
    """Reduce ``results_path_stub`` to one filename-safe segment (no directory components)."""
    if not isinstance(results_path_stub, str):
        return default
    if '\x00' in results_path_stub:
        return default
    base = os.path.basename(results_path_stub.replace('\\', '/'))
    if not base:
        return default
    safe = ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in base)
    safe = safe.strip('._') or default
    return safe[:200]


def _resolved_image_log_path(
    output_dir: FCPath, results_path_stub: object, timestamp: str
) -> FCPath:
    """Resolve ``<output_dir>/logs/<stub>_<timestamp>.log`` and ensure it stays under ``logs``."""
    logs_dir = (output_dir / 'logs').resolve()
    for stub in (
        _safe_stub_for_image_log(results_path_stub),
        _safe_stub_for_image_log('', default='image'),
    ):
        candidate = (output_dir / 'logs' / f'{stub}_{timestamp}.log').resolve()
        if candidate.is_relative_to(logs_dir):
            return candidate
    raise ValueError(f'Refusing image log path outside output_dir/logs (root={logs_dir!r})')


def process_task(
    task_id: str, task_data: dict[str, Any], worker_data: WorkerData
) -> tuple[bool, Any]:
    """Reproject the images named in a single cloud_tasks task.

    Parameters:
        task_id: The ID of the task.
        task_data: The data for the task. It is expected to contain the following keys:
            - "mode": ``'rings'`` or ``'body'``. Selects the mosaic factory and
              reprojection function for this task.
            - "dataset_name": The name of the dataset.
            - "files": The files to process. Each is a dict with the keys:
                - "image_file_url": The URL of the image file.
                - "label_file_url": The URL of the label file.
                - "results_path_stub": The path stub for the results.
                - "index_file_row": The row from the index file for the image file.
            - "arguments": A dict of every per-task parameter (all of the CLI
              arguments previously accepted by ``sd_mosaic``'s rings/body modes
              except ``--config-file`` and ``--nav-results-root``). Keys use the
              same snake_case names as the argparse destinations; for example
              ``output_dir``, ``prefix``, ``format``, ``overwrite``,
              ``no_write_output_files``, ``image_name``, and the body- or
              ring-specific fields consumed by
              :func:`spindoctor.cli.reproj.factories.build_body_mosaic` /
              :func:`spindoctor.cli.reproj.factories.build_ring_mosaic` and
              :func:`spindoctor.cli.reproj.reproject.reproject_one_ring`.
        worker_data: The data for the worker (parsed CLI namespace in ``args``,
            which holds only ``config_file`` and ``nav_results_root``). After worker
            startup, ``nav_results_root_path`` may be set to a precomputed
            :class:`filecache.FCPath` for offset loading.

    Returns:
        Tuple of ``(retry, result)``. ``retry`` is always ``False``. ``result`` is
        ``{'status': 'success'}`` on success, or ``{'status': 'error', 'status_error': ...}``
        (and optionally ``status_exception``) on failure.
    """
    if not isinstance(task_data, dict):
        return False, {'status': 'error', 'status_error': 'invalid_task_data_type'}

    mode = task_data.get('mode')
    if mode is None:
        return False, {'status': 'error', 'status_error': 'no_mode'}
    if mode not in ('rings', 'body'):
        return False, {
            'status': 'error',
            'status_error': 'invalid_mode',
            'status_exception': f'mode must be "rings" or "body", got {mode!r}',
        }

    cli_args = cast(argparse.Namespace, worker_data.args)
    load_default_and_user_config(cli_args, DEFAULT_CONFIG)

    nav_results_root_path = cast(FCPath | None, getattr(worker_data, 'nav_results_root_path', None))

    dataset_name = task_data.get('dataset_name')
    if dataset_name is None:
        return False, {'status': 'error', 'status_error': 'no_dataset_name'}
    if not isinstance(dataset_name, str):
        return False, {'status': 'error', 'status_error': 'invalid_dataset_name'}
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
    if not isinstance(files, list):
        return False, {'status': 'error', 'status_error': 'invalid_files_type'}

    task_arguments = task_data.get('arguments')
    if task_arguments is None:
        return False, {'status': 'error', 'status_error': 'no_arguments'}
    if not isinstance(task_arguments, dict):
        return False, {'status': 'error', 'status_error': 'invalid_arguments_type'}

    output_dir_str = task_arguments.get('output_dir')
    if output_dir_str is None:
        return False, {'status': 'error', 'status_error': 'no_output_dir'}
    if not isinstance(output_dir_str, str):
        return False, {'status': 'error', 'status_error': 'invalid_output_dir_type'}

    prefix: str = task_arguments.get('prefix', '')
    fmt: str = task_arguments.get('format', 'fits')
    overwrite: bool = task_arguments.get('overwrite', False)
    no_write_output_files: bool = task_arguments.get('no_write_output_files', False)
    image_name_override: str | None = task_arguments.get('image_name', None)

    output_dir = FCPath(output_dir_str)
    task_args = argparse.Namespace(**task_arguments)
    mosaic: BodyMosaic | RingMosaic
    if mode == 'body':
        mosaic = build_body_mosaic(task_args)
    else:
        mosaic = build_ring_mosaic(task_args)

    for file in files:
        if not isinstance(file, dict):
            return False, {'status': 'error', 'status_error': 'invalid_file_entry_type'}
        image_file_url = file.get('image_file_url', None)
        label_file_url = file.get('label_file_url', None)
        results_path_stub = file.get('results_path_stub', None)
        index_file_row = file.get('index_file_row', None)
        if index_file_row is not None and not isinstance(index_file_row, dict):
            return False, {'status': 'error', 'status_error': 'invalid_index_file_row_type'}
        index_row: dict[str, Any] = index_file_row if isinstance(index_file_row, dict) else {}
        if image_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_image_file_url'}
        if not isinstance(image_file_url, str):
            return False, {'status': 'error', 'status_error': 'invalid_image_file_url'}
        if label_file_url is None:
            return False, {'status': 'error', 'status_error': 'no_label_file_url'}
        if not isinstance(label_file_url, str):
            return False, {'status': 'error', 'status_error': 'invalid_label_file_url'}
        if results_path_stub is None:
            return False, {'status': 'error', 'status_error': 'no_results_path_stub'}
        if not isinstance(results_path_stub, str):
            return False, {'status': 'error', 'status_error': 'invalid_results_path_stub'}
        image_file = ImageFile(
            image_file_url=FCPath(image_file_url),
            label_file_url=FCPath(label_file_url),
            results_path_stub=results_path_stub,
            index_file_row=index_row,
        )

        out_path = per_image_output_path(
            output_dir,
            prefix,
            image_file,
            fmt=fmt,
            subject_name=mosaic.body_name,
        )

        if not overwrite and out_path.exists():
            MAIN_LOGGER.debug('Skipping (exists): %s', out_path)
            continue

        timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        try:
            image_log_path = _resolved_image_log_path(
                output_dir, image_file.results_path_stub, timestamp
            )
        except ValueError as exc:
            return False, {
                'status': 'error',
                'status_error': 'invalid_image_log_path',
                'status_exception': str(exc),
            }
        image_log_path.parent.mkdir(parents=True, exist_ok=True)
        local_handlers = image_log_handlers(image_log_path, cli_args, DEFAULT_CONFIG)

        try:
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
                    if mode == 'body':
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
        finally:
            for handler in local_handlers:
                handler.close()

    return False, {'status': 'success'}  # No retry under any circumstances


async def async_main() -> None:
    """Async CLI entry for the cloud_tasks reprojection worker.

    Parses only ``--config-file`` and ``--nav-results-root`` from ``sys.argv``;
    all other parameters (mode, output directory, mosaic geometry, etc.) are
    read per-task from ``task_data``. A single worker process can therefore
    handle a queue that mixes ring and body tasks. Before the worker starts,
    default and user config are loaded and
    ``worker._data.nav_results_root_path`` is precomputed for tasks.
    """
    argparser = argparse.ArgumentParser(
        prog='sd_mosaic_cloud_tasks',
        description='Cloud Tasks reprojection worker for ring and body mosaics.',
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
            'Root directory of sd_offset results. When provided, pre-computed offsets '
            'from _metadata.json files are applied before reprojection.'
        ),
    )

    worker = Worker(process_task, args=sys.argv[1:], argparser=argparser)
    init_cli_args = cast(argparse.Namespace, worker._data.args)
    load_default_and_user_config(init_cli_args, DEFAULT_CONFIG)
    # ``WorkerData`` has no typed field; ``process_task`` reads via ``getattr``.
    worker._data.nav_results_root_path = _resolve_nav_results_root_fcpath(init_cli_args)  # type: ignore[attr-defined]
    await worker.start()


def main() -> None:  # Required for setuptools entry points
    """Synchronous entry point; runs ``asyncio.run(async_main())``."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()
