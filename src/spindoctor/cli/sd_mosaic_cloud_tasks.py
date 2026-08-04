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
from typing import Any, cast

import pdslogger
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
    build_cloud_task_logging,
    build_image_log_handlers,
    get_nav_results_root,
    load_default_and_user_config,
)
from spindoctor.config.program_names import SD_MOSAIC
from spindoctor.dataset import dataset_name_to_inst_name
from spindoctor.dataset.dataset import ImageFile
from spindoctor.obs import ObsSnapshotInst, inst_name_to_obs_class
from spindoctor.reproj.bodies import BodyMosaic, BodyReprojResult
from spindoctor.reproj.rings import RingMosaic, RingReprojResult

PROGRAM_NAME = SD_MOSAIC
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""


def _resolve_nav_results_root_fcpath(cli_args: argparse.Namespace) -> FCPath | None:
    """Return the nav results root as an ``FCPath``, or ``None`` if unset."""
    try:
        nav_results_root_str = get_nav_results_root(cli_args, DEFAULT_CONFIG)
    except ValueError:
        return None
    return FileCache(None).new_path(nav_results_root_str)


def _log_image_exception(msg: str, *args: object) -> None:
    """Log an exception about one image, into that image's log.

    Reprojecting an image is allowed to fail without failing the task, so this
    record is the only account of what happened to it.  It belongs to the image
    rather than to the run, which matters here: a cloud task has no main log,
    so a failure reported there would be reported nowhere.

    Parameters:
        msg: Message template, with pdslogger-style ``%s`` placeholders.
        *args: Values substituted into ``msg``.
    """
    IMAGE_LOGGER.exception(msg, *args, stacktrace=False, more=traceback.format_exc())


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
        Tuple of ``(retry, result)``. ``retry`` is always ``False``. ``result``
        is ``{'status': 'error', 'status_error': ...}`` (and optionally
        ``status_exception``) when the task itself could not run.  Otherwise it
        is ``{'status': 'success'}`` with ``n_done``, ``n_skipped`` and
        ``n_failed``: an individual image is allowed to fail without failing
        the task, so the counts are what distinguish a task that reprojected
        its images from one that failed every one of them.  ``n_uncorrected``
        counts images reprojected without a navigation offset, tallied by
        reason under ``uncorrected_reasons``: those images do produce a
        product, and a batch registered entirely on uncorrected pointing is
        otherwise indistinguishable from a good one.  An image whose
        ``results_path_stub`` was refused is additionally named under
        ``rejected_stubs``, since no log could be opened to record it.
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

    # The task's own output directory is the fallback log root, because a
    # worker is not required to have a navigation results root and its logs
    # should not disappear when it does not.
    run_logging = build_cloud_task_logging(
        PROGRAM_NAME,
        cli_args,
        DEFAULT_CONFIG,
        fallback_log_root=FCPath(output_dir_str) / 'logs',
    )

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

    # A task has no run log to report these to, so they are counted and
    # returned: a task that skipped or failed every image would otherwise be
    # indistinguishable from one that reprojected them all.
    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_uncorrected = 0
    uncorrected_reasons: dict[str, int] = {}
    rejected_stubs: list[dict[str, str]] = []

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
            n_skipped += 1
            continue

        try:
            # The log path is not reported anywhere: a cloud task has no
            # console to report it to, and naming the file inside itself tells
            # a later reader nothing they did not have to know already.
            local_handlers, _ = build_image_log_handlers(
                'reproj',
                f'{mosaic.body_name}/{image_file.results_path_stub}',
                run_logging.sinks,
                run_logging.levels,
                timestamp=run_logging.timestamp,
            )
        except ValueError as exc:
            # results_path_stub comes from task data; a stub that would put the
            # log outside the log root is a bad entry rather than a retryable
            # failure.  It fails its own image and the task carries on, because
            # abandoning the batch would discard the images already reprojected
            # and let one malformed entry cost the whole task.  Reported in the
            # result rather than logged: the reason there is nowhere to write
            # this image's log is precisely that its log path was refused.
            rejected_stubs.append({'results_path_stub': results_path_stub, 'reason': str(exc)})
            n_failed += 1
            continue

        try:
            with IMAGE_LOGGER.open(
                f'REPROJECT {image_file.image_file_url}',
                handler=local_handlers,
                level=run_logging.levels.image_section_level(),
            ):
                try:
                    image_path = image_file.image_file_path.absolute()
                    obs = obs_class.from_file(image_path, extfov_margin_vu=(0, 0))

                    lookup = load_offset_if_any(nav_results_root_path, image_file)
                    if lookup.offset is not None:
                        apply_offset_to_obs(
                            cast(ObsSnapshotInst, obs), lookup.offset[0], lookup.offset[1]
                        )
                    elif lookup.reason is not None:
                        # A task has no run log, so the count is what carries
                        # this out: a batch reprojected entirely on uncorrected
                        # pointing looks exactly like a good one otherwise.
                        n_uncorrected += 1
                        uncorrected_reasons[lookup.reason] = (
                            uncorrected_reasons.get(lookup.reason, 0) + 1
                        )

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
                        IMAGE_LOGGER.info('Saved reproj: %s', out_path)
                    n_done += 1
                except Exception:
                    _log_image_exception('Error reprojecting %s', image_file.image_file_url)
                    n_failed += 1
        finally:
            for handler in local_handlers:
                if handler is not pdslogger.NULL_HANDLER:
                    handler.close()

    # No retry under any circumstances.  The status reports that the task ran,
    # not that every image in it reprojected; the counts say which.
    task_result: dict[str, Any] = {
        'status': 'success',
        'n_done': n_done,
        'n_skipped': n_skipped,
        'n_failed': n_failed,
        'n_uncorrected': n_uncorrected,
    }
    if uncorrected_reasons:
        task_result['uncorrected_reasons'] = uncorrected_reasons
    if rejected_stubs:
        task_result['rejected_stubs'] = rejected_stubs
    return False, task_result


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
