#!/usr/bin/env python3
"""sd_mosaic -- reproject a dataset of images and combine them into a mosaic.

Entry points
------------
sd_mosaic         -- dispatches on the first positional argument (``rings`` or ``body``)
sd_mosaic_rings   -- equivalent to ``sd_mosaic rings ...``
sd_mosaic_body    -- equivalent to ``sd_mosaic body ...``

Two-pass workflow
-----------------
1. Reprojection pass: for each image in the dataset, load the observation,
   optionally apply a navigation offset from ``--nav-results-root``, call
   ``BodyMosaic.reproject()`` / ``RingMosaic.reproject()``, and save the result.
   Per-image logs are written under ``<output-dir>/logs/``. Existing files are
   skipped unless ``--overwrite`` is given.

2. Mosaic pass: re-iterate the same image list, load each reprojection file
   that exists, call ``mosaic.add()`` (body mode passes resolution merge
   parameters and max incidence/emission/resolution from the CLI explicitly),
   then save the final mosaic.

Either pass may be skipped with ``--skip-reproject`` / ``--skip-mosaic``.
"""

import argparse
import cProfile
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import cast

from filecache import FCPath, FileCache

# Allow running directly from the source tree:
#   python src/spindoctor/cli/sd_mosaic.py rings ...
package_source_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, package_source_path)

from spindoctor.cli.program_names import SD_MOSAIC
from spindoctor.cli.reproj.args import (
    add_body_args,
    add_common_env_args,
    add_common_output_args,
    add_ring_args,
)
from spindoctor.cli.reproj.factories import build_body_mosaic, build_ring_mosaic
from spindoctor.cli.reproj.offsets import apply_offset_to_obs, load_offset_if_any
from spindoctor.cli.reproj.paths import mosaic_output_path, per_image_output_path
from spindoctor.cli.reproj.reproject import reproject_one_body, reproject_one_ring
from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    get_nav_results_root,
    image_log_handlers,
    load_default_and_user_config,
    setup_logging,
)
from spindoctor.dataset import dataset_name_to_class, dataset_name_to_inst_name, dataset_names
from spindoctor.dataset.dataset import DataSet, ImageFile
from spindoctor.obs import ObsSnapshotInst, inst_name_to_obs_class
from spindoctor.reproj.bodies import USE_MOSAIC_LIMITS, BodyMosaicData, BodyReprojResult
from spindoctor.reproj.rings import RingMosaicData, RingReprojResult
from spindoctor.support.file import json_as_string
from spindoctor.support.misc import log_run_environment

PROGRAM_NAME = SD_MOSAIC
"""Program identity: names the main log directory and the
``logging.programs`` configuration block for this program."""


def _reproject_image_log_handlers(
    output_dir: FCPath,
    image_file: ImageFile,
    args: argparse.Namespace,
) -> tuple[list[logging.Handler], FCPath]:
    """Return ``(handlers, log_path)`` for per-image reprojection logs.

    Writes a timestamped file next to the npz/fits products, under
    ``<output_dir>/logs/``, using the same ``output_dir`` as
    :func:`spindoctor.cli.reproj.paths.per_image_output_path`.
    """
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    image_log_path = (
        FCPath(output_dir) / 'logs' / (image_file.results_path_stub + '_' + timestamp + '.log')
    )
    image_log_path.parent.mkdir(parents=True, exist_ok=True)
    return image_log_handlers(image_log_path, args, DEFAULT_CONFIG), image_log_path


def _log_main_exception(msg: str, *args: object) -> None:
    """Log an exception with full traceback (frames plus final error line).

    ``PdsLogger.exception()`` records only ``traceback.format_tb``, which omits
    the ``SomeError: ...`` line at the end. Pass ``traceback.format_exc()`` as
    ``more=`` and disable the default partial stack to avoid duplicating frames.
    """
    MAIN_LOGGER.exception(msg, *args, stacktrace=False, more=traceback.format_exc())


def _run_reproject_pass(
    *,
    args: argparse.Namespace,
    nav_results_root_path: FCPath | None,
    output_dir: FCPath,
    prefix: str,
    fmt: str,
    subject_name: str,
    obs_class: type[ObsSnapshotInst],
    reproject_fn: Callable[[ObsSnapshotInst, str], BodyReprojResult | RingReprojResult],
) -> tuple[int, int]:
    """Reproject each selected image: path checks, per-image logs, offset, save.

    Parameters:
        args: Parsed CLI namespace.
        nav_results_root_path: Optional root for ``sd_offset`` metadata (offsets).
        output_dir: Mosaic / per-image output directory.
        prefix: Output filename prefix.
        fmt: Output format (``fits`` or ``npz``).
        subject_name: Body or planet segment for :func:`per_image_output_path`.
        obs_class: Observation class for :meth:`~spindoctor.obs.ObsSnapshotInst.from_file`.
        reproject_fn: Callable taking ``(obs, image_name)`` and returning a
            reprojection result with ``save()``.

    Returns:
        ``(n_done, n_skipped)`` counts for the pass (dry-run does not increment
        ``n_done``; skipped-existing increments ``n_skipped``).
    """
    assert DATASET is not None
    n_done = 0
    n_skipped = 0
    for imagefiles in DATASET.yield_image_files_from_arguments(args):
        image_file = imagefiles.image_files[0]
        out_path = per_image_output_path(
            output_dir, prefix, image_file, fmt=fmt, subject_name=subject_name
        )

        if not args.overwrite and out_path.exists():
            MAIN_LOGGER.debug('Skipping (exists): %s', out_path)
            n_skipped += 1
            continue

        if args.dry_run:
            MAIN_LOGGER.info(
                '[dry-run] Would reproject %s -> %s', image_file.image_file_url, out_path
            )
            continue

        local_handlers, image_log_path = _reproject_image_log_handlers(output_dir, image_file, args)
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
                        args.image_name
                        if args.image_name is not None
                        else image_file.image_file_path.stem
                    )
                    obs_inst = cast(ObsSnapshotInst, obs)
                    result = reproject_fn(obs_inst, img_label)

                    if not args.no_write_output_files:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        result.save(out_path)
                        MAIN_LOGGER.info('Saved reproj: %s', out_path)
                    n_done += 1
                except Exception:
                    _log_main_exception('Error reprojecting %s', image_file.image_file_url)
                finally:
                    if local_handlers:
                        MAIN_LOGGER.info('Wrote reprojection log to %s', image_log_path)
        finally:
            for handler in local_handlers:
                handler.close()

    return n_done, n_skipped


DATASET: DataSet | None = None
DATASET_NAME: str | None = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser(mode: str) -> argparse.ArgumentParser:
    """Build an ArgumentParser for the given mode (``'rings'`` or ``'body'``)."""
    description = {
        'rings': 'Reproject ring images and build a ring mosaic.',
        'body': 'Reproject body images and build a body mosaic.',
    }[mode]

    parser = argparse.ArgumentParser(
        prog=f'sd_mosaic_{mode}',
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_env_args(parser)
    add_common_output_args(parser)
    parser.add_argument(
        '--output-cloud-tasks-file',
        type=str,
        default=None,
        help=(
            'Write a JSON task descriptions file suitable for loading into a '
            'cloud_tasks queue (consumed by sd_mosaic_{rings,body}_cloud_tasks); '
            'do not perform any reprojection or mosaic passes.'
        ),
    )
    if mode == 'rings':
        add_ring_args(parser)
    else:
        add_body_args(parser)
    return parser


# Keys that live on the CLI namespace but must NOT be forwarded to cloud_tasks
# workers: environment/logging controls handled by the worker's own CLI, and
# sd_mosaic flow-control flags that have no meaning for a single per-image
# task. Dataset-selection arguments (added dynamically by each DataSet) are
# excluded because we iterate them here and the worker only sees concrete file
# URLs.
_CLI_ONLY_TASK_EXCLUDES: frozenset[str] = frozenset(
    {
        'config_file',
        'nav_results_root',
        'pds3_holdings_root',
        'log_level',
        'profile',
        'skip_reproject',
        'skip_mosaic',
        'dry_run',
        'output_cloud_tasks_file',
    }
)


def _task_argument_keys(mode: str) -> list[str]:
    """Return the argparse destinations that belong in each task's ``arguments`` dict.

    Built by re-applying the same output/body/ring arg-group helpers to a throw-away
    parser, so the list stays in sync with :mod:`spindoctor.cli.reproj.args`.

    Parameters:
        mode: ``'rings'`` or ``'body'``.

    Returns:
        A list of argparse ``dest`` names, in declaration order.
    """
    aux = argparse.ArgumentParser(add_help=False)
    add_common_output_args(aux)
    if mode == 'rings':
        add_ring_args(aux)
    else:
        add_body_args(aux)
    return [a.dest for a in aux._actions if a.dest not in {'help', *_CLI_ONLY_TASK_EXCLUDES}]


def _write_cloud_tasks_file(mode: str, args: argparse.Namespace) -> None:
    """Write a cloud_tasks JSON file describing one task per selected image group.

    Parameters:
        mode: ``'rings'`` or ``'body'``.
        args: Parsed CLI namespace; ``args.output_cloud_tasks_file`` is the target path.
    """
    assert DATASET is not None
    assert DATASET_NAME is not None

    task_keys = _task_argument_keys(mode)
    task_arguments = {k: getattr(args, k) for k in task_keys if hasattr(args, k)}

    tasks_json = []
    for imagefile_idx, imagefiles in enumerate(DATASET.yield_image_files_from_arguments(args)):
        task_id = f'{DATASET_NAME}-{imagefiles.image_files[0].label_file_name}-{imagefile_idx}'
        task_files = [
            {
                'image_file_url': f.image_file_url.as_posix(),
                'label_file_url': f.label_file_url.as_posix(),
                'results_path_stub': f.results_path_stub,
                'index_file_row': f.index_file_row,
            }
            for f in imagefiles.image_files
        ]
        tasks_json.append(
            {
                'task_id': task_id,
                'data': {
                    'mode': mode,
                    'arguments': task_arguments,
                    'dataset_name': DATASET_NAME,
                    'files': task_files,
                },
            }
        )

    cloud_tasks_path = FCPath(args.output_cloud_tasks_file)
    with cloud_tasks_path.open('w') as f:
        f.write(json_as_string(tasks_json))


def parse_args(command_list: list[str]) -> tuple[str, argparse.Namespace]:
    """Parse ``command_list`` (the portion after the program name).

    Returns:
        A ``(mode, args)`` tuple where ``mode`` is ``'rings'`` or ``'body'``.
    """
    global DATASET, DATASET_NAME

    if len(command_list) < 1 or command_list[0] not in ('rings', 'body'):
        print('Usage: sd_mosaic <rings|body> <dataset_name> [options]', file=sys.stderr)
        sys.exit(1)

    mode = command_list[0]
    rest = command_list[1:]

    if len(rest) < 1:
        print(f'Usage: sd_mosaic {mode} <dataset_name> [options]', file=sys.stderr)
        sys.exit(1)

    DATASET_NAME = rest[0].lower()
    if DATASET_NAME not in dataset_names():
        print(f'Unknown dataset "{DATASET_NAME}"', file=sys.stderr)
        print(f'Valid datasets: {", ".join(dataset_names())}', file=sys.stderr)
        sys.exit(1)

    try:
        DATASET = dataset_name_to_class(DATASET_NAME)()
    except KeyError:
        print(f'Unknown dataset "{DATASET_NAME}"', file=sys.stderr)
        sys.exit(1)

    parser = _build_parser(mode)
    # The dataset class adds its own selection arguments (index file, volume, etc.)
    DATASET.add_selection_arguments(parser)
    args = parser.parse_args(rest[1:])
    return mode, args


# ---------------------------------------------------------------------------
# Body workflow
# ---------------------------------------------------------------------------


def _run_body(args: argparse.Namespace, nav_results_root_path: FCPath | None) -> None:
    assert DATASET is not None

    inst_name = dataset_name_to_inst_name(DATASET_NAME)  # type: ignore[arg-type]  # DATASET_NAME is set at runtime from argv; dataset_name_to_inst_name is typed for a Literal union of known dataset keys only (false-positive arg-type).
    obs_class = inst_name_to_obs_class(inst_name)

    mosaic = build_body_mosaic(args)
    output_dir = FCPath(args.output_dir)
    prefix: str = args.prefix
    fmt: str = args.format

    # ---- Pass 1: reprojection ------------------------------------------------
    if not args.skip_reproject:
        MAIN_LOGGER.info('=== Reprojection pass (body=%s) ===', mosaic.body_name)
        n_done, n_skipped = _run_reproject_pass(
            args=args,
            nav_results_root_path=nav_results_root_path,
            output_dir=output_dir,
            prefix=prefix,
            fmt=fmt,
            subject_name=mosaic.body_name,
            obs_class=obs_class,
            reproject_fn=lambda obs, image_name: reproject_one_body(
                obs, mosaic, image_name=image_name
            ),
        )
        MAIN_LOGGER.info('Reprojection pass complete: %d done, %d skipped.', n_done, n_skipped)

    # ---- Pass 2: mosaic ------------------------------------------------------
    if not args.skip_mosaic:
        MAIN_LOGGER.info('=== Mosaic pass (body=%s) ===', mosaic.body_name)
        n_added = 0
        for imagefiles in DATASET.yield_image_files_from_arguments(args):
            image_file = imagefiles.image_files[0]
            reproj_path = per_image_output_path(
                output_dir,
                prefix,
                image_file,
                fmt=fmt,
                subject_name=mosaic.body_name,
            )
            if not reproj_path.exists():
                MAIN_LOGGER.info(
                    'Skipping mosaic add for %s: no reprojection file at %s.',
                    image_file.image_file_url,
                    reproj_path,
                )
                continue
            try:
                result = BodyReprojResult.load(reproj_path)
                mosaic.add(
                    result,
                    resolution_threshold=float(args.resolution_threshold),
                    copy_slop=int(args.copy_slop),
                    max_incidence=(
                        math.radians(float(args.max_incidence))
                        if args.max_incidence is not None
                        else USE_MOSAIC_LIMITS
                    ),
                    max_emission=(
                        math.radians(float(args.max_emission))
                        if args.max_emission is not None
                        else USE_MOSAIC_LIMITS
                    ),
                    max_resolution=(
                        float(args.max_resolution)
                        if args.max_resolution is not None
                        else USE_MOSAIC_LIMITS
                    ),
                )
                n_added += 1
            except Exception:
                MAIN_LOGGER.info(
                    'Skipping mosaic add for %s: failed while loading or adding reproj from %s.',
                    image_file.image_file_url,
                    reproj_path,
                )
                _log_main_exception('Error loading reproj %s', reproj_path)

        MAIN_LOGGER.info('Added %d reprojections to mosaic.', n_added)

        if n_added > 0 and not args.dry_run and not args.no_write_output_files:
            mosaic_data: BodyMosaicData = mosaic.to_bounded()
            out_mosaic = mosaic_output_path(output_dir, prefix, fmt, subject_name=mosaic.body_name)
            out_mosaic.parent.mkdir(parents=True, exist_ok=True)
            mosaic_data.save(out_mosaic)
            MAIN_LOGGER.info('Saved mosaic: %s', out_mosaic)


# ---------------------------------------------------------------------------
# Ring workflow
# ---------------------------------------------------------------------------


def _run_rings(args: argparse.Namespace, nav_results_root_path: FCPath | None) -> None:
    assert DATASET is not None

    inst_name = dataset_name_to_inst_name(DATASET_NAME)  # type: ignore[arg-type]  # DATASET_NAME is set at runtime from argv; dataset_name_to_inst_name is typed for a Literal union of known dataset keys only (false-positive arg-type).
    obs_class = inst_name_to_obs_class(inst_name)

    mosaic = build_ring_mosaic(args)
    output_dir = FCPath(args.output_dir)
    prefix: str = args.prefix
    fmt: str = args.format

    # ---- Pass 1: reprojection ------------------------------------------------
    if not args.skip_reproject:
        MAIN_LOGGER.info('=== Reprojection pass (rings, planet=%s) ===', mosaic.body_name)
        n_done, n_skipped = _run_reproject_pass(
            args=args,
            nav_results_root_path=nav_results_root_path,
            output_dir=output_dir,
            prefix=prefix,
            fmt=fmt,
            subject_name=mosaic.body_name,
            obs_class=obs_class,
            reproject_fn=lambda obs, image_name: reproject_one_ring(
                obs, args, mosaic, image_name=image_name
            ),
        )
        MAIN_LOGGER.info('Reprojection pass complete: %d done, %d skipped.', n_done, n_skipped)

    # ---- Pass 2: mosaic ------------------------------------------------------
    if not args.skip_mosaic:
        MAIN_LOGGER.info('=== Mosaic pass (rings, planet=%s) ===', mosaic.body_name)
        n_added = 0
        for imagefiles in DATASET.yield_image_files_from_arguments(args):
            image_file = imagefiles.image_files[0]
            reproj_path = per_image_output_path(
                output_dir,
                prefix,
                image_file,
                fmt=fmt,
                subject_name=mosaic.body_name,
            )
            if not reproj_path.exists():
                MAIN_LOGGER.info(
                    'Skipping mosaic add for %s: no reprojection file at %s.',
                    image_file.image_file_url,
                    reproj_path,
                )
                continue
            try:
                result = RingReprojResult.load(reproj_path)
                mosaic.add(result)
                n_added += 1
            except Exception:
                MAIN_LOGGER.info(
                    'Skipping mosaic add for %s: failed while loading or adding reproj from %s.',
                    image_file.image_file_url,
                    reproj_path,
                )
                _log_main_exception('Error loading reproj %s', reproj_path)

        MAIN_LOGGER.info('Added %d reprojections to mosaic.', n_added)

        if n_added > 0 and not args.dry_run and not args.no_write_output_files:
            mosaic_data: RingMosaicData = mosaic.to_sparse()
            out_mosaic = mosaic_output_path(output_dir, prefix, fmt, subject_name=mosaic.body_name)
            out_mosaic.parent.mkdir(parents=True, exist_ok=True)
            mosaic_data.save(out_mosaic)
            MAIN_LOGGER.info('Saved mosaic: %s', out_mosaic)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def main() -> None:
    """Dispatch on ``rings`` or ``body`` first positional argument."""
    command_list = sys.argv[1:]
    mode, args = parse_args(command_list)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    load_default_and_user_config(args, DEFAULT_CONFIG)

    nav_results_root_str: str | None = None
    if args.nav_results_root is not None:
        nav_results_root_str = args.nav_results_root
    else:
        nav_results_root_str = get_nav_results_root(args, DEFAULT_CONFIG)

    nav_results_root_path: FCPath | None = None
    if nav_results_root_str is not None:
        nav_results_root_path = FileCache(None).new_path(nav_results_root_str)

    try:
        setup_logging(args, DEFAULT_CONFIG, nav_results_root_str or '')
    except (TypeError, ValueError) as exc:
        print(f'Invalid logging configuration: {exc}', file=sys.stderr)
        sys.exit(1)

    # Apply the --log-level console override to the program loggers directly.
    # PdsLogger.set_level accepts a level-name string, so there is no need to
    # reach through the stdlib root logger (which would also re-level every
    # third-party library) to honour the flag.
    log_level = args.log_level
    if log_level is not None and isinstance(log_level, str):
        MAIN_LOGGER.set_level(log_level.upper())
        IMAGE_LOGGER.set_level(log_level.upper())

    start = time.time()
    log_run_environment(MAIN_LOGGER, sys.argv[1:])

    if args.output_cloud_tasks_file:
        MAIN_LOGGER.info('Writing cloud_tasks file to %s', args.output_cloud_tasks_file)
        _write_cloud_tasks_file(mode, args)
        MAIN_LOGGER.info('Wrote cloud_tasks file to %s', args.output_cloud_tasks_file)
        if args.profile:
            pr.disable()
            pr.print_stats(sort='cumulative')
        return

    try:
        if mode == 'body':
            _run_body(args, nav_results_root_path)
        else:
            _run_rings(args, nav_results_root_path)
    finally:
        if args.profile:
            pr.disable()
            pr.print_stats(sort='cumulative')

    MAIN_LOGGER.info('Total elapsed time %.2f sec', time.time() - start)


def rings_main() -> None:
    """Entry point for ``sd_mosaic_rings``; prepends ``rings`` to argv."""
    sys.argv = [sys.argv[0], 'rings', *sys.argv[1:]]
    main()


def body_main() -> None:
    """Entry point for ``sd_mosaic_body``; prepends ``body`` to argv."""
    sys.argv = [sys.argv[0], 'body', *sys.argv[1:]]
    main()


if __name__ == '__main__':
    main()
