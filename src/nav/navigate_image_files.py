"""Top-level driver that navigates a single image and writes results.

Given an observation class and an ``ImageFiles`` batch of size one, this
module reads the image, builds a ``NavOrchestrator`` configured with the
caller's model and technique filters, runs ``orchestrator.navigate(obs)``,
and writes the curated metadata (and a summary PNG when requested) to
``nav_results_root``.

This is the function ``nav_offset`` and ``nav_offset_cloud_tasks`` invoke
once per image.  Errors from image loading, missing SPICE coverage, or
unexpected exceptions during navigation are captured into the output
metadata so the driver always returns a structured result (never a raised
exception that crashes the worker process).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from PIL import Image

from nav.config import DEFAULT_CONFIG, IMAGE_LOGGER, MAIN_LOGGER, image_log_handlers
from nav.dataset.dataset import ImageFiles
from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import (
    NavOrchestrator,
    NavResult,
    build_metadata_dict,
)
from nav.obs import ObsSnapshotInst
from nav.support.file import json_as_string
from nav.support.misc import log_run_environment
from nav.support.summary_png import render_annotated_summary_rgb

__all__ = ['build_metadata_from_result', 'navigate_image_files', 'write_summary_png']


_SPICE_DATA_HINTS = (
    'SPICE(CKINSUFFDATA)',
    'SPICE(SPKINSUFFDATA)',
    'SPICE(NOFRAMECONNECT)',
)


def navigate_image_files(
    obs_class: type[ObsSnapshotInst],
    image_files: ImageFiles,
    nav_results_root: FCPath,
    *,
    nav_models: list[str] | None = None,
    nav_techniques: list[str] | None = None,
    write_output_files: bool = True,
    log_arguments: argparse.Namespace | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Navigate one image batch and optionally write the result files.

    Top-level driver.  Reads the image, asks every registered
    ``NavModel`` subclass to construct whatever instances apply to the
    observation, runs the orchestrator with the caller's model and
    technique filters, and writes ``_metadata.json`` plus
    ``_summary.png`` if requested.

    Parameters:
        obs_class: Concrete ``ObsSnapshotInst`` subclass for the mission.
        image_files: ``ImageFiles`` batch.  Exactly one image per batch is
            supported; calling code should split larger batches.
        nav_results_root: Directory to write ``_metadata.json`` and
            ``_summary.png`` results to.  May be a ``FileCache`` URL.
        nav_models: Glob-pattern list selecting which ``NavModel`` instances
            run; ``None`` means all.  Patterns may use a leading ``!`` for
            exclusion.
        nav_techniques: Glob-pattern list selecting which ``NavTechnique``
            instances run.  ``None`` means all.
        write_output_files: When True, write the metadata JSON and summary
            PNG; when False, perform a dry run and return the metadata
            dict only.
        log_arguments: Parsed CLI arguments used to resolve the per-image
            log-file level.  ``None`` defaults to the configured INFO
            level.

    Returns:
        Tuple ``(success, metadata)`` where ``success`` is True for an
        ``ok`` ``NavResult.status`` and False otherwise.  ``metadata`` is
        the curated JSON-friendly dict.
    """
    logger = IMAGE_LOGGER

    if len(image_files.image_files) != 1:
        logger.error(
            'Expected exactly one image per batch; got %d',
            len(image_files.image_files),
        )
        return False, {
            'status': 'error',
            'status_error': 'expected_one_image_per_batch',
            'status_exception': (
                f'Expected exactly one image per batch; got {len(image_files.image_files)}'
            ),
        }

    image_file = image_files.image_files[0]
    image_url = image_file.image_file_url
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    extra_params = image_file.extra_params
    public_metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = nav_results_root / (image_file.results_path_stub + '_summary.png')

    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    image_log_path = (
        nav_results_root / 'logs' / (image_file.results_path_stub + '_' + timestamp + '.log')
    )
    local_handlers = image_log_handlers(image_log_path, log_arguments, DEFAULT_CONFIG)

    try:
        with logger.open(str(image_url), handler=local_handlers):
            log_run_environment(logger, sys.argv[1:])
            try:
                snapshot = obs_class.from_file(image_url, **extra_params)
            except (OSError, RuntimeError) as exc:
                metadata = _metadata_for_load_error(image_path, image_name, exc, logger)
                if write_output_files:
                    public_metadata_file.write_text(json_as_string(metadata))
                MAIN_LOGGER.info('Wrote log to %s', image_log_path)
                return False, metadata
            snapshot_inst = cast(ObsSnapshotInst, snapshot)
            orchestrator = NavOrchestrator(
                build_models_for_obs(snapshot_inst),
                only_models=nav_models or '*',
                only_techniques=nav_techniques or '*',
            )
            nav_result = orchestrator.navigate(snapshot_inst)
            metadata = build_metadata_from_result(nav_result, image_path, image_name)
            if write_output_files:
                logger.info('Writing metadata to %s', public_metadata_file)
                public_metadata_file.write_text(json_as_string(metadata))
                write_summary_png(snapshot_inst, nav_result, summary_png_file, logger)
            MAIN_LOGGER.info('Wrote log to %s', image_log_path)
            return nav_result.status == 'success', metadata
    finally:
        for handler in local_handlers:
            handler.close()


def _metadata_for_load_error(
    image_path: Path,
    image_name: str,
    exc: BaseException,
    logger: Any,
) -> dict[str, Any]:
    """Build a metadata dict for an image-load or kernel-coverage failure."""
    message = str(exc)
    if any(hint in message for hint in _SPICE_DATA_HINTS):
        logger.exception('No SPICE kernel available for "%s": %s', image_path, message)
        status_error = 'missing_spice_data'
    else:
        logger.exception('Error reading image "%s": %s', image_path, message)
        status_error = 'image_read_error'
    return {
        'status': 'error',
        'status_error': status_error,
        'status_exception': message,
        'observation': {
            'image_path': str(image_path),
            'image_name': image_name,
        },
    }


def build_metadata_from_result(
    result: NavResult, image_path: Path, image_name: str
) -> dict[str, Any]:
    """Build the JSON metadata dict from a NavResult.

    Used by both the autonomous pipeline and the manual-nav driver so a
    manually-picked offset writes the same ``_metadata.json`` schema.

    Parameters:
        result: NavResult to curate.
        image_path: Absolute path to the source image; written to the
            ``observation.image_path`` field.
        image_name: Basename of the source image; written to the
            ``observation.image_name`` field.
    """
    metadata: dict[str, Any] = {
        'status': result.status,
        'observation': {
            'image_path': str(image_path),
            'image_name': image_name,
        },
        'navigation_result': build_metadata_dict(result),
    }
    if result.offset_px is not None:
        metadata['offset'] = list(result.offset_px)
    metadata['confidence'] = result.confidence
    return metadata


def write_summary_png(
    obs: ObsSnapshotInst,
    result: NavResult,
    png_path: FCPath,
    logger: Any,
) -> None:
    """Composite the source image with the orchestrator's annotation overlay.

    Thin driver around
    :func:`nav.support.summary_png.render_annotated_summary_rgb`; writes
    the resulting RGB to ``png_path`` as a PNG.

    Parameters:
        obs: Observation snapshot used as the background.
        result: Navigation result; ``offset_px`` shifts the overlay to the
            best-fit pose, ``annotations`` carries every NavModel's overlay.
        png_path: Destination path; supports ``FCPath`` URLs.
        logger: ``pdslogger`` to emit one INFO line on success.
    """
    overlay_offset = result.offset_px if result.offset_px is not None else (0.0, 0.0)
    rgb = render_annotated_summary_rgb(obs, result.annotations, overlay_offset)
    buf = BytesIO()
    Image.fromarray(rgb, mode='RGB').save(buf, format='PNG')
    png_path.write_bytes(buf.getvalue())
    logger.info('Wrote summary PNG to %s', png_path)
