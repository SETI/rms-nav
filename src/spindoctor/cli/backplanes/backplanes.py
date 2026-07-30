import json
from typing import Any, cast

import oops
import pdslogger
from filecache import FCPath

from spindoctor.config import (
    DEFAULT_CONFIG,
    IMAGE_LOGGER,
    MAIN_LOGGER,
    RunLogging,
    build_image_log_handlers,
    run_logging_for_root,
)
from spindoctor.dataset.dataset import ImageFiles
from spindoctor.obs import ObsSnapshot, ObsSnapshotInst

from .backplanes_bodies import create_body_backplanes
from .backplanes_rings import create_ring_backplanes
from .merge import merge_sources_into_master
from .writer import write_fits


def generate_backplanes_image_files(
    obs_class: type[ObsSnapshotInst],
    image_files: ImageFiles,
    *,
    nav_results_root: FCPath,
    backplane_results_root: FCPath,
    write_output_files: bool = True,
    run_logging: RunLogging | None = None,
) -> dict[str, Any]:
    """Generate backplanes for a single image batch using prior offset metadata.

    Parameters:
        obs_class: Observation snapshot class for the instrument.
        image_files: List of images; must have exactly one image in the batch.
        nav_results_root: Root containing previously written navigation metadata JSONs.
        backplane_results_root: Destination root for FITS and label files.
        write_output_files: Whether to write outputs to storage.
        run_logging: This run's resolved logging, giving the level and sinks
            the per-image log is written with.  ``None`` resolves the
            configuration's defaults against the backplane results root.

    Returns:
        ``{'status': 'success'}``, or ``{'status': 'skipped'}`` with the
        navigation status that caused the skip.  An image can be skipped for a
        reason its caller has no other way to learn: the reason is reported to
        the run's log, and a cloud task has no run log, so returning it is what
        keeps a batch that quietly skipped everything distinguishable from one
        that processed it.
    """

    logger = IMAGE_LOGGER
    config = DEFAULT_CONFIG
    if run_logging is None:
        run_logging = run_logging_for_root(backplane_results_root / 'logs')

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}'
        )

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    fits_file_path = backplane_results_root / (image_file.results_path_stub + '_backplanes.fits')

    # Decide whether there is work before opening a log for it, so a skipped
    # image does not leave behind a file containing only its own header.  This
    # raises if the metadata is missing or unreadable; the caller reports that.
    metadata_text = metadata_file.read_text()
    nav_metadata = cast(dict[str, Any], json.loads(metadata_text))

    status = nav_metadata.get('status', None)
    if status != 'success':
        nav_error = nav_metadata.get('status_error', 'unknown')
        MAIN_LOGGER.warning(
            'Skipping backplanes for "%s": status=%s error=%s',
            image_path,
            status,
            nav_error,
        )
        return {
            'status': 'skipped',
            'status_error': 'nav_status_not_success',
            'nav_status': status,
            'nav_status_error': nav_error,
        }

    local_handlers, image_log_path = build_image_log_handlers(
        'backplanes',
        image_file.results_path_stub,
        run_logging.sinks,
        run_logging.levels,
        timestamp=run_logging.timestamp,
    )
    try:
        with logger.open(
            f'Processing image: {image_path!s}',
            handler=local_handlers,
            level=run_logging.levels.image_section_level(),
        ):
            # Build observation in original FOV
            # TODO We only support snapshots for backplane generation for now
            obs = obs_class.from_file(image_path, extfov_margin_vu=(0, 0))
            if not isinstance(obs, ObsSnapshot):
                raise TypeError(f'Expected ObsSnapshot, got {type(obs).__name__}')
            snapshot = obs

            # Apply offset via OffsetFOV; metadata uses (dv, du)
            if 'offset' not in nav_metadata:
                raise ValueError(f'{image_path}: "offset" field not found in metadata')
            if nav_metadata['offset'] is None:
                logger.warning('%s: "offset" field is None, using (0, 0)', image_path)
                dv, du = 0, 0
            else:
                dv, du = nav_metadata['offset']
            snapshot.fov = oops.fov.OffsetFOV(snapshot.fov, uv_offset=(float(du), float(dv)))

            # Compute bodies backplanes
            bodies_result = create_body_backplanes(snapshot, config, logger=logger)
            # Compute rings backplanes (if enabled/configured)
            rings_result = create_ring_backplanes(snapshot, config, logger=logger)

            # Merge all sources (distance-aware)
            master_by_type, body_id_map = merge_sources_into_master(
                snapshot,
                bodies_result=bodies_result,
                rings_result=rings_result,
            )

            if write_output_files:
                write_fits(
                    fits_file_path=fits_file_path,
                    snapshot=snapshot,
                    master_by_type=master_by_type,
                    body_id_map=body_id_map,
                    config=config,
                    bodies_result=bodies_result,
                    rings_result=rings_result,
                    logger=logger,
                )
    finally:
        for handler in local_handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
        if image_log_path is not None:
            MAIN_LOGGER.info('Wrote log to %s', image_log_path)

    return {'status': 'success'}
