import json
from typing import Any, cast

from filecache import FCPath
import oops

from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER
from nav.dataset.dataset import ImageFiles
from nav.obs import ObsSnapshot, ObsSnapshotInst

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
) -> None:
    """Generate backplanes for a single image batch using prior offset metadata.

    Parameters:
        obs_class: Observation snapshot class for the instrument.
        image_files: List of images; must have exactly one image in the batch.
        nav_results_root: Root containing previously written navigation metadata JSONs.
        backplane_results_root: Destination root for FITS and label files.
        write_output_files: Whether to write outputs to storage.
    """

    logger = DEFAULT_LOGGER
    config = DEFAULT_CONFIG

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}'
        )

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    fits_file_path = backplane_results_root / (
        image_file.results_path_stub + '_backplanes.fits'
    )

    with logger.open(f'Processing image: {str(image_path)}'):
        # This will raise an exception if the metadata file is not found or not valid JSON
        metadata_text = metadata_file.read_text()
        nav_metadata = cast(dict[str, Any], json.loads(metadata_text))

        status = nav_metadata.get('status', None)
        if status != 'success':
            logger.warning(
                'Skipping backplanes for "%s": status=%s error=%s',
                image_path,
                status,
                nav_metadata.get('status_error', 'unknown'),
            )
            return

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
            logger.warning(f'{image_path}: "offset" field is None, using (0, 0)')
            dv, du = 0, 0
        else:
            dv, du = nav_metadata['offset']
        snapshot.fov = oops.fov.OffsetFOV(
            snapshot.fov, uv_offset=(float(du), float(dv))
        )

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
