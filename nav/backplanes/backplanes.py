import json
from typing import Any, cast

from filecache import FCPath
import numpy as np
import oops

from nav.config import DEFAULT_CONFIG
from nav.config.logger import DEFAULT_LOGGER
from nav.dataset.dataset import ImageFiles
from nav.obs import ObsSnapshotInst

from .backplanes_bodies import create_body_backplanes
from .backplanes_rings import create_ring_backplanes
from .merge import merge_sources_into_master
from .writer import write_fits_and_label


def generate_backplanes_image_files(
    obs_class: type[ObsSnapshotInst],
    image_files: ImageFiles,
    *,
    nav_results_root: FCPath,
    backplane_results_root: FCPath,
    write_output_files: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Generate backplanes for a single image batch using prior offset metadata.

    Parameters:
        obs_class: Observation snapshot class for the instrument.
        image_files: List of images; must have exactly one image in the batch.
        metadata_root: Root containing previously written navigation metadata JSONs.
        results_root: Destination root for FITS and label files.
        write_output_files: Whether to write outputs to storage.
    """

    logger = DEFAULT_LOGGER
    config = DEFAULT_CONFIG

    if len(image_files.image_files) != 1:
        logger.error("Expected exactly one image per batch; got %d", len(image_files.image_files))
        return False, {
            'status': 'error',
            'status_error': 'expected_one_image_per_batch',
            'status_exception':
                f'Expected exactly one image per batch; got {len(image_files.image_files)}',
        }

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    fits_file_path = backplane_results_root / (image_file.results_path_stub + '_backplanes.fits')
    label_file_path = backplane_results_root / (image_file.results_path_stub + '_backplanes.xml')

    with logger.open(str(image_path)):
        # Gate on metadata existence
        try:
            metadata_text = metadata_file.read_text()
            nav_metadata = cast(dict[str, Any], json.loads(metadata_text))
        except FileNotFoundError:
            logger.warning('Offset metadata not found: %s', metadata_file)
            return False, {
                'status': 'warning',
                'status_error': 'metadata_missing',
                'status_exception': f'Offset metadata not found: {metadata_file}',
                'image_path': str(image_path),
                'image_name': image_name,
            }
        except Exception as e:  # JSON parse etc.
            logger.warning('Error reading metadata "%s": %s', metadata_file, e)
            return False, {
                'status': 'warning',
                'status_error': 'metadata_read_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        status = nav_metadata.get('status', None)
        if status != 'success':
            logger.warning('Skipping backplanes for "%s": status=%s error=%s',
                           image_path, status,
                           nav_metadata.get('status_error', 'unknown'))
            return False, {
                'status': 'warning',
                'status_error': 'prior_status_not_success',
                'status_exception': nav_metadata.get('status_exception', ''),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Build observation in original FOV
        try:
            snapshot = obs_class.from_file(image_path, extfov_margin_vu=(0, 0))
        except Exception as e:
            logger.exception('Error reading image "%s"', image_path)
            return False, {
                'status': 'error',
                'status_error': 'image_read_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Apply offset via OffsetFOV; metadata uses (dv, du)
        try:
            dv, du = nav_metadata.get('offset', (0.0, 0.0))
            snapshot.fov = oops.fov.OffsetFOV(snapshot.fov, uv_offset=(float(du), float(dv)))
        except Exception as e:
            logger.error('Unable to apply OffsetFOV; continuing with unshifted FOV: %s', e)
            return False, {
                'status': 'error',
                'status_error': 'offset_apply_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Compute bodies backplanes
        bodies_result = create_body_backplanes(snapshot, config)

        # Compute rings backplanes (if enabled/configured)
        rings_result = create_ring_backplanes(snapshot, config)

        # Merge all sources (distance-aware)
        master_by_type, body_id_map, merge_info = merge_sources_into_master(
            snapshot,
            bodies_result=bodies_result,
            rings_result=rings_result,
        )

        # Fallback for environments lacking geometry: create zero arrays for configured types
        if not master_by_type:
            try:
                zero = snapshot.make_fov_zeros(dtype=float).astype('float32')
            except Exception:
                zero = np.zeros(snapshot.data.shape, dtype='float32')
            expected_types: set[str] = set()
            expected_types.update(
                [bp['name'] for bp in getattr(config.backplanes, 'bodies', [])]
            )
            expected_types.update(
                [
                    bp['name']
                    for bp in getattr(config.backplanes, 'rings', [])
                    if bp.get('name') != 'distance'
                ]
            )
            for t in sorted(expected_types):
                master_by_type[t] = zero.copy()

        out_metadata: dict[str, Any] = {
            'status': 'success',
            'image_path': str(image_path),
            'image_name': image_name,
            'backplane_types': sorted(list(master_by_type.keys())),
            'merge': merge_info,
        }

        if write_output_files:
            write_fits_and_label(
                fits_file_path=fits_file_path,
                label_file_path=label_file_path,
                snapshot=snapshot,
                master_by_type=master_by_type,
                body_id_map=body_id_map,
                config=config,
                logger=logger,
            )

        return True, out_metadata
