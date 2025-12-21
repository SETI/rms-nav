from pathlib import Path
from typing import Any, Optional, cast

from filecache import FCPath
from PIL import Image

from nav.config import DEFAULT_LOGGER
from nav.obs import ObsSnapshotInst
from nav.dataset.dataset import ImageFiles
from nav.nav_master import NavMaster
from nav.support.file import json_as_string


def navigate_image_files(obs_class: type[ObsSnapshotInst],
                         image_files: ImageFiles,
                         nav_results_root: FCPath,
                         *,
                         nav_models: Optional[list[str]] = None,
                         nav_techniques: Optional[list[str]] = None,
                         write_output_files: bool = True) -> tuple[bool, dict[str, Any]]:
    """Navigate a set of image files.

    Parameters:
        obs_class: The observation snapshot class.
        image_files: The image files to navigate.
        nav_results_root: The directory to write the navigation results to; may be a FileCache URL.
        nav_models: The models to use for navigation; or None if all models are to be used.
        nav_techniques: The techniques to use for navigation; or None if all techniques are to be
            used.
        write_output_files: Whether to write output files. False performs the navigation as
            a dry run but doesn't write any results.

    Returns:
        A tuple containing a boolean indicating success or failure and a dictionary containing the
        public metadata for the navigation.
    """

    logger = DEFAULT_LOGGER

    if len(image_files.image_files) != 1:
        logger.error("Expected exactly one image per batch; got %d", len(image_files.image_files))
        return False, {
            'status': 'error',
            'status_error': 'expected_one_image_per_batch',
            'status_exception':
                f'Expected exactly one image per batch; got {len(image_files.image_files)}',
        }

    image_file = image_files.image_files[0]
    image_url = image_file.image_file_url
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    extra_params = image_file.extra_params
    public_metadata_file = nav_results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = nav_results_root / (image_file.results_path_stub + '_summary.png')

    with logger.open(str(image_url)):
        try:
            snapshot = obs_class.from_file(image_url, **extra_params)
        except (OSError, RuntimeError) as e:
            if ('SPICE(CKINSUFFDATA)' in str(e) or
                'SPICE(SPKINSUFFDATA)' in str(e) or
                'SPICE(NOFRAMECONNECT)' in str(e)):
                logger.exception('No SPICE kernel available for "%s": %s', image_path, str(e))
                metadata = {
                    'status': 'error',
                    'status_error': 'missing_spice_data',
                    'status_exception': str(e),
                    'observation': {
                        'image_path': str(image_path),
                        'image_name': image_name,
                    }
                }
            else:
                logger.exception('Error reading image "%s": %s', image_path, str(e))
                metadata = {
                    'status': 'error',
                    'status_error': 'image_read_error',
                    'status_exception': str(e),
                    'observation': {
                        'image_path': str(image_path),
                        'image_name': image_name,
                    }
                }
            public_metadata_file.write_text(json_as_string(metadata))
            return False, metadata

        nm = NavMaster(snapshot, nav_models=nav_models, nav_techniques=nav_techniques)
        nm.compute_all_models()

        nm.navigate()

        metadata = nm.metadata_serializable()
        metadata['status'] = 'success'

        if write_output_files:
            logger.info(f'Writing metadata to {public_metadata_file}')
            public_metadata_file.write_text(json_as_string(metadata))
            logger.info(f'Writing summary PNG to {summary_png_file}')
            overlay = nm.create_overlay()
            png_local = cast(Path, summary_png_file.get_local_path())
            im = Image.fromarray(overlay)
            im.save(png_local)
            summary_png_file.upload()

        return True, metadata
