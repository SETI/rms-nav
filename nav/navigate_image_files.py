from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from PIL import Image

from nav.config import DEFAULT_LOGGER
from nav.obs import ObsSnapshotInst
from nav.dataset.dataset import ImageFiles
from nav.nav_master import NavMaster
from nav.support.file import json_as_string


def navigate_image_files(obs_class: type[ObsSnapshotInst],
                         image_files: ImageFiles,
                         results_root: FCPath,
                         nav_models: list[str],
                         nav_techniques: list[str],
                         write_output_files: bool = True) -> tuple[bool, dict[str, Any]]:

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
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    public_metadata_file = results_root / (image_file.results_path_stub + '_metadata.json')
    summary_png_file = results_root / (image_file.results_path_stub + '_summary.png')

    with logger.open(str(image_path)):
        try:
            snapshot = obs_class.from_file(image_path)
        except (OSError, RuntimeError) as e:
            if ('SPICE(CKINSUFFDATA)' in str(e) or
                'SPICE(SPKINSUFFDATA)' in str(e) or
                'SPICE(NOFRAMECONNECT)' in str(e)):
                logger.exception('No SPICE kernel available for "%s"', image_path)
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
                logger.exception('Error reading image "%s"', image_path)
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

        if write_output_files:
            overlay = nm.create_overlay()

            metadata = nm.metadata_serializable()
            metadata['status'] = 'success'
            public_metadata_file.write_text(json_as_string(metadata))

            png_local = cast(Path, summary_png_file.get_local_path())
            im = Image.fromarray(overlay)
            im.save(png_local)
            summary_png_file.upload()

        return True, metadata
