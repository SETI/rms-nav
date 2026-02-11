import json
import shutil
from pathlib import Path
from typing import Any, cast

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from nav.dataset.dataset import DataSet, ImageFiles
from nav.support.file import json_as_string


def generate_bundle_data_files(
    dataset: DataSet,
    image_files: ImageFiles,
    *,
    nav_results_root: FCPath,
    backplane_results_root: FCPath,
    bundle_results_root: FCPath,
    logger: PdsLogger,
) -> None:
    """Generate PDS4 bundle data files for a single image batch.

    Parameters:
        dataset: The dataset instance to get bundle-specific methods from.
        image_files: List of images; must have exactly one image in the batch.
        nav_results_root: Root containing navigation metadata JSONs and summary PNGs.
        backplane_results_root: Root containing backplane FITS files and metadata JSONs.
        bundle_results_root: Destination root for bundle files.
        logger: Logger for diagnostic messages.
    """

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}'
        )

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    results_path_stub = image_file.results_path_stub

    metadata_file = nav_results_root / (results_path_stub + '_metadata.json')
    backplane_metadata_file = backplane_results_root / (
        results_path_stub + '_backplane_metadata.json'
    )

    with logger.open(f'Generating PDS4 bundle data files for {image_path!s}'):
        # Read navigation metadata
        metadata_text = metadata_file.read_text()
        nav_metadata = cast(dict[str, Any], json.loads(metadata_text))

        status = nav_metadata.get('status', None)
        if status != 'success':
            # TODO Figure out what to do with non-navigated images
            logger.warning(
                'Skipping bundle generation for "%s": status=%s error=%s',
                image_path,
                status,
                nav_metadata.get('status_error', 'unknown'),
            )
            return

        # Read backplane metadata
        backplane_metadata_text = backplane_metadata_file.read_text()
        bp_stats = cast(dict[str, Any], json.loads(backplane_metadata_text))

        pds4_path_stub = dataset.pds4_path_stub(image_file)
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()

        # TODO Clean up the nav metadata to only include the necessary fields

        # Combine metadata for supplemental file
        combined_metadata: dict[str, Any] = {
            'navigation': nav_metadata,
            'backplanes': bp_stats,
        }

        # Get template variables from dataset
        template_vars = dataset.pds4_template_variables(
            image_file=image_file,
            nav_metadata=nav_metadata,
            backplane_metadata=bp_stats,
        )

        # Determine output paths
        bundle_root = bundle_results_root / bundle_name
        data_dir = bundle_root / 'data'
        browse_dir = bundle_root / 'browse'
        label_file_path = data_dir / (pds4_path_stub + '_backplanes.lblx')
        suppl_file_path = data_dir / (pds4_path_stub + '_supplemental.txt')
        browse_label_path = browse_dir / (pds4_path_stub + '_summary.lblx')
        browse_image_path = browse_dir / (pds4_path_stub + '_summary.png')

        # Add file path variables to template_vars
        fits_file_path = backplane_results_root / (results_path_stub + '_backplanes.fits')
        summary_png_source = nav_results_root / (results_path_stub + '_summary.png')
        template_vars['BACKPLANE_FILENAME'] = label_file_path.name.replace('.lblx', '.fits')
        template_vars['BACKPLANE_PATH'] = str(fits_file_path)
        template_vars['BACKPLANE_SUPPL_FILENAME'] = suppl_file_path.name
        template_vars['BACKPLANE_SUPPL_PATH'] = str(suppl_file_path)
        template_vars['BROWSE_FULL_FILENAME'] = browse_image_path.name
        template_vars['BROWSE_FULL_PATH'] = str(browse_image_path)

        # Generate supplemental file (JSON format) - must be written before template
        suppl_file_path.write_text(json_as_string(combined_metadata))
        logger.info('Generated supplemental file: %s', suppl_file_path)

        # Generate PDS4 label file
        template_path = Path(template_dir) / 'data.lblx'
        template = pdstemplate.PdsTemplate(str(template_path))
        template.write(template_vars, label_file_path)
        logger.info('Generated PDS4 label: %s', label_file_path)

        # Copy summary PNG to browse directory and generate browse label
        if summary_png_source.exists():
            # Copy the summary PNG file
            summary_png_local = cast(Path, summary_png_source.get_local_path())
            browse_image_local = cast(Path, browse_image_path.get_local_path())
            # TODO This needs to be updated for cloud storage
            browse_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_png_local, browse_image_local)
            browse_image_path.upload()
            logger.info('Copied summary image: %s', browse_image_path)

            # Generate browse label
            browse_template_path = Path(template_dir) / 'browse.lblx'
            browse_template = pdstemplate.PdsTemplate(str(browse_template_path))
            browse_template.write(template_vars, browse_label_path)
            browse_label_path.upload()
            logger.info('Generated browse label: %s', browse_label_path)
        else:
            logger.warning('Summary PNG not found: %s', summary_png_source)
