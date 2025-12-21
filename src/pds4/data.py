import json
import shutil
from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from pdslogger import PdsLogger
import pdstemplate

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
    write_output_files: bool = True,
) -> tuple[bool, dict[str, Any] | None]:
    """Generate PDS4 bundle data files for a single image batch.

    Parameters:
        dataset: The dataset instance for bundle-specific methods.
        image_files: List of images; must have exactly one image in the batch.
        nav_results_root: Root containing navigation metadata JSONs.
        backplane_results_root: Root containing backplane FITS files.
        bundle_results_root: Destination root for bundle files.
        logger: Logger for diagnostic messages.
        write_output_files: Whether to write output files.

    Returns:
        Tuple of (success boolean, metadata dictionary or None).
        Returns (False, None) if image should be skipped (missing files, etc.).
        Returns (True, metadata) on success.
        Raises exceptions for unexpected errors.
    """

    # TODO Move this to the top-level program
    pdstemplate.PdsTemplate.set_logger(logger)

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}')

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    results_path_stub = image_file.results_path_stub

    metadata_file = nav_results_root / (results_path_stub + '_metadata.json')
    backplane_metadata_file = backplane_results_root / (
        results_path_stub + '_backplane_metadata.json')

    with logger.open(str(image_path)):
        # Read navigation metadata
        try:
            metadata_text = metadata_file.read_text()
            nav_metadata = cast(dict[str, Any], json.loads(metadata_text))
        except FileNotFoundError:
            logger.warning('Offset metadata not found: %s', metadata_file)
            return False, None

        status = nav_metadata.get('status', None)
        if status != 'success':
            logger.warning('Skipping bundle generation for "%s": status=%s error=%s',
                           image_path, status,
                           nav_metadata.get('status_error', 'unknown'))
            return False, None

        # Read backplane metadata file
        try:
            backplane_metadata_text = backplane_metadata_file.read_text()
            bp_stats = cast(dict[str, Any], json.loads(backplane_metadata_text))
        except FileNotFoundError:
            logger.warning('Backplane metadata file not found: %s', backplane_metadata_file)
            return False, None

        # Get PDS4 path stub (includes path and filename prefix)
        pds4_path_stub = dataset.pds4_path_stub(image_file)
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()

        # Extract directory and filename prefix from pds4_path_stub
        # pds4_path_stub is like "1234xxxxxx/123456xxxx/1234567890w"
        if '/' in pds4_path_stub:
            pds4_dir, pds4_filename_prefix = pds4_path_stub.rsplit('/', 1)
        else:
            pds4_dir = ''
            pds4_filename_prefix = pds4_path_stub

        # Combine metadata
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
        if pds4_dir:
            data_dir = bundle_root / 'data' / pds4_dir
            browse_dir = bundle_root / 'browse' / pds4_dir
        else:
            data_dir = bundle_root / 'data'
            browse_dir = bundle_root / 'browse'
        label_file_path = data_dir / f'{pds4_filename_prefix}_backplanes.lblx'
        suppl_file_path = data_dir / f'{pds4_filename_prefix}_supplemental.txt'
        browse_label_path = browse_dir / f'{pds4_filename_prefix}_summary.lblx'
        browse_image_path = browse_dir / f'{pds4_filename_prefix}_summary.png'

        # Add file path variables to template_vars
        fits_file_path = backplane_results_root / (results_path_stub + '_backplanes.fits')
        summary_png_source = nav_results_root / (results_path_stub + '_summary.png')
        template_vars['BACKPLANE_FILENAME'] = label_file_path.name.replace('.lblx', '.fits')
        template_vars['BACKPLANE_PATH'] = str(fits_file_path)
        template_vars['BACKPLANE_SUPPL_FILENAME'] = suppl_file_path.name
        template_vars['BACKPLANE_SUPPL_PATH'] = str(suppl_file_path)
        template_vars['BROWSE_FULL_FILENAME'] = browse_image_path.name
        template_vars['BROWSE_FULL_PATH'] = str(browse_image_path)

        if write_output_files:
            # Generate supplemental file (JSON format) - must be written before template
            suppl_file_path.write_text(json_as_string(combined_metadata))
            logger.info('Generated supplemental file: %s', suppl_file_path)

            # Generate PDS4 label file
            template_path = Path(template_dir) / 'data.lblx'
            if not template_path.exists():
                raise FileNotFoundError(f'Template file not found: {template_path}')

            template = pdstemplate.PdsTemplate(str(template_path))
            template.write(template_vars, label_file_path)
            logger.info('Generated PDS4 label: %s', label_file_path)

            # Copy summary PNG to browse directory and generate browse label
            if summary_png_source.exists():
                # Copy the summary PNG file
                summary_png_local = summary_png_source.get_local_path()
                browse_image_local = browse_image_path.get_local_path()
                # TODO This needs to be updated for cloud storage
                browse_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_png_local, browse_image_local)
                browse_image_path.upload()
                logger.info('Copied summary image: %s', browse_image_path)

                # Generate browse label
                browse_template_path = Path(template_dir) / 'browse.lblx'
                if browse_template_path.exists():
                    browse_template = pdstemplate.PdsTemplate(str(browse_template_path))
                    browse_template.write(template_vars, browse_label_path)
                    browse_label_path.upload()
                    logger.info('Generated browse label: %s', browse_label_path)
                else:
                    logger.warning('Browse template not found: %s', browse_template_path)
            else:
                logger.warning('Summary PNG not found: %s', summary_png_source)

        out_metadata: dict[str, Any] = {
            'image_path': str(image_path),
            'image_name': image_name,
            'label_file': str(label_file_path),
            'supplemental_file': str(suppl_file_path),
        }

        return True, out_metadata
