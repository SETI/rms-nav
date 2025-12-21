import json
from pathlib import Path
from typing import Any, cast

from astropy.io import fits
from filecache import FCPath
import numpy as np
from pdslogger import PdsLogger
import pdstemplate

from nav.dataset.dataset import DataSet, ImageFiles
from nav.support.file import json_as_string
from nav.support.types import NDArrayFloatType, NDArrayIntType

from .backplane_summary import calculate_backplane_statistics


def generate_bundle_data_files(
    dataset: DataSet,
    image_files: ImageFiles,
    *,
    nav_results_root: FCPath,
    backplane_results_root: FCPath,
    bundle_results_root: FCPath,
    logger: PdsLogger,
    write_output_files: bool = True,
) -> tuple[bool, dict[str, Any]]:
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
        Tuple of (success boolean, metadata dictionary).
    """

    if len(image_files.image_files) != 1:
        logger.error("Expected exactly one image per batch; got %d",
                     len(image_files.image_files))
        return False, {
            'status': 'error',
            'status_error': 'expected_one_image_per_batch',
            'status_exception':
                f'Expected exactly one image per batch; got {len(image_files.image_files)}',
        }

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    image_name = image_path.name
    results_path_stub = image_file.results_path_stub

    # Get supplemental filename from end of results_path_stub
    results_path_stub_end = Path(results_path_stub).name

    metadata_file = nav_results_root / (results_path_stub + '_metadata.json')
    fits_file_path = backplane_results_root / (results_path_stub + '_backplanes.fits')

    with logger.open(str(image_path)):
        # Read navigation metadata
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
        except Exception as e:
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
            logger.warning('Skipping bundle generation for "%s": status=%s error=%s',
                           image_path, status,
                           nav_metadata.get('status_error', 'unknown'))
            return False, {
                'status': 'warning',
                'status_error': 'prior_status_not_success',
                'status_exception': nav_metadata.get('status_exception', ''),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Read backplane FITS file
        try:
            local_fits_path = cast(str, fits_file_path.retrieve())
            with fits.open(local_fits_path) as hdul:
                body_id_map: NDArrayIntType | None = None
                backplane_arrays: dict[str, NDArrayFloatType] = {}

                # Parse HDUs: locate BODY_ID_MAP and backplanes
                for hdu in hdul[1:]:  # Skip primary HDU
                    name = (hdu.name or '').upper()
                    if name == 'BODY_ID_MAP':
                        body_id_map = np.asarray(hdu.data, dtype=np.int32)
                        logger.debug('Found BODY_ID_MAP in FITS file')
                    else:
                        arr = np.asarray(hdu.data, dtype=np.float64)
                        backplane_arrays[name] = arr
                        logger.debug('Found backplane %s in FITS file', name)

                if body_id_map is None:
                    logger.warning('No BODY_ID_MAP found in FITS file %s', fits_file_path)
                    body_id_map = np.zeros((1, 1), dtype=np.int32)

        except FileNotFoundError:
            logger.warning('Backplane FITS file not found: %s', fits_file_path)
            return False, {
                'status': 'warning',
                'status_error': 'fits_file_missing',
                'status_exception': f'Backplane FITS file not found: {fits_file_path}',
                'image_path': str(image_path),
                'image_name': image_name,
            }
        except Exception as e:
            logger.exception('Error reading FITS file "%s"', fits_file_path)
            return False, {
                'status': 'error',
                'status_error': 'fits_read_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Calculate backplane statistics
        try:
            bp_stats = calculate_backplane_statistics(
                body_id_map=body_id_map,
                backplane_arrays=backplane_arrays,
                logger=logger,
            )
        except Exception as e:
            logger.exception('Error calculating backplane statistics')
            return False, {
                'status': 'error',
                'status_error': 'statistics_calculation_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Get PDS4 path stub
        pds4_path_stub = dataset.pds4_path_stub(image_file)
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()

        # Combine metadata
        combined_metadata: dict[str, Any] = {
            'navigation': nav_metadata,
            'backplanes': bp_stats,
            'pds4_path_stub': pds4_path_stub,
        }

        # Get template variables from dataset
        try:
            template_vars = dataset.pds4_template_variables(
                image_file=image_file,
                nav_metadata=nav_metadata,
                backplane_metadata=bp_stats,
            )
        except Exception as e:
            logger.exception('Error getting template variables from dataset')
            return False, {
                'status': 'error',
                'status_error': 'template_variables_error',
                'status_exception': str(e),
                'image_path': str(image_path),
                'image_name': image_name,
            }

        # Determine output paths
        bundle_root = bundle_results_root / bundle_name
        data_dir = bundle_root / 'data' / pds4_path_stub
        label_file_path = data_dir / f'{image_name.rsplit(".", 1)[0]}_backplanes.lblx'
        suppl_file_path = data_dir / f'{results_path_stub_end}_supplemental.txt'
        metadata_output_path = bundle_results_root / (results_path_stub + '_bundle_metadata.json')

        # Add file path variables to template_vars
        template_vars['BACKPLANE_FILENAME'] = label_file_path.name.replace('.lblx', '.fits')
        template_vars['BACKPLANE_PATH'] = str(fits_file_path)
        template_vars['BACKPLANE_SUPPL_FILENAME'] = suppl_file_path.name
        template_vars['BACKPLANE_SUPPL_PATH'] = str(suppl_file_path)

        if write_output_files:
            try:
                # Generate supplemental file (JSON format) - must be written before template
                suppl_file_path.write_text(json_as_string(combined_metadata))
                logger.info('Generated supplemental file: %s', suppl_file_path)

                # Generate PDS4 label file
                template_path = Path(template_dir) / 'data.lblx'
                if not template_path.exists():
                    logger.error('Template file not found: %s', template_path)
                    return False, {
                        'status': 'error',
                        'status_error': 'template_not_found',
                        'status_exception': f'Template file not found: {template_path}',
                        'image_path': str(image_path),
                        'image_name': image_name,
                    }

                template = pdstemplate.PdsTemplate(str(template_path))
                template.write(template_vars, label_file_path)
                logger.info('Generated PDS4 label: %s', label_file_path)

                # Write combined metadata file
                metadata_output_path.write_text(json_as_string(combined_metadata))
                logger.debug('Wrote combined metadata: %s', metadata_output_path)

            except Exception as e:
                logger.exception('Error writing output files')
                return False, {
                    'status': 'error',
                    'status_error': 'write_error',
                    'status_exception': str(e),
                    'image_path': str(image_path),
                    'image_name': image_name,
                }

        out_metadata: dict[str, Any] = {
            'status': 'success',
            'image_path': str(image_path),
            'image_name': image_name,
            'pds4_path_stub': pds4_path_stub,
            'label_file': str(label_file_path),
            'supplemental_file': str(suppl_file_path),
        }

        return True, out_metadata
