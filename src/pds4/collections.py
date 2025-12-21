import csv
import json
from pathlib import Path
from typing import Any, cast

from filecache import FCPath
from pdslogger import PdsLogger
import pdstemplate

from nav.dataset.dataset import DataSet


def generate_collection_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> tuple[bool, dict[str, Any]]:
    """Generate collection CSV and label files for the bundle.

    Parameters:
        bundle_results_root: Root directory of the bundle.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        Tuple of (success boolean, metadata dictionary).
    """

    try:
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()
        bundle_root = bundle_results_root / bundle_name

        # Scan for all label files in data directory
        data_dir = bundle_root / 'data'
        label_files: list[FCPath] = []
        lidvids: list[str] = []

        if not data_dir.exists():
            logger.warning('Data directory does not exist: %s', data_dir)
            return False, {
                'status': 'error',
                'status_error': 'data_directory_missing',
                'status_exception': f'Data directory does not exist: {data_dir}',
            }

        # Recursively scan for .lblx files
        for label_file in data_dir.rglob('*.lblx'):
            label_files.append(label_file)

        # Sort by image name (extracted from filename)
        def get_image_name_from_label(path: FCPath) -> str:
            # Extract image name from filename
            # (e.g., "1234567890w_backplanes.lblx" -> "1234567890w")
            name = path.stem
            if '_backplanes' in name:
                return name.split('_backplanes')[0]
            return name

        label_files.sort(key=get_image_name_from_label)
        logger.info('Found %d label files in bundle', len(label_files))

        # Extract LIDVIDs from label files (this would need to parse XML or
        # read from metadata - for now, generate placeholder)
        # TODO: Parse actual LIDVIDs from label files
        for label_file in label_files:
            # Placeholder: would need to parse XML to get actual LIDVID
            lidvids.append(f'urn:nasa:pds:{bundle_name}:data:placeholder::1.0')

        # Generate collection_data.csv
        collection_data_csv = bundle_root / 'data' / 'collection_data.csv'
        try:
            collection_data_local = cast(Path, collection_data_csv.get_local_path())
            collection_data_local.parent.mkdir(parents=True, exist_ok=True)
            with collection_data_local.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Member Status', 'LIDVID_LID'])
                for lidvid in lidvids:
                    writer.writerow(['P', lidvid])
            collection_data_csv.upload()
            logger.info('Generated collection_data.csv: %s', collection_data_csv)
        except Exception as e:
            logger.exception('Error writing collection_data.csv')
            return False, {
                'status': 'error',
                'status_error': 'collection_data_csv_error',
                'status_exception': str(e),
            }

        # Generate collection_browse.csv (similar structure)
        collection_browse_csv = bundle_root / 'browse' / 'collection_browse.csv'
        try:
            collection_browse_local = cast(Path, collection_browse_csv.get_local_path())
            collection_browse_local.parent.mkdir(parents=True, exist_ok=True)
            with collection_browse_local.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Member Status', 'LIDVID_LID'])
                # Browse products would be extracted similarly
                # For now, placeholder
                for lidvid in lidvids:
                    browse_lidvid = lidvid.replace(':data:', ':browse:')
                    writer.writerow(['P', browse_lidvid])
            collection_browse_csv.upload()
            logger.info('Generated collection_browse.csv: %s', collection_browse_csv)
        except Exception as e:
            logger.exception('Error writing collection_browse.csv')
            return False, {
                'status': 'error',
                'status_error': 'collection_browse_csv_error',
                'status_exception': str(e),
            }

        # Generate collection label files using templates
        template_base = Path(template_dir)

        # Collection data label
        collection_data_template = template_base / 'collection_data.lblx'
        if collection_data_template.exists():
            try:
                template = pdstemplate.PdsTemplate(str(collection_data_template))
                collection_data_label = bundle_root / 'data' / 'collection_data.lblx'
                collection_data_label_local = cast(
                    Path, collection_data_label.get_local_path())
                template_vars = {
                    'EARLIEST_START_DATE_TIME': '',  # TODO: Calculate from all images
                    'LATEST_STOP_DATE_TIME': '',  # TODO: Calculate from all images
                }
                template.write(template_vars, str(collection_data_label_local))
                collection_data_label.upload()
                logger.info('Generated collection_data.lblx')
            except Exception as e:
                logger.warning('Error generating collection_data.lblx: %s', e)

        # Collection browse label
        collection_browse_template = template_base / 'collection_browse.lblx'
        if collection_browse_template.exists():
            try:
                template = pdstemplate.PdsTemplate(str(collection_browse_template))
                collection_browse_label = bundle_root / 'browse' / 'collection_browse.lblx'
                collection_browse_label_local = cast(Path,
                                                     collection_browse_label.get_local_path())
                template_vars = {}
                template.write(template_vars, str(collection_browse_label_local))
                collection_browse_label.upload()
                logger.info('Generated collection_browse.lblx')
            except Exception as e:
                logger.warning('Error generating collection_browse.lblx: %s', e)

        return True, {
            'status': 'success',
            'num_products': len(lidvids),
        }

    except Exception as e:
        logger.exception('Error generating collection files')
        return False, {
            'status': 'error',
            'status_error': 'collection_generation_error',
            'status_exception': str(e),
        }


def generate_global_index_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> tuple[bool, dict[str, Any]]:
    """Generate global index files for bodies and rings.

    Parameters:
        bundle_results_root: Root directory of the bundle.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        Tuple of (success boolean, metadata dictionary).
    """

    try:
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()
        bundle_root = bundle_results_root / bundle_name

        # Scan for all bundle metadata files
        metadata_files: list[FCPath] = []
        for metadata_file in bundle_results_root.rglob('*_bundle_metadata.json'):
            metadata_files.append(metadata_file)

        # Sort by image name (extracted from filename)
        def get_image_name_from_metadata(path: FCPath) -> str:
            # Extract image name from filename
            # (e.g., "1234567890w_bundle_metadata.json" -> "1234567890w")
            return path.name.replace('_bundle_metadata.json', '')

        metadata_files.sort(key=get_image_name_from_metadata)
        logger.info('Found %d bundle metadata files', len(metadata_files))

        # Collect body and ring statistics
        body_index_rows: list[dict[str, Any]] = []
        ring_index_rows: list[dict[str, Any]] = []
        ring_backplane_types: set[str] = set()

        for metadata_file in metadata_files:
            try:
                metadata_text = metadata_file.read_text()
                metadata = json.loads(metadata_text)
                backplanes = metadata.get('backplanes', {})
                bodies = backplanes.get('bodies', {})
                rings = backplanes.get('rings', {})

                # Extract LID and path from metadata or filename
                # TODO: Parse actual LID from label files
                results_stub = metadata_file.name.replace('_bundle_metadata.json', '')
                lid = f'urn:nasa:pds:{bundle_name}:data:placeholder::1.0'
                # Path relative to data directory
                pds4_path_stub = metadata.get('pds4_path_stub', results_stub)
                path_to_image = f'{pds4_path_stub}/{results_stub}_backplanes.fits'

                # Body index: one line per image per body backplane
                for body_name, body_stats in bodies.items():
                    for bp_type, bp_values in body_stats.items():
                        body_index_rows.append({
                            'LID': lid,
                            'body_name': body_name,
                            'backplane_type': bp_type,
                            'path_to_image_file': path_to_image,
                            'min_value': bp_values.get('min'),
                            'max_value': bp_values.get('max'),
                        })

                # Ring index: one line per image
                ring_row: dict[str, Any] = {
                    'LID': lid,
                    'path_to_image_file': path_to_image,
                }
                for ring_type, ring_values in rings.items():
                    ring_backplane_types.add(ring_type)
                    ring_row[f'{ring_type}_min'] = ring_values.get('min')
                    ring_row[f'{ring_type}_max'] = ring_values.get('max')
                if rings:
                    ring_index_rows.append(ring_row)

            except Exception as e:
                logger.warning('Error reading metadata file %s: %s', metadata_file, e)
                continue

        # Generate global_index_bodies.tab
        supplemental_dir = bundle_root / 'document' / 'supplemental'
        bodies_tab = supplemental_dir / 'global_index_bodies.tab'
        try:
            bodies_tab_local = cast(Path, bodies_tab.get_local_path())
            bodies_tab_local.parent.mkdir(parents=True, exist_ok=True)
            with bodies_tab_local.open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['LID', 'body_name', 'backplane_type',
                                 'path_to_image_file', 'min_value', 'max_value'])
                for row in body_index_rows:
                    writer.writerow([
                        row['LID'],
                        row['body_name'],
                        row['backplane_type'],
                        row['path_to_image_file'],
                        row['min_value'],
                        row['max_value'],
                    ])
            bodies_tab.upload()
            logger.info('Generated global_index_bodies.tab with %d rows', len(body_index_rows))
        except Exception as e:
            logger.exception('Error writing global_index_bodies.tab')
            return False, {
                'status': 'error',
                'status_error': 'bodies_index_error',
                'status_exception': str(e),
            }

        # Generate global_index_rings.tab
        rings_tab = supplemental_dir / 'global_index_rings.tab'
        try:
            rings_tab_local = cast(Path, rings_tab.get_local_path())
            rings_tab_local.parent.mkdir(parents=True, exist_ok=True)
            with rings_tab_local.open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                # Build header: LID, path_to_image_file, then min/max for each ring type
                header = ['LID', 'path_to_image_file']
                for ring_type in sorted(ring_backplane_types):
                    header.append(f'{ring_type}_min')
                    header.append(f'{ring_type}_max')
                writer.writerow(header)

                for row in ring_index_rows:
                    row_data = [row['LID'], row['path_to_image_file']]
                    for ring_type in sorted(ring_backplane_types):
                        row_data.append(row.get(f'{ring_type}_min', ''))
                        row_data.append(row.get(f'{ring_type}_max', ''))
                    writer.writerow(row_data)
            rings_tab.upload()
            logger.info('Generated global_index_rings.tab with %d rows', len(ring_index_rows))
        except Exception as e:
            logger.exception('Error writing global_index_rings.tab')
            return False, {
                'status': 'error',
                'status_error': 'rings_index_error',
                'status_exception': str(e),
            }

        # Generate label files using templates
        template_base = Path(template_dir)

        # Global index bodies label
        bodies_template = template_base / 'global_index_bodies.lblx'
        if bodies_template.exists():
            try:
                template = pdstemplate.PdsTemplate(str(bodies_template))
                bodies_label = supplemental_dir / 'global_index_bodies.lblx'
                bodies_label_local = cast(Path, bodies_label.get_local_path())
                template_vars = {
                    'FILE_RECORDS': len(body_index_rows),
                }
                template.write(template_vars, str(bodies_label_local))
                bodies_label.upload()
                logger.info('Generated global_index_bodies.lblx')
            except Exception as e:
                logger.warning('Error generating global_index_bodies.lblx: %s', e)

        # Global index rings label
        rings_template = template_base / 'global_index_rings.lblx'
        if rings_template.exists():
            try:
                template = pdstemplate.PdsTemplate(str(rings_template))
                rings_label = supplemental_dir / 'global_index_rings.lblx'
                rings_label_local = cast(Path, rings_label.get_local_path())
                template_vars = {
                    'FILE_RECORDS': len(ring_index_rows),
                }
                template.write(template_vars, str(rings_label_local))
                rings_label.upload()
                logger.info('Generated global_index_rings.lblx')
            except Exception as e:
                logger.warning('Error generating global_index_rings.lblx: %s', e)

        return True, {
            'status': 'success',
            'num_body_rows': len(body_index_rows),
            'num_ring_rows': len(ring_index_rows),
        }

    except Exception as e:
        logger.exception('Error generating global index files')
        return False, {
            'status': 'error',
            'status_error': 'global_index_generation_error',
            'status_exception': str(e),
        }
