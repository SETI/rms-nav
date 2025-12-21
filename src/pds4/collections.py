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
) -> dict[str, Any]:
    """Generate collection CSV and label files for the bundle.

    Parameters:
        bundle_results_root: Root directory of the bundle.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        Metadata dictionary with 'num_products' key.
    """

    pdstemplate.PdsTemplate.set_logger(logger)

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name

    # Scan for all label files in data directory
    data_dir = bundle_root / 'data'
    label_files: list[FCPath] = []
    lidvids: list[str] = []

    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')

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
    collection_data_local = cast(Path, collection_data_csv.get_local_path())
    collection_data_local.parent.mkdir(parents=True, exist_ok=True)
    with collection_data_local.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Member Status', 'LIDVID_LID'])
        for lidvid in lidvids:
            writer.writerow(['P', lidvid])
    collection_data_csv.upload()
    logger.info('Generated collection_data.csv: %s', collection_data_csv)

    # Generate collection label files using templates
    template_base = Path(template_dir)

    # Collection data label
    collection_data_template = template_base / 'collection_data.lblx'
    if collection_data_template.exists():
        template = pdstemplate.PdsTemplate(str(collection_data_template))
        collection_data_label = bundle_root / 'data' / 'collection_data.lblx'
        collection_data_label_local = cast(
            Path, collection_data_label.get_local_path())
        template_vars = {
            'COLLECTION_DATA_CSV_PATH': str(collection_data_csv),
            'EARLIEST_START_DATE_TIME': '',  # TODO: Calculate from all images
            'LATEST_STOP_DATE_TIME': '',  # TODO: Calculate from all images
        }
        try:
            template.write(template_vars, str(collection_data_label_local))
        except Exception:
            logger.exception('Error creating label collection_data.lblx: %s',
                             collection_data_label_local)
            raise
        collection_data_label.upload()
        logger.info('Generated collection_data.lblx')

    # Generate collection_browse.csv (must be written before collection_browse.lblx)
    collection_browse_csv = bundle_root / 'browse' / 'collection_browse.csv'
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

    # Collection browse label
    collection_browse_template = template_base / 'collection_browse.lblx'
    if collection_browse_template.exists():
        template = pdstemplate.PdsTemplate(str(collection_browse_template))
        collection_browse_label = bundle_root / 'browse' / 'collection_browse.lblx'
        collection_browse_label_local = cast(Path,
                                             collection_browse_label.get_local_path())
        template_vars = {
            'COLLECTION_BROWSE_CSV_PATH': str(collection_browse_csv),
        }
        try:
            template.write(template_vars, str(collection_browse_label_local))
        except Exception:
            logger.exception('Error creating label collection_browse.lblx: %s',
                             collection_browse_label_local)
            raise
        collection_browse_label.upload()
        logger.info('Generated collection_browse.lblx')

    return {
        'num_products': len(lidvids),
    }


def generate_global_index_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> dict[str, Any]:
    """Generate global index files for bodies and rings.

    Parameters:
        bundle_results_root: Root directory of the bundle.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        Metadata dictionary with 'num_body_rows' and 'num_ring_rows' keys.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    config = dataset.config

    # Get configured backplane types from config
    bodies_cfg = getattr(config.backplanes, 'bodies', [])
    body_backplane_types = [bp['name'] for bp in bodies_cfg]
    rings_cfg = getattr(config.backplanes, 'rings', [])
    ring_backplane_types = [bp['name'] for bp in rings_cfg]

    # Scan for all supplemental files
    supplemental_files: list[FCPath] = []
    data_dir = bundle_root / 'data'
    for suppl_file in data_dir.rglob('*_supplemental.txt'):
        supplemental_files.append(suppl_file)

    # Sort by image name (extracted from filename)
    def get_image_name_from_supplemental(path: FCPath) -> str:
        # Extract image name from filename
        # (e.g., "1234567890w_supplemental.txt" -> "1234567890w")
        return path.name.replace('_supplemental.txt', '')

    supplemental_files.sort(key=get_image_name_from_supplemental)
    logger.info('Found %d supplemental files', len(supplemental_files))

    # Collect body and ring statistics
    body_index_rows: list[dict[str, Any]] = []
    ring_index_rows: list[dict[str, Any]] = []

    for suppl_file in supplemental_files:
        try:
            suppl_text = suppl_file.read_text()
            metadata = json.loads(suppl_text)
        except Exception:
            logger.exception('Error reading supplemental file %s', suppl_file)
            continue

        backplanes = metadata.get('backplanes', {})
        bodies = backplanes.get('bodies', {})
        rings = backplanes.get('rings', {})

        # Derive pds4_path_stub from supplemental file path
        # Supplemental file is at: bundle_root/data/pds4_path_stub_supplemental.txt
        # Get relative path from data directory and remove _supplemental.txt suffix
        suppl_relative = suppl_file.relative_to(data_dir)
        pds4_path_stub = str(suppl_relative).replace('_supplemental.txt', '')
        # Replace path separators with forward slashes for consistency
        pds4_path_stub = pds4_path_stub.replace('\\', '/')

        # Extract LID from metadata or filename
        # TODO: Parse actual LID from label files
        lid = f'urn:nasa:pds:{bundle_name}:data:placeholder::1.0'
        # pds4_path_stub includes path and filename prefix
        # Path relative to data directory
        path_to_image = f'{pds4_path_stub}_backplanes.fits'

        # Body index: one line per image per body
        for body_name, body_data in bodies.items():
            body_row: dict[str, Any] = {
                'LID': lid,
                'body_name': body_name,
                'path_to_image_file': path_to_image,
            }
            body_backplanes = body_data.get('backplanes', {})
            # Add min/max columns for each configured backplane type
            for bp_type in body_backplane_types:
                bp_values = body_backplanes.get(bp_type, {})
                body_row[f'{bp_type}_min'] = bp_values.get('min', '')
                body_row[f'{bp_type}_max'] = bp_values.get('max', '')
            body_index_rows.append(body_row)

        # Ring index: one line per image
        ring_backplanes = rings.get('backplanes', {})
        if ring_backplanes:
            ring_row: dict[str, Any] = {
                'LID': lid,
                'path_to_image_file': path_to_image,
            }
            # Add min/max columns for each configured ring backplane type
            for ring_type in ring_backplane_types:
                ring_values = ring_backplanes.get(ring_type, {})
                ring_row[f'{ring_type}_min'] = ring_values.get('min', '')
                ring_row[f'{ring_type}_max'] = ring_values.get('max', '')
            ring_index_rows.append(ring_row)

    # Generate global_index_bodies.tab
    supplemental_dir = bundle_root / 'document' / 'supplemental'
    bodies_tab = supplemental_dir / 'global_index_bodies.tab'
    bodies_tab_local = cast(Path, bodies_tab.get_local_path())
    bodies_tab_local.parent.mkdir(parents=True, exist_ok=True)
    with bodies_tab_local.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Build header: LID, body_name, path_to_image_file, then min/max for each backplane type
        header = ['LID', 'body_name', 'path_to_image_file']
        for bp_type in body_backplane_types:
            header.append(f'{bp_type}_min')
            header.append(f'{bp_type}_max')
        writer.writerow(header)

        for row in body_index_rows:
            row_data = [row['LID'], row['body_name'], row['path_to_image_file']]
            for bp_type in body_backplane_types:
                row_data.append(row.get(f'{bp_type}_min', ''))
                row_data.append(row.get(f'{bp_type}_max', ''))
            writer.writerow(row_data)
    bodies_tab.upload()
    logger.info('Generated global_index_bodies.tab with %d rows', len(body_index_rows))

    # Generate global_index_rings.tab
    rings_tab = supplemental_dir / 'global_index_rings.tab'
    rings_tab_local = cast(Path, rings_tab.get_local_path())
    rings_tab_local.parent.mkdir(parents=True, exist_ok=True)
    with rings_tab_local.open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Build header: LID, path_to_image_file, then min/max for each ring type
        header = ['LID', 'path_to_image_file']
        for ring_type in ring_backplane_types:
            header.append(f'{ring_type}_min')
            header.append(f'{ring_type}_max')
        writer.writerow(header)

        for row in ring_index_rows:
            row_data = [row['LID'], row['path_to_image_file']]
            for ring_type in ring_backplane_types:
                row_data.append(row.get(f'{ring_type}_min', ''))
                row_data.append(row.get(f'{ring_type}_max', ''))
            writer.writerow(row_data)
    rings_tab.upload()
    logger.info('Generated global_index_rings.tab with %d rows', len(ring_index_rows))

    # Generate label files using templates
    template_base = Path(template_dir)

    # Global index bodies label
    bodies_template = template_base / 'global_index_bodies.lblx'
    if bodies_template.exists():
        template = pdstemplate.PdsTemplate(str(bodies_template))
        bodies_label = supplemental_dir / 'global_index_bodies.lblx'
        bodies_label_local = cast(Path, bodies_label.get_local_path())
        template_vars = {
            'FILE_RECORDS': len(body_index_rows),
        }
        template.write(template_vars, str(bodies_label_local))
        bodies_label.upload()
        logger.info('Generated global_index_bodies.lblx')

    # Global index rings label
    rings_template = template_base / 'global_index_rings.lblx'
    if rings_template.exists():
        template = pdstemplate.PdsTemplate(str(rings_template))
        rings_label = supplemental_dir / 'global_index_rings.lblx'
        rings_label_local = cast(Path, rings_label.get_local_path())
        template_vars = {
            'FILE_RECORDS': len(ring_index_rows),
        }
        template.write(template_vars, str(rings_label_local))
        rings_label.upload()
        logger.info('Generated global_index_rings.lblx')

    return {
        'num_body_rows': len(body_index_rows),
        'num_ring_rows': len(ring_index_rows),
    }
