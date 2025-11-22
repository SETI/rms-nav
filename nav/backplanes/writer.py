from typing import Any

from filecache import FCPath
from astropy.io import fits
import numpy as np
import pdstemplate

from nav.config import Config
from nav.config.logger import DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from pathlib import Path


def write_fits_and_label(
    *,
    fits_file_path: FCPath,
    label_file_path: FCPath,
    snapshot: ObsSnapshot,
    master_by_type: dict[str, np.ndarray],
    body_id_map: np.ndarray,
    config: Config,
    logger: Any = DEFAULT_LOGGER,
) -> None:
    """Write FITS and PDS4 label using FCPath.

    Parameters:
        fits_file_path: The FITS file path.
        label_file_path: The PDS4 label file path.
        snapshot: The observation snapshot.
        master_by_type: The master by type.
        body_id_map: The body id map.
    """

    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = []
    primary = fits.PrimaryHDU()
    hdus.append(primary)

    # BODY_ID_MAP first (after Primary)
    id_hdu = fits.ImageHDU(data=body_id_map.astype('int32'), name='BODY_ID_MAP')
    hdus.append(id_hdu)

    # Backplane arrays
    units_map: dict[str, str] = {}
    # Try to pull declared units from config
    for bp in getattr(config.backplanes, 'bodies', []):
        units_map[bp['name']] = bp.get('units', '')
    for rp in getattr(config.backplanes, 'rings', []):
        if rp['name'] != 'distance':  # not written as a master backplane type
            units_map[rp['name']] = rp.get('units', '')

    # Filter out zero-only backplanes
    filtered_master = {k: v for k, v in master_by_type.items() if np.any(v != 0.0)}

    for name, arr in filtered_master.items():
        hdu = fits.ImageHDU(data=arr.astype('float32'), name=name.upper())
        if name in units_map and units_map[name]:
            hdu.header['BUNIT'] = units_map[name]
        hdus.append(hdu)

    hdul = fits.HDUList(hdus)
    local_path = fits_file_path.get_local_path()
    hdul.writeto(local_path, overwrite=True)
    fits_file_path.upload()

    # PDS4 label via PdsTemplate
    try:
        template_path = Path(__file__).resolve().parent / 'templates' / 'backplanes.lblx'
        template = pdstemplate.PdsTemplate(str(template_path))
        xml_meta: dict[str, Any] = {
            'PRODUCT_ID': fits_file_path.name,
            'FILE_NAME': fits_file_path.name,
            'LINES': snapshot.data.shape[0],
            'LINE_SAMPLES': snapshot.data.shape[1],
            'BANDS': len(filtered_master) + 1,  # + BODY_ID_MAP
            'BACKPLANE_TYPES': sorted([k.upper() for k in filtered_master.keys()]),
            'TARGETS': [],
        }
        # Populate targets if mapping exists (optional)
        target_lids = getattr(config.backplanes, 'target_lids', {})
        for naif_id, lid in target_lids.items():
            xml_meta.setdefault('TARGETS', []).append({'NAIF_ID': int(naif_id), 'LID': lid})

        local_label = label_file_path.get_local_path()
        template.write(xml_meta, local_label)
        label_file_path.upload()
    except Exception as e:
        logger.error('Failed to write PDS4 label for %s: %s', fits_file_path, e)
