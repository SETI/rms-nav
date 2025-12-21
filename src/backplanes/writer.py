from typing import Any

from filecache import FCPath
from astropy.io import fits
import numpy as np
from pdslogger import PdsLogger
import pdstemplate

from nav.config import Config
from nav.config.logger import DEFAULT_LOGGER
from nav.obs import ObsSnapshot
from pathlib import Path


def write_fits(
    *,
    fits_file_path: FCPath,
    snapshot: ObsSnapshot,
    master_by_type: dict[str, np.ndarray],
    body_id_map: np.ndarray,
    config: Config,
    logger: PdsLogger = DEFAULT_LOGGER,
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

    # BODY_ID_MAP first (after Primary) - only include if not empty
    has_body_id_map = np.any(body_id_map != 0)
    if has_body_id_map:
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
