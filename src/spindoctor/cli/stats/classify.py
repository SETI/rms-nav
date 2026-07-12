"""Image-name classification and time helpers for the statistics system."""

import re

import julian

__all__ = ['date_from_image_et', 'instrument_from_image_name']

# Filename shapes for the four supported instruments, tested against the bare
# image name with any calibration suffix and extension removed:
#   Cassini ISS      N1454725799_1 / W1728613298_8
#   Voyager ISS      C3250013 (7-digit FDS)
#   Galileo SSI      C0349632000R / C0059881800S (10-digit SCLK + R/S)
#   New Horizons     lor_0003103486_0x630_sci
_COISS_RE = re.compile(r'^[NW]\d{10}(_\d+)?$')
_VGISS_RE = re.compile(r'^C\d{7}$')
_GOSSI_RE = re.compile(r'^C\d{10}[A-Z]?$')
_NHLORRI_RE = re.compile(r'^LOR_\d+.*$')


def instrument_from_image_name(image_name: str) -> str:
    """Classify an image filename into one of the supported instruments.

    Fallback only: ingest prefers the ``observation.instrument`` field the
    pipeline records in each metadata document and calls this just for
    documents that lack the field.

    Parameters:
        image_name: Image filename, with or without directory, calibration
            suffix, or extension (``N1454725799_1_CALIB.IMG`` and
            ``N1454725799_1`` classify identically).

    Returns:
        One of ``'coiss'``, ``'vgiss'``, ``'gossi'``, ``'nhlorri'``, or
        ``'unknown'``.
    """
    name = image_name.rsplit('/', 1)[-1].upper()
    name = name.split('.', 1)[0]
    for suffix in ('_CALIB', '_GEOMED', '_CLEANED', '_RAW'):
        name = name.removesuffix(suffix)
    if _COISS_RE.match(name):
        return 'coiss'
    if _VGISS_RE.match(name):
        return 'vgiss'
    if _GOSSI_RE.match(name):
        return 'gossi'
    if _NHLORRI_RE.match(name):
        return 'nhlorri'
    return 'unknown'


def date_from_image_et(image_et: float | None) -> str | None:
    """UTC calendar date (``YYYY-MM-DD``) for a SPICE ET epoch.

    Parameters:
        image_et: TDB seconds past J2000 (the ``provenance.image_et``
            metadata field), or None.

    Returns:
        The UTC date string, or None when ``image_et`` is None.
    """
    if image_et is None:
        return None
    iso = str(julian.iso_from_tai(julian.tai_from_tdb(image_et), digits=0))
    return iso[:10]
