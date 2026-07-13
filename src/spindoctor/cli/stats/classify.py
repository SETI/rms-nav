"""Time helpers for the statistics system."""

import julian

__all__ = ['date_from_image_et']


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
