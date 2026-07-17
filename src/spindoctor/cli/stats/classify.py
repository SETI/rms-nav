"""Time helpers for the statistics system."""

import julian

__all__ = ['date_from_image_et', 'datetime_from_image_et']


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


def datetime_from_image_et(image_et: float | None) -> str | None:
    """UTC calendar date and time (``YYYY-MM-DDTHH:MM:SS``) for a SPICE ET epoch.

    The same instant as :func:`date_from_image_et`, to the second.  The
    report shows this where a bare date would collapse many images taken
    the same day into one indistinguishable bound.

    Parameters:
        image_et: TDB seconds past J2000 (the ``provenance.image_et``
            metadata field), or None.

    Returns:
        The UTC timestamp string, or None when ``image_et`` is None.
    """
    if image_et is None:
        return None
    # digits=None truncates to whole seconds; digits=0 would leave a
    # trailing '.' with no fractional part behind it.
    return str(julian.iso_from_tai(julian.tai_from_tdb(image_et), digits=None))
