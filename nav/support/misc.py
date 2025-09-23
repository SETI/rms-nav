import datetime
import math
from typing import Any

import oops


def ra_rad_to_hms(ra: float) -> str:
    """Converts right ascension in radians to a formatted string in hours, minutes, and seconds.

    Parameters:
        ra: Right ascension value in radians.

    Returns:
        Formatted string in the form "HHhMMmSS.SSSs".

    Raises:
        ValueError: If the right ascension value is negative.
    """

    if ra < 0:
        raise ValueError(f'ra cannot be negative, got {ra}')

    ra = ra % math.tau
    ra_deg = ra * oops.DPR / 15  # In hours
    hh = int(ra_deg)
    mm = int((ra_deg - hh) * 60)
    ss = int((ra_deg - hh - mm / 60.) * 3600 * 1000 + .5) / 1000
    if ss >= 60:
        mm += 1
        ss -= 60
    if mm >= 60:
        hh += 1
        mm -= 60
    if hh >= 24:
        hh -= 24

    return f"{hh:02d}h{mm:02d}m{ss:06.3f}s"


def dec_rad_to_dms(dec: float) -> str:
    """Converts declination in radians to a formatted string in degrees, minutes, and seconds.

    Parameters:
        dec: Declination value in radians.

    Returns:
        Formatted string in the form "+/-DDDdMMmSS.SSSs".
    """

    dec_deg = dec * oops.DPR  # In degrees
    is_neg = False
    if dec_deg < 0:
        is_neg = True
        dec_deg = -dec_deg
    dd = int(dec_deg)
    mm = int((dec_deg - dd) * 60)
    ss = int((dec_deg - dd - mm / 60.) * 3600 * 1000 + .5) / 1000
    if ss >= 60:
        mm += 1
        ss -= 60
    if mm >= 60:
        dd += 1
        mm -= 60
    # TODO Check this - does this make sense for both dec and rad?
    if dd >= 180:
        dd -= 360
    elif dd <= -180:
        dd += 360
    if dd < 0:
        is_neg = not is_neg
        dd = -dd
    neg = '-' if is_neg else '+'

    return f"{neg}{dd:03d}d{mm:02d}m{ss:06.3f}s"


def now_iso() -> str:
    """Returns the current time as an ISO 8601 formatted string with timezone information.

    Returns:
        Current time as an ISO 8601 formatted string.
    """

    return datetime.datetime.now().astimezone().isoformat()


def now_dt() -> datetime.datetime:
    """Returns the current time as a datetime object with timezone information.

    Returns:
        Current time as a timezone-aware datetime object.
    """

    return datetime.datetime.now().astimezone()


def dt_delta_str(start_time: datetime.datetime,
                 end_time: datetime.datetime) -> str:
    """Returns the difference between two datetime objects as a string representation.

    Parameters:
        start_time: The starting datetime.
        end_time: The ending datetime.

    Returns:
        String representation of the time difference.
    """

    return str(end_time - start_time)


def flatten_list(lst: list[Any]) -> list[Any]:
    """Flattens a list of lists into a single list.

    Parameters:
        lst: The list to flatten.

    Returns:
        A flattened list.
    """

    return [x for sublist in lst for x in sublist]


def safe_lstrip_zero(s: str) -> str:
    """Strips leading zeros from a string but leaves one zero behind if that's all there is.

    Parameters:
        s: The string to strip leading zeros from.

    Returns:
        The string with leading zeros stripped.
    """

    if not s:
        return s

    ret = s.lstrip('0')
    if ret == '':
        ret = '0'
    return ret
