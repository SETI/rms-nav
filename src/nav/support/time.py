import datetime
from typing import cast

import julian


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


def et_to_utc(et: float, digits: int = 3) -> str:
    """Returns the UTC time for a given ET time.

    Parameters:
        et: The SPICE ET time (equivalent to TDB).
        digits: The number of digits to include after the decimal point.

    Returns:
        The UTC time as a string.
    """

    return cast(str, julian.iso_from_tai(julian.tai_from_tdb(et), digits=digits))


def utc_to_et(utc: str) -> float:
    """Returns the ET time (TDB seconds) for a given UTC time string.

    Parameters:
        utc: The UTC time as an ISO 8601 formatted string (e.g.,
            "2008-01-01 12:00:00" or "2008-01-01T12:00:00").

    Returns:
        The SPICE ET time (equivalent to TDB) in seconds as a float.
    """

    result = julian.tdb_from_tai(julian.tai_from_iso(utc))
    return float(result)
