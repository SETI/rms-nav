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
        et: The ET time.

    Returns:
        The UTC time as a string.
    """

    return cast(str, julian.iso_from_tai(julian.tai_from_tdb(et), digits=digits))
