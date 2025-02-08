import datetime

import oops


def ra_rad_to_hms(ra: float) -> str:
    """Convert right ascension in radians to a pretty string."""

    if ra < 0:
        raise ValueError(f'ra cannot be negative, got {ra}')

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
    """Convert declination in radians to a pretty string."""

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
    """Return the current time as an ISO 8601 string."""

    return datetime.datetime.now().astimezone().isoformat()


def now_dt() -> datetime.datetime:
    """Return the current time as a DateTime."""

    return datetime.datetime.now().astimezone()

def dt_delta_str(start_time: datetime.datetime,
                 end_time: datetime.datetime) -> str:
    """Return the difference between two DateTimes as a float"""

    return str(end_time - start_time)
