"""The values a navigation metadata record carries, and the one reader of each.

``navigate_image_files`` writes one JSON record per navigated image.  Two very
different consumers read it back: the reprojection and backplane stages, which
classify the pointing the record supplies, and the results index, which stores
the record column by column so those same stages can read one row instead of
one file.  A value whose usable domain is decided twice has two answers, and a
record the two answers differ about supplies one pointing read as a document and
another read as a row.  So the domain of each such value is decided here, once,
and the reader and the store call the same function.

The store's rule follows from that and is an invariant rather than a
convention:

* every value a reader can use is stored, in the form the reader reads it as;
* nothing else is stored, so a stored value means to a reader exactly what the
  document's own value meant.

A record rebuilt from those columns therefore classifies exactly as its document
does.

What is not decided here is whether a value is *right*.  Whether nine numbers
form a proper rotation is
:func:`spindoctor.support.cmatrix.validated_record_rotation`'s question, and
both readers ask it of the same nine numbers, so a recorded matrix that is not a
rotation is refused identically however it was stored.  This module decides only
which values survive to be asked about.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    'INVALID_OFFSET_TYPE',
    'MALFORMED_OFFSET',
    'MISSING_OFFSET_KEY',
    'NON_FINITE_OFFSET',
    'NULL_OFFSET',
    'UNKNOWN_STATUS',
    'RecordOffset',
    'finite_float',
    'record_offset',
    'record_rotation_values',
    'record_status',
    'record_status_error',
]

UNKNOWN_STATUS = 'unknown'
"""What a record naming no outcome of its own is read as naming.

A record whose top-level ``status`` is absent, null, empty or not a string names
no outcome.  Every reader reports it as this value and the results index stores
it as this value, so the two never disagree about what such a record said.  The
navigator's own vocabulary is ``success``, ``failed``, ``conflicted`` and
``error``, none of which is this.
"""

MISSING_OFFSET_KEY = 'missing_offset_key'
"""The record carries no ``offset`` field at all."""

NULL_OFFSET = 'null_offset'
"""The record carries an ``offset`` field holding null."""

INVALID_OFFSET_TYPE = 'invalid_offset_type'
"""The offset holds a boolean, which is an ``int`` in Python and a pixel nowhere."""

NON_FINITE_OFFSET = 'non_finite_offset'
"""The offset holds NaN or an infinity."""

MALFORMED_OFFSET = 'malformed_offset'
"""The offset is not two values a reader can read as pixels."""


def finite_float(value: Any) -> float | None:
    """Read one recorded number as a finite float, or nothing.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The float, or None when the value is absent, is not a number, is a
        boolean, or is not finite.  A boolean is refused although it converts
        without complaint: it is an ``int`` in Python and a measurement nowhere.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def record_status(nav_metadata: dict[str, Any]) -> str:
    """Read the outcome a navigation record names.

    Parameters:
        nav_metadata: The parsed metadata record.

    Returns:
        The record's own top-level ``status``, or :data:`UNKNOWN_STATUS` when it
        names none -- an absent field, a null one, an empty string, or a value
        that is not a string.  The nested copy inside ``navigation_result``
        never stands in for it: the pointing ladder's first question is whether
        the record's own field says ``success``, and a reader that borrowed the
        nested copy would apply a corrected pointing to a record that supplies
        none.
    """
    status = nav_metadata.get('status')
    if isinstance(status, str) and status:
        return status
    return UNKNOWN_STATUS


def record_status_error(nav_metadata: dict[str, Any]) -> str:
    """Read what a navigation record says went wrong.

    Parameters:
        nav_metadata: The parsed metadata record.

    Returns:
        The record's ``status_error``, or :data:`UNKNOWN_STATUS` when it names
        none.  Every failed and conflicted navigation writes no such field, so
        the default is the common case rather than an edge one.
    """
    error = nav_metadata.get('status_error')
    if isinstance(error, str) and error:
        return error
    return UNKNOWN_STATUS


@dataclass(frozen=True)
class RecordOffset:
    """The pixel offset a navigation record supplies, classified.

    Parameters:
        pair: The recorded ``(dv, du)`` offset in pixels, or None when the
            record supplies none a reader can use.
        reason: Why it supplies none, or None when ``pair`` is usable.  One of
            :data:`MISSING_OFFSET_KEY`, :data:`NULL_OFFSET`,
            :data:`INVALID_OFFSET_TYPE`, :data:`NON_FINITE_OFFSET` and
            :data:`MALFORMED_OFFSET`.
    """

    pair: tuple[float, float] | None
    reason: str | None


def record_offset(nav_metadata: dict[str, Any]) -> RecordOffset:
    """Read the pixel offset a navigation record supplies.

    The value is read as written: two elements, each convertible to a finite
    float and neither a boolean.  A sequence of any other length is refused
    whole rather than truncated, because a reader that took the first two of
    three would apply a pointing built from part of a value nobody wrote.

    Parameters:
        nav_metadata: The parsed metadata record.

    Returns:
        The classified offset.  A pair is returned exactly when a reader can
        apply it, so a store holding these records stores this pair and stores
        nothing where there is none.
    """
    if 'offset' not in nav_metadata:
        return RecordOffset(pair=None, reason=MISSING_OFFSET_KEY)
    offset = nav_metadata['offset']
    if offset is None:
        return RecordOffset(pair=None, reason=NULL_OFFSET)
    if isinstance(offset, (str, bytes)) or not isinstance(offset, Sequence):
        return RecordOffset(pair=None, reason=MALFORMED_OFFSET)
    if len(offset) != 2:
        return RecordOffset(pair=None, reason=MALFORMED_OFFSET)
    dv_raw, du_raw = offset[0], offset[1]
    if isinstance(dv_raw, bool) or isinstance(du_raw, bool):
        return RecordOffset(pair=None, reason=INVALID_OFFSET_TYPE)
    try:
        dv = float(dv_raw)
        du = float(du_raw)
    except (TypeError, ValueError):
        return RecordOffset(pair=None, reason=MALFORMED_OFFSET)
    if not math.isfinite(dv) or not math.isfinite(du):
        return RecordOffset(pair=None, reason=NON_FINITE_OFFSET)
    return RecordOffset(pair=(dv, du), reason=None)


def record_rotation_values(value: Any) -> list[Any] | None:
    """Read the nine row-major values a recorded rotation is written as.

    A record writes a rotation as nine row-major values, and a 3x3 nesting of
    them is read as the same nine.  Nothing else is a rotation this reader can
    evaluate, and nothing else is stored.

    Booleans are refused here rather than left to the validator, because they
    are the one element type the two readers would otherwise judge differently:
    ``numpy`` promotes a boolean beside a number to that number's type, so a
    single ``True`` among eight floats would read as ``1.0`` from a document
    while a store that refuses booleans held nothing at all.

    Parameters:
        value: The recorded value.

    Returns:
        The nine values in row-major order, exactly as they were recorded, or
        None when the value is not one of the two shapes.  Whether the nine are
        real, finite numbers forming a proper rotation is decided by the
        validator that both readers apply to them.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    entries = list(value)
    if len(entries) == 3 and all(
        not isinstance(row, (str, bytes)) and isinstance(row, Sequence) and len(row) == 3
        for row in entries
    ):
        entries = [entry for row in entries for entry in row]
    if len(entries) != 9:
        return None
    if any(isinstance(entry, bool) for entry in entries):
        return None
    return entries
