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

What is not decided here is whether a value is *right*.  Whether a 3x3 array of
real numbers is a proper rotation is
:func:`spindoctor.support.cmatrix.validated_record_rotation`'s question, and
both readers ask it of the same array, so a recorded matrix that is not a
rotation is refused identically however it was stored.  This module decides only
which values survive to be asked about.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from filecache import FCPath

from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'INVALID_OFFSET_TYPE',
    'MALFORMED_OFFSET',
    'MISSING_OFFSET_KEY',
    'NON_FINITE_OFFSET',
    'NULL_OFFSET',
    'REAL_NUMBER_DTYPE_KINDS',
    'UNKNOWN_STATUS',
    'NavRecord',
    'RecordOffset',
    'finite_float',
    'record_offset',
    'record_rotation_matrix',
    'record_status',
    'record_status_error',
]

REAL_NUMBER_DTYPE_KINDS = frozenset({'i', 'u', 'f'})
"""Array dtype kinds a recorded value may hold and still be read as numbers.

Signed integers, unsigned integers and floats.  Booleans are excluded although
they convert to float without complaint, and so are text, objects, complex
numbers and every date-like kind: none of them is a measurement, and nine
``True`` values would otherwise read as an identity rotation.
"""

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


@dataclass(frozen=True)
class NavRecord:
    """One image's navigation record, and where that record is kept.

    A record read from its document and one rebuilt from an index row are the
    same thing to every consumer, so both arrive in this shape and neither
    carries which storage produced it.

    Parameters:
        path: The document.  For a record rebuilt from a row this is the
            document the ingest recorded reading, or -- for a row written
            before anything recorded one -- where the stub says the document
            lives.  It is what a message about this record names, so that an
            operator is always told a file they can open.
        stub: The image's results path stub: its identity under the root, and
            the name of its log.
        metadata: The record, in the shape the navigator writes it.  A record
            rebuilt from a row carries the fields its columns hold and no
            others, which is the part of the document its consumer reads.
    """

    path: FCPath
    stub: str
    metadata: dict[str, Any]


def finite_float(value: Any) -> float | None:
    """Read one recorded number as a finite float, or nothing.

    Parameters:
        value: The value as it was parsed.

    Returns:
        The float, or None when the value is absent, is not a number, is a
        boolean, is not finite, or is an integer too large to be one.  A
        boolean is refused although it converts without complaint: it is an
        ``int`` in Python and a measurement nowhere.  JSON puts no bound on an
        integer literal, so a recorded integer of several hundred digits is a
        value a reader cannot use rather than an error for a caller to meet.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except OverflowError:
        return None
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
    except (OverflowError, TypeError, ValueError):
        # ``OverflowError`` among them: a JSON integer of more than about three
        # hundred digits is a value no reader can turn into a pixel, and a
        # record carrying one is a malformed record rather than an exception
        # for a caller that asked for a classification.
        return RecordOffset(pair=None, reason=MALFORMED_OFFSET)
    if not math.isfinite(dv) or not math.isfinite(du):
        return RecordOffset(pair=None, reason=NON_FINITE_OFFSET)
    return RecordOffset(pair=(dv, du), reason=None)


def _records_booleans(entry: Any) -> bool:
    """Whether one recorded entry is a boolean, or is written wholly of them.

    Parameters:
        entry: One of the nine entries, exactly as it was recorded.

    Returns:
        True when the entry on its own assembles into an array of boolean
        kind, which a bare ``True`` and every nesting of nothing but booleans
        does.  An entry no array can be made of at all is not one of those; the
        assembly of the nine refuses it.
    """
    try:
        return bool(np.asarray(entry).dtype.kind == 'b')
    except ValueError:
        return False


def _nine_recorded_entries(value: Any) -> list[Any] | None:
    """Read the nine entries a recorded rotation is written as, uncoerced.

    Parameters:
        value: The recorded value.

    Returns:
        The nine entries in row-major order, exactly as they were recorded, or
        None when the value is neither nine entries nor a 3x3 nesting of them,
        or when any entry is written wholly of booleans.  Booleans are refused
        one entry at a time rather than on the nine assembled together, because
        ``numpy`` promotes a boolean beside a number to that number's type: a
        single ``True`` among eight floats would otherwise assemble into a
        float array and read as ``1.0``.  The question is asked of the array
        each entry makes rather than of the entry's own Python type, because an
        entry is a container in every nesting deeper than the two a record is
        written in, and a type test walks straight past those.
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
    if any(_records_booleans(entry) for entry in entries):
        return None
    return entries


def record_rotation_matrix(value: Any) -> NDArrayFloatType | None:
    """Read a recorded rotation as the 3x3 matrix of numbers it denotes.

    A record writes a rotation as nine row-major values, and a 3x3 nesting of
    them is read as the same nine.  Either shape is then assembled into one 3x3
    array, which accepts any further nesting an array library can reconcile
    into that shape -- nine one-element rows among them.  That the accepted
    domain is wider than the two written shapes is deliberate: the recorded
    value denotes exactly one matrix in every one of them, and a reader that
    refused one of the denoting shapes while applying another would classify a
    record by how its nine numbers were bracketed rather than by what they say.

    This is the whole of what "a recorded matrix a reader can evaluate" means,
    for the reader that applies one and for the store that holds one alike.
    Asking it twice is how a rotation the reader applied came to be stored as
    nothing, so the question has exactly this one answer.

    Parameters:
        value: The recorded value.

    Returns:
        The 3x3 matrix as float64, or None when the recorded value is not nine
        entries or a 3x3 nesting of them, when those entries are of shapes no
        3x3 array can be made of, when they are not real numbers (text,
        booleans, nulls, objects, or an integer too large to be a float, all of
        which assemble into an array of some other kind), or when any of them
        is not finite.  Whether the matrix that survives is a proper rotation
        is decided by the validator both readers apply to it.
    """
    entries = _nine_recorded_entries(value)
    if entries is None:
        return None
    try:
        assembled = np.asarray(entries).reshape(3, 3)
    except ValueError:
        # Nine entries of shapes no single array can hold -- a row of two
        # beside eight scalars -- is a malformed record like any other, and not
        # an exception for a caller asking what a record denotes.
        return None
    if assembled.dtype.kind not in REAL_NUMBER_DTYPE_KINDS:
        return None
    matrix: NDArrayFloatType = assembled.astype(np.float64)
    if not bool(np.all(np.isfinite(matrix))):
        return None
    return matrix
