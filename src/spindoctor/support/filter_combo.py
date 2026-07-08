"""Canonical form of a multi-filter optical combo string.

Most missions in this pipeline carry one or two filter names per exposure
(e.g. Cassini ISS uses two filter wheels per camera; Voyager and Galileo use
one).  Multiple call sites need a canonical string for the same combo:

- ``mag_offset_table`` lookup keys.
- Test image-library sidecar ``filter_combo`` field.
- Per-image log lines and metadata fields.

This module supplies the single ``canonicalize`` rule (alphabetic-sort joined
by ``'+'``) so the spelling never drifts between consumers.
"""

from collections.abc import Sequence

__all__ = ['canonicalize']


def canonicalize(filters: Sequence[str | None]) -> str:
    """Return a canonical string representation of a filter combo.

    Drops ``None`` entries, sorts the remaining filter names alphabetically,
    joins them with ``'+'``.  Duplicate names are preserved (so
    ``['CL', 'CL']`` becomes ``'CL+CL'``).  An empty sequence (or a
    sequence containing only ``None`` entries) returns ``'NONE'``.

    Parameters:
        filters: Iterable of filter name strings; ``None`` entries are
            dropped.

    Returns:
        Canonical ``'+'``-joined sorted filter combo string, or ``'NONE'``
        if no non-None filters are given.

    Examples:
        ``canonicalize([])`` -> ``'NONE'``
        ``canonicalize(['CL1'])`` -> ``'CL1'``
        ``canonicalize(['CL2', 'CL1'])`` -> ``'CL1+CL2'``
        ``canonicalize(['CL', 'CL'])`` -> ``'CL+CL'``
        ``canonicalize(['F1', None, 'F2'])`` -> ``'F1+F2'``
    """
    kept = [name for name in filters if name is not None]
    if not kept:
        return 'NONE'
    return '+'.join(sorted(kept))
