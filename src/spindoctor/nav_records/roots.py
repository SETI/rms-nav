"""What makes two spellings of a results root one root.

A results root reaches a program as a command-line value, a configuration key or
an environment variable, and the three routinely differ by a trailing slash or by
being relative to the working directory.  Everything that keys anything on a root
-- a record source deciding which of its roots a selection names, an index row
recording where it was ingested from, a consumer filtering on the root it was
itself pointed at -- only meets what it is looking for if both sides spell the
root the same way.  So one function spells it, and everything calls that one.

The rule is about identity rather than about storage, so it lives here, where a
reader of documents and a reader of rows can both reach it without either of them
reaching a database.
"""

from collections.abc import Sequence
from pathlib import Path

from filecache import FCPath

__all__ = [
    'distinct_roots',
    'normalize_root_url',
]


def normalize_root_url(root: str | Path | FCPath) -> str:
    """Return the form of a results root that the index stores and compares.

    The rule is one absolute, resolved POSIX rendering, so that a root named
    relatively on one run and absolutely on the next, named with a trailing
    slash by one program and without by another, named through a link by one
    operator and at its own location by another, or written with ``~`` or a
    ``..`` in it, is one root.  That rendering carries no trailing separator
    except on the filesystem root itself, whose separator is its whole name.

    Resolving here is what lets everything downstream stop thinking about
    paths.  A root is canonical from the moment it is spelled, so joining a
    validated key onto it gives one answer for every call in the seam, and no
    reader needs a rule of its own about what a join may produce.  The
    resolution applies to every kind of root without a branch: a remote
    location has no links and no relative form, and the storage layer returns
    such a URL unchanged and reaches no network to do it.

    Two spellings are refused here rather than rendered.  An empty one renders
    as whatever directory the process happens to be in, so a program handed one
    -- which is what an unset variable in ``--nav-results-root "$ROOT"`` hands it
    -- would walk the working directory, write its documents under a root nobody
    named, and report a completed pass.  One carrying a null byte renders
    perfectly well and then fails at the first call that reaches the filesystem,
    which is a failure charged to a directory listing rather than to the word
    that caused it.  Every caller reads a root through here, so both are refused
    once for the whole surface.

    Parameters:
        root: The results root as its holder spelled it: a local path, an
            :class:`FCPath`, or a cloud URL.

    Returns:
        The normalized root URL.

    Raises:
        ValueError: If the spelling is not a location: empty, carrying a null
            byte, or one the storage layer itself refuses to resolve.
    """
    spelled = str(root)
    if spelled == '':
        raise ValueError('a results root spelled as nothing at all is not a location')
    if '\x00' in spelled:
        raise ValueError(f'a results root carrying a null byte is not a location: {spelled!r}')
    return FCPath(root).expanduser().resolve().as_posix()


def distinct_roots(roots: Sequence[str]) -> list[str]:
    """Normalize the given roots and drop the repeats, keeping their order.

    ``/data/x`` and ``/data/x/`` are one root, and a command line naming both
    means the tree once.  Walking it twice reads every document twice and gives
    one root two ingest runs; in a pass divided into cloud tasks it also hands
    every document out in two shares, leaves the first of the two runs
    unfinished forever, and -- since a completion stamps the newest run and then
    finds nothing outstanding -- tells the operator that a root it has just
    finished was never divided up.

    Every mode of a pass reads the roots through this, and so does the driver
    that reports which roots it was given: a run that named a root two ways and
    then reported on it once reads as a root having gone missing.

    Parameters:
        roots: The roots as their holder spelled them.

    Returns:
        The normalized roots, first spelling first.

    Raises:
        ValueError: If a root is not a location: one the storage layer refuses
            to resolve, one carrying a null byte, or an empty spelling, which
            is the working directory rather than a root anybody named.
    """
    distinct: dict[str, None] = {}
    for root in roots:
        distinct.setdefault(normalize_root_url(root), None)
    return list(distinct)
