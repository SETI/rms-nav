"""Root identity and per-root ingest bookkeeping for the results index.

Every row of the index names the results root it was ingested from, and every
consumer filters on the root it was itself pointed at.  The two only meet if
both spell the root the same way, so one function spells it and everything
calls that one: a results root reaches a program as a command-line value, a
configuration key or an environment variable, and the three routinely differ by
a trailing slash or by being relative to the working directory.

Absence of a row is only meaningful once the root is known to have been
ingested.  "No row for this stub" means "this image was never navigated" if and
only if the whole root was walked, and means nothing at all otherwise, so a
consumer asks :func:`require_ingested_roots` before it reads absence as an
answer.  An ingest records its root in ``ingest_runs`` when it starts and stamps
the finish time when it completes, so a run that died halfway leaves a row that
says as much.
"""

from pathlib import Path

import sqlalchemy
from filecache import FCPath

from spindoctor.results_index.schema import INGEST_RUNS

__all__ = ['ingested_roots', 'normalize_root_url', 'require_ingested_roots']


def normalize_root_url(root: str | Path | FCPath) -> str:
    """Return the form of a results root that the index stores and compares.

    The rule is one absolute POSIX rendering with no trailing separator, so that
    a root named relatively on one run and absolutely on the next, or named with
    a trailing slash by one program and without by another, is one root.

    Parameters:
        root: The results root as its holder spelled it: a local path, an
            :class:`FCPath`, or a cloud URL.

    Returns:
        The normalized root URL.
    """
    normalized = FCPath(root).absolute().as_posix()
    stripped = normalized.rstrip('/')
    # A root that is nothing but separators is the filesystem root, which is the
    # one root whose trailing separator is the whole name.
    return stripped or '/'


def ingested_roots(connection: sqlalchemy.Connection) -> list[str]:
    """Return every root whose newest ingest run completed.

    A root is listed once, on the strength of its newest run alone: an ingest
    that started and died leaves the root unusable however many earlier runs
    finished, because the tree it half-walked is the tree a consumer would be
    reading absence from.

    Parameters:
        connection: An open connection to the index.

    Returns:
        The normalized root URLs, in name order.
    """
    newest = (
        sqlalchemy.select(
            INGEST_RUNS.c.root_url,
            sqlalchemy.func.max(INGEST_RUNS.c.run_id).label('run_id'),
        )
        .group_by(INGEST_RUNS.c.root_url)
        .subquery()
    )
    completed = (
        sqlalchemy.select(INGEST_RUNS.c.root_url)
        .join(newest, INGEST_RUNS.c.run_id == newest.c.run_id)
        .where(INGEST_RUNS.c.finished_utc.is_not(None))
        .order_by(INGEST_RUNS.c.root_url)
    )
    return [str(row.root_url) for row in connection.execute(completed)]


def require_ingested_roots(
    connection: sqlalchemy.Connection, roots: list[str], *, url: str
) -> None:
    """Verify that every named root has been fully ingested into this index.

    Parameters:
        connection: An open connection to the index.
        roots: The normalized root URLs the caller means to read.
        url: The index URL, so the message says which index was asked.

    Raises:
        ValueError: If any named root has no completed ingest run, naming the
            roots that are missing and the roots the index does hold.  Absence
            must never be read as "nothing was navigated" under that root.
    """
    available = ingested_roots(connection)
    missing = [root for root in roots if root not in available]
    if not missing:
        return
    held = ', '.join(available) if available else '(none)'
    raise ValueError(
        f'{url}: the results index has no completed ingest of {", ".join(missing)}. '
        f'It holds: {held}. Run sd_stats_ingest over that root first; until then the '
        f'index cannot say whether an image under it was navigated.'
    )
