"""Root identity and per-root ingest bookkeeping for the results index.

Every row of the index names the results root it was ingested from, and every
consumer filters on the root it was itself pointed at.  The two only meet if
both spell the root the same way, so one function spells it and everything
calls that one: a results root reaches a program as a command-line value, a
configuration key or an environment variable, and the three routinely differ by
a trailing slash or by being relative to the working directory.

Absence of a row is only meaningful once the root is known to have been
ingested.  "No row in ``images`` for this stub" means "this image was never
navigated" only if the whole root was walked, and means nothing at all
otherwise, so a consumer asks :func:`require_ingested_roots` before it reads
absence as an answer.  An ingest records its root in ``ingest_runs`` when it
starts and stamps the finish time when it completes, so a run that died halfway
leaves a row that says as much.

A completed run is necessary and not sufficient.  Both tables that record a file
have to be asked: a document the ingest refused is a file that exists, and it
has a row in ``failed_files`` and none in ``images``, so a consumer reading the
absence from ``images`` alone reports a navigated image as one nothing
navigated.  What a completed run makes true is that absence from *both* tables
means no file was there to read -- for every directory that run listed.
"""

from dataclasses import dataclass
from pathlib import Path

import sqlalchemy
from filecache import FCPath

from spindoctor.results_index.engine import masked_url
from spindoctor.results_index.schema import INGEST_RUNS

__all__ = [
    'NewestPass',
    'ingested_roots',
    'newest_pass',
    'normalize_root_url',
    'require_ingested_roots',
]


def normalize_root_url(root: str | Path | FCPath) -> str:
    """Return the form of a results root that the index stores and compares.

    The rule is one absolute POSIX rendering, so that a root named relatively on
    one run and absolutely on the next, or named with a trailing slash by one
    program and without by another, is one root.  That rendering carries no
    trailing separator except on the filesystem root itself, whose separator is
    its whole name.

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
            byte, or one the storage layer itself refuses to render absolute.
    """
    spelled = str(root)
    if spelled == '':
        raise ValueError('a results root spelled as nothing at all is not a location')
    if '\x00' in spelled:
        raise ValueError(f'a results root carrying a null byte is not a location: {spelled!r}')
    return FCPath(root).absolute().as_posix()


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


@dataclass(frozen=True)
class NewestPass:
    """What the newest ingest pass over one root recorded about its own reach.

    Parameters:
        finished_utc: When that pass finished, as it stamped it, and None when
            the root has no run row or the run never finished.  It is the age
            of the answer a consumer reads out of the index, which is the one
            fact that says whether the tree has moved on since.
        directories_missed: How many directories that pass did not list.
    """

    finished_utc: str | None
    directories_missed: int


def newest_pass(connection: sqlalchemy.Connection, root_url: str) -> NewestPass:
    """Return what the newest pass over one root recorded about itself.

    A directory nobody enumerated holds files nobody recorded, and absence of a
    row under it therefore says nothing about the image whose document is there.
    A consumer that reads absence as a positive answer -- "this image was never
    navigated" -- asks for this and says so when the count is not zero, because
    the run completed all the same and nothing else in the index shows the gap.
    It asks for the finish time in the same breath and for the same reason: the
    index answers as of that moment and detects no change since, so the moment
    belongs with the answer rather than in the head of whoever exported the URL.

    Both come from the newest run row of the named root alone.  One database
    serves several roots, and the newest run in the table is routinely another
    root's.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to ask about.

    Returns:
        What that pass recorded, with no finish time and a count of zero when
        the root has no run row at all.
    """
    newest = (
        sqlalchemy.select(INGEST_RUNS.c.finished_utc, INGEST_RUNS.c.directories_missed)
        .where(INGEST_RUNS.c.root_url == root_url)
        .order_by(INGEST_RUNS.c.run_id.desc())
        .limit(1)
    )
    row = connection.execute(newest).first()
    if row is None:
        return NewestPass(finished_utc=None, directories_missed=0)
    missed = row.directories_missed
    return NewestPass(
        finished_utc=None if row.finished_utc is None else str(row.finished_utc),
        directories_missed=0 if missed is None else int(missed),
    )


def require_ingested_roots(
    connection: sqlalchemy.Connection, roots: list[str], *, url: str
) -> None:
    """Verify that every named root has been fully ingested into this index.

    The index URL is masked here rather than by each caller.  This refusal is
    printed to a terminal and written to run logs, an index URL can carry a
    database password, and a caller that forgets to mask one is a leak in every
    program that consumes the index.  The roots are named as they are: a results
    root is not a connection URL, has no credentials to hide, and is the one
    string the reader has to correct.

    Parameters:
        connection: An open connection to the index.
        roots: The normalized root URLs the caller means to read.
        url: The index URL, so the message says which index was asked.

    Raises:
        ValueError: If any named root has no completed ingest run, naming the
            roots that are missing and the roots the index does hold.  Under
            such a root, absence of a row must never be read as "nothing was
            navigated".  Under an ingested one it may be, but only once the
            refusal table has been asked too: a completed run makes absence
            from both tables meaningful, not absence from ``images`` alone.
    """
    available = ingested_roots(connection)
    missing = [root for root in roots if root not in available]
    if not missing:
        return
    held = ', '.join(available) if available else '(none)'
    raise ValueError(
        f'{masked_url(url)}: the results index has no completed ingest of {", ".join(missing)}. '
        f'It holds: {held}. Run sd_stats_ingest over that root first; until then the '
        f'index cannot say whether an image under it was navigated.'
    )
