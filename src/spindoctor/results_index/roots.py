"""Root identity and per-root ingest bookkeeping for the results index.

Every row of the index names the results root it was ingested from, and every
consumer filters on the root it was itself pointed at.  The two only meet if
both spell the root the same way, and one function spells it:
:func:`~spindoctor.nav_records.normalize_root_url`, which is a rule about root
identity rather than about a database and is therefore kept where a reader of
documents can reach it too.  It is re-exported from here, so that a consumer of
the index reads the root and the bookkeeping about it out of one place.

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
means no file was there to read: a pass that could not list a directory stops
rather than finishing, so a run that has a finish time listed the whole root.
"""

from collections.abc import Sequence

import sqlalchemy
from sqlalchemy.engine import Engine

from spindoctor.nav_records import normalize_root_url
from spindoctor.results_index.engine import open_index, reporting_a_failed_read
from spindoctor.results_index.masking import masked_url
from spindoctor.results_index.schema import INGEST_RUNS

__all__ = [
    'RootNotIngestedError',
    'ingested_roots',
    'newest_finish_time',
    'normalize_root_url',
    'open_index_for_roots',
    'require_ingested_roots',
    'unfinished_roots',
]


class RootNotIngestedError(ValueError):
    """A root the index holds no completed ingest of.

    A ``ValueError`` like every other refusal this layer raises, so a caller
    that reports them all alike is unaffected.  It is a type of its own because
    two callers report it differently from the refusals beside it: it names a
    value the operator typed, and a program with a usage convention reports a
    bad value as a usage error rather than as a failed run.
    """


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


def unfinished_roots(connection: sqlalchemy.Connection) -> list[str]:
    """Return every root the index knows of whose newest ingest run did not complete.

    The other half of :func:`ingested_roots` over the same table, and the half a
    consumer needs to be able to say what it left out.  A run that reads only
    the roots with a completed run covers less than the index holds rows for,
    and a narrowing nobody can name is one an operator reads as an answer about
    the whole index.

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
    unfinished = (
        sqlalchemy.select(INGEST_RUNS.c.root_url)
        .join(newest, INGEST_RUNS.c.run_id == newest.c.run_id)
        .where(INGEST_RUNS.c.finished_utc.is_(None))
        .order_by(INGEST_RUNS.c.root_url)
    )
    return [str(row.root_url) for row in connection.execute(unfinished)]


def newest_finish_time(connection: sqlalchemy.Connection, root_url: str) -> str | None:
    """Return when the newest pass over one root finished.

    The index answers as of that moment and detects no change since, so a
    consumer reports the moment with the answer rather than leaving it in the
    head of whoever exported the URL.

    It comes from the newest run row of the named root alone.  One database
    serves several roots, and the newest run in the table is routinely another
    root's.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to ask about.

    Returns:
        The finish time that pass stamped, and None when the root has no run
        row at all or its newest run never finished.
    """
    newest = (
        sqlalchemy.select(INGEST_RUNS.c.finished_utc)
        .where(INGEST_RUNS.c.root_url == root_url)
        .order_by(INGEST_RUNS.c.run_id.desc())
        .limit(1)
    )
    row = connection.execute(newest).first()
    if row is None or row.finished_utc is None:
        return None
    return str(row.finished_utc)


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
        RootNotIngestedError: If any named root has no completed ingest run,
            naming the roots that are missing and the roots the index does hold.
            Under such a root, absence of a row must never be read as "nothing
            was navigated".  Under an ingested one it may be, but only once the
            refusal table has been asked too: a completed run makes absence from
            both tables meaningful, not absence from ``images`` alone.
    """
    available = ingested_roots(connection)
    missing = [root for root in roots if root not in available]
    if not missing:
        return
    held = ', '.join(available) if available else '(none)'
    raise RootNotIngestedError(
        f'{masked_url(url)}: the results index has no completed ingest of {", ".join(missing)}. '
        f'It holds: {held}. Run sd_stats_ingest over that root first; until then the '
        f'index cannot say whether an image under it was navigated.'
    )


def open_index_for_roots(url: str, roots: Sequence[str]) -> Engine:
    """Open an index and refuse it if a root the caller means to read is not in it.

    Every consumer that reads rows under a named root opens the index this way,
    because every one of them needs the same two things to be true before it
    reads anything and needs them in the same order: the URL names an index this
    build can read, and the roots it is about to ask about have been ingested
    into it.  Written out at each call site, the sequence lost a step at a time
    -- an engine left undisposed on the refusal, a root checked after the first
    query rather than before it.

    Nothing here falls back to the documents.  A URL that cannot be opened, or a
    root nobody has ingested, fails the run: turning either into a slow read of
    the tree would make a misconfigured run a silently different one.  Nor does
    anything here create an index: a consumer that created one would answer every
    question with "nothing was navigated", so creating is the writer's alone and
    the writer requires no root to be ingested.

    Parameters:
        url: Connection URL of the results index.
        roots: The normalized roots the caller means to read.  Empty means the
            caller reads whatever the index holds and names no root of its own,
            which the report does.

    Returns:
        The open index, which the caller closes when it is done with it.

    Raises:
        RootNotIngestedError: If any named root has no completed ingest run in
            the index.  A caller that reports a value the operator typed
            differently from a run that failed catches this one first; the rest
            catch the ``ValueError`` it is a kind of.
        ValueError: If the URL cannot be opened, does not name an index, or names
            one written by another version of the schema, or if the index cannot
            be read.  The engine is disposed of before any refusal leaves here,
            so a refused caller has nothing to close.
    """
    engine = open_index(url, create=False)
    try:
        with reporting_a_failed_read(url), engine.connect() as connection:
            require_ingested_roots(connection, list(roots), url=url)
    except Exception:
        engine.dispose()
        raise
    return engine
