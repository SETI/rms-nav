"""What the index already holds about a root, and how rows go back into it.

Reading and writing sit together because they are the same two tables read from
both ends.  A pass opens by asking what it has already paid to read -- an
ingested document and a refused one alike, since neither is worth downloading
again unchanged -- and closes by replacing the rows of the documents it did
read.

An image is written whole or not at all: the delete cascades to the child
tables and runs inside the caller's transaction, so a concurrent worker never
sees half of one image.  A chunk whose write fails is rewritten one image at a
time, which puts every writable document in and identifies the one the database
will not accept, instead of costing the chunk and then the run.

A refusal replaces whatever an earlier pass recorded, in a table of its own.  A
consumer reads absence of an ``images`` row as "this image was never
navigated", so a file with no usable data must leave that answer alone rather
than leaving behind a row nothing backs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sqlalchemy
from pdslogger import PdsLogger

from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest_rows import ImageRows
from spindoctor.results_index import FAILED_FILES, FEATURE_SOURCES, IMAGES, TECHNIQUES

__all__ = ['UNWRITABLE']

UNWRITABLE = 'the database would not accept its rows'
"""Reason counted against a document the database refused to store.

A reason of its own rather than the driver's message, which names the individual
file and the individual value and would therefore tally as one reason per file.
It is also not one of the "not a current-schema navigation document" reasons:
such a document read exactly as the schema says, and only the storage refused
it.
"""


@dataclass(frozen=True)
class _RecordedFile:
    """What the index already holds about one file of a root.

    Parameters:
        mtime_ns: Modification time recorded when it was last read.
        size_bytes: Size recorded when it was last read.
        has_summary_png: Whether a summary PNG was recorded beside it, or None
            for a refused file, which has no image row to carry the flag.
        from_images: Whether the record is an ingested image rather than a
            refused file.
    """

    mtime_ns: int | None
    size_bytes: int | None
    has_summary_png: bool | None
    from_images: bool


def _recorded_files(connection: sqlalchemy.Connection, root_url: str) -> dict[str, _RecordedFile]:
    """What the index already holds about one root's files.

    Both tables are read, because both record a file this ingest has already
    paid to read: an ingested document and a refused one alike are skipped when
    nothing about them has changed.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to read.

    Returns:
        Stub to what is recorded for it.
    """
    images = sqlalchemy.select(
        IMAGES.c.results_path_stub,
        IMAGES.c.mtime_ns,
        IMAGES.c.size_bytes,
        IMAGES.c.has_summary_png,
    ).where(IMAGES.c.root_url == root_url)
    recorded = {
        str(row.results_path_stub): _RecordedFile(
            mtime_ns=row.mtime_ns,
            size_bytes=row.size_bytes,
            has_summary_png=None if row.has_summary_png is None else bool(row.has_summary_png),
            from_images=True,
        )
        for row in connection.execute(images)
    }
    failed = sqlalchemy.select(
        FAILED_FILES.c.results_path_stub, FAILED_FILES.c.mtime_ns, FAILED_FILES.c.size_bytes
    ).where(FAILED_FILES.c.root_url == root_url)
    for row in connection.execute(failed):
        recorded.setdefault(
            str(row.results_path_stub),
            _RecordedFile(
                mtime_ns=row.mtime_ns,
                size_bytes=row.size_bytes,
                has_summary_png=None,
                from_images=False,
            ),
        )
    return recorded


def _write_image(connection: sqlalchemy.Connection, rows: ImageRows) -> None:
    """Replace one image and its child rows.

    The delete cascades to the child tables, so the image is written whole or
    not at all.  It runs inside the caller's transaction, which is what keeps a
    concurrent worker from ever seeing half of one image.  The refusal a
    previous pass may have recorded goes with it, because the file reads now.

    Parameters:
        connection: A connection inside an open transaction.
        rows: The rows to write.
    """
    root_url = rows.image['root_url']
    stub = rows.image['results_path_stub']
    connection.execute(
        IMAGES.delete().where(
            IMAGES.c.root_url == root_url,
            IMAGES.c.results_path_stub == stub,
        )
    )
    connection.execute(IMAGES.insert(), [rows.image])
    if rows.techniques:
        connection.execute(TECHNIQUES.insert(), rows.techniques)
    if rows.feature_sources:
        connection.execute(FEATURE_SOURCES.insert(), rows.feature_sources)
    connection.execute(
        FAILED_FILES.delete().where(
            FAILED_FILES.c.root_url == root_url,
            FAILED_FILES.c.results_path_stub == stub,
        )
    )


def _write_refusal(connection: sqlalchemy.Connection, refusal: dict[str, Any]) -> None:
    """Record one file that could not be read, and drop whatever it used to say.

    A document that ingested on an earlier pass and no longer reads has an
    ``images`` row that no file backs.  It goes, because a consumer applies
    what it finds there; what replaces it is a refusal, in a table no consumer
    reads absence from.

    Parameters:
        connection: A connection inside an open transaction.
        refusal: The ``failed_files`` row to write.
    """
    root_url = refusal['root_url']
    stub = refusal['results_path_stub']
    connection.execute(
        IMAGES.delete().where(
            IMAGES.c.root_url == root_url,
            IMAGES.c.results_path_stub == stub,
        )
    )
    connection.execute(
        FAILED_FILES.delete().where(
            FAILED_FILES.c.root_url == root_url,
            FAILED_FILES.c.results_path_stub == stub,
        )
    )
    connection.execute(FAILED_FILES.insert(), [refusal])


def _connection_was_lost(exc: BaseException) -> bool:
    """Whether a failure says the database went away rather than refusing a row.

    The two need telling apart.  A row the database will not accept is one
    document's problem and the pass goes on without it; a connection that is
    gone would refuse every remaining image the same way, and a pass that
    "completed" without them leaves every consumer reading the absence of their
    rows as "this image was never navigated".

    Parameters:
        exc: The failure to classify.

    Returns:
        True when the driver reported the connection as no longer usable.
    """
    return isinstance(exc, sqlalchemy.exc.DBAPIError) and bool(exc.connection_invalidated)


def _write_chunk(
    engine: sqlalchemy.Engine,
    pending: Sequence[ImageRows],
    refused: Sequence[dict[str, Any]],
    *,
    counts: IngestCounts,
    logger: PdsLogger,
) -> int:
    """Write one chunk's images and refusals, isolating a row the database refuses.

    The chunk is one transaction, which is what bounds the cost of a crash and
    keeps a writer from holding a lock for the length of a run.  A document the
    database will not store -- an identifier too large for its column, a value
    a backend's type will not hold -- would take the whole chunk down with it
    and then the run, leaving the root's ingest unfinished and every consumer
    refusing it.  So a chunk that fails is written again one image at a time,
    which puts every writable document in and identifies the one that is not.

    Parameters:
        engine: The open index.
        pending: The images to write.
        refused: The ``failed_files`` rows to write.
        counts: Accumulator the write failures are added to.
        logger: Logger for the per-file failures.

    Returns:
        How many images were written.

    Raises:
        Exception: Whatever the database raised, if it says the connection is
            gone rather than the row unacceptable.
    """
    try:
        with engine.begin() as connection:
            for rows in pending:
                _write_image(connection, rows)
            for refusal in refused:
                _write_refusal(connection, refusal)
    except Exception as exc:
        if _connection_was_lost(exc):
            raise
        logger.debug('Retrying a chunk one image at a time after %s: %s', type(exc).__name__, exc)
        return _write_separately(engine, pending, refused, counts=counts, logger=logger)
    return len(pending)


def _write_separately(
    engine: sqlalchemy.Engine,
    pending: Sequence[ImageRows],
    refused: Sequence[dict[str, Any]],
    *,
    counts: IngestCounts,
    logger: PdsLogger,
) -> int:
    """Write one chunk's rows in a transaction each, counting what will not go in.

    A write failure is counted but not recorded in ``failed_files``, exactly as
    a retrieval that failed is not: the document read, so nothing about it says
    the next pass will not store it, and a recorded refusal would be skipped for
    as long as the file did not change.

    Parameters:
        engine: The open index.
        pending: The images to write.
        refused: The ``failed_files`` rows to write.
        counts: Accumulator the write failures are added to.
        logger: Logger for the per-file failures.

    Returns:
        How many images were written.

    Raises:
        Exception: Whatever the database raised, if it says the connection is
            gone rather than the row unacceptable.
    """
    written = 0
    for rows in pending:
        source_file = str(rows.image['source_file'])
        try:
            with engine.begin() as connection:
                _write_image(connection, rows)
        except Exception as exc:
            if _connection_was_lost(exc):
                raise
            counts.record_failure(UNWRITABLE, source_file)
            logger.debug('Skipping %s: %s: %s', source_file, type(exc).__name__, exc)
            continue
        written += 1
    for refusal in refused:
        try:
            with engine.begin() as connection:
                _write_refusal(connection, refusal)
        except Exception as exc:
            if _connection_was_lost(exc):
                raise
            logger.debug(
                'Could not record the refusal of %s: %s', refusal['results_path_stub'], exc
            )
    return written
