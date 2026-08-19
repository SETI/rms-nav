"""What the index already holds about a root, and how rows go back into it.

Reading and writing sit together because they are the same two tables read from
both ends.  A pass opens by asking what it has already paid to read -- an
ingested document and a refused one alike, since neither is worth downloading
again unchanged -- and closes by replacing the rows of the documents it did
read.

An image is written whole or not at all: the delete cascades to the child
tables and runs inside the caller's transaction, so a concurrent worker never
sees half of one image.  A chunk whose write fails is rewritten one image at a
time, which is what names the document the database would not take.

A row the database will not accept ends the pass.  It is not a property of the
file: the document read exactly as the schema says, and what refused it is this
code's own writer or the column set it writes into, either of which will refuse
the next document of the same shape the same way.  Charged to the file it would
be counted and left out of both tables, and the pass would stamp its run
finished -- after which absence of an ``images`` row reads as "this image was
never navigated", which is the one answer the index exists to give and would
now be wrong for an image nobody can name.  So the pass stops where it happened,
the root's run keeps its NULL finish time, and every consumer says the root has
no completed ingest until the writer is fixed and the pass rerun.

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

from spindoctor.nav_records.facts import NOT_A_NAVIGATION_DOCUMENT, ImageFacts
from spindoctor.results_index import FAILED_FILES, FEATURE_SOURCES, IMAGES, TECHNIQUES

__all__ = ['UnwritableRowError']


class UnwritableRowError(RuntimeError):
    """The database would not accept the rows of a document that read cleanly.

    Carries the file and the driver's own sentence, because what has to be
    fixed is the writer or the column set rather than the file: a value one
    backend holds and another refuses is a property of the schema, and the next
    document of the same shape fails the same way.

    Ends the pass rather than being charged to the file.  A file counted and
    left out is in neither table, and a finished run over an index missing it
    turns absence of an ``images`` row -- which every consumer reads as "this
    image was never navigated" -- into an answer nobody can tell from the truth.
    """


_RECORDED_LOOKUP_BATCH_SIZE = 500
"""How many stubs one restricted lookup names at a time.

Each stub is a bind parameter, and every backend limits how many one statement
may carry.  A pass over a whole root names none of them and is unaffected; a
pass over a share of one names its own, and the share is whatever the caller
divided the root into.
"""


@dataclass(frozen=True)
class _RecordedFile:
    """What the index already holds about one file of a root.

    Parameters:
        mtime_ns: Modification time recorded when it was last read.
        size_bytes: Size recorded when it was last read.
        from_images: Whether the record is an ingested image rather than a
            refused file.
    """

    mtime_ns: int | None
    size_bytes: int | None
    from_images: bool


def _stub_restrictions(
    column: sqlalchemy.Column[Any], stubs: Sequence[str] | None
) -> list[sqlalchemy.ColumnElement[bool]]:
    """Return the extra WHERE terms that narrow a lookup to named stubs.

    Parameters:
        column: The table's stub column to restrict.
        stubs: The stubs to ask about, or None to ask about the whole root.

    Returns:
        One term per batch of stubs, or a single term matching nothing when an
        empty sequence was given.  A caller that names no stub is asking about
        no file, which is not the same question as asking about all of them.
    """
    if stubs is None:
        return [sqlalchemy.true()]
    if len(stubs) == 0:
        return [sqlalchemy.false()]
    return [
        column.in_(stubs[start : start + _RECORDED_LOOKUP_BATCH_SIZE])
        for start in range(0, len(stubs), _RECORDED_LOOKUP_BATCH_SIZE)
    ]


def _recorded_files(
    connection: sqlalchemy.Connection, root_url: str, *, stubs: Sequence[str] | None = None
) -> dict[str, _RecordedFile]:
    """What the index already holds about one root's files.

    Both tables are read, because both record a file this ingest has already
    paid to read: an ingested document and a refused one alike are skipped when
    nothing about them has changed.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to read.
        stubs: Ask only about these stubs.  A pass over a share of a root reads
            what is recorded for its own files rather than for the whole root,
            which on an archive-scale root is the difference between one lookup
            and one lookup per worker over every row in it.  None asks about
            every file of the root.

    Returns:
        Stub to what is recorded for it.
    """
    recorded: dict[str, _RecordedFile] = {}
    for restriction in _stub_restrictions(IMAGES.c.results_path_stub, stubs):
        images = sqlalchemy.select(
            IMAGES.c.results_path_stub,
            IMAGES.c.mtime_ns,
            IMAGES.c.size_bytes,
        ).where(IMAGES.c.root_url == root_url, restriction)
        for row in connection.execute(images):
            recorded[str(row.results_path_stub)] = _RecordedFile(
                mtime_ns=row.mtime_ns,
                size_bytes=row.size_bytes,
                from_images=True,
            )
    for restriction in _stub_restrictions(FAILED_FILES.c.results_path_stub, stubs):
        failed = sqlalchemy.select(
            FAILED_FILES.c.results_path_stub,
            FAILED_FILES.c.mtime_ns,
            FAILED_FILES.c.size_bytes,
        ).where(FAILED_FILES.c.root_url == root_url, restriction)
        for row in connection.execute(failed):
            recorded.setdefault(
                str(row.results_path_stub),
                _RecordedFile(
                    mtime_ns=row.mtime_ns,
                    size_bytes=row.size_bytes,
                    from_images=False,
                ),
            )
    return recorded


def _refusals_the_tree_answers_for(connection: sqlalchemy.Connection, root_url: str) -> int:
    """How many of one root's refusals an error filter reads out of the tree.

    Three terms, because a row of ``failed_files`` is only a divergence when all
    three hold:

    - The row is this root's, since one index serves several roots.
    - Its reason is the schema family
      (:data:`~spindoctor.nav_records.facts.NOT_A_NAVIGATION_DOCUMENT`),
      which is a JSON object the tree reads a ``status`` out of and this index
      records no status for.  The other reasons are a file no JSON object came
      out of, and the tree excludes such a file from every error filter exactly
      as this index answers none for it, so the two agree and the row is not a
      gap.
    - It carries a subtree, because a selection asks about the subtrees it
      enumerated and :func:`~spindoctor.results_index.selection._stub_query`
      restricts both arms by ``IN`` over them, which is false for NULL.  A
      refusal under no subtree is in no selection's answer either way.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to count under.

    Returns:
        The rows meeting all three, whichever pass wrote them.  That is a
        different number from what any one pass refused: an unchanged file is
        skipped rather than read, so a pass after the one that refused it
        refuses nothing and tallies nothing.
    """
    total = (
        sqlalchemy.select(sqlalchemy.func.count())
        .select_from(FAILED_FILES)
        .where(
            FAILED_FILES.c.root_url == root_url,
            FAILED_FILES.c.reason.startswith(NOT_A_NAVIGATION_DOCUMENT, autoescape=True),
            FAILED_FILES.c.subtree.is_not(None),
        )
    )
    return int(connection.execute(total).scalar_one())


def _report_refusals(engine: sqlalchemy.Engine, root_url: str, *, logger: PdsLogger) -> None:
    """Report how many of one root's documents the tree answers for and this does not.

    Said at the end of every pass over a root, and said as the root's own total
    rather than as this pass's, because the pass's tally is zero on every pass
    after the one that read the file.  It is otherwise a number an operator can
    reach only by querying ``failed_files``.

    What it counts is the refusals a selection can actually be short by, not
    every refusal: a file no JSON object came out of is one the tree excludes
    from every error filter too, and counting it would report a gap where the
    two agree.  It is still the whole root's count, and a selection enumerates
    subtrees, so it bounds one selection's shortfall rather than measuring it.

    The report is informational, so a failure of it costs the report and nothing
    else.  It runs after the root's rows are written and its run is stamped, and
    a database that went away between the two would otherwise take with it the
    remaining roots of the run and the counts of the one just finished.

    Parameters:
        engine: The open index.
        root_url: The normalized root the pass covered.
        logger: Logger for the count.
    """
    try:
        with engine.connect() as connection:
            refused = _refusals_the_tree_answers_for(connection, root_url)
    except sqlalchemy.exc.SQLAlchemyError as exc:
        # The driver's own sentence rather than the exception's rendering, which
        # wraps it in the statement, the bound parameters and a documentation
        # link -- several lines of machinery around the one that says what
        # happened.  Compared against None rather than taken for its truth:
        # str(None) is 'None', which would read as a driver that said so.
        driver_error = getattr(exc, 'orig', None)
        detail = (str(driver_error).strip() if driver_error is not None else '') or str(exc)
        logger.warning(
            'Could not count the refused documents under %s (%s: %s). The pass itself is '
            'unaffected; query failed_files for the count.',
            root_url,
            type(exc).__name__,
            detail,
        )
        return
    logger.info(
        'Documents under %s an error filter reads from the results tree and not from this '
        'index: %d, whichever pass recorded them. Each is a JSON object the ingest refused, '
        'so this index records no status for it and no error filter answered here selects '
        'its image. The count is the whole root, so it bounds rather than measures how '
        'short a selection over some of its subtrees comes.',
        root_url,
        refused,
    )


def _write_image(connection: sqlalchemy.Connection, rows: ImageFacts) -> None:
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
    pending: Sequence[ImageFacts],
    refused: Sequence[dict[str, Any]],
    *,
    logger: PdsLogger,
) -> int:
    """Write one chunk's images and refusals, naming a row the database refuses.

    The chunk is one transaction, which is what bounds the cost of a crash and
    keeps a writer from holding a lock for the length of a run.  A chunk that
    fails says only that one of its documents would not go in, so it is written
    again one image at a time: every writable document of it is stored, and the
    one that is not is named.

    Parameters:
        engine: The open index.
        pending: The images to write.
        refused: The ``failed_files`` rows to write.
        logger: Logger for the per-file failures.

    Returns:
        How many images were written.

    Raises:
        UnwritableRowError: If the database will not accept some document's
            rows, naming it.
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
        return _write_separately(engine, pending, refused)
    return len(pending)


def _write_separately(
    engine: sqlalchemy.Engine, pending: Sequence[ImageFacts], refused: Sequence[dict[str, Any]]
) -> int:
    """Write one chunk's rows in a transaction each, naming what will not go in.

    Every document that goes in is committed on its own, so the work already
    paid for is kept and a rerun after the fix reads only what is left.  The
    first one that will not go in ends the pass: it is a fault of the writer or
    of the column set, so the documents after it in this chunk would be written
    against the same fault, and a pass that carried on would finish having left
    an unknown number of images out of both tables.

    Parameters:
        engine: The open index.
        pending: The images to write.
        refused: The ``failed_files`` rows to write.

    Returns:
        How many images were written.

    Raises:
        UnwritableRowError: If the database will not accept some document's
            rows, naming it and what the driver said.
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
            raise UnwritableRowError(
                f'{source_file}: the database would not accept its rows '
                f'({type(exc).__name__}: {exc})'
            ) from exc
        written += 1
    for refusal in refused:
        try:
            with engine.begin() as connection:
                _write_refusal(connection, refusal)
        except Exception as exc:
            if _connection_was_lost(exc):
                raise
            raise UnwritableRowError(
                f'{refusal["results_path_stub"]}: the database would not accept the record of '
                f'its refusal ({type(exc).__name__}: {exc})'
            ) from exc
    return written
