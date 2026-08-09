"""One ingest pass over each of the named results roots.

Each root is walked once; the files the index has already read are dropped from
what the pass reads; the rest are ingested in chunks; and a root that was
listed completely is pruned of the rows whose documents have left the tree.

What a row is keyed by
----------------------

``(root_url, results_path_stub)``.  The stub is derived from the document's own
location -- its path under the ingest root with ``_metadata.json`` removed --
and never from anything inside the document, because that is precisely how the
navigator chose the path it wrote to.  The root is the ``nav_results_root``,
normalized once by :func:`~spindoctor.results_index.normalize_root_url` so that
a consumer's spelling of the same root matches.  Ingesting a subdirectory of a
results root would produce stubs no consumer's lookup can match, so the root
identity is part of the contract rather than a convenience.

What is read again
------------------

A file whose recorded ``(mtime_ns, size_bytes)`` still matches the listing, and
whose summary PNG is as the last pass recorded it, is not read at all.  A
backend whose listing supplies neither metric cannot answer that question, so
such a root is re-read in full, with a warning saying so.

What leaving the tree costs
---------------------------

Presence has to mean what absence means.  A document deleted from the tree
leaves a row that would answer for an image the tree no longer holds, so the
rows of one root whose stub the walk did not find are deleted with it.  That is
sound only for a pass that listed the whole root: a worker handed a share of a
root has no evidence about the stubs outside its share, and deleting on that
evidence would delete its peers' work.  The prune therefore reads a complete
listing and refuses anything else.
"""

from collections.abc import Sequence

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.stats.ingest.chunks import _batched, _ingest_chunk
from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest.runs import _finish_run, _start_run
from spindoctor.cli.stats.ingest.store import _recorded_files, _RecordedFile
from spindoctor.cli.stats.ingest.walk import _ListedFile, _RootListing, _walk_root
from spindoctor.results_index import FAILED_FILES, IMAGES, normalize_root_url

__all__ = ['INGEST_COMMIT_CHUNK_SIZE', 'distinct_roots', 'ingest_metadata_files']

INGEST_COMMIT_CHUNK_SIZE = 512
"""How many images are written per database transaction.

Independent of the retrieval batch size: one bounds a download, the other
bounds how much work a crash costs and how long a writer holds its lock.  An
image's own rows are always written inside one transaction, so a concurrent
worker never sees half of an image.
"""


def distinct_roots(roots: Sequence[str]) -> list[str]:
    """Normalize the given roots and drop the repeats, keeping their order.

    ``/data/x`` and ``/data/x/`` are one root, and a command line naming both
    means the tree once.  Walking it twice reads every document twice and gives
    one root two ingest runs; in a pass divided into cloud tasks it also hands
    every document out in two shares, leaves the first of the two runs
    unfinished forever, and -- since a completion stamps the newest run and then
    finds nothing outstanding -- tells the operator that a root it has just
    finished was never divided up.

    Every mode of the pass reads the roots through this, and so does the driver
    that reports which roots it was given: a run that named a root two ways and
    then reported on it once reads as a root having gone missing.

    Parameters:
        roots: The roots as their holder spelled them.

    Returns:
        The normalized roots, first spelling first.

    Raises:
        ValueError: If a root is not a location: one the storage layer refuses
            to render absolute, one carrying a null byte, or an empty spelling,
            which is the working directory rather than a root anybody named.
    """
    distinct: dict[str, None] = {}
    for root in roots:
        distinct.setdefault(normalize_root_url(root), None)
    return list(distinct)


def _is_unchanged(
    listed: _ListedFile, recorded: _RecordedFile | None, summary_stubs: set[str]
) -> bool:
    """Whether a listed file is exactly what the index already read.

    The summary PNG is part of the comparison because ``has_summary_png`` is a
    column of the row and comes from the walk rather than from the document: a
    summary written after the document was ingested changes the row that ought
    to be stored, while changing nothing about the document itself.

    Parameters:
        listed: The file as this walk saw it.
        recorded: What the index holds about it, or None when it holds nothing.
        summary_stubs: Stubs this walk saw a summary PNG for.

    Returns:
        True when the file need not be read again.
    """
    if recorded is None:
        return False
    if (recorded.mtime_ns, recorded.size_bytes) != (listed.mtime_ns, listed.size_bytes):
        return False
    if recorded.from_images:
        return recorded.has_summary_png == (listed.results_path_stub in summary_stubs)
    return True


def _files_to_read(
    files: Sequence[_ListedFile],
    summary_stubs: set[str],
    recorded: dict[str, _RecordedFile],
    *,
    force: bool,
    has_file_metrics: bool,
) -> list[_ListedFile]:
    """Select the metadata files this pass has to read.

    A file the last pass refused is skipped on the same evidence as one it
    ingested: it has not changed, so reading it produces the same refusal.
    ``force`` re-reads both.

    Written over the files rather than over a whole-root listing, because a pass
    over a share of a root selects from its share by exactly this rule and must
    not be able to reach for anything a complete listing would have carried.

    Parameters:
        files: The metadata files this pass is responsible for.
        summary_stubs: Stubs the walk saw a summary PNG for.
        recorded: Stub to what the index already holds about it.
        force: Whether to re-read every document regardless.
        has_file_metrics: Whether the listing reported a size and modification
            time for every one of those files.  A listing that reports neither
            cannot answer "has this changed", so all of them are read.

    Returns:
        The files to read, in the order given.
    """
    if force or not has_file_metrics:
        return list(files)
    return [
        listed
        for listed in files
        if not _is_unchanged(listed, recorded.get(listed.results_path_stub), summary_stubs)
    ]


def _prune_missing(
    engine: sqlalchemy.Engine,
    root_url: str,
    listing: _RootListing,
    recorded: dict[str, _RecordedFile],
    *,
    logger: PdsLogger,
) -> int:
    """Delete the rows of one root whose document has left the tree.

    Absence of an ``images`` row is what a consumer reads as "this image was
    never navigated", so presence has to mean the tree still holds the result.
    A re-navigation that renames or removes documents otherwise leaves rows
    that answer confidently for images nothing produced.

    Parameters:
        engine: The open index.
        root_url: Normalized URL of the root being ingested.
        listing: The walk this prune is entitled to act on.
        recorded: What the index held about the root before this pass.
        logger: Logger for the count removed.

    Returns:
        How many image rows were deleted.

    Raises:
        ValueError: If the listing covers part of a root rather than all of it.
            A pass over a share of a root knows nothing about the stubs outside
            its share, and would delete another worker's rows on that evidence.
    """
    if not listing.covers_whole_root:
        raise ValueError(
            f'{root_url}: rows may only be removed on the evidence of a complete listing '
            f'of the root, and this walk did not produce one'
        )
    found = {listed.results_path_stub for listed in listing.metadata_files}
    gone = sorted(stub for stub in recorded if stub not in found)
    if not gone:
        return 0
    removed = sum(1 for stub in gone if recorded[stub].from_images)
    logger.info('Removing %d row(s) under %s whose document has left the tree', removed, root_url)
    for batch in _batched(gone, INGEST_COMMIT_CHUNK_SIZE):
        with engine.begin() as connection:
            connection.execute(
                IMAGES.delete().where(
                    IMAGES.c.root_url == root_url,
                    IMAGES.c.results_path_stub.in_(batch),
                )
            )
            connection.execute(
                FAILED_FILES.delete().where(
                    FAILED_FILES.c.root_url == root_url,
                    FAILED_FILES.c.results_path_stub.in_(batch),
                )
            )
    return removed


def ingest_metadata_files(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool = False,
    logger: PdsLogger,
) -> IngestCounts:
    """Ingest every metadata document under the given results roots.

    Each root is walked once, and each file whose recorded size and
    modification time still match the walk is skipped without being read,
    whether the last pass ingested it or refused it.  A document that cannot be
    read as a current-schema navigation document is counted against its own
    file and the run continues.

    A root this walk lists completely is also pruned: the rows of documents the
    tree no longer holds are deleted, so that presence of a row means what
    absence of one means.  A root the walk could not list is left alone
    entirely, and its ingest run is deliberately not completed, because a
    mistyped or unmounted root is not an empty one.

    Two spellings of one root are one root, and are walked once.

    Parameters:
        engine: The open index, which must already carry the schema.
        roots: Navigation results roots -- local directories or any URL the
            ``filecache`` layer accepts.  Each is normalized to the form the
            rows record and consumers compare against.
        force: Re-read every document, ignoring the recorded file metrics.
        logger: Logger for the per-root scan summary and per-file failures.

    Returns:
        What the pass did, summed over every root.
    """
    total = IngestCounts()
    # The normalized form is what is walked, not the string as typed.  It is
    # the same location, absolute and spelled once: walking the typed form
    # would record a relative source_file beside an absolute root_url, and a
    # relative local root is one the storage layer refuses outright.  Two
    # spellings of one root are one root here as they are at a fan-out, so the
    # same command line means the same thing in every mode.
    for root_url in distinct_roots(roots):
        root = FCPath(root_url)
        counts = IngestCounts()
        run_id = _start_run(engine, root_url)
        logger.info('Ingesting %s', root_url)
        listing = _walk_root(root, logger=logger)
        counts.files_seen = len(listing.metadata_files)
        counts.directories_missed = listing.directories_missed
        if not listing.root_listed:
            # The run row keeps its NULL finish time, so every consumer treats
            # this root as one nobody has ingested rather than as one that
            # holds nothing.
            counts.roots_unreadable = 1
            total.add(counts)
            continue
        with engine.connect() as connection:
            recorded = _recorded_files(connection, root_url)
        to_read = _files_to_read(
            listing.metadata_files,
            listing.summary_stubs,
            recorded,
            force=force,
            has_file_metrics=listing.has_file_metrics,
        )
        counts.files_skipped = counts.files_seen - len(to_read)
        for chunk in _batched(to_read, INGEST_COMMIT_CHUNK_SIZE):
            _ingest_chunk(
                engine,
                root,
                chunk,
                root_url=root_url,
                summary_stubs=listing.summary_stubs,
                counts=counts,
                logger=logger,
            )
        if listing.covers_whole_root:
            counts.files_removed = _prune_missing(
                engine, root_url, listing, recorded, logger=logger
            )
        _finish_run(engine, run_id, counts)
        logger.info(
            'Ingested %d, skipped %d unchanged, failed %d, removed %d of %d file(s) under %s',
            counts.files_ingested,
            counts.files_skipped,
            counts.files_failed,
            counts.files_removed,
            counts.files_seen,
            root_url,
        )
        total.add(counts)
    return total
