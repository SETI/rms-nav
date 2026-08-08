"""Read a navigation results tree into the results index.

One pass over a results root reads every ``*_metadata.json`` document under it
and writes one ``images`` row per document, with its per-technique and feature
inventory rows beside it.  Nothing else reads the tree afterwards: a consumer
that needs a few fields per image reads a row.

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

One walk feeds everything
-------------------------

The recursive listing collects both suffixes in a single pass and carries each
entry's size and modification time with it, so the summary-PNG flag and the two
file metrics all come from the walk.  There is no per-file stat and no per-file
existence check: on a cloud root each of those is a paid round trip per image
per run, which is the cost this index exists to remove.

A file whose recorded ``(mtime_ns, size_bytes)`` still matches the listing is
not read at all.  A backend whose listing supplies neither metric cannot answer
that question, so such a root is re-read in full, with a warning saying so.

What a failed document costs
----------------------------

A results tree holds ``*_metadata.json`` files that are not per-image navigation
documents -- products of other tools, and documents written by an older
metadata schema.  Each is counted as an error for its own file and no more: the
run continues, and the closing summary tallies the failures by reason, so
several hundred documents that were never navigation results read as exactly
that rather than as a broken ingest.
"""

import datetime
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.stats.ingest_rows import (
    ImageRows,
    MetadataDocumentError,
    MetadataSource,
    rows_from_metadata,
)
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    INGEST_RUNS,
    SCHEMA_VERSION,
    TECHNIQUES,
    normalize_root_url,
)

__all__ = [
    'INGEST_COMMIT_CHUNK_SIZE',
    'INGEST_RETRIEVE_BATCH_SIZE',
    'METADATA_SUFFIX',
    'SUMMARY_PNG_SUFFIX',
    'IngestCounts',
    'ingest_metadata_files',
]

METADATA_SUFFIX = '_metadata.json'
"""Suffix of the per-image navigation document under the results root."""

SUMMARY_PNG_SUFFIX = '_summary.png'
"""Suffix of the per-image summary PNG written beside the document."""

INGEST_RETRIEVE_BATCH_SIZE = 64
"""How many metadata files are retrieved in one batched download.

A cloud backend downloads a batch in parallel, so the batch size trades peak
memory and per-request concurrency against the number of round trips.
"""

INGEST_COMMIT_CHUNK_SIZE = 512
"""How many images are written per database transaction.

Independent of the retrieval batch size: one bounds a download, the other
bounds how much work a crash costs and how long a writer holds its lock.  An
image's own rows are always written inside one transaction, so a concurrent
worker never sees half of an image.
"""


@dataclass
class IngestCounts:
    """What one ingest pass did.

    Parameters:
        files_seen: Metadata files the walk found.
        files_ingested: Documents read and written as rows.
        files_skipped: Documents whose recorded size and modification time
            still matched the listing, so they were never read.
        files_failed: Files that are not current-schema navigation documents.
        failures_by_reason: How many files failed for each distinct reason, so
            a tree full of documents that were never navigation results reads
            as that rather than as an ingest that went wrong.
    """

    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    failures_by_reason: dict[str, int] = field(default_factory=dict)

    def add(self, other: 'IngestCounts') -> None:
        """Fold another pass's counts into this one.

        Parameters:
            other: The counts to add.
        """
        self.files_seen += other.files_seen
        self.files_ingested += other.files_ingested
        self.files_skipped += other.files_skipped
        self.files_failed += other.files_failed
        for reason, count in other.failures_by_reason.items():
            self.failures_by_reason[reason] = self.failures_by_reason.get(reason, 0) + count

    def record_failure(self, reason: str) -> None:
        """Count one file that could not be ingested.

        Parameters:
            reason: What was wrong with it, with nothing file-specific in it.
        """
        self.files_failed += 1
        self.failures_by_reason[reason] = self.failures_by_reason.get(reason, 0) + 1


@dataclass(frozen=True)
class _ListedFile:
    """One metadata file the walk found, with the metrics it reported.

    Parameters:
        results_path_stub: Path under the root with the suffix removed.
        mtime_ns: Modification time in nanoseconds, or None when unreported.
        size_bytes: Size in bytes, or None when unreported.
    """

    results_path_stub: str
    mtime_ns: int | None
    size_bytes: int | None


@dataclass
class _RootListing:
    """Everything one walk of a results root found.

    Parameters:
        metadata_files: The metadata files, in stub order.
        summary_stubs: Stubs that also have a summary PNG.
        has_file_metrics: Whether every metadata file reported both a size and
            a modification time.  A listing that reports neither cannot answer
            "has this changed", so such a root is re-read in full.
    """

    metadata_files: list[_ListedFile] = field(default_factory=list)
    summary_stubs: set[str] = field(default_factory=set)
    has_file_metrics: bool = True


# ---------------------------------------------------------------------------
# One walk of a results root
# ---------------------------------------------------------------------------


def _metrics_of(entry_metadata: dict[str, Any] | None) -> tuple[int | None, int | None]:
    """The modification time and size a listing entry reports.

    Parameters:
        entry_metadata: The listing entry's metadata, or None when the backend
            reported none.

    Returns:
        ``(mtime_ns, size_bytes)``, each None when unreported.  The time is
        converted from the seconds a listing reports; the conversion is exact
        enough for its only purpose, which is noticing that a file changed.
    """
    if entry_metadata is None:
        return None, None
    mtime = entry_metadata.get('mtime')
    size = entry_metadata.get('size')
    mtime_ns = None if mtime is None else round(float(mtime) * 1_000_000_000)
    size_bytes = None if size is None else int(size)
    return mtime_ns, size_bytes


def _list_directory(directory: FCPath, prefix: str, listing: _RootListing) -> None:
    """Collect one directory's result files and descend into its subdirectories.

    Parameters:
        directory: The directory to list.
        prefix: The path of that directory under the root, ending in ``/`` (or
            empty at the root itself).
        listing: Accumulator the walk fills in.
    """
    try:
        entries = list(directory.iterdir_metadata())
    except (FileNotFoundError, NotADirectoryError):
        # A directory that is not there, or that stopped being a directory
        # between the parent listing and this call, holds no result files.
        return
    for path, entry_metadata in entries:
        name = path.name
        relative = f'{prefix}{name}'
        is_dir = entry_metadata['is_dir'] if entry_metadata is not None else path.is_dir()
        if is_dir:
            _list_directory(path, f'{relative}/', listing)
        elif name.endswith(METADATA_SUFFIX):
            mtime_ns, size_bytes = _metrics_of(entry_metadata)
            if mtime_ns is None or size_bytes is None:
                listing.has_file_metrics = False
            listing.metadata_files.append(
                _ListedFile(
                    results_path_stub=relative[: -len(METADATA_SUFFIX)],
                    mtime_ns=mtime_ns,
                    size_bytes=size_bytes,
                )
            )
        elif name.endswith(SUMMARY_PNG_SUFFIX):
            listing.summary_stubs.add(relative[: -len(SUMMARY_PNG_SUFFIX)])


def _walk_root(root: FCPath, *, logger: PdsLogger) -> _RootListing:
    """Walk one results root once, collecting both result-file suffixes.

    Parameters:
        root: The results root.
        logger: Logger for the scan summary and the degraded-listing warning.

    Returns:
        What the walk found, with the metadata files in stub order.
    """
    listing = _RootListing()
    _list_directory(root, '', listing)
    listing.metadata_files.sort(key=lambda listed: listed.results_path_stub)
    logger.info(
        'Results scan found %d metadata and %d summary PNG file(s) under %s',
        len(listing.metadata_files),
        len(listing.summary_stubs),
        root.as_posix(),
    )
    if not listing.has_file_metrics and listing.metadata_files:
        logger.warning(
            'Listing of %s reports no size or modification time, so every document '
            'is re-read: this root cannot be ingested incrementally',
            root.as_posix(),
        )
    return listing


# ---------------------------------------------------------------------------
# Writing the rows
# ---------------------------------------------------------------------------


def _recorded_metrics(
    connection: sqlalchemy.Connection, root_url: str
) -> dict[str, tuple[int | None, int | None]]:
    """The size and modification time already recorded for one root's images.

    Parameters:
        connection: An open connection to the index.
        root_url: The normalized root to read.

    Returns:
        Stub to ``(mtime_ns, size_bytes)`` for every row of that root.
    """
    selected = sqlalchemy.select(
        IMAGES.c.results_path_stub, IMAGES.c.mtime_ns, IMAGES.c.size_bytes
    ).where(IMAGES.c.root_url == root_url)
    return {
        str(row.results_path_stub): (row.mtime_ns, row.size_bytes)
        for row in connection.execute(selected)
    }


def _write_image(connection: sqlalchemy.Connection, rows: ImageRows) -> None:
    """Replace one image and its child rows.

    The delete cascades to the child tables, so the image is written whole or
    not at all.  It runs inside the caller's transaction, which is what keeps a
    concurrent worker from ever seeing half of one image.

    Parameters:
        connection: A connection inside an open transaction.
        rows: The rows to write.
    """
    connection.execute(
        IMAGES.delete().where(
            IMAGES.c.root_url == rows.image['root_url'],
            IMAGES.c.results_path_stub == rows.image['results_path_stub'],
        )
    )
    connection.execute(IMAGES.insert(), [rows.image])
    if rows.techniques:
        connection.execute(TECHNIQUES.insert(), rows.techniques)
    if rows.feature_sources:
        connection.execute(FEATURE_SOURCES.insert(), rows.feature_sources)


def _batched(items: Sequence[_ListedFile], size: int) -> Iterator[Sequence[_ListedFile]]:
    """Yield consecutive slices of a sequence.

    Parameters:
        items: The sequence to slice.
        size: Maximum slice length.

    Yields:
        The slices, in order.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_document(local_path: Path, source: MetadataSource) -> ImageRows:
    """Read one retrieved metadata file into rows.

    Parameters:
        local_path: Local path the retrieval produced.
        source: Where the document came from.

    Returns:
        The rows the document becomes.

    Raises:
        MetadataDocumentError: If the file cannot be read, does not parse as
            JSON, does not parse to a JSON object, or is not a current-schema
            navigation document.
    """
    try:
        text = local_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise MetadataDocumentError('unreadable', source_file=source.source_file) from exc
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetadataDocumentError('not valid JSON', source_file=source.source_file) from exc
    if not isinstance(parsed, dict):
        raise MetadataDocumentError('not a JSON object', source_file=source.source_file)
    return rows_from_metadata(parsed, source)


def _ingest_chunk(
    engine: sqlalchemy.Engine,
    root: FCPath,
    chunk: Sequence[_ListedFile],
    *,
    root_url: str,
    summary_stubs: set[str],
    counts: IngestCounts,
    logger: PdsLogger,
) -> None:
    """Retrieve, read and write one chunk of metadata files.

    The whole chunk is one transaction, so a crash costs one chunk rather than
    a whole archive-scale run, and no writer holds a lock for the length of one.

    Parameters:
        engine: The open index.
        root: The results root, which the retrieval is relative to.
        chunk: The files to ingest.
        root_url: Normalized URL of the root, as the rows record it.
        summary_stubs: Stubs the walk saw a summary PNG for.
        counts: Accumulator this chunk's outcomes are added to.
        logger: Logger for per-file failures.
    """
    pending: list[ImageRows] = []
    for batch in _batched(chunk, INGEST_RETRIEVE_BATCH_SIZE):
        sub_paths: list[str | Path] = [
            f'{listed.results_path_stub}{METADATA_SUFFIX}' for listed in batch
        ]
        # retrieve() rather than get_local_path(): on a cloud root the latter
        # names a file it never downloads.  exception_on_fail=False keeps one
        # unreadable file from ending the run.
        local_paths = cast(
            list[Path | Exception], root.retrieve(sub_paths, exception_on_fail=False)
        )
        for listed, local_path in zip(batch, local_paths, strict=True):
            source = MetadataSource(
                root_url=root_url,
                results_path_stub=listed.results_path_stub,
                source_file=(root / f'{listed.results_path_stub}{METADATA_SUFFIX}').as_posix(),
                mtime_ns=listed.mtime_ns,
                size_bytes=listed.size_bytes,
                has_summary_png=listed.results_path_stub in summary_stubs,
            )
            if isinstance(local_path, BaseException):
                counts.record_failure('could not be retrieved')
                logger.debug('Skipping %s: could not be retrieved', source.source_file)
                continue
            try:
                pending.append(_read_document(local_path, source))
            except MetadataDocumentError as exc:
                counts.record_failure(exc.reason)
                logger.debug('Skipping %s', exc)
    if not pending:
        return
    with engine.begin() as connection:
        for rows in pending:
            _write_image(connection, rows)
    counts.files_ingested += len(pending)


def _start_run(engine: sqlalchemy.Engine, root_url: str) -> int:
    """Record that an ingest of one root has begun.

    The row is committed before the walk, so a consumer that looks while the
    run is in flight sees a root whose newest run has not finished and refuses
    to read absence from it.

    Parameters:
        engine: The open index.
        root_url: Normalized URL of the root being ingested.

    Returns:
        The run's surrogate identifier.
    """
    with engine.begin() as connection:
        result = connection.execute(
            INGEST_RUNS.insert().values(
                root_url=root_url,
                started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                finished_utc=None,
                schema_version=SCHEMA_VERSION,
            )
        )
    return int(cast(tuple[Any, ...], result.inserted_primary_key)[0])


def _finish_run(engine: sqlalchemy.Engine, run_id: int, counts: IngestCounts) -> None:
    """Stamp an ingest run as complete and record what it covered.

    Parameters:
        engine: The open index.
        run_id: The run to complete.
        counts: What the run did.
    """
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.update()
            .where(INGEST_RUNS.c.run_id == run_id)
            .values(
                finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                files_seen=counts.files_seen,
                files_ingested=counts.files_ingested,
                files_skipped=counts.files_skipped,
                files_failed=counts.files_failed,
            )
        )


def _files_to_read(
    listing: _RootListing,
    recorded: dict[str, tuple[int | None, int | None]],
    *,
    force: bool,
) -> list[_ListedFile]:
    """Select the metadata files this pass has to read.

    Parameters:
        listing: What the walk found.
        recorded: Stub to the size and modification time already recorded.
        force: Whether to re-read every document regardless.

    Returns:
        The files to read, in stub order.
    """
    if force or not listing.has_file_metrics:
        return list(listing.metadata_files)
    return [
        listed
        for listed in listing.metadata_files
        if recorded.get(listed.results_path_stub) != (listed.mtime_ns, listed.size_bytes)
    ]


def ingest_metadata_files(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool = False,
    logger: PdsLogger,
) -> IngestCounts:
    """Ingest every metadata document under the given results roots.

    Each root is walked once, and each document whose recorded size and
    modification time still match the walk is skipped without being read.  A
    document that cannot be read as a current-schema navigation document is
    counted against its own file and the run continues.

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
    for root_str in roots:
        root_url = normalize_root_url(root_str)
        root = FCPath(root_str)
        counts = IngestCounts()
        run_id = _start_run(engine, root_url)
        logger.info('Ingesting %s', root_url)
        listing = _walk_root(root, logger=logger)
        counts.files_seen = len(listing.metadata_files)
        with engine.connect() as connection:
            recorded = _recorded_metrics(connection, root_url)
        to_read = _files_to_read(listing, recorded, force=force)
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
        _finish_run(engine, run_id, counts)
        logger.info(
            'Ingested %d, skipped %d unchanged, failed %d of %d file(s) under %s',
            counts.files_ingested,
            counts.files_skipped,
            counts.files_failed,
            counts.files_seen,
            root_url,
        )
        total.add(counts)
    return total
