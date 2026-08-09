"""Retrieving a chunk of metadata files, reading each one, and writing them.

Retrieval is batched because a cloud backend downloads a batch in parallel, so
the batch size trades peak memory and per-request concurrency against the
number of round trips.  It is ``retrieve()`` that is called rather than
``get_local_path()``, which on a cloud root names a file the cache would hold
and downloads nothing.

What a failed document costs
----------------------------

A results tree holds ``*_metadata.json`` files that are not per-image
navigation documents -- products of other tools, and documents written by an
older metadata schema.  Each is counted as an error for its own file and no
more: the run continues, and the closing summary tallies the failures by
reason, so several hundred documents that were never navigation results read as
exactly that rather than as a broken ingest.

A refused file is recorded in ``failed_files`` with everything the walk knows
about it and nothing the document would have said: the same two metrics an
ingested one records, so an unchanged refusal is skipped on the next pass
instead of being downloaded and parsed again forever, plus the volume it lives
under, which is what a selection filter asks of a file it never opens.  A
retrieval that never delivered the file is counted without being recorded: it
says nothing about the file that will still be true next pass, and a recorded
refusal is skipped for as long as the file does not change.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.stats.ingest.counts import IngestCounts
from spindoctor.cli.stats.ingest.store import _write_chunk
from spindoctor.cli.stats.ingest.walk import METADATA_SUFFIX, _ListedFile
from spindoctor.cli.stats.ingest_rows import (
    NOT_A_NAVIGATION_DOCUMENT,
    ImageRows,
    MetadataDocumentError,
    MetadataSource,
    _volume_of,
    rows_from_metadata,
)

__all__ = ['INGEST_RETRIEVE_BATCH_SIZE']

_Item = TypeVar('_Item')
"""What one slice of a batched sequence holds."""

INGEST_RETRIEVE_BATCH_SIZE = 64
"""How many metadata files are retrieved in one batched download.

A cloud backend downloads a batch in parallel, so the batch size trades peak
memory and per-request concurrency against the number of round trips.
"""


def _batched(items: Sequence[_Item], size: int) -> Iterator[Sequence[_Item]]:
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
    except Exception as exc:
        # The decoder reports more than malformed syntax.  Twenty thousand
        # nested objects exhaust the recursion limit rather than failing to
        # parse, and a decoder that runs out of memory says so its own way.
        # None of them is a reason to end the run.
        raise MetadataDocumentError(
            f'{NOT_A_NAVIGATION_DOCUMENT} ({type(exc).__name__} while parsing it)',
            source_file=source.source_file,
        ) from exc
    if not isinstance(parsed, dict):
        raise MetadataDocumentError('not a JSON object', source_file=source.source_file)
    try:
        return rows_from_metadata(parsed, source)
    except MetadataDocumentError:
        raise
    except Exception as exc:
        # The converter checks every shape the document schema declares, so
        # reaching here means a shape nobody enumerated.  One such file costs
        # itself; letting the exception out would cost every other file in the
        # tree, and would leave the root's ingest run unfinished, after which
        # every consumer refuses the root.
        raise MetadataDocumentError(
            f'{NOT_A_NAVIGATION_DOCUMENT} ({type(exc).__name__} while reading it)',
            source_file=source.source_file,
        ) from exc


def _ingest_chunk(
    engine: sqlalchemy.Engine,
    root: FCPath,
    chunk: Sequence[_ListedFile],
    *,
    root_url: str,
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
        counts: Accumulator this chunk's outcomes are added to.
        logger: Logger for per-file failures.
    """
    pending: list[ImageRows] = []
    refused: list[dict[str, Any]] = []
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
            )
            if isinstance(local_path, BaseException):
                # Nothing was read, so nothing is known about the file beyond
                # the listing.  A retrieval that failed once is worth trying
                # again, so no refusal is recorded for it.
                counts.record_failure('could not be retrieved', source.source_file)
                logger.debug('Skipping %s: could not be retrieved', source.source_file)
                continue
            try:
                pending.append(_read_document(local_path, source))
            except MetadataDocumentError as exc:
                counts.record_failure(exc.reason, source.source_file)
                logger.debug('Skipping %s', exc)
                refused.append(
                    {
                        'root_url': root_url,
                        'results_path_stub': listed.results_path_stub,
                        'reason': exc.reason,
                        # The walk knows this whatever the file says, and a
                        # selection filter asks about the file rather than
                        # about its contents: a refused document is one the
                        # tree still holds, under a volume.  The volume is
                        # derived by the same function the images row uses, so
                        # the two tables can never disagree about which volume
                        # a stub is under.
                        'volume': _volume_of(listed.results_path_stub),
                        'mtime_ns': listed.mtime_ns,
                        'size_bytes': listed.size_bytes,
                    }
                )
    if not pending and not refused:
        return
    counts.files_ingested += _write_chunk(engine, pending, refused, counts=counts, logger=logger)
