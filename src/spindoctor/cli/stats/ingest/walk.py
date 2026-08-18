"""One walk of a results root, feeding everything a pass takes from the tree.

The recursive listing collects the navigation documents in a single pass and
carries each entry's size and modification time with it, so both file metrics
come from the walk.  There is no per-file stat and no per-file existence check:
on a cloud root each of those is a paid round trip per image per run, which is
the cost this index exists to remove.

A walk that cannot list a directory ends the pass there and then.  A directory
nobody enumerated holds documents nobody recorded, and absence of a row is
exactly what every consumer reads as "this image was never navigated", so a
pass that finished around the gap would stamp that reading as an answer.
Stopping costs a run; finishing costs a wrong answer that outlives it.

A directory the walk has already listed under another name is a different
thing, and not a gap: its documents are in the listing under the path the walk
met first, and descending a second time would only write them again under
stubs no consumer asks about.  The walk declines it, says so, and goes on.
"""

from dataclasses import dataclass, field
from typing import Any

from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.support.nav_document import METADATA_SUFFIX

__all__ = ['METADATA_SUFFIX', 'UnlistableDirectoryError']


class UnlistableDirectoryError(Exception):
    """A directory under a results root that the walk could not list.

    Raised where the walk meets the directory rather than where the damage
    would show, which is what keeps the cost to the run that has read nothing
    yet: a pass stopped at prune time has already read every document under an
    archive-scale root, and discards hours of retrieval to say what it could
    have said in the first minute.

    A transient failure -- a share that stops answering for a moment, a
    permission fixed a minute later -- therefore costs a whole pass rather than
    degrading one.  That is the trade this exception is: an ingest is
    reproducible from the tree and can simply be run again, and the answer it
    would otherwise leave behind is one no later pass corrects.

    Parameters:
        directory: The directory that would not be listed.
        reason: What the storage layer said when it refused.
    """

    def __init__(self, directory: str, reason: str) -> None:
        super().__init__(
            f'{directory} could not be listed ({reason}), so the documents under it were '
            f'never seen and an image beneath it would read as one nothing was ever '
            f'written for'
        )


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
        has_file_metrics: Whether every metadata file reported both a size and
            a modification time.  A listing that reports neither cannot answer
            "has this changed", so such a root is re-read in full.
        root_listed: Whether the root itself could be listed.  A root that is
            not there is a different thing from a root that is empty, and only
            the second one has been ingested when the walk ends.  It is the one
            directory whose refusal is reported rather than raised, because a
            pass over several roots accounts for each of them separately and a
            mistyped root is the commonest thing an operator types.
    """

    metadata_files: list[_ListedFile] = field(default_factory=list)
    has_file_metrics: bool = True
    root_listed: bool = True


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


def _is_directory(path: FCPath, entry_metadata: dict[str, Any] | None) -> bool:
    """Whether one listing entry is a directory.

    Parameters:
        path: The entry.
        entry_metadata: What the listing said about it, or None when the
            backend reported nothing.

    Returns:
        True when the entry is a directory.  A backend that reported no
        metadata for the entry, or metadata that does not say, is asked
        directly; an entry that is no longer there to answer about is a file
        as far as this pass is concerned, since a deletion landing mid-walk is
        ordinary and there is nothing under such an entry to have missed.

    Raises:
        UnlistableDirectoryError: If the storage layer refuses to say what the
            entry is for any other reason.  An entry it will not answer about
            may be a directory holding documents, and calling it a file is
            exactly the silent gap this pass refuses: the walk would go on, the
            run would complete, and every document under it would read as an
            image nothing had navigated.
    """
    is_dir = None if entry_metadata is None else entry_metadata.get('is_dir')
    if is_dir is not None:
        return bool(is_dir)
    try:
        return bool(path.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        raise UnlistableDirectoryError(path.as_posix(), str(exc)) from exc


def _directory_identity(directory: FCPath) -> tuple[int, int] | None:
    """Return what makes one directory the same directory as another.

    Parameters:
        directory: The directory the walk is about to list.

    Returns:
        The device and inode numbers the filesystem gives it, or None when no
        identity can be taken: a cloud location, which has no links for a walk
        to go round in, or a directory the filesystem would not answer about,
        which the listing itself is about to fail on anyway.
    """
    if not directory.is_local():
        return None
    try:
        status = directory.stat()
    except OSError:
        return None
    return status.st_dev, status.st_ino


def _list_directory(
    directory: FCPath,
    prefix: str,
    listing: _RootListing,
    visited: dict[tuple[int, int], str],
    *,
    logger: PdsLogger,
) -> bool:
    """Collect one directory's documents and descend into its subdirectories.

    Parameters:
        directory: The directory to list.
        prefix: The path of that directory under the root, ending in ``/`` (or
            empty at the root itself).
        listing: Accumulator the walk fills in.
        visited: Where this walk has already listed each directory it has
            listed, by identity, which it adds this one to.
        logger: Logger for a directory reached a second way.

    Returns:
        Whether the directory was listed here.  False for the root of the pass
        when it could not be listed, which its caller reports as a root nobody
        ingested, and False for a directory already listed under another name,
        which is not a gap and which the recursive caller has nothing to do
        about.

    Raises:
        UnlistableDirectoryError: If a directory under the root could not be
            listed.  The documents beneath it are then documents this pass
            cannot see, and a row's absence is what a consumer reads as an
            answer, so the pass stops instead of finishing around them.
    """
    identity = _directory_identity(directory)
    if identity is not None:
        listed_as = visited.get(identity)
        if listed_as is not None:
            # A second path to a directory this walk has already listed -- a
            # link back to an ancestor, or one volume reachable two ways.
            # Descending would write the same documents again under a second
            # set of stubs, one row per document per level until the
            # filesystem stops the loop at its own link limit, and no consumer
            # asks about any of them: a stub comes from the image's own
            # volume and filespec, which name the directory once.  So the walk
            # declines it, and the root is still wholly listed, because every
            # document under it is in this listing under the path met first.
            logger.info(
                'Not listing %s, which is %s reached a second way and already listed',
                directory.as_posix(),
                listed_as,
            )
            return False
        visited[identity] = directory.as_posix()
    try:
        entries = list(directory.iterdir_metadata())
    except OSError as exc:
        # Every way a directory can refuse to be listed: it is not there, it
        # stopped being a directory between the parent listing and this call,
        # this user may not read it, the share it lives on has gone away.  All
        # of them mean the same thing to the walk -- it can see no result file
        # here, which is not the same as there being none -- and it is the root
        # alone that is reported rather than raised.
        if not prefix:
            return False
        raise UnlistableDirectoryError(directory.as_posix(), str(exc)) from exc
    for path, entry_metadata in entries:
        name = path.name
        relative = f'{prefix}{name}'
        is_dir = _is_directory(path, entry_metadata)
        if is_dir:
            _list_directory(path, f'{relative}/', listing, visited, logger=logger)
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
        # Every other file is passed over without being counted anywhere.  A
        # results tree holds the summary picture a navigation that reached a
        # result drew, and whatever else an operator has put there, and none of
        # them is a file this pass reads or a gap in what it listed.
    return True


def _walk_root(root: FCPath, *, logger: PdsLogger) -> _RootListing:
    """Walk one results root once, collecting its navigation documents.

    Parameters:
        root: The results root.
        logger: Logger for the scan summary and the degraded-listing warning.

    Returns:
        What the walk found, with the metadata files in stub order.  A listing
        this returns for a root it listed is a complete account of that root,
        which is what licenses the prune to act on what it does not hold.

    Raises:
        UnlistableDirectoryError: If a directory under the root could not be
            listed, which ends the whole pass rather than this root's part of
            it.
    """
    listing = _RootListing()
    listing.root_listed = _list_directory(root, '', listing, {}, logger=logger)
    listing.metadata_files.sort(key=lambda listed: listed.results_path_stub)
    if not listing.root_listed:
        logger.error(
            'Results root %s could not be listed, so nothing under it has been ingested: '
            'check the spelling of the root',
            root.as_posix(),
        )
        return listing
    logger.info(
        'Results scan found %d metadata file(s) under %s',
        len(listing.metadata_files),
        root.as_posix(),
    )
    if not listing.has_file_metrics and listing.metadata_files:
        logger.warning(
            'Listing of %s reports no size or modification time, so every document '
            'is re-read: this root cannot be ingested incrementally',
            root.as_posix(),
        )
    return listing
