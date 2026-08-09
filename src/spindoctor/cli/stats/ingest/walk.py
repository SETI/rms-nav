"""One walk of a results root, feeding everything a pass takes from the tree.

The recursive listing collects the navigation documents in a single pass and
carries each entry's size and modification time with it, so both file metrics
come from the walk.  There is no per-file stat and no per-file existence check:
on a cloud root each of those is a paid round trip per image per run, which is
the cost this index exists to remove.

A walk also reports what it did not see.  A directory it could not list, and
one it had already listed under another name, leave it knowing about part of
the root rather than all of it -- which is not evidence that a stub it did not
find has left the tree, and not evidence a prune may act on.
"""

from dataclasses import dataclass, field
from typing import Any

from filecache import FCPath
from pdslogger import PdsLogger

__all__ = ['METADATA_SUFFIX']

METADATA_SUFFIX = '_metadata.json'
"""Suffix of the per-image navigation document under the results root."""


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
            the second one has been ingested when the walk ends.
        directories_missed: How many directories under the root this walk did
            not list on their own account -- ones it could not list, and ones
            it had already walked under another name.  The walk then knows
            about some of the root rather than all of it, which is not evidence
            that a stub it did not see is gone.
    """

    metadata_files: list[_ListedFile] = field(default_factory=list)
    has_file_metrics: bool = True
    root_listed: bool = True
    directories_missed: int = 0

    @property
    def covers_whole_root(self) -> bool:
        """Whether this listing is a complete account of the root.

        Returns:
            True when every directory under the root, and the root itself, was
            listed.  Only such a listing is evidence that a recorded stub it
            does not hold has left the tree.
        """
        return self.root_listed and not self.directories_missed


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
        directly; one that cannot answer either is treated as a file, since
        descending into something that is not there would only add a directory
        the walk could not list.
    """
    is_dir = None if entry_metadata is None else entry_metadata.get('is_dir')
    if is_dir is not None:
        return bool(is_dir)
    try:
        return bool(path.is_dir())
    except OSError:
        return False


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
    directory: FCPath, prefix: str, listing: _RootListing, visited: set[tuple[int, int]]
) -> bool:
    """Collect one directory's documents and descend into its subdirectories.

    Parameters:
        directory: The directory to list.
        prefix: The path of that directory under the root, ending in ``/`` (or
            empty at the root itself).
        listing: Accumulator the walk fills in.
        visited: Identities of the directories this walk has already listed,
            which it adds this one to.

    Returns:
        Whether the directory was listed on its own account.  False both for
        one that could not be listed and for one already walked under another
        name, because the caller's bookkeeping is the same either way: the walk
        did not enumerate this directory here, so it must not act on what it
        did not see.
    """
    identity = _directory_identity(directory)
    if identity is not None:
        if identity in visited:
            # A link back to a directory already walked.  Descending would
            # write the same documents again under a second set of stubs --
            # one row per document per level, until the filesystem stops the
            # loop at its own link limit -- so the walk stops here instead.
            return False
        visited.add(identity)
    try:
        entries = list(directory.iterdir_metadata())
    except OSError:
        # Every way a directory can refuse to be listed: it is not there, it
        # stopped being a directory between the parent listing and this call,
        # this user may not read it, the share it lives on has gone away.  All
        # of them mean the same thing to the walk -- it can see no result file
        # here, which is not the same as there being none -- and treating only
        # some of them that way ends the run on the commonest of them.
        return False
    for path, entry_metadata in entries:
        name = path.name
        relative = f'{prefix}{name}'
        is_dir = _is_directory(path, entry_metadata)
        if is_dir:
            if not _list_directory(path, f'{relative}/', listing, visited):
                listing.directories_missed += 1
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
        # results tree holds the summary PNG of each document and whatever else
        # an operator has put there, and none of them is a file this pass reads
        # or a gap in what it listed.
    return True


def _walk_root(root: FCPath, *, logger: PdsLogger) -> _RootListing:
    """Walk one results root once, collecting its navigation documents.

    Parameters:
        root: The results root.
        logger: Logger for the scan summary and the degraded-listing warning.

    Returns:
        What the walk found, with the metadata files in stub order.
    """
    listing = _RootListing()
    listing.root_listed = _list_directory(root, '', listing, set())
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
    if listing.directories_missed:
        logger.warning(
            '%d director(ies) under %s were not listed, so this pass covers some of the '
            'root rather than all of it and removes no row from it: absence of a row under '
            'one of them is not evidence that its image was never navigated',
            listing.directories_missed,
            root.as_posix(),
        )
    if not listing.has_file_metrics and listing.metadata_files:
        logger.warning(
            'Listing of %s reports no size or modification time, so every document '
            'is re-read: this root cannot be ingested incrementally',
            root.as_posix(),
        )
    return listing
