"""Finding the navigation documents under a results root by listing directories.

The walk carries the metrics
----------------------------

The recursive listing collects the navigation documents in a single pass and
carries each entry's size and modification time with it, so both file metrics
come from the walk.  There is no per-file stat: on a cloud root each of those is
a paid round trip per image per run, against one round trip per directory for a
listing that returns up to a thousand entries with their metrics.  Those two
metrics are exactly what decides whether a document has changed since it was last
read, so a discovery that could not supply them would make every consumer re-read
every document.

A directory nobody can list
---------------------------

A walk that cannot list a directory ends there and then.  A directory nobody
enumerated holds documents nobody read, and absence is what a consumer reads as
"this image was never navigated", so a pass that finished around the gap would
stamp that reading as an answer.  Stopping costs a run; finishing costs a wrong
answer that outlives it.  A kernel set or a summary that quietly covers less
than the tree is worse than one that stops and says so.

A directory the walk has already listed under another name is a different thing,
and not a gap: its documents are in the listing under the path the walk met
first, and descending a second time would only report them again under stubs no
consumer asks about.  The walk declines it, says so, and goes on.  That also
stops a link pointing back up the tree from being followed until the filesystem
runs out of link depth.

Symbolic links inside a results tree
------------------------------------

Which of the two paths to such a directory the walk met first is whatever the
directory listings returned first, and that is not defined.  A document reached
two ways is therefore recorded under one of two stubs, either of them, and a
later pass may choose the other -- which makes it a document that has left the
tree under the stub the earlier pass recorded, so the rows written by one pass
are deleted by the next.  Do not put symbolic links inside a results tree.  A
results *root* that is a link is a different matter and is handled: a root is
resolved to the location it names before anything is read.
"""

from collections.abc import Iterator
from typing import Any

from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.nav_records.document import METADATA_SUFFIX
from spindoctor.nav_records.record import ListedRecord

__all__ = [
    'UnlistableDirectoryError',
    'UnlistableRootError',
    'walk_from',
]


class UnlistableDirectoryError(Exception):
    """A directory under a results root that the walk could not list.

    Raised where the walk meets the directory rather than where the damage
    would show, which is what keeps the cost to the run that has read nothing
    yet: a pass stopped at the end has already read every document under an
    archive-scale root, and discards hours of retrieval to say what it could
    have said in the first minute.

    A transient failure -- a share that stops answering for a moment, a
    permission fixed a minute later -- therefore costs a whole pass rather than
    degrading one.  That is the trade this exception is: what a pass produces is
    reproducible from the tree and can simply be produced again, and the answer
    it would otherwise leave behind is one no later pass corrects.

    Parameters:
        directory: The directory that would not be listed, which is kept as
            :attr:`directory` so that a caller can tell the directory it asked
            about from one further down.  A caller absorbing the first of those
            is answering about a location the tree does not hold; one absorbing
            the second would be answering out of a walk that stopped partway.
        reason: What the storage layer said when it refused.
    """

    def __init__(self, directory: str, reason: str) -> None:
        super().__init__(
            f'{directory} could not be listed ({reason}), so the documents under it were '
            f'never seen and an image beneath it would read as one nothing was ever '
            f'written for'
        )
        self.directory = directory


class UnlistableRootError(UnlistableDirectoryError):
    """A results root the walk could not list at all.

    Its own class because it is the one refusal a caller may reasonably absorb:
    a pass over several roots accounts for each of them separately, and a
    mistyped root is the commonest thing an operator types.  Every other
    consumer lets it end the run, exactly as it lets the directory case end one,
    which is why it is a kind of that rather than a thing beside it.

    Parameters:
        root: The root that would not be listed, kept as :attr:`directory` for
            the reason the base class keeps one.
        reason: What the storage layer said when it refused.
    """

    def __init__(self, root: str, reason: str) -> None:
        # The base class's message is about a directory inside a root that was
        # otherwise read.  Nothing under this one was read at all, and the
        # likeliest cause is the spelling rather than the storage, so the
        # message says that instead.
        Exception.__init__(
            self,
            f'results root {root} could not be listed ({reason}), so nothing under it has '
            f'been read: check the spelling of the root',
        )
        self.directory = root


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
        as far as this walk is concerned, since a deletion landing mid-walk is
        ordinary and there is nothing under such an entry to have missed.

    Raises:
        UnlistableDirectoryError: If the storage layer refuses to say what the
            entry is for any other reason.  An entry it will not answer about
            may be a directory holding documents, and calling it a file is
            exactly the silent gap this walk refuses: the walk would go on, the
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


def _entries_of(
    directory: FCPath, unlistable: type[UnlistableDirectoryError]
) -> list[tuple[FCPath, dict[str, Any] | None]]:
    """List one directory, or refuse in the terms its caller asked for.

    Parameters:
        directory: The directory to list.
        unlistable: The refusal to raise when it will not be listed, which is
            :class:`UnlistableRootError` for the root of a walk and
            :class:`UnlistableDirectoryError` for everything under it.

    Returns:
        The entries, each paired with whatever the listing reported about it.

    Raises:
        UnlistableDirectoryError: If the directory could not be listed.  Every
            way that can happen -- it is not there, it stopped being a directory
            between the parent listing and this call, this user may not read it,
            the share it lives on has gone away -- means the same thing to the
            walk: it can see no document here, which is not the same as there
            being none.
    """
    try:
        return list(directory.iterdir_metadata())
    except OSError as exc:
        raise unlistable(directory.as_posix(), str(exc)) from exc


def walk_from(
    directory: FCPath,
    prefix: str,
    visited: dict[tuple[int, int], str],
    *,
    unlistable: type[UnlistableDirectoryError],
    logger: PdsLogger,
) -> Iterator[ListedRecord]:
    """Yield the documents under one directory, descending as it goes.

    Parameters:
        directory: The directory to list.
        prefix: The path of that directory under the root, ending in ``/`` (or
            empty at the root itself), which is what makes each entry's stub.
        visited: Where this walk has already listed each directory it has
            listed, by identity, which it adds this one to.
        unlistable: The refusal to raise when this directory will not be listed.
            Its subdirectories always refuse as directories.
        logger: Logger for a directory reached a second way.

    Yields:
        One entry per navigation document, in the order the listings return
        them.  The walk descends the moment it meets a subdirectory, so a
        directory's own documents and the documents beneath it interleave.

    Raises:
        UnlistableDirectoryError: If this directory or any under it could not be
            listed.  The documents beneath it are then documents this walk
            cannot see, and a record's absence is what a consumer reads as an
            answer, so the walk stops instead of finishing around them.
    """
    identity = _directory_identity(directory)
    if identity is not None:
        listed_as = visited.get(identity)
        if listed_as is not None:
            # A second path to a directory this walk has already listed -- a
            # link back to an ancestor, or one subtree reachable two ways.
            # Descending would report the same documents again under a second
            # set of stubs, one per document per level until the filesystem
            # stops the loop at its own link limit, and no consumer asks about
            # any of them: a stub comes from the image's own subtree and
            # filespec, which name the directory once.  So the walk declines
            # it, and the root is still wholly listed, because every document
            # under it is in this listing under the path met first.
            logger.info(
                'Not listing %s, which is %s reached a second way and already listed',
                directory.as_posix(),
                listed_as,
            )
            return
        visited[identity] = directory.as_posix()
    for path, entry_metadata in _entries_of(directory, unlistable):
        name = path.name
        relative = f'{prefix}{name}'
        if _is_directory(path, entry_metadata):
            yield from walk_from(
                path,
                f'{relative}/',
                visited,
                unlistable=UnlistableDirectoryError,
                logger=logger,
            )
        elif name.endswith(METADATA_SUFFIX):
            mtime_ns, size_bytes = _metrics_of(entry_metadata)
            yield ListedRecord(
                stub=relative[: -len(METADATA_SUFFIX)],
                path=path,
                mtime_ns=mtime_ns,
                size_bytes=size_bytes,
            )
        # Every other file is passed over without being counted anywhere.  A
        # results tree holds the summary picture a navigation that reached a
        # result drew, and whatever else an operator has put there, and none of
        # them is a file this walk reads or a gap in what it listed.
