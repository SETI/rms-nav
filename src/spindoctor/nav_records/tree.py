"""The navigation records as documents under one or more results roots.

The authoritative storage: the documents as navigation left them, complete and
always current.  Everything a program asks of this source is answered by looking
at the tree, which is why a question about every image costs a read of every
file, and why the listing is worth so much more than the names it carries.

The walk carries the metrics
----------------------------

The recursive listing collects the navigation documents in a single pass and
carries each entry's size and modification time with it, so both file metrics
come from the walk.  There is no per-file stat and no per-file existence check:
on a cloud root each of those is a paid round trip per image per run, against
one round trip per directory for a listing that returns up to a thousand entries
with their metrics.  Those two metrics are exactly what decides whether a
document has changed since it was last read, so a discovery that could not
supply them would make every consumer re-read every document.

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

Batched underneath, lazy on top
-------------------------------

Documents are retrieved in groups and yielded one at a time.  Batched because on
a cloud root each file is a separate round trip and a backend downloads a batch
in parallel; lazy because the caller should not have to hold a mission in memory
to read the first record of one.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from filecache import FCPath
from pdslogger import NullLogger, PdsLogger

from spindoctor.nav_records.document import (
    COULD_NOT_RETRIEVE,
    METADATA_SUFFIX,
    NOT_A_JSON_OBJECT,
    NOT_VALID_JSON,
    UNREADABLE,
    document_path,
    read_document,
    stub_refusal,
)
from spindoctor.nav_records.record import ListedRecord, NavRecord, UnreadableFile
from spindoctor.nav_records.roots import distinct_roots
from spindoctor.nav_records.selection import Selection
from spindoctor.nav_records.source import (
    in_batches,
    refuse_what_a_listing_cannot_answer,
    root_for_stub,
    root_for_stubs,
    selected_roots,
)
from spindoctor.support.nav_record import record_midtime_et

__all__ = [
    'NAMES_NO_INSTRUMENT',
    'RECORDS_NO_MIDTIME',
    'RETRIEVE_BATCH_SIZE',
    'TreeRecordSource',
    'UnlistableDirectoryError',
    'UnlistableRootError',
]

RETRIEVE_BATCH_SIZE = 64
"""How many documents are retrieved in one batched download.

A cloud backend downloads a batch in parallel, so the batch size trades peak
memory and per-request concurrency against the number of round trips.
"""

NAMES_NO_INSTRUMENT = 'names no instrument to attribute it to a mission'
"""Why a document that read perfectly well is still not one mission's record.

Not one of the reasons in :mod:`spindoctor.nav_records.document`, which are the
ways a file yields no document at all.  This one is a document, read and parsed,
that no mission-filtered read can place: only a document that *names* a mission
can be another mission's, so one naming none is reported rather than passed over
silently, which is how a truncated or corrupted document would otherwise vanish
from every mission's run without a trace.
"""

RECORDS_NO_MIDTIME = 'records no exposure midtime to place it in a span of time'
"""Why a document that read perfectly well is still not one span's record.

The time filter's half of :data:`NAMES_NO_INSTRUMENT`, and reported for the same
reason: only a document that records a usable midtime can be shown to lie outside
a span, so one recording none is reported rather than passed over.  A truncated
document, or one whose midtime is a NaN that would otherwise satisfy every bound
at once, would vanish from every time-bounded run without a trace.
"""


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
        directory: The directory that would not be listed.
        reason: What the storage layer said when it refused.
    """

    def __init__(self, directory: str, reason: str) -> None:
        super().__init__(
            f'{directory} could not be listed ({reason}), so the documents under it were '
            f'never seen and an image beneath it would read as one nothing was ever '
            f'written for'
        )


class UnlistableRootError(UnlistableDirectoryError):
    """A results root the walk could not list at all.

    Its own class because it is the one refusal a caller may reasonably absorb:
    a pass over several roots accounts for each of them separately, and a
    mistyped root is the commonest thing an operator types.  Every other
    consumer lets it end the run, exactly as it lets the directory case end one,
    which is why it is a kind of that rather than a thing beside it.

    Parameters:
        root: The root that would not be listed.
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


def _walk_from(
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
            yield from _walk_from(
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


class TreeRecordSource:
    """The navigation records as documents under one or more results roots.

    Parameters:
        roots: The results roots to read, in the order questions are answered
            about them.  Two spellings of one root are one root, and the source
            holds each of them once.
        logger: Logger for what the walk declines to descend into.  Nothing else
            here logs: a file that yielded no record is yielded to the caller,
            which reports it in its own terms.  None says nothing at all, which
            is what a caller holding no logger of its own needs -- a layer that
            must not acquire a voice its own caller did not configure has none
            to lend.

    Raises:
        ValueError: If no root is given, or if one of them is not a location.
    """

    def __init__(
        self, roots: Sequence[str | Path | FCPath], *, logger: PdsLogger | None = None
    ) -> None:
        held = distinct_roots([str(root) for root in roots])
        if not held:
            raise ValueError('a record source over the documents needs at least one results root')
        self._roots = tuple(held)
        self._logger = NullLogger() if logger is None else logger

    def __enter__(self) -> 'TreeRecordSource':
        """Enter a run's use of this source.

        Returns:
            The source itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a run's use of this source, closing it.

        Parameters:
            exc_type: The exception's class, when the run is leaving on one.
            exc: The exception, when the run is leaving on one.
            traceback: Its traceback, when the run is leaving on one.
        """
        self.close()

    @property
    def roots(self) -> tuple[str, ...]:
        """The roots this source answers about, in the order it holds them.

        Returns:
            The normalized root URLs.
        """
        return self._roots

    def record(self, stub: str) -> NavRecord:
        """Read one image's document.

        Parameters:
            stub: The image's results path stub.

        Returns:
            The record, paired with the document it was read from.

        Raises:
            ValueError: If this source holds more than one root, naming them
                all: a stub is a key under a root and says nothing about which
                root it is a key under, so there is no single document to read.
                Also if the file is not valid JSON, or is valid JSON that is not
                an object.
            FileNotFoundError: If the document is not there, or if the stub is
                not a key under a root at all.  Both mean the same thing to a
                caller -- this image has no readable record here -- and the
                message says which of the two it was.
        """
        root = FCPath(root_for_stub(self._roots, stub))
        refusal = stub_refusal(stub)
        if refusal is not None:
            raise FileNotFoundError(
                f'{stub}: does not name a navigation document under '
                f'{root} ({refusal}), so none can be read for this image'
            )
        path = document_path(root, stub)
        return NavRecord(path=path, stub=stub, metadata=read_document(path))

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Return every document the selection covers, without opening one.

        The roots are walked in the order this source holds them, and each
        root's documents arrive in the order its directory listings return them.
        What the selection asks for is checked before anything is walked, so a
        selection this source cannot honour is refused where it is asked rather
        than partway through a caller's loop.

        Parameters:
            selection: Which documents to list.  Only ``roots`` and ``subtrees``
                may be set.

        Returns:
            One entry per document, with the size and modification time its
            directory listing reported, produced as the walk goes.

        Raises:
            ValueError: If the selection carries ``stubs``, ``instrument``,
                ``start_et`` or ``stop_et``, naming which.  A listing opens no
                document, so it cannot answer what a document says, and a
                restriction silently ignored is a wrong answer rather than a
                missing feature: a caller would read a listing of the whole root
                as a listing of one mission.  Also if the selection names a root
                this source does not hold.
            UnlistableDirectoryError: If a directory under a selected root could
                not be listed, or :class:`UnlistableRootError` if a selected
                root could not be listed at all.  Raised from the walk, so it
                reaches the caller as it reads.
        """
        refuse_what_a_listing_cannot_answer(selection)
        return self._listing_of(selected_roots(self._roots, selection.roots), selection.subtrees)

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Return the records the selection covers, yielded one at a time.

        Documents are retrieved :data:`RETRIEVE_BATCH_SIZE` at a time and
        yielded singly, so a caller holds one record where the source holds one
        batch.  What the selection asks for is checked before anything is read.

        Parameters:
            selection: Which records to yield.  A selection naming stubs reads
                exactly those, in the order it names them; one naming none takes
                its stubs from the listing of the selected roots.  Either way
                every stub is a key under the root: a named one was checked
                where the selection was written, and a listed one came from a
                document under the root itself.

        Returns:
            One record per document that yielded one, and one
            :class:`~spindoctor.nav_records.record.UnreadableFile` per file that
            did not.  A document of another mission is passed over silently when
            the selection names one, and so is a record outside the selection's
            time bounds; a document that names no mission, or that records no
            midtime a bound could be applied to, is reported instead, because
            only a document that says which it is can be shown not to be this
            run's.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, since a stub is a key under a root; or if it
                names a root this source does not hold.
            UnlistableDirectoryError: If a directory under a selected root could
                not be listed, or :class:`UnlistableRootError` if a selected
                root could not be listed at all.  Raised from the walk as the
                caller reads, and not at all when the selection names its own
                stubs, which lists nothing.
        """
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._records_of(roots, selection)
        root = root_for_stubs(roots, selection.stubs)
        return self._records_of_root(FCPath(root), iter(selection.stubs), selection)

    def _listing_of(self, roots: Sequence[str], subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Walk each root in turn, yielding its documents as they are found.

        Parameters:
            roots: The normalized roots to walk, in the order to walk them.
            subtrees: The top-level directories of each to descend, or empty for
                the whole of each root.

        Yields:
            One entry per document found.
        """
        for root_url in roots:
            yield from self._listing_of_root(FCPath(root_url), subtrees)

    def _records_of(
        self, roots: Sequence[str], selection: Selection
    ) -> Iterator[NavRecord | UnreadableFile]:
        """Read each root's documents in turn, taking its stubs from its listing.

        Parameters:
            roots: The normalized roots to read, in the order to read them.
            selection: The selection, for the subtrees to walk and the
                restrictions a document has to be opened to answer.

        Yields:
            One record or one unreadable file per document the selection covers.
        """
        for root_url in roots:
            root = FCPath(root_url)
            stubs = (listed.stub for listed in self._listing_of_root(root, selection.subtrees))
            yield from self._records_of_root(root, stubs, selection)

    def describe(self) -> str:
        """Return where these records came from, for the run log.

        Returns:
            The roots the documents were read under, in the order this source
            holds them.
        """
        return f'the navigation documents under {", ".join(self._roots)}'

    def close(self) -> None:
        """Release nothing: reading documents holds nothing open."""

    def _listing_of_root(self, root: FCPath, subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Yield one root's documents, restricted to the named top-level directories.

        Parameters:
            root: The results root to walk.
            subtrees: The top-level directories to descend, or empty for the
                whole root.

        Yields:
            One entry per document found.

        Raises:
            UnlistableRootError: If the whole root was asked for and could not
                be listed.
            UnlistableDirectoryError: If any directory under it could not be
                listed, a named subtree among them.  A subtree the root does not
                hold is refused rather than passed over: a run restricted to a
                directory that is not there would otherwise report a clean pass
                over nothing.
        """
        # One record of where this walk has been, shared by every subtree of the
        # root, so that two subtrees that are one directory reached two ways are
        # listed once between them rather than once each.
        visited: dict[tuple[int, int], str] = {}
        if not subtrees:
            yield from _walk_from(
                root, '', visited, unlistable=UnlistableRootError, logger=self._logger
            )
            return
        for subtree in subtrees:
            yield from _walk_from(
                root / subtree,
                f'{subtree}/',
                visited,
                unlistable=UnlistableDirectoryError,
                logger=self._logger,
            )

    def _records_of_root(
        self, root: FCPath, stubs: Iterator[str], selection: Selection
    ) -> Iterator[NavRecord | UnreadableFile]:
        """Retrieve one root's documents in batches and yield them one at a time.

        Parameters:
            root: The results root the stubs are keys under.
            stubs: The stubs to read, which arrive lazily when they come from a
                listing.
            selection: The selection, for the restrictions a document has to be
                opened to answer.

        Yields:
            One record or one unreadable file per stub that survives the
            selection's restrictions.
        """
        for batch in in_batches(stubs, RETRIEVE_BATCH_SIZE):
            sub_paths: list[str | Path] = [f'{stub}{METADATA_SUFFIX}' for stub in batch]
            # retrieve() rather than get_local_path(): on a cloud root the
            # latter names a file it never downloads.  exception_on_fail=False
            # keeps one file that never arrived from ending the pass.
            local_paths = cast(
                list[Path | Exception], root.retrieve(sub_paths, exception_on_fail=False)
            )
            for stub, local_path in zip(batch, local_paths, strict=True):
                found = self._record_of(root, stub, local_path, selection)
                if found is not None:
                    yield found

    def _record_of(
        self, root: FCPath, stub: str, local_path: Path | Exception, selection: Selection
    ) -> NavRecord | UnreadableFile | None:
        """Read one retrieved document, or say why it yielded no record.

        Parameters:
            root: The results root the stub is a key under.
            stub: The image's results path stub.
            local_path: What the retrieval produced for it: a local file, or the
                exception that says it never arrived.
            selection: The selection, for the restrictions a document has to be
                opened to answer.

        Returns:
            The record, or the unreadable file it is instead, or None when the
            document is outside the selection -- another mission's, or outside
            its time bounds.  Those two are passed over rather than reported,
            because neither is this run's business.  A document that cannot be
            placed at all -- naming no mission under a mission filter, recording
            no usable midtime under a time bound -- is an unreadable file
            instead: it cannot be shown to be another run's, so passing over it
            would drop it out of every run there is.
        """
        path = document_path(root, stub)
        if isinstance(local_path, BaseException):
            return UnreadableFile(path=path, stub=stub, reason=COULD_NOT_RETRIEVE)
        metadata = self._read(FCPath(local_path))
        if isinstance(metadata, str):
            return UnreadableFile(path=path, stub=stub, reason=metadata)
        if selection.instrument is not None:
            observation = metadata.get('observation')
            instrument = observation.get('instrument') if isinstance(observation, dict) else None
            if not isinstance(instrument, str):
                return UnreadableFile(path=path, stub=stub, reason=NAMES_NO_INSTRUMENT)
            if instrument != selection.instrument:
                return None
        if selection.bounded_in_time:
            midtime = record_midtime_et(metadata)
            if midtime is None:
                return UnreadableFile(path=path, stub=stub, reason=RECORDS_NO_MIDTIME)
            if not self._within(midtime, selection):
                return None
        return NavRecord(path=path, stub=stub, metadata=metadata)

    @staticmethod
    def _read(path: FCPath) -> dict[str, Any] | str:
        """Read one document, or return the reason no document came out of it.

        Parameters:
            path: The retrieved file.

        Returns:
            The document, or one of the reason constants in
            :mod:`spindoctor.nav_records.document`.
        """
        try:
            return read_document(path)
        except (OSError, UnicodeDecodeError):
            return UNREADABLE
        except json.JSONDecodeError:
            return NOT_VALID_JSON
        except ValueError:
            # What is left of ValueError once the decoder's own failure is taken
            # out is the document rule itself: valid JSON that is not an object.
            return NOT_A_JSON_OBJECT
        except Exception:
            # The decoder reports more than malformed syntax.  Twenty thousand
            # nested objects exhaust the recursion limit rather than failing to
            # parse, and a decoder that runs out of memory says so its own way.
            # Neither is a reason to end a pass, and what happened in both is
            # that no value came out of the file.
            return NOT_VALID_JSON

    @staticmethod
    def _within(midtime: float, selection: Selection) -> bool:
        """Whether one exposure midtime lies inside the selection's time bounds.

        Parameters:
            midtime: The record's exposure midtime, which the caller has already
                read and found usable.
            selection: The selection, which places at least one bound.

        Returns:
            True when the midtime is inside every bound the selection places.
            Both bounds are inclusive.
        """
        if selection.start_et is not None and midtime < selection.start_et:
            return False
        return not (selection.stop_et is not None and midtime > selection.stop_et)
