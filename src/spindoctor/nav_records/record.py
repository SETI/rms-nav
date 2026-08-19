"""What a navigation record is, and what stands in its place when there is none.

A navigation pass writes one document per image, and every program downstream of
navigation reads what that document says.  The same thing can be kept two ways --
as the document itself, or as a row of an ingested results index -- so the shapes
a reader meets are declared here, in a module that knows about neither storage.

Three of them, for the four questions a program asks:

* :class:`NavRecord` is a record, paired with the document it stands for.
* :class:`UnreadableFile` is a file where a record should have been.  It is a
  value rather than an exception because a stream of a mission's records must be
  able to report one without ending: the file names no image, so there is nothing
  for a run to omit and no image to attribute a failure to, but a run that passed
  over it in silence would report itself clean while covering less than the tree.
* :class:`ListedRecord` is what a listing knows about a document without opening
  it: where it is, and the two metrics that say whether it has changed.

A stream of per-image facts answers in two of these three: the facts themselves,
which are :mod:`spindoctor.nav_records.facts`'s, and an
:class:`UnreadableFile` for a file no facts came out of.
"""

from dataclasses import dataclass
from typing import Any

from filecache import FCPath

__all__ = [
    'ListedRecord',
    'NavRecord',
    'UnreadableFile',
]


@dataclass(frozen=True)
class NavRecord:
    """One image's navigation record, and where that record is kept.

    A record read from its document and one rebuilt from an index row are the
    same thing to every consumer, so both arrive in this shape and neither
    carries which storage produced it.

    Parameters:
        path: The document.  For a record rebuilt from a row this is the
            document the ingest recorded reading, or -- for a row written before
            anything recorded one -- where the stub says the document lives.  It
            is what a message about this record names, so that an operator is
            always told a file they can open.
        stub: The image's results path stub: its identity under the root, and the
            name of its log.
        metadata: The record, in the shape the navigator writes it.  A record
            rebuilt from a row carries the fields its columns hold and no others,
            which is the part of the document its consumer reads.
    """

    path: FCPath
    stub: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class UnreadableFile:
    """A file a record could not be read out of, and why.

    Yielded into the same stream as the records rather than raised, so that one
    unreadable file costs itself and not the rest of the pass.  What a consumer
    owes it is a report: a kernel set or a summary that quietly covered less than
    the tree is worse than one that says how much less.

    Parameters:
        path: The file, which is what an operator would open.
        stub: Its results path stub, which is the only name it has under the
            root.  A file that is not a record still has one, because a stub
            comes from where the file is rather than from what it says.
        reason: Why no record came out of it, in terms a run log can print.
    """

    path: FCPath
    stub: str
    reason: str


@dataclass(frozen=True)
class ListedRecord:
    """One document a listing found, and what the listing knew about it.

    The two metrics come from the directory entry itself rather than from a stat
    of the file: on a cloud root a listing returns them for up to a thousand
    entries in one round trip, and asking per file would cost one round trip per
    image.  They are what decides whether a document has changed since it was
    last read, so a listing that reports neither cannot answer that question and
    every document under it has to be read again.

    Parameters:
        stub: The document's results path stub -- its path under the root with
            the document suffix removed.
        path: Where the document is.
        mtime_ns: Modification time in nanoseconds, or None when the listing
            reported none.
        size_bytes: Size in bytes, or None when the listing reported none.
    """

    stub: str
    path: FCPath
    mtime_ns: int | None
    size_bytes: int | None

    @property
    def has_metrics(self) -> bool:
        """Whether the listing reported both metrics for this document.

        Returns:
            True when a later pass can decide from the listing alone whether this
            document has changed, which needs both the size and the modification
            time.
        """
        return self.mtime_ns is not None and self.size_bytes is not None
