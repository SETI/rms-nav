"""The document a navigation record is written to, and how one is read back.

:mod:`spindoctor.support.nav_record` decides what the values of a record mean.
This module is about the file those values are written to: what it is named,
where the document of one image lives under a results root, and how one is read.
It knows nothing about a database, so every reader of a document shares it
whether or not the program it belongs to can read an index.

The three rules here are each written once because each of them was written more
than once before, and a rule about which files may be read is not a rule while
two readers hold different versions of it:

* **What a document is named.**  A results path stub is a document's path under
  its root with the suffix taken off, so the walk that lists documents, the
  ingest that records their stubs and every reader that rebuilds a path from one
  have to agree about which suffix that is.
* **Which paths a root may be read at.**  A stub is a key, and reading a key back
  as a path is where a key carrying ``..`` becomes a file outside the root.
* **What makes a file a document.**  Valid JSON holding an object, which is what
  every reader needs before it can read a field off one.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from filecache import FCPath

from spindoctor.support.nav_record import NavRecord

__all__ = [
    'ABSOLUTE_PATH_FRAGMENT',
    'METADATA_SUFFIX',
    'NULL_BYTE_IN_PATH',
    'PATH_OUTSIDE_ROOT',
    'ResolvedDocumentPath',
    'read_document',
    'read_documents',
    'resolved_document_path',
    'stub_for_document',
]

METADATA_SUFFIX = '_metadata.json'
"""What a per-image navigation document is named."""

NULL_BYTE_IN_PATH = 'metadata path contains null byte'
"""Why a stub naming a path with a null byte may not be read."""

ABSOLUTE_PATH_FRAGMENT = 'metadata path fragment is absolute'
"""Why a stub that is an absolute path may not be read under a root."""

PATH_OUTSIDE_ROOT = 'resolved metadata path is outside root'
"""Why a stub whose path escapes its root may not be read."""


@dataclass(frozen=True)
class ResolvedDocumentPath:
    """Where one stub's document lives, or why it may not be read there.

    Parameters:
        path: The document's location, and None when the stub does not name one
            this root may be read at.
        refusal: Which rule the stub broke, and None when it broke none.  It is
            one of the three constants in this module, so that a caller reports
            the refusal in its own terms rather than passing on a sentence
            written here.
        resolved: What the stub resolved to when it escaped its root, for the
            report that has to show it, and None otherwise.
        root: The root it escaped, and None otherwise.
    """

    path: FCPath | None
    refusal: str | None = None
    resolved: FCPath | None = None
    root: FCPath | None = None


def resolved_document_path(
    nav_results_root: str | Path | FCPath, results_path_stub: str
) -> ResolvedDocumentPath:
    """Resolve one stub's document under a results root, refusing what escapes it.

    Refused are a null byte, a stub that is an absolute path of its own, and any
    resolution landing outside the root, which a stub carrying ``..`` segments
    does.  One rule decides which paths a results root may be read at, for every
    reader, rather than each reader deciding for itself.

    Nothing is logged and nothing is raised: which of those a refusal becomes is
    the caller's to decide, and the callers decide differently.  A per-image
    reader reports it against the image and carries on with no pointing; a reader
    asked for the record itself has nothing to hand back and raises.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.
        results_path_stub: The image's results path stub.

    Returns:
        The resolution, carrying either the path or the rule that refused it.
    """
    relative_name = f'{results_path_stub}{METADATA_SUFFIX}'
    if '\x00' in relative_name:
        return ResolvedDocumentPath(None, refusal=NULL_BYTE_IN_PATH)
    if Path(relative_name).is_absolute():
        return ResolvedDocumentPath(None, refusal=ABSOLUTE_PATH_FRAGMENT)
    root = FCPath(nav_results_root).expanduser().resolve()
    candidate = (root / relative_name).resolve()
    if not candidate.is_relative_to(root):
        return ResolvedDocumentPath(None, refusal=PATH_OUTSIDE_ROOT, resolved=candidate, root=root)
    return ResolvedDocumentPath(candidate)


def stub_for_document(root: FCPath, path: FCPath) -> str:
    """Return the results path stub of one document under a root.

    Parameters:
        root: The navigation results root.
        path: The document.

    Returns:
        The file's path relative to the root, without the document suffix.  The
        full path is used when it does not lie under the root, which cannot
        happen for a document the root's own listing produced.
    """
    relative = path.as_posix().removeprefix(root.as_posix()).lstrip('/')
    return relative.removesuffix(METADATA_SUFFIX)


def read_document(path: FCPath) -> dict[str, Any]:
    """Read one navigation document.

    Parameters:
        path: The file to read.

    Returns:
        The document.

    Raises:
        ValueError: If the file does not hold a JSON object.  A file holding
            valid JSON that is not an object is not a document, and reading
            fields off one would fail later and further away.
        OSError: If it cannot be read.  A file that is not there raises
            :exc:`FileNotFoundError`, which is what a caller distinguishing an
            unnavigated image from an unreadable one catches.
    """
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f'holds a {type(document).__name__}, not a JSON object')
    return cast(dict[str, Any], document)


def read_documents(root: FCPath, mission: str) -> tuple[list[NavRecord], list[tuple[FCPath, str]]]:
    """Read every document of one mission under a results root.

    A file that cannot be read as JSON, or that holds JSON that is not a
    document, is returned for the caller to report rather than raised on: it
    names no image, so there is nothing for a report to say about it and nothing
    an omission reason could be recorded against.  A document of another mission
    is simply not this run's business and is passed over silently -- but only a
    document that *names* a mission can be another mission's.  One with no
    readable instrument at all is unreadable, not foreign: skipping it silently
    would let a truncated or corrupted document vanish from every mission's run
    without a trace.

    Parameters:
        root: The navigation results root.
        mission: The instrument identity to keep.

    Returns:
        The mission's records, ordered by the path of the document each was read
        from, and one entry per file that could not be read at all, pairing it
        with why.
    """
    records: list[NavRecord] = []
    unreadable: list[tuple[FCPath, str]] = []
    for path in sorted(root.rglob(f'*{METADATA_SUFFIX}'), key=lambda entry: entry.as_posix()):
        stub = stub_for_document(root, path)
        try:
            metadata = read_document(path)
        except (OSError, ValueError) as exc:
            unreadable.append((path, str(exc)))
            continue
        observation = metadata.get('observation')
        instrument = observation.get('instrument') if isinstance(observation, dict) else None
        if not isinstance(instrument, str):
            unreadable.append((path, 'names no instrument to attribute it to a mission'))
            continue
        if instrument != mission:
            continue
        records.append(NavRecord(path=path, stub=stub, metadata=metadata))
    return records, unreadable
