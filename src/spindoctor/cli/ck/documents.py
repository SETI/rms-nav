"""Where the C-kernel generator gets one mission's navigation documents.

The generator reads every navigation document under a results root, keeps the
ones its mission wrote, and builds a segment from some of them.  On a local
tree that is one file read per image; on a cloud root it is one paid round trip
per image, and a Cassini-scale root holds several hundred thousand.  An
ingested results index answers the same question with one query, so the seam is
explicit: :class:`DocumentSource` names it, :class:`TreeDocumentSource` reads
files, and :class:`IndexDocumentSource` reads rows.

This is a **bulk** reader rather than a per-image lookup -- every document of
one mission under one root, which the run then filters by time -- so the index
path issues one statement for the images and one for the files the ingest
refused, and nothing else touches storage.

What a rebuilt document carries
-------------------------------

Only the fields this program reads: ``observation`` (``image_name``,
``instrument``, ``camera``, ``shutter_mode``), the top-level ``status``,
``offset`` and ``confidence``, and from ``navigation_result`` its
``status_reason``, ``sigma_px``, ``confidence_rank``, ``rotation_deg``, the
recorded ``provenance.spice_kernels``, and the ``times`` and ``pointing``
blocks.  It is not the document; it is the part of the document a corrected
kernel and its report are built from.

A field the row does not carry is rebuilt as a field the document did not have,
rather than as one holding null, because the readers distinguish the two.
``offset`` is the exception and is rebuilt as a key holding null, which is what
the navigator writes for an image that measured none.  The ``status`` column is
NOT NULL and stands in for a document that named no outcome with
:data:`~spindoctor.support.nav_record.UNKNOWN_STATUS`; that value is rebuilt as
the absent field it stands for, so a document with no status is refused by this
path exactly as the file path refuses it.

Where the two paths differ
--------------------------

One class of difference survives, and it has one cause: **a value the ingest
could not store is rebuilt as absent, so this path reports an image the file
path refuses outright.**  The readers here refuse a malformed value loudly --
an offset of three numbers, a sigma holding text, a confidence that is not a
number -- and end the run naming the document, because a value of the wrong
kind is a defect in the record rather than an image without a solution.  Ingest
stores each of those as NULL, exactly as it stores an absent one, so the row
cannot say which it was; the rebuilt document then reads as one that recorded
nothing there, and the image is reported with that value blank.

Nothing a segment is built from can differ.  A C-matrix, its baseline, the
frame identities and the exposure epochs are stored in the shapes their
producer writes and are rebuilt in those shapes, and a row whose matrix the
readers refuse is refused here too.

A file the ingest refused entirely is not a row at all: it is recorded in
``failed_files``, and this source reports every such file under the root as one
it could not read, with the reason the ingest recorded.  That count decides the
exit status, as an unreadable file does on the file path.  Such a file names no
mission, since ingest could not read one out of it, so it is reported whichever
mission is being written -- which is what the file path does with a document it
cannot attribute to a mission.
"""

from typing import Any, Protocol

import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Engine

from spindoctor.cli.ck.inputs import METADATA_SUFFIX, Document, read_documents
from spindoctor.results_index import (
    FAILED_FILES,
    IMAGES,
    masked_url,
    normalize_root_url,
    open_index,
    reporting_a_failed_read,
    require_ingested_roots,
)
from spindoctor.support.nav_record import UNKNOWN_STATUS

__all__ = [
    'DocumentSource',
    'IndexDocumentSource',
    'TreeDocumentSource',
    'build_document_source',
]

_ROW_COLUMNS = (
    IMAGES.c.results_path_stub,
    IMAGES.c.source_file,
    IMAGES.c.image_name,
    IMAGES.c.instrument,
    IMAGES.c.camera,
    IMAGES.c.shutter_mode,
    IMAGES.c.status,
    IMAGES.c.status_reason,
    IMAGES.c.offset_dv,
    IMAGES.c.offset_du,
    IMAGES.c.sigma_dv,
    IMAGES.c.sigma_du,
    IMAGES.c.rotation_deg,
    IMAGES.c.confidence,
    IMAGES.c.confidence_rank,
    IMAGES.c.spice_kernels,
    IMAGES.c.start_et,
    IMAGES.c.stop_et,
    IMAGES.c.midtime_et,
    IMAGES.c.exposure_s,
    IMAGES.c.sclk_midtime,
    IMAGES.c.camera_frame,
    IMAGES.c.camera_frame_id,
    IMAGES.c.ck_frame_id,
    IMAGES.c.cmatrix,
    IMAGES.c.cmatrix_original,
)
"""Every column a rebuilt document is made of, read in one SELECT."""


class DocumentSource(Protocol):
    """One mission's navigation documents, however they are stored."""

    def read_documents(self, mission: str) -> tuple[list[Document], list[tuple[FCPath, str]]]:
        """Return one mission's documents, and the files that could not be read.

        Parameters:
            mission: The instrument identity to keep.

        Returns:
            The documents, ordered by their path under the root, and one entry
            per file that could not be read at all, pairing it with why.
        """
        ...

    def describe(self) -> str:
        """Return where these documents came from, for the run log.

        Returns:
            The results root, and the index they were read out of when they
            were read out of one.  Any password in an index URL is masked.
        """
        ...

    def close(self) -> None:
        """Release whatever the source holds open."""
        ...


class TreeDocumentSource:
    """The documents as files under a results root.

    Parameters:
        nav_results_root: The navigation results root to walk.
    """

    def __init__(self, nav_results_root: FCPath) -> None:
        self._root = nav_results_root

    def read_documents(self, mission: str) -> tuple[list[Document], list[tuple[FCPath, str]]]:
        """Walk the root and read every document of one mission.

        Parameters:
            mission: The instrument identity to keep.

        Returns:
            The documents and the unreadable files, as
            :func:`~spindoctor.cli.ck.inputs.read_documents` produces them.
        """
        return read_documents(self._root, mission)

    def describe(self) -> str:
        """Return the root the documents were read from.

        Returns:
            The root as a POSIX path.
        """
        return self._root.as_posix()

    def close(self) -> None:
        """Release nothing: a walk holds nothing open."""


class IndexDocumentSource:
    """The documents as rows of a results index.

    Parameters:
        engine: The open index.
        root_url: Normalized URL of the results root whose rows to read.
        url: The index URL, kept for the messages that name it.
    """

    def __init__(self, engine: Engine, root_url: str, url: str) -> None:
        self._engine = engine
        self._root_url = root_url
        self._url = url
        self._root = FCPath(root_url)

    def read_documents(self, mission: str) -> tuple[list[Document], list[tuple[FCPath, str]]]:
        """Read one mission's rows, and the files the ingest refused.

        Both statements are keyed by the root as well as by what they select.
        One index serves several results roots, and a query that asked only for
        the mission would write another root's images into this run's kernels.

        Parameters:
            mission: The instrument identity to keep, matched against the
                ``instrument`` column exactly as the file path matches it
                against ``observation.instrument``.

        Returns:
            The documents, ordered by their stub, and one entry per file the
            ingest refused under this root, pairing it with the recorded
            reason.

        Raises:
            ValueError: If the index cannot be read, naming it with any
                password masked.
        """
        images = (
            sqlalchemy.select(*_ROW_COLUMNS)
            .where(IMAGES.c.root_url == self._root_url, IMAGES.c.instrument == mission)
            .order_by(IMAGES.c.results_path_stub)
        )
        refused = (
            sqlalchemy.select(FAILED_FILES.c.results_path_stub, FAILED_FILES.c.reason)
            .where(FAILED_FILES.c.root_url == self._root_url)
            .order_by(FAILED_FILES.c.results_path_stub)
        )
        with reporting_a_failed_read(self._url), self._engine.connect() as connection:
            documents = [self._document_of(row) for row in connection.execute(images)]
            unreadable = [
                (self._path_of(str(row.results_path_stub)), str(row.reason))
                for row in connection.execute(refused)
            ]
        return documents, unreadable

    def describe(self) -> str:
        """Return the root and the index the documents were read out of.

        Returns:
            The root, followed by the index URL with any password masked.
        """
        return f'{self._root_url} in the results index {masked_url(self._url)}'

    def close(self) -> None:
        """Dispose of the engine and its connection pool."""
        self._engine.dispose()

    def _path_of(self, stub: str) -> FCPath:
        """Return where the document of one stub lives under this root.

        Parameters:
            stub: The image's results path stub.

        Returns:
            The metadata file's location, which is the stub under the root with
            the document suffix restored.

        """
        return self._root / f'{stub}{METADATA_SUFFIX}'

    def _document_of(self, row: sqlalchemy.Row[Any]) -> Document:
        """Rebuild one row's document, with the file it was read from.

        Parameters:
            row: One row of the index, carrying every column of
                ``_ROW_COLUMNS``.

        Returns:
            The document.
        """
        stub = str(row.results_path_stub)
        # The path the ingest recorded reading, so a message about this
        # document names the file an operator would open.  A row written before
        # anything recorded one falls back to where the stub says it lives.
        path = self._path_of(stub) if row.source_file is None else FCPath(str(row.source_file))
        return Document(path=path, stub=stub, metadata=_metadata_from_row(row))


def _present(values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Return the named values that are not NULL, as one block of a document.

    A column the row does not carry is a field the document did not have, and
    every reader here distinguishes an absent field from one holding null, so
    an absent value is left out rather than written as null.

    Parameters:
        values: Field names paired with what the row holds for each.

    Returns:
        The fields that carry a value, in the order they were given.
    """
    return {name: value for name, value in values if value is not None}


def _pair_or_none(first: Any, second: Any) -> list[float] | None:
    """Return a recorded ``[dv, du]`` pair, or None when the row has no pair.

    Parameters:
        first: The first member's column value.
        second: The second member's column value.

    Returns:
        The pair, or None when either half is NULL.  Ingest stores a pair it
        could not read whole as two NULLs, so half a pair is not a pair.
    """
    if first is None or second is None:
        return None
    return [float(first), float(second)]


def _metadata_from_row(row: sqlalchemy.Row[Any]) -> dict[str, Any]:
    """Rebuild the navigation document one index row records.

    Parameters:
        row: One row of the index, carrying every column of ``_ROW_COLUMNS``.

    Returns:
        The rebuilt document, holding the fields this program reads.
    """
    metadata: dict[str, Any] = {
        'observation': _present(
            (
                ('image_name', row.image_name),
                ('instrument', row.instrument),
                ('camera', row.camera),
                ('shutter_mode', row.shutter_mode),
            )
        )
    }
    # A document that named no outcome is stored as the sentinel status and is
    # rebuilt as the absent field it stands for, so this path refuses such a
    # record where the file path refuses it.  A document naming that same word
    # for itself is rebuilt without the field too, which is the one case the
    # sentinel cannot tell apart.
    if row.status != UNKNOWN_STATUS:
        metadata['status'] = row.status
    # Written whichever way the pair went, since the navigator writes the key
    # for an image that measured no offset as well as for one that did.
    metadata['offset'] = _pair_or_none(row.offset_dv, row.offset_du)
    metadata.update(_present((('confidence', row.confidence),)))
    navigation_result: dict[str, Any] = _present(
        (
            ('status_reason', row.status_reason),
            ('sigma_px', _pair_or_none(row.sigma_dv, row.sigma_du)),
            ('confidence_rank', row.confidence_rank),
            # Presence alone decides whether the run fitted a camera rotation,
            # whose pivot no record carries and which therefore cannot be
            # written as an attitude.
            ('rotation_deg', row.rotation_deg),
        )
    )
    provenance = _present((('spice_kernels', row.spice_kernels),))
    if provenance:
        navigation_result['provenance'] = provenance
    times = _present(
        (
            ('start_et', row.start_et),
            ('stop_et', row.stop_et),
            ('midtime_et', row.midtime_et),
            ('exposure_s', row.exposure_s),
            ('sclk_midtime', row.sclk_midtime),
        )
    )
    if times:
        navigation_result['times'] = times
    # The block exists whenever any of its columns does, not only when the
    # corrected attitude does: its producer writes the baseline and the frame
    # identities for every navigated image and the corrected attitude only
    # where one was computed, so a block with none of them is a record that had
    # no pointing block, and one missing only the attitude is a result that
    # fitted a camera rotation.
    pointing = _present(
        (
            ('cmatrix', row.cmatrix),
            ('cmatrix_original', row.cmatrix_original),
            ('camera_frame', row.camera_frame),
            ('camera_frame_id', row.camera_frame_id),
            ('ck_frame_id', row.ck_frame_id),
        )
    )
    if pointing:
        navigation_result['pointing'] = pointing
    if navigation_result:
        metadata['navigation_result'] = navigation_result
    return metadata


def build_document_source(
    nav_results_root: FCPath, *, results_db_url: str | None
) -> DocumentSource:
    """Build the source a run reads its navigation documents through.

    With no index URL the source walks the results tree, which is the default.
    With one, the index is opened and the root is checked against its ingest
    bookkeeping before anything is read: a root the index has not fully
    ingested cannot say what it holds, so it is refused rather than read short.
    A URL that cannot be opened fails here; nothing falls back to walking the
    tree, which would turn a misconfigured run into a slow, silently different
    one.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.
        results_db_url: Connection URL of the results index, or None to read
            the tree.

    Returns:
        The source, which the caller closes when it is done with it.

    Raises:
        ValueError: If the index cannot be opened, is not an index, or was
            written by another version of the schema; or if the root has no
            completed ingest run in it.
    """
    if results_db_url is None:
        return TreeDocumentSource(nav_results_root)
    root_url = normalize_root_url(nav_results_root)
    engine = open_index(results_db_url, create=False)
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, [root_url], url=results_db_url)
    except Exception:
        engine.dispose()
        raise
    return IndexDocumentSource(engine, root_url, results_db_url)
