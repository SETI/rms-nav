"""Where a program gets its navigation records, whichever storage holds them.

A navigation pass writes one document per image.  Five programs read those
records back, and each of them can read either the documents or an ingested
results index: reading a document is one file read per image, which on a cloud
root is one paid round trip per image and a Cassini-scale root holds several
hundred thousand.  This module is the seam between the two storages.  It answers
in two shapes, because the programs ask in two shapes:

* **One image**, by its results path stub.  The reprojection and backplane
  stages ask this inside a per-image loop, so the index answers it in one
  statement over the primary key.
* **One mission**, in bulk.  The kernel writer asks this once per run and then
  filters by time, so the index answers it in two statements per run -- one for
  the images, one for the files the ingest refused -- rather than in one file
  read per image.  Two statements per run against several hundred thousand round
  trips is the whole point; the count of statements is not.

:class:`RecordSource` names the seam, :class:`TreeRecordSource` reads documents
and :class:`IndexRecordSource` reads rows.  Neither decides anything about a
record it hands back: the classification that reads a pointing out of one, the
eligibility that reads a status, and the arithmetic that reads a matrix are the
caller's, unchanged from one storage to the other.  What makes that safe is that
both storages answer in one shape, rebuilt through the one column-to-field
correspondence in :mod:`spindoctor.results_index.rebuild`.

Both sources answer both shapes.  A consumer that reads one image today and a
mission tomorrow does not acquire a second seam, and a consumer written against
either shape works over both storages the day it is written.

What differs between the two storages, and what may not
-------------------------------------------------------

**A record read from a document carries every field the document has; one
rebuilt from a row carries the columns its consumer selected.**  That is what
makes a row cheap, and it is why a consumer names its columns rather than
selecting the whole table: nobody reads forty fields.  A consumer that reads a
field it did not select reads it as absent, which is why the columns a consumer
names are pinned by a test rather than left to be noticed in production.

**A value the ingest could not store is rebuilt as absent**, which is the one
class of difference the seam cannot close.  It belongs to what a column can
hold; :mod:`spindoctor.results_index.rebuild` states it, and each consumer's
documentation says what it does about it.

**A file the ingest refused is not a row.**  It is recorded in ``failed_files``,
and it is neither an image that was never navigated nor one whose record can be
read: the document may well record a perfectly good pointing, and reading the
refusal as absence would build a corrected product from the documents and an
uncorrected one from the index without saying so.  Both shapes therefore report
a refusal as a refusal.  The per-image shape fails the image, naming the stub,
the index and the reason the ingest recorded.  The bulk shape reports every
refused file under the root as one it could not read, with that reason, which is
what the walk does with a document it cannot attribute to a mission.

**Nothing falls back.**  A URL that cannot be opened, or a root nobody has
ingested, fails the run rather than quietly reading the tree instead.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Engine

from spindoctor.results_index.engine import reporting_a_failed_read
from spindoctor.results_index.masking import masked_url
from spindoctor.results_index.rebuild import record_from_row
from spindoctor.results_index.roots import normalize_root_url, open_index_for_roots
from spindoctor.results_index.schema import FAILED_FILES, IMAGES
from spindoctor.support.nav_document import (
    METADATA_SUFFIX,
    read_document,
    read_documents,
    resolved_document_path,
)
from spindoctor.support.nav_record import NavRecord

__all__ = [
    'IndexRecordSource',
    'RecordSource',
    'TreeRecordSource',
    'build_record_source',
]


class RecordSource(Protocol):
    """One results root's navigation records, however they are stored.

    Implementations answer for exactly one root: the document-backed one reads
    under it and the index-backed one filters on it, so neither can serve a
    record belonging to another root.
    """

    def read_record(self, results_path_stub: str) -> dict[str, Any]:
        """Return the navigation record of one image.

        Parameters:
            results_path_stub: The image's stub, which is its identity under the
                root.

        Returns:
            The record, as a mapping.

        Raises:
            FileNotFoundError: If nothing recorded this image.
            ValueError: If something recorded this image and the source cannot
                say what it recorded.
        """
        ...

    def read_records(self, mission: str) -> tuple[list[NavRecord], list[tuple[FCPath, str]]]:
        """Return one mission's records, and the files that could not be read.

        Parameters:
            mission: The instrument identity to keep.

        Returns:
            The records, ordered by the path of the document each stands for,
            and one entry per file that could not be read at all, pairing it
            with why.
        """
        ...

    def describe(self) -> str:
        """Return where these records came from, for the run log.

        Returns:
            The results root, and the index the records were read out of when
            they were read out of one.  Any password in an index URL is masked.
        """
        ...

    def close(self) -> None:
        """Release whatever the source holds open."""
        ...


class TreeRecordSource:
    """The records as documents under a results root.

    Parameters:
        nav_results_root: The navigation results root to read.
    """

    def __init__(self, nav_results_root: str | Path | FCPath) -> None:
        self._root = FCPath(nav_results_root)

    def read_record(self, results_path_stub: str) -> dict[str, Any]:
        """Read one image's document.

        Parameters:
            results_path_stub: The image's stub.

        Returns:
            The document.

        Raises:
            FileNotFoundError: If the document is not there, or if the stub does
                not name a path this root may be read at.  Both mean the same
                thing to a caller -- this image has no readable record here --
                and the message says which of the two it was.
            ValueError: If the file is not valid JSON, or is valid JSON that is
                not an object.
        """
        resolved = resolved_document_path(self._root, results_path_stub)
        if resolved.path is None:
            raise FileNotFoundError(
                f'{results_path_stub}: does not name a navigation document under '
                f'{self._root} ({resolved.refusal}), so none can be read for this image'
            )
        return read_document(resolved.path)

    def read_records(self, mission: str) -> tuple[list[NavRecord], list[tuple[FCPath, str]]]:
        """Walk the root and read every document of one mission.

        Parameters:
            mission: The instrument identity to keep.

        Returns:
            The records and the unreadable files, as :func:`read_documents`
            produces them.
        """
        return read_documents(self._root, mission)

    def describe(self) -> str:
        """Return the root the documents were read from.

        Returns:
            The root as a POSIX path.
        """
        return self._root.as_posix()

    def close(self) -> None:
        """Release nothing: reading documents holds nothing open."""


class IndexRecordSource:
    """The records as rows of a results index.

    The columns are the consumer's, because a row is only cheaper than a
    document while it carries less: a per-image lookup that dragged back every
    JSON column would spend on the matrix and the kernel list what it saved on
    the round trip.

    Parameters:
        engine: The open index, which this source disposes of when it is closed.
        root_url: Normalized URL of the results root whose rows to read.
        url: The index URL, kept for the messages that name it.
        columns: The columns of ``images`` this consumer reads.  Each must be a
            column :mod:`spindoctor.results_index.rebuild` knows a place for, or
            the rebuilt record silently lacks the field it was selected for.
    """

    def __init__(
        self,
        engine: Engine,
        root_url: str,
        url: str,
        columns: Sequence[sqlalchemy.Column[Any]],
    ) -> None:
        self._engine = engine
        self._root_url = root_url
        self._url = masked_url(url)
        self._raw_url = url
        self._root = FCPath(root_url)
        self._columns = tuple(columns)
        # An index is a snapshot of its last ingest, so a row can be absent
        # because nothing navigated the image or because the image was navigated
        # after that ingest.  Neither the row nor its absence can say which, so
        # the message says what was searched and leaves the reader able to tell.
        self._storage = (
            f'the results index {self._url}, a snapshot of its last ingest of {root_url}'
        )

    def read_record(self, results_path_stub: str) -> dict[str, Any]:
        """Rebuild one image's navigation record from its row.

        Parameters:
            results_path_stub: The image's stub.

        Returns:
            The rebuilt record.

        Raises:
            FileNotFoundError: If the index holds no row for this stub under this
                root, naming both and the index, and saying that the index is a
                snapshot: the row is absent either because nothing navigated the
                image or because it was navigated after the last ingest, and the
                message names the snapshot so the two can be told apart.  A
                missing document raises the same way, so the caller reports both
                the same way.
            ValueError: If the index records the document for this stub as one
                the ingest refused.  Deliberately not the same exception: a
                caller reports a missing record as an image nothing navigated,
                and this image was navigated -- the index simply cannot say what
                it recorded.
        """
        row = self._row(results_path_stub)
        if row.record_stub is None:
            self._refuse_a_document_the_ingest_refused(results_path_stub, row.refusal_reason)
            raise FileNotFoundError(
                f'{results_path_stub}: no navigation record for this image in {self._storage}'
            )
        return record_from_row(row)

    def read_records(self, mission: str) -> tuple[list[NavRecord], list[tuple[FCPath, str]]]:
        """Read one mission's rows, and the files the ingest refused.

        Both statements are keyed by the root as well as by what they select.
        One index serves several results roots, and a query that asked only for
        the mission would write another root's images into this run's products.

        A refused file names no mission, since the ingest could not read one out
        of it, so it is reported whichever mission is being read -- which is
        what the walk does with a document it cannot attribute to a mission.

        Parameters:
            mission: The instrument identity to keep, matched against the
                ``instrument`` column exactly as the walk matches it against
                ``observation.instrument``.

        Returns:
            The records, ordered by the path of the document each stands for as
            the walk orders them, and one entry per file the ingest refused
            under this root, pairing it with the recorded reason.

        Raises:
            ValueError: If the index cannot be read, naming it with any password
                masked.
        """
        # Neither statement orders: ordering is done below, on the rebuilt
        # paths.  A server sorts text under its own collation, and a locale
        # collation orders a separator against an underscore differently from
        # the codepoint order the walk sorts its paths by, so an ORDER BY here
        # would hand back one order from SQLite and another from PostgreSQL for
        # the same tree.  Sorting the paths is the one key the two storages
        # share, and it is what lets a run be held to the walk on any backend.
        images = sqlalchemy.select(*self._bulk_columns()).where(
            IMAGES.c.root_url == self._root_url, IMAGES.c.instrument == mission
        )
        refused = sqlalchemy.select(FAILED_FILES.c.results_path_stub, FAILED_FILES.c.reason).where(
            FAILED_FILES.c.root_url == self._root_url
        )
        with reporting_a_failed_read(self._raw_url), self._engine.connect() as connection:
            records = [self._record_of(row) for row in connection.execute(images)]
            unreadable = [
                (self._path_of(str(row.results_path_stub)), str(row.reason))
                for row in connection.execute(refused)
            ]
        records.sort(key=lambda record: record.path.as_posix())
        unreadable.sort(key=lambda entry: entry[0].as_posix())
        return records, unreadable

    def describe(self) -> str:
        """Return the root and the index the records were read out of.

        Returns:
            The root, followed by the index URL with any password masked.
        """
        return f'{self._root_url} in the results index {self._url}'

    def close(self) -> None:
        """Dispose of the engine, closing every connection it pooled."""
        self._engine.dispose()

    def _bulk_columns(self) -> tuple[sqlalchemy.ColumnElement[Any], ...]:
        """Return the columns a bulk read selects.

        The stub and the recorded source file are added to the consumer's own
        columns rather than asked of it: a bulk read hands back records paired
        with where each is kept, so it needs both whatever the consumer reads.

        Returns:
            The columns, with the two added ones first and neither repeated if
            the consumer named it too.
        """
        added = (IMAGES.c.results_path_stub, IMAGES.c.source_file)
        names = {column.name for column in added}
        return (*added, *(column for column in self._columns if column.name not in names))

    def _row(self, results_path_stub: str) -> sqlalchemy.Row[Any]:
        """Read what the index holds about one image, from both of its tables.

        One query rather than a record lookup followed by a refusal lookup: an
        image with no record is the common case on a partially navigated root,
        and it is the case that would pay the second round trip -- against the
        stage whose whole purpose is removing one per image.  The key is selected
        as a row of its own and both tables are joined onto it, so exactly one
        row comes back whether the index holds a record, a refusal or neither.

        Parameters:
            results_path_stub: The image's stub.

        Returns:
            The row.  ``record_stub`` carries the stub when the index holds a
            navigation record for it and nothing otherwise, and
            ``refusal_reason`` carries the recorded reason when the index holds a
            refusal for it and nothing otherwise; a stub the index knows nothing
            about answers to neither.  Both halves are read rather than one,
            because a stub in both tables is a record with a stale refusal beside
            it and must be read as the record it is.

        Raises:
            ValueError: If the index cannot be read at all -- a lost connection,
                a table the account may not read, a partially restored database.
                Translated here for the same reason the selection seam translates
                it: a caller of this module reports the failure against one
                image, and the database layer's own exception types are ones it
                cannot name.
        """
        key = sqlalchemy.select(
            sqlalchemy.literal(self._root_url, sqlalchemy.Text).label('root_url'),
            sqlalchemy.literal(results_path_stub, sqlalchemy.Text).label('results_path_stub'),
        ).subquery()
        statement = (
            sqlalchemy.select(
                IMAGES.c.results_path_stub.label('record_stub'),
                FAILED_FILES.c.reason.label('refusal_reason'),
                *self._columns,
            )
            .select_from(key)
            .outerjoin(
                IMAGES,
                sqlalchemy.and_(
                    IMAGES.c.root_url == key.c.root_url,
                    IMAGES.c.results_path_stub == key.c.results_path_stub,
                ),
            )
            .outerjoin(
                FAILED_FILES,
                sqlalchemy.and_(
                    FAILED_FILES.c.root_url == key.c.root_url,
                    FAILED_FILES.c.results_path_stub == key.c.results_path_stub,
                ),
            )
        )
        with reporting_a_failed_read(self._raw_url), self._engine.connect() as connection:
            row = connection.execute(statement).first()
        # The key is selected as a row of its own and both tables are joined onto
        # it, so the statement answers with one row for every stub, including one
        # neither table knows.
        assert row is not None
        return row

    def _refuse_a_document_the_ingest_refused(self, stub: str, reason: Any) -> None:
        """Fail an image whose document the ingest recorded as unreadable.

        Parameters:
            stub: The stub whose record was not found.
            reason: What the index records as the reason it could not read that
                image's document, or None when it records no refusal for it.

        Raises:
            ValueError: If the index records this stub as a document the ingest
                refused, naming the stub, the index and the recorded reason.  A
                refusal means the index cannot answer for this image, which is a
                different fact from nothing having navigated it, and reading the
                one as the other builds a product from the document under one
                storage and from uncorrected pointing under the other.
        """
        if reason is None:
            return
        raise ValueError(
            f'{stub}: {self._storage} records the navigation document for this image as '
            f'one the ingest could not read ({reason}), so the index cannot say what it '
            f'recorded. Read the navigation documents instead, or fix the document and '
            f'ingest that root again.'
        )

    def _path_of(self, stub: str) -> FCPath:
        """Return where the document of one stub lives under this root.

        Parameters:
            stub: The image's results path stub.

        Returns:
            The document's location, which is the stub under the root with the
            document suffix restored.
        """
        return self._root / f'{stub}{METADATA_SUFFIX}'

    def _record_of(self, row: sqlalchemy.Row[Any]) -> NavRecord:
        """Rebuild one row's record, with the document it stands for.

        Parameters:
            row: One row of the index, carrying the consumer's columns and the
                two a bulk read adds.

        Returns:
            The record.
        """
        stub = str(row.results_path_stub)
        # The path the ingest recorded reading, so a message about this record
        # names the file an operator would open.  A row written before anything
        # recorded one falls back to where the stub says it lives.
        path = self._path_of(stub) if row.source_file is None else FCPath(str(row.source_file))
        return NavRecord(path=path, stub=stub, metadata=record_from_row(row))


def build_record_source(
    nav_results_root: str | Path | FCPath,
    *,
    results_db_url: str | None,
    columns: Sequence[sqlalchemy.Column[Any]],
) -> RecordSource:
    """Build the source a run reads its navigation records through.

    With no index URL the source reads documents, which is every program's
    default.  With one, the index is opened and the root is checked against its
    ingest bookkeeping before anything is read: a root the index has not fully
    ingested cannot say what it holds, so it is refused rather than read short.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.
        results_db_url: Connection URL of the results index, or None to read the
            documents.
        columns: The columns of ``images`` this consumer reads.  Ignored when the
            documents are read, which carry every field whatever is selected.

    Returns:
        The source, which the caller closes when it is done with it.

    Raises:
        ValueError: If the index cannot be opened, is not an index, or was
            written by another version of the schema; or if the root has no
            completed ingest run in it.
    """
    if results_db_url is None:
        return TreeRecordSource(nav_results_root)
    root_url = normalize_root_url(nav_results_root)
    engine = open_index_for_roots(results_db_url, [root_url])
    return IndexRecordSource(engine, root_url, results_db_url, columns)
