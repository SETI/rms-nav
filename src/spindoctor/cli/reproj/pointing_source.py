"""Where a reprojection or backplane reader gets one image's navigation record.

Both readers need the same two things per image: the whole record (the backplane
stage reads its status before it decides there is work to do) and the classified
pointing that record supplies.  How those arrive differs.  Reading one
``_metadata.json`` per image costs one round trip per image on a cloud root,
which a Cassini-scale run pays several hundred thousand times; an ingested
results index answers the same questions with one row.  So the seam is explicit:
:class:`PointingSource` names it, :class:`FilePointingSource` reads documents,
and :class:`IndexPointingSource` reads rows.

Neither implementation classifies anything itself.  The index-backed one rebuilds
the shape of the document from the row and hands it to the one classifier,
:func:`spindoctor.cli.reproj.offsets.select_pointing`, so the two paths cannot
disagree about which pointing a record supplies -- there is only one ladder, and
it is the same code in both modes.  That holds only while the columns the
rebuild reads say what the document's own fields said, which is why ingest fills
every one of them through :mod:`spindoctor.support.nav_record`, the module these
readers read the same fields through.  A column filled by a rule of its own,
even one that agreed when it was written, is a second reader of the record.

What the rebuilt record carries
-------------------------------

Only the fields the index holds for these two readers: the top-level ``status``,
``status_error`` and ``offset``, and the ``navigation_result`` ``times`` and
``pointing`` blocks the C-matrix mechanism needs.  It is not the document; it is
the part of the document that decides a pointing.  The ``pointing`` block's
``camera_frame`` name is left out deliberately, though the index carries it for
another reader: the frame identity a recorded attitude is gated against is taken
from the observation, never from the record, so nothing here would consult a
rebuilt name.

A field the row does not carry is rebuilt as a field the document did not have,
rather than as one holding null, because the record is read as a document would
be.  ``offset`` is the one exception: it is rebuilt as a key holding null,
because ingest stores an absent, null, malformed and non-finite offset alike as
NULL and the rebuild has to render the pair as one of them.  Which one is
chosen makes no difference to the pointing, since none of the four supplies
one; it does decide the name the shortfall is counted under, and the null-valued
key is what makes ``null_offset`` the row's reason for all four, as class 1
below states.  The ``status`` column
is NOT NULL and stands in for a document that named no outcome with
:data:`~spindoctor.support.nav_record.UNKNOWN_STATUS`; that value is rebuilt as
the absent field it stands for, and both are then read as naming no outcome by
:func:`~spindoctor.support.nav_record.record_status`, so a document naming that
word for itself and one naming nothing are reported alike rather than one of
them as nothing.

Where the two paths differ
--------------------------

The rule this seam is held to: **a record the two storages classify differently
may differ in the reason and in nothing else.**  The reason is a name a
run-level tally counts under; the mechanism, the matrices, the midtime and the
offset are what a product is built from.  A difference in any of those is a
defect in this module or in what ingest stores, not an entry for the list below.

A row is not a file, and ingest has already refused some documents.  Every
reason the file path can report either has an index-path equivalent or is
unreachable there, and an unreachable one surfaces under whichever reason
describes the row that ingest actually wrote:

+---------------------------------------+------------------------------------------+
| File-path reason                      | Index path                               |
+=======================================+==========================================+
| ``no_metadata``                       | no row for the stub under this root      |
+---------------------------------------+------------------------------------------+
| ``navigation_did_not_succeed``        | row whose ``status`` is not ``success``  |
+---------------------------------------+------------------------------------------+
| ``null_offset``                       | success row with ``offset_dv`` or        |
|                                       | ``offset_du`` NULL                       |
+---------------------------------------+------------------------------------------+
| ``no_pointing_block``                 | row with no pointing column set          |
+---------------------------------------+------------------------------------------+
| ``no_cmatrix_rotation_fitted``        | row with a pointing column set and       |
|                                       | ``cmatrix`` NULL                         |
+---------------------------------------+------------------------------------------+
| ``malformed_pointing``                | row carrying a ``cmatrix`` the validator |
|                                       | refuses, or one with no ``midtime_et``   |
+---------------------------------------+------------------------------------------+
| ``pool_already_corrected`` and the    | identical: they are decided when the     |
| gate reasons                          | selection is applied, from the three     |
|                                       | recorded values both paths carry         |
+---------------------------------------+------------------------------------------+
| ``unusable_metadata_path``            | unreachable: a stub is a key, not a path |
+---------------------------------------+------------------------------------------+
| ``unreadable_metadata``,              | unreachable: ingest refused such a file, |
| ``invalid_json``,                     | so it has no record row and a refusal    |
| ``metadata_not_an_object``            | row instead, and the lookup fails the    |
|                                       | image rather than classifying it         |
+---------------------------------------+------------------------------------------+
| ``missing_offset_key``,               | reported as ``null_offset``: one column  |
| ``invalid_offset_type``,              | pair holds all five, and none of them    |
| ``non_finite_offset``,                | supplies a pointing                      |
| ``malformed_offset``                  |                                          |
+---------------------------------------+------------------------------------------+

A document the ingest refused
-----------------------------

Ingest reads a file it cannot make a navigation record of as a refusal rather
than as a record, and writes it to ``failed_files`` instead of to ``images``:
a document naming no ``observation.instrument`` or no ``image_name``, one
whose declared containers are of another shape, one naming a technique twice,
and any other file the converter cannot read whole.  The file path reads such a
document perfectly well and supplies whatever pointing it records.  So a lookup
that found no record row and stopped there would report a navigated image as one
nothing navigated, and would build a corrected product through documents and an
uncorrected one through the index without saying so.

The lookup therefore asks both tables at once -- one query, the stub and root
selected as a row of their own with each table joined onto it -- and a stub
recorded as a refusal fails the image, naming the stub, the index and the reason
the ingest recorded.  Both halves in one query rather than a second lookup where
the first found nothing: an image with no record is the common case on a
partially navigated root, and it is exactly that image that would pay the extra
round trip the index exists to remove.  A refusal is an answer the index cannot
give, which is a different fact from "no such image was navigated", and the two
are reported differently.  The image is refused rather than quietly read from
its document:
a source that fell back to files for some images would make ``--results-db``
mean a different thing per image, and one round trip per image is the cost the
index exists to remove.  Failing one image does not fail the run -- both
consumers contain a per-image failure -- so a root whose ingest refused some of
its documents is re-ingested rather than worked around.

The one refusal ingest deliberately records nowhere is a file it could not
retrieve: nothing is known about it that will still be true next pass, and a
recorded refusal is skipped for as long as the file does not change.  Such a
file has no row in either table and reads as an image nothing navigated.

The three classes read differently
----------------------------------

Three classes of record the index does hold are classified under a different
reason by the two storages, and that the reason is the whole of the difference
is asserted rather than observed: a member whose mechanism, matrices, midtime or
offset differed would be a defect in this module or in what ingest stores, not a
fourth class.  The membership is measured rather than argued -- both sources are
driven over every shape a record's fields can take, and what survives defines it
-- and each member is a shape no navigation produces, named here so that a
record hand-built into a results tree is not read as agreeing when it does not.

1. **An ``offset`` no reader can use.**  Absent, null, a boolean pair, a
   non-finite pair, or anything else that is not two values convertible to
   finite pixels.  The document is classified under which of those it was;
   the row, which holds one NULL pair for all of them, under ``null_offset``.
2. **A ``cmatrix`` no column can hold** -- one whose recorded value is not one
   3x3 matrix of finite real numbers in some nesting an array library
   reconciles into that shape.  Nine values, a 3x3 nesting of them and nine
   rows of one all denote the same matrix and are all held; a value of any
   other shape, and one whose nine entries are not finite real numbers, is
   held by neither storage.
   The document is ``malformed_pointing``; the row is
   ``no_cmatrix_rotation_fitted`` when something else of the block survives
   (which is what a fitted-rotation result looks like) and ``no_pointing_block``
   when nothing does.  The file path also puts one line in the run log for it
   and the index path does not.  A ``cmatrix`` that *is* nine finite numbers and
   is not a rotation is stored, and the validator then refuses it in both paths
   alike, which is why ``malformed_pointing`` has a row of its own above.
3. **A ``pointing`` block none of whose four columned fields survives** -- one
   holding only ``camera_frame``, or frame identities written as floats or
   booleans, which the integer columns refuse.  The document is
   ``no_cmatrix_rotation_fitted``, because the block exists and carries no
   corrected attitude; the row is ``no_pointing_block``, because the block left
   no trace in it.  The block a navigation writes always carries the baseline
   and both frame identities as integers.

Everything else agrees.  For every record the navigator wrote and ingest stored,
and for every hand-built shape outside those three classes, the mechanism, the
matrices, the midtime, the offset, the outcome and the error are identical in
the two paths.
"""

import json
from typing import Any, Protocol

import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Engine

from spindoctor.cli.reproj.offsets import (
    NO_METADATA,
    NO_METADATA_MESSAGE,
    PointingSelection,
    load_pointing_if_any,
    none_selection,
    resolved_nav_metadata_path,
    select_pointing,
)
from spindoctor.config import IMAGE_LOGGER
from spindoctor.dataset.dataset import ImageFile
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
    'FilePointingSource',
    'IndexPointingSource',
    'PointingSource',
    'build_pointing_source',
]

_ROW_COLUMNS = (
    IMAGES.c.status,
    IMAGES.c.status_error,
    IMAGES.c.offset_dv,
    IMAGES.c.offset_du,
    IMAGES.c.start_et,
    IMAGES.c.stop_et,
    IMAGES.c.midtime_et,
    IMAGES.c.exposure_s,
    IMAGES.c.sclk_start,
    IMAGES.c.sclk_midtime,
    IMAGES.c.sclk_stop,
    IMAGES.c.camera_frame_id,
    IMAGES.c.ck_frame_id,
    IMAGES.c.cmatrix,
    IMAGES.c.cmatrix_original,
)
"""Every column a rebuilt navigation record is made of, read in one SELECT."""


class PointingSource(Protocol):
    """One image's navigation record, however it is stored.

    Implementations answer for exactly one results root: the file-backed one
    reads under it, and the index-backed one filters on it, so neither can serve
    a record belonging to another root.
    """

    def read_record(self, image_file: ImageFile) -> dict[str, Any]:
        """Return the navigation record for one image.

        Parameters:
            image_file: The image to look up.

        Returns:
            The record, as a mapping.

        Raises:
            FileNotFoundError: If nothing recorded this image.
            ValueError: If something recorded this image and the source cannot
                say what it recorded.
        """
        ...

    def load_pointing(self, image_file: ImageFile) -> PointingSelection:
        """Return the pointing one image's navigation record supplies.

        A record that supplies none is not an error: the selection carries the
        reason instead, and the caller reports and counts it.

        Parameters:
            image_file: The image to look up.

        Returns:
            The classified selection.

        Raises:
            ValueError: If something recorded this image and the source cannot
                say what it recorded, which is not a record supplying no
                pointing and must not be counted as one.
        """
        ...

    def close(self) -> None:
        """Release whatever the source holds open."""
        ...


class FilePointingSource:
    """A navigation record read from its ``_metadata.json`` document.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under, or None
            to look for no pointing at all -- which is what a reprojection run
            given no navigation results does, and is a choice rather than a
            shortfall.
    """

    def __init__(self, nav_results_root: str | FCPath | None) -> None:
        """Record the root this source reads its documents under."""
        self._nav_results_root = nav_results_root

    def read_record(self, image_file: ImageFile) -> dict[str, Any]:
        """Read and parse one image's metadata document.

        Parameters:
            image_file: The image to look up.

        Returns:
            The parsed document.

        Raises:
            FileNotFoundError: If the document does not exist; if the stub does
                not name a path under the root, which a stub carrying a null
                byte, an absolute fragment or a ``..`` escape does not; or if
                this source was built with no navigation results root and
                therefore has nowhere to look. All three mean the same thing to
                the caller -- this image has no readable record -- and the
                image's log carries which of them it was.
            ValueError: If the document is not valid JSON, or is valid JSON that
                is not an object.
        """
        if self._nav_results_root is None:
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: no navigation results root was resolved, '
                f'so no navigation record can be read for this image'
            )
        # Resolved through the one guard both readers share rather than joined
        # here: the class would otherwise apply different rules about which
        # paths a results root may be read at depending on which of its two
        # methods was called.
        metadata_file = resolved_nav_metadata_path(self._nav_results_root, image_file)
        if metadata_file is None:
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: does not name a navigation record under '
                f'{self._nav_results_root}, so none can be read for this image'
            )
        return _json_object_from_text(metadata_file.read_text(), source=str(metadata_file))

    def load_pointing(self, image_file: ImageFile) -> PointingSelection:
        """Load and classify one image's recorded pointing from its document.

        Parameters:
            image_file: The image to look up.

        Returns:
            The classified selection, carrying the reason for every way the
            document could fail to supply a pointing.
        """
        return load_pointing_if_any(self._nav_results_root, image_file)

    def close(self) -> None:
        """Release nothing: a file-backed source holds nothing open."""


class IndexPointingSource:
    """A navigation record rebuilt from one row of the results index.

    One lookup is one SELECT on the index's primary key, filtered on the root as
    well as the stub: one index can hold several results roots, and two roots may
    hold the same stub.

    Parameters:
        engine: An open index, which this source disposes of when it is closed.
        root_url: The normalized results root whose rows this source answers
            from.
    """

    def __init__(self, engine: Engine, root_url: str) -> None:
        """Take the open index and the root this source answers from.

        The index is named for messages here rather than by each caller, and is
        rendered with its credentials hidden: these messages reach run logs and
        bug reports, and a connection URL can carry a database password.
        """
        self._engine = engine
        self._root_url = root_url
        self._url = masked_url(engine.url.render_as_string(hide_password=False))
        # An index is a snapshot of its last ingest, so a row can be absent
        # because nothing navigated the image or because the image was
        # navigated after that ingest.  Neither the row nor its absence can say
        # which, so the message says what was searched and leaves the reader
        # able to tell.
        self._storage = (
            f'the results index {self._url}, a snapshot of its last ingest of {root_url}'
        )

    def _row(self, image_file: ImageFile) -> sqlalchemy.Row[Any]:
        """Read what the index holds about one image, from both of its tables.

        One query rather than a record lookup followed by a refusal lookup: an
        image with no record is the common case on a partially navigated root,
        and it is the case that would pay the second round trip -- against the
        stage whose whole purpose is removing one per image.  The key is
        selected as a row of its own and both tables are joined onto it, so
        exactly one row comes back whether the index holds a record, a refusal
        or neither.

        Parameters:
            image_file: The image to look up.

        Returns:
            The row.  ``record_stub`` carries the stub when the index holds a
            navigation record for it and nothing otherwise, and
            ``refusal_reason`` carries the recorded reason when the index holds
            a refusal for it and nothing otherwise; a stub the index knows
            nothing about answers to neither.  Both halves are read rather than
            one, because a stub in both tables is a record with a stale refusal
            beside it and must be read as the record it is.

        Raises:
            ValueError: If the index cannot be read at all -- a lost
                connection, a table the account may not read, a partially
                restored database.  Translated here for the same reason the
                selection seam translates it: a caller of this module reports
                the failure against one image, and the database layer's own
                exception types are ones it cannot name.
        """
        key = sqlalchemy.select(
            sqlalchemy.literal(self._root_url, sqlalchemy.Text).label('root_url'),
            sqlalchemy.literal(image_file.results_path_stub, sqlalchemy.Text).label(
                'results_path_stub'
            ),
        ).subquery()
        statement = (
            sqlalchemy.select(
                IMAGES.c.results_path_stub.label('record_stub'),
                FAILED_FILES.c.reason.label('refusal_reason'),
                *_ROW_COLUMNS,
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
        url = self._engine.url.render_as_string(hide_password=False)
        with reporting_a_failed_read(url), self._engine.connect() as connection:
            row = connection.execute(statement).first()
        # The key is selected as a row of its own and both tables are joined
        # onto it, so the statement answers with one row for every stub,
        # including one neither table knows.
        assert row is not None
        return row

    def _refuse_a_document_the_ingest_refused(self, image_file: ImageFile, reason: Any) -> None:
        """Fail an image whose document the ingest recorded as one it could not read.

        Parameters:
            image_file: The image whose record was not found.
            reason: What the index records as the reason it could not read that
                image's document, or None when it records no refusal for it.

        Raises:
            ValueError: If the index records this stub as a document the ingest
                refused, naming the stub, the index and the recorded reason. A
                refusal means the index cannot answer for this image, which is
                a different fact from nothing having navigated it, and reading
                the one as the other builds a product from the document under
                one storage and from uncorrected pointing under the other.
        """
        if reason is None:
            return
        raise ValueError(
            f'{image_file.results_path_stub}: {self._storage} records the navigation '
            f'document for this image as one the ingest could not read ({reason}), so '
            f'the index cannot say what it recorded. Read the navigation documents '
            f'instead, or fix the document and ingest that root again.'
        )

    def read_record(self, image_file: ImageFile) -> dict[str, Any]:
        """Rebuild one image's navigation record from its row.

        Parameters:
            image_file: The image to look up.

        Returns:
            The rebuilt record.

        Raises:
            FileNotFoundError: If the index holds no row for this stub under this
                root, naming both and the index, and saying that the index is a
                snapshot: the row is absent either because nothing navigated
                the image or because it was navigated after the last ingest,
                and the message names the snapshot so the two can be told
                apart. A missing document raises the same way, so the caller
                reports both the same way.
            ValueError: If the index records the document for this stub as one
                the ingest refused. Deliberately not the same exception: a
                caller reports a missing record as an image nothing navigated,
                and this image was navigated -- the index simply cannot say
                what it recorded.
        """
        row = self._row(image_file)
        if row.record_stub is None:
            self._refuse_a_document_the_ingest_refused(image_file, row.refusal_reason)
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: no navigation record for this image in '
                f'{self._storage}'
            )
        return _record_from_row(row)

    def load_pointing(self, image_file: ImageFile) -> PointingSelection:
        """Read and classify one image's recorded pointing from its row.

        Parameters:
            image_file: The image to look up.

        Returns:
            The classified selection, from the same classifier the file-backed
            source uses.

        Raises:
            ValueError: If the index records the document for this stub as one
                the ingest refused. A record that supplies no pointing is
                classified and counted; a document the index cannot answer for
                is neither, because the same document read as a file may well
                supply one and the product would differ in silence.
        """
        row = self._row(image_file)
        if row.record_stub is None:
            self._refuse_a_document_the_ingest_refused(image_file, row.refusal_reason)
            IMAGE_LOGGER.warning(NO_METADATA_MESSAGE, image_file.image_file_url, self._storage)
            return none_selection(NO_METADATA)
        return select_pointing(_record_from_row(row), subject=image_file.image_file_url.as_posix())

    def close(self) -> None:
        """Dispose of the engine, closing every connection it pooled."""
        self._engine.dispose()


def _json_object_from_text(text: str, *, source: str) -> dict[str, Any]:
    """Parse text as a JSON object.

    Parameters:
        text: The document text.
        source: Path or URL of the document, for the refusal message.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the text is not valid JSON, or is valid JSON that is not
            an object. ``json.JSONDecodeError`` is itself a ``ValueError``, so
            one type covers both.
    """
    record = json.loads(text)
    if not isinstance(record, dict):
        raise ValueError(f'{source}: navigation metadata is not a JSON object')
    return record


def _present(values: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Return the named values that are not NULL, as one block of a record.

    A column the row does not carry is a field the document did not have, and a
    reader distinguishes an absent field from a present one, so an absent value
    is left out rather than written as null.

    Parameters:
        values: Field names paired with what the row holds for each.

    Returns:
        The fields that carry a value, in the order they were given.
    """
    return {name: value for name, value in values if value is not None}


def _record_from_row(row: sqlalchemy.Row[Any]) -> dict[str, Any]:
    """Rebuild the navigation record one index row records.

    The result carries the fields the pointing classifier and the backplane
    status check read, in the shapes the navigator writes them: the top-level
    ``status``, ``status_error`` and ``offset``, and the ``times`` and
    ``pointing`` blocks of ``navigation_result``.

    Parameters:
        row: One row of the index, carrying every column of ``_ROW_COLUMNS``.

    Returns:
        The rebuilt record.
    """
    # The offset key is always written, with a null value where the row carries
    # no usable pair.  Ingest stores an absent, null, malformed and non-finite
    # offset alike as NULL, so the rebuild has to render the pair as one of
    # them.  None of the four supplies a pointing, so the choice cannot change a
    # product; it does decide the name the shortfall is counted under, and this
    # one is what makes ``null_offset`` the row's reason for all of them rather
    # than ``missing_offset_key``.
    offset: list[float] | None = None
    if row.offset_dv is not None and row.offset_du is not None:
        offset = [row.offset_dv, row.offset_du]
    # ``status`` and ``status_error`` are rendered back as the fields they
    # stand for, absent ones included, so the record reads as the document it
    # came from.  The status column is NOT NULL and records a document that
    # named no outcome as ``UNKNOWN_STATUS``, which is that document's absent
    # field and is rebuilt as one; a document naming that same word for itself
    # is rebuilt without the field too, and both are then read as naming no
    # outcome by the one function every consumer reads the field through.
    record: dict[str, Any] = _present((('status_error', row.status_error),))
    if row.status != UNKNOWN_STATUS:
        record['status'] = row.status
    record['offset'] = offset
    navigation_result: dict[str, Any] = {}
    times = _present(
        (
            ('start_et', row.start_et),
            ('stop_et', row.stop_et),
            ('midtime_et', row.midtime_et),
            ('exposure_s', row.exposure_s),
            ('sclk_start', row.sclk_start),
            ('sclk_midtime', row.sclk_midtime),
            ('sclk_stop', row.sclk_stop),
        )
    )
    if times:
        navigation_result['times'] = times
    # The pointing block exists whenever any of its columns does, not only when
    # the corrected attitude does.  Its producer writes the baseline and the
    # frame identities for every navigated image and the corrected attitude only
    # where one was computed, so a block with none of the four is a record that
    # had no pointing block, and a block missing only the corrected attitude is a
    # result that fitted a camera rotation.  Keying on the corrected attitude
    # alone cannot tell those two apart.
    pointing = _present(
        (
            ('cmatrix', row.cmatrix),
            ('cmatrix_original', row.cmatrix_original),
            ('camera_frame_id', row.camera_frame_id),
            ('ck_frame_id', row.ck_frame_id),
        )
    )
    if pointing:
        navigation_result['pointing'] = pointing
    if navigation_result:
        record['navigation_result'] = navigation_result
    return record


def build_pointing_source(
    nav_results_root: str | FCPath | None,
    *,
    results_db_url: str | None,
) -> PointingSource:
    """Build the source a program reads its navigation records through.

    With no index URL the source reads documents, which is every program's
    default. With one, the index is opened and the root is checked against its
    ingest bookkeeping before a single lookup is made: a root the index has not
    fully ingested cannot answer "this image was never navigated", so it is
    refused rather than read. A URL that cannot be opened fails here; nothing
    falls back to reading files, which would turn a misconfigured run into a
    slow, silently different one.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under. None
            means no pointing is looked for at all, which only a run with no
            index can ask for -- an index is keyed by root, so there is nothing
            to look a row up under.
        results_db_url: Connection URL of the results index, or None to read
            documents.

    Returns:
        The source, which the caller closes when it is done with it.

    Raises:
        ValueError: If an index URL was resolved with no navigation results
            root; if the index cannot be opened, is not an index, or was written
            by another version of the schema; or if the root has no completed
            ingest run in it.
    """
    if results_db_url is None:
        return FilePointingSource(nav_results_root)
    if nav_results_root is None:
        raise ValueError(
            f'the results index {masked_url(results_db_url)} was named with no navigation '
            f'results root; the index is keyed by root, so there is no root to read rows '
            f'under. Name one, or pass --results-db none to read navigation documents.'
        )
    root_url = normalize_root_url(nav_results_root)
    engine = open_index(results_db_url, create=False)
    try:
        with engine.connect() as connection:
            require_ingested_roots(connection, [root_url], url=results_db_url)
    except Exception:
        engine.dispose()
        raise
    return IndexPointingSource(engine, root_url)
