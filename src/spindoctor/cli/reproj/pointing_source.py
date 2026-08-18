"""Where a reprojection or backplane reader gets one image's navigation record.

Both readers need the same two things per image: the whole record (the backplane
stage reads its status before it decides there is work to do) and the classified
pointing that record supplies.  How those arrive differs.  Reading one
``_metadata.json`` per image costs one round trip per image on a cloud root,
which a Cassini-scale run pays several hundred thousand times; an ingested
results index answers the same questions with one row.  So the seam is explicit:
:class:`PointingSource` names it, :class:`FilePointingSource` reads documents,
and :class:`IndexPointingSource` reads rows.

Neither implementation classifies anything itself, and neither reads its storage
itself.  Both are thin wrappers over
:mod:`spindoctor.results_index.record_source`, the one seam every program reads
navigation records through, and both hand what it returns to the one classifier,
:func:`spindoctor.cli.reproj.offsets.select_pointing`.  So the two paths cannot
disagree about which pointing a record supplies -- there is only one ladder, and
it is the same code in both modes -- and neither can disagree with the other
programs about what a row says, because no rebuild of a row lives here.  That
holds only while the columns the rebuild reads say what the document's own fields
said, which is why ingest fills every one of them through
:mod:`spindoctor.support.nav_record`, the module these readers read the same
fields through.  A column filled by a rule of its own, even one that agreed when
it was written, is a second reader of the record.

What the rebuilt record carries
-------------------------------

:data:`_ROW_COLUMNS` is this consumer's declaration of what it reads, and a
rebuilt record carries those columns and nothing else: the top-level ``status``,
``status_error`` and ``offset``, and the ``navigation_result`` ``times`` and
``pointing`` blocks the C-matrix mechanism needs.  It is not the document; it is
the part of the document that decides a pointing.  The ``pointing`` block's
``camera_frame`` is not among them although the index carries it for another
reader: the frame identity a recorded attitude is gated against is taken from the
observation, never from the record, so nothing here would consult a rebuilt name.
Selecting no column for it is the whole of how that is arranged.

How an absent column, a NOT NULL status and half a pair are rebuilt is
:mod:`spindoctor.results_index.rebuild`'s to state, since every consumer depends
on the same answers.  Two of them decide a reason this module reports, so they
are worth repeating here: ``offset`` is rebuilt as a key holding null, which is
what makes ``null_offset`` this path's reason for all four unusable offsets, as
class 1 below states; and the sentinel ``status`` is rebuilt as the absent field
it stands for, so a document naming no outcome and one naming that word for
itself are reported alike rather than one of them as nothing.

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

from typing import Any, Protocol

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
    IMAGES,
    IndexRecordSource,
    RecordSource,
    masked_url,
    normalize_root_url,
    open_index_for_roots,
)
from spindoctor.support.nav_document import read_document

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
"""Every column these two readers read, and the whole of what a lookup selects.

A row is only cheaper than a document while it carries less, so this is a
declaration rather than a convenience: what is not here is not read, and what is
read is here.  A test holds the list to the fields
:mod:`spindoctor.results_index.rebuild` knows a place for, since a column
selected that no field is rebuilt from would be paid for and dropped.
"""


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
        # Resolved through the one guard every reader of a document shares
        # rather than joined here: the class would otherwise apply different
        # rules about which paths a results root may be read at depending on
        # which of its two methods was called.  This wrapper of the guard is the
        # one that reports a refused path against the image, which is why this
        # method calls it rather than asking the seam for the document.
        metadata_file = resolved_nav_metadata_path(self._nav_results_root, image_file)
        if metadata_file is None:
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: does not name a navigation record under '
                f'{self._nav_results_root}, so none can be read for this image'
            )
        return read_document(metadata_file)

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

    Everything about reading the row is
    :class:`~spindoctor.results_index.record_source.IndexRecordSource`'s: the one
    statement over the primary key, the root the key is filtered on, the refusal
    table read in the same statement, and the rebuild.  What is left here is what
    this consumer does with what comes back, which is to classify it.

    Parameters:
        engine: An open index, which this source disposes of when it is closed.
        root_url: The normalized results root whose rows this source answers
            from.
    """

    def __init__(self, engine: Engine, root_url: str) -> None:
        """Build the seam over this index, and name it for the messages.

        The index is named for messages here rather than by each caller, and is
        rendered with its credentials hidden: these messages reach run logs and
        bug reports, and a connection URL can carry a database password.
        """
        url = engine.url.render_as_string(hide_password=False)
        self._records: RecordSource = IndexRecordSource(engine, root_url, url, _ROW_COLUMNS)
        # An index is a snapshot of its last ingest, so a row can be absent
        # because nothing navigated the image or because the image was
        # navigated after that ingest.  Neither the row nor its absence can say
        # which, so the message says what was searched and leaves the reader
        # able to tell.
        self._storage = (
            f'the results index {masked_url(url)}, a snapshot of its last ingest of {root_url}'
        )

    def read_record(self, image_file: ImageFile) -> dict[str, Any]:
        """Rebuild one image's navigation record from its row.

        Parameters:
            image_file: The image to look up.

        Returns:
            The rebuilt record.

        Raises:
            FileNotFoundError: If the index holds no row for this stub under this
                root. A missing document raises the same way, so the caller
                reports both the same way.
            ValueError: If the index records the document for this stub as one
                the ingest refused. Deliberately not the same exception: a
                caller reports a missing record as an image nothing navigated,
                and this image was navigated -- the index simply cannot say
                what it recorded.
        """
        return self._records.read_record(image_file.results_path_stub)

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
                supply one and the product would differ in silence.  The seam
                raises this and it is deliberately not caught: only the absence
                of a row is a classifiable answer.
        """
        try:
            record = self._records.read_record(image_file.results_path_stub)
        except FileNotFoundError:
            IMAGE_LOGGER.warning(NO_METADATA_MESSAGE, image_file.image_file_url, self._storage)
            return none_selection(NO_METADATA)
        return select_pointing(record, subject=image_file.image_file_url.as_posix())

    def close(self) -> None:
        """Close the seam, which disposes of the index it holds open."""
        self._records.close()


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
    return IndexPointingSource(open_index_for_roots(results_db_url, [root_url]), root_url)
