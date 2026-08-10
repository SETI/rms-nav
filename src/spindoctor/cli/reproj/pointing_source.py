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
rebuild reads say what the document's own fields said: the ladder's first
question is whether the top-level ``status`` is ``success``, and a column
answering it from anywhere else would be a classification made at ingest time,
which no reader of the document could arrive at.

What the rebuilt record carries
-------------------------------

Only the fields the index holds for these two readers: the top-level ``status``,
``status_error`` and ``offset``, and the ``navigation_result`` ``times`` and
``pointing`` blocks the C-matrix mechanism needs.  It is not the document; it is
the part of the document that decides a pointing.  The ``pointing`` block's
``camera_frame`` name is deliberately absent, because no reader consults it: the
frame identity a recorded attitude is gated against is taken from the
observation, never from the record.

A field the row does not carry is rebuilt as a field the document did not have,
rather than as one holding null, because a reader tells those apart: the
backplane stage reports an absent ``status_error`` as ``unknown`` and would
report a null one as null.  The one exception is ``offset``, which is rebuilt as
a key holding null, and the reasoning is below.  The ``status`` column is NOT
NULL and stands in for a document that named no outcome with
:data:`~spindoctor.results_index.schema.UNKNOWN_STATUS`, so that value is
rebuilt as the absent field it stands for.

Where the two paths differ
--------------------------

A row is not a file, and ingest has already refused some documents and coerced
some values.  Every reason the file path can report either has an index-path
equivalent or is unreachable there, and an unreachable one surfaces under
whichever reason describes the row that ingest actually wrote:

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
| ``invalid_json``,                     | so it has no row, and it surfaces as     |
| ``metadata_not_an_object``            | ``no_metadata``                          |
+---------------------------------------+------------------------------------------+
| ``invalid_offset_type``,              | unreachable: ingest coerces such an      |
| ``non_finite_offset``,                | offset to NULL, and it surfaces as       |
| ``malformed_offset``                  | ``null_offset``                          |
+---------------------------------------+------------------------------------------+
| ``missing_offset_key``                | unreachable: a rebuilt record always     |
|                                       | carries the key, and a row with no       |
|                                       | offset surfaces as ``null_offset``       |
+---------------------------------------+------------------------------------------+

Four further differences have no row of their own, because each is the same
record classified differently rather than one reason reported in place of
another.  Each is a shape no navigation produces, and each is named here so that
a record hand-built into a results tree is not read as agreeing when it does
not.

Two differ only in what the outcome is called, so a run-level tally counts the
image under the other class and the product is the same:

* A ``cmatrix`` ingest cannot store -- one that is not nine finite numbers --
  is ``malformed_pointing`` via files and ``no_cmatrix_rotation_fitted`` via
  the index, because it becomes a NULL ``cmatrix`` beside a stored baseline,
  which is what a fitted-rotation result looks like.  Both fall back to the
  offset; they differ in what they call it.  (A ``cmatrix`` that *is* nine
  finite numbers and is not a rotation is stored, and the validator then
  refuses it in both paths alike, which is why ``malformed_pointing`` has a row
  of its own above.)
* A ``pointing`` block carrying none of the four fields the index has columns
  for -- one holding only ``camera_frame``, say -- is
  ``no_cmatrix_rotation_fitted`` via files and ``no_pointing_block`` via the
  index, since the rebuilt record has no block to distinguish.  The block a
  navigation writes always carries the baseline and both frame identities, so
  this is a block none of them wrote.

Two differ in the *product* and not only in what it is called:

* A ``cmatrix`` written as a 3x3 nesting -- a shape this module's classifier
  accepts and the navigator never writes -- selects the C-matrix mechanism via
  files and the offset via the index, so the two products are built on
  different pointing.
* A ``status: success`` record carrying no ``offset`` key at all is refused by
  the backplane stage via files, which raises rather than building geometry on
  a record shaped like a defect, and produces backplanes via the index, which
  cannot see the difference between an absent offset and a null one and reports
  the commoner of them.  With a usable ``cmatrix`` beside it the index path
  applies the corrected attitude and writes the product the file path refuses.
  The rebuild renders an absent offset as a null one deliberately: ingest
  stores an absent, null, malformed and non-finite offset alike as NULL, and
  the other three are records the file path builds products from, so rendering
  the pair as an absent key would refuse three reachable shapes to agree about
  one that is not.  No navigation writes it: a result carrying no offset is
  never a success, so a document whose status is ``success`` always carries
  one.

These are real behavioral differences and are stated rather than papered over.
For every record the navigator wrote and ingest stored, everything a product is
built from -- the mechanism, the matrices, the midtime, the offset -- is
identical in the two paths, and so is every field the readers report about it.
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
    IMAGES,
    UNKNOWN_STATUS,
    masked_url,
    normalize_root_url,
    open_index,
    reporting_a_failed_read,
    require_ingested_roots,
)

__all__ = [
    'FilePointingSource',
    'IndexPointingSource',
    'PointingSource',
    'build_pointing_source',
]

_METADATA_SUFFIX = '_metadata.json'
"""Suffix the navigator appends to a results path stub to name its document."""

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

    def _row(self, image_file: ImageFile) -> sqlalchemy.Row[Any] | None:
        """Read the one row recording an image, or None when there is none.

        Parameters:
            image_file: The image to look up.

        Returns:
            The row, or None when the index holds none for this stub under this
            root.

        Raises:
            ValueError: If the index cannot be read at all -- a lost
                connection, a table the account may not read, a partially
                restored database.  Translated here for the same reason the
                selection seam translates it: a caller of this module reports
                the failure against one image, and the database layer's own
                exception types are ones it cannot name.
        """
        statement = sqlalchemy.select(*_ROW_COLUMNS).where(
            IMAGES.c.root_url == self._root_url,
            IMAGES.c.results_path_stub == image_file.results_path_stub,
        )
        url = self._engine.url.render_as_string(hide_password=False)
        with reporting_a_failed_read(url), self._engine.connect() as connection:
            return connection.execute(statement).first()

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
        """
        row = self._row(image_file)
        if row is None:
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
        """
        row = self._row(image_file)
        if row is None:
            IMAGE_LOGGER.warning(NO_METADATA_MESSAGE, image_file.image_file_url, self._storage)
            return none_selection(NO_METADATA)
        return select_pointing(_record_from_row(row), subject=str(image_file.image_file_url))

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
    # no usable pair.  Ingest stores a malformed, non-finite or absent offset
    # alike as NULL, so the index cannot tell those apart, and reporting the
    # commonest of them is what keeps a genuine null offset from being read as a
    # defect-shaped record with no offset field at all.
    offset: list[float] | None = None
    if row.offset_dv is not None and row.offset_du is not None:
        offset = [row.offset_dv, row.offset_du]
    # ``status`` and ``status_error`` are rendered back as the fields they
    # stand for, absent ones included: a reader distinguishes a document that
    # named an outcome from one that named none, and reports the second by
    # defaulting rather than by printing whatever the column held.  The status
    # column is NOT NULL and records a document that named no outcome as
    # ``UNKNOWN_STATUS``, which is that document's absent field and is rebuilt
    # as one.
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
