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
it is the same code in both modes.

What the rebuilt record carries
-------------------------------

Only the fields the index holds for these two readers: the top-level ``status``,
``status_error`` and ``offset``, and the ``navigation_result`` ``times`` and
``pointing`` blocks the C-matrix mechanism needs.  It is not the document; it is
the part of the document that decides a pointing.  The ``pointing`` block's
``camera_frame`` name is deliberately absent, because no reader consults it: the
frame identity a recorded attitude is gated against is taken from the
observation, never from the record.

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

Two further differences have no row of their own, because they are not one
reason reported in place of another but the same record classified differently.
Both come of ingest storing a rotation only in the nine row-major floats its
producer writes, so anything else becomes a NULL ``cmatrix`` beside a stored
baseline -- which is what a fitted-rotation result looks like:

* A ``cmatrix`` ingest cannot store -- one that is not nine finite numbers --
  is ``malformed_pointing`` via files and ``no_cmatrix_rotation_fitted`` via
  the index.  Both fall back to the offset; they differ in what they call it.
  (A ``cmatrix`` that *is* nine finite numbers and is not a rotation is stored,
  and the validator then refuses it in both paths alike, which is why
  ``malformed_pointing`` has a row of its own above.)
* A ``cmatrix`` written as a 3x3 nesting -- a shape this module's classifier
  accepts and the navigator never writes -- selects the C-matrix mechanism via
  files and the offset via the index.  This is the one record class whose
  *product* differs between the two, and it is a shape no navigation produces.

These are real behavioral differences and are stated rather than papered over.
For every record the navigator wrote and ingest stored, everything a product is
built from -- the mechanism, the matrices, the midtime, the offset -- is
identical in the two paths.
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
    select_pointing,
)
from spindoctor.config import IMAGE_LOGGER
from spindoctor.dataset.dataset import ImageFile
from spindoctor.results_index import (
    IMAGES,
    masked_url,
    normalize_root_url,
    open_index,
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
            FileNotFoundError: If the document does not exist, or if this source
                was built with no navigation results root and therefore has
                nowhere to look.
            ValueError: If the document is not valid JSON, or is valid JSON that
                is not an object.
        """
        if self._nav_results_root is None:
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: no navigation results root was resolved, '
                f'so no navigation record can be read for this image'
            )
        metadata_file = FCPath(self._nav_results_root) / (
            image_file.results_path_stub + _METADATA_SUFFIX
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

    def _row(self, image_file: ImageFile) -> sqlalchemy.Row[Any] | None:
        """Read the one row recording an image, or None when there is none.

        Parameters:
            image_file: The image to look up.

        Returns:
            The row, or None when the index holds none for this stub under this
            root.
        """
        statement = sqlalchemy.select(*_ROW_COLUMNS).where(
            IMAGES.c.root_url == self._root_url,
            IMAGES.c.results_path_stub == image_file.results_path_stub,
        )
        with self._engine.connect() as connection:
            return connection.execute(statement).first()

    def read_record(self, image_file: ImageFile) -> dict[str, Any]:
        """Rebuild one image's navigation record from its row.

        Parameters:
            image_file: The image to look up.

        Returns:
            The rebuilt record.

        Raises:
            FileNotFoundError: If the index holds no row for this stub under this
                root, naming both and the index. The index is a snapshot of a
                fully ingested root, so absence of a row means the image was
                never navigated -- the same thing a missing document means, and
                raised the same way so the caller reports it the same way.
        """
        row = self._row(image_file)
        if row is None:
            raise FileNotFoundError(
                f'{image_file.results_path_stub}: the results index {self._url} holds no '
                f'navigation record for this image under {self._root_url}'
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
            IMAGE_LOGGER.warning(NO_METADATA_MESSAGE, image_file.image_file_url)
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
    record: dict[str, Any] = {
        'status': row.status,
        'status_error': row.status_error,
        'offset': offset,
    }
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
