"""The one correspondence between a row's columns and a record's fields.

Two things are held here, and the first is what makes the second worth having.

**Every column of ``images`` is deliberately one of three things**: a field of a
record, part of the identity of the document a row stands for, or a value the
ingest computed rather than copied.  A column added to the schema and to the
ingest, and then to nobody's consumer, is the failure this catches: it would read
as absent from every rebuilt record for as long as it took somebody to notice,
and no coverage measurement would say so, because the code that would have read
it does not exist yet.

**A record rebuilt from a document's own columns says what the document said.**
That is the invariant the whole seam rests on: consumers classify a pointing,
decide an eligibility and build a kernel from a rebuilt record, and every one of
those is entitled to the same answer whichever storage it read.  It is asserted
here against the forward direction -- the ingest's own flattening -- so that a
change to either side has to keep the round trip.
"""

from typing import Any

import pytest
import sqlalchemy
from tests.spindoctor.cli.stats.conftest import metadata_document

from spindoctor.cli.ck.inputs import RECORD_COLUMNS as CK_COLUMNS
from spindoctor.cli.reproj.pointing_source import _ROW_COLUMNS as REPROJECTION_COLUMNS
from spindoctor.cli.stats.ingest_rows import MetadataSource, rows_from_metadata
from spindoctor.results_index import IMAGES, RECORD_FIELDS, record_from_row
from spindoctor.results_index.rebuild import DERIVED_COLUMNS, IDENTITY_COLUMNS
from spindoctor.support.nav_record import (
    UNKNOWN_STATUS,
    record_offset,
    record_status,
    record_status_error,
)

ROOT_URL = '/data/nav-results'
"""The root a source document is ingested under."""

STUB = 'COISS_2001/N1454725799'
"""The stub that document is recorded against."""


def _mapped_columns() -> set[str]:
    """Return every column name the rebuild knows a place for.

    Returns:
        The names, from the correspondence itself rather than from a second list
        of them.
    """
    return {column for entry in RECORD_FIELDS for column in entry.columns}


def _row_of(document: dict[str, Any], columns: tuple[Any, ...]) -> sqlalchemy.Row[Any]:
    """Flatten a document the way the ingest does and hand back one row of it.

    The row is built in memory rather than through a database, because what is
    being tested is the correspondence rather than a query: a real row would add
    a backend's type coercion to the comparison and hide which side of it a
    difference came from.

    Parameters:
        document: The navigation document to flatten.
        columns: The columns the consumer selects.

    Returns:
        A row carrying those columns, with the values the ingest stores.
    """
    source = MetadataSource(
        root_url=ROOT_URL,
        results_path_stub=STUB,
        source_file=f'{ROOT_URL}/{STUB}_metadata.json',
        mtime_ns=1,
        size_bytes=2,
    )
    stored = rows_from_metadata(document, source).image
    names = [column.name for column in columns]
    statement = sqlalchemy.select(*(sqlalchemy.literal(stored[name]).label(name) for name in names))
    with sqlalchemy.create_engine('sqlite://').connect() as connection:
        row = connection.execute(statement).one()
    return row


def test_every_column_is_a_record_field_an_identity_or_a_derived_value() -> None:
    """A column belonging to none of the three is one no rebuild can hand back."""
    sorted_out = _mapped_columns() | IDENTITY_COLUMNS | DERIVED_COLUMNS
    unsorted = sorted(set(IMAGES.columns.keys()) - sorted_out)
    assert unsorted == []


def test_no_column_is_sorted_into_two_of_the_three() -> None:
    """The three groups are a partition, so a column cannot be a field and derived."""
    mapped = _mapped_columns()
    assert sorted((mapped & IDENTITY_COLUMNS) | (mapped & DERIVED_COLUMNS)) == []


def test_no_identity_column_is_also_derived() -> None:
    """The other pair of the partition, asserted for the same reason."""
    assert sorted(IDENTITY_COLUMNS & DERIVED_COLUMNS) == []


def test_the_correspondence_names_only_real_columns() -> None:
    """A mapping for a column that does not exist is a field nothing can fill."""
    assert sorted(_mapped_columns() - set(IMAGES.columns.keys())) == []


@pytest.mark.parametrize(
    ('consumer', 'columns'),
    [('reprojection', REPROJECTION_COLUMNS), ('kernel writer', CK_COLUMNS)],
)
def test_every_column_a_consumer_selects_is_one_the_rebuild_places(
    consumer: str, columns: tuple[Any, ...]
) -> None:
    """A column selected and not placed is paid for on every read and dropped.

    The consumer would read the field it selected the column for as absent, which
    is the same thing the row says when the value really is absent -- so the
    mistake is invisible in the answer and shows up as a product built from a
    field nothing filled.

    Parameters:
        consumer: Which consumer's list is under test, for the failure message.
        columns: The columns it selects.
    """
    unplaced = sorted({column.name for column in columns} - _mapped_columns())
    assert unplaced == [], f'{consumer} selects columns the rebuild has no place for'


def test_a_rebuilt_record_names_the_outcome_the_document_named() -> None:
    """The value every consumer gates on, through the accessor both paths use."""
    document = metadata_document(status='failed', status_reason='no features')
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert record_status(rebuilt) == record_status(document)


def test_a_rebuilt_record_supplies_the_offset_the_document_supplied() -> None:
    """What a product is built from, so a difference here is a wrong product."""
    document = metadata_document(offset=[1.25, -2.5])
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert record_offset(rebuilt).pair == record_offset(document).pair


def test_a_rebuilt_record_names_the_error_the_document_named() -> None:
    """The vocabulary a selection filter matches verbatim."""
    document = metadata_document(status='error', status_error='missing_spice_data')
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert record_status_error(rebuilt) == record_status_error(document)


def test_a_document_naming_no_outcome_is_rebuilt_as_one_naming_none() -> None:
    """The status column is NOT NULL, so a sentinel stands in and must be undone.

    Left in place it would be read as a document whose status was the word
    ``unknown``, which no navigation writes.
    """
    document = metadata_document(status='')
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert 'status' not in rebuilt


def test_a_document_naming_no_outcome_still_reads_as_naming_none() -> None:
    """Which is what the accessor every consumer reads the field through says."""
    document = metadata_document(status='')
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert record_status(rebuilt) == UNKNOWN_STATUS


def test_an_offset_the_document_did_not_record_is_rebuilt_as_a_null_offset() -> None:
    """The one field written even when the row carries no value for it.

    A key holding null is what the navigator writes for an image that measured
    no offset, and it is what makes the shortfall count under one reason rather
    than under whichever of four an absent key would have named.
    """
    document = metadata_document(status='failed', offset=None)
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert rebuilt['offset'] is None


def test_a_block_whose_columns_are_all_absent_is_not_written() -> None:
    """A missing block and one holding nothing readable are one record.

    Several consumers read the presence of the pointing block to tell a record
    that never had one from a result that fitted a camera rotation, so an empty
    block written for tidiness would be read as a fact.
    """
    document = metadata_document(pointing={})
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert 'pointing' not in rebuilt.get('navigation_result', {})


def test_a_block_with_one_column_set_is_written() -> None:
    """The other half of the rule above, so it is not passing for want of a block."""
    document = metadata_document(pointing={'camera_frame_id': -82360})
    rebuilt = record_from_row(_row_of(document, REPROJECTION_COLUMNS))
    assert rebuilt['navigation_result']['pointing'] == {'camera_frame_id': -82360}


def test_half_a_recorded_pair_is_not_a_pair() -> None:
    """Ingest stores a pair it could not read whole as two NULLs.

    Rebuilding one half as a pair would hand a consumer a value the document
    never recorded, and a sigma of one axis is not a sigma.
    """
    document = metadata_document()
    row = _row_of(document, (IMAGES.c.sigma_dv,))
    assert 'navigation_result' not in record_from_row(row)


def test_a_column_that_is_no_record_field_is_passed_over() -> None:
    """A consumer selects its key alongside what it reads, and gets its record.

    Refusing the key instead would make every consumer strip its own row before
    handing it over, which is a rule that would be forgotten once.
    """
    document = metadata_document()
    row = _row_of(document, (IMAGES.c.results_path_stub, IMAGES.c.status))
    assert record_from_row(row) == {'status': 'success'}


def test_a_field_no_column_was_selected_for_is_absent() -> None:
    """What a consumer reads is what it asked for, and the rest is not invented."""
    document = metadata_document(times={'midtime_et': 100.0})
    row = _row_of(document, (IMAGES.c.status,))
    assert 'navigation_result' not in record_from_row(row)
