"""Tests for reading a navigation results tree into the results index.

What these pin is mostly what ingest does *not* do: it does not key an image by
its name, does not round the offset it stores, does not merge the two reason
vocabularies, does not read a document whose file has not changed, does not
stat a file it already listed, and does not stop when a file turns out not to
be a navigation document at all.
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath

from spindoctor.cli.stats.ingest import INGEST_COMMIT_CHUNK_SIZE, ingest_metadata_files
from spindoctor.cli.stats.ingest import chunks as chunks_module
from spindoctor.cli.stats.ingest import driver as driver_module
from spindoctor.cli.stats.ingest_rows import (
    MetadataDocumentError,
    MetadataSource,
    rows_from_metadata,
)
from spindoctor.nav_records import METADATA_SUFFIX, RETRIEVE_BATCH_SIZE
from spindoctor.results_index import (
    IMAGES,
    INGEST_RUNS,
    TECHNIQUES,
    UNKNOWN_STATUS,
    open_index,
)

from .conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    run_rows,
    technique,
    write_metadata,
    write_summary_png,
)

SOURCE = MetadataSource(
    root_url='/data/nav-results',
    results_path_stub='COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    source_file='/data/nav-results/x_metadata.json',
    mtime_ns=1234567890123456789,
    size_bytes=4096,
)


def _rows(connection: sqlalchemy.Connection, statement: Any) -> list[Any]:
    """Execute a statement and return its rows.

    Parameters:
        connection: An open connection.
        statement: The statement to run.

    Returns:
        The rows.
    """
    return list(connection.execute(statement))


# ---------------------------------------------------------------------------
# One document into rows
# ---------------------------------------------------------------------------


def test_the_stub_and_root_key_the_row() -> None:
    """The key comes from where the file is, never from the document."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['results_path_stub'] == SOURCE.results_path_stub


def test_the_root_is_recorded_as_given() -> None:
    """The other half of the key is the root the walk was told to read."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['root_url'] == SOURCE.root_url


def test_the_subtree_is_the_stubs_first_segment() -> None:
    """A stub under a directory yields it without string surgery in SQL."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['subtree'] == 'COISS_2001'


def test_a_bare_basename_stub_has_no_subtree() -> None:
    """The simulated dataset produces a stub with no separator, and no subtree."""
    source = MetadataSource(
        root_url='/data/nav-results',
        results_path_stub='sim_scene_000042',
        source_file='/data/nav-results/sim_scene_000042_metadata.json',
        mtime_ns=1,
        size_bytes=2,
    )
    rows = rows_from_metadata(metadata_document(instrument='sim'), source)
    assert rows.image['subtree'] is None


def test_the_stored_offset_is_the_top_level_one() -> None:
    """The authoritative offset is the top-level field, not the display copy."""
    document = metadata_document(offset=[3.14159265358979, -2.71828182845905])
    document['navigation_result']['offset_px'] = [3.142, -2.718]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['offset_dv'] == 3.14159265358979


def test_the_stored_offset_is_not_rounded() -> None:
    """A fifteen-digit offset round-trips through the row unchanged."""
    document = metadata_document(offset=[3.14159265358979, -2.71828182845905])
    document['navigation_result']['offset_px'] = [3.142, -2.718]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['offset_du'] == -2.71828182845905


def test_the_stored_confidence_is_the_top_level_one() -> None:
    """Confidence follows the offset: the value, not the rounded display copy."""
    document = metadata_document(confidence=0.876543210987654)
    document['navigation_result']['confidence'] = 0.877
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['confidence'] == 0.876543210987654


def test_a_missing_top_level_offset_is_null() -> None:
    """A document with no offset stores none, whatever the display copy says."""
    document = metadata_document(status='failed', status_reason='no_features', offset=None)
    document['navigation_result']['offset_px'] = [9.0, 9.0]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['offset_dv'] is None


def test_a_non_finite_offset_is_null() -> None:
    """A malformed offset is stored as no offset rather than as a NaN."""
    document = metadata_document()
    document['offset'] = [float('nan'), 1.0]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['offset_dv'] is None


def test_the_stored_status_is_the_documents_own() -> None:
    """The column holds the top-level field and nothing standing in for it.

    A consumer rebuilds a record from the row and classifies it with the same
    ladder that reads the document, and that ladder's first question is whether
    the top-level ``status`` is ``success``.  A column carrying the nested copy
    where the document named nothing would answer ``success`` for a document
    that never did, and a record supplying no pointing through its file would
    then apply a corrected attitude through the index.
    """
    document = metadata_document(offset=[1.0, 2.0])
    del document['status']
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status'] != 'success'


def test_a_document_naming_no_status_is_recorded_as_naming_none() -> None:
    """The column is NOT NULL, so "this document did not say" needs a value."""
    document = metadata_document(offset=[1.0, 2.0])
    del document['status']
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status'] == UNKNOWN_STATUS


def test_the_nested_status_is_ignored_even_when_it_names_an_outcome() -> None:
    """A guard on the two above, which a document with no nested copy would pass."""
    document = metadata_document(offset=[1.0, 2.0])
    del document['status']
    assert document['navigation_result']['status'] == 'success'


def test_a_document_naming_a_status_keeps_it() -> None:
    """The control: the field the ladder reads survives into the column verbatim."""
    rows = rows_from_metadata(metadata_document(status='conflicted'), SOURCE)
    assert rows.image['status'] == 'conflicted'


def test_status_error_is_stored_verbatim() -> None:
    """The selection filter matches this token exactly, so nothing may touch it."""
    document = metadata_document(
        status='error', status_reason=None, status_error='missing_spice_data', offset=None
    )
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status_error'] == 'missing_spice_data'


def test_status_error_does_not_reach_the_reason_column() -> None:
    """The two vocabularies stay in their own columns rather than merging."""
    document = metadata_document(
        status='error', status_reason=None, status_error='missing_spice_data', offset=None
    )
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status_reason'] is None


def test_status_reason_does_not_reach_the_error_column() -> None:
    """And the reverse, so a filter on one never matches a value of the other."""
    document = metadata_document(status='failed', status_reason='no_features', offset=None)
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status_error'] is None


def test_an_empty_reason_is_stored_as_nothing() -> None:
    """An empty reason must be NULL, so a COALESCE over the pair falls through."""
    document = metadata_document(status='failed', status_reason='', offset=None)
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status_reason'] is None


@pytest.mark.parametrize(
    'recorded', [None, '', 42, UNKNOWN_STATUS], ids=['null', 'empty', 'number', 'the-word']
)
def test_a_document_naming_no_error_stores_no_error(recorded: Any) -> None:
    """Read through the consumers' own function rather than a rule of the column.

    Every one of these names no error to the readers, the word ``unknown``
    included, since that is what they report a record naming none under.  A
    column deciding for itself which fields name an error would agree with them
    until one of the two changed.

    Parameters:
        recorded: The recorded ``status_error``.
    """
    document = metadata_document(status='failed', offset=None)
    document['status_error'] = recorded
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['status_error'] is None


@pytest.mark.parametrize(
    ('sigma', 'expected'),
    [
        ([0.5, 0.75], (0.5, 0.75)),
        ([0.5, 0.75, 1.0], (None, None)),
        ([0.5], (None, None)),
        ([], (None, None)),
    ],
    ids=['pair', 'three', 'one', 'empty'],
)
def test_a_recorded_sigma_pair_is_refused_whole_unless_it_is_a_pair(
    sigma: Any, expected: tuple[float | None, float | None]
) -> None:
    """Two of three recorded numbers are not the pair anybody wrote.

    The per-axis uncertainty an operator reads off a report has to be the one
    the navigation recorded; taking the first two of three would report an
    uncertainty from a value of some other shape as though it were that pair.

    Parameters:
        sigma: The recorded ``navigation_result.sigma_px``.
        expected: The ``(sigma_dv, sigma_du)`` the columns must hold.
    """
    document = metadata_document()
    document['navigation_result']['sigma_px'] = sigma
    rows = rows_from_metadata(document, SOURCE)
    assert (rows.image['sigma_dv'], rows.image['sigma_du']) == expected


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [([1.5, -2.5], (1.5, -2.5)), ([1.5, -2.5, 9.0], (None, None)), ([1.5], (None, None))],
    ids=['pair', 'three', 'one'],
)
def test_a_techniques_offset_is_refused_whole_unless_it_is_a_pair(
    offset: Any, expected: tuple[float | None, float | None]
) -> None:
    """The same rule for the per-technique estimates a report compares.

    Parameters:
        offset: The recorded ``per_technique[].offset_px``.
        expected: The ``(offset_dv, offset_du)`` the row must hold.
    """
    entry = technique('StarFieldFromCatalogNav', (0.0, 0.0))
    entry['offset_px'] = offset
    document = metadata_document(per_technique=[entry])
    rows = rows_from_metadata(document, SOURCE)
    assert (rows.techniques[0]['offset_dv'], rows.techniques[0]['offset_du']) == expected


def test_the_covariance_block_is_the_offset_block() -> None:
    """A twist-fitted 3x3 matrix contributes only its 2x2 offset block."""
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0],
    ]
    rows = rows_from_metadata(document, SOURCE)
    assert (
        rows.image['covariance_vv'],
        rows.image['covariance_vu'],
        rows.image['covariance_uu'],
    ) == (1.0, 2.0, 4.0)


def test_the_rotation_columns_are_read() -> None:
    """A twist-fitted result records a rotation, and the index carries it."""
    document = metadata_document()
    document['navigation_result']['rotation_deg'] = 0.125
    document['navigation_result']['sigma_rotation_deg'] = 0.004
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['rotation_deg'] == 0.125


def test_the_image_number_is_ingested() -> None:
    """The range filter compares a column, so the number is computed here."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['image_number'] == 1454725799


def test_the_file_metrics_come_from_the_walk() -> None:
    """The incremental skip compares these two against the next listing."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert (rows.image['mtime_ns'], rows.image['size_bytes']) == (
        SOURCE.mtime_ns,
        SOURCE.size_bytes,
    )


def test_the_technique_flags_are_booleans() -> None:
    """A boolean column holds a boolean; an integer flag is a type error later."""
    document = metadata_document(
        per_technique=[technique('BodyLimbNav', (1.0, 1.0), spurious=True)]
    )
    rows = rows_from_metadata(document, SOURCE)
    assert rows.techniques[0]['spurious'] is True


def test_the_child_rows_carry_the_image_key() -> None:
    """A child row names the image by the pair that keys it."""
    document = metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))])
    rows = rows_from_metadata(document, SOURCE)
    assert rows.techniques[0]['results_path_stub'] == SOURCE.results_path_stub


def test_the_feature_inventory_is_aggregated() -> None:
    """Per-feature detail is not retained; the counts per source are."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    gated = next(row for row in rows.feature_sources if row['source_model'] == 'stars')
    assert gated['n_gated'] == 1


def test_the_corrected_pointing_columns_are_read_when_present() -> None:
    """No document in the tree carries these yet, so a fixture exercises them."""
    document = metadata_document()
    document['navigation_result']['times'] = {
        'start_et': 170000000.5,
        'stop_et': 170000002.5,
        'exposure_s': 2.0,
        'sclk_start': '1/1294561202.100',
        'sclk_midtime': '1/1294561203.100',
        'sclk_stop': '1/1294561204.100',
    }
    document['navigation_result']['pointing'] = {
        'camera_frame_id': -82360,
        'ck_frame_id': -82000,
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'cmatrix_original': [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['sclk_midtime'] == '1/1294561203.100'


def test_a_corrected_pointing_matrix_is_stored_as_nine_floats() -> None:
    """The producer writes a row-major matrix, and the column holds one."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['cmatrix'] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize('field', ['cmatrix', 'cmatrix_original'])
def test_a_matrix_written_as_a_nesting_is_stored_as_the_nine_it_denotes(field: str) -> None:
    """Both shapes a record can write a rotation in reach the column.

    The classifier reads a 3x3 nesting as the nine values it denotes, so a
    store that held only the flat form would answer with no corrected attitude
    for a record the classifier applies one from -- two products from one
    document.  The baseline is stored under the same rule as the corrected
    attitude, because it is gated against the observation the same way.

    Parameters:
        field: Which recorded matrix is written as a nesting.
    """
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        field: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image[field] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_a_matrix_of_nine_numbers_that_is_not_a_rotation_is_still_stored() -> None:
    """Whether nine numbers are a rotation is the reader's question, not a column's.

    Stored, the value is refused by the one validator both readers apply to
    it; refused here, the row would look like a result that fitted a camera
    rotation and the two paths would call the same record different things.
    """
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    }
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['cmatrix'] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


@pytest.mark.parametrize(
    'value',
    [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]],
        [True] * 9,
        ['1.0', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '0.0', '1.0'],
        'not a matrix',
        [[1.0, 2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
        [float('nan'), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [10**400, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ],
    ids=[
        'eight',
        'ragged',
        'booleans',
        'strings',
        'text',
        'nine-shapes-that-are-not-one-matrix',
        'non-finite',
        'integer-too-large-for-a-float',
    ],
)
def test_a_matrix_no_reader_could_use_is_stored_as_nothing(value: Any) -> None:
    """Nothing a reader would refuse is kept, so a stored value is always usable.

    Parameters:
        value: The recorded matrix under test.
    """
    document = metadata_document()
    document['navigation_result']['pointing'] = {'cmatrix': value}
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['cmatrix'] is None


@pytest.mark.parametrize('field', ['cmatrix', 'cmatrix_original'])
def test_a_matrix_written_as_rows_of_one_is_stored_as_the_nine_it_denotes(field: str) -> None:
    """Whatever shape the readers assemble one matrix from, the column holds it.

    Nine rows of one reshape into the same 3x3 the flat nine do, so a reader
    applies the rotation; a column that judged the entries one at a time
    instead of assembling them would hold nothing, and the same record would be
    reprojected on its corrected attitude through a document and on an
    ``OffsetFOV`` through a row.

    Parameters:
        field: Which of the two recorded matrices is written that way.
    """
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'cmatrix_original': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    document['navigation_result']['pointing'][field] = [
        [value] for value in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image[field] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_an_unreadable_matrix_costs_the_matrix_and_not_the_document() -> None:
    """A value no reader can use is a NULL column, never a refused file.

    An integer too large for a float is the case that used to raise out of the
    conversion, and the raise cost the whole document: every other column of a
    real navigation record was lost, and the image then read as one nothing had
    navigated.
    """
    document = metadata_document(offset=[1.5, -2.5])
    document['navigation_result']['pointing'] = {'cmatrix': [10**400] * 9}
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['offset_dv'] == 1.5


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [
        ([1.5, -2.5], (1.5, -2.5)),
        (['1.5', '-2.5'], (1.5, -2.5)),
        ([1, -2], (1.0, -2.0)),
        ([1.5, -2.5, 9.0], (None, None)),
        ([1.5], (None, None)),
        ([True, False], (None, None)),
        ([float('nan'), 1.0], (None, None)),
    ],
    ids=['pair', 'numeric-strings', 'integers', 'three', 'one', 'booleans', 'non-finite'],
)
def test_the_offset_column_holds_what_a_reader_would_apply(
    offset: Any, expected: tuple[float | None, float | None]
) -> None:
    """Exactly the pair a consumer applies, and nothing where it applies none.

    A store that took the first two of three would build a product on a
    pointing nobody recorded; one that refused a pair the reader converts would
    leave an index-backed run uncorrected where the document-backed one is
    corrected.

    Parameters:
        offset: The recorded top-level offset.
        expected: The ``(offset_dv, offset_du)`` the columns must hold.
    """
    document = metadata_document()
    document['offset'] = offset
    rows = rows_from_metadata(document, SOURCE)
    assert (rows.image['offset_dv'], rows.image['offset_du']) == expected


def test_a_document_with_no_pointing_stores_nulls() -> None:
    """An image that never navigated has no corrected attitude to record."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['cmatrix'] is None


def test_a_camera_frame_id_is_read_as_an_integer() -> None:
    """The frame identifiers are integers, and a boolean is not one."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {'camera_frame_id': True}
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['camera_frame_id'] is None


def test_the_shutter_mode_is_stored_as_recorded() -> None:
    """Two cameras exposed together share one bus attitude, and this says so.

    A kernel writer pairs the two and keeps one correction; a column that did
    not carry the mode would leave it pairing on the camera name alone, which
    every image of that camera would match.
    """
    document = metadata_document()
    document['observation']['shutter_mode'] = 'BOTSIM'
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['shutter_mode'] == 'BOTSIM'


def test_a_document_recording_no_shutter_mode_stores_none() -> None:
    """Most datasets record none, and a missing mode pairs nothing."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['shutter_mode'] is None


def test_the_recorded_kernels_are_stored_in_the_order_recorded() -> None:
    """A correction overlays one original kernel, named among these."""
    document = metadata_document()
    document['navigation_result']['provenance']['spice_kernels'] = [
        'cas00172.tsc',
        'naif0012.tls',
    ]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['spice_kernels'] == ['cas00172.tsc', 'naif0012.tls']


def test_a_kernel_list_holding_anything_but_names_is_stored_as_none() -> None:
    """Its reader refuses such a block, and an emptied list is not that block.

    Storing the readable members would hand a consumer a shorter list than the
    document holds, and storing an empty one would say the run recorded no
    kernels -- which its reader refuses for an image carrying an attitude.
    """
    document = metadata_document()
    document['navigation_result']['provenance']['spice_kernels'] = ['cas00172.tsc', 7]
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['spice_kernels'] is None


def test_a_run_that_recorded_no_kernels_stores_the_empty_list() -> None:
    """An empty list is a statement about the run, and its reader refuses it."""
    document = metadata_document()
    document['navigation_result']['provenance']['spice_kernels'] = []
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['spice_kernels'] == []


def test_the_camera_frame_name_is_stored_as_recorded() -> None:
    """A kernel writer looks the name up among the frame kernels it furnishes."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {'camera_frame': 'CASSINI_ISS_NAC'}
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['camera_frame'] == 'CASSINI_ISS_NAC'


def test_a_camera_frame_name_that_is_not_text_is_stored_as_none() -> None:
    """``str(None)`` is ``'None'``, which would name a frame nothing defines."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {'camera_frame': -82360}
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['camera_frame'] is None


def test_a_document_without_an_image_name_is_refused() -> None:
    """This is what a file that is not a navigation document looks like."""
    with pytest.raises(MetadataDocumentError, match=r'no observation\.image_name'):
        rows_from_metadata({'observation': {}}, SOURCE)


def test_a_document_without_an_instrument_is_refused() -> None:
    """Half a document is no more ingestible than none."""
    with pytest.raises(MetadataDocumentError, match=r'no observation\.instrument'):
        rows_from_metadata(metadata_document(instrument=None), SOURCE)


def test_a_refusal_names_the_file() -> None:
    """A run that meets hundreds of these has to be able to name each one."""
    with pytest.raises(MetadataDocumentError, match=SOURCE.source_file):
        rows_from_metadata({'observation': {}}, SOURCE)


def test_a_refusal_carries_the_reason_without_the_file() -> None:
    """The reason is tallied across files, so it may not carry a file name."""
    with pytest.raises(MetadataDocumentError) as caught:
        rows_from_metadata({'observation': {}}, SOURCE)
    assert caught.value.reason == (
        'not a current-schema navigation document (no observation.image_name)'
    )


# ---------------------------------------------------------------------------
# The walk and the writer
# ---------------------------------------------------------------------------


def test_two_volumes_with_one_basename_produce_two_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Keying on the image name alone silently loses one of these."""
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document())
    write_metadata(root, 'COISS_2002/data/N1294561202_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        subtrees = _rows(connection, sqlalchemy.select(IMAGES.c.subtree).order_by(IMAGES.c.subtree))
    engine.dispose()
    assert [row.subtree for row in subtrees] == ['COISS_2001', 'COISS_2002']


def test_each_colliding_image_is_independently_retrievable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The pair is a key, so one of the two can be read without the other."""
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1294561202_1_CALIB', metadata_document(status='success'))
    write_metadata(
        root,
        'COISS_2002/data/N1294561202_1_CALIB',
        metadata_document(status='failed', status_reason='no_features', offset=None),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.status).where(
                IMAGES.c.results_path_stub == 'COISS_2002/data/N1294561202_1_CALIB'
            ),
        )
    engine.dispose()
    assert [row.status for row in found] == ['failed']


def test_a_bare_basename_stub_ingests_with_a_null_subtree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A simulated scene lives at the root of the tree and names no subtree."""
    root = tmp_path / 'results'
    write_metadata(root, 'sim_scene_000042', metadata_document(instrument='sim'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.subtree))
    engine.dispose()
    assert [row.subtree for row in found] == [None]


def _ingest_a_file_named_only_by_the_suffix(
    tmp_path: Path, logger: pdslogger.PdsLogger
) -> list[Any]:
    """Ingest a tree holding a file whose whole name is the document suffix.

    It is the last bracket case of the suffix test: a name that ends in the
    suffix and is nothing else, so trimming the suffix leaves an empty stub.
    The pass treats it as any other document, which puts it in under the empty
    stub with no subtree above it -- a row every subtree-restricted query passes
    over, which is what a selection is.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the pass reports through.

    Returns:
        The stub and subtree of every row the pass wrote.
    """
    root = tmp_path / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / METADATA_SUFFIX).write_text(json.dumps(metadata_document()), encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub, IMAGES.c.subtree))
    engine.dispose()
    return found


def test_a_file_named_only_by_the_suffix_ingests_under_an_empty_stub(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Trimming a suffix off a name of exactly that length leaves nothing."""
    found = _ingest_a_file_named_only_by_the_suffix(tmp_path, quiet_logger)
    assert [row.results_path_stub for row in found] == ['']


def test_a_file_named_only_by_the_suffix_ingests_with_a_null_subtree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """So it is under no subtree, and no enumeration of subtrees reaches it."""
    found = _ingest_a_file_named_only_by_the_suffix(tmp_path, quiet_logger)
    assert [row.subtree for row in found] == [None]


def _tree_with_a_file_that_is_not_a_document(tmp_path: Path) -> Path:
    """Write a root holding one document and four files that are not one.

    A results root holds the summary PNG a navigation that reached a result
    drew, and whatever else an operator has left there.  None of them is a file
    the pass reads, and the walk has to pass over each without adding it to any
    tally.

    The clutter is chosen to bracket the suffix test rather than to be merely
    unlike a document.  ``notes.txt`` and the summary PNG are unlike one in
    every way; ``scene_index.json`` is JSON and is not a navigation document,
    which is what separates the document suffix from the file extension; and
    ``..._metadata.json.tmp`` -- a partial write, an editor's backup, a
    ``.json.gz`` -- carries the suffix without ending in it, which is what
    separates ending in the suffix from containing it.  A name that only
    contains it yields a stub with the suffix's length cut off the end of a
    longer name, naming nothing, which the pass then retrieves, fails on,
    records nothing for, and retrieves again on every pass afterwards.

    The remaining bracket case is a name that is the suffix and nothing else,
    which is a document as far as the walk is concerned; what it ingests as is
    two tests above.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_summary_png(root, 'VOL/N1454725799_1_CALIB')
    (root / 'VOL' / 'notes.txt').write_text('nothing to ingest here', encoding='utf-8')
    (root / 'VOL' / 'scene_index.json').write_text('{"not": "a document"}', encoding='utf-8')
    (root / 'VOL' / f'N1454725799_1_CALIB{METADATA_SUFFIX}.tmp').write_text(
        '{"half": "written"', encoding='utf-8'
    )
    return root


def test_a_file_that_is_not_a_document_is_not_a_file_this_pass_saw(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The walk counts the documents, and a tree holds far more than documents.

    Counted as one of them, a summary PNG, a JSON file that is not a navigation
    result, or a half-written document would be retrieved, refused for not being
    a navigation document, and tallied against the root on every pass.
    """
    root = _tree_with_a_file_that_is_not_a_document(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert (counts.files_seen, counts.files_ingested, counts.files_failed) == (1, 1, 0)


def test_a_file_that_is_not_a_document_is_no_part_of_what_the_walk_found(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """What the entry loop collects, over a tree of ordinary clutter.

    Every entry is a directory to descend into, a document to collect, or a
    file to pass over, and only the last leaves the listing as it was.  A name
    that ends in the document suffix is what says which, and neither half of
    that is enough on its own: a file that merely ends in ``.json`` is not a
    navigation document, and one that merely contains the suffix yields a stub
    with the suffix's length cut off the end of a longer name, naming nothing,
    which the pass then retrieves, fails on, records nothing for, and retrieves
    again on every pass afterwards.
    """
    root = _tree_with_a_file_that_is_not_a_document(tmp_path)
    listing = driver_module._listing_of_root(root.as_posix(), logger=quiet_logger)
    assert listing is not None
    assert [found.stub for found in listing.documents] == ['VOL/N1454725799_1_CALIB']


def test_re_ingesting_an_image_replaces_its_child_rows(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A document that lost a technique must not leave the old row behind."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(
        root, stub, metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))])
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(root, stub, metadata_document(per_technique=[]))
    ingest_tree(url, [root], logger=quiet_logger, force=True)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(TECHNIQUES.c.technique_name))
    engine.dispose()
    assert [row.technique_name for row in found] == []


def test_an_unchanged_file_is_not_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second pass over an unchanged tree costs one listing and no reads."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    retrievals: list[Any] = []
    real_retrieve = FCPath.retrieve

    def counted(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        retrievals.append(sub_path)
        return real_retrieve(self, sub_path, **kwargs)

    monkeypatch.setattr(FCPath, 'retrieve', counted)
    ingest_tree(url, [root], logger=quiet_logger)
    assert retrievals == []


def _watch_per_file_questions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path ingest asks the storage layer about individually.

    ``exists`` is never needed at all, so it is forbidden outright.  ``stat``
    is recorded rather than forbidden: a local walk asks it about a directory
    it is entering, to recognize one it has already walked, and that question
    is asked once per directory and never on a cloud root.  What may not
    happen is asking it about a *file*, which on a cloud root is a paid round
    trip per image per run.

    Parameters:
        monkeypatch: Fixture the recorders are installed through.

    Returns:
        The list the paths accumulate in, in call order.
    """
    asked: list[str] = []
    real_stat = FCPath.stat

    def recorded(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        asked.append(self.as_posix())
        return real_stat(self, *args, **kwargs)

    def forbidden(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('ingest asked the backend whether one file exists')

    monkeypatch.setattr(FCPath, 'stat', recorded)
    monkeypatch.setattr(FCPath, 'exists', forbidden)
    return asked


def test_an_unchanged_file_is_not_stat_ed_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing already carried the metrics; asking again is a round trip."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    asked = _watch_per_file_questions(monkeypatch)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_skipped == 1
    assert [path for path in asked if path.endswith(METADATA_SUFFIX)] == []


def test_a_first_ingest_asks_about_no_single_file_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk feeds presence and both metrics, so a first pass is one listing."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    asked = _watch_per_file_questions(monkeypatch)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1
    assert [path for path in asked if path.endswith(METADATA_SUFFIX)] == []


def test_a_touched_file_is_read_again(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A document that changed on disk is what the second pass exists to catch."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(
        root, stub, metadata_document(status='failed', status_reason='no_features', offset=None)
    )
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


SAME_LENGTH_BEFORE = metadata_document(image_name='N1454725799_1_CALIB.IMG')
"""A document whose serialization is the same length as the one below."""

SAME_LENGTH_AFTER = metadata_document(image_name='N1454725798_1_CALIB.IMG')
"""The same document with one digit of the image name changed.

An edit of exactly the same byte length is what leaves the size half of the
incremental comparison saying nothing, so it is the only edit that asks the
modification time whether the file changed.
"""


def _rewrite_at_the_same_length(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest a document, rewrite it to the same length, and ingest again.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    path = write_metadata(root, stub, SAME_LENGTH_BEFORE)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=logger)
    written = path.stat().st_mtime_ns
    path.write_text(json.dumps(SAME_LENGTH_AFTER), encoding='utf-8')
    # A rewrite this quick can land in the same nanosecond the first write did,
    # which would make the two passes agree for a reason the test is not about.
    os.utime(path, ns=(written + 1_000_000_000, written + 1_000_000_000))
    return url


def test_the_same_length_rewrite_really_is_the_same_length() -> None:
    """Otherwise the test below would be passing on the size half after all."""
    before = len(json.dumps(SAME_LENGTH_BEFORE).encode('utf-8'))
    after = len(json.dumps(SAME_LENGTH_AFTER).encode('utf-8'))
    assert before == after


def test_a_file_rewritten_to_the_same_length_is_read_again(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The modification time is the half of the comparison that catches this.

    A results tree is rewritten in place by a re-navigation, and a document
    that changed without changing length is the ordinary case: one status word
    for another, one digit of an offset for another. Comparing size alone would
    skip such a file for as long as it existed.
    """
    url = _rewrite_at_the_same_length(tmp_path, quiet_logger)
    counts = ingest_tree(url, [tmp_path / 'results'], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_file_rewritten_to_the_same_length_updates_its_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Re-reading is only worth anything if the row that comes back is the new one."""
    url = _rewrite_at_the_same_length(tmp_path, quiet_logger)
    ingest_tree(url, [tmp_path / 'results'], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.image_name))
    engine.dispose()
    assert [row.image_name for row in found] == ['N1454725798_1_CALIB.IMG']


def test_a_touched_file_updates_its_row(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Re-reading is only useful if the row that comes back is the new one."""
    root = tmp_path / 'results'
    stub = 'VOL/N1454725799_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    write_metadata(
        root, stub, metadata_document(status='failed', status_reason='no_features', offset=None)
    )
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.status))
    engine.dispose()
    assert [row.status for row in found] == ['failed']


def test_force_re_reads_an_unchanged_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The escape hatch for a tree whose metrics cannot be trusted."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_ingested == 1


def test_force_skips_nothing(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A forced pass reads everything, so nothing is counted as skipped."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    counts = ingest_tree(url, [root], logger=quiet_logger, force=True)
    assert counts.files_skipped == 0


def test_a_listing_without_metrics_re_reads_everything(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that cannot say whether a file changed gets no skip at all."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    real_iterdir = FCPath.iterdir_metadata

    def stripped(self: FCPath) -> Any:
        for path, entry in real_iterdir(self):
            if entry is not None and not entry['is_dir']:
                yield path, {'is_dir': False}
            else:
                yield path, entry

    monkeypatch.setattr(FCPath, 'iterdir_metadata', stripped)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_listing_without_metrics_warns(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently re-reading a whole archive every run would be a mystery."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    real_iterdir = FCPath.iterdir_metadata

    def stripped(self: FCPath) -> Any:
        for path, entry in real_iterdir(self):
            if entry is not None and not entry['is_dir']:
                yield path, {'is_dir': False}
            else:
                yield path, entry

    monkeypatch.setattr(FCPath, 'iterdir_metadata', stripped)
    warnings: list[str] = []
    monkeypatch.setattr(
        quiet_logger, 'warning', lambda message, *args: warnings.append(str(message))
    )
    ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert any('cannot be ingested incrementally' in message for message in warnings)


def test_a_malformed_document_is_counted_as_an_error(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A results tree holds files that were never navigation documents."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_failed == 1


def test_a_malformed_document_does_not_abort_the_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One unreadable file among hundreds must not cost the other hundreds."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'bad_metadata.json').write_text('not json at all', encoding='utf-8')
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_failures_are_tallied_by_reason(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Hundreds of files nobody wanted must read differently from a real fault."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    (root / 'params_metadata.json').write_text('{"params": {}}', encoding='utf-8')
    (root / 'broken_metadata.json').write_text('not json at all', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {
        'not a current-schema navigation document (no observation.image_name)': 2,
        'not valid JSON': 1,
    }


def test_a_document_that_is_not_an_object_is_counted(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Valid JSON that is a list is still not a navigation document."""
    root = tmp_path / 'results'
    root.mkdir()
    (root / 'list_metadata.json').write_text('[1, 2, 3]', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'not a JSON object': 1}


def test_an_ingest_run_is_recorded_at_the_start(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer that looks mid-run must see a root that is not ready."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    seen: list[Any] = []

    real_listing = driver_module._listing_of_root

    def watching(listed_root: Any, **kwargs: Any) -> Any:
        with engine.connect() as connection:
            seen.extend(_rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc)))
        return real_listing(listed_root, **kwargs)

    monkeypatch.setattr(driver_module, '_listing_of_root', watching)
    ingest_metadata_files(engine, [root.as_posix()], logger=quiet_logger)
    engine.dispose()
    assert [row.finished_utc for row in seen] == [None]


def test_an_ingest_run_is_completed_at_the_end(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The finish time is what makes absence of a row mean "not navigated"."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_an_ingest_run_records_what_it_covered(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The counts are the record of what a root's index actually contains."""
    root = tmp_path / 'results'
    root.mkdir()
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'edges_metadata.json').write_text('{"edges": []}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(
                INGEST_RUNS.c.files_seen, INGEST_RUNS.c.files_ingested, INGEST_RUNS.c.files_failed
            ),
        )
    engine.dispose()
    assert [tuple(row) for row in found] == [(2, 1, 1)]


def test_each_root_gets_its_own_run(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A run covers one root, because a consumer asks about one root."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_metadata(first, 'VOL/N1454725799_1_CALIB', metadata_document())
    write_metadata(second, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.root_url))
    engine.dispose()
    assert sorted(row.root_url for row in found) == sorted([first.as_posix(), second.as_posix()])


def test_a_chunk_boundary_is_crossed_mid_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More images than one transaction holds must all still arrive."""
    monkeypatch.setattr(driver_module, 'INGEST_COMMIT_CHUNK_SIZE', 3)
    monkeypatch.setattr(chunks_module, 'RETRIEVE_BATCH_SIZE', 2)
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    url = index_url(tmp_path / 'index.sqlite3')
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_ingested == 7


def test_a_chunk_boundary_leaves_every_row_readable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting is not the same as having committed."""
    monkeypatch.setattr(driver_module, 'INGEST_COMMIT_CHUNK_SIZE', 3)
    monkeypatch.setattr(chunks_module, 'RETRIEVE_BATCH_SIZE', 2)
    root = tmp_path / 'results'
    for index in range(7):
        write_metadata(
            root,
            f'VOL/N145472579{index}_1_CALIB',
            metadata_document(image_name=f'N145472579{index}_1_CALIB.IMG'),
        )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 7


def test_the_batch_and_chunk_sizes_are_independent() -> None:
    """One bounds a download and the other a transaction; neither implies the other."""
    assert (RETRIEVE_BATCH_SIZE, INGEST_COMMIT_CHUNK_SIZE) == (64, 512)


def _cloud_style_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubs: list[str]) -> Path:
    """Make a local tree behave the way a cloud root behaves.

    On a cloud root, ``get_local_path`` names the file the cache *would* hold
    and does not put anything there; only ``retrieve`` downloads.  Here the
    documents are written to one directory and the root the ingest is handed
    lists them but holds no readable file, so a caller that names a file
    instead of retrieving it gets a path with nothing behind it.

    Parameters:
        tmp_path: Directory both trees live under.
        monkeypatch: Fixture the retrieval is redirected through.
        stubs: Results path stubs the tree holds.

    Returns:
        The root to hand the ingest.
    """
    origin = tmp_path / 'origin'
    root = tmp_path / 'results'
    for stub in stubs:
        write_metadata(origin, stub, metadata_document())
        # The listing sees a file of the right name, size and time; its
        # contents are not the document, so reading it in place fails.
        placeholder = root / f'{stub}{METADATA_SUFFIX}'
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.write_text('not the document', encoding='utf-8')

    def downloading(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        paths = sub_path if isinstance(sub_path, list) else [sub_path]
        return [Path(origin / str(one)) for one in paths]

    monkeypatch.setattr(FCPath, 'retrieve', downloading)
    return root


def test_a_cloud_style_root_downloads_its_files(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_local_path`` names a file on a cloud root; it does not fetch one.

    An ingest that named the file instead of retrieving it would read whatever
    happened to be at that path, which on a cloud root is nothing.
    """
    root = _cloud_style_root(tmp_path, monkeypatch, ['VOL/N1454725799_1_CALIB'])
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_cloud_style_root_reads_the_downloaded_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the row holds what the downloaded document said, not the placeholder."""
    root = _cloud_style_root(tmp_path, monkeypatch, ['VOL/N1454725799_1_CALIB'])
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.image_name))
    engine.dispose()
    assert [row.image_name for row in found] == ['N1454725799_1_CALIB.IMG']


def test_an_unretrievable_file_is_counted_not_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batched retrieval reports its failures rather than raising on one.

    The stand-in honors ``exception_on_fail`` rather than swallowing it, which
    is what makes this a test of the call and not only of the handling: the
    storage layer raises unless it is asked not to, so a caller that stops
    passing the keyword ends the run here instead of counting one file.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def failing(
        self: FCPath, sub_path: Any = None, *, exception_on_fail: bool = True, **kwargs: Any
    ) -> Any:
        errors = [FileNotFoundError('gone') for _ in sub_path]
        if exception_on_fail:
            raise errors[0]
        return errors

    monkeypatch.setattr(FCPath, 'retrieve', failing)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'could not be retrieved': 1}


def _tree_with_a_dangling_symlink(tmp_path: Path) -> Path:
    """Build a results tree holding one real document and one broken link.

    A dangling symlink is the ordinary way a file the walk listed cannot be
    retrieved: the listing names it, and the download finds nothing behind it.
    A re-navigation that moved its output and a partially restored backup both
    leave them.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    link = root / 'VOL' / f'N1454725800_1_CALIB{METADATA_SUFFIX}'
    link.symlink_to(tmp_path / 'nowhere' / f'gone{METADATA_SUFFIX}')
    return root


def test_a_dangling_symlink_costs_only_its_own_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same guarantee against a real broken file rather than a stand-in."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


def test_a_dangling_symlink_is_counted_as_unretrievable(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """And it is counted, so a tree full of them does not read as a clean pass."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'could not be retrieved': 1}


def test_a_dangling_symlink_leaves_a_completed_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run that ended on one would leave every consumer refusing the root."""
    root = _tree_with_a_dangling_symlink(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_a_missing_root_is_reported_rather_than_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A root that is not there holds no documents, and says so by counting none."""
    counts = ingest_tree(
        index_url(tmp_path / 'index.sqlite3'), [tmp_path / 'absent'], logger=quiet_logger
    )
    assert counts.files_seen == 0


def test_a_root_is_normalized_before_it_is_stored(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing separator must not make one root into two."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    ingest_metadata_files(engine, [f'{root.as_posix()}/'], logger=quiet_logger)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.root_url))
    engine.dispose()
    assert [row.root_url for row in found] == [root.as_posix()]


def _ingest_two_spellings_of_one_root(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest one root named twice, with and without a trailing separator.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    try:
        ingest_metadata_files(engine, [root.as_posix(), f'{root.as_posix()}/'], logger=logger)
    finally:
        engine.dispose()
    return url


def test_two_spellings_of_one_root_are_walked_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A trailing separator is not another root, in this mode as in the others.

    Walked twice, every document under it is read twice and the tree is listed
    twice -- the most expensive thing an ingest does, and a paid round trip per
    directory on a cloud root.
    """
    url = _ingest_two_spellings_of_one_root(tmp_path, quiet_logger)
    assert len(run_rows(url)) == 1


def test_two_spellings_of_one_root_read_their_documents_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what the second pass over the same tree costs."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    engine = open_index(index_url(tmp_path / 'index.sqlite3'), create=True)
    try:
        counts = ingest_metadata_files(
            engine, [root.as_posix(), f'{root.as_posix()}/'], logger=quiet_logger
        )
    finally:
        engine.dispose()
    assert counts.files_seen == 1


def _ingest_a_relative_root(tmp_path: Path, logger: pdslogger.PdsLogger) -> str:
    """Ingest a root named relatively to the working directory.

    A relative root is a documented spelling of the option, and the walk is
    handed the same normalized form the rows are keyed by rather than the
    string as typed.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL, for whatever the caller means to read from it.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    engine = open_index(url, create=True)
    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        ingest_metadata_files(engine, ['results'], logger=logger)
    finally:
        os.chdir(previous)
        engine.dispose()
    return url


def test_a_relative_root_is_ingested(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The walk is handed the normalized root, which is the absolute one.

    Handing it the string as typed asks the storage layer for a relative local
    URL, which it refuses outright -- a traceback out of a console entry point,
    over a spelling the option documents.
    """
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == ['VOL/N1454725799_1_CALIB']


def test_a_relative_root_completes_its_run(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run left unfinished is a root every consumer afterwards refuses."""
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc))
    engine.dispose()
    assert [row.finished_utc is not None for row in found] == [True]


def test_a_relative_root_records_an_absolute_source_file(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The source file names the same location the root does, not a shorter one."""
    url = _ingest_a_relative_root(tmp_path, quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.source_file))
    engine.dispose()
    assert [str(row.source_file).startswith(tmp_path.as_posix()) for row in found] == [True]


def test_a_second_ingest_of_the_same_root_adds_no_row(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Ingest is idempotent: the same tree twice is the same one row per image."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    ingest_tree(url, [root], logger=quiet_logger, force=True)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(sqlalchemy.func.count()).select_from(IMAGES))
    engine.dispose()
    assert found[0][0] == 1


def test_the_source_file_records_where_the_document_came_from(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Provenance: which file on which root produced this row."""
    root = tmp_path / 'results'
    path = write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.source_file))
    engine.dispose()
    assert [row.source_file for row in found] == [path.as_posix()]


def test_an_offset_survives_the_database_bit_for_bit(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Fifteen significant digits, through the column and back."""
    root = tmp_path / 'results'
    write_metadata(
        root,
        'VOL/N1454725799_1_CALIB',
        metadata_document(offset=[3.14159265358979, -2.71828182845905]),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.offset_dv))
    engine.dispose()
    assert found[0][0] == 3.14159265358979


def test_a_deep_tree_is_walked_to_the_bottom(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A real results tree is volume, then range, then image."""
    root = tmp_path / 'results'
    stub = 'COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB'
    write_metadata(root, stub, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.results_path_stub))
    engine.dispose()
    assert [row.results_path_stub for row in found] == [stub]


def test_a_file_that_is_not_a_result_is_left_alone(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only the two result suffixes are collected; a tree holds other files."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'VOL' / 'notes.txt').write_text('ignore me', encoding='utf-8')
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_seen == 1


def test_the_written_row_survives_a_plain_sqlite_reader(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The index is an ordinary database; opening it directly is supported."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    connection = sqlite3.connect(database)
    try:
        names = [row[0] for row in connection.execute('SELECT image_name FROM images')]
    finally:
        connection.close()
    assert names == ['N1454725799_1_CALIB.IMG']


def test_the_excluded_set_is_stored_as_json(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The column is JSON on both backends, so a direct query can reach inside."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document(excluded=['BodyBlobNav']))
    database = tmp_path / 'index.sqlite3'
    ingest_tree(index_url(database), [root], logger=quiet_logger)
    connection = sqlite3.connect(database)
    try:
        stored = connection.execute('SELECT excluded_from_consensus FROM images').fetchone()[0]
    finally:
        connection.close()
    assert json.loads(stored) == ['BodyBlobNav']


def test_the_excluded_set_is_stored_in_name_order() -> None:
    """The report joins these names into one cell, so their order is its output.

    A document lists them in whatever order the ensemble dropped them, which is
    not stable between two runs over the same image. Sorting here is what makes
    two reports of the same tree comparable.
    """
    document = metadata_document(excluded=['StarRefineNav', 'BodyBlobNav', 'RingEdgeNav'])
    rows = rows_from_metadata(document, SOURCE)
    assert rows.image['excluded_from_consensus'] == [
        'BodyBlobNav',
        'RingEdgeNav',
        'StarRefineNav',
    ]
