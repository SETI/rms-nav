"""Tests for reading a navigation results tree into the results index.

What these pin is mostly what ingest does *not* do: it does not key an image by
its name, does not round the offset it stores, does not merge the two reason
vocabularies, does not read a document whose file has not changed, does not
stat a file it already listed, and does not stop when a file turns out not to
be a navigation document at all.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath

from spindoctor.cli.stats import ingest as ingest_module
from spindoctor.cli.stats.ingest import (
    INGEST_COMMIT_CHUNK_SIZE,
    INGEST_RETRIEVE_BATCH_SIZE,
    METADATA_SUFFIX,
    ingest_metadata_files,
)
from spindoctor.cli.stats.ingest_rows import (
    MetadataDocumentError,
    MetadataSource,
    rows_from_metadata,
)
from spindoctor.results_index import IMAGES, INGEST_RUNS, TECHNIQUES, open_index

from .conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
)

SOURCE = MetadataSource(
    root_url='/data/nav-results',
    results_path_stub='COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    source_file='/data/nav-results/x_metadata.json',
    mtime_ns=1234567890123456789,
    size_bytes=4096,
    has_summary_png=True,
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


def test_the_volume_is_the_stubs_first_segment() -> None:
    """A volume-qualified stub yields the volume without string surgery in SQL."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['volume'] == 'COISS_2001'


def test_a_bare_basename_stub_has_no_volume() -> None:
    """The simulated dataset produces a stub with no separator, and no volume."""
    source = MetadataSource(
        root_url='/data/nav-results',
        results_path_stub='sim_scene_000042',
        source_file='/data/nav-results/sim_scene_000042_metadata.json',
        mtime_ns=1,
        size_bytes=2,
        has_summary_png=False,
    )
    rows = rows_from_metadata(metadata_document(instrument='sim'), source)
    assert rows.image['volume'] is None


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


def test_the_summary_png_flag_comes_from_the_walk() -> None:
    """Nothing in the document says whether a summary was written beside it."""
    rows = rows_from_metadata(metadata_document(), SOURCE)
    assert rows.image['has_summary_png'] is True


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
        volumes = _rows(connection, sqlalchemy.select(IMAGES.c.volume).order_by(IMAGES.c.volume))
    engine.dispose()
    assert [row.volume for row in volumes] == ['COISS_2001', 'COISS_2002']


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


def test_a_bare_basename_stub_ingests_with_a_null_volume(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A simulated scene lives at the root of the tree and names no volume."""
    root = tmp_path / 'results'
    write_metadata(root, 'sim_scene_000042', metadata_document(instrument='sim'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(connection, sqlalchemy.select(IMAGES.c.volume))
    engine.dispose()
    assert [row.volume for row in found] == [None]


def test_the_summary_png_is_seen_by_the_walk(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One listing collects both suffixes, so nothing asks a second time."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    (root / 'VOL' / 'N1454725799_1_CALIB_summary.png').write_bytes(b'\x89PNG')
    write_metadata(root, 'VOL/N1454725800_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    engine = open_index(url)
    with engine.connect() as connection:
        found = _rows(
            connection,
            sqlalchemy.select(IMAGES.c.results_path_stub, IMAGES.c.has_summary_png).order_by(
                IMAGES.c.results_path_stub
            ),
        )
    engine.dispose()
    assert [bool(row.has_summary_png) for row in found] == [True, False]


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


def test_an_unchanged_file_is_not_stat_ed_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing already carried the metrics; asking again is a round trip."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)

    def forbidden(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('ingest asked the backend about one file at a time')

    monkeypatch.setattr(FCPath, 'stat', forbidden)
    monkeypatch.setattr(FCPath, 'exists', forbidden)
    counts = ingest_tree(url, [root], logger=quiet_logger)
    assert counts.files_skipped == 1


def test_a_first_ingest_asks_about_no_single_file_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk feeds presence and both metrics, so a first pass is one listing."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def forbidden(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('ingest asked the backend about one file at a time')

    monkeypatch.setattr(FCPath, 'stat', forbidden)
    monkeypatch.setattr(FCPath, 'exists', forbidden)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.files_ingested == 1


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

    real_walk = ingest_module._walk_root

    def watching(walk_root: Any, **kwargs: Any) -> Any:
        with engine.connect() as connection:
            seen.extend(_rows(connection, sqlalchemy.select(INGEST_RUNS.c.finished_utc)))
        return real_walk(walk_root, **kwargs)

    monkeypatch.setattr(ingest_module, '_walk_root', watching)
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
    monkeypatch.setattr(ingest_module, 'INGEST_COMMIT_CHUNK_SIZE', 3)
    monkeypatch.setattr(ingest_module, 'INGEST_RETRIEVE_BATCH_SIZE', 2)
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
    monkeypatch.setattr(ingest_module, 'INGEST_COMMIT_CHUNK_SIZE', 3)
    monkeypatch.setattr(ingest_module, 'INGEST_RETRIEVE_BATCH_SIZE', 2)
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
    assert (INGEST_RETRIEVE_BATCH_SIZE, INGEST_COMMIT_CHUNK_SIZE) == (64, 512)


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
    """A batched retrieval reports its failures rather than raising on one."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())

    def failing(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        return [FileNotFoundError('gone') for _ in sub_path]

    monkeypatch.setattr(FCPath, 'retrieve', failing)
    counts = ingest_tree(index_url(tmp_path / 'index.sqlite3'), [root], logger=quiet_logger)
    assert counts.failures_by_reason == {'could not be retrieved': 1}


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
