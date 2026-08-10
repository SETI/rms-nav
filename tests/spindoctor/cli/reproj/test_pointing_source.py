"""Reading one image's navigation record from a file and from an index.

The guarantee under test is equivalence: for every record ingest stored, the
two sources classify the same pointing, carry the same recorded values, and
write the same warning to the image's log.  Where they cannot -- because ingest
refused a document, or coerced a value the file path can still read -- the
difference is asserted rather than assumed, so a later change that silently
widened it fails here.

Every lookup is filtered on the root as well as the stub, and a two-root fixture
whose second root differs in exactly the value under test is what proves it: a
query that dropped the root would answer from whichever row it found first, and
the two directions of the same assertion cannot both be satisfied by one row.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pdslogger
import pytest
from filecache import FCPath
from tests.spindoctor.cli.reproj.conftest import (
    BOOLEAN_OFFSET_STUB,
    CAMERA_FRAME_ONLY_STUB,
    CMATRIX,
    CMATRIX_ORIGINAL,
    CMATRIX_STUB,
    FAILED_STUB,
    FITTED_STUB,
    MALFORMED_OFFSET_STUB,
    MIDTIME_ET,
    NAN_MIDTIME_STUB,
    NESTED_CMATRIX_STUB,
    NO_MIDTIME_STUB,
    NO_OFFSET_KEY_STUB,
    NO_POINTING_STUB,
    NO_STATUS_ERROR_STUB,
    NO_TOP_LEVEL_STATUS_STUB,
    NON_FINITE_OFFSET_STUB,
    NOT_A_ROTATION_STUB,
    NULL_OFFSET_STUB,
    OFFSET,
    POINTING,
    SUCCESS_NO_OFFSET_KEY_STUB,
    TIMES,
    UNNAVIGATED_STUB,
    build_tree,
    document,
    image_file,
    index_for,
)

from spindoctor.cli.reproj.offsets import PointingMechanism, PointingSelection
from spindoctor.cli.reproj.pointing_source import (
    FilePointingSource,
    IndexPointingSource,
    PointingSource,
    build_pointing_source,
)
from spindoctor.config import IMAGE_LOGGER, LogLevels, LogSinks, build_image_log_handlers
from spindoctor.results_index import INGEST_RUNS, normalize_root_url, open_index

_STAMP = '2026-08-08T12-00-00'


def _selection(sources: dict[str, PointingSource], mode: str, stub: str) -> PointingSelection:
    """Look one stub up through one of the two sources.

    Parameters:
        sources: The pair of sources over the fixture tree.
        mode: ``'file'`` or ``'index'``.
        stub: The results path stub to look up.

    Returns:
        The classified selection.
    """
    return sources[mode].load_pointing(image_file(stub))


# ---------------------------------------------------------------------------
# The reasons both paths reach, asserted from the same fixture
# ---------------------------------------------------------------------------

_SAME_REASON = [
    (CMATRIX_STUB, None),
    (FITTED_STUB, 'no_cmatrix_rotation_fitted'),
    (NO_POINTING_STUB, 'no_pointing_block'),
    (FAILED_STUB, 'navigation_did_not_succeed'),
    (NULL_OFFSET_STUB, 'null_offset'),
    (NO_MIDTIME_STUB, 'malformed_pointing'),
    (NOT_A_ROTATION_STUB, 'malformed_pointing'),
    (NAN_MIDTIME_STUB, 'malformed_pointing'),
    (UNNAVIGATED_STUB, 'no_metadata'),
]


@pytest.mark.parametrize(('stub', 'reason'), _SAME_REASON)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_report_the_same_reason(
    sources: dict[str, PointingSource], mode: str, stub: str, reason: str | None
) -> None:
    """Each reachable reason is reported the same whichever storage answers."""
    assert _selection(sources, mode, stub).reason == reason


@pytest.mark.parametrize(
    ('stub', 'mechanism'),
    [
        (CMATRIX_STUB, PointingMechanism.CMATRIX),
        (FITTED_STUB, PointingMechanism.OFFSET),
        (NO_POINTING_STUB, PointingMechanism.OFFSET),
        (FAILED_STUB, PointingMechanism.NONE),
        (NULL_OFFSET_STUB, PointingMechanism.NONE),
        (NO_MIDTIME_STUB, PointingMechanism.OFFSET),
        (NOT_A_ROTATION_STUB, PointingMechanism.OFFSET),
        (NAN_MIDTIME_STUB, PointingMechanism.OFFSET),
        (UNNAVIGATED_STUB, PointingMechanism.NONE),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_select_the_same_mechanism(
    sources: dict[str, PointingSource], mode: str, stub: str, mechanism: PointingMechanism
) -> None:
    """And the mechanism, which is what decides how the product is built."""
    assert _selection(sources, mode, stub).mechanism is mechanism


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_carry_the_same_offset(sources: dict[str, PointingSource], mode: str) -> None:
    """The offset a fallback would apply survives both storages unrounded."""
    assert _selection(sources, mode, NO_POINTING_STUB).offset == (OFFSET[0], OFFSET[1])


# ---------------------------------------------------------------------------
# The recorded attitude, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_cmatrix_survives_bit_for_bit(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """A rotation of full-mantissa float64 comes back exactly as recorded.

    The reader's flip gate holds the recovered rotation to 1e-9, so a storage
    that rounded even the last place would refuse records the file path
    accepts.
    """
    selection = _selection(sources, mode, CMATRIX_STUB)
    assert selection.cmatrix is not None
    assert np.array_equal(selection.cmatrix, np.asarray(CMATRIX).reshape(3, 3))


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_baseline_survives_bit_for_bit(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """So does the as-flown attitude the gate is computed against."""
    selection = _selection(sources, mode, CMATRIX_STUB)
    assert selection.cmatrix_original is not None
    assert np.array_equal(selection.cmatrix_original, np.asarray(CMATRIX_ORIGINAL).reshape(3, 3))


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_midtime_survives_exactly(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """And the midtime, which the reader gates against the observation at 1e-6 s.

    Asserted exactly rather than approximately: a gate that tight leaves the
    stored value no room to be nearly right.
    """
    assert _selection(sources, mode, CMATRIX_STUB).midtime_et == MIDTIME_ET


def test_the_index_carries_the_recorded_midtime_not_a_recomputed_one(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The stored midtime is what the document said, not the shutter midpoint.

    The two agree for the frame the rest of these tests use, which is exactly
    why the distinction needs a record where they do not: an implementation
    that recomputed the value from ``start_et`` and ``stop_et`` would pass
    every other assertion here and then gate real records against an epoch
    nobody recorded.
    """
    displaced = dict(TIMES)
    displaced['midtime_et'] = TIMES['start_et'] + 0.001
    root = tmp_path / 'nav'
    build_tree(
        root,
        {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET, times=displaced, pointing=POINTING)},
    )
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(root))
        selection = source.load_pointing(image_file(CMATRIX_STUB))
        assert selection.midtime_et == displaced['midtime_et']
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# The reasons only one path can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('stub', 'file_reason'),
    [
        (MALFORMED_OFFSET_STUB, 'malformed_offset'),
        (NO_OFFSET_KEY_STUB, 'missing_offset_key'),
        (NON_FINITE_OFFSET_STUB, 'non_finite_offset'),
        (BOOLEAN_OFFSET_STUB, 'invalid_offset_type'),
    ],
)
def test_the_file_path_distinguishes_every_unusable_offset(
    sources: dict[str, PointingSource], stub: str, file_reason: str
) -> None:
    """Reading the document tells the four unusable-offset shapes apart."""
    assert _selection(sources, 'file', stub).reason == file_reason


@pytest.mark.parametrize(
    'stub',
    [MALFORMED_OFFSET_STUB, NO_OFFSET_KEY_STUB, NON_FINITE_OFFSET_STUB, BOOLEAN_OFFSET_STUB],
)
def test_the_index_path_reports_them_all_as_a_null_offset(
    sources: dict[str, PointingSource], stub: str
) -> None:
    """Ingest stores every one of them as NULL, so the index reports one reason.

    A real behavioral difference between the paths, asserted rather than
    papered over: the product is the same either way, because none of the four
    supplies a pointing.
    """
    assert _selection(sources, 'index', stub).reason == 'null_offset'


@pytest.mark.parametrize(
    'stub',
    [MALFORMED_OFFSET_STUB, NO_OFFSET_KEY_STUB, NON_FINITE_OFFSET_STUB, BOOLEAN_OFFSET_STUB],
)
def test_neither_path_supplies_a_pointing_for_an_unusable_offset(
    sources: dict[str, PointingSource], stub: str
) -> None:
    """Which is what makes the differing reason a name rather than a product."""
    assert _selection(sources, 'index', stub).mechanism is PointingMechanism.NONE


def test_the_classifier_accepts_a_rotation_written_as_a_nesting(
    sources: dict[str, PointingSource],
) -> None:
    """A 3x3 nesting is a shape the classifier reads and the producer never writes."""
    assert _selection(sources, 'file', NESTED_CMATRIX_STUB).mechanism is PointingMechanism.CMATRIX


def test_the_index_cannot_store_a_rotation_written_as_a_nesting(
    sources: dict[str, PointingSource],
) -> None:
    """And is the one record class whose product differs between the paths.

    Ingest stores a rotation only in the nine row-major floats its producer
    writes, so a nested one becomes a NULL ``cmatrix`` beside a stored
    baseline, which the classifier then reads as a fitted rotation and answers
    with the offset. Pinned rather than left to be discovered: no navigation
    writes this shape, and one that started to would change what an
    index-backed run built.
    """
    assert _selection(sources, 'index', NESTED_CMATRIX_STUB).mechanism is PointingMechanism.OFFSET


def test_a_document_that_is_not_an_object_has_no_row_at_all(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Ingest refuses it, so the index reports the image as never navigated.

    The file path calls the same document ``metadata_not_an_object``; the
    index has nothing to call it, which is why that reason is unreachable
    there.
    """
    root = tmp_path / 'nav'
    path = root / f'{NO_POINTING_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(root))
        assert source.load_pointing(image_file(NO_POINTING_STUB)).reason == 'no_metadata'
    finally:
        engine.dispose()


def test_the_file_path_calls_that_same_document_something_else(tmp_path: Path) -> None:
    """Which is the difference the module docstring records."""
    root = tmp_path / 'nav'
    path = root / f'{NO_POINTING_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    source = FilePointingSource(FCPath(root))
    assert source.load_pointing(image_file(NO_POINTING_STUB)).reason == 'metadata_not_an_object'


# ---------------------------------------------------------------------------
# The two NULL-cmatrix cases
# ---------------------------------------------------------------------------


def test_a_fitted_rotation_row_is_not_read_as_a_missing_pointing_block(
    sources: dict[str, PointingSource],
) -> None:
    """A result that fitted a camera rotation stores no cmatrix and a baseline.

    Both that record and one with no pointing block at all leave ``cmatrix``
    NULL, so the column cannot separate them; the recorded baseline can, and
    the reason is what a run-level tally counts each class under.
    """
    assert _selection(sources, 'index', FITTED_STUB).reason == 'no_cmatrix_rotation_fitted'


def test_a_record_with_no_pointing_block_is_not_read_as_a_fitted_rotation(
    sources: dict[str, PointingSource],
) -> None:
    """The other half of that distinction, which one column alone would lose."""
    assert _selection(sources, 'index', NO_POINTING_STUB).reason == 'no_pointing_block'


def test_a_pointing_block_of_uncolumned_fields_is_read_as_a_fitted_rotation(
    sources: dict[str, PointingSource],
) -> None:
    """Reading the document, a block with no cmatrix is a fitted rotation.

    Whatever else the block holds: the classifier keys on the absence of the
    corrected attitude, not on what is there beside it.
    """
    selection = _selection(sources, 'file', CAMERA_FRAME_ONLY_STUB)
    assert selection.reason == 'no_cmatrix_rotation_fitted'


def test_the_index_reads_that_same_block_as_no_block_at_all(
    sources: dict[str, PointingSource],
) -> None:
    """Because none of its fields is a column, so the row records no block.

    A behavioral difference, pinned rather than left to be found: the mechanism
    and the product are the same, and a run-level tally counts the image under
    the other class.  No navigation writes such a block.
    """
    selection = _selection(sources, 'index', CAMERA_FRAME_ONLY_STUB)
    assert selection.reason == 'no_pointing_block'


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_that_block_selects_the_recorded_offset_either_way(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """Which is what makes the differing reason a name rather than a product."""
    assert _selection(sources, mode, CAMERA_FRAME_ONLY_STUB).mechanism is PointingMechanism.OFFSET


# ---------------------------------------------------------------------------
# The outcome a document names, and the one it does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_document_naming_no_outcome_supplies_no_pointing(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """A nested copy of the status does not stand in for the top-level field.

    The ladder's first question is whether the document's own ``status`` is
    ``success``; a document that names none supplies no pointing at all.  The
    document under test carries a nested ``success`` beside a usable recorded
    attitude, so a column standing that copy in for the field would apply the
    corrected attitude here and nothing at all through the document.
    """
    selection = _selection(sources, mode, NO_TOP_LEVEL_STATUS_STUB)
    assert selection.reason == 'navigation_did_not_succeed'


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_document_naming_no_outcome_leaves_the_pointing_uncorrected(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """The half of that which decides the product rather than the tally."""
    selection = _selection(sources, mode, NO_TOP_LEVEL_STATUS_STUB)
    assert selection.mechanism is PointingMechanism.NONE


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_document_naming_no_outcome_reads_back_as_naming_none(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """And the record reads back with no status, which is what the skip reports.

    The backplane stage puts this value in its skip result and in the run log,
    and a cloud task has no other channel for it.
    """
    record = sources[mode].read_record(image_file(NO_TOP_LEVEL_STATUS_STUB))
    assert record.get('status') is None


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_an_unsuccessful_record_naming_no_error_reads_back_as_naming_none(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """An absent ``status_error`` is absent, not null.

    Every failed and conflicted navigation writes no such field, and the
    backplane stage reports an absent one as ``unknown`` by defaulting.  A
    rebuilt record carrying the key with a null value would default to nothing
    and report ``None`` instead, on the commonest unsuccessful record there is.
    """
    record = sources[mode].read_record(image_file(NO_STATUS_ERROR_STUB))
    assert 'status_error' not in record


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_recorded_error_still_reads_back(sources: dict[str, PointingSource], mode: str) -> None:
    """The control for the absence above, which an empty record would pass."""
    record = sources[mode].read_record(image_file(FAILED_STUB))
    assert record['status_error'] == 'missing_spice_data'


# ---------------------------------------------------------------------------
# A success carrying no offset field, which the two paths build differently
# ---------------------------------------------------------------------------


def test_the_file_path_sees_that_a_success_carries_no_offset_field(
    sources: dict[str, PointingSource],
) -> None:
    """Reading the document, the absent key is visible as an absent key.

    The backplane stage refuses such a record rather than building geometry on
    something shaped like a defect.
    """
    assert _selection(sources, 'file', SUCCESS_NO_OFFSET_KEY_STUB).offset_key_present is False


def test_the_index_reports_that_same_record_as_carrying_a_null_offset(
    sources: dict[str, PointingSource],
) -> None:
    """Because ingest stores an absent offset and a null one alike.

    The rebuild renders the pair as a key holding null deliberately: a
    malformed, non-finite or genuinely null offset is stored the same way, and
    those are records the backplane stage builds products from, so rendering
    the pair as an absent key would refuse three reachable shapes to agree
    about one that no navigation writes.  This is the record whose *product*
    differs between the paths, and it is pinned here so that a change to either
    side has to say so.
    """
    assert _selection(sources, 'index', SUCCESS_NO_OFFSET_KEY_STUB).offset_key_present is True


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_still_select_the_recorded_attitude_for_it(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """The difference is the refusal, not the pointing: both read the C-matrix."""
    selection = _selection(sources, mode, SUCCESS_NO_OFFSET_KEY_STUB)
    assert selection.mechanism is PointingMechanism.CMATRIX


# ---------------------------------------------------------------------------
# The warnings an image's own log carries
# ---------------------------------------------------------------------------


def _log_of(source: PointingSource, stub: str, log_root: Path) -> str:
    """Look one stub up inside an image scope and return the image's log.

    Parameters:
        source: The source to look up through.
        stub: The results path stub.
        log_root: Root the per-image log is written under.

    Returns:
        The text of the image's log.
    """
    handlers, path = build_image_log_handlers(
        'reproj',
        stub,
        LogSinks(log_root=FCPath(log_root)),
        LogLevels(),
        timestamp=_STAMP,
    )
    try:
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            source.load_pointing(image_file(stub))
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


@pytest.mark.parametrize(
    ('stub', 'expected'),
    [
        (UNNAVIGATED_STUB, 'No navigation record for'),
        (FAILED_STUB, "status='error'"),
        (NULL_OFFSET_STUB, 'null offset'),
        (NO_MIDTIME_STUB, 'malformed pointing block'),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_image_log_says_the_same_thing_either_way(
    sources: dict[str, PointingSource],
    tmp_path: Path,
    mode: str,
    stub: str,
    expected: str,
) -> None:
    """The per-image warnings keep their message shapes in the index path.

    A reader of one image's log, and anything scraping those logs, sees the
    same line whichever storage the run was pointed at.
    """
    assert expected in _log_of(sources[mode], stub, tmp_path / f'logs_{mode}')


def test_the_missing_record_warning_names_the_storage_that_was_searched(
    sources: dict[str, PointingSource], tmp_path: Path
) -> None:
    """An index that holds no row names itself, not a results root it never read.

    An index is a snapshot of its last ingest, so a row can be absent because
    nothing navigated the image or because the image was navigated since; a
    message naming a results root would rule the second out for a reader when
    it is the likelier of the two.
    """
    log_text = _log_of(sources['index'], UNNAVIGATED_STUB, tmp_path / 'logs_index')
    assert 'a snapshot of its last ingest' in log_text


def test_the_same_warning_names_the_documents_when_documents_were_searched(
    sources: dict[str, PointingSource], tmp_path: Path
) -> None:
    """The other half: reading documents, the message names the results root."""
    log_text = _log_of(sources['file'], UNNAVIGATED_STUB, tmp_path / 'logs_file')
    assert 'the navigation results under' in log_text


# ---------------------------------------------------------------------------
# The whole record, for the backplane stage
# ---------------------------------------------------------------------------


def test_the_record_carries_the_status_the_backplane_stage_skips_on(
    sources: dict[str, PointingSource],
) -> None:
    """The stage reads the status before it decides there is work to do."""
    assert sources['index'].read_record(image_file(FAILED_STUB))['status'] == 'error'


def test_the_record_carries_the_error_the_skip_reports(
    sources: dict[str, PointingSource],
) -> None:
    """And the error it names in the skip it returns."""
    record = sources['index'].read_record(image_file(FAILED_STUB))
    assert record['status_error'] == 'missing_spice_data'


def test_a_stub_absent_from_the_index_raises(sources: dict[str, PointingSource]) -> None:
    """Exactly as a missing document does, so the caller reports it the same way."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['index'].read_record(image_file(UNNAVIGATED_STUB))
    assert UNNAVIGATED_STUB in str(excinfo.value)


def test_the_raise_names_the_index_it_asked(sources: dict[str, PointingSource]) -> None:
    """So a reader can tell a wrong index from an unnavigated image."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['index'].read_record(image_file(UNNAVIGATED_STUB))
    assert 'index.sqlite3' in str(excinfo.value)


def test_a_missing_document_raises_in_the_file_path(
    sources: dict[str, PointingSource],
) -> None:
    """The behavior the index path is matched against, naming the document."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['file'].read_record(image_file(UNNAVIGATED_STUB))
    assert UNNAVIGATED_STUB in str(excinfo.value)


def test_a_stub_that_escapes_the_root_reads_no_record(tmp_path: Path) -> None:
    """A stub resolving outside the root names no record this source may read.

    The same rule both of this class's methods apply, rather than one guarded
    lookup and one that joins whatever it was handed onto the root.
    """
    root = tmp_path / 'nav'
    root.mkdir(parents=True)
    (tmp_path / 'elsewhere_metadata.json').write_text(json.dumps({'status': 'success'}))
    escaping = image_file('../elsewhere')
    with pytest.raises(FileNotFoundError) as excinfo:
        FilePointingSource(FCPath(root)).read_record(escaping)
    assert 'does not name a navigation record' in str(excinfo.value)


def test_a_stub_that_stays_under_the_root_still_reads_its_record(tmp_path: Path) -> None:
    """The control for the refusal above, which a source refusing all would pass."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    record = FilePointingSource(FCPath(root)).read_record(image_file(CMATRIX_STUB))
    assert record['status'] == 'success'


def test_an_index_that_stops_answering_is_reported_as_a_refusal(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """A lost index is a refusal naming it, not the database layer's own exception.

    The callers of this module report a lookup failure against one image, and
    they cannot name SQLAlchemy's exception types: they deliberately do not
    import it.  The index's own tables are dropped here, which is the shape a
    partially restored database has.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    engine = index_for([root], database, logger=quiet_ingest_logger)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql('DROP TABLE images')
        source = IndexPointingSource(engine, normalize_root_url(root))
        with pytest.raises(ValueError, match='could not be read') as excinfo:
            source.read_record(image_file(CMATRIX_STUB))
        assert 'index.sqlite3' in str(excinfo.value)
    finally:
        engine.dispose()


def test_a_file_source_with_no_root_has_nowhere_to_read(tmp_path: Path) -> None:
    """A reprojection run given no navigation results can supply no record."""
    with pytest.raises(FileNotFoundError) as excinfo:
        FilePointingSource(None).read_record(image_file(CMATRIX_STUB))
    assert 'no navigation results root' in str(excinfo.value)


def test_a_document_that_is_not_an_object_is_refused_by_name(tmp_path: Path) -> None:
    """Reading a record has to return one, so a JSON array is refused here.

    Left to the caller it becomes an attribute error from the middle of a
    batch run, naming neither the file nor what is wrong with it.
    """
    root = tmp_path / 'nav'
    path = root / f'{CMATRIX_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    with pytest.raises(ValueError, match='not a JSON object'):
        FilePointingSource(FCPath(root)).read_record(image_file(CMATRIX_STUB))


# ---------------------------------------------------------------------------
# One index, several roots
# ---------------------------------------------------------------------------


def _two_roots(tmp_path: Path, logger: pdslogger.PdsLogger) -> tuple[Path, Path, Any]:
    """Build two results roots holding the same stub with different offsets.

    Parameters:
        tmp_path: Directory both roots and the index are written under.
        logger: Logger the ingest reports through.

    Returns:
        The two roots and the open index over both of them.
    """
    first = tmp_path / 'nav_a'
    second = tmp_path / 'nav_b'
    build_tree(first, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[1.5, -2.5])})
    build_tree(second, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[9.25, 8.75])})
    return first, second, index_for([first, second], tmp_path / 'index.sqlite3', logger=logger)


def test_a_source_reads_its_own_roots_offset(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """One index can hold several roots, and a lookup answers from one of them."""
    first, _second, engine = _two_roots(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(first))
        assert source.load_pointing(image_file(CMATRIX_STUB)).offset == (1.5, -2.5)
    finally:
        engine.dispose()


def test_the_other_source_reads_the_other_roots_offset(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The same assertion from the other side.

    Together with the previous test this is what a query filtering on the stub
    alone cannot satisfy: one row cannot be both answers, so dropping the root
    predicate fails one of the two whichever row the database returns first.
    """
    _first, second, engine = _two_roots(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(second))
        assert source.load_pointing(image_file(CMATRIX_STUB)).offset == (9.25, 8.75)
    finally:
        engine.dispose()


def test_a_stub_recorded_only_under_another_root_is_absent(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """An image navigated under one root was not navigated under the other.

    The backplane raise is what carries this, and a lookup that forgot the root
    would find the neighbour's row and build a product from it.
    """
    first = tmp_path / 'nav_a'
    second = tmp_path / 'nav_b'
    build_tree(first, {NO_POINTING_STUB: document(NO_POINTING_STUB, offset=OFFSET)})
    build_tree(second, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    engine = index_for([first, second], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(first))
        with pytest.raises(FileNotFoundError, match=CMATRIX_STUB):
            source.read_record(image_file(CMATRIX_STUB))
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Choosing a source
# ---------------------------------------------------------------------------


def test_no_url_reads_documents(tmp_path: Path) -> None:
    """No index is every program's default, and it reads the tree."""
    source = build_pointing_source(FCPath(tmp_path), results_db_url=None)
    assert isinstance(source, FilePointingSource)


def test_a_url_reads_rows(tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger) -> None:
    """A resolved URL makes the same program read one row per image."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    source = build_pointing_source(FCPath(root), results_db_url=f'sqlite:///{database.as_posix()}')
    try:
        assert isinstance(source, IndexPointingSource)
    finally:
        source.close()


def test_an_unopenable_url_fails_rather_than_falling_back(tmp_path: Path) -> None:
    """A misconfigured index is a failed run, not a slow and different one."""
    missing = tmp_path / 'nowhere' / 'index.sqlite3'
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(FCPath(tmp_path), results_db_url=f'sqlite:///{missing.as_posix()}')
    assert 'sd_stats_ingest' in str(excinfo.value)


def test_an_unopenable_url_leaves_no_database_behind(tmp_path: Path) -> None:
    """And it does not create the index it was told to read."""
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='sd_stats_ingest') as excinfo:
        build_pointing_source(FCPath(tmp_path), results_db_url=f'sqlite:///{missing.as_posix()}')
    assert 'index.sqlite3' in str(excinfo.value)
    assert not missing.exists()


def test_an_index_with_no_root_to_read_under_is_refused(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Rows are keyed by root, so an index with no root has nothing to answer."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError, match='no navigation results root'):
        build_pointing_source(None, results_db_url=f'sqlite:///{database.as_posix()}')


def test_a_root_nobody_ingested_is_refused(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Absence of a row means "never navigated" only under a fully ingested root.

    Under one nobody ingested it means nothing at all, so the consumer refuses
    rather than reporting every image as unnavigated.
    """
    ingested = tmp_path / 'nav_a'
    other = tmp_path / 'nav_b'
    build_tree(ingested, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    other.mkdir(parents=True, exist_ok=True)
    database = tmp_path / 'index.sqlite3'
    index_for([ingested], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(FCPath(other), results_db_url=f'sqlite:///{database.as_posix()}')
    assert normalize_root_url(other) in str(excinfo.value)


def test_the_refusal_names_the_roots_the_index_does_hold(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """So the reader can see which root they meant to point at."""
    ingested = tmp_path / 'nav_a'
    other = tmp_path / 'nav_b'
    build_tree(ingested, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    other.mkdir(parents=True, exist_ok=True)
    database = tmp_path / 'index.sqlite3'
    index_for([ingested], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(FCPath(other), results_db_url=f'sqlite:///{database.as_posix()}')
    assert normalize_root_url(ingested) in str(excinfo.value)


def test_a_run_that_died_halfway_leaves_a_root_unreadable(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """A root whose newest ingest never finished has not been ingested.

    Its rows are whatever the dead pass wrote, so absence under it says
    nothing, and a consumer must not read it as an answer.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    url = f'sqlite:///{database.as_posix()}'
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            # Restricted to the root under test rather than applied to every
            # row: a fixture that mutates all of them cannot tell a query
            # reading the right row from one reading whichever it found first.
            connection.execute(
                INGEST_RUNS.update()
                .where(INGEST_RUNS.c.root_url == normalize_root_url(root))
                .values(finished_utc=None)
            )
    finally:
        engine.dispose()
    with pytest.raises(ValueError, match='no completed ingest'):
        build_pointing_source(FCPath(root), results_db_url=url)


# ---------------------------------------------------------------------------
# What the rebuilt record does not carry
# ---------------------------------------------------------------------------


def test_the_rebuilt_record_carries_the_recorded_frame_identities(
    sources: dict[str, PointingSource],
) -> None:
    """The pointing block the index holds is rebuilt whole, not just its matrices."""
    record = sources['index'].read_record(image_file(CMATRIX_STUB))
    assert record['navigation_result']['pointing']['ck_frame_id'] == POINTING['ck_frame_id']


def test_the_rebuilt_record_omits_the_camera_frame_name(
    sources: dict[str, PointingSource],
) -> None:
    """No reader consults it: the frame identity comes from the observation.

    Asserted so that a reader which started consulting it would be caught here
    rather than by a product built on a name the index never stored.  The
    document it was ingested from is checked to carry the name, so what is
    asserted is that the rebuild drops it rather than that nothing ever had it.
    """
    ingested = sources['file'].read_record(image_file(CMATRIX_STUB))
    assert 'camera_frame' in ingested['navigation_result']['pointing']
    record = sources['index'].read_record(image_file(CMATRIX_STUB))
    assert 'camera_frame' not in record['navigation_result']['pointing']
