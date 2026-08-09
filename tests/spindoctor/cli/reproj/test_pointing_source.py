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
    CMATRIX,
    CMATRIX_ORIGINAL,
    CMATRIX_STUB,
    FAILED_STUB,
    FITTED_STUB,
    MALFORMED_OFFSET_STUB,
    MIDTIME_ET,
    NO_MIDTIME_STUB,
    NO_OFFSET_KEY_STUB,
    NO_POINTING_STUB,
    NON_FINITE_OFFSET_STUB,
    NOT_A_ROTATION_STUB,
    NULL_OFFSET_STUB,
    OFFSET,
    POINTING,
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
from spindoctor.results_index import normalize_root_url, open_index

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
        (UNNAVIGATED_STUB, 'no metadata found'),
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
    """The behavior the index path is matched against."""
    with pytest.raises(FileNotFoundError):
        sources['file'].read_record(image_file(UNNAVIGATED_STUB))


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
    with pytest.raises(ValueError):
        build_pointing_source(FCPath(tmp_path), results_db_url=f'sqlite:///{missing.as_posix()}')
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
            connection.exec_driver_sql('UPDATE ingest_runs SET finished_utc = NULL')
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
    rather than by a product built on a name the index never stored.
    """
    record = sources['index'].read_record(image_file(CMATRIX_STUB))
    assert 'camera_frame' not in record['navigation_result']['pointing']
