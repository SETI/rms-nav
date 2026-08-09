"""Tests for the one query that answers an enumeration's selection filters.

The rows are written directly rather than ingested, because what is under test
is what the query reads and not what a walk records: a refusal with no image row
beside it, an image whose ``status_error`` is NULL, a stub with no volume, and a
second root holding a row for every stub the first one holds.

That second root is why every test here builds two.  The index is keyed by root
and stub together, and a query that filtered on the stub alone would answer with
the other root's rows while every single-root assertion stayed green.  The other
root is therefore stocked so that every filter's answer changes if one of its
rows leaks: it holds a fatal SPICE error and a summary PNG for each stub, and a
document for the one stub the first root deliberately has none of.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.pool import QueuePool
from tests.spindoctor.results_index.conftest import image_row, opened, sqlite_url_for

from spindoctor.results_index import FAILED_FILES, IMAGES, INGEST_RUNS, SCHEMA_VERSION
from spindoctor.results_index.selection import ResultStubs, _stub_query, read_result_stubs

ROOT = '/data/nav-results'
"""The root under test, spelled as :func:`normalize_root_url` renders it."""

OTHER_ROOT = '/data/other-nav-results'
"""A second ingested root, holding a row for every stub the first one holds."""

VOLUME = 'COISS_2001'
"""The selected volume."""

OTHER_VOLUME = 'COISS_2002'
"""A volume of the same root that the enumeration did not select."""

SUCCESS_WITH_PNG = f'{VOLUME}/data/a/N1000000001_1_CALIB'
SUCCESS_NO_PNG = f'{VOLUME}/data/a/N1000000002_1_CALIB'
FAILURE = f'{VOLUME}/data/a/N1000000003_1_CALIB'
SPICE_ERROR = f'{VOLUME}/data/b/N1000000004_1_CALIB'
NONSPICE_ERROR = f'{VOLUME}/data/b/N1000000005_1_CALIB'
ERROR_WITHOUT_STATUS_ERROR = f'{VOLUME}/data/b/N1000000006_1_CALIB'
REFUSED = f'{VOLUME}/data/b/N1000000007_1_CALIB'
REFUSED_WITH_PNG = f'{VOLUME}/data/b/N1000000012_1_CALIB'
REFUSED_OTHER_VOLUME = f'{OTHER_VOLUME}/data/b/N1000000008_1_CALIB'
UNSELECTED_VOLUME_IMAGE = f'{OTHER_VOLUME}/data/a/N1000000009_1_CALIB'
NO_VOLUME = 'scene_0001'
ONLY_IN_THE_OTHER_ROOT = f'{VOLUME}/data/c/N1000000010_1_CALIB'
ONLY_REFUSED_IN_THE_OTHER_ROOT = f'{VOLUME}/data/c/N1000000011_1_CALIB'

EVERY_STUB = (
    SUCCESS_WITH_PNG,
    SUCCESS_NO_PNG,
    FAILURE,
    SPICE_ERROR,
    NONSPICE_ERROR,
    ERROR_WITHOUT_STATUS_ERROR,
    REFUSED,
    REFUSED_WITH_PNG,
    REFUSED_OTHER_VOLUME,
    UNSELECTED_VOLUME_IMAGE,
    NO_VOLUME,
    ONLY_IN_THE_OTHER_ROOT,
)
"""Every stub either root holds, so the other root can be stocked with all of them."""

SPICE = 'missing_spice_data'
"""The ``status_error`` value the SPICE filters tell apart."""


def _volume_of(stub: str) -> str | None:
    """Return the volume column value a stub gets, as the ingest derives it.

    Parameters:
        stub: The results path stub.

    Returns:
        Its first path segment, or None when it carries no separator.
    """
    volume, separator, _rest = stub.partition('/')
    return volume if separator else None


def _document(stub: str, **columns: Any) -> dict[str, Any]:
    """Return an ``images`` row for one stub of the root under test.

    Every optional column is named even where it is None, because a batched
    insert takes its column list from the first row it is handed: a row that
    omits a key another row carries is stored with that column NULL, silently.

    Parameters:
        stub: The results path stub the row is keyed by.
        columns: Column values replacing the defaults.

    Returns:
        A mapping ready to insert.
    """
    return image_row(
        root_url=ROOT,
        results_path_stub=stub,
        volume=_volume_of(stub),
        **{'has_summary_png': False, 'status': 'success', 'status_error': None, **columns},
    )


def _completed_run(root_url: str) -> dict[str, Any]:
    """Return an ``ingest_runs`` row saying a pass over one root completed.

    Parameters:
        root_url: The root the run covered.

    Returns:
        A mapping ready to insert.
    """
    stamp = datetime.now(UTC).isoformat()
    return {
        'root_url': root_url,
        'started_utc': stamp,
        'finished_utc': stamp,
        'schema_version': SCHEMA_VERSION,
    }


@pytest.fixture
def two_roots(tmp_path: Path) -> str:
    """Build an index holding two ingested roots and return its URL.

    Parameters:
        tmp_path: Directory the index file is written into.

    Returns:
        The connection URL of the index.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    documents = [
        _document(SUCCESS_WITH_PNG, has_summary_png=True),
        _document(SUCCESS_NO_PNG),
        _document(FAILURE, status='failure'),
        _document(SPICE_ERROR, status='error', status_error=SPICE),
        _document(NONSPICE_ERROR, status='error', status_error='bad_pointing'),
        _document(ERROR_WITHOUT_STATUS_ERROR, status='error', status_error=None),
        _document(UNSELECTED_VOLUME_IMAGE, has_summary_png=True),
        _document(NO_VOLUME, has_summary_png=True),
    ]
    refusals = [
        {
            'root_url': root_url,
            'results_path_stub': stub,
            'reason': 'not a current-schema navigation document',
            'volume': _volume_of(stub),
            'has_summary_png': has_summary_png,
            'mtime_ns': 1,
            'size_bytes': 2,
        }
        for root_url, stub, has_summary_png in (
            (ROOT, REFUSED, False),
            (ROOT, REFUSED_WITH_PNG, True),
            (ROOT, REFUSED_OTHER_VOLUME, False),
            # The other root refuses a file the root under test holds nothing
            # for, so a refusal read without its root shows up as a document
            # that exists under a root that has none.
            (OTHER_ROOT, ONLY_REFUSED_IN_THE_OTHER_ROOT, True),
        )
    ]
    # Everything the other root holds would change an answer if it leaked: a
    # fatal SPICE error and a summary PNG for every stub, including the one stub
    # the root under test deliberately holds nothing for.
    other_documents = [
        image_row(
            root_url=OTHER_ROOT,
            results_path_stub=stub,
            volume=_volume_of(stub),
            has_summary_png=True,
            status='error',
            status_error=SPICE,
        )
        for stub in EVERY_STUB
    ]
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), documents)
        connection.execute(IMAGES.insert(), other_documents)
        connection.execute(FAILED_FILES.insert(), refusals)
        connection.execute(INGEST_RUNS.insert(), [_completed_run(ROOT), _completed_run(OTHER_ROOT)])
    return url


def _stubs(url: str, **filters: bool) -> ResultStubs:
    """Read the root under test, selecting its one selected volume.

    Parameters:
        url: The index to read.
        filters: Error filters to apply.

    Returns:
        What the query answered.
    """
    return read_result_stubs(url, ROOT, [VOLUME], **filters)


def test_an_ingested_document_counts_as_present(two_roots: str) -> None:
    """The ordinary case: a row means the file it was read from exists."""
    assert SUCCESS_WITH_PNG in _stubs(two_roots).with_metadata


def test_a_refused_document_counts_as_present(two_roots: str) -> None:
    """A file the ingest could not read is still a file the tree walk finds."""
    assert REFUSED in _stubs(two_roots).with_metadata


def test_a_stub_the_root_holds_nothing_for_is_absent(two_roots: str) -> None:
    """Only the other root holds it, and the other root is not this one."""
    assert ONLY_IN_THE_OTHER_ROOT not in _stubs(two_roots).with_metadata


def test_another_roots_refusal_is_not_this_roots_document(two_roots: str) -> None:
    """The refusals are keyed by root and stub together, exactly as the images are."""
    assert ONLY_REFUSED_IN_THE_OTHER_ROOT not in _stubs(two_roots).with_metadata


def test_the_query_does_not_fetch_an_unselected_volume(two_roots: str) -> None:
    """The volume restriction is in the query and not only in what is kept from it.

    The saving is that an enumeration of one volume does not pay to read the
    rows of every other volume of the root, and a restriction applied only to
    the rows already fetched answers identically while paying for all of them.
    """
    query = _stub_query(ROOT, [VOLUME], sqlalchemy.false())
    with opened(two_roots) as engine, engine.connect() as connection:
        fetched = {str(row[0]) for row in connection.execute(query)}
    assert UNSELECTED_VOLUME_IMAGE not in fetched


def test_a_summary_png_is_read_from_the_document_it_sits_beside(two_roots: str) -> None:
    """The walk records the PNG on the row of the file it was found with."""
    assert _stubs(two_roots).with_summary_png == frozenset({SUCCESS_WITH_PNG, REFUSED_WITH_PNG})


def test_a_summary_png_beside_a_refused_document_is_present(two_roots: str) -> None:
    """A PNG is found beside a file, not read out of it.

    The tree walk finds ``X_summary.png`` whatever ``X_metadata.json`` turned
    out to contain, so a summary beside a document the ingest refused has to
    read as present here too, or an entire results root written by another tool
    answers both PNG filters backwards.
    """
    assert REFUSED_WITH_PNG in _stubs(two_roots).with_summary_png


def test_a_refused_document_with_no_summary_png_is_not_in_the_png_set(two_roots: str) -> None:
    """The flag is the walk's answer for that file, not a constant for the table."""
    assert REFUSED not in _stubs(two_roots).with_summary_png


def test_a_document_with_no_summary_png_is_not_in_the_png_set(two_roots: str) -> None:
    """The other root has a PNG for this stub, and answers for its own root only."""
    assert SUCCESS_NO_PNG not in _stubs(two_roots).with_summary_png


def test_an_image_of_an_unselected_volume_is_not_read(two_roots: str) -> None:
    """The tree walk lists the selected volumes' directories and no others."""
    assert UNSELECTED_VOLUME_IMAGE not in _stubs(two_roots).with_metadata


def test_a_refusal_of_an_unselected_volume_is_not_read(two_roots: str) -> None:
    """A refused file is under a volume like any other, and is restricted to it."""
    assert REFUSED_OTHER_VOLUME not in _stubs(two_roots).with_metadata


def test_the_query_does_not_fetch_an_unselected_volumes_refusal(two_roots: str) -> None:
    """The restriction is in the query for the refusals as much as for the images.

    A root whose non-navigation files outnumber its results is exactly the tree
    the refusal table was made for, and a single-volume enumeration that fetched
    every refusal in the root would pay for all of them on every run.
    """
    query = _stub_query(ROOT, [VOLUME], sqlalchemy.false())
    with opened(two_roots) as engine, engine.connect() as connection:
        fetched = {str(row[0]) for row in connection.execute(query)}
    assert REFUSED_OTHER_VOLUME not in fetched


def test_a_stub_with_no_volume_is_not_read(two_roots: str) -> None:
    """A scene name with no volume above it sits outside every walked directory."""
    assert NO_VOLUME not in _stubs(two_roots).with_metadata


def test_selecting_a_volume_reads_that_volume(two_roots: str) -> None:
    """The unselected volume's images are readable when it is the one selected."""
    stubs = read_result_stubs(two_roots, ROOT, [OTHER_VOLUME])
    assert stubs.with_metadata == frozenset({UNSELECTED_VOLUME_IMAGE, REFUSED_OTHER_VOLUME})


def test_no_volume_selected_reads_nothing(two_roots: str) -> None:
    """An enumeration that selected no volume walks no directory."""
    assert read_result_stubs(two_roots, ROOT, []).with_metadata == frozenset()


def test_no_error_filter_matches_nothing(two_roots: str) -> None:
    """The set is the filters' answer, and no filter was asked."""
    assert _stubs(two_roots).matching_error == frozenset()


def test_the_error_filter_matches_every_fatal_error(two_roots: str) -> None:
    """Whatever ended the run, as long as it ended it."""
    assert _stubs(two_roots, has_offset_error=True).matching_error == frozenset(
        {SPICE_ERROR, NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR}
    )


def test_the_error_filter_does_not_match_a_run_that_finished(two_roots: str) -> None:
    """A navigation that failed to find an offset is not a fatal error."""
    assert FAILURE not in _stubs(two_roots, has_offset_error=True).matching_error


def test_the_spice_filter_matches_only_the_spice_error(two_roots: str) -> None:
    """The value is matched verbatim, which is what the column exists for."""
    assert _stubs(two_roots, has_offset_spice_error=True).matching_error == frozenset({SPICE_ERROR})


def test_the_nonspice_filter_matches_a_document_with_no_status_error(two_roots: str) -> None:
    """A fatal error that named no cause is not a SPICE error.

    The trap this pins is SQL's own: comparing NULL with anything yields NULL,
    so an inequality alone drops the row rather than keeping it, and the filter
    silently loses every fatal error whose cause went unrecorded.
    """
    matching = _stubs(two_roots, has_offset_nonspice_error=True).matching_error
    assert ERROR_WITHOUT_STATUS_ERROR in matching


def test_the_nonspice_filter_rejects_the_spice_error(two_roots: str) -> None:
    """The two error filters partition the fatal errors between them."""
    assert _stubs(two_roots, has_offset_nonspice_error=True).matching_error == frozenset(
        {NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR}
    )


def test_the_error_filters_answer_for_this_root_only(two_roots: str) -> None:
    """Every stub is a SPICE error in the other root and none of them is here."""
    matching = _stubs(two_roots, has_offset_spice_error=True).matching_error
    assert SUCCESS_WITH_PNG not in matching


def test_a_refused_document_matches_no_error_filter(two_roots: str) -> None:
    """Nothing was read from it, so it records no status to match."""
    assert REFUSED not in _stubs(two_roots, has_offset_error=True).matching_error


def test_a_root_with_no_completed_ingest_is_refused(two_roots: str) -> None:
    """Absence of a row means "never navigated" only under a root that was walked."""
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        read_result_stubs(two_roots, '/data/never-ingested', [VOLUME])
    assert '/data/never-ingested' in str(excinfo.value)


def test_an_index_that_cannot_be_opened_is_an_error(tmp_path: Path) -> None:
    """A resolved URL that will not open fails the run rather than reading files."""
    url = sqlite_url_for(tmp_path / 'absent.sqlite3')
    with pytest.raises(ValueError, match='sd_stats_ingest') as excinfo:
        read_result_stubs(url, ROOT, [VOLUME])
    assert 'absent.sqlite3' in str(excinfo.value)


def test_a_relative_root_names_the_root_the_ingest_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative root is a documented spelling of the option, not a second root.

    The root is built under ``tmp_path`` rather than named absolutely, because
    resolving a relative name means changing into its parent, and a directory
    named in the source exists on the machine that wrote it and nowhere else.
    """
    root = tmp_path / 'nav-results'
    root.mkdir()
    url = sqlite_url_for(tmp_path / 'relative.sqlite3')
    root_url = root.as_posix()
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                image_row(
                    root_url=root_url,
                    results_path_stub=SUCCESS_WITH_PNG,
                    volume=VOLUME,
                    has_summary_png=True,
                )
            ],
        )
        connection.execute(INGEST_RUNS.insert(), [_completed_run(root_url)])
    monkeypatch.chdir(tmp_path)
    stubs = read_result_stubs(url, root.name, [VOLUME])
    assert SUCCESS_WITH_PNG in stubs.with_metadata


def test_a_trailing_separator_names_the_root_the_ingest_recorded(two_roots: str) -> None:
    """One program writes the trailing slash on the root and another does not."""
    stubs = read_result_stubs(two_roots, f'{ROOT}/', [VOLUME])
    assert SUCCESS_WITH_PNG in stubs.with_metadata


def _read_recording_the_engine(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[sqlalchemy.Engine, QueuePool]:
    """Answer the filters, recording the engine built for it and its pool.

    Parameters:
        url: The index to read.
        monkeypatch: Fixture the recording hook is installed through.

    Returns:
        The engine the read built, and the pool it started with.
    """
    built: list[tuple[sqlalchemy.Engine, QueuePool]] = []
    create_engine = sqlalchemy.create_engine

    def recording(*args: Any, **kwargs: Any) -> sqlalchemy.Engine:
        made = create_engine(*args, **kwargs)
        # A file-backed SQLite engine pools its connections, and only a pooling
        # implementation can report what it is holding.
        assert isinstance(made.pool, QueuePool)
        built.append((made, made.pool))
        return made

    monkeypatch.setattr(sqlalchemy, 'create_engine', recording)
    _stubs(url)
    return built[0]


def test_the_query_does_not_leave_the_index_open(
    two_roots: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer outlives the connection, and nothing else does.

    A filter is constructed once per enumeration and holds its answer for the
    whole of it, so an undisposed pool would keep a SQLite connection, or a
    server session, for the length of a navigation run.  Disposal replaces the
    pool, which is the observable proof that it happened.
    """
    engine, pool = _read_recording_the_engine(two_roots, monkeypatch)
    assert engine.pool is not pool


def test_the_query_returns_its_connection(two_roots: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool the read built holds no connection afterwards."""
    _engine, pool = _read_recording_the_engine(two_roots, monkeypatch)
    assert pool.checkedin() == 0
