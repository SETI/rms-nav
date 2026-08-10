"""Tests for the one query that answers an enumeration's selection filters.

The rows are written directly rather than ingested, because what is under test
is what the query reads and not what a walk records: a refusal with no image row
beside it, an image whose ``status_error`` is NULL, a stub with no volume, and a
second root holding a row for every stub the first one holds.

That second root is why every test here builds two.  The index is keyed by root
and stub together, and a query that filtered on the stub alone would answer with
the other root's rows while every single-root assertion stayed green.  The other
root is therefore stocked so that every filter's answer changes if one of its
rows leaks: it holds a fatal SPICE error for each stub, and a document for the
one stub the first root deliberately has none of.

The run rows are stocked the same way, because the run table is keyed by root as
well and is read by a query of its own.  The other root is always passed over
last, so its run is the newest in the whole index, and its pass always missed
directories and always finished at a different moment: a count or a finish time
read from the newest run rather than from this root's newest run is therefore
the other root's, and says so.
"""

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy.pool import QueuePool
from tests.spindoctor.results_index.conftest import image_row, opened, sqlite_url_for

from spindoctor.results_index import FAILED_FILES, IMAGES, INGEST_RUNS, SCHEMA_VERSION, selection
from spindoctor.results_index.selection import ResultStubs, _stub_query, read_result_stubs

ROOT = '/data/nav-results'
"""The root under test, spelled as :func:`normalize_root_url` renders it."""

OTHER_ROOT = '/data/other-nav-results'
"""A second ingested root, holding a row for every stub the first one holds."""

VOLUME = 'COISS_2001'
"""The selected volume."""

OTHER_VOLUME = 'COISS_2002'
"""A volume of the same root that the enumeration did not select."""

SUCCESS = f'{VOLUME}/data/a/N1000000001_1_CALIB'
SECOND_SUCCESS = f'{VOLUME}/data/a/N1000000002_1_CALIB'
FAILURE = f'{VOLUME}/data/a/N1000000003_1_CALIB'
SPICE_ERROR = f'{VOLUME}/data/b/N1000000004_1_CALIB'
NONSPICE_ERROR = f'{VOLUME}/data/b/N1000000005_1_CALIB'
ERROR_WITHOUT_STATUS_ERROR = f'{VOLUME}/data/b/N1000000006_1_CALIB'
REFUSED = f'{VOLUME}/data/b/N1000000007_1_CALIB'
REFUSED_OTHER_VOLUME = f'{OTHER_VOLUME}/data/b/N1000000008_1_CALIB'
UNSELECTED_VOLUME_IMAGE = f'{OTHER_VOLUME}/data/a/N1000000009_1_CALIB'
NO_VOLUME = 'scene_0001'
ONLY_IN_THE_OTHER_ROOT = f'{VOLUME}/data/c/N1000000010_1_CALIB'
ONLY_REFUSED_IN_THE_OTHER_ROOT = f'{VOLUME}/data/c/N1000000011_1_CALIB'

EVERY_STUB = (
    SUCCESS,
    SECOND_SUCCESS,
    FAILURE,
    SPICE_ERROR,
    NONSPICE_ERROR,
    ERROR_WITHOUT_STATUS_ERROR,
    REFUSED,
    REFUSED_OTHER_VOLUME,
    UNSELECTED_VOLUME_IMAGE,
    NO_VOLUME,
    ONLY_IN_THE_OTHER_ROOT,
)
"""Every stub either root holds, so the other root can be stocked with all of them."""

SPICE = 'missing_spice_data'
"""The ``status_error`` value the SPICE filters tell apart."""

INGESTED = '2026-02-03T04:05:06+00:00'
"""When the newest pass over the root under test finished."""

OTHER_ROOT_INGESTED = '2026-03-04T05:06:07+00:00'
"""When the newest pass over the other root finished, which is not when this one did."""

OTHER_ROOT_MISSED = 7
"""How many directories the other root's newest pass did not list.

It is not zero, and that pass is the newest in the index, so a count read from
the newest run of the table rather than from the newest run of the root under
test reports this number instead of that root's own.
"""


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
        **{'status': 'success', 'status_error': None, **columns},
    )


def _completed_run(
    root_url: str, *, directories_missed: int = 0, finished_utc: str | None = None
) -> dict[str, Any]:
    """Return an ``ingest_runs`` row saying a pass over one root completed.

    Parameters:
        root_url: The root the run covered.
        directories_missed: How many directories that pass did not list.
        finished_utc: When the pass finished, defaulting to now for the tests
            that do not read the time back.

    Returns:
        A mapping ready to insert.
    """
    stamp = datetime.now(UTC).isoformat() if finished_utc is None else finished_utc
    return {
        'root_url': root_url,
        'started_utc': stamp,
        'finished_utc': stamp,
        'directories_missed': directories_missed,
        'schema_version': SCHEMA_VERSION,
    }


def _other_roots_newest_run() -> dict[str, Any]:
    """Return the run row that makes a root-blind run query answer wrongly.

    It is inserted last, so it is the newest run in the index, and it records a
    finish time and a missed count that the root under test never records.

    Returns:
        A mapping ready to insert.
    """
    return _completed_run(
        OTHER_ROOT, directories_missed=OTHER_ROOT_MISSED, finished_utc=OTHER_ROOT_INGESTED
    )


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
        _document(SUCCESS),
        _document(SECOND_SUCCESS),
        _document(FAILURE, status='failure'),
        _document(SPICE_ERROR, status='error', status_error=SPICE),
        _document(NONSPICE_ERROR, status='error', status_error='bad_pointing'),
        _document(ERROR_WITHOUT_STATUS_ERROR, status='error', status_error=None),
        _document(UNSELECTED_VOLUME_IMAGE),
        _document(NO_VOLUME),
    ]
    refusals = [
        {
            'root_url': root_url,
            'results_path_stub': stub,
            'reason': 'not a current-schema navigation document',
            'volume': _volume_of(stub),
            'mtime_ns': 1,
            'size_bytes': 2,
        }
        for root_url, stub in (
            (ROOT, REFUSED),
            (ROOT, REFUSED_OTHER_VOLUME),
            # The other root refuses a file the root under test holds nothing
            # for, so a refusal read without its root shows up as a document
            # that exists under a root that has none.
            (OTHER_ROOT, ONLY_REFUSED_IN_THE_OTHER_ROOT),
        )
    ]
    # Everything the other root holds would change an answer if it leaked: a
    # fatal SPICE error for every stub, including the one stub the root under
    # test deliberately holds nothing for.
    other_documents = [
        image_row(
            root_url=OTHER_ROOT,
            results_path_stub=stub,
            volume=_volume_of(stub),
            status='error',
            status_error=SPICE,
        )
        for stub in EVERY_STUB
    ]
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), documents)
        connection.execute(IMAGES.insert(), other_documents)
        connection.execute(FAILED_FILES.insert(), refusals)
        connection.execute(
            INGEST_RUNS.insert(),
            [_completed_run(ROOT, finished_utc=INGESTED), _other_roots_newest_run()],
        )
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
    assert SUCCESS in _stubs(two_roots).with_metadata


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


def test_the_negative_error_filter_matches_every_document_recording_no_error(
    two_roots: str,
) -> None:
    """A run that succeeded and a run that finished without one, together.

    This is the selection ``the images this root has a navigated result for``
    is spelled from, so it has to reach an outcome that is not a success as
    surely as it reaches one that is.
    """
    assert _stubs(two_roots, has_no_offset_error=True).matching_error == frozenset(
        {SUCCESS, SECOND_SUCCESS, FAILURE}
    )


def test_the_negative_error_filter_matches_no_fatal_error(two_roots: str) -> None:
    """The two halves of the vocabulary exclude each other, without exhausting.

    No document is in both, which is what this pins.  Documents outside both
    are what the sibling file is for: a file the ingest refused records neither
    an error nor the absence of one.

    A fatal error that named no cause is the row a SQL inequality is most
    likely to mishandle, and it belongs on the other side of this exclusion.
    """
    matching = _stubs(two_roots, has_no_offset_error=True).matching_error
    assert ERROR_WITHOUT_STATUS_ERROR not in matching


def test_the_negative_error_filter_answers_for_this_root_only(tmp_path: Path) -> None:
    """Each root disagrees with the other about both of its stubs.

    It gets a root pair of its own rather than joining the shared fixture,
    because the disagreement has to run the other way round to be visible: the
    shared decoy records a fatal error for every stub, which is what makes a
    filter phrased in the positive answer differently when it drops the root,
    and is exactly why one phrased in the negative would not.

    So this pair holds one stub that errored here and finished there, and one
    that finished here and errored there, and the answer is stated as the whole
    set.  The second stub is what makes the assertion a positive one: a query
    that read no row at all -- the volume predicate emptied, the root predicate
    inverted -- answers with the empty set, which a test that only denied the
    other root's stub would have accepted.

    Parameters:
        tmp_path: Directory the index file is written into.
    """
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(
            IMAGES.insert(),
            [
                _document(SUCCESS, status='error', status_error=SPICE),
                _document(SECOND_SUCCESS, status='success', status_error=None),
                image_row(
                    root_url=OTHER_ROOT,
                    results_path_stub=SUCCESS,
                    volume=VOLUME,
                    status='success',
                    status_error=None,
                ),
                image_row(
                    root_url=OTHER_ROOT,
                    results_path_stub=SECOND_SUCCESS,
                    volume=VOLUME,
                    status='error',
                    status_error=SPICE,
                ),
            ],
        )
        connection.execute(INGEST_RUNS.insert(), [_completed_run(ROOT), _completed_run(OTHER_ROOT)])
    assert _stubs(url, has_no_offset_error=True).matching_error == frozenset({SECOND_SUCCESS})


@pytest.mark.parametrize(
    'naming_an_error',
    ['has_offset_error', 'has_offset_spice_error', 'has_offset_nonspice_error'],
)
def test_asking_for_a_fatal_error_and_for_none_selects_nothing(
    two_roots: str, naming_an_error: str
) -> None:
    """This layer conjoins the pair rather than rejecting it, as it does the rest.

    ``ResultsFilter`` refuses a contradictory pair before it opens anything,
    but this function is exported and takes the flags as it finds them, so what
    it does with a pair nothing can satisfy is part of its contract: the empty
    selection that describes it, rather than one of the two filters silently
    winning.

    Parameters:
        two_roots: The index to read.
        naming_an_error: The flag paired with ``has_no_offset_error``.
    """
    stubs = _stubs(two_roots, has_no_offset_error=True, **{naming_an_error: True})
    assert SUCCESS in stubs.with_metadata
    assert stubs.matching_error == frozenset()


def test_a_refused_document_matches_the_negative_error_filter_no_more_than_the_rest(
    two_roots: str,
) -> None:
    """Nothing was read from it, so it records neither an error nor the lack of one.

    The refusal arm contributes a literal rather than a predicate, and this is
    what says the literal is false for a filter phrased in the negative too.
    """
    assert REFUSED not in _stubs(two_roots, has_no_offset_error=True).matching_error


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
    assert SUCCESS not in matching


def test_a_refused_document_matches_no_error_filter(two_roots: str) -> None:
    """Nothing was read from it, so it records no status to match."""
    assert REFUSED not in _stubs(two_roots, has_offset_error=True).matching_error


def test_a_complete_pass_leaves_nothing_for_the_caller_to_report(two_roots: str) -> None:
    """A pass that listed the whole root makes absence mean what it says.

    The other root's pass is the newest in the index and missed directories, so
    a count read without naming this root reports that gap against a root that
    has none.
    """
    assert _stubs(two_roots).directories_missed == 0


def test_the_answer_carries_the_time_of_the_pass_that_recorded_it(two_roots: str) -> None:
    """The index detects no change since that moment, so the moment travels with it.

    It is read from this root's newest run: the other root was passed over
    afterwards, and its finish time says nothing about how old this answer is.
    """
    assert _stubs(two_roots).ingested_utc == INGESTED


def _index_with_runs(tmp_path: Path, counts: list[int]) -> str:
    """Build an index whose root was passed over once per count, in that order.

    A second root is passed over after all of them, missing directories and
    finishing at a moment of its own, so that a count or a finish time read from
    the newest run in the index rather than from the newest run over this root
    is that second root's and is visibly wrong.

    Parameters:
        tmp_path: Directory the index file is written into.
        counts: How many directories each completed pass over the root under
            test did not list, oldest pass first.

    Returns:
        The connection URL of the index.
    """
    url = sqlite_url_for(tmp_path / 'missed.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), [_document(SECOND_SUCCESS)])
        connection.execute(
            INGEST_RUNS.insert(),
            [
                _completed_run(ROOT, directories_missed=missed, finished_utc=INGESTED)
                for missed in counts
            ]
            + [_other_roots_newest_run()],
        )
    return url


def test_a_pass_that_missed_a_directory_hands_the_count_back(tmp_path: Path) -> None:
    """Absence under a directory nobody listed is not an answer, and says so.

    The run completed, so nothing else in the index shows the gap; the count on
    the run row is the only place it appears, and the caller that reads absence
    as "this image was never navigated" is the one that has to be told.
    """
    stubs = read_result_stubs(_index_with_runs(tmp_path, [0, 3]), ROOT, [VOLUME])
    assert stubs.directories_missed == 3


def test_the_count_comes_from_the_newest_pass(tmp_path: Path) -> None:
    """A later complete pass answers for the tree an earlier half-read one left.

    The count belongs to a pass and not to a root: the whole root was listed
    this time, so absence under it means what it says again, and reporting the
    older pass's gap would cry wolf on every enumeration until the end of time.
    """
    stubs = read_result_stubs(_index_with_runs(tmp_path, [3, 0]), ROOT, [VOLUME])
    assert stubs.directories_missed == 0


def test_another_roots_gap_is_not_this_roots(tmp_path: Path) -> None:
    """The count is read from the runs over the named root and no others.

    One index serves several roots, and the newest run in it is routinely
    another root's: a count read without the root warns of a gap this root does
    not have, or -- with the two passes the other way round -- reports none when
    this root has one.
    """
    stubs = read_result_stubs(_index_with_runs(tmp_path, [0]), ROOT, [VOLUME])
    assert stubs.directories_missed == 0


def test_another_roots_pass_is_not_when_this_answer_was_recorded(tmp_path: Path) -> None:
    """The finish time comes from the same run row as the count, and by the same rule."""
    stubs = read_result_stubs(_index_with_runs(tmp_path, [0]), ROOT, [VOLUME])
    assert stubs.ingested_utc == INGESTED


def test_a_root_with_no_completed_ingest_is_refused(two_roots: str) -> None:
    """Absence of a row means "never navigated" only under a root that was walked."""
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        read_result_stubs(two_roots, '/data/never-ingested', [VOLUME])
    assert '/data/never-ingested' in str(excinfo.value)


def _without_a_table(url: str, table: str) -> str:
    """Take one table away from an index that is otherwise sound.

    This is the shape of an index whose account was granted the rows it reports
    on and not the bookkeeping beside them, and of one restored from a partial
    dump.  A connection lost between the open and the query fails the same way
    and cannot be provoked as cheaply.

    Parameters:
        url: The index to alter.
        table: Name of the table to drop.

    Returns:
        The same URL, for the caller to read.
    """
    with opened(url) as engine, engine.begin() as connection:
        connection.execute(sqlalchemy.text(f'DROP TABLE {table}'))
    return url


def test_a_failing_run_query_is_reported_rather_than_raised_as_it_came(two_roots: str) -> None:
    """The index opened and then would not answer, which is still a refusal.

    Every consumer of this module catches one type, because the alternative is
    that a program which deliberately never imports the database layer has to
    name its exceptions to report on them.
    """
    with pytest.raises(ValueError, match='could not be read'):
        read_result_stubs(_without_a_table(two_roots, 'ingest_runs'), ROOT, [VOLUME])


def test_a_failing_stub_query_is_reported_the_same_way(two_roots: str) -> None:
    """The refusal covers every query the read issues, not only the first."""
    with pytest.raises(ValueError, match='could not be read'):
        read_result_stubs(_without_a_table(two_roots, 'failed_files'), ROOT, [VOLUME])


def test_a_failure_names_what_the_database_said(two_roots: str) -> None:
    """A refusal that did not name the missing table would leave nothing to act on."""
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(_without_a_table(two_roots, 'failed_files'), ROOT, [VOLUME])
    assert 'failed_files' in str(excinfo.value)


def test_no_database_exception_escapes_the_read(two_roots: str) -> None:
    """The type is the one the caller can name, and not a subclass of the one it cannot."""
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(_without_a_table(two_roots, 'ingest_runs'), ROOT, [VOLUME])
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def _timed_out() -> sqlalchemy.exc.SQLAlchemyError:
    """Return a database failure that carries no driver exception under it.

    Every failure a dropped table provokes is a statement a driver answered, so
    it has an ``orig`` to report.  A pool that ran out of connections, a
    connection used after it closed, and a result read after it was consumed do
    not: they are raised by the database layer itself, before or after any
    driver was asked anything, and they are what a production index under load
    produces.

    Returns:
        The failure, with the sentence it carries as its own.
    """
    return sqlalchemy.exc.TimeoutError(
        'QueuePool limit of size 5 overflow 10 reached, connection timed out'
    )


def test_a_failure_with_no_driver_message_reports_the_sentence_it_has(tmp_path: Path) -> None:
    """The report carries the driver's sentence, and this failure's own is all there is."""
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with (
        pytest.raises(ValueError, match='connection timed out'),
        selection._reporting_a_failed_read(url),
    ):
        raise _timed_out()


def test_a_failure_with_no_driver_message_does_not_report_the_word_none(tmp_path: Path) -> None:
    """``str(None)`` is ``'None'``, which reads as a driver that answered that."""
    url = sqlite_url_for(tmp_path / 'index.sqlite3')
    with (
        pytest.raises(ValueError) as excinfo,
        selection._reporting_a_failed_read(url),
    ):
        raise _timed_out()
    assert 'None' not in str(excinfo.value)


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
                    results_path_stub=SUCCESS,
                    volume=VOLUME,
                )
            ],
        )
        connection.execute(INGEST_RUNS.insert(), [_completed_run(root_url)])
    monkeypatch.chdir(tmp_path)
    stubs = read_result_stubs(url, root.name, [VOLUME])
    assert SUCCESS in stubs.with_metadata


def test_a_trailing_separator_names_the_root_the_ingest_recorded(two_roots: str) -> None:
    """One program writes the trailing slash on the root and another does not."""
    stubs = read_result_stubs(two_roots, f'{ROOT}/', [VOLUME])
    assert SUCCESS in stubs.with_metadata


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


ENUMERATION_MEMBERS = {
    'a document the ingest refused': 'the ingest refused',
    'a status the index falls back for': 'absent, empty, or not a string',
    'a file with no row': 'has no row at all',
    'a document that left the tree': 'the tree no longer holds',
    'a document rewritten in place': 'rewritten in place',
}
"""The answers the index gives differently from the tree, and what names each.

Every place that states the enumeration states each member in its own words,
which is why the member is identified by a phrase it carries rather than by its
position or by the number of members: a list compared by length agrees with a
list that dropped one member and gained another, and agrees with itself after a
member is deleted and an unrelated paragraph is emphasized in its place.

The phrase is chosen to be the one wording every statement of that member shares
and no other member's carries.  Adding a member means adding it here, to the
module docstring, to the navigation guide and to the plan's two lists, in one
commit; rewording one out of any of those lists fails, which is the point.
"""

PLAN = Path(__file__).resolve().parents[3] / 'plans' / 'RESULTS_DB_PLAN.md'
"""The plan, which states the enumeration twice: in Phase 5 and in criterion 1."""

NAVIGATION_GUIDE = (
    Path(__file__).resolve().parents[3] / 'docs' / 'user_guide' / 'user_guide_navigation.rst'
)
"""The guide, which states the enumeration where an operator will meet it.

The member the guide is most easily written without is the one that costs an
operator most: a selection answered from an index is short by every document
the ingest refused, and nothing in the run says so.
"""


def _plan_lines() -> list[str]:
    """Return the plan's lines, skipping the test when the plan is not there.

    The plan is a repository document rather than a packaged one, so a checkout
    always has it and an installed tree never does.

    Returns:
        The lines of the plan file.
    """
    if not PLAN.is_file():
        pytest.skip(f'{PLAN} is not in this tree')
    return PLAN.read_text(encoding='utf-8').splitlines()


def _normalized(text: str) -> str:
    """Return one entry with the markup the four lists spell differently removed.

    Parameters:
        text: The entry as its list writes it.

    Returns:
        The text with emphasis and literal markers dropped, its line breaks
        collapsed to single spaces, and its case folded, so that one phrase
        identifies a member whether the list around it is Python, ``.rst`` or
        Markdown, and whether the phrase is wrapped across two lines.
    """
    stripped = text.replace('*', '').replace('`', '')
    return re.sub(r'\s+', ' ', stripped).strip().casefold()


def _lead_paragraphs(
    lines: Sequence[str], lead: re.Pattern[str], *, opens: str | None = None, closes: str | None
) -> list[str]:
    """Return each paragraph of one region that a member's own lead opens.

    An entry is its lead line and the lines wrapped under it, up to the blank
    line that closes the paragraph: the identity of a member belongs where a
    reader meets it, which is the paragraph that announces it, and not in a
    sub-list or an aside further down.

    Parameters:
        lines: The lines of the file, or of the docstring.
        lead: Pattern matching the opening line of an entry.
        opens: Text identifying the line the region starts after, or None to
            start at the first line.
        closes: Prefix of the line that ends the region, or None to read to the
            end.

    Returns:
        One entry per lead, each as a single line of text.
    """
    inside = opens is None
    collected: list[list[str]] = []
    open_paragraph = False
    for line in lines:
        if not inside:
            inside = opens is not None and opens in line
            continue
        if closes is not None and line.startswith(closes):
            break
        if lead.match(line):
            collected.append([line.strip()])
            open_paragraph = True
        elif not line.strip():
            open_paragraph = False
        elif open_paragraph:
            collected[-1].append(line.strip())
    return [' '.join(parts) for parts in collected]


def _phase_five_members() -> list[str]:
    """Return the enumeration as the Phase 5 entry states it.

    Returns:
        One entry per member.
    """
    return _lead_paragraphs(
        _plan_lines(),
        re.compile(r'^  \d+\. '),
        opens='**What the index answers differently',
        closes='- **The answer says how old',
    )


def _criterion_one_members() -> list[str]:
    """Return the enumeration as acceptance criterion 1 restates it.

    Returns:
        One entry per member.
    """
    return _lead_paragraphs(
        _plan_lines(),
        re.compile(r'^   \d+\. '),
        opens='## 5. Acceptance criteria',
        closes='2. No pipeline',
    )


def _docstring_members() -> list[str]:
    """Return the enumeration as the module docstring states it.

    Returns:
        One entry per member.
    """
    docstring = selection.__doc__ or ''
    return _lead_paragraphs(docstring.splitlines(), re.compile(r'^- \*\*'), closes=None)


def _navigation_guide_members() -> list[str]:
    """Return the enumeration as the navigation guide states it.

    The guide's members are the bold-led paragraphs of its account of
    ``--results-db``, which runs from the sentence introducing that option's
    answers to the end of the selection section.

    Returns:
        One entry per member.
    """
    if not NAVIGATION_GUIDE.is_file():
        pytest.skip(f'{NAVIGATION_GUIDE} is not in this tree')
    return _lead_paragraphs(
        NAVIGATION_GUIDE.read_text(encoding='utf-8').splitlines(),
        re.compile(r'^\*\*'),
        opens='Given ``--results-db``',
        closes='Miscellaneous',
    )


def _members_stated_in(entries: Sequence[str]) -> list[str]:
    """Return the member each entry of one list states, entry by entry.

    Parameters:
        entries: The list's entries.

    Returns:
        One name per member an entry carries the identifying phrase of, so that
        an entry stating none contributes nothing and an entry stating two
        contributes both.
    """
    return [
        name
        for entry in entries
        for name, phrase in ENUMERATION_MEMBERS.items()
        if _normalized(phrase) in _normalized(entry)
    ]


ENUMERATION_LISTS = [
    pytest.param(_docstring_members, id='module-docstring'),
    pytest.param(_navigation_guide_members, id='navigation-guide'),
    pytest.param(_phase_five_members, id='plan-phase-5'),
    pytest.param(_criterion_one_members, id='plan-criterion-1'),
]
"""Every place a member of the enumeration has to be stated, and how to read it.

The guide is one of them because an operator reading it is the person a silently
short selection is served to.  The plan states the enumeration twice, and both
are here, because acceptance criterion 1 restates it as the list rather than as
a pointer at one.
"""


@pytest.mark.parametrize('members', ENUMERATION_LISTS)
def test_every_list_states_each_member_of_the_enumeration_once(
    members: Callable[[], list[str]],
) -> None:
    """Matched member by member, because a length agrees with the wrong list.

    A selection answered from an index differs from the same selection answered
    from the tree, silently and by however many documents the ingest refused, so
    the operator choosing between them is told which members of this list apply
    to their root.  Deleting the paragraph that says so and emphasizing another
    in its place leaves the count where it was; it does not leave this set where
    it was.
    """
    assert sorted(_members_stated_in(members())) == sorted(ENUMERATION_MEMBERS)


@pytest.mark.parametrize('members', ENUMERATION_LISTS)
def test_no_entry_of_a_list_states_anything_outside_the_enumeration(
    members: Callable[[], list[str]],
) -> None:
    """An entry naming no member is a member added to one list and not the rest.

    It is also what a paragraph promoted into the enumeration by emphasis looks
    like: the snapshot, which the docstring states after the list precisely
    because the age of the pass decides nothing about the members, reads as one
    of them the moment its lead is emphasized like theirs.
    """
    assert [entry for entry in members() if not _members_stated_in([entry])] == []
