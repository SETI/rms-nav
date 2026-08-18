"""Tests for the one query that answers an enumeration's selection filters.

The rows are written directly rather than ingested, because what is under test
is what the query reads and not what a walk records: a refusal with no image row
beside it, an image whose ``status_error`` is NULL, a stub with no subtree, and a
second root holding a row for every stub the first one holds.

That second root is why every test here builds two.  The index is keyed by root
and stub together, and a query that filtered on the stub alone would answer with
the other root's rows while every single-root assertion stayed green.  The other
root is therefore stocked so that every filter's answer changes if one of its
rows leaks: it holds a fatal SPICE error for each stub, and a document for each
of the two stubs the first root deliberately has none of -- one under the
selected subtree and one under the subtree the enumeration passes over.

The run rows are stocked the same way, because the run table is keyed by root as
well and is read by a query of its own.  The other root is always passed over
last, so its run is the newest in the whole index, and its pass always missed
directories and always finished at a different moment: a count or a finish time
read from the newest run rather than from this root's newest run is therefore
the other root's, and says so.
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

SUBTREE = 'COISS_2001'
"""The selected subtree."""

OTHER_SUBTREE = 'COISS_2002'
"""A subtree of the same root that the enumeration did not select."""

SUCCESS = f'{SUBTREE}/data/a/N1000000001_1_CALIB'
SECOND_SUCCESS = f'{SUBTREE}/data/a/N1000000002_1_CALIB'
FAILURE = f'{SUBTREE}/data/a/N1000000003_1_CALIB'
SPICE_ERROR = f'{SUBTREE}/data/b/N1000000004_1_CALIB'
NONSPICE_ERROR = f'{SUBTREE}/data/b/N1000000005_1_CALIB'
ERROR_WITHOUT_STATUS_ERROR = f'{SUBTREE}/data/b/N1000000006_1_CALIB'
REFUSED = f'{SUBTREE}/data/b/N1000000007_1_CALIB'
REFUSED_OTHER_SUBTREE = f'{OTHER_SUBTREE}/data/b/N1000000008_1_CALIB'
UNSELECTED_SUBTREE_IMAGE = f'{OTHER_SUBTREE}/data/a/N1000000009_1_CALIB'
NO_SUBTREE = 'scene_0001'
ONLY_IN_THE_OTHER_ROOT = f'{SUBTREE}/data/c/N1000000010_1_CALIB'
ONLY_REFUSED_IN_THE_OTHER_ROOT = f'{SUBTREE}/data/c/N1000000011_1_CALIB'
ONLY_IN_THE_OTHER_ROOTS_OTHER_SUBTREE = f'{OTHER_SUBTREE}/data/c/N1000000012_1_CALIB'
"""A stub only the other root holds, under the subtree the enumeration did not select.

It is what makes the answer to ``select the other subtree`` a root-aware one:
the two roots agree about every other stub of that subtree, so a query that
dropped its root predicate would answer identically without it.
"""

EVERY_STUB = (
    SUCCESS,
    SECOND_SUCCESS,
    FAILURE,
    SPICE_ERROR,
    NONSPICE_ERROR,
    ERROR_WITHOUT_STATUS_ERROR,
    REFUSED,
    REFUSED_OTHER_SUBTREE,
    UNSELECTED_SUBTREE_IMAGE,
    NO_SUBTREE,
    ONLY_IN_THE_OTHER_ROOT,
    ONLY_IN_THE_OTHER_ROOTS_OTHER_SUBTREE,
)
"""Every stub either root holds, so the other root can be stocked with all of them."""

SPICE = 'missing_spice_data'
"""The ``status_error`` value the SPICE filters tell apart."""

INGESTED = '2026-02-03T04:05:06+00:00'
"""When the newest pass over the root under test finished."""

EARLIER_INGESTED = '2026-01-02T03:04:05+00:00'
"""When an earlier pass over that same root finished."""

OTHER_ROOT_INGESTED = '2026-03-04T05:06:07+00:00'
"""When the newest pass over the other root finished, which is not when this one did.

That pass is the newest in the index, so a finish time read from the newest run
of the table rather than from the newest run of the root under test reports this
moment instead of that root's own.
"""


def _subtree_of(stub: str) -> str | None:
    """Return the subtree column value a stub gets, as the ingest derives it.

    Parameters:
        stub: The results path stub.

    Returns:
        Its first path segment, or None when it carries no separator.
    """
    subtree, separator, _rest = stub.partition('/')
    return subtree if separator else None


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
        subtree=_subtree_of(stub),
        **{'status': 'success', 'status_error': None, **columns},
    )


def _completed_run(root_url: str, *, finished_utc: str | None = None) -> dict[str, Any]:
    """Return an ``ingest_runs`` row saying a pass over one root completed.

    Parameters:
        root_url: The root the run covered.
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
        'schema_version': SCHEMA_VERSION,
    }


def _other_roots_newest_run() -> dict[str, Any]:
    """Return the run row that makes a root-blind run query answer wrongly.

    It is inserted last, so it is the newest run in the index, and it records a
    finish time the root under test never records.

    Returns:
        A mapping ready to insert.
    """
    return _completed_run(OTHER_ROOT, finished_utc=OTHER_ROOT_INGESTED)


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
        _document(UNSELECTED_SUBTREE_IMAGE),
        _document(NO_SUBTREE),
    ]
    refusals = [
        {
            'root_url': root_url,
            'results_path_stub': stub,
            'reason': 'not a current-schema navigation document',
            'subtree': _subtree_of(stub),
            'mtime_ns': 1,
            'size_bytes': 2,
        }
        for root_url, stub in (
            (ROOT, REFUSED),
            (ROOT, REFUSED_OTHER_SUBTREE),
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
            subtree=_subtree_of(stub),
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
    """Read the root under test, selecting its one selected subtree.

    Parameters:
        url: The index to read.
        filters: Error filters to apply.

    Returns:
        What the query answered.
    """
    return read_result_stubs(url, ROOT, [SUBTREE], **filters)


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


def test_the_query_does_not_fetch_an_unselected_subtree(two_roots: str) -> None:
    """The subtree restriction is in the query and not only in what is kept from it.

    The saving is that an enumeration of one subtree does not pay to read the
    rows of every other subtree of the root, and a restriction applied only to
    the rows already fetched answers identically while paying for all of them.
    """
    query = _stub_query(ROOT, [SUBTREE], sqlalchemy.false())
    with opened(two_roots) as engine, engine.connect() as connection:
        fetched = {str(row[0]) for row in connection.execute(query)}
    assert UNSELECTED_SUBTREE_IMAGE not in fetched


def test_an_image_of_an_unselected_subtree_is_not_read(two_roots: str) -> None:
    """The tree walk lists the selected subtrees' directories and no others."""
    assert UNSELECTED_SUBTREE_IMAGE not in _stubs(two_roots).with_metadata


def test_a_refusal_of_an_unselected_subtree_is_not_read(two_roots: str) -> None:
    """A refused file is under a subtree like any other, and is restricted to it."""
    assert REFUSED_OTHER_SUBTREE not in _stubs(two_roots).with_metadata


def test_the_query_does_not_fetch_an_unselected_subtrees_refusal(two_roots: str) -> None:
    """The restriction is in the query for the refusals as much as for the images.

    A root whose non-navigation files outnumber its results is exactly the tree
    the refusal table was made for, and a single-subtree enumeration that fetched
    every refusal in the root would pay for all of them on every run.
    """
    query = _stub_query(ROOT, [SUBTREE], sqlalchemy.false())
    with opened(two_roots) as engine, engine.connect() as connection:
        fetched = {str(row[0]) for row in connection.execute(query)}
    assert REFUSED_OTHER_SUBTREE not in fetched


def test_a_stub_with_no_subtree_is_not_read(two_roots: str) -> None:
    """A scene name with no subtree above it sits outside every walked directory."""
    assert NO_SUBTREE not in _stubs(two_roots).with_metadata


def test_selecting_a_subtree_reads_that_subtree_of_this_root(two_roots: str) -> None:
    """The unselected subtree's files are readable when it is the one selected.

    Stated as the whole set, and the other root holds a file of that subtree
    this one does not, so the subtree predicate is pinned together with the root
    predicate: a query restricted by subtree alone answers with the other root's
    file as well.
    """
    stubs = read_result_stubs(two_roots, ROOT, [OTHER_SUBTREE])
    assert stubs.with_metadata == frozenset({UNSELECTED_SUBTREE_IMAGE, REFUSED_OTHER_SUBTREE})


def test_no_subtree_selected_reads_nothing(two_roots: str) -> None:
    """An enumeration that selected no subtree walks no directory."""
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
    that read no row at all -- the subtree predicate emptied, the root predicate
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
                    subtree=SUBTREE,
                    status='success',
                    status_error=None,
                ),
                image_row(
                    root_url=OTHER_ROOT,
                    results_path_stub=SECOND_SUCCESS,
                    subtree=SUBTREE,
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


def test_the_answer_carries_the_time_of_the_pass_that_recorded_it(two_roots: str) -> None:
    """The index detects no change since that moment, so the moment travels with it.

    It is read from this root's newest run: the other root was passed over
    afterwards, and its finish time says nothing about how old this answer is.
    """
    assert _stubs(two_roots).ingested_utc == INGESTED


def _index_with_runs(tmp_path: Path, finish_times: list[str]) -> str:
    """Build an index whose root was passed over once per time, in that order.

    A second root is passed over after all of them, finishing at a moment of its
    own, so that a finish time read from the newest run in the index rather than
    from the newest run over this root is that second root's and is visibly
    wrong.

    Parameters:
        tmp_path: Directory the index file is written into.
        finish_times: When each completed pass over the root under test
            finished, oldest pass first.

    Returns:
        The connection URL of the index.
    """
    url = sqlite_url_for(tmp_path / 'passes.sqlite3')
    with opened(url, create=True) as engine, engine.begin() as connection:
        connection.execute(IMAGES.insert(), [_document(SECOND_SUCCESS)])
        connection.execute(
            INGEST_RUNS.insert(),
            [_completed_run(ROOT, finished_utc=finished) for finished in finish_times]
            + [_other_roots_newest_run()],
        )
    return url


def test_the_time_comes_from_the_newest_pass_over_the_root(tmp_path: Path) -> None:
    """A root passed over twice is as old as the second pass, not the first.

    The age belongs to a pass and not to a root, and an answer dated by the
    oldest pass over the root reads as stale on every enumeration until the end
    of time.
    """
    url = _index_with_runs(tmp_path, [EARLIER_INGESTED, INGESTED])
    stubs = read_result_stubs(url, ROOT, [SUBTREE])
    assert stubs.ingested_utc == INGESTED


def test_another_roots_pass_is_not_when_this_answer_was_recorded(tmp_path: Path) -> None:
    """The finish time is read from the runs over the named root and no others.

    One index serves several roots, and the newest run in it is routinely
    another root's: a time read without the root dates this answer by whenever
    somebody last passed over an unrelated tree.
    """
    stubs = read_result_stubs(_index_with_runs(tmp_path, [INGESTED]), ROOT, [SUBTREE])
    assert stubs.ingested_utc == INGESTED


def test_a_root_with_no_completed_ingest_is_refused(two_roots: str) -> None:
    """Absence of a row means "never navigated" only under a root that was walked."""
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        read_result_stubs(two_roots, '/data/never-ingested', [SUBTREE])
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
        read_result_stubs(_without_a_table(two_roots, 'ingest_runs'), ROOT, [SUBTREE])


def test_a_failing_stub_query_is_reported_the_same_way(two_roots: str) -> None:
    """The refusal covers every query the read issues, not only the first."""
    with pytest.raises(ValueError, match='could not be read'):
        read_result_stubs(_without_a_table(two_roots, 'failed_files'), ROOT, [SUBTREE])


def test_a_failure_names_what_the_database_said(two_roots: str) -> None:
    """A refusal that did not name the missing table would leave nothing to act on."""
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(_without_a_table(two_roots, 'failed_files'), ROOT, [SUBTREE])
    assert 'failed_files' in str(excinfo.value)


def test_no_database_exception_escapes_the_read(two_roots: str) -> None:
    """The type is the one the caller can name, and not a subclass of the one it cannot."""
    with pytest.raises(ValueError) as excinfo:
        read_result_stubs(_without_a_table(two_roots, 'ingest_runs'), ROOT, [SUBTREE])
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_an_index_that_cannot_be_opened_is_an_error(tmp_path: Path) -> None:
    """A resolved URL that will not open fails the run rather than reading files."""
    url = sqlite_url_for(tmp_path / 'absent.sqlite3')
    with pytest.raises(ValueError, match='sd_stats_ingest') as excinfo:
        read_result_stubs(url, ROOT, [SUBTREE])
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
                    subtree=SUBTREE,
                )
            ],
        )
        connection.execute(INGEST_RUNS.insert(), [_completed_run(root_url)])
    monkeypatch.chdir(tmp_path)
    stubs = read_result_stubs(url, root.name, [SUBTREE])
    assert SUCCESS in stubs.with_metadata


def test_a_trailing_separator_names_the_root_the_ingest_recorded(two_roots: str) -> None:
    """One program writes the trailing slash on the root and another does not."""
    stubs = read_result_stubs(two_roots, f'{ROOT}/', [SUBTREE])
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
