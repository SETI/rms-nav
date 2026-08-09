"""The selection filters answered from a results index, against the same tree.

Every filter has two implementations behind it, and the point of the index one
is that it answers what the tree one answers.  So the same fixture tree drives
both: it is written to disk, ingested into an index, and every filter is then
asked of the tree and of the index and held to one stated answer.  Stating the
answer rather than only comparing the two matters, because two implementations
that are wrong in the same way agree.

The tree covers what the filters distinguish -- a success, a run that finished
without one, three shapes of fatal error, a document that is not valid JSON, a
document that is valid JSON and not an object, and an image with no result files
at all -- and it is ingested alongside a second root holding a fatal SPICE error
and a summary PNG for every one of those stubs.  A query that filtered on the
stub without its root would answer with that second root's rows, and no
single-root fixture can see it happen.

Three answers the index gives differently from the tree have tests of their own
rather than being left out of the matrix, because each is a property of what the
index records and silently changing it is what the tests are here to catch.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from tests.spindoctor.cli.stats.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
    write_summary_png,
)

from spindoctor.cli.stats.ingest import store
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import (
    _SPICE_STATUS_ERROR,
    RESULTS_FILTER_BATCH_SIZE,
    ResultsFilter,
)
from spindoctor.results_index import SPICE_STATUS_ERROR

VOLUMES = ['COISS_2001', 'COISS_2002']
"""The volumes the enumeration selected."""

SUCCESS_WITH_PNG = 'COISS_2001/data/a/N1000000001_1_CALIB'
SUCCESS_NO_PNG = 'COISS_2001/data/a/N1000000002_1_CALIB'
FAILURE = 'COISS_2001/data/a/N1000000003_1_CALIB'
SPICE_ERROR = 'COISS_2001/data/b/N1000000004_1_CALIB'
NONSPICE_ERROR = 'COISS_2001/data/b/N1000000005_1_CALIB'
ERROR_WITHOUT_STATUS_ERROR = 'COISS_2001/data/b/N1000000006_1_CALIB'
MALFORMED = 'COISS_2001/data/b/N1000000007_1_CALIB'
NOT_AN_OBJECT = 'COISS_2001/data/b/N1000000008_1_CALIB'
NO_RESULT = 'COISS_2001/data/c/N1000000009_1_CALIB'
OTHER_VOLUME = 'COISS_2002/data/a/N1000000010_1_CALIB'

CANDIDATES = (
    SUCCESS_WITH_PNG,
    SUCCESS_NO_PNG,
    FAILURE,
    SPICE_ERROR,
    NONSPICE_ERROR,
    ERROR_WITHOUT_STATUS_ERROR,
    MALFORMED,
    NOT_AN_OBJECT,
    NO_RESULT,
    OTHER_VOLUME,
)
"""The images offered to the filter, in the order an enumeration yields them."""

WITH_A_DOCUMENT = tuple(stub for stub in CANDIDATES if stub != NO_RESULT)
"""Every candidate whose metadata file exists, however well it reads."""

WITH_A_PNG = (SUCCESS_WITH_PNG, FAILURE, NONSPICE_ERROR, MALFORMED, OTHER_VOLUME)
"""Every candidate a summary PNG was written for, in enumeration order.

One of them is a document the ingest refuses, because a PNG is found beside a
file rather than read out of it: the walk finds ``X_summary.png`` whatever
``X_metadata.json`` turned out to contain.
"""

WITHOUT_A_PNG = tuple(stub for stub in CANDIDATES if stub not in WITH_A_PNG)
"""Every candidate no summary PNG was written for."""

FATAL_ERRORS = (SPICE_ERROR, NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR)
"""Every candidate whose document records a fatal error."""


def _logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps the ingest and the filter quiet.

    Returns:
        A logger discarding everything written to it.
    """
    return pdslogger.NullLogger()


def _write_tree(root: Path) -> None:
    """Write the fixture results tree under one root.

    Parameters:
        root: The results root to write into.
    """
    write_metadata(root, SUCCESS_WITH_PNG, metadata_document(image_name='N1000000001_1.IMG'))
    write_summary_png(root, SUCCESS_WITH_PNG)
    write_metadata(root, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    write_metadata(
        root,
        FAILURE,
        metadata_document(image_name='N1000000003_1.IMG', status='failure', offset=None),
    )
    write_summary_png(root, FAILURE)
    write_metadata(
        root,
        SPICE_ERROR,
        metadata_document(
            image_name='N1000000004_1.IMG',
            status='error',
            status_error=SPICE_STATUS_ERROR,
            offset=None,
        ),
    )
    write_metadata(
        root,
        NONSPICE_ERROR,
        metadata_document(
            image_name='N1000000005_1.IMG',
            status='error',
            status_error='unhandled_exception',
            offset=None,
        ),
    )
    write_summary_png(root, NONSPICE_ERROR)
    write_metadata(
        root,
        ERROR_WITHOUT_STATUS_ERROR,
        metadata_document(image_name='N1000000006_1.IMG', status='error', offset=None),
    )
    _write_bytes(root, MALFORMED, b'{"status": "error"')
    # A summary PNG sits beside a document the ingest refuses. This is the
    # ordinary shape of a results root written by an older metadata schema --
    # every image in one has a summary beside a document the ingest will not
    # read -- so both PNG filters have to answer for it as the walk does.
    write_summary_png(root, MALFORMED)
    _write_bytes(root, NOT_AN_OBJECT, b'[1, 2, 3]')
    write_metadata(root, OTHER_VOLUME, metadata_document(image_name='N1000000010_1.IMG'))
    write_summary_png(root, OTHER_VOLUME)


def _write_bytes(root: Path, stub: str, content: bytes) -> None:
    """Write a metadata file that is not a readable navigation document.

    Parameters:
        root: The results root to write into.
        stub: The image's results path stub.
        content: Exactly what the file holds.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_decoy_tree(root: Path) -> None:
    """Write a second root holding an answer-changing row for every stub.

    Every candidate gets a fatal SPICE error and a summary PNG here, and the one
    candidate the tree under test has no result files for gets a document the
    ingest refuses with a summary PNG beside it, so that a refusal read without
    its root changes both the presence answer and the PNG answer.  Any filter
    that read this root's rows for the other root's stubs therefore answers
    differently, which is what makes the composite key testable at all.

    Parameters:
        root: The second results root to write into.
    """
    for stub in CANDIDATES:
        if stub == NO_RESULT:
            _write_bytes(root, stub, b'{"status": "error"')
            write_summary_png(root, stub)
            continue
        write_metadata(
            root,
            stub,
            metadata_document(
                image_name=f'{Path(stub).name}.IMG',
                status='error',
                status_error=SPICE_STATUS_ERROR,
                offset=None,
            ),
        )
        write_summary_png(root, stub)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Write the fixture results tree and return its root.

    Parameters:
        tmp_path: Directory the roots are written under.

    Returns:
        The results root under test.
    """
    root = tmp_path / 'results'
    _write_tree(root)
    return root


@pytest.fixture
def indexed(tree: Path, tmp_path: Path) -> str:
    """Ingest the fixture tree and a second root, and return the index URL.

    Parameters:
        tree: The results root under test.
        tmp_path: Directory the index and the second root live under.

    Returns:
        The connection URL of the index.
    """
    decoy = tmp_path / 'other-results'
    _write_decoy_tree(decoy)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [tree, decoy], logger=_logger())
    return url


def _candidate_files(root: Path) -> list[ImageFile]:
    """Build the images an enumeration would offer the filter.

    Parameters:
        root: The results root, only so the stand-in URLs point somewhere.

    Returns:
        One :class:`ImageFile` per candidate, in enumeration order.
    """
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in CANDIDATES
    ]


def _select(results_filter: ResultsFilter, images: list[ImageFile]) -> list[str]:
    """Run images through the filter exactly as the enumeration does.

    Parameters:
        results_filter: The filter under test.
        images: The candidates, in enumeration order.

    Returns:
        The stubs that passed, in enumeration order.
    """
    kept = [image for image in images if results_filter.passes_presence(image.results_path_stub)]
    selected: list[ImageFile] = []
    for start in range(0, len(kept), RESULTS_FILTER_BATCH_SIZE):
        batch = kept[start : start + RESULTS_FILTER_BATCH_SIZE]
        selected.extend(results_filter.filter_batch(batch))
    return [image.results_path_stub for image in selected]


def _selection(root: Path, flags: dict[str, bool], *, results_db_url: str | None) -> list[str]:
    """Answer one filter combination over the fixture tree.

    Parameters:
        root: The results root under test.
        flags: The selection flags to apply.
        results_db_url: The index to answer from, or None to read the tree.

    Returns:
        The stubs that passed, in enumeration order.
    """
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=_logger(),
        results_db_url=results_db_url,
        **flags,
    )
    return _select(results_filter, _candidate_files(root))


_MATRIX = [
    pytest.param({'has_offset_file': True}, list(WITH_A_DOCUMENT), id='offset-file'),
    pytest.param({'has_no_offset_file': True}, [NO_RESULT], id='no-offset-file'),
    pytest.param({'has_png_file': True}, list(WITH_A_PNG), id='png-file'),
    pytest.param({'has_no_png_file': True}, list(WITHOUT_A_PNG), id='no-png-file'),
    pytest.param({'has_offset_error': True}, list(FATAL_ERRORS), id='offset-error'),
    pytest.param({'has_offset_spice_error': True}, [SPICE_ERROR], id='spice-error'),
    pytest.param(
        {'has_offset_nonspice_error': True},
        [NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR],
        id='nonspice-error',
    ),
    pytest.param(
        {'has_offset_file': True, 'has_no_png_file': True},
        [stub for stub in WITHOUT_A_PNG if stub != NO_RESULT],
        id='offset-file-and-no-png',
    ),
    pytest.param(
        {'has_no_offset_file': True, 'has_png_file': True}, [], id='no-offset-file-and-png'
    ),
    pytest.param(
        {'has_no_offset_file': True, 'has_no_png_file': True},
        [NO_RESULT],
        id='no-offset-file-and-no-png',
    ),
    pytest.param(
        {'has_offset_file': True, 'has_png_file': True}, list(WITH_A_PNG), id='both-files'
    ),
    pytest.param(
        {'has_offset_error': True, 'has_png_file': True}, [NONSPICE_ERROR], id='error-and-png'
    ),
    pytest.param(
        {'has_offset_spice_error': True, 'has_no_png_file': True},
        [SPICE_ERROR],
        id='spice-error-and-no-png',
    ),
    pytest.param(
        {'has_offset_nonspice_error': True, 'has_png_file': True},
        [NONSPICE_ERROR],
        id='nonspice-error-and-png',
    ),
]
"""Every filter flag, alone and paired, with the selection it makes.

The pairings are not decoration: an absence filter alone takes the batched
``exists()`` path and never walks the tree, and the same flag beside a presence
filter is answered from the walked sets instead.  Both of those modes have to
land on the index path's one answer.
"""


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_tree_answers_each_filter(
    tree: Path, flags: dict[str, bool], expected: list[str]
) -> None:
    """What reading the results tree selects, stated rather than compared."""
    assert _selection(tree, flags, results_db_url=None) == expected


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_index_answers_each_filter_the_same_way(
    tree: Path, indexed: str, flags: dict[str, bool], expected: list[str]
) -> None:
    """One query reaches the answer the walk and the per-image reads reach."""
    assert _selection(tree, flags, results_db_url=indexed) == expected


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_index_path_reads_no_file_at_all(
    tree: Path,
    indexed: str,
    flags: dict[str, bool],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every route to the results tree is broken, and the answer is unchanged.

    The saving is the whole point of the index: no walk per volume, no batched
    existence check, no metadata download.  Counting round trips would prove the
    same thing more weakly, since a path taken once still costs a round trip per
    enumeration on a cloud root.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the index path must not read the results tree')

    monkeypatch.setattr(FCPath, 'walk', refuse)
    monkeypatch.setattr(FCPath, 'exists', refuse)
    monkeypatch.setattr(FCPath, 'retrieve', refuse)
    assert _selection(tree, flags, results_db_url=indexed) == expected


_FLAG_CASES = [pytest.param(case.values[0], id=case.id) for case in _MATRIX]
"""The same filter combinations, for the assertions that do not need the answer."""


@pytest.mark.parametrize('flags', _FLAG_CASES)
def test_the_index_path_leaves_nothing_for_the_batch_stage(
    tree: Path, indexed: str, flags: dict[str, bool]
) -> None:
    """The enumeration buffers images only to amortize round trips an index has none of."""
    results_filter = ResultsFilter(
        VOLUMES, str(tree), logger=_logger(), results_db_url=indexed, **flags
    )
    assert results_filter.needs_batch_filtering is False


@pytest.mark.parametrize(
    'flags',
    [
        pytest.param({'has_offset_file': True, 'has_no_offset_file': True}, id='offset-file-pair'),
        pytest.param({'has_png_file': True, 'has_no_png_file': True}, id='png-file-pair'),
        pytest.param(
            {'has_offset_spice_error': True, 'has_offset_nonspice_error': True}, id='error-pair'
        ),
        pytest.param(
            {'has_offset_error': True, 'has_no_offset_file': True}, id='error-and-no-offset-file'
        ),
    ],
)
def test_a_contradictory_pair_is_refused_before_the_index_is_opened(
    tree: Path, tmp_path: Path, flags: dict[str, bool]
) -> None:
    """The flags are validated first, so the refusal is the same with or without one.

    The URL names a database that does not exist, so a constructor that opened
    the index before checking its flags would report that instead.
    """
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(ValueError, match=r'mutually exclusive|contradicts') as excinfo:
        ResultsFilter(VOLUMES, str(tree), logger=_logger(), results_db_url=absent, **flags)
    assert 'not-an-index.sqlite3' not in str(excinfo.value)


def test_a_root_with_no_completed_ingest_is_refused(tree: Path, indexed: str) -> None:
    """Absence of a row is only an answer under a root somebody ingested."""
    other_root = tree.parent / 'never-ingested'
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        ResultsFilter(
            VOLUMES,
            str(other_root),
            logger=_logger(),
            results_db_url=indexed,
            has_offset_file=True,
        )
    assert other_root.as_posix() in str(excinfo.value)


def test_an_index_that_cannot_be_opened_is_not_a_reason_to_read_files(
    tree: Path, tmp_path: Path
) -> None:
    """A misconfigured run fails; it does not become a slow, silently different one."""
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(ValueError, match='sd_stats_ingest') as excinfo:
        ResultsFilter(
            VOLUMES, str(tree), logger=_logger(), results_db_url=absent, has_offset_file=True
        )
    assert 'not-an-index.sqlite3' in str(excinfo.value)


def test_both_paths_match_the_same_spice_error() -> None:
    """The two implementations tell a SPICE failure apart by the same value.

    They hold it separately because the tree path may not import the index
    package, so nothing but this holds the two spellings together.
    """
    assert SPICE_STATUS_ERROR == _SPICE_STATUS_ERROR


def test_a_summary_png_with_no_document_reads_as_present_in_the_tree(tmp_path: Path) -> None:
    """The walk finds the file, whatever else the image does or does not have."""
    root = tmp_path / 'results'
    write_summary_png(root, SUCCESS_WITH_PNG)
    images = [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SUCCESS_WITH_PNG,
        )
    ]
    results_filter = ResultsFilter(VOLUMES, str(root), logger=_logger(), has_png_file=True)
    assert _select(results_filter, images) == [SUCCESS_WITH_PNG]


def test_a_summary_png_with_no_document_reads_as_absent_in_the_index(tmp_path: Path) -> None:
    """The flag lives on the row of the document the PNG was found beside.

    A PNG with no document beside it is recorded nowhere, so the index answers
    that no summary exists for it.  This is one of the three answers the index
    gives differently from the tree, and it is pinned here rather than left to
    be discovered.
    """
    root = tmp_path / 'results'
    write_summary_png(root, SUCCESS_WITH_PNG)
    root.mkdir(parents=True, exist_ok=True)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SUCCESS_WITH_PNG,
        )
    ]
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_png_file=True
    )
    assert _select(results_filter, images) == []


def _error_document_that_is_not_a_navigation_document(root: Path) -> list[ImageFile]:
    """Write a JSON object carrying a fatal error and nothing else, and its image.

    Parameters:
        root: The results root to write into.

    Returns:
        The one candidate image, ready to filter.
    """
    write_metadata(root, SPICE_ERROR, {'status': 'error', 'status_error': SPICE_STATUS_ERROR})
    return [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SPICE_ERROR,
        )
    ]


def test_a_document_that_is_not_a_navigation_document_matches_the_tree_error_filter(
    tmp_path: Path,
) -> None:
    """The tree path reads the two fields out of any JSON object it can parse."""
    root = tmp_path / 'results'
    images = _error_document_that_is_not_a_navigation_document(root)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), has_offset_spice_error=True
    )
    assert _select(results_filter, images) == [SPICE_ERROR]


def test_a_document_that_is_not_a_navigation_document_matches_no_index_error_filter(
    tmp_path: Path,
) -> None:
    """The ingest refused it, so the index holds no status for it to match.

    It still counts as a file that exists, which is what the presence filters
    ask, and that equivalence is what the refusal table is read for.
    """
    root = tmp_path / 'results'
    images = _error_document_that_is_not_a_navigation_document(root)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=_logger(),
        results_db_url=url,
        has_offset_spice_error=True,
    )
    assert _select(results_filter, images) == []


def test_a_summary_png_written_after_a_refusal_is_seen_by_the_next_pass(
    tmp_path: Path,
) -> None:
    """The flag is part of what makes a refused file unchanged, as it is for an image.

    A refused file whose metrics still match is skipped without being read,
    which is what stops a tree of non-navigation documents from being downloaded
    on every run.  A summary PNG written beside it after the refusal was
    recorded changes nothing about the file and everything about the row that
    ought to be stored, so it has to be part of the comparison or the PNG stays
    invisible until the document itself changes.
    """
    root = tmp_path / 'results'
    _write_bytes(root, MALFORMED, b'{"status": "error"')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    write_summary_png(root, MALFORMED)
    ingest_tree(url, [root], logger=_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=MALFORMED,
        )
    ]
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_png_file=True
    )
    assert _select(results_filter, images) == [MALFORMED]


def _status_only_in_the_navigation_result(root: Path) -> list[ImageFile]:
    """Write a document whose outcome is recorded only under ``navigation_result``.

    A document written by an older metadata schema is the plausible shape of
    this: the outcome is there, in the place the rest of the index reads an
    outcome from, and the top-level field the tree path reads is not.

    Parameters:
        root: The results root to write into.

    Returns:
        The one candidate image, ready to filter.
    """
    document = metadata_document(image_name='N1000000004_1.IMG', offset=None)
    del document['status']
    document['navigation_result']['status'] = 'error'
    write_metadata(root, SPICE_ERROR, document)
    return [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SPICE_ERROR,
        )
    ]


def test_a_status_only_in_the_navigation_result_matches_no_tree_error_filter(
    tmp_path: Path,
) -> None:
    """The tree path reads the top-level field and no other."""
    root = tmp_path / 'results'
    images = _status_only_in_the_navigation_result(root)
    results_filter = ResultsFilter(VOLUMES, str(root), logger=_logger(), has_offset_error=True)
    assert _select(results_filter, images) == []


def test_a_status_only_in_the_navigation_result_matches_the_index_error_filter(
    tmp_path: Path,
) -> None:
    """The recorded status falls back to the navigation result, so the index matches.

    This is one of the answers the index gives differently, pinned rather than
    left to be discovered: the column the error filters read is the column the
    whole index reads an outcome from, and it is written from whichever of the
    two places the document put it.
    """
    root = tmp_path / 'results'
    images = _status_only_in_the_navigation_result(root)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_error=True
    )
    assert _select(results_filter, images) == [SPICE_ERROR]


def _one_image_tree(tmp_path: Path) -> tuple[Path, list[ImageFile]]:
    """Write a results root holding one navigated image.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The root, and the one candidate image ready to filter.
    """
    root = tmp_path / 'results'
    write_metadata(root, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    return root, [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SUCCESS_NO_PNG,
        )
    ]


def test_a_file_the_pass_could_not_retrieve_reads_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is recorded for it, deliberately, so the next pass tries it again.

    A recorded refusal would be skipped for as long as the file did not change,
    and a download that failed once says nothing that will still be true then.
    The cost is that the file reads as absent until a pass reads it, which is one
    of the answers the index gives differently.
    """
    root, images = _one_image_tree(tmp_path)

    def refuse(self: FCPath, *args: Any, **kwargs: Any) -> list[Exception]:
        return [OSError('the backend did not answer') for _ in args[0]]

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(FCPath, 'retrieve', refuse)
    ingest_tree(url, [root], logger=_logger())
    monkeypatch.undo()
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
    )
    assert _select(results_filter, images) == []


def test_a_document_the_database_would_not_store_reads_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row the database refuses is counted and recorded nowhere, on the same grounds.

    The document read exactly as the schema says, so nothing about it says the
    next pass will not store it, and a recorded refusal would stop the next pass
    from trying.  It therefore reads as absent, exactly as a file nobody could
    retrieve does.
    """
    root, images = _one_image_tree(tmp_path)

    def refuse(connection: Any, rows: Any) -> None:
        raise sqlalchemy.exc.IntegrityError('INSERT', {}, Exception('refused'))

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(store, '_write_image', refuse)
    ingest_tree(url, [root], logger=_logger())
    monkeypatch.undo()
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
    )
    assert _select(results_filter, images) == []


def test_importing_the_dataset_package_does_not_import_sqlalchemy() -> None:
    """The navigation critical path stays free of the database layer.

    Every navigation run imports :mod:`spindoctor.dataset`, and most of them
    name no index at all.  The index-backed filter is therefore imported inside
    the branch that has a URL, and this is what says so.

    It runs in a subprocess because the assertion is about a fresh interpreter:
    anything else in the test session has already imported SQLAlchemy, and the
    same check inside this process would pass no matter what the package does.
    """
    probe = (
        'import json, sys\n'
        'import spindoctor.dataset\n'
        'print(json.dumps(sorted(name for name in sys.modules '
        'if name.split(".")[0] == "sqlalchemy")))\n'
    )
    completed = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []
