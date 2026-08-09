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

Every answer the index gives differently from the tree has a test of its own
rather than being left out of the matrix, because each is a property of what the
index records and silently changing it is what the tests are here to catch.  The
list they cover is the one :mod:`spindoctor.results_index.selection` enumerates,
and a member added there is added here.
"""

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
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
    SelectionError,
)
from spindoctor.results_index import (
    INGEST_RUNS,
    SPICE_STATUS_ERROR,
    normalize_root_url,
    open_index,
    selection,
)
from spindoctor.results_index.selection import ResultStubs

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
    that no summary exists for it.  This is one of the answers the index gives
    differently from the tree, and it is pinned here rather than left to be
    discovered.
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


OTHER_ROOT_MISSED = 5
"""How many directories the second root's pass did not list.

The second root is ingested last, so its run is the newest in the index: a count
read from the newest run of the table rather than from the newest run over the
root being enumerated is this one, and reports a gap the enumerated root does
not have -- or, with the two the other way round, reports none where it does.
"""


def _stamp_run(url: str, root: Path, **values: Any) -> None:
    """Record something about the pass over one root, and about no other root.

    A run row is one root's, exactly as an image row is, so an update without a
    root names every pass in the index and makes the two roots indistinguishable
    in the column it writes.

    Parameters:
        url: The index to write into.
        root: The results root whose newest pass is being described.
        values: Column values to record on it.
    """
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.update()
                .where(INGEST_RUNS.c.root_url == normalize_root_url(root))
                .values(**values)
            )
    finally:
        engine.dispose()


def _index_of_two_roots(tmp_path: Path, root: Path, *, missed: int) -> str:
    """Ingest a tree, and a second tree whose pass missed directories after it.

    The count is written rather than provoked, so that the test is about what a
    consumer does with a count and not about what makes a walk record one.  The
    second root carries a count of its own for the same reason the fixture tree
    is ingested beside a decoy: a query answering from the wrong root's run row
    passes every single-root assertion.

    Parameters:
        tmp_path: Directory the index file is written into.
        root: The results root to ingest and describe.
        missed: How many directories the pass over that root is recorded as
            having missed.

    Returns:
        The connection URL of the index.
    """
    decoy = tmp_path / 'other-results'
    write_metadata(decoy, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root, decoy], logger=_logger())
    _stamp_run(url, root, directories_missed=missed)
    _stamp_run(url, decoy, directories_missed=OTHER_ROOT_MISSED)
    return url


def _reporting_logger() -> pdslogger.PdsLogger:
    """Return a logger whose output a test reads back.

    Returns:
        A logger of its own, so raising its level cannot affect another test.
    """
    return pdslogger.PdsLogger(f'results_filter_test_{uuid.uuid4().hex}')


def test_an_ingest_that_missed_a_directory_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Absence under a directory nobody listed is not an answer, and says so.

    The run completed, so nothing else in the index shows the gap; a run that
    missed a directory otherwise makes an absence filter re-navigate every image
    under it without a word.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=2)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert 'did not list 2 directories' in capsys.readouterr().out


def test_the_report_of_a_gap_says_that_nothing_was_removed_either(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass that missed a directory removes no row anywhere under the root.

    That is the half of the cost an operator can act on: a document deleted
    since the pass before keeps its row for as long as the directory stays
    unlistable, so ``--has-offset-file`` hands on an image whose document is
    gone, and this is the only place it is said.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=2)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert 'no row was removed anywhere under the root' in capsys.readouterr().out


def test_a_complete_ingest_is_reported_as_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass that listed the whole root leaves absence meaning what it says.

    The other root's pass is the newest in the index and missed directories, so
    a count read without naming this root warns about a root that has no gap.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert 'did not list' not in capsys.readouterr().out


def test_the_report_says_how_old_the_index_answer_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The index detects no change since its pass, so its age is what says it is usable.

    An exported URL makes a snapshot answer a resume idiom on every machine that
    exports it, and how long ago that snapshot was taken is the fact that
    decides whether this run is affected by it.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    _stamp_run(url, root, finished_utc=(datetime.now(UTC) - timedelta(days=2)).isoformat())
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert '2 days ago' in capsys.readouterr().out


def test_the_report_names_the_moment_as_well_as_the_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The interval is what a reader compares against; the stamp names the pass to re-run."""
    root, _images = _one_image_tree(tmp_path)
    stamp = '2026-02-03T04:05:06+00:00'
    url = _index_of_two_roots(tmp_path, root, missed=0)
    _stamp_run(url, root, finished_utc=stamp)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert stamp in capsys.readouterr().out


def test_the_age_is_that_of_this_roots_pass_and_not_another(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second root was passed over afterwards, and says nothing about this answer."""
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    _stamp_run(url, root, finished_utc='2026-02-03T04:05:06+00:00')
    _stamp_run(url, tmp_path / 'other-results', finished_utc='2026-03-04T05:06:07+00:00')
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert '2026-03-04T05:06:07+00:00' not in capsys.readouterr().out


def test_a_finish_time_that_will_not_parse_is_reported_as_it_stands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reader can act on a value the index really holds and not on a fiction.

    Nothing this pipeline writes puts an unreadable stamp in that column, and an
    index restored from somewhere else is exactly where one would come from.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    _stamp_run(url, root, finished_utc='whenever it was')
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert 'ingested whenever it was' in capsys.readouterr().out


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


UNLISTABLE = 'COISS_2001/data/c'
"""A directory under the root that one pass cannot list."""


def _refusing_to_list(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    """Make one directory refuse to be listed, as a permission or a share can.

    Provoked rather than written into the run row, because what is under test is
    what the prune does when the walk comes back incomplete, and that is decided
    by the walk and not by the count it records.

    Parameters:
        monkeypatch: Fixture the refusal is installed through.
        directory: The directory that will refuse.
    """
    listing = FCPath.iterdir_metadata
    refused = directory.as_posix()

    def refuse(self: FCPath) -> Any:
        if self.as_posix() == refused:
            raise OSError('this directory may not be listed')
        return listing(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refuse)


def _tree_of_two_documents(tmp_path: Path) -> tuple[Path, list[ImageFile]]:
    """Write a root holding two documents in two directories, and an empty third.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The root, and the two candidate images in enumeration order.
    """
    root = tmp_path / 'results'
    write_metadata(root, SUCCESS_NO_PNG, metadata_document(image_name='N1000000002_1.IMG'))
    write_metadata(root, SPICE_ERROR, metadata_document(image_name='N1000000004_1.IMG'))
    (root / UNLISTABLE).mkdir(parents=True, exist_ok=True)
    return root, [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in (SUCCESS_NO_PNG, SPICE_ERROR)
    ]


def _index_after_a_document_left_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, listing_the_whole_root: bool
) -> tuple[Path, list[ImageFile], str]:
    """Ingest a root, delete one of its documents, and ingest it again.

    The second pass either lists the whole root or finds one directory it cannot
    list.  Both passes complete and stamp a finish time, so a consumer accepts
    the root either way; what differs is whether the pass had the evidence to
    remove the row of the document that has gone.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        monkeypatch: Fixture the unlistable directory is installed through.
        listing_the_whole_root: Whether the second pass lists every directory.

    Returns:
        The root, the two candidate images, and the connection URL of the index.
    """
    root, images = _tree_of_two_documents(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    (root / f'{SPICE_ERROR}_metadata.json').unlink()
    if not listing_the_whole_root:
        _refusing_to_list(monkeypatch, root / UNLISTABLE)
    ingest_tree(url, [root], logger=_logger())
    monkeypatch.undo()
    return root, images, url


def test_a_document_that_left_the_tree_reads_as_absent_in_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk finds what is there now, which is the answer the index is held to."""
    root, images, _url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=True
    )
    results_filter = ResultsFilter(VOLUMES, str(root), logger=_logger(), has_offset_file=True)
    assert _select(results_filter, images) == [SUCCESS_NO_PNG]


def test_a_document_that_left_the_tree_is_pruned_by_a_pass_that_listed_it_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete pass has the evidence to remove the row, and removes it.

    This is what makes the divergence below a consequence of the incomplete
    listing rather than a property of an index: presence means what absence
    means again as soon as one pass lists the whole root.
    """
    root, images, url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=True
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
    )
    assert _select(results_filter, images) == [SUCCESS_NO_PNG]


def test_a_document_that_left_the_tree_survives_a_pass_that_missed_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unlistable directory holds every stale row of the root, not only its own.

    A pass that did not list the whole root has no evidence about the stubs it
    did not see, so it removes none of them, and the row of a document deleted
    from a directory it did list survives with them.  The index then hands a
    presence filter an image whose document is not there, for as long as that
    one directory stays unlistable.
    """
    root, images, url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=False
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
    )
    assert _select(results_filter, images) == [SUCCESS_NO_PNG, SPICE_ERROR]


def test_the_tree_offers_a_document_that_left_it_to_the_absence_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been written for that image now, so the resume idiom picks it up."""
    root, images, _url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=False
    )
    results_filter = ResultsFilter(VOLUMES, str(root), logger=_logger(), has_no_offset_file=True)
    assert _select(results_filter, images) == [SPICE_ERROR]


def test_the_absence_filter_skips_a_document_the_tree_no_longer_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction of the same stale row, and the costlier one.

    ``--has-no-offset-file`` is the resume idiom, so an image whose document was
    deleted is one the run silently declines to navigate again.
    """
    root, images, url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=False
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_no_offset_file=True
    )
    assert _select(results_filter, images) == []


OTHER_STATUS_ERROR = 'some_other_problem'
"""A fatal error of the same text length as the SPICE one.

A document rewritten from one to the other is a different document of exactly
the same size, which is half of what makes the pass skip it.
"""


def _fatal_error_document(status_error: str) -> dict[str, Any]:
    """Build the fatal-error document the rewritten stub carries.

    Parameters:
        status_error: The fatal error the document records.

    Returns:
        The document.
    """
    return metadata_document(
        image_name='N1000000004_1.IMG', status='error', status_error=status_error, offset=None
    )


def _index_after_a_document_was_rewritten_in_place(
    tmp_path: Path, *, keeping_its_size: bool = True
) -> tuple[Path, list[ImageFile], str]:
    """Ingest a root, rewrite one document in place, and ingest it again.

    The rewrite keeps the document's modification time, and by default its
    length: a tree restored with ``cp -p`` or ``rsync --times``, a document
    patched in place and stamped back, or a backend whose listing reports the
    same time for two writes.  Those are the two metrics the pass compares, so
    the second pass has nothing to tell it that the file it already read is not
    the file that is there now.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        keeping_its_size: Whether the rewritten document is the same length as
            the one it replaces.  False makes the rewrite visible again, which
            is what says the divergence is about the metrics and not about
            rewriting.

    Returns:
        The root, the one candidate image, and the connection URL of the index.
    """
    root = tmp_path / 'results'
    document = write_metadata(root, SPICE_ERROR, _fatal_error_document(SPICE_STATUS_ERROR))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    before = document.stat()
    replacement = OTHER_STATUS_ERROR if keeping_its_size else f'{OTHER_STATUS_ERROR}_and_then_some'
    write_metadata(root, SPICE_ERROR, _fatal_error_document(replacement))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (document.stat().st_size == before.st_size) is keeping_its_size
    ingest_tree(url, [root], logger=_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / f'{SPICE_ERROR}.IMG'),
            label_file_url=FCPath(root / f'{SPICE_ERROR}.LBL'),
            results_path_stub=SPICE_ERROR,
        )
    ]
    return root, images, url


def test_a_document_rewritten_in_place_is_not_read_again(tmp_path: Path) -> None:
    """The mechanism, stated on its own: the pass skips a file it has already read.

    Nothing about a listing distinguishes a document from another of the same
    length written at the same recorded time, and reading it to find out is the
    download the skip exists to avoid.
    """
    root = tmp_path / 'results'
    document = write_metadata(root, SPICE_ERROR, _fatal_error_document(SPICE_STATUS_ERROR))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    before = document.stat()
    write_metadata(root, SPICE_ERROR, _fatal_error_document(OTHER_STATUS_ERROR))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert ingest_tree(url, [root], logger=_logger()).files_skipped == 1


def test_a_document_rewritten_in_place_reads_as_it_is_now_in_the_tree(tmp_path: Path) -> None:
    """The walk opens the file, so it answers from the document that is there."""
    root, images, _url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), has_offset_spice_error=True
    )
    assert _select(results_filter, images) == []


def test_a_document_rewritten_in_place_reads_as_its_previous_self_in_the_index(
    tmp_path: Path,
) -> None:
    """The row still records what the document said when it was last read.

    No number of completed passes corrects this one, which is what makes it a
    member of the enumeration rather than the snapshot's age: a pass that
    finished a second ago answers from the document before the rewrite.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert _select(results_filter, images) == [SPICE_ERROR]


def test_a_rewrite_that_changes_the_length_is_read_again(tmp_path: Path) -> None:
    """The divergence is the equal metrics, not the rewrite.

    A document rewritten to a different length is one the pass has evidence
    about, and the row it leaves is what the tree would answer.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(
        tmp_path, keeping_its_size=False
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert _select(results_filter, images) == []


def test_a_forced_pass_corrects_a_document_rewritten_in_place(tmp_path: Path) -> None:
    """Reading every document regardless is what an operator has to reach for.

    It is the remedy because the alternative is reading the file to find out
    whether it needs reading, which is the cost the skip exists to avoid.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    ingest_tree(url, [root], logger=_logger(), force=True)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert _select(results_filter, images) == []


def _index_without_a_table(tmp_path: Path, root: Path, table: str) -> str:
    """Ingest a tree into an index and then take one of its tables away.

    This is the shape of an index whose account was granted the rows it reports
    on and not the bookkeeping beside them, and of one restored from a partial
    dump.  A connection lost between the open and the query fails the same way
    and cannot be provoked as cheaply.

    Parameters:
        tmp_path: Directory the index file is written into.
        root: The results root to ingest.
        table: Name of the table to drop.

    Returns:
        The connection URL of the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=_logger())
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(f'DROP TABLE {table}'))
    finally:
        engine.dispose()
    return url


def test_an_index_that_will_not_answer_refuses_the_selection(tmp_path: Path) -> None:
    """An index that opened and then failed is a misconfigured run like any other.

    The type is the one a program reporting the message catches, so a database
    failure reaches an operator as the sentence that says what to change rather
    than as a traceback out of an enumeration.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
        )


def test_an_index_that_will_not_answer_raises_no_database_exception(tmp_path: Path) -> None:
    """This module never imports the database layer, so it may not raise its types."""
    root, _images = _one_image_tree(tmp_path)
    url = _index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=_logger(), results_db_url=url, has_offset_file=True
        )
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_an_index_that_cannot_be_opened_refuses_the_selection(tmp_path: Path) -> None:
    """The three refusals are one type, so a program catches one and reports all three."""
    root, _images = _one_image_tree(tmp_path)
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(SelectionError, match='sd_stats_ingest'):
        ResultsFilter(
            VOLUMES, str(root), logger=_logger(), results_db_url=absent, has_offset_file=True
        )


def test_a_root_the_index_does_not_cover_refuses_the_selection(tree: Path, indexed: str) -> None:
    """Absence under a root nobody ingested is the third of the three."""
    with pytest.raises(SelectionError, match='no completed ingest'):
        ResultsFilter(
            VOLUMES,
            str(tree.parent / 'never-ingested'),
            logger=_logger(),
            results_db_url=indexed,
            has_offset_file=True,
        )


def test_a_contradictory_pair_refuses_the_selection(tree: Path) -> None:
    """The flags are the fourth, and the one that needs no index at all."""
    with pytest.raises(SelectionError, match='mutually exclusive'):
        ResultsFilter(
            VOLUMES,
            str(tree),
            logger=_logger(),
            has_offset_file=True,
            has_no_offset_file=True,
        )


def test_the_volumes_are_fixed_at_the_boundary_for_the_index(tree: Path, indexed: str) -> None:
    """A caller is free to hand over an iterator, which one read would empty.

    Which path reads the volumes depends on the flags and on whether a URL was
    given, so the sequence is fixed once at the constructor rather than left to
    whichever path happens to consume it.
    """
    results_filter = ResultsFilter(
        VOLUMES, str(tree), logger=_logger(), results_db_url=indexed, has_offset_file=True
    )
    from_iterator = ResultsFilter(
        iter(VOLUMES), str(tree), logger=_logger(), results_db_url=indexed, has_offset_file=True
    )
    images = _candidate_files(tree)
    assert _select(from_iterator, images) == _select(results_filter, images)


def test_the_volumes_are_fixed_at_the_boundary_for_the_tree(tree: Path) -> None:
    """The walked path is handed the same fixed sequence, for the same reason."""
    results_filter = ResultsFilter(VOLUMES, str(tree), logger=_logger(), has_offset_file=True)
    from_iterator = ResultsFilter(iter(VOLUMES), str(tree), logger=_logger(), has_offset_file=True)
    images = _candidate_files(tree)
    assert _select(from_iterator, images) == _select(results_filter, images)


def _reads_recorded_by(reads: list[list[str]]) -> Any:
    """Return a stand-in read that reads its volumes twice and records both.

    Reading twice is the contract under test.  ``volumes`` arrives here as
    whatever the constructor passed on, and an iterator passed through would be
    empty the second time -- which is exactly what the boundary exists to stop,
    and what an end-to-end comparison of a list against an iterator cannot see,
    since a single read serves both correctly.

    Parameters:
        reads: List each read appends its result to.

    Returns:
        A callable with the signature of
        :func:`~spindoctor.results_index.selection.read_result_stubs`.
    """

    def recording(
        url: str, nav_results_root: Any, volumes: Iterable[str], **flags: bool
    ) -> ResultStubs:
        reads.append(list(volumes))
        reads.append(list(volumes))
        return ResultStubs(
            with_metadata=frozenset(), with_summary_png=frozenset(), matching_error=frozenset()
        )

    return recording


def test_the_index_path_is_handed_volumes_it_can_be_read_twice_from(
    tree: Path, indexed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary hands on a sequence, not whatever iterable it was given."""
    reads: list[list[str]] = []
    monkeypatch.setattr(selection, 'read_result_stubs', _reads_recorded_by(reads))
    ResultsFilter(
        iter(VOLUMES), str(tree), logger=_logger(), results_db_url=indexed, has_offset_file=True
    )
    assert reads[1] == VOLUMES


def test_the_tree_path_is_handed_volumes_it_can_be_read_twice_from(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walked path is handed the same fixed sequence, for the same reason."""
    reads: list[list[str]] = []

    def recording(self: ResultsFilter, volumes: Sequence[str]) -> None:
        reads.append(list(volumes))
        reads.append(list(volumes))

    monkeypatch.setattr(ResultsFilter, '_scan_volumes', recording)
    ResultsFilter(iter(VOLUMES), str(tree), logger=_logger(), has_offset_file=True)
    assert reads[1] == VOLUMES


def _reported_line(out: str) -> str:
    """Return the line reporting what the index holds.

    Parameters:
        out: Everything the filter wrote.

    Returns:
        The one line naming the counts and the age of the answer.
    """
    return next(line for line in out.splitlines() if 'Results index holds' in line)


def test_a_finish_time_in_the_future_is_reported_as_it_stands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two machines disagreeing by seconds is ordinary, and is not an interval.

    The pass is finished by whichever machine ran the ingest and the stamp is
    read by another, so a workstation a few seconds behind a cloud worker reads
    a moment that has not happened yet.  Reporting one as "less than a minute
    ago" would state an interval that is not one.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    stamp = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    _stamp_run(url, root, finished_utc=stamp)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert _reported_line(capsys.readouterr().out).endswith(stamp)


def test_a_finish_time_with_no_offset_is_read_as_utc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An index restored from elsewhere is where a stamp with no offset comes from.

    Every pass this pipeline runs writes an offset, so this is the same input
    the unreadable stamp is: a column filled by something else.  Without a
    reading for it, subtracting it raises out of the constructor, which is a
    crash of the enumeration rather than an answer about the index.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    naive = (datetime.now(UTC) - timedelta(days=2)).replace(tzinfo=None).isoformat()
    _stamp_run(url, root, finished_utc=naive)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert _reported_line(capsys.readouterr().out).endswith(f'{naive} (2 days ago)')


def test_a_recorded_finish_time_of_nothing_is_reported_as_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The column is not null, so an empty string is a value a consumer meets.

    It says nothing about when the pass finished, and the report says that
    rather than naming a moment or leaving the sentence unfinished.
    """
    root, _images = _one_image_tree(tmp_path)
    url = _index_of_two_roots(tmp_path, root, missed=0)
    _stamp_run(url, root, finished_utc='')
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=_reporting_logger(),
        results_db_url=url,
        has_no_offset_file=True,
    )
    assert 'at a time this index does not record' in capsys.readouterr().out
