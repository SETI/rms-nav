"""Every answer the index gives differently from the tree, one test each.

Each of these is a property of what one ingest pass could read and record rather
than of the query, and each is enumerated in
:mod:`spindoctor.results_index.selection`, in the plan, and here.  They are
tested in both directions -- what the tree answers and what the index answers --
because what makes each a divergence is that the two differ, and an assertion
about one of them alone would pass if the other silently changed to match.

A member added to the enumeration is added here.  One that stops being a
divergence keeps its tests, asserting the agreement instead: what made it worth
a test is that the two answers could differ, and that is as true of the answer
that now matches as of the one that did not.
"""

import os
from pathlib import Path
from typing import Any

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
from tests.spindoctor.dataset.conftest import (
    MALFORMED,
    SPICE_ERROR,
    SUCCESS_NO_PNG,
    SUCCESS_WITH_PNG,
    UNLISTABLE,
    VOLUMES,
    null_logger,
    one_image_tree,
    refusing_to_list,
    select_from,
    write_bytes,
)

from spindoctor.cli.stats.ingest import store
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import ResultsFilter
from spindoctor.results_index import SPICE_STATUS_ERROR


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
    results_filter = ResultsFilter(VOLUMES, str(root), logger=null_logger(), has_png_file=True)
    assert select_from(results_filter, images) == [SUCCESS_WITH_PNG]


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
    ingest_tree(url, [root], logger=null_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=SUCCESS_WITH_PNG,
        )
    ]
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_png_file=True
    )
    assert select_from(results_filter, images) == []


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
        VOLUMES, str(root), logger=null_logger(), has_offset_spice_error=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


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
    ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES,
        str(root),
        logger=null_logger(),
        results_db_url=url,
        has_offset_spice_error=True,
    )
    assert select_from(results_filter, images) == []


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
    write_bytes(root, MALFORMED, b'{"status": "error"')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    write_summary_png(root, MALFORMED)
    ingest_tree(url, [root], logger=null_logger())
    images = [
        ImageFile(
            image_file_url=FCPath(root / 'x.IMG'),
            label_file_url=FCPath(root / 'x.LBL'),
            results_path_stub=MALFORMED,
        )
    ]
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_png_file=True
    )
    assert select_from(results_filter, images) == [MALFORMED]


def _status_only_in_the_navigation_result(root: Path) -> list[ImageFile]:
    """Write a document whose outcome is recorded only under ``navigation_result``.

    The outcome is in the nested copy and the top-level field both paths read
    is absent.  Neither path takes the nested copy for it, which is what keeps
    the two agreeing, and is asserted here because a column that borrowed it
    would be a classification made at ingest time -- an answer no reader of the
    document could arrive at.

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
    results_filter = ResultsFilter(VOLUMES, str(root), logger=null_logger(), has_offset_error=True)
    assert select_from(results_filter, images) == []


def test_a_status_only_in_the_navigation_result_matches_no_index_error_filter(
    tmp_path: Path,
) -> None:
    """And the index reads the same field, so it answers the same way.

    The ``status`` column holds the document's own top-level field and nothing
    standing in for it.  A column that fell back to the nested copy would match
    an error filter here for a document that matches none in the tree, and --
    since the pointing readers rebuild a record from these same columns -- would
    apply a corrected attitude to an image whose document supplies no pointing
    at all.
    """
    root = tmp_path / 'results'
    images = _status_only_in_the_navigation_result(root)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_error=True
    )
    assert select_from(results_filter, images) == []


def test_a_status_only_in_the_navigation_result_is_still_a_row(tmp_path: Path) -> None:
    """The control for the agreement above, which an uningested document passes.

    The document is a navigation document and ingests; what it is not is a
    document naming an outcome, so it matches no filter that names one.
    """
    root = tmp_path / 'results'
    images = _status_only_in_the_navigation_result(root)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


def test_a_file_the_pass_could_not_retrieve_reads_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is recorded for it, deliberately, so the next pass tries it again.

    A recorded refusal would be skipped for as long as the file did not change,
    and a download that failed once says nothing that will still be true then.
    The cost is that the file reads as absent until a pass reads it, which is one
    of the answers the index gives differently.
    """
    root, images = one_image_tree(tmp_path)

    def refuse(self: FCPath, *args: Any, **kwargs: Any) -> list[Exception]:
        return [OSError('the backend did not answer') for _ in args[0]]

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(FCPath, 'retrieve', refuse)
    ingest_tree(url, [root], logger=null_logger())
    monkeypatch.undo()
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == []


def test_a_document_the_database_would_not_store_reads_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row the database refuses is counted and recorded nowhere, on the same grounds.

    The document read exactly as the schema says, so nothing about it says the
    next pass will not store it, and a recorded refusal would stop the next pass
    from trying.  It therefore reads as absent, exactly as a file nobody could
    retrieve does.
    """
    root, images = one_image_tree(tmp_path)

    def refuse(connection: Any, rows: Any) -> None:
        raise sqlalchemy.exc.IntegrityError('INSERT', {}, Exception('refused'))

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(store, '_write_image', refuse)
    ingest_tree(url, [root], logger=null_logger())
    monkeypatch.undo()
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == []


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
    ingest_tree(url, [root], logger=null_logger())
    (root / f'{SPICE_ERROR}_metadata.json').unlink()
    if not listing_the_whole_root:
        refusing_to_list(monkeypatch, root / UNLISTABLE)
    ingest_tree(url, [root], logger=null_logger())
    monkeypatch.undo()
    return root, images, url


def test_a_document_that_left_the_tree_reads_as_absent_in_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk finds what is there now, which is the answer the index is held to."""
    root, images, _url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=True
    )
    results_filter = ResultsFilter(VOLUMES, str(root), logger=null_logger(), has_offset_file=True)
    assert select_from(results_filter, images) == [SUCCESS_NO_PNG]


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
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == [SUCCESS_NO_PNG]


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
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == [SUCCESS_NO_PNG, SPICE_ERROR]


def test_the_tree_offers_a_document_that_left_it_to_the_absence_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been written for that image now, so the resume idiom picks it up."""
    root, images, _url = _index_after_a_document_left_the_tree(
        tmp_path, monkeypatch, listing_the_whole_root=False
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), has_no_offset_file=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


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
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_no_offset_file=True
    )
    assert select_from(results_filter, images) == []


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
    ingest_tree(url, [root], logger=null_logger())
    before = document.stat()
    replacement = OTHER_STATUS_ERROR if keeping_its_size else f'{OTHER_STATUS_ERROR}_and_then_some'
    write_metadata(root, SPICE_ERROR, _fatal_error_document(replacement))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (document.stat().st_size == before.st_size) is keeping_its_size
    ingest_tree(url, [root], logger=null_logger())
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
    ingest_tree(url, [root], logger=null_logger())
    before = document.stat()
    write_metadata(root, SPICE_ERROR, _fatal_error_document(OTHER_STATUS_ERROR))
    os.utime(document, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert ingest_tree(url, [root], logger=null_logger()).files_skipped == 1


def test_a_document_rewritten_in_place_reads_as_it_is_now_in_the_tree(tmp_path: Path) -> None:
    """The walk opens the file, so it answers from the document that is there."""
    root, images, _url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), has_offset_spice_error=True
    )
    assert select_from(results_filter, images) == []


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
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


def test_a_rewrite_that_changes_the_length_is_read_again(tmp_path: Path) -> None:
    """The divergence is the equal metrics, not the rewrite.

    A document rewritten to a different length is one the pass has evidence
    about, and the row it leaves is what the tree would answer.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(
        tmp_path, keeping_its_size=False
    )
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert select_from(results_filter, images) == []


def test_a_forced_pass_corrects_a_document_rewritten_in_place(tmp_path: Path) -> None:
    """Reading every document regardless is what an operator has to reach for.

    It is the remedy because the alternative is reading the file to find out
    whether it needs reading, which is the cost the skip exists to avoid.
    """
    root, images, url = _index_after_a_document_was_rewritten_in_place(tmp_path)
    ingest_tree(url, [root], logger=null_logger(), force=True)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_spice_error=True
    )
    assert select_from(results_filter, images) == []
