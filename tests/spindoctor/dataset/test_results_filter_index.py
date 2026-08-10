"""The selection filters answered from a results index, against the same tree.

The parity matrix is here: every filter flag asked of the tree and of the index
over the one fixture tree, each held to a stated answer, plus the assertions
that the index path reads no file at all and leaves nothing for the batch stage.
So is every document shape the two paths could plausibly read differently and do
not, each asked of both over a tree of its own.  The refusals are here too,
because a filter that cannot answer has to say so in one type a program can
catch, and so is the guarantee that the navigation critical path never imports
the database layer.

Two files carry the rest: the answers the index gives differently from the tree,
each with a test of its own, and what the filter reports about the pass that
filled the index.
"""

import itertools
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import sqlalchemy
from filecache import FCPath
from tests.spindoctor.cli.stats.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.dataset.conftest import (
    ERROR_WITHOUT_STATUS_ERROR,
    FATAL_ERRORS,
    NO_RESULT,
    NONSPICE_ERROR,
    SPICE_ERROR,
    VOLUMES,
    WITH_A_DOCUMENT,
    WITHOUT_A_FATAL_ERROR,
    candidate_files,
    index_without_a_table,
    null_logger,
    one_image_tree,
    select_from,
    selection_of,
)

from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import (
    _SPICE_STATUS_ERROR,
    ResultsFilter,
    SelectionError,
)
from spindoctor.results_index import SPICE_STATUS_ERROR, selection
from spindoctor.results_index.selection import ResultStubs

_MATRIX = [
    pytest.param({'has_offset_file': True}, list(WITH_A_DOCUMENT), id='offset-file'),
    pytest.param({'has_no_offset_file': True}, [NO_RESULT], id='no-offset-file'),
    pytest.param({'has_offset_error': True}, list(FATAL_ERRORS), id='offset-error'),
    pytest.param({'has_no_offset_error': True}, list(WITHOUT_A_FATAL_ERROR), id='no-offset-error'),
    pytest.param({'has_offset_spice_error': True}, [SPICE_ERROR], id='spice-error'),
    pytest.param(
        {'has_offset_nonspice_error': True},
        [NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR],
        id='nonspice-error',
    ),
    pytest.param(
        {'has_offset_error': True, 'has_offset_file': True},
        list(FATAL_ERRORS),
        id='offset-error-and-offset-file',
    ),
    pytest.param(
        {'has_offset_spice_error': True, 'has_offset_file': True},
        [SPICE_ERROR],
        id='spice-error-and-offset-file',
    ),
    pytest.param(
        {'has_no_offset_error': True, 'has_offset_file': True},
        list(WITHOUT_A_FATAL_ERROR),
        id='no-offset-error-and-offset-file',
    ),
]
"""Every filter flag, alone and paired, with the selection it makes.

The flags reach the tree two ways, and both have to land on the index path's
one answer: the absence filter alone takes the batched ``exists()`` path and
never walks the tree, while the presence and error filters walk it.  The
presence filter is answered from the walked set; an error filter is answered
from the document itself, the walked set only pruning the candidates it is then
retrieved for.  No error filter can reach the batched ``exists()`` path at all:
each folds presence in, which is what asks for the walk.

The pairings are the combinations a user has a reason to write.  ``images this
run navigated to a result`` is the last of them, and it is the pair rather than
a flag because presence and outcome are separate questions in this vocabulary;
that each error filter folds presence in, so that naming it changes nothing, is
what the first two pin.
"""


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_tree_answers_each_filter(
    tree: Path, flags: dict[str, bool], expected: list[str]
) -> None:
    """What reading the results tree selects, stated rather than compared."""
    assert selection_of(tree, flags, results_db_url=None) == expected


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_index_answers_each_filter_the_same_way(
    tree: Path, indexed: str, flags: dict[str, bool], expected: list[str]
) -> None:
    """One query reaches the answer the walk and the per-image reads reach."""
    assert selection_of(tree, flags, results_db_url=indexed) == expected


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
    assert selection_of(tree, flags, results_db_url=indexed) == expected


def test_an_image_with_no_document_is_not_one_recording_no_error_in_the_tree(tree: Path) -> None:
    """The negative error filter asks what a document records, so it needs one.

    An image nothing has been written for is what ``has_no_offset_file``
    selects.  Reading its absence as an outcome would put it in both selections
    at once and leave no way to ask for either without the other.
    """
    kept = selection_of(tree, {'has_no_offset_error': True}, results_db_url=None)
    assert NO_RESULT not in kept


@pytest.mark.parametrize(
    'flag',
    [
        'has_offset_error',
        'has_no_offset_error',
        'has_offset_spice_error',
        'has_offset_nonspice_error',
    ],
)
def test_an_error_filter_prunes_a_missing_document_against_the_walked_set(
    tree: Path, flag: str
) -> None:
    """Every error filter folds presence in, and this is where that shows.

    The fold-in changes no answer: without it the retrieval of a document that
    is not there fails and the batch stage drops the image anyway.  What it
    changes is the cost, and only here -- an image nothing has been written for
    is settled by a set already in memory rather than by one retrieval per
    candidate, which on a cloud root is a paid round trip per image.  The
    population that pays it is every image the run has yet to navigate, so it
    is the common case rather than the corner.

    Parameters:
        tree: The results root under test.
        flag: The error filter, one per flag that reads a document.
    """
    results_filter = ResultsFilter(
        VOLUMES, str(tree), logger=null_logger(), results_db_url=None, **{flag: True}
    )
    assert results_filter.passes_presence(NO_RESULT) is False


def test_an_image_with_no_row_is_not_one_recording_no_error_in_the_index(
    tree: Path, indexed: str
) -> None:
    """Absence of a row is absence of a document, and reads the same way here."""
    kept = selection_of(tree, {'has_no_offset_error': True}, results_db_url=indexed)
    assert NO_RESULT not in kept


_NO_OUTCOME_STUB = 'COISS_2001/data/b/N1000000011_1_CALIB'
"""The one image of the tree written for the document naming no outcome.

Under a selected volume, since a stub outside one is answered by neither path
and would make every expectation below the empty selection for the wrong
reason.  It is not a stub of the shared fixture tree: that tree states an answer
per filter for ten documents at once, and what is asked here is what one
document shape answers.
"""

_NO_STATUS_FIELD = object()
"""Stands for the document that carries no top-level ``status`` field at all."""

_NO_OUTCOME_STATUSES = [
    pytest.param(_NO_STATUS_FIELD, id='absent'),
    pytest.param(None, id='null'),
    pytest.param('', id='empty'),
    pytest.param(42, id='not-a-string'),
]
"""Every shape of a top-level ``status`` that names no outcome.

The four are kept apart because one reader tells them apart by type and another
by SQL: a document carrying no field, one carrying null, one carrying the empty
string and one carrying a number reach the store as four different values and
have to leave it as one.
"""

_NO_OUTCOME_MATRIX = [
    pytest.param({'has_offset_file': True}, [_NO_OUTCOME_STUB], id='offset-file'),
    pytest.param({'has_offset_error': True}, [], id='offset-error'),
    pytest.param({'has_no_offset_error': True}, [_NO_OUTCOME_STUB], id='no-offset-error'),
    pytest.param({'has_offset_spice_error': True}, [], id='spice-error'),
    pytest.param({'has_offset_nonspice_error': True}, [], id='nonspice-error'),
]
"""What every filter that reads a document answers about one naming no outcome.

The presence filter leads, because it is what says the document is there to be
read: without it every empty answer below would also be the answer for a root
holding nothing.  ``has_no_offset_error`` carries the same weight from the other
side -- a document the ingest refused matches it no more than it matches the
rest, so an answer of the image itself is an answer read off a stored outcome.
"""


def _naming_no_outcome(root: Path, status: Any) -> list[ImageFile]:
    """Write a document that names no outcome of its own, and return its image.

    The outcome sits in the nested ``navigation_result`` copy instead, which is
    what makes the answers evidence of anything: a path that read the nested
    copy wherever the top-level field names nothing would answer the error
    filters with this image and the negative one without it.

    Parameters:
        root: The results root to write into.
        status: The top-level ``status`` the document carries, or
            :data:`_NO_STATUS_FIELD` for one carrying no such field.

    Returns:
        The one candidate image, ready to filter.
    """
    document = metadata_document(image_name='N1000000011_1.IMG', offset=None)
    if status is _NO_STATUS_FIELD:
        del document['status']
    else:
        document['status'] = status
    document['navigation_result']['status'] = 'error'
    write_metadata(root, _NO_OUTCOME_STUB, document)
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{_NO_OUTCOME_STUB}.IMG'),
            label_file_url=FCPath(root / f'{_NO_OUTCOME_STUB}.LBL'),
            results_path_stub=_NO_OUTCOME_STUB,
        )
    ]


@pytest.mark.parametrize('status', _NO_OUTCOME_STATUSES)
@pytest.mark.parametrize(('flags', 'expected'), _NO_OUTCOME_MATRIX)
def test_the_tree_reads_a_document_naming_no_outcome(
    tmp_path: Path, status: Any, flags: dict[str, bool], expected: list[str]
) -> None:
    """The walk reads the top-level field and no other, so it finds no outcome.

    Parameters:
        tmp_path: Directory the root is written under.
        status: The shape of top-level ``status`` the document carries.
        flags: The selection flags to apply.
        expected: The stubs the filter selects.
    """
    root = tmp_path / 'results'
    images = _naming_no_outcome(root, status)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=None, **flags
    )
    assert select_from(results_filter, images) == expected


@pytest.mark.parametrize('status', _NO_OUTCOME_STATUSES)
@pytest.mark.parametrize(('flags', 'expected'), _NO_OUTCOME_MATRIX)
def test_the_index_reads_a_document_naming_no_outcome_the_same_way(
    tmp_path: Path, status: Any, flags: dict[str, bool], expected: list[str]
) -> None:
    """The stored status is the document's own field, so the query answers alike.

    Every shape above is stored as the one value a record naming no outcome is
    read as, which is an outcome the error filters name nowhere.  A store that
    borrowed the nested copy would be making a classification no reader of the
    document could arrive at, and -- since the pointing readers rebuild a record
    from these same columns -- would hand a corrected attitude to an image whose
    document supplies no pointing at all.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        status: The shape of top-level ``status`` the document carries.
        flags: The selection flags to apply.
        expected: The stubs the filter selects.
    """
    root = tmp_path / 'results'
    images = _naming_no_outcome(root, status)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_db_url=url, **flags
    )
    assert select_from(results_filter, images) == expected


_FLAG_CASES = [pytest.param(case.values[0], id=case.id) for case in _MATRIX]
"""The same filter combinations, for the assertions that do not need the answer."""


@pytest.mark.parametrize('flags', _FLAG_CASES)
def test_the_index_path_leaves_nothing_for_the_batch_stage(
    tree: Path, indexed: str, flags: dict[str, bool]
) -> None:
    """The enumeration buffers images only to amortize round trips an index has none of."""
    results_filter = ResultsFilter(
        VOLUMES, str(tree), logger=null_logger(), results_db_url=indexed, **flags
    )
    assert results_filter.needs_batch_filtering is False


CONTRADICTORY_PAIRS = [
    pytest.param({'has_offset_file': True, 'has_no_offset_file': True}, id='offset-file-pair'),
    pytest.param(
        {'has_offset_spice_error': True, 'has_offset_nonspice_error': True}, id='error-pair'
    ),
    pytest.param(
        {'has_offset_error': True, 'has_no_offset_file': True}, id='error-and-no-offset-file'
    ),
    pytest.param({'has_offset_error': True, 'has_no_offset_error': True}, id='error-and-no-error'),
    pytest.param(
        {'has_offset_spice_error': True, 'has_no_offset_error': True},
        id='spice-error-and-no-error',
    ),
    pytest.param(
        {'has_offset_nonspice_error': True, 'has_no_offset_error': True},
        id='nonspice-error-and-no-error',
    ),
    pytest.param(
        {'has_no_offset_error': True, 'has_no_offset_file': True},
        id='no-error-and-no-offset-file',
    ),
    pytest.param(
        {'has_offset_spice_error': True, 'has_no_offset_file': True},
        id='spice-error-and-no-offset-file',
    ),
    pytest.param(
        {'has_offset_nonspice_error': True, 'has_no_offset_file': True},
        id='nonspice-error-and-no-offset-file',
    ),
]
"""Every pair of selection flags no image could satisfy.

That it is every one of them is asserted rather than claimed: the six flags make
fifteen pairs, and the test below puts each of the fifteen to the constructor and
holds the ones it refuses to exactly this list.  A list short by a pair leaves
the message that pair produces unasserted, which is how two of the four
contradictions came to have a path through the message builder nothing read.
"""

CONTRADICTION_REFUSAL = r'mutually exclusive|cannot be combined with'
"""The two shapes a refusal of contradictory selection flags takes.

Two flags that exclude each other and nothing else are mutually exclusive.  One
flag that excludes several which are satisfiable together is named against the
ones it excludes instead: ``has_offset_error`` and ``has_offset_spice_error``
are a pair the constructor accepts, so a message calling the three of them
mutually exclusive would assert an exclusion between two flags that have none.
"""

SELECTION_FLAGS = (
    'has_offset_file',
    'has_no_offset_file',
    'has_offset_error',
    'has_no_offset_error',
    'has_offset_spice_error',
    'has_offset_nonspice_error',
)
"""The six results-file selection flags."""

COMBINATIONS = [
    names
    for size in range(2, len(SELECTION_FLAGS) + 1)
    for names in itertools.combinations(SELECTION_FLAGS, size)
]
"""Every combination of two or more of them, which is what a user may type.

A user types flags rather than pairs, and a refusal is about the combination
they typed: three of these carry two contradictions at once, and the
contradiction a run happens to notice first is not the whole of what is wrong
with the selection.
"""


def _contradicted_within(names: Sequence[str]) -> set[str]:
    """Return the flags of one combination that another flag of it contradicts.

    Parameters:
        names: The flags the user typed.

    Returns:
        Every flag belonging to a contradictory pair both of whose flags are in
        the combination.  A flag that contradicts nothing else present is not
        one of them: it is a narrowing the selection could have satisfied, and
        naming it in a refusal would send its user to change the wrong thing.
    """
    given = set(names)
    return {
        name
        for case in CONTRADICTORY_PAIRS
        for pair in [set(cast(dict[str, bool], case.values[0]))]
        if pair <= given
        for name in pair
    }


def _refusal_of(tree: Path, names: Sequence[str]) -> str | None:
    """Build a filter over the given flags and return what it refused, if it did.

    Parameters:
        tree: The results root, read only by the combinations that are not
            refused.
        names: The flags to turn on.

    Returns:
        The refusal's message, or None when the combination was accepted.
    """
    flags = dict.fromkeys(names, True)
    try:
        ResultsFilter(VOLUMES, str(tree), logger=null_logger(), results_db_url=None, **flags)
    except SelectionError as exc:
        return str(exc)
    return None


def test_exactly_the_combinations_holding_a_contradictory_pair_are_refused(tree: Path) -> None:
    """The list of contradictions is the whole of what the constructor refuses.

    Asserted over every combination rather than over the pairs alone, so that a
    contradiction reachable only by three flags together, and a combination
    refused for no pair anybody wrote down, are both failures here.
    """
    refused = {names for names in COMBINATIONS if _refusal_of(tree, names) is not None}
    assert refused == {names for names in COMBINATIONS if _contradicted_within(names)}


def test_a_refusal_names_every_flag_of_the_combination_it_cannot_satisfy(tree: Path) -> None:
    """A flag left out of the message reads as one the run accepted.

    A combination carrying two contradictions is refused for both, because a
    user who removes the flag the first one named and runs again would otherwise
    meet the second, and a run refused a pair at a time costs a run per pair.
    """
    unnamed = {
        names: sorted(name for name in _contradicted_within(names) if name not in (message or ''))
        for names in COMBINATIONS
        if (message := _refusal_of(tree, names)) is not None
    }
    assert {names: missing for names, missing in unnamed.items() if missing} == {}


def _pairs_called_mutually_exclusive(tree: Path) -> set[frozenset[str]]:
    """Return every pair of flags a refusal claims cannot hold together.

    "Mutually exclusive" is a claim about each pair of the flags it leads, so a
    clause leading three of them claims three pairs.

    Parameters:
        tree: The results root the combinations are put to.

    Returns:
        One frozen pair per claim any refusal makes.
    """
    claimed: set[frozenset[str]] = set()
    for names in COMBINATIONS:
        message = _refusal_of(tree, names)
        if message is None:
            continue
        for clause in message.split('; '):
            if 'mutually exclusive' not in clause:
                continue
            lead = clause.split(':')[0]
            named = sorted(name for name in SELECTION_FLAGS if name in lead)
            claimed.update(frozenset(pair) for pair in itertools.combinations(named, 2))
    return claimed


def test_no_refusal_calls_a_satisfiable_pair_mutually_exclusive(tree: Path) -> None:
    """A refusal that says more than it means sends its user to change the wrong flag.

    ``has_offset_error`` and ``has_offset_spice_error`` are a pair the
    constructor accepts, so a message that named them and
    ``has_no_offset_file`` together as mutually exclusive would tell a user that
    a selection this program answers is impossible.  The refusal that names one
    flag against the ones it excludes says only what is true, and this is what
    holds every "mutually exclusive" clause to a pair that really is one.
    """
    exclusive = {frozenset(cast(dict[str, bool], case.values[0])) for case in CONTRADICTORY_PAIRS}
    unfounded = _pairs_called_mutually_exclusive(tree) - exclusive
    assert sorted(sorted(pair) for pair in unfounded) == []


@pytest.mark.parametrize('flags', CONTRADICTORY_PAIRS)
def test_a_contradictory_pair_is_refused_before_the_index_is_opened(
    tree: Path, tmp_path: Path, flags: dict[str, bool]
) -> None:
    """The flags are validated first, so the refusal is the same with or without one.

    The URL names a database that does not exist, so a constructor that opened
    the index before checking its flags would report that instead.
    """
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(ValueError, match=CONTRADICTION_REFUSAL) as excinfo:
        ResultsFilter(VOLUMES, str(tree), logger=null_logger(), results_db_url=absent, **flags)
    assert 'not-an-index.sqlite3' not in str(excinfo.value)


@pytest.mark.parametrize('flags', CONTRADICTORY_PAIRS)
def test_a_refusal_names_every_flag_that_made_the_selection_impossible(
    tree: Path, flags: dict[str, bool]
) -> None:
    """The message is the whole diagnosis, so it names what the user typed.

    A message naming a category rather than a flag -- "the offset-error
    filters" -- leaves the user to work out which of six flags belongs to it,
    and one of the six is named for the absence of an error, so the category
    reads as something they did not ask for.

    Parameters:
        tree: The results root, which is never read: the flags are refused
            first.
        flags: The contradictory pair.
    """
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(VOLUMES, str(tree), logger=null_logger(), results_db_url=None, **flags)
    assert [name for name in flags if name not in str(excinfo.value)] == []


def test_a_root_with_no_completed_ingest_is_refused(tree: Path, indexed: str) -> None:
    """Absence of a row is only an answer under a root somebody ingested."""
    other_root = tree.parent / 'never-ingested'
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        ResultsFilter(
            VOLUMES,
            str(other_root),
            logger=null_logger(),
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
            VOLUMES, str(tree), logger=null_logger(), results_db_url=absent, has_offset_file=True
        )
    assert 'not-an-index.sqlite3' in str(excinfo.value)


def test_both_paths_match_the_same_spice_error() -> None:
    """The two implementations tell a SPICE failure apart by the same value.

    They hold it separately because the tree path may not import the index
    package, so nothing but this holds the two spellings together.
    """
    assert SPICE_STATUS_ERROR == _SPICE_STATUS_ERROR


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


def test_an_index_that_will_not_answer_refuses_the_selection(tmp_path: Path) -> None:
    """An index that opened and then failed is a misconfigured run like any other.

    The type is the one a program reporting the message catches, so a database
    failure reaches an operator as the sentence that says what to change rather
    than as a traceback out of an enumeration.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )


def test_an_index_that_will_not_answer_raises_no_database_exception(tmp_path: Path) -> None:
    """This module never imports the database layer, so it may not raise its types."""
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_an_index_that_cannot_be_opened_refuses_the_selection(tmp_path: Path) -> None:
    """The three refusals are one type, so a program catches one and reports all three."""
    root, _images = one_image_tree(tmp_path)
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(SelectionError, match='sd_stats_ingest'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_db_url=absent, has_offset_file=True
        )


def test_a_root_the_index_does_not_cover_refuses_the_selection(tree: Path, indexed: str) -> None:
    """Absence under a root nobody ingested is the third of the three."""
    with pytest.raises(SelectionError, match='no completed ingest'):
        ResultsFilter(
            VOLUMES,
            str(tree.parent / 'never-ingested'),
            logger=null_logger(),
            results_db_url=indexed,
            has_offset_file=True,
        )


def test_a_contradictory_pair_refuses_the_selection(tree: Path) -> None:
    """The flags are the fourth, and the one that needs no index at all."""
    with pytest.raises(SelectionError, match='mutually exclusive'):
        ResultsFilter(
            VOLUMES,
            str(tree),
            logger=null_logger(),
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
        VOLUMES, str(tree), logger=null_logger(), results_db_url=indexed, has_offset_file=True
    )
    from_iterator = ResultsFilter(
        iter(VOLUMES), str(tree), logger=null_logger(), results_db_url=indexed, has_offset_file=True
    )
    images = candidate_files(tree)
    assert select_from(from_iterator, images) == select_from(results_filter, images)


def test_the_volumes_are_fixed_at_the_boundary_for_the_tree(tree: Path) -> None:
    """The walked path is handed the same fixed sequence, for the same reason."""
    results_filter = ResultsFilter(VOLUMES, str(tree), logger=null_logger(), has_offset_file=True)
    from_iterator = ResultsFilter(
        iter(VOLUMES), str(tree), logger=null_logger(), has_offset_file=True
    )
    images = candidate_files(tree)
    assert select_from(from_iterator, images) == select_from(results_filter, images)


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
        return ResultStubs(with_metadata=frozenset(), matching_error=frozenset())

    return recording


def test_the_index_path_is_handed_volumes_it_can_be_read_twice_from(
    tree: Path, indexed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary hands on a sequence, not whatever iterable it was given."""
    reads: list[list[str]] = []
    monkeypatch.setattr(selection, 'read_result_stubs', _reads_recorded_by(reads))
    ResultsFilter(
        iter(VOLUMES), str(tree), logger=null_logger(), results_db_url=indexed, has_offset_file=True
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
    ResultsFilter(iter(VOLUMES), str(tree), logger=null_logger(), has_offset_file=True)
    assert reads[1] == VOLUMES


def test_the_refusal_does_not_repeat_the_query_that_failed(tmp_path: Path) -> None:
    """The wrapper exists so that the sentence to act on is what a reader meets.

    The database layer renders a failed statement with its SQL, its bound
    parameters and a link to its own documentation.  Reported through a program
    that catches this type, that puts the advice under a page of machinery.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )
    assert 'SELECT' not in str(excinfo.value)


def test_the_refusal_carries_what_the_driver_said(tmp_path: Path) -> None:
    """Which table is missing is the whole of what makes the failure actionable."""
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError, match='no such table: failed_files'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )


def test_a_database_failure_of_another_class_is_translated_too(tmp_path: Path) -> None:
    """Every way the layer fails is one type at this seam, not the operational ones.

    A missing table raises one class on SQLite and another on PostgreSQL, and a
    value the driver will not bind raises a third on both.  A caller that never
    imports the database layer cannot name any of them, so the guarantee is the
    family and not a member of it.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    unbindable = cast(list[str], [object()])
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            unbindable, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )


def test_a_database_failure_of_another_class_raises_no_database_exception(
    tmp_path: Path,
) -> None:
    """This module never imports the database layer, so it may not raise its types."""
    root, _images = one_image_tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    unbindable = cast(list[str], [object()])
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            unbindable, str(root), logger=null_logger(), results_db_url=url, has_offset_file=True
        )
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)
