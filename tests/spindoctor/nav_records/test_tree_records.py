"""What the record source reads, in what batches, and what it will not read.

One stub names one document.  A stream reads what the listing found, in batches,
and hands back one record at a time -- batched underneath because on a cloud root
each file is a round trip, lazy on top because the caller should not have to hold
a mission to read the first record of one.

A file that yielded no record is reported into the stream rather than raised on:
it names no image, so there is nothing for a run to omit and nothing an omission
reason could be recorded against.  What is passed over silently is what is simply
not this run's business -- another mission's document, and a record outside the
selection's span.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest

from spindoctor.nav_records import tree as tree_module
from spindoctor.nav_records import (
    METADATA_SUFFIX,
    NAMES_NO_INSTRUMENT,
    RECORDS_NO_MIDTIME,
    RETRIEVE_BATCH_SIZE,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)

from .conftest import (
    FIRST_STUB,
    MISSION,
    OTHER_MISSION,
    SECOND_STUB,
    count_reads,
    count_retrievals,
    document,
    failing_retrievals,
    reasons_of,
    records_of,
    stubs_of,
    timed_document,
    tree_source,
    two_volume_tree,
    unlistable_subdirectory,
    write_document,
    write_text,
)

# ---------------------------------------------------------------------------
# One image by its stub
# ---------------------------------------------------------------------------


def test_one_stub_reads_its_own_document(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The per-image shape: one stub, one file read, one record."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert source.record(FIRST_STUB).metadata['status'] == 'success'


def test_one_stub_carries_its_own_identity_back(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A record is paired with the stub it answers for, whatever the document says."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert source.record(FIRST_STUB).stub == FIRST_STUB


def test_a_stub_with_no_document_raises_the_error_a_caller_distinguishes(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unnavigated image is told apart from an unreadable document."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError):
        source.record('VOL1/N0000000000_1_CALIB')


def test_a_stub_that_escapes_its_root_reads_as_an_image_with_no_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A key is not a path, and a key holding ``..`` names a file outside the root."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError, match='does not name a navigation document'):
        source.record('../../elsewhere/N1454725799')


def test_a_stub_that_escapes_its_root_is_told_which_rule_refused_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Two things mean "no readable record here", and the message says which."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError) as excinfo:
        source.record('../../elsewhere/N1454725799')
    assert 'names a parent directory' in str(excinfo.value)


def test_a_stub_that_resolves_back_inside_its_root_is_refused_too(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It names a document this root does hold, under a key no consumer stores.

    A stub is the identity of an image, so two spellings of one document are two
    identities: read under this one, the record comes back where an index keyed
    on the canonical spelling says the image was never navigated.
    """
    root = tmp_path / 'results'
    write_document(root, 'BARESCENE', document())
    source = tree_source(root, quiet_logger)
    with pytest.raises(FileNotFoundError, match='names a parent directory'):
        source.record('VOL1/../BARESCENE')


def test_a_stub_spelled_as_an_absolute_path_reads_no_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Joining an absolute fragment discards the root, so it would read anything."""
    planted = tmp_path / 'elsewhere'
    write_document(planted, 'STOLEN', document())
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    stub = (planted / 'STOLEN').as_posix()
    with pytest.raises(FileNotFoundError, match='fragment is absolute'):
        source.record(stub)


def test_a_selection_naming_a_stub_that_escapes_its_root_is_refused_where_it_is_written(
    tmp_path: Path,
) -> None:
    """A queue task carries stubs, and a task file is written by hand.

    Refused in the selection rather than in a source, so the walk and the query
    refuse it alike: there is one place the key can be wrong and neither storage
    is reached from it.
    """
    write_document(tmp_path / 'elsewhere', 'STOLEN', document())
    with pytest.raises(ValueError, match='is not a results path stub'):
        Selection(stubs=('../elsewhere/STOLEN',))


def test_a_selection_naming_a_subtree_that_escapes_its_root_is_refused(
    tmp_path: Path,
) -> None:
    """The other parameter a caller can walk out of a root with."""
    with pytest.raises(ValueError, match='is not a subtree of a results root'):
        Selection(subtrees=('..',))


def test_a_document_that_is_not_a_json_object_refuses_the_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Reading a field off an array would fail later and further away."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '[1, 2]')
    with pytest.raises(ValueError, match='not a JSON object'):
        tree_source(root, quiet_logger).record(FIRST_STUB)


def test_one_stub_is_refused_by_a_source_holding_more_than_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub is a key under a root, and says nothing about which root."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError, match='key under one root'):
        source.record(FIRST_STUB)


def test_the_refusal_of_a_bare_stub_names_every_root_the_source_holds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming one of two would leave the reader unable to see the ambiguity."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError) as excinfo:
        source.record(FIRST_STUB)
    assert first.as_posix() in str(excinfo.value)
    assert second.as_posix() in str(excinfo.value)


# ---------------------------------------------------------------------------
# The stream, and what it does in batches
# ---------------------------------------------------------------------------


def _many_documents(tmp_path: Path, count: int) -> Path:
    """Write a results tree holding more documents than one retrieval batch.

    Parameters:
        tmp_path: Directory the tree lives under.
        count: How many documents to write.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for index in range(count):
        write_document(root, f'VOL1/N{index:010d}_1_CALIB', document())
    return root


def test_every_record_of_a_tree_larger_than_one_batch_comes_back(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Batching must not lose the tail of a batch or the last part-batch."""
    count = RETRIEVE_BATCH_SIZE * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    assert len(list(source.records(Selection()))) == count


def test_every_record_of_a_tree_larger_than_one_batch_is_distinct(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A count alone would pass on a batch handed back twice."""
    count = RETRIEVE_BATCH_SIZE * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    assert len(set(stubs_of(source.records(Selection())))) == count


def test_a_tree_larger_than_one_batch_is_retrieved_in_batches(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retrieval per batch, not one per file: that is what a cloud root pays."""
    count = RETRIEVE_BATCH_SIZE * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    calls = count_retrievals(monkeypatch)
    list(source.records(Selection()))
    assert calls == [RETRIEVE_BATCH_SIZE, RETRIEVE_BATCH_SIZE, RETRIEVE_BATCH_SIZE, 1]


def test_the_first_record_of_a_stream_costs_one_document_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream is lazy on top, so a caller does not have to hold a mission."""
    source = tree_source(_many_documents(tmp_path, RETRIEVE_BATCH_SIZE * 3), quiet_logger)
    read = count_reads(monkeypatch)
    stream = source.records(Selection())
    next(stream)
    assert len(read) == 1


def test_the_first_record_of_a_stream_costs_one_retrieval(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batched underneath: one batch is fetched, not the whole tree and not one file."""
    source = tree_source(_many_documents(tmp_path, RETRIEVE_BATCH_SIZE * 3), quiet_logger)
    calls = count_retrievals(monkeypatch)
    stream = source.records(Selection())
    next(stream)
    assert calls == [RETRIEVE_BATCH_SIZE]


def test_a_stream_of_named_stubs_reads_exactly_those(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """What a queue task carries: a worker reads the files it was handed."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = list(source.records(Selection(stubs=(SECOND_STUB,))))
    assert stubs_of(found) == [SECOND_STUB]


def test_a_stream_of_named_stubs_reads_them_in_the_order_they_were_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """They name images outright rather than narrowing to them."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = list(source.records(Selection(stubs=(SECOND_STUB, FIRST_STUB))))
    assert stubs_of(found) == [SECOND_STUB, FIRST_STUB]


def test_a_stream_of_named_stubs_lists_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker holding its own list must not pay for a walk of the root."""
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    unlistable_subdirectory(monkeypatch, PermissionError)
    assert stubs_of(source.records(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_named_stubs_are_refused_by_a_source_holding_more_than_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub is a key under a root, however many of them are named at once."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError, match='under one root'):
        list(source.records(Selection(stubs=(FIRST_STUB,))))


def test_named_stubs_are_read_when_the_selection_narrows_to_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Narrowing a two-root source to one is what makes a bare stub a key again."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    selection = Selection(roots=(str(second),), stubs=(FIRST_STUB,))
    assert stubs_of(source.records(selection)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# A file that yielded no record
# ---------------------------------------------------------------------------


def test_a_file_that_never_arrived_is_reported_rather_than_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file that would not download must not cost the rest of the pass."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    failing_retrievals(monkeypatch)
    assert reasons_of(list(source.records(Selection()))) == [
        'could not be retrieved',
        'could not be retrieved',
    ]


def test_a_file_whose_bytes_will_not_read_as_text_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It names no image, so there is nothing for a run to omit and nothing to raise."""
    root = tmp_path / 'results'
    path = root / f'{FIRST_STUB}{METADATA_SUFFIX}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'\xff\xfe\x00\x01')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['unreadable']


def test_a_file_that_is_not_valid_json_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A results tree holds files that are not per-image navigation documents."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not valid JSON']


def test_a_document_nested_past_the_recursion_limit_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The decoder reports more than malformed syntax, and this is not a reason to stop.

    Twenty thousand nested arrays exhaust the recursion limit rather than
    failing to parse, and what happened is the same thing the reason states: no
    value came out of the file.
    """
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '[' * 20000 + ']' * 20000)
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not valid JSON']


def test_a_fault_in_the_reader_itself_is_not_reported_as_a_bad_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed document is a property of the file; a broken reader is not.

    Reported as a bad document it would say every file in the tree was
    malformed, and an operator would go looking at a tree that is fine.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())

    def broken(path: Any) -> dict[str, Any]:
        raise TypeError('the reader was changed and no longer works')

    monkeypatch.setattr(tree_module, 'read_document', broken)
    with pytest.raises(TypeError, match='no longer works'):
        list(tree_source(root, quiet_logger).records(Selection()))


def test_a_file_holding_json_that_is_not_an_object_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Valid JSON that is not a document is unreadable for the same reason."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '[1, 2]')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not a JSON object']


def test_an_unreadable_file_still_carries_the_stub_it_would_have_answered_for(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub comes from where the file is rather than from what it says."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(found) == [FIRST_STUB]


def test_one_unreadable_file_does_not_cost_the_records_beside_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole reason it is a value in the stream rather than an exception."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    write_text(root, SECOND_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# One mission of several
# ---------------------------------------------------------------------------


def _tree_of_two_missions(tmp_path: Path) -> Path:
    """Write a results tree holding this mission's document, another's, and a mute one.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_document(root, 'VOL1/A_mine', document())
    write_document(root, 'VOL1/B_theirs', document(observation={'instrument': OTHER_MISSION}))
    write_document(root, 'VOL1/C_mute', document(observation={'image_name': 'N1.IMG'}))
    return root


def test_only_this_missions_document_becomes_a_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run is per mission, and another mission's images are not its business."""
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert stubs_of(records_of(found)) == ['VOL1/A_mine']


def test_a_document_of_another_mission_is_passed_over_silently(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It is passed over rather than reported: nothing about it went wrong.

    Reporting it would put a line about every image of every other mission into
    a run log, and would make a per-mission pass over a shared tree read as a
    pass that could not read most of what it found.
    """
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert 'VOL1/B_theirs' not in stubs_of(found)


def test_a_document_naming_no_instrument_is_reported_rather_than_passed_over(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only a document that names a mission can be another mission's.

    One with no readable instrument is unreadable, not foreign: skipping it
    silently would let a truncated document vanish from every mission's run
    without a trace.
    """
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert reasons_of(found) == ['names no instrument to attribute it to a mission']


def test_the_mute_document_is_the_one_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A count of one refusal would pass if the wrong file were the one refused."""
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert [entry.stub for entry in found if isinstance(entry, UnreadableFile)] == ['VOL1/C_mute']


@pytest.mark.parametrize(
    'observation',
    [
        pytest.param(None, id='no-observation'),
        pytest.param('later', id='observation-not-a-block'),
        pytest.param({'image_name': 'A_CALIB'}, id='no-instrument'),
        pytest.param({'instrument': None}, id='instrument-null'),
    ],
)
def test_every_way_a_document_can_name_no_instrument_is_reported(
    observation: Any, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Enumerating only some of them lets the others through as another mission's.

    Parameters:
        observation: What the document records where its observation belongs.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    contents = document(status='error')
    if observation is None:
        del contents['observation']
    else:
        contents['observation'] = observation
    write_document(root, FIRST_STUB, contents)
    found = list(tree_source(root, quiet_logger).records(Selection(instrument=MISSION)))
    assert reasons_of(found) == ['names no instrument to attribute it to a mission']


def test_a_document_naming_no_instrument_is_a_record_when_no_mission_is_asked_for(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run reading every mission has nothing to attribute it to and nothing to refuse."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document(observation={'image_name': 'N1.IMG'}))
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# One span of time
# ---------------------------------------------------------------------------


def test_a_record_inside_the_range_is_kept(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The ordinary case the exclusions below are exceptions to."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(0.5))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=0.0, stop_et=1.0)))
    assert stubs_of(found) == [FIRST_STUB]


@pytest.mark.parametrize('midtime', [pytest.param(1.0, id='start'), pytest.param(3.0, id='stop')])
def test_a_record_at_the_range_edge_is_kept(
    midtime: float, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Both bounds are inclusive, so an exposure exactly on one is inside.

    Parameters:
        midtime: The midtime to record, which is one of the two bounds.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0, stop_et=3.0)))
    assert len(found) == 1


@pytest.mark.parametrize('midtime', [pytest.param(0.5, id='before'), pytest.param(3.5, id='after')])
def test_a_record_outside_the_range_is_passed_over(
    midtime: float, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It is not this run's business, so it is not this run's refusal either.

    Parameters:
        midtime: The midtime to record, which is outside the range.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0, stop_et=3.0)))
    assert found == []


@pytest.mark.parametrize(
    'midtime',
    [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='inf'),
        pytest.param(float('-inf'), id='minus-inf'),
        pytest.param(None, id='null'),
        pytest.param(True, id='boolean'),
        pytest.param('later', id='text'),
    ],
)
def test_a_record_with_an_unusable_midtime_is_reported_rather_than_passed_over(
    midtime: Any, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only a document that can be placed in time can be shown to be outside a span.

    A NaN would otherwise fall inside every range at once, and every one of
    these would otherwise vanish from every time-bounded run without a trace --
    which is exactly what the mission filter refuses to do with a document that
    names no mission.

    Parameters:
        midtime: The value to record, of any type.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=0.0, stop_et=1.0)))
    assert reasons_of(found) == [RECORDS_NO_MIDTIME]


def test_an_unplaceable_record_is_reported_the_way_a_mission_less_one_is(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The two filters are one rule, and a run reading both must see both shortfalls."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document(observation={'image_name': 'N1.IMG'}))
    write_document(root, SECOND_STUB, timed_document(None))
    selection = Selection(instrument=MISSION, start_et=0.0, stop_et=1.0)
    found = list(tree_source(root, quiet_logger).records(selection))
    assert sorted(reasons_of(found)) == sorted([NAMES_NO_INSTRUMENT, RECORDS_NO_MIDTIME])


def test_a_record_with_an_unusable_midtime_is_kept_when_no_bound_is_given(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """With no range to satisfy there is nothing to place the image against."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(float('nan')))
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert len(found) == 1


def test_a_lower_bound_alone_still_drops_a_record_before_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A half-bounded range is a range, and a two-sided test would miss it."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(0.5))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0)))
    assert found == []


def test_an_upper_bound_alone_still_drops_a_record_after_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The other half of the same rule."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(2.0))
    found = list(tree_source(root, quiet_logger).records(Selection(stop_et=1.0)))
    assert found == []


# ---------------------------------------------------------------------------
# What the source says about itself
# ---------------------------------------------------------------------------


def test_the_source_names_its_roots_for_the_run_log(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The run log has to say which storage answered, and where."""
    root = two_volume_tree(tmp_path)
    assert root.as_posix() in tree_source(root, quiet_logger).describe()


def test_the_source_is_a_context_manager(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A caller closes every source the same way, whatever that source holds open."""
    root = two_volume_tree(tmp_path)
    with tree_source(root, quiet_logger) as source:
        assert sorted(stubs_of(source.listing(Selection()))) == [FIRST_STUB, SECOND_STUB]


def test_two_spellings_of_one_root_bind_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A source holding one root twice would answer for every image twice."""
    root = two_volume_tree(tmp_path)
    source = TreeRecordSource([root.as_posix(), f'{root.as_posix()}/'], logger=quiet_logger)
    assert len(source.roots) == 1
