"""Hermetic tests for ``spindoctor.cli.ck.inputs``.

What a run reads before it writes anything: the metadata documents a
navigation pass left, the time range that selects among them, and the kernel
directories their provenance names.  Each of these is a place where a value
that is not what it claims to be would otherwise reach the writer -- a NaN
midtime satisfies every time range at once, and a basename two directories
hold is two different kernels.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.ck import inputs


def test_a_kernel_directory_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """Named rather than silently contributing nothing."""
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        inputs.kernel_paths([str(tmp_path / 'gone')])


def test_a_kernel_directory_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """A path that exists and is not a directory fails the same way."""
    path = tmp_path / 'notadir'
    path.write_text('')
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        inputs.kernel_paths([str(path)])


def test_a_basename_no_directory_holds_is_refused(tmp_path: Path) -> None:
    """A kernel an image names and the run cannot find is named, not guessed at."""
    with pytest.raises(ValueError, match='is not in any of the kernel directories'):
        inputs.resolve_one('cas00172.tsc', {})


def test_a_basename_two_directories_hold_is_refused(tmp_path: Path) -> None:
    """Two files of one name are two different kernels, and the record says which."""
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    for root in (first, second):
        root.mkdir()
        (root / 'cas00172.tsc').write_text('')
    paths = inputs.kernel_paths([str(first), str(second)])
    with pytest.raises(ValueError, match='is in more than one kernel directory'):
        inputs.resolve_one('cas00172.tsc', paths)


def test_a_metadata_file_that_is_not_json_is_counted(tmp_path: Path) -> None:
    """It names no image, so it is reported rather than given a report row."""
    (tmp_path / f'broken{inputs.METADATA_SUFFIX}').write_text('{not json')
    documents, unreadable = inputs.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert documents == []
    assert len(unreadable) == 1


def test_a_metadata_file_holding_a_json_array_is_counted(tmp_path: Path) -> None:
    """Valid JSON that is not a document is unreadable for the same reason."""
    (tmp_path / f'listy{inputs.METADATA_SUFFIX}').write_text('[1, 2]')
    _documents, unreadable = inputs.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert len(unreadable) == 1


def test_a_document_from_another_mission_is_not_considered(tmp_path: Path) -> None:
    """A run is per mission, and another mission's images are not its business."""
    (tmp_path / f'other{inputs.METADATA_SUFFIX}').write_text(
        json.dumps({'status': 'success', 'observation': {'instrument': 'vgiss'}})
    )
    documents, unreadable = inputs.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert documents == []
    assert len(unreadable) == 0


def _timed(midtime: Any) -> inputs.Document:
    """Build a document recording one exposure midtime.

    Parameters:
        midtime: The value to record, of any type.

    Returns:
        The document.
    """
    return inputs.Document(
        path=FCPath('x_metadata.json'),
        stub='x',
        metadata={'navigation_result': {'times': {'midtime_et': midtime}}},
    )


@pytest.mark.parametrize(
    'midtime',
    [float('nan'), float('inf'), float('-inf'), None, True, 'later'],
    ids=['nan', 'inf', 'minus-inf', 'null', 'boolean', 'text'],
)
def test_an_unusable_midtime_cannot_be_placed_in_time(midtime: Any) -> None:
    """A NaN would otherwise fall inside every time range at once."""
    selected, undated = inputs.select_by_time([_timed(midtime)], 0.0, 1.0)
    assert selected == []
    assert undated == 1


def test_a_midtime_at_the_range_edge_is_selected() -> None:
    """Both bounds are inclusive, so an exposure exactly on one is inside."""
    selected, _undated = inputs.select_by_time([_timed(1.0)], 1.0, 1.0)
    assert len(selected) == 1


def test_an_unusable_midtime_is_kept_when_no_bound_is_given() -> None:
    """With no range to satisfy there is nothing to place the image against."""
    selected, undated = inputs.select_by_time([_timed(float('nan'))], None, None)
    assert len(selected) == 1
    assert undated == 0


@pytest.mark.parametrize(
    'metadata',
    [
        {},
        {'navigation_result': 'later'},
        {'navigation_result': {}},
        {'navigation_result': {'times': 'later'}},
    ],
    ids=['no-result', 'result-not-a-block', 'no-times', 'times-not-a-block'],
)
def test_a_document_with_no_usable_times_cannot_be_placed_in_time(
    metadata: dict[str, Any],
) -> None:
    """A load-error document records no exposure and satisfies no range."""
    document = inputs.Document(path=FCPath('x_metadata.json'), stub='x', metadata=metadata)
    selected, undated = inputs.select_by_time([document], 0.0, 1.0)
    assert selected == []
    assert undated == 1


@pytest.mark.parametrize(
    'metadata',
    [
        {},
        {'navigation_result': 'later'},
        {'navigation_result': {}},
        {'navigation_result': {'provenance': 'later'}},
        {'navigation_result': {'provenance': {}}},
        {'navigation_result': {'provenance': {'spice_kernels': 'naif0012.tls'}}},
        {'navigation_result': {'provenance': {'spice_kernels': [1, None]}}},
    ],
    ids=[
        'no-result',
        'result-not-a-block',
        'no-provenance',
        'provenance-not-a-block',
        'no-kernel-list',
        'list-is-text',
        'list-holds-no-text',
    ],
)
def test_a_document_recording_no_kernel_names_contributes_none(
    metadata: dict[str, Any],
) -> None:
    """Gathering the run's kernels never fails on one document's record.

    A document that cannot name kernels is refused where it is read as a
    navigated image, by name; refusing it here would refuse it for the whole
    run's pool instead, which says nothing about which document was at fault.
    """
    document = inputs.Document(path=FCPath('x_metadata.json'), stub='x', metadata=metadata)
    assert inputs.recorded_basenames([document]) == ()


def test_a_recorded_kernel_no_directory_holds_is_skipped(tmp_path: Path) -> None:
    """Provenance names every kernel a batch ever furnished, not just this one's.

    A kernel that is genuinely needed and genuinely missing is refused where it
    is used -- an undefined frame is named by the assignment step -- so
    skipping here loses nothing and lets a run proceed on a directory set that
    holds only what this mission needs.
    """
    (tmp_path / 'present.tls').write_text('')
    paths = inputs.kernel_paths([str(tmp_path)])
    assert inputs.furnish_supporting_kernels(['absent.tls'], paths, frozenset({'.tls'})) == ()
