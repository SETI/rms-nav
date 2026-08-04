"""Tests that a missing or unusable navigation offset is reported per image.

Every reason an offset could not be loaded describes one image, so it belongs
in that image's log.  It matters where these land: a cloud task has no run log,
so the same warning bound to the main logger would be discarded, and a
reprojection that silently used uncorrected pointing looks exactly like one
that did not.
"""

import json
from pathlib import Path

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.cli.reproj.offsets import OffsetLookup, load_offset_if_any
from spindoctor.config import IMAGE_LOGGER, LogLevels, LogSinks, build_image_log_handlers
from spindoctor.dataset.dataset import ImageFile

_STAMP = '2026-07-29T12-00-00'
_STUB = 'COISS_2001/N1234567890_1'


def _image_file(stub: str = _STUB) -> ImageFile:
    """Build an image file naming the stub its metadata is looked up under.

    Parameters:
        stub: The results path stub, defaulting to a well-formed one.

    Returns:
        The image file.
    """
    return ImageFile(
        image_file_url=FCPath('/nowhere/N1234567890_1.IMG'),
        label_file_url=FCPath('/nowhere/N1234567890_1.LBL'),
        results_path_stub=stub,
        index_file_row={},
    )


def _write_metadata(nav_root: FCPath, metadata: object) -> None:
    """Write navigation metadata where the offset loader will look for it.

    Parameters:
        nav_root: The navigation results root.
        metadata: The object to serialize as that image's metadata.
    """
    _metadata_path(nav_root).write_text(json.dumps(metadata))


def _metadata_path(nav_root: FCPath) -> FCPath:
    """Return the metadata path for the image these tests navigate.

    Parameters:
        nav_root: The navigation results root.

    Returns:
        The path, with its parent directory created.
    """
    path = nav_root / f'{_STUB}_metadata.json'
    Path(path.as_posix()).parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_and_capture(nav_root: FCPath, log_root: FCPath) -> tuple[OffsetLookup, str]:
    """Load the offset inside an image scope and return it with the image log.

    Parameters:
        nav_root: Root the metadata is read from.
        log_root: Root the per-image log is written under.

    Returns:
        Tuple of the lookup outcome and the text of the image's log.
    """
    handlers, path = build_image_log_handlers(
        'reproj',
        _STUB,
        LogSinks(log_root=log_root),
        LogLevels(),
        timestamp=_STAMP,
    )
    try:
        # The handlers must be attached to the section, not merely built, or
        # the records take the handler-less print fallback and the log file
        # stays empty.
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            lookup = load_offset_if_any(nav_root, _image_file())
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        return lookup, str(stream.read())


def test_a_usable_offset_is_returned(tmp_path: Path) -> None:
    """The stored offset is loaded when navigation succeeded."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, {'status': 'success', 'offset': [1.5, -2.5]})
    lookup, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert lookup.offset == (1.5, -2.5)


@pytest.mark.parametrize(
    'metadata',
    [
        {'status': 'error', 'offset': [1.0, 2.0]},
        {'status': 'success', 'offset': None},
        {'status': 'success', 'offset': [1.0]},
        {'status': 'success', 'offset': 'nope'},
    ],
)
def test_an_unusable_offset_is_refused(tmp_path: Path, metadata: dict[str, object]) -> None:
    """Navigation that did not produce a usable offset yields none."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, metadata)
    lookup, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert lookup.offset is None


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'status': 'error', 'offset': [1.0, 2.0]}, "status='error'"),
        ({'status': 'success', 'offset': None}, 'null offset'),
        ({'status': 'success', 'offset': [1.0]}, 'malformed offset'),
    ],
)
def test_the_reason_reaches_the_image_log(
    tmp_path: Path, metadata: dict[str, object], expected: str
) -> None:
    """Why the pointing was left uncorrected is recorded against the image."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, metadata)
    _, log_text = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert expected in log_text


def test_a_missing_metadata_file_is_reported_to_the_image_log(tmp_path: Path) -> None:
    """An image with no navigation result says so in its own log."""
    _, log_text = _load_and_capture(FCPath(tmp_path) / 'nav', FCPath(tmp_path) / 'logs')
    assert 'no metadata found' in log_text


def test_unparsable_metadata_is_reported_to_the_image_log(tmp_path: Path) -> None:
    """Metadata that is not JSON is reported against the image, not the run."""
    nav_root = FCPath(tmp_path) / 'nav'
    _metadata_path(nav_root).write_text('{not json')
    _, log_text = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert 'Invalid JSON' in log_text


def test_metadata_that_is_not_an_object_is_reported_to_the_image_log(tmp_path: Path) -> None:
    """A JSON document of the wrong shape is reported against the image."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, [1, 2, 3])
    _, log_text = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert 'not a JSON object' in log_text


def test_no_nav_root_loads_nothing(tmp_path: Path) -> None:
    """Reprojecting without navigation results asks for no offset at all."""
    assert load_offset_if_any(None, _image_file()).offset is None


def _plant_loadable_metadata(path: FCPath) -> None:
    """Write metadata that would load successfully if it were reached.

    A stub naming a file that does not exist yields no offset whether or not
    anything refused it, so a test for the refusal has to put a real, valid,
    loadable file at the far end.  Then the only thing between the loader and
    an offset from outside the results root is the guard under test.

    Parameters:
        path: Where to write the metadata.
    """
    Path(path.as_posix()).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'status': 'success', 'offset': [9.0, 9.0]}))


def test_a_traversing_stub_does_not_load_an_offset_from_outside_the_root(
    tmp_path: Path,
) -> None:
    """A stub climbing out of the results root is refused, not followed.

    The planted file is valid and would load: without the guard this returns
    ``(9.0, 9.0)`` read from outside the root the caller named.
    """
    nav_root = FCPath(tmp_path) / 'nav'
    Path((nav_root).as_posix()).mkdir(parents=True, exist_ok=True)
    _plant_loadable_metadata(FCPath(tmp_path) / 'evil' / 'N1234567890_1_metadata.json')
    assert load_offset_if_any(nav_root, _image_file('../evil/N1234567890_1')).offset is None


def test_an_absolute_stub_does_not_load_an_offset_from_outside_the_root(
    tmp_path: Path,
) -> None:
    """Nor does one naming an absolute path, which would ignore the root."""
    nav_root = FCPath(tmp_path) / 'nav'
    Path(nav_root.as_posix()).mkdir(parents=True, exist_ok=True)
    planted = FCPath(tmp_path) / 'elsewhere' / 'N1234567890_1_metadata.json'
    _plant_loadable_metadata(planted)
    stub = planted.as_posix().removesuffix('_metadata.json')
    assert load_offset_if_any(nav_root, _image_file(stub)).offset is None


def test_the_refusal_is_reported_to_the_image_log(tmp_path: Path) -> None:
    """And says why, naming the stub as the thing to look at."""
    nav_root = FCPath(tmp_path) / 'nav'
    Path(nav_root.as_posix()).mkdir(parents=True, exist_ok=True)
    _plant_loadable_metadata(FCPath(tmp_path) / 'evil' / 'N1234567890_1_metadata.json')
    handlers, path = build_image_log_handlers(
        'reproj', _STUB, LogSinks(log_root=FCPath(tmp_path) / 'logs'), LogLevels(), timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            load_offset_if_any(nav_root, _image_file('../evil/N1234567890_1'))
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        assert 'path traversal' in stream.read()


def test_a_stub_with_a_null_byte_is_refused(tmp_path: Path) -> None:
    """A null byte cannot reach the filesystem call at all."""
    nav_root = FCPath(tmp_path) / 'nav'
    Path(nav_root.as_posix()).mkdir(parents=True, exist_ok=True)
    assert load_offset_if_any(nav_root, _image_file('N123\x001')).offset is None


# ---------------------------------------------------------------------------
# The reason the run is told
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('metadata', 'reason'),
    [
        ({'status': 'error', 'offset': [1.0, 2.0]}, 'navigation_did_not_succeed'),
        ({'status': 'success', 'offset': None}, 'null_offset'),
        ({'status': 'success', 'offset': [1.0]}, 'malformed_offset'),
    ],
)
def test_the_lookup_carries_a_reason_for_the_run(
    tmp_path: Path, metadata: dict[str, object], reason: str
) -> None:
    """The caller is told why, in a form it can report and count.

    The prose belongs in the image's log; this is what a run-level line and a
    tally are built from.
    """
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, metadata)
    lookup, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert lookup.reason == reason


def test_a_missing_metadata_file_has_its_own_reason(tmp_path: Path) -> None:
    """Nothing navigated this image is distinct from navigating it badly."""
    lookup, _ = _load_and_capture(FCPath(tmp_path) / 'nav', FCPath(tmp_path) / 'logs')
    assert lookup.reason == 'no_metadata'


def test_a_loaded_offset_carries_no_reason(tmp_path: Path) -> None:
    """Success is not something to explain."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, {'status': 'success', 'offset': [1.5, -2.5]})
    lookup, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert lookup.reason is None


def test_asking_for_no_offset_is_not_a_missing_offset() -> None:
    """Reprojecting without navigation results is a choice, not a shortfall.

    Counting it as uncorrected pointing would report every image of a
    deliberately unnavigated run as an anomaly.
    """
    assert load_offset_if_any(None, _image_file()).reason is None
