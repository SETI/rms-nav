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

from spindoctor.cli.reproj.offsets import load_offset_if_any
from spindoctor.config import IMAGE_LOGGER, LogLevels, LogSinks, build_image_log_handlers
from spindoctor.dataset.dataset import ImageFile

_STAMP = '2026-07-29T12-00-00'
_STUB = 'COISS_2001/N1234567890_1'


def _image_file() -> ImageFile:
    """Build an image file naming the stub the metadata is looked up under.

    Returns:
        The image file.
    """
    return ImageFile(
        image_file_url=FCPath('/nowhere/N1234567890_1.IMG'),
        label_file_url=FCPath('/nowhere/N1234567890_1.LBL'),
        results_path_stub=_STUB,
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


def _load_and_capture(nav_root: FCPath, log_root: FCPath) -> tuple[tuple[float, float] | None, str]:
    """Load the offset inside an image scope and return it with the image log.

    Parameters:
        nav_root: Root the metadata is read from.
        log_root: Root the per-image log is written under.

    Returns:
        Tuple of the loaded offset and the text of the image's log.
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
            offset = load_offset_if_any(nav_root, _image_file())
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        return offset, str(stream.read())


def test_a_usable_offset_is_returned(tmp_path: Path) -> None:
    """The stored offset is loaded when navigation succeeded."""
    nav_root = FCPath(tmp_path) / 'nav'
    _write_metadata(nav_root, {'status': 'success', 'offset': [1.5, -2.5]})
    offset, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert offset == (1.5, -2.5)


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
    offset, _ = _load_and_capture(nav_root, FCPath(tmp_path) / 'logs')
    assert offset is None


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
    assert load_offset_if_any(None, _image_file()) is None


def test_a_traversing_stub_is_refused(tmp_path: Path) -> None:
    """A stub resolving outside the results root does not load an offset."""
    escaping = ImageFile(
        image_file_url=FCPath('/nowhere/N1234567890_1.IMG'),
        label_file_url=FCPath('/nowhere/N1234567890_1.LBL'),
        results_path_stub='../../escaped/N1234567890_1',
        index_file_row={},
    )
    assert load_offset_if_any(FCPath(tmp_path), escaping) is None
