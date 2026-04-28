"""Tests for ``nav.navigate_image_files.navigate_image_files``.

The driver wires the per-image batch loader together with the
orchestrator and the metadata curator.  These tests exercise the
happy / image-load-failure / status=failed paths against a fake
observation class so no holdings are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath

from nav.dataset.dataset import ImageFile, ImageFiles
from nav.navigate_image_files import navigate_image_files


class _FakeSnapshot:
    """Minimal stand-in for ObsSnapshotInst used by the driver tests."""

    def __init__(self, *, blank: bool = False, midtime: float = 100.0) -> None:
        rng = np.random.default_rng(seed=99)
        if blank:
            self.data = np.zeros((32, 32), np.float64)
        else:
            self.data = rng.standard_normal(size=(32, 32)) + 100.0
        self._sensor_mask = np.ones(self.data.shape, bool)
        self.midtime = midtime

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


class _FakeObsClass:
    """``obs_class`` shim whose ``from_file`` returns a controllable snapshot."""

    raise_on_load: BaseException | None = None
    blank: bool = False

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> _FakeSnapshot:
        if cls.raise_on_load is not None:
            raise cls.raise_on_load
        return _FakeSnapshot(blank=cls.blank)


def _make_image_files(tmp_path: Path) -> ImageFiles:
    """Build an ImageFiles batch with a single placeholder image."""
    img_path = tmp_path / 'fake_image.IMG'
    img_path.write_bytes(b'\x00')
    label_path = tmp_path / 'fake_image.LBL'
    label_path.write_bytes(b'\x00')
    return ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(label_path)),
                results_path_stub='fake_image',
            )
        ]
    )


def test_navigate_image_files_no_features_path(tmp_path: Path) -> None:
    """A clean image with no registered models yields a status='failed' result.

    The fake observation class returns an empty NavModel set (no
    real-scene models are registered), so the orchestrator reports
    ``NO_FEATURES_EXTRACTED``.
    """
    _FakeObsClass.raise_on_load = None
    _FakeObsClass.blank = False
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        _FakeObsClass,  # type: ignore[arg-type]
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status'] == 'failed'
    assert metadata['confidence'] == 0.0
    assert 'navigation_result' in metadata
    nav_result = metadata['navigation_result']
    # Without registered real-scene models the classifier still runs.
    assert nav_result['status_reason'] in (
        'no_features_extracted',
        'no_signal_in_image',
        'no_feasible_techniques',
    )


def test_navigate_image_files_writes_metadata(tmp_path: Path) -> None:
    """``write_output_files=True`` writes the metadata JSON to disk."""
    _FakeObsClass.raise_on_load = None
    _FakeObsClass.blank = False
    image_files = _make_image_files(tmp_path)
    results_root = tmp_path / 'results'
    results_root.mkdir(exist_ok=True)
    success, _metadata = navigate_image_files(
        _FakeObsClass,  # type: ignore[arg-type]
        image_files,
        FCPath(str(results_root)),
        write_output_files=True,
    )
    metadata_path = results_root / 'fake_image_metadata.json'
    assert metadata_path.exists()
    assert success is False  # no real-scene models registered


def test_navigate_image_files_blank_image_yields_no_signal(tmp_path: Path) -> None:
    """A blank image yields ``status_reason == 'no_signal_in_image'``."""
    _FakeObsClass.raise_on_load = None
    _FakeObsClass.blank = True
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        _FakeObsClass,  # type: ignore[arg-type]
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['navigation_result']['status_reason'] == 'no_signal_in_image'


def test_navigate_image_files_image_load_failure_records_status(tmp_path: Path) -> None:
    """An OSError during ``from_file`` records ``status='error'`` metadata."""
    _FakeObsClass.raise_on_load = OSError('cannot read fixture image')
    _FakeObsClass.blank = False
    image_files = _make_image_files(tmp_path)
    try:
        success, metadata = navigate_image_files(
            _FakeObsClass,  # type: ignore[arg-type]
            image_files,
            FCPath(str(tmp_path / 'results')),
            write_output_files=False,
        )
    finally:
        _FakeObsClass.raise_on_load = None
    assert success is False
    assert metadata['status'] == 'error'
    assert metadata['status_error'] == 'image_read_error'
    assert 'cannot read fixture image' in metadata['status_exception']


def test_navigate_image_files_spice_load_failure_records_missing_kernel(
    tmp_path: Path,
) -> None:
    """A SPICE-data error is classified as ``status_error='missing_spice_data'``."""
    _FakeObsClass.raise_on_load = RuntimeError('SPICE(SPKINSUFFDATA) coverage missing')
    _FakeObsClass.blank = False
    image_files = _make_image_files(tmp_path)
    try:
        success, metadata = navigate_image_files(
            _FakeObsClass,  # type: ignore[arg-type]
            image_files,
            FCPath(str(tmp_path / 'results')),
            write_output_files=False,
        )
    finally:
        _FakeObsClass.raise_on_load = None
    assert success is False
    assert metadata['status_error'] == 'missing_spice_data'


def test_navigate_image_files_rejects_multi_image_batch(tmp_path: Path) -> None:
    """A batch containing more than one image yields an error metadata block."""
    img_path = tmp_path / 'a.IMG'
    img_path.write_bytes(b'\x00')
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(img_path)),
                results_path_stub='a',
            ),
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(img_path)),
                results_path_stub='b',
            ),
        ]
    )
    success, metadata = navigate_image_files(
        _FakeObsClass,  # type: ignore[arg-type]
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status_error'] == 'expected_one_image_per_batch'
