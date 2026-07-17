"""Tests for ``spindoctor.navigate_image_files.navigate_image_files``.

The driver wires the per-image batch loader together with the
orchestrator and the metadata curator.  These tests exercise the
happy / image-load-failure / status=failed paths against a fake
observation class so no holdings are required.

Also covers the annotation-compositing summary-PNG renderer
(``write_summary_png`` and the rendering helper now exposed via
``spindoctor.support.summary_png.grayscale_to_rgb_with_quantile_stretch``)
end-to-end against a synthetic ``Annotations`` collection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from filecache import FCPath
from PIL import Image

from spindoctor.annotation import Annotation, Annotations
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.navigate_image_files import (
    build_timing_section,
    navigate_image_files,
    write_summary_png,
)
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.summary_png import (
    grayscale_to_rgb_with_quantile_stretch as _grayscale_to_rgb_with_quantile_stretch,
)


class _FakeSnapshot:
    """Minimal stand-in for ObsSnapshotInst used by the driver tests."""

    def __init__(self, *, blank: bool = False, midtime: float = 100.0) -> None:
        rng = np.random.default_rng(seed=99)
        if blank:
            self.data = np.zeros((32, 32), np.float64)
        else:
            self.data = rng.standard_normal(size=(32, 32)) + 100.0
        # The orchestrator reads ``obs.extdata`` (extfov-shaped); this
        # fake uses zero extfov margin so ``extdata`` is the same shape
        # as ``data``.
        self.extdata = self.data
        self._sensor_mask = np.ones(self.data.shape, bool)
        self.midtime = midtime
        # Stands in for ObsInst.camera, written to observation.camera.
        self.camera = 'NAC'

    def extfov_data_sensor_mask(self) -> np.ndarray:
        return self._sensor_mask


def _make_fake_obs_class(
    *,
    blank: bool = False,
    raise_on_load: BaseException | None = None,
) -> type:
    """Build a fresh per-test ``obs_class`` shim with controllable behavior.

    Each call returns a brand-new class with the requested behavior baked in
    as class-level constants (read-only).  The previous design used a single
    shared `_FakeObsClass` whose mutable state raced under pytest-xdist.
    """
    captured_blank = blank
    captured_raise = raise_on_load

    class _FakeObsClass:
        @classmethod
        def from_file(cls, path: Any, **kwargs: Any) -> _FakeSnapshot:
            if captured_raise is not None:
                raise captured_raise
            return _FakeSnapshot(blank=captured_blank)

    return _FakeObsClass


def _make_image_files(
    tmp_path: Path, *, image_et: float | None = None, camera: str | None = None
) -> ImageFiles:
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
                image_et=image_et,
                camera=camera,
            )
        ]
    )


def test_navigate_image_files_no_features_path(tmp_path: Path) -> None:
    """A clean image with no registered models yields a status='failed' result.

    The fake observation class returns an empty NavModel set (no
    real-scene models are registered), so the orchestrator reports
    ``NO_FEATURES_EXTRACTED``.
    """
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status'] == 'failed'
    assert metadata['confidence'] == 0.0
    # The fake observation class is not in the instrument registry.
    assert metadata['observation']['instrument'] == 'unknown'
    assert metadata['observation']['image_shape'] == [32, 32]
    assert metadata['timing']['elapsed_s'] >= 0.0
    assert metadata['timing']['start_iso8601'].endswith('Z')
    assert metadata['timing']['end_iso8601'].endswith('Z')
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
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    results_root = tmp_path / 'results'
    success, _metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(results_root)),
        write_output_files=True,
    )
    metadata_path = results_root / 'fake_image_metadata.json'
    assert metadata_path.exists()
    assert success is False  # no real-scene models registered


def test_navigate_image_files_records_camera(tmp_path: Path) -> None:
    """The observation's camera is written to observation.camera."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['observation']['camera'] == 'NAC'


def test_navigate_image_files_load_error_records_index_epoch_and_camera(tmp_path: Path) -> None:
    """A SPICE-kernel failure keeps the epoch and camera the index supplied."""
    obs_class = _make_fake_obs_class(
        raise_on_load=RuntimeError('SPICE(NOFRAMECONNECT) -- insufficient information')
    )
    image_files = _make_image_files(tmp_path, image_et=-11784634.984, camera='WAC')
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['status_error'] == 'missing_spice_data'
    assert metadata['observation']['image_et'] == -11784634.984
    assert metadata['observation']['camera'] == 'WAC'


def test_navigate_image_files_load_error_without_index_details(tmp_path: Path) -> None:
    """An image with no index row records neither epoch nor camera."""
    obs_class = _make_fake_obs_class(raise_on_load=OSError('boom'))
    image_files = _make_image_files(tmp_path, image_et=None, camera=None)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert 'image_et' not in metadata['observation']
    assert 'camera' not in metadata['observation']


def test_navigate_image_files_prefers_observation_camera(tmp_path: Path) -> None:
    """On a successful load the observation's own camera wins over the index."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path, camera='WAC')
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    # The fake snapshot reports NAC; the index row here says WAC.
    assert metadata['observation']['camera'] == 'NAC'


def test_navigate_image_files_blank_image_yields_no_signal(tmp_path: Path) -> None:
    """A blank image yields ``status_reason == 'no_signal_in_image'``."""
    obs_class = _make_fake_obs_class(blank=True)
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['navigation_result']['status_reason'] == 'no_signal_in_image'


def test_navigate_image_files_image_load_failure_records_status(tmp_path: Path) -> None:
    """An OSError during ``from_file`` records ``status='error'`` metadata."""
    obs_class = _make_fake_obs_class(raise_on_load=OSError('cannot read fixture image'))
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status'] == 'error'
    assert metadata['status_error'] == 'image_read_error'
    assert 'cannot read fixture image' in metadata['status_exception']
    assert metadata['observation']['instrument'] == 'unknown'
    # The image never loaded, so no shape is recorded; timing still is.
    assert 'image_shape' not in metadata['observation']
    assert metadata['timing']['elapsed_s'] >= 0.0


def test_navigate_image_files_spice_load_failure_records_missing_kernel(
    tmp_path: Path,
) -> None:
    """A SPICE-data error is classified as ``status_error='missing_spice_data'``."""
    obs_class = _make_fake_obs_class(
        raise_on_load=RuntimeError('SPICE(SPKINSUFFDATA) coverage missing')
    )
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status_error'] == 'missing_spice_data'


def test_navigate_image_files_writes_summary_png(tmp_path: Path) -> None:
    """``write_output_files=True`` writes a non-empty summary PNG to disk."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    results_root = tmp_path / 'results'
    navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(results_root)),
        write_output_files=True,
    )
    png_path = results_root / 'fake_image_summary.png'
    assert png_path.exists()
    with Image.open(png_path) as img:
        assert img.mode == 'RGB'
        assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# _grayscale_to_rgb_with_quantile_stretch
# ---------------------------------------------------------------------------


def test_grayscale_to_rgb_stretch_shape_and_dtype() -> None:
    """Output is uint8 RGB with the input's spatial shape."""
    image = np.linspace(0.0, 1.0, num=64, dtype=np.float64).reshape(8, 8)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


def test_grayscale_to_rgb_stretch_handles_constant_image() -> None:
    """A constant image renders as an all-zero RGB field.

    The 0.001 / 0.999 quantiles collapse to the constant value, so the
    stretch helper bumps ``white`` to ``nextafter(black, inf)``; the
    subsequent ``(value - black) / (white - black)`` evaluates to 0
    everywhere and ``(0 * 255).astype(uint8)`` is uniformly zero.
    """
    image = np.full((4, 4), 5.0, dtype=np.float64)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rgb, expected)


def test_grayscale_to_rgb_stretch_treats_non_finite_as_zero() -> None:
    """NaN / inf samples are masked out before percentile selection."""
    image = np.array([[0.0, np.nan], [1.0, np.inf]], dtype=np.float64)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    assert rgb.shape == (2, 2, 3)
    # Non-finite pixels are remapped to zero before the stretch — so they
    # appear at or near the black end (0).
    assert int(rgb[0, 1, 0]) == 0


def test_grayscale_to_rgb_stretch_preserves_few_bright_pixels() -> None:
    """A handful of bright pixels keep their relative brightness ordering.

    A 1024 x 1024 dark-sky image with 8 bright stars at distinct
    intensities would clip every star to 255 under a fixed 0.001 / 0.999
    quantile stretch (0.1 % of 1 M = 1 048 pixels excluded as "white,"
    far more than the 8 brights).  The adaptive helper limits the clip
    count to half the bright-outlier count, so the brightest few
    saturate but the dimmer half keeps distinct gray values that
    preserve their brightness ordering.
    """
    rng = np.random.default_rng(seed=42)
    image = rng.normal(loc=0.0, scale=0.001, size=(1024, 1024)).astype(np.float64)
    bright_intensities = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    bright_positions = [
        (100, 100),
        (200, 200),
        (300, 300),
        (400, 400),
        (500, 500),
        (600, 600),
        (700, 700),
        (800, 800),
    ]
    for (v, u), intensity in zip(bright_positions, bright_intensities, strict=True):
        image[v, u] = intensity
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    bright_grays = [int(rgb[v, u, 0]) for v, u in bright_positions]
    # The two dimmest stars keep distinct, sub-saturation gray values
    # (under the old fixed-0.999 stretch they would both clip to 255).
    assert bright_grays[0] < bright_grays[1] < 255
    # Brightness ordering monotonically rises through every bright pixel.
    assert bright_grays == sorted(bright_grays)


def test_grayscale_to_rgb_stretch_keeps_default_clip_when_many_brights() -> None:
    """With many bright pixels the original 0.1 % clip is unchanged.

    A bright disc filling a quarter of the FOV has tens of thousands of
    bright pixels — far more than the 0.1 % default clip count.  The
    adaptive helper falls back to the default behavior in that regime so
    a body-fills-FOV scene's visualization is unchanged from before this
    fix.
    """
    image = np.zeros((512, 512), dtype=np.float64)
    # Bright disc: 256 x 256 = 65536 bright pixels (~25 % of the image,
    # vastly more than the 0.1 % default clip count of 262).
    image[128:384, 128:384] = 100.0
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    # Every disc pixel saturates to 255 because they are all the same
    # value and far above the 99.9 percentile of the full image.
    assert int(rgb[256, 256, 0]) == 255
    # Background is at the dark end.
    assert int(rgb[0, 0, 0]) == 0


# ---------------------------------------------------------------------------
# write_summary_png — direct fixture-driven exercise
# ---------------------------------------------------------------------------


class _FakeObsForRender:
    """Minimal stand-in for ObsSnapshot that satisfies the renderer + Annotations."""

    def __init__(self, *, image: np.ndarray) -> None:
        self.data = image
        self.extdata = image
        self._shape = image.shape
        self.data_shape_vu = image.shape
        self.extdata_shape_vu = image.shape
        self.extfov_margin_vu = (0, 0)
        self.extfov_margin_v = 0
        self.extfov_margin_u = 0

    def extract_offset_array(
        self, array: np.ndarray, _offset: tuple[float, float] | tuple[int, int] | None
    ) -> np.ndarray:
        # Zero extfov margin: the offset slice is the array itself.
        return array


def _make_render_result(
    *, annotations: Annotations, offset_px: tuple[float, float] | None = None
) -> NavResult:
    """Build a minimal NavResult that carries an annotation collection."""
    classifier = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )
    provenance = Provenance(
        spindoctor_version='0.0.0',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-28T00:00:00Z',
    )
    if offset_px is not None:
        return NavResult.success(
            offset_px=offset_px,
            covariance_px2=np.eye(2),
            confidence=0.5,
            confidence_rank='medium',
            status_reason=NavStatusReason.OK,
            per_technique=[],
            feature_inventory=[],
            image_classifier=classifier,
            provenance=provenance,
            annotations=annotations,
        )
    return NavResult.failed(
        status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
        image_classifier=classifier,
        provenance=provenance,
        annotations=annotations,
    )


def test_write_summary_png_image_only_when_no_annotations(tmp_path: Path) -> None:
    """With an empty annotation collection the PNG carries the source image only."""
    rng = np.random.default_rng(seed=11)
    image = rng.standard_normal((20, 24)) + 50.0
    obs = _FakeObsForRender(image=image)
    result = _make_render_result(annotations=Annotations())
    png_path = FCPath(str(tmp_path / 'out.png'))
    write_summary_png(obs, result, png_path, _CapturingLogger())  # type: ignore[arg-type]
    with Image.open(BytesIO(png_path.read_bytes())) as img:
        assert img.mode == 'RGB'
        assert img.size == (24, 20)


def test_write_summary_png_composites_overlay(tmp_path: Path) -> None:
    """A red overlay is visible at the overlay's True pixels in the PNG."""
    rng = np.random.default_rng(seed=12)
    image = rng.standard_normal((16, 16)) + 10.0
    obs = _FakeObsForRender(image=image)
    overlay_mask = np.zeros((16, 16), dtype=bool)
    overlay_mask[5:8, 5:8] = True
    annotation = Annotation(
        obs=obs,  # type: ignore[arg-type]
        overlay=overlay_mask,
        overlay_color=(255, 0, 0),
    )
    annotations = Annotations()
    annotations.add_annotations(annotation)
    result = _make_render_result(annotations=annotations, offset_px=(0.0, 0.0))
    png_path = FCPath(str(tmp_path / 'overlay.png'))
    write_summary_png(obs, result, png_path, _CapturingLogger())  # type: ignore[arg-type]
    with Image.open(BytesIO(png_path.read_bytes())) as raw:
        img = np.asarray(raw)
    inside = img[6, 6]
    outside = img[0, 0]
    assert inside[0] == 255
    assert inside[1] == 0
    assert inside[2] == 0
    assert outside[0] == outside[1]
    assert outside[1] == outside[2]


class _CapturingLogger:
    """Minimal stand-in for pdslogger that records info messages."""

    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, fmt: str, *args: Any) -> None:
        self.infos.append(fmt % args if args else fmt)


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
    obs_class = _make_fake_obs_class()
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status_error'] == 'expected_one_image_per_batch'
    # Even the validation early-return records the instrument and timing.
    assert metadata['observation']['instrument'] == 'unknown'
    assert metadata['timing']['elapsed_s'] >= 0.0


def test_build_timing_section_formats_utc() -> None:
    """The timing section carries UTC ISO8601 strings and float seconds."""
    start = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    end = datetime(2026, 7, 11, 12, 0, 2, 500000, tzinfo=UTC)
    timing = build_timing_section(start, end)
    assert timing['start_iso8601'] == '2026-07-11T12:00:00Z'
    assert timing['end_iso8601'] == '2026-07-11T12:00:02.500000Z'
    assert timing['elapsed_s'] == 2.5
