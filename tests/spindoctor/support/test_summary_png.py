"""Unit tests for the summary-PNG metadata block and per-box star stretch."""

from __future__ import annotations

import numpy as np

from spindoctor.support.summary_png import (
    SummaryMetadata,
    _apply_local_stretch_boxes,
    _extfov_box_to_data_slice,
    _local_stretch_patch,
    build_summary_metadata_lines,
)


class _FakeObs:
    """Minimal obs stand-in exposing the ext-FOV geometry the stretch needs."""

    def __init__(self, *, data_shape_vu: tuple[int, int], margin_vu: tuple[int, int]) -> None:
        self.data_shape_vu = data_shape_vu
        self.extfov_margin_v = margin_vu[0]
        self.extfov_margin_u = margin_vu[1]


class _FakeAnnotation:
    """Annotation stand-in carrying only ``stretch_boxes``."""

    def __init__(self, stretch_boxes: list[tuple[int, int, int, int]]) -> None:
        self.stretch_boxes = stretch_boxes


class _FakeAnnotations:
    """Annotations stand-in exposing an ``annotations`` list."""

    def __init__(self, annotations: list[_FakeAnnotation]) -> None:
        self.annotations = annotations


# ---------------------------------------------------------------------------
# build_summary_metadata_lines
# ---------------------------------------------------------------------------


def test_metadata_lines_success_lists_techniques() -> None:
    """A successful nav names its contributing techniques on the Nav line."""
    meta = SummaryMetadata(
        image_name='W1569489327_1_CALIB.IMG',
        filter_name='CL1+CL2',
        exposure_s=0.38,
        status='success',
        techniques=('RingEdgeNav', 'StarFieldFromCatalogNav'),
        confidence=0.87,
        confidence_rank='high',
    )
    lines = build_summary_metadata_lines(meta)
    assert lines[0] == 'W1569489327_1_CALIB.IMG'
    assert 'Nav: success [RingEdgeNav, StarFieldFromCatalogNav]' in lines


def test_metadata_lines_success_carries_filter_and_exposure() -> None:
    """Filter and exposure lines are present when the fields are set."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='CL1+CL2',
        exposure_s=0.38,
        status='success',
        techniques=('RingEdgeNav',),
        confidence=0.5,
        confidence_rank='medium',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Filter: CL1+CL2' in lines


def test_metadata_lines_success_carries_exposure_line() -> None:
    """The exposure value is rendered in seconds."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='CL1+CL2',
        exposure_s=0.38,
        status='success',
        techniques=('RingEdgeNav',),
        confidence=0.5,
        confidence_rank='medium',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Exposure: 0.38 s' in lines


def test_metadata_lines_confidence_value_and_tier() -> None:
    """The confidence line carries both the numeric value and the tier."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='',
        exposure_s=None,
        status='success',
        techniques=('RingEdgeNav',),
        confidence=0.873,
        confidence_rank='high',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Confidence: 0.873 (high)' in lines


def test_metadata_lines_omits_absent_filter() -> None:
    """No filter line is emitted for an instrument with no filter."""
    meta = SummaryMetadata(
        image_name='lorri.fit',
        filter_name='',
        exposure_s=1.0,
        status='success',
        techniques=('StarFieldFromCatalogNav',),
        confidence=0.4,
        confidence_rank='low',
    )
    lines = build_summary_metadata_lines(meta)
    assert not any(line.startswith('Filter:') for line in lines)


def test_metadata_lines_omits_absent_exposure() -> None:
    """No exposure line is emitted when the exposure is unknown."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='CL1',
        exposure_s=None,
        status='failed',
        techniques=(),
        confidence=0.0,
        confidence_rank='failed',
    )
    lines = build_summary_metadata_lines(meta)
    assert not any(line.startswith('Exposure:') for line in lines)


def test_metadata_lines_failed_reports_status_without_techniques() -> None:
    """A failed nav reports the bare status and never lists techniques."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='CL1',
        exposure_s=2.0,
        status='failed',
        techniques=(),
        confidence=0.0,
        confidence_rank='failed',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Nav: failed' in lines


def test_metadata_lines_failed_confidence_shown() -> None:
    """A failed nav still shows the (zero) confidence and failed tier."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='CL1',
        exposure_s=2.0,
        status='failed',
        techniques=(),
        confidence=0.0,
        confidence_rank='failed',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Confidence: 0.000 (failed)' in lines


def test_metadata_lines_confidence_na_when_none() -> None:
    """A missing confidence renders as ``n/a`` beside the tier."""
    meta = SummaryMetadata(
        image_name='x.IMG',
        filter_name='',
        exposure_s=None,
        status='conflicted',
        techniques=(),
        confidence=None,
        confidence_rank='conflicted',
    )
    lines = build_summary_metadata_lines(meta)
    assert 'Confidence: n/a (conflicted)' in lines


# ---------------------------------------------------------------------------
# _extfov_box_to_data_slice
# ---------------------------------------------------------------------------


def test_extfov_box_to_data_slice_zero_margin_zero_offset() -> None:
    """With no margin and no offset the box maps to itself."""
    obs = _FakeObs(data_shape_vu=(100, 100), margin_vu=(0, 0))
    box = (10, 20, 15, 26)
    assert _extfov_box_to_data_slice(obs, box, (0.0, 0.0)) == (10, 20, 15, 26)  # type: ignore[arg-type]


def test_extfov_box_to_data_slice_subtracts_margin() -> None:
    """A non-zero ext-FOV margin shifts the box back into FOV coordinates."""
    obs = _FakeObs(data_shape_vu=(100, 100), margin_vu=(5, 7))
    box = (10, 20, 15, 26)
    assert _extfov_box_to_data_slice(obs, box, (0.0, 0.0)) == (5, 13, 10, 19)  # type: ignore[arg-type]


def test_extfov_box_to_data_slice_applies_offset() -> None:
    """The overlay offset shifts the box by the same round-to-int convention."""
    obs = _FakeObs(data_shape_vu=(100, 100), margin_vu=(5, 5))
    box = (10, 10, 14, 14)
    # v0 = margin - round(offset); data = extfov - v0 = extfov - margin + round(offset)
    assert _extfov_box_to_data_slice(obs, box, (2.0, -3.0)) == (7, 2, 11, 6)  # type: ignore[arg-type]


def test_extfov_box_to_data_slice_clips_to_fov() -> None:
    """A box partly outside the FOV is clipped to the in-bounds region."""
    obs = _FakeObs(data_shape_vu=(12, 12), margin_vu=(0, 0))
    box = (-3, 8, 4, 20)
    assert _extfov_box_to_data_slice(obs, box, (0.0, 0.0)) == (0, 8, 4, 12)  # type: ignore[arg-type]


def test_extfov_box_to_data_slice_returns_none_when_outside() -> None:
    """A box entirely outside the FOV maps to None."""
    obs = _FakeObs(data_shape_vu=(12, 12), margin_vu=(0, 0))
    box = (20, 20, 25, 25)
    assert _extfov_box_to_data_slice(obs, box, (0.0, 0.0)) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _local_stretch_patch
# ---------------------------------------------------------------------------


def test_local_stretch_patch_reaches_full_range() -> None:
    """The stretched patch spans the full 0..255 range."""
    image = np.full((10, 10), 100.0)
    image[3:7, 3:7] = 100.0
    image[5, 5] = 101.0
    patch = _local_stretch_patch(image, (3, 3, 7, 7))
    assert patch is not None
    assert int(patch.max()) == 255


def test_local_stretch_patch_darkest_pixel_is_zero() -> None:
    """The box minimum maps to 0 after the stretch."""
    image = np.full((10, 10), 100.0)
    image[5, 5] = 140.0
    patch = _local_stretch_patch(image, (3, 3, 7, 7))
    assert patch is not None
    assert int(patch.min()) == 0


def test_local_stretch_patch_flat_box_returns_none() -> None:
    """A perfectly flat box has nothing to stretch and returns None."""
    image = np.full((10, 10), 50.0)
    assert _local_stretch_patch(image, (2, 2, 6, 6)) is None


def test_local_stretch_patch_all_nonfinite_returns_none() -> None:
    """A box with no finite pixels returns None."""
    image = np.full((10, 10), np.nan)
    assert _local_stretch_patch(image, (2, 2, 6, 6)) is None


# ---------------------------------------------------------------------------
# _apply_local_stretch_boxes
# ---------------------------------------------------------------------------


def test_apply_local_stretch_reveals_faint_star() -> None:
    """A faint star, invisible after a global stretch, brightens inside its box."""
    # Bright frame; the star box holds a dim peak just above a dark local floor.
    image = np.full((40, 40), 5000.0)
    image[10:17, 10:17] = 10.0  # dark patch (space around the star)
    image[13, 13] = 60.0  # the faint star peak
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)  # pretend global stretch left box dark
    obs = _FakeObs(data_shape_vu=(40, 40), margin_vu=(0, 0))
    annotations = _FakeAnnotations([_FakeAnnotation([(10, 10, 17, 17)])])
    _apply_local_stretch_boxes(rgb, image, obs, annotations, (0.0, 0.0))  # type: ignore[arg-type]
    # The star peak is the box maximum, so it saturates to 255 after the stretch.
    assert int(rgb[13, 13, 0]) == 255


def test_apply_local_stretch_leaves_outside_box_untouched() -> None:
    """Pixels outside every stretch box keep their pre-stretch value."""
    image = np.full((40, 40), 5000.0)
    image[10:17, 10:17] = 10.0
    image[13, 13] = 60.0
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    obs = _FakeObs(data_shape_vu=(40, 40), margin_vu=(0, 0))
    annotations = _FakeAnnotations([_FakeAnnotation([(10, 10, 17, 17)])])
    _apply_local_stretch_boxes(rgb, image, obs, annotations, (0.0, 0.0))  # type: ignore[arg-type]
    assert int(rgb[30, 30, 0]) == 0
