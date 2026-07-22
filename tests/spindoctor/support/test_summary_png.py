"""Unit tests for the summary-PNG metadata block and per-box star stretch."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from spindoctor.support.summary_png import (
    SummaryMetadata,
    _apply_local_stretch_boxes,
    _draw_metadata_block,
    _extfov_box_to_data_slice,
    _least_crowded_corner,
    _load_summary_font,
    _local_stretch_patch,
    _text_overlap_area,
    _wrap_lines_to_width,
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


# ---------------------------------------------------------------------------
# _text_overlap_area
# ---------------------------------------------------------------------------


def test_text_overlap_area_counts_intersection() -> None:
    """Overlap is the intersection area of the block and a label box."""
    # Block rows/cols [0,10); label [5,20)x[5,20) -> 5x5 = 25 px^2 overlap.
    area = _text_overlap_area((0, 0, 10, 10), [(5, 5, 20, 20)])
    assert area == 25


def test_text_overlap_area_zero_when_disjoint() -> None:
    """A label box that misses the block contributes no overlap."""
    area = _text_overlap_area((0, 0, 10, 10), [(50, 50, 60, 60)])
    assert area == 0


# ---------------------------------------------------------------------------
# _least_crowded_corner
# ---------------------------------------------------------------------------


def test_least_crowded_corner_darkest_wins_without_text() -> None:
    """With no label boxes the darkest corner is chosen (brightness only)."""
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    rgb[76:96, 76:96] = 0  # bottom-right corner is darkest
    xy = _least_crowded_corner(rgb, 20, 20, text_bboxes=None)
    assert xy == (76, 76)


def test_least_crowded_corner_avoids_text_blocked_corner() -> None:
    """A corner whose block region overlaps a label is passed over."""
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    # A label covering the top-left corner region (v,u = rows,cols).
    text = [(0, 0, 30, 30)]
    xy = _least_crowded_corner(rgb, 20, 20, text_bboxes=text)
    assert xy != (4, 4)
    # The chosen corner must be free of the label box.
    assert _text_overlap_area((xy[1], xy[0], xy[1] + 20, xy[0] + 20), text) == 0


def test_least_crowded_corner_darker_free_corner_wins() -> None:
    """Among text-free corners the darker one wins on brightness."""
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    rgb[76:96, 76:96] = 0  # bottom-right is the darkest free corner
    text = [(0, 0, 30, 30)]  # blocks only the top-left corner
    xy = _least_crowded_corner(rgb, 20, 20, text_bboxes=text)
    assert xy == (76, 76)


def test_least_crowded_corner_all_conflict_picks_least_overlap() -> None:
    """When every corner conflicts, the least-overlapping corner wins."""
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Big label boxes fully cover three corners; a tiny one clips the fourth.
    text = [
        (0, 0, 24, 24),  # top-left block fully covered (400 px^2)
        (0, 76, 24, 100),  # top-right fully covered
        (76, 0, 100, 24),  # bottom-left fully covered
        (76, 76, 80, 80),  # bottom-right clipped only (4x4 = 16 px^2)
    ]
    xy = _least_crowded_corner(rgb, 20, 20, text_bboxes=text)
    assert xy == (76, 76)


# ---------------------------------------------------------------------------
# _wrap_lines_to_width
# ---------------------------------------------------------------------------


def test_wrap_lines_splits_line_wider_than_budget() -> None:
    """A line wider than the pixel budget is broken across several lines."""
    image = Image.new('RGB', (200, 40))
    draw = ImageDraw.Draw(image)
    font = _load_summary_font(14)
    source = 'aaaa bbbb cccc dddd eeee ffff'
    wrapped = _wrap_lines_to_width(draw, [source], font, 40)
    widths = [
        right - left
        for left, _top, right, _bottom in (
            draw.textbbox((0, 0), line, font=font) for line in wrapped
        )
    ]
    assert len(wrapped) > 1
    assert ' '.join(wrapped) == source
    assert max(widths) <= 40


def test_wrap_lines_keeps_narrow_line_intact() -> None:
    """A line already within the budget is returned as a single line."""
    image = Image.new('RGB', (200, 40))
    draw = ImageDraw.Draw(image)
    font = _load_summary_font(14)
    wrapped = _wrap_lines_to_width(draw, ['ab'], font, 400)
    assert wrapped == ['ab']


# ---------------------------------------------------------------------------
# _draw_metadata_block
# ---------------------------------------------------------------------------


def test_draw_metadata_block_renders_yellow_text() -> None:
    """On a roomy frame the block paints yellow text pixels."""
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    _draw_metadata_block(rgb, ['Hello'])
    yellow = (rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] < 80)
    assert bool(yellow.any())


def test_draw_metadata_block_skips_when_frame_too_small() -> None:
    """A frame too small to hold the block is left untouched."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    before = rgb.copy()
    _draw_metadata_block(rgb, ['A metadata line far too wide for this frame'])
    assert np.array_equal(rgb, before)


def test_draw_metadata_block_empty_lines_is_noop() -> None:
    """No lines means the image is not touched."""
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    before = rgb.copy()
    _draw_metadata_block(rgb, [])
    assert np.array_equal(rgb, before)
