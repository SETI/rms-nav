"""Tests for :mod:`spindoctor.annotation.annotation_text_info`.

Covers the ``AnnotationTextInfo`` accessors and string form, the cached
``_load_font`` helper, and the ``_draw_text`` placement contract: each of the
twelve anchor / arrow position constants places the text on the documented
side of the reference point, earlier locations in ``text_loc`` win, the
avoid mask and the shared annotation-number mask veto conflicting spots, and
text that would run off the image edge is skipped without drawing or
crashing.

The tests render into small in-memory RGB canvases and assert pixel-level
facts (ink present inside the expected bounding box, nothing outside it)
instead of comparing golden images. Fonts come from matplotlib's bundled
DejaVuSans.ttf (matplotlib is a required dependency), so no system font,
network, or SPICE resource is touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from spindoctor.annotation.annotation_text_info import (
    TEXTINFO_BOTTOM,
    TEXTINFO_BOTTOM_ARROW,
    TEXTINFO_BOTTOM_LEFT,
    TEXTINFO_BOTTOM_RIGHT,
    TEXTINFO_CENTER,
    TEXTINFO_LEFT,
    TEXTINFO_LEFT_ARROW,
    TEXTINFO_RIGHT,
    TEXTINFO_RIGHT_ARROW,
    TEXTINFO_TOP,
    TEXTINFO_TOP_ARROW,
    TEXTINFO_TOP_LEFT,
    TEXTINFO_TOP_RIGHT,
    AnnotationTextInfo,
    TextLocInfo,
    _load_font,
)
from spindoctor.support.types import NDArrayBoolType, NDArrayIntType

FONT_DIR = str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf')
FONT_NAME = 'DejaVuSans.ttf'

# Pixel tolerance for ink-inside-bounding-box assertions.
TOL = 1

# Arrow geometry from the placement contract: a 15-pixel leader plus a
# 2-pixel gap separates the text from the reference point.
ARROW_SPAN = 17


def _make_info(
    text: str = 'Hello',
    *,
    locs: list[TextLocInfo],
    ref_vu: tuple[int, int] | None = (50, 50),
    color: tuple[int, ...] = (255, 0, 0),
    font_size: int = 10,
) -> AnnotationTextInfo:
    """Build an ``AnnotationTextInfo`` with test-friendly defaults.

    Parameters:
        text: The annotation text.
        locs: Candidate text locations in priority order.
        ref_vu: Reference point (v, u) or None for an unanchored label.
        color: RGB color tuple for the text.
        font_size: Font size in points.

    Returns:
        The configured ``AnnotationTextInfo`` instance.
    """

    return AnnotationTextInfo(text, locs, ref_vu, color=color, font=FONT_NAME, font_size=font_size)


class _Canvas:
    """A small in-memory render target mirroring ``Annotations._add_text``.

    Parameters:
        shape: The (v, u) canvas shape in pixels.
        track_text: Whether to allocate the annotation-number mask that makes
            successive placements avoid each other.
    """

    def __init__(self, shape: tuple[int, int] = (100, 100), *, track_text: bool = True) -> None:
        self.shape = shape
        self.text_layer: NDArrayIntType = np.zeros((*shape, 3), dtype=np.uint8)
        self.graphic_layer: NDArrayIntType = np.zeros((*shape, 3), dtype=np.uint8)
        self.ann_num_mask: NDArrayIntType | None = (
            np.zeros((*shape, 3), dtype=np.int64) if track_text else None
        )
        self._image = Image.fromarray(self.text_layer, mode='RGB')
        self._draw = ImageDraw.Draw(self._image)

    def place(
        self,
        text_info: AnnotationTextInfo,
        *,
        ann_num: int = 0,
        extfov: tuple[int, int] = (0, 0),
        offset: tuple[float, float] = (0.0, 0.0),
        avoid_mask: NDArrayBoolType | None = None,
        show_all_positions: bool = False,
    ) -> bool:
        """Run ``_draw_text`` against this canvas.

        Parameters:
            text_info: The annotation text to place.
            ann_num: Annotation number recorded in the placement mask.
            extfov: Extended field-of-view margins (v, u).
            offset: Navigation offset (dv, du) applied to coordinates.
            avoid_mask: Optional mask of pixels the text must not cover.
            show_all_positions: Whether to draw every valid candidate location.

        Returns:
            The boolean result of ``_draw_text``.
        """

        return text_info._draw_text(
            ann_num=ann_num,
            extfov=extfov,
            offset=offset,
            avoid_mask=avoid_mask,
            text_layer=self.text_layer,
            graphic_layer=self.graphic_layer,
            ann_num_mask=self.ann_num_mask,
            text_draw=self._draw,
            tt_dir=FONT_DIR,
            show_all_positions=show_all_positions,
        )

    @property
    def text_pixels(self) -> NDArrayIntType:
        """The rendered text layer as a (v, u, 3) uint8 array."""

        return np.array(self._image, dtype=np.uint8)


def _ink_rows_cols(pixels: NDArrayIntType) -> tuple[NDArrayIntType, NDArrayIntType]:
    """Return the row and column indices of every non-black pixel.

    Parameters:
        pixels: A (v, u, 3) image array.

    Returns:
        Two arrays holding the v and u indices of pixels with any nonzero
        channel.
    """

    rows, cols = np.nonzero(pixels.max(axis=2))
    return rows, cols


def _text_extent(text: str, font_size: int) -> tuple[int, int]:
    """Measure the rendered extent of ``text`` with the test font.

    Parameters:
        text: The text to measure.
        font_size: Font size in points.

    Returns:
        The (height, width) of the ink bounding box in pixels.
    """

    font = ImageFont.truetype(os.path.join(FONT_DIR, FONT_NAME), font_size)
    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, anchor='la', font=font)
    return int(y1 - y0), int(x1 - x0)


def _expected_text_box(
    position: str, anchor_v: int, anchor_u: int, height: int, width: int
) -> tuple[int, int, int, int]:
    """Compute the contractual ink bounding box for a placement constant.

    Encodes the documented meaning of each anchor: ``left`` puts the text to
    the left of the reference point (vertically centered), ``top_right``
    above and to the right, ``center`` centered on it, and the ``*_arrow``
    variants push the text a further leader-plus-gap span away.

    Parameters:
        position: One of the twelve TEXTINFO placement constants.
        anchor_v: The v coordinate the location entry names.
        anchor_u: The u coordinate the location entry names.
        height: Rendered text height in pixels.
        width: Rendered text width in pixels.

    Returns:
        Inclusive (v_min, v_max, u_min, u_max) bounds for the text ink.
    """

    half_v = height // 2
    half_u = width // 2
    if position == TEXTINFO_LEFT:
        return anchor_v - half_v, anchor_v - half_v + height, anchor_u - width, anchor_u
    if position == TEXTINFO_LEFT_ARROW:
        return (
            anchor_v - half_v,
            anchor_v - half_v + height,
            anchor_u - width - ARROW_SPAN,
            anchor_u - ARROW_SPAN,
        )
    if position == TEXTINFO_RIGHT:
        return anchor_v - half_v, anchor_v - half_v + height, anchor_u, anchor_u + width
    if position == TEXTINFO_RIGHT_ARROW:
        return (
            anchor_v - half_v,
            anchor_v - half_v + height,
            anchor_u + ARROW_SPAN,
            anchor_u + width + ARROW_SPAN,
        )
    if position == TEXTINFO_TOP:
        return anchor_v - height, anchor_v, anchor_u - half_u, anchor_u - half_u + width
    if position == TEXTINFO_TOP_ARROW:
        return (
            anchor_v - height - ARROW_SPAN,
            anchor_v - ARROW_SPAN,
            anchor_u - half_u,
            anchor_u - half_u + width,
        )
    if position == TEXTINFO_BOTTOM:
        return anchor_v, anchor_v + height, anchor_u - half_u, anchor_u - half_u + width
    if position == TEXTINFO_BOTTOM_ARROW:
        return (
            anchor_v + ARROW_SPAN,
            anchor_v + height + ARROW_SPAN,
            anchor_u - half_u,
            anchor_u - half_u + width,
        )
    if position == TEXTINFO_CENTER:
        return (
            anchor_v - half_v,
            anchor_v - half_v + height,
            anchor_u - half_u,
            anchor_u - half_u + width,
        )
    if position == TEXTINFO_TOP_LEFT:
        return anchor_v - height, anchor_v, anchor_u - width, anchor_u
    if position == TEXTINFO_TOP_RIGHT:
        return anchor_v - height, anchor_v, anchor_u, anchor_u + width
    if position == TEXTINFO_BOTTOM_LEFT:
        return anchor_v, anchor_v + height, anchor_u - width, anchor_u
    if position == TEXTINFO_BOTTOM_RIGHT:
        return anchor_v, anchor_v + height, anchor_u, anchor_u + width
    raise ValueError(f'Unexpected test position {position!r}')


def _assert_ink_within(pixels: NDArrayIntType, box: tuple[int, int, int, int]) -> None:
    """Assert that ink exists and every ink pixel lies inside ``box``.

    Parameters:
        pixels: A (v, u, 3) image array.
        box: Inclusive (v_min, v_max, u_min, u_max) bounds, expanded by the
            module tolerance before comparison.
    """

    rows, cols = _ink_rows_cols(pixels)
    assert rows.size > 0
    assert int(rows.min()) >= box[0] - TOL
    assert int(rows.max()) <= box[1] + TOL
    assert int(cols.min()) >= box[2] - TOL
    assert int(cols.max()) <= box[3] + TOL


# ---------------------------------------------------------------------------
# Accessors, string form, TextLocInfo
# ---------------------------------------------------------------------------


def test_properties_return_constructor_values() -> None:
    """Each documented property echoes the constructor argument."""

    locs = [TextLocInfo(TEXTINFO_CENTER, 50, 50)]
    info = _make_info('Mimas', locs=locs, ref_vu=(12, 34), font_size=14)
    assert info.text == 'Mimas'
    assert info.text_loc == locs
    assert info.ref_vu == (12, 34)
    assert info.color == (255, 0, 0)
    assert info.font == FONT_NAME
    assert info.font_size == 14


def test_rgba_color_roundtrips() -> None:
    """A four-component RGBA color is stored and returned unchanged."""

    info = AnnotationTextInfo(
        'x',
        [TextLocInfo(TEXTINFO_CENTER, 50, 50)],
        None,
        color=(1, 2, 3, 4),
        font=FONT_NAME,
        font_size=10,
    )
    assert info.color == (1, 2, 3, 4)


def test_str_summarizes_text_and_placement() -> None:
    """The string form names the class, the text, and the reference point."""

    info = _make_info('Enceladus', locs=[TextLocInfo(TEXTINFO_CENTER, 5, 6)], ref_vu=(5, 6))
    rendered = str(info)
    assert 'AnnotationTextInfo' in rendered
    assert 'Enceladus' in rendered
    assert 'Ref vu: (5, 6)' in rendered


def test_str_truncates_after_ten_locations() -> None:
    """More than ten candidate locations are elided with an ellipsis."""

    locs = [TextLocInfo(TEXTINFO_CENTER, v, v) for v in range(11)]
    info = _make_info(locs=locs)
    assert str(info).endswith('...')


def test_str_short_location_list_is_not_truncated() -> None:
    """Ten or fewer candidate locations are all shown."""

    locs = [TextLocInfo(TEXTINFO_CENTER, v, v) for v in range(10)]
    info = _make_info(locs=locs)
    assert '...' not in str(info)


def test_repr_matches_str() -> None:
    """``__repr__`` delegates to ``__str__``."""

    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)])
    assert repr(info) == str(info)


def test_text_loc_info_field_names() -> None:
    """``TextLocInfo`` carries a placement label and its (v, u) coordinates."""

    assert TextLocInfo._fields == ('label', 'label_v', 'label_u')


# ---------------------------------------------------------------------------
# _load_font
# ---------------------------------------------------------------------------


def test_load_font_returns_freetype_font() -> None:
    """A valid path and size yield a PIL FreeTypeFont."""

    font = _load_font(os.path.join(FONT_DIR, FONT_NAME), 10)
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_load_font_caches_identical_arguments() -> None:
    """Repeated loads of the same path and size return the cached object."""

    first = _load_font(os.path.join(FONT_DIR, FONT_NAME), 11)
    second = _load_font(os.path.join(FONT_DIR, FONT_NAME), 11)
    assert first is second


def test_load_font_distinct_sizes_are_not_shared() -> None:
    """The cache is keyed on size as well as path."""

    small = _load_font(os.path.join(FONT_DIR, FONT_NAME), 9)
    large = _load_font(os.path.join(FONT_DIR, FONT_NAME), 21)
    assert small is not large


def test_load_font_missing_file_raises_with_config_hint() -> None:
    """A bad path raises FileNotFoundError naming the path and the config key."""

    bad_path = os.path.join(FONT_DIR, 'NoSuchFont.ttf')
    with pytest.raises(FileNotFoundError, match=r'NoSuchFont\.ttf.*truetype_font_dir'):
        _load_font(bad_path, 12)


# ---------------------------------------------------------------------------
# _draw_text: anchor placement contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'position',
    [
        TEXTINFO_LEFT,
        TEXTINFO_RIGHT,
        TEXTINFO_TOP,
        TEXTINFO_BOTTOM,
        TEXTINFO_CENTER,
        TEXTINFO_TOP_LEFT,
        TEXTINFO_TOP_RIGHT,
        TEXTINFO_BOTTOM_LEFT,
        TEXTINFO_BOTTOM_RIGHT,
    ],
)
def test_plain_positions_place_ink_on_documented_side(position: str) -> None:
    """Each non-arrow constant places all ink in the documented region.

    Parameters:
        position: The TEXTINFO placement constant under test.
    """

    height, width = _text_extent('Hello', 10)
    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(position, 50, 50)])
    placed = canvas.place(info)
    assert placed is True
    _assert_ink_within(canvas.text_pixels, _expected_text_box(position, 50, 50, height, width))


@pytest.mark.parametrize(
    'position',
    [TEXTINFO_LEFT_ARROW, TEXTINFO_RIGHT_ARROW, TEXTINFO_TOP_ARROW, TEXTINFO_BOTTOM_ARROW],
)
def test_arrow_positions_offset_text_and_draw_leader(position: str) -> None:
    """Arrow constants push the text out and draw a leader in the graphic layer.

    The text sits a leader-plus-gap span beyond the plain anchor position and
    the arrow ink stays inside the corridor between the text and the
    reference point.

    Parameters:
        position: The TEXTINFO arrow placement constant under test.
    """

    height, width = _text_extent('Hello', 10)
    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(position, 50, 50)])
    placed = canvas.place(info)
    assert placed is True
    _assert_ink_within(canvas.text_pixels, _expected_text_box(position, 50, 50, height, width))

    arrow_rows, arrow_cols = _ink_rows_cols(canvas.graphic_layer)
    assert arrow_rows.size > 0
    if position in (TEXTINFO_LEFT_ARROW, TEXTINFO_RIGHT_ARROW):
        assert int(arrow_rows.min()) >= 50 - 7
        assert int(arrow_rows.max()) <= 50 + 7
        if position == TEXTINFO_LEFT_ARROW:
            assert int(arrow_cols.min()) >= 50 - ARROW_SPAN
            assert int(arrow_cols.max()) <= 50 + 2
        else:
            assert int(arrow_cols.min()) >= 50 - 2
            assert int(arrow_cols.max()) <= 50 + ARROW_SPAN
    else:
        assert int(arrow_cols.min()) >= 50 - 7
        assert int(arrow_cols.max()) <= 50 + 7
        if position == TEXTINFO_TOP_ARROW:
            assert int(arrow_rows.min()) >= 50 - ARROW_SPAN
            assert int(arrow_rows.max()) <= 50 + 2
        else:
            assert int(arrow_rows.min()) >= 50 - 2
            assert int(arrow_rows.max()) <= 50 + ARROW_SPAN


def test_ink_uses_requested_color_channels() -> None:
    """Pure red text leaves the green and blue channels untouched."""

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)], color=(255, 0, 0))
    canvas.place(info)
    pixels = canvas.text_pixels
    assert int(pixels[:, :, 0].max()) > 0
    assert int(pixels[:, :, 1].max()) == 0
    assert int(pixels[:, :, 2].max()) == 0


def test_offset_and_extfov_shift_the_anchor() -> None:
    """Location coordinates are extended-FOV values shifted by offset - extfov.

    With extfov margins (10, 10) and a navigation offset of (5, -3), an
    anchor at extended coordinates (60, 60) lands at display coordinates
    (55, 47).

    """

    height, width = _text_extent('Hello', 10)
    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 60, 60)], ref_vu=(60, 60))
    placed = canvas.place(info, extfov=(10, 10), offset=(5.0, -3.0))
    assert placed is True
    box = _expected_text_box(TEXTINFO_CENTER, 55, 47, height, width)
    _assert_ink_within(canvas.text_pixels, box)


def test_ref_vu_none_places_normally() -> None:
    """An unanchored label (ref_vu None) skips the FOV check and still places."""

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)], ref_vu=None)
    placed = canvas.place(info)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size > 0


# ---------------------------------------------------------------------------
# _draw_text: priority, collision avoidance
# ---------------------------------------------------------------------------


def test_first_valid_location_wins() -> None:
    """Locations earlier in text_loc take priority; only the first is drawn."""

    canvas = _Canvas()
    info = _make_info(
        locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30), TextLocInfo(TEXTINFO_CENTER, 70, 70)],
        ref_vu=(30, 30),
    )
    placed = canvas.place(info)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert int(rows.max()) < 50


def test_show_all_positions_draws_every_valid_location() -> None:
    """show_all_positions renders the text at each candidate that fits."""

    canvas = _Canvas()
    info = _make_info(
        locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30), TextLocInfo(TEXTINFO_CENTER, 70, 70)],
        ref_vu=(30, 30),
    )
    placed = canvas.place(info, show_all_positions=True)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert int(rows.min()) < 50
    assert int(rows.max()) > 50


def test_avoid_mask_forces_fallback_location() -> None:
    """A masked-off first choice pushes the text to the next candidate."""

    canvas = _Canvas()
    avoid = np.zeros(canvas.shape, dtype=bool)
    avoid[10:50, 10:60] = True
    info = _make_info(
        locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30), TextLocInfo(TEXTINFO_CENTER, 70, 70)],
        ref_vu=(30, 30),
    )
    placed = canvas.place(info, avoid_mask=avoid)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert int(rows.min()) > 50


def test_avoid_mask_everywhere_returns_false() -> None:
    """With every candidate masked off the text is not placed at all."""

    canvas = _Canvas()
    avoid = np.ones(canvas.shape, dtype=bool)
    info = _make_info(
        locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30), TextLocInfo(TEXTINFO_CENTER, 70, 70)],
        ref_vu=(30, 30),
    )
    placed = canvas.place(info, avoid_mask=avoid)
    assert placed is False
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size == 0


def test_ann_num_mask_records_placement() -> None:
    """A successful placement stamps ann_num + 1 into the shared mask."""

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)])
    canvas.place(info, ann_num=3)
    assert canvas.ann_num_mask is not None
    assert int(canvas.ann_num_mask.max()) == 4


def test_second_annotation_avoids_first() -> None:
    """Text already stamped into the mask pushes later text to its fallback."""

    canvas = _Canvas()
    first = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30)], ref_vu=(30, 30))
    placed_first = canvas.place(first, ann_num=0)
    assert placed_first is True
    before = canvas.text_pixels

    second = _make_info(
        locs=[TextLocInfo(TEXTINFO_CENTER, 30, 30), TextLocInfo(TEXTINFO_CENTER, 70, 70)],
        ref_vu=(30, 30),
    )
    placed_second = canvas.place(second, ann_num=1)
    assert placed_second is True
    new_ink = canvas.text_pixels.astype(np.int64) - before.astype(np.int64)
    rows, _ = np.nonzero(new_ink.max(axis=2) > 0)
    assert rows.size > 0
    assert int(rows.min()) > 50


# ---------------------------------------------------------------------------
# _draw_text: edges and degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'anchor',
    [(1, 1), (1, 50), (1, 98), (50, 1), (50, 98), (98, 1), (98, 50), (98, 98)],
)
def test_anchor_at_image_edge_fails_cleanly(anchor: tuple[int, int]) -> None:
    """Text that would cross any image edge is skipped without drawing.

    Parameters:
        anchor: The (v, u) anchor point near an edge or corner of the canvas.
    """

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, anchor[0], anchor[1])], ref_vu=anchor)
    placed = canvas.place(info)
    assert placed is False
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size == 0


@pytest.mark.parametrize('ref_vu', [(150, 50), (50, 150)])
def test_ref_point_outside_fov_skips_without_drawing(ref_vu: tuple[int, int]) -> None:
    """A reference point outside the layer is not labeled but counts as done.

    ``_draw_text`` documents returning True only on successful placement, but
    an off-FOV reference is deliberately treated as 'nothing to label' (the
    in-code comment) so the caller does not warn about it.

    Parameters:
        ref_vu: The out-of-bounds reference point.
    """

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)], ref_vu=ref_vu)
    placed = canvas.place(info)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size == 0


def test_empty_text_places_without_ink() -> None:
    """An empty string is accepted and renders no pixels."""

    canvas = _Canvas()
    info = _make_info('', locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)])
    placed = canvas.place(info)
    assert placed is True
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size == 0


def test_multiline_text_renders_taller_than_single_line() -> None:
    """A two-line label occupies a taller ink span than its first line alone."""

    single = _Canvas()
    single.place(_make_info('AB', locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)], font_size=12))
    single_rows, _ = _ink_rows_cols(single.text_pixels)

    multi = _Canvas()
    placed = multi.place(
        _make_info('AB\nCD', locs=[TextLocInfo(TEXTINFO_CENTER, 50, 50)], font_size=12)
    )
    assert placed is True
    multi_rows, _ = _ink_rows_cols(multi.text_pixels)
    single_span = int(single_rows.max()) - int(single_rows.min())
    multi_span = int(multi_rows.max()) - int(multi_rows.min())
    assert multi_span > single_span


def test_unknown_position_label_raises() -> None:
    """A location entry with an unrecognized label raises ValueError."""

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo('diagonal', 50, 50)])
    with pytest.raises(ValueError, match='Unknown text position: diagonal'):
        canvas.place(info)


def test_empty_location_list_returns_false() -> None:
    """With no candidate locations at all the text cannot be placed."""

    canvas = _Canvas()
    info = _make_info(locs=[])
    placed = canvas.place(info)
    assert placed is False


def test_show_all_positions_reports_failure_when_nothing_fits() -> None:
    """Per the Returns contract, no successful placement must yield False.

    The docstring says the method returns 'True if the text was successfully
    placed, False otherwise', but with show_all_positions=True the for-else
    fallthrough skips the failure return and reports True even though nothing
    was drawn.

    """

    canvas = _Canvas()
    info = _make_info(locs=[TextLocInfo(TEXTINFO_CENTER, 1, 1)], ref_vu=None)
    placed = canvas.place(info, show_all_positions=True)
    rows, _ = _ink_rows_cols(canvas.text_pixels)
    assert rows.size == 0
    assert placed is False
