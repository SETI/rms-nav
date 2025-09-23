from collections import namedtuple
import functools
import os
from typing import cast

import numpy as np
from PIL import ImageDraw, ImageFont

from nav.config import DEFAULT_CONFIG
from nav.support.image import draw_line_arrow
from nav.support.types import NDArrayBoolType, NDArrayIntType


TEXTINFO_LEFT = 'left'
TEXTINFO_LEFT_ARROW = 'left_arrow'
TEXTINFO_RIGHT = 'right'
TEXTINFO_RIGHT_ARROW = 'right_arrow'
TEXTINFO_TOP = 'top'
TEXTINFO_TOP_ARROW = 'top_arrow'
TEXTINFO_BOTTOM = 'bottom'
TEXTINFO_BOTTOM_ARROW = 'bottom_arrow'
TEXTINFO_CENTER = 'center'

TextLocInfo = namedtuple('TextLocInfo', ['label', 'label_v', 'label_u'])


@functools.cache
def _load_font(path: str,
               size: int) -> ImageFont.FreeTypeFont:
    """Loads and caches a font for text rendering.

    Parameters:
        path: Path to the font file.
        size: Font size in points.

    Returns:
        A FreeTypeFont object for the specified font and size.
    """

    # TODO Add error handling
    return ImageFont.truetype(path, size)


class AnnotationTextInfo:
    def __init__(self,
                 text: str,
                 text_loc: list[TextLocInfo],
                 ref_vu: tuple[int, int] | None,
                 *,
                 color: tuple[int, ...],
                 font: str,
                 font_size: int):
        """Initializes text annotation information.

        Parameters:
            text: The text to display.
            text_loc: List of possible text locations with positioning information.
            ref_vu: Optional reference point (v, u) that the text is associated with.
            color: RGB or RGBA color tuple for the text.
            font: Font filename to use for rendering.
            font_size: Font size in points.
        """

        self._config = DEFAULT_CONFIG
        self._text = text
        self._text_loc = text_loc
        self._ref_vu = ref_vu
        self._color = tuple(color)
        self._font = font
        self._font_size = font_size

    @property
    def text(self) -> str:
        """Returns the annotation text."""
        return self._text

    @property
    def text_loc(self) -> list[TextLocInfo]:
        """Returns the list of possible text locations."""
        return self._text_loc

    @property
    def ref_vu(self) -> tuple[int, int] | None:
        """Returns the reference point (v, u) that the text is associated with."""
        return self._ref_vu

    @property
    def color(self) -> tuple[int, ...]:
        """Returns the RGB or RGBA color tuple for the text."""
        return self._color

    @property
    def font(self) -> str:
        """Returns the font filename used for rendering the text."""
        return self._font

    @property
    def font_size(self) -> int:
        """Returns the font size in points."""
        return self._font_size

    def _draw_text(self,
                   *,
                   ann_num: int,
                   extfov: tuple[int, int],
                   offset: tuple[int, int],
                   avoid_mask: NDArrayBoolType | None,
                   text_layer: NDArrayIntType,
                   graphic_layer: NDArrayIntType,
                   ann_num_mask: NDArrayIntType | None,
                   text_draw: ImageDraw.ImageDraw,
                   tt_dir: str,
                   show_all_positions: bool) -> bool:
        """Try to place the text in a location that doesn't conflict with other elements.

        Parameters:
            ann_num: Annotation number for identification.
            extfov: Extended field of view margins (v, u).
            offset: Offset to apply to coordinates (v, u).
            avoid_mask: Mask of areas to avoid when placing text.
            text_layer: Image layer for rendering text.
            graphic_layer: Image layer for rendering arrows.
            ann_num_mask: Mask tracking where annotations have been placed.
            text_draw: ImageDraw object for rendering text.
            tt_dir: Directory containing TrueType fonts.
            show_all_positions: Whether to try all positions or stop after finding the
                first valid one.

        Returns:
            True if the text was successfully placed, False otherwise.
        """

        if (self.ref_vu is not None and
            (self.ref_vu[0] - extfov[0] - offset[0] < 0 or
             self.ref_vu[0] - extfov[0] - offset[0] >= text_layer.shape[0] or
             self.ref_vu[1] - extfov[1] - offset[1] < 0 or
             self.ref_vu[1] - extfov[1] - offset[1] >= text_layer.shape[1])):
            # The thing we're labeling isn't in the FOV, so don't bother labeling it
            return True

        font = _load_font(os.path.join(tt_dir, self.font), self.font_size)

        text_size = cast(tuple[int, int, int, int],
                         text_draw.textbbox((0, 0), self.text,
                                            anchor='la', font=font))
        text_offset_u = text_size[0]
        text_offset_v = text_size[1]
        text_width_u = text_size[2] - text_size[0]
        text_width_v = text_size[3] - text_size[1]

        horiz_arrow_gap = 2
        horiz_arrow_len = 15
        vert_arrow_gap = 2
        vert_arrow_len = 15
        arrow_thickness = 1.5
        arrow_head_length = 6
        arrow_head_angle = 30
        edge_margin = 3  # Margin at edge of FOV
        text_margin = 2  # Margin around text and arrow so text doesn't get too close

        v_margin_min = edge_margin
        v_margin_max = text_layer.shape[0] - edge_margin - 1
        u_margin_min = edge_margin
        u_margin_max = text_layer.shape[1] - edge_margin - 1

        # Run through the possible positions in order. For each, figure out where the
        # text (and optionally, arrow) goes. Then see if that location would conflict
        # with existing text or the avoid mask. If it doesn't conflict, put the text
        # (and optionally, arrow) and quit. This gives priority to the positions
        # earliest in the list.
        for text_pos, text_v, text_u in self.text_loc:
            text_v = text_v - extfov[0] - offset[0]
            text_u = text_u - extfov[1] - offset[1]
            arrow_u0 = arrow_u1 = arrow_v0 = arrow_v1 = None
            if text_pos == TEXTINFO_LEFT:
                v = text_v - text_width_v // 2
                u = text_u - text_width_u
            elif text_pos == TEXTINFO_LEFT_ARROW:
                v = text_v - text_width_v // 2
                u = text_u - text_width_u - horiz_arrow_len - horiz_arrow_gap
                arrow_v0 = arrow_v1 = text_v
                arrow_u0 = text_u - horiz_arrow_len
                arrow_u1 = text_u
            elif text_pos == TEXTINFO_RIGHT:
                v = text_v - text_width_v // 2
                u = text_u
            elif text_pos == TEXTINFO_RIGHT_ARROW:
                v = text_v - text_width_v // 2
                u = text_u + horiz_arrow_len + horiz_arrow_gap
                arrow_v0 = arrow_v1 = text_v
                arrow_u0 = text_u + horiz_arrow_len
                arrow_u1 = text_u
            elif text_pos == TEXTINFO_TOP:
                v = text_v - text_width_v
                u = text_u - text_width_u // 2
            elif text_pos == TEXTINFO_TOP_ARROW:
                v = text_v - text_width_v - vert_arrow_len - vert_arrow_gap
                u = text_u - text_width_u // 2
                arrow_u0 = arrow_u1 = text_u
                arrow_v0 = text_v - vert_arrow_len
                arrow_v1 = text_v
            elif text_pos == TEXTINFO_BOTTOM:
                v = text_v
                u = text_u - text_width_u // 2
            elif text_pos == TEXTINFO_BOTTOM_ARROW:
                v = text_v + vert_arrow_len + vert_arrow_gap
                u = text_u - text_width_u // 2
                arrow_u0 = arrow_u1 = text_u
                arrow_v0 = text_v + vert_arrow_len
                arrow_v1 = text_v
            elif text_pos == TEXTINFO_CENTER:
                v = text_v - text_width_v // 2
                u = text_u - text_width_u // 2
            else:
                raise ValueError(f'Unknown text position: {text_pos}')

            # TODO This does not handle the case of the thing we're pointing at being off
            # the offset image while the text is still visible. For example,
            # inst_id = 'coiss'; URL = URL_CASSINI_ISS_STARS_02; offset = (-9, 28)

            # This mess with text_offset_u and text_offset_v is because textbbox and text
            # don't support the "lt" anchor for multi-line text, so we have to use "la",
            # which has additional left and top margin, and then subtract it off
            # ourselves.
            # See: https://github.com/python-pillow/Pillow/issues/5080
            v0_margin = v - text_offset_v - text_margin
            v1_margin = v - text_offset_v + text_width_v + text_margin
            u0_margin = u - text_offset_u - text_margin
            u1_margin = u - text_offset_u + text_width_u + text_margin
            if (v0_margin < v_margin_min or v1_margin > v_margin_max or
                u0_margin < u_margin_min or u1_margin > u_margin_max):
                # Text would run off edge
                continue

            if (avoid_mask is not None and
                np.any(avoid_mask[v0_margin:v1_margin, u0_margin:u1_margin])):
                # Conflicts with something the program doesn't want us to overwrite
                continue
            if (ann_num_mask is not None and
                np.any(ann_num_mask[v0_margin:v1_margin, u0_margin:u1_margin])):
                # Conflicts with text or arrows we've already drawn
                continue

            if arrow_u0 is not None:
                assert arrow_u1 is not None
                assert arrow_v0 is not None
                assert arrow_v1 is not None
                # Calculate head width from angle and add a little margin of error
                head_width = int(np.ceil(
                    arrow_head_length * np.sin(np.deg2rad(arrow_head_angle)) * 2)) + 2
                if arrow_v0 == arrow_v1:  # Horizontal arrow
                    arrow_box_v0 = arrow_v0 - head_width // 2
                    arrow_box_v1 = arrow_v0 + head_width // 2
                    arrow_box_u0 = arrow_u0
                    arrow_box_u1 = arrow_u1
                else:  # Vertical arrow
                    arrow_box_u0 = arrow_u0 - head_width // 2
                    arrow_box_u1 = arrow_u0 + head_width // 2
                    arrow_box_v0 = arrow_v0
                    arrow_box_v1 = arrow_v1
                arrow_box_v0, arrow_box_v1 = (min(arrow_box_v0, arrow_box_v1),
                                              max(arrow_box_v0, arrow_box_v1))
                arrow_box_u0, arrow_box_u1 = (min(arrow_box_u0, arrow_box_u1),
                                              max(arrow_box_u0, arrow_box_u1))

                if (not v_margin_min <= arrow_box_v0 <= v_margin_max or
                    not v_margin_min <= arrow_box_v1 <= v_margin_max or
                    not u_margin_min <= arrow_box_u0 <= u_margin_max or
                    not u_margin_min <= arrow_box_u1 <= u_margin_max):
                    # Arrow would run off edge
                    continue

                if ann_num_mask is not None:
                    if np.any(ann_num_mask[arrow_box_v0:arrow_box_v1+1,
                                           arrow_box_u0:arrow_box_u1+1]):
                        # Conflicts with text or arrows we've already drawn
                        continue
                    ann_num_mask[arrow_box_v0:arrow_box_v1+1,
                                 arrow_box_u0:arrow_box_u1+1] = ann_num+1
                draw_line_arrow(graphic_layer, self.color,
                                arrow_u0, arrow_v0, arrow_u1, arrow_v1,
                                thickness=arrow_thickness,
                                arrow_head_length=arrow_head_length,
                                arrow_head_angle=arrow_head_angle)

            text_draw.text((u - text_offset_u, v - text_offset_v), self.text,
                           anchor='la', fill=self.color, font=font)
            if ann_num_mask is not None:
                ann_num_mask[v0_margin:v1_margin, u0_margin:u1_margin] = ann_num+1

            if not show_all_positions:
                break
        else:
            if not show_all_positions:
                return False

        return True
