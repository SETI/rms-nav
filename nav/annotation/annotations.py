import functools
import os
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .annotation import Annotation
from .annotation_text_info import (TEXTINFO_LEFT,
                                   TEXTINFO_LEFT_ARROW,
                                   TEXTINFO_RIGHT,
                                   TEXTINFO_RIGHT_ARROW,
                                   TEXTINFO_TOP,
                                   TEXTINFO_TOP_ARROW,
                                   TEXTINFO_BOTTOM,
                                   TEXTINFO_BOTTOM_ARROW,
                                   TEXTINFO_CENTER)

from nav.util.image import draw_line_arrow
from nav.util.types import NDArrayBoolType, NDArrayIntType


@functools.cache
def _load_font(path: str,
               size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class Annotations:
    def __init__(self) -> None:
        self._annotations: list[Annotation] = []

    def add_annotations(self,
                        annotations: Annotation | list[Annotation] | None
                        ) -> None:
        if annotations is None:
            return
        if isinstance(annotations, Annotations):
            ann_list = annotations._annotations
        elif not isinstance(annotations, list):
            ann_list = [annotations]
        else:
            ann_list = annotations
        for ann in ann_list:
            if len(self._annotations):
                if ann.overlay.shape != self._annotations[-1].overlay.shape:
                    raise ValueError(
                        'Annotation does not have same shape as previous: '
                        f'{ann.overlay.shape} vs '
                        f'{self._annotations[-1].overlay.shape}')
            self._annotations.append(ann)

    def combine(self,
                extfov: tuple[int, int],
                include_text: bool = True,
                offset: tuple[int, int] = (0, 0),
                text_use_avoid_mask: bool = True,
                text_avoid_other_text: bool = True,
                text_show_all_positions: bool = False
                ) -> NDArrayIntType | None:
        """Combine all annotations into a single graphic overlay."""

        if len(self._annotations) == 0:
            return None

        res = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        all_avoid_mask = np.zeros(self._annotations[0].overlay.shape, dtype=np.bool_)

        for annotation in self._annotations:
            res[annotation.overlay] = annotation._overlay_color
            if text_use_avoid_mask and annotation.avoid_mask is not None:
                all_avoid_mask |= annotation.avoid_mask

        if include_text:
            self._add_text(res, extfov, offset, all_avoid_mask,
                           text_avoid_other_text, text_show_all_positions)

        return res[extfov[0]+offset[0]:res.shape[0]-extfov[0]+offset[0],
                   extfov[1]+offset[1]:res.shape[1]-extfov[1]+offset[1]]

    def _add_text(self,
                  res: NDArrayIntType,
                  extfov: tuple[int, int],
                  offset: tuple[int, int],
                  avoid_mask: NDArrayBoolType,
                  text_avoid_other_text: bool,
                  text_show_all_positions: bool) -> None:
        """Add label text to an existing overlay."""

        text_layer = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        graphic_layer = np.zeros(self._annotations[0].overlay.shape + (3,),
                                 dtype=np.uint8)
        if text_avoid_other_text:
            ann_num_mask = np.zeros(self._annotations[0].overlay.shape, dtype=np.int_)
        else:
            ann_num_mask = None

        text_im = Image.frombuffer('RGB',
                                   (text_layer.shape[1], text_layer.shape[0]),
                                text_layer, 'raw', 'RGB', 0, 1)
        text_draw = ImageDraw.Draw(text_im)

        for ann_num, annotation in enumerate(self._annotations):
            # XXX ann_num is really not enough because we want a number for each
            # text_info
            if not annotation.text_info_list:
                continue

            tt_dir = cast(str, annotation.config.general.truetype_font_dir)

            # We first try to place the label avoiding the masked pixels, but if that
            # fails we go ahead and put the text in those places.
            for text_info in annotation.text_info_list:
                found_place = False
                for avoid in [True, False]:
                    ret = self._try_text_info(text_info, ann_num, extfov, offset,
                                              avoid_mask if avoid else None,
                                              text_layer, graphic_layer, ann_num_mask,
                                              text_draw, tt_dir,
                                              text_show_all_positions)
                    if ret:
                        found_place = True
                        break
                if not found_place:
                    annotation.config.logger.warning(
                        f'Could not find place for text annotation {text_info.text}')

        text_layer = (np.array(text_im.getdata()).astype('uint8').
                      reshape(text_layer.shape))
        text_layer[graphic_layer != 0] = graphic_layer[graphic_layer != 0]

        res[text_layer != 0] = text_layer[text_layer != 0]

    def _try_text_info(self,
                       text_info: list[tuple[str, int, int]],
                       ann_num: int,
                       extfov: tuple[int, int],
                       offset: tuple[int, int],
                       avoid_mask: NDArrayBoolType | None,
                       text_layer: NDArrayIntType,
                       graphic_layer: NDArrayIntType,
                       ann_num_mask: NDArrayIntType,
                       text_draw: ImageDraw.ImageDraw,
                       tt_dir: str,
                       show_all_positions: bool) -> bool:
        """Try to place the text for a text_info in a place that doesn't conflict."""

        font = _load_font(os.path.join(tt_dir, text_info.font), text_info.font_size)

        text_size = cast(tuple[int, int, int, int],
                         text_draw.textbbox((0, 0), text_info.text,
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

        # We take the offset into account when deciding where to put labels, so that
        # the final offset model, which will be cropped to the size of the original
        # image, doesn't have text running off the edges.
        v_margin_min = extfov[0] + offset[0] + edge_margin
        v_margin_max = (text_layer.shape[0] - extfov[0] + offset[0] -
                        edge_margin - 1)
        u_margin_min = extfov[1] + offset[1] + edge_margin
        u_margin_max = (text_layer.shape[1] - extfov[1] + offset[1] -
                        edge_margin - 1)

        # Run through the possible positions in order. For each, figure out where the
        # text (and optionally, arrow) goes. Then see if that location would conflict
        # with existing text or the avoid mask. If it doesn't conflict, put the text
        # (and optionally, arrow) and quit. This gives priority to the positions
        # earliest in the list.
        for text_pos, text_v, text_u in text_info.text_loc:
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
                draw_line_arrow(graphic_layer, text_info.color,
                                arrow_u0, arrow_v0, arrow_u1, arrow_v1,
                                thickness=arrow_thickness,
                                arrow_head_length=arrow_head_length,
                                arrow_head_angle=arrow_head_angle)

            text_draw.text((u - text_offset_u, v - text_offset_v), text_info.text,
                           anchor='la', fill=text_info.color, font=font)
            if ann_num_mask is not None:
                ann_num_mask[v0_margin:v1_margin, u0_margin:u1_margin] = ann_num+1

            if not show_all_positions:
                break
        else:
            if not show_all_positions:
                return False

        return True
