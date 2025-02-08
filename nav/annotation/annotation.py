import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nav.config import Config, DEFAULT_CONFIG
from nav.util.image import shift_array, draw_line_arrow
from nav.util.types import NDArrayBoolType, NDArrayIntType

from .annotation_text_info import (AnnotationTextInfo,
                                   TEXTINFO_LEFT,
                                   TEXTINFO_LEFT_ARROW,
                                   TEXTINFO_RIGHT,
                                   TEXTINFO_RIGHT_ARROW,
                                   TEXTINFO_TOP,
                                   TEXTINFO_TOP_ARROW,
                                   TEXTINFO_BOTTOM,
                                   TEXTINFO_BOTTOM_ARROW,
                                   TEXTINFO_CENTER)


class Annotation:
    def __init__(self,
                 overlay: NDArrayBoolType,
                 *,
                 thicken_overlay: int = 0,
                 text_info: Optional[AnnotationTextInfo |
                                     list[AnnotationTextInfo]] = None,
                 config: Optional[Config] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._overlay = overlay
        if thicken_overlay > 0:
            for u_offset in range(-thicken_overlay, thicken_overlay + 1):
                for v_offset in range(-thicken_overlay, thicken_overlay + 1):
                    if u_offset == 0 and v_offset == 0:
                        continue
                    self._overlay = (self._overlay |
                                     shift_array(overlay, (v_offset, u_offset)))

        if not isinstance(text_info, (list, tuple)):
            text_info = [text_info]
        self._text_info = text_info


    @property
    def config(self) -> Config | None:
        return self._config

    @property
    def overlay(self) -> NDArrayBoolType:
        return self._overlay

    @property
    def text_info(self) -> AnnotationTextInfo | None:
        return self._text_info

    def add_text_info(self,
                      text_info: (AnnotationTextInfo |
                                  list[AnnotationTextInfo])) -> None:
        if not isinstance(text_info, (list, tuple)):
            text_info = [text_info]
        self._text_info.extend(text_info)

class Annotations:
    def __init__(self) -> None:
        self._annotations: list[Annotation] = []

    def add_annotation(self,
                       annotation: Annotation | None) -> None:
        if annotation is None:
            return
        if len(self._annotations):
            if annotation.overlay.shape != self._annotations[-1].overlay.shape:
                raise ValueError(
                    'Annotation does not have same shape as previous: '
                    f'{annotation.overlay.shape} vs '
                    f'{self._annotations[-1].overlay.shape}')
        self._annotations.append(annotation)

    def combine(self,
                extfov: tuple[int, int],
                overlay_color: tuple[int, int, int] = (255, 0, 0),
                include_text: bool = True,
                offset: tuple[int, int] = (0, 0),
                ) -> NDArrayIntType | None:

        if len(self._annotations) == 0:
            return None

        res = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        for annotation in self._annotations:
            res[annotation.overlay] = overlay_color

        if include_text:
            self._add_text(res, extfov, offset)

        return res[extfov[0]+offset[0]:res.shape[0]-extfov[0]+offset[0],
                   extfov[1]+offset[1]:res.shape[1]-extfov[1]+offset[1]]

    def _add_text(self,
                  res: NDArrayIntType,
                  extfov: tuple[int, int],
                  offset: tuple[int, int]) -> None:

        text_layer = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        graphic_layer = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        ann_num_mask = np.zeros(self._annotations[0].overlay.shape, dtype=np.int_)

        text_im = Image.frombuffer('RGB',
                                    (text_layer.shape[1], text_layer.shape[0]),
                                    text_layer, 'raw', 'RGB', 0, 1)
        text_draw = ImageDraw.Draw(text_im)

        for ann_num, annotation in enumerate(self._annotations):
            if not annotation.text_info:
                continue

            tt_dir = annotation.config.general('truetype_font_dir')

            for text_info in annotation.text_info:
                font = ImageFont.truetype(os.path.join(tt_dir, text_info.font),
                                          text_info.font_size)

                text_size = text_draw.textbbox((0, 0), text_info.text,
                                               anchor='lt', font=font)[2:]
                horiz_arrow_gap = 2
                horiz_arrow_len = 15
                vert_arrow_gap = 2
                vert_arrow_len = 15
                arrow_thickness = 1.5
                arrow_head_length = 6
                arrow_head_angle = 30
                edge_margin = 3
                v_margin_min = extfov[0] + offset[0] + edge_margin
                v_margin_max = (text_layer.shape[0] - extfov[0] + offset[0] -
                                edge_margin - 1)
                u_margin_min = extfov[1] + offset[1] + edge_margin
                u_margin_max = (text_layer.shape[1] - extfov[1] + offset[1] -
                                edge_margin - 1)

                for text_pos, text_v, text_u in text_info.text_loc:
                    arrow_u0 = arrow_u1 = arrow_v0 = arrow_v1 = None
                    if text_pos == TEXTINFO_LEFT:
                        v = text_v - text_size[1] // 2
                        u = text_u - text_size[0]
                    elif text_pos == TEXTINFO_LEFT_ARROW:
                        v = text_v - text_size[1] // 2
                        u = text_u - text_size[0] - horiz_arrow_len - horiz_arrow_gap
                        arrow_v0 = arrow_v1 = text_v
                        arrow_u0 = text_u - horiz_arrow_len
                        arrow_u1 = text_u
                    elif text_pos == TEXTINFO_RIGHT:
                        v = text_v - text_size[1] // 2
                        u = text_u
                    elif text_pos == TEXTINFO_RIGHT_ARROW:
                        v = text_v - text_size[1] // 2
                        u = text_u + horiz_arrow_len + horiz_arrow_gap
                        arrow_v0 = arrow_v1 = text_v
                        arrow_u0 = text_u + horiz_arrow_len
                        arrow_u1 = text_u
                    elif text_pos == TEXTINFO_TOP:
                        v = text_v - text_size[1]
                        u = text_u - text_size[0] // 2
                    elif text_pos == TEXTINFO_TOP_ARROW:
                        v = text_v - text_size[1] - vert_arrow_len - vert_arrow_gap
                        u = text_u - text_size[0] // 2
                        arrow_u0 = arrow_u1 = text_u
                        arrow_v0 = text_v - vert_arrow_len
                        arrow_v1 = text_v
                    elif text_pos == TEXTINFO_BOTTOM:
                        v = text_v
                        u = text_u - text_size[0] // 2
                    elif text_pos == TEXTINFO_BOTTOM_ARROW:
                        v = text_v + vert_arrow_len + vert_arrow_gap
                        u = text_u - text_size[0] // 2
                        arrow_u0 = arrow_u1 = text_u
                        arrow_v0 = text_v + vert_arrow_len
                        arrow_v1 = text_v
                    elif text_pos == TEXTINFO_CENTER:
                        v = text_v - text_size[1] // 2
                        u = text_u - text_size[0] // 2
                    else:
                        raise ValueError(f'Unknown text position: {text_pos}')

                    if (v < v_margin_min or v+text_size[1] > v_margin_max or
                        u < u_margin_min or u+text_size[0] > u_margin_max):
                        continue
                    if (arrow_u0 is not None and
                        not v_margin_min <= arrow_v0 <= v_margin_max or
                        not v_margin_min <= arrow_v1 <= v_margin_max or
                        not u_margin_min <= arrow_u0 <= u_margin_max or
                        not u_margin_min <= arrow_u1 <= u_margin_max):
                        continue

                    text_draw.text((u, v), text_info.text,
                                   anchor='lt', fill=text_info.color, font=font)
                    ann_num_mask[v:v+text_size[1], u:u+text_size[0]] = ann_num+1

                    if arrow_u0 is not None:
                        draw_line_arrow(graphic_layer, text_info.color,
                                        arrow_u0, arrow_v0, arrow_u1, arrow_v1,
                                        thickness=arrow_thickness,
                                        arrow_head_length=arrow_head_length,
                                        arrow_head_angle=arrow_head_angle)

                    break
                else:
                    print(f'Warning: Could not find place for label {text_info.text}')

        text_layer = (np.array(text_im.getdata()).astype('uint8').
                      reshape(text_layer.shape))
        text_layer[graphic_layer != 0] = graphic_layer[graphic_layer != 0]

        res[text_layer != 0] = text_layer[text_layer != 0]

        # plt.figure()
        # plt.imshow(text_layer)
        # plt.show()
