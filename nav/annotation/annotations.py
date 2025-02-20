from typing import cast

import numpy as np
from PIL import Image, ImageDraw

from .annotation import Annotation

from nav.util.types import NDArrayBoolType, NDArrayIntType


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
                offset: tuple[int, int] = (0, 0),
                include_text: bool = True,
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
                    ret = text_info._draw_text(ann_num=ann_num,
                                               extfov=extfov, offset=offset,
                                               avoid_mask=avoid_mask if avoid else None,
                                               text_layer=text_layer,
                                               graphic_layer=graphic_layer,
                                               ann_num_mask=ann_num_mask,
                                               text_draw=text_draw,
                                               tt_dir=tt_dir,
                                               show_all_positions=text_show_all_positions)
                    if ret:
                        found_place = True
                        break
                if not found_place:
                    annotation.config.logger.warning(
                        f'Could not find place for text annotation {text_info.text}')

        text_layer = (np.array(text_im.getdata()).astype('uint8')
                      .reshape(text_layer.shape))
        text_layer[graphic_layer != 0] = graphic_layer[graphic_layer != 0]

        res[text_layer != 0] = text_layer[text_layer != 0]
