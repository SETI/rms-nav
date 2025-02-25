from typing import Union, cast

import numpy as np
from PIL import Image, ImageDraw

from .annotation import Annotation

from nav.obs import ObsSnapshot
from nav.support.types import NDArrayBoolType, NDArrayIntType


class Annotations:
    def __init__(self) -> None:
        self._annotations: list[Annotation] = []

    def add_annotations(self,
                        annotations: Union[Annotation,
                                           list[Annotation],
                                           'Annotations',
                                           None]
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
                if ann.obs != self._annotations[-1].obs:
                    raise ValueError('Annotation does not have same Obs as previous')
            self._annotations.append(ann)

    @property
    def annotations(self) -> list[Annotation]:
        """Return the list of annotations."""
        return self._annotations

    def combine(self,
                offset: tuple[float, float] | None = None,
                include_text: bool = True,
                text_use_avoid_mask: bool = True,
                text_avoid_other_text: bool = True,
                text_show_all_positions: bool = False
                ) -> NDArrayIntType | None:
        """Combine all annotations into a single graphic overlay."""

        if len(self.annotations) == 0:
            return None

        obs = self.annotations[0].obs

        data_shape = obs.data_shape_vu

        res = np.zeros(data_shape + (3,), dtype=np.uint8)
        all_avoid_mask = np.zeros(data_shape, dtype=np.bool_)

        if offset is None:
            int_offset = (0, 0)
        else:
            int_offset = (int(offset[0]), int(offset[1]))

        for annotation in self._annotations:
            # TODO This does not handle z-depth. In other words, an overlay does not
            # get hidden by other overlays in front of it. This can best be seen with
            # two moons that are partially occluding each other.
            overlay = annotation.overlay[
                obs.extfov_margin_v+int_offset[0]:
                    obs.extfov_margin_v+data_shape[0]+int_offset[0],
                obs.extfov_margin_u+int_offset[1]:
                    obs.extfov_margin_u+data_shape[1]+int_offset[1]]

            res[overlay] = annotation.overlay_color
            if text_use_avoid_mask and annotation.avoid_mask is not None:
                avoid_mask = annotation.avoid_mask[
                    obs.extfov_margin_v+int_offset[0]:
                        obs.extfov_margin_v+data_shape[0]+int_offset[0],
                    obs.extfov_margin_u+int_offset[1]:
                        obs.extfov_margin_u+data_shape[1]+int_offset[1]]
                all_avoid_mask |= avoid_mask

        if include_text:
            self._add_text(obs, res, int_offset, all_avoid_mask,
                           text_avoid_other_text, text_show_all_positions)

        return res

    def _add_text(self,
                  obs: ObsSnapshot,
                  res: NDArrayIntType,
                  offset: tuple[int, int],
                  avoid_mask: NDArrayBoolType,
                  text_avoid_other_text: bool,
                  text_show_all_positions: bool) -> None:
        """Add label text to an existing overlay."""

        text_layer = np.zeros_like(res, dtype=np.uint8)
        graphic_layer = np.zeros_like(res, dtype=np.uint8)
        if text_avoid_other_text:
            ann_num_mask = np.zeros(self._annotations[0].overlay.shape, dtype=np.int_)
        else:
            ann_num_mask = None

        text_im = Image.frombuffer('RGB',
                                   (text_layer.shape[1], text_layer.shape[0]),
                                   text_layer, 'raw', 'RGB', 0, 1)
        text_draw = ImageDraw.Draw(text_im)

        for ann_num, annotation in enumerate(self._annotations):
            # TODO ann_num is not really used for anything right now. Eventually it could
            # be used for backtracking to know which annotation to try to move in order
            # to place one that is overconstrained. However, ann_num is really not enough
            # because we want a number for each text_info, so this code should probably
            # just go away at some point.

            if not annotation.text_info_list:
                continue

            tt_dir = cast(str, annotation.config.general.truetype_font_dir)

            # We first try to place the label avoiding the masked pixels, but if that
            # fails we go ahead and put the text in those places.
            for text_info in annotation.text_info_list:
                found_place = False
                for avoid in [True, False]:
                    ret = text_info._draw_text(ann_num=ann_num,
                                               extfov=obs.extfov_margin_vu,
                                               offset=offset,
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
                    annotation.config.logger.debug(
                        'Count not find place avoiding other items for text annotation '
                        f'{text_info.text!r}')
                if not found_place:
                    annotation.config.logger.warning(
                        'Could not find final place for text annotation '
                        f'{text_info.text!r}')

        text_layer = (np.array(text_im.getdata()).astype('uint8')
                      .reshape(text_layer.shape))
        text_layer[graphic_layer != 0] = graphic_layer[graphic_layer != 0]

        res[text_layer != 0] = text_layer[text_layer != 0]
