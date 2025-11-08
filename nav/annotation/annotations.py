from typing import Optional, Union, cast

import numpy as np
from PIL import Image, ImageDraw

from nav.config import Config
from nav.obs import ObsSnapshot
from .annotation import Annotation
from nav.support.nav_base import NavBase
from nav.support.types import NDArrayBoolType, NDArrayIntType


class Annotations(NavBase):
    """Manages a collection of annotation objects for an observation.

    This class provides functionality to combine multiple annotations into a single
    overlay image and handle text placement.
    """
    def __init__(self,
                 *,
                 config: Optional[Config] = None) -> None:
        """Initializes an empty annotations collection."""
        super().__init__(config=config)
        self._annotations: list[Annotation] = []

    def add_annotations(self,
                        annotations: Union[Annotation,
                                           list[Annotation],
                                           'Annotations',
                                           None]
                        ) -> None:
        """Adds one or more annotations to this collection.

        Parameters:
            annotations: The annotation(s) to add. Can be a single Annotation, a list
                of Annotations, another Annotations object, or None.

        Raises:
            ValueError: If an annotation is for a different observation than existing
            annotations.
        """
        if annotations is None:
            return
        if isinstance(annotations, Annotations):
            ann_list = annotations.annotations
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
        """Combines all annotations into a single graphic overlay image.

        Parameters:
            offset: Optional offset (dv,du) to apply to all annotations
            include_text: Whether to include text annotations
            text_use_avoid_mask: Whether to use avoid masks for text placement
            text_avoid_other_text: Whether text should avoid other text
            text_show_all_positions: Whether to show all possible text positions

        Returns:
            A combined RGB array containing all annotations, or None if no annotations
            exist.
        """

        log_level = self._config.general.get('log_level_annotate')
        with self.logger.open('ANNOTATE IMAGE', level=log_level):
            if len(self.annotations) == 0:
                self.logger.info('No annotations to annotate')
                return None

            obs = self.annotations[0].obs

            data_shape = obs.data_shape_vu

            res = np.zeros(data_shape + (3,), dtype=np.uint8)
            all_avoid_mask = np.zeros(data_shape, dtype=bool)

            for annotation in self.annotations:
                # TODO This does not handle z-depth. In other words, an overlay does not
                # get hidden by other overlays in front of it. This can best be seen with
                # two moons that are partially occluding each other.
                overlay = obs.extract_offset_array(annotation.overlay, offset)

                res[overlay] = annotation.overlay_color
                if text_use_avoid_mask and annotation.avoid_mask is not None:
                    avoid_mask = obs.extract_offset_array(annotation.avoid_mask, offset)
                    all_avoid_mask |= avoid_mask

            if include_text:
                self._add_text(obs, res, offset, all_avoid_mask,
                               text_avoid_other_text, text_show_all_positions)

            return res

    def _add_text(self,
                  obs: ObsSnapshot,
                  res: NDArrayIntType,
                  offset: tuple[float, float],
                  avoid_mask: NDArrayBoolType,
                  text_avoid_other_text: bool,
                  text_show_all_positions: bool) -> None:
        """Adds label text to an existing overlay image.

        Parameters:
            obs: The observation snapshot
            res: The target image array to modify
            offset: Offset to apply to annotations
            avoid_mask: Mask indicating areas to avoid when placing text
            text_avoid_other_text: Whether text should avoid other text
            text_show_all_positions: Whether to show all possible text positions
        """

        text_layer = np.zeros_like(res, dtype=np.uint8)
        graphic_layer = np.zeros_like(res, dtype=np.uint8)
        if text_avoid_other_text:
            ann_num_mask = np.zeros(res.shape, dtype=int)
        else:
            ann_num_mask = None

        text_im = Image.fromarray(text_layer, mode='RGB')
        text_draw = ImageDraw.Draw(text_im)

        for ann_num, annotation in enumerate(self.annotations):
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
                    self.logger.debug(
                        'Could not find place avoiding other items for text annotation '
                        f'{text_info.text!r}')
                if not found_place:
                    self.logger.warning(
                        'Could not find final place for text annotation '
                        f'{text_info.text!r}')

        # This ensures text_layer is writeable
        text_layer = (np.array(text_im.getdata()).astype(np.uint8)
                      .reshape(text_layer.shape))
        text_layer[graphic_layer != 0] = graphic_layer[graphic_layer != 0]

        res[text_layer != 0] = text_layer[text_layer != 0]
