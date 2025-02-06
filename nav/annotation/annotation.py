from typing import Optional

import numpy as np

from nav.util.image import shift_array
from nav.util.types import NDArrayBoolType, NDArrayIntType

from .text_info import TextInfo


class Annotation:
    def __init__(self,
                 overlay: NDArrayBoolType,
                 *,
                 thicken_overlay: int = 0,
                 text_info: Optional[TextInfo] = None) -> None:
        self._overlay = overlay
        if thicken_overlay > 0:
            for u_offset in range(-thicken_overlay, thicken_overlay + 1):
                for v_offset in range(-thicken_overlay, thicken_overlay + 1):
                    if u_offset == 0 and v_offset == 0:
                        continue
                    self._overlay = (self._overlay |
                                     shift_array(overlay, (v_offset, u_offset)))

        self._text_info = text_info

    @property
    def overlay(self) -> NDArrayBoolType:
        return self._overlay

    @property
    def text_info(self) -> TextInfo | None:
        return self._text_info


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
                overlay_color: tuple[int, int, int] = (255, 0, 0)
                ) -> NDArrayIntType | None:
        if len(self._annotations) == 0:
            return None
        res = np.zeros(self._annotations[0].overlay.shape + (3,), dtype=np.uint8)
        for annotation in self._annotations:
            res[annotation.overlay] = overlay_color

        return res
