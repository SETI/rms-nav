from typing import Optional

from nav.config import Config, DEFAULT_CONFIG
from nav.util.image import shift_array
from nav.util.types import NDArrayBoolType

from .annotation_text_info import AnnotationTextInfo


class Annotation:
    def __init__(self,
                 overlay: NDArrayBoolType,
                 overlay_color: tuple[int, int, int],
                 *,
                 thicken_overlay: int = 0,
                 text_info: Optional[AnnotationTextInfo |
                                     list[AnnotationTextInfo]] = None,
                 avoid_mask: Optional[NDArrayBoolType] = None,
                 config: Optional[Config] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._overlay = overlay
        self._overlay_color = overlay_color
        self._avoid_mask = avoid_mask
        if thicken_overlay > 0:
            for u_offset in range(-thicken_overlay, thicken_overlay + 1):
                for v_offset in range(-thicken_overlay, thicken_overlay + 1):
                    if u_offset == 0 and v_offset == 0:
                        continue
                    self._overlay = (self._overlay |
                                     shift_array(overlay, (v_offset, u_offset)))

        if text_info is None:
            self._text_info = []
        elif isinstance(text_info, (list, tuple)):
            self._text_info = text_info
        else:
            self._text_info = [text_info]

    @property
    def config(self) -> Config:
        return self._config

    @property
    def overlay(self) -> NDArrayBoolType:
        return self._overlay

    @property
    def text_info_list(self) -> list[AnnotationTextInfo]:
        return self._text_info

    @property
    def avoid_mask(self) -> NDArrayBoolType | None:
        return self._avoid_mask

    def add_text_info(self,
                      text_info: (AnnotationTextInfo |
                                  list[AnnotationTextInfo])) -> None:
        if not isinstance(text_info, (list, tuple)):
            text_info = [text_info]
        self._text_info.extend(text_info)
