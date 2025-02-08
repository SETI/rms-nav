from typing import Optional, cast

from nav.config import Config, DEFAULT_CONFIG


TEXTINFO_LEFT = 'left'
TEXTINFO_LEFT_ARROW = 'left_arrow'
TEXTINFO_RIGHT = 'right'
TEXTINFO_RIGHT_ARROW = 'right_arrow'
TEXTINFO_TOP = 'top'
TEXTINFO_TOP_ARROW = 'top_arrow'
TEXTINFO_BOTTOM = 'bottom'
TEXTINFO_BOTTOM_ARROW = 'bottom_arrow'
TEXTINFO_CENTER = 'center'


class AnnotationTextInfo:
    def __init__(self,
                 text: str,
                 text_loc: list[tuple[str, int, int]],
                 *,
                 color: tuple[int, ...],
                 font: str,
                 font_size: int):

        self._config = DEFAULT_CONFIG
        self._text = text
        self._text_loc = text_loc
        self._color = tuple(color)
        self._font = font
        self._font_size = font_size

    @property
    def text(self) -> str:
        return self._text

    @property
    def text_loc(self) -> list[tuple[str, int, int]]:
        return self._text_loc

    @property
    def color(self) -> tuple[int, ...]:
        return self._color

    @property
    def font(self) -> str:
        return self._font

    @property
    def font_size(self) -> int:
        return self._font_size
