from typing import Optional

from nav.config import Config, DEFAULT_CONFIG
from nav.obs import ObsSnapshot
from nav.support.image import shift_array
from nav.support.types import NDArrayBoolType

from .annotation_text_info import AnnotationTextInfo


class Annotation:
    """Represents an annotation for an observation image.

    This class handles overlays and text annotations to be displayed on observation
    images.

    Parameters:
        obs: The observation snapshot to annotate
        overlay: Boolean mask indicating where the overlay should appear
        overlay_color: RGB color tuple for the overlay
        thicken_overlay: Number of pixels to thicken the overlay by
        text_info: Text annotation information
        avoid_mask: Boolean mask indicating areas where text should not be placed
        config: Configuration object
    """
    def __init__(self,
                 obs: ObsSnapshot,
                 overlay: NDArrayBoolType,
                 overlay_color: tuple[int, int, int],
                 *,
                 thicken_overlay: int = 0,
                 text_info: Optional[AnnotationTextInfo |
                                     list[AnnotationTextInfo]] = None,
                 avoid_mask: Optional[NDArrayBoolType] = None,
                 config: Optional[Config] = None) -> None:

        self._config = config or DEFAULT_CONFIG
        self._obs = obs
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

        if overlay.shape != obs.extdata_shape_vu:
            raise ValueError(
                f'Annotation overlay shape ({overlay.shape}) does not agree with Obs '
                f'shape ({obs.extdata_shape_vu})')

    @property
    def config(self) -> Config:
        """Returns the configuration object for this annotation."""
        return self._config

    @property
    def obs(self) -> ObsSnapshot:
        """Returns the observation snapshot associated with this annotation."""
        return self._obs

    @property
    def overlay(self) -> NDArrayBoolType:
        """Returns the boolean mask representing the overlay area."""
        return self._overlay

    @property
    def overlay_color(self) -> tuple[int, int, int]:
        """Returns the RGB color tuple for the overlay."""
        return self._overlay_color

    @property
    def avoid_mask(self) -> NDArrayBoolType | None:
        """Returns the mask indicating areas where text should not be placed, if any."""
        return self._avoid_mask

    @property
    def text_info_list(self) -> list[AnnotationTextInfo]:
        """Returns the list of text annotation information objects."""
        return self._text_info

    def add_text_info(self,
                      text_info: (AnnotationTextInfo |
                                  list[AnnotationTextInfo])) -> None:
        """Adds text annotation information to this annotation.

        Parameters:
            text_info: One or more text annotation information objects to add
        """

        if not isinstance(text_info, (list, tuple)):
            text_info = [text_info]
        self._text_info.extend(text_info)
