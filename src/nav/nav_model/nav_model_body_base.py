import numpy as np
import scipy.ndimage as ndimage

from nav.annotation import (
    Annotation,
    Annotations,
    AnnotationTextInfo,
    TextLocInfo,
    TEXTINFO_LEFT_ARROW,
    TEXTINFO_RIGHT_ARROW,
    TEXTINFO_BOTTOM_ARROW,
    TEXTINFO_TOP_ARROW,
)
from nav.support.image import shift_array
from nav.support.types import NDArrayBoolType, NDArrayFloatType
from .nav_model import NavModel


class NavModelBodyBase(NavModel):
    """Base class for body navigation models.

    Provides shared helpers to compute a limb mask and to create annotations
    consistent with the standard body model implementation.
    """

    def _compute_limb_mask_from_body_mask(
        self, body_mask: NDArrayBoolType
    ) -> NDArrayBoolType:
        """Compute limb mask as body pixels adjacent to at least one non-body neighbor."""
        neighbor = (
            shift_array(~body_mask, (-1, 0))
            | shift_array(~body_mask, (1, 0))
            | shift_array(~body_mask, (0, -1))
            | shift_array(~body_mask, (0, 1))
        )
        return body_mask & neighbor

    def _create_annotations(
        self,
        u_center: int,
        v_center: int,
        model: NDArrayFloatType,
        limb_mask: NDArrayBoolType,
        body_mask: NDArrayBoolType,
    ) -> Annotations:
        """Creates annotation objects for labeling the body in visualizations.

        This is functionally equivalent to the implementation used by the
        normal body model, so annotation behavior is consistent across models.
        """
        obs = self._obs
        body_name = getattr(self, '_body_name', 'BODY')
        body_config = self._config.bodies

        text_loc: list[TextLocInfo] = []
        v_center_extfov = v_center + obs.extfov_margin_v
        u_center_extfov = u_center + obs.extfov_margin_u

        v_center_extfov_clipped = np.clip(v_center_extfov, 0, body_mask.shape[0] - 1)
        u_center_extfov_clipped = np.clip(u_center_extfov, 0, body_mask.shape[1] - 1)
        if not body_mask[v_center_extfov_clipped].any():
            body_mask_u_min = 0
            body_mask_u_max = body_mask.shape[1] - 1
        else:
            body_mask_u_min = int(np.argmax(body_mask[v_center_extfov_clipped]))
            body_mask_u_max = int(
                (
                    body_mask.shape[1]
                    - np.argmax(body_mask[v_center_extfov_clipped, ::-1])
                    - 1
                )
            )
        body_mask_v_min = int(np.argmax(body_mask[:, u_center_extfov_clipped]))
        body_mask_v_max = int(
            (
                body_mask.shape[0]
                - np.argmax(body_mask[::-1, u_center_extfov_clipped])
                - 1
            )
        )
        body_mask_u_ctr = (body_mask_u_min + body_mask_u_max) // 2
        body_mask_v_ctr = (body_mask_v_min + body_mask_v_max) // 2

        # Scan around center to place labels on limb
        for orig_dist in range(
            0, max(body_mask_v_ctr - body_mask_v_min, body_config.label_scan_v)
        ):
            for neg in [-1, 1]:
                dist = orig_dist * neg
                v = body_mask_v_ctr + dist
                if not 0 <= v < body_mask.shape[0]:
                    continue

                # Left side
                u = int(np.argmax(body_mask[v]))
                if u > 0:
                    angle = (
                        np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov))
                        % 360
                    )
                    if 135 < angle < 225:  # Left side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_LEFT_ARROW, v, u - body_config.label_horiz_gap
                            )
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u
                            )
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u
                            )
                        )

                # Right side
                u = body_mask.shape[1] - int(np.argmax(body_mask[v, ::-1])) - 1
                if u < body_mask.shape[1] - 1:
                    angle = (
                        np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov))
                        % 360
                    )
                    if angle > 315 or angle < 45:  # Right side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_RIGHT_ARROW, v, u + body_config.label_horiz_gap
                            )
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u
                            )
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(
                                TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u
                            )
                        )

                if orig_dist == 0:
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_TOP_ARROW,
                            body_mask_v_min - body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_BOTTOM_ARROW,
                            body_mask_v_max + body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    break

        # Coarse scan for additional candidates
        for v_orig_dist in range(
            0, body_mask_v_ctr - body_mask_v_min, body_config.label_grid_v
        ):
            for v_neg in [-1, 1]:
                v_dist = v_orig_dist * v_neg
                v = body_mask_v_ctr + v_dist
                if not 0 <= v < body_mask.shape[0]:
                    continue
                for u_orig_dist in range(
                    0, body_mask_u_ctr - body_mask_u_min, body_config.label_grid_u
                ):
                    for u_neg in [-1, 1]:
                        u_dist = u_orig_dist * u_neg
                        u = body_mask_u_ctr + u_dist
                        if not 0 <= u < body_mask.shape[1]:
                            continue
                        if not body_mask[v, u]:
                            continue
                        if u < model.shape[1] // 2:
                            text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW, v, u))
                        else:
                            text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u))
                if v_orig_dist == 0:
                    break

        text_info = AnnotationTextInfo(
            body_name,
            text_loc=text_loc,
            ref_vu=None,
            font=body_config.label_font,
            font_size=body_config.label_font_size,
            color=body_config.label_font_color,
        )

        text_avoid_mask = ndimage.maximum_filter(
            body_mask, body_config.label_mask_enlarge
        )

        annotation = Annotation(
            obs,
            limb_mask,
            body_config.label_limb_color,
            thicken_overlay=body_config.outline_thicken,
            avoid_mask=text_avoid_mask,
            text_info=text_info,
            config=self._config,
        )

        annotations = Annotations()
        annotations.add_annotations(annotation)
        return annotations
