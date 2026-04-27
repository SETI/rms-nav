"""Compose template-bearing features into a single ext-FOV image and mask.

Some downstream consumers (the manual-navigation dialog, the summary-PNG
overlay) want a single 2-D representation of "what the predicted scene
looks like" rather than a per-feature collection.  This module builds that
composite by Z-buffer painting every feature that carries a
``template_img`` / ``template_mask`` payload, ordered by subject range so
nearer features paint over farther ones.

Polyline-only features (limbs, terminators, ring edges) are skipped; their
geometry is rendered separately by the annotation pipeline.
"""

from __future__ import annotations

import numpy as np

from nav.feature.feature import NavFeature
from nav.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['compose_template_features']


def compose_template_features(
    features: list[NavFeature], extfov_shape_vu: tuple[int, int]
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Z-buffer paint template features into a single ext-FOV image+mask.

    Features are sorted by ``subject_range_km`` ascending (nearer features
    last), so the closest body's pixels overwrite farther bodies' on
    overlap.  Each feature's template is placed at its
    ``geometry.bbox_extfov_vu`` location; pixels marked True in the
    feature's ``template_mask`` carry the template's value into the
    composite, and the composite mask becomes the OR of every painted
    feature's mask.

    Features without a ``template_img`` or ``template_mask`` are skipped.

    Parameters:
        features: The feature list (may include non-template features).
        extfov_shape_vu: Shape ``(v, u)`` of the ext-FOV array to build.

    Returns:
        Tuple ``(image, mask)`` where ``image`` is float64 in ext-FOV
        coordinates and ``mask`` is a boolean array of the same shape.
    """
    image: NDArrayFloatType = np.zeros(extfov_shape_vu, dtype=np.float64)
    mask: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
    template_features = [
        f for f in features if f.template_img is not None and f.template_mask is not None
    ]
    # Nearer features paint last so they overwrite farther ones on overlap.
    ordered = sorted(template_features, key=lambda f: f.subject_range_km, reverse=True)
    for feature in ordered:
        v_min, u_min, v_max, u_max = _bbox_clamped(feature.geometry.bbox_extfov_vu, extfov_shape_vu)
        if v_max <= v_min or u_max <= u_min:
            continue
        assert feature.template_img is not None
        assert feature.template_mask is not None
        template_img = feature.template_img
        template_mask = feature.template_mask
        # Slice the part of the template that fits inside ext-FOV.
        bbox = feature.geometry.bbox_extfov_vu
        t_v_lo = v_min - bbox[0]
        t_u_lo = u_min - bbox[1]
        t_v_hi = t_v_lo + (v_max - v_min)
        t_u_hi = t_u_lo + (u_max - u_min)
        if t_v_hi > template_img.shape[0] or t_u_hi > template_img.shape[1]:
            raise ValueError(
                f'feature {feature.feature_id!r}: declared bbox '
                f'{feature.geometry.bbox_extfov_vu!r} extends past template '
                f'shape {template_img.shape!r}'
            )
        sub_img = template_img[t_v_lo:t_v_hi, t_u_lo:t_u_hi]
        sub_mask = template_mask[t_v_lo:t_v_hi, t_u_lo:t_u_hi]
        target_image_slice = image[v_min:v_max, u_min:u_max]
        target_mask_slice = mask[v_min:v_max, u_min:u_max]
        np.copyto(target_image_slice, sub_img, where=sub_mask)
        np.logical_or(target_mask_slice, sub_mask, out=target_mask_slice)
    return image, mask


def _bbox_clamped(
    bbox_extfov_vu: tuple[int, int, int, int], extfov_shape_vu: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clamp a feature's bbox to lie inside ``extfov_shape_vu``."""
    v_min, u_min, v_max, u_max = bbox_extfov_vu
    h, w = extfov_shape_vu
    return (
        max(0, int(v_min)),
        max(0, int(u_min)),
        min(int(h), int(v_max)),
        min(int(w), int(u_max)),
    )
