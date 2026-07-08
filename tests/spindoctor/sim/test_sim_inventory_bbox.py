"""Inventory bounding boxes follow the projected, rotated silhouette (SIM-7).

A body's inventory bbox uses the per-axis half-extents of the axis1/axis2
ellipse rotated in-plane by rotation_z (axis3 is the depth axis and does not
project): sqrt((a cos t)^2 + (b sin t)^2) along v and
sqrt((a sin t)^2 + (b cos t)^2) along u.  These tests plant an elongated,
tilted body and check the bbox against both the analytic values and the
rendered silhouette.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render import render_combined_model

_CENTER = 64.0
_SEMI_MAJOR = 30.0  # axis1 / 2
_SEMI_MINOR = 10.0  # axis2 / 2
_ROTATION_Z_DEG = 30.0

_COS_T = np.cos(np.radians(_ROTATION_Z_DEG))
_SIN_T = np.sin(np.radians(_ROTATION_Z_DEG))
_HALF_V = float(np.hypot(_SEMI_MAJOR * _COS_T, _SEMI_MINOR * _SIN_T))
_HALF_U = float(np.hypot(_SEMI_MAJOR * _SIN_T, _SEMI_MINOR * _COS_T))

# Silhouette extents are quantized to pixel centers and the sub-pixel
# positioning shift can spread the edge by one pixel.
_PIXEL_TOLERANCE = 1.5


def _scene() -> dict[str, Any]:
    """A noiseless scene with one fully lit, elongated, rotated body."""
    return {
        'size_v': 128,
        'size_u': 128,
        'random_seed': 3,
        'instrument': 'coiss_nac',
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
        'bodies': [
            {
                'name': 'SLAB',
                'center_v': _CENTER,
                'center_u': _CENTER,
                'axis1': 2 * _SEMI_MAJOR,
                'axis2': 2 * _SEMI_MINOR,
                'axis3': 2 * _SEMI_MINOR,
                'rotation_z': _ROTATION_Z_DEG,
                'phase_angle': 0.0,
                'anti_aliasing': 0.0,
            }
        ],
    }


def _inventory() -> dict[str, float]:
    """Render the scene and return the planted body's inventory entry."""
    _, meta = render_combined_model(_scene())
    inventory: dict[str, float] = meta['inventory']['SLAB']
    return inventory


def _mask_extents() -> tuple[float, float, float, float]:
    """Render the scene and return (v_min, v_max, u_min, u_max) of the mask."""
    _, meta = render_combined_model(_scene())
    mask = meta['body_masks'][0]
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return float(rows.min()), float(rows.max()), float(cols.min()), float(cols.max())


def test_bbox_v_size_matches_rotated_ellipse() -> None:
    """The v pixel size equals the analytic projected extent along v."""
    assert _inventory()['v_pixel_size'] == pytest.approx(2 * _HALF_V)


def test_bbox_u_size_matches_rotated_ellipse() -> None:
    """The u pixel size equals the analytic projected extent along u."""
    assert _inventory()['u_pixel_size'] == pytest.approx(2 * _HALF_U)


def test_bbox_v_limits_match_rotated_ellipse() -> None:
    """The v min/max limits bracket the center by the analytic half-extent."""
    inventory = _inventory()
    assert inventory['v_min_unclipped'] == pytest.approx(_CENTER - _HALF_V)
    assert inventory['v_max_unclipped'] == pytest.approx(_CENTER + _HALF_V)


def test_bbox_u_limits_match_rotated_ellipse() -> None:
    """The u min/max limits bracket the center by the analytic half-extent."""
    inventory = _inventory()
    assert inventory['u_min_unclipped'] == pytest.approx(_CENTER - _HALF_U)
    assert inventory['u_max_unclipped'] == pytest.approx(_CENTER + _HALF_U)


def test_silhouette_falls_inside_bbox() -> None:
    """Every rendered body pixel lies within the inventory bbox."""
    inventory = _inventory()
    v_min, v_max, u_min, u_max = _mask_extents()
    assert v_min >= inventory['v_min_unclipped'] - _PIXEL_TOLERANCE
    assert v_max <= inventory['v_max_unclipped'] + _PIXEL_TOLERANCE
    assert u_min >= inventory['u_min_unclipped'] - _PIXEL_TOLERANCE
    assert u_max <= inventory['u_max_unclipped'] + _PIXEL_TOLERANCE


def test_bbox_hugs_silhouette() -> None:
    """The bbox is tight: the silhouette reaches each bbox edge."""
    inventory = _inventory()
    v_min, v_max, u_min, u_max = _mask_extents()
    assert v_min <= inventory['v_min_unclipped'] + _PIXEL_TOLERANCE
    assert v_max >= inventory['v_max_unclipped'] - _PIXEL_TOLERANCE
    assert u_min <= inventory['u_min_unclipped'] + _PIXEL_TOLERANCE
    assert u_max >= inventory['u_max_unclipped'] - _PIXEL_TOLERANCE
