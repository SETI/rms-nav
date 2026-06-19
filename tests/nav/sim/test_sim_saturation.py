"""Saturation clipping and column bloom for the simulator (B4).

These tests drive ``apply_saturation`` directly for the clip/bloom mechanics and
render a bright scene end-to-end to confirm saturated pixels land on the
orchestrator's saturation mask.
"""

from typing import Any

import numpy as np

from nav.sim.render import apply_saturation, render_combined_model
from nav.support.image_quality import saturation_mask

_SATURATION = 4095.0


def test_clip_caps_at_saturation() -> None:
    """Pixels above the full well are clipped down to it."""
    img = np.array([[0.0, 2000.0], [5000.0, 9000.0]], dtype=np.float64)
    apply_saturation(img, saturation_dn=_SATURATION)
    assert float(img.max()) == _SATURATION


def test_clip_leaves_subsaturated_pixels_untouched() -> None:
    """Pixels below the full well are unchanged by clipping."""
    img = np.array([[0.0, 2000.0], [5000.0, 9000.0]], dtype=np.float64)
    apply_saturation(img, saturation_dn=_SATURATION)
    assert float(img[0, 1]) == 2000.0


def test_no_bloom_keeps_single_saturated_pixel() -> None:
    """Without bloom, only the originally over-full pixel saturates."""
    img = np.zeros((21, 21), dtype=np.float64)
    img[10, 10] = 50.0 * _SATURATION
    apply_saturation(img, saturation_dn=_SATURATION, bloom_length=0)
    assert int((img >= _SATURATION).sum()) == 1


def test_bloom_spreads_along_column() -> None:
    """Column bloom turns a saturated pixel into a vertical streak."""
    img = np.zeros((21, 21), dtype=np.float64)
    img[10, 10] = 50.0 * _SATURATION
    apply_saturation(img, saturation_dn=_SATURATION, bloom_length=3)
    saturated = np.argwhere(img >= _SATURATION)
    distinct_v = {int(v) for v, _ in saturated}
    distinct_u = {int(u) for _, u in saturated}
    assert len(distinct_v) > 1
    assert distinct_u == {10}


def test_bloom_conserves_excess_floor() -> None:
    """Bloom never lowers the count of saturated pixels below the no-bloom case."""
    base = np.zeros((21, 21), dtype=np.float64)
    base[10, 10] = 50.0 * _SATURATION
    bloomed = base.copy()
    apply_saturation(base, saturation_dn=_SATURATION, bloom_length=0)
    apply_saturation(bloomed, saturation_dn=_SATURATION, bloom_length=4)
    assert int((bloomed >= _SATURATION).sum()) >= int((base >= _SATURATION).sum())


def _bright_scene(*, size: int = 64) -> dict[str, Any]:
    """A coiss scene bright enough to drive the lit body into saturation."""
    return {
        'size_v': size,
        'size_u': size,
        'random_seed': 1,
        'instrument': 'coiss_nac',
        'noise': {'signal_full_scale_frac': 2.0},
        'bodies': [
            {
                'name': 'B',
                'center_v': size / 2,
                'center_u': size / 2,
                'axis1': size * 0.6,
                'axis2': size * 0.5,
                'axis3': size * 0.5,
            }
        ],
    }


def test_render_saturates_at_full_well() -> None:
    """A bright scene clips at the camera full well, not above it."""
    img, _ = render_combined_model(_bright_scene())
    assert float(img.max()) == _SATURATION


def test_render_saturation_lands_on_mask() -> None:
    """Saturated render pixels are flagged by the orchestrator's mask."""
    img, _ = render_combined_model(_bright_scene())
    mask = saturation_mask(img, full_well_dn=_SATURATION)
    assert int(mask.sum()) > 0
