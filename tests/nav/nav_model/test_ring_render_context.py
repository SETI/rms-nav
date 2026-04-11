"""Unit tests for RingsRenderContext and RingRenderResult.

Simple construction and field-access tests. Both are plain dataclasses;
no complex logic to test beyond that fields are correctly set and that
RingsRenderContext is frozen.
"""

import numpy as np
import pytest

from nav.nav_model.rings.ring_render_context import RingsRenderContext
from nav.nav_model.rings.ring_render_result import RingRenderResult

# ---------------------------------------------------------------------------
# RingsRenderContext
# ---------------------------------------------------------------------------


def _make_context(**kwargs: object) -> RingsRenderContext:
    """Create a minimal RingsRenderContext with sensible defaults."""
    defaults: dict[str, object] = {
        'obs': object(),
        'ring_target': 'saturn:ring',
        'epoch': 252460865.0,
        'resolutions': np.ones((10, 10), dtype=np.float64),
        'fade_width_pix': 100.0,
        'all_edge_radii': (),
    }
    defaults.update(kwargs)
    return RingsRenderContext(**defaults)  # type: ignore[arg-type]


def test_context_fields_set_correctly() -> None:
    """All constructor arguments appear as attributes."""
    obs = object()
    resolutions = np.ones((5, 7), dtype=np.float64)
    edge_radii = ((80_000.0, 'IEG'), (90_000.0, 'OEG'))
    ctx = RingsRenderContext(
        obs=obs,
        ring_target='uranus:ring',
        epoch=1.0,
        resolutions=resolutions,
        fade_width_pix=50.0,
        all_edge_radii=edge_radii,
    )
    assert ctx.obs is obs
    assert ctx.ring_target == 'uranus:ring'
    assert ctx.epoch == pytest.approx(1.0)
    assert ctx.resolutions is resolutions
    assert ctx.fade_width_pix == pytest.approx(50.0)
    assert ctx.all_edge_radii == edge_radii


def test_context_is_frozen() -> None:
    """RingsRenderContext is frozen: attribute assignment raises an error."""
    ctx = _make_context()
    with pytest.raises(AttributeError, match=r"cannot assign to field 'epoch'"):
        ctx.epoch = 999.0  # type: ignore[misc]


def test_context_resolutions_shape_preserved() -> None:
    """Resolutions array is stored by reference, shape preserved."""
    res = np.full((3, 4), 5.0)
    ctx = _make_context(resolutions=res)
    assert ctx.resolutions.shape == (3, 4)
    assert ctx.resolutions[0, 0] == pytest.approx(5.0)


def test_context_empty_edge_radii() -> None:
    """Empty all_edge_radii tuple is valid."""
    ctx = _make_context(all_edge_radii=())
    assert ctx.all_edge_radii == ()


def test_context_multiple_edge_radii() -> None:
    """all_edge_radii preserves order and contents."""
    radii = ((100_000.0, 'IEG'), (110_000.0, 'OEG'), (120_000.0, 'IER'))
    ctx = _make_context(all_edge_radii=radii)
    assert len(ctx.all_edge_radii) == 3
    assert ctx.all_edge_radii[0] == (100_000.0, 'IEG')
    assert ctx.all_edge_radii[2] == (120_000.0, 'IER')


# ---------------------------------------------------------------------------
# RingRenderResult
# ---------------------------------------------------------------------------


def _make_result(**kwargs: object) -> RingRenderResult:
    """Create a minimal RingRenderResult with sensible defaults.

    Omits ``edge_info_list`` from defaults so tests can exercise the dataclass
    field default; pass ``edge_info_list=...`` when a non-default list is needed.
    """
    defaults: dict[str, object] = {
        'model_img': np.zeros((10, 10), dtype=np.float64),
        'model_mask': np.zeros((10, 10), dtype=bool),
        'uncertainty': 1.5,
    }
    defaults.update(kwargs)
    return RingRenderResult(**defaults)  # type: ignore[arg-type]


def test_result_fields_set_correctly() -> None:
    """All constructor arguments appear as attributes."""
    img = np.ones((5, 5), dtype=np.float64)
    mask = np.ones((5, 5), dtype=bool)
    ei: list[tuple[np.ndarray, str, str]] = [(np.ones((5, 5), dtype=bool), 'label', 'IEG')]
    result = RingRenderResult(
        model_img=img,
        model_mask=mask,
        uncertainty=2.3,
        edge_info_list=ei,
    )
    assert result.model_img is img
    assert result.model_mask is mask
    assert result.uncertainty == pytest.approx(2.3)
    assert len(result.edge_info_list) == 1


@pytest.mark.parametrize(
    'uncertainty',
    [0.0, 1.0e100],
    ids=['zero', 'large'],
)
def test_result_uncertainty_edge_cases(uncertainty: float) -> None:
    """Uncertainty (km RMS) may be zero or very large if finite and non-negative."""
    result = _make_result(uncertainty=uncertainty)
    assert result.uncertainty == pytest.approx(uncertainty)


def test_result_negative_uncertainty_raises() -> None:
    """Negative uncertainty is rejected (RMS must be non-negative)."""
    with pytest.raises(ValueError, match='uncertainty'):
        _make_result(uncertainty=-0.5)


def test_result_edge_info_list_dataclass_default() -> None:
    """``RingRenderResult.edge_info_list`` defaults to ``[]`` when omitted.

    ``_make_result`` does not pass ``edge_info_list``, so the dataclass default
    factory runs (not an explicit empty list argument).
    """
    result = _make_result()
    assert result.edge_info_list == []


def test_result_is_not_frozen() -> None:
    """RingRenderResult is NOT frozen -- fields can be updated."""
    result = _make_result()
    # Should not raise
    result.uncertainty = 99.0
    assert result.uncertainty == pytest.approx(99.0)


def test_result_model_img_shape() -> None:
    """model_img shape is preserved when mask matches."""
    img = np.zeros((7, 13), dtype=np.float64)
    mask = np.zeros((7, 13), dtype=bool)
    result = _make_result(model_img=img, model_mask=mask)
    assert result.model_img.shape == (7, 13)
