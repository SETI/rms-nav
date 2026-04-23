"""Tests for ``nav.ui.common``."""

import os
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

# If this file is collected without ``tests/nav/ui/conftest.py`` (unusual), still
# prefer headless Qt before importing bindings.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFormLayout
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(
        f'PyQt6 QApplication init failed (display/EGL/platform): {exc}',
        allow_module_level=True,
    )

try:
    from nav.ui.common import apply_linear_gamma_stretch, build_stretch_controls
except (ImportError, OSError) as exc:
    pytest.skip(
        f'nav.ui.common import failed (Qt dependency): {exc}',
        allow_module_level=True,
    )


@pytest.fixture
def qapp() -> QApplication:
    """Ensure a ``QApplication`` exists for widget construction."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


def _stretch_callbacks() -> tuple[
    Callable[[float], None],
    Callable[[float], None],
    Callable[[float], None],
]:
    def _black(_x: float) -> None:
        pass

    def _white(_x: float) -> None:
        pass

    def _gamma(_x: float) -> None:
        pass

    return _black, _white, _gamma


def _base_stretch_kwargs() -> dict[str, Any]:
    on_b, on_w, on_g = _stretch_callbacks()
    return {
        'img_min': 0.0,
        'img_max': 1.0,
        'black_init': 0.1,
        'white_init': 0.9,
        'gamma_init': 1.0,
        'on_black_changed': on_b,
        'on_white_changed': on_w,
        'on_gamma_changed': on_g,
    }


def test_build_stretch_controls_accepts_valid_numeric_args(qapp: QApplication) -> None:
    """Typical finite numeric arguments construct sliders and labels."""
    form = QFormLayout()
    result = build_stretch_controls(form, **_base_stretch_kwargs())
    assert result['slider_black'].value() == 100
    assert result['slider_white'].value() == 900


def test_build_stretch_controls_rejects_bool_black_init(qapp: QApplication) -> None:
    """``bool`` is rejected even though it is a subclass of ``int``."""
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['black_init'] = True
    with pytest.raises(TypeError, match='black_init must be int or float, not bool'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_numeric_img_max(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['img_max'] = 'bad'
    with pytest.raises(TypeError, match='img_max must be int or float, not str'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_finite_gamma_init(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['gamma_init'] = float('inf')
    with pytest.raises(ValueError, match=r'gamma_init must be a finite number'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_bool_value_label_min_width(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['value_label_min_width'] = True
    with pytest.raises(TypeError, match='value_label_min_width must be int, not bool'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_positive_value_label_min_width(
    qapp: QApplication,
) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['value_label_min_width'] = 0
    with pytest.raises(ValueError, match='value_label_min_width must be > 0, got 0'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_negative_slider_horizontal_stretch(
    qapp: QApplication,
) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['slider_horizontal_stretch'] = -1
    with pytest.raises(ValueError, match='slider_horizontal_stretch must be >= 0, got -1'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_allows_degenerate_img_range(qapp: QApplication) -> None:
    """``img_max <= img_min`` still uses internal ``img_min + 1.0`` upper bound."""
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['img_min'] = 5.0
    kw['img_max'] = 5.0
    kw['black_init'] = 5.0
    kw['white_init'] = 6.0
    result = build_stretch_controls(form, **kw)
    assert result['slider_black'].value() == 0
    assert result['slider_white'].value() == 1000


# ---------------------------------------------------------------------------
# apply_linear_gamma_stretch
# ---------------------------------------------------------------------------


def test_apply_linear_gamma_stretch_linear_gamma_one() -> None:
    """gamma=1.0 is a simple linear normalisation."""
    data = np.array([0.0, 0.5, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


def test_apply_linear_gamma_stretch_clips_below_black() -> None:
    data = np.array([-1.0, 0.0, 0.5])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.5])


def test_apply_linear_gamma_stretch_clips_above_white() -> None:
    data = np.array([0.5, 1.0, 2.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result, [0.5, 1.0, 1.0])


def test_apply_linear_gamma_stretch_gamma_half_brightens_midtones() -> None:
    """gamma=0.5 maps 0.25 -> 0.5 (brightens relative to linear)."""
    data = np.array([0.0, 0.25, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=0.5)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-7)


def test_apply_linear_gamma_stretch_gamma_two_darkens_midtones() -> None:
    """gamma=2.0 maps 0.5 -> 0.25 (darkens relative to linear)."""
    data = np.array([0.0, 0.5, 1.0])
    result = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=2.0)
    np.testing.assert_allclose(result, [0.0, 0.25, 1.0], atol=1e-7)


def test_apply_linear_gamma_stretch_degenerate_range() -> None:
    """When white <= black, range is clamped to black + 1e-6; all non-black values -> 1."""
    data = np.array([5.0, 5.0, 6.0])
    result = apply_linear_gamma_stretch(data, black=5.0, white=5.0, gamma=1.0)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_apply_linear_gamma_stretch_non_positive_gamma_treated_as_one() -> None:
    """gamma <= 0 is treated as 1.0."""
    data = np.array([0.0, 0.5, 1.0])
    result_bad = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=0.0)
    result_ref = apply_linear_gamma_stretch(data, black=0.0, white=1.0, gamma=1.0)
    np.testing.assert_allclose(result_bad, result_ref)
