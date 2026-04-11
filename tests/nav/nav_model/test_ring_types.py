"""Tests for ``RingBaseOrbitMode`` and ``RingPerturbationMode`` validation."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from nav.nav_model.rings.ring_types import RingBaseOrbitMode, RingPerturbationMode


def test_ring_base_orbit_mode_rejects_bool_a() -> None:
    """``RingBaseOrbitMode.__post_init__`` rejects bool for numeric orbit fields."""
    with pytest.raises(TypeError, match=r'RingBaseOrbitMode\.a must be a real number'):
        RingBaseOrbitMode(a=True, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)


def test_ring_base_orbit_mode_rejects_non_numeric_ae() -> None:
    """Non-real ``ae`` raises TypeError before range checks."""
    with pytest.raises(TypeError, match=r'RingBaseOrbitMode\.ae must be a real number'):
        RingBaseOrbitMode(
            a=1.0,
            ae='x',  # type: ignore[arg-type]
            long_peri=0.0,
            rate_peri=0.0,
            rms=0.0,
        )


def test_ring_perturbation_mode_rejects_non_finite_phase() -> None:
    """``phase`` must be finite so ``parsed_modes_for_backplane`` stays safe."""
    with pytest.raises(ValueError, match=r'RingPerturbationMode\.phase must be a finite'):
        RingPerturbationMode(mode_num=2, amplitude=1.0, phase=float('nan'), pattern_speed=0.0)


def test_ring_perturbation_mode_rejects_bool_pattern_speed() -> None:
    """``pattern_speed`` rejects bool despite ``bool`` being a subclass of ``int``."""
    with pytest.raises(
        ValueError, match=r'RingPerturbationMode\.pattern_speed must be int or float'
    ):
        RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=True)


def test_ring_perturbation_mode_accepts_finite_phase_and_speed() -> None:
    """Typical finite values construct successfully."""
    m = RingPerturbationMode(mode_num=2, amplitude=1.0, phase=-180.0, pattern_speed=math.pi)
    assert m.phase == -180.0
    assert m.pattern_speed == math.pi


def test_base_orbit_mode_is_frozen() -> None:
    """``RingBaseOrbitMode`` rejects field assignment after construction."""
    m = RingBaseOrbitMode(a=1.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)
    with pytest.raises(AttributeError) as exc_info:
        m.a = 2.0  # type: ignore[misc]
    assert isinstance(exc_info.value, FrozenInstanceError)
    msg = str(exc_info.value)
    assert 'cannot assign' in msg
    assert "field 'a'" in msg


def test_ring_perturbation_mode_is_frozen() -> None:
    """``RingPerturbationMode`` rejects field assignment after construction."""
    p = RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=0.0)
    with pytest.raises(AttributeError) as exc_info:
        p.mode_num = 3  # type: ignore[misc]
    assert isinstance(exc_info.value, FrozenInstanceError)
    msg = str(exc_info.value)
    assert 'cannot assign' in msg
    assert "field 'mode_num'" in msg


@pytest.mark.parametrize('mode_num', ['a', 3.5], ids=['str_mode', 'float_mode'])
def test_perturbation_mode_invalid_mode_num(mode_num: Any) -> None:
    """``RingPerturbationMode`` rejects non-int ``mode_num`` (constructor)."""
    with pytest.raises(ValueError, match=r'RingPerturbationMode\.mode_num must be int'):
        RingPerturbationMode(
            mode_num=mode_num,
            amplitude=1.0,
            phase=0.0,
            pattern_speed=0.0,
        )


def test_perturbation_mode_non_numeric_amplitude() -> None:
    """``RingPerturbationMode`` rejects non-numeric ``amplitude`` (constructor)."""
    with pytest.raises(ValueError, match=r'RingPerturbationMode\.amplitude must be'):
        RingPerturbationMode(
            mode_num=2,
            amplitude='x',  # type: ignore[arg-type]
            phase=0.0,
            pattern_speed=0.0,
        )
