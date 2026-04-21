"""Tests for ``RingBaseOrbitMode``, ``RingPerturbationMode``, and ``RingEdgeData``."""

import math
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from nav.nav_model.rings.ring_types import (
    RingBaseOrbitMode,
    RingEdgeData,
    RingPerturbationMode,
)


def test_ring_base_orbit_mode_rejects_bool_a() -> None:
    """``RingBaseOrbitMode.__post_init__`` rejects bool for numeric orbit fields."""
    with pytest.raises(TypeError, match=r'RingBaseOrbitMode\.a must be a real number'):
        RingBaseOrbitMode(a=True, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)


def test_ring_base_orbit_mode_rejects_non_finite_long_peri() -> None:
    """``long_peri`` must be finite."""
    with pytest.raises(TypeError, match=r'RingBaseOrbitMode\.long_peri must be a finite'):
        RingBaseOrbitMode(
            a=1.0,
            ae=0.0,
            long_peri=float('nan'),
            rate_peri=0.0,
            rms=0.0,
        )


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


def test_ring_perturbation_mode_rejects_non_finite_amplitude() -> None:
    """``amplitude`` must be finite."""
    with pytest.raises(ValueError, match=r'RingPerturbationMode\.amplitude must be a finite'):
        RingPerturbationMode(mode_num=2, amplitude=float('nan'), phase=0.0, pattern_speed=0.0)


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


def _sample_base_orbit() -> RingBaseOrbitMode:
    return RingBaseOrbitMode(a=1.0, ae=0.0, long_peri=0.0, rate_peri=0.0, rms=0.0)


def _sample_perturbation() -> RingPerturbationMode:
    return RingPerturbationMode(mode_num=2, amplitude=1.0, phase=0.0, pattern_speed=0.0)


def test_ring_edge_data_none_base_orbit_raises() -> None:
    """``RingEdgeData`` rejects ``None`` for ``base_orbit``."""
    with pytest.raises(ValueError, match=r'RingEdgeData: base_orbit must not be None'):
        RingEdgeData(base_orbit=None, perturbations=())  # type: ignore[arg-type]


def test_ring_edge_data_wrong_base_orbit_type_raises() -> None:
    """``RingEdgeData`` requires ``RingBaseOrbitMode`` for ``base_orbit``."""
    with pytest.raises(
        TypeError,
        match=r'RingEdgeData: base_orbit must be an instance of RingBaseOrbitMode',
    ):
        RingEdgeData(
            base_orbit='not an orbit',  # type: ignore[arg-type]
            perturbations=(),
        )


def test_ring_edge_data_none_perturbations_raises() -> None:
    """``RingEdgeData`` rejects ``None`` for ``perturbations``."""
    with pytest.raises(ValueError, match=r'RingEdgeData: perturbations must not be None'):
        RingEdgeData(base_orbit=_sample_base_orbit(), perturbations=None)  # type: ignore[arg-type]


def test_ring_edge_data_list_perturbations_becomes_tuple() -> None:
    """A list of perturbations is normalized to a ``tuple``."""
    p = _sample_perturbation()
    edge = RingEdgeData(
        base_orbit=_sample_base_orbit(),
        perturbations=[p],  # type: ignore[arg-type]
    )
    assert isinstance(edge.perturbations, tuple)
    assert edge.perturbations == (p,)


def test_ring_edge_data_perturbation_wrong_element_type_raises() -> None:
    """Each perturbation entry must be a ``RingPerturbationMode``."""
    with pytest.raises(
        TypeError,
        match=r'RingEdgeData: perturbations\[0\] must be RingPerturbationMode',
    ):
        RingEdgeData(
            base_orbit=_sample_base_orbit(),
            perturbations=(1.0,),  # type: ignore[arg-type]
        )


def test_ring_edge_data_perturbations_str_raises() -> None:
    """``str`` is not a valid ``perturbations`` sequence."""
    with pytest.raises(
        TypeError,
        match=r'RingEdgeData: perturbations must be a sequence of RingPerturbationMode',
    ):
        RingEdgeData(
            base_orbit=_sample_base_orbit(),
            perturbations='bad',  # type: ignore[arg-type]
        )
