"""Tests for ``nav.nav_technique.confidence_config.load_confidence_spec``."""

from __future__ import annotations

from typing import Any

import pytest

from nav.config.config import Config
from nav.nav_technique.confidence import ConfidenceSpec, ConfidenceTerm
from nav.nav_technique.confidence_config import (
    ConfidenceConfigError,
    load_confidence_spec,
    load_technique_tuning,
)


def test_load_confidence_spec_returns_spec_for_shipped_technique() -> None:
    """The bundled ``config_510_techniques.yaml`` builds a spec for every shipped technique."""
    techniques = dict(Config().category('techniques'))
    spec = load_confidence_spec(techniques, 'BodyDiscCorrelateNav')
    assert isinstance(spec, ConfidenceSpec)
    assert spec.alpha0 == pytest.approx(-2.0)
    feature_names = [t.feature for t in spec.terms]
    assert 'ncc_peak' in feature_names
    assert spec.hard_zero_if == {'at_edge': True, 'spurious': True}


def test_load_confidence_spec_loads_hard_cap_when_present() -> None:
    """``BodyBlobNav`` declares ``hard_cap = 0.4`` in the bundled YAML."""
    techniques = dict(Config().category('techniques'))
    spec = load_confidence_spec(techniques, 'BodyBlobNav')
    assert spec.hard_cap == pytest.approx(0.4)


def test_load_confidence_spec_loads_default_term_offsets() -> None:
    """A term that omits ``offset`` / ``divisor`` / ``cap_at`` gets sensible defaults."""
    techniques = dict(Config().category('techniques'))
    spec = load_confidence_spec(techniques, 'BodyLimbNav')
    visible_arc_fraction = next(t for t in spec.terms if t.feature == 'visible_limb_arc_fraction')
    assert visible_arc_fraction.offset == 0.0
    assert visible_arc_fraction.divisor == 1.0
    assert visible_arc_fraction.cap_at is None


def test_load_confidence_spec_raises_on_unknown_technique() -> None:
    """An unknown technique name surfaces with a descriptive error."""
    techniques: dict[str, Any] = {'KnownNav': {'alpha0': -1.0, 'terms': []}}
    with pytest.raises(ConfidenceConfigError, match=r"missing block for technique 'UnknownNav'"):
        load_confidence_spec(techniques, 'UnknownNav')


def test_load_confidence_spec_raises_on_missing_alpha0() -> None:
    """A block without ``alpha0`` fails validation at load time."""
    techniques: dict[str, Any] = {'BadNav': {'terms': []}}
    with pytest.raises(ConfidenceConfigError, match=r'missing required key alpha0'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_unknown_block_keys() -> None:
    """Typos in block keys surface as unknown-keys errors."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'halfd_zero_if': {'at_edge': True},
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'unknown keys'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_unknown_term_keys() -> None:
    """Typos in term keys surface as unknown-keys errors."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [{'feature': 'x', 'alpha': 1.0, 'divsor': 2.0}],
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'unknown keys'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_non_finite_alpha() -> None:
    """Non-finite numeric values fail with a typed message."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [{'feature': 'x', 'alpha': float('inf')}],
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be finite'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_zero_divisor() -> None:
    """``ConfidenceTerm.divisor`` must be non-zero (re-raised as ConfidenceConfigError)."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [{'feature': 'x', 'alpha': 1.0, 'divisor': 0.0}],
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'divisor must be non-zero'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_constructs_terms_correctly() -> None:
    """A minimal block round-trips into a ConfidenceSpec with the expected terms."""
    techniques: dict[str, Any] = {
        'GoodNav': {
            'alpha0': 0.5,
            'terms': [
                {
                    'feature': 'a',
                    'alpha': 2.0,
                    'offset': 1.0,
                    'divisor': 4.0,
                    'cap_at': 0.8,
                }
            ],
            'hard_zero_if': {'at_edge': True},
            'hard_cap': 0.9,
        }
    }
    spec = load_confidence_spec(techniques, 'GoodNav')
    assert spec.alpha0 == pytest.approx(0.5)
    assert spec.terms == (
        ConfidenceTerm(feature='a', alpha=2.0, offset=1.0, divisor=4.0, cap_at=0.8),
    )
    assert spec.hard_zero_if == {'at_edge': True}
    assert spec.hard_cap == pytest.approx(0.9)


def test_load_confidence_spec_rejects_non_mapping_block() -> None:
    """A scalar where a mapping is expected fails with a typed error."""
    techniques: dict[str, Any] = {'BadNav': 'not a mapping'}
    with pytest.raises(ConfidenceConfigError, match=r'expected a mapping'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_bool_alpha() -> None:
    """``True`` / ``False`` are not accepted as numeric alpha values."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [{'feature': 'x', 'alpha': True}],
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be numeric, got bool'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_confidence_spec_rejects_non_bool_hard_zero() -> None:
    """``hard_zero_if`` values must be bool, not int 0/1."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'hard_zero_if': {'at_edge': 1},
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be bool'):
        load_confidence_spec(techniques, 'BadNav')


def test_load_technique_tuning_returns_shipped_block() -> None:
    """``BodyLimbNav.tuning`` ships with the documented placeholders."""
    techniques = dict(Config().category('techniques'))
    tuning = load_technique_tuning(techniques, 'BodyLimbNav')
    assert tuning['min_arc_vertices'] == pytest.approx(30.0)
    assert tuning['spurious_dt_rms_factor'] == pytest.approx(5.0)
    assert tuning['spurious_dt_floor_px'] == pytest.approx(3.0)
    assert tuning['spurious_min_inliers'] == 6
    assert tuning['spurious_min_inlier_fraction'] == pytest.approx(0.20)


def test_load_technique_tuning_returns_empty_when_no_tuning_block() -> None:
    """Missing ``tuning`` returns an empty dict, not an error."""
    techniques: dict[str, Any] = {'GoodNav': {'alpha0': -1.0, 'terms': []}}
    assert load_technique_tuning(techniques, 'GoodNav') == {}


def test_load_technique_tuning_returns_empty_when_technique_missing() -> None:
    """Test-only techniques opt out by name; the loader returns empty."""
    assert load_technique_tuning({}, 'UnknownNav') == {}


def test_load_technique_tuning_rejects_bool_value() -> None:
    """``True`` / ``False`` are not accepted as numeric tuning values."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'tuning': {'foo': True},
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be numeric, got bool'):
        load_technique_tuning(techniques, 'BadNav')


def test_load_technique_tuning_rejects_non_numeric_value() -> None:
    """A string value (or other non-numeric) is rejected."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'tuning': {'foo': 'bar'},
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be numeric'):
        load_technique_tuning(techniques, 'BadNav')


def test_load_technique_tuning_rejects_non_finite_value() -> None:
    """Non-finite numeric values fail with a typed message."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'tuning': {'foo': float('inf')},
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'must be finite'):
        load_technique_tuning(techniques, 'BadNav')


def test_load_technique_tuning_rejects_non_mapping_block() -> None:
    """A list where a mapping is expected fails with a typed error."""
    techniques: dict[str, Any] = {
        'BadNav': {
            'alpha0': -1.0,
            'terms': [],
            'tuning': [1, 2, 3],
        }
    }
    with pytest.raises(ConfidenceConfigError, match=r'expected a mapping'):
        load_technique_tuning(techniques, 'BadNav')
