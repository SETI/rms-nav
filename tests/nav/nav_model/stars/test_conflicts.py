"""Tests for ``nav.nav_model.stars.conflicts``."""

from __future__ import annotations

from typing import Any, cast

import pytest

from nav.nav_model.stars.conflicts import _conflict_body_list, parse_ring_occlusion_annuli


def test_parse_ring_occlusion_annuli_normalises_keys() -> None:
    """Planet keys are normalised to upper case in the returned mapping."""
    result = parse_ring_occlusion_annuli({'saturn': [[100.0, 200.0]]})
    assert list(result.keys()) == ['SATURN']
    assert result['SATURN'] == [(100.0, 200.0)]


def test_parse_ring_occlusion_annuli_returns_empty_for_none() -> None:
    """A ``None`` input is treated as the empty mapping."""
    assert parse_ring_occlusion_annuli(None) == {}


def test_parse_ring_occlusion_annuli_returns_empty_for_empty() -> None:
    """An empty mapping returns an empty mapping."""
    assert parse_ring_occlusion_annuli({}) == {}


def test_parse_ring_occlusion_annuli_rejects_non_sequence_pair() -> None:
    """A scalar where a length-2 sequence was expected raises ``ValueError``."""
    bad: Any = {'SATURN': [123.0]}
    with pytest.raises(ValueError, match='must be a length-2 sequence'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_wrong_length() -> None:
    """A length-3 sequence raises ``ValueError`` with the actual length."""
    with pytest.raises(ValueError, match='must have exactly 2 elements'):
        parse_ring_occlusion_annuli({'SATURN': [[1.0, 2.0, 3.0]]})


def test_parse_ring_occlusion_annuli_rejects_non_numeric() -> None:
    """A non-numeric inner radius raises ``ValueError`` naming it."""
    bad: Any = {'SATURN': [['inner', 200.0]]}
    with pytest.raises(ValueError, match='must be numeric'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_inner_ge_outer() -> None:
    """Degenerate annuli (inner >= outer) raise ``ValueError``."""
    with pytest.raises(ValueError, match=r'inner 100\.0 km >= outer 100\.0 km'):
        parse_ring_occlusion_annuli({'SATURN': [[100.0, 100.0]]})


def test_parse_ring_occlusion_annuli_rejects_bool() -> None:
    """A bool is rejected even though Python admits it as int."""
    bad: Any = {'SATURN': [[True, 200.0]]}
    with pytest.raises(ValueError, match='got bool'):
        parse_ring_occlusion_annuli(bad)


def test_parse_ring_occlusion_annuli_rejects_inf() -> None:
    """Non-finite radii raise ``ValueError`` naming the bad value."""
    with pytest.raises(ValueError, match='must be finite'):
        parse_ring_occlusion_annuli({'SATURN': [[float('inf'), 200.0]]})


class _FakeObsWithSaturn:
    """Stand-in for an observation whose closest planet is Saturn."""

    closest_planet = 'SATURN'


class _FakeObsNoPlanet:
    """Stand-in for an observation with no closest planet."""

    closest_planet: str | None = None


class _FakeSaturnConfig:
    """Stand-in for the project config exposing only ``satellites``."""

    @staticmethod
    def satellites(planet: str) -> list[str]:
        """Return Saturn's satellites in canonical order."""
        return ['MIMAS', 'TETHYS'] if planet == 'SATURN' else []


class _FakeEmptyConfig:
    """Stand-in for a config that knows about no satellites."""

    @staticmethod
    def satellites(planet: str) -> list[str]:
        """Return an empty list for any planet."""
        del planet
        return []


def test_conflict_body_list_includes_planet_and_satellites() -> None:
    """``_conflict_body_list`` returns the planet plus its satellites."""
    out = _conflict_body_list(cast(Any, _FakeObsWithSaturn()), cast(Any, _FakeSaturnConfig()))
    assert out == ['SATURN', 'MIMAS', 'TETHYS']


def test_conflict_body_list_returns_empty_when_no_closest_planet() -> None:
    """When ``obs.closest_planet`` is ``None`` the body list is empty."""
    out = _conflict_body_list(cast(Any, _FakeObsNoPlanet()), cast(Any, _FakeEmptyConfig()))
    assert out == []
