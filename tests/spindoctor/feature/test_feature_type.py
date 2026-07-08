"""Tests for ``spindoctor.feature.feature_type.NavFeatureType``."""

import pytest

from spindoctor.feature.feature_type import NavFeatureType


@pytest.mark.parametrize(
    'name',
    [
        'STAR',
        'LIMB_ARC',
        'TERMINATOR_ARC',
        'BODY_DISC',
        'BODY_BLOB',
        'RING_EDGE',
        'RING_ANNULUS',
        'TITAN_LIMB',
        'CARTOGRAPHIC_MODEL',
    ],
)
def test_navfeaturetype_has_value(name: str) -> None:
    """The plan's full 9-value enumeration is present."""
    assert hasattr(NavFeatureType, name)


def test_navfeaturetype_count_matches_plan() -> None:
    """Exactly 9 values are defined."""
    assert len(list(NavFeatureType)) == 9


def test_navfeaturetype_distinct_values() -> None:
    """No two enum members share a string value."""
    values = [member.value for member in NavFeatureType]
    assert len(set(values)) == len(values)
