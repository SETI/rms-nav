"""Tests for ``nav.support.status_reason.NavStatusReason``."""

import pytest

from nav.support.status_reason import NavStatusReason


@pytest.mark.parametrize(
    'name',
    [
        'OK',
        'RANK_1_ONLY',
        'CONFLICTED_TECHNIQUES',
        'NO_SIGNAL_IN_IMAGE',
        'IMAGE_OVEREXPOSED',
        'MISSING_DATA_DOMINANT',
        'IMAGE_CORRUPT',
        'KERNELS_UNAVAILABLE',
        'INSTRUMENT_NOT_CONFIGURED',
        'NO_FEATURES_EXTRACTED',
        'ALL_FEATURES_GATED',
        'NO_FEASIBLE_TECHNIQUES',
        'ALL_TECHNIQUES_SPURIOUS',
        'FINAL_CONFIDENCE_BELOW_THRESHOLD',
        'UNOBSERVABLE_OFFSET',
    ],
)
def test_navstatusreason_has_value(name: str) -> None:
    """The plan's full 15-value enumeration is present."""
    assert hasattr(NavStatusReason, name)


def test_navstatusreason_count_matches_plan() -> None:
    """Exactly 15 values are defined; adding a value must update tests."""
    assert len(list(NavStatusReason)) == 15


@pytest.mark.parametrize(
    ('member', 'expected_value'),
    [
        (NavStatusReason.OK, 'ok'),
        (NavStatusReason.RANK_1_ONLY, 'rank_1_only'),
        (NavStatusReason.UNOBSERVABLE_OFFSET, 'unobservable_offset'),
    ],
)
def test_navstatusreason_value_lowercase_snake_case(
    member: NavStatusReason, expected_value: str
) -> None:
    """Each value's string form is lowercase snake_case."""
    assert member.value == expected_value


def test_navstatusreason_distinct_values() -> None:
    """No two enum members share a string value."""
    values = [member.value for member in NavStatusReason]
    assert len(set(values)) == len(values)
