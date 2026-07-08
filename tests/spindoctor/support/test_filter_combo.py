"""Tests for ``spindoctor.support.filter_combo.canonicalize``."""

import pytest

from spindoctor.support.filter_combo import canonicalize


@pytest.mark.parametrize(
    ('inputs', 'expected'),
    [
        ([], 'NONE'),
        ([None], 'NONE'),
        ([None, None], 'NONE'),
        (['CL1'], 'CL1'),
        (['CL2', 'CL1'], 'CL1+CL2'),
        (['CL', 'CL'], 'CL+CL'),
        (['F1', None, 'F2'], 'F1+F2'),
        (['F3', 'F2', 'F1'], 'F1+F2+F3'),
        ([None, 'CL'], 'CL'),
    ],
)
def test_canonicalize(inputs: list[str | None], expected: str) -> None:
    """Canonical form drops None, sorts, joins with '+'."""
    assert canonicalize(inputs) == expected
