"""The one domain every reader and the store share for a record's values.

These are the boundaries the parity between a document and its index row rests
on.  Two implementations that agree today are how the two storages came to
supply different pointing for one record, so the domain is pinned here as its
own contract rather than only through the consumers that call it.
"""

from typing import Any

import pytest

from spindoctor.support.nav_record import (
    INVALID_OFFSET_TYPE,
    MALFORMED_OFFSET,
    MISSING_OFFSET_KEY,
    NON_FINITE_OFFSET,
    NULL_OFFSET,
    UNKNOWN_STATUS,
    finite_float,
    record_offset,
    record_rotation_values,
    record_status,
    record_status_error,
)

_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# One recorded number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(1.5, 1.5), (-2, -2.0), (0, 0.0)],
    ids=['float', 'negative-int', 'zero'],
)
def test_a_recorded_number_is_read_as_a_float(value: Any, expected: float) -> None:
    """An integer is a number a reader can use, and is read as one.

    Parameters:
        value: The recorded value.
        expected: The float it denotes.
    """
    assert finite_float(value) == expected


@pytest.mark.parametrize(
    'value',
    [None, True, False, float('nan'), float('inf'), float('-inf'), '1.5', [1.5], {}],
    ids=['none', 'true', 'false', 'nan', 'inf', 'neg-inf', 'text', 'list', 'object'],
)
def test_a_value_that_is_not_a_finite_number_is_read_as_nothing(value: Any) -> None:
    """Booleans included: an ``int`` in Python is a measurement nowhere.

    Parameters:
        value: The recorded value.
    """
    assert finite_float(value) is None


# ---------------------------------------------------------------------------
# The outcome a record names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'status', ['success', 'failed', 'conflicted', 'error', 'unknown'], ids=lambda s: str(s)
)
def test_a_record_naming_an_outcome_is_read_as_naming_it(status: str) -> None:
    """Including the word a record naming none is read as, which is not special.

    Parameters:
        status: The recorded outcome.
    """
    assert record_status({'status': status}) == status


@pytest.mark.parametrize(
    'record',
    [{}, {'status': None}, {'status': ''}, {'status': 42}, {'status': ['success']}],
    ids=['absent', 'null', 'empty', 'number', 'list'],
)
def test_a_record_naming_no_outcome_is_read_as_naming_none(record: dict[str, Any]) -> None:
    """Every way of naming no outcome reads as the same one word.

    Parameters:
        record: The record under test.
    """
    assert record_status(record) == UNKNOWN_STATUS


def test_the_nested_copy_of_the_status_does_not_stand_in_for_the_field() -> None:
    """The ladder's first question is the record's own field, and only that one.

    A reader that borrowed the nested copy would apply a corrected pointing to
    a record whose own field supplies none.
    """
    assert record_status({'navigation_result': {'status': 'success'}}) == UNKNOWN_STATUS


@pytest.mark.parametrize(
    'record',
    [{}, {'status_error': None}, {'status_error': ''}, {'status_error': 42}],
    ids=['absent', 'null', 'empty', 'number'],
)
def test_a_record_naming_no_error_is_read_as_naming_none(record: dict[str, Any]) -> None:
    """An absent error field and one holding null both name no error.

    Parameters:
        record: The record under test.
    """
    assert record_status_error(record) == UNKNOWN_STATUS


def test_a_record_naming_an_error_is_read_as_naming_it() -> None:
    """The control for the defaults above, which an empty reader would pass."""
    assert record_status_error({'status_error': 'missing_spice_data'}) == 'missing_spice_data'


# ---------------------------------------------------------------------------
# The offset a record supplies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('offset', 'pair'),
    [
        ([1.5, -2.5], (1.5, -2.5)),
        ([1, -2], (1.0, -2.0)),
        (['1.5', '-2.5'], (1.5, -2.5)),
        ((1.5, -2.5), (1.5, -2.5)),
    ],
    ids=['floats', 'integers', 'numeric-strings', 'tuple'],
)
def test_a_usable_offset_is_read_as_its_two_numbers(offset: Any, pair: tuple[float, float]) -> None:
    """Whatever a reader can convert to two finite pixels, it applies.

    Parameters:
        offset: The recorded value.
        pair: The ``(dv, du)`` it denotes.
    """
    assert record_offset({'offset': offset}).pair == pair


@pytest.mark.parametrize(
    ('record', 'reason'),
    [
        ({}, MISSING_OFFSET_KEY),
        ({'offset': None}, NULL_OFFSET),
        ({'offset': [True, False]}, INVALID_OFFSET_TYPE),
        ({'offset': [1.0, True]}, INVALID_OFFSET_TYPE),
        ({'offset': [float('nan'), 1.0]}, NON_FINITE_OFFSET),
        ({'offset': [1.0, float('inf')]}, NON_FINITE_OFFSET),
        ({'offset': [1.0, 2.0, 3.0]}, MALFORMED_OFFSET),
        ({'offset': [1.0]}, MALFORMED_OFFSET),
        ({'offset': []}, MALFORMED_OFFSET),
        ({'offset': [1.0, None]}, MALFORMED_OFFSET),
        ({'offset': '1.0,2.0'}, MALFORMED_OFFSET),
        ({'offset': 1.5}, MALFORMED_OFFSET),
        ({'offset': {'dv': 1.0, 'du': 2.0}}, MALFORMED_OFFSET),
    ],
    ids=[
        'absent',
        'null',
        'booleans',
        'one-boolean',
        'nan',
        'inf',
        'three',
        'one',
        'empty',
        'null-element',
        'text',
        'scalar',
        'object',
    ],
)
def test_an_unusable_offset_is_read_as_nothing_and_says_why(
    record: dict[str, Any], reason: str
) -> None:
    """Each way of failing to be a pair has its own name for the run's tally.

    Parameters:
        record: The record under test.
        reason: The name the shortfall is reported under.
    """
    assert record_offset(record).reason == reason


@pytest.mark.parametrize(
    'record',
    [{}, {'offset': None}, {'offset': [1.0, 2.0, 3.0]}, {'offset': [True, False]}],
    ids=['absent', 'null', 'three', 'booleans'],
)
def test_an_unusable_offset_supplies_no_pair_at_all(record: dict[str, Any]) -> None:
    """Never part of one: two of three recorded numbers are nobody's measurement.

    Parameters:
        record: The record under test.
    """
    assert record_offset(record).pair is None


def test_a_usable_offset_carries_no_reason() -> None:
    """The control: a reason beside a pair would be a shortfall that is not one."""
    assert record_offset({'offset': [1.5, -2.5]}).reason is None


# ---------------------------------------------------------------------------
# The shape a recorded rotation is written in
# ---------------------------------------------------------------------------


def test_nine_row_major_values_are_read_as_they_were_written() -> None:
    """The shape the producer writes, kept in order."""
    assert record_rotation_values(_ROTATION) == _ROTATION


def test_a_nesting_is_read_as_the_nine_it_denotes() -> None:
    """Three rows of three are the same nine values in the same order."""
    nested = [_ROTATION[0:3], _ROTATION[3:6], _ROTATION[6:9]]
    assert record_rotation_values(nested) == _ROTATION


def test_the_values_are_returned_as_recorded_rather_than_coerced() -> None:
    """Whether they are real, finite numbers is the validator's question.

    Coercing here would answer it twice, and the second answer is the one that
    drifts.
    """
    assert record_rotation_values([1, 0, 0, 0, 1, 0, 0, 0, 1]) == [1, 0, 0, 0, 1, 0, 0, 0, 1]


@pytest.mark.parametrize(
    'value',
    [
        _ROTATION[:8],
        [*_ROTATION, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'not a matrix',
        b'not a matrix',
        1.0,
        None,
        {'row0': [1.0, 0.0, 0.0]},
    ],
    ids=[
        'eight',
        'ten',
        'ragged',
        'two-rows',
        'text',
        'bytes',
        'scalar',
        'none',
        'object',
    ],
)
def test_anything_that_is_neither_shape_is_read_as_nothing(value: Any) -> None:
    """A ragged nesting included, which handed to an array library raises.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_values(value) is None


@pytest.mark.parametrize(
    'value',
    [[True] * 9, [True, *_ROTATION[1:]], [[True, False, False], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
    ids=['all', 'one-among-numbers', 'nested'],
)
def test_a_boolean_anywhere_in_a_matrix_is_read_as_nothing(value: Any) -> None:
    """The one element type the reader and the store would otherwise judge apart.

    An array library promotes a boolean beside a number to that number's type,
    so a single ``True`` among eight floats reads as ``1.0`` from a document
    while a column refusing booleans holds nothing at all.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_values(value) is None
