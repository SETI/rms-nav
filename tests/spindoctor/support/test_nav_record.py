"""The one domain every reader and the store share for a record's values.

These are the boundaries the parity between a document and its index row rests
on.  Two implementations that agree today are how the two storages came to
supply different pointing for one record, so the domain is pinned here as its
own contract rather than only through the consumers that call it.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.support.nav_record import (
    INVALID_OFFSET_TYPE,
    MALFORMED_OFFSET,
    MISSING_OFFSET_KEY,
    NON_FINITE_OFFSET,
    NULL_OFFSET,
    UNKNOWN_STATUS,
    finite_float,
    record_midtime_et,
    record_offset,
    record_rotation_matrix,
    record_status,
    record_status_error,
)

_ROTATION = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

_TOO_BIG_FOR_A_FLOAT = 10**400
"""A JSON integer literal no float can hold.

JSON puts no bound on an integer, and ``float()`` of one this size raises
rather than returning an infinity, so it is a value that reaches every reader
and is a number to none of them.
"""


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


@pytest.mark.parametrize(
    'value',
    [_TOO_BIG_FOR_A_FLOAT, -_TOO_BIG_FOR_A_FLOAT],
    ids=['positive', 'negative'],
)
def test_an_integer_too_large_for_a_float_is_read_as_nothing(value: int) -> None:
    """It is refused, not raised over, which is what the docstring promises.

    ``float()`` of such an integer raises ``OverflowError``, which is neither a
    ``TypeError`` nor a ``ValueError``; letting it out would turn every reader
    of a recorded number into one that can raise, from a function whose whole
    contract is to answer with a number or with nothing.

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
        ({'offset': [_TOO_BIG_FOR_A_FLOAT, 1.0]}, MALFORMED_OFFSET),
        ({'offset': [1.0, _TOO_BIG_FOR_A_FLOAT]}, MALFORMED_OFFSET),
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
        'huge-integer-first',
        'huge-integer-second',
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


@pytest.mark.parametrize(
    'value',
    [
        _ROTATION,
        [_ROTATION[0:3], _ROTATION[3:6], _ROTATION[6:9]],
        [[value] for value in _ROTATION],
        [
            [[value] for value in _ROTATION[0:3]],
            [[value] for value in _ROTATION[3:6]],
            [[value] for value in _ROTATION[6:9]],
        ],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ],
    ids=['row-major', 'nesting', 'one-element-rows', 'nested-one-element-rows', 'integers'],
)
def test_every_shape_one_matrix_can_be_made_of_is_read_as_that_matrix(value: Any) -> None:
    """Whatever an array library can reconcile into one 3x3 of numbers.

    The producer writes nine row-major floats and a 3x3 nesting denotes the
    same nine, but a reader that assembles an array accepts more than those
    two, and every one of them has to mean the same thing to the store: a
    rotation the reader applies and the store held nothing for is a corrected
    product through a document and an uncorrected one through a row.

    Parameters:
        value: The recorded value under test.
    """
    matrix = record_rotation_matrix(value)
    assert matrix is not None
    assert np.array_equal(matrix, np.asarray(_ROTATION).reshape(3, 3))


def test_the_matrix_is_read_as_float_whatever_the_record_wrote() -> None:
    """One dtype leaves the reader, so the store holds what the reader evaluates."""
    matrix = record_rotation_matrix([1, 0, 0, 0, 1, 0, 0, 0, 1])
    assert matrix is not None
    assert matrix.dtype == np.float64


def test_a_full_mantissa_value_survives_the_reading_exactly() -> None:
    """Assembling the array must not round: the reader's gate holds it to 1e-9."""
    values = [0.9636758075215185, *_ROTATION[1:]]
    matrix = record_rotation_matrix(values)
    assert matrix is not None
    assert matrix[0][0] == 0.9636758075215185


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
    assert record_rotation_matrix(value) is None


@pytest.mark.parametrize(
    'value',
    [
        [[1.0, 2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
        [[1.0, 2.0, 3.0], 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [_ROTATION[0:3], _ROTATION[0:3], _ROTATION[0:3], *_ROTATION[0:6]],
    ],
    ids=['a-row-of-two-among-eight', 'a-row-among-scalars', 'three-rows-among-six-scalars'],
)
def test_nine_entries_of_shapes_no_matrix_holds_are_read_as_nothing(value: Any) -> None:
    """Nine entries are not nine numbers, and assembling them raises.

    These pass the count and reach the array library, which refuses to make one
    homogeneous array of them.  That refusal is a malformed record like any
    other: letting it out would put a bare exception through a classifier whose
    callers absorb only what it declares.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_matrix(value) is None


@pytest.mark.parametrize(
    'value',
    [
        [str(value) for value in _ROTATION],
        ['1.0', *_ROTATION[1:]],
        [None, *_ROTATION[1:]],
        [{}, *_ROTATION[1:]],
        [_TOO_BIG_FOR_A_FLOAT, *_ROTATION[1:]],
    ],
    ids=['all-text', 'one-text', 'one-null', 'one-object', 'one-huge-integer'],
)
def test_nine_entries_that_are_not_all_numbers_are_read_as_nothing(value: Any) -> None:
    """A single non-number makes the whole assembled array something else.

    Text, nulls and objects each assemble into an array of a kind that is not a
    number, and an integer too large for a float does the same rather than
    raising, so all of them are refused by the one rule.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_matrix(value) is None


@pytest.mark.parametrize(
    'value',
    [
        [float('nan'), *_ROTATION[1:]],
        [float('inf'), *_ROTATION[1:]],
        [*_ROTATION[:8], float('-inf')],
    ],
    ids=['nan', 'inf', 'neg-inf'],
)
def test_a_non_finite_entry_makes_the_matrix_unreadable(value: Any) -> None:
    """NaN defeats every comparison a rotation check makes, so it never gets there.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_matrix(value) is None


def _rows_of_one(flat: list[Any]) -> list[list[Any]]:
    """Rewrite nine row-major values as nine rows of one.

    Parameters:
        flat: The nine values, row-major.

    Returns:
        Nine rows of one, a nesting the assembly reconciles into the same 3x3.
    """
    return [[value] for value in flat]


@pytest.mark.parametrize(
    'value',
    [
        [True] * 9,
        [True, *_ROTATION[1:]],
        [[True, False, False], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        _rows_of_one([True] * 9),
        _rows_of_one([True, *_ROTATION[1:]]),
        _rows_of_one(_rows_of_one([True, *_ROTATION[1:]])),
        [
            _rows_of_one([True, *_ROTATION[1:3]]),
            _rows_of_one(_ROTATION[3:6]),
            _rows_of_one(_ROTATION[6:9]),
        ],
    ],
    ids=[
        'all',
        'one-among-numbers',
        'nested',
        'rows-of-one-all',
        'rows-of-one-among-numbers',
        'rows-of-one-nested-twice',
        'three-rows-of-one-element-rows',
    ],
)
def test_a_boolean_anywhere_in_a_matrix_is_read_as_nothing(value: Any) -> None:
    """The one element type an assembled array would silently make a number of.

    An array library promotes a boolean beside a number to that number's type,
    so a single ``True`` among eight floats would assemble into ``1.0`` and
    nine of them into an identity rotation.  Every nesting the assembly
    reconciles into a 3x3 is covered, not only the two a record is written in:
    the deeper ones hand a container to any test of an entry's own type, and
    the promotion happens all the same.

    Parameters:
        value: The recorded value under test.
    """
    assert record_rotation_matrix(value) is None


def test_nine_finite_numbers_that_are_no_rotation_are_still_read() -> None:
    """Whether they are a rotation is the validator's question, not this one.

    The store holds these nine, and the validator both readers apply then
    refuses them identically; a reader that refused them here would leave the
    two storages disagreeing about which records exist at all.
    """
    assert record_rotation_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]) is not None


# ---------------------------------------------------------------------------
# One recorded exposure midtime
# ---------------------------------------------------------------------------


def test_a_recorded_midtime_is_read_as_the_number_it_records() -> None:
    """The ordinary case the refusals below are exceptions to."""
    assert record_midtime_et({'navigation_result': {'times': {'midtime_et': 1.5}}}) == 1.5


def test_a_recorded_midtime_of_zero_is_a_midtime() -> None:
    """J2000 itself is a perfectly good epoch, and is falsy in Python."""
    assert record_midtime_et({'navigation_result': {'times': {'midtime_et': 0}}}) == 0.0


@pytest.mark.parametrize(
    'midtime',
    [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='inf'),
        pytest.param(float('-inf'), id='minus-inf'),
        pytest.param(None, id='null'),
        pytest.param(True, id='boolean'),
        pytest.param('later', id='text'),
    ],
)
def test_a_value_no_reader_can_place_in_time_is_read_as_no_midtime(midtime: Any) -> None:
    """Every comparison against a NaN is False, so a NaN satisfies every range at once.

    An infinity is the same defect one step along: it falls inside any
    half-bounded range it can have no business in.  Both are read as no midtime
    rather than passed on, which is what makes a bounded selection able to say
    that it could not place the image.

    Parameters:
        midtime: The value recorded where a midtime belongs.
    """
    assert record_midtime_et({'navigation_result': {'times': {'midtime_et': midtime}}}) is None


def test_a_recorded_integer_no_float_can_hold_is_read_as_no_midtime() -> None:
    """JSON puts no bound on an integer literal, and asking whether one is finite raises.

    A midtime of several hundred digits is a value a reader cannot use rather
    than an error for a caller to meet.  Read as an error it would end a
    time-bounded stream over the documents on one malformed file, while the same
    file ingested into the index places no image at all -- so the two storages
    would disagree about a document neither of them can place.
    """
    recorded = {'navigation_result': {'times': {'midtime_et': 10**400}}}
    assert record_midtime_et(recorded) is None


@pytest.mark.parametrize(
    'metadata',
    [
        pytest.param({}, id='no-result'),
        pytest.param({'navigation_result': 'later'}, id='result-not-a-block'),
        pytest.param({'navigation_result': {}}, id='no-times'),
        pytest.param({'navigation_result': {'times': 'later'}}, id='times-not-a-block'),
    ],
)
def test_a_record_with_no_usable_times_records_no_midtime(metadata: dict[str, Any]) -> None:
    """A load-error record records no exposure, and each way of recording none counts.

    Parameters:
        metadata: A record holding nothing a midtime can be read out of.
    """
    assert record_midtime_et(metadata) is None
