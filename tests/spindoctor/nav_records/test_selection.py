"""What a caller may ask a record source for, and what it may not.

A selection is a value, so what is tested here is what it accepts and what it
refuses at the moment it is written, rather than what any source does with it.
Every refusal is here rather than in the two backends, and that is what makes
them agree: a walk and a query cannot be made to refuse alike by writing the
same refusal twice, so there is one place a selection can be wrong and neither
of them is reached from it.

What each refusal is for, since none of them is a type check for its own sake:
a swapped range selects nothing, and a run that read nothing for that reason is
indistinguishable from a clean run over a quiet span; a NaN bound is a number no
comparison is true of, so the walk keeps every record and the query keeps none;
a value of the wrong type reaches a database layer that refuses it in a
vocabulary the caller cannot name; and a key that is not a key names a file
under another root, or under none.
"""

from typing import Any

import pytest

from spindoctor.nav_records import Selection


def test_a_selection_naming_nothing_narrows_nothing() -> None:
    """The empty selection covers every record the source holds."""
    assert Selection() == Selection(roots=(), subtrees=(), stubs=())


def test_a_selection_naming_nothing_places_no_time_bound() -> None:
    """With no bound to satisfy, a record recording no midtime is still a record."""
    assert Selection().bounded_in_time is False


def test_a_lower_bound_alone_places_a_time_bound() -> None:
    """One bound is enough to make a record with no midtime unplaceable."""
    assert Selection(start_et=0.0).bounded_in_time is True


def test_an_upper_bound_alone_places_a_time_bound() -> None:
    """The other half of the same rule, which a one-sided test would miss."""
    assert Selection(stop_et=1.0).bounded_in_time is True


def test_a_bound_of_zero_is_a_bound() -> None:
    """J2000 itself is a perfectly good bound, and is falsy in Python."""
    assert Selection(start_et=0.0, stop_et=0.0).bounded_in_time is True


def test_an_inverted_time_range_is_refused_where_it_is_written() -> None:
    """A run that selected nothing would read as a clean run over a quiet span."""
    with pytest.raises(ValueError, match='the time range is inverted'):
        Selection(start_et=1.0, stop_et=0.0)


def test_the_refusal_names_both_ends_of_the_range() -> None:
    """The two numbers are the whole of what the reader has to swap back."""
    with pytest.raises(ValueError) as excinfo:
        Selection(start_et=5.0, stop_et=2.0)
    assert '5.0' in str(excinfo.value)
    assert '2.0' in str(excinfo.value)


@pytest.mark.parametrize(
    'bound',
    [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='inf'),
        pytest.param(float('-inf'), id='minus-inf'),
    ],
)
def test_a_non_finite_lower_bound_is_refused(bound: float) -> None:
    """A NaN is the sharpest disagreement the two storages can have.

    Every comparison against one is false, so the walk -- which keeps what it
    cannot show to be outside -- keeps every record, and the query, which keeps
    what it can show to be inside, keeps none.  An infinite bound is refused
    beside it because ``None`` is how a selection says it places no bound.

    Parameters:
        bound: The value to refuse.
    """
    with pytest.raises(ValueError, match='finite number of seconds'):
        Selection(start_et=bound)


def test_a_non_finite_upper_bound_is_refused() -> None:
    """The other half of the same rule, which a one-sided test would miss."""
    with pytest.raises(ValueError, match='finite number of seconds'):
        Selection(stop_et=float('nan'))


@pytest.mark.parametrize('field', [pytest.param('start_et'), pytest.param('stop_et')])
def test_a_bound_no_float_can_hold_is_refused_as_a_selection(field: str) -> None:
    """Python puts no bound on an integer, and asking whether one is finite raises.

    Refused as a selection rather than left to raise an arithmetic error out of
    the seam: an ``OverflowError`` reaching a caller from a value it wrote names
    neither the field nor the rule, and it is the one way a bound can be
    unusable without being a float at all.

    Parameters:
        field: The bound to hand an integer no float can hold.
    """
    named: dict[str, Any] = {field: 10**400}
    with pytest.raises(ValueError, match='finite number of seconds'):
        Selection(**named)


def test_a_boolean_time_bound_is_refused() -> None:
    """A boolean is an integer in Python, and would bound a span at one second."""
    with pytest.raises(ValueError, match='a time bound is a number of seconds'):
        Selection(start_et=True)


def test_a_mission_that_is_not_text_is_refused() -> None:
    """A query builder handed one raises in a vocabulary its caller cannot name."""
    with pytest.raises(ValueError, match='a mission is named by text'):
        Selection(instrument=[])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'field', [pytest.param('roots'), pytest.param('subtrees'), pytest.param('stubs')]
)
def test_a_single_name_where_several_belong_is_refused(field: str) -> None:
    """A string is a sequence of one-character strings, and would narrow to those.

    Parameters:
        field: The field to hand one string rather than a tuple of them.
    """
    named: dict[str, Any] = {field: '/data/results'}
    with pytest.raises(ValueError, match='names zero or more values as a tuple'):
        Selection(**named)


@pytest.mark.parametrize(
    'field', [pytest.param('roots'), pytest.param('subtrees'), pytest.param('stubs')]
)
def test_a_name_that_is_not_text_is_refused(field: str) -> None:
    """Each of them is compared against text a storage holds.

    Parameters:
        field: The field to hand something that is not a name.
    """
    named: dict[str, Any] = {field: (7,)}
    with pytest.raises(ValueError, match='each of them is text'):
        Selection(**named)


def test_a_subtree_of_more_than_one_component_is_refused() -> None:
    """A walk joins it and a query compares it to one component: two answers."""
    with pytest.raises(ValueError, match='is not a subtree of a results root'):
        Selection(subtrees=('VOL1/data',))


def test_a_stub_that_walks_out_of_its_root_is_refused() -> None:
    """A queue task carries stubs, and a task file can be written by hand."""
    with pytest.raises(ValueError, match='is not a results path stub'):
        Selection(stubs=('../elsewhere/N1454725799',))


def test_the_refusal_of_a_stub_names_the_stub() -> None:
    """A task carries hundreds, so a refusal naming none is unactionable."""
    with pytest.raises(ValueError) as excinfo:
        Selection(stubs=('VOL1/OK', '/etc/passwd'))
    assert '/etc/passwd' in str(excinfo.value)


def test_a_range_whose_ends_are_equal_is_not_inverted() -> None:
    """Both bounds are inclusive, so one instant is a range holding one instant."""
    assert Selection(start_et=1.0, stop_et=1.0).stop_et == 1.0


def test_a_selection_cannot_be_changed_after_it_is_written() -> None:
    """One selection is handed to two sources, and neither may alter it."""
    selection = Selection(instrument='coiss')
    with pytest.raises(AttributeError):
        selection.instrument = 'vgiss'  # type: ignore[misc]
