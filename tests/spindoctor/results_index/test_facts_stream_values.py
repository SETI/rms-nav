"""The values the agreement of the two storages is made of.

An agreement is worth what the documents behind it carry, so the values a
comparison could pass while losing are read out one at a time: a covariance
matrix no pair of per-axis sigmas states, a list whose order is not its sorted
order, an epoch a navigation run recorded against one an image that never
loaded has none of, and an outcome named against one that is absent.

Every read a query blind to the root could answer out of the wrong row is made
against each of the two roots in turn, which record different values under the
same stubs.
"""

import pytest
from tests.spindoctor.results_index.conftest import (
    BOTH_ROOTS,
    ERROR_STUB,
    FIRST_VALUES,
    SUCCESS_STUB,
    UNLOADED_STUB,
    TwoRoots,
    facts_from_index,
    facts_of,
)

from spindoctor.nav_records import Selection


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_twist_covariance_survives_whole(two_roots: TwoRoots, which: str) -> None:
    """Its rotation row and column carry terms no per-axis sigma states.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = two_roots.values(which)
    found = facts_of(facts_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['covariance_px2'] == values.twist_covariance


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_a_per_technique_covariance_survives_whole(two_roots: TwoRoots, which: str) -> None:
    """Stored as a matrix, so a reader derives the sigmas rather than the reverse.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = two_roots.values(which)
    found = facts_of(facts_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert [row['covariance_px2'] for row in found.techniques] == [
        values.technique_covariance,
        values.technique_covariance,
    ]


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_an_exclusion_list_keeps_the_order_it_was_written_in(
    two_roots: TwoRoots, which: str
) -> None:
    """Which is not sorted order, so a storage that sorted the list is caught.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = two_roots.values(which)
    found = facts_of(facts_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['excluded_from_consensus'] == values.excluded


def test_the_exclusion_list_under_test_is_not_in_sorted_order() -> None:
    """Without which the test above would hold whether or not order survived."""
    assert FIRST_VALUES.excluded != sorted(FIRST_VALUES.excluded)


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_each_roots_frame_id_comes_back_from_that_root(two_roots: TwoRoots, which: str) -> None:
    """The two roots name two frames, so a read of one may not answer with the other.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    values = two_roots.values(which)
    found = facts_of(facts_from_index(two_roots, which, Selection()), SUCCESS_STUB)
    assert found.image['camera_frame_id'] == values.camera_frame_id


def test_an_image_that_never_loaded_records_no_epoch(two_roots: TwoRoots) -> None:
    """An epoch is an observation's midtime, and this image built no observation.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), UNLOADED_STUB)
    assert found.image['provenance_image_et'] is None


def test_an_image_that_never_loaded_is_placed_nowhere_in_time(two_roots: TwoRoots) -> None:
    """The column a date filter compares against is NULL rather than standing in.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), UNLOADED_STUB)
    assert found.image['image_et'] is None


def test_a_navigated_image_records_its_epoch_as_provenance(two_roots: TwoRoots) -> None:
    """The case the column has to tell apart from the one above.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), SUCCESS_STUB)
    assert found.image['provenance_image_et'] == 0.0


def test_an_image_naming_no_outcome_carries_no_error(two_roots: TwoRoots) -> None:
    """An absent outcome and a recorded error are different facts.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), UNLOADED_STUB)
    assert found.image['status_error'] is None


def test_a_run_that_ended_in_an_error_names_it(two_roots: TwoRoots) -> None:
    """The vocabulary an error filter matches verbatim.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    found = facts_of(facts_from_index(two_roots, 'first', Selection()), ERROR_STUB)
    assert found.image['status_error'] == 'SPICE(SPKINSUFFDATA)'
