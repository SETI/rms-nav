"""The acceptance test of the facts seam: one tree, two storages, one answer.

Over one results tree, the facts a source reading the documents yields and the
facts a source reading an index ingested from that same tree yields are the same
facts -- every field of the image, every technique row, every feature-source row,
and the same refusal for every file neither can read.  A consumer cannot see
which storage answered, so anything the two disagree about is a report that
changes when an operator points it at a database.

Every comparison is made against each of the two roots in turn, and the files
neither storage reads are named one at a time as well, so that an agreement here
is an agreement about something rather than about an empty stream.
"""

from typing import Any

import pytest
from tests.spindoctor.results_index.conftest import (
    BOTH_ROOTS,
    EXTRA_STUB,
    NESTED_STUB,
    REFUSED_STUB,
    TORN_STUB,
    TwoRoots,
    facts_from_index,
    facts_from_tree,
    stub_of,
    technique_key,
)

from spindoctor.nav_records import (
    NOT_A_NAVIGATION_DOCUMENT,
    NOT_VALID_JSON,
    ImageFacts,
    Selection,
    UnreadableFile,
)


def _feature_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return what identifies one feature-source row within its image.

    Parameters:
        row: The row.

    Returns:
        The feature type and the source that offered it.
    """
    return str(row['feature_type']), str(row['source_model']), str(row['source_name'])


def _images(found: list[ImageFacts | UnreadableFile]) -> list[dict[str, Any]]:
    """Return each image's own values, in the order the stream was sorted into.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One mapping per image.
    """
    return [one.image for one in found if isinstance(one, ImageFacts)]


def _techniques(found: list[ImageFacts | UnreadableFile]) -> list[list[dict[str, Any]]]:
    """Return each image's technique rows, sorted within the image.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One list per image, each in technique-name order.
    """
    return [
        sorted(one.techniques, key=technique_key) for one in found if isinstance(one, ImageFacts)
    ]


def _feature_sources(found: list[ImageFacts | UnreadableFile]) -> list[list[dict[str, Any]]]:
    """Return each image's feature-source rows, sorted within the image.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        One list per image, each in feature-source order.
    """
    return [
        sorted(one.feature_sources, key=_feature_key)
        for one in found
        if isinstance(one, ImageFacts)
    ]


def _refusals(found: list[ImageFacts | UnreadableFile]) -> list[tuple[str, str]]:
    """Return every file no facts came out of, and why.

    Parameters:
        found: What a stream yielded, sorted.

    Returns:
        The stub and reason of each.
    """
    return [(one.stub, one.reason) for one in found if isinstance(one, UnreadableFile)]


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_cover_the_same_files(two_roots: TwoRoots, which: str) -> None:
    """Every file the selection covers, from either storage.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = [stub_of(one) for one in facts_from_tree(two_roots, which, Selection())]
    from_index = [stub_of(one) for one in facts_from_index(two_roots, which, Selection())]
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_image_field(two_roots: TwoRoots, which: str) -> None:
    """Field by field, which is the whole of what a consumer reads.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _images(facts_from_tree(two_roots, which, Selection()))
    from_index = _images(facts_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_technique_row(two_roots: TwoRoots, which: str) -> None:
    """The child rows a reader of documents gets for nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _techniques(facts_from_tree(two_roots, which, Selection()))
    from_index = _techniques(facts_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_agree_on_every_feature_source_row(
    two_roots: TwoRoots, which: str
) -> None:
    """The aggregated inventory, which is the other table the merge reads.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _feature_sources(facts_from_tree(two_roots, which, Selection()))
    from_index = _feature_sources(facts_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


@pytest.mark.parametrize('which', BOTH_ROOTS)
def test_the_two_storages_refuse_the_same_files_for_the_same_reason(
    two_roots: TwoRoots, which: str
) -> None:
    """A file that is no navigation document, reported alike by both.

    Parameters:
        two_roots: The two ingested roots and their index.
        which: The root to read.
    """
    from_tree = _refusals(facts_from_tree(two_roots, which, Selection()))
    from_index = _refusals(facts_from_index(two_roots, which, Selection()))
    assert from_index == from_tree


def test_the_comparison_covers_a_refused_file(two_roots: TwoRoots) -> None:
    """Without which the agreement above would be an agreement about nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(facts_from_tree(two_roots, 'first', Selection())))
    assert reasons[REFUSED_STUB].startswith(NOT_A_NAVIGATION_DOCUMENT)


def test_the_comparison_covers_a_file_that_is_not_json(two_roots: TwoRoots) -> None:
    """The other family of reason, which says the parse produced nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(facts_from_tree(two_roots, 'first', Selection())))
    assert reasons[TORN_STUB] == NOT_VALID_JSON


def test_the_comparison_covers_a_file_the_decoder_gave_up_on(two_roots: TwoRoots) -> None:
    """Whichever way the decoder gave up, both storages state the one reason for it.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(facts_from_tree(two_roots, 'first', Selection())))
    assert reasons[NESTED_STUB] == NOT_VALID_JSON


def test_the_index_gives_that_file_the_reason_the_documents_give_it(
    two_roots: TwoRoots,
) -> None:
    """Named on its own as well, because it is the case the two once differed on.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    reasons = dict(_refusals(facts_from_index(two_roots, 'first', Selection())))
    assert reasons[NESTED_STUB] == NOT_VALID_JSON


def test_a_refusal_is_reported_although_the_other_root_holds_a_record_there(
    two_roots: TwoRoots,
) -> None:
    """A record under one root is no evidence about another root's refusal.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    stubs = [stub for stub, _reason in _refusals(facts_from_index(two_roots, 'first', Selection()))]
    assert REFUSED_STUB in stubs


def test_the_other_roots_extra_image_is_not_this_roots(two_roots: TwoRoots) -> None:
    """A stream that dropped the root would yield an image this root never held.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    stubs = [stub_of(one) for one in facts_from_index(two_roots, 'first', Selection())]
    assert EXTRA_STUB not in stubs
