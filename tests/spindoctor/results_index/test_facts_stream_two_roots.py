"""Two roots read as one stream of facts.

A source over one root cannot show everything the key does: the merge pairs the
wrong rows only where one stream spans two roots.  Two shapes make that visible.
Roots whose stubs interleave put key order and stub order at odds, so a sort that
named the stub alone hands the image stream and the child streams back
interleaved differently, and an image whose rows are met out of turn is given
none of them.  Roots where one's last stub is the next one's first put two
adjacent image groups under one stub, which is the simplest shape a merge key
that lost its root half mis-pairs under.

The rows are written directly rather than ingested from trees, because what each
shape turns on is which stubs each root holds and nothing about the documents.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest
from tests.spindoctor.conftest import index_url
from tests.spindoctor.results_index.conftest import (
    COLUMNS,
    child_name,
    feature_source_row,
    image_row,
    key_of,
    technique_row,
)

from spindoctor.nav_records import ImageFacts, Selection, normalize_root_url
from spindoctor.results_index import (
    FEATURE_SOURCES,
    IMAGES,
    TECHNIQUES,
    IndexRecordSource,
    open_index,
)

INTERLEAVED_FIRST = ('VOL/S1_CALIB', 'VOL/S3_CALIB')
"""What the first of the interleaved roots holds."""

INTERLEAVED_SECOND = ('VOL/S2_CALIB',)
"""What the second holds: one stub that sorts between the first root's two.

Key order and stub order therefore disagree, which is the shape a sort that lost
its root component hands back interleaved: the merge meets one image's rows
before the image it is assembling, waits, and gives that image none of its own.
"""

BOUNDARY_FIRST = ('VOL/S1_CALIB', 'VOL/S5_CALIB')
"""What the first of the boundary roots holds."""

BOUNDARY_SECOND = ('VOL/S5_CALIB', 'VOL/S6_CALIB')
"""What the second holds, beginning at the stub the first one ends on.

Two adjacent image groups sharing a stub is the shape a merge key that lost its
root half mis-pairs under: the first root's image takes the second root's rows
as well as its own, and the second root's image comes back with none.
"""


class TwoRootPairs:
    """Four results roots in one index, in two pairs, each read as one stream.

    Parameters:
        interleaved: The pair whose stubs interleave between the two roots.
        boundary: The pair where one root's last stub is the other's first.
        url: The index holding all four.
    """

    def __init__(
        self, interleaved: tuple[Path, Path], boundary: tuple[Path, Path], url: str
    ) -> None:
        self.interleaved = interleaved
        self.boundary = boundary
        self.url = url


@pytest.fixture
def two_root_pairs(tmp_path: Path) -> TwoRootPairs:
    """Write one index whose four roots make up the two shapes read below.

    Written as rows rather than ingested from trees, because what each shape
    turns on is which stubs each root holds and nothing about the documents.
    The child rows go in back to front, so insertion order pairs every image
    with the wrong one.

    Parameters:
        tmp_path: Directory the index is written under.

    Returns:
        The two pairs and the index holding all four roots.
    """
    held = {
        'boundary-a': BOUNDARY_FIRST,
        'boundary-b': BOUNDARY_SECOND,
        'interleaved-a': INTERLEAVED_FIRST,
        'interleaved-b': INTERLEAVED_SECOND,
    }
    roots = {name: tmp_path / name for name in held}
    keys = [
        (normalize_root_url(roots[name]), stub) for name, stubs in held.items() for stub in stubs
    ]
    url = index_url(tmp_path / 'pairs.sqlite3')
    engine = open_index(url, create=True)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.insert(),
                [
                    image_row(root_url=root_url, results_path_stub=stub, subtree='VOL')
                    for root_url, stub in keys
                ],
            )
            connection.execute(
                TECHNIQUES.insert(),
                [
                    technique_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        technique_name=child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
            connection.execute(
                FEATURE_SOURCES.insert(),
                [
                    feature_source_row(
                        root_url=root_url,
                        results_path_stub=stub,
                        source_name=child_name(root_url, stub),
                    )
                    for root_url, stub in reversed(keys)
                ],
            )
    finally:
        engine.dispose()
    return TwoRootPairs(
        (roots['interleaved-a'], roots['interleaved-b']),
        (roots['boundary-a'], roots['boundary-b']),
        url,
    )


def _children_of(url: str, roots: Sequence[Path], pick: str) -> dict[tuple[str, str], list[str]]:
    """Return the child names the merge gave each image of a stream over two roots.

    Parameters:
        url: The index to read.
        roots: The roots the source is opened over, read as one stream.
        pick: ``'techniques'`` or ``'feature_sources'``.

    Returns:
        The names, by the whole key of the image they were merged onto.
    """
    engine = open_index(url)
    with IndexRecordSource(engine, list(roots), url, COLUMNS) as source:
        found = [one for one in source.facts(Selection()) if isinstance(one, ImageFacts)]
    if pick == 'techniques':
        return {
            key_of(one): [str(row['technique_name']) for row in one.techniques] for one in found
        }
    return {key_of(one): [str(row['source_name']) for row in one.feature_sources] for one in found}


def _each_images_own_child(
    roots: Sequence[Path], held: Sequence[Sequence[str]]
) -> dict[tuple[str, str], list[str]]:
    """Return the one child row each image of these roots was written with.

    Parameters:
        roots: The roots, in the order their stubs are given.
        held: What each of them holds.

    Returns:
        The child name of each image, by the whole key.
    """
    return {
        (normalize_root_url(root), stub): [child_name(normalize_root_url(root), stub)]
        for root, stubs in zip(roots, held, strict=True)
        for stub in stubs
    }


def test_two_roots_in_one_stream_give_each_image_its_own_technique_rows(
    two_root_pairs: TwoRootPairs,
) -> None:
    """The stubs interleave, so key order and stub order are two different orders.

    A sort that named the stub alone would hand the image stream and the child
    streams back interleaved differently, and an image whose rows are met out of
    turn is given none of them.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.interleaved, 'techniques'
    ) == _each_images_own_child(two_root_pairs.interleaved, (INTERLEAVED_FIRST, INTERLEAVED_SECOND))


def test_two_roots_in_one_stream_give_each_image_its_own_feature_source_rows(
    two_root_pairs: TwoRootPairs,
) -> None:
    """The other child table, merged onto the same stream by the same rule.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.interleaved, 'feature_sources'
    ) == _each_images_own_child(two_root_pairs.interleaved, (INTERLEAVED_FIRST, INTERLEAVED_SECOND))


def test_the_interleaved_roots_really_do_disagree_with_stub_order() -> None:
    """Without which the two tests above would hold whatever the sort named."""
    assert [*INTERLEAVED_FIRST, *INTERLEAVED_SECOND] != sorted(
        [*INTERLEAVED_FIRST, *INTERLEAVED_SECOND]
    )


def test_a_stub_the_next_root_begins_with_keeps_the_first_roots_rows_off_it(
    two_root_pairs: TwoRootPairs,
) -> None:
    """One root's last stub is the next root's first, so the two groups adjoin.

    A merge comparing stubs alone takes both groups of child rows for the first
    of the two images and leaves the second with none, and there is no simpler
    shape it goes wrong under.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    assert _children_of(
        two_root_pairs.url, two_root_pairs.boundary, 'techniques'
    ) == _each_images_own_child(two_root_pairs.boundary, (BOUNDARY_FIRST, BOUNDARY_SECOND))


def test_the_boundary_roots_really_do_share_a_stub_where_they_meet() -> None:
    """Without which the test above would hold whatever the merge compared."""
    assert BOUNDARY_FIRST[-1] == BOUNDARY_SECOND[0]


def test_a_selection_naming_one_root_narrows_a_stream_over_two(
    two_root_pairs: TwoRootPairs,
) -> None:
    """A source is free to hold two roots and be asked about one of them.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    first, _second = two_root_pairs.interleaved
    engine = open_index(two_root_pairs.url)
    held = list(two_root_pairs.interleaved)
    with IndexRecordSource(engine, held, two_root_pairs.url, COLUMNS) as source:
        found = sorted(key_of(one) for one in source.facts(Selection(roots=(str(first),))))
    assert found == sorted((normalize_root_url(first), stub) for stub in INTERLEAVED_FIRST)


def test_named_stubs_over_a_source_holding_two_roots_are_refused(
    two_root_pairs: TwoRootPairs,
) -> None:
    """A stub is a key under a root, so two roots would answer one name twice.

    Parameters:
        two_root_pairs: The four roots and the index holding them.
    """
    engine = open_index(two_root_pairs.url)
    with (
        IndexRecordSource(
            engine, list(two_root_pairs.interleaved), two_root_pairs.url, COLUMNS
        ) as source,
        pytest.raises(ValueError, match='selection of keys'),
    ):
        source.facts(Selection(stubs=(INTERLEAVED_FIRST[0],)))
