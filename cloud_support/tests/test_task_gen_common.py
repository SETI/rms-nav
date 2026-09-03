"""Unit tests for the shared machinery behind the cloud-task generators.

Only the volume grouping has anything to decide.  The rest of the module runs
sd_offset and copies what it wrote, and is exercised by running it.

Run these with ``pytest cloud_support/tests``; the project's default test paths
cover the package alone.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import task_gen_common as common

# The Cassini archive's shape: 125 volumes of wildly uneven size, a few of them
# tiny, summing to about 443,000 images.  Sizes are what makes grouping hard, so
# the pattern here spans the real range rather than a uniform one.
UNEVEN_VOLUMES = [
    (f'COISS_{2000 + index:04d}', size)
    for index, size in enumerate([4285, 40, 5886, 129, 3745, 1021, 4705, 625, 3296, 890], start=1)
]


def test_a_volume_is_never_split_and_never_lost() -> None:
    groups = common.group_volumes(UNEVEN_VOLUMES, 5000)

    grouped = [volume for volumes, _ in groups for volume in volumes]
    assert grouped == [volume for volume, _ in UNEVEN_VOLUMES]


def test_each_group_reports_the_count_of_the_volumes_in_it() -> None:
    groups = common.group_volumes(UNEVEN_VOLUMES, 5000)

    sizes = dict(UNEVEN_VOLUMES)
    assert [count for _, count in groups] == [
        sum(sizes[volume] for volume in volumes) for volumes, _ in groups
    ]


def test_the_groups_come_out_near_the_target_rather_than_near_each_extreme() -> None:
    # 443,177 images at a target of 50,000 is the whole Cassini archive. Filling
    # each group to the target in turn leaves the remainder in the last one; the
    # even spread is what keeps the last group from being half or double the rest.
    volumes = [(f'COISS_{2000 + index:04d}', 3545) for index in range(125)]

    groups = common.group_volumes(volumes, 50000)

    assert len(groups) == 9
    # No two groups differ by more than the one volume that separates them.
    assert max(count for _, count in groups) - min(count for _, count in groups) <= 3545


def test_a_volume_that_would_overshoot_is_held_for_the_next_group() -> None:
    # Closing at 900 is nearer 1000 than closing at 1800 would be, so the third
    # volume starts the next group instead of finishing this one.
    volumes = [('A', 500), ('B', 400), ('C', 900), ('D', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A', 'B'], ['C'], ['D']]


def test_a_remainder_too_small_for_a_queue_joins_the_group_before_it() -> None:
    volumes = [('A', 1000), ('B', 1000), ('C', 30)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B', 'C']]


def test_a_remainder_worth_a_queue_stands_on_its_own() -> None:
    volumes = [('A', 1000), ('B', 1000), ('C', 600)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B'], ['C']]


def test_a_volume_holding_no_images_is_passed_over() -> None:
    volumes = [('A', 1000), ('EMPTY', 0), ('B', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['A'], ['B']]


def test_a_volume_larger_than_a_whole_group_is_a_group() -> None:
    volumes = [('HUGE', 5000), ('A', 1000), ('B', 1000)]

    groups = common.group_volumes(volumes, 1000)

    assert [volumes for volumes, _ in groups] == [['HUGE'], ['A'], ['B']]


def test_no_volumes_makes_no_groups() -> None:
    assert common.group_volumes([], 1000) == []


def test_a_target_of_zero_is_refused() -> None:
    with pytest.raises(ValueError) as exc:
        common.group_volumes([('A', 10)], 0)
    assert 'must be positive' in str(exc.value)
