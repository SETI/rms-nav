"""Tests for how much of a pass over a results tree runs at once."""

import re
from typing import Any

import pytest

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.nav_records import RETRIEVE_BATCH_SIZE, TreeTuning
from spindoctor.nav_records.tree import RETRIEVE_THREADS
from spindoctor.nav_records.walk import WALK_DIRECTORIES_AT_ONCE, WALK_THREADS


class _Section:
    """A configuration section that answers only what it was given.

    Parameters:
        values: The settings this section names.
    """

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


def test_the_module_defaults_are_the_tuning_defaults() -> None:
    """One source, so a default cannot be raised in one place and not the other."""
    tuning = TreeTuning()
    assert tuning.walk_threads == WALK_THREADS
    assert tuning.walk_directories_at_once == WALK_DIRECTORIES_AT_ONCE
    assert tuning.retrieve_threads == RETRIEVE_THREADS
    assert tuning.retrieve_batch_size == RETRIEVE_BATCH_SIZE


def test_the_shipped_configuration_names_every_setting() -> None:
    """A setting the shipped file omits is one nobody knows they can change."""
    section = DEFAULT_CONFIG.results_index
    for name in (
        'walk_threads',
        'walk_directories_at_once',
        'retrieve_threads',
        'retrieve_batch_size',
    ):
        assert getattr(section, name, None) is not None, name


def test_the_shipped_configuration_is_what_the_defaults_say() -> None:
    """The file and the dataclass agree, so neither is quietly the real one."""
    assert TreeTuning.from_config_section(DEFAULT_CONFIG.results_index) == TreeTuning()


def test_a_section_that_omits_a_setting_leaves_it_at_the_default() -> None:
    """An operator changing one number does not have to restate the rest."""
    tuning = TreeTuning.from_config_section(_Section(walk_threads=4))
    assert tuning.walk_threads == 4
    assert tuning.retrieve_threads == TreeTuning().retrieve_threads


def test_no_section_at_all_is_the_defaults() -> None:
    """A caller with no configuration to consult still gets a working pass."""
    assert TreeTuning.from_config_section(None) == TreeTuning()


@pytest.mark.parametrize('field', ['walk_threads', 'walk_directories_at_once', 'retrieve_threads'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True, '8', None])
def test_a_setting_that_is_not_a_positive_integer_is_refused(field: str, value: Any) -> None:
    """A pass tuned to zero threads does not run slowly; it does not run.

    Parameters:
        field: The setting to give the bad value to.
        value: A value that is not a count of things.
    """
    with pytest.raises(ValueError, match=field):
        TreeTuning(**{field: value})


def test_a_batch_smaller_than_the_pool_it_feeds_is_refused() -> None:
    """Not a slow configuration but a pool that cannot fill, so it is said early."""
    with pytest.raises(ValueError, match='retrieve_batch_size'):
        TreeTuning(retrieve_threads=64, retrieve_batch_size=8)


def test_a_batch_equal_to_the_pool_is_allowed() -> None:
    """The bound is what cannot fill the pool, not what fills it exactly once."""
    assert TreeTuning(retrieve_threads=8, retrieve_batch_size=8).retrieve_batch_size == 8


def test_a_configured_value_that_cannot_work_is_refused_by_name() -> None:
    """An operator reading the failure has to be told which setting to change."""
    with pytest.raises(ValueError, match=re.escape('results_index.walk_threads')):
        TreeTuning.from_config_section(_Section(walk_threads=0))
