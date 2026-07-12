"""Unit tests for the instrument-name <-> observation-class registry."""

import pytest

from spindoctor.obs import (
    ObsCassiniISS,
    inst_name_to_obs_class,
    inst_names,
    obs_class_to_inst_name,
)


def test_inst_names_round_trip() -> None:
    """Every registered name maps to a class that maps back to the name."""
    for name in inst_names():
        obs_class = inst_name_to_obs_class(name)
        assert obs_class_to_inst_name(obs_class) == name


def test_inst_name_to_obs_class_is_case_insensitive() -> None:
    assert inst_name_to_obs_class('COISS') is ObsCassiniISS


def test_inst_name_to_obs_class_unknown_raises() -> None:
    with pytest.raises(KeyError, match='unknown instrument name'):
        inst_name_to_obs_class('nonexistent')


def test_obs_class_to_inst_name_unregistered_returns_unknown() -> None:
    """An unregistered class maps to 'unknown' rather than raising."""

    class _Unregistered:
        pass

    assert obs_class_to_inst_name(_Unregistered) == 'unknown'
