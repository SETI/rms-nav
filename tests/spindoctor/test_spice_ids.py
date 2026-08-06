"""Tests for the shared CK-object-to-spacecraft-clock mapping.

The mapping exists because ``cspyce.ckmeta`` computes a clock id from a CK
object id instead of validating one, so both the attitude computation and the
C-kernel writer check what it resolves against a recorded table rather than
trusting it.  These tests pin the table against SPICE and pin both consumers
to the table, so the two checks cannot drift apart into one that bites and one
that no longer does.

``ckmeta`` needs no furnished kernels, so all of this is hermetic.
"""

from typing import cast

import cspyce
import pytest

from spindoctor.cli.ck.segment import resolve_sclk_id
from spindoctor.spice_ids import CK_OBJECT_SCLK_ID
from spindoctor.support.cmatrix import _CASSINI_CK_FRAME_ID, _ck_object_sclk_id

# The recorded pairs, in a stable order so a failure names the same case run
# to run.  Voyager 1 is the pair integer division gets wrong: ``-31100 //
# 1000`` is -32 in Python, which is the other spacecraft.
_RECORDED_PAIRS = sorted(CK_OBJECT_SCLK_ID.items())

# A CK object no mission owns, used to pin what ``ckmeta`` does with one.
_NONEXISTENT_CK_FRAME_ID = -999999


@pytest.mark.parametrize(('ck_frame_id', 'sclk_id'), _RECORDED_PAIRS)
def test_every_recorded_pair_agrees_with_spice(ck_frame_id: int, sclk_id: int) -> None:
    """Each recorded CK object resolves to the clock recorded beside it.

    A typo in either half of a pair would otherwise only surface as time tags
    encoding another spacecraft's clock, on every record of every kernel.
    """
    assert int(cspyce.ckmeta(ck_frame_id, 'SCLK')) == sclk_id


@pytest.mark.parametrize(('ck_frame_id', 'sclk_id'), _RECORDED_PAIRS)
def test_the_attitude_computation_reads_the_recorded_pair(ck_frame_id: int, sclk_id: int) -> None:
    """The attitude computation's clock for an object is the recorded one."""
    assert _ck_object_sclk_id(ck_frame_id) == sclk_id


@pytest.mark.parametrize(('ck_frame_id', 'sclk_id'), _RECORDED_PAIRS)
def test_the_kernel_writer_reads_the_recorded_pair(ck_frame_id: int, sclk_id: int) -> None:
    """The writer's resolved clock for an object is the recorded one."""
    assert resolve_sclk_id(ck_frame_id) == sclk_id


def test_ckmeta_computes_a_clock_for_a_ck_object_that_does_not_exist() -> None:
    """``ckmeta`` computes rather than validates, which is why the table exists.

    A nonexistent CK object still yields a plausible clock id and no error, so
    trusting the round trip would write time tags from the wrong clock while
    every call reported success.
    """
    assert int(cspyce.ckmeta(_NONEXISTENT_CK_FRAME_ID, 'SCLK')) == -999


def test_the_recorded_table_is_read_only() -> None:
    """Neither consumer can edit the table the other one checks against."""
    with pytest.raises(TypeError, match='does not support item assignment'):
        cast(dict[int, int], CK_OBJECT_SCLK_ID)[_CASSINI_CK_FRAME_ID] = -1
