"""Hermetic tests for the Voyager frozen-attitude side of the assignment step.

A Voyager observation frame is frozen from a single tolerance-snapped pointing
lookup, so reproducing its baseline means making the same lookup: the same
clock tick, the same tolerance, the same fixed platform-to-camera rotation on
top.  These tests write discrete and interpolating baselines around a Voyager
exposure and pin which of the two lookups answers, how far each reaches, and
that the coverage filter admits everything the widest lookup can serve.
"""

from pathlib import Path

import cspyce
import pytest
from tests.spindoctor.cli.ck.assignment_helpers import (
    VOYAGER_KERNEL_NAME,
    VOYAGER_MIDTIME_ET,
    VOYAGER_SCLK_ID,
    WRONG_RAD,
    discrete_entry,
    image_entry,
    snapped_et,
    turned,
    voyager_recorded,
    write_candidate,
    write_discrete_candidate,
)
from tests.spindoctor.cli.ck.ck_helpers import (
    ET0,
    VOYAGER_CAMERA_FRAME,
    VOYAGER_CK_FRAME_ID,
    KernelPool,
)

from spindoctor.cli.ck import assignment
from spindoctor.cli.ck.assignment import assign_images, baseline_attitudes
from spindoctor.cli.ck.images import OmissionReason
from spindoctor.cli.ck.index import SNAPPED_LOOKUP_TOL_TICKS, build_ck_index

_VOYAGER_START_ET = ET0 + 1.003
_VOYAGER_IMAGE_NAME = 'C1205021_CALIB.IMG'

# A discrete baseline holds pointing only at its records, so a lookup answers
# only within its tolerance of one.  The offsets are derived from the module's
# own tolerance constants, so they keep bracketing the bound if it moves: one
# record well inside the base tolerance, one just outside it by an amount only
# the exposure term can reach, and one beyond the exposure term as well.
_BASE_TOL_TICKS = assignment._SNAPPED_TOL_TICKS
_EXTRA_TICKS = 0.2
_INSIDE_BASE_TOLERANCE_TICKS = _BASE_TOL_TICKS / 8.0
_JUST_OUTSIDE_BASE_TOLERANCE_TICKS = _BASE_TOL_TICKS + _EXTRA_TICKS
_BEYOND_THE_EXPOSURE_TERM_TICKS = _BASE_TOL_TICKS + 1.0

# The exposure term is exposure / divisor ticks.  The long exposure adds
# 0.3125 ticks, which reaches a record _EXTRA_TICKS past the base tolerance
# and not one a whole tick past it; a Voyager frame's own 0.48 s adds 0.01
# and reaches neither.
_LONG_ENOUGH_EXPOSURE_S = 0.3125 * assignment._SNAPPED_TOL_EXPOSURE_DIVISOR
_TOO_SHORT_EXPOSURE_S = 0.48


def test_a_frozen_baseline_reproduces_through_the_snapped_lookup(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A Voyager baseline is reproduced at the whole tick, not at the midtime."""
    root = tmp_path / 'CK'
    write_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=VOYAGER_SCLK_ID,
    )
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=voyager_recorded(snapped_et(VOYAGER_MIDTIME_ET)),
        kernels=(VOYAGER_KERNEL_NAME,),
        image_name=_VOYAGER_IMAGE_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=_VOYAGER_START_ET,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None


def test_a_frozen_baseline_read_at_the_midtime_reproduces_through_the_wider_lookup(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The wider lookup asks at the continuous tick, so a midtime reading pairs too.

    The frame oops falls back to encodes the epoch continuously rather than
    rounding it to a whole tick, and a baseline that interpolates between its
    records answers such a request at the epoch itself.  An image navigated
    through that fallback therefore records the attitude at its own midtime,
    and the second attempt is what pairs it.
    """
    root = tmp_path / 'CK'
    write_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=VOYAGER_SCLK_ID,
    )
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=voyager_recorded(VOYAGER_MIDTIME_ET),
        kernels=(VOYAGER_KERNEL_NAME,),
        image_name=_VOYAGER_IMAGE_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=_VOYAGER_START_ET,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None


def test_a_frozen_baseline_holding_another_attitude_does_not_reproduce(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Neither Voyager lookup pairs a kernel that holds a different attitude."""
    root = tmp_path / 'CK'
    write_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=VOYAGER_SCLK_ID,
        attitude=turned(WRONG_RAD),
    )
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=voyager_recorded(snapped_et(VOYAGER_MIDTIME_ET)),
        kernels=(VOYAGER_KERNEL_NAME,),
        image_name=_VOYAGER_IMAGE_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=_VOYAGER_START_ET,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_a_frozen_baseline_outside_the_first_tolerance_reproduces_at_the_second(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An image navigated through the wider fallback tolerance still pairs.

    The kernel's first record is ten seconds after the exposure, which is
    beyond the tolerance the primary lookup uses and inside the one the frame
    oops falls back to; the attitude that lookup answers with is the one at the
    kernel's first record.
    """
    root = tmp_path / 'CK'
    write_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=VOYAGER_SCLK_ID,
        start_et=ET0 + 10.0,
    )
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=voyager_recorded(ET0 + 10.0),
        kernels=(VOYAGER_KERNEL_NAME,),
        image_name=_VOYAGER_IMAGE_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=_VOYAGER_START_ET,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None


def test_a_discrete_baseline_answers_only_within_a_tolerance(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A record a hundred ticks away is out of reach with no tolerance at all.

    This is the premise of the tolerance tests below, and the difference
    between a discrete baseline and every other kernel these tests write: an
    interpolating baseline answers any epoch inside its window whatever
    tolerance is asked for, so nothing about a tolerance can be measured on
    one.
    """
    path = write_discrete_candidate(
        tmp_path / 'CK',
        VOYAGER_KERNEL_NAME,
        offset_ticks=_INSIDE_BASE_TOLERANCE_TICKS,
        midtime_et=VOYAGER_MIDTIME_ET,
    )
    pool.furnish(path)
    tick = float(cspyce.sce2t(VOYAGER_SCLK_ID, VOYAGER_MIDTIME_ET))
    with pytest.raises(OSError, match='CKINSUFFDATA'):
        cspyce.ckgp(VOYAGER_CK_FRAME_ID, tick, 0.0, 'J2000')


def test_a_discrete_baseline_reproduces_within_the_base_tolerance(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A Voyager image is paired with the discrete baseline its lookup reaches."""
    root = tmp_path / 'CK'
    write_discrete_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        offset_ticks=_INSIDE_BASE_TOLERANCE_TICKS,
        midtime_et=VOYAGER_MIDTIME_ET,
    )
    index = build_ck_index([root])
    assignments = assign_images([discrete_entry(_INSIDE_BASE_TOLERANCE_TICKS)], index)
    assert assignments[0].baseline is not None


def test_the_snapped_lookup_reaches_a_record_within_its_base_tolerance(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The first of the two lookups answers, not just the wider fallback.

    Both lookups find the same record on a discrete baseline, so which of them
    reached it is visible only in how many attitudes come back.
    """
    path = write_discrete_candidate(
        tmp_path / 'CK',
        VOYAGER_KERNEL_NAME,
        offset_ticks=_INSIDE_BASE_TOLERANCE_TICKS,
        midtime_et=VOYAGER_MIDTIME_ET,
    )
    pool.furnish(path)
    entry = discrete_entry(_INSIDE_BASE_TOLERANCE_TICKS)
    assert entry.pointing is not None
    assert len(baseline_attitudes(entry.pointing)) == 2


@pytest.mark.parametrize(
    ('offset_ticks', 'exposure_s', 'expected_lookups'),
    [
        (_JUST_OUTSIDE_BASE_TOLERANCE_TICKS, _LONG_ENOUGH_EXPOSURE_S, 2),
        (_JUST_OUTSIDE_BASE_TOLERANCE_TICKS, _TOO_SHORT_EXPOSURE_S, 1),
        (_BEYOND_THE_EXPOSURE_TERM_TICKS, _LONG_ENOUGH_EXPOSURE_S, 1),
    ],
    ids=['term-reaches-it', 'term-too-small', 'term-is-not-larger'],
)
def test_the_snapped_tolerance_grows_with_the_exposure(
    pool: KernelPool,
    tmp_path: Path,
    offset_ticks: float,
    exposure_s: float,
    expected_lookups: int,
) -> None:
    """The exposure term is what reaches a record just beyond the base tolerance.

    A 15 s exposure adds 0.3125 ticks of tolerance, which reaches a record a
    fifth of a tick past the base tolerance and not one a whole tick past it;
    a 0.48 s exposure adds 0.01 and reaches neither.  The third case is what
    pins the size of the term rather than its presence.

    Parameters:
        offset_ticks: How far past the midtime's whole tick the record sits.
        exposure_s: The recorded exposure duration.
        expected_lookups: How many of the two lookups answer.
    """
    path = write_discrete_candidate(
        tmp_path / 'CK',
        VOYAGER_KERNEL_NAME,
        offset_ticks=offset_ticks,
        midtime_et=VOYAGER_MIDTIME_ET,
    )
    pool.furnish(path)
    entry = discrete_entry(offset_ticks, exposure_s=exposure_s)
    assert entry.pointing is not None
    assert len(baseline_attitudes(entry.pointing)) == expected_lookups


def test_an_image_at_the_edge_of_the_snapped_lookup_is_still_assigned(
    pool: KernelPool, tmp_path: Path
) -> None:
    """The coverage filter and the lookup reach as far as each other.

    The record sits a tick inside the widest snapped tolerance, which is
    nearly the furthest a navigated Voyager frame can have been frozen from.
    The image has to survive the coverage filter and then reproduce; if the two
    tolerances ever stopped being one value, it would be dropped before any
    kernel was furnished and reported as having no baseline.  The last tick is
    left out because the filter measures from the exposure midtime and the
    lookup from that midtime rounded to a whole tick, so the two agree only to
    within half a tick of the extreme edge.
    """
    offset_ticks = SNAPPED_LOOKUP_TOL_TICKS - 1.0
    root = tmp_path / 'CK'
    write_discrete_candidate(
        root,
        VOYAGER_KERNEL_NAME,
        offset_ticks=offset_ticks,
        midtime_et=VOYAGER_MIDTIME_ET,
    )
    index = build_ck_index([root])
    assignments = assign_images([discrete_entry(offset_ticks)], index)
    assert assignments[0].baseline is not None
