"""Hermetic tests for ``spindoctor.cli.ck.segment``.

Each test builds its own baseline C-kernel from a known attitude history, plants
a known correction, has the writer produce the corrected segment, and then reads
the corrected kernel back through SPICE.  The assertions compare attitudes to
the composed truth matrix itself rather than to the magnitude of the correction,
because a magnitude survives a reversed composition, a dropped conjugation and a
sign flip alike.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cspyce
import numpy as np
import pytest
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    TICKS_PER_SECOND,
    VOYAGER_CAMERA_FRAME,
    VOYAGER_CK_FRAME_ID,
    KernelPool,
    axis_rotation,
    baseline_angular_velocity,
    baseline_attitude,
    baseline_segment,
    rotation_angle_between,
    sweeping_attitude,
    write_baseline_ck,
    write_ck,
)

from spindoctor.cli.ck.pointing import ImagePointing
from spindoctor.cli.ck.segment import (
    BaselineCoverageGapError,
    CkSegment,
    build_segment,
    resolve_sclk_id,
)
from spindoctor.support.types import NDArrayFloatType

# The correction the tests plant, expressed in the CK object's own frame.  It is
# far larger than a navigated correction ever is and turns about an axis shared
# with neither the baseline attitude nor the baseline angular velocity, so a
# composition applied in the wrong order or on the wrong side cannot come out
# equal to the right one by symmetry.
_CORRECTION_AXIS = np.array([-0.8, 0.1, 0.59])
_CORRECTION_RAD = np.radians(5.0)

# The baseline kernels carry records every half second from ET0, and the
# exposure under test runs from ET0 + 1 to ET0 + 3, so its records land on
# baseline records and its interior sample epochs do not.
_BASELINE_STEP_S = 0.5
_BASELINE_RECORDS = 9
_START_ET = ET0 + 1.0
_STOP_ET = ET0 + 3.0

# Two attitudes agree when the rotation between them is smaller than this.  It
# is the plan's reproduction bound; the quaternion round trip through SPICE
# costs about 1e-16 radians, so the bound is not tight against numerical noise.
_ANGLE_TOL_RAD = 1e-9

_IMAGE_NAME = 'N1484573295_1.IMG'

# The clocks the two test CK objects encode their time tags against: what
# resolve_sclk_id returns for CASSINI_CK_FRAME_ID and VOYAGER_CK_FRAME_ID.
_CASSINI_SCLK_ID = -82
_VOYAGER_SCLK_ID = -31


@dataclass(frozen=True)
class _Case:
    """One planted correction, its baseline kernel, and the corrected kernel.

    Parameters:
        pointing: The recorded pointing handed to the writer.
        segment: The segment the writer built.
        correction: The correction planted in the CK object's own frame.
        baseline_path: The baseline kernel, furnished when the case is built.
        corrected_path: The corrected kernel, written but not yet furnished.
    """

    pointing: ImagePointing
    segment: CkSegment
    correction: NDArrayFloatType
    baseline_path: Path
    corrected_path: Path


def _build_case(
    pool: KernelPool,
    *,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    camera_frame: str = CASSINI_CAMERA_FRAME,
    attitude: Callable[[float], NDArrayFloatType] = baseline_attitude,
    with_angular_velocity: bool = True,
    baseline_records: int = _BASELINE_RECORDS,
    start_et: float = _START_ET,
    stop_et: float = _STOP_ET,
    midtime_et: float | None = None,
    exposure_s: float | None = None,
    correction_rad: float = _CORRECTION_RAD,
    image_name: str = _IMAGE_NAME,
) -> _Case:
    """Plant a correction, write the corrected kernel, and report both.

    The baseline kernel is left furnished so a test can read baseline values
    before swapping in the corrected one.

    Parameters:
        pool: The test's kernel pool.
        ck_frame_id: SPICE id of the object under test.
        camera_frame: SPICE name of the camera frame the C-matrix is in.
        attitude: The baseline J2000-to-CK-object rotation at an epoch.
        with_angular_velocity: Whether the baseline kernel carries angular
            velocity.
        baseline_records: Number of half-second records in the baseline.
        start_et: Exposure start.
        stop_et: Exposure stop.
        midtime_et: Exposure midtime; the arithmetic midpoint when None.
        exposure_s: Exposure duration; ``stop_et - start_et`` when None.
        correction_rad: Size of the planted correction.
        image_name: Basename recorded for the image.

    Returns:
        The case.
    """
    sclk_id = resolve_sclk_id(ck_frame_id)
    epochs = [ET0 + step * _BASELINE_STEP_S for step in range(baseline_records)]
    baseline_path = pool.root / 'baseline.bc'
    write_baseline_ck(
        baseline_path,
        ck_frame_id=ck_frame_id,
        sclk_id=sclk_id,
        epochs=epochs,
        attitude=attitude,
        angular_velocity=baseline_angular_velocity if with_angular_velocity else None,
    )
    pool.furnish(baseline_path)
    midtime = (start_et + stop_et) / 2.0 if midtime_et is None else midtime_et
    correction = axis_rotation(_CORRECTION_AXIS, correction_rad)
    camera_from_ck = np.asarray(
        cspyce.pxform(str(cspyce.frmnam(ck_frame_id)), camera_frame, midtime), dtype=np.float64
    )
    pointing = ImagePointing(
        image_name=image_name,
        cmatrix=camera_from_ck @ correction @ attitude(midtime),
        cmatrix_original=camera_from_ck @ attitude(midtime),
        camera_frame=camera_frame,
        ck_frame_id=ck_frame_id,
        start_et=start_et,
        stop_et=stop_et,
        midtime_et=midtime,
        exposure_s=stop_et - start_et if exposure_s is None else exposure_s,
    )
    segment = build_segment(pointing)
    corrected_path = pool.root / 'corrected.bc'
    write_ck(corrected_path, [segment])
    return _Case(
        pointing=pointing,
        segment=segment,
        correction=correction,
        baseline_path=baseline_path,
        corrected_path=corrected_path,
    )


def _swap_to_corrected(pool: KernelPool, case: _Case) -> None:
    """Unload the baseline kernel and furnish the corrected one.

    Parameters:
        pool: The test's kernel pool.
        case: The case whose kernels are swapped.
    """
    pool.unload(case.baseline_path)
    pool.furnish(case.corrected_path)


def _tick(sclk_id: int, et: float) -> float:
    """Return the encoded SCLK for an epoch.

    Parameters:
        sclk_id: The spacecraft clock.
        et: TDB seconds past J2000.

    Returns:
        The encoded time tag.
    """
    return float(cspyce.sce2c(sclk_id, et))


def _attitude_from_pool(ck_frame_id: int, sclk_id: int, et: float) -> NDArrayFloatType:
    """Read the furnished pointing for a CK object at one epoch.

    Parameters:
        ck_frame_id: SPICE id of the object.
        sclk_id: The spacecraft clock.
        et: TDB seconds past J2000.

    Returns:
        The 3x3 J2000-to-CK-object rotation.
    """
    cmat, _clkout = cspyce.ckgp(ck_frame_id, _tick(sclk_id, et), 0.0, 'J2000')
    return np.asarray(cmat, dtype=np.float64)


@pytest.mark.parametrize(
    'query_et',
    [_START_ET, (_START_ET + _STOP_ET) / 2.0, _STOP_ET, _START_ET + 0.7],
    ids=['start', 'midtime', 'stop', 'interior'],
)
def test_corrected_attitude_matches_composed_truth(pool: KernelPool, query_et: float) -> None:
    """The written attitude is the planted correction on the baseline history.

    Parameters:
        query_et: Epoch the corrected kernel is queried at, TDB seconds past
            J2000.
    """
    case = _build_case(pool)
    _swap_to_corrected(pool, case)
    read = _attitude_from_pool(CASSINI_CK_FRAME_ID, _CASSINI_SCLK_ID, query_et)
    truth = case.correction @ baseline_attitude(query_et)
    assert rotation_angle_between(read, truth) < _ANGLE_TOL_RAD


def test_baseline_attitude_is_not_the_corrected_one(pool: KernelPool) -> None:
    """The correction really moved the pointing, by the angle that was planted."""
    case = _build_case(pool)
    _swap_to_corrected(pool, case)
    read = _attitude_from_pool(CASSINI_CK_FRAME_ID, _CASSINI_SCLK_ID, case.pointing.midtime_et)
    moved = rotation_angle_between(read, baseline_attitude(case.pointing.midtime_et))
    assert moved == pytest.approx(_CORRECTION_RAD, abs=_ANGLE_TOL_RAD)


def test_angular_velocity_is_copied_unchanged(pool: KernelPool) -> None:
    """Angular velocity is the baseline's own vectors, not rotated by delta."""
    case = _build_case(pool)
    record_ets = [_START_ET, (_START_ET + _STOP_ET) / 2.0, _STOP_ET]
    baseline_av = [
        np.asarray(cspyce.ckgpav(CASSINI_CK_FRAME_ID, _tick(_CASSINI_SCLK_ID, et), 0.0, 'J2000')[1])
        for et in record_ets
    ]
    _swap_to_corrected(pool, case)
    corrected_av = [
        np.asarray(cspyce.ckgpav(CASSINI_CK_FRAME_ID, _tick(_CASSINI_SCLK_ID, et), 0.0, 'J2000')[1])
        for et in record_ets
    ]
    assert case.segment.has_angular_velocity is True
    assert np.array_equal(corrected_av[0], baseline_av[0])
    assert np.array_equal(corrected_av[1], baseline_av[1])
    assert np.array_equal(corrected_av[2], baseline_av[2])


def test_written_coverage_is_exactly_the_exposure(pool: KernelPool) -> None:
    """The file advertises the exposure and not one tick more.

    ``ckcov`` reports what the segment descriptor claims, which is what a
    consumer and the file-mirroring step both read; it is independent of the
    records the segment actually holds, so a descriptor widened beyond them
    would go unnoticed by any assertion made on the record array.
    """
    case = _build_case(pool)
    cover = cspyce.ckcov(
        str(case.corrected_path), CASSINI_CK_FRAME_ID, False, 'SEGMENT', 0.0, 'SCLK'
    )
    assert len(cover) == 2
    assert float(cover[0]) == _tick(_CASSINI_SCLK_ID, _START_ET)
    assert float(cover[1]) == _tick(_CASSINI_SCLK_ID, _STOP_ET)


def test_pointing_outside_the_written_window_falls_through(pool: KernelPool) -> None:
    """A consumer gets nothing from the corrected kernel outside the exposure."""
    case = _build_case(pool)
    _swap_to_corrected(pool, case)
    with pytest.raises(OSError, match='CKINSUFFDATA'):
        cspyce.ckgp(CASSINI_CK_FRAME_ID, _tick(_CASSINI_SCLK_ID, _STOP_ET + 0.25), 0.0, 'J2000')


def test_the_correction_reaches_every_lookup_over_a_furnished_baseline(
    pool: KernelPool,
) -> None:
    """The overlay's attitude is what SPICE answers, through ckgp, ckgpav and sxform.

    The baseline stays furnished, which is how these kernels are meant to be
    used and the only arrangement in which a segment declaring no angular
    velocity is visibly skipped: SPICE would then answer ``ckgpav`` and
    ``sxform`` -- the call oops makes -- from the baseline underneath, with the
    uncorrected attitude, while ``ckgp`` alone reported the correction.
    """
    case = _build_case(pool)
    pool.furnish(case.corrected_path)
    midtime = case.pointing.midtime_et
    tick = _tick(_CASSINI_SCLK_ID, midtime)
    truth = case.correction @ baseline_attitude(midtime)
    from_ckgp = np.asarray(cspyce.ckgp(CASSINI_CK_FRAME_ID, tick, 0.0, 'J2000')[0], np.float64)
    from_ckgpav = np.asarray(cspyce.ckgpav(CASSINI_CK_FRAME_ID, tick, 0.0, 'J2000')[0], np.float64)
    ck_frame = str(cspyce.frmnam(CASSINI_CK_FRAME_ID))
    from_sxform = np.asarray(cspyce.sxform('J2000', ck_frame, midtime), np.float64)[:3, :3]
    assert rotation_angle_between(from_ckgp, truth) < _ANGLE_TOL_RAD
    assert rotation_angle_between(from_ckgpav, truth) < _ANGLE_TOL_RAD
    assert rotation_angle_between(from_sxform, truth) < _ANGLE_TOL_RAD


@pytest.mark.parametrize(
    ('ck_frame_id', 'camera_frame'),
    [
        (CASSINI_CK_FRAME_ID, CASSINI_CAMERA_FRAME),
        (VOYAGER_CK_FRAME_ID, VOYAGER_CAMERA_FRAME),
    ],
    ids=['evaluated-chain', 'frozen-attitude'],
)
def test_spice_answers_the_camera_frame_with_the_recorded_cmatrix(
    pool: KernelPool, ck_frame_id: int, camera_frame: str
) -> None:
    """The one question a consumer asks, answered with the number that was recorded.

    Everything the writer does is between the recorded C-matrix and this
    lookup: the rotation from the camera to the CK object, the correction in
    that object's coordinates, the quaternion conversion, the segment write,
    and the frame chain back out to the camera.  A consumer makes none of those
    steps -- it furnishes the file and asks for the camera frame -- so this is
    the assertion that corresponds to their experience, and any of those steps
    going wrong moves it.

    Parameters:
        ck_frame_id: SPICE id of the object under test.
        camera_frame: SPICE name of the camera frame the C-matrix is in.
    """
    case = _build_case(pool, ck_frame_id=ck_frame_id, camera_frame=camera_frame)
    pool.furnish(case.corrected_path)
    read = np.asarray(
        cspyce.pxform('J2000', camera_frame, case.pointing.midtime_et), dtype=np.float64
    )
    assert rotation_angle_between(read, case.pointing.cmatrix) < _ANGLE_TOL_RAD


def test_a_baseline_without_angular_velocity_is_refused(pool: KernelPool) -> None:
    """An exposure whose baseline supplies no angular velocity gets no segment.

    The alternative is a segment declaring none, which ``ckgpav`` and
    ``sxform`` skip in favor of any other kernel covering the same object and
    epoch -- so whether the correction is delivered would depend on what else
    the consumer furnished.
    """
    with pytest.raises(ValueError, match='angular velocity at only some of them'):
        _build_case(pool, with_angular_velocity=False)


def test_angular_velocity_missing_from_part_of_the_exposure_is_refused(
    pool: KernelPool,
) -> None:
    """An exposure straddling a baseline that loses angular velocity gets no segment.

    One flag covers the whole segment, so a segment cannot say that half its
    records have angular velocity.  Zeros for the rest would claim a parked
    platform the baseline never measured, and declaring none would withhold the
    correction from ``sxform``, so the image is refused instead.
    """
    sclk_id = resolve_sclk_id(CASSINI_CK_FRAME_ID)
    baseline_path = pool.root / 'baseline.bc'
    write_ck(
        baseline_path,
        [
            baseline_segment(
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=sclk_id,
                epochs=[ET0 + step * 0.5 for step in range(5)],
                attitude=baseline_attitude,
                angular_velocity=baseline_angular_velocity,
                segid='with-av',
            ),
            baseline_segment(
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=sclk_id,
                epochs=[ET0 + 2.5 + step * 0.5 for step in range(5)],
                attitude=baseline_attitude,
                angular_velocity=None,
                segid='without-av',
            ),
        ],
    )
    pool.furnish(baseline_path)
    correction = axis_rotation(_CORRECTION_AXIS, _CORRECTION_RAD)
    midtime = (_START_ET + _STOP_ET) / 2.0
    camera_from_ck = np.asarray(
        cspyce.pxform(str(cspyce.frmnam(CASSINI_CK_FRAME_ID)), CASSINI_CAMERA_FRAME, midtime),
        dtype=np.float64,
    )
    pointing = ImagePointing(
        image_name=_IMAGE_NAME,
        cmatrix=camera_from_ck @ correction @ baseline_attitude(midtime),
        cmatrix_original=camera_from_ck @ baseline_attitude(midtime),
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=_START_ET,
        stop_et=_STOP_ET,
        midtime_et=midtime,
        exposure_s=_STOP_ET - _START_ET,
    )
    # The premise: angular velocity at the first record, none at the last.
    cspyce.ckgpav(CASSINI_CK_FRAME_ID, _tick(sclk_id, _START_ET), 0.0, 'J2000')
    with pytest.raises(OSError, match='CKINSUFFDATA'):
        cspyce.ckgpav(CASSINI_CK_FRAME_ID, _tick(sclk_id, _STOP_ET), 0.0, 'J2000')
    with pytest.raises(ValueError, match='angular velocity at only some of them'):
        build_segment(pointing)


def test_a_record_outside_the_baseline_coverage_is_refused(pool: KernelPool) -> None:
    """Baseline pointing is read at the record epoch, never snapped to a distant one.

    The lookup tolerance is zero, so an exposure the baseline does not cover
    fails instead of being corrected against whatever attitude happens to be
    nearest.
    """
    with pytest.raises(BaselineCoverageGapError, match='supplies no pointing for CK object'):
        _build_case(pool, start_et=ET0 + 10.0, stop_et=ET0 + 12.0)


def test_an_exposure_whose_midtime_alone_is_covered_is_refused(pool: KernelPool) -> None:
    """The condition that reaches a run: the midtime paired the image with this baseline.

    The exposure straddles the end of the baseline's coverage, so the lookup
    that made the pairing succeeds and the one at the exposure stop does not.
    """
    covered_to = ET0 + (_BASELINE_RECORDS - 1) * _BASELINE_STEP_S
    with pytest.raises(BaselineCoverageGapError, match='which is a record epoch of this exposure'):
        _build_case(pool, start_et=covered_to - 1.0, stop_et=covered_to + 1.0)
    # The premise: the midtime this image was paired on is covered, and the
    # exposure stop is a second past the end of the same baseline.
    midtime_lookup = cspyce.ckgp.flag(
        CASSINI_CK_FRAME_ID, _tick(_CASSINI_SCLK_ID, covered_to), 0.0, 'J2000'
    )
    assert bool(midtime_lookup[-1]) is True
    stop_lookup = cspyce.ckgp.flag(
        CASSINI_CK_FRAME_ID, _tick(_CASSINI_SCLK_ID, covered_to + 1.0), 0.0, 'J2000'
    )
    assert bool(stop_lookup[-1]) is False


def test_a_coverage_gap_is_not_reported_as_a_missing_angular_velocity(
    pool: KernelPool,
) -> None:
    """The two arrive from SPICE the same way and mean opposite things.

    One omits an image and the other stops the run, so a gap that came back as
    the angular-velocity refusal would end a batch the report should have
    carried on past.
    """
    with pytest.raises(BaselineCoverageGapError) as raised:
        _build_case(pool, start_et=ET0 + 10.0, stop_et=ET0 + 12.0)
    assert 'angular velocity' not in str(raised.value)


def test_a_record_epoch_that_is_not_a_time_is_refused(
    pool: KernelPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tag SPICE cannot answer for is not the same as an epoch it does not cover.

    ``ckgp`` reports a non-finite time tag exactly as it reports an uncovered
    epoch, so a tag that is not a time would otherwise be read as a coverage
    gap and quietly omit the image.
    """
    sclk_id = resolve_sclk_id(CASSINI_CK_FRAME_ID)
    baseline_path = pool.root / 'baseline.bc'
    write_baseline_ck(
        baseline_path,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        sclk_id=sclk_id,
        epochs=[ET0 + step * _BASELINE_STEP_S for step in range(_BASELINE_RECORDS)],
        attitude=baseline_attitude,
        angular_velocity=baseline_angular_velocity,
    )
    pool.furnish(baseline_path)
    midtime = (_START_ET + _STOP_ET) / 2.0
    camera_from_ck = np.asarray(
        cspyce.pxform(str(cspyce.frmnam(CASSINI_CK_FRAME_ID)), CASSINI_CAMERA_FRAME, midtime),
        dtype=np.float64,
    )
    pointing = ImagePointing(
        image_name=_IMAGE_NAME,
        cmatrix=camera_from_ck @ baseline_attitude(midtime),
        cmatrix_original=camera_from_ck @ baseline_attitude(midtime),
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=_START_ET,
        stop_et=_STOP_ET,
        midtime_et=midtime,
        exposure_s=_STOP_ET - _START_ET,
    )
    # The premise, measured rather than assumed: this baseline covers the
    # exposure, and the lookup still comes back empty-handed for such a tag.
    assert bool(cspyce.ckgp.flag(CASSINI_CK_FRAME_ID, float('nan'), 0.0, 'J2000')[-1]) is False
    monkeypatch.setattr(cspyce, 'sce2c', lambda _sclk_id, _et: float('nan'))
    with pytest.raises(ValueError, match='is not a finite time'):
        build_segment(pointing)


def test_no_baseline_furnished_at_all_is_not_a_coverage_gap(pool: KernelPool) -> None:
    """A pool without the kernel is a setup failure, not one exposure's condition.

    It has to stay distinguishable, because a coverage gap omits one image and
    carries on: a run furnishing no C-kernel would otherwise report every image
    it considered as having outlasted its baseline.
    """
    pointing = ImagePointing(
        image_name=_IMAGE_NAME,
        cmatrix=np.eye(3),
        cmatrix_original=np.eye(3),
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=_START_ET,
        stop_et=_STOP_ET,
        midtime_et=(_START_ET + _STOP_ET) / 2.0,
        exposure_s=_STOP_ET - _START_ET,
    )
    with pytest.raises(OSError, match='NOLOADEDFILES'):
        build_segment(pointing)


def test_a_ck_object_with_no_frame_name_is_refused(pool: KernelPool) -> None:
    """Without the frame kernel there is no rotation to the camera, and no guess at one."""
    pointing = ImagePointing(
        image_name=_IMAGE_NAME,
        cmatrix=np.eye(3),
        cmatrix_original=np.eye(3),
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=_START_ET,
        stop_et=_STOP_ET,
        midtime_et=(_START_ET + _STOP_ET) / 2.0,
        exposure_s=_STOP_ET - _START_ET,
    )
    pool.unload(pool.root / 'test.tf')
    with pytest.raises(KeyError, match='FRAMEIDNOTFOUND'):
        build_segment(pointing)


def test_segment_stores_its_records_read_only(pool: KernelPool) -> None:
    """A caller cannot edit a validated record set after the fact."""
    case = _build_case(pool)
    assert case.segment.sclkdp.flags.writeable is False
    assert case.segment.quats.flags.writeable is False
    assert case.segment.avvs is not None
    assert case.segment.avvs.flags.writeable is False


def test_quaternion_records_are_sign_continuous(pool: KernelPool) -> None:
    """A sequence whose raw quaternions flip sign is written continuous."""
    case = _build_case(pool, attitude=sweeping_attitude)
    record_ets = [_START_ET, (_START_ET + _STOP_ET) / 2.0, _STOP_ET]
    raw = np.vstack([cspyce.m2q(case.correction @ sweeping_attitude(et)) for et in record_ets])
    raw_dots = [float(np.dot(raw[index], raw[index + 1])) for index in range(len(raw) - 1)]
    written = np.asarray(case.segment.quats, dtype=np.float64)
    written_dots = [
        float(np.dot(written[index], written[index + 1])) for index in range(len(written) - 1)
    ]
    _swap_to_corrected(pool, case)
    interior_et = _START_ET + 0.3
    read = _attitude_from_pool(CASSINI_CK_FRAME_ID, _CASSINI_SCLK_ID, interior_et)
    truth = case.correction @ sweeping_attitude(interior_et)
    assert min(raw_dots) < 0.0
    assert min(written_dots) >= 0.0
    assert rotation_angle_between(read, truth) < _ANGLE_TOL_RAD


def test_coincident_exposure_epochs_yield_one_record(pool: KernelPool) -> None:
    """Three epochs that are one floating-point value yield one valid record.

    This is the only way the single-record path is reached.  ``sce2c`` encodes
    a fractional tick, so an exposure merely shorter than a tick still encodes
    to three distinct time tags: with the test clock's 1/256 s tick, a 1 ms
    exposure is 0.256 ticks and produces three records.  Only epochs that do
    not differ as doubles collapse, which no real exposure produces -- the
    shortest Cassini ISS exposure is 5 ms.
    """
    start_et = ET0 + 2.0
    stop_et = start_et + 1.0e-9
    midtime_et = start_et + 5.0e-10
    case = _build_case(
        pool,
        start_et=start_et,
        stop_et=stop_et,
        midtime_et=midtime_et,
        exposure_s=1.0e-9,
    )
    _swap_to_corrected(pool, case)
    read = _attitude_from_pool(CASSINI_CK_FRAME_ID, _CASSINI_SCLK_ID, start_et)
    truth = case.correction @ baseline_attitude(case.pointing.midtime_et)
    assert stop_et == start_et
    assert midtime_et == start_et
    assert case.segment.record_count == 1
    assert case.segment.begtim == case.segment.endtim
    assert rotation_angle_between(read, truth) < _ANGLE_TOL_RAD


def test_a_sub_tick_exposure_still_yields_three_records(pool: KernelPool) -> None:
    """An exposure shorter than a clock tick is not degenerate.

    The encoded time tags differ in their fractional part, so the ordinary
    three-record path applies.
    """
    start_et = ET0 + 2.0
    exposure_s = 1.0e-3
    case = _build_case(
        pool,
        start_et=start_et,
        stop_et=start_et + exposure_s,
        midtime_et=start_et + exposure_s / 2.0,
        exposure_s=exposure_s,
    )
    ticks = np.asarray(case.segment.sclkdp, dtype=np.float64)
    assert exposure_s * TICKS_PER_SECOND < 1.0
    assert case.segment.record_count == 3
    assert float(np.min(np.diff(ticks))) > 0.0


@pytest.mark.parametrize(
    ('exposure_s', 'expected_records'),
    [(2.0, 3), (9.999, 3), (10.0, 11), (25.0, 27)],
    ids=['short', 'just-under-threshold', 'at-threshold', 'long'],
)
def test_record_count_follows_the_cadence(
    pool: KernelPool, exposure_s: float, expected_records: int
) -> None:
    """Long exposures get a one-second cadence; short ones get three records.

    Ten seconds exactly is on the cadence side of the boundary, because it is
    an ordinary commanded exposure and three records over ten seconds have
    already lost measurable fidelity.

    Parameters:
        exposure_s: Exposure duration under test, in seconds.
        expected_records: Records the segment should hold for that duration.
    """
    start_et = ET0 + 1.0
    stop_et = start_et + exposure_s
    case = _build_case(
        pool,
        baseline_records=int(exposure_s / _BASELINE_STEP_S) + 6,
        start_et=start_et,
        stop_et=stop_et,
    )
    assert case.segment.record_count == expected_records
    assert case.segment.begtim == _tick(_CASSINI_SCLK_ID, start_et)
    assert case.segment.endtim == _tick(_CASSINI_SCLK_ID, stop_et)


def test_an_exposure_needing_more_records_than_a_segment_holds_is_refused(
    pool: KernelPool,
) -> None:
    """A span no instrument commands is refused rather than expanded.

    The cadence arithmetic has no bound of its own: a span of ten million
    seconds asks for ten million records, and a longer one exhausts memory
    before anything can report why.
    """
    start_et = ET0 + 1.0
    with pytest.raises(ValueError, match='more than the 10000 a segment may hold'):
        _build_case(pool, start_et=start_et, stop_et=start_et + 1.0e7)


@pytest.mark.parametrize('query_et', [_START_ET, _STOP_ET], ids=['start', 'stop'])
def test_frozen_attitude_object_is_constant_across_the_exposure(
    pool: KernelPool, query_et: float
) -> None:
    """A frozen-attitude object carries one attitude, not the baseline history.

    Parameters:
        query_et: Epoch the corrected kernel is queried at, TDB seconds past
            J2000.
    """
    case = _build_case(pool, ck_frame_id=VOYAGER_CK_FRAME_ID, camera_frame=VOYAGER_CAMERA_FRAME)
    _swap_to_corrected(pool, case)
    read = _attitude_from_pool(VOYAGER_CK_FRAME_ID, _VOYAGER_SCLK_ID, query_et)
    truth = case.correction @ baseline_attitude(case.pointing.midtime_et)
    assert rotation_angle_between(read, truth) < _ANGLE_TOL_RAD


def test_frozen_attitude_object_carries_zero_angular_velocity(pool: KernelPool) -> None:
    """A constant-attitude segment declares an angular velocity of zero.

    Zero rather than none: an attitude that does not change has an angular
    velocity, and it is zero.  A segment declaring none would be skipped by
    ``ckgpav`` and ``sxform`` in favor of any other kernel covering the same
    object and epoch, which would answer with the uncorrected attitude.
    """
    case = _build_case(pool, ck_frame_id=VOYAGER_CK_FRAME_ID, camera_frame=VOYAGER_CAMERA_FRAME)
    assert case.segment.has_angular_velocity is True
    assert case.segment.avvs is not None
    assert np.array_equal(case.segment.avvs, np.zeros((case.segment.record_count, 3)))


def test_a_frozen_correction_reaches_sxform(pool: KernelPool) -> None:
    """A frozen segment answers ``sxform`` with its own attitude and zero rate."""
    case = _build_case(pool, ck_frame_id=VOYAGER_CK_FRAME_ID, camera_frame=VOYAGER_CAMERA_FRAME)
    pool.furnish(case.corrected_path)
    midtime = case.pointing.midtime_et
    ck_frame = str(cspyce.frmnam(VOYAGER_CK_FRAME_ID))
    xform = np.asarray(cspyce.sxform('J2000', ck_frame, midtime), np.float64)
    truth = case.correction @ baseline_attitude(midtime)
    _matrix, omega = cspyce.xf2rav(xform)
    assert rotation_angle_between(xform[:3, :3], truth) < _ANGLE_TOL_RAD
    assert np.array_equal(np.asarray(omega, np.float64), np.zeros(3))


def test_frozen_attitude_ignores_the_baseline_time_variation(pool: KernelPool) -> None:
    """The frozen segment's start and stop attitudes are the same rotation."""
    case = _build_case(pool, ck_frame_id=VOYAGER_CK_FRAME_ID, camera_frame=VOYAGER_CAMERA_FRAME)
    _swap_to_corrected(pool, case)
    at_start = _attitude_from_pool(VOYAGER_CK_FRAME_ID, _VOYAGER_SCLK_ID, _START_ET)
    at_stop = _attitude_from_pool(VOYAGER_CK_FRAME_ID, _VOYAGER_SCLK_ID, _STOP_ET)
    assert rotation_angle_between(at_start, at_stop) < _ANGLE_TOL_RAD


@pytest.mark.parametrize(
    ('ck_frame_id', 'sclk_id'),
    [(-82000, -82), (-77001, -77), (-98000, -98), (-31100, -31), (-32100, -32)],
    ids=['cassini', 'galileo', 'new-horizons', 'voyager-1', 'voyager-2'],
)
def test_resolve_sclk_id(ck_frame_id: int, sclk_id: int) -> None:
    """Each CK object resolves to its own spacecraft clock.

    Voyager 1 is the case integer division gets wrong: ``-31100 // 1000`` is
    -32 in Python, which is the other spacecraft.

    Parameters:
        ck_frame_id: SPICE id of the object a corrected kernel targets.
        sclk_id: Spacecraft clock that object's time tags are encoded against.
    """
    assert resolve_sclk_id(ck_frame_id) == sclk_id


def test_resolve_sclk_id_refuses_an_unknown_object() -> None:
    """An object outside the known set is refused rather than looked up."""
    with pytest.raises(ValueError, match='CK object -999999 is not one this writer knows'):
        resolve_sclk_id(-999999)


def test_resolve_sclk_id_refuses_a_clock_that_is_not_the_expected_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resolved clock is checked, not trusted; a wrong one is silent otherwise."""
    monkeypatch.setattr(cspyce, 'ckmeta', lambda ckid, meta: -99)
    with pytest.raises(
        ValueError, match='CK object -82000 resolves to spacecraft clock -99, not the expected -82'
    ):
        resolve_sclk_id(CASSINI_CK_FRAME_ID)


def test_segment_refuses_repeated_time_tags() -> None:
    """Time tags that do not strictly increase are refused before SPICE sees them."""
    quats = np.vstack([cspyce.m2q(np.eye(3))] * 2)
    with pytest.raises(ValueError, match='not strictly increasing'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='repeat',
            sclkdp=np.array([1.0, 1.0]),
            quats=quats,
            avvs=None,
        )


def test_segment_refuses_a_quaternion_count_mismatch() -> None:
    """A record set whose arrays disagree on length is refused."""
    quats = np.vstack([cspyce.m2q(np.eye(3))])
    with pytest.raises(ValueError, match=r'quats must have shape \(2, 4\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='mismatch',
            sclkdp=np.array([1.0, 2.0]),
            quats=quats,
            avvs=None,
        )


def test_segment_refuses_an_angular_velocity_count_mismatch() -> None:
    """Angular velocity that does not cover every record is refused."""
    quats = np.vstack([cspyce.m2q(np.eye(3))] * 2)
    with pytest.raises(ValueError, match=r'avvs must have shape \(2, 3\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='mismatch',
            sclkdp=np.array([1.0, 2.0]),
            quats=quats,
            avvs=np.zeros((1, 3)),
        )


@pytest.mark.parametrize('sclkdp', [np.float64(1.0), 1.0], ids=['numpy-scalar', 'python-float'])
def test_segment_refuses_a_scalar_time_tag(sclkdp: object) -> None:
    """A scalar where the record array belongs is refused, and by the documented type.

    A zero-dimensional array has no length to read, so the shape guard has to
    run before anything indexes it; otherwise the failure is an ``IndexError``
    from inside the guard's own arithmetic.

    Parameters:
        sclkdp: Scalar passed where the time tag array belongs.
    """
    with pytest.raises(ValueError, match=r'sclkdp must hold at least one time tag; got shape \(\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='scalar',
            sclkdp=cast(NDArrayFloatType, sclkdp),
            quats=np.vstack([cspyce.m2q(np.eye(3))]),
            avvs=None,
        )


@pytest.mark.parametrize('bad', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_segment_refuses_a_non_finite_time_tag(bad: float) -> None:
    """A non-finite time tag is refused rather than handed to ckw03.

    The ordering check is no defense: a NaN reads as strictly increasing
    because every comparison against it is False, and an infinity satisfies
    the ordering outright.

    Parameters:
        bad: Non-finite value planted in the second time tag.
    """
    quats = np.vstack([cspyce.m2q(np.eye(3))] * 2)
    with pytest.raises(ValueError, match=r'sclkdp holds a non-finite value at index \(1,\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='non-finite',
            sclkdp=np.array([1.0, bad]),
            quats=quats,
            avvs=None,
        )


@pytest.mark.parametrize('bad', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_segment_refuses_a_non_finite_quaternion(bad: float) -> None:
    """A non-finite quaternion component is refused, naming where it sits.

    Parameters:
        bad: Non-finite value planted in the second record's third component.
    """
    quats = np.vstack([cspyce.m2q(np.eye(3))] * 2)
    quats[1, 2] = bad
    with pytest.raises(ValueError, match=r'quats holds a non-finite value at index \(1, 2\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='non-finite',
            sclkdp=np.array([1.0, 2.0]),
            quats=quats,
            avvs=None,
        )


@pytest.mark.parametrize('bad', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_segment_refuses_a_non_finite_angular_velocity(bad: float) -> None:
    """A non-finite angular velocity component is refused, naming where it sits.

    Parameters:
        bad: Non-finite value planted in the first record's second component.
    """
    quats = np.vstack([cspyce.m2q(np.eye(3))] * 2)
    avvs = np.zeros((2, 3))
    avvs[0, 1] = bad
    with pytest.raises(ValueError, match=r'avvs holds a non-finite value at index \(0, 1\)'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='non-finite',
            sclkdp=np.array([1.0, 2.0]),
            quats=quats,
            avvs=avvs,
        )


def test_segment_refuses_an_empty_record_set() -> None:
    """A segment with no records is refused."""
    with pytest.raises(ValueError, match='sclkdp must hold at least one time tag'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='empty',
            sclkdp=np.array([]),
            quats=np.zeros((0, 4)),
            avvs=None,
        )


def test_segment_refuses_an_overlong_identifier() -> None:
    """A segment identifier longer than SPICE stores is refused."""
    quats = np.vstack([cspyce.m2q(np.eye(3))])
    with pytest.raises(ValueError, match='longer than the 40 characters SPICE stores'):
        CkSegment(
            ck_frame_id=CASSINI_CK_FRAME_ID,
            segid='X' * 41,
            sclkdp=np.array([1.0]),
            quats=quats,
            avvs=None,
        )


def test_build_segment_refuses_an_overlong_image_name(pool: KernelPool) -> None:
    """An image name that would not fit the segment identifier is refused."""
    with pytest.raises(ValueError, match='longer than the 40 characters'):
        _build_case(pool, image_name='N' * 41)
