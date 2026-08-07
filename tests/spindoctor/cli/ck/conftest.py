"""Hermetic SPICE fixtures and helpers for the C-kernel writer tests.

Every kernel these tests use is written here: a minimal leapseconds kernel, a
spacecraft clock kernel for the two clocks under test, a frame kernel defining
one CK-class frame per mission with a camera frame fixed to it at a rotation
nowhere near the identity, and the baseline C-kernels themselves, produced
through the writer's own segment primitives.  Nothing reads the holdings and
nothing needs a kernel from disk, so the whole suite runs with no environment
set.

Being hermetic about files is not enough, because the SPICE kernel pool is
process-global.  A test that had loaded a real image through oops leaves real
mission kernels furnished, and they define the same objects these kernels do --
a real Cassini frame kernel names -82000, a real clock kernel names -82 -- so
what these tests measure would depend on which file ran first in the worker.
Every test here therefore runs against an emptied pool, and the pool it
borrowed is furnished again afterwards.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from tests.kernel_pool import isolated_kernel_pool

from spindoctor.cli.ck.pointing import NDArrayFloatType
from spindoctor.cli.ck.segment import CkSegment, write_segment

# The two CK objects the tests exercise: a bus whose baseline attitude varies
# across the exposure, and a scan platform standing in for the frozen-attitude
# mission.  Both ids are real, because the writer refuses any other.
CASSINI_CK_FRAME_ID = -82000
CASSINI_CAMERA_FRAME = 'SD_TEST_NAC'
VOYAGER_CK_FRAME_ID = -31100
VOYAGER_CAMERA_FRAME = 'SD_TEST_VGCAM'

# The epoch the test kernels are built around, and the clock tick size the test
# SCLK declares.  ET0 is far enough from J2000 that a nanosecond exposure is
# below the resolution of a double, which is what makes the degenerate
# single-record path reachable without hand-feeding three identical epochs.
ET0 = 5.0e8
TICKS_PER_SECOND = 256.0

# The baseline attitude turns at a constant rate about a fixed axis, which
# makes SPICE's spherical interpolation exact at every epoch between records
# and lets a test assert against the analytic attitude rather than against
# another interpolation.
_BASELINE_AXIS = np.array([0.30, -0.50, 0.81])
_BASELINE_RATE_RAD_S = 3.0e-3
_BASELINE_ORIENTATION_AXIS = np.array([0.6, 0.7, -0.39])
_BASELINE_ORIENTATION_RAD = 1.13

# Angular velocity that varies from record to record and points along none of
# the axes above, so a copy that quietly rotated it would not come back equal.
_AV_AT_ET0 = np.array([1.7e-3, -4.2e-4, 9.1e-4])
_AV_RATE = np.array([2.3e-5, 6.1e-5, -1.9e-5])

_LSK_TEXT = """KPL/LSK

Minimal leapseconds kernel written by the SpinDoctor C-kernel writer tests.

\\begindata
DELTET/DELTA_T_A = 32.184
DELTET/K         = 1.657D-3
DELTET/EB        = 1.671D-2
DELTET/M         = ( 6.239996D0 1.99096871D-7 )
DELTET/DELTA_AT  = ( 10, @1972-JAN-1
                     32, @1999-JAN-1
                     37, @2017-JAN-1 )
\\begintext
"""

_SCLK_TEXT = """KPL/SCLK

Minimal spacecraft clock kernel written by the SpinDoctor C-kernel writer
tests.  Both clocks tick at 1/256 second and run on TDB, so an encoded tick is
exactly 256 times the TDB seconds past J2000.

\\begindata
SCLK_KERNEL_ID           = ( @2000-01-01/00:00:00 )

SCLK_DATA_TYPE_82        = ( 1 )
SCLK01_TIME_SYSTEM_82    = ( 1 )
SCLK01_N_FIELDS_82       = ( 2 )
SCLK01_MODULI_82         = ( 4294967296 256 )
SCLK01_OFFSETS_82        = ( 0 0 )
SCLK01_OUTPUT_DELIM_82   = ( 1 )
SCLK_PARTITION_START_82  = ( 0.0000000000000E+00 )
SCLK_PARTITION_END_82    = ( 1.0995116277750E+12 )
SCLK01_COEFFICIENTS_82   = ( 0.0000000000000E+00 0.0000000000000E+00 1.0000000000000E+00 )

SCLK_DATA_TYPE_31        = ( 1 )
SCLK01_TIME_SYSTEM_31    = ( 1 )
SCLK01_N_FIELDS_31       = ( 2 )
SCLK01_MODULI_31         = ( 4294967296 256 )
SCLK01_OFFSETS_31        = ( 0 0 )
SCLK01_OUTPUT_DELIM_31   = ( 1 )
SCLK_PARTITION_START_31  = ( 0.0000000000000E+00 )
SCLK_PARTITION_END_31    = ( 1.0995116277750E+12 )
SCLK01_COEFFICIENTS_31   = ( 0.0000000000000E+00 0.0000000000000E+00 1.0000000000000E+00 )
\\begintext
"""

_FK_TEXT = """KPL/FK

Minimal frame kernel written by the SpinDoctor C-kernel writer tests.  Each
camera frame sits at a large fixed rotation from the CK object's frame, so a
test cannot pass by confusing the two.

\\begindata
FRAME_SD_TEST_BUS         = -82000
FRAME_-82000_NAME         = 'SD_TEST_BUS'
FRAME_-82000_CLASS        = 3
FRAME_-82000_CLASS_ID     = -82000
FRAME_-82000_CENTER       = -82
CK_-82000_SCLK            = -82
CK_-82000_SPK             = -82

FRAME_SD_TEST_NAC         = -82361
FRAME_-82361_NAME         = 'SD_TEST_NAC'
FRAME_-82361_CLASS        = 4
FRAME_-82361_CLASS_ID     = -82361
FRAME_-82361_CENTER       = -82
TKFRAME_-82361_SPEC       = 'ANGLES'
TKFRAME_-82361_RELATIVE   = 'SD_TEST_BUS'
TKFRAME_-82361_ANGLES     = ( -90.0, 60.0, 30.0 )
TKFRAME_-82361_AXES       = ( 3, 1, 3 )
TKFRAME_-82361_UNITS      = 'DEGREES'

FRAME_SD_TEST_PLATFORM    = -31100
FRAME_-31100_NAME         = 'SD_TEST_PLATFORM'
FRAME_-31100_CLASS        = 3
FRAME_-31100_CLASS_ID     = -31100
FRAME_-31100_CENTER       = -31
CK_-31100_SCLK            = -31
CK_-31100_SPK             = -31

FRAME_SD_TEST_VGCAM       = -31101
FRAME_-31101_NAME         = 'SD_TEST_VGCAM'
FRAME_-31101_CLASS        = 4
FRAME_-31101_CLASS_ID     = -31101
FRAME_-31101_CENTER       = -31
TKFRAME_-31101_SPEC       = 'ANGLES'
TKFRAME_-31101_RELATIVE   = 'SD_TEST_PLATFORM'
TKFRAME_-31101_ANGLES     = ( 20.0, -35.0, 110.0 )
TKFRAME_-31101_AXES       = ( 1, 2, 3 )
TKFRAME_-31101_UNITS      = 'DEGREES'
\\begintext
"""


class KernelPool:
    """The kernels one test furnished, so teardown can unload just those.

    Parameters:
        root: Directory the test may write its kernels into.
    """

    def __init__(self, root: Path) -> None:
        """Start with an empty record of furnished kernels."""
        self.root = root
        self._furnished: list[Path] = []

    def furnish(self, path: Path) -> None:
        """Furnish one kernel and remember it.

        Parameters:
            path: The kernel to furnish.
        """
        cspyce.furnsh(str(path))
        self._furnished.append(path)

    def unload(self, path: Path) -> None:
        """Unload one furnished kernel.

        Parameters:
            path: The kernel to unload.
        """
        cspyce.unload(str(path))
        self._furnished.remove(path)

    def unload_all(self) -> None:
        """Unload every kernel this pool furnished, most recent first."""
        for path in reversed(list(self._furnished)):
            self.unload(path)


SUPPORT_KERNEL_NAMES = ('test.tls', 'test.tsc', 'test.tf')
"""The hermetic leapseconds, spacecraft clock and frame kernels, in load order."""


def write_support_kernels(root: Path) -> tuple[Path, ...]:
    """Write the hermetic LSK, SCLK and FK into a directory.

    Parameters:
        root: Directory to write them into.

    Returns:
        Their paths, in the order they should be furnished.
    """
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in zip(SUPPORT_KERNEL_NAMES, (_LSK_TEXT, _SCLK_TEXT, _FK_TEXT), strict=True):
        path = root / name
        path.write_text(text)
        written.append(path)
    return tuple(written)


@pytest.fixture(scope='module', autouse=True)
def empty_kernel_pool() -> Iterator[None]:
    """Run the tests in each module here against a pool holding nothing else.

    Autouse rather than requested, because the tests that read the pool itself
    -- what a meta-kernel furnished, which files define a frame -- take no
    fixture at all and are exactly the ones a foreign kernel misleads.

    Module-scoped rather than per test, because emptying and restoring a pool
    that oops has filled costs about a second and no foreign kernel can appear
    between two tests of one module: nothing else is running.  Every test here
    already unloads what it furnished, and one that stopped would now be
    visible to the next test rather than hidden by a reset.

    Yields:
        Nothing; the module's tests run against an empty pool.
    """
    with isolated_kernel_pool():
        yield


@pytest.fixture
def pool(tmp_path: Path, empty_kernel_pool: None) -> Iterator[KernelPool]:
    """Furnish the hermetic LSK, SCLK and FK, and unload them afterwards.

    Parameters:
        empty_kernel_pool: Requested by name so the pool is emptied before
            these kernels go into it, whatever order pytest would otherwise
            pick.

    Yields:
        The record of what was furnished, for a test that furnishes more.
    """
    kernels = KernelPool(tmp_path)
    for path in write_support_kernels(tmp_path):
        kernels.furnish(path)
    try:
        yield kernels
    finally:
        kernels.unload_all()


def axis_rotation(axis: NDArrayFloatType, angle_rad: float) -> NDArrayFloatType:
    """Return the rotation of ``angle_rad`` about ``axis``.

    Parameters:
        axis: Rotation axis; need not be normalized.
        angle_rad: Rotation angle in radians.

    Returns:
        The 3x3 rotation matrix, active on vectors.
    """
    unit = np.asarray(axis, dtype=np.float64) / float(np.linalg.norm(axis))
    rotation: NDArrayFloatType = np.asarray(cspyce.axisar(unit, angle_rad), dtype=np.float64)
    return rotation


def baseline_attitude(et: float) -> NDArrayFloatType:
    """Return the baseline J2000-to-CK-object attitude at one epoch.

    The attitude turns at a constant rate about a fixed axis, so SPICE's
    interpolation between any two records reproduces this function exactly and
    a test can assert against it directly.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation.
    """
    turned = axis_rotation(_BASELINE_AXIS, _BASELINE_RATE_RAD_S * (et - ET0))
    start = axis_rotation(_BASELINE_ORIENTATION_AXIS, _BASELINE_ORIENTATION_RAD)
    attitude: NDArrayFloatType = turned @ start
    return attitude


def sweeping_attitude(et: float) -> NDArrayFloatType:
    """Return a baseline attitude whose rotation angle crosses 180 degrees.

    The angle passes through 180 degrees two seconds after ET0, which is the
    midtime of the exposure the tests use.  A quaternion sequence taken from
    this attitude flips sign between adjacent records there, because ``m2q``
    fixes the scalar component non-negative and the scalar component changes
    sign as the rotation angle passes 180 degrees.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The 3x3 rotation.
    """
    angle_rad = np.radians(160.0) + np.radians(10.0) * (et - ET0)
    return axis_rotation(_BASELINE_AXIS, float(angle_rad))


def baseline_angular_velocity(et: float) -> NDArrayFloatType:
    """Return the baseline angular velocity at one epoch, in J2000.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The 3-vector, varying from record to record.
    """
    velocity: NDArrayFloatType = _AV_AT_ET0 + _AV_RATE * (et - ET0)
    return velocity


def baseline_segment(
    *,
    ck_frame_id: int,
    sclk_id: int,
    epochs: Sequence[float],
    attitude: Callable[[float], NDArrayFloatType],
    angular_velocity: Callable[[float], NDArrayFloatType] | None,
    segid: str = 'baseline',
) -> CkSegment:
    """Build one baseline segment from an attitude history.

    The quaternions come straight from ``cspyce.m2q`` rather than through the
    writer's sign-continuity helper, so a baseline is exactly what a real
    kernel is: whatever the producer stored.

    Parameters:
        ck_frame_id: SPICE id of the object the segment describes.
        sclk_id: Spacecraft clock the time tags are encoded against.
        epochs: Record epochs, TDB seconds past J2000, strictly increasing.
        attitude: The J2000-to-CK-object rotation at an epoch.
        angular_velocity: The angular velocity at an epoch, or None for a
            segment carrying none.
        segid: Segment identifier.

    Returns:
        The segment.
    """
    ticks = [float(cspyce.sce2c(sclk_id, et)) for et in epochs]
    quats = np.vstack([np.asarray(cspyce.m2q(attitude(et)), dtype=np.float64) for et in epochs])
    avvs = None
    if angular_velocity is not None:
        avvs = np.vstack([angular_velocity(et) for et in epochs])
    return CkSegment(
        ck_frame_id=ck_frame_id,
        segid=segid,
        sclkdp=np.asarray(ticks, dtype=np.float64),
        quats=quats,
        avvs=avvs,
    )


def write_type1_ck(
    path: Path,
    *,
    ck_frame_id: int,
    ticks: Sequence[float],
    attitude: Callable[[float], NDArrayFloatType],
    sclk_id: int,
) -> None:
    """Write a discrete (type 1) C-kernel holding pointing at given time tags.

    Every other kernel these tests write is type 3, which interpolates and
    therefore answers any epoch inside its window whatever tolerance is asked
    for.  A type 1 kernel answers only within a tolerance of a record it
    actually holds, which is what the real Voyager ISS baselines are and what
    makes a lookup tolerance decide anything at all.

    Parameters:
        path: File to create.
        ck_frame_id: SPICE id of the object the segment describes.
        ticks: Encoded SCLK time tags of the records, strictly increasing.
        attitude: The J2000-to-CK-object rotation at an epoch.
        sclk_id: Spacecraft clock the time tags are encoded against, used to
            turn each tag back into the epoch the attitude is taken at.
    """
    quats = np.vstack(
        [
            np.asarray(cspyce.m2q(attitude(float(cspyce.sct2e(sclk_id, tick)))), dtype=np.float64)
            for tick in ticks
        ]
    )
    handle = cspyce.ckopn(str(path), 'baseline', 0)
    try:
        cspyce.ckw01(
            handle,
            float(ticks[0]),
            float(ticks[-1]),
            ck_frame_id,
            'J2000',
            False,
            'discrete-baseline',
            np.asarray(ticks, dtype=np.float64),
            quats,
            np.zeros((len(ticks), 3), dtype=np.float64),
        )
    finally:
        cspyce.ckcls(handle)


def write_ck(path: Path, segments: Sequence[CkSegment]) -> None:
    """Write segments to a new C-kernel.

    The handle is closed even when a write raises.  SPICE caps the number of
    DAF files open at once, so a handle leaked by a test that exercises a
    failure path would go on to break an unrelated ``ckopn`` later in the same
    worker, with an error naming neither the leak nor its cause.

    Parameters:
        path: File to create.
        segments: The segments to add, in order.
    """
    handle = cspyce.ckopn(str(path), 'baseline', 0)
    try:
        for segment in segments:
            write_segment(handle, segment)
    finally:
        cspyce.ckcls(handle)


def write_baseline_ck(
    path: Path,
    *,
    ck_frame_id: int,
    sclk_id: int,
    epochs: Sequence[float],
    attitude: Callable[[float], NDArrayFloatType],
    angular_velocity: Callable[[float], NDArrayFloatType] | None,
) -> None:
    """Write the baseline C-kernel a corrected segment is measured against.

    Parameters:
        path: File to create.
        ck_frame_id: SPICE id of the object the segment describes.
        sclk_id: Spacecraft clock the time tags are encoded against.
        epochs: Record epochs, TDB seconds past J2000, strictly increasing.
        attitude: The J2000-to-CK-object rotation at an epoch.
        angular_velocity: The angular velocity at an epoch, or None to write a
            segment carrying none.
    """
    write_ck(
        path,
        [
            baseline_segment(
                ck_frame_id=ck_frame_id,
                sclk_id=sclk_id,
                epochs=epochs,
                attitude=attitude,
                angular_velocity=angular_velocity,
            )
        ],
    )


def image_metadata(
    *,
    image_name: str,
    cmatrix: NDArrayFloatType | None,
    cmatrix_original: NDArrayFloatType,
    camera_frame: str,
    ck_frame_id: int,
    start_et: float,
    stop_et: float,
    midtime_et: float | None = None,
    exposure_s: float | None = None,
    status: str = 'success',
    instrument: str | None = None,
    camera: str | None = None,
    shutter_mode: str | None = None,
    kernels: Sequence[str] | None = (),
    rotation_deg: float | None = None,
    sclk_midtime: str | None = None,
    offset: Sequence[float] | None = None,
    sigma_px: Sequence[float] | None = None,
    confidence: float | None = None,
    confidence_rank: str | None = None,
    status_reason: str | None = None,
) -> dict[str, Any]:
    """Build a per-image metadata document shaped like the pipeline's own.

    Only the fields the generator reads are populated; the rest of the schema
    is irrelevant to it and would only make a test harder to read.

    Parameters:
        image_name: Basename recorded for the image.
        cmatrix: The corrected C-matrix, or None to omit it, which is what an
            image that navigated without one records.
        cmatrix_original: The uncorrected C-matrix.
        camera_frame: SPICE name of the camera frame.
        ck_frame_id: SPICE id of the object a corrected kernel targets.
        start_et: Exposure start, TDB seconds past J2000.
        stop_et: Exposure stop, TDB seconds past J2000.
        midtime_et: Exposure midtime; the arithmetic midpoint when None.
        exposure_s: Exposure duration; ``stop_et - start_et`` when None.
        status: The navigation status.
        instrument: The registered instrument identity; the field is omitted
            when None.
        camera: The camera that took the image; the field is omitted when
            None.
        shutter_mode: The shutter mode; the field is omitted when None.
        kernels: The SPICE kernel basenames recorded in the provenance block;
            the provenance block itself is omitted when None.
        rotation_deg: A fitted camera rotation; the field is omitted when
            None.
        sclk_midtime: The spacecraft clock string recorded for the midtime;
            the field is omitted when None.
        offset: The navigated ``[dv, du]`` offset; the top-level field is
            omitted when None.
        sigma_px: The recorded ``[dv, du]`` uncertainty; the field is omitted
            when None.
        confidence: The recorded confidence; the top-level field is omitted
            when None.
        confidence_rank: The recorded rank; the field is omitted when None.
        status_reason: The recorded status reason; the field is omitted when
            None.

    Returns:
        The metadata dict.
    """
    pointing: dict[str, Any] = {}
    if cmatrix is not None:
        pointing['cmatrix'] = [float(value) for value in np.asarray(cmatrix).reshape(9)]
    pointing['cmatrix_original'] = [
        float(value) for value in np.asarray(cmatrix_original).reshape(9)
    ]
    pointing['camera_frame'] = camera_frame
    pointing['ck_frame_id'] = ck_frame_id
    midtime = (start_et + stop_et) / 2.0 if midtime_et is None else midtime_et
    times: dict[str, Any] = {
        'start_et': start_et,
        'stop_et': stop_et,
        'midtime_et': midtime,
        'exposure_s': stop_et - start_et if exposure_s is None else exposure_s,
    }
    if sclk_midtime is not None:
        times['sclk_midtime'] = sclk_midtime
    navigation_result: dict[str, Any] = {'pointing': pointing, 'times': times}
    if kernels is not None:
        navigation_result['provenance'] = {'spice_kernels': list(kernels)}
    if rotation_deg is not None:
        navigation_result['rotation_deg'] = rotation_deg
    if sigma_px is not None:
        navigation_result['sigma_px'] = list(sigma_px)
    if confidence_rank is not None:
        navigation_result['confidence_rank'] = confidence_rank
    if status_reason is not None:
        navigation_result['status_reason'] = status_reason
    observation: dict[str, Any] = {'image_name': image_name}
    if instrument is not None:
        observation['instrument'] = instrument
    if camera is not None:
        observation['camera'] = camera
    if shutter_mode is not None:
        observation['shutter_mode'] = shutter_mode
    metadata: dict[str, Any] = {
        'status': status,
        'observation': observation,
        'navigation_result': navigation_result,
    }
    if offset is not None:
        metadata['offset'] = list(offset)
    if confidence is not None:
        metadata['confidence'] = confidence
    return metadata


def rotation_angle_between(first: NDArrayFloatType, second: NDArrayFloatType) -> float:
    """Return the angle of the rotation carrying one attitude onto another.

    The angle is taken through a quaternion rather than through the trace,
    because the trace form loses half its digits as the angle goes to zero,
    which is exactly the regime these comparisons live in.  It is zero if and
    only if the two matrices are equal, so it detects a wrong direction as
    readily as a wrong magnitude.

    Parameters:
        first: A 3x3 rotation.
        second: A 3x3 rotation.

    Returns:
        The angle in radians, in [0, pi].
    """
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    quat = np.asarray(cspyce.m2q(relative), dtype=np.float64)
    return float(2.0 * np.arctan2(float(np.linalg.norm(quat[1:])), abs(float(quat[0]))))
