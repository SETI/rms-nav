"""Hermetic SPICE fixtures and helpers for the C-kernel writer tests.

Every kernel these tests use is written here: a minimal leapseconds kernel, a
spacecraft clock kernel for the two clocks under test, a frame kernel defining
one CK-class frame per mission with a camera frame fixed to it at a rotation
nowhere near the identity, and the baseline C-kernels themselves, produced
through the writer's own segment primitives.  Nothing reads the holdings and
nothing needs a kernel from disk, so the whole suite runs with no environment
set.

The SPICE kernel pool is process-global, so the pool fixture unloads exactly
what it furnished instead of calling ``kclear``, which would also discard
whatever an unrelated test furnished earlier in the same worker.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cspyce
import numpy as np
import pytest

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


@pytest.fixture
def pool(tmp_path: Path) -> Iterator[KernelPool]:
    """Furnish the hermetic LSK, SCLK and FK, and unload them afterwards."""
    kernels = KernelPool(tmp_path)
    for name, text in (('test.tls', _LSK_TEXT), ('test.tsc', _SCLK_TEXT), ('test.tf', _FK_TEXT)):
        path = tmp_path / name
        path.write_text(text)
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
