"""Shared builders for the assignment tests.

The assignment step is exercised from two test files -- the Cassini frame-chain
tests and the Voyager frozen-attitude tests -- and both build the same things:
candidate C-kernels holding either the baseline attitude or a decoy turned away
from it, and image entries recording the attitude a navigation run would have
frozen.  Those builders live here, in a plain module, so neither test file
imports the other.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    VOYAGER_CAMERA_FRAME,
    VOYAGER_CK_FRAME_ID,
    axis_rotation,
    baseline_attitude,
    image_metadata,
    write_baseline_ck,
    write_type1_ck,
)

from spindoctor.cli.ck.images import ImageEntry
from spindoctor.cli.ck.pointing import NDArrayFloatType

# The clocks the two test objects are tagged against, matching the test SCLK.
CASSINI_SCLK_ID = -82
VOYAGER_SCLK_ID = -31

# Candidate kernels carry half-second records over four seconds from ET0, and
# the exposure under test sits inside that.
RECORD_STEP_S = 0.5
RECORDS = 9
START_ET = ET0 + 1.0
EXPOSURE_S = 2.0
MIDTIME_ET = START_ET + EXPOSURE_S / 2.0

# Axis for the rotations that turn a baseline into something else.  It is
# shared with neither the baseline's own axis nor its orientation.
DECOY_AXIS = np.array([-0.8, 0.1, 0.59])

# A decoy far outside any tolerance.
WRONG_RAD = np.radians(5.0)

# Real names from the holdings.  The decoy sorts after the true baseline, so it
# would win the tie-break if the reproduction test did not exclude it.
TRUE_NAME = '03236_04002ra.bc'
DECOY_NAME = 'zz04002_04009ra.bc'

IMAGE_NAME = 'N1484573295_1.IMG'

# A midtime that does not land on a whole clock tick, so the snapped lookup
# answers at a measurably different epoch from the midtime itself.
VOYAGER_START_ET = ET0 + 1.003
VOYAGER_MIDTIME_ET = VOYAGER_START_ET + EXPOSURE_S / 2.0

VOYAGER_KERNEL_NAME = 'vg1_sat_version1_type1_iss_sedr.bc'
VOYAGER_IMAGE_NAME = 'C1205021_CALIB.IMG'


def turned(angle_rad: float) -> Callable[[float], NDArrayFloatType]:
    """Return an attitude history turned from the baseline by a fixed rotation.

    Parameters:
        angle_rad: How far from the baseline to turn.

    Returns:
        The attitude at an epoch.
    """
    rotation = axis_rotation(DECOY_AXIS, angle_rad)

    def attitude(et: float) -> NDArrayFloatType:
        turned_attitude: NDArrayFloatType = rotation @ baseline_attitude(et)
        return turned_attitude

    return attitude


def write_candidate(
    directory: Path,
    name: str,
    *,
    attitude: Callable[[float], NDArrayFloatType] = baseline_attitude,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    sclk_id: int = CASSINI_SCLK_ID,
    start_et: float = ET0,
    records: int = RECORDS,
) -> Path:
    """Write one candidate C-kernel into a directory, creating the directory.

    Parameters:
        directory: Directory to write into.
        name: Basename of the kernel.
        attitude: The J2000-to-CK-object rotation at an epoch.
        ck_frame_id: SPICE id of the object the kernel describes.
        sclk_id: The spacecraft clock its time tags are encoded against.
        start_et: First record epoch, TDB seconds past J2000.
        records: Number of half-second records.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    write_baseline_ck(
        path,
        ck_frame_id=ck_frame_id,
        sclk_id=sclk_id,
        epochs=[start_et + step * RECORD_STEP_S for step in range(records)],
        attitude=attitude,
        angular_velocity=None,
    )
    return path


def camera_from_object(ck_frame_id: int, camera_frame: str) -> NDArrayFloatType:
    """Return the fixed rotation from a CK object's frame to a camera frame.

    The frame kernel defining both frames must be furnished.

    Parameters:
        ck_frame_id: SPICE id of the object.
        camera_frame: SPICE name of the camera frame.

    Returns:
        The 3x3 rotation, which the frame kernel defines as a constant.
    """
    matrix: NDArrayFloatType = np.asarray(
        cspyce.pxform(str(cspyce.frmnam(ck_frame_id)), camera_frame, 0.0), dtype=np.float64
    )
    return matrix


def image_entry(
    *,
    cmatrix_original: NDArrayFloatType,
    kernels: tuple[str, ...],
    image_name: str = IMAGE_NAME,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    camera_frame: str = CASSINI_CAMERA_FRAME,
    start_et: float = START_ET,
    **overrides: Any,
) -> ImageEntry:
    """Read one image's entry from metadata built around a recorded baseline.

    Parameters:
        cmatrix_original: The uncorrected attitude the image recorded.
        kernels: The kernel basenames its provenance recorded.
        image_name: Basename recorded for the image.
        ck_frame_id: SPICE id of the object its correction targets.
        camera_frame: SPICE name of its camera frame.
        start_et: Exposure start, TDB seconds past J2000.
        overrides: Further keyword arguments for the metadata builder.

    Returns:
        The entry.
    """
    defaults: dict[str, Any] = {
        'image_name': image_name,
        'cmatrix': axis_rotation(DECOY_AXIS, 0.5) @ np.asarray(cmatrix_original),
        'cmatrix_original': cmatrix_original,
        'camera_frame': camera_frame,
        'ck_frame_id': ck_frame_id,
        'start_et': start_et,
        'stop_et': start_et + EXPOSURE_S,
        'camera': 'NAC',
        'kernels': kernels,
    }
    defaults.update(overrides)
    return ImageEntry.from_metadata(image_metadata(**defaults))


def cassini_recorded(*, midtime_et: float = MIDTIME_ET) -> NDArrayFloatType:
    """Return the attitude a Cassini image navigated against, in camera terms.

    The frame chain is evaluated the way ``pxform`` evaluates it: the constant
    rotation from the bus frame to the camera frame, composed onto the bus
    attitude the baseline kernel holds at the midtime.  The test frame kernel
    must be furnished.

    Parameters:
        midtime_et: The exposure midtime.

    Returns:
        The 3x3 J2000-to-camera rotation.
    """
    recorded: NDArrayFloatType = camera_from_object(
        CASSINI_CK_FRAME_ID, CASSINI_CAMERA_FRAME
    ) @ baseline_attitude(midtime_et)
    return recorded


def voyager_recorded(snapped_et: float) -> NDArrayFloatType:
    """Return the attitude a Voyager image froze, in camera terms.

    Parameters:
        snapped_et: The epoch the pointing lookup actually answered at.

    Returns:
        The 3x3 J2000-to-camera rotation.
    """
    recorded: NDArrayFloatType = camera_from_object(
        VOYAGER_CK_FRAME_ID, VOYAGER_CAMERA_FRAME
    ) @ baseline_attitude(snapped_et)
    return recorded


def snapped_et(midtime_et: float) -> float:
    """Return the epoch the whole-tick pointing lookup lands on.

    Parameters:
        midtime_et: The exposure midtime.

    Returns:
        TDB seconds past J2000 of the encoded whole tick.
    """
    return float(cspyce.sct2e(VOYAGER_SCLK_ID, float(cspyce.sce2t(VOYAGER_SCLK_ID, midtime_et))))


def write_discrete_candidate(
    directory: Path, name: str, *, offset_ticks: float, midtime_et: float
) -> Path:
    """Write a discrete baseline holding one record near an epoch.

    Parameters:
        directory: Directory to write into.
        name: Basename of the kernel.
        offset_ticks: How far after the midtime's whole tick the record sits.
        midtime_et: The exposure midtime the record is placed relative to.

    Returns:
        The path written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    tick = float(cspyce.sce2t(VOYAGER_SCLK_ID, midtime_et)) + offset_ticks
    write_type1_ck(
        path,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        ticks=[tick],
        attitude=baseline_attitude,
        sclk_id=VOYAGER_SCLK_ID,
    )
    return path


def discrete_entry(offset_ticks: float, *, exposure_s: float = EXPOSURE_S) -> ImageEntry:
    """Build the entry for an image whose baseline holds one nearby record.

    Parameters:
        offset_ticks: How far after the midtime's whole tick the record sits.
        exposure_s: The recorded exposure, which widens the lookup tolerance.

    Returns:
        The entry, recording the attitude at the record's own epoch.
    """
    midtime = VOYAGER_MIDTIME_ET
    tick = float(cspyce.sce2t(VOYAGER_SCLK_ID, midtime)) + offset_ticks
    record_et = float(cspyce.sct2e(VOYAGER_SCLK_ID, tick))
    start = midtime - exposure_s / 2.0
    return image_entry(
        cmatrix_original=voyager_recorded(record_et),
        kernels=(VOYAGER_KERNEL_NAME,),
        image_name=VOYAGER_IMAGE_NAME,
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=start,
        stop_et=start + exposure_s,
        midtime_et=midtime,
        exposure_s=exposure_s,
    )
