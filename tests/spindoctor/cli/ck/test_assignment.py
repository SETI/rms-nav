"""Hermetic tests for ``spindoctor.cli.ck.assignment``.

Each test writes its own candidate C-kernels -- one holding the attitude an
image really navigated against and one or more holding something else -- and
lets the assignment step work out which is which by furnishing them and asking.
The decoys are named so that they would win the tie-break if reproduction were
skipped, so a test cannot pass by chance of ordering.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    VOYAGER_CAMERA_FRAME,
    VOYAGER_CK_FRAME_ID,
    KernelPool,
    axis_rotation,
    baseline_attitude,
    image_metadata,
    write_baseline_ck,
)

from spindoctor.cli.ck.assignment import (
    Assignment,
    assign_images,
    attitudes_reproduce,
    group_for_output,
    output_basename,
    reproduces_baseline,
    rotation_angle_rad,
)
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import CkFile, KernelClass, build_ck_index
from spindoctor.cli.ck.pointing import NDArrayFloatType

# The clocks the two test objects are tagged against, matching the test SCLK.
_CASSINI_SCLK_ID = -82
_VOYAGER_SCLK_ID = -31

# Candidate kernels carry half-second records over four seconds from ET0, and
# the exposure under test sits inside that.
_RECORD_STEP_S = 0.5
_RECORDS = 9
_START_ET = ET0 + 1.0
_EXPOSURE_S = 2.0
_MIDTIME_ET = _START_ET + _EXPOSURE_S / 2.0

# Axis for the rotations that turn a baseline into something else.  It is
# shared with neither the baseline's own axis nor its orientation.
_DECOY_AXIS = np.array([-0.8, 0.1, 0.59])

# A decoy far outside any tolerance, and the two that bracket the reproduction
# bound of 1e-9 radians.
_WRONG_RAD = np.radians(5.0)
_JUST_OUTSIDE_RAD = 1e-8
_WELL_INSIDE_RAD = 1e-10

# Real names from the holdings.  The decoy sorts after the true baseline, so it
# would win the tie-break if the reproduction test did not exclude it.
_TRUE_NAME = '03236_04002ra.bc'
_DECOY_NAME = 'zz04002_04009ra.bc'

_IMAGE_NAME = 'N1484573295_1.IMG'


def _turned(angle_rad: float) -> Callable[[float], NDArrayFloatType]:
    """Return an attitude history turned from the baseline by a fixed rotation.

    Parameters:
        angle_rad: How far from the baseline to turn.

    Returns:
        The attitude at an epoch.
    """
    rotation = axis_rotation(_DECOY_AXIS, angle_rad)

    def attitude(et: float) -> NDArrayFloatType:
        turned: NDArrayFloatType = rotation @ baseline_attitude(et)
        return turned

    return attitude


def _write_candidate(
    directory: Path,
    name: str,
    *,
    attitude: Callable[[float], NDArrayFloatType] = baseline_attitude,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    sclk_id: int = _CASSINI_SCLK_ID,
    start_et: float = ET0,
    records: int = _RECORDS,
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
        epochs=[start_et + step * _RECORD_STEP_S for step in range(records)],
        attitude=attitude,
        angular_velocity=None,
    )
    return path


def _camera_from_object(ck_frame_id: int, camera_frame: str) -> NDArrayFloatType:
    """Return the fixed rotation from a CK object's frame to a camera frame.

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


def _entry(
    *,
    cmatrix_original: NDArrayFloatType,
    kernels: tuple[str, ...],
    image_name: str = _IMAGE_NAME,
    ck_frame_id: int = CASSINI_CK_FRAME_ID,
    camera_frame: str = CASSINI_CAMERA_FRAME,
    start_et: float = _START_ET,
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
        'cmatrix': axis_rotation(_DECOY_AXIS, 0.5) @ np.asarray(cmatrix_original),
        'cmatrix_original': cmatrix_original,
        'camera_frame': camera_frame,
        'ck_frame_id': ck_frame_id,
        'start_et': start_et,
        'stop_et': start_et + _EXPOSURE_S,
        'camera': 'NAC',
        'kernels': kernels,
    }
    defaults.update(overrides)
    return ImageEntry.from_metadata(image_metadata(**defaults))


def _cassini_recorded(pool: KernelPool, *, midtime_et: float = _MIDTIME_ET) -> NDArrayFloatType:
    """Return the attitude a Cassini image navigated against, in camera terms.

    The frame chain is evaluated the way ``pxform`` evaluates it: the constant
    rotation from the bus frame to the camera frame, composed onto the bus
    attitude the baseline kernel holds at the midtime.

    Parameters:
        pool: The test's kernel pool, which must have the frame kernel
            furnished.
        midtime_et: The exposure midtime.

    Returns:
        The 3x3 J2000-to-camera rotation.
    """
    recorded: NDArrayFloatType = _camera_from_object(
        CASSINI_CK_FRAME_ID, CASSINI_CAMERA_FRAME
    ) @ baseline_attitude(midtime_et)
    return recorded


def test_the_recorded_baseline_is_what_pxform_gives(pool: KernelPool, tmp_path: Path) -> None:
    """The tests' idea of the recorded attitude is SPICE's own.

    Everything below plants ``cmatrix_original`` analytically, so this pins
    that the planted value is what the frame chain actually evaluates to.
    """
    path = _write_candidate(tmp_path / 'CK-reconstructed', _TRUE_NAME)
    pool.furnish(path)
    evaluated = np.asarray(
        cspyce.pxform('J2000', CASSINI_CAMERA_FRAME, _MIDTIME_ET), dtype=np.float64
    )
    assert rotation_angle_rad(_cassini_recorded(pool), evaluated) < 1e-12


def test_assignment_keeps_the_reproducing_candidate(pool: KernelPool, tmp_path: Path) -> None:
    """The candidate that answers the recorded attitude is the one assigned."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    _write_candidate(root, _DECOY_NAME, attitude=_turned(_WRONG_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME, _DECOY_NAME))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == _TRUE_NAME


def test_assignment_reports_no_reproducing_baseline(pool: KernelPool, tmp_path: Path) -> None:
    """An image whose baseline no candidate reproduces gets no segment."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _DECOY_NAME, attitude=_turned(_WRONG_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_DECOY_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is None
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_a_baseline_that_drifted_past_the_bound_does_not_reproduce(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel differing by ten nanoradians is a different kernel."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME, attitude=_turned(_JUST_OUTSIDE_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_a_baseline_within_the_bound_reproduces(pool: KernelPool, tmp_path: Path) -> None:
    """Floating-point noise between two evaluations is not a difference."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME, attitude=_turned(_WELL_INSIDE_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is None


def test_a_candidate_not_covering_the_midtime_is_not_assigned(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel whose window ends before the exposure supplied nothing."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME, start_et=ET0 + 100.0)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_an_image_naming_no_kernels_is_not_assigned(pool: KernelPool, tmp_path: Path) -> None:
    """A recorded kernel list with nothing in it resolves to no candidate."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=())
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_the_tie_break_prefers_the_reconstructed_kernel(pool: KernelPool, tmp_path: Path) -> None:
    """Among kernels that all reproduce, the class decides first."""
    reconstructed = tmp_path / 'CK-reconstructed'
    gapfill = tmp_path / 'CK-gapfill'
    predicted = tmp_path / 'CK-predicted'
    # The reconstructed name sorts first, so only the class rank can pick it.
    _write_candidate(reconstructed, '03236_04002ra.bc')
    _write_candidate(gapfill, 'zz03001_04001pa_gapfill_v01.bc')
    _write_candidate(predicted, 'zz04009_04051px.bc')
    index = build_ck_index([reconstructed, gapfill, predicted])
    entry = _entry(
        cmatrix_original=_cassini_recorded(pool),
        kernels=('03236_04002ra.bc', 'zz03001_04001pa_gapfill_v01.bc', 'zz04009_04051px.bc'),
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.kernel_class is KernelClass.RECONSTRUCTED


def test_the_tie_break_prefers_the_greatest_basename_within_a_class(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Two reproducing kernels of one class are separated by their names."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, '03236_04002ra.bc')
    _write_candidate(root, '04002_04009ra.bc')
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_cassini_recorded(pool),
        kernels=('03236_04002ra.bc', '04002_04009ra.bc'),
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == '04002_04009ra.bc'


def test_a_simultaneous_pair_yields_one_segment_and_one_loser(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Both frames reproduce the same baseline; only the narrow angle one is written."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    recorded = _cassini_recorded(pool)
    entries = [
        _entry(
            cmatrix_original=recorded,
            kernels=(_TRUE_NAME,),
            image_name='N1484573295_1.IMG',
            camera='NAC',
            shutter_mode='BOTSIM',
        ),
        _entry(
            cmatrix_original=recorded,
            kernels=(_TRUE_NAME,),
            image_name='W1484573295_1.IMG',
            camera='WAC',
            shutter_mode='BOTSIM',
        ),
    ]
    assignments = assign_images(entries, index)
    assert assignments[0].baseline is not None
    assert assignments[1].baseline is None
    assert assignments[1].omission_reason is OmissionReason.BOTSIM_LOSER


def test_a_fitted_rotation_is_omitted_as_unsupported(pool: KernelPool, tmp_path: Path) -> None:
    """An image with a fitted rotation is reported without being looked up."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_cassini_recorded(pool),
        kernels=(_TRUE_NAME,),
        cmatrix=None,
        rotation_deg=0.25,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.ROTATION_UNSUPPORTED


def test_an_ineligible_image_keeps_its_own_reason(pool: KernelPool, tmp_path: Path) -> None:
    """An image that did not navigate is reported as not eligible."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_cassini_recorded(pool),
        kernels=(_TRUE_NAME,),
        cmatrix=None,
        status='failed',
    )
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NOT_ELIGIBLE


def test_assignments_come_back_in_the_order_the_images_were_given(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Every image considered appears exactly once, in order."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    recorded = _cassini_recorded(pool)
    names = ('N1484573295_1.IMG', 'N1484573296_1.IMG', 'N1484573297_1.IMG')
    entries = [
        _entry(cmatrix_original=recorded, kernels=(_TRUE_NAME,), image_name=name) for name in names
    ]
    assignments = assign_images(entries, index)
    assert tuple(assignment.image_name for assignment in assignments) == names


def test_assign_refuses_two_images_with_the_same_name(pool: KernelPool, tmp_path: Path) -> None:
    """Two records of one name cannot be told apart in the report."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    with pytest.raises(ValueError, match='two images have the same name'):
        assign_images([entry, entry], index)


def test_assign_refuses_a_pool_that_already_holds_a_kernel(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A stray C-kernel would answer the lookups alongside the candidate."""
    root = tmp_path / 'CK-reconstructed'
    path = _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    pool.furnish(path)
    with pytest.raises(ValueError, match='already furnished'):
        assign_images([entry], index)


def test_the_candidate_pool_is_left_as_it_was_found(pool: KernelPool, tmp_path: Path) -> None:
    """Every candidate furnished for a test is unloaded again."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assign_images([entry], index)
    assert int(cspyce.ktotal('CK')) == 0


def _voyager_recorded(snapped_et: float) -> NDArrayFloatType:
    """Return the attitude a Voyager image froze, in camera terms.

    Parameters:
        snapped_et: The epoch the pointing lookup actually answered at.

    Returns:
        The 3x3 J2000-to-camera rotation.
    """
    recorded: NDArrayFloatType = _camera_from_object(
        VOYAGER_CK_FRAME_ID, VOYAGER_CAMERA_FRAME
    ) @ baseline_attitude(snapped_et)
    return recorded


def _snapped_et(midtime_et: float) -> float:
    """Return the epoch the whole-tick pointing lookup lands on.

    Parameters:
        midtime_et: The exposure midtime.

    Returns:
        TDB seconds past J2000 of the encoded whole tick.
    """
    return float(cspyce.sct2e(_VOYAGER_SCLK_ID, float(cspyce.sce2t(_VOYAGER_SCLK_ID, midtime_et))))


# A midtime that does not land on a whole clock tick, so the snapped lookup
# answers at a measurably different epoch from the midtime itself.
_VOYAGER_START_ET = ET0 + 1.003
_VOYAGER_MIDTIME_ET = _VOYAGER_START_ET + _EXPOSURE_S / 2.0


def test_a_frozen_baseline_reproduces_through_the_snapped_lookup(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A Voyager baseline is reproduced at the whole tick, not at the midtime."""
    root = tmp_path / 'CK'
    _write_candidate(
        root,
        'vg1_sat_version1_type1_iss_sedr.bc',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=_VOYAGER_SCLK_ID,
    )
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_voyager_recorded(_snapped_et(_VOYAGER_MIDTIME_ET)),
        kernels=('vg1_sat_version1_type1_iss_sedr.bc',),
        image_name='C1205021_CALIB.IMG',
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
    _write_candidate(
        root,
        'vg1_sat_version1_type1_iss_sedr.bc',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=_VOYAGER_SCLK_ID,
    )
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_voyager_recorded(_VOYAGER_MIDTIME_ET),
        kernels=('vg1_sat_version1_type1_iss_sedr.bc',),
        image_name='C1205021_CALIB.IMG',
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
    _write_candidate(
        root,
        'vg1_sat_version1_type1_iss_sedr.bc',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=_VOYAGER_SCLK_ID,
        attitude=_turned(_WRONG_RAD),
    )
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_voyager_recorded(_snapped_et(_VOYAGER_MIDTIME_ET)),
        kernels=('vg1_sat_version1_type1_iss_sedr.bc',),
        image_name='C1205021_CALIB.IMG',
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
    _write_candidate(
        root,
        'vg1_sat_version1_type1_iss_sedr.bc',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        sclk_id=_VOYAGER_SCLK_ID,
        start_et=ET0 + 10.0,
    )
    index = build_ck_index([root])
    entry = _entry(
        cmatrix_original=_voyager_recorded(ET0 + 10.0),
        kernels=('vg1_sat_version1_type1_iss_sedr.bc',),
        image_name='C1205021_CALIB.IMG',
        ck_frame_id=VOYAGER_CK_FRAME_ID,
        camera_frame=VOYAGER_CAMERA_FRAME,
        start_et=_VOYAGER_START_ET,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None


def test_reproduces_baseline_is_false_when_nothing_is_furnished(pool: KernelPool) -> None:
    """A lookup the kernels cannot answer is a candidate that did not supply it."""
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assert entry.pointing is not None
    assert reproduces_baseline(entry.pointing) is False


@pytest.mark.parametrize(
    ('basename', 'expected'),
    [
        ('03236_04002ra.bc', '03236_04002ra_nav.bc'),
        ('03001_04001pa_gapfill_v01.bc', '03001_04001pa_gapfill_v01_nav.bc'),
        ('vg2_sat_version1_type1_iss_sedr.bc', 'vg2_sat_version1_type1_iss_sedr_nav.bc'),
        ('V1SAT_VERSION2_TYPE3_UVS_SEDR.ck', 'V1SAT_VERSION2_TYPE3_UVS_SEDR_nav.ck'),
    ],
    ids=['reconstructed', 'gapfill', 'voyager', 'other-extension'],
)
def test_output_basename_marks_a_real_name(basename: str, expected: str) -> None:
    """The corrected file carries the original's name with the marker before the extension.

    Parameters:
        basename: A real original basename from the holdings.
        expected: The corrected name it maps to.
    """
    assert output_basename(basename) == expected


@pytest.mark.parametrize(
    'basename',
    [
        '',
        '.',
        '..',
        'CK-reconstructed/03236_04002ra.bc',
        '/holdings/03236_04002ra.bc',
    ],
    ids=['empty', 'dot', 'parent', 'relative-path', 'absolute-path'],
)
def test_output_basename_refuses_a_name_that_is_not_a_basename(basename: str) -> None:
    """A path or a directory is not a basename; where to write it is the caller's business.

    Parameters:
        basename: A string no original C-kernel is named.
    """
    with pytest.raises(ValueError, match='not a bare C-kernel basename'):
        output_basename(basename)


@pytest.mark.parametrize(
    'basename',
    ['03236_04002ra', '03236_04002ra.lbl', '.bc'],
    ids=['no-extension', 'label', 'extension-only'],
)
def test_output_basename_refuses_a_name_that_is_not_a_kernel(basename: str) -> None:
    """A file that is not a C-kernel has no corrected counterpart.

    Parameters:
        basename: A basename carrying no C-kernel extension.
    """
    with pytest.raises(ValueError, match='does not end in a C-kernel extension'):
        output_basename(basename)


def test_output_basename_refuses_correcting_a_correction() -> None:
    """Re-running over an output directory would measure against a corrected baseline."""
    with pytest.raises(ValueError, match='already a corrected kernel'):
        output_basename('03236_04002ra_nav.bc')


def test_an_assignment_names_the_file_its_segment_goes_into(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An assigned image reports the corrected file that will carry it."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].output_name == '03236_04002ra_nav.bc'


def test_an_omitted_image_names_no_file(pool: KernelPool, tmp_path: Path) -> None:
    """An image with no baseline goes into no corrected file at all."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _DECOY_NAME, attitude=_turned(_WRONG_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_DECOY_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].output_name is None


def test_group_for_output_carries_every_image_of_one_baseline(
    pool: KernelPool, tmp_path: Path
) -> None:
    """One corrected file mirrors one original and holds all of its images."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _TRUE_NAME)
    index = build_ck_index([root])
    recorded = _cassini_recorded(pool)
    entries = [
        _entry(cmatrix_original=recorded, kernels=(_TRUE_NAME,), image_name=name)
        for name in ('N1484573295_1.IMG', 'N1484573296_1.IMG')
    ]
    groups = group_for_output(assign_images(entries, index))
    assert len(groups) == 1
    assert groups[0].name == '03236_04002ra_nav.bc'
    assert len(groups[0].assignments) == 2


def test_group_for_output_skips_omitted_images(pool: KernelPool, tmp_path: Path) -> None:
    """An original no image reproduces yields no file at all."""
    root = tmp_path / 'CK-reconstructed'
    _write_candidate(root, _DECOY_NAME, attitude=_turned(_WRONG_RAD))
    index = build_ck_index([root])
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_DECOY_NAME,))
    assert group_for_output(assign_images([entry], index)) == ()


def test_group_for_output_refuses_two_originals_with_one_corrected_name(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Two directories holding one basename would write one file twice."""
    first = tmp_path / 'CK-reconstructed'
    second = tmp_path / 'CK-gapfill'
    _write_candidate(first, _TRUE_NAME)
    _write_candidate(second, _TRUE_NAME, attitude=_turned(_WRONG_RAD))
    index = build_ck_index([first, second])
    entries = [
        _entry(
            cmatrix_original=_cassini_recorded(pool),
            kernels=(_TRUE_NAME,),
            image_name='N1484573295_1.IMG',
        ),
        _entry(
            cmatrix_original=_camera_from_object(CASSINI_CK_FRAME_ID, CASSINI_CAMERA_FRAME)
            @ _turned(_WRONG_RAD)(_MIDTIME_ET),
            kernels=(_TRUE_NAME,),
            image_name='N1484573296_1.IMG',
        ),
    ]
    assignments = assign_images(entries, index)
    with pytest.raises(ValueError, match='would both be corrected to'):
        group_for_output(assignments)


def test_an_assignment_needs_a_baseline_or_a_reason(pool: KernelPool) -> None:
    """An assignment that is neither written nor omitted has no disposition."""
    entry = _entry(cmatrix_original=_cassini_recorded(pool), kernels=(_TRUE_NAME,))
    with pytest.raises(ValueError, match='not both and not neither'):
        Assignment(entry=entry, baseline=None, omission_reason=None)


def test_an_assignment_cannot_write_an_image_with_no_pointing(pool: KernelPool) -> None:
    """A baseline assigned to an ineligible image would have nothing to write."""
    entry = _entry(
        cmatrix_original=_cassini_recorded(pool),
        kernels=(_TRUE_NAME,),
        cmatrix=None,
        status='failed',
    )
    ck_file = CkFile(
        path=Path('/holdings/CK-reconstructed') / _TRUE_NAME,
        kernel_class=KernelClass.RECONSTRUCTED,
        coverage=(),
    )
    with pytest.raises(ValueError, match='carries no pointing to write'):
        Assignment(entry=entry, baseline=ck_file, omission_reason=None)


def test_attitudes_agreeing_within_the_bound_reproduce() -> None:
    """A rotation smaller than the bound is the same attitude."""
    recorded = baseline_attitude(ET0)
    assert attitudes_reproduce(recorded, axis_rotation(_DECOY_AXIS, 9e-10) @ recorded) is True


def test_attitudes_differing_beyond_the_bound_do_not_reproduce() -> None:
    """A rotation larger than the bound is a different attitude."""
    recorded = baseline_attitude(ET0)
    assert attitudes_reproduce(recorded, axis_rotation(_DECOY_AXIS, 1.1e-9) @ recorded) is False


@pytest.mark.parametrize(
    'value', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf']
)
def test_rotation_angle_refuses_a_non_finite_attitude(value: float) -> None:
    """A NaN answers every comparison with False and would read as agreement.

    Parameters:
        value: The non-finite value planted in the matrix.
    """
    broken = np.eye(3).copy()
    broken[0, 0] = value
    with pytest.raises(ValueError, match='holds a non-finite value'):
        rotation_angle_rad(np.eye(3), broken)


@pytest.mark.parametrize(
    'shape', [(2, 2), (9,), (3, 3, 1)], ids=['two-by-two', 'flat-nine', 'nested']
)
def test_rotation_angle_refuses_a_matrix_of_the_wrong_shape(shape: tuple[int, ...]) -> None:
    """Anything but a 3x3 is refused by name rather than by a numpy message.

    Parameters:
        shape: A shape an attitude never has.
    """
    with pytest.raises(ValueError, match='is not a 3x3 matrix'):
        rotation_angle_rad(np.ones(shape), np.eye(3))
