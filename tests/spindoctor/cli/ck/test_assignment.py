"""Hermetic tests for ``spindoctor.cli.ck.assignment``.

Each test writes its own candidate C-kernels -- one holding the attitude an
image really navigated against and one or more holding something else -- and
lets the assignment step work out which is which by furnishing them and asking.
The decoys are named so that they would win the tie-break if reproduction were
skipped, so a test cannot pass by chance of ordering.

The Voyager frozen-attitude lookups have a file of their own,
``test_assignment_voyager``; what is here is the frame-chain reproduction test,
the tie-break, the simultaneous-exposure rule, the refusals, and the naming of
the corrected outputs.
"""

from pathlib import Path

import cspyce
import numpy as np
import pytest
from filecache import FCPath
from tests.spindoctor.cli.ck.assignment_helpers import (
    DECOY_AXIS,
    DECOY_NAME,
    IMAGE_NAME,
    MIDTIME_ET,
    TRUE_NAME,
    WRONG_RAD,
    camera_from_object,
    cassini_recorded,
    image_entry,
    turned,
    write_candidate,
)
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    KernelPool,
    axis_rotation,
    baseline_attitude,
)

from spindoctor.cli.ck.assignment import (
    Assignment,
    assign_images,
    attitudes_reproduce,
    baseline_attitudes,
    group_for_output,
    output_basename,
    reproduces_baseline,
    rotation_angle_rad,
)
from spindoctor.cli.ck.images import OmissionReason
from spindoctor.cli.ck.index import (
    CkFile,
    CkIndex,
    CoverageInterval,
    KernelClass,
    build_ck_index,
)

# The two decoy angles that bracket the reproduction bound of 1e-9 radians.
_JUST_OUTSIDE_RAD = 1e-8
_WELL_INSIDE_RAD = 1e-10

# An object whose clock id SPICE computes as 0, which no SCLK kernel defines,
# so a real kernel naming it beside the spacecraft leaves its coverage
# unreadable.  No image corrects it.
_CLOCKLESS_OBJECT_ID = -1


def test_the_recorded_baseline_is_what_pxform_gives(pool: KernelPool, tmp_path: Path) -> None:
    """The tests' idea of the recorded attitude is SPICE's own.

    Everything below plants ``cmatrix_original`` analytically, so this pins
    that the planted value is what the frame chain actually evaluates to.
    """
    path = write_candidate(tmp_path / 'CK-reconstructed', TRUE_NAME)
    pool.furnish(path)
    evaluated = np.asarray(
        cspyce.pxform('J2000', CASSINI_CAMERA_FRAME, MIDTIME_ET), dtype=np.float64
    )
    assert rotation_angle_rad(cassini_recorded(), evaluated) < 1e-12


def test_assignment_keeps_the_reproducing_candidate(pool: KernelPool, tmp_path: Path) -> None:
    """The candidate that answers the recorded attitude is the one assigned."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    write_candidate(root, DECOY_NAME, attitude=turned(WRONG_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME, DECOY_NAME))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None
    assert assignments[0].baseline.basename == TRUE_NAME


def test_assignment_reports_no_reproducing_baseline(pool: KernelPool, tmp_path: Path) -> None:
    """An image whose baseline no candidate reproduces gets no segment."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, DECOY_NAME, attitude=turned(WRONG_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(DECOY_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is None
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_a_baseline_that_drifted_past_the_bound_does_not_reproduce(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel differing by ten nanoradians is a different kernel."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME, attitude=turned(_JUST_OUTSIDE_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_a_baseline_within_the_bound_reproduces(pool: KernelPool, tmp_path: Path) -> None:
    """Floating-point noise between two evaluations is not a difference."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME, attitude=turned(_WELL_INSIDE_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is None


def test_a_candidate_not_covering_the_midtime_is_not_assigned(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel whose window ends before the exposure supplied nothing."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME, start_et=ET0 + 100.0)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_an_image_naming_no_indexed_kernel_is_not_assigned(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A kernel list naming nothing the index holds resolves to no candidate."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=('naif0012.tls',))
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE


def test_the_tie_break_prefers_the_reconstructed_kernel(pool: KernelPool, tmp_path: Path) -> None:
    """Among kernels that all reproduce, the class decides first."""
    reconstructed = tmp_path / 'CK-reconstructed'
    gapfill = tmp_path / 'CK-gapfill'
    predicted = tmp_path / 'CK-predicted'
    # The reconstructed name sorts first, so only the class rank can pick it.
    write_candidate(reconstructed, '03236_04002ra.bc')
    write_candidate(gapfill, 'zz03001_04001pa_gapfill_v01.bc')
    write_candidate(predicted, 'zz04009_04051px.bc')
    index = build_ck_index([reconstructed, gapfill, predicted])
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
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
    write_candidate(root, '03236_04002ra.bc')
    write_candidate(root, '04002_04009ra.bc')
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
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
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    recorded = cassini_recorded()
    entries = [
        image_entry(
            cmatrix_original=recorded,
            kernels=(TRUE_NAME,),
            image_name='N1484573295_1.IMG',
            camera='NAC',
            shutter_mode='BOTSIM',
        ),
        image_entry(
            cmatrix_original=recorded,
            kernels=(TRUE_NAME,),
            image_name='W1484573295_1.IMG',
            camera='WAC',
            shutter_mode='BOTSIM',
        ),
    ]
    assignments = assign_images(entries, index)
    assert assignments[0].baseline is not None
    assert assignments[1].baseline is None
    assert assignments[1].omission_reason is OmissionReason.BOTSIM_LOSER


def test_a_winner_whose_baseline_does_not_reproduce_suppresses_nothing(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A narrow angle frame that writes nothing yields the bus to its partner.

    The narrow angle frame records an attitude no candidate reproduces, so it
    receives no segment; its wide angle partner's correction then conflicts
    with nothing and is written rather than omitted as a loser.
    """
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    recorded = cassini_recorded()
    drifted = axis_rotation(DECOY_AXIS, WRONG_RAD) @ recorded
    entries = [
        image_entry(
            cmatrix_original=drifted,
            kernels=(TRUE_NAME,),
            image_name='N1484573295_1.IMG',
            camera='NAC',
            shutter_mode='BOTSIM',
        ),
        image_entry(
            cmatrix_original=recorded,
            kernels=(TRUE_NAME,),
            image_name='W1484573295_1.IMG',
            camera='WAC',
            shutter_mode='BOTSIM',
        ),
    ]
    assignments = assign_images(entries, index)
    assert assignments[0].omission_reason is OmissionReason.NO_REPRODUCING_BASELINE
    assert assignments[1].baseline is not None
    assert assignments[1].omission_reason is None


def test_a_fitted_rotation_is_omitted_as_unsupported(pool: KernelPool, tmp_path: Path) -> None:
    """An image with a fitted rotation is reported without being looked up."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
        kernels=(TRUE_NAME,),
        cmatrix=None,
        rotation_deg=0.25,
    )
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.ROTATION_UNSUPPORTED


def test_an_ineligible_image_keeps_its_own_reason(pool: KernelPool, tmp_path: Path) -> None:
    """An image that did not navigate is reported as not eligible."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
        kernels=(TRUE_NAME,),
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
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    recorded = cassini_recorded()
    names = ('N1484573295_1.IMG', 'N1484573296_1.IMG', 'N1484573297_1.IMG')
    entries = [
        image_entry(cmatrix_original=recorded, kernels=(TRUE_NAME,), image_name=name)
        for name in names
    ]
    assignments = assign_images(entries, index)
    assert tuple(assignment.image_name for assignment in assignments) == names


def test_assign_refuses_two_images_with_the_same_name(pool: KernelPool, tmp_path: Path) -> None:
    """Two records of one name cannot be told apart in the report, and are named."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    with pytest.raises(ValueError, match='named more than once') as refusal:
        assign_images([entry, entry], index)
    assert IMAGE_NAME in str(refusal.value)


def test_assign_refuses_a_pool_that_already_holds_a_kernel(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A stray C-kernel would answer the lookups alongside the candidate."""
    root = tmp_path / 'CK-reconstructed'
    path = write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    pool.furnish(path)
    with pytest.raises(ValueError, match='already furnished'):
        assign_images([entry], index)


def test_assign_refuses_an_object_whose_coverage_the_index_could_not_read(
    pool: KernelPool, tmp_path: Path
) -> None:
    """A clock kernel the index needed and did not have is named, not blamed on drift.

    An object whose coverage could not be expressed in TDB offers no candidate
    at all, so every image correcting it would otherwise be reported as having
    no reproducing baseline -- which is the report reserved for holdings that
    changed since navigation ran.
    """
    root = tmp_path / 'CK-reconstructed'
    path = write_candidate(root, TRUE_NAME)
    index = CkIndex(
        files=(
            CkFile(
                path=FCPath(path),
                kernel_class=KernelClass.RECONSTRUCTED,
                coverage=(),
                unreadable_objects=(CASSINI_CK_FRAME_ID,),
            ),
        )
    )
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    with pytest.raises(ValueError, match='could not read the coverage'):
        assign_images([entry], index)


def test_assign_ignores_an_unreadable_object_no_image_needs(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An object nothing corrects does not stop a run, however unreadable it is."""
    root = tmp_path / 'CK-reconstructed'
    path = write_candidate(root, TRUE_NAME)
    covering = build_ck_index([root]).files[0].coverage
    index = CkIndex(
        files=(
            CkFile(
                path=FCPath(path),
                kernel_class=KernelClass.RECONSTRUCTED,
                coverage=covering,
                unreadable_objects=(_CLOCKLESS_OBJECT_ID,),
            ),
        )
    )
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None


def test_an_image_with_no_pointing_names_no_object_to_read(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An image that did not navigate corrects nothing, unreadable or otherwise.

    It carries no pointing at all, so it names no CK object, and an index that
    could not read some object's coverage does not turn it into a refusal.
    """
    root = tmp_path / 'CK-reconstructed'
    path = write_candidate(root, TRUE_NAME)
    index = CkIndex(
        files=(
            CkFile(
                path=FCPath(path),
                kernel_class=KernelClass.RECONSTRUCTED,
                coverage=(),
                unreadable_objects=(CASSINI_CK_FRAME_ID,),
            ),
        )
    )
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
        kernels=(TRUE_NAME,),
        cmatrix=None,
        status='failed',
    )
    assignments = assign_images([entry], index)
    assert assignments[0].omission_reason is OmissionReason.NOT_ELIGIBLE


def test_the_candidate_pool_is_left_as_it_was_found(pool: KernelPool, tmp_path: Path) -> None:
    """Every candidate furnished for a test is unloaded again."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assign_images([entry], index)
    assert int(cspyce.ktotal('CK')) == 0


def test_reproduces_baseline_is_false_when_nothing_is_furnished(pool: KernelPool) -> None:
    """A lookup the kernels cannot answer is a candidate that did not supply it."""
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
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


@pytest.mark.parametrize(
    'basename',
    ['03236_04002ra_nav.bc', '03236_04002RA_NAV.BC'],
    ids=['lower-case-marker', 'upper-case-marker'],
)
def test_output_basename_refuses_correcting_a_correction(basename: str) -> None:
    """Re-running over an output directory would measure against a corrected baseline.

    The marker is read case-blind, as the index reads it: an upper-cased copy
    of a corrected kernel is still a corrected kernel.

    Parameters:
        basename: A corrected kernel's name, in either case.
    """
    with pytest.raises(ValueError, match='already a corrected kernel'):
        output_basename(basename)


def test_an_assignment_names_the_file_its_segment_goes_into(
    pool: KernelPool, tmp_path: Path
) -> None:
    """An assigned image reports the corrected file that will carry it."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].output_name == '03236_04002ra_nav.bc'


def test_an_omitted_image_names_no_file(pool: KernelPool, tmp_path: Path) -> None:
    """An image with no baseline goes into no corrected file at all."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, DECOY_NAME, attitude=turned(WRONG_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(DECOY_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].output_name is None


def test_group_for_output_carries_every_image_of_one_baseline(
    pool: KernelPool, tmp_path: Path
) -> None:
    """One corrected file mirrors one original and holds all of its images."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    recorded = cassini_recorded()
    entries = [
        image_entry(cmatrix_original=recorded, kernels=(TRUE_NAME,), image_name=name)
        for name in ('N1484573295_1.IMG', 'N1484573296_1.IMG')
    ]
    groups = group_for_output(assign_images(entries, index))
    assert len(groups) == 1
    assert groups[0].name == '03236_04002ra_nav.bc'
    assert len(groups[0].assignments) == 2


def test_group_for_output_skips_omitted_images(pool: KernelPool, tmp_path: Path) -> None:
    """An original no image reproduces yields no file at all."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, DECOY_NAME, attitude=turned(WRONG_RAD))
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(DECOY_NAME,))
    assert group_for_output(assign_images([entry], index)) == ()


def test_group_for_output_refuses_two_originals_with_one_corrected_name(
    pool: KernelPool, tmp_path: Path
) -> None:
    """Two directories holding one basename would write one file twice."""
    first = tmp_path / 'CK-reconstructed'
    second = tmp_path / 'CK-gapfill'
    write_candidate(first, TRUE_NAME)
    write_candidate(second, TRUE_NAME, attitude=turned(WRONG_RAD))
    index = build_ck_index([first, second])
    entries = [
        image_entry(
            cmatrix_original=cassini_recorded(),
            kernels=(TRUE_NAME,),
            image_name='N1484573295_1.IMG',
        ),
        image_entry(
            cmatrix_original=camera_from_object(CASSINI_CK_FRAME_ID, CASSINI_CAMERA_FRAME)
            @ turned(WRONG_RAD)(MIDTIME_ET),
            kernels=(TRUE_NAME,),
            image_name='N1484573296_1.IMG',
        ),
    ]
    assignments = assign_images(entries, index)
    with pytest.raises(ValueError, match='would both be corrected to'):
        group_for_output(assignments)


def test_an_assignment_needs_a_baseline_or_a_reason(pool: KernelPool) -> None:
    """An assignment that is neither written nor omitted has no disposition."""
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    with pytest.raises(ValueError, match='not both and not neither'):
        Assignment(entry=entry, baseline=None, omission_reason=None)


def test_an_assignment_cannot_write_an_image_with_no_pointing(pool: KernelPool) -> None:
    """A baseline assigned to an ineligible image would have nothing to write."""
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
        kernels=(TRUE_NAME,),
        cmatrix=None,
        status='failed',
    )
    ck_file = CkFile(
        path=FCPath('/holdings/CK-reconstructed') / TRUE_NAME,
        kernel_class=KernelClass.RECONSTRUCTED,
        coverage=(),
    )
    with pytest.raises(ValueError, match='carries no pointing to write'):
        Assignment(entry=entry, baseline=ck_file, omission_reason=None)


def test_attitudes_agreeing_within_the_bound_reproduce() -> None:
    """A rotation smaller than the bound is the same attitude."""
    recorded = baseline_attitude(ET0)
    assert attitudes_reproduce(recorded, axis_rotation(DECOY_AXIS, 9e-10) @ recorded) is True


def test_attitudes_differing_beyond_the_bound_do_not_reproduce() -> None:
    """A rotation larger than the bound is a different attitude."""
    recorded = baseline_attitude(ET0)
    assert attitudes_reproduce(recorded, axis_rotation(DECOY_AXIS, 1.1e-9) @ recorded) is False


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


def test_a_chain_evaluated_object_has_one_lookup(pool: KernelPool, tmp_path: Path) -> None:
    """An object read by evaluating a frame chain is asked once, not twice."""
    path = write_candidate(tmp_path / 'CK-reconstructed', TRUE_NAME)
    pool.furnish(path)
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assert entry.pointing is not None
    assert len(baseline_attitudes(entry.pointing)) == 1


def test_baseline_attitudes_is_empty_when_nothing_answers(pool: KernelPool) -> None:
    """A candidate that answers no lookup contributes no attitude to compare."""
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assert entry.pointing is not None
    assert baseline_attitudes(entry.pointing) == ()


def test_assign_refuses_an_undefined_camera_frame(pool: KernelPool, tmp_path: Path) -> None:
    """A frame kernel that was never furnished is not a baseline that drifted.

    Reported as drift it would empty a whole run and blame the holdings, so it
    is refused once, before any candidate is tried.
    """
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    pool.unload(pool.root / 'test.tf')
    with pytest.raises(ValueError, match='is not defined by the furnished kernels'):
        assign_images([entry], index)


def test_assign_refuses_an_undefined_ck_object_frame(pool: KernelPool, tmp_path: Path) -> None:
    """The object a corrected kernel targets has to be a frame the pool knows."""
    root = tmp_path / 'CK-reconstructed'
    write_candidate(root, TRUE_NAME)
    index = build_ck_index([root])
    entry = image_entry(
        cmatrix_original=cassini_recorded(),
        kernels=(TRUE_NAME,),
        ck_frame_id=-98000,
        camera_frame=CASSINI_CAMERA_FRAME,
    )
    with pytest.raises(ValueError, match='has no frame name in the furnished kernels'):
        assign_images([entry], index)


def test_a_candidate_named_by_a_url_is_fetched_before_it_is_furnished(
    pool: KernelPool, tmp_path: Path
) -> None:
    """SPICE furnishes a kernel by local name, so a kernel named by a URL is fetched.

    A kernel tree can live somewhere SPICE cannot open, which is how the
    holdings are named in continuous integration.  Handing the name through
    unchanged furnishes nothing, and every image is then reported as having no
    reproducing baseline.

    The URL is spelled rather than mocked deliberately.  A ``file://`` path is
    local to the file cache, so this reaches no network and creates no cache
    state, and faking the fetch would assert only that it was called rather
    than that a URL-spelled name resolves to something SPICE can open.
    """
    root = tmp_path / 'CK-reconstructed'
    path = write_candidate(root, TRUE_NAME)
    index = CkIndex(
        files=(
            CkFile(
                path=FCPath(f'file://{path}'),
                kernel_class=KernelClass.RECONSTRUCTED,
                coverage=(
                    CoverageInterval(
                        ck_frame_id=CASSINI_CK_FRAME_ID, start_et=ET0, stop_et=ET0 + 4.0
                    ),
                ),
            ),
        )
    )
    entry = image_entry(cmatrix_original=cassini_recorded(), kernels=(TRUE_NAME,))
    assignments = assign_images([entry], index)
    assert assignments[0].baseline is not None
