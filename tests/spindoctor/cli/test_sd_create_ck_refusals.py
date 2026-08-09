"""What the ``sd_create_ck`` driver refuses, reports, and exits with.

The products of a clean run are pinned in ``test_sd_create_ck``; this module
holds the other half of the driver's contract: the runs it stops by name, the
omissions it reports instead of stopping, the exit statuses a batch wrapper
reads, and the guarantee that a refused run leaves the output directory as it
found it.
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath
from tests.kernel_pool import isolated_kernel_pool
from tests.spindoctor.cli.ck.ck_helpers import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
)
from tests.spindoctor.cli.sd_create_ck_helpers import (
    BASELINE_A,
    IMAGE_A_ET,
    IMAGE_B_ET,
    KERNEL_NAMES,
    image_document,
    report_rows,
    run_driver,
    run_log,
    run_stopped,
    utc_of,
    write_metadata,
)

from spindoctor.cli import sd_create_ck
from spindoctor.cli.ck import inputs
from spindoctor.cli.ck.assignment import Assignment
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import CkFile, KernelClass
from spindoctor.cli.ck.pointing import ImagePointing


@pytest.fixture(scope='module', autouse=True)
def empty_kernel_pool() -> Iterator[None]:
    """Run this module against an emptied SPICE pool, and put the pool back.

    A real C-kernel another test file left furnished in the worker defines the
    same objects the hermetic kernels here do, so it would answer the driver's
    reproduction lookups beside the candidate under test -- which the driver
    rightly refuses to run with.  The isolation is the same loan the ck
    package's conftest makes.

    Yields:
        Nothing; the module's tests run against an empty pool.
    """
    with isolated_kernel_pool():
        yield


# ---------------------------------------------------------------------------
# Where the corrected kernels may be written
# ---------------------------------------------------------------------------


def test_a_remote_output_directory_is_refused() -> None:
    """SPICE creates a kernel by name on the local filesystem and nowhere else."""
    with pytest.raises(ValueError, match='not a local directory'):
        sd_create_ck.local_output_path(FCPath('gs://bucket/out/orig_nav.bc'))


# ---------------------------------------------------------------------------
# What the run says when something goes wrong
# ---------------------------------------------------------------------------


def test_an_unreadable_metadata_file_is_named_in_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """It names no image, so the run log is the only place it can be reported."""
    (run_tree['results'] / 'vol' / f'broken{inputs.METADATA_SUFFIX}').write_text('{not json')
    run_driver(run_tree, monkeypatch, expected_exit=1)
    log = run_log(run_tree)
    assert 'broken_metadata.json' in log
    assert 'Could not read 1 metadata file(s)' in log


def test_an_unreadable_metadata_file_makes_the_run_exit_non_zero(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A batch wrapper cannot otherwise tell a clean run from a skipped one."""
    (run_tree['results'] / 'vol' / f'broken{inputs.METADATA_SUFFIX}').write_text('[1, 2]')
    run_driver(run_tree, monkeypatch, expected_exit=1)


def test_the_run_still_writes_what_it_could_read(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """One unreadable file is reported, not a reason to abandon the others."""
    (run_tree['results'] / 'vol' / f'broken{inputs.METADATA_SUFFIX}').write_text('{not json')
    run_driver(run_tree, monkeypatch, expected_exit=1)
    assert (run_tree['output'] / 'orig_a_nav.bc').exists()


def test_an_image_that_cannot_be_placed_in_time_is_reported(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A time range was asked for and this image cannot be shown to satisfy it."""
    metadata = image_document(
        image_name='E_CALIB',
        midtime=IMAGE_A_ET,
        cmatrix_original=np.eye(3),
        cmatrix=np.eye(3),
    )
    del metadata['navigation_result']['times']['midtime_et']
    write_metadata(run_tree['results'], 'vol/E_CALIB', metadata)
    run_driver(run_tree, monkeypatch, '--start-time', utc_of(run_tree, IMAGE_A_ET - 100.0))
    assert 'Ignored 1 image(s) that recorded no exposure midtime' in run_log(run_tree)


def test_a_run_selecting_nothing_says_so_and_writes_nothing(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A time range that excludes every image is not an error, and not silent."""
    run_driver(run_tree, monkeypatch, '--start-time', utc_of(run_tree, IMAGE_B_ET + 10000.0))
    assert 'No images selected; nothing to write' in run_log(run_tree)
    assert list(run_tree['output'].glob('*.bc')) == []


def test_a_run_whose_every_metadata_file_is_unreadable_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An empty selection is clean only when every file could be read.

    A run whose selection is empty because its metadata could not be read must
    not report itself clean: a batch wrapper would then read a skipped input as
    a quiet span.
    """
    tree = {
        'kernels': tmp_path / 'kernels',
        'results': tmp_path / 'results',
        'output': tmp_path / 'output',
    }
    tree['kernels'].mkdir()
    tree['results'].mkdir()
    (tree['results'] / f'broken{inputs.METADATA_SUFFIX}').write_text('{not json')
    run_driver(tree, monkeypatch, expected_exit=1)


@pytest.mark.parametrize('flag', ['--start-time', '--stop-time'], ids=['start', 'stop'])
def test_a_malformed_time_bound_is_refused_naming_its_flag(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None, flag: str
) -> None:
    """A mistyped bound reads as the argument refusal it is, not as a defect inside.

    Parameters:
        flag: The selection flag carrying the unreadable value.
    """
    refusal = run_stopped(
        run_tree, monkeypatch, f"{flag} 'garbage' is not a UTC time SPICE can read", flag, 'garbage'
    )
    assert 'garbage' in refusal


def test_a_run_correcting_nothing_writes_no_meta_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A meta-kernel naming no correction would furnish only the originals."""
    for name in ('A_CALIB', 'B_CALIB', 'D_CALIB'):
        (run_tree['results'] / 'vol' / f'{name}_metadata.json').unlink()
    run_driver(run_tree, monkeypatch)
    assert 'no meta-kernel written' in run_log(run_tree)
    assert not (run_tree['output'] / 'coiss_nav.tm').exists()


def test_a_run_correcting_nothing_still_writes_the_report(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The report is the answer, and 'nothing was corrected' is an answer."""
    for name in ('A_CALIB', 'B_CALIB', 'D_CALIB'):
        (run_tree['results'] / 'vol' / f'{name}_metadata.json').unlink()
    run_driver(run_tree, monkeypatch)
    assert report_rows(run_tree)['C_CALIB']['omission_reason'] == 'not_eligible'


def _write_unreadable_image(run_tree: dict[str, Path]) -> None:
    """Add an image document the entry reader must refuse to a run tree.

    Parameters:
        run_tree: The directories the fixture built.
    """
    metadata = image_document(
        image_name='F_CALIB',
        midtime=IMAGE_A_ET,
        cmatrix_original=np.eye(3),
        cmatrix=np.eye(3),
    )
    metadata['navigation_result']['provenance']['spice_kernels'] = []
    write_metadata(run_tree['results'], 'vol/F_CALIB', metadata)


def test_a_document_that_is_not_a_navigated_image_stops_the_run(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The closed reason set has no entry for a record the reader cannot read."""
    _write_unreadable_image(run_tree)
    run_stopped(run_tree, monkeypatch, 'spice_kernels')


def test_a_document_that_is_not_a_navigated_image_is_named_in_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Named before it propagates, so the operator knows which file to look at."""
    _write_unreadable_image(run_tree)
    run_stopped(run_tree, monkeypatch, 'spice_kernels')
    assert 'F_CALIB_metadata.json: cannot be read as a navigated image' in run_log(run_tree)


def test_an_exposure_its_baseline_does_not_cover_is_reported(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The run finishes and the image is one row of the report, like any omission.

    Not ``no_reproducing_baseline``: this baseline did reproduce, at the
    midtime, which is what paired the image with it, and that reason is the
    detector for holdings that changed since navigation ran.
    """
    run_driver(straddling_tree, monkeypatch)
    assert report_rows(straddling_tree)['G_CALIB']['omission_reason'] == 'baseline_coverage_gap'


def test_an_exposure_its_baseline_does_not_cover_is_not_reported_as_drift(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The drift detector stays clean, which is what makes it worth watching."""
    run_driver(straddling_tree, monkeypatch)
    assert 'Images omitted, no_reproducing_baseline: 0' in run_log(straddling_tree)


def test_an_exposure_its_baseline_does_not_cover_names_no_source_file(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """It received no segment, so no file can be said to carry one."""
    run_driver(straddling_tree, monkeypatch)
    assert report_rows(straddling_tree)['G_CALIB']['source_bc'] == ''


def test_an_exposure_its_baseline_does_not_cover_appears_once(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Every image considered appears exactly once, this one included."""
    run_driver(straddling_tree, monkeypatch)
    with (straddling_tree['output'] / 'coiss_ck_report.csv').open() as stream:
        names = [row['image_name'] for row in csv.DictReader(stream)]
    assert names == ['G_CALIB']


def test_an_exposure_its_baseline_does_not_cover_writes_no_kernel(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Its baseline is left with nothing to correct, and an empty file claims a correction."""
    run_driver(straddling_tree, monkeypatch)
    assert list(straddling_tree['output'].glob('*.bc')) == []


def test_an_exposure_its_baseline_does_not_cover_survives_strict_scope(
    straddling_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    pool_restored: None,
    strict_log_scope: None,
) -> None:
    """The omission is logged where a scope is open for it, not while building.

    Logging it through the image logger during the build would raise
    ``LogScopeError`` under this documented setting, since no image scope is
    open there; the reporting pass opens one per image and reports it from
    inside.
    """
    run_driver(straddling_tree, monkeypatch)
    logs = list((straddling_tree['output'] / 'logs').rglob('*G_CALIB*'))
    assert len(logs) == 1
    assert 'baseline_coverage_gap' in logs[0].read_text()


def test_an_exposure_its_baseline_does_not_cover_is_named_in_the_run_log(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With the record epoch, which the reason itself does not carry."""
    run_driver(straddling_tree, monkeypatch)
    log = run_log(straddling_tree)
    assert 'G_CALIB: no corrected segment written (baseline_coverage_gap)' in log
    assert 'G_CALIB: the furnished baseline supplies no pointing for CK object -82000' in log


def test_an_exposure_its_baseline_does_not_cover_is_counted(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The end-of-run counts carry the new reason like the other four."""
    run_driver(straddling_tree, monkeypatch)
    assert 'Images omitted, baseline_coverage_gap: 1' in run_log(straddling_tree)


def test_a_baseline_left_with_nothing_to_correct_says_so(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An operator sees why a baseline they expected to be mirrored was not."""
    run_driver(straddling_tree, monkeypatch)
    assert f'No image is left to correct {BASELINE_A}' in run_log(straddling_tree)


def _run_refused(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the driver over a tree whose second output file cannot be built.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
    """
    run_stopped(tree, monkeypatch, 'angular velocity at only some of them')


def test_a_refusal_leaves_no_corrected_kernel_behind(
    refused_second_file_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The first file was buildable; a run that wrote as it built would have written it."""
    _run_refused(refused_second_file_tree, monkeypatch)
    assert list(refused_second_file_tree['output'].glob('*.bc')) == []


def test_a_refusal_leaves_no_meta_kernel_behind(
    refused_second_file_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A meta-kernel is written after the kernels, and there are none to name."""
    _run_refused(refused_second_file_tree, monkeypatch)
    assert not (refused_second_file_tree['output'] / 'coiss_nav.tm').exists()


def test_a_refusal_leaves_no_report_behind(
    refused_second_file_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Nothing is written, so nothing claims to say what was: the run is repeatable."""
    _run_refused(refused_second_file_tree, monkeypatch)
    assert not (refused_second_file_tree['output'] / 'coiss_ck_report.csv').exists()


def test_a_refusal_names_the_image_it_could_not_build(
    refused_second_file_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Named in the run log, which is the only record such a run leaves."""
    _run_refused(refused_second_file_tree, monkeypatch)
    assert 'B_CALIB: could not build the corrected segment' in run_log(refused_second_file_tree)


# ---------------------------------------------------------------------------
# An output path the run cannot write
#
# The run writes orig_a_nav.bc and then orig_b_nav.bc, so blocking the second
# is what tells a run that judges its destinations first from one that judges
# each file as it reaches it.  The second would refuse either way; only the
# first says whether anything was left behind when it did.
# ---------------------------------------------------------------------------


def _block_second_output(tree: dict[str, Path], *, with_link_to: Path | None = None) -> Path:
    """Put something at the run's second output path, and return it.

    Parameters:
        tree: The directories the fixture built.
        with_link_to: A target to make the blocker a symbolic link to, absent
            itself.  Without it the blocker is an ordinary file.

    Returns:
        The blocked path.
    """
    path = tree['output'] / 'orig_b_nav.bc'
    path.parent.mkdir(parents=True, exist_ok=True)
    if with_link_to is None:
        path.write_bytes(b'not a kernel')
    else:
        path.symlink_to(with_link_to)
    return path


def test_an_occupied_second_output_path_stops_the_run(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And names it, since an operator has to know which file to move."""
    _block_second_output(run_tree)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')


def test_an_occupied_second_output_path_leaves_the_first_unwritten(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The whole point: a run that judged each file as it reached it wrote this one."""
    _block_second_output(run_tree)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'orig_a_nav.bc').exists()


def test_an_occupied_second_output_path_leaves_no_meta_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A meta-kernel would name a corrected set that was never written."""
    _block_second_output(run_tree)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'coiss_nav.tm').exists()


def test_an_occupied_second_output_path_leaves_no_report(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Nothing was written, so nothing claims to say what was: the run is repeatable."""
    _block_second_output(run_tree)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'coiss_ck_report.csv').exists()


def test_an_occupied_second_output_path_is_left_alone(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The file the operator has to choose between is not the one that changed."""
    blocked = _block_second_output(run_tree)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert blocked.read_bytes() == b'not a kernel'


def test_a_run_names_every_output_path_it_cannot_write(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The set is cleared in one pass, not one rerun per occupied file.

    This is what the whole set being judged buys over judging each file just
    before it is written: both of those refuse before anything is on disk, but
    only the set-level one reaches the second path to report it.
    """
    run_tree['output'].mkdir(parents=True, exist_ok=True)
    (run_tree['output'] / 'orig_a_nav.bc').write_bytes(b'not a kernel')
    _block_second_output(run_tree)
    refusal = run_stopped(run_tree, monkeypatch, r'orig_a_nav\.bc already exists')
    assert 'orig_b_nav.bc already exists' in refusal


def test_a_dangling_link_at_the_second_output_path_stops_the_run(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """``Path.exists`` follows the link and reports the absent target as absent."""
    _block_second_output(run_tree, with_link_to=run_tree['output'] / 'elsewhere.bc')
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')


def test_a_dangling_link_at_the_second_output_path_creates_no_target(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the harm: the write would create a file the run never named."""
    target = run_tree['output'] / 'elsewhere.bc'
    _block_second_output(run_tree, with_link_to=target)
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')
    assert not target.exists()


def test_a_dangling_link_at_the_second_output_path_leaves_the_first_unwritten(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The set is judged before the first file, whatever occupies a later path."""
    _block_second_output(run_tree, with_link_to=run_tree['output'] / 'elsewhere.bc')
    run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')
    assert not (run_tree['output'] / 'orig_a_nav.bc').exists()


# ---------------------------------------------------------------------------
# Carrying a build-time omission into the report
# ---------------------------------------------------------------------------


def _entry(image_name: str, *, corrected: bool) -> ImageEntry:
    """Build one image entry, with or without pointing to write.

    Parameters:
        image_name: Basename recorded for the image.
        corrected: Whether it carries a pointing solution.

    Returns:
        The entry.
    """
    pointing = None
    if corrected:
        pointing = ImagePointing(
            image_name=image_name,
            cmatrix=np.eye(3),
            cmatrix_original=np.eye(3),
            camera_frame=CASSINI_CAMERA_FRAME,
            ck_frame_id=CASSINI_CK_FRAME_ID,
            start_et=IMAGE_A_ET - 1.0,
            stop_et=IMAGE_A_ET + 1.0,
            midtime_et=IMAGE_A_ET,
            exposure_s=2.0,
        )
    return ImageEntry(
        image_name=image_name,
        status='success' if corrected else 'failed',
        camera='NAC',
        shutter_mode='NACONLY',
        rotation_fitted=False,
        kernel_basenames=KERNEL_NAMES if corrected else (),
        pointing=pointing,
        ineligibility_reason=None if corrected else OmissionReason.NOT_ELIGIBLE,
    )


def _assignment(image_name: str, *, corrected: bool) -> Assignment:
    """Build one assignment, either carrying a baseline or a reason it has none.

    Parameters:
        image_name: Basename recorded for the image.
        corrected: Whether the image was paired with a baseline.

    Returns:
        The assignment.
    """
    if not corrected:
        return Assignment(
            entry=_entry(image_name, corrected=False),
            baseline=None,
            omission_reason=OmissionReason.NOT_ELIGIBLE,
        )
    baseline = CkFile(
        path=FCPath(f'/kernels/{BASELINE_A}'),
        kernel_class=KernelClass.RECONSTRUCTED,
        coverage=(),
    )
    return Assignment(
        entry=_entry(image_name, corrected=True), baseline=baseline, omission_reason=None
    )


def test_a_build_omission_replaces_that_image_s_baseline() -> None:
    """The report is written from the assignments, so the reason has to reach them."""
    assignments = (_assignment('A_CALIB', corrected=True), _assignment('B_CALIB', corrected=True))
    revised = sd_create_ck.apply_build_omissions(
        assignments, {'A_CALIB': OmissionReason.BASELINE_COVERAGE_GAP}
    )
    assert revised[0].omission_reason is OmissionReason.BASELINE_COVERAGE_GAP
    assert revised[0].baseline is None


def test_a_build_omission_leaves_the_other_images_alone() -> None:
    """And leaves them in the order the report expects them in."""
    assignments = (_assignment('A_CALIB', corrected=True), _assignment('B_CALIB', corrected=True))
    revised = sd_create_ck.apply_build_omissions(
        assignments, {'A_CALIB': OmissionReason.BASELINE_COVERAGE_GAP}
    )
    assert revised[1] is assignments[1]


def test_no_build_omissions_changes_nothing() -> None:
    """A run where every assigned image built is the ordinary one."""
    assignments = (_assignment('A_CALIB', corrected=True),)
    assert sd_create_ck.apply_build_omissions(assignments, {}) == assignments


def test_a_build_omission_naming_an_unknown_image_is_refused() -> None:
    """Its reason would reach no row, and the run would report nothing amiss."""
    assignments = (_assignment('A_CALIB', corrected=True),)
    with pytest.raises(ValueError, match='would reach no row of the report'):
        sd_create_ck.apply_build_omissions(
            assignments, {'Z_CALIB': OmissionReason.BASELINE_COVERAGE_GAP}
        )


def test_a_build_omission_naming_an_already_omitted_image_is_refused() -> None:
    """No segment was built for it, so no build could have omitted it.

    Overwriting the reason it already carries would replace the one the report
    is meant to show with one the run did not measure.
    """
    assignments = (_assignment('C_CALIB', corrected=False),)
    with pytest.raises(ValueError, match='C_CALIB'):
        sd_create_ck.apply_build_omissions(
            assignments, {'C_CALIB': OmissionReason.BASELINE_COVERAGE_GAP}
        )


def test_two_documents_naming_one_image_are_refused(tmp_path: Path) -> None:
    """One set of facts would silently stand in for the other's.

    The documents are the ones an image that failed to load leaves, which
    record a name and a status and no epoch, so no leapseconds kernel is
    needed to read them.
    """
    metadata: dict[str, Any] = {'status': 'failed', 'observation': {'image_name': 'A_CALIB'}}
    documents = [
        inputs.Document(
            path=FCPath(str(tmp_path / f'{stub}_metadata.json')), stub=stub, metadata=metadata
        )
        for stub in ('vol/first', 'vol/second')
    ]
    with pytest.raises(ValueError, match='two documents name the image'):
        sd_create_ck.image_facts(documents)


def test_an_image_with_no_pointing_has_no_segment_to_build() -> None:
    """The guard on the one caller that could pass an omitted image.

    An assignment that names a baseline always carries pointing, so this is
    reachable only from an assignment that names a reason instead.
    """
    entry = ImageEntry(
        image_name='H_CALIB',
        status='failed',
        camera=None,
        shutter_mode=None,
        rotation_fitted=False,
        kernel_basenames=(),
        pointing=None,
        ineligibility_reason=OmissionReason.NOT_ELIGIBLE,
    )
    assignment = Assignment(entry=entry, baseline=None, omission_reason=OmissionReason.NOT_ELIGIBLE)
    with pytest.raises(ValueError, match='no pointing to write'):
        sd_create_ck.pointing_of(assignment)
