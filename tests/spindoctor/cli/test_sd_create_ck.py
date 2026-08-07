"""End-to-end tests for the ``sd_create_ck`` driver.

One run, one mission, four images: two that navigate against different original
kernels, one that did not navigate at all, and one whose recorded baseline no
candidate reproduces.  What is asserted is everything the run leaves behind --
the corrected kernels and their comment areas, the meta-kernel's load order, the
report's one row per image, and the two logs -- because those are the products,
and each of them can be wrong on its own while the others look right.

Every kernel is written by the test.  The driver furnishes the pool itself and,
being a program, does not unload it again, so the pool fixture here records what
was furnished on the way in and unloads whatever the run added on the way out.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cspyce
import numpy as np
import pytest
from filecache import FCPath
from tests.spindoctor.cli.ck.conftest import (
    CASSINI_CAMERA_FRAME,
    CASSINI_CK_FRAME_ID,
    ET0,
    baseline_angular_velocity,
    baseline_attitude,
    image_metadata,
    write_baseline_ck,
    write_support_kernels,
)

from spindoctor.cli import sd_create_ck
from spindoctor.cli.ck import inputs
from spindoctor.cli.ck.assignment import Assignment
from spindoctor.cli.ck.comments import read_comment_area
from spindoctor.cli.ck.images import ImageEntry, OmissionReason
from spindoctor.cli.ck.index import CkFile, KernelClass
from spindoctor.cli.ck.pointing import ImagePointing

# The two exposures are far enough apart that no original kernel covers both,
# so each image has exactly one candidate and the two land in different files.
_IMAGE_A_ET = ET0
_IMAGE_B_ET = ET0 + 1000.0
_EXPOSURE_S = 2.0

_CASSINI_SCLK_ID = -82

_BASELINE_A = 'orig_a.bc'
_BASELINE_B = 'orig_b.bc'
_KERNEL_NAMES = ('test.tf', 'test.tls', 'test.tsc', _BASELINE_A, _BASELINE_B)


class _Furnished:
    """The pool as it was before a run, so a run's additions can be undone."""

    def __init__(self) -> None:
        """Record every kernel currently furnished."""
        self.before = self._loaded()

    @staticmethod
    def _loaded() -> list[str]:
        """Return the paths of every furnished kernel, in load order."""
        return [str(cspyce.kdata(at, 'ALL')[0]) for at in range(int(cspyce.ktotal('ALL')))]

    def restore(self) -> None:
        """Unload every kernel furnished since this object was built."""
        for path in reversed(self._loaded()):
            if path not in self.before:
                cspyce.unload(path)


@pytest.fixture
def pool_restored() -> Iterator[None]:
    """Undo whatever the driver furnished, leaving the process pool as found."""
    guard = _Furnished()
    try:
        yield
    finally:
        guard.restore()


def _camera_attitude(et: float) -> Any:
    """Return the camera attitude the baseline kernels give at one epoch.

    Parameters:
        et: TDB seconds past J2000.

    Returns:
        The 3x3 J2000-to-camera rotation, which is what the metadata records
        as the uncorrected attitude.
    """
    return np.asarray(cspyce.pxform('J2000', CASSINI_CAMERA_FRAME, et), dtype=np.float64)


def _corrected(attitude: Any) -> Any:
    """Return an attitude a small correction away from another.

    Parameters:
        attitude: The uncorrected 3x3 rotation.

    Returns:
        The corrected rotation, turned by a milliradian about a fixed axis so
        that a segment built from it differs measurably from its baseline.
    """
    axis = np.array([0.2, 0.5, -0.84])
    turn = np.asarray(cspyce.axisar(axis / np.linalg.norm(axis), 1.0e-3), dtype=np.float64)
    corrected: Any = turn @ np.asarray(attitude, dtype=np.float64)
    return corrected


def _write_metadata(root: Path, stub: str, metadata: dict[str, Any]) -> Path:
    """Write one per-image metadata document under a results root.

    Parameters:
        root: The navigation results root.
        stub: The image's results path stub.
        metadata: The document.

    Returns:
        The file written.
    """
    path = root / f'{stub}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def _image(
    *,
    image_name: str,
    midtime: float,
    cmatrix_original: Any,
    cmatrix: Any | None,
    status: str = 'success',
) -> dict[str, Any]:
    """Build one Cassini image's metadata document.

    Parameters:
        image_name: Basename recorded for the image.
        midtime: Exposure midtime, TDB seconds past J2000.
        cmatrix_original: The uncorrected attitude recorded for it.
        cmatrix: The corrected attitude, or None for an image without one.
        status: The navigation status.

    Returns:
        The document.
    """
    return image_metadata(
        image_name=image_name,
        cmatrix=cmatrix,
        cmatrix_original=cmatrix_original,
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=midtime - _EXPOSURE_S / 2.0,
        stop_et=midtime + _EXPOSURE_S / 2.0,
        status=status,
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        kernels=_KERNEL_NAMES,
        sclk_midtime='1/1484573295.118',
        offset=(-3.25, 1.125),
        sigma_px=(0.0625, 0.0313),
        confidence=0.8125,
        confidence_rank='high',
        status_reason='ensemble_agreement',
    )


def _write_kernels(kernels: Path, *, angular_velocity_in_b: bool = True) -> tuple[Any, Any]:
    """Write the hermetic kernels and the two baselines, and read what they give.

    The kernels have to be furnished to build the baselines and to read the
    attitudes the metadata records, and they are unloaded again before the
    driver runs: the driver refuses to identify a clock kernel while another
    already defines that clock, which is the whole point of that refusal.

    Parameters:
        kernels: The kernel directory to write into.
        angular_velocity_in_b: Whether the second baseline carries angular
            velocity.  Without it, an image assigned to that baseline
            reproduces its attitude and then cannot be built into a segment.

    Returns:
        The uncorrected camera attitudes at the two exposure midtimes.
    """
    support = write_support_kernels(kernels)
    for path in support:
        cspyce.furnsh(str(path))
    baselines = []
    try:
        for name, centre, with_av in (
            (_BASELINE_A, _IMAGE_A_ET, True),
            (_BASELINE_B, _IMAGE_B_ET, angular_velocity_in_b),
        ):
            path = kernels / name
            write_baseline_ck(
                path,
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=_CASSINI_SCLK_ID,
                epochs=[centre - 10.0, centre, centre + 10.0],
                attitude=baseline_attitude,
                angular_velocity=baseline_angular_velocity if with_av else None,
            )
            baselines.append(path)
        cspyce.furnsh(str(baselines[0]))
        cspyce.furnsh(str(baselines[1]))
        return _camera_attitude(_IMAGE_A_ET), _camera_attitude(_IMAGE_B_ET)
    finally:
        for path in reversed([*support, *baselines]):
            cspyce.unload(str(path))


@pytest.fixture
def run_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a kernel directory, a results root and four images for one run."""
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    original_a, original_b = _write_kernels(kernels)
    drifted = _corrected(original_a)
    _write_metadata(
        results,
        'vol/A_CALIB',
        _image(
            image_name='A_CALIB',
            midtime=_IMAGE_A_ET,
            cmatrix_original=original_a,
            cmatrix=_corrected(original_a),
        ),
    )
    _write_metadata(
        results,
        'vol/B_CALIB',
        _image(
            image_name='B_CALIB',
            midtime=_IMAGE_B_ET,
            cmatrix_original=original_b,
            cmatrix=_corrected(original_b),
        ),
    )
    _write_metadata(
        results,
        'vol/C_CALIB',
        _image(
            image_name='C_CALIB',
            midtime=_IMAGE_A_ET,
            cmatrix_original=original_a,
            cmatrix=None,
            status='failed',
        ),
    )
    _write_metadata(
        results,
        'vol/D_CALIB',
        _image(
            image_name='D_CALIB',
            midtime=_IMAGE_A_ET,
            cmatrix_original=drifted,
            cmatrix=_corrected(drifted),
        ),
    )
    return {'kernels': kernels, 'results': results, 'output': output}


@pytest.fixture
def refused_second_file_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a run whose second output file cannot be built at all.

    Two images navigate against two baselines, and the second baseline carries
    no angular velocity, which no segment can express.  The corrected files are
    written in name order, so the first one is buildable and the second is the
    refusal: a run that wrote as it went would leave the first file behind.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    original_a, original_b = _write_kernels(kernels, angular_velocity_in_b=False)
    for stub, name, midtime, original in (
        ('vol/A_CALIB', 'A_CALIB', _IMAGE_A_ET, original_a),
        ('vol/B_CALIB', 'B_CALIB', _IMAGE_B_ET, original_b),
    ):
        _write_metadata(
            results,
            stub,
            _image(
                image_name=name,
                midtime=midtime,
                cmatrix_original=original,
                cmatrix=_corrected(original),
            ),
        )
    return {'kernels': kernels, 'results': results, 'output': output}


@pytest.fixture
def straddling_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a run whose one image outlasts the baseline that reproduces it.

    The baseline covers a second either side of the exposure midtime and the
    exposure runs for four, so the midtime reproduces -- which is what pairs
    the image with this baseline -- and the segment's start and stop records
    then have no pointing to read.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    support = write_support_kernels(kernels)
    for path in support:
        cspyce.furnsh(str(path))
    baseline = kernels / _BASELINE_A
    try:
        write_baseline_ck(
            baseline,
            ck_frame_id=CASSINI_CK_FRAME_ID,
            sclk_id=_CASSINI_SCLK_ID,
            epochs=[_IMAGE_A_ET - 1.0, _IMAGE_A_ET, _IMAGE_A_ET + 1.0],
            attitude=baseline_attitude,
            angular_velocity=baseline_angular_velocity,
        )
        cspyce.furnsh(str(baseline))
        original = _camera_attitude(_IMAGE_A_ET)
    finally:
        for path in reversed([*support, baseline]):
            cspyce.unload(str(path))
    metadata = image_metadata(
        image_name='G_CALIB',
        cmatrix=_corrected(original),
        cmatrix_original=original,
        camera_frame=CASSINI_CAMERA_FRAME,
        ck_frame_id=CASSINI_CK_FRAME_ID,
        start_et=_IMAGE_A_ET - 2.0,
        stop_et=_IMAGE_A_ET + 2.0,
        status='success',
        instrument='coiss',
        camera='NAC',
        shutter_mode='NACONLY',
        kernels=_KERNEL_NAMES,
        sclk_midtime='1/1484573295.118',
        offset=(-3.25, 1.125),
        sigma_px=(0.0625, 0.0313),
        confidence=0.8125,
        confidence_rank='high',
        status_reason='ensemble_agreement',
    )
    _write_metadata(results, 'vol/G_CALIB', metadata)
    return {'kernels': kernels, 'results': results, 'output': output}


def _run(
    tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
    expected_exit: int = 0,
) -> None:
    """Run the driver over a prepared tree.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
        extra: Additional arguments.
        expected_exit: The exit status the run should end with.
    """
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            str(tree['results']),
            '--kernel-dir',
            str(tree['kernels']),
            '--output-dir',
            str(tree['output']),
            '--log-root',
            str(tree['output'] / 'logs'),
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        sd_create_ck.main()
    assert exit_info.value.code == expected_exit


def _utc(tree: dict[str, Path], et: float) -> str:
    """Return one epoch as a UTC string, with the leapseconds kernel furnished.

    The run tree deliberately leaves nothing furnished, so a test that needs to
    express an epoch in UTC furnishes the leapseconds kernel for that one call.

    Parameters:
        tree: The directories the fixture built.
        et: TDB seconds past J2000.

    Returns:
        The epoch as an ISO calendar UTC string.
    """
    lsk = str(tree['kernels'] / 'test.tls')
    cspyce.furnsh(lsk)
    try:
        return str(cspyce.et2utc(et, 'ISOC', 3))
    finally:
        cspyce.unload(lsk)


def _report_rows(tree: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Read the report the run wrote, keyed by image name.

    Parameters:
        tree: The directories the fixture built.

    Returns:
        One entry per row.
    """
    with (tree['output'] / 'coiss_ck_report.csv').open() as stream:
        return {row['image_name']: row for row in csv.DictReader(stream)}


# ---------------------------------------------------------------------------
# The corrected kernels
# ---------------------------------------------------------------------------


def test_one_corrected_kernel_per_original(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Each original an image navigated against gets a mirror of its own."""
    _run(run_tree, monkeypatch)
    written = sorted(path.name for path in run_tree['output'].glob('*.bc'))
    assert written == ['orig_a_nav.bc', 'orig_b_nav.bc']


def test_no_kernel_is_written_for_an_original_nothing_reproduces(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The drifted image writes nothing at all rather than an empty file."""
    _run(run_tree, monkeypatch)
    assert not (run_tree['output'] / 'orig_a_nav_nav.bc').exists()


def test_the_corrected_kernel_describes_the_object(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A plain reader finds the bus in the file, with no writer code present."""
    _run(run_tree, monkeypatch)
    path = run_tree['output'] / 'orig_a_nav.bc'
    assert [int(value) for value in cspyce.ckobj(str(path))] == [CASSINI_CK_FRAME_ID]


# ---------------------------------------------------------------------------
# The comment area
# ---------------------------------------------------------------------------


def test_the_comment_area_names_the_baseline(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Read back through the same DAF interface a consumer would use."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(_BASELINE_A in line for line in lines)


def test_the_comment_area_names_the_clock_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the one the run chose from the image's own provenance."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any('test.tsc' in line for line in lines)


def test_the_comment_area_names_the_image_it_carries(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And only that image, not every image the run considered."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(line.startswith('A_CALIB') for line in lines)
    assert not any(line.startswith('B_CALIB') for line in lines)


def test_each_file_names_the_image_it_actually_carries(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Checked on the second file too, where taking the run's first image would
    read as a plausible comment area on a file that does not hold that image."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_b_nav.bc')
    assert any(line.startswith('B_CALIB') for line in lines)
    assert not any(line.startswith('A_CALIB') for line in lines)


def test_the_comment_area_names_the_generator_version(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A file says what wrote it, so a regeneration can be told from an old one."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    assert any(line.startswith('Generator version:') for line in lines)


def test_the_comment_area_names_the_configuration_hash(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And under what configuration it ran."""
    _run(run_tree, monkeypatch)
    lines = read_comment_area(run_tree['output'] / 'orig_a_nav.bc')
    stated = [line for line in lines if line.startswith('Configuration hash:')]
    assert len(stated[0].split()[-1]) == 64


# ---------------------------------------------------------------------------
# The meta-kernel
# ---------------------------------------------------------------------------


def test_the_meta_kernel_furnishes_originals_before_corrections(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Furnished for real, because load order is the only thing it decides."""
    _run(run_tree, monkeypatch)
    meta = run_tree['output'] / 'coiss_nav.tm'
    cspyce.furnsh(str(meta))
    try:
        loaded = [str(cspyce.kdata(at, 'CK')[0]) for at in range(int(cspyce.ktotal('CK')))]
    finally:
        cspyce.unload(str(meta))
    assert loaded == [
        str(run_tree['kernels'] / _BASELINE_A),
        str(run_tree['kernels'] / _BASELINE_B),
        str(run_tree['output'] / 'orig_a_nav.bc'),
        str(run_tree['output'] / 'orig_b_nav.bc'),
    ]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_every_image_considered_appears_exactly_once(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Four images considered, four rows, whatever became of each."""
    _run(run_tree, monkeypatch)
    rows = _report_rows(run_tree)
    assert sorted(rows) == ['A_CALIB', 'B_CALIB', 'C_CALIB', 'D_CALIB']


def test_a_corrected_image_names_the_file_carrying_its_segment(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """And it is the mirror of the original that image navigated against."""
    _run(run_tree, monkeypatch)
    rows = _report_rows(run_tree)
    assert rows['A_CALIB']['source_bc'] == 'orig_a_nav.bc'
    assert rows['B_CALIB']['source_bc'] == 'orig_b_nav.bc'


def test_a_corrected_image_carries_no_omission_reason(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A row says one thing or the other, never both."""
    _run(run_tree, monkeypatch)
    assert _report_rows(run_tree)['A_CALIB']['omission_reason'] == ''


def test_an_image_that_did_not_navigate_is_reported_as_not_eligible(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With no source file, so it cannot be read as corrected."""
    _run(run_tree, monkeypatch)
    row = _report_rows(run_tree)['C_CALIB']
    assert row['omission_reason'] == 'not_eligible'
    assert row['source_bc'] == ''


def test_an_image_whose_baseline_no_candidate_reproduces_is_reported(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the detector for a kernel set that changed since navigation."""
    _run(run_tree, monkeypatch)
    assert _report_rows(run_tree)['D_CALIB']['omission_reason'] == 'no_reproducing_baseline'


def test_the_report_carries_the_measurement_the_metadata_recorded(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The offset column is the recorded one, unrounded."""
    _run(run_tree, monkeypatch)
    row = _report_rows(run_tree)['A_CALIB']
    assert row['offset_dv'] == '-3.25'
    assert row['confidence_rank'] == 'high'


# ---------------------------------------------------------------------------
# Time selection
# ---------------------------------------------------------------------------


def test_a_time_range_selects_the_images_inside_it(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An image outside the range was never considered and gets no row."""
    stop = _utc(run_tree, _IMAGE_A_ET + 100.0)
    _run(run_tree, monkeypatch, '--stop-time', stop)
    assert 'B_CALIB' not in _report_rows(run_tree)


def test_a_time_range_still_writes_the_images_inside_it(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Only the original the surviving images navigated against is mirrored."""
    stop = _utc(run_tree, _IMAGE_A_ET + 100.0)
    _run(run_tree, monkeypatch, '--stop-time', stop)
    assert sorted(path.name for path in run_tree['output'].glob('*.bc')) == ['orig_a_nav.bc']


# ---------------------------------------------------------------------------
# Both logs
# ---------------------------------------------------------------------------


def _run_log(tree: dict[str, Path]) -> str:
    """Return what the run wrote to its main log.

    Parameters:
        tree: The directories the fixture built.

    Returns:
        The main log's text.
    """
    logs = list((tree['output'] / 'logs' / 'sd_create_ck').glob('main_*.log'))
    assert len(logs) == 1
    return logs[0].read_text()


def test_an_omission_reaches_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An operator watching a batch sees it without opening a per-image log."""
    _run(run_tree, monkeypatch)
    assert 'D_CALIB: no corrected segment written (no_reproducing_baseline)' in _run_log(run_tree)


def test_the_run_log_counts_each_omission_reason(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """One line per reason at the end, so a batch's shape is readable at a glance."""
    _run(run_tree, monkeypatch)
    assert 'Images omitted, no_reproducing_baseline: 1' in _run_log(run_tree)


def test_the_run_log_counts_the_images_corrected(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Against which the omissions can be read as a fraction of the batch."""
    _run(run_tree, monkeypatch)
    assert 'Images corrected 2' in _run_log(run_tree)


def test_an_omission_reaches_the_image_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With the detail, in the log of the image it happened to."""
    _run(run_tree, monkeypatch)
    logs = list((run_tree['output'] / 'logs').rglob('*D_CALIB*'))
    assert len(logs) == 1
    assert 'no_reproducing_baseline' in logs[0].read_text()


def test_a_corrected_image_records_its_file_in_its_own_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """So an image's own log answers what became of it either way."""
    _run(run_tree, monkeypatch)
    logs = list((run_tree['output'] / 'logs').rglob('*A_CALIB*'))
    assert 'orig_a_nav.bc' in logs[0].read_text()


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
    _run(run_tree, monkeypatch, expected_exit=1)
    log = _run_log(run_tree)
    assert 'broken_metadata.json' in log
    assert 'Could not read 1 metadata file(s)' in log


def test_an_unreadable_metadata_file_makes_the_run_exit_non_zero(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A batch wrapper cannot otherwise tell a clean run from a skipped one."""
    (run_tree['results'] / 'vol' / f'broken{inputs.METADATA_SUFFIX}').write_text('[1, 2]')
    _run(run_tree, monkeypatch, expected_exit=1)


def test_the_run_still_writes_what_it_could_read(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """One unreadable file is reported, not a reason to abandon the others."""
    (run_tree['results'] / 'vol' / f'broken{inputs.METADATA_SUFFIX}').write_text('{not json')
    _run(run_tree, monkeypatch, expected_exit=1)
    assert (run_tree['output'] / 'orig_a_nav.bc').exists()


def test_an_image_that_cannot_be_placed_in_time_is_reported(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A time range was asked for and this image cannot be shown to satisfy it."""
    metadata = _image(
        image_name='E_CALIB',
        midtime=_IMAGE_A_ET,
        cmatrix_original=np.eye(3),
        cmatrix=np.eye(3),
    )
    del metadata['navigation_result']['times']['midtime_et']
    _write_metadata(run_tree['results'], 'vol/E_CALIB', metadata)
    _run(run_tree, monkeypatch, '--start-time', _utc(run_tree, _IMAGE_A_ET - 100.0))
    assert 'Ignored 1 image(s) that recorded no exposure midtime' in _run_log(run_tree)


def test_a_run_selecting_nothing_says_so_and_writes_nothing(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A time range that excludes every image is not an error, and not silent."""
    _run(run_tree, monkeypatch, '--start-time', _utc(run_tree, _IMAGE_B_ET + 10000.0))
    assert 'No images selected; nothing to write' in _run_log(run_tree)
    assert list(run_tree['output'].glob('*.bc')) == []


def test_a_run_correcting_nothing_writes_no_meta_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A meta-kernel naming no correction would furnish only the originals."""
    for name in ('A_CALIB', 'B_CALIB', 'D_CALIB'):
        (run_tree['results'] / 'vol' / f'{name}_metadata.json').unlink()
    _run(run_tree, monkeypatch)
    assert 'no meta-kernel written' in _run_log(run_tree)
    assert not (run_tree['output'] / 'coiss_nav.tm').exists()


def test_a_run_correcting_nothing_still_writes_the_report(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The report is the answer, and 'nothing was corrected' is an answer."""
    for name in ('A_CALIB', 'B_CALIB', 'D_CALIB'):
        (run_tree['results'] / 'vol' / f'{name}_metadata.json').unlink()
    _run(run_tree, monkeypatch)
    assert _report_rows(run_tree)['C_CALIB']['omission_reason'] == 'not_eligible'


def test_a_document_that_is_not_a_navigated_image_stops_the_run(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The closed reason set has no entry for a record the reader cannot read."""
    metadata = _image(
        image_name='F_CALIB',
        midtime=_IMAGE_A_ET,
        cmatrix_original=np.eye(3),
        cmatrix=np.eye(3),
    )
    metadata['navigation_result']['provenance']['spice_kernels'] = []
    _write_metadata(run_tree['results'], 'vol/F_CALIB', metadata)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            str(run_tree['results']),
            '--kernel-dir',
            str(run_tree['kernels']),
            '--output-dir',
            str(run_tree['output']),
            '--log-root',
            str(run_tree['output'] / 'logs'),
        ],
    )
    with pytest.raises(ValueError, match='spice_kernels'):
        sd_create_ck.main()


def test_a_document_that_is_not_a_navigated_image_is_named_in_the_run_log(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Named before it propagates, so the operator knows which file to look at."""
    metadata = _image(
        image_name='F_CALIB',
        midtime=_IMAGE_A_ET,
        cmatrix_original=np.eye(3),
        cmatrix=np.eye(3),
    )
    metadata['navigation_result']['provenance']['spice_kernels'] = []
    _write_metadata(run_tree['results'], 'vol/F_CALIB', metadata)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            str(run_tree['results']),
            '--kernel-dir',
            str(run_tree['kernels']),
            '--output-dir',
            str(run_tree['output']),
            '--log-root',
            str(run_tree['output'] / 'logs'),
        ],
    )
    with pytest.raises(ValueError, match='spice_kernels'):
        sd_create_ck.main()
    assert 'F_CALIB_metadata.json: cannot be read as a navigated image' in _run_log(run_tree)


def test_an_exposure_its_baseline_does_not_cover_is_reported(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The run finishes and the image is one row of the report, like any omission.

    Not ``no_reproducing_baseline``: this baseline did reproduce, at the
    midtime, which is what paired the image with it, and that reason is the
    detector for holdings that changed since navigation ran.
    """
    _run(straddling_tree, monkeypatch)
    assert _report_rows(straddling_tree)['G_CALIB']['omission_reason'] == 'baseline_coverage_gap'


def test_an_exposure_its_baseline_does_not_cover_is_not_reported_as_drift(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The drift detector stays clean, which is what makes it worth watching."""
    _run(straddling_tree, monkeypatch)
    assert 'Images omitted, no_reproducing_baseline: 0' in _run_log(straddling_tree)


def test_an_exposure_its_baseline_does_not_cover_names_no_source_file(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """It received no segment, so no file can be said to carry one."""
    _run(straddling_tree, monkeypatch)
    assert _report_rows(straddling_tree)['G_CALIB']['source_bc'] == ''


def test_an_exposure_its_baseline_does_not_cover_appears_once(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Every image considered appears exactly once, this one included."""
    _run(straddling_tree, monkeypatch)
    with (straddling_tree['output'] / 'coiss_ck_report.csv').open() as stream:
        names = [row['image_name'] for row in csv.DictReader(stream)]
    assert names == ['G_CALIB']


def test_an_exposure_its_baseline_does_not_cover_writes_no_kernel(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Its baseline is left with nothing to correct, and an empty file claims a correction."""
    _run(straddling_tree, monkeypatch)
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
    _run(straddling_tree, monkeypatch)
    logs = list((straddling_tree['output'] / 'logs').rglob('*G_CALIB*'))
    assert len(logs) == 1
    assert 'baseline_coverage_gap' in logs[0].read_text()


def test_an_exposure_its_baseline_does_not_cover_is_named_in_the_run_log(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """With the record epoch, which the reason itself does not carry."""
    _run(straddling_tree, monkeypatch)
    log = _run_log(straddling_tree)
    assert 'G_CALIB: no corrected segment written (baseline_coverage_gap)' in log
    assert 'G_CALIB: the furnished baseline supplies no pointing for CK object -82000' in log


def test_an_exposure_its_baseline_does_not_cover_is_counted(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The end-of-run counts carry the new reason like the other four."""
    _run(straddling_tree, monkeypatch)
    assert 'Images omitted, baseline_coverage_gap: 1' in _run_log(straddling_tree)


def test_a_baseline_left_with_nothing_to_correct_says_so(
    straddling_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """An operator sees why a baseline they expected to be mirrored was not."""
    _run(straddling_tree, monkeypatch)
    assert f'No image is left to correct {_BASELINE_A}' in _run_log(straddling_tree)


def _run_stopped(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Run the driver over a tree it must refuse, and hold it to the reason.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
        message: Text the refusal must name.
    """
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            str(tree['results']),
            '--kernel-dir',
            str(tree['kernels']),
            '--output-dir',
            str(tree['output']),
            '--log-root',
            str(tree['output'] / 'logs'),
        ],
    )
    with pytest.raises(ValueError, match=message):
        sd_create_ck.main()


def _run_refused(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the driver over a tree whose second output file cannot be built.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
    """
    _run_stopped(tree, monkeypatch, 'angular velocity at only some of them')


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
    assert 'B_CALIB: could not build the corrected segment' in _run_log(refused_second_file_tree)


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
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')


def test_an_occupied_second_output_path_leaves_the_first_unwritten(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The whole point: a run that judged each file as it reached it wrote this one."""
    _block_second_output(run_tree)
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'orig_a_nav.bc').exists()


def test_an_occupied_second_output_path_leaves_no_meta_kernel(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """A meta-kernel would name a corrected set that was never written."""
    _block_second_output(run_tree)
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'coiss_nav.tm').exists()


def test_an_occupied_second_output_path_leaves_no_report(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Nothing was written, so nothing claims to say what was: the run is repeatable."""
    _block_second_output(run_tree)
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert not (run_tree['output'] / 'coiss_ck_report.csv').exists()


def test_an_occupied_second_output_path_is_left_alone(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The file the operator has to choose between is not the one that changed."""
    blocked = _block_second_output(run_tree)
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc already exists')
    assert blocked.read_bytes() == b'not a kernel'


def test_a_dangling_link_at_the_second_output_path_stops_the_run(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """``Path.exists`` follows the link and reports the absent target as absent."""
    _block_second_output(run_tree, with_link_to=run_tree['output'] / 'elsewhere.bc')
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')


def test_a_dangling_link_at_the_second_output_path_creates_no_target(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """Which is the harm: the write would create a file the run never named."""
    target = run_tree['output'] / 'elsewhere.bc'
    _block_second_output(run_tree, with_link_to=target)
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')
    assert not target.exists()


def test_a_dangling_link_at_the_second_output_path_leaves_the_first_unwritten(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """The set is judged before the first file, whatever occupies a later path."""
    _block_second_output(run_tree, with_link_to=run_tree['output'] / 'elsewhere.bc')
    _run_stopped(run_tree, monkeypatch, r'orig_b_nav\.bc is a symbolic link')
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
            start_et=_IMAGE_A_ET - 1.0,
            stop_et=_IMAGE_A_ET + 1.0,
            midtime_et=_IMAGE_A_ET,
            exposure_s=2.0,
        )
    return ImageEntry(
        image_name=image_name,
        status='success' if corrected else 'failed',
        camera='NAC',
        shutter_mode='NACONLY',
        rotation_fitted=False,
        kernel_basenames=_KERNEL_NAMES if corrected else (),
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
        path=FCPath(f'/kernels/{_BASELINE_A}'),
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


# ---------------------------------------------------------------------------
# Paths a consumer has to be able to resolve
# ---------------------------------------------------------------------------


def test_the_meta_kernel_names_absolute_paths(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None
) -> None:
    """SPICE resolves a relative name against the consumer's directory, not ours."""
    monkeypatch.chdir(run_tree['output'].parent)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            'results',
            '--kernel-dir',
            'kernels',
            '--output-dir',
            'output',
            '--log-root',
            'output/logs',
        ],
    )
    with pytest.raises(SystemExit):
        sd_create_ck.main()
    meta = str(run_tree['output'] / 'coiss_nav.tm')
    cspyce.furnsh(meta)
    try:
        named = [str(cspyce.kdata(at, 'CK')[0]) for at in range(int(cspyce.ktotal('CK')))]
    finally:
        cspyce.unload(meta)
    assert [name for name in named if not name.startswith('/')] == []


def test_a_meta_kernel_written_from_a_relative_run_still_furnishes(
    run_tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pool_restored: None, tmp_path: Path
) -> None:
    """Furnished from somewhere else entirely, which is where a consumer is."""
    monkeypatch.chdir(run_tree['output'].parent)
    monkeypatch.setattr(
        'sys.argv',
        [
            'sd_create_ck',
            'coiss',
            '--nav-results-root',
            'results',
            '--kernel-dir',
            'kernels',
            '--output-dir',
            'output',
            '--log-root',
            'output/logs',
        ],
    )
    with pytest.raises(SystemExit):
        sd_create_ck.main()
    monkeypatch.chdir(tmp_path.parent)
    meta = str(run_tree['output'] / 'coiss_nav.tm')
    cspyce.furnsh(meta)
    try:
        assert int(cspyce.ktotal('CK')) == 4
    finally:
        cspyce.unload(meta)


def test_a_remote_directory_is_left_as_it_was_given() -> None:
    """It is already absolute, and there is no local directory to resolve it against."""
    assert sd_create_ck.absolute_directory('gs://bucket/kernels') == 'gs://bucket/kernels'
