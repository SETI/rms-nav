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
from spindoctor.cli.ck.comments import read_comment_area

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


@pytest.fixture
def run_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a kernel directory, a results root and four images for one run.

    The kernels have to be furnished to build the baselines and to read the
    attitudes the metadata records, and they are unloaded again before the
    driver runs: the driver refuses to identify a clock kernel while another
    already defines that clock, which is the whole point of that refusal.
    """
    kernels = tmp_path / 'kernels'
    results = tmp_path / 'results'
    output = tmp_path / 'output'
    support = write_support_kernels(kernels)
    for path in support:
        cspyce.furnsh(str(path))
    baselines = []
    try:
        for name, centre in ((_BASELINE_A, _IMAGE_A_ET), (_BASELINE_B, _IMAGE_B_ET)):
            path = kernels / name
            write_baseline_ck(
                path,
                ck_frame_id=CASSINI_CK_FRAME_ID,
                sclk_id=_CASSINI_SCLK_ID,
                epochs=[centre - 10.0, centre, centre + 10.0],
                attitude=baseline_attitude,
                angular_velocity=baseline_angular_velocity,
            )
            baselines.append(path)
        cspyce.furnsh(str(baselines[0]))
        cspyce.furnsh(str(baselines[1]))
        original_a = _camera_attitude(_IMAGE_A_ET)
        original_b = _camera_attitude(_IMAGE_B_ET)
        drifted = _corrected(original_a)
    finally:
        for path in reversed([*support, *baselines]):
            cspyce.unload(str(path))
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


def _run(tree: dict[str, Path], monkeypatch: pytest.MonkeyPatch, *extra: str) -> None:
    """Run the driver over a prepared tree.

    Parameters:
        tree: The directories the fixture built.
        monkeypatch: Used to set the command line.
        extra: Additional arguments.
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
    assert exit_info.value.code == 0


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
# What the run does with inputs that are not what they should be
# ---------------------------------------------------------------------------


def test_a_kernel_directory_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """Named rather than silently contributing nothing."""
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        sd_create_ck.kernel_paths([str(tmp_path / 'gone')])


def test_a_kernel_directory_that_is_a_file_is_refused(tmp_path: Path) -> None:
    """A path that exists and is not a directory fails the same way."""
    path = tmp_path / 'notadir'
    path.write_text('')
    with pytest.raises(ValueError, match='does not exist or is not a directory'):
        sd_create_ck.kernel_paths([str(path)])


def test_a_basename_no_directory_holds_is_refused(tmp_path: Path) -> None:
    """A kernel an image names and the run cannot find is named, not guessed at."""
    with pytest.raises(ValueError, match='is not in any of the kernel directories'):
        sd_create_ck.resolve_one('cas00172.tsc', {})


def test_a_basename_two_directories_hold_is_refused(tmp_path: Path) -> None:
    """Two files of one name are two different kernels, and the record says which."""
    first = tmp_path / 'a'
    second = tmp_path / 'b'
    for root in (first, second):
        root.mkdir()
        (root / 'cas00172.tsc').write_text('')
    paths = sd_create_ck.kernel_paths([str(first), str(second)])
    with pytest.raises(ValueError, match='is in more than one kernel directory'):
        sd_create_ck.resolve_one('cas00172.tsc', paths)


def test_a_metadata_file_that_is_not_json_is_counted(tmp_path: Path) -> None:
    """It names no image, so it is reported rather than given a report row."""
    (tmp_path / f'broken{sd_create_ck.METADATA_SUFFIX}').write_text('{not json')
    documents, unreadable = sd_create_ck.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert documents == []
    assert unreadable == 1


def test_a_metadata_file_holding_a_json_array_is_counted(tmp_path: Path) -> None:
    """Valid JSON that is not a document is unreadable for the same reason."""
    (tmp_path / f'listy{sd_create_ck.METADATA_SUFFIX}').write_text('[1, 2]')
    _documents, unreadable = sd_create_ck.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert unreadable == 1


def test_a_document_from_another_mission_is_not_considered(tmp_path: Path) -> None:
    """A run is per mission, and another mission's images are not its business."""
    (tmp_path / f'other{sd_create_ck.METADATA_SUFFIX}').write_text(
        json.dumps({'status': 'success', 'observation': {'instrument': 'vgiss'}})
    )
    documents, unreadable = sd_create_ck.read_documents(FCPath(str(tmp_path)), 'coiss')
    assert documents == []
    assert unreadable == 0


def _timed(midtime: Any) -> sd_create_ck.Document:
    """Build a document recording one exposure midtime.

    Parameters:
        midtime: The value to record, of any type.

    Returns:
        The document.
    """
    return sd_create_ck.Document(
        path=FCPath('x_metadata.json'),
        stub='x',
        metadata={'navigation_result': {'times': {'midtime_et': midtime}}},
    )


@pytest.mark.parametrize(
    'midtime',
    [float('nan'), float('inf'), float('-inf'), None, True, 'later'],
    ids=['nan', 'inf', 'minus-inf', 'null', 'boolean', 'text'],
)
def test_an_unusable_midtime_cannot_be_placed_in_time(midtime: Any) -> None:
    """A NaN would otherwise fall inside every time range at once."""
    selected, undated = sd_create_ck.select_by_time([_timed(midtime)], 0.0, 1.0)
    assert selected == []
    assert undated == 1


def test_a_midtime_at_the_range_edge_is_selected() -> None:
    """Both bounds are inclusive, so an exposure exactly on one is inside."""
    selected, _undated = sd_create_ck.select_by_time([_timed(1.0)], 1.0, 1.0)
    assert len(selected) == 1


def test_an_unusable_midtime_is_kept_when_no_bound_is_given() -> None:
    """With no range to satisfy there is nothing to place the image against."""
    selected, undated = sd_create_ck.select_by_time([_timed(float('nan'))], None, None)
    assert len(selected) == 1
    assert undated == 0


def test_a_remote_output_directory_is_refused() -> None:
    """SPICE creates a kernel by name on the local filesystem and nowhere else."""
    with pytest.raises(ValueError, match='not a local directory'):
        sd_create_ck.local_output_path(FCPath('gs://bucket/out/orig_nav.bc'))
