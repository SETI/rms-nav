import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.cli.stats.classify import datetime_from_image_et
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.dataset_pds3 import DataSetPDS3
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.dataset.dataset_pds3_galileo_ssi import DataSetPDS3GalileoSSI
from spindoctor.dataset.dataset_pds3_newhorizons_lorri import DataSetPDS3NewHorizonsLORRI
from spindoctor.dataset.dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS


@pytest.fixture
def ds() -> DataSetPDS3CassiniISS:
    return DataSetPDS3CassiniISS('/fake/holdings')


class _FakeIndexTable:
    """Stand-in for a PdsTable serving canned index rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def dicts_by_row(self) -> list[dict[str, Any]]:
        """Return the canned rows in index order."""
        return self._rows


class _FakeIndexCache:
    """Stand-in for the index FileCache: echoes URLs back as local paths."""

    def retrieve(self, urls: list[str]) -> list[Path]:
        """Return the label/table URL pair as paths without any I/O."""
        return [Path(urls[0]), Path(urls[1])]


def _install_fake_index(
    ds: DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
    volume_filespecs: dict[str, list[str]],
) -> list[str]:
    """Serve synthetic index rows per volume; returns the log of volumes read."""
    volumes_read: list[str] = []

    def fake_read_pds_table(fn: Path, columns: tuple[str, ...] | None = None) -> _FakeIndexTable:
        for vol, specs in volume_filespecs.items():
            if vol in str(fn):
                volumes_read.append(vol)
                return _FakeIndexTable([{'FILE_SPECIFICATION_NAME': s} for s in specs])
        raise AssertionError(f'Unexpected index read: {fn}')

    monkeypatch.setattr(ds, '_index_filecache', _FakeIndexCache())
    monkeypatch.setattr(ds, '_read_pds_table', fake_read_pds_table)
    return volumes_read


def _coiss_filespecs(camera: str, numbers: list[int]) -> list[str]:
    """Index filespecs for one camera, in the index's per-camera sorted order."""
    range_dir = f'{numbers[0]:010d}_{numbers[-1]:010d}'
    return [f'data/{range_dir}/{camera}{num:010d}_1.IMG' for num in numbers]


def _yielded_names(groups: list[Any]) -> list[str]:
    """Base image names (no suffix) of the yielded single-image groups."""
    return [g.image_files[0].image_file_name.split('_')[0] for g in groups]


def test_last_image_num_keeps_wac_frames(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # COISS indexes sort all N* rows before all W* rows, so the image-number
    # sequence resets partway through the volume. In-range WAC frames after the
    # reset must still be yielded (issue #136).
    numbers = [1000000100, 1000000101, 1000000102, 1000000103, 1000000104]
    _install_fake_index(
        ds,
        monkeypatch,
        {'COISS_2001': _coiss_filespecs('N', numbers) + _coiss_filespecs('W', numbers)},
    )

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001'], img_end_num=1000000102))

    assert _yielded_names(groups) == [
        'N1000000100',
        'N1000000101',
        'N1000000102',
        'W1000000100',
        'W1000000101',
        'W1000000102',
    ]


def test_scan_stops_after_first_volume_fully_past_range(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Volumes are chronological, so the scan stops after the first volume whose
    # rows are all past img_end_num -- without reading any later volume's index.
    in_range = [1000000100, 1000000101]
    past_range = [1000000200, 1000000201]
    far_past_range = [1000000300, 1000000301]
    volumes_read = _install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': _coiss_filespecs('N', in_range) + _coiss_filespecs('W', in_range),
            'COISS_2002': _coiss_filespecs('N', past_range),
            'COISS_2003': _coiss_filespecs('N', far_past_range),
        },
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002', 'COISS_2003'], img_end_num=1000000101
        )
    )

    assert len(groups) == 4
    assert volumes_read == ['COISS_2001', 'COISS_2002']


def test_img_name_list_range_clamp_still_stops_scan(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit image-name list clamps the number range; rows rejected by the
    # name list must still report past-the-range so the volume scan terminates.
    numbers = [1000000100, 1000000101, 1000000102]
    later = [1000000200, 1000000201]
    volumes_read = _install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': _coiss_filespecs('N', numbers) + _coiss_filespecs('W', numbers),
            'COISS_2002': _coiss_filespecs('N', later),
        },
    )

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002'], img_name_list=['N1000000101']
        )
    )

    assert _yielded_names(groups) == ['N1000000101']
    assert volumes_read == ['COISS_2001', 'COISS_2002']


# --- Results-based filters (--has-offset-file and friends) ---


_FILTER_NUMS = [1000000100, 1000000101, 1000000102]


def _install_two_camera_index(ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a COISS_2001 index with three NAC and three WAC frames."""
    _install_fake_index(
        ds,
        monkeypatch,
        {'COISS_2001': _coiss_filespecs('N', _FILTER_NUMS) + _coiss_filespecs('W', _FILTER_NUMS)},
    )


def _write_result_file(
    results_root: Path,
    volume: str,
    numbers: list[int],
    camera: str,
    num: int,
    suffix: str,
    content: str = 'x',
) -> None:
    """Write one synthetic result file where the pipeline would put it.

    The Cassini results path stub comes from the label filespec, so the
    filename carries the ``_CALIB`` label suffix.
    """
    range_dir = f'{numbers[0]:010d}_{numbers[-1]:010d}'
    path = results_root / volume / 'data' / range_dir / f'{camera}{num:010d}_1_CALIB{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_has_offset_file_keeps_only_navigated_in_order(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only the frames with an existing metadata file are yielded, in normal
    # enumeration order (NAC rows before WAC rows within the volume). A summary
    # PNG alone must not satisfy the offset-file filter.
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'W', 1000000100, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000100, '_summary.png')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_file=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'W1000000100']


def test_has_no_offset_file_excludes_navigated(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'W', 1000000100, '_metadata.json')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_no_offset_file=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == [
        'N1000000100',
        'N1000000102',
        'W1000000101',
        'W1000000102',
    ]


def test_has_png_file_ands_with_has_no_offset_file(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The flags AND together: PNG must exist and the metadata file must not.
    _install_two_camera_index(ds, monkeypatch)
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000100, '_summary.png')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_summary.png')
    _write_result_file(tmp_path, 'COISS_2001', _FILTER_NUMS, 'N', 1000000101, '_metadata.json')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_png_file=True,
            has_no_offset_file=True,
            nav_results_root=str(tmp_path),
        )
    )

    assert _yielded_names(groups) == ['N1000000100']


def _write_error_metadata(tmp_path: Path) -> None:
    """Write metadata files: one success, one SPICE error, one non-SPICE error."""
    contents = {
        1000000100: {'status': 'success'},
        1000000101: {'status': 'error', 'status_error': 'missing_spice_data'},
        1000000102: {'status': 'error', 'status_error': 'image_read_error'},
    }
    for num, metadata in contents.items():
        _write_result_file(
            tmp_path,
            'COISS_2001',
            _FILTER_NUMS,
            'N',
            num,
            '_metadata.json',
            json.dumps(metadata),
        )


def test_has_offset_error_matches_any_fatal_error(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # WAC frames have no metadata file at all and are excluded by the implied
    # presence filter; the success metadata is excluded by its status.
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101', 'N1000000102']


def test_has_offset_spice_error_matches_only_spice(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'], has_offset_spice_error=True, nav_results_root=str(tmp_path)
        )
    )

    assert _yielded_names(groups) == ['N1000000101']


def test_has_offset_nonspice_error_matches_only_nonspice(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_two_camera_index(ds, monkeypatch)
    _write_error_metadata(tmp_path)

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001'],
            has_offset_nonspice_error=True,
            nav_results_root=str(tmp_path),
        )
    )

    assert _yielded_names(groups) == ['N1000000102']


@pytest.mark.parametrize(
    'flags',
    [
        {'has_offset_file': True, 'has_no_offset_file': True},
        {'has_png_file': True, 'has_no_png_file': True},
        {'has_offset_spice_error': True, 'has_offset_nonspice_error': True},
        {'has_offset_error': True, 'has_no_offset_file': True},
    ],
)
def test_contradictory_results_flags_raise(
    ds: DataSetPDS3CassiniISS,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    flags: dict[str, bool],
) -> None:
    _install_two_camera_index(ds, monkeypatch)

    with pytest.raises(ValueError, match=r'mutually exclusive|contradicts'):
        list(
            ds.yield_image_files_index(
                volumes=['COISS_2001'], nav_results_root=str(tmp_path), **flags
            )
        )


# --- Uniform random sampling (--choose-random-images) ---


def test_choose_random_images_pool_spans_all_volumes(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool is built from every selected volume before sampling. With
    # shuffle replaced by reverse, the yields are deterministic and cross the
    # volume boundary, in shuffled (not enumeration) order.
    nums1 = [1000000100, 1000000101, 1000000102]
    nums2 = [1000000200, 1000000201, 1000000202]
    volumes_read = _install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': _coiss_filespecs('N', nums1),
            'COISS_2002': _coiss_filespecs('N', nums2),
        },
    )
    monkeypatch.setattr(random, 'shuffle', lambda pool: pool.reverse())

    groups = list(
        ds.yield_image_files_index(volumes=['COISS_2001', 'COISS_2002'], choose_random_images=4)
    )

    assert _yielded_names(groups) == [
        'N1000000202',
        'N1000000201',
        'N1000000200',
        'N1000000102',
    ]
    assert volumes_read == ['COISS_2001', 'COISS_2002']


def test_choose_random_images_returns_requested_count(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_two_camera_index(ds, monkeypatch)

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001'], choose_random_images=2))

    names = _yielded_names(groups)
    assert len(names) == 2
    assert len(set(names)) == 2
    all_names = {f'{camera}{num}' for camera in 'NW' for num in _FILTER_NUMS}
    assert set(names) <= all_names


def test_choose_random_images_with_offset_filter(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Random sampling combined with a results filter: only navigated images are
    # candidates, and asking for more than exist returns exactly the candidates.
    nums1 = [1000000100, 1000000101, 1000000102]
    nums2 = [1000000200, 1000000201, 1000000202]
    _install_fake_index(
        ds,
        monkeypatch,
        {
            'COISS_2001': _coiss_filespecs('N', nums1),
            'COISS_2002': _coiss_filespecs('N', nums2),
        },
    )
    _write_result_file(tmp_path, 'COISS_2001', nums1, 'N', 1000000101, '_metadata.json')
    _write_result_file(tmp_path, 'COISS_2002', nums2, 'N', 1000000201, '_metadata.json')

    groups = list(
        ds.yield_image_files_index(
            volumes=['COISS_2001', 'COISS_2002'],
            choose_random_images=10,
            has_offset_file=True,
            nav_results_root=str(tmp_path),
        )
    )

    assert sorted(_yielded_names(groups)) == ['N1000000101', 'N1000000201']


@pytest.mark.parametrize('bad', [0, -1, -5])
def test_choose_random_images_rejects_non_positive_programmatic(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch, bad: int
) -> None:
    # 0 would silently disable sampling (yielding everything) and a negative
    # value would yield exactly one frame; both are rejected at the boundary.
    _install_two_camera_index(ds, monkeypatch)

    with pytest.raises(ValueError, match='positive integer'):
        list(ds.yield_image_files_index(volumes=['COISS_2001'], choose_random_images=bad))


@pytest.mark.parametrize('bad', ['0', '-3'])
def test_choose_random_images_argparse_rejects_non_positive(bad: str) -> None:
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(['--choose-random-images', bad])


def test_choose_random_images_argparse_accepts_positive() -> None:
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    arguments = parser.parse_args(['--choose-random-images', '5'])

    assert arguments.choose_random_images == 5


def test_selection_arguments_include_results_filters() -> None:
    parser = argparse.ArgumentParser()
    DataSetPDS3CassiniISS.add_selection_arguments(parser)

    arguments = parser.parse_args(['--has-offset-file', '--has-no-png-file'])

    assert arguments.has_offset_file is True
    assert arguments.has_no_png_file is True
    assert arguments.has_offset_spice_error is False


def test_yielded_imagefile_carries_label_resolver(
    ds: DataSetPDS3CassiniISS, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_index(
        ds, monkeypatch, {'COISS_2001': _coiss_filespecs('N', [1000000100, 1000000100])[:1]}
    )

    groups = list(ds.yield_image_files_index(volumes=['COISS_2001']))

    assert len(groups) == 1
    assert groups[0].image_files[0].image_url_resolver is not None


# --- ^IMAGE pointer parsing (issue #12) ---


def _write_label(tmp_path: Path, name: str, text: str) -> Path:
    """Write a synthetic PDS3 label file and return its path."""
    label_path = tmp_path / name
    label_path.write_text(text)
    return label_path


@pytest.mark.parametrize(
    ('pointer_line', 'expected'),
    [
        ('^IMAGE = ("N1454725799_1.IMG",4)', 'N1454725799_1.IMG'),
        ('^IMAGE                          = ("C3250013_GEOMED.IMG", 2)', 'C3250013_GEOMED.IMG'),
        ('^IMAGE = "LOR_0003103486_0X630_SCI.FIT"', 'LOR_0003103486_0X630_SCI.FIT'),
        ('^IMAGE = N1454725799_1.IMG', 'N1454725799_1.IMG'),
    ],
)
def test_image_filename_from_label_pointer_forms(
    tmp_path: Path, pointer_line: str, expected: str
) -> None:
    label_path = _write_label(
        tmp_path,
        'test.LBL',
        f'PDS_VERSION_ID = PDS3\r\n^IMAGE_HEADER = ("OTHER.IMG",1)\r\n{pointer_line}\r\nEND\r\n',
    )
    assert DataSetPDS3CassiniISS._image_filename_from_label(label_path) == expected


def test_image_filename_from_label_attached_offset_is_none(tmp_path: Path) -> None:
    label_text = 'PDS_VERSION_ID = PDS3\r\n^IMAGE = 3\r\nEND\r\n'
    label_path = _write_label(tmp_path, 'test.LBL', label_text)
    assert DataSetPDS3CassiniISS._image_filename_from_label(label_path) is None


def test_image_filename_from_label_missing_pointer_is_none(tmp_path: Path) -> None:
    label_path = _write_label(tmp_path, 'test.LBL', 'PDS_VERSION_ID = PDS3\r\nEND\r\n')
    assert DataSetPDS3CassiniISS._image_filename_from_label(label_path) is None


def test_image_url_from_label_case_insensitive_match_keeps_guess(
    ds: DataSetPDS3CassiniISS, tmp_path: Path
) -> None:
    # NH LORRI labels name the .fit file in uppercase while the holdings store it
    # in lowercase; a case-only difference must not rewrite the URL.
    label_path = _write_label(
        tmp_path,
        'lor_0003103486_0x630_sci.lbl',
        '^IMAGE = ("LOR_0003103486_0X630_SCI.FIT", 14)\r\nEND\r\n',
    )
    guess = FCPath('/holdings/data/lor_0003103486_0x630_sci.fit')
    assert ds._image_url_from_label(guess, label_path) is None


def test_image_url_from_label_corrects_differing_name(
    ds: DataSetPDS3CassiniISS, tmp_path: Path
) -> None:
    label_path = _write_label(
        tmp_path,
        'n1454725799_1.lbl',
        '^IMAGE = ("N1454725799_1_FULL.IMG", 4)\r\nEND\r\n',
    )
    guess = FCPath('/holdings/data/n1454725799_1.img')
    resolved = ds._image_url_from_label(guess, label_path)
    assert resolved is not None
    # The pointer name adopts the label filename's (lowercase) case convention.
    assert resolved.as_posix() == '/holdings/data/n1454725799_1_full.img'


def test_image_url_from_label_no_pointer_keeps_guess(
    ds: DataSetPDS3CassiniISS, tmp_path: Path
) -> None:
    label_path = _write_label(tmp_path, 'test.LBL', 'PDS_VERSION_ID = PDS3\r\nEND\r\n')
    guess = FCPath('/holdings/data/TEST.IMG')
    assert ds._image_url_from_label(guess, label_path) is None


# --- ImageFile.resolve_image_url ---


def _make_imagefile(
    image_url: FCPath,
    label_path: Path,
    resolver: Callable[[FCPath, Path], FCPath | None],
) -> ImageFile:
    """Build an ImageFile with the given provisional URL and resolver."""
    return ImageFile(
        image_file_url=image_url,
        label_file_url=FCPath(label_path),
        results_path_stub='stub',
        image_url_resolver=resolver,
    )


def test_resolve_image_url_replaces_url_and_runs_once(tmp_path: Path) -> None:
    label_path = _write_label(tmp_path, 'IMG.LBL', 'END\r\n')
    corrected = FCPath(tmp_path / 'CORRECTED.IMG')
    calls: list[FCPath] = []

    def resolver(image_url: FCPath, _label_path: Path) -> FCPath:
        calls.append(image_url)
        return corrected

    imagefile = _make_imagefile(FCPath(tmp_path / 'GUESS.IMG'), label_path, resolver)

    assert imagefile.resolve_image_url() == corrected
    assert imagefile.image_file_url == corrected
    assert imagefile.resolve_image_url() == corrected
    assert len(calls) == 1


def test_resolve_image_url_none_keeps_guess(tmp_path: Path) -> None:
    label_path = _write_label(tmp_path, 'IMG.LBL', 'END\r\n')
    guess = FCPath(tmp_path / 'GUESS.IMG')

    imagefile = _make_imagefile(guess, label_path, lambda _url, _path: None)

    assert imagefile.resolve_image_url() == guess


def test_image_file_path_uses_resolved_url(tmp_path: Path) -> None:
    label_path = _write_label(tmp_path, 'IMG.LBL', 'END\r\n')
    real_image = tmp_path / 'REAL.IMG'
    real_image.write_bytes(b'image')

    imagefile = _make_imagefile(
        FCPath(tmp_path / 'WRONG.IMG'), label_path, lambda _url, _path: FCPath(real_image)
    )

    assert imagefile.image_file_path == real_image


def test_resolve_image_url_falls_back_when_resolver_raises(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unretrievable label must not abort the image before the pipeline's
    # per-image error boundary: the provisional guess is kept and a warning
    # is logged, and the resolver never runs again.
    label_path = _write_label(tmp_path, 'IMG.LBL', 'END\r\n')
    guess = FCPath(tmp_path / 'GUESS.IMG')
    calls: list[FCPath] = []

    def resolver(image_url: FCPath, _label_path: Path) -> FCPath:
        calls.append(image_url)
        raise FileNotFoundError('label missing from holdings')

    imagefile = _make_imagefile(guess, label_path, resolver)

    assert imagefile.resolve_image_url() == guess
    assert imagefile.resolve_image_url() == guess
    assert len(calls) == 1
    captured = capsys.readouterr()
    assert 'Image URL resolution from label' in captured.out + captured.err


# --- observation epoch from the index row ---


@pytest.mark.parametrize(
    ('dataset_class', 'row', 'expected_iso'),
    [
        # Cassini ISS index times are day-of-year; IMAGE_MID_TIME wins.
        (
            DataSetPDS3CassiniISS,
            {'IMAGE_MID_TIME': '1999-230T02:28:20.833', 'IMAGE_TIME': '1999-230T02:28:22.835'},
            '1999-08-18T02:28:21',
        ),
        # Voyager ISS index times are plain ISO.
        (DataSetPDS3VoyagerISS, {'IMAGE_TIME': '1978-12-11T00:29:23'}, '1978-12-11T00:29:23'),
        # Galileo SSI index times carry a trailing Z.
        (DataSetPDS3GalileoSSI, {'IMAGE_TIME': '1996-06-03T17:05:38.015Z'}, '1996-06-03T17:05:38'),
        # New Horizons LORRI has no IMAGE_TIME; START_TIME is the epoch.
        (
            DataSetPDS3NewHorizonsLORRI,
            {'START_TIME': '2006-02-24T16:12:48.306'},
            '2006-02-24T16:12:48',
        ),
    ],
)
def test_image_et_from_index_row_parses_each_instrument_format(
    dataset_class: type[DataSetPDS3], row: dict[str, Any], expected_iso: str
) -> None:
    """Every instrument's index time format is read without SPICE."""
    image_et = dataset_class.image_et_from_index_row(row)
    assert image_et is not None
    assert datetime_from_image_et(image_et) == expected_iso


def test_image_et_from_index_row_prefers_first_available_column() -> None:
    """A missing preferred column falls through to the next one."""
    row = {'IMAGE_TIME': '1999-230T02:28:20.833'}
    image_et = DataSetPDS3CassiniISS.image_et_from_index_row(row)
    assert image_et is not None
    assert datetime_from_image_et(image_et) == '1999-08-18T02:28:21'


def test_image_et_from_index_row_skips_masked_value() -> None:
    """A null cell (flagged by PdsTable's companion mask) is skipped."""
    row = {
        'IMAGE_MID_TIME': 'UNK',
        'IMAGE_MID_TIME_mask': True,
        'IMAGE_TIME': '1999-230T02:28:20.833',
    }
    image_et = DataSetPDS3CassiniISS.image_et_from_index_row(row)
    assert image_et is not None
    assert datetime_from_image_et(image_et) == '1999-08-18T02:28:21'


def test_image_et_from_index_row_unparsable_value() -> None:
    """An unreadable time yields None rather than raising."""
    assert DataSetPDS3CassiniISS.image_et_from_index_row({'IMAGE_MID_TIME': 'UNK'}) is None


def test_image_et_from_index_row_without_index() -> None:
    """An image not enumerated from an index has no epoch."""
    assert DataSetPDS3CassiniISS.image_et_from_index_row({}) is None


# --- camera from the index row ---


@pytest.mark.parametrize(
    ('dataset_class', 'row', 'expected'),
    [
        (DataSetPDS3CassiniISS, {'INSTRUMENT_ID': 'ISSNA'}, 'NAC'),
        (DataSetPDS3CassiniISS, {'INSTRUMENT_ID': 'ISSWA'}, 'WAC'),
        # Voyager indexes name the camera instead of carrying an id.
        (DataSetPDS3VoyagerISS, {'INSTRUMENT_NAME': 'NARROW ANGLE CAMERA'}, 'NAC'),
        (DataSetPDS3VoyagerISS, {'INSTRUMENT_NAME': 'WIDE ANGLE CAMERA'}, 'WAC'),
        (DataSetPDS3GalileoSSI, {'INSTRUMENT_ID': 'SSI'}, 'SSI'),
        (DataSetPDS3NewHorizonsLORRI, {'INSTRUMENT_ID': 'LORRI'}, 'LORRI'),
    ],
)
def test_camera_from_index_row_maps_each_instrument(
    dataset_class: type[DataSetPDS3], row: dict[str, Any], expected: str
) -> None:
    """Every instrument's index names its camera, without SPICE."""
    assert dataset_class.camera_from_index_row(row) == expected


def test_camera_from_index_row_tolerates_padding_and_case() -> None:
    """Index values are matched stripped and upper-cased."""
    assert DataSetPDS3CassiniISS.camera_from_index_row({'INSTRUMENT_ID': ' issna '}) == 'NAC'


def test_camera_from_index_row_unrecognized_value() -> None:
    """An unknown camera yields None rather than a name nothing else knows."""
    assert DataSetPDS3CassiniISS.camera_from_index_row({'INSTRUMENT_ID': 'ISSXX'}) is None


def test_camera_from_index_row_skips_masked_value() -> None:
    """A null cell is skipped rather than read."""
    row = {'INSTRUMENT_ID': 'ISSNA', 'INSTRUMENT_ID_mask': True}
    assert DataSetPDS3CassiniISS.camera_from_index_row(row) is None


def test_camera_from_index_row_without_index() -> None:
    """An image not enumerated from an index has no camera."""
    assert DataSetPDS3CassiniISS.camera_from_index_row({}) is None
