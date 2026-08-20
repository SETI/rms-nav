"""What a PDS3 label and an index row say about one image.

Enumerating a selection is one question; what a single image turns out to be is
another, and it is answered without SPICE and without opening the image.  A
label names the file its data really lives in, which is not always the name the
holdings layout predicts; an index row carries the observation's epoch and the
camera that took it, in a spelling that differs per instrument and may be masked
or absent altogether.  Each of those readings has to answer None rather than
raise when the value is not there, because an image that cannot be placed in
time is still an image the enumeration yields.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.dataset_pds3 import DataSetPDS3
from spindoctor.dataset.dataset_pds3_cassini_iss import DataSetPDS3CassiniISS
from spindoctor.dataset.dataset_pds3_galileo_ssi import DataSetPDS3GalileoSSI
from spindoctor.dataset.dataset_pds3_newhorizons_lorri import DataSetPDS3NewHorizonsLORRI
from spindoctor.dataset.dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS
from spindoctor.nav_records.derived import datetime_from_image_et

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


def _image_file_with_epoch(image_et: float | None) -> ImageFile:
    """Build an ``ImageFile`` carrying one enumerated epoch.

    Parameters:
        image_et: The epoch the enumeration read out of the index row.

    Returns:
        The image file, with its epoch settled as construction settles it.
    """
    return ImageFile(
        image_file_url=FCPath('/holdings/N1454725799_1.IMG'),
        label_file_url=FCPath('/holdings/N1454725799_1.LBL'),
        results_path_stub='stub',
        image_et=image_et,
    )


@pytest.mark.parametrize(
    'recorded',
    [float('nan'), float('inf'), float('-inf')],
    ids=['nan', 'inf', 'negative-inf'],
)
def test_an_epoch_no_reader_could_place_reads_as_no_epoch(recorded: float) -> None:
    """An index value that is not a finite number places the image nowhere.

    The index file is not this project's to fix, so such a value is a fact
    about someone else's data rather than a defect here, and refusing it
    would fail a whole enumeration over one row.  Every comparison against a
    NaN is False, so a NaN epoch would fall inside every time range at once
    and an infinite one inside a half-bounded range it can have no business
    in.  None is already what this field says when no epoch is known.

    Parameters:
        recorded: The non-finite epoch the enumeration supplied.
    """
    assert _image_file_with_epoch(recorded).image_et is None


def test_a_readable_epoch_survives_untouched() -> None:
    """The control: a finite epoch is carried through exactly as enumerated."""
    assert _image_file_with_epoch(221309426.8040615).image_et == 221309426.8040615


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
