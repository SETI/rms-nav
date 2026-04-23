"""Tests for ``reproj_cli.paths``."""

from filecache import FCPath

from nav.dataset.dataset import ImageFile
from reproj_cli.paths import mosaic_output_path, per_image_output_path


def _dummy_image_file(stem: str) -> ImageFile:
    url = FCPath(f'/data/{stem}.IMG')
    return ImageFile(
        image_file_url=url,
        label_file_url=FCPath(f'/data/{stem}.LBL'),
        results_path_stub='stub',
    )


def test_per_image_output_includes_subject_with_prefix() -> None:
    img = _dummy_image_file('N1635282917_1_CALIB')
    p = per_image_output_path(
        '/out', 'mimas_2004', img, 'fits', subject_name='MIMAS'
    )
    assert p.name == 'mimas_2004_MIMAS_N1635282917_1_CALIB_reproj.fits'


def test_per_image_output_includes_subject_no_prefix() -> None:
    img = _dummy_image_file('N1')
    p = per_image_output_path(FCPath('/tmp'), '', img, 'npz', subject_name='ENCELADUS')
    assert p.name == 'ENCELADUS_N1_reproj.npz'


def test_mosaic_output_includes_subject() -> None:
    p = mosaic_output_path('/mos', 'fring', 'fits', subject_name='SATURN')
    assert p.name == 'fring_SATURN_mosaic.fits'


def test_mosaic_output_subject_only() -> None:
    p = mosaic_output_path('/mos', '', 'npz', subject_name='SATURN')
    assert p.name == 'SATURN_mosaic.npz'


def test_subject_name_sanitizes_separators() -> None:
    img = _dummy_image_file('x')
    p = per_image_output_path('/o', 'p', img, 'fits', subject_name='foo:bar/baz')
    assert 'foo_bar_baz' in p.name
