"""Tests for ``spindoctor.cli.reproj.paths``."""

from filecache import FCPath

from spindoctor.cli.reproj.paths import mosaic_output_path, per_image_output_path
from spindoctor.dataset.dataset import ImageFile


def _dummy_image_file(stem: str) -> ImageFile:
    """Build a minimal :class:`~spindoctor.dataset.dataset.ImageFile` for path tests.

    Args:
        stem: Fake image stem used in ``image_file_url`` / ``label_file_url`` paths.

    Returns:
        An ``ImageFile`` with ``FCPath`` URLs and a fixed ``results_path_stub``.
    """
    url = FCPath(f'/data/{stem}.IMG')
    return ImageFile(
        image_file_url=url,
        label_file_url=FCPath(f'/data/{stem}.LBL'),
        results_path_stub='stub',
    )


def test_per_image_output_includes_subject_with_prefix() -> None:
    """Verify per-image output filename includes subject with prefix."""
    img = _dummy_image_file('N1635282917_1_CALIB')
    p = per_image_output_path('/out', 'mimas_2004', img, fmt='fits', subject_name='MIMAS')
    assert p.name == 'mimas_2004_MIMAS_N1635282917_1_CALIB_reproj.fits'


def test_per_image_output_includes_subject_no_prefix() -> None:
    """Verify per-image output uses subject and stem when prefix is empty."""
    img = _dummy_image_file('N1')
    p = per_image_output_path(FCPath('/tmp'), '', img, fmt='npz', subject_name='ENCELADUS')
    assert p.name == 'ENCELADUS_N1_reproj.npz'


def test_mosaic_output_includes_subject() -> None:
    """Verify mosaic filename includes prefix and subject."""
    p = mosaic_output_path('/mos', 'fring', 'fits', subject_name='SATURN')
    assert p.name == 'fring_SATURN_mosaic.fits'


def test_mosaic_output_subject_only() -> None:
    """Verify mosaic filename is subject-only when prefix is empty."""
    p = mosaic_output_path('/mos', '', 'npz', subject_name='SATURN')
    assert p.name == 'SATURN_mosaic.npz'


def test_subject_name_sanitizes_separators() -> None:
    """Verify subject colons and slashes become underscores in the output basename."""
    img = _dummy_image_file('x')
    p = per_image_output_path('/o', 'p', img, fmt='fits', subject_name='foo:bar/baz')
    assert p.name == 'p_foo_bar_baz_x_reproj.fits'
