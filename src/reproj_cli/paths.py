"""Output-path helpers for nav_mosaic.

Conventions
-----------
Per-image reprojection file:
    <output_dir>/<prefix>_<subject>_<image_stem>_reproj.<ext>

``subject`` is the mosaic body/planet name (e.g. ``MIMAS``, ``SATURN``), with
spaces, colons, and slashes replaced by underscores for safe filenames.

Per-image reprojection log (``nav_mosaic`` pass 1):
    <output_dir>/logs/<results_path_stub>_<timestamp>.log

Final mosaic file:
    <output_dir>/<prefix>_<subject>_mosaic.<ext>

If ``prefix`` is empty the leading underscore is omitted before ``subject``, e.g.
``MIMAS_<image_stem>_reproj.fits`` and ``SATURN_mosaic.fits``.
"""

from filecache import FCPath

from nav.dataset.dataset import ImageFile


def _subject_filename_segment(subject_name: str) -> str:
    """Return ``subject_name`` normalized for use between underscores in filenames."""
    s = (
        subject_name.strip()
        .replace(' ', '_')
        .replace(':', '_')
        .replace('/', '_')
        .replace('\\', '_')
    )
    return s if s else 'unknown'


def per_image_output_path(
    output_dir: str | FCPath,
    prefix: str,
    image_file: ImageFile,
    fmt: str,
    *,
    subject_name: str,
) -> FCPath:
    """Return the output path for a single reprojected image.

    Parameters:
        output_dir: Directory that will contain the output files.
        prefix: Optional filename prefix (may be empty).
        image_file: The source image file object; its URL stem is used.
        fmt: File format extension, either ``'fits'`` or ``'npz'``.
        subject_name: Body or planet name from the mosaic (e.g. ``MIMAS``, ``SATURN``).

    Returns:
        An :class:`filecache.FCPath` pointing to the reprojection file.
    """
    stem = FCPath(image_file.image_file_url).stem
    sub = _subject_filename_segment(subject_name)
    if prefix:
        filename = f'{prefix}_{sub}_{stem}_reproj.{fmt}'
    else:
        filename = f'{sub}_{stem}_reproj.{fmt}'
    return FCPath(output_dir) / filename


def mosaic_output_path(
    output_dir: str | FCPath,
    prefix: str,
    fmt: str,
    *,
    subject_name: str,
) -> FCPath:
    """Return the output path for the final mosaic file.

    Parameters:
        output_dir: Directory that will contain the output files.
        prefix: Optional filename prefix (may be empty).
        fmt: File format extension, either ``'fits'`` or ``'npz'``.
        subject_name: Body or planet name from the mosaic (e.g. ``MIMAS``, ``SATURN``).

    Returns:
        An :class:`filecache.FCPath` pointing to the mosaic file.
    """
    sub = _subject_filename_segment(subject_name)
    if prefix:
        filename = f'{prefix}_{sub}_mosaic.{fmt}'
    else:
        filename = f'{sub}_mosaic.{fmt}'
    return FCPath(output_dir) / filename
