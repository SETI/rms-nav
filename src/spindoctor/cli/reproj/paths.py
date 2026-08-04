"""Output-path helpers for sd_mosaic.

Conventions
-----------
Per-image reprojection file:
    <output_dir>/<prefix>_<subject>_<image_stem>_reproj.<ext>

``subject`` is the mosaic body/planet name (e.g. ``MIMAS``, ``SATURN``), with
spaces, colons, and slashes replaced by underscores for safe filenames.

Per-image reprojection log (``sd_mosaic`` pass 1), built by
:func:`spindoctor.config.logging_config.image_log_path` rather than here:
    <log_root>/reproj/<subject>/<results_path_stub>_<timestamp>.log

Final mosaic file:
    <output_dir>/<prefix>_<subject>_mosaic.<ext>

If ``prefix`` is empty the leading underscore is omitted before ``subject``, e.g.
``MIMAS_<image_stem>_reproj.fits`` and ``SATURN_mosaic.fits``.
"""

import re

from filecache import FCPath

from spindoctor.dataset.dataset import ImageFile

_ALLOWED_OUTPUT_FMTS: frozenset[str] = frozenset({'fits', 'npz'})
_PREFIX_INVALID = re.compile(r'[\x00-\x1f\x7f/\\\\]')


def _subject_filename_segment(subject_name: str) -> str:
    """Normalize a label for use as a single path component in output filenames.

    Parameters:
        subject_name: Raw body/planet (or similar) label from the CLI or mosaic.

    Returns:
        A filename-safe string: non-alphanumeric characters (except ``.``, ``-``,
        ``_``) and path/meta characters are replaced with underscores, runs collapsed,
        and leading/trailing underscores stripped. Returns ``'unknown'`` when the
        result would otherwise be empty (e.g. ``_subject_filename_segment('   ')``).
    """
    s = subject_name.strip()
    out = ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in s)
    while '__' in out:
        out = out.replace('__', '_')
    out = out.strip('_')
    return out if out else 'unknown'


def _validate_output_prefix(prefix: str) -> None:
    if not isinstance(prefix, str):
        raise TypeError(f'prefix must be str, got {type(prefix).__name__}')
    if '\x00' in prefix:
        raise ValueError('prefix must not contain null bytes')
    if _PREFIX_INVALID.search(prefix):
        raise ValueError(f'prefix must not contain path separators; got {prefix!r}')


def _validate_output_fmt(fmt: str) -> str:
    if not isinstance(fmt, str):
        raise TypeError(f'fmt must be str, got {type(fmt).__name__}')
    ext = fmt.lower()
    if ext not in _ALLOWED_OUTPUT_FMTS:
        raise ValueError(f'fmt must be one of {sorted(_ALLOWED_OUTPUT_FMTS)}, got {fmt!r}')
    return ext


def _ensure_output_under_dir(output_dir: str | FCPath, filename: str) -> FCPath:
    if not filename or filename in {'.', '..'}:
        raise ValueError(f'invalid output filename: {filename!r}')
    if '/' in filename or '\\' in filename:
        raise ValueError(f'filename must not contain path separators: {filename!r}')
    base = FCPath(output_dir).expanduser().resolve()
    candidate = (base / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f'Refusing output path outside output_dir={output_dir!r} for filename={filename!r} '
            f'(resolves to {candidate!r})'
        ) from exc
    return candidate


def per_image_output_path(
    output_dir: str | FCPath,
    prefix: str,
    image_file: ImageFile,
    *,
    fmt: str,
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

    Raises:
        TypeError: If ``prefix`` or ``fmt`` is not a :class:`str`.
        ValueError: If ``fmt`` or ``prefix`` is invalid or the resolved path would
            escape ``output_dir``.
    """
    stem = _subject_filename_segment(FCPath(image_file.image_file_url).stem)
    sub = _subject_filename_segment(subject_name)
    _validate_output_prefix(prefix)
    ext = _validate_output_fmt(fmt)
    if prefix:
        filename = f'{prefix}_{sub}_{stem}_reproj.{ext}'
    else:
        filename = f'{sub}_{stem}_reproj.{ext}'
    return _ensure_output_under_dir(output_dir, filename)


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

    Raises:
        TypeError: If ``prefix`` or ``fmt`` is not a :class:`str`.
        ValueError: If ``fmt`` or ``prefix`` is invalid or the resolved path would
            escape ``output_dir``.
    """
    sub = _subject_filename_segment(subject_name)
    _validate_output_prefix(prefix)
    ext = _validate_output_fmt(fmt)
    if prefix:
        filename = f'{prefix}_{sub}_mosaic.{ext}'
    else:
        filename = f'{sub}_mosaic.{ext}'
    return _ensure_output_under_dir(output_dir, filename)
