"""Serialization helpers for spindoctor.reproj result dataclasses.

Provides format-agnostic save/load utilities used by BodyReprojResult,
BodyMosaicData, RingReprojResult, and RingMosaicData.

All path arguments accept ``str``, :class:`pathlib.Path`, or :class:`filecache.FCPath`.
Each path is normalized to an :class:`~filecache.FCPath` at entry. Writes use
:meth:`~filecache.FCPath.get_local_path`, then NumPy or Astropy to write the
file, then :meth:`~filecache.FCPath.upload`. Reads resolve a local cache path
via :meth:`~filecache.FCPath.get_local_path` before loading.

Supported formats
-----------------
npz
    NumPy's native compressed (``np.savez_compressed``) or uncompressed
    (``np.savez``) archive. MaskedArrays are stored as two entries:
    ``<name>__data`` (the underlying array) and ``<name>__mask`` (bool).
    Scalars and strings are stored as 0-D unicode or numeric arrays.
    Tuples of length 2 of numeric types are stored as 1-D length-2 arrays.
    Tuples of strings (including empty) are stored as a 1-D Unicode string array.
    ``image_dtype`` / ``metadata_dtype`` are stored as 0-D unicode arrays
    containing the dtype ``str`` attribute (e.g. ``'<f8'``).

fits
    FITS via ``astropy.io.fits``. Scalar metadata (strings, numbers,
    dtype names, 2-tuples of scalars encoded as paired header keys) are stored in the
    PrimaryHDU header. Each array (and its mask when applicable) occupies
    a separate ImageHDU named ``<FIELDNAME>`` and ``<FIELDNAME>_MASK``.
    Tuple-of-string payloads use a 1-D ``uint8`` ImageHDU (UTF-8 with a ``NUL``
    terminator after every entry, so the entry count is unambiguous).

Schema evolution
----------------
Every file includes a ``__kind__`` key/header card identifying the
dataclass (e.g. ``'BodyMosaicData'``) and a ``__version__`` integer
(currently ``1``). ``load()`` raises ``ValueError`` when ``__kind__``
does not match the expected kind. Future schema changes should bump
``__version__`` and handle older versions inside the appropriate
``load()`` implementation.
"""

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from filecache import FCPath

from spindoctor.config import IMAGE_LOGGER
from spindoctor.reproj.ring_orbit_model import RingOrbitModel
from spindoctor.support.types import PathLike

_CURRENT_VERSION = 1
_logger = IMAGE_LOGGER


def _as_fcpath(path: PathLike) -> FCPath:
    """Normalize a path-like value to :class:`~filecache.FCPath`."""
    if isinstance(path, FCPath):
        return path
    return FCPath(path)


def tuple_of_strings_field(value: Any) -> tuple[str, ...]:
    """Normalize a loaded ``contributing_image_names`` value to ``tuple[str, ...]``."""
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        flat = np.ravel(value)
        if flat.size == 0:
            return ()
        # FITS: UTF-8 bytes written by ``_fits_encode_value`` for tuple-of-strings
        # fields; every entry carries a trailing NUL terminator.
        if flat.dtype == np.uint8:
            raw = flat.tobytes()
            if not raw:
                return ()
            if not raw.endswith(b'\0'):
                raise ValueError(
                    'Malformed tuple-of-strings FITS payload: missing entry terminator'
                )
            try:
                return tuple(p.decode('utf-8') for p in raw.split(b'\0')[:-1])
            except UnicodeDecodeError as exc:
                _logger.error(
                    'Invalid UTF-8 in FITS tuple-of-strings buffer (%d bytes)',
                    len(raw),
                )
                raise ValueError('Invalid UTF-8 in tuple-of-strings FITS payload') from exc
        return tuple(str(x) for x in flat.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value)
    return (str(value),)


# ---------------------------------------------------------------------------
# Format inference
# ---------------------------------------------------------------------------


def _path_starts_with_zip_magic(fcpath: FCPath) -> bool:
    """True if the local file begins with ZIP magic (NumPy ``.npz`` archives)."""
    try:
        local = cast(Path, fcpath.get_local_path())
        with local.open('rb') as fb:
            return fb.read(4) == b'PK\x03\x04'
    except OSError:
        return False


def infer_format(
    path: PathLike,
    format_: str | None,
) -> str:
    """Infer the serialization format from the file extension or explicit override.

    Parameters:
        path: File path (``str``, ``Path``, or ``FCPath``); used for extension-based
            inference when format_ is None.
        format_: Explicit format string ('npz' or 'fits'). When not None, this
            value is returned verbatim after validation.

    Returns:
        'npz' or 'fits'.

    Raises:
        ValueError: If the format cannot be inferred or is not supported.
    """
    if format_ is not None:
        if format_ not in ('npz', 'fits'):
            raise ValueError(f"format must be 'npz' or 'fits', got {format_!r}")
        return format_

    fcpath = _as_fcpath(path)
    suffix = fcpath.suffix.lower()
    if suffix in ('.npz',):
        return 'npz'
    if suffix in ('.fits', '.fit'):
        if _path_starts_with_zip_magic(fcpath):
            return 'npz'
        return 'fits'
    if suffix in ('.fz',):
        return 'fits'

    # Try removing a second extension (e.g. .fits.gz → .fits)
    stem = fcpath.stem
    second_suffix = Path(stem).suffix.lower()
    if second_suffix in ('.fits', '.fit'):
        if _path_starts_with_zip_magic(fcpath):
            return 'npz'
        return 'fits'

    raise ValueError(
        f'Cannot infer format from path {fcpath!r}. '
        "Use format_='npz' or format_='fits' to specify explicitly."
    )


# ---------------------------------------------------------------------------
# RingOrbitModel serialization helpers
# ---------------------------------------------------------------------------

_ORBIT_MODEL_FROM_DICT_KEYS: tuple[str, ...] = (
    'name',
    'a',
    'e',
    'w0',
    'dw',
    'mean_motion',
    'epoch_utc',
)


def orbit_model_to_dict(om: RingOrbitModel | None) -> dict[str, Any]:
    """Serialize a RingOrbitModel to a plain dict of primitives.

    Parameters:
        om: The orbit model to serialize, or None.

    Returns:
        Dict with keys ``name``, ``a``, ``e``, ``w0``, ``dw``,
        ``mean_motion``, ``epoch_utc``, and ``is_none`` (bool flag).
    """
    if om is None:
        return {'is_none': True}
    return {
        'is_none': False,
        'name': om.name,
        'a': om.a,
        'e': om.e,
        'w0': om.w0,
        'dw': om.dw,
        'mean_motion': om.mean_motion,
        'epoch_utc': om.epoch_utc,
    }


def orbit_model_from_dict(d: dict[str, Any]) -> RingOrbitModel | None:
    """Deserialize a RingOrbitModel from a plain dict.

    Parameters:
        d: Dict produced by ``orbit_model_to_dict``.

    Returns:
        RingOrbitModel instance, or None if the original was None.

    Raises:
        ValueError: If ``is_none`` is not true but required keys are missing
            (see :func:`orbit_model_to_dict` for the expected schema).
    """
    if d.get('is_none') is True:
        return None
    missing = [k for k in _ORBIT_MODEL_FROM_DICT_KEYS if k not in d]
    if missing:
        raise ValueError(
            'orbit_model dict is missing required key(s) '
            f'{missing!r}; when is_none is not True, keys must match those emitted by '
            'orbit_model_to_dict() (name, a, e, w0, dw, mean_motion, epoch_utc).'
        )
    return RingOrbitModel(
        name=str(d['name']),
        a=float(d['a']),
        e=float(d['e']),
        w0=float(d['w0']),
        dw=float(d['dw']),
        mean_motion=float(d['mean_motion']),
        epoch_utc=str(d['epoch_utc']),
    )


# ---------------------------------------------------------------------------
# npz helpers
# ---------------------------------------------------------------------------


def save_npz(
    path: PathLike,
    kind: str,
    version: int,
    payload: dict[str, Any],
    *,
    compress: bool,
) -> None:
    """Save a payload dict to an npz archive.

    MaskedArrays are split into ``<name>__data`` and ``<name>__mask``
    entries. Tuples of length 2 of numeric types are stored as 1-D length-2 arrays.
    Tuples of strings are stored as a 1-D Unicode string array. Strings
    and dtype names are stored as 0-D unicode arrays. Dicts (e.g. from
    ``orbit_model_to_dict``) are flattened with ``<name>__<key>`` entries.

    Parameters:
        path: Output path (``str``, ``Path``, or ``FCPath``).
        kind: Dataclass kind string (e.g. 'BodyMosaicData').
        version: Schema version integer.
        payload: Mapping of field name → value. Supported types: ndarray,
            MaskedArray, numeric 2-tuples, tuple[str, ...] (including empty),
            str, float, int, bool, np.dtype, dict, None.
        compress: If True use ``np.savez_compressed``; otherwise
            ``np.savez``.
    """
    fcpath = _as_fcpath(path)
    local_path = cast(Path, fcpath.get_local_path())

    arrays: dict[str, np.ndarray] = {}
    arrays['__kind__'] = np.array(kind)
    arrays['__version__'] = np.array(version)

    for name, value in payload.items():
        _npz_encode_value(arrays, name, value)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(local_path, **arrays)  # type: ignore[arg-type]
    fcpath.upload()


def _npz_encode_value(arrays: dict[str, np.ndarray], name: str, value: Any) -> None:
    """Encode a single payload value into the arrays dict for npz storage.

    Internal helper called only within this module. Not intended for import.

    Parameters:
        arrays: Target dict being built for np.savez / np.savez_compressed.
        name: Key name for this value.
        value: Value to encode.
    """
    if value is None:
        arrays[name + '__none'] = np.array(True)
    elif isinstance(value, ma.MaskedArray):
        arrays[name + '__data'] = np.asarray(value.data)
        arrays[name + '__mask'] = np.asarray(ma.getmaskarray(value))
    elif isinstance(value, np.ndarray):
        arrays[name] = value
    elif isinstance(value, np.dtype):
        arrays[name] = np.array(value.str)
    elif isinstance(value, tuple):
        if len(value) > 0 and isinstance(value[0], str):
            arrays[name] = np.asarray(value, dtype=np.str_)
        elif len(value) == 0:
            arrays[name] = np.array([], dtype=np.str_)
        else:
            arrays[name] = np.array(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _npz_encode_value(arrays, f'{name}__{k}', v)
    elif isinstance(value, (bool, int, float, str)):
        arrays[name] = np.array(value)
    else:
        raise TypeError(
            f'Unsupported type for npz serialization: {type(value).__name__!r} for field {name!r}'
        )


def load_npz(
    path: PathLike,
    expected_kind: str,
) -> dict[str, Any]:
    """Load an npz archive and reassemble MaskedArrays.

    Parameters:
        path: Input path (``str``, ``Path``, or ``FCPath``).
        expected_kind: The expected dataclass kind string. Raises
            ``ValueError`` if the file's ``__kind__`` does not match.

    Returns:
        Dict with MaskedArrays reassembled from ``__data`` / ``__mask``
        pairs. Other arrays are returned as plain ndarrays or scalars
        unwrapped from 0-D arrays.

    Raises:
        ValueError: If sentinels are missing, ``__kind__`` mismatches, or the
            file is missing other required keys.
    """
    fcpath = _as_fcpath(path)
    local_path = cast(Path, fcpath.get_local_path())
    with np.load(local_path, allow_pickle=False) as raw:
        if '__kind__' not in raw:
            raise ValueError('Missing file sentinel __kind__ - file is truncated or wrong format')
        if '__version__' not in raw:
            raise ValueError(
                'Missing file sentinel __version__ - file is truncated or wrong format'
            )
        kind = str(raw['__kind__'])
        if kind != expected_kind:
            raise ValueError(f'Kind mismatch: file contains {kind!r}, expected {expected_kind!r}')

        version = int(raw['__version__'])
        _ = version  # retained for future schema migration

        # Collect all keys; reconstruct MaskedArrays and None sentinels
        result: dict[str, Any] = {}
        keys = set(raw.files)
        keys.discard('__kind__')
        keys.discard('__version__')

        # First pass: identify MA pairs and None sentinels
        data_keys = {k[:-6] for k in keys if k.endswith('__data')}
        mask_keys = {k[:-6] for k in keys if k.endswith('__mask')}
        none_keys = {k[:-6] for k in keys if k.endswith('__none')}
        ma_keys = data_keys & mask_keys

        orphan_keys = data_keys.symmetric_difference(mask_keys)
        if orphan_keys:
            raise ValueError(
                f'Unmatched "__data"/"__mask" sentinel pairs in npz file. '
                f'Orphaned base names: {sorted(orphan_keys)!r}. '
                'Each "<name>__data" entry must have a matching "<name>__mask" and vice versa.'
            )

        handled: set[str] = set()
        for base in ma_keys:
            result[base] = ma.MaskedArray(
                np.array(raw[base + '__data']),
                mask=np.array(raw[base + '__mask']),
            )
            handled.add(base + '__data')
            handled.add(base + '__mask')

        for base in none_keys:
            result[base] = None
            handled.add(base + '__none')

        for k in keys - handled:
            arr = raw[k]
            if arr.ndim == 0:
                # Unwrap scalars: unicode → str, others → Python scalar
                v = arr.item()
                result[k] = v
            else:
                result[k] = np.array(arr)

        return result


# ---------------------------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------------------------


def _fits_load_key_to_payload(key: str) -> str:
    """Normalize a FITS header or HDU name to the lowercase keys used in npz payloads."""
    k = key.strip()
    if k.upper().startswith('HIERARCH '):
        k = k[9:].strip()
    return k.lower()


def _fits_image_hdu_data(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` suitable for :class:`astropy.io.fits.ImageHDU` (no bool dtype).

    Astropy's FITS stack has no BITPIX for numpy bool; store booleans as ``uint8``
    (0/1). Callers that reload should cast back to ``bool_`` when needed.
    """
    if arr.dtype.kind == 'b':
        return arr.astype(np.uint8, copy=False)
    return arr


def save_fits(
    path: PathLike,
    kind: str,
    version: int,
    payload: dict[str, Any],
) -> None:
    """Save a payload dict to a FITS file.

    Scalar metadata (strings, numbers, 2-tuples of scalars as paired header cards,
    dtype names) are stored in the PrimaryHDU header. Each array and its mask
    (for MaskedArrays) are stored as separate ImageHDUs with EXTNAME set to
    ``<FIELDNAME>`` and ``<FIELDNAME>_MASK`` respectively. Tuple-of-string fields
    use a 1-D string ImageHDU.

    Parameters:
        path: Output path (``str``, ``Path``, or ``FCPath``).
        kind: Dataclass kind string (e.g. 'BodyMosaicData').
        version: Schema version integer.
        payload: Mapping of field name → value (same supported types as :func:`save_npz`,
            with tuple-of-strings encoded as an ImageHDU).
    """
    fcpath = _as_fcpath(path)
    local_path = cast(Path, fcpath.get_local_path())

    # Long / mixed-case scalar keys become FITS HIERARCH cards; astropy warns by
    # default even though the output is standards-conformant for modern FITS.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        primary_hdr = fits.Header()
        primary_hdr['KIND'] = kind
        primary_hdr['VERSION'] = version

        hdus: list[fits.ImageHDU] = []

        for name, value in payload.items():
            _fits_encode_value(primary_hdr, hdus, name.upper(), value)

        hdul = fits.HDUList([fits.PrimaryHDU(header=primary_hdr), *hdus])
        hdul.writeto(local_path, overwrite=True)
    fcpath.upload()


def _fits_encode_header_scalar(hdr: Any, name: str, value: Any) -> None:
    """Encode one scalar into a FITS header, handling non-finite floats.

    FITS headers cannot hold NaN/inf values; a non-finite float is written
    under ``<NAME>_NONF`` as its string form and reconstructed by
    :func:`load_fits`.

    Parameters:
        hdr: The PrimaryHDU header.
        name: Header keyword (already uppercased/suffixed by the caller).
        value: The bool, int, float, or str value to write.
    """
    if isinstance(value, float) and not np.isfinite(value):
        hdr[name + '_NONF'] = repr(value)
    else:
        hdr[name] = bool(value) if isinstance(value, bool) else value


def _fits_encode_value(
    hdr: Any,
    hdus: list[Any],
    name: str,
    value: Any,
) -> None:
    """Encode a single payload value into a FITS header or HDU list.

    Internal helper called only within this module. Not intended for import.

    Parameters:
        hdr: The PrimaryHDU header (for scalar values).
        hdus: List of ImageHDUs being built (for array values).
        name: Field name in UPPER_CASE.
        value: Value to encode.
    """
    if value is None:
        hdr[name + '_NONE'] = True
    elif isinstance(value, ma.MaskedArray):
        hdus.append(fits.ImageHDU(data=_fits_image_hdu_data(np.asarray(value.data)), name=name))
        hdus.append(
            fits.ImageHDU(
                data=ma.getmaskarray(value).astype(np.uint8),
                name=name + '_MASK',
            )
        )
    elif isinstance(value, np.ndarray):
        hdus.append(fits.ImageHDU(data=_fits_image_hdu_data(value), name=name))
    elif isinstance(value, np.dtype):
        hdr[name] = value.str
    elif isinstance(value, tuple) and (len(value) == 0 or isinstance(value[0], str)):
        # ImageHDU cannot represent NumPy Unicode dtypes; store UTF-8 bytes
        # (uint8) with a NUL terminator after EVERY entry, so N entries carry
        # N terminators and ``('',)`` stays distinguishable from ``()``.
        raw = ''.join(str(s) + '\0' for s in value).encode('utf-8')
        raw_arr = np.frombuffer(raw, dtype=np.uint8).copy()
        hdus.append(fits.ImageHDU(data=_fits_image_hdu_data(raw_arr), name=name))
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(
                f'Field {name!r}: only 2-tuples are supported for FITS header encoding, '
                f'got tuple of length {len(value)}'
            )
        _fits_encode_header_scalar(hdr, name + '_0', value[0])
        _fits_encode_header_scalar(hdr, name + '_1', value[1])
    elif isinstance(value, dict):
        for k, v in value.items():
            _fits_encode_value(hdr, hdus, f'{name}__{k.upper()}', v)
    elif isinstance(value, (bool, int, float)):
        _fits_encode_header_scalar(hdr, name, value)
    elif isinstance(value, str):
        hdr[name] = value
    else:
        raise TypeError(
            f'Unsupported type for FITS header serialization: {type(value).__name__!r} '
            f'for field {name!r}'
        )


def load_fits(
    path: PathLike,
    expected_kind: str,
) -> dict[str, Any]:
    """Load a FITS file and reassemble MaskedArrays.

    Parameters:
        path: Input path (``str``, ``Path``, or ``FCPath``).
        expected_kind: The expected dataclass kind string.

    Returns:
        Dict with MaskedArrays, plain ndarrays, and scalar values
        reconstructed from the FITS file. Keys are lowercased to match
        :func:`load_npz` / dataclass ``load()`` conventions. Header ``*_NONE``
        sentinels for top-level ``None`` values do not collide with flattened
        nested keys such as ``orbit_model__is_none``.

    Raises:
        ValueError: If required sentinels are missing or the file's ``KIND``
            header card does not match ``expected_kind``.
    """
    fcpath = _as_fcpath(path)
    local_path = cast(Path, fcpath.get_local_path())
    with fits.open(local_path) as hdul:
        primary_hdr = hdul[0].header

        if 'KIND' not in primary_hdr:
            raise ValueError('Missing file sentinel KIND - file is truncated or wrong format')
        if 'VERSION' not in primary_hdr:
            raise ValueError('Missing file sentinel VERSION - file is truncated or wrong format')
        kind = str(primary_hdr['KIND'])
        if kind != expected_kind:
            raise ValueError(f'Kind mismatch: file contains {kind!r}, expected {expected_kind!r}')

        version = int(primary_hdr['VERSION'])
        _ = version  # retained for future schema migration

        # Collect ImageHDUs by EXTNAME (avoid clobbering when EXTNAME is missing/blank).
        hdu_map: dict[str, Any] = {}
        extname_first_index: dict[str, int] = {}
        for idx, hdu in enumerate(hdul[1:], start=1):
            extname = (hdu.name or '').strip()
            if not extname:
                extname = str(hdu.header.get('EXTNAME', '') or '').strip()
            if not extname:
                extname = f'__UNNAMED{idx}'
            if extname in hdu_map:
                raise ValueError(
                    f'Duplicate FITS EXTNAME {extname!r}: HDU indices '
                    f'{extname_first_index[extname]} and {idx}'
                )
            extname_first_index[extname] = idx
            hdu_map[extname] = hdu.data

        result: dict[str, Any] = {}

        # Scalars from primary header (iterate keywords; more reliable than ``.cards``
        # for COMMENT/HIERARCH edge cases).
        skip_cards = {
            'SIMPLE',
            'BITPIX',
            'NAXIS',
            'NAXIS1',
            'NAXIS2',
            'NAXIS3',
            'NAXIS4',
            'EXTEND',
            'XTENSION',
            'PCOUNT',
            'GCOUNT',
            'EXTNAME',
            'EXTVER',
            'LONGSTRN',
            'COMMENT',
            'HISTORY',
            'KIND',
            'VERSION',
            'END',
        }
        for kw in primary_hdr:
            if kw in skip_cards:
                continue
            result[kw] = primary_hdr[kw]

        # Reconstruct MaskedArrays from HDU pairs
        orphan_masks = [
            name[:-5] for name in hdu_map if name.endswith('_MASK') and name[:-5] not in hdu_map
        ]
        if orphan_masks:
            raise ValueError(
                f'Orphaned "_MASK" HDUs in FITS file with no matching base HDU. '
                f'Missing base names: {sorted(orphan_masks)!r}. '
                'Each "<NAME>_MASK" HDU must have a corresponding "<NAME>" HDU.'
            )
        for name, _data in list(hdu_map.items()):
            if name.endswith('_MASK'):
                base = name[:-5]
                if base in hdu_map:
                    result[base] = ma.MaskedArray(hdu_map[base], mask=hdu_map[name].astype(bool))
                    hdu_map.pop(base, None)
                    hdu_map.pop(name, None)

        for name, data in hdu_map.items():
            if name not in result:
                result[name] = data

    # Reconstruct non-finite float header values first (written as
    # ``<NAME>_NONF`` string cards because FITS headers cannot hold NaN/inf),
    # so the tuple reconstruction below sees the plain ``_0``/``_1`` keys.
    for k in [k for k in result if k.endswith('_NONF')]:
        result[k[:-5]] = float(result.pop(k))

    # Post-process header scalars: reconstruct tuples and None sentinels
    keys_to_delete: list[str] = []
    keys_to_add: dict[str, Any] = {}
    for k in list(result.keys()):
        if k.endswith('_NONE'):
            base = k[:-5]
            # ``__`` in ``base`` marks flattened nested keys, e.g. ``ORBIT_MODEL__IS_NONE``
            # from ``orbit_model.is_none``; skip those. True top-level None sentinels look
            # like ``FIELD_NONE`` (``base`` has no ``__``) and map the field to Python ``None``.
            if '__' in base:
                continue
            keys_to_add[base] = None
            keys_to_delete.append(k)
        elif k.endswith('_0'):
            base = k[:-2]
            k1 = base + '_1'
            if k1 in result:
                keys_to_add[base] = (result[k], result[k1])
                keys_to_delete.append(k)
                keys_to_delete.append(k1)

    for k in keys_to_delete:
        result.pop(k, None)
    result.update(keys_to_add)

    return {_fits_load_key_to_payload(k): v for k, v in result.items()}


# ---------------------------------------------------------------------------
# Dtype verification helper
# ---------------------------------------------------------------------------


def _dtype_matches_declared(actual: np.dtype, declared: np.dtype) -> bool:
    """True if ``actual`` matches ``declared`` up to byte order (FITS is often big-endian)."""
    if actual == declared:
        return True
    a = np.dtype(actual)
    d = np.dtype(declared)
    if a.itemsize != d.itemsize:
        return False
    # Same width: kinds must match (endian swap ok; reject int32 vs uint32, float vs int, ...).
    return a.kind == d.kind


def verify_dtype(
    arrays: dict[str, ma.MaskedArray | np.ndarray],
    *,
    image_dtype: np.dtype,
    metadata_dtype: np.dtype,
    image_fields: list[str],
    metadata_fields: list[str],
    float64_fields: list[str] | None = None,
) -> None:
    """Verify that loaded arrays have the expected dtypes.

    Parameters:
        arrays: Dict of field name → array.
        image_dtype: Expected dtype for image arrays.
        metadata_dtype: Expected dtype for metadata arrays.
        image_fields: List of field names that should have ``image_dtype``.
        metadata_fields: List of field names that should have
            ``metadata_dtype``.
        float64_fields: Optional list of field names that must always be
            ``float64`` regardless of ``metadata_dtype`` (e.g. ``time``).

    Raises:
        ValueError: If any array dtype does not match its expected dtype,
            or if ``image_number`` is not ``uint16``, or if any mask is
            not ``bool_``.
    """
    for fname in image_fields:
        arr = arrays.get(fname)
        if arr is None:
            continue
        actual = arr.dtype
        if not _dtype_matches_declared(actual, image_dtype):
            raise ValueError(
                f'image_dtype mismatch for field {fname!r}: '
                f'file declares {image_dtype} but array is {actual}'
            )

    for fname in metadata_fields:
        arr = arrays.get(fname)
        if arr is None:
            continue
        actual = arr.dtype
        if not _dtype_matches_declared(actual, metadata_dtype):
            raise ValueError(
                f'metadata_dtype mismatch for field {fname!r}: '
                f'file declares {metadata_dtype} but array is {actual}'
            )

    if float64_fields:
        for fname in float64_fields:
            arr = arrays.get(fname)
            if arr is None:
                continue
            actual = arr.dtype
            if not (actual.kind == 'f' and actual.itemsize == 8):
                raise ValueError(
                    f'dtype mismatch for field {fname!r}: must be float64, got {actual}'
                )

    imgnum = arrays.get('image_number')
    if imgnum is not None:
        actual = imgnum.dtype
        if not (actual.kind == 'u' and actual.itemsize == 2):
            raise ValueError(f'image_number must be uint16, got {actual}')

    for fname, arr in arrays.items():
        if not isinstance(arr, ma.MaskedArray):
            continue
        mask = ma.getmaskarray(arr)
        if mask.dtype != np.dtype(np.bool_):
            raise ValueError(f'Mask for field {fname!r} must be bool_, got {mask.dtype}')
