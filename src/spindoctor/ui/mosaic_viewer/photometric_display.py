"""Photometric display modes for ring/body mosaic viewer windows.

Converts stored mosaic/reprojection pixels to intrinsic brightness or applies an
alternate viewing model for on-screen display. Core correction models live in
:mod:`spindoctor.reproj.photometric_model`.
"""

import math

import numpy as np
import numpy.ma as ma

from spindoctor.reproj.photometric_model import photometric_model_from_name

_KNOWN_PHOTOMETRY_MODES: frozenset[str] = frozenset(
    {
        'as_saved',
        'intrinsic',
        'lambert',
        'lommel_seeliger',
        'lommelseeliger',
        'minnaert',
    }
)


def _normalize_photometry_mode(mode: str) -> str:
    return str(mode).strip().lower().replace('-', '_').replace(' ', '_')


def _validate_photometry_mode(mode: str) -> str:
    n = _normalize_photometry_mode(mode)
    if n not in _KNOWN_PHOTOMETRY_MODES:
        raise ValueError(
            f'Unknown photometric display mode {mode!r}; expected one of '
            f'{sorted(_KNOWN_PHOTOMETRY_MODES)} (see photometric_model_from_name).'
        )
    return n


def compute_body_display_image(
    *,
    mode: str,
    image_ma: ma.MaskedArray,
    photometric_model_name: str | None,
    phase_deg: ma.MaskedArray,
    emission_deg: ma.MaskedArray,
    incidence_deg: ma.MaskedArray,
) -> ma.MaskedArray:
    """Return image data for the mosaic viewer for a photometric display mode.

    Parameters:
        mode: ``'as_saved'``, ``'intrinsic'``, or a supported model name
            (``lambert``, ``lommel_seeliger`` / ``lommelseeliger``, ``minnaert``).
        image_ma: Stored image (possibly already corrected when saved).
        photometric_model_name: Model name stored with the file, if any.
        phase_deg, emission_deg, incidence_deg: Per-pixel angles in **degrees**
            (same shape as ``image_ma``).

    Returns:
        Masked array, same shape as ``image_ma``. ``as_saved`` returns ``image_ma``
        unchanged. Other modes return float64 pixel values under the original mask.
    """
    mode_l = _validate_photometry_mode(mode)
    if mode_l == 'as_saved':
        return image_ma

    mask = ma.getmaskarray(image_ma)
    data = np.asarray(image_ma.filled(np.nan), dtype=np.float64)
    inc = np.deg2rad(np.asarray(incidence_deg.filled(np.nan), dtype=np.float64))
    emi = np.deg2rad(np.asarray(emission_deg.filled(np.nan), dtype=np.float64))
    pha = np.deg2rad(np.asarray(phase_deg.filled(np.nan), dtype=np.float64))

    try:
        file_model = photometric_model_from_name(photometric_model_name)
    except ValueError:
        file_model = None
    inc_w = np.nan_to_num(inc, nan=0.0)
    emi_w = np.nan_to_num(emi, nan=0.0)
    pha_w = np.nan_to_num(pha, nan=0.0)

    if mode_l == 'intrinsic' and file_model is None:
        return image_ma

    work = np.nan_to_num(data, nan=0.0)

    if file_model is not None:
        intrinsic = file_model.uncorrect(work, incidence=inc_w, emission=emi_w, phase=pha_w)
    else:
        intrinsic = work

    if mode_l == 'intrinsic':
        out = intrinsic
    else:
        view_model = photometric_model_from_name(mode_l)
        assert view_model is not None
        incore = np.nan_to_num(intrinsic, nan=0.0)
        out = view_model.correct(incore, incidence=inc_w, emission=emi_w, phase=pha_w)

    out_f = np.asarray(out, dtype=np.float64)
    out_ma = ma.masked_array(out_f, mask=mask)
    if hasattr(image_ma, 'fill_value'):
        out_ma.fill_value = image_ma.fill_value
    return out_ma


def compute_ring_display_image(
    *,
    mode: str,
    image_ma: ma.MaskedArray,
    photometric_model_name: str | None,
    mean_phase_deg: ma.MaskedArray,
    mean_emission_deg: ma.MaskedArray,
    mean_incidence_deg: float | None,
) -> ma.MaskedArray:
    """Return image data for the ring mosaic viewer for a photometric display mode.

    ``mean_phase_deg`` and ``mean_emission_deg`` are per-longitude columns (deg); they
    are broadcast over radius rows to match ``image_ma``. ``mean_incidence_deg`` is a
    scalar mean (deg) for the reprojection / mosaic slice. If incidence is unknown,
    modes other than ``as_saved`` return the stored image unchanged (same geometry
    cannot be inferred for undo/apply).

    Parameters:
        mode: Same values as :func:`compute_body_display_image` (e.g. ``as_saved``,
            ``intrinsic``, ``lambert``).
        image_ma: 2-D ring image (radius x longitude), masked array.
        photometric_model_name: Model name stored with the file, if any.
        mean_phase_deg: 1-D masked array of mean phase (deg) per longitude column.
        mean_emission_deg: 1-D masked array of mean emission (deg) per longitude column.
        mean_incidence_deg: Scalar mean incidence (deg), or ``None`` / non-finite when
            unknown.

    Returns:
        ``ma.MaskedArray`` with the same shape and dtype semantics as
        :func:`compute_body_display_image`. When ``mean_incidence_deg`` is missing or
        not finite, returns ``image_ma`` unchanged for any mode other than ``as_saved``.
    """
    mode_l = _validate_photometry_mode(mode)
    if mode_l == 'as_saved':
        return image_ma

    if image_ma.ndim != 2:
        raise ValueError(
            f'compute_ring_display_image: image_ma must be 2-D for broadcast with '
            f'mean_phase_deg / mean_emission_deg; got shape {image_ma.shape}'
        )

    if mean_incidence_deg is None or not math.isfinite(float(mean_incidence_deg)):
        return image_ma

    mask = ma.getmaskarray(image_ma)
    n_r, n_c = image_ma.shape

    def _broadcast_columns(col_ma: ma.MaskedArray) -> ma.MaskedArray:
        col_f = np.asarray(ma.filled(col_ma, np.nan), dtype=np.float64)
        grid = np.broadcast_to(col_f[np.newaxis, :], (n_r, n_c))
        col_mask = ma.getmaskarray(col_ma)
        m2 = np.broadcast_to(col_mask[np.newaxis, :], (n_r, n_c))
        return ma.masked_array(grid, mask=m2)

    phase_deg = _broadcast_columns(mean_phase_deg)
    emission_deg = _broadcast_columns(mean_emission_deg)
    inc_scalar = float(mean_incidence_deg)
    incidence_deg = ma.masked_array(
        np.full((n_r, n_c), inc_scalar, dtype=np.float64),
        mask=mask,
    )

    return compute_body_display_image(
        mode=mode_l,
        image_ma=image_ma,
        photometric_model_name=photometric_model_name,
        phase_deg=phase_deg,
        emission_deg=emission_deg,
        incidence_deg=incidence_deg,
    )
