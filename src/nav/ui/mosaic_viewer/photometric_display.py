"""Photometric display modes for ring/body mosaic viewer windows.

Converts stored mosaic/reprojection pixels to intrinsic brightness or applies an
alternate viewing model for on-screen display. Core correction models live in
:mod:`nav.reproj.photometric_model`.
"""

import math

import numpy as np
import numpy.ma as ma

from nav.reproj.photometric_model import photometric_model_from_name


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
        mode: ``'as_saved'`` (file pixels), ``'intrinsic'`` (undo file model only),
            or a model name understood by
            :func:`~nav.reproj.photometric_model.photometric_model_from_name`
            (undo file model, then apply that model).
        image_ma: Stored image (possibly already corrected when saved).
        photometric_model_name: Model name stored with the file, if any.
        phase_deg, emission_deg, incidence_deg: Per-pixel angles in **degrees**
            (same shape as ``image_ma``).

    Returns:
        Masked array of the same shape and dtype as ``image_ma``.
    """
    mode_l = str(mode).strip().lower()
    if mode_l == 'as_saved':
        return image_ma

    mask = ma.getmaskarray(image_ma)
    data = np.asarray(image_ma.filled(np.nan), dtype=np.float64)
    inc = np.deg2rad(np.asarray(incidence_deg.filled(np.nan), dtype=np.float64))
    emi = np.deg2rad(np.asarray(emission_deg.filled(np.nan), dtype=np.float64))
    pha = np.deg2rad(np.asarray(phase_deg.filled(np.nan), dtype=np.float64))

    file_model = photometric_model_from_name(photometric_model_name)
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
        if view_model is None:
            out = intrinsic
        else:
            incore = np.nan_to_num(intrinsic, nan=0.0)
            out = view_model.correct(incore, incidence=inc_w, emission=emi_w, phase=pha_w)

    out = np.asarray(out, dtype=np.asarray(image_ma).dtype)
    out = np.where(mask, np.nan, out)
    out_ma = ma.masked_array(out, mask=mask)
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
    """
    mode_l = str(mode).strip().lower()
    if mode_l == 'as_saved':
        return image_ma

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
