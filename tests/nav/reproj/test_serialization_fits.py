"""FITS serialization edge cases in ``nav.reproj._serialization``."""

import math
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest
from astropy.io import fits

from nav.reproj._serialization import infer_format, load_fits, orbit_model_to_dict, save_fits
from nav.reproj.rings import RingReprojResult


def test_infer_format_detects_npz_renamed_to_fits(tmp_path: Path) -> None:
    """``.fits`` paths containing ZIP (npz) bytes must load as npz, not FITS."""
    npz_source = tmp_path / 'real.npz'
    np.savez_compressed(npz_source, x=np.array([1]))
    disguised = tmp_path / 'disguised.fits'
    disguised.write_bytes(npz_source.read_bytes())
    assert infer_format(disguised, None) == 'npz'


def test_save_fits_bool_ndarray_is_stored_as_uint8(tmp_path: Path) -> None:
    """Boolean image data must be coerced for astropy ImageHDU (no bool BITPIX)."""
    path = tmp_path / 'bools.fits'
    save_fits(
        path,
        'TestKind',
        1,
        {'anti': np.array([True, False, True], dtype=np.bool_)},
    )
    with fits.open(path) as hdul:
        assert hdul[1].name == 'ANTI'
        np.testing.assert_array_equal(hdul[1].data, np.array([1, 0, 1], dtype=np.uint8))


def test_load_fits_lowercase_keys_match_npz_conventions(tmp_path: Path) -> None:
    """load_fits must yield lowercase keys so dataclass ``load()`` matches npz."""
    path = tmp_path / 'keys.fits'
    save_fits(
        path,
        'RingReprojResult',
        1,
        {'orbit_model': orbit_model_to_dict(None), 'body_name': 'SATURN'},
    )
    d = load_fits(path, 'RingReprojResult')
    assert 'body_name' in d
    assert d['body_name'] == 'SATURN'
    assert 'orbit_model__is_none' in d
    assert d['orbit_model__is_none'] is True


def test_ring_reproj_result_fits_roundtrip(tmp_path: Path) -> None:
    """RingReprojResult save/load via FITS matches npz-style field names and data."""
    n_full, n_rad, valid = 32, 5, [0, 1]
    lon_res = math.pi / 16
    rad_res = 5.0
    ri, ro = 1000.0, 1020.0
    antimask = np.zeros(n_full, dtype=np.bool_)
    antimask[valid] = True
    shape = (n_rad, len(valid))
    img = ma.MaskedArray(np.ones(shape, dtype=np.float32))
    r1 = RingReprojResult(
        body_name='SATURN',
        longitude_resolution=lon_res,
        radius_resolution=rad_res,
        radius_inner=ri,
        radius_outer=ro,
        longitude_antimask=antimask,
        img=img,
        mean_radial_resolution=np.ones(len(valid), dtype=np.float32),
        mean_angular_resolution=np.ones(len(valid), dtype=np.float32),
        mean_phase=np.ones(len(valid), dtype=np.float32),
        mean_emission=np.ones(len(valid), dtype=np.float32),
        incidence=0.4,
        time=123.0,
        orbit_model=None,
        image_dtype=np.dtype(np.float32),
        metadata_dtype=np.dtype(np.float32),
        photometric_model_name='lambert',
        image_name='N123456',
    )
    path = tmp_path / 'ring.fits'
    r1.save(path, format='fits')
    r2 = RingReprojResult.load(path, format='fits')
    assert r2.body_name == r1.body_name
    assert r2.longitude_resolution == pytest.approx(r1.longitude_resolution)
    assert r2.radius_resolution == pytest.approx(r1.radius_resolution)
    assert r2.radius_inner == pytest.approx(r1.radius_inner)
    assert r2.radius_outer == pytest.approx(r1.radius_outer)
    assert r2.incidence == pytest.approx(r1.incidence)
    assert r2.time == pytest.approx(r1.time)
    assert r2.orbit_model is None
    assert r2.image_name == r1.image_name
    assert r2.photometric_model_name == r1.photometric_model_name
    np.testing.assert_array_equal(r2.longitude_antimask, r1.longitude_antimask)
    np.testing.assert_array_equal(ma.getdata(r2.img), ma.getdata(r1.img))
    np.testing.assert_array_equal(ma.getmaskarray(r2.img), ma.getmaskarray(r1.img))
