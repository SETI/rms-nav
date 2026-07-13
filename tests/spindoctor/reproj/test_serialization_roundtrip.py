"""Spec tests for FITS/npz round-trip completeness in ``spindoctor.reproj._serialization``.

Contracts under test (module docstring of ``_serialization``, the dev guide
"Serialization" section, and the ``save`` / ``load`` docstrings on
``RingReprojResult`` / ``RingMosaicData``):

- Every documented dataclass field survives a save/load cycle bit-exact (arrays) or
  exactly (scalars/strings); FITS may flip array byte order but not kind or width.
- ``__kind__`` / ``__version__`` sentinels are written and checked; mismatched or
  missing sentinels raise ``ValueError``.
- Load-time dtype verification raises ``ValueError`` naming the offending field.
- ``RingOrbitModel`` round-trips through flattened ``orbit_model__*`` entries;
  ``None`` round-trips through the ``is_none`` sentinel.
- Format inference follows the documented extension rules.

Synthetic reprojection results reuse the ``_make_ring_repro`` builder from
``tests.spindoctor.reproj.test_rings``.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest
from astropy.io import fits as astropy_fits
from tests.spindoctor.reproj.test_rings import (
    _LON_RES,
    _RAD_RES,
    _RADIUS_INNER,
    _RADIUS_OUTER,
    _make_ring_repro,
)

from spindoctor.reproj._serialization import (
    infer_format,
    load_npz,
    orbit_model_from_dict,
    orbit_model_to_dict,
    save_npz,
    tuple_of_strings_field,
    verify_dtype,
)
from spindoctor.reproj.ring_orbit_model import RingOrbitModel
from spindoctor.reproj.rings import RingMosaic, RingMosaicData, RingReprojResult

_ORBIT_MODEL = RingOrbitModel(
    name='roundtrip-model',
    a=140221.3,
    e=0.00235,
    w0=0.4224,
    dw=0.0471,
    mean_motion=10.157,
    epoch_utc='2007-01-01',
)


def _make_mosaic() -> RingMosaic:
    """Return a small SATURN RingMosaic matching the shared test grid constants."""
    return RingMosaic(
        body_name='SATURN',
        radius_inner=_RADIUS_INNER,
        radius_outer=_RADIUS_OUTER,
        longitude_resolution=_LON_RES,
        radius_resolution=_RAD_RES,
    )


def _make_mosaic_data() -> RingMosaicData:
    """Return sparse RingMosaicData built from two synthetic reprojections."""
    mosaic = _make_mosaic()
    img_a = ma.MaskedArray(np.arange(10, dtype=np.float64).reshape(5, 2))
    img_a[0, 0] = ma.masked
    mosaic.add(
        _make_ring_repro(
            valid_lon_bins=[3, 4],
            img_values=img_a,
            mean_radial_resolution=7.5,
            incidence=0.2,
            time=1000.5,
            image_name='img_a',
        )
    )
    mosaic.add(
        _make_ring_repro(
            valid_lon_bins=[10],
            img_values=42.0,
            mean_radial_resolution=3.25,
            incidence=0.6,
            time=2000.25,
            image_name='img_b',
        )
    )
    return mosaic.to_sparse()


def _assert_repro_equal(loaded: RingReprojResult, original: RingReprojResult) -> None:
    """Assert that every documented RingReprojResult field round-tripped exactly.

    Parameters:
        loaded: The result reconstructed by ``RingReprojResult.load``.
        original: The result that was saved.
    """
    assert loaded.body_name == original.body_name
    assert loaded.longitude_resolution == original.longitude_resolution
    assert loaded.radius_resolution == original.radius_resolution
    assert loaded.radius_inner == original.radius_inner
    assert loaded.radius_outer == original.radius_outer
    np.testing.assert_array_equal(loaded.longitude_antimask, original.longitude_antimask)
    assert loaded.longitude_antimask.dtype == np.dtype(np.bool_)
    np.testing.assert_array_equal(ma.getdata(loaded.img), ma.getdata(original.img))
    np.testing.assert_array_equal(ma.getmaskarray(loaded.img), ma.getmaskarray(original.img))
    np.testing.assert_array_equal(loaded.mean_radial_resolution, original.mean_radial_resolution)
    np.testing.assert_array_equal(loaded.mean_angular_resolution, original.mean_angular_resolution)
    np.testing.assert_array_equal(loaded.mean_phase, original.mean_phase)
    np.testing.assert_array_equal(loaded.mean_emission, original.mean_emission)
    assert loaded.incidence == original.incidence
    assert loaded.time == original.time
    assert loaded.orbit_model == original.orbit_model
    assert loaded.image_dtype == original.image_dtype
    assert loaded.metadata_dtype == original.metadata_dtype
    assert loaded.photometric_model_name == original.photometric_model_name
    assert loaded.image_name == original.image_name


def _assert_mosaic_data_equal(loaded: RingMosaicData, original: RingMosaicData) -> None:
    """Assert that every documented RingMosaicData field round-tripped exactly.

    Parameters:
        loaded: The data reconstructed by ``RingMosaicData.load``.
        original: The data that was saved.
    """
    assert loaded.body_name == original.body_name
    assert loaded.ring_body_name == original.ring_body_name
    assert loaded.shadow_body_name == original.shadow_body_name
    assert loaded.longitude_resolution == original.longitude_resolution
    assert loaded.radius_resolution == original.radius_resolution
    assert loaded.radius_inner == original.radius_inner
    assert loaded.radius_outer == original.radius_outer
    np.testing.assert_array_equal(loaded.longitude_antimask, original.longitude_antimask)
    np.testing.assert_array_equal(ma.getdata(loaded.img), ma.getdata(original.img))
    np.testing.assert_array_equal(ma.getmaskarray(loaded.img), ma.getmaskarray(original.img))
    assert loaded.longitude_range == original.longitude_range
    for field in (
        'mean_radial_resolution',
        'mean_angular_resolution',
        'mean_phase',
        'mean_emission',
        'image_number',
        'time',
    ):
        got = getattr(loaded, field)
        want = getattr(original, field)
        np.testing.assert_array_equal(ma.getdata(got), ma.getdata(want))
        np.testing.assert_array_equal(ma.getmaskarray(got), ma.getmaskarray(want))
    assert loaded.mean_incidence == original.mean_incidence
    # FITS may flip byte order; the documented contract is width and kind.
    assert loaded.image_number.dtype.kind == 'u'
    assert loaded.image_number.dtype.itemsize == 2
    assert loaded.time.dtype.kind == 'f'
    assert loaded.time.dtype.itemsize == 8
    assert loaded.image_dtype == original.image_dtype
    assert loaded.metadata_dtype == original.metadata_dtype
    assert loaded.contributing_image_names == original.contributing_image_names
    assert loaded.orbit_model_name == original.orbit_model_name
    assert loaded.photometric_model_name == original.photometric_model_name


class TestRingReprojResultRoundTrip:
    """Field-complete save/load round trips for RingReprojResult."""

    @pytest.mark.parametrize('compress', [True, False])
    def test_npz_round_trip_all_fields(self, tmp_path: Path, compress: bool) -> None:
        """All fields survive an npz round trip, compressed or not."""
        img = ma.MaskedArray(np.arange(15, dtype=np.float64).reshape(5, 3))
        img[2, 1] = ma.masked
        original = _make_ring_repro(
            valid_lon_bins=[2, 7, 8],
            img_values=img,
            mean_radial_resolution=np.array([1.5, 2.5, 3.5]),
            incidence=0.775,
            time=987654.125,
            photometric_model_name='lambert',
            image_name='N171',
        )
        path = tmp_path / 'repro.npz'
        original.save(path, compress=compress)
        _assert_repro_equal(RingReprojResult.load(path), original)

    def test_fits_round_trip_all_fields(self, tmp_path: Path) -> None:
        """All fields survive a FITS round trip."""
        img = ma.MaskedArray(np.arange(15, dtype=np.float64).reshape(5, 3))
        img[0, 0] = ma.masked
        original = _make_ring_repro(
            valid_lon_bins=[2, 7, 8],
            img_values=img,
            mean_phase=np.array([0.25, 0.5, 0.75]),
            incidence=0.775,
            time=987654.125,
            photometric_model_name='lambert',
            image_name='N171',
        )
        path = tmp_path / 'repro.fits'
        original.save(path, format_='fits')
        _assert_repro_equal(RingReprojResult.load(path, format_='fits'), original)

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_orbit_model_round_trips_by_value(self, tmp_path: Path, fmt: str) -> None:
        """A non-None orbit model is rebuilt equal to the original in both formats."""
        original = _make_ring_repro(
            valid_lon_bins=[5],
            radius_inner=-10.0,
            radius_outer=10.0,
            orbit_model=_ORBIT_MODEL,
        )
        path = tmp_path / f'om.{fmt}'
        original.save(path, format_=fmt)
        loaded = RingReprojResult.load(path, format_=fmt)
        assert loaded.orbit_model == _ORBIT_MODEL
        assert loaded.orbit_model is not None
        assert loaded.orbit_model.a == _ORBIT_MODEL.a
        assert loaded.orbit_model.epoch_utc == _ORBIT_MODEL.epoch_utc

    def test_npz_round_trip_nan_incidence(self, tmp_path: Path) -> None:
        """A NaN incidence (produced by an all-masked reprojection) survives npz."""
        original = _make_ring_repro(valid_lon_bins=[3, 4], incidence=float('nan'))
        path = tmp_path / 'nan_inc.npz'
        original.save(path)
        assert math.isnan(RingReprojResult.load(path).incidence)

    def test_fits_round_trip_nan_incidence(self, tmp_path: Path) -> None:
        """A NaN incidence should survive a FITS round trip like it does for npz."""
        original = _make_ring_repro(valid_lon_bins=[3, 4], incidence=float('nan'))
        path = tmp_path / 'nan_inc.fits'
        original.save(path, format_='fits')
        assert math.isnan(RingReprojResult.load(path, format_='fits').incidence)

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_non_finite_image_values_survive(self, tmp_path: Path, fmt: str) -> None:
        """Unmasked NaN and inf pixels in img round trip bit-exact in both formats."""
        img = ma.MaskedArray(np.ones((5, 2), dtype=np.float64))
        img[0, 0] = np.nan
        img[1, 1] = np.inf
        original = _make_ring_repro(valid_lon_bins=[1, 2], img_values=img)
        path = tmp_path / f'nonfinite.{fmt}'
        original.save(path, format_=fmt)
        loaded = RingReprojResult.load(path, format_=fmt)
        assert math.isnan(ma.getdata(loaded.img)[0, 0])
        assert math.isinf(ma.getdata(loaded.img)[1, 1])

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_empty_reprojection_round_trips(self, tmp_path: Path, fmt: str) -> None:
        """A reprojection with zero valid longitude columns round trips in both formats."""
        original = _make_ring_repro(valid_lon_bins=[])
        path = tmp_path / f'empty.{fmt}'
        original.save(path, format_=fmt)
        loaded = RingReprojResult.load(path, format_=fmt)
        assert loaded.img.shape == (5, 0)
        assert loaded.mean_phase.shape == (0,)
        assert not loaded.longitude_antimask.any()

    def test_fits_preserves_metadata_dtype_width_and_kind(self, tmp_path: Path) -> None:
        """FITS may flip byte order but metadata arrays stay 4-byte floats."""
        original = _make_ring_repro(valid_lon_bins=[2, 3])
        path = tmp_path / 'dtypes.fits'
        original.save(path, format_='fits')
        loaded = RingReprojResult.load(path, format_='fits')
        assert loaded.mean_radial_resolution.dtype.kind == 'f'
        assert loaded.mean_radial_resolution.dtype.itemsize == 4
        assert ma.getdata(loaded.img).dtype.kind == 'f'
        assert ma.getdata(loaded.img).dtype.itemsize == 8


class TestRingMosaicDataRoundTrip:
    """Field-complete save/load round trips for RingMosaicData."""

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_round_trip_all_fields(self, tmp_path: Path, fmt: str) -> None:
        """Every field of a two-image sparse mosaic survives both formats."""
        original = _make_mosaic_data()
        path = tmp_path / f'mosaic.{fmt}'
        original.save(path, format_=fmt)
        _assert_mosaic_data_equal(RingMosaicData.load(path, format_=fmt), original)

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_bounded_longitude_range_round_trips(self, tmp_path: Path, fmt: str) -> None:
        """The (start, end) longitude_range tuple from to_bounded survives exactly."""
        mosaic = _make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[3, 4], image_name='img_a'))
        original = mosaic.to_bounded(longitude_range=(3 * _LON_RES, 5 * _LON_RES))
        path = tmp_path / f'bounded.{fmt}'
        original.save(path, format_=fmt)
        loaded = RingMosaicData.load(path, format_=fmt)
        assert loaded.longitude_range == original.longitude_range

    @pytest.mark.parametrize('fmt', ['npz', 'fits'])
    def test_empty_mosaic_round_trips(self, tmp_path: Path, fmt: str) -> None:
        """A mosaic with no images saves and reloads with zero-width sparse arrays."""
        original = _make_mosaic().to_sparse()
        path = tmp_path / f'empty_mosaic.{fmt}'
        original.save(path, format_=fmt)
        loaded = RingMosaicData.load(path, format_=fmt)
        assert loaded.img.shape == original.img.shape
        assert not loaded.longitude_antimask.any()
        assert loaded.contributing_image_names == ()
        assert loaded.image_number.dtype.kind == 'u'
        assert loaded.image_number.dtype.itemsize == 2
        assert loaded.time.dtype.kind == 'f'
        assert loaded.time.dtype.itemsize == 8

    def test_fits_single_empty_image_name_round_trips(self, tmp_path: Path) -> None:
        """contributing_image_names == ('',) must survive a FITS round trip."""
        mosaic = _make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[3], image_name=''))
        original = mosaic.to_sparse()
        assert original.contributing_image_names == ('',)
        path = tmp_path / 'empty_name.fits'
        original.save(path, format_='fits')
        loaded = RingMosaicData.load(path, format_='fits')
        assert loaded.contributing_image_names == ('',)

    def test_npz_single_empty_image_name_round_trips(self, tmp_path: Path) -> None:
        """contributing_image_names == ('',) survives an npz round trip."""
        mosaic = _make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[3], image_name=''))
        original = mosaic.to_sparse()
        path = tmp_path / 'empty_name.npz'
        original.save(path)
        assert RingMosaicData.load(path).contributing_image_names == ('',)

    def test_fits_mixed_empty_names_round_trip(self, tmp_path: Path) -> None:
        """Empty names between non-empty ones survive the NUL-separated FITS encoding."""
        original = dataclasses.replace(
            _make_mosaic_data(), contributing_image_names=('img_a', '', 'img_b')
        )
        path = tmp_path / 'mixed_names.fits'
        original.save(path, format_='fits')
        loaded = RingMosaicData.load(path, format_='fits')
        assert loaded.contributing_image_names == ('img_a', '', 'img_b')


class TestCorruptAndMismatchedInputs:
    """Documented rejection of wrong-kind, truncated, and dtype-corrupted files."""

    def test_npz_kind_mismatch_rejected(self, tmp_path: Path) -> None:
        """Loading a RingMosaicData npz as RingReprojResult raises ValueError."""
        path = tmp_path / 'mosaic.npz'
        _make_mosaic_data().save(path)
        with pytest.raises(ValueError, match='Kind mismatch'):
            RingReprojResult.load(path)

    def test_fits_kind_mismatch_rejected(self, tmp_path: Path) -> None:
        """Loading a RingReprojResult FITS file as RingMosaicData raises ValueError."""
        path = tmp_path / 'repro.fits'
        _make_ring_repro(valid_lon_bins=[1]).save(path, format_='fits')
        with pytest.raises(ValueError, match='Kind mismatch'):
            RingMosaicData.load(path, format_='fits')

    def test_npz_missing_kind_sentinel_rejected(self, tmp_path: Path) -> None:
        """An npz without __kind__ is rejected as truncated or wrong format."""
        path = tmp_path / 'no_kind.npz'
        np.savez(path, foo=np.array([1]))
        with pytest.raises(ValueError, match='Missing file sentinel __kind__'):
            load_npz(path, 'RingReprojResult')

    def test_npz_missing_version_sentinel_rejected(self, tmp_path: Path) -> None:
        """An npz without __version__ is rejected as truncated or wrong format."""
        path = tmp_path / 'no_version.npz'
        np.savez(path, __kind__=np.array('RingReprojResult'))
        with pytest.raises(ValueError, match='Missing file sentinel __version__'):
            load_npz(path, 'RingReprojResult')

    def test_missing_required_keys_rejected(self, tmp_path: Path) -> None:
        """A well-formed npz lacking required RingReprojResult keys raises ValueError."""
        path = tmp_path / 'partial.npz'
        save_npz(path, 'RingReprojResult', 1, {'body_name': 'SATURN'}, compress=False)
        with pytest.raises(ValueError, match='missing required keys'):
            RingReprojResult.load(path)

    def test_npz_orphaned_data_entry_rejected(self, tmp_path: Path) -> None:
        """A __data entry without its __mask twin raises ValueError."""
        path = tmp_path / 'orphan.npz'
        np.savez(
            path,
            __kind__=np.array('K'),
            __version__=np.array(1),
            img__data=np.zeros(3),
        )
        with pytest.raises(ValueError, match='Unmatched "__data"/"__mask"'):
            load_npz(path, 'K')

    def test_fits_duplicate_extname_rejected(self, tmp_path: Path) -> None:
        """Two HDUs sharing an EXTNAME raise ValueError instead of silently clobbering."""
        path = tmp_path / 'dup.fits'
        primary = astropy_fits.PrimaryHDU()
        primary.header['KIND'] = 'K'
        primary.header['VERSION'] = 1
        hdus = astropy_fits.HDUList(
            [
                primary,
                astropy_fits.ImageHDU(data=np.zeros(3), name='IMG'),
                astropy_fits.ImageHDU(data=np.zeros(3), name='IMG'),
            ]
        )
        hdus.writeto(path, overwrite=True)
        from spindoctor.reproj._serialization import load_fits

        with pytest.raises(ValueError, match="Duplicate FITS EXTNAME 'IMG'"):
            load_fits(path, 'K')

    def test_fits_orphaned_mask_hdu_rejected(self, tmp_path: Path) -> None:
        """A _MASK HDU without its base HDU raises ValueError."""
        path = tmp_path / 'orphan_mask.fits'
        primary = astropy_fits.PrimaryHDU()
        primary.header['KIND'] = 'K'
        primary.header['VERSION'] = 1
        hdus = astropy_fits.HDUList(
            [primary, astropy_fits.ImageHDU(data=np.zeros(3, np.uint8), name='FOO_MASK')]
        )
        hdus.writeto(path, overwrite=True)
        from spindoctor.reproj._serialization import load_fits

        with pytest.raises(ValueError, match='Orphaned "_MASK" HDUs'):
            load_fits(path, 'K')

    def test_image_dtype_mismatch_rejected_on_load(self, tmp_path: Path) -> None:
        """A file whose img dtype disagrees with its declared image_dtype is rejected."""
        bad = dataclasses.replace(
            _make_ring_repro(valid_lon_bins=[1, 2]), image_dtype=np.dtype(np.float32)
        )
        path = tmp_path / 'bad_dtype.npz'
        bad.save(path)
        with pytest.raises(ValueError, match="image_dtype mismatch for field 'img'"):
            RingReprojResult.load(path)

    def test_image_number_dtype_mismatch_rejected_on_load(self, tmp_path: Path) -> None:
        """A file whose image_number is not uint16 is rejected."""
        data = _make_mosaic_data()
        n = data.image_number.shape[0]
        bad = dataclasses.replace(data, image_number=ma.MaskedArray(np.zeros(n, dtype=np.uint32)))
        path = tmp_path / 'bad_imgnum.npz'
        bad.save(path)
        with pytest.raises(ValueError, match='image_number must be uint16'):
            RingMosaicData.load(path)


class TestInferFormat:
    """Documented extension-based format inference."""

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('data.npz', 'npz'),
            ('data.fits', 'fits'),
            ('data.fit', 'fits'),
            ('data.fz', 'fits'),
            ('data.fits.gz', 'fits'),
        ],
    )
    def test_extension_mapping(self, name: str, expected: str) -> None:
        """Each documented extension maps to its format (files need not exist)."""
        assert infer_format(name, None) == expected

    def test_unknown_extension_rejected(self) -> None:
        """An unrecognized extension raises ValueError telling the user to pass format_."""
        with pytest.raises(ValueError, match='Cannot infer format'):
            infer_format('data.dat', None)

    def test_invalid_explicit_format_rejected(self) -> None:
        """An unsupported explicit format_ raises ValueError."""
        with pytest.raises(ValueError, match="format must be 'npz' or 'fits'"):
            infer_format('data.npz', 'hdf5')

    def test_explicit_format_overrides_extension(self) -> None:
        """An explicit format_ wins over the file extension."""
        assert infer_format('data.fits', 'npz') == 'npz'


class TestSerializationHelpers:
    """tuple_of_strings_field, orbit_model dict helpers, and verify_dtype."""

    def test_tuple_of_strings_none_becomes_empty(self) -> None:
        """None normalizes to the empty tuple."""
        assert tuple_of_strings_field(None) == ()

    def test_tuple_of_strings_from_unicode_array(self) -> None:
        """A 1-D unicode array (npz encoding) becomes a tuple of str."""
        assert tuple_of_strings_field(np.array(['a', 'bb'])) == ('a', 'bb')

    def test_tuple_of_strings_from_utf8_bytes(self) -> None:
        """A uint8 array of NUL-terminated UTF-8 (FITS encoding) decodes correctly."""
        raw = np.frombuffer(b'x\0y\0', dtype=np.uint8).copy()
        assert tuple_of_strings_field(raw) == ('x', 'y')

    def test_tuple_of_strings_missing_terminator_rejected(self) -> None:
        """A payload without the per-entry NUL terminator raises ValueError."""
        raw = np.frombuffer(b'x\0y', dtype=np.uint8).copy()
        with pytest.raises(ValueError, match='missing entry terminator'):
            tuple_of_strings_field(raw)

    def test_tuple_of_strings_invalid_utf8_rejected(self) -> None:
        """Invalid UTF-8 bytes raise the documented ValueError."""
        raw = np.array([0xFF, 0xFE, 0x00], dtype=np.uint8)
        with pytest.raises(ValueError, match='Invalid UTF-8'):
            tuple_of_strings_field(raw)

    def test_tuple_of_strings_from_list(self) -> None:
        """A plain list normalizes to a tuple of str."""
        assert tuple_of_strings_field(['a', 'b']) == ('a', 'b')

    def test_orbit_model_none_round_trips_through_sentinel(self) -> None:
        """None serializes to the is_none sentinel and deserializes back to None."""
        d = orbit_model_to_dict(None)
        assert d == {'is_none': True}
        assert orbit_model_from_dict(d) is None

    def test_orbit_model_dict_round_trip(self) -> None:
        """A model round-trips through to_dict/from_dict by value."""
        assert orbit_model_from_dict(orbit_model_to_dict(_ORBIT_MODEL)) == _ORBIT_MODEL

    def test_orbit_model_missing_keys_rejected(self) -> None:
        """A non-None dict lacking required keys raises ValueError."""
        with pytest.raises(ValueError, match='missing required key'):
            orbit_model_from_dict({'is_none': False, 'name': 'x'})

    def test_verify_dtype_allows_endian_swap(self) -> None:
        """Big-endian arrays (as FITS produces) match a little-endian declaration."""
        arrays = {'img': np.zeros(3, dtype='>f8')}
        verify_dtype(
            arrays,
            image_dtype=np.dtype('<f8'),
            metadata_dtype=np.dtype(np.float32),
            image_fields=['img'],
            metadata_fields=[],
        )

    def test_verify_dtype_rejects_kind_mismatch(self) -> None:
        """Same-width but different-kind arrays (int64 vs float64) are rejected."""
        arrays = {'img': np.zeros(3, dtype=np.int64)}
        with pytest.raises(ValueError, match="image_dtype mismatch for field 'img'"):
            verify_dtype(
                arrays,
                image_dtype=np.dtype(np.float64),
                metadata_dtype=np.dtype(np.float32),
                image_fields=['img'],
                metadata_fields=[],
            )

    def test_verify_dtype_enforces_float64_fields(self) -> None:
        """Fields listed in float64_fields must be 8-byte floats regardless of dtypes."""
        arrays = {'time': np.zeros(3, dtype=np.float32)}
        with pytest.raises(ValueError, match='must be float64'):
            verify_dtype(
                arrays,
                image_dtype=np.dtype(np.float64),
                metadata_dtype=np.dtype(np.float32),
                image_fields=[],
                metadata_fields=[],
                float64_fields=['time'],
            )
