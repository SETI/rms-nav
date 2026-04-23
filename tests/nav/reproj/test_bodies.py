"""Tests for BodyMosaic, BodyReprojResult, and BodyMosaicData.

Geometry-dependent tests (reproject()) are deferred to integration tests.
These unit tests cover static utilities, mosaic accumulation, dynamic expansion,
longitude wraparound, and retrieval methods using synthetically constructed
BodyReprojResult objects.
"""

import math
from types import SimpleNamespace
from typing import Literal

import numpy as np
import numpy.ma as ma
import pytest

from nav.reproj.bodies import BodyMosaic, BodyMosaicData, BodyReprojResult
from nav.ui.mosaic_viewer.common import load_body_file

# Convenient resolution (rad/pixel) for tests: 0.1 rad ~ 5.7 deg
_LAT_RES = 0.1  # rad/pixel
_LON_RES = 0.1  # rad/pixel

_N_FULL_LAT = int(math.pi / _LAT_RES)  # 31 for 0.1 rad
_N_FULL_LON = int(2.0 * math.pi / _LON_RES)  # 62 for 0.1 rad


# =========================================================================
# Helpers
# =========================================================================


def _make_repro(
    *,
    body_name: str = 'MIMAS',
    lat_range: tuple[int, int],
    lon_range: tuple[int, int],
    img_values: float | np.ndarray | None = 1.0,
    eff_res_values: float | np.ndarray = 1.0,
    res_values: float | np.ndarray | None = None,
    mask: np.ndarray | None = None,
    latlon_type: Literal['centric', 'graphic', 'squashed'] = 'centric',
    lon_direction: Literal['east', 'west'] = 'east',
    lat_resolution: float = _LAT_RES,
    lon_resolution: float = _LON_RES,
    time: float = 0.0,
    image_name: str = '',
) -> BodyReprojResult:
    """Build a synthetic BodyReprojResult for use in tests.

    Parameters:
        body_name: Name of the body (e.g. ``'MIMAS'``).
        lat_range: (min_lat_bin, max_lat_bin) in full-grid coordinates.
        lon_range: (min_lon_bin, max_lon_bin) in full-grid coordinates.
        img_values: Scalar or array for image pixel values.
        eff_res_values: Scalar or array for effective resolution.
        res_values: Scalar or array for resolution (defaults to eff_res_values).
        mask: Boolean mask array; True means invalid.
        latlon_type: Latitude/longitude coordinate system (``'centric'``,
            ``'graphic'``, or ``'squashed'``).
        lon_direction: Longitude direction (``'east'`` or ``'west'``).
        lat_resolution: Latitude bin size in radians per pixel.
        lon_resolution: Longitude bin size in radians per pixel.
        time: Scalar observation midtime (TDB seconds).
        image_name: Label carried on the synthetic reprojection result.
    """
    n_lat = lat_range[1] - lat_range[0] + 1
    n_lon = lon_range[1] - lon_range[0] + 1
    shape = (n_lat, n_lon)

    def _fill(v: float | np.ndarray | None, default: float = 0.0) -> ma.MaskedArray:
        """Return a masked array of shape ``shape`` filled with v (or default) and masked by
        ``mask``."""
        if v is None:
            v = default
        if np.isscalar(v):
            arr = np.full(shape, v, dtype=np.float32)
        else:
            arr = np.array(v, dtype=np.float32)
        mval = ma.MaskedArray(arr, mask=(mask if mask is not None else False))
        return mval

    if res_values is None:
        res_values = eff_res_values

    return BodyReprojResult(
        body_name=body_name,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
        latlon_type=latlon_type,
        lon_direction=lon_direction,
        lat_idx_range=(lat_range[0], lat_range[1]),
        lon_idx_range=(lon_range[0], lon_range[1]),
        img=_fill(img_values, 0.0),
        resolution=_fill(res_values),
        eff_resolution=_fill(eff_res_values),
        phase=_fill(0.5),
        emission=_fill(0.3),
        incidence=_fill(0.4),
        time=time,
        photometric_model_name=None,
        image_dtype=np.dtype(np.float32),
        metadata_dtype=np.dtype(np.float32),
        image_name=image_name,
    )


# =========================================================================
# Outside-FOV early exit
# =========================================================================


class TestBodyReprojOutsideFov:
    """Tests for ``_body_reproj_result_outside_fov`` empty result."""

    def test_empty_result_shape_and_masked_img(self) -> None:
        """Non-mask-only empty result matches minimal idx range and is fully masked."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        obs = SimpleNamespace(midtime=123.45)
        r = mosaic._body_reproj_result_outside_fov(
            obs,
            mask_only=False,
            image_name='stem',
            navigation_uncertainty=0.0,
        )
        assert r.time == pytest.approx(123.45)
        assert r.image_name == 'stem'
        assert r.lat_idx_range == (0, 1)
        assert r.lon_idx_range == (0, 1)
        assert r.img.shape == (2, 2)
        assert bool(ma.getmaskarray(r.img).all())


# =========================================================================
# Static utility tests
# =========================================================================


class TestGenerateLatitudes:
    """Tests for BodyMosaic.generate_latitudes."""

    def test_default_range_starts_near_south_pole(self) -> None:
        """Default range starts at the minimum allowed latitude."""
        lats = BodyMosaic.generate_latitudes(lat_resolution=_LAT_RES)
        assert lats[0] >= -math.pi / 2.0

    def test_default_range_ends_near_north_pole(self) -> None:
        """Default range ends at the maximum allowed latitude."""
        lats = BodyMosaic.generate_latitudes(lat_resolution=_LAT_RES)
        assert lats[-1] <= math.pi / 2.0

    def test_values_on_grid_boundaries(self) -> None:
        """All returned latitudes are multiples of lat_resolution."""
        lats = BodyMosaic.generate_latitudes(lat_resolution=_LAT_RES)
        for lat in lats:
            residual = lat % _LAT_RES
            assert residual < 1e-10 or abs(residual - _LAT_RES) < 1e-10

    def test_custom_range_restriction(self) -> None:
        """Custom range excludes values outside the specified limits."""
        lats = BodyMosaic.generate_latitudes(
            latitude_start=0.0, latitude_end=0.5, lat_resolution=_LAT_RES
        )
        assert lats[0] >= 0.0
        assert lats[-1] <= 0.5

    def test_single_latitude_at_zero(self) -> None:
        """Range [0.0, 0.0] with any resolution returns exactly [0.0]."""
        lats = BodyMosaic.generate_latitudes(
            latitude_start=0.0, latitude_end=0.05, lat_resolution=_LAT_RES
        )
        assert len(lats) == 1
        assert pytest.approx(lats[0]) == 0.0


class TestGenerateLongitudes:
    """Tests for BodyMosaic.generate_longitudes."""

    def test_default_range_covers_full_circle(self) -> None:
        """Default range is close to 0..2pi."""
        lons = BodyMosaic.generate_longitudes(lon_resolution=_LON_RES)
        assert lons[0] >= 0.0
        assert lons[-1] < 2.0 * math.pi

    def test_custom_range(self) -> None:
        """Custom range excludes values outside limits."""
        lons = BodyMosaic.generate_longitudes(
            longitude_start=1.0, longitude_end=2.0, lon_resolution=_LON_RES
        )
        assert lons[0] >= 1.0
        assert lons[-1] <= 2.0


# =========================================================================
# Constructor validation tests
# =========================================================================


class TestBodyMosaicConstructor:
    """Tests for BodyMosaic constructor parameter validation."""

    def test_invalid_latlon_type_raises(self) -> None:
        """Invalid latlon_type raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match='latlon_type'):
            BodyMosaic(body_name='MIMAS', latlon_type='invalid')  # type: ignore[arg-type]

    def test_invalid_lon_direction_raises(self) -> None:
        """Invalid lon_direction raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match='lon_direction'):
            BodyMosaic(body_name='MIMAS', lon_direction='north')  # type: ignore[arg-type]

    def test_dynamic_false_none_lat_range_uses_full_valid_lat(self) -> None:
        """dynamic=False with lat_range=None uses the full valid latitude extent."""
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
            dynamic=False,
            lat_range=None,
            lon_range=(0.0, 1.0),
        )
        assert mosaic is not None

    def test_dynamic_false_none_lon_range_uses_full_valid_lon(self) -> None:
        """dynamic=False with lon_range=None uses the full valid longitude extent."""
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
            dynamic=False,
            lat_range=(0.0, 1.0),
            lon_range=None,
        )
        assert mosaic is not None

    def test_valid_defaults_construct_successfully(self) -> None:
        """Default constructor with only body_name succeeds."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        assert mosaic is not None

    def test_body_name_stored_correctly(self) -> None:
        """body_name is accessible from the mosaic."""
        mosaic = BodyMosaic(body_name='ENCELADUS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        assert mosaic.body_name == 'ENCELADUS'


# =========================================================================
# Basic add and retrieval tests
# =========================================================================


class TestBodyMosaicAddAndRetrieve:
    """Tests for add(), to_bounded(), to_full(), and bounds property."""

    def test_empty_mosaic_has_none_bounds(self) -> None:
        """Bounds is None when no data has been added."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        assert mosaic.bounds is None

    def test_add_single_result_updates_bounds(self) -> None:
        """After adding one result, bounds reflects the data extents."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        repro = _make_repro(lat_range=(5, 7), lon_range=(10, 12))
        mosaic.add(repro)

        bounds = mosaic.bounds
        assert bounds is not None
        (lat_min, lat_max), (lon_min, lon_max) = bounds
        assert pytest.approx(lat_min, abs=1e-10) == 5 * _LAT_RES - math.pi / 2.0
        assert pytest.approx(lat_max, abs=1e-10) == 7 * _LAT_RES - math.pi / 2.0
        assert pytest.approx(lon_min, abs=1e-10) == 10 * _LON_RES
        assert pytest.approx(lon_max, abs=1e-10) == 12 * _LON_RES

    def test_to_bounded_default_returns_data_bounds(self) -> None:
        """to_bounded() with no arguments returns current data bounds."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        img_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        repro = _make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=img_val)
        mosaic.add(repro)

        result = mosaic.to_bounded()
        assert isinstance(result, BodyMosaicData)
        assert result.img.shape == (3, 3)

    def test_to_bounded_returns_masked_array(self) -> None:
        """to_bounded() returns a MaskedArray for img."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12)))
        result = mosaic.to_bounded()
        assert isinstance(result.img, ma.MaskedArray)

    def test_to_bounded_data_values_correct(self) -> None:
        """Values from add() appear correctly in to_bounded()."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        img_val = np.arange(9, dtype=np.float32).reshape(3, 3) + 1.0
        repro = _make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=img_val)
        mosaic.add(repro)

        result = mosaic.to_bounded()
        np.testing.assert_allclose(result.img.data, img_val)

    def test_to_bounded_masked_pixels_are_masked(self) -> None:
        """Pixels outside the reprojected region are masked in to_bounded()."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mask = np.array([[False, False, True], [False, False, False], [True, False, False]])
        repro = _make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=1.0, mask=mask)
        mosaic.add(repro)

        result = mosaic.to_bounded()
        assert ma.getmaskarray(result.img)[0, 2]
        assert ma.getmaskarray(result.img)[2, 0]
        assert not ma.getmaskarray(result.img)[0, 0]

    def test_to_full_shape(self) -> None:
        """to_full() returns array with full latitude x longitude shape."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12)))

        result = mosaic.to_full()
        assert result.img.shape == (_N_FULL_LAT, _N_FULL_LON)

    def test_to_full_data_in_correct_location(self) -> None:
        """Data from add() appears at the correct position in to_full()."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        img_val = np.ones((3, 3), dtype=np.float32) * 42.0
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=img_val))

        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[5:8, 10:13], img_val)

    def test_to_full_empty_regions_are_masked(self) -> None:
        """Regions with no data are masked in to_full()."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12)))

        result = mosaic.to_full()
        # Lat bin 0 should have no data
        assert np.all(ma.getmaskarray(result.img)[0, :])

    def test_mosaic_lat_range_in_data(self) -> None:
        """BodyMosaicData lat_range reflects the actual data bounds."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12)))

        result = mosaic.to_bounded()
        lat_min_expected = 5 * _LAT_RES - math.pi / 2.0
        lat_max_expected = 7 * _LAT_RES - math.pi / 2.0
        assert pytest.approx(result.lat_range[0], abs=1e-10) == lat_min_expected
        assert pytest.approx(result.lat_range[1], abs=1e-10) == lat_max_expected


# =========================================================================
# Merge strategy tests
# =========================================================================


class TestMergeStrategies:
    """Tests for pixel conflict resolution during add()."""

    def test_better_resolution_wins(self) -> None:
        """New pixel replaces existing when its effective resolution is better."""
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
        )
        first = _make_repro(
            lat_range=(5, 5), lon_range=(10, 10), img_values=1.0, eff_res_values=2.0
        )
        second = _make_repro(
            lat_range=(5, 5), lon_range=(10, 10), img_values=9.0, eff_res_values=0.5
        )  # better res
        mosaic.add(first)
        mosaic.add(second)

        result = mosaic.to_bounded()
        assert pytest.approx(float(result.img.data[0, 0])) == 9.0

    def test_worse_resolution_does_not_win(self) -> None:
        """Worse resolution does not replace existing data."""
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
        )
        first = _make_repro(
            lat_range=(5, 5), lon_range=(10, 10), img_values=1.0, eff_res_values=0.5
        )  # good res
        second = _make_repro(
            lat_range=(5, 5), lon_range=(10, 10), img_values=9.0, eff_res_values=2.0
        )  # worse res
        mosaic.add(first)
        mosaic.add(second)

        result = mosaic.to_bounded()
        assert pytest.approx(float(result.img.data[0, 0])) == 1.0

    def test_empty_pixels_filled_regardless_of_resolution(self) -> None:
        """Empty pixels are always filled, even with worse resolution data."""
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
        )
        # First add fills rows 0-1 only (partial coverage)
        mask1 = np.array([[False, False], [True, True]])
        first = _make_repro(
            lat_range=(5, 6), lon_range=(10, 11), img_values=1.0, eff_res_values=0.5, mask=mask1
        )
        mosaic.add(first)

        # Second add fills rows 1 only (empty region) with worse resolution
        mask2 = np.array([[True, True], [False, False]])
        second = _make_repro(
            lat_range=(5, 6), lon_range=(10, 11), img_values=5.0, eff_res_values=2.0, mask=mask2
        )
        mosaic.add(second)

        result = mosaic.to_bounded()
        # Row 0: set by first (img=1.0, res=0.5), not replaced because already filled
        assert pytest.approx(float(result.img.data[0, 0])) == 1.0
        # Row 1: was empty, filled by second (img=5.0) even though res is worse
        assert pytest.approx(float(result.img.data[1, 0])) == 5.0


# =========================================================================
# Dynamic expansion tests
# =========================================================================


class TestDynamicExpansion:
    """Tests for dynamic=True expansion behavior."""

    def test_expansion_beyond_initial_lat(self) -> None:
        """Dynamic mosaic expands when new data arrives outside current lat range."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=1.0))
        mosaic.add(_make_repro(lat_range=(3, 5), lon_range=(10, 12), img_values=2.0))

        bounds = mosaic.bounds
        assert bounds is not None
        (lat_min, lat_max), _ = bounds
        # Should now cover lat bins 3-7
        assert pytest.approx(lat_min, abs=1e-10) == 3 * _LAT_RES - math.pi / 2.0
        assert pytest.approx(lat_max, abs=1e-10) == 7 * _LAT_RES - math.pi / 2.0

    def test_expansion_beyond_initial_lon(self) -> None:
        """Dynamic mosaic expands when new data arrives outside current lon range."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=1.0))
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(14, 16), img_values=2.0))

        bounds = mosaic.bounds
        assert bounds is not None
        _, (lon_min, lon_max) = bounds
        # Should now cover lon bins 10-16
        assert pytest.approx(lon_min, abs=1e-10) == 10 * _LON_RES
        assert pytest.approx(lon_max, abs=1e-10) == 16 * _LON_RES

    def test_dynamic_false_clips_outside_data(self) -> None:
        """dynamic=False clips data that falls outside the fixed lat/lon range."""
        lat_r = (3 * _LAT_RES - math.pi / 2.0, 7 * _LAT_RES - math.pi / 2.0)
        lon_r = (10 * _LON_RES, 14 * _LON_RES)
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
            lat_range=lat_r,
            lon_range=lon_r,
            dynamic=False,
        )
        # Data outside lat range (bins 0-2 are below the mosaic range of 3-7)
        mosaic.add(_make_repro(lat_range=(0, 2), lon_range=(10, 12), img_values=99.0))

        bounds = mosaic.bounds
        # No data should have been stored (all outside the fixed range)
        assert bounds is None

    def test_data_within_fixed_range_is_stored(self) -> None:
        """dynamic=False stores data that falls within the fixed range."""
        lat_r = (3 * _LAT_RES - math.pi / 2.0, 9 * _LAT_RES - math.pi / 2.0)
        lon_r = (8 * _LON_RES, 16 * _LON_RES)
        mosaic = BodyMosaic(
            body_name='MIMAS',
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
            lat_range=lat_r,
            lon_range=lon_r,
            dynamic=False,
        )
        mosaic.add(_make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=42.0))

        result = mosaic.to_full()
        assert pytest.approx(float(result.img.data[5, 10])) == 42.0

    def test_expansion_preserves_existing_data(self) -> None:
        """Expansion via dynamic=True does not corrupt previously stored data."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        mosaic.add(
            _make_repro(lat_range=(5, 7), lon_range=(10, 12), img_values=7.0, eff_res_values=1.0)
        )
        mosaic.add(
            _make_repro(lat_range=(8, 9), lon_range=(10, 12), img_values=3.0, eff_res_values=1.0)
        )

        result = mosaic.to_full()
        # Original data at bins (5,10) should still be 7.0
        assert pytest.approx(float(result.img.data[5, 10])) == 7.0
        # New data at bins (8,10) should be 3.0
        assert pytest.approx(float(result.img.data[8, 10])) == 3.0


# =========================================================================
# Longitude wraparound tests
# =========================================================================


class TestLongitudeWraparound:
    """Tests for longitude wraparound handling (0/2pi boundary)."""

    def test_wraparound_data_stores_correctly(self) -> None:
        """Data spanning the 0/2pi boundary is stored in the shifted buffer."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        # Add data near the end of the circle (last few bins)
        last_bins = _N_FULL_LON - 3  # e.g. 59 for N=62
        mosaic.add(
            _make_repro(
                lat_range=(5, 5),
                lon_range=(last_bins, _N_FULL_LON - 1),
                img_values=11.0,
                eff_res_values=1.0,
            )
        )
        # Add data near the beginning of the circle (first few bins)
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(0, 2), img_values=22.0, eff_res_values=1.0)
        )

        result = mosaic.to_full()
        # Data at last bins should be 11.0
        assert pytest.approx(float(result.img.data[5, last_bins])) == 11.0
        # Data at first bins should be 22.0
        assert pytest.approx(float(result.img.data[5, 0])) == 22.0

    def test_to_bounded_with_wraparound_range(self) -> None:
        """to_bounded() with a wrapping lon_range returns correct data."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        last_bins = _N_FULL_LON - 3
        mosaic.add(
            _make_repro(
                lat_range=(5, 5),
                lon_range=(last_bins, _N_FULL_LON - 1),
                img_values=11.0,
                eff_res_values=1.0,
            )
        )
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(0, 2), img_values=22.0, eff_res_values=1.0)
        )

        # Retrieve with a wrapping range
        lon_wrap_start = last_bins * _LON_RES
        lon_wrap_end = 2 * _LON_RES
        result = mosaic.to_bounded(lon_range=(lon_wrap_start, lon_wrap_end))
        # Should return 3 + 3 = 6 columns total
        assert result.img.shape == (1, 6)

    def test_second_add_extends_wraparound(self) -> None:
        """Second add extending a wrapped range expands array correctly."""
        mosaic = BodyMosaic(
            body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES, dynamic=True
        )
        last_bins = _N_FULL_LON - 2  # bins at 60, 61
        mosaic.add(
            _make_repro(
                lat_range=(5, 5),
                lon_range=(last_bins, _N_FULL_LON - 1),
                img_values=11.0,
                eff_res_values=1.0,
            )
        )
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(0, 1), img_values=22.0, eff_res_values=1.0)
        )
        # Extend further into the beginning
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(2, 3), img_values=33.0, eff_res_values=1.0)
        )

        result = mosaic.to_full()
        assert pytest.approx(float(result.img.data[5, 2])) == 33.0
        assert pytest.approx(float(result.img.data[5, 0])) == 22.0
        assert pytest.approx(float(result.img.data[5, _N_FULL_LON - 1])) == 11.0


# =========================================================================
# Metadata tests
# =========================================================================


class TestBodyMosaicMetadata:
    """Tests for metadata fields in BodyMosaicData (resolution, time, etc.)."""

    def test_resolution_stored_correctly(self) -> None:
        """Resolution values from add() appear in to_bounded() result."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        repro = _make_repro(
            lat_range=(5, 5), lon_range=(10, 10), res_values=2.5, eff_res_values=3.0
        )
        mosaic.add(repro)
        result = mosaic.to_bounded()
        assert pytest.approx(float(result.resolution.data[0, 0])) == 2.5

    def test_time_stored_correctly(self) -> None:
        """Time values from add() appear in to_bounded() result."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        repro = _make_repro(lat_range=(5, 5), lon_range=(10, 10), time=12345.0)
        mosaic.add(repro)
        result = mosaic.to_bounded()
        assert pytest.approx(float(result.time.data[0, 0])) == 12345.0

    def test_image_number_increments(self) -> None:
        """image_number increments on each non-overlapping add."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(10, 10), img_values=1.0, eff_res_values=1.0)
        )
        mosaic.add(
            _make_repro(lat_range=(5, 5), lon_range=(15, 15), img_values=2.0, eff_res_values=1.0)
        )

        result = mosaic.to_full()
        assert int(result.image_number.data[5, 10]) == 0
        assert int(result.image_number.data[5, 15]) == 1

    def test_photometric_model_name_in_data(self) -> None:
        """photometric_model_name is stored in BodyMosaicData."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10)))
        result = mosaic.to_bounded()
        assert result.photometric_model_name is None


class TestContributingImageNamesBodies:
    """Tests for ``contributing_image_names`` on body mosaics."""

    def test_two_adds_record_names_in_order(self) -> None:
        """Each add that bumps image_count appends repro.image_name."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10), image_name='obs_a'))
        mosaic.add(_make_repro(lat_range=(6, 6), lon_range=(15, 15), image_name='obs_b'))
        result = mosaic.to_bounded()
        assert result.contributing_image_names == ('obs_a', 'obs_b')

    def test_contributing_image_names_npz_roundtrip(self, tmp_path) -> None:
        """contributing_image_names survives BodyMosaicData npz save/load."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10), image_name='stem_x'))
        path = tmp_path / 'body_mosaic.npz'
        mosaic.to_bounded().save(path)
        loaded = BodyMosaicData.load(path)
        assert loaded.contributing_image_names == ('stem_x',)

    def test_contributing_image_names_fits_roundtrip(self, tmp_path) -> None:
        """contributing_image_names survives BodyMosaicData FITS save/load."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10), image_name='N888'))
        path = tmp_path / 'body_mosaic.fits'
        mosaic.to_bounded().save(path, format='fits')
        loaded = BodyMosaicData.load(path, format='fits')
        assert loaded.contributing_image_names == ('N888',)

    def test_load_body_file_passes_contributing_names(self, tmp_path) -> None:
        """BodyDisplayData from load_body_file includes contributing_image_names."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10), image_name='mimas_obs'))
        path = tmp_path / 'body_for_display.fits'
        mosaic.to_bounded().save(path, format='fits')
        dd = load_body_file(str(path))
        assert dd.is_mosaic is True
        assert dd.contributing_image_names == ('mimas_obs',)
