"""Tests for RingMosaic, RingReprojResult, and RingMosaicData.

Geometry-dependent tests (reproject()) are deferred to integration tests.
These unit tests cover static utilities, sparse mosaic accumulation, batched
insertion, merge strategies, and retrieval methods using synthetically
constructed RingReprojResult objects.
"""

import math
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pytest

from nav.reproj.photometric_model import LambertModel
from nav.reproj.ring_orbit_model import RingOrbitModel
from nav.reproj.rings import RingMosaic, RingMosaicData, RingMosaicMergeStrategy, RingReprojResult
from nav.ui.mosaic_viewer.common import load_ring_file

# Convenient resolution values for tests.
# pi/16 gives exactly 32 full-circle longitude bins (2*pi / (pi/16) = 32).
_LON_RES = math.pi / 16  # rad/pixel -- ~11.25 deg/pix
_RAD_RES = 5.0  # km/pixel
_RADIUS_INNER = 1000.0  # km
_RADIUS_OUTER = 1020.0  # km

# Derived constants
_N_FULL_LON = 32  # int(2*pi / _LON_RES) = int(32.0) = 32
_N_RADIUS = 5  # ceil((1020-1000 + slop) / 5.0) = 5


# =========================================================================
# Helpers
# =========================================================================


def _make_ring_repro(
    *,
    body_name: str = 'SATURN',
    longitude_resolution: float = _LON_RES,
    radius_resolution: float = _RAD_RES,
    radius_inner: float = _RADIUS_INNER,
    radius_outer: float = _RADIUS_OUTER,
    valid_lon_bins: list[int],
    img_values: float | np.ndarray = 1.0,
    mean_radial_resolution: float | np.ndarray = 10.0,
    mean_angular_resolution: float | np.ndarray = 0.001,
    mean_phase: float | np.ndarray = 0.5,
    mean_emission: float | np.ndarray = 0.3,
    incidence: float = 0.4,
    time: float = 0.0,
    n_full_lon: int = _N_FULL_LON,
    n_radius: int = _N_RADIUS,
    image_name: str = '',
) -> RingReprojResult:
    """Build a synthetic RingReprojResult for use in tests.

    Parameters:
        valid_lon_bins: Indices into the full-circle longitude array where
            this reprojection has data. Must be sorted.
        img_values: Scalar or array [n_radius, len(valid_lon_bins)] of pixel
            values for the image data.
        mean_radial_resolution: Scalar or 1-D array [len(valid_lon_bins)].
        n_full_lon: Total number of longitude bins in 0..2pi.
        n_radius: Number of radius bins.
        image_name: Label carried on the synthetic reprojection result.
    """
    n_valid = len(valid_lon_bins)

    antimask = np.zeros(n_full_lon, dtype=np.bool_)
    for b in valid_lon_bins:
        antimask[b] = True

    shape = (n_radius, n_valid)

    def _fill_2d(v: float | np.ndarray) -> ma.MaskedArray:
        if isinstance(v, ma.MaskedArray):
            return ma.MaskedArray(
                np.asarray(v.data, dtype=np.float32),
                mask=ma.getmaskarray(v),
            )
        if np.isscalar(v):
            arr = np.full(shape, v, dtype=np.float32)
        else:
            arr = np.asarray(v, dtype=np.float32)
        return ma.MaskedArray(arr)

    def _fill_1d(v: float | np.ndarray) -> np.ndarray:
        if np.isscalar(v):
            return np.full(n_valid, v, dtype=np.float32)
        return np.asarray(v, dtype=np.float32)

    return RingReprojResult(
        body_name=body_name,
        longitude_resolution=longitude_resolution,
        radius_resolution=radius_resolution,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        longitude_antimask=antimask,
        img=_fill_2d(img_values),
        mean_radial_resolution=_fill_1d(mean_radial_resolution),
        mean_angular_resolution=_fill_1d(mean_angular_resolution),
        mean_phase=_fill_1d(mean_phase),
        mean_emission=_fill_1d(mean_emission),
        incidence=incidence,
        time=time,
        orbit_model=None,
        image_dtype=np.dtype(np.float32),
        metadata_dtype=np.dtype(np.float32),
        image_name=image_name,
    )


# =========================================================================
# Static utility tests
# =========================================================================


class TestGenerateLongitudes:
    """Tests for RingMosaic.generate_longitudes."""

    def test_default_range_covers_full_circle(self) -> None:
        """Default range spans 0 to near 2*pi."""
        lons = RingMosaic.generate_longitudes(longitude_resolution=_LON_RES)
        assert lons[0] >= 0.0
        assert lons[-1] < 2.0 * math.pi

    def test_custom_range(self) -> None:
        """Custom range excludes values outside the specified limits."""
        lons = RingMosaic.generate_longitudes(
            longitude_start=1.0,
            longitude_end=2.0,
            longitude_resolution=_LON_RES,
        )
        assert lons[0] >= 1.0
        assert lons[-1] <= 2.0

    def test_values_on_grid_boundaries(self) -> None:
        """All returned longitudes are multiples of longitude_resolution."""
        lons = RingMosaic.generate_longitudes(longitude_resolution=_LON_RES)
        for lon in lons:
            residual = lon % _LON_RES
            assert residual < 1e-10 or abs(residual - _LON_RES) < 1e-10


class TestGenerateRadii:
    """Tests for RingMosaic.generate_radii."""

    def test_returns_correct_endpoints(self) -> None:
        """Returns values from inner to outer radius."""
        radii = RingMosaic.generate_radii(
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            radius_resolution=_RAD_RES,
        )
        assert radii[0] == pytest.approx(_RADIUS_INNER)
        assert radii[-1] <= _RADIUS_OUTER + _RAD_RES  # at most one step beyond

    def test_step_size_correct(self) -> None:
        """Step between successive radii equals radius_resolution."""
        radii = RingMosaic.generate_radii(
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            radius_resolution=_RAD_RES,
        )
        diffs = np.diff(radii)
        np.testing.assert_allclose(diffs, _RAD_RES)

    def test_length_matches_n_radius(self) -> None:
        """Number of radii matches the expected bin count."""
        radii = RingMosaic.generate_radii(
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            radius_resolution=_RAD_RES,
        )
        assert len(radii) == _N_RADIUS


# =========================================================================
# Constructor tests
# =========================================================================


class TestRingMosaicConstructor:
    """Tests for RingMosaic constructor parameter derivation and storage."""

    def test_ring_body_name_derived_from_planet(self) -> None:
        """ring_body_name is lowercased body_name + ':ring'."""
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )
        assert mosaic.ring_body_name == 'saturn:ring'

    def test_shadow_body_name_derived_from_planet(self) -> None:
        """shadow_body_name is the lowercased planet name."""
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )
        assert mosaic.shadow_body_name == 'saturn'

    def test_body_name_stored(self) -> None:
        """body_name attribute is preserved."""
        mosaic = RingMosaic(
            body_name='URANUS',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )
        assert mosaic.body_name == 'URANUS'

    def test_empty_mosaic_antimask_all_false(self) -> None:
        """A freshly created mosaic has an all-False longitude antimask."""
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )
        result = mosaic.to_sparse()
        assert not np.any(result.longitude_antimask)

    def test_empty_mosaic_has_none_bounds(self) -> None:
        """Bounds is None when no data has been added."""
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )
        assert mosaic.bounds is None


# =========================================================================
# Sparse add and retrieval tests
# =========================================================================


class TestRingMosaicAddAndRetrieve:
    """Tests for add(), to_sparse(), to_bounded(), and to_full()."""

    def _make_mosaic(self) -> RingMosaic:
        """Create and return a RingMosaic for SATURN with predefined radii and resolution."""
        return RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )

    def test_add_single_result_updates_antimask(self) -> None:
        """After adding one result, antimask is True at valid longitudes."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))

        result = mosaic.to_sparse()
        assert result.longitude_antimask[5]
        assert result.longitude_antimask[6]
        assert result.longitude_antimask[7]
        assert not result.longitude_antimask[4]
        assert not result.longitude_antimask[8]

    def test_to_sparse_img_shape(self) -> None:
        """to_sparse() img has shape [n_radius, n_valid_longitudes]."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        result = mosaic.to_sparse()
        assert result.img.shape == (_N_RADIUS, 3)

    def test_to_sparse_img_values_correct(self) -> None:
        """Data values from add() appear correctly in to_sparse()."""
        mosaic = self._make_mosaic()
        img = np.arange(_N_RADIUS * 3, dtype=np.float32).reshape(_N_RADIUS, 3) + 1.0
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7], img_values=img))
        result = mosaic.to_sparse()
        np.testing.assert_allclose(result.img.data, img)

    def test_to_full_shape(self) -> None:
        """to_full() returns array with shape [n_radius, n_full_longitude_bins]."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        result = mosaic.to_full()
        assert result.img.shape == (_N_RADIUS, _N_FULL_LON)

    def test_to_full_data_at_correct_position(self) -> None:
        """Data from add() appears at the correct longitude position in to_full()."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7], img_values=42.0))
        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5:8], 42.0)

    def test_to_full_invalid_longitudes_are_masked(self) -> None:
        """Longitudes with no data are masked in to_full()."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        result = mosaic.to_full()
        # Longitude bin 0 should be entirely masked
        assert np.all(ma.getmaskarray(result.img)[:, 0])
        # Longitude bins 5-7 should have data
        assert not np.any(ma.getmaskarray(result.img)[:, 5:8])

    def test_to_bounded_shape(self) -> None:
        """to_bounded() with an explicit range returns the correct column count."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        lon_start = 5 * _LON_RES
        lon_end = 7 * _LON_RES
        result = mosaic.to_bounded(longitude_range=(lon_start, lon_end))
        assert result.img.shape == (_N_RADIUS, 3)

    def test_to_bounded_data_correct(self) -> None:
        """to_bounded() returns correct data values for the requested range."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7], img_values=77.0))
        lon_start = 5 * _LON_RES
        lon_end = 7 * _LON_RES
        result = mosaic.to_bounded(longitude_range=(lon_start, lon_end))
        np.testing.assert_allclose(result.img.data, 77.0)

    def test_to_bounded_empty_columns_masked(self) -> None:
        """Columns with no data are masked when included in to_bounded()."""
        mosaic = self._make_mosaic()
        # Only bins 5 and 7 have data (bin 6 is absent)
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 7], img_values=1.0))
        lon_start = 5 * _LON_RES
        lon_end = 7 * _LON_RES
        result = mosaic.to_bounded(longitude_range=(lon_start, lon_end))
        assert result.img.shape == (_N_RADIUS, 3)
        # Column 1 (bin 6) should be masked
        assert np.all(ma.getmaskarray(result.img)[:, 1])
        # Columns 0 and 2 should have data
        assert not np.any(ma.getmaskarray(result.img)[:, 0])
        assert not np.any(ma.getmaskarray(result.img)[:, 2])

    def test_bounds_correct_after_add(self) -> None:
        """bounds property returns correct longitude range after add."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        bounds = mosaic.bounds
        assert bounds is not None
        lon_min, lon_max = bounds
        assert lon_min == pytest.approx(5 * _LON_RES)
        assert lon_max == pytest.approx(7 * _LON_RES)

    def test_add_result_returns_masked_array(self) -> None:
        """to_sparse() img is a MaskedArray."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        result = mosaic.to_sparse()
        assert isinstance(result.img, ma.MaskedArray)


# =========================================================================
# Sparse growth via batched insert tests
# =========================================================================


class TestSparseGrowth:
    """Tests for batched np.insert-based sparse expansion when adding new lons."""

    def _make_mosaic(self) -> RingMosaic:
        return RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )

    def test_second_add_extends_sparse_antimask(self) -> None:
        """Adding a second result with new longitudes extends the antimask."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        mosaic.add(_make_ring_repro(valid_lon_bins=[7, 8, 9]))

        result = mosaic.to_sparse()
        for b in [5, 6, 7, 8, 9]:
            assert result.longitude_antimask[b], f'bin {b} should be True'
        assert result.img.shape == (_N_RADIUS, 5)

    def test_second_add_preserves_original_data(self) -> None:
        """Data from the first add() is intact after a second add()."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7], img_values=11.0))
        mosaic.add(_make_ring_repro(valid_lon_bins=[8, 9], img_values=22.0))

        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5:8], 11.0)

    def test_second_add_new_longitudes_correct_values(self) -> None:
        """Data from the second add() appears at the correct new positions."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6], img_values=11.0))
        mosaic.add(_make_ring_repro(valid_lon_bins=[9, 10], img_values=22.0))

        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 9:11], 22.0)

    def test_add_disjoint_then_overlapping(self) -> None:
        """Adding disjoint then overlapping results correctly expands storage."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[2, 3]))
        mosaic.add(_make_ring_repro(valid_lon_bins=[10, 11]))
        mosaic.add(_make_ring_repro(valid_lon_bins=[3, 4, 5, 10]))

        result = mosaic.to_sparse()
        for b in [2, 3, 4, 5, 10, 11]:
            assert result.longitude_antimask[b]
        assert result.img.shape == (_N_RADIUS, 6)

    def test_add_all_existing_longitudes_does_not_grow(self) -> None:
        """Adding only existing longitudes does not increase sparse width."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6, 7]))

        result = mosaic.to_sparse()
        assert result.img.shape == (_N_RADIUS, 3)


# =========================================================================
# Merge strategy tests
# =========================================================================


class TestRingMergeStrategies:
    """Tests for BEST_RESOLUTION and MOST_COVERAGE_THEN_RESOLUTION merge strategies."""

    def _make_mosaic(
        self,
        strategy: RingMosaicMergeStrategy = RingMosaicMergeStrategy.BEST_RESOLUTION,
    ) -> RingMosaic:
        """Create and return a RingMosaic for SATURN with predefined radii and resolution."""
        return RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
            merge_strategy=strategy,
        )

    def test_best_resolution_better_resolution_wins(self) -> None:
        """BEST_RESOLUTION: new column replaces when mean radial res is better."""
        mosaic = self._make_mosaic(RingMosaicMergeStrategy.BEST_RESOLUTION)
        mosaic.add(
            _make_ring_repro(valid_lon_bins=[5], img_values=1.0, mean_radial_resolution=20.0)
        )
        mosaic.add(_make_ring_repro(valid_lon_bins=[5], img_values=9.0, mean_radial_resolution=5.0))
        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5], 9.0)

    def test_best_resolution_worse_resolution_does_not_win(self) -> None:
        """BEST_RESOLUTION: existing data kept when new data has worse resolution."""
        mosaic = self._make_mosaic(RingMosaicMergeStrategy.BEST_RESOLUTION)
        mosaic.add(_make_ring_repro(valid_lon_bins=[5], img_values=1.0, mean_radial_resolution=5.0))
        mosaic.add(
            _make_ring_repro(valid_lon_bins=[5], img_values=9.0, mean_radial_resolution=20.0)
        )
        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5], 1.0)

    def test_most_coverage_fills_sparse_column_with_more_valid_radii(self) -> None:
        """MOST_COVERAGE: column with more valid radii wins regardless of resolution."""
        mosaic = self._make_mosaic(RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION)

        # First: all radii masked except 2
        img1 = ma.MaskedArray(
            np.ones((_N_RADIUS, 1), dtype=np.float32) * 1.0,
            mask=np.array([[True], [True], [True], [False], [False]]),
        )
        mosaic.add(
            _make_ring_repro(valid_lon_bins=[5], img_values=img1, mean_radial_resolution=5.0)
        )

        # Second: all radii valid (better coverage), worse resolution
        mosaic.add(
            _make_ring_repro(valid_lon_bins=[5], img_values=9.0, mean_radial_resolution=20.0)
        )
        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5], 9.0)

    def test_most_coverage_same_valid_radii_resolution_tiebreaker(self) -> None:
        """MOST_COVERAGE: when valid radii equal, better resolution wins."""
        mosaic = self._make_mosaic(RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION)
        mosaic.add(
            _make_ring_repro(valid_lon_bins=[5], img_values=1.0, mean_radial_resolution=20.0)
        )
        mosaic.add(_make_ring_repro(valid_lon_bins=[5], img_values=9.0, mean_radial_resolution=5.0))
        result = mosaic.to_full()
        np.testing.assert_allclose(result.img.data[:, 5], 9.0)


# =========================================================================
# Metadata tests
# =========================================================================


class TestRingMosaicMetadata:
    """Tests for per-longitude metadata fields (resolution, time, image_number)."""

    def _make_mosaic(self) -> RingMosaic:
        return RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )

    def test_mean_radial_resolution_stored(self) -> None:
        """mean_radial_resolution values from add() appear in to_sparse()."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6], mean_radial_resolution=7.5))
        result = mosaic.to_sparse()
        np.testing.assert_allclose(result.mean_radial_resolution, 7.5)

    def test_time_stored_correctly(self) -> None:
        """time values from add() appear correctly per longitude column."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6], time=12345.0))
        result = mosaic.to_full()
        assert result.time.data[5] == pytest.approx(12345.0)
        assert result.time.data[6] == pytest.approx(12345.0)

    def test_image_number_increments_on_new_columns(self) -> None:
        """image_number reflects which add() call contributed each longitude."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5], img_values=1.0, mean_radial_resolution=1.0))
        mosaic.add(_make_ring_repro(valid_lon_bins=[9], img_values=2.0, mean_radial_resolution=1.0))
        result = mosaic.to_full()
        assert int(result.image_number.data[5]) == 0
        assert int(result.image_number.data[9]) == 1

    def test_body_name_in_result(self) -> None:
        """body_name is preserved in the mosaic data result."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5]))
        result = mosaic.to_sparse()
        assert result.body_name == 'SATURN'

    def test_ring_body_name_in_result(self) -> None:
        """ring_body_name appears in the mosaic data result."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5]))
        result = mosaic.to_sparse()
        assert result.ring_body_name == 'saturn:ring'


# =========================================================================
# Contributing image names
# =========================================================================


class TestContributingImageNamesRings:
    """Tests for ``contributing_image_names`` on ring mosaics."""

    def _make_mosaic(self) -> RingMosaic:
        """Return a small ``RingMosaic`` for SATURN using module test constants.

        Uses ``_RADIUS_INNER``, ``_RADIUS_OUTER``, ``_LON_RES``, and ``_RAD_RES`` for
        the radial/longitude grid; no orbit model or photometry.

        Returns:
            A fresh :class:`~nav.reproj.rings.RingMosaic` instance.
        """
        return RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
        )

    def test_two_adds_record_names_in_order(self) -> None:
        """Each successful add appends repro.image_name to the mosaic list."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[1, 2], image_name='cassini_001'))
        mosaic.add(_make_ring_repro(valid_lon_bins=[10, 11], image_name='cassini_002'))
        data = mosaic.to_sparse()
        assert data.contributing_image_names == ('cassini_001', 'cassini_002')

    def test_contributing_image_names_npz_roundtrip(self, tmp_path: Path) -> None:
        """contributing_image_names survives RingMosaicData npz save/load."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[5, 6], image_name='stem_a'))
        path = tmp_path / 'ring_mosaic.npz'
        mosaic.to_sparse().save(path)
        loaded = RingMosaicData.load(path)
        assert loaded.contributing_image_names == ('stem_a',)

    def test_contributing_image_names_fits_roundtrip(self, tmp_path: Path) -> None:
        """contributing_image_names survives RingMosaicData FITS save/load."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[4], image_name='N999'))
        path = tmp_path / 'ring_mosaic.fits'
        mosaic.to_sparse().save(path, format='fits')
        loaded = RingMosaicData.load(path, format='fits')
        assert loaded.contributing_image_names == ('N999',)

    def test_load_ring_file_passes_contributing_names(self, tmp_path: Path) -> None:
        """RingDisplayData from load_ring_file includes contributing_image_names."""
        mosaic = self._make_mosaic()
        mosaic.add(_make_ring_repro(valid_lon_bins=[3], image_name='cassini_img'))
        path = tmp_path / 'for_display.fits'
        mosaic.to_sparse().save(path, format='fits')
        dd = load_ring_file(str(path))
        assert dd.is_mosaic is True
        assert dd.contributing_image_names == ('cassini_img',)

    def test_orbit_model_name_npz_roundtrip(self, tmp_path: Path) -> None:
        """RingMosaicData saves and restores orbit_model_name for display metadata."""
        model = RingOrbitModel(
            name='test_ring_model',
            a=100000.0,
            e=0.0,
            w0=0.0,
            dw=0.0,
            mean_motion=1.0,
            epoch_utc='2000-01-01T12:00:00',
        )
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
            orbit_model=model,
        )
        mosaic.add(_make_ring_repro(valid_lon_bins=[5]))
        path = tmp_path / 'with_om.npz'
        mosaic.to_sparse().save(path)
        loaded = RingMosaicData.load(path)
        assert loaded.orbit_model_name == 'test_ring_model'

        dd = load_ring_file(str(path))
        assert dd.orbit_model_name == 'test_ring_model'

    def test_photometric_model_name_npz_roundtrip(self, tmp_path: Path) -> None:
        """RingMosaicData saves photometric_model_name; load_ring_file exposes it."""
        mosaic = RingMosaic(
            body_name='SATURN',
            radius_inner=_RADIUS_INNER,
            radius_outer=_RADIUS_OUTER,
            longitude_resolution=_LON_RES,
            radius_resolution=_RAD_RES,
            photometric_model=LambertModel(),
        )
        mosaic.add(_make_ring_repro(valid_lon_bins=[5]))
        path = tmp_path / 'with_ph.npz'
        mosaic.to_sparse().save(path)
        loaded = RingMosaicData.load(path)
        assert loaded.photometric_model_name == 'lambert'
        dd = load_ring_file(str(path))
        assert dd.photometric_model_name == 'lambert'

    def test_load_ring_file_longitude_origin_sparse_reproj(self, tmp_path: Path) -> None:
        """Sparse ring reproj columns map to global bins, not longitude starting at 0."""
        repro = _make_ring_repro(valid_lon_bins=[10, 11, 12])
        path = tmp_path / 'lon_sparse.npz'
        repro.save(path)
        dd = load_ring_file(str(path))
        step_deg = _LON_RES * 180.0 / math.pi
        assert dd.longitude_column_origin_deg == pytest.approx(10 * step_deg)
        assert dd.longitude_extent_hi_deg == pytest.approx(10 * step_deg + 3 * step_deg)
