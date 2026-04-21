"""Tests for create_cartographic_model and CartographicModelResult.

Geometry-dependent coordinate-mapping tests use a mocked oops.backplane.Backplane
to avoid requiring a real observation. Full end-to-end tests are deferred to
integration tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.ma as ma
import pytest

from nav.reproj.bodies import BodyMosaicData
from nav.reproj.cartographic_model import CartographicModelResult, create_cartographic_model

_LAT_RES = 0.1  # rad/pixel
_LON_RES = 0.1  # rad/pixel


# =========================================================================
# Helpers
# =========================================================================


def _make_mosaic(
    n_lat: int = 5,
    n_lon: int = 10,
    *,
    lat_min: float = -0.2,
    lon_min: float = 0.5,
    img_values: float | np.ndarray = 0.5,
    mask_all: bool = False,
    eff_res: float = 2.0,
) -> BodyMosaicData:
    """Build a synthetic BodyMosaicData for testing."""
    shape = (n_lat, n_lon)
    lat_max = lat_min + (n_lat - 1) * _LAT_RES
    lon_max = lon_min + (n_lon - 1) * _LON_RES

    if np.isscalar(img_values):
        scalar_val = img_values if isinstance(img_values, float) else float(img_values)  # type: ignore[arg-type]
        img_data = np.full(shape, scalar_val, dtype=np.float32)
    else:
        img_data = np.asarray(img_values, dtype=np.float32)

    img_mask = np.ones(shape, dtype=bool) if mask_all else np.zeros(shape, dtype=bool)
    data_mask = np.zeros(shape, dtype=bool)

    return BodyMosaicData(
        img=ma.MaskedArray(img_data, mask=img_mask),
        lat_range=(lat_min, lat_max),
        lon_range=(lon_min, lon_max),
        resolution=ma.MaskedArray(np.ones(shape, dtype=np.float32), mask=data_mask),
        eff_resolution=ma.MaskedArray(np.full(shape, eff_res, dtype=np.float32), mask=data_mask),
        phase=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
        emission=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
        incidence=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
        image_number=ma.MaskedArray(np.zeros(shape, dtype=np.int32), mask=data_mask),
        time=ma.MaskedArray(np.zeros(shape, dtype=np.float64), mask=data_mask),
        lat_resolution=_LAT_RES,
        lon_resolution=_LON_RES,
        photometric_model_name=None,
    )


def _setup_mock_backplane(
    mock_bp: MagicMock,
    img_shape: tuple[int, int],
    mosaic: BodyMosaicData,
    *,
    lat_row: int = 2,
    lon_col: int = 3,
    center_res: float = 2.0,
) -> None:
    """Configure mock backplane to map every image pixel to (lat_row, lon_col) in mosaic."""
    lat_val = mosaic.lat_range[0] + lat_row * mosaic.lat_resolution
    lat_arr = np.full(img_shape, lat_val, dtype=np.float64)
    mock_lat = MagicMock()
    mock_lat.mvals = ma.MaskedArray(lat_arr, mask=False)
    mock_bp.latitude.return_value = mock_lat

    lon_val = mosaic.lon_range[0] + lon_col * mosaic.lon_resolution
    lon_arr = np.full(img_shape, lon_val, dtype=np.float64)
    mock_lon = MagicMock()
    mock_lon.mvals = ma.MaskedArray(lon_arr, mask=False)
    mock_bp.longitude.return_value = mock_lon

    mock_bp.center_resolution.return_value.vals = center_res


# =========================================================================
# None return
# =========================================================================


class TestReturnNone:
    def test_returns_none_when_mosaic_all_masked(self) -> None:
        """Returns None immediately if every mosaic pixel is masked."""
        mosaic = _make_mosaic(mask_all=True)
        obs = MagicMock()
        result = create_cartographic_model(mosaic, obs, body_name='MIMAS')
        assert result is None

    def test_does_not_call_backplane_when_mosaic_all_masked(self) -> None:
        """No backplane is created when mosaic has no valid data."""
        mosaic = _make_mosaic(mask_all=True)
        obs = MagicMock()
        with patch('oops.backplane.Backplane') as mock_bp_class:
            create_cartographic_model(mosaic, obs, body_name='MIMAS')
            mock_bp_class.assert_not_called()


# =========================================================================
# Model image shape and values
# =========================================================================


class TestModelImage:
    def test_model_img_shape_matches_backplane(self) -> None:
        """Output model shape matches the backplane latitude array shape."""
        mosaic = _make_mosaic()
        obs = MagicMock()
        img_shape = (6, 8)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            _setup_mock_backplane(mock_bp, img_shape, mosaic)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert result.model_img.shape == img_shape

    def test_model_img_samples_mosaic_value(self) -> None:
        """Each image pixel receives the mosaic value at its mapped lat/lon."""
        n_lat, n_lon = 5, 10
        img_data = np.zeros((n_lat, n_lon), dtype=np.float32)
        img_data[2, 3] = 0.8  # distinctive value at row 2, col 3
        mosaic = _make_mosaic(n_lat=n_lat, n_lon=n_lon, img_values=img_data)
        obs = MagicMock()
        img_shape = (4, 5)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            # Map every image pixel to mosaic (row=2, col=3)
            _setup_mock_backplane(mock_bp, img_shape, mosaic, lat_row=2, lon_col=3)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert pytest.approx(float(result.model_img[0, 0]), abs=1e-4) == 0.8

    def test_model_img_is_zero_where_body_mask(self) -> None:
        """Pixels with the body surface masked out have model value 0."""
        mosaic = _make_mosaic(img_values=1.0)
        obs = MagicMock()
        img_shape = (3, 4)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            # Latitude backplane is fully masked (body not visible)
            lat_arr = np.zeros(img_shape, dtype=np.float64)
            mock_lat = MagicMock()
            mock_lat.mvals = ma.MaskedArray(lat_arr, mask=True)
            mock_bp.latitude.return_value = mock_lat

            lon_arr = np.zeros(img_shape, dtype=np.float64)
            mock_lon = MagicMock()
            mock_lon.mvals = ma.MaskedArray(lon_arr, mask=False)
            mock_bp.longitude.return_value = mock_lon
            mock_bp.center_resolution.return_value.vals = 1.0

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert np.all(result.model_img == 0.0)

    def test_model_img_is_zero_outside_mosaic_bounds(self) -> None:
        """Image pixels whose lat/lon fall outside the mosaic return 0."""
        mosaic = _make_mosaic(img_values=1.0)
        obs = MagicMock()
        img_shape = (3, 4)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            # Latitude is far outside the mosaic lat range
            lat_arr = np.full(img_shape, mosaic.lat_range[1] + 1.0, dtype=np.float64)
            mock_lat = MagicMock()
            mock_lat.mvals = ma.MaskedArray(lat_arr, mask=False)
            mock_bp.latitude.return_value = mock_lat

            lon_arr = np.full(img_shape, mosaic.lon_range[0], dtype=np.float64)
            mock_lon = MagicMock()
            mock_lon.mvals = ma.MaskedArray(lon_arr, mask=False)
            mock_bp.longitude.return_value = mock_lon
            mock_bp.center_resolution.return_value.vals = 1.0

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert np.all(result.model_img == 0.0)


# =========================================================================
# Resolution ratio
# =========================================================================


class TestResolutionRatio:
    def test_resolution_ratio_equal_resolution(self) -> None:
        """Resolution ratio is 1.0 when mosaic and image resolutions match."""
        mosaic = _make_mosaic(eff_res=3.0)
        obs = MagicMock()
        img_shape = (4, 5)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            _setup_mock_backplane(mock_bp, img_shape, mosaic, center_res=3.0)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert pytest.approx(result.resolution_ratio, abs=1e-5) == 1.0

    def test_resolution_ratio_coarser_mosaic(self) -> None:
        """Resolution ratio > 1 when mosaic is coarser than the image."""
        mosaic = _make_mosaic(eff_res=4.0)
        obs = MagicMock()
        img_shape = (4, 5)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            _setup_mock_backplane(mock_bp, img_shape, mosaic, center_res=2.0)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert pytest.approx(result.resolution_ratio, abs=1e-5) == 2.0

    def test_resolution_ratio_finer_mosaic(self) -> None:
        """Resolution ratio < 1 when mosaic is finer than the image."""
        mosaic = _make_mosaic(eff_res=1.0)
        obs = MagicMock()
        img_shape = (4, 5)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            _setup_mock_backplane(mock_bp, img_shape, mosaic, center_res=4.0)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert pytest.approx(result.resolution_ratio, abs=1e-5) == 0.25

    def test_resolution_ratio_is_one_when_eff_resolution_all_masked(self) -> None:
        """Resolution ratio defaults to 1.0 when mosaic eff_resolution is all masked."""
        n_lat, n_lon = 5, 10
        shape = (n_lat, n_lon)
        data_mask = np.zeros(shape, dtype=bool)
        img_mask = np.zeros(shape, dtype=bool)
        mosaic = BodyMosaicData(
            img=ma.MaskedArray(np.ones(shape, dtype=np.float32), mask=img_mask),
            lat_range=(-0.2, -0.2 + (n_lat - 1) * _LAT_RES),
            lon_range=(0.5, 0.5 + (n_lon - 1) * _LON_RES),
            resolution=ma.MaskedArray(np.ones(shape, dtype=np.float32), mask=data_mask),
            eff_resolution=ma.MaskedArray(
                np.ones(shape, dtype=np.float32), mask=np.ones(shape, dtype=bool)
            ),
            phase=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
            emission=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
            incidence=ma.MaskedArray(np.zeros(shape, dtype=np.float32), mask=data_mask),
            image_number=ma.MaskedArray(np.zeros(shape, dtype=np.int32), mask=data_mask),
            time=ma.MaskedArray(np.zeros(shape, dtype=np.float64), mask=data_mask),
            lat_resolution=_LAT_RES,
            lon_resolution=_LON_RES,
            photometric_model_name=None,
        )
        obs = MagicMock()
        img_shape = (4, 5)

        with patch('oops.backplane.Backplane') as mock_bp_class:
            mock_bp = MagicMock()
            mock_bp_class.return_value = mock_bp
            _setup_mock_backplane(mock_bp, img_shape, mosaic)

            result = create_cartographic_model(mosaic, obs, body_name='MIMAS')

        assert result is not None
        assert pytest.approx(result.resolution_ratio, abs=1e-5) == 1.0


# =========================================================================
# CartographicModelResult dataclass
# =========================================================================


class TestCartographicModelResult:
    def test_frozen_dataclass(self) -> None:
        """CartographicModelResult is immutable (frozen dataclass)."""
        img = np.zeros((3, 4), dtype=np.float32)
        result = CartographicModelResult(model_img=img, resolution_ratio=1.5)
        with pytest.raises(AttributeError):
            result.resolution_ratio = 2.0  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """Result fields are accessible as expected."""
        img = np.ones((3, 4), dtype=np.float32) * 0.7
        result = CartographicModelResult(model_img=img, resolution_ratio=2.3)
        assert result.model_img.shape == (3, 4)
        assert pytest.approx(result.resolution_ratio, abs=1e-5) == 2.3
