"""Spec-first contract tests for create_cartographic_model.

Contracts under test come from the docstrings in
``src/spindoctor/reproj/cartographic_model.py`` and the "Cartographic model
projection" section of ``docs/dev_guide/dev_guide_reprojection.rst``: for each
image pixel the mosaic is sampled at the pixel's lat/lon via bilinear
interpolation, ``col = ((bp_longitude - lon_min) % (2 * pi)) / lon_resolution``
handles longitude wraparound, pixels off the body or outside the mosaic are 0.0,
and ``resolution_ratio`` is the median mosaic effective resolution over the
image-center resolution (falling back to 1.0 on degenerate inputs).

All geometry is supplied through a fake ``oops.backplane.Backplane`` so the
tests run hermetically (no SPICE kernels, no holdings).
"""

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pytest

from spindoctor.reproj.bodies import BodyMosaicData
from spindoctor.reproj.cartographic_model import create_cartographic_model

_LAT_RES = 0.1  # rad/pixel
_LON_RES = 0.1  # rad/pixel
_TWO_PI = 2.0 * math.pi


def _make_mosaic_data(
    img: npt.NDArray[np.float64],
    *,
    img_mask: npt.NDArray[np.bool_] | bool = False,
    lat_min: float = -0.2,
    lon_min: float = 0.5,
    eff_res: npt.NDArray[np.float64] | float = 2.0,
    eff_res_mask: npt.NDArray[np.bool_] | bool = False,
) -> BodyMosaicData:
    """Build a synthetic BodyMosaicData around a given image array.

    Parameters:
        img: Mosaic image array of shape (n_lat, n_lon).
        img_mask: Mask for the mosaic image (True = no data).
        lat_min: Latitude of mosaic row 0 (rad).
        lon_min: Longitude of mosaic column 0 (rad); the grid may extend past 2*pi.
        eff_res: Scalar or per-cell effective resolution (km/pixel).
        eff_res_mask: Mask for the effective-resolution array.
    """
    n_lat, n_lon = img.shape
    shape = (n_lat, n_lon)
    zeros = np.zeros(shape, dtype=np.float32)
    eff = np.broadcast_to(np.asarray(eff_res, dtype=np.float32), shape)
    return BodyMosaicData(
        body_name='MIMAS',
        img=ma.MaskedArray(np.asarray(img, dtype=np.float64), mask=img_mask),
        lat_resolution=_LAT_RES,
        lon_resolution=_LON_RES,
        lat_range=(lat_min, lat_min + (n_lat - 1) * _LAT_RES),
        lon_range=(lon_min, lon_min + (n_lon - 1) * _LON_RES),
        latlon_type='centric',
        lon_direction='east',
        resolution=ma.MaskedArray(np.ones(shape, dtype=np.float32)),
        eff_resolution=ma.MaskedArray(eff.copy(), mask=eff_res_mask),
        phase=ma.MaskedArray(zeros.copy()),
        emission=ma.MaskedArray(zeros.copy()),
        incidence=ma.MaskedArray(zeros.copy()),
        time=ma.MaskedArray(np.zeros(shape, dtype=np.float64)),
        image_number=ma.MaskedArray(np.zeros(shape, dtype=np.uint16)),
        photometric_model_name=None,
        image_dtype=np.dtype(np.float64),
        metadata_dtype=np.dtype(np.float32),
    )


class _CartoBackplane:
    """Backplane stand-in returning explicit lat/lon arrays for the image."""

    def __init__(
        self,
        latitude: npt.NDArray[np.float64],
        longitude: npt.NDArray[np.float64],
        *,
        lat_mask: npt.NDArray[np.bool_] | bool = False,
        lon_mask: npt.NDArray[np.bool_] | bool = False,
        center_res: float = 2.0,
    ) -> None:
        """Store the fake backplane arrays.

        Parameters:
            latitude: Per-pixel body latitude (rad), image shaped.
            longitude: Per-pixel body longitude (rad), image shaped.
            lat_mask: Mask for the latitude backplane (True = off body).
            lon_mask: Mask for the longitude backplane.
            center_res: Scalar image-center resolution (km/pixel).
        """
        self._lat = ma.MaskedArray(np.asarray(latitude, dtype=np.float64), mask=lat_mask)
        self._lon = ma.MaskedArray(np.asarray(longitude, dtype=np.float64), mask=lon_mask)
        self._center_res = center_res

    def latitude(self, name: str, lat_type: str = 'centric') -> SimpleNamespace:
        """Return the latitude backplane wrapper.

        Parameters:
            name: Body name (ignored).
            lat_type: Latitude type (ignored).
        """
        return SimpleNamespace(mvals=self._lat)

    def longitude(
        self, name: str, direction: str = 'east', lon_type: str = 'centric'
    ) -> SimpleNamespace:
        """Return the longitude backplane wrapper.

        Parameters:
            name: Body name (ignored).
            direction: Longitude direction (ignored).
            lon_type: Longitude type (ignored).
        """
        return SimpleNamespace(mvals=self._lon)

    def center_resolution(self, name: str) -> SimpleNamespace:
        """Return the scalar center-resolution wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._center_res)


def _run(
    mosaic: BodyMosaicData,
    bp: _CartoBackplane,
) -> tuple[npt.NDArray[np.float32], float]:
    """Run create_cartographic_model against a fake backplane.

    Parameters:
        mosaic: Mosaic data to project.
        bp: Fake backplane supplying the image geometry.
    """
    with patch('oops.backplane.Backplane', new=lambda obs: bp):
        result = create_cartographic_model(mosaic, object(), body_name='MIMAS')
    assert result is not None
    return result.model_img, result.resolution_ratio


def _cell_coords(
    mosaic: BodyMosaicData,
    row: float,
    col: float,
    shape: tuple[int, int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return constant lat/lon image arrays targeting a fractional mosaic cell.

    Parameters:
        mosaic: Mosaic whose grid defines the coordinate mapping.
        row: Fractional mosaic row to target.
        col: Fractional mosaic column to target.
        shape: Image shape for the returned arrays.
    """
    lat = np.full(shape, mosaic.lat_range[0] + row * _LAT_RES)
    lon = np.full(shape, mosaic.lon_range[0] + col * _LON_RES)
    return lat, lon


# =========================================================================
# Input validation
# =========================================================================


class TestValidation:
    """Argument validation happens before any backplane is created."""

    def test_non_string_body_name_raises_type_error(self) -> None:
        """A non-str body_name raises TypeError."""
        mosaic = _make_mosaic_data(np.ones((2, 2)))
        with pytest.raises(TypeError, match='body_name must be str'):
            create_cartographic_model(mosaic, object(), body_name=123)  # type: ignore[arg-type]

    def test_empty_body_name_raises_value_error(self) -> None:
        """A blank body_name raises ValueError."""
        mosaic = _make_mosaic_data(np.ones((2, 2)))
        with pytest.raises(ValueError, match='body_name must be a non-empty string'):
            create_cartographic_model(mosaic, object(), body_name='   ')

    def test_bad_latlon_type_raises_value_error(self) -> None:
        """An unknown latlon_type raises ValueError."""
        mosaic = _make_mosaic_data(np.ones((2, 2)))
        with pytest.raises(ValueError, match="latlon_type must be 'centric'"):
            create_cartographic_model(
                mosaic,
                object(),
                body_name='MIMAS',
                latlon_type='polar',  # type: ignore[arg-type]
            )

    def test_bad_lon_direction_raises_value_error(self) -> None:
        """An unknown lon_direction raises ValueError."""
        mosaic = _make_mosaic_data(np.ones((2, 2)))
        with pytest.raises(ValueError, match="lon_direction must be 'east' or 'west'"):
            create_cartographic_model(
                mosaic,
                object(),
                body_name='MIMAS',
                lon_direction='up',  # type: ignore[arg-type]
            )


# =========================================================================
# Sampling and interpolation
# =========================================================================


class TestSampling:
    """Bilinear sampling of the mosaic at each image pixel's lat/lon."""

    def test_exact_cell_returns_cell_value(self) -> None:
        """A pixel mapping exactly onto a mosaic cell receives that cell's value."""
        img = np.zeros((4, 6))
        img[2, 3] = 0.8
        mosaic = _make_mosaic_data(img)
        lat, lon = _cell_coords(mosaic, 2.0, 3.0, (3, 5))
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 0.8, rtol=1e-6)

    def test_midpoint_between_rows_is_average(self) -> None:
        """A pixel halfway between two rows receives the average of both values."""
        img = np.zeros((4, 6))
        img[1, 3] = 2.0
        img[2, 3] = 4.0
        mosaic = _make_mosaic_data(img)
        lat, lon = _cell_coords(mosaic, 1.5, 3.0, (2, 2))
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 3.0, rtol=1e-6)

    def test_midpoint_between_columns_is_average(self) -> None:
        """A pixel halfway between two columns receives the average of both values."""
        img = np.zeros((4, 6))
        img[2, 3] = 2.0
        img[2, 4] = 6.0
        mosaic = _make_mosaic_data(img)
        lat, lon = _cell_coords(mosaic, 2.0, 3.5, (2, 2))
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 4.0, rtol=1e-6)

    def test_masked_mosaic_cell_contributes_zero(self) -> None:
        """Masked mosaic cells enter the interpolation as 0.0."""
        img = np.full((4, 6), 2.0)
        img_mask = np.zeros((4, 6), dtype=bool)
        img_mask[2, 3] = True
        mosaic = _make_mosaic_data(img, img_mask=img_mask)
        lat, lon = _cell_coords(mosaic, 2.0, 2.5, (2, 2))  # between valid col 2 and masked col 3
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 1.0, rtol=1e-6)

    def test_longitude_wraparound_samples_across_two_pi(self) -> None:
        """Longitudes expressed modulo 2*pi map onto a mosaic extending past 2*pi."""
        img = np.zeros((4, 6))
        img[1, 4] = 5.0
        mosaic = _make_mosaic_data(img, lon_min=6.0)  # columns span 6.0..6.5 rad, past 2*pi
        lat = np.full((2, 2), mosaic.lat_range[0] + 1.0 * _LAT_RES)
        lon = np.full((2, 2), (6.0 + 4.0 * _LON_RES) - _TWO_PI)  # same angle, wrapped into [0, 2pi)
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 5.0, rtol=1e-5)

    def test_single_cell_mosaic(self) -> None:
        """A 1x1 mosaic is sampled only by pixels mapping exactly onto its cell."""
        mosaic = _make_mosaic_data(np.array([[7.0]]))
        lat = np.array([[mosaic.lat_range[0], mosaic.lat_range[0] + _LAT_RES]])
        lon = np.full((1, 2), mosaic.lon_range[0])
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        assert model[0, 0] == pytest.approx(7.0)
        assert model[0, 1] == pytest.approx(0.0)


class TestOutOfBoundsAndMasks:
    """Pixels off the body or outside the mosaic coverage are 0.0."""

    def test_latitude_outside_mosaic_is_zero(self) -> None:
        """Pixels whose latitude is beyond the mosaic extent get 0.0."""
        mosaic = _make_mosaic_data(np.full((4, 6), 3.0))
        lat, lon = _cell_coords(mosaic, 4.5, 2.0, (2, 3))  # rows run 0..3 only
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 0.0)

    def test_body_masked_pixel_is_zero(self) -> None:
        """Pixels masked in the latitude backplane get 0.0 even with in-range coords."""
        mosaic = _make_mosaic_data(np.full((4, 6), 3.0))
        lat, lon = _cell_coords(mosaic, 2.0, 3.0, (2, 3))
        lat_mask = np.zeros((2, 3), dtype=bool)
        lat_mask[0, 1] = True
        model, _ = _run(mosaic, _CartoBackplane(lat, lon, lat_mask=lat_mask))
        assert model[0, 1] == pytest.approx(0.0)
        assert model[0, 0] == pytest.approx(3.0)

    def test_last_row_boundary_is_inclusive(self) -> None:
        """A pixel on the last mosaic row (row n_lat - 1) is in bounds.

        The target latitude is nudged a few ULPs below the exact boundary so that
        float round-off in the row-coordinate division cannot push the sample past
        the inclusive ``row_coords <= n_lat - 1`` bound being tested.
        """
        img = np.zeros((4, 6))
        img[3, :] = 1.5
        mosaic = _make_mosaic_data(img)
        boundary_lat = mosaic.lat_range[0] + 3.0 * _LAT_RES
        lat_val = boundary_lat - 4.0 * float(np.spacing(boundary_lat))
        lat = np.full((2, 2), lat_val)
        lon = np.full((2, 2), mosaic.lon_range[0] + 2.0 * _LON_RES)
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 1.5, rtol=1e-6)

    def test_just_beyond_last_row_is_zero(self) -> None:
        """A pixel slightly past the last mosaic row is out of bounds."""
        mosaic = _make_mosaic_data(np.full((4, 6), 1.5))
        lat, lon = _cell_coords(mosaic, 3.01, 2.0, (2, 2))
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        np.testing.assert_allclose(model, 0.0)


# =========================================================================
# Output format and resolution ratio
# =========================================================================


class TestOutputFormat:
    """Model image shape/dtype and the resolution_ratio field."""

    def test_model_shape_matches_image_and_dtype_float32(self) -> None:
        """The model image matches the backplane shape and is float32."""
        mosaic = _make_mosaic_data(np.ones((4, 6)))
        lat, lon = _cell_coords(mosaic, 2.0, 3.0, (5, 7))
        model, _ = _run(mosaic, _CartoBackplane(lat, lon))
        assert model.shape == (5, 7)
        assert model.dtype == np.dtype(np.float32)

    def test_resolution_ratio_is_median_over_center(self) -> None:
        """resolution_ratio == median(unmasked eff_resolution) / center_resolution."""
        eff = np.arange(1.0, 25.0).reshape(4, 6)
        eff_mask = eff > 12.0  # unmasked values 1..12, median 6.5
        mosaic = _make_mosaic_data(np.ones((4, 6)), eff_res=eff, eff_res_mask=eff_mask)
        lat, lon = _cell_coords(mosaic, 2.0, 3.0, (2, 2))
        _, ratio = _run(mosaic, _CartoBackplane(lat, lon, center_res=2.0))
        assert ratio == pytest.approx(6.5 / 2.0)

    def test_zero_center_resolution_falls_back_to_one(self) -> None:
        """A non-positive image-center resolution yields resolution_ratio == 1.0."""
        mosaic = _make_mosaic_data(np.ones((4, 6)), eff_res=3.0)
        lat, lon = _cell_coords(mosaic, 2.0, 3.0, (2, 2))
        _, ratio = _run(mosaic, _CartoBackplane(lat, lon, center_res=0.0))
        assert ratio == pytest.approx(1.0)

    def test_fully_masked_mosaic_returns_none_without_backplane(self) -> None:
        """An all-masked mosaic returns None before any backplane is constructed."""
        mosaic = _make_mosaic_data(np.ones((4, 6)), img_mask=True)
        with patch('oops.backplane.Backplane') as bp_cls:
            result = create_cartographic_model(mosaic, object(), body_name='MIMAS')
        assert result is None
        bp_cls.assert_not_called()
