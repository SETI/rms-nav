"""Ring reprojection and mosaicing utilities.

This module provides the RingMosaic class for reprojecting planetary ring
images onto radius/longitude grids and accumulating them into sparse mosaics.

Thread safety: RingMosaic.reproject() is not thread-safe because it may
temporarily mutate obs.fov and oops global precision settings. Concurrent
calls from separate threads will interfere.
"""

import enum
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.ma as ma
import oops
import scipy.ndimage as nd
from polymath import Scalar, Vector3

from nav.reproj._context_managers import _reduced_oops_precision
from nav.reproj._serialization import (
    infer_format,
    load_fits,
    load_npz,
    orbit_model_from_dict,
    orbit_model_to_dict,
    save_fits,
    save_npz,
    verify_dtype,
)
from nav.reproj.ring_orbit_model import RingOrbitModel
from nav.support.image import array_unzoom, array_zoom
from nav.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType, PathLike

_LOGGING_NAME = 'nav.' + __name__


class RingMosaicMergeStrategy(enum.Enum):
    """Strategy for resolving per-longitude-column conflicts when adding data to a RingMosaic.

    Members:
        BEST_RESOLUTION: Replace an existing longitude column only when the new
            data has strictly better mean radial resolution for that column.
        MOST_COVERAGE_THEN_RESOLUTION: Fill empty (missing) longitude columns
            first; for already-present columns, replace only when the new data
            has better mean radial resolution.
    """

    BEST_RESOLUTION = 'best_resolution'
    MOST_COVERAGE_THEN_RESOLUTION = 'most_coverage_then_resolution'


# Main ring radius bounds used when no explicit range is given
RINGS_MIN_RADIUS = oops.body.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.body.SATURN_MAIN_RINGS[1]

# Slop values: must be smaller than the smallest resolution we will ever use
_LONGITUDE_SLOP = 1e-6  # rad
_RADIUS_SLOP = 1e-6  # km

_MAX_LONGITUDE = math.pi * 2.0 - _LONGITUDE_SLOP * 2

# Module-level defaults for RingMosaic parameters
DEFAULT_LONGITUDE_RESOLUTION = 0.02 * math.pi / 180.0  # 0.02 degrees in rad
DEFAULT_RADIUS_RESOLUTION = 5.0  # km
DEFAULT_ZOOM = (1, 1)
DEFAULT_MARGIN = 3


@dataclass(frozen=True)
class RingReprojResult:
    """Data returned by RingMosaic.reproject().

    The image and per-longitude metadata arrays are always sparse: only
    longitude columns with valid data are included. Use
    ``longitude_antimask`` to reconstruct the actual longitude values.

    Attributes:
        body_name: The name of the planet whose ring was reprojected.
        img: Sparse reprojected image [radius, valid_longitude].
        longitude_resolution: Longitude resolution (rad/pixel).
        radius_resolution: Radius resolution (km/pixel).
        radius_inner: Inner radius of the reprojection (km).
        radius_outer: Outer radius of the reprojection (km).
        longitude_antimask: Boolean array of length ``n_full_lon`` (the
            total number of longitude bins from 0 to 2*pi). True at each
            longitude bin that has reprojected data.
        mean_radial_resolution: Mean radial resolution per valid longitude
            (km/pixel).
        mean_angular_resolution: Mean angular resolution per valid longitude
            (rad/pixel).
        mean_phase: Mean phase angle per valid longitude (rad).
        mean_emission: Mean emission angle per valid longitude (rad).
        incidence: Scalar incidence angle over the ring plane (rad).
        time: Midtime of the observation (TDB seconds).
        orbit_model: The RingOrbitModel used for co-rotating longitude
            conversion, or None for inertial longitude.
        image_dtype: NumPy dtype used for the ``img`` array.
        metadata_dtype: NumPy dtype used for geometry arrays
            (``mean_radial_resolution``, ``mean_angular_resolution``,
            ``mean_phase``, ``mean_emission``). Mosaic ``time`` columns are
            always ``float64`` regardless of this setting.
    """

    body_name: str
    img: ma.MaskedArray
    longitude_resolution: float
    radius_resolution: float
    radius_inner: float
    radius_outer: float
    longitude_antimask: NDArrayBoolType
    mean_radial_resolution: NDArrayFloatType
    mean_angular_resolution: NDArrayFloatType
    mean_phase: NDArrayFloatType
    mean_emission: NDArrayFloatType
    incidence: float
    time: float
    orbit_model: RingOrbitModel | None
    image_dtype: np.dtype
    metadata_dtype: np.dtype

    def save(
        self,
        path: PathLike,
        *,
        format: str | None = None,  # noqa: A002
        compress: bool = True,
    ) -> None:
        """Save this RingReprojResult to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format`` is given.
            format: Explicit format: ``'npz'`` or ``'fits'``.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            result = ring_mosaic.reproject(obs)
            result.save('ring_reproj.npz')
            reloaded = RingReprojResult.load('ring_reproj.npz')
        """
        fmt = infer_format(path, format)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'img': self.img,
            'longitude_resolution': self.longitude_resolution,
            'radius_resolution': self.radius_resolution,
            'radius_inner': self.radius_inner,
            'radius_outer': self.radius_outer,
            'longitude_antimask': self.longitude_antimask,
            'mean_radial_resolution': self.mean_radial_resolution,
            'mean_angular_resolution': self.mean_angular_resolution,
            'mean_phase': self.mean_phase,
            'mean_emission': self.mean_emission,
            'incidence': self.incidence,
            'time': self.time,
            'orbit_model': orbit_model_to_dict(self.orbit_model),
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
        }
        if fmt == 'npz':
            save_npz(path, 'RingReprojResult', 1, payload, compress=compress)
        else:
            save_fits(path, 'RingReprojResult', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format: str | None = None,  # noqa: A002
    ) -> 'RingReprojResult':
        """Load a RingReprojResult from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new RingReprojResult with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``.

        Example::

            result = RingReprojResult.load('ring_reproj.npz')
        """
        fmt = infer_format(path, format)
        d = (
            load_npz(path, 'RingReprojResult')
            if fmt == 'npz'
            else load_fits(path, 'RingReprojResult')
        )

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in (
                    'img',
                    'mean_radial_resolution',
                    'mean_angular_resolution',
                    'mean_phase',
                    'mean_emission',
                )
            },
            image_dtype,
            metadata_dtype,
            image_fields=['img'],
            metadata_fields=[
                'mean_radial_resolution',
                'mean_angular_resolution',
                'mean_phase',
                'mean_emission',
            ],
        )

        # Reconstruct orbit model from flattened dict keys
        orbit_d: dict[str, Any] = {}
        prefix = 'orbit_model__'
        for k, v in d.items():
            if k.startswith(prefix):
                orbit_d[k[len(prefix) :]] = v
        orbit_model = orbit_model_from_dict(orbit_d) if orbit_d else None

        return cls(
            body_name=str(d['body_name']),
            img=d['img'],
            longitude_resolution=float(d['longitude_resolution']),
            radius_resolution=float(d['radius_resolution']),
            radius_inner=float(d['radius_inner']),
            radius_outer=float(d['radius_outer']),
            longitude_antimask=np.asarray(d['longitude_antimask'], dtype=np.bool_),
            mean_radial_resolution=d['mean_radial_resolution'],
            mean_angular_resolution=d['mean_angular_resolution'],
            mean_phase=d['mean_phase'],
            mean_emission=d['mean_emission'],
            incidence=float(d['incidence']),
            time=float(d['time']),
            orbit_model=orbit_model,
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
        )


@dataclass(frozen=True)
class RingMosaicData:
    """Mosaic data returned by RingMosaic retrieval methods.

    Attributes:
        body_name: The name of the planet.
        ring_body_name: The oops body name for the ring plane
            (e.g. 'saturn:ring').
        shadow_body_name: The oops body name for the shadow-casting planet
            (e.g. 'saturn').
        longitude_resolution: Longitude resolution (rad/pixel).
        radius_resolution: Radius resolution (km/pixel).
        radius_inner: Inner radius (km).
        radius_outer: Outer radius (km).
        longitude_antimask: Boolean array of length ``n_full_lon``.
            True at longitude bins that have data.
        img: Mosaic image [radius, longitude] as a MaskedArray.
        longitude_range: (start, end) of longitudes in img (rad), or None
            for the full circle.
        mean_radial_resolution: Mean radial resolution per longitude column
            as a MaskedArray.
        mean_angular_resolution: Mean angular resolution per column as a
            MaskedArray.
        mean_phase: Mean phase angle per column as a MaskedArray (rad).
        mean_emission: Mean emission angle per column as a MaskedArray (rad).
        mean_incidence: Scalar mean incidence angle (rad) over all images.
        image_number: Per-longitude index identifying which add() call
            contributed the data, as a MaskedArray. Always stored as
            ``uint16``, capping a single mosaic at 65 535 contributing images.
        time: Per-longitude observation midtime (TDB seconds) as a
            MaskedArray. Always stored as ``float64``.
        image_dtype: NumPy dtype used for the ``img`` array.
        metadata_dtype: NumPy dtype used for geometry arrays
            (``mean_radial_resolution``, ``mean_angular_resolution``,
            ``mean_phase``, ``mean_emission``).
    """

    body_name: str
    ring_body_name: str
    shadow_body_name: str
    longitude_resolution: float
    radius_resolution: float
    radius_inner: float
    radius_outer: float
    longitude_antimask: NDArrayBoolType
    img: ma.MaskedArray
    longitude_range: tuple[float, float] | None
    mean_radial_resolution: ma.MaskedArray
    mean_angular_resolution: ma.MaskedArray
    mean_phase: ma.MaskedArray
    mean_emission: ma.MaskedArray
    mean_incidence: float
    image_number: ma.MaskedArray
    time: ma.MaskedArray
    image_dtype: np.dtype
    metadata_dtype: np.dtype

    def save(
        self,
        path: PathLike,
        *,
        format: str | None = None,  # noqa: A002
        compress: bool = True,
    ) -> None:
        """Save this RingMosaicData to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format`` is given.
            format: Explicit format: ``'npz'`` or ``'fits'``.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            data = ring_mosaic.to_bounded(longitude_range=(0.0, math.pi))
            data.save('saturn_rings.npz')
            data.save('saturn_rings.fits', format='fits')
            reloaded = RingMosaicData.load('saturn_rings.npz')
        """
        fmt = infer_format(path, format)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'ring_body_name': self.ring_body_name,
            'shadow_body_name': self.shadow_body_name,
            'longitude_resolution': self.longitude_resolution,
            'radius_resolution': self.radius_resolution,
            'radius_inner': self.radius_inner,
            'radius_outer': self.radius_outer,
            'longitude_antimask': self.longitude_antimask,
            'img': self.img,
            'longitude_range': self.longitude_range,
            'mean_radial_resolution': self.mean_radial_resolution,
            'mean_angular_resolution': self.mean_angular_resolution,
            'mean_phase': self.mean_phase,
            'mean_emission': self.mean_emission,
            'mean_incidence': self.mean_incidence,
            'image_number': self.image_number,
            'time': self.time,
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
        }
        if fmt == 'npz':
            save_npz(path, 'RingMosaicData', 1, payload, compress=compress)
        else:
            save_fits(path, 'RingMosaicData', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format: str | None = None,  # noqa: A002
    ) -> 'RingMosaicData':
        """Load a RingMosaicData from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new RingMosaicData with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``, or if ``image_number`` is not uint16,
                or if ``time`` is not float64.

        Example::

            data = RingMosaicData.load('saturn_rings.npz')
        """
        fmt = infer_format(path, format)
        d = load_npz(path, 'RingMosaicData') if fmt == 'npz' else load_fits(path, 'RingMosaicData')

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in (
                    'img',
                    'mean_radial_resolution',
                    'mean_angular_resolution',
                    'mean_phase',
                    'mean_emission',
                    'time',
                    'image_number',
                )
            },
            image_dtype,
            metadata_dtype,
            image_fields=['img'],
            metadata_fields=[
                'mean_radial_resolution',
                'mean_angular_resolution',
                'mean_phase',
                'mean_emission',
            ],
            float64_fields=['time'],
        )

        lon_range_raw = d.get('longitude_range')
        longitude_range: tuple[float, float] | None
        if lon_range_raw is None:
            longitude_range = None
        else:
            longitude_range = (float(lon_range_raw[0]), float(lon_range_raw[1]))

        return cls(
            body_name=str(d['body_name']),
            ring_body_name=str(d['ring_body_name']),
            shadow_body_name=str(d['shadow_body_name']),
            longitude_resolution=float(d['longitude_resolution']),
            radius_resolution=float(d['radius_resolution']),
            radius_inner=float(d['radius_inner']),
            radius_outer=float(d['radius_outer']),
            longitude_antimask=np.asarray(d['longitude_antimask'], dtype=np.bool_),
            img=d['img'],
            longitude_range=longitude_range,
            mean_radial_resolution=d['mean_radial_resolution'],
            mean_angular_resolution=d['mean_angular_resolution'],
            mean_phase=d['mean_phase'],
            mean_emission=d['mean_emission'],
            mean_incidence=float(d['mean_incidence']),
            image_number=d['image_number'],
            time=d['time'],
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
        )


class RingMosaic:
    """Sparse ring mosaic accumulator.

    Reprojects ring images onto a radius/longitude grid and accumulates them
    into a mosaic using true sparse longitude storage. Only longitude columns
    that actually contain reprojected data are stored; new columns are
    inserted in sorted order using batched ``np.insert`` calls.

    Parameters:
        body_name: The planet name (e.g. 'SATURN'). Used to derive the
            oops ring body name and shadow body name for backplane calls.
        radius_inner: Inner radius of the mosaic (km).
        radius_outer: Outer radius of the mosaic (km).
        longitude_resolution: Longitude bin width (rad/pixel).
        radius_resolution: Radius bin height (km/pixel).
        merge_strategy: How to resolve conflicts when the same longitude
            column appears in multiple reprojections.
        orbit_model: Default RingOrbitModel for co-rotating frame
            reprojections. Can be overridden per-call in reproject().
        image_dtype: NumPy dtype for the reprojected brightness ``img``
            array. Defaults to ``np.float64``.
        metadata_dtype: NumPy dtype for geometry arrays
            (``mean_radial_resolution``, ``mean_angular_resolution``,
            ``mean_phase``, ``mean_emission``). Defaults to ``np.float32``.

    Notes:
        reproject() is not thread-safe because it mutates obs.fov and oops
        global precision settings.

        ``time`` is always stored as ``float64`` regardless of
        ``metadata_dtype``. ``image_number`` is always stored as
        ``uint16``, capping a single mosaic at 65 535 contributing images.
        ``add()`` raises ``OverflowError`` if that limit is exceeded.
    """

    def __init__(
        self,
        body_name: str,
        radius_inner: float,
        radius_outer: float,
        *,
        longitude_resolution: float = DEFAULT_LONGITUDE_RESOLUTION,
        radius_resolution: float = DEFAULT_RADIUS_RESOLUTION,
        merge_strategy: RingMosaicMergeStrategy = (
            RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        ),
        orbit_model: RingOrbitModel | None = None,
        image_dtype: np.typing.DTypeLike = np.float64,
        metadata_dtype: np.typing.DTypeLike = np.float32,
    ) -> None:
        """Initialize an empty RingMosaic."""
        self._body_name = body_name
        self._ring_body_name = body_name.lower() + ':ring'
        self._shadow_body_name = body_name.lower()
        self._radius_inner = radius_inner
        self._radius_outer = radius_outer
        self._lon_resolution = longitude_resolution
        self._rad_resolution = radius_resolution
        self._merge_strategy = merge_strategy
        self._orbit_model = orbit_model
        self._image_dtype: np.dtype = np.dtype(image_dtype)
        self._metadata_dtype: np.dtype = np.dtype(metadata_dtype)

        self._n_radius = math.ceil((radius_outer - radius_inner + _RADIUS_SLOP) / radius_resolution)
        self._n_full_lon = int(math.pi * 2.0 / longitude_resolution)

        # Sparse storage: only valid longitude columns are held.
        # _antimask[i] is True iff longitude bin i has data.
        self._antimask: NDArrayBoolType = np.zeros(self._n_full_lon, dtype=np.bool_)
        self._img_sparse: ma.MaskedArray = ma.MaskedArray(
            np.empty((self._n_radius, 0), dtype=self._image_dtype),
            mask=np.ones((self._n_radius, 0), dtype=np.bool_),
        )
        self._mean_radial_res: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_angular_res: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_phase: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_emission: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_incidence: float = 0.0
        self._image_number: NDArrayIntType = np.empty(0, dtype=np.uint16)
        self._time: NDArrayFloatType = np.empty(0, dtype=np.float64)
        self._image_count: int = 0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def body_name(self) -> str:
        """The planet name supplied at construction."""
        return self._body_name

    @property
    def ring_body_name(self) -> str:
        """oops body name for the ring plane (e.g. 'saturn:ring')."""
        return self._ring_body_name

    @property
    def shadow_body_name(self) -> str:
        """oops body name for the shadow-casting planet (e.g. 'saturn')."""
        return self._shadow_body_name

    @property
    def bounds(self) -> tuple[float, float] | None:
        """Longitude extent of current mosaic data as (min, max) in radians.

        Returns None if no data has been added yet.
        """
        true_indices = np.where(self._antimask)[0]
        if len(true_indices) == 0:
            return None
        return (
            float(true_indices[0]) * self._lon_resolution,
            float(true_indices[-1]) * self._lon_resolution,
        )

    # ------------------------------------------------------------------
    # Static grid generation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_longitudes(
        longitude_start: float = 0.0,
        longitude_end: float = _MAX_LONGITUDE,
        *,
        longitude_resolution: float = DEFAULT_LONGITUDE_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a longitude array aligned to grid boundaries.

        Parameters:
            longitude_start: Start of the range (rad). Defaults to 0.
            longitude_end: End of the range (rad). Defaults to just under
                2*pi.
            longitude_resolution: Step size (rad/pixel).

        Returns:
            1-D array of longitudes (rad) on resolution boundaries, with no
            value less than longitude_start or greater than longitude_end.
        """
        start_idx = math.ceil(longitude_start / longitude_resolution)
        end_idx = math.floor(longitude_end / longitude_resolution)
        return np.arange(start_idx, end_idx + 1) * longitude_resolution

    @staticmethod
    def generate_radii(
        radius_inner: float,
        radius_outer: float,
        *,
        radius_resolution: float = DEFAULT_RADIUS_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a radius array from inner to outer.

        Parameters:
            radius_inner: Inner radius (km).
            radius_outer: Outer radius (km).
            radius_resolution: Step size (km/pixel).

        Returns:
            1-D array of radii (km) starting at radius_inner with step
            radius_resolution, ending at or just before radius_outer.
        """
        n = math.ceil((radius_outer - radius_inner + _RADIUS_SLOP) / radius_resolution)
        return np.arange(n) * radius_resolution + radius_inner

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def longitude_radius_to_pixels(
        obs: Any,
        longitude: NDArrayFloatType,
        radius: NDArrayFloatType,
        *,
        orbit_model: RingOrbitModel | None = None,
        ring_body_name: str = 'saturn:ring',
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Convert longitude/radius pairs to image pixel coordinates.

        If orbit_model is given, the longitude values are treated as
        co-rotating and are converted to inertial before the coordinate
        lookup.

        Parameters:
            obs: The Observation whose FOV is used.
            longitude: Longitude array (rad).
            radius: Radius array (km).
            orbit_model: RingOrbitModel for co-rotating frame conversion,
                or None for inertial longitude.
            ring_body_name: oops ring body name for the surface lookup.

        Returns:
            Tuple of (u, v) floating-point pixel coordinate arrays.
        """
        longitude = Scalar(longitude)
        radius = Scalar(radius)

        if orbit_model is not None:
            longitude = orbit_model.corotating_to_inertial(longitude, obs.midtime)

        if len(longitude) == 0:
            return Scalar([]), Scalar([])

        ring_surface = oops.Body.lookup(ring_body_name.replace(':', '_').upper()).surface
        obs_event = oops.Event(obs.midtime, (Vector3.ZERO, Vector3.ZERO), obs.path, obs.frame)
        _, obs_event = ring_surface.photon_to_event_by_coords(obs_event, (radius, longitude))

        uv = obs.fov.uv_from_los(obs_event.neg_arr_ap)
        u, v = uv.to_scalars()
        return u.vals, v.vals

    @staticmethod
    def orbit_pixels(
        obs: Any,
        orbit_model: RingOrbitModel,
        *,
        ring_body_name: str = 'saturn:ring',
        longitude_step: float = 0.002 * math.pi / 180.0,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Return (u, v) pixel pairs for an orbit model feature in an image.

        Computes the full 0..2pi longitude range, converts each to image
        coordinates, and filters to those inside the FOV.

        Parameters:
            obs: The Observation (with its FOV already configured as needed).
            orbit_model: The ring orbit model to trace.
            ring_body_name: oops ring body name.
            longitude_step: Longitude step for sampling (rad).

        Returns:
            Tuple of (u_pixels, v_pixels) integer pixel coordinate arrays.
        """
        longitudes, radii = orbit_model.longitude_radius(obs.midtime, step=longitude_step)

        bp = obs.ext_bp
        u_min = 0
        v_min = 0
        u_max = obs.extdata_shape_xy[0] - 1
        v_max = obs.extdata_shape_xy[1] - 1

        bp_radius = bp.ring_radius(ring_body_name)
        bp_longitude = bp.ring_longitude(ring_body_name)
        min_bp_radius = bp_radius.min()
        max_bp_radius = bp_radius.max()
        min_bp_longitude = bp_longitude.min()
        max_bp_longitude = bp_longitude.max()

        goodr = (radii >= min_bp_radius) & (radii <= max_bp_radius)
        goodl = (longitudes >= min_bp_longitude) & (longitudes <= max_bp_longitude)
        good = goodr & goodl

        longitudes = longitudes[good]
        radii = radii[good]

        u_pix, v_pix = RingMosaic.longitude_radius_to_pixels(
            obs, longitudes, radii, ring_body_name=ring_body_name
        )

        in_fov = (u_pix >= u_min) & (u_pix <= u_max) & (v_pix >= v_min) & (v_pix <= v_max)
        return u_pix[in_fov], v_pix[in_fov]

    # ------------------------------------------------------------------
    # Reprojection
    # ------------------------------------------------------------------

    def reproject(
        self,
        obs: Any,
        data: ma.MaskedArray | None = None,
        *,
        longitude_range: tuple[float, float] | None = None,
        radius_range: tuple[float, float] | None = None,
        margin: int = DEFAULT_MARGIN,
        zoom_amt: int | tuple[int, int] = DEFAULT_ZOOM,
        orbit_model: RingOrbitModel | None = None,
        uv_range: tuple[int, int, int, int] | None = None,
        omit_shadow: bool = True,
    ) -> RingReprojResult:
        """Reproject the ring in an image to a sparse radius/longitude grid.

        Always returns a sparse RingReprojResult: only longitude columns
        with valid data are stored in the result. The returned
        longitude_antimask indicates which of the ``n_full_lon`` bins are
        present.

        Parameters:
            obs: The Observation (with its FOV already configured as needed).
            data: Image data to reproject. If None, uses obs.data.
            longitude_range: (start, end) longitude limits (rad). Defaults
                to the full 0..2pi range.
            radius_range: (inner, outer) radius limits (km). Defaults to
                the mosaic's own radius_inner/outer.
            margin: Number of edge pixels to exclude. Must be >= 1.
            zoom_amt: Positive integer or (radial, longitude) tuple giving
                the zoom factor for sub-pixel interpolation. Negative values
                select spline interpolation order (not yet supported).
            orbit_model: RingOrbitModel for co-rotating frame conversion.
                Overrides the default set at construction. None uses inertial
                longitude.
            uv_range: (start_u, end_u, start_v, end_v) to restrict the
                image region reprojected.
            omit_shadow: If True, mask pixels inside the planet's shadow.

        Returns:
            RingReprojResult with sparse image and metadata arrays.

        Raises:
            NotImplementedError: If negative zoom_amt (spline order) is
                requested.
            ValueError: If n_longitude_bins_zoom is not a multiple of
                l_zoom_amt (internal consistency check).
        """
        logger = logging.getLogger(_LOGGING_NAME + '.reproject')
        logger.debug(
            'longitude_range=%s radius_range=%s zoom=%s',
            longitude_range,
            radius_range,
            zoom_amt,
        )

        if orbit_model is None:
            orbit_model = self._orbit_model

        if data is None:
            data = obs.data
        data = data.view(ma.MaskedArray)

        radius_inner = self._radius_inner if radius_range is None else radius_range[0]
        radius_outer = self._radius_outer if radius_range is None else radius_range[1]

        if longitude_range is None:
            longitude_start = 0.0
            longitude_end = _MAX_LONGITUDE
        else:
            longitude_start, longitude_end = longitude_range

        if not isinstance(zoom_amt, (list, tuple)):
            zoom_amt = (zoom_amt, zoom_amt)

        if zoom_amt[0] > 0:
            r_zoom_amt: int = int(zoom_amt[0])
            r_spline_order = 0
        else:
            r_zoom_amt = 1
            r_spline_order = -int(zoom_amt[0])

        if zoom_amt[1] > 0:
            l_zoom_amt: int = int(zoom_amt[1])
            l_spline_order = 0
        else:
            l_zoom_amt = 1
            l_spline_order = -int(zoom_amt[1])

        with _reduced_oops_precision():
            return self._reproject_inner(
                obs=obs,
                data=data,
                longitude_start=longitude_start,
                longitude_end=longitude_end,
                radius_inner=radius_inner,
                radius_outer=radius_outer,
                margin=margin,
                r_zoom_amt=r_zoom_amt,
                l_zoom_amt=l_zoom_amt,
                r_spline_order=r_spline_order,
                l_spline_order=l_spline_order,
                orbit_model=orbit_model,
                uv_range=uv_range,
                omit_shadow=omit_shadow,
                logger=logger,
            )

    def _reproject_inner(
        self,
        *,
        obs: Any,
        data: ma.MaskedArray,
        longitude_start: float,
        longitude_end: float,
        radius_inner: float,
        radius_outer: float,
        margin: int,
        r_zoom_amt: int,
        l_zoom_amt: int,
        r_spline_order: int,
        l_spline_order: int,
        orbit_model: RingOrbitModel | None,
        uv_range: tuple[int, int, int, int] | None,
        omit_shadow: bool,
        logger: logging.Logger,
    ) -> RingReprojResult:
        """Inner reprojection logic, runs with reduced oops precision."""
        if r_spline_order != 0 or l_spline_order != 0:
            raise NotImplementedError(
                'Spline interpolation (negative zoom_amt) is not yet supported'
            )

        meshgrid = None
        start_u = 0
        end_u = obs.data_shape_xy[0] - 1
        start_v = 0
        end_v = obs.data_shape_xy[1]
        if uv_range is not None:
            start_u, end_u, start_v, end_v = uv_range
            meshgrid = oops.Meshgrid.for_fov(
                obs.fov,
                origin=(start_u + 0.5, start_v + 0.5),
                limit=(end_u + 0.5, end_v + 0.5),
                swap=True,
            )

        bp = oops.backplane.Backplane(obs, meshgrid)

        logger.debug('Computing radius backplane')
        bp_radius = bp.ring_radius(self._ring_body_name)
        logger.debug('Computing longitude backplane')
        bp_longitude = bp.ring_longitude(self._ring_body_name)

        if ma.is_masked(data):
            bp_radius = bp_radius.remask_or(data.mask)
            bp_longitude = bp_longitude.remask_or(data.mask)

        logger.debug('Computing geometry backplanes')
        bp_radial_res = bp.ring_radial_resolution(self._ring_body_name)
        bp_angular_res = bp.ring_angular_resolution(self._ring_body_name)
        bp_phase = bp.phase_angle(self._ring_body_name)
        bp_emission = bp.emission_angle(self._ring_body_name)
        bp_incidence = bp.incidence_angle(self._ring_body_name)
        if ma.is_masked(data):
            bp_radial_res = bp_radial_res.remask_or(data.mask)
            bp_angular_res = bp_angular_res.remask_or(data.mask)
            bp_phase = bp_phase.remask_or(data.mask)
            bp_emission = bp_emission.remask_or(data.mask)
            bp_incidence = bp_incidence.remask_or(data.mask)

        if omit_shadow:
            logger.debug('Computing shadow mask')
            shadow = bp.where_inside_shadow(self._ring_body_name, self._shadow_body_name).vals
            data = ma.masked_where(shadow, data)

        if orbit_model is not None:
            bp_longitude = orbit_model.inertial_to_corotating(bp_longitude, obs.midtime)

        n_radius_bins = math.ceil(
            (radius_outer - radius_inner + _RADIUS_SLOP) / self._rad_resolution
        )
        n_radius_bins_zoom = n_radius_bins * r_zoom_amt

        full_min_lon_bin = int(np.floor(longitude_start / self._lon_resolution))
        full_max_lon_bin = int(np.floor(longitude_end / self._lon_resolution))
        n_full_lon_bins = full_max_lon_bin - full_min_lon_bin + 1

        restr_bp_lon = bp_longitude[
            (bp_longitude >= longitude_start)
            & (bp_longitude <= longitude_end)
            & (bp_radius >= radius_inner)
            & (bp_radius <= radius_outer)
        ]

        bp_lon_binned = np.floor(
            (restr_bp_lon.vals - longitude_start) / self._lon_resolution
        ).astype('int')
        full_good_antimask = np.zeros(n_full_lon_bins, dtype=np.bool_)
        full_good_antimask[bp_lon_binned] = True

        filter_width = int(1.0 / np.degrees(self._lon_resolution))
        if filter_width > 1:
            full_good_antimask = nd.maximum_filter1d(full_good_antimask, filter_width, mode='wrap')

        full_lon_bins: NDArrayIntType = np.arange(n_full_lon_bins, dtype=np.int32)
        if l_zoom_amt == 1:
            full_lon_bins_zoom: NDArrayFloatType = full_lon_bins.astype(np.float64)
        else:
            full_lon_bins_zoom = np.arange(n_full_lon_bins * l_zoom_amt) / float(l_zoom_amt)

        full_good_antimask_zoom = array_zoom(full_good_antimask, (l_zoom_amt,))

        lon_bins_restr: NDArrayIntType = full_lon_bins[full_good_antimask]
        lon_bins_restr_zoom: NDArrayFloatType
        if l_zoom_amt == 1:
            lon_bins_restr_zoom = lon_bins_restr.astype(np.float64)
        else:
            lon_bins_restr_zoom = full_lon_bins_zoom[full_good_antimask_zoom]

        n_lon_bins = len(lon_bins_restr)
        n_lon_bins_zoom = len(lon_bins_restr_zoom)
        if (n_lon_bins_zoom % l_zoom_amt) != 0:
            raise ValueError(
                f'n_lon_bins_zoom ({n_lon_bins_zoom}) is not a multiple of '
                f'l_zoom_amt ({l_zoom_amt})'
            )

        long_bins = np.tile(np.arange(n_lon_bins), n_radius_bins)
        long_bins_act = np.tile(
            lon_bins_restr * self._lon_resolution + longitude_start, n_radius_bins
        )

        if r_zoom_amt == 1 and l_zoom_amt == 1:
            long_bins_zoom = long_bins
            long_bins_act_zoom = long_bins_act
        else:
            long_bins_zoom = np.tile(np.arange(n_lon_bins_zoom), n_radius_bins_zoom)
            long_bins_act_zoom = np.tile(
                lon_bins_restr_zoom * self._lon_resolution + longitude_start,
                n_radius_bins_zoom,
            )

        rad_bins = np.repeat(np.arange(n_radius_bins), n_lon_bins)
        if r_zoom_amt == 1 and l_zoom_amt == 1:
            rad_bins_zoom = rad_bins
        else:
            rad_bins_zoom = np.repeat(np.arange(n_radius_bins_zoom), n_lon_bins_zoom)

        rad_bins_act: NDArrayFloatType
        rad_bins_act_zoom: NDArrayFloatType
        if orbit_model is not None:
            inertial_lons_act = orbit_model.corotating_to_inertial(long_bins_act, obs.midtime)
            rad_bins_act = (
                rad_bins * self._rad_resolution
                + radius_inner
                - orbit_model.a
                + orbit_model.radius_at_longitude(inertial_lons_act, obs.midtime)
            )
            if r_zoom_amt == 1 and l_zoom_amt == 1:
                rad_bins_act_zoom = rad_bins_act
            else:
                inertial_lons_zoom = orbit_model.corotating_to_inertial(
                    long_bins_act_zoom, obs.midtime
                )
                rad_offset_zoom = orbit_model.radius_at_longitude(inertial_lons_zoom, obs.midtime)
                rad_bins_act_zoom = (
                    rad_bins_zoom / float(r_zoom_amt) * self._rad_resolution
                    + radius_inner
                    - orbit_model.a
                    + rad_offset_zoom
                )
        else:
            rad_bins_act = rad_bins * self._rad_resolution + radius_inner
            if r_zoom_amt == 1 and l_zoom_amt == 1:
                rad_bins_act_zoom = rad_bins_act
            else:
                raise NotImplementedError(
                    'Zoom with non-corotating inertial radius is not yet implemented'
                )

        u_frac_zoom, v_frac_zoom = self.longitude_radius_to_pixels(
            obs,
            long_bins_act_zoom,
            rad_bins_act_zoom,
            orbit_model=orbit_model,
            ring_body_name=self._ring_body_name,
        )

        u_frac_zoom_rect = u_frac_zoom.reshape((n_radius_bins_zoom, n_lon_bins_zoom))
        u_frac = u_frac_zoom_rect[::r_zoom_amt, ::l_zoom_amt].reshape(long_bins_act.shape)
        v_frac_zoom_rect = v_frac_zoom.reshape((n_radius_bins_zoom, n_lon_bins_zoom))
        v_frac = v_frac_zoom_rect[::r_zoom_amt, ::l_zoom_amt].reshape(long_bins_act.shape)

        u_frac -= start_u
        v_frac -= start_v
        u_pix = np.floor(u_frac).astype('int')
        v_pix = np.floor(v_frac).astype('int')

        good = (
            (u_pix >= margin)
            & (u_pix <= end_u - start_u - margin)
            & (v_pix >= margin)
            & (v_pix <= end_v - start_v - margin)
        )
        u_frac = u_frac[good]
        v_frac = v_frac[good]
        u_pix = u_pix[good]
        v_pix = v_pix[good]

        if r_zoom_amt == 1 and l_zoom_amt == 1:
            u_pix_zoom, v_pix_zoom = u_pix, v_pix
            good_zoom = good
        else:
            u_frac_zoom -= start_u
            v_frac_zoom -= start_v
            u_pix_zoom = np.floor(u_frac_zoom).astype('int')
            v_pix_zoom = np.floor(v_frac_zoom).astype('int')
            good_zoom = (
                (u_pix_zoom >= margin)
                & (u_pix_zoom <= end_u - start_u - margin)
                & (v_pix_zoom >= margin)
                & (v_pix_zoom <= end_v - start_v - margin)
            )
            u_pix_zoom = u_pix_zoom[good_zoom]
            v_pix_zoom = v_pix_zoom[good_zoom]

        good_rad = rad_bins[good]
        good_lon = long_bins[good]
        zoomed = r_zoom_amt != 1 or l_zoom_amt != 1
        good_rad_zoom = rad_bins_zoom[good_zoom] if zoomed else good_rad
        good_lon_zoom = long_bins_zoom[good_zoom] if zoomed else good_lon

        restr_data = data[v_pix_zoom, u_pix_zoom]
        repro_img: ma.MaskedArray = ma.zeros(
            (n_radius_bins_zoom, n_lon_bins_zoom), dtype=self._image_dtype
        )
        repro_img[:, :] = ma.masked
        repro_img[good_rad_zoom, good_lon_zoom] = restr_data
        repro_img = ma.MaskedArray(array_unzoom(repro_img, (r_zoom_amt, l_zoom_amt)))

        good_lon_antimask = ~ma.getmaskarray(ma.sum(repro_img, axis=0))

        repro_radial_res = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_radial_res[:] = ma.masked
        repro_radial_res[good_rad, good_lon] = bp_radial_res.mvals[v_pix, u_pix]
        radial_res_antimask = ~ma.getmaskarray(ma.mean(repro_radial_res, axis=0))
        good_lon_antimask &= radial_res_antimask

        repro_angular_res = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_angular_res[:] = ma.masked
        repro_angular_res[good_rad, good_lon] = bp_angular_res.mvals[v_pix, u_pix]

        repro_phase = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_phase[:] = ma.masked
        repro_phase[good_rad, good_lon] = bp_phase.mvals[v_pix, u_pix]

        repro_emission = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_emission[:] = ma.masked
        repro_emission[good_rad, good_lon] = bp_emission.mvals[v_pix, u_pix]

        repro_incidence = float(ma.mean(bp_incidence.mvals[v_pix, u_pix]))

        # Compress to sparse representation
        repro_img = repro_img[:, good_lon_antimask]
        repro_mean_radial_res = ma.mean(repro_radial_res[:, good_lon_antimask], axis=0).filled(0.0)
        repro_mean_angular_res = ma.mean(repro_angular_res[:, good_lon_antimask], axis=0).filled(
            0.0
        )
        repro_mean_phase = ma.mean(repro_phase[:, good_lon_antimask], axis=0).filled(0.0)
        repro_mean_emission = ma.mean(repro_emission[:, good_lon_antimask], axis=0).filled(0.0)

        # Full antimask for the longitude range covered
        new_antimask = np.zeros(self._n_full_lon, dtype=np.bool_)
        new_antimask[lon_bins_restr[good_lon_antimask] + full_min_lon_bin] = True

        logger.debug('Reprojection complete: %d valid longitudes', int(good_lon_antimask.sum()))

        return RingReprojResult(
            body_name=self._body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            longitude_antimask=new_antimask,
            img=repro_img,
            mean_radial_resolution=np.asarray(repro_mean_radial_res, dtype=self._metadata_dtype),
            mean_angular_resolution=np.asarray(repro_mean_angular_res, dtype=self._metadata_dtype),
            mean_phase=np.asarray(repro_mean_phase, dtype=self._metadata_dtype),
            mean_emission=np.asarray(repro_mean_emission, dtype=self._metadata_dtype),
            incidence=repro_incidence,
            time=obs.midtime,
            orbit_model=orbit_model,
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
        )

    # ------------------------------------------------------------------
    # Mosaic accumulation
    # ------------------------------------------------------------------

    def add(self, repro: RingReprojResult) -> None:
        """Add a reprojected image to the mosaic.

        New longitude columns are inserted into the sparse arrays using a
        single batched ``np.insert`` call. Existing columns are updated
        according to the merge strategy.

        Parameters:
            repro: Sparse reprojection result from reproject().

        Raises:
            OverflowError: If the number of images added would exceed the
                uint16 maximum of 65 535.
        """
        if self._image_count > np.iinfo(np.uint16).max:
            raise OverflowError(
                f'image_count {self._image_count} exceeds uint16 max '
                f'{np.iinfo(np.uint16).max}; cannot add more images'
            )

        valid_bins = np.where(repro.longitude_antimask)[0]
        if len(valid_bins) == 0:
            return

        existing_bins = np.where(self._antimask)[0]

        new_mask = ~self._antimask[valid_bins]
        new_bins = valid_bins[new_mask]
        old_bins = valid_bins[~new_mask]

        if len(new_bins) > 0:
            self._insert_new_columns(new_bins, valid_bins, repro)

        if len(old_bins) > 0:
            self._update_existing_columns(old_bins, valid_bins, existing_bins, repro)

        self._antimask[valid_bins] = True
        self._image_count += 1
        self._mean_incidence = repro.incidence

    def _insert_new_columns(
        self,
        new_bins: NDArrayIntType,
        valid_bins: NDArrayIntType,
        repro: RingReprojResult,
    ) -> None:
        """Insert columns for brand-new longitude bins via batched np.insert."""
        # For each new bin, find where it sits among the currently-stored
        # columns (which correspond to sorted existing_bins).
        existing_bins = np.where(self._antimask)[0]
        insert_positions = np.searchsorted(existing_bins, new_bins)

        # Indices of new_bins in the repro valid_bins array
        new_repro_idx = np.searchsorted(valid_bins, new_bins)

        # Expand all sparse arrays by inserting the new columns
        new_img_cols = repro.img[:, new_repro_idx]  # [n_radius, n_new]
        new_img_data = ma.getdata(new_img_cols)
        new_img_mask = ma.getmaskarray(new_img_cols)

        img_data = np.insert(ma.getdata(self._img_sparse), insert_positions, new_img_data, axis=1)
        img_mask = np.insert(
            ma.getmaskarray(self._img_sparse), insert_positions, new_img_mask, axis=1
        )
        self._img_sparse = ma.MaskedArray(img_data, mask=img_mask)

        self._mean_radial_res = np.insert(
            self._mean_radial_res, insert_positions, repro.mean_radial_resolution[new_repro_idx]
        )
        self._mean_angular_res = np.insert(
            self._mean_angular_res, insert_positions, repro.mean_angular_resolution[new_repro_idx]
        )
        self._mean_phase = np.insert(
            self._mean_phase, insert_positions, repro.mean_phase[new_repro_idx]
        )
        self._mean_emission = np.insert(
            self._mean_emission, insert_positions, repro.mean_emission[new_repro_idx]
        )
        self._image_number = np.insert(
            self._image_number,
            insert_positions,
            np.full(len(new_bins), self._image_count, dtype=np.uint16),
        )
        self._time = np.insert(
            self._time,
            insert_positions,
            np.full(len(new_bins), repro.time, dtype=np.float64),
        )

    def _update_existing_columns(
        self,
        old_bins: NDArrayIntType,
        valid_bins: NDArrayIntType,
        existing_bins: NDArrayIntType,
        repro: RingReprojResult,
    ) -> None:
        """Update columns that already exist in the mosaic per merge strategy."""
        # sparse index of each old bin in current storage
        existing_bins_current = np.where(self._antimask)[0]
        sparse_idx = np.searchsorted(existing_bins_current, old_bins)
        # index of each old bin in repro.valid_bins
        repro_idx = np.searchsorted(valid_bins, old_bins)

        for k in range(len(old_bins)):
            si = sparse_idx[k]
            ri = repro_idx[k]
            if self._should_replace(si, ri, repro):
                self._img_sparse[:, si] = repro.img[:, ri]
                self._mean_radial_res[si] = repro.mean_radial_resolution[ri]
                self._mean_angular_res[si] = repro.mean_angular_resolution[ri]
                self._mean_phase[si] = repro.mean_phase[ri]
                self._mean_emission[si] = repro.mean_emission[ri]
                self._image_number[si] = self._image_count
                self._time[si] = repro.time

    def _should_replace(
        self,
        sparse_idx: int,
        repro_idx: int,
        repro: RingReprojResult,
    ) -> bool:
        """Return True if the repro column should replace the existing sparse column."""
        if self._merge_strategy == RingMosaicMergeStrategy.BEST_RESOLUTION:
            return bool(repro.mean_radial_resolution[repro_idx] < self._mean_radial_res[sparse_idx])

        # MOST_COVERAGE_THEN_RESOLUTION
        existing_valid = int((~ma.getmaskarray(self._img_sparse[:, sparse_idx])).sum())
        new_valid = int((~ma.getmaskarray(repro.img[:, repro_idx])).sum())
        if new_valid != existing_valid:
            return new_valid > existing_valid
        return bool(repro.mean_radial_resolution[repro_idx] < self._mean_radial_res[sparse_idx])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def to_sparse(self) -> RingMosaicData:
        """Return the mosaic in its native sparse representation.

        The returned img has shape [n_radius, n_valid_longitudes] and the
        longitude_antimask has length n_full_lon with True at each stored
        column.

        Returns:
            RingMosaicData with the sparse internal arrays.
        """
        return self._build_result(
            img=self._img_sparse,
            antimask=self._antimask.copy(),
            longitude_range=None,
            per_lon_1d=True,
        )

    def to_full(self) -> RingMosaicData:
        """Return the mosaic as a dense full-circle array.

        The returned img has shape [n_radius, n_full_lon]. Longitude columns
        with no data are fully masked.

        Returns:
            RingMosaicData with a full-circle dense array.
        """
        full_img_data = np.zeros((self._n_radius, self._n_full_lon), dtype=self._image_dtype)
        full_img_mask = np.ones((self._n_radius, self._n_full_lon), dtype=np.bool_)
        full_mean_rad_res = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_ang_res = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_phase = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_emission = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_img_number = np.zeros(self._n_full_lon, dtype=np.uint16)
        full_time = np.zeros(self._n_full_lon, dtype=np.float64)

        full_mask_1d = np.ones(self._n_full_lon, dtype=np.bool_)

        valid_bins = np.where(self._antimask)[0]
        if len(valid_bins) > 0:
            full_img_data[:, valid_bins] = ma.getdata(self._img_sparse)
            full_img_mask[:, valid_bins] = ma.getmaskarray(self._img_sparse)
            full_mean_rad_res[valid_bins] = self._mean_radial_res
            full_mean_ang_res[valid_bins] = self._mean_angular_res
            full_mean_phase[valid_bins] = self._mean_phase
            full_mean_emission[valid_bins] = self._mean_emission
            full_img_number[valid_bins] = self._image_number
            full_time[valid_bins] = self._time
            full_mask_1d[valid_bins] = False

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=self._antimask.copy(),
            img=ma.MaskedArray(full_img_data, mask=full_img_mask),
            longitude_range=None,
            mean_radial_resolution=ma.MaskedArray(full_mean_rad_res, mask=full_mask_1d),
            mean_angular_resolution=ma.MaskedArray(full_mean_ang_res, mask=full_mask_1d),
            mean_phase=ma.MaskedArray(full_mean_phase, mask=full_mask_1d),
            mean_emission=ma.MaskedArray(full_mean_emission, mask=full_mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(full_img_number, mask=full_mask_1d),
            time=ma.MaskedArray(full_time, mask=full_mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
        )

    def to_bounded(
        self,
        *,
        longitude_range: tuple[float, float],
    ) -> RingMosaicData:
        """Return the mosaic restricted to the given longitude range.

        The returned img has shape [n_radius, n_bins_in_range] where every
        bin from the start to end of the range is included. Bins without
        data are fully masked.

        Parameters:
            longitude_range: (start, end) in radians.

        Returns:
            RingMosaicData covering exactly the requested longitude range.
        """
        lon_start, lon_end = longitude_range
        start_bin = round(lon_start / self._lon_resolution)
        end_bin = round(lon_end / self._lon_resolution)
        n_bins = end_bin - start_bin + 1

        bounded_img_data = np.zeros((self._n_radius, n_bins), dtype=self._image_dtype)
        bounded_img_mask = np.ones((self._n_radius, n_bins), dtype=np.bool_)
        bounded_mean_rad_res = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_ang_res = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_phase = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_emission = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_img_number = np.zeros(n_bins, dtype=np.uint16)
        bounded_time = np.zeros(n_bins, dtype=np.float64)
        bounded_mask_1d = np.ones(n_bins, dtype=np.bool_)

        bounded_antimask = np.zeros(self._n_full_lon, dtype=np.bool_)

        valid_global_bins = np.where(self._antimask)[0]
        in_range = (valid_global_bins >= start_bin) & (valid_global_bins <= end_bin)
        range_global_bins = valid_global_bins[in_range]

        for _k, gb in enumerate(range_global_bins):
            lb = gb - start_bin
            sparse_k = int(np.searchsorted(valid_global_bins, gb))
            bounded_img_data[:, lb] = ma.getdata(self._img_sparse)[:, sparse_k]
            bounded_img_mask[:, lb] = ma.getmaskarray(self._img_sparse)[:, sparse_k]
            bounded_mean_rad_res[lb] = self._mean_radial_res[sparse_k]
            bounded_mean_ang_res[lb] = self._mean_angular_res[sparse_k]
            bounded_mean_phase[lb] = self._mean_phase[sparse_k]
            bounded_mean_emission[lb] = self._mean_emission[sparse_k]
            bounded_img_number[lb] = self._image_number[sparse_k]
            bounded_time[lb] = self._time[sparse_k]
            bounded_mask_1d[lb] = False
            bounded_antimask[gb] = True

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=bounded_antimask,
            img=ma.MaskedArray(bounded_img_data, mask=bounded_img_mask),
            longitude_range=longitude_range,
            mean_radial_resolution=ma.MaskedArray(bounded_mean_rad_res, mask=bounded_mask_1d),
            mean_angular_resolution=ma.MaskedArray(bounded_mean_ang_res, mask=bounded_mask_1d),
            mean_phase=ma.MaskedArray(bounded_mean_phase, mask=bounded_mask_1d),
            mean_emission=ma.MaskedArray(bounded_mean_emission, mask=bounded_mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(bounded_img_number, mask=bounded_mask_1d),
            time=ma.MaskedArray(bounded_time, mask=bounded_mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
        )

    def _build_result(
        self,
        *,
        img: ma.MaskedArray,
        antimask: NDArrayBoolType,
        longitude_range: tuple[float, float] | None,
        per_lon_1d: bool,
    ) -> RingMosaicData:
        """Build a RingMosaicData for to_sparse()."""
        if per_lon_1d:
            mask_1d = np.zeros(img.shape[1], dtype=np.bool_)
        else:
            mask_1d = np.ones(img.shape[1], dtype=np.bool_)

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=antimask,
            img=img.copy(),
            longitude_range=longitude_range,
            mean_radial_resolution=ma.MaskedArray(self._mean_radial_res.copy(), mask=mask_1d),
            mean_angular_resolution=ma.MaskedArray(self._mean_angular_res.copy(), mask=mask_1d),
            mean_phase=ma.MaskedArray(self._mean_phase.copy(), mask=mask_1d),
            mean_emission=ma.MaskedArray(self._mean_emission.copy(), mask=mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(self._image_number.copy(), mask=mask_1d),
            time=ma.MaskedArray(self._time.copy(), mask=mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
        )
