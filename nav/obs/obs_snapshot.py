from typing import Optional, cast

import numpy as np
import oops
from oops.observation.snapshot import Snapshot
from oops.meshgrid import Meshgrid
from oops.backplane import Backplane

from nav.config import Config
from nav.support.image import pad_array
from nav.support.types import DTypeLike, NDArrayType, NDArrayFloatType, NDArrayBoolType, NPType

from .obs import Obs


class ObsSnapshot(Obs, Snapshot):
    """Provides cached Backplane and Meshgrid operations for snapshot observations.

    This class extends both Obs and Snapshot to provide navigation-specific functionality
    for snapshot observations, including FOV management and backplane caching.
    """

    def __init__(self,
                 snapshot: Snapshot,
                 *,
                 extfov_margin_vu: Optional[int | tuple[int, int]] = None,
                 config: Optional[Config] = None) -> None:
        """Initialize an ObsSnapshot by wrapping an existing Snapshot.

        Parameters:
            snapshot: The Snapshot object to wrap.
            extfov_margin_vu: The extended field of view margins in (v, u) pixels.
            config: The configuration object to use. If None, uses DEFAULT_CONFIG.

        Warning:
            The provided snapshot object should not be used after this call,
            as its internal state is transferred to this instance.
        """

        if not isinstance(snapshot, Snapshot):
            raise TypeError(f"Expected Snapshot, got {type(snapshot).__name__}")

        # Because the oops image read routines create a Snapshot and not a
        # Navigation class, we need to create a Navigation class that looks
        # like a Snapshot but then adds our own attributes. This cool trick
        # with __dict__ makes Navigation have the exact same set of attributes
        # as the original Snapshot. Warning: Do not use that Snapshot anymore!
        self.__dict__ = snapshot.__dict__

        super().__init__(config=config)

        if self.data.ndim != 2:
            raise ValueError(f'Data shape must be 2D, got {self.data.shape}')

        self._data_shape_uv = cast(tuple[int, int], self.data.shape[::-1])
        self._data_shape_vu = cast(tuple[int, int], self.data.shape)
        self._fov_vu_min = (0, 0)
        self._fov_vu_max = (self._data_shape_vu[0]-1, self._data_shape_vu[1]-1)

        if extfov_margin_vu is None:
            self._extfov_margin_vu = (0, 0)
        elif isinstance(extfov_margin_vu, int):
            self._extfov_margin_vu = (int(extfov_margin_vu), int(extfov_margin_vu))
        else:
            self._extfov_margin_vu = (int(extfov_margin_vu[0]), int(extfov_margin_vu[1]))

        if self._extfov_margin_vu[0] < 0 or self._extfov_margin_vu[1] < 0:
            raise ValueError('extfov_margin_vu must be non-negative (v,u).')

        self._extdata = pad_array(self.data, self._extfov_margin_vu)
        self._extdata_shape_uv = cast(tuple[int, int], self._extdata.shape[::-1])
        self._extdata_shape_vu = cast(tuple[int, int], self._extdata.shape)
        self._extfov_vu_min = (-self._extfov_margin_vu[0], -self._extfov_margin_vu[1])
        self._extfov_vu_max = (self._data_shape_vu[0] + self._extfov_margin_vu[0] - 1,
                               self._data_shape_vu[1] + self._extfov_margin_vu[1] - 1)
        self.reset_all()

        closest_planet = None
        closest_dist = 1e38
        for planet in self._config.planets:
            dist = self.body_distance(planet)
            if dist < closest_dist:
                closest_planet = planet
                closest_dist = dist

        self._closest_planet = closest_planet

    def make_fov_zeros(self,
                       dtype: DTypeLike = np.float64) -> NDArrayFloatType:
        """Creates a zero-filled array matching the original FOV dimensions.

        Parameters:
            dtype: Data type for the array elements.

        Returns:
            A zero-filled array with the same shape as the original data.
        """

        return np.zeros(self.data.shape, dtype=dtype)

    def make_extfov_zeros(self,
                          dtype: DTypeLike = np.float64) -> NDArrayFloatType:
        """Creates a zero-filled array matching the extended FOV dimensions.

        Parameters:
            dtype: Data type for the array elements.

        Returns:
            A zero-filled array with the same shape as the extended FOV.
        """

        return np.zeros(self.extdata.shape, dtype=dtype)

    def make_extfov_false(self) -> NDArrayBoolType:
        """Creates a boolean array of False values matching the extended FOV dimensions.

        Returns:
            A boolean array of False values with the same shape as the extended FOV.
        """

        return np.zeros(self.extdata.shape, dtype=bool)

    def unpad_array_to_extfov(self,
                              array: NDArrayType[NPType]) -> NDArrayType[NPType]:
        """Unpads an array to be the size of the extended FOV.

        This is most useful for using the result of np.unpackbits.

        Returns:
            The unpadded array.
        """

        return array[:self.extdata_shape_vu[0], :self.extdata_shape_vu[1]]

    def clip_fov(self,
                 u: int,
                 v: int) -> tuple[int, int]:
        """Clips coordinates to ensure they are within the original FOV boundaries.

        Parameters:
            u: U coordinate to clip
            v: V coordinate to clip

        Returns:
            A tuple of (u, v) coordinates clipped to the FOV boundaries.
        """

        return (int(np.clip(u, self.fov_u_min, self.fov_u_max)),
                int(np.clip(v, self.fov_v_min, self.fov_v_max)))

    def clip_extfov(self,
                    u: int,
                    v: int) -> tuple[int, int]:
        """Clips coordinates to ensure they are within the extended FOV boundaries.

        Parameters:
            u: U coordinate to clip
            v: V coordinate to clip

        Returns:
            A tuple of (u, v) coordinates clipped to the extended FOV boundaries.
        """

        return (int(np.clip(u, self.extfov_u_min, self.extfov_u_max)),
                int(np.clip(v, self.extfov_v_min, self.extfov_v_max)))

    @property
    def data_shape_uv(self) -> tuple[int, int]:
        return self._data_shape_uv

    @property
    def data_shape_vu(self) -> tuple[int, int]:
        return self._data_shape_vu

    @property
    def data_shape_u(self) -> int:
        return self._data_shape_uv[0]

    @property
    def data_shape_v(self) -> int:
        return self._data_shape_uv[1]

    @property
    def fov_vu_min(self) -> tuple[int, int]:
        return self._fov_vu_min

    @property
    def fov_vu_max(self) -> tuple[int, int]:
        return self._fov_vu_max

    @property
    def fov_v_min(self) -> int:
        return self._fov_vu_min[0]

    @property
    def fov_v_max(self) -> int:
        return self._fov_vu_max[0]

    @property
    def fov_u_min(self) -> int:
        return self._fov_vu_min[1]

    @property
    def fov_u_max(self) -> int:
        return self._fov_vu_max[1]

    @property
    def extfov_margin_vu(self) -> tuple[int, int]:
        return self._extfov_margin_vu

    @property
    def extfov_margin_v(self) -> int:
        return self._extfov_margin_vu[0]

    @property
    def extfov_margin_u(self) -> int:
        return self._extfov_margin_vu[1]

    @property
    def extdata(self) -> NDArrayFloatType:
        return self._extdata

    @property
    def extdata_shape_uv(self) -> tuple[int, int]:
        return self._extdata_shape_uv

    @property
    def extdata_shape_vu(self) -> tuple[int, int]:
        return self._extdata_shape_vu

    @property
    def extfov_vu_min(self) -> tuple[int, int]:
        return self._extfov_vu_min

    @property
    def extfov_vu_max(self) -> tuple[int, int]:
        return self._extfov_vu_max

    @property
    def extfov_v_min(self) -> int:
        return self._extfov_vu_min[0]

    @property
    def extfov_v_max(self) -> int:
        return self._extfov_vu_max[0]

    @property
    def extfov_u_min(self) -> int:
        return self._extfov_vu_min[1]

    @property
    def extfov_u_max(self) -> int:
        return self._extfov_vu_max[1]

    @property
    def closest_planet(self) -> str | None:
        return self._closest_planet

    def reset_all(self) -> None:
        """Resets all cached Backplanes and Meshgrids to their initial state.

        Clears all cached computations, forcing them to be regenerated on next access.
        """

        # Standard FOV
        self._meshgrid = None
        self._bp = None

        # Extended FOV
        self._ext_meshgrid = None
        self._ext_bp = None

        # Standard FOV at four corners only
        self._corner_meshgrid = None
        self._corner_bp = None

        # Extended FOV at four corners only
        self._ext_corner_meshgrid = None
        self._ext_corner_bp = None

        # Center pixel only
        self._center_meshgrid = None
        self._center_bp = None

    @property
    def bp(self) -> Backplane:
        """Returns a Backplane for the entire original FOV, creating it if needed."""

        if self._bp is None:
            self._bp = Backplane(self, meshgrid=self.meshgrid())

        return self._bp

    @property
    def ext_bp(self) -> Backplane:
        """Create a Backplane for the entire extended FOV."""

        if self._ext_bp is None:
            if self._extfov_margin_vu == (0, 0):
                self._ext_bp = self.bp
            else:
                ext_meshgrid = Meshgrid.for_fov(
                    self.fov,
                    origin=(self.extfov_u_min+.5, self.extfov_v_min+.5),
                    limit=(self.extfov_u_max+.5, self.extfov_v_max+.5),
                    swap=True)
                self._ext_bp = Backplane(self, meshgrid=ext_meshgrid)

        return self._ext_bp

    @property
    def corner_bp(self) -> Backplane:
        """Create a Backplane with points only in the four corners of the original FOV."""

        if self._corner_bp is None:
            corner_meshgrid = Meshgrid.for_fov(
                self.fov,
                origin=(0, 0),
                limit=self._data_shape_uv,
                undersample=self._data_shape_uv,
                swap=True)
            self._corner_bp = Backplane(self, meshgrid=corner_meshgrid)

        return self._corner_bp

    @property
    def ext_corner_bp(self) -> Backplane:
        """Create a Backplane with points only in the four corners of the extended FOV."""

        if self._ext_corner_bp is None:
            if self._extfov_margin_vu == (0, 0):
                self._ext_corner_bp = self.corner_bp
            else:
                ext_corner_meshgrid = Meshgrid.for_fov(
                    self.fov,
                    origin=(self.extfov_u_min, self.extfov_v_min),
                    limit=(self.extfov_u_max+1, self.extfov_v_max+1),
                    undersample=(self._extdata_shape_uv[0], self._extdata_shape_uv[1]),
                    swap=True)
                self._ext_corner_bp = Backplane(self, meshgrid=ext_corner_meshgrid)

        return self._ext_corner_bp

    @property
    def center_bp(self) -> Backplane:
        """Create a Backplane with only a single point in the center."""

        if self._center_bp is None:
            center_meshgrid = Meshgrid.for_fov(
                self.fov,
                origin=(self._data_shape_uv[0]//2, self._data_shape_uv[1]//2),
                limit=(self._data_shape_uv[0]//2, self._data_shape_uv[1]//2),
                swap=True)
            self._center_bp = Backplane(self, meshgrid=center_meshgrid)
        return self._center_bp

    def extract_offset_image(self,
                             img: NDArrayType[NPType],
                             offset: tuple[float, float]) -> NDArrayType[NPType]:
        """Extracts a subimage from the given img at the given offset (dv,du).

        Parameters:
            img: Image to extract the subimage from. This must be the same shape as the
                extended FOV.
            offset: Offset (dv,du) to extract the subimage at.

        Returns:
            The extracted subimage. This will be the same shape as the original FOV.
        """

        if img.shape != self.extdata_shape_vu:
            raise ValueError(f'img shape {img.shape} must equal extdata shape '
                             f'{self.extdata_shape_vu}')
        v_size, u_size = self.extdata_shape_vu
        offset_v = int(offset[0]) + self.extfov_margin_v
        offset_u = int(offset[1]) + self.extfov_margin_u
        if (offset_v < 0 or offset_u < 0 or
            offset_v + self.data_shape_v > v_size or
            offset_u + self.data_shape_u > u_size):
            raise ValueError('offset produces out-of-bounds subimage slice')
        return img[offset_v:offset_v+self.data_shape_v,
                   offset_u:offset_u+self.data_shape_u]

    def _ra_dec_limits(self,
                       bp: Backplane,
                       apparent: bool = True) -> tuple[float, float, float, float]:
        """Find the RA and DEC limits of an observation.

        Parameters:
            bp: Backplane to use.
            apparent: Whether to compensate for aberration and light travel time.

        Returns:
            ra_min, ra_max, dec_min, dec_max (radians)
        """

        ra = bp.right_ascension(apparent=apparent)
        dec = bp.declination(apparent=apparent)

        ra_min = ra.min()
        ra_max = ra.max()
        if ra_max-ra_min > oops.PI:
            # Wrap around
            ra_min = ra[np.where(ra > np.pi)].min()
            ra_max = ra[np.where(ra < np.pi)].max()

        dec_min = dec.min()
        dec_max = dec.max()
        if dec_max-dec_min > oops.PI:
            # Wrap around
            dec_min = dec[np.where(dec > np.pi)].min()
            dec_max = dec[np.where(dec < np.pi)].max()

        ra_min = ra_min.vals
        ra_max = ra_max.vals
        dec_min = dec_min.vals
        dec_max = dec_max.vals

        return ra_min, ra_max, dec_min, dec_max

    def ra_dec_limits(self,
                      apparent: bool = True) -> tuple[float, float, float, float]:
        """Finds the right ascension and declination limits of the observation using the
        standard FOV.

        Parameters:
            apparent: Whether to compensate for aberration and light travel time.

        Returns:
            A tuple containing (ra_min, ra_max, dec_min, dec_max) in radians.
        """

        return self._ra_dec_limits(self.corner_bp, apparent=apparent)

    def ra_dec_limits_ext(self,
                          apparent: bool = True) -> tuple[float, float, float, float]:
        """Finds the right ascension and declination limits of the observation using the
        extended FOV.

        Parameters:
            apparent: Whether to compensate for aberration and light travel time.

        Returns:
            A tuple containing (ra_min, ra_max, dec_min, dec_max) in radians.
        """

        return self._ra_dec_limits(self.ext_corner_bp, apparent=apparent)

    def sun_body_distance(self,
                          body: str) -> float:
        """Computes the distance from the Sun to the specified celestial body in
        kilometers.

        Parameters:
            body: Name of the celestial body.

        Returns:
            Distance in kilometers from the Sun to the specified body.
        """

        target_sun_path = oops.path.Path.as_waypoint(body.upper()).wrt('SUN')
        sun_event = target_sun_path.event_at_time(self.midtime)
        return cast(float, sun_event.pos.norm().vals)

    def body_distance(self,
                      body: str) -> float:
        """Computes the distance from the spacecraft to the specified celestial body.

        Parameters:
            body: Name of the celestial body.

        Returns:
            Distance in kilometers from the spacecraft to the specified body.
        """

        return cast(float, self.center_bp.center_distance(body).vals)
