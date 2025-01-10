from typing import Any, Optional, cast

import numpy as np
import oops
from oops.observation.snapshot import Snapshot
from oops.meshgrid import Meshgrid
from oops.backplane import Backplane

from nav.config.global_config import ALL_PLANETS
from nav.util.image import pad_array
from nav.util.types import NDArrayFloatType, DTypeLike

from .obs import Obs


class ObsSnapshot(Obs, Snapshot):
    """Class that defines cached Backplane and Meshgrid operations on Snapshots."""

    def __init__(self,
                 snapshot: Snapshot,
                 extfov_margin: Optional[int | tuple[int, int]] = None,
                 **kwargs: Any):
        # Because the oops image read routines create a Snapshot and not a
        # Navigation class, we need to create a Navigation class that looks
        # like a Snapshot but then adds our own attributes. This cool trick
        # with __dict__ makes Navigation have the exact same set of attributes
        # as the original Snapshot. Warning: Do not use that Snapshot anymore!
        self.__dict__ = snapshot.__dict__

        super().__init__(**kwargs)

        assert len(self.data.shape) == 2
        self._data_shape_uv = cast(tuple[int, int], self.data.shape[::-1])
        self._fov_uv_min = (0, 0)
        self._fov_uv_max = (self._data_shape_uv[0] - 1, self._data_shape_uv[1] - 1)

        if extfov_margin is None:
            self._extfov_margin = (0, 0)
        elif isinstance(extfov_margin, int):
            self._extfov_margin = (int(extfov_margin), int(extfov_margin))
        else:
            self._extfov_margin = (int(extfov_margin[0]), int(extfov_margin[1]))

        self._extdata = pad_array(self.data, self._extfov_margin)
        self._extdata_shape_uv = cast(tuple[int, int], self._extdata.shape[::-1])
        self._extfov_uv_min = (-self._extfov_margin[0], -self._extfov_margin[1])
        self._extfov_uv_max = (self._data_shape_uv[0] + self._extfov_margin[0] - 1,
                               self._data_shape_uv[1] + self._extfov_margin[1] - 1)
        self.reset_all()

        closest_planet = None
        closest_dist = 1e38
        for planet in ALL_PLANETS:
            dist = self.body_distance(planet)
            if dist < closest_dist:
                closest_planet = planet
                closest_dist = dist
        assert closest_planet is not None

        self._closest_planet = closest_planet

    def make_fov_zeros(self,
                       dtype: DTypeLike = np.float64) -> NDArrayFloatType:
        return np.zeros(self.data.shape, dtype=dtype)

    def make_extfov_zeros(self,
                          dtype: DTypeLike = np.float64) -> NDArrayFloatType:
        return np.zeros(self.extdata.shape, dtype=dtype)

    def clip_fov(self,
                 u: int,
                 v: int) -> tuple[int, int]:
        return (int(np.clip(u, self.fov_u_min, self.fov_u_max)),
                int(np.clip(v, self.fov_v_min, self.fov_v_max)))

    def clip_extfov(self,
                    u: int,
                    v: int) -> tuple[int, int]:
        return (int(np.clip(u, self.extfov_u_min, self.extfov_u_max)),
                int(np.clip(v, self.extfov_v_min, self.extfov_v_max)))

    @property
    def data_shape_uv(self) -> tuple[int, int]:
        return self._data_shape_uv

    @property
    def data_shape_u(self) -> int:
        return self._data_shape_uv[0]

    @property
    def data_shape_v(self) -> int:
        return self._data_shape_uv[1]

    @property
    def fov_uv_min(self) -> tuple[int, int]:
        return self._fov_uv_min

    @property
    def fov_uv_max(self) -> tuple[int, int]:
        return self._fov_uv_max

    @property
    def fov_u_min(self) -> int:
        return self._fov_uv_min[0]

    @property
    def fov_u_max(self) -> int:
        return self._fov_uv_max[0]

    @property
    def fov_v_min(self) -> int:
        return self._fov_uv_min[1]

    @property
    def fov_v_max(self) -> int:
        return self._fov_uv_max[1]

    @property
    def extfov_margin(self) -> tuple[int, int]:
        return self._extfov_margin

    @property
    def extfov_margin_u(self) -> int:
        return self._extfov_margin[0]

    @property
    def extfov_margin_v(self) -> int:
        return self._extfov_margin[1]

    @property
    def extdata(self) -> NDArrayFloatType:
        return self._extdata

    @property
    def extdata_shape_uv(self) -> tuple[int, int]:
        return self._extdata_shape_uv

    @property
    def extfov_uv_min(self) -> tuple[int, int]:
        return self._extfov_uv_min

    @property
    def extfov_uv_max(self) -> tuple[int, int]:
        return self._extfov_uv_max

    @property
    def extfov_u_min(self) -> int:
        return self._extfov_uv_min[0]

    @property
    def extfov_u_max(self) -> int:
        return self._extfov_uv_max[0]

    @property
    def extfov_v_min(self) -> int:
        return self._extfov_uv_min[1]

    @property
    def extfov_v_max(self) -> int:
        return self._extfov_uv_max[1]

    @property
    def closest_planet(self) -> str | None:
        return self._closest_planet

    def reset_all(self) -> None:
        """Reset all Backplanes and Meshgrids to unknown."""

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
        """Create a Backplane for the entire original FOV."""

        if self._bp is None:
            self._bp = Backplane(self, meshgrid=self.meshgrid())

        return self._bp

    @property
    def ext_bp(self) -> Backplane:
        """Create a Backplane for the entire extended FOV."""

        if self._ext_bp is None:
            if self._extfov_margin == (0, 0):
                self._ext_bp = self.bp
            else:
                ext_meshgrid = Meshgrid.for_fov(
                    self.fov,
                    origin=(self._extfov_uv_min[0]+.5, self._extfov_uv_min[1]+.5),
                    limit=(self._extfov_uv_max[0]+.5, self._extfov_uv_max[1]+.5),
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
            if self._extfov_margin == (0, 0):
                self._ext_corner_bp = self.corner_bp
            else:
                ext_corner_meshgrid = Meshgrid.for_fov(
                    self.fov,
                    origin=self._extfov_uv_min,
                    limit=(self._extfov_uv_max[0]+1, self._extfov_uv_max[1]+1),
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

    def _ra_dec_limits(self,
                       bp: Backplane) -> tuple[float, float, float, float]:
        """Find the RA and DEC limits of an observation.

        Returns:
            ra_min, ra_max, dec_min, dec_max (radians)
        """

        ra = bp.right_ascension(apparent=True)
        dec = bp.declination(apparent=True)

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

    def ra_dec_limits(self) -> tuple[float, float, float, float]:
        """Find the RA and DEC limits of an observation using the standard FOV.

        Returns:
            ra_min, ra_max, dec_min, dec_max (radians)
        """

        return self._ra_dec_limits(self.corner_bp)

    def ra_dec_limits_ext(self) -> tuple[float, float, float, float]:
        """Find the RA and DEC limits of an observation using the extended FOV.

        Returns:
            ra_min, ra_max, dec_min, dec_max (radians)
        """

        return self._ra_dec_limits(self.ext_corner_bp)

    def sun_body_distance(self,
                          body: str) -> float:
        """Compute the distance from the Sun to an object in km."""

        target_sun_path = oops.path.Path.as_waypoint(body.upper()).wrt('SUN')
        sun_event = target_sun_path.event_at_time(self.midtime)
        return cast(float, sun_event.pos.norm().vals)

    def body_distance(self,
                      body: str) -> float:
        """Compute the distance from the spacecraft to an object in km."""

        return cast(float, self.center_bp.center_distance(body).vals)
