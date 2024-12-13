from typing import Optional

import numpy as np

import oops
from oops.observation.snapshot import Snapshot
from oops.meshgrid import Meshgrid
from oops.backplane import Backplane

from nav.util.image import pad_image

from .obs import Obs


class ObsSnapshot(Obs, Snapshot):
    """Class the defines cached Backplane and Meshgrid operations."""

    def __init__(self,
                 snapshot: Snapshot,
                 extfov_margin: Optional[int | tuple[int, int]] = None):
        # Because the oops image read routines create a Snapshot and not a
        # Navigation class, we need to create a Navigation class that looks
        # like a Snapshot but then adds our own attributes. This cool trick
        # with __dict__ makes Navigation have the exact same set of attributes
        # as the original Snapshot. Warning: Do not use that Snapshot anymore!
        self.__dict__ = snapshot.__dict__

        assert len(self.data.shape) == 2
        self._data_shape_uv = self.data.shape[::-1]
        self._fov_uv_min = (0, 0)
        self._fov_uv_max = (self._data_shape_uv[0] - 1, self._data_shape_uv[1] - 1)

        if extfov_margin is None:
            self._extfov_margin = (0, 0)
        elif isinstance(extfov_margin, int):
            self._extfov_margin = (int(extfov_margin), int(extfov_margin))
        else:
            self._extfov_margin = (int(extfov_margin[0]), int(extfov_margin[1]))

        self._extdata = pad_image(self.data, self._extfov_margin)
        self._extdata_shape_uv = self._extdata.shape[::-1]
        self._extfov_uv_min = (-self._extfov_margin[0], -self._extfov_margin[1])
        self._extfov_uv_max = (self._data_shape_uv[0] + self._extfov_margin[0] - 1,
                               self._data_shape_uv[1] + self._extfov_margin[1] - 1)
        self.reset_all()

    def reset_all(self):
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
    def corner_bp(self):
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
    def ext_corner_bp(self):
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
    def center_bp(self):
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

    def sun_distance(self,
                     planet: str) -> float:
        """Compute the distance from the Sun to a planet in AU."""

        target_sun_path = oops.path.Path.as_waypoint(planet.upper()).wrt("SUN")
        sun_event = target_sun_path.event_at_time(self.midtime)
        solar_range = sun_event.pos.norm().vals / oops.AU

        return solar_range
