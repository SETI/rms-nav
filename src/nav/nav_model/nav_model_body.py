"""Catalog-driven body NavModel.

Renders one body's predicted appearance from SPICE, classifies it
against the design's emission rules (limb arc vs. body disc vs. blob vs.
terminator), and emits one ``NavFeature`` per surviving feature type.

The pipeline:

1. Builds an oversampled meshgrid around the predicted body bounding
   box so the limb silhouette is anti-aliased.
2. Extracts the limb and terminator polylines from the discrete
   silhouette masks.
3. Looks up the per-body shape parameters in
   ``nav.nav_model.body_shape.BODY_SHAPE_TABLE``.
4. Decides which features to emit by computing
   ``limb_uncertainty_px`` and the ``visible_lit_fraction`` /
   ``overflow_fraction`` quantities the design specifies.

The feature-by-feature emission rules:

- ``LIMB_ARC`` is emitted when ``limb_uncertainty_px <=
  LIMB_ARC_MAX_UNCERTAINTY_PX`` and there are surviving limb vertices.
- ``BODY_BLOB`` is emitted when the predicted disc diameter is at least
  the body's ``min_blob_diameter_px`` *and* the limb uncertainty is
  too high for ``LIMB_ARC``.
- ``BODY_DISC`` is emitted alongside ``LIMB_ARC`` when the body fits
  inside the FOV with at least ``BODY_DISC_MIN_VISIBLE_LIT_FRACTION``
  of its lit side visible and ``overflow_fraction`` below
  ``BODY_DISC_MAX_OVERFLOW_FRACTION``.
- ``TERMINATOR_ARC`` is emitted when the terminator polyline has at
  least ``TERMINATOR_MIN_VERTICES`` surviving vertices and the
  phase-angle factor (``sin(phase_angle)``) is above
  ``TERMINATOR_MIN_PHASE_FACTOR``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polymath
from oops import Meshgrid
from oops.backplane import Backplane

from nav.annotation import Annotations
from nav.config import Config
from nav.feature.constants import (
    INCIDENCE_FACTOR_ANGLE_CAP_DEG,
    INCIDENCE_FACTOR_CLIP_DEG,
    MAX_INCIDENCE_FACTOR_CAP,
)
from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import (
    BodyBlobFlags,
    BodyDiscFlags,
    LimbArcFlags,
    TerminatorArcFlags,
)
from nav.feature.geometry import (
    BodyBlobGeometry,
    BodyDiscGeometry,
    LimbPolyline,
    TerminatorPolyline,
)
from nav.nav_model.body_shape import BODY_SHAPE_TABLE, DEFAULT_BODY_SHAPE, BodyShape
from nav.nav_model.nav_model import NavModel
from nav.nav_model.nav_model_body_base import NavModelBodyBase
from nav.support.constants import HALFPI
from nav.support.image import filter_downsample, shift_array
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from oops import Observation

    from nav.nav_orchestrator.nav_context import NavContext

from nav.support.filters import NavFilterKind, NavFilterSpec

__all__ = [
    'BODY_DISC_MAX_OVERFLOW_FRACTION',
    'BODY_DISC_MIN_VISIBLE_LIT_FRACTION',
    'BODY_POSITION_SLOP_FRAC',
    'LIMB_ARC_MAX_UNCERTAINTY_PX',
    'TERMINATOR_MIN_PHASE_FACTOR',
    'TERMINATOR_MIN_VERTICES',
    'NavModelBody',
]


BODY_POSITION_SLOP_FRAC: float = 0.05
"""Inflation factor for the body bbox before clipping.

The ``oops.inventory`` bounding box is sometimes a half-pixel too small.
Inflating it by 5% before clipping into the extfov keeps anti-aliased
limb pixels from being lost on the boundary.
"""


LIMB_ARC_MAX_UNCERTAINTY_PX: float = 2.0
"""Cap on the limb normal-sigma at which LIMB_ARC remains useful.

Above this value the feature is unreliable enough that BODY_BLOB or
BODY_DISC is the right emission instead.  Phase-5 calibration may
tighten this; the default is the design's "limb fits within a couple of
pixels" guideline.
"""


BODY_DISC_MIN_VISIBLE_LIT_FRACTION: float = 0.4
"""Minimum lit-and-in-FOV fraction for BODY_DISC emission.

Below 40% of the lit hemisphere visible, the disc match is too
asymmetric to be useful; BODY_BLOB or LIMB_ARC carries the load.
"""


BODY_DISC_MAX_OVERFLOW_FRACTION: float = 0.3
"""Maximum overflow fraction for BODY_DISC emission.

A body whose disc is more than 30% off-frame loses too much template
support for the correlation peak to be sharp.
"""


TERMINATOR_MIN_VERTICES: int = 8
"""Minimum surviving vertices for TERMINATOR_ARC emission."""


TERMINATOR_MIN_PHASE_FACTOR: float = 0.05
"""Minimum ``sin(phase_angle)`` for TERMINATOR_ARC emission.

Below sin(phase) ~= 0.05 (phase < 3 deg) the terminator is too close to
the limb to be photometrically distinguishable.
"""


@dataclass(frozen=True)
class _PolylineSampler:
    """Bundle of sampled limb / terminator polyline data.

    Encapsulates the per-vertex outputs of the silhouette extraction so
    multiple feature emitters can share the same data without
    re-running the discrete-mask traversal.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    incidence_rad: NDArrayFloatType
    km_per_pixel: NDArrayFloatType


class NavModelBody(NavModelBodyBase):
    """Catalog-driven body NavModel.

    Parameters:
        name: Model instance name (e.g. ``'body:MIMAS'``).
        obs: Observation snapshot.
        body_name: SPICE body name.
        inventory: Optional pre-computed inventory entry; pulled from
            ``obs.inventory`` on demand otherwise.
        config: Optional ``Config`` override.
    """

    _abstract = False

    def __init__(
        self,
        name: str,
        obs: Observation,
        body_name: str,
        *,
        inventory: dict[str, Any] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._body_name = body_name.upper()
        self._inventory = inventory
        self._model_img: NDArrayFloatType | None = None
        self._body_mask: NDArrayBoolType | None = None
        self._limb_mask: NDArrayBoolType | None = None
        self._terminator_mask: NDArrayBoolType | None = None
        self._limb_sampler: _PolylineSampler | None = None
        self._terminator_sampler: _PolylineSampler | None = None
        self._km_per_pixel_at_limb: float = 0.0
        self._predicted_diameter_px: float = 0.0
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._subject_range_km: float = float('inf')
        self._visible_lit_fraction: float = 0.0
        self._overflow_fraction: float = 1.0
        self._phase_angle_factor: float = 0.0

    @classmethod
    def instances_for_obs(cls, obs: Observation) -> list[NavModel]:
        """Return one NavModelBody per body whose bbox lies inside extfov.

        Calls ``obs.inventory`` once with the planet + satellites list
        from ``config.satellites`` and constructs a NavModel for every
        entry whose ``inventory_body_in_extfov`` predicate fires.

        Parameters:
            obs: Observation snapshot.

        Returns:
            One ``NavModelBody`` per body present in the extfov.
        """
        from nav.config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG
        planet = getattr(obs, 'closest_planet', None)
        if planet is None:
            return []
        body_list: list[str] = [planet, *list(config.satellites(planet))]
        inventory_method = getattr(obs, 'inventory', None)
        if not callable(inventory_method):
            return []
        try:
            inv = inventory_method(body_list, return_type='full')
        except (TypeError, AttributeError, ValueError):
            return []
        in_extfov = getattr(obs, 'inventory_body_in_extfov', None)
        if not callable(in_extfov):
            return []
        out: list[NavModel] = []
        for body_name in body_list:
            entry = inv.get(body_name)
            if entry is None:
                continue
            if not in_extfov(entry):
                continue
            out.append(cls(f'body:{body_name}', obs, body_name, inventory=entry))
        return out

    def create_model(self) -> None:
        """Render the silhouette, masks, and polylines used by ``to_features``."""
        start_time = now_dt()
        self._metadata.clear()
        self._metadata['start_time'] = start_time.isoformat()
        self._metadata['end_time'] = None
        self._metadata['elapsed_time_sec'] = None
        self._metadata['body_name'] = self._body_name
        with self._logger.open(f'CREATE BODY MODEL FOR: {self._body_name}'):
            self._render()
        end_time = now_dt()
        self._metadata['end_time'] = end_time.isoformat()
        self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _render(self) -> None:
        """Populate masks, polyline samplers, and metadata."""
        obs = self.obs
        ext_bp = obs.ext_bp
        body_name = self._body_name
        body_config = self._config.bodies
        if self._inventory is None:
            self._inventory = obs.inventory([body_name], return_type='full')[body_name]
        inventory = self._inventory

        sub_solar_lon = float(np.degrees(ext_bp.sub_solar_longitude(body_name).vals))
        sub_solar_lat = float(np.degrees(ext_bp.sub_solar_latitude(body_name).vals))
        sub_observer_lon = float(np.degrees(ext_bp.sub_observer_longitude(body_name).vals))
        sub_observer_lat = float(np.degrees(ext_bp.sub_observer_latitude(body_name).vals))
        phase_angle_deg = float(np.degrees(ext_bp.center_phase_angle(body_name).vals))
        self._metadata['sub_solar_lon_deg'] = sub_solar_lon
        self._metadata['sub_solar_lat_deg'] = sub_solar_lat
        self._metadata['sub_observer_lon_deg'] = sub_observer_lon
        self._metadata['sub_observer_lat_deg'] = sub_observer_lat
        self._metadata['phase_angle_deg'] = phase_angle_deg
        self._phase_angle_factor = float(np.sin(np.radians(phase_angle_deg)))
        self._subject_range_km = float(inventory.get('range', float('inf')))

        bb_area = float(inventory['u_pixel_size'] * inventory['v_pixel_size'])
        self._metadata['bbox_area_px'] = bb_area
        self._metadata['size_ok'] = bool(bb_area >= body_config.min_bounding_box_area)

        u_min_unc = int(inventory['u_min_unclipped'])
        u_max_unc = int(inventory['u_max_unclipped'])
        v_min_unc = int(inventory['v_min_unclipped'])
        v_max_unc = int(inventory['v_max_unclipped'])
        u_slop = int((u_max_unc - u_min_unc) * BODY_POSITION_SLOP_FRAC)
        v_slop = int((v_max_unc - v_min_unc) * BODY_POSITION_SLOP_FRAC)
        u_min = u_min_unc - u_slop
        u_max = u_max_unc + u_slop
        v_min = v_min_unc - v_slop
        v_max = v_max_unc + v_slop
        u_min, v_min = obs.clip_extfov(u_min, v_min)
        u_max, v_max = obs.clip_extfov(u_max, v_max)
        if u_min == u_max == obs.extfov_u_max:
            u_min -= 1
        if u_min == u_max == obs.extfov_u_min:
            u_max += 1
        if v_min == v_max == obs.extfov_v_max:
            v_min -= 1
        if v_min == v_max == obs.extfov_v_min:
            v_max += 1

        guaranteed_visible = (
            u_min >= obs.extfov_margin_u
            and u_max <= obs.data_shape_u - 1 - obs.extfov_margin_u
            and v_min >= obs.extfov_margin_v
            and v_max <= obs.data_shape_v - 1 - obs.extfov_margin_v
        )
        self._metadata['guaranteed_visible_in_fov'] = guaranteed_visible

        model_img, limb_mask, terminator_mask, body_mask, sampler_data = (
            self._build_backplane_model(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)
        )
        self._model_img = model_img
        self._body_mask = body_mask
        self._limb_mask = limb_mask
        self._terminator_mask = terminator_mask

        u_center_data = (u_min_unc + u_max_unc) / 2.0
        v_center_data = (v_min_unc + v_max_unc) / 2.0
        self._predicted_center_vu = (
            float(v_center_data + obs.extfov_margin_v),
            float(u_center_data + obs.extfov_margin_u),
        )
        diameter = max(
            float(inventory['u_pixel_size']),
            float(inventory['v_pixel_size']),
        )
        self._predicted_diameter_px = diameter
        self._bbox_extfov_vu = (
            int(v_min + obs.extfov_margin_v),
            int(u_min + obs.extfov_margin_u),
            int(v_max + obs.extfov_margin_v + 1),
            int(u_max + obs.extfov_margin_u + 1),
        )

        self._km_per_pixel_at_limb = sampler_data['km_per_pixel_mean']
        self._limb_sampler = sampler_data['limb_sampler']
        self._terminator_sampler = sampler_data['terminator_sampler']
        self._visible_lit_fraction = float(sampler_data['visible_lit_fraction'])
        self._overflow_fraction = float(sampler_data['overflow_fraction'])
        self._metadata['km_per_pixel_at_limb'] = self._km_per_pixel_at_limb
        self._metadata['predicted_diameter_px'] = self._predicted_diameter_px
        self._metadata['visible_lit_fraction'] = self._visible_lit_fraction
        self._metadata['overflow_fraction'] = self._overflow_fraction

    def _build_backplane_model(
        self,
        *,
        u_min: int,
        u_max: int,
        v_min: int,
        v_max: int,
    ) -> tuple[
        NDArrayFloatType,
        NDArrayBoolType,
        NDArrayBoolType,
        NDArrayBoolType,
        dict[str, Any],
    ]:
        """Build the silhouette, limb / terminator masks, and samplers.

        Renders an oversampled Lambert silhouette over the body bbox,
        downsamples to the extfov grid, extracts the discrete limb and
        terminator masks, and samples per-vertex polyline data.

        Parameters:
            u_min, u_max, v_min, v_max: Body bounding box in extfov
                coordinates (already clipped).

        Returns:
            ``(model_img, limb_mask, terminator_mask, body_mask, info)``
            where ``info`` carries the polyline samplers, the mean
            km/pixel at the limb, and the visibility / overflow
            fractions.
        """
        obs = self.obs
        body_name = self._body_name
        body_config = self._config.bodies
        inventory = self._inventory
        assert inventory is not None  # populated in _render

        oversample_u = max(
            int(
                np.floor(
                    body_config.oversample_edge_limit / max(np.ceil(inventory['u_pixel_size']), 1)
                )
            ),
            1,
        )
        oversample_v = max(
            int(
                np.floor(
                    body_config.oversample_edge_limit / max(np.ceil(inventory['v_pixel_size']), 1)
                )
            ),
            1,
        )
        oversample_u = min(oversample_u, body_config.oversample_maximum)
        oversample_v = min(oversample_v, body_config.oversample_maximum)
        restr_u_min = u_min + 1.0 / (2 * oversample_u)
        restr_u_max = u_max + 1 - 1.0 / (2 * oversample_u)
        restr_v_min = v_min + 1.0 / (2 * oversample_v)
        restr_v_max = v_max + 1 - 1.0 / (2 * oversample_v)
        restr_meshgrid = Meshgrid.for_fov(
            obs.fov,
            origin=(restr_u_min, restr_v_min),
            limit=(restr_u_max, restr_v_max),
            oversample=(oversample_u, oversample_v),
            swap=True,
        )
        restr_bp = Backplane(obs, meshgrid=restr_meshgrid)

        oversampled_incidence_mvals = restr_bp.incidence_angle(body_name).mvals
        downsampled_incidence_mvals = filter_downsample(
            oversampled_incidence_mvals, oversample_v, oversample_u
        )
        incidence_scalar = polymath.Scalar(downsampled_incidence_mvals)

        body_mask_invalid = incidence_scalar.expand_mask().mask
        body_mask_valid = ~body_mask_invalid
        limb_mask_neighbor = (
            shift_array(body_mask_invalid, (-1, 0))
            | shift_array(body_mask_invalid, (1, 0))
            | shift_array(body_mask_invalid, (0, -1))
            | shift_array(body_mask_invalid, (0, 1))
        )
        limb_mask_local: NDArrayBoolType = body_mask_valid & limb_mask_neighbor

        # Terminator mask: pixels whose incidence crosses 90 deg
        incidence_vals = incidence_scalar.vals
        is_lit = (incidence_vals < HALFPI) & body_mask_valid
        is_dark = (incidence_vals >= HALFPI) & body_mask_valid
        # A pixel is on the terminator if it is lit and any neighbour is dark.
        terminator_local: NDArrayBoolType = is_lit & (
            shift_array(is_dark, (-1, 0))
            | shift_array(is_dark, (1, 0))
            | shift_array(is_dark, (0, -1))
            | shift_array(is_dark, (0, 1))
        )

        if not body_mask_valid.any() or incidence_scalar[body_mask_valid].min() >= HALFPI:
            local_model: NDArrayFloatType = np.zeros_like(body_mask_valid, dtype=np.float64)
            local_model[body_mask_valid] = 0.01
        else:
            if body_config.use_lambert:
                lambert_oversampled = restr_bp.lambert_law(body_name).mvals.filled(0.0)
                local_model = filter_downsample(lambert_oversampled, oversample_v, oversample_u)
                local_model = local_model + 0.05
                local_model[body_mask_invalid] = 0.0
            else:
                local_model = body_mask_valid.astype(np.float64)
            if body_config.use_albedo and body_name in body_config.geometric_albedo:
                albedo = float(body_config.geometric_albedo[body_name])
                local_model = local_model * albedo

        # Promote to extfov-shaped arrays.
        ext_v0 = v_min + obs.extfov_margin_v
        ext_u0 = u_min + obs.extfov_margin_u
        v_size = local_model.shape[0]
        u_size = local_model.shape[1]
        model_img = obs.make_extfov_zeros()
        limb_mask = obs.make_extfov_false()
        body_mask = obs.make_extfov_false()
        terminator_mask = obs.make_extfov_false()
        v_slice = slice(ext_v0, ext_v0 + v_size)
        u_slice = slice(ext_u0, ext_u0 + u_size)
        model_img[v_slice, u_slice] = local_model
        limb_mask[v_slice, u_slice] = limb_mask_local
        terminator_mask[v_slice, u_slice] = terminator_local
        body_mask[v_slice, u_slice] = body_mask_valid

        km_per_pixel_arr: NDArrayFloatType = (
            restr_bp.resolution(body_name).mvals.filled(0.0)
            if body_mask_valid.any()
            else np.zeros_like(body_mask_valid, dtype=np.float64)
        )
        # Downsample km/pixel to the same grid as the masks.
        km_per_pixel_local = filter_downsample(km_per_pixel_arr, oversample_v, oversample_u)

        limb_sampler = _build_polyline_sampler(
            local_mask=limb_mask_local,
            incidence_local=incidence_vals,
            km_per_pixel_local=km_per_pixel_local,
            ext_v0=ext_v0,
            ext_u0=ext_u0,
        )
        terminator_sampler = _build_polyline_sampler(
            local_mask=terminator_local,
            incidence_local=incidence_vals,
            km_per_pixel_local=km_per_pixel_local,
            ext_v0=ext_v0,
            ext_u0=ext_u0,
        )

        sensor_v0 = obs.extfov_margin_v
        sensor_v1 = obs.extfov_margin_v + obs.data_shape_v
        sensor_u0 = obs.extfov_margin_u
        sensor_u1 = obs.extfov_margin_u + obs.data_shape_u
        in_sensor = np.zeros_like(body_mask, dtype=bool)
        in_sensor[sensor_v0:sensor_v1, sensor_u0:sensor_u1] = True
        lit_mask = body_mask.copy()
        if local_model.size > 0:
            lit_arr = obs.make_extfov_false()
            lit_arr[v_slice, u_slice] = is_lit
            lit_mask = lit_arr
        body_total = int(np.count_nonzero(body_mask))
        body_visible = int(np.count_nonzero(body_mask & in_sensor))
        # Per Part 1: ``visible_lit_fraction`` is the fraction of the
        # *whole predicted disc* (lit and dark) whose cos(incidence) >= 0
        # *and* which lies inside the sensor FOV.
        lit_visible_in_fov = int(np.count_nonzero(lit_mask & in_sensor))
        visible_lit_fraction = lit_visible_in_fov / max(body_total, 1)
        overflow_fraction = 1.0 - (body_visible / max(body_total, 1))

        if limb_mask_local.any() and km_per_pixel_local[limb_mask_local].size:
            km_per_pixel_mean = float(np.mean(km_per_pixel_local[limb_mask_local]))
        else:
            km_per_pixel_mean = 0.0

        info: dict[str, Any] = {
            'limb_sampler': limb_sampler,
            'terminator_sampler': terminator_sampler,
            'km_per_pixel_mean': km_per_pixel_mean,
            'visible_lit_fraction': visible_lit_fraction,
            'overflow_fraction': overflow_fraction,
        }
        return model_img, limb_mask, terminator_mask, body_mask, info

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the body's NavFeatures per the design's gate rules."""
        del context
        if self._body_mask is None:
            return []
        shape = BODY_SHAPE_TABLE.get(self._body_name, DEFAULT_BODY_SHAPE)
        features: list[NavFeature] = []
        limb_uncertainty_px = self._limb_uncertainty_px(shape)
        limb_arc_emitted = False
        if (
            self._limb_sampler is not None
            and self._limb_sampler.vertices_vu.size > 0
            and limb_uncertainty_px <= LIMB_ARC_MAX_UNCERTAINTY_PX
        ):
            features.append(
                _build_limb_arc(
                    body_name=self._body_name,
                    sampler=self._limb_sampler,
                    shape=shape,
                    bbox=self._bbox_extfov_vu,
                    subject_range_km=self._subject_range_km,
                    psf_sigma_px=self._star_psf_sigma(),
                    source_model=self.name,
                )
            )
            limb_arc_emitted = True
        else:
            if self._predicted_diameter_px >= shape.min_blob_diameter_px:
                features.append(self._build_blob_feature(shape))

        if self._should_emit_disc(limb_arc_emitted):
            features.append(self._build_disc_feature(shape))

        terminator_feature = self._maybe_build_terminator(shape)
        if terminator_feature is not None:
            features.append(terminator_feature)
        return features

    def to_annotations(self, context: NavContext) -> Annotations:
        """Reuse the shared body annotation helper."""
        del context
        if self._model_img is None or self._body_mask is None or self._limb_mask is None:
            return Annotations()
        v_center, u_center = self._predicted_center_vu
        return self._create_annotations(
            round(u_center - self.obs.extfov_margin_u),
            round(v_center - self.obs.extfov_margin_v),
            self._model_img,
            self._limb_mask,
            self._body_mask,
        )

    def _star_psf_sigma(self) -> float:
        """Return the per-pixel PSF sigma from ``obs.star_psf()``."""
        psf = self.obs.star_psf()
        if hasattr(psf, 'sigma'):
            return float(psf.sigma)
        fwhm_method = getattr(psf, 'fwhm', None)
        if callable(fwhm_method):
            return float(fwhm_method()) / 2.3548200450309493
        raise AttributeError(f'PSF {type(psf).__name__} exposes neither sigma nor fwhm()')

    def _limb_uncertainty_px(self, shape: BodyShape) -> float:
        """Return the design's ``limb_uncertainty_px`` for this body."""
        if self._km_per_pixel_at_limb <= 0.0:
            return float('inf')
        return shape.ellipsoid_residual_km / self._km_per_pixel_at_limb

    def _should_emit_disc(self, limb_arc_emitted: bool) -> bool:
        """Return True when BODY_DISC should be emitted alongside other features."""
        if not limb_arc_emitted:
            return False
        if self._visible_lit_fraction < BODY_DISC_MIN_VISIBLE_LIT_FRACTION:
            return False
        return not self._overflow_fraction > BODY_DISC_MAX_OVERFLOW_FRACTION

    def _build_disc_feature(self, shape: BodyShape) -> NavFeature:
        """Construct the BODY_DISC feature (template + geometry + flags)."""
        assert self._model_img is not None
        assert self._body_mask is not None
        return NavFeature(
            feature_id=f'body_disc:{self._body_name}',
            feature_type=NavFeatureType.BODY_DISC,
            source_model=self.name,
            geometry=BodyDiscGeometry(
                bbox_extfov_vu=self._bbox_extfov_vu,
                predicted_center_vu=self._predicted_center_vu,
                overflow_fraction=self._overflow_fraction,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=float(min(0.5, shape.albedo_variation)),
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=_disc_reliability(
                visible_lit_fraction=self._visible_lit_fraction,
                overflow_fraction=self._overflow_fraction,
                diameter_px=self._predicted_diameter_px,
            ),
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=self._visible_lit_fraction,
                overflow_fraction=self._overflow_fraction,
            ),
            usable_types=frozenset({NavFeatureType.BODY_DISC}),
            flags=BodyDiscFlags(
                body_name=self._body_name,
                overflow_fov_fraction=self._overflow_fraction,
            ),
            template_img=self._model_img,
            template_mask=self._body_mask,
        )

    def _build_blob_feature(self, shape: BodyShape) -> NavFeature:
        """Construct the BODY_BLOB feature."""
        del shape
        # Estimate per-pixel SNR from the brightest pixel in the model.
        assert self._model_img is not None
        assert self._body_mask is not None
        snr = float(self._model_img[self._body_mask].mean()) if self._body_mask.any() else 0.0
        sigma_centroid = self._predicted_diameter_px / (
            2.0 * math.sqrt(max(int(self._body_mask.sum()), 1)) * max(snr, 1e-6)
        )
        cov = (sigma_centroid * sigma_centroid) * np.eye(2, dtype=np.float64)
        return NavFeature(
            feature_id=f'body_blob:{self._body_name}',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model=self.name,
            geometry=BodyBlobGeometry(
                predicted_center_vu=self._predicted_center_vu,
                bbox_extfov_vu=self._bbox_extfov_vu,
                predicted_diameter_px=self._predicted_diameter_px,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=cov,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=_blob_reliability(snr=snr, diameter_px=self._predicted_diameter_px),
            reliability_reasons=NavReliabilityBreakdown(
                blob_snr=min(1.0, snr / 10.0),
                blob_extent_px=min(1.0, self._predicted_diameter_px / 30.0),
            ),
            usable_types=frozenset({NavFeatureType.BODY_BLOB}),
            flags=BodyBlobFlags(
                body_name=self._body_name,
                predicted_diameter_px=self._predicted_diameter_px,
            ),
        )

    def _maybe_build_terminator(self, shape: BodyShape) -> NavFeature | None:
        """Build TERMINATOR_ARC when the design's terminator gates pass."""
        sampler = self._terminator_sampler
        if sampler is None or sampler.vertices_vu.shape[0] < TERMINATOR_MIN_VERTICES:
            return None
        if self._phase_angle_factor < TERMINATOR_MIN_PHASE_FACTOR:
            return None
        sigma_normal_per_vertex_px = _sigma_normal_per_vertex(
            sampler=sampler,
            shape=shape,
            psf_sigma_px=self._star_psf_sigma(),
            include_albedo=True,
        )
        sigma_tangent_per_vertex_px = np.full_like(sigma_normal_per_vertex_px, 0.5)
        return NavFeature(
            feature_id=f'terminator_arc:{self._body_name}',
            feature_type=NavFeatureType.TERMINATOR_ARC,
            source_model=self.name,
            geometry=TerminatorPolyline(
                vertices_vu=sampler.vertices_vu,
                normals_vu=sampler.normals_vu,
                sigma_normal_per_vertex_px=sigma_normal_per_vertex_px,
                sigma_tangent_per_vertex_px=sigma_tangent_per_vertex_px,
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=_terminator_reliability(
                visible_arc_fraction=_visible_arc_fraction(sampler),
                albedo_variation=shape.albedo_variation,
                phase_factor=self._phase_angle_factor,
            ),
            reliability_reasons=NavReliabilityBreakdown(
                visible_arc_fraction=_visible_arc_fraction(sampler),
                albedo_penalty=min(1.0, shape.albedo_variation),
            ),
            usable_types=frozenset({NavFeatureType.TERMINATOR_ARC}),
            flags=TerminatorArcFlags(
                body_name=self._body_name,
                visible_arc_fraction=_visible_arc_fraction(sampler),
                phase_angle_factor=min(1.0, self._phase_angle_factor),
            ),
        )


def _build_limb_arc(
    *,
    body_name: str,
    sampler: _PolylineSampler,
    shape: BodyShape,
    bbox: tuple[int, int, int, int],
    subject_range_km: float,
    psf_sigma_px: float,
    source_model: str,
) -> NavFeature:
    """Construct the LIMB_ARC NavFeature for one body."""
    sigma_normal_per_vertex_px = _sigma_normal_per_vertex(
        sampler=sampler, shape=shape, psf_sigma_px=psf_sigma_px, include_albedo=False
    )
    sigma_tangent_per_vertex_px = np.full_like(sigma_normal_per_vertex_px, 0.5)
    visible_arc_fraction = _visible_arc_fraction(sampler)
    incidence_factor_mean = float(np.mean(_incidence_factor_array(sampler.incidence_rad)))
    return NavFeature(
        feature_id=f'limb_arc:{body_name}',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model=source_model,
        geometry=LimbPolyline(
            vertices_vu=sampler.vertices_vu,
            normals_vu=sampler.normals_vu,
            sigma_normal_per_vertex_px=sigma_normal_per_vertex_px,
            sigma_tangent_per_vertex_px=sigma_tangent_per_vertex_px,
            bbox_extfov_vu=bbox,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=_limb_reliability(
            visible_arc_fraction=visible_arc_fraction,
            visible_arc_px=float(sampler.vertices_vu.shape[0]),
            mean_incidence_factor=incidence_factor_mean,
        ),
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
            incidence_factor=min(1.0, incidence_factor_mean / MAX_INCIDENCE_FACTOR_CAP),
        ),
        usable_types=frozenset({NavFeatureType.LIMB_ARC}),
        flags=LimbArcFlags(
            body_name=body_name,
            visible_arc_fraction=visible_arc_fraction,
        ),
    )


def _build_polyline_sampler(
    *,
    local_mask: NDArrayBoolType,
    incidence_local: NDArrayFloatType,
    km_per_pixel_local: NDArrayFloatType,
    ext_v0: int,
    ext_u0: int,
) -> _PolylineSampler:
    """Sample a polyline along the True pixels of ``local_mask``.

    The local mask is a 1-pixel-wide ridge inside the body silhouette; we
    return a parallel array of vertex coordinates in extfov space, the
    outward-pointing normal at each vertex (from the discrete-mask
    gradient), the per-vertex incidence angle, and the per-vertex km/px
    scale.
    """
    vs, us = np.where(local_mask)
    if vs.size == 0:
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return _PolylineSampler(
            vertices_vu=empty,
            normals_vu=empty,
            incidence_rad=np.empty(0, dtype=np.float64),
            km_per_pixel=np.empty(0, dtype=np.float64),
        )
    vertices_vu: NDArrayFloatType = np.stack(
        [vs.astype(np.float64) + ext_v0, us.astype(np.float64) + ext_u0], axis=1
    )
    rows, cols = local_mask.shape
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        # Outward normal: gradient of body-mask values around the vertex.
        # The body-side neighbour is True; the off-body neighbour is False.
        v_dir = 0.0
        u_dir = 0.0
        if v > 0 and not local_mask[v - 1, u]:
            v_dir = -1.0
        elif v < rows - 1 and not local_mask[v + 1, u]:
            v_dir = 1.0
        if u > 0 and not local_mask[v, u - 1]:
            u_dir = -1.0
        elif u < cols - 1 and not local_mask[v, u + 1]:
            u_dir = 1.0
        norm = math.hypot(v_dir, u_dir) or 1.0
        normals_vu[i, 0] = v_dir / norm
        normals_vu[i, 1] = u_dir / norm
    incidence_rad = incidence_local[vs, us].astype(np.float64)
    km_per_pixel = km_per_pixel_local[vs, us].astype(np.float64)
    return _PolylineSampler(
        vertices_vu=vertices_vu,
        normals_vu=normals_vu,
        incidence_rad=incidence_rad,
        km_per_pixel=km_per_pixel,
    )


def _incidence_factor_array(incidence_rad: NDArrayFloatType) -> NDArrayFloatType:
    """Return the design's ``incidence_factor`` array, capped per the constants."""
    deg = np.degrees(incidence_rad)
    deg_clipped = np.clip(deg, 0.0, INCIDENCE_FACTOR_CLIP_DEG)
    factor = 1.0 / np.cos(np.radians(np.minimum(deg_clipped, INCIDENCE_FACTOR_ANGLE_CAP_DEG))) - 1.0
    factor = np.clip(factor, 0.0, MAX_INCIDENCE_FACTOR_CAP)
    out: NDArrayFloatType = factor.astype(np.float64)
    return out


def _sigma_normal_per_vertex(
    *,
    sampler: _PolylineSampler,
    shape: BodyShape,
    psf_sigma_px: float,
    include_albedo: bool,
) -> NDArrayFloatType:
    """Compute the per-vertex normal-sigma per the design.

    Implements the formula from Part 1's "Position covariance per
    feature type" section, including the limb-softness term that uses
    the per-vertex km/px scale and the optional albedo / photometric
    contribution for terminator arcs.
    """
    incidence_factor = _incidence_factor_array(sampler.incidence_rad)
    km_per_pixel = np.where(sampler.km_per_pixel > 0.0, sampler.km_per_pixel, np.nan)
    limb_softness_km = psf_sigma_px * km_per_pixel
    base = (
        shape.ellipsoid_residual_km**2
        + shape.crater_scale_km**2
        + (incidence_factor * limb_softness_km) ** 2
        + shape.spice_orbital_residual_km**2
    )
    if include_albedo:
        albedo_term = (shape.albedo_variation * limb_softness_km) ** 2
        photometric_term = (limb_softness_km * 0.5) ** 2
        base = base + albedo_term + photometric_term
    sigma_km = np.sqrt(np.maximum(base, 0.0))
    sigma_px = sigma_km / km_per_pixel
    return np.nan_to_num(sigma_px, nan=LIMB_ARC_MAX_UNCERTAINTY_PX, posinf=1e3, neginf=1e3)


def _visible_arc_fraction(sampler: _PolylineSampler) -> float:
    """Fraction of polyline vertices that are usable.

    The sampler already drops shadow / off-FOV vertices during
    construction; for the purposes of the reliability reason we report
    1.0 when any vertices survive.  When the table-driven shadow
    extractor is wired, this will be replaced by
    ``len(survivors) / len(total)``.
    """
    return 1.0 if sampler.vertices_vu.shape[0] > 0 else 0.0


def _limb_reliability(
    *, visible_arc_fraction: float, visible_arc_px: float, mean_incidence_factor: float
) -> float:
    """Sigmoid-of-sum reliability for LIMB_ARC features."""
    z = (
        -1.0
        + 1.5 * visible_arc_fraction
        + 1.0 * _sigmoid(visible_arc_px / 50.0)
        - 0.7 * mean_incidence_factor
    )
    return float(_sigmoid(z))


def _terminator_reliability(
    *, visible_arc_fraction: float, albedo_variation: float, phase_factor: float
) -> float:
    """Reliability of TERMINATOR_ARC mirroring the design's formula."""
    base = _sigmoid(-1.0 + 1.5 * visible_arc_fraction - 1.5 * albedo_variation)
    return float(base * min(1.0, phase_factor))


def _disc_reliability(
    *, visible_lit_fraction: float, overflow_fraction: float, diameter_px: float
) -> float:
    """Reliability of BODY_DISC per the design (no scoring alpha coefficients yet)."""
    sigmoid_term = _sigmoid(diameter_px / 30.0 - 1.0)
    return float(visible_lit_fraction * (1.0 - overflow_fraction) * sigmoid_term)


def _blob_reliability(*, snr: float, diameter_px: float) -> float:
    """Reliability of BODY_BLOB per the design (cap at 0.4)."""
    return float(_sigmoid(snr) * _sigmoid(diameter_px / 8.0 - 1.0) * 0.4)


def _sigmoid(x: float) -> float:
    """Logistic sigmoid (numerically safe)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
