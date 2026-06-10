"""Catalog-driven body NavModel.

Renders one body's predicted appearance from SPICE, classifies it
against the design's emission rules (limb arc vs. body disc vs. blob vs.
terminator), and emits one ``NavFeature`` per surviving feature type.

The pipeline:

1. Builds an oversampled meshgrid around the predicted body bounding
   box so the limb silhouette is anti-aliased.
2. Extracts the limb and terminator polylines from the discrete
   silhouette masks.
3. Looks up the per-body shape parameters via
   :func:`nav.nav_model.body_shape.shape_for_body`, which merges the
   operator-curated ``config_220_body_shape.yaml`` over the hard-coded
   ``BODY_SHAPE_TABLE`` profiles.
4. Decides which features to emit by computing
   ``limb_uncertainty_px`` and the ``visible_lit_fraction`` /
   ``overflow_fraction`` quantities the design specifies.

The feature-by-feature emission rules:

- ``LIMB_ARC`` is emitted when ``limb_uncertainty_px <=
  LIMB_ARC_MAX_UNCERTAINTY_PX`` and there are surviving limb vertices.
- ``BODY_BLOB`` is emitted when the predicted disc diameter is at
  least ``max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)``
  *and* the limb uncertainty is too high for ``LIMB_ARC``.  The
  per-body shape floor can override the global default upward but
  not downward.
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
from nav.config import DEFAULT_CONFIG, Config
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
from nav.nav_model.body_shape import BodyShape, shape_for_body
from nav.nav_model.nav_model import NavModel
from nav.nav_model.nav_model_body_base import NavModelBodyBase
from nav.nav_model.stars.predicted_snr import psf_sigma_px
from nav.support.constants import HALFPI
from nav.support.image import filter_downsample, shift_array
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from oops import Observation

    from nav.nav_orchestrator.nav_context import NavContext

from nav.support.filters import NavFilterKind, NavFilterSpec

__all__ = [
    'BODY_BLOB_MIN_DIAMETER_PX',
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


LIMB_ARC_MAX_UNCERTAINTY_PX: float = 3.0
"""Cap on the limb normal-sigma at which LIMB_ARC remains useful.

Above this value the per-vertex normal uncertainty is too large for
the DT-based limb fit; the extractor switches to ``BODY_BLOB`` so the
brightness-weighted-centroid technique still has something to work
with.  The numeric value is a config default pending calibration
against the operator-curated image library.
"""


BODY_BLOB_MIN_DIAMETER_PX: float = 8.0
"""Minimum predicted disc diameter (px) at which BODY_BLOB is emitted.

Below this diameter the predicted body silhouette is too small for a
brightness-weighted centroid to pin the body to better than ~1 px, so
the extractor emits no body feature for the image.  The per-body
shape table can override this floor for known irregular / gas-giant
bodies.
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
            self._log_geometry_summary()
            end_time = now_dt()
            self._metadata['end_time'] = end_time.isoformat()
            self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()
            self._logger.info('Model created in %.3f s', self._metadata['elapsed_time_sec'])

    def _log_geometry_summary(self) -> None:
        """Emit INFO-level geometry summary plus DEBUG-level bbox detail."""
        meta = self._metadata
        self._logger.info(
            'Subsolar (lon, lat) = (%.2f, %.2f) deg; '
            'subobserver (lon, lat) = (%.2f, %.2f) deg; phase = %.2f deg',
            meta.get('sub_solar_lon_deg', float('nan')),
            meta.get('sub_solar_lat_deg', float('nan')),
            meta.get('sub_observer_lon_deg', float('nan')),
            meta.get('sub_observer_lat_deg', float('nan')),
            meta.get('phase_angle_deg', float('nan')),
        )
        self._logger.info(
            'Subject range = %.0f km; predicted diameter = %.2f px; km/px at limb = %.4f',
            self._subject_range_km,
            self._predicted_diameter_px,
            self._km_per_pixel_at_limb,
        )
        self._logger.info(
            'Visible-lit fraction = %.3f; silhouette overflow = %.3f; guaranteed-visible = %s',
            self._visible_lit_fraction,
            self._overflow_fraction,
            meta.get('guaranteed_visible_in_fov', False),
        )
        self._logger.debug(
            'Bbox (extfov vu) = %s; bbox area = %.1f px; size_ok = %s',
            self._bbox_extfov_vu,
            meta.get('bbox_area_px', 0.0),
            meta.get('size_ok', False),
        )

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
        self._logger.debug(
            'Body %s: backplane oversample (u, v) = (%d, %d); edge_limit = %s, max = %s',
            self._body_name,
            oversample_u,
            oversample_v,
            body_config.oversample_edge_limit,
            body_config.oversample_maximum,
        )
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
        # Terminator mask: pixels whose incidence crosses 90 deg
        incidence_vals = incidence_scalar.vals
        is_lit = (incidence_vals < HALFPI) & body_mask_valid
        is_dark = (incidence_vals >= HALFPI) & body_mask_valid
        # The geometric limb is the silhouette boundary (lit + unlit).
        # The DT-fit LIMB_ARC technique can only see the *lit* limb in
        # the image — the unlit limb merges into dark space and has no
        # gradient.  Including unlit-side vertices in the polyline gives
        # the LM no useful signal there, but the high sigma assigned to
        # those vertices widens their Tukey inlier band so they happily
        # lock onto the terminator (a strong DT minimum) when the LM
        # shifts.  Filtering to lit-side vertices removes that hazard
        # at the source.  See Cassini Tethys N1574928113 for the
        # calibration case.
        geometric_limb_mask: NDArrayBoolType = body_mask_valid & limb_mask_neighbor
        limb_mask_local: NDArrayBoolType = geometric_limb_mask & is_lit
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

        # km/pixel is queried at the oversampled grid and downsampled to
        # match the masks.  When the predicted silhouette is empty the
        # backplane query is skipped (it would return masked values
        # anyway) and the downsampled-shape zeros array is used
        # directly — feeding it back through ``filter_downsample`` would
        # try to downsample an already-downsampled array, asserting on
        # the (downsampled-)shape vs oversample divisibility.
        if body_mask_valid.any():
            km_per_pixel_arr = restr_bp.resolution(body_name).mvals.filled(0.0)
            km_per_pixel_local = filter_downsample(km_per_pixel_arr, oversample_v, oversample_u)
        else:
            km_per_pixel_local = np.zeros_like(body_mask_valid, dtype=np.float64)

        limb_sampler = _build_polyline_sampler(
            local_mask=limb_mask_local,
            region_mask=body_mask_valid,
            incidence_local=incidence_vals,
            km_per_pixel_local=km_per_pixel_local,
            ext_v0=ext_v0,
            ext_u0=ext_u0,
        )
        terminator_sampler = _build_polyline_sampler(
            local_mask=terminator_local,
            region_mask=is_lit,
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
        # ``visible_lit_fraction`` measures the fraction of the *whole
        # predicted disc* (lit + dark together) whose cos(incidence) >= 0
        # AND which lies inside the sensor FOV — not the lit hemisphere
        # alone, which would always score ~1.0 for a fully-in-frame
        # body and lose discriminating power for the BODY_DISC gate.
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
        with self._logger.open(f'EMIT BODY FEATURES: {self._body_name}'):
            if self._body_mask is None:
                self._logger.debug('body_mask is None — emitting no features')
                return []
            shape = shape_for_body(self._body_name, config=self._config)
            features: list[NavFeature] = []
            limb_uncertainty_px = self._limb_uncertainty_px(shape)
            self._logger.debug(
                'ellipsoid_rms_residual = %.3f km; limb_uncertainty = %.3f px '
                '(arc max %.3f); shape_class = %s',
                shape.ellipsoid_rms_residual_km,
                limb_uncertainty_px,
                LIMB_ARC_MAX_UNCERTAINTY_PX,
                shape.shape_class_hint,
            )
            limb_arc_emitted = False
            blob_min_px = max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)
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
                        psf_sigma_px=psf_sigma_px(self.obs.star_psf()),
                        source_model=self.name,
                    )
                )
                limb_arc_emitted = True
            elif self._predicted_diameter_px >= blob_min_px:
                features.append(self._build_blob_feature(shape))
            else:
                self._logger.debug(
                    'No body feature emitted: limb_uncertainty %.3f > %.3f and '
                    'predicted_diameter %.3f < %.3f',
                    limb_uncertainty_px,
                    LIMB_ARC_MAX_UNCERTAINTY_PX,
                    self._predicted_diameter_px,
                    blob_min_px,
                )

            if self._should_emit_disc(limb_arc_emitted):
                features.append(self._build_disc_feature(shape))

            terminator_feature = self._maybe_build_terminator(shape)
            if terminator_feature is not None:
                features.append(terminator_feature)

            kinds = ', '.join(sorted({f.feature_type.name for f in features})) or 'none'
            self._logger.info('Emitted %d feature(s) [%s]', len(features), kinds)
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

    def _limb_uncertainty_px(self, shape: BodyShape) -> float:
        """Return the design's ``limb_uncertainty_px`` for this body."""
        if self._km_per_pixel_at_limb <= 0.0:
            return float('inf')
        return shape.ellipsoid_rms_residual_km / self._km_per_pixel_at_limb

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
        # ``compose_template_features`` expects the template payload to be a
        # postage-stamp sized to ``bbox_extfov_vu``.  ``self._model_img`` is
        # an extfov-shaped buffer with non-zero values only inside the body
        # bbox, so crop here to the body's bbox.  The .copy() detaches from
        # the parent so the feature can freeze its own array safely.
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        template_img = self._model_img[v_min:v_max, u_min:u_max].copy()
        template_mask = self._body_mask[v_min:v_max, u_min:u_max].copy()
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
            template_img=template_img,
            template_mask=template_mask,
        )

    def _build_blob_feature(self, shape: BodyShape) -> NavFeature:
        """Construct the BODY_BLOB feature.

        Three phase-aware decisions live here:

        * **Lit-weighted predicted centroid.** At high phase the
          measured brightness centroid sits at the centroid of the lit
          hemisphere, not at the geometric body center.  Predicting the
          lit-weighted centroid up front means the navigation offset
          the technique recovers is just the spacecraft pointing error,
          not pointing error plus the systematic phase bias.  The
          weighted centroid is computed over ``_model_img *
          _body_mask`` (the rendered model already encodes the local
          incidence-driven brightness from an *ellipsoidal* body) and
          collapses to the geometric center at phase 0 where the model
          is uniform.
        * **Inflated centroid covariance.** The lit-weighted centroid
          assumes an ellipsoidal body of *known* rotational
          orientation; the same scene on an irregular body whose pose
          we do not know carries a residual centroid bias the
          ellipsoidal model cannot remove.  The bias scales with the
          shape's RMS departure from an ellipsoid (in km) and with how
          much of the body the lit hemisphere fails to sample
          (``1 + 2 * sin^2(phase / 2)`` runs from 1 at full-phase to
          3 at full crescent — even at phase 0 the orientation is
          unknown, so a residual-scale bias is always present).  This
          irregularity sigma is added to the photon-noise-limited
          centroid sigma in quadrature so the joint-fit covariance
          reflects both terms.
        * **``phase_irregularity_factor`` on the flags.** Surfaces the
          dimensionless ``sin(phase/2) * residual / radius`` so the
          BLOB confidence formula can down-weight irregular high-phase
          blobs without the technique having to recompute it.  Raw
          ``phase_angle_deg`` is also recorded for diagnostic
          inspection but the confidence formula consumes the combined
          factor.
        """
        # Estimate per-pixel SNR from the brightest pixel in the model.
        assert self._model_img is not None
        assert self._body_mask is not None
        snr = float(self._model_img[self._body_mask].mean()) if self._body_mask.any() else 0.0
        sigma_centroid = self._predicted_diameter_px / (
            2.0 * math.sqrt(max(int(self._body_mask.sum()), 1)) * max(snr, 1e-6)
        )
        phase_angle_deg = float(self._metadata.get('phase_angle_deg', 0.0))
        if not math.isfinite(phase_angle_deg):
            phase_angle_deg = 0.0
        # Clamp to the BodyBlobFlags valid range; phase outside [0, 180]
        # is a SPICE-corner-case artefact, never a physical value.
        phase_angle_deg = max(0.0, min(180.0, phase_angle_deg))
        phase_irregularity_factor = self._phase_irregularity_factor(shape, phase_angle_deg)
        sigma_irregular_px = phase_irregularity_factor * (self._predicted_diameter_px / 2.0)
        sigma_total_px = math.sqrt(sigma_centroid * sigma_centroid + sigma_irregular_px**2)
        cov = (sigma_total_px * sigma_total_px) * np.eye(2, dtype=np.float64)
        predicted_center_vu = self._lit_weighted_centroid_vu()
        return NavFeature(
            feature_id=f'body_blob:{self._body_name}',
            feature_type=NavFeatureType.BODY_BLOB,
            source_model=self.name,
            geometry=BodyBlobGeometry(
                predicted_center_vu=predicted_center_vu,
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
                phase_angle_deg=phase_angle_deg,
                phase_irregularity_factor=phase_irregularity_factor,
            ),
        )

    def _phase_irregularity_factor(self, shape: BodyShape, phase_angle_deg: float) -> float:
        """Return the dimensionless phase-and-irregularity coupling.

        Computed as ``(ellipsoid_rms_residual_km / body_radius_km) *
        (1 + 2 * sin^2(phase / 2))``.  Two physical effects compose:

        * The ``residual / radius`` ratio is the *fractional* shape
          uncertainty of the body relative to its ellipsoidal model.
          For a regular Mimas (residual ~ 1 km, radius ~ 200 km)
          this is ~ 0.005; for irregular Hyperion / Phoebe (residual
          ~ 10 km, radius ~ 100-135 km) it is ~ 0.07-0.10.
        * The ``1 + 2 * sin^2(phase / 2)`` factor captures the
          orientation-uncertainty bound.  At phase 0 the entire
          hemisphere is lit and the factor is ``1`` — we still do
          not know the body's rotational orientation so a full
          ``residual_km``-scale centroid bias is in play even at
          full phase.  At phase 90 the factor is ``2`` because only
          half the body is lit and the dark side could be hiding an
          equal amount of shape irregularity.  At phase 180 the
          factor is ``3`` since only the crescent is lit, leaving
          most of the body's irregularity hidden.

        ``body_radius_km`` is derived from the predicted disc geometry
        rather than from a separate static-data lookup so a body
        absent from the YAML or hard-coded shape table still gets a
        meaningful factor (the residual itself falls back to
        ``DEFAULT_BODY_SHAPE`` upstream).
        """
        if self._predicted_diameter_px <= 0.0 or self._km_per_pixel_at_limb <= 0.0:
            return 0.0
        body_radius_km = self._km_per_pixel_at_limb * (self._predicted_diameter_px / 2.0)
        if body_radius_km <= 0.0:
            return 0.0
        sin_half_phase = math.sin(math.radians(phase_angle_deg) / 2.0)
        phase_factor = 1.0 + 2.0 * sin_half_phase * sin_half_phase
        residual_fraction = shape.ellipsoid_rms_residual_km / body_radius_km
        return float(max(0.0, residual_fraction * phase_factor))

    def _lit_weighted_centroid_vu(self) -> tuple[float, float]:
        """Return the brightness-weighted centroid of the rendered body.

        Falls back to the geometric center when the model is empty or
        the body mask is all-False (degenerate render).  The centroid
        is in the same extfov coordinate frame ``_predicted_center_vu``
        uses, so the BLOB feature's geometry stays self-consistent.
        """
        assert self._model_img is not None
        assert self._body_mask is not None
        weights = np.where(self._body_mask, self._model_img, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return self._predicted_center_vu
        v_indices, u_indices = np.indices(weights.shape, dtype=np.float64)
        lit_v = float((weights * v_indices).sum() / total)
        lit_u = float((weights * u_indices).sum() / total)
        return (lit_v, lit_u)

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
            psf_sigma_px=psf_sigma_px(self.obs.star_psf()),
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
        ),
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
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
    region_mask: NDArrayBoolType,
    incidence_local: NDArrayFloatType,
    km_per_pixel_local: NDArrayFloatType,
    ext_v0: int,
    ext_u0: int,
) -> _PolylineSampler:
    """Sample a polyline along the True pixels of ``local_mask``.

    The local mask is a 1-pixel-wide ridge along a feature boundary; we
    return a parallel array of vertex coordinates in extfov space, the
    outward-pointing normal at each vertex, the per-vertex incidence
    angle, and the per-vertex km/px scale.

    The outward normal is the discrete gradient of ``region_mask`` (the
    body silhouette for the limb, or the lit mask for the terminator),
    not of the ridge itself: a one-pixel ridge has no consistent
    interior/exterior orientation, so its gradient sign depends on the
    diagonal orientation of the ridge rather than on which side is the
    body interior.  With the body-side True / space-side False
    convention, ``n_v = region[v-1, u] - region[v+1, u]`` and
    ``n_u = region[v, u-1] - region[v, u+1]`` point from inside to
    outside.  Out-of-image neighbours are treated as space (False).

    Parameters:
        local_mask: 1-pixel-wide ridge whose True pixels become vertices.
        region_mask: Body-side / lit-side mask whose discrete gradient
            defines the outward normal.  Same shape as ``local_mask``.
        incidence_local: Per-pixel incidence angle (radians).
        km_per_pixel_local: Per-pixel km/px scale.
        ext_v0: Extfov v-offset of the local grid origin.
        ext_u0: Extfov u-offset of the local grid origin.
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
    rows, cols = region_mask.shape
    region = region_mask.astype(np.float64)
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        # Outward normal: discrete gradient of the body-side / lit-side
        # mask, pointing from inside (True) to outside (False).  Out-of-
        # image neighbours count as space (0.0).
        up = region[v - 1, u] if v > 0 else 0.0
        down = region[v + 1, u] if v < rows - 1 else 0.0
        left = region[v, u - 1] if u > 0 else 0.0
        right = region[v, u + 1] if u < cols - 1 else 0.0
        v_dir = up - down
        u_dir = left - right
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
        shape.ellipsoid_rms_residual_km**2
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


def _limb_reliability(*, visible_arc_fraction: float, visible_arc_px: float) -> float:
    """Sigmoid-of-sum reliability for LIMB_ARC features.

    The score answers a feature-existence question: is this limb arc a
    target a downstream technique should bother running on?  Per-vertex
    geometric softness (high incidence at the terminator-adjacent end
    of the limb) lives in :func:`_sigma_normal_per_vertex`, where the
    LM fit weights individual vertices by their normal sigma; folding
    it into the reliability scalar as well would double-count the same
    physics and, because ``incidence_factor`` saturates near the cap
    on every fully-lit body, would penalize the cleanest possible
    geometries the hardest.
    """
    z = -1.0 + 1.5 * visible_arc_fraction + 1.0 * _sigmoid(visible_arc_px / 50.0)
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
