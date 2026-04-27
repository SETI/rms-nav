"""Simulated-body NavModel.

Renders a body from operator-supplied geometric parameters (centre, axes,
rotation, lighting) rather than from SPICE.  Used by the simulated-image
GUI to compose synthetic test scenes; the rendered body becomes a
``BODY_DISC`` ``NavFeature`` that the standard pipeline can navigate
against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from oops import Observation

from nav.annotation import Annotations
from nav.config import Config
from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import BodyDiscFlags
from nav.feature.geometry import BodyDiscGeometry
from nav.nav_model.nav_model_body_base import NavModelBodyBase
from nav.sim.sim_body import create_simulated_body
from nav.support.filters import NavFilterKind, NavFilterSpec
from nav.support.time import now_dt
from nav.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from nav.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelBodySimulated']


class NavModelBodySimulated(NavModelBodyBase):
    """Body NavModel rendered from operator-supplied simulation parameters.

    Parameters:
        name: Name of this model instance.
        obs: Observation containing image geometry (used for output shapes
            and extfov margins).
        body_name: Logical body name used in metadata and labels.
        sim_params: Dictionary of simulation parameters.  Expected keys:

            - ``name``
            - ``center_v``, ``center_u`` (pixel coordinates of the centre)
            - ``range`` (km; subject distance, defaults to inf)
            - ``axis1``, ``axis2``, ``axis3`` (km; ellipsoid semi-axes)
            - ``rotation_z`` (deg; rotation about the line of sight)
            - ``rotation_tilt`` (deg; tilt of the body)
            - ``illumination_angle`` (deg)
            - ``phase_angle`` (deg)

            Crater and anti-aliasing keys are accepted but ignored;
            anti-aliasing is always maximal here.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        body_name: str,
        sim_params: dict[str, Any],
        *,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._body_name = body_name.upper()
        self._sim_params: dict[str, Any] = dict(sim_params)
        self._model_img: NDArrayFloatType | None = None
        self._body_mask: NDArrayBoolType | None = None
        self._limb_mask: NDArrayBoolType | None = None
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._subject_range_km: float = float('inf')
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)

    def create_model(self) -> None:
        """Render the simulated body and populate masks, annotations, metadata."""
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None
        metadata['body_name'] = self._body_name
        self._metadata.clear()
        self._metadata.update(metadata)
        with self._logger.open(f'CREATE SIMULATED BODY MODEL FOR: {self._body_name}'):
            self._render()
        end_time = now_dt()
        self._metadata['end_time'] = end_time.isoformat()
        self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _render(self) -> None:
        """Generate the simulated image and the matching masks."""
        p = self._sim_params
        data_size_v = int(self.obs.data_shape_v)
        data_size_u = int(self.obs.data_shape_u)
        ext_margin_v = int(self.obs.extfov_margin_v)
        ext_margin_u = int(self.obs.extfov_margin_u)
        rotation_z_rad = float(np.radians(p.get('rotation_z', 0.0)))
        rotation_tilt_rad = float(np.radians(p.get('rotation_tilt', 0.0)))
        illumination_angle_rad = float(np.radians(p.get('illumination_angle', 0.0)))
        phase_angle_rad = float(np.radians(p.get('phase_angle', 0.0)))
        center_v = float(p.get('center_v', data_size_v / 2.0))
        center_u = float(p.get('center_u', data_size_u / 2.0))
        axis1 = float(p.get('axis1', 0.0))
        axis2 = float(p.get('axis2', 0.0))
        axis3 = float(p.get('axis3', min(axis1, axis2)))
        sim_img = create_simulated_body(
            size=(data_size_v, data_size_u),
            center=(center_v, center_u),
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            rotation_z=rotation_z_rad,
            rotation_tilt=rotation_tilt_rad,
            illumination_angle=illumination_angle_rad,
            phase_angle=phase_angle_rad,
            anti_aliasing=1,
        )
        body_mask = sim_img > 0.0
        limb_mask = self._compute_limb_mask_from_body_mask(body_mask)
        model_img_full = self.obs.make_extfov_zeros()
        limb_mask_full = self.obs.make_extfov_false()
        body_mask_full = self.obs.make_extfov_false()
        slice_v = slice(ext_margin_v, ext_margin_v + data_size_v)
        slice_u = slice(ext_margin_u, ext_margin_u + data_size_u)
        model_img_full[slice_v, slice_u] = sim_img
        limb_mask_full[slice_v, slice_u] = limb_mask
        body_mask_full[slice_v, slice_u] = body_mask
        self._model_img = model_img_full
        self._body_mask = body_mask_full
        self._limb_mask = limb_mask_full
        self._predicted_center_vu = (
            center_v + ext_margin_v,
            center_u + ext_margin_u,
        )
        self._subject_range_km = float(p.get('range', float('inf')))
        # Extfov-coord bounding box of the body silhouette.
        self._bbox_extfov_vu = (
            int(ext_margin_v),
            int(ext_margin_u),
            int(ext_margin_v + data_size_v),
            int(ext_margin_u + data_size_u),
        )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit a single BODY_DISC feature carrying the rendered template."""
        if self._model_img is None or self._body_mask is None:
            return []
        feature = NavFeature(
            feature_id=f'body_disc:{self._body_name}',
            feature_type=NavFeatureType.BODY_DISC,
            source_model=self.name,
            geometry=BodyDiscGeometry(
                bbox_extfov_vu=self._bbox_extfov_vu,
                predicted_center_vu=self._predicted_center_vu,
                overflow_fraction=0.0,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=1.0,
            reliability_reasons=NavReliabilityBreakdown(
                visible_lit_fraction=1.0, overflow_fraction=0.0
            ),
            usable_types=frozenset({NavFeatureType.BODY_DISC}),
            flags=BodyDiscFlags(body_name=self._body_name, overflow_fov_fraction=0.0),
            template_img=self._model_img,
            template_mask=self._body_mask,
        )
        return [feature]

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit body silhouette + label annotations for the summary PNG."""
        if self._model_img is None or self._body_mask is None or self._limb_mask is None:
            return Annotations()
        center_v = float(self._sim_params.get('center_v', self.obs.data_shape_v / 2.0))
        center_u = float(self._sim_params.get('center_u', self.obs.data_shape_u / 2.0))
        return self._create_annotations(
            round(center_u),
            round(center_v),
            self._model_img,
            self._limb_mask,
            self._body_mask,
        )
