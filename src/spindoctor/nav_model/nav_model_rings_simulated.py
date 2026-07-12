"""Simulated-rings NavModel.

Renders a ring system from operator-supplied parameters (centre, edge radii,
shading) rather than from SPICE.  Used by the simulated-image GUI to
compose synthetic ring scenes; the rendered annulus becomes a
``RING_ANNULUS`` ``NavFeature`` for the standard pipeline.

The data model classes ``RingFeature`` and ``RingEdgeData`` are shared with
the catalog-driven ring rendering path; only the image generation differs
(pixel-space here, vs. backplane-based for the real model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import oops

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import RingAnnulusFlags, RingEdgeFlags
from spindoctor.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_rings_base import NavModelRingsBase
from spindoctor.nav_model.rings import RingFeature
from spindoctor.sim.sim_ring import compute_border_atop_simulated, render_ring
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelRingsSimulated']

# Per-vertex ring-edge uncertainty for a simulated ring.  The rendered edge is
# sharp and noise-free, so the predicted edge sits within ~1 px of the image
# edge; the along-edge sigma reflects the one-pixel polyline sampling resolution.
_RING_EDGE_SIGMA_RADIAL_PX: float = 1.0
_RING_EDGE_SIGMA_ALONG_PX: float = 0.5
# A polyline whose max deviation from its best-fit line is below this is treated
# as straight (rank-1 constraint); a curved ring arc exceeds it and constrains
# the offset in both axes.
_RING_EDGE_FLAT_CURVATURE_PX: float = 1.0


def _ring_edge_is_straight(vertices_vu: NDArrayFloatType) -> bool:
    """Return True when the polyline's deviation from a line is below threshold.

    Computed by SVD of the centred vertices: the smaller singular direction's
    spread is the max perpendicular deviation from the best-fit line.
    """
    if vertices_vu.shape[0] < 3:
        return True
    centred = vertices_vu - vertices_vu.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    deviations = centred @ vt[1]
    return bool(float(np.max(np.abs(deviations))) <= _RING_EDGE_FLAT_CURVATURE_PX)


def _ring_edge_polyline(
    edge_mask: NDArrayBoolType,
    center_vu: tuple[float, float],
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Extract ring-edge vertices and outward radial normals from an edge mask.

    Each ``True`` pixel becomes a vertex; the normal points radially outward from
    the ring centre (the direction across the edge the technique fits along).

    Parameters:
        edge_mask: Extfov-shape boolean 1-pixel edge mask.
        center_vu: Ring centre ``(v, u)`` in extfov coordinates.

    Returns:
        ``(vertices_vu, normals_vu)`` each shaped ``(N, 2)``; empty when no edge.
    """
    if not edge_mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    vs, us = np.where(edge_mask)
    vertices_vu = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    radial = vertices_vu - np.asarray(center_vu, dtype=np.float64)[None, :]
    norms = np.hypot(radial[:, 0], radial[:, 1])
    norms[norms == 0.0] = 1.0
    normals_vu = radial / norms[:, None]
    return vertices_vu, normals_vu


class NavModelRingsSimulated(NavModelRingsBase):
    """Ring NavModel rendered from operator-supplied simulation parameters.

    Parameters:
        name: Name of this model instance.
        obs: Observation containing image geometry.
        ring_name: Logical ring system name used in metadata and labels.
        sim_params: Dictionary of simulation parameters.  Expected keys:

            - ``name``, ``feature_type``
            - ``center_v``, ``center_u``
            - ``range``
            - ``shading_distance``
            - ``inner_data``, ``outer_data`` — lists of dicts with ``mode``,
              ``a``, ``rms``, ``ae``, ``long_peri``, ``rate_peri`` keys.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: oops.Observation,
        ring_name: str,
        sim_params: dict[str, Any],
        *,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._ring_name = ring_name.upper()
        self._sim_params: dict[str, Any] = dict(sim_params)
        self._model_img: NDArrayFloatType | None = None
        self._ring_mask: NDArrayBoolType | None = None
        self._ring_feature: RingFeature | None = None
        self._predicted_center_vu: tuple[float, float] = (0.0, 0.0)
        self._subject_range_km: float = float('inf')
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def instances_for_obs(
        cls, obs: oops.Observation, *, config: Config | None = None
    ) -> list[NavModel]:
        """Build one simulated ring model per ring of a simulated obs.

        Reads ``obs.sim_params['rings']``; returns an empty list for a real obs
        so the SPICE-backed ``NavModelRings`` handles those instead.

        Parameters:
            obs: Observation snapshot.
            config: Configuration passed to the constructed instances.  None
                uses ``DEFAULT_CONFIG``.

        Returns:
            One ``NavModelRingsSimulated`` per ring in the sim scene.
        """
        if not getattr(obs, 'is_simulated', False):
            return []
        sim_params = getattr(obs, 'sim_params', None)
        if not isinstance(sim_params, dict):
            return []
        out: list[NavModel] = []
        for ring_params in sim_params.get('rings', []) or []:
            if not isinstance(ring_params, dict):
                continue
            ring_name = str(ring_params.get('name', 'SIM-RING'))
            out.append(cls(f'rings_sim:{ring_name}', obs, ring_name, ring_params, config=config))
        return out

    def create_model(self) -> None:
        """Render the simulated rings and populate masks, annotations, metadata."""
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None
        self._metadata.clear()
        self._metadata.update(metadata)
        log_level = self._config.general.get('log_level_model_rings')
        with self._logger.open(
            f'CREATE SIMULATED RINGS MODEL FOR: {self._ring_name}',
            level=log_level,
        ):
            self._render()
        end_time = now_dt()
        self._metadata['end_time'] = end_time.isoformat()
        self._metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _render(self) -> None:
        """Generate the simulated ring image and the matching mask."""
        obs = self.obs
        p = self._sim_params
        time = obs.sim_time
        epoch = obs.sim_epoch
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)
        ext_margin_v = int(obs.extfov_margin_v)
        ext_margin_u = int(obs.extfov_margin_u)
        feature_config = _sim_params_to_feature_config(p)
        ring_feature = RingFeature.from_config(self._ring_name, feature_config)
        self._ring_feature = ring_feature
        self._logger.debug(
            'Simulated rings: parsed feature %r type=%s',
            self._ring_name,
            ring_feature.feature_type.value,
        )
        sim_img: NDArrayFloatType = obs.make_extfov_zeros()
        center_v_data = float(p.get('center_v', data_size_v / 2.0))
        center_u_data = float(p.get('center_u', data_size_u / 2.0))
        center_v_extfov = center_v_data + ext_margin_v
        center_u_extfov = center_u_data + ext_margin_u
        ring_params_extfov = dict(p)
        ring_params_extfov['center_v'] = center_v_extfov
        ring_params_extfov['center_u'] = center_u_extfov
        self._logger.debug(
            'Simulated rings: render_ring at extfov center (%.2f, %.2f) shade_solid=True',
            center_v_extfov,
            center_u_extfov,
        )
        # Solid shading fakes the normal ring modelling process where the
        # space between edges is opaque without further information.
        render_ring(
            sim_img,
            ring_params_extfov,
            0.0,
            0.0,
            time=time,
            epoch=epoch,
            shade_solid=True,
        )
        ring_mask: NDArrayBoolType = sim_img != 0.0
        self._logger.debug(
            'Simulated rings: render complete, %d / %d pixels in mask',
            int(np.count_nonzero(ring_mask)),
            ring_mask.size,
        )
        self._model_img = sim_img
        self._ring_mask = ring_mask
        self._predicted_center_vu = (center_v_extfov, center_u_extfov)
        self._subject_range_km = float(p.get('range', float('inf')))
        self._bbox_extfov_vu = (
            int(ext_margin_v),
            int(ext_margin_u),
            int(ext_margin_v + data_size_v),
            int(ext_margin_u + data_size_u),
        )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the ring features.

        Emits a ``RING_ANNULUS`` carrying the rendered template (for the
        correlation path) whenever the render put any ring pixels on the
        ext-FOV.  The template payload convention (``compose_template_features``)
        is a postage stamp local to ``bbox_extfov_vu``, so the ext-FOV-sized
        render is cropped to the mask's tight bbox before emission --
        passing the full ext-FOV image with an interior bbox displaces the
        painted ring by the bbox origin and the annulus NCC recovers a
        garbage offset.  Also emits one ``RING_EDGE`` per rendered edge
        (inner / outer) -- a per-vertex polyline with outward radial normals that
        ``RingEdgeNav`` fits against the image-edge distance transform, so a
        curved ring arc recovers the planted offset in both axes.
        """
        if self._model_img is None or self._ring_mask is None or self._ring_feature is None:
            return []
        features: list[NavFeature] = []
        if self._ring_mask.any():
            mask_vs, mask_us = np.where(self._ring_mask)
            bbox = (
                int(mask_vs.min()),
                int(mask_us.min()),
                int(mask_vs.max()) + 1,
                int(mask_us.max()) + 1,
            )
            template_img = self._model_img[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
            template_mask = self._ring_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]].copy()
            features.append(
                NavFeature(
                    feature_id=f'ring_annulus:{self._ring_name}',
                    feature_type=NavFeatureType.RING_ANNULUS,
                    source_model=self.name,
                    geometry=RingAnnulusGeometry(
                        bbox_extfov_vu=bbox,
                        predicted_center_vu=self._predicted_center_vu,
                    ),
                    subject_range_km=self._subject_range_km,
                    position_cov_px=None,
                    intensity_sigma_rel=0.0,
                    preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
                    reliability=1.0,
                    reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
                    usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
                    flags=RingAnnulusFlags(
                        planet_name=self._ring_name,
                        constituent_edge_count=(
                            int(self._ring_feature.outer_edge is not None)
                            + int(self._ring_feature.inner_edge is not None)
                        ),
                    ),
                    template_img=template_img,
                    template_mask=template_mask,
                )
            )
        for edge_type, edge_mask in self._iter_edge_masks():
            edge_feature = self._build_ring_edge_feature(edge_type, edge_mask)
            if edge_feature is not None:
                features.append(edge_feature)
        return features

    def _build_ring_edge_feature(
        self,
        edge_type: str,
        edge_mask: NDArrayBoolType,
    ) -> NavFeature | None:
        """Build a RING_EDGE feature from an edge mask, or ``None`` if empty."""
        vertices_vu, normals_vu = _ring_edge_polyline(edge_mask, self._predicted_center_vu)
        n = vertices_vu.shape[0]
        if n == 0:
            return None
        sigma_radial = np.full(n, _RING_EDGE_SIGMA_RADIAL_PX, dtype=np.float64)
        sigma_along = np.full(n, _RING_EDGE_SIGMA_ALONG_PX, dtype=np.float64)
        return NavFeature(
            feature_id=f'ring_edge:{self._ring_name}:{edge_type}',
            feature_type=NavFeatureType.RING_EDGE,
            source_model=self.name,
            geometry=RingEdgePolyline(
                vertices_vu=vertices_vu,
                normals_vu=normals_vu,
                sigma_radial_per_vertex_px=sigma_radial,
                sigma_along_edge_per_vertex_px=sigma_along,
                is_straight_line=_ring_edge_is_straight(vertices_vu),
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=1.0,
            reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
            usable_types=frozenset({NavFeatureType.RING_EDGE}),
            flags=RingEdgeFlags(
                is_straight_line=_ring_edge_is_straight(vertices_vu),
                edge_name=f'{self._ring_name}:{edge_type}',
                planet_name=self._ring_name,
            ),
        )

    def _iter_edge_masks(self) -> list[tuple[str, NDArrayBoolType]]:
        """Return ``(edge_type, edge_mask_extfov)`` for each rendered ring edge."""
        assert self._ring_feature is not None
        obs = self.obs
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)
        center_v = float(self._sim_params.get('center_v', data_size_v / 2.0))
        center_u = float(self._sim_params.get('center_u', data_size_u / 2.0))
        time = obs.sim_time
        epoch = obs.sim_epoch
        out: list[tuple[str, NDArrayBoolType]] = []
        for edge_data, edge_type in (
            (self._ring_feature.inner_edge, 'inner'),
            (self._ring_feature.outer_edge, 'outer'),
        ):
            if edge_data is None:
                continue
            base = edge_data.base_orbit
            edge_mask = compute_border_atop_simulated(
                data_size_v,
                data_size_u,
                center_v,
                center_u,
                a=base.a,
                ae=base.ae,
                long_peri=base.long_peri,
                rate_peri=base.rate_peri,
                epoch=epoch,
                time=time,
            )
            edge_mask_extfov: NDArrayBoolType = obs.make_extfov_false()
            edge_mask_extfov[
                obs.extfov_margin_v : obs.extfov_margin_v + data_size_v,
                obs.extfov_margin_u : obs.extfov_margin_u + data_size_u,
            ] = edge_mask
            out.append((edge_type, edge_mask_extfov))
        return out

    def to_annotations(self, context: NavContext) -> Annotations:
        """Emit ring-edge polyline + label annotations."""
        if self._model_img is None or self._ring_mask is None or self._ring_feature is None:
            return Annotations()
        return self._build_simulated_edge_annotations()

    def _build_simulated_edge_annotations(self) -> Annotations:
        """Render annotations for simulated inner and outer ring edges."""
        assert self._ring_feature is not None
        assert self._ring_mask is not None
        obs = self.obs
        ring_feature = self._ring_feature
        data_size_v = int(obs.data_shape_v)
        data_size_u = int(obs.data_shape_u)
        center_v = float(self._sim_params.get('center_v', data_size_v / 2.0))
        center_u = float(self._sim_params.get('center_u', data_size_u / 2.0))
        time = obs.sim_time
        epoch = obs.sim_epoch
        labels = ring_feature.edge_labels
        feature_name = ring_feature.name or 'UNNAMED'
        edge_info_list: list[tuple[NDArrayBoolType, str, str]] = []
        for edge_data, edge_type in (
            (ring_feature.inner_edge, 'inner'),
            (ring_feature.outer_edge, 'outer'),
        ):
            if edge_data is None:
                continue
            label = labels[edge_type]
            label_text = f'{feature_name} {label}'
            base = edge_data.base_orbit
            edge_mask = compute_border_atop_simulated(
                data_size_v,
                data_size_u,
                center_v,
                center_u,
                a=base.a,
                ae=base.ae,
                long_peri=base.long_peri,
                rate_peri=base.rate_peri,
                epoch=epoch,
                time=time,
            )
            edge_mask_extfov: NDArrayBoolType = obs.make_extfov_false()
            edge_mask_extfov[
                obs.extfov_margin_v : obs.extfov_margin_v + data_size_v,
                obs.extfov_margin_u : obs.extfov_margin_u + data_size_u,
            ] = edge_mask
            edge_info_list.append((edge_mask_extfov, label_text, label))
        return self._create_edge_annotations(obs, edge_info_list, self._ring_mask)


def _sim_params_to_feature_config(p: dict[str, Any]) -> dict[str, Any]:
    """Convert GUI sim_params dict to a config dict for ``RingFeature.from_config``.

    The GUI stores ring parameters in a flat dict with ``inner_data`` and
    ``outer_data`` lists.  This adapter rewrites that into the canonical
    feature-config format the validator expects.

    Parameters:
        p: GUI sim_params dictionary.

    Returns:
        Feature config dict suitable for ``RingFeature.from_config``.
    """
    raw_inner = p.get('inner_data') or None
    raw_outer = p.get('outer_data') or None
    feature_type_raw = p.get('feature_type', 'RINGLET')
    config: dict[str, Any] = {
        'feature_type': feature_type_raw,
        'name': p.get('name'),
    }
    if raw_inner is not None:
        config['inner_data'] = raw_inner
    if raw_outer is not None:
        config['outer_data'] = raw_outer
    return config
