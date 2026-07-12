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

from spindoctor.annotation import Annotations
from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyDiscFlags, LimbArcFlags
from spindoctor.feature.geometry import BodyDiscGeometry, LimbPolyline
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import LIMB_ARC_MIN_VERTICES
from spindoctor.nav_model.nav_model_body_base import BODY_BLOB_MIN_DIAMETER_PX, NavModelBodyBase
from spindoctor.sim.sim_body import create_simulated_body
from spindoctor.sim.sim_body_polyhedral import mesh_spec_from_params, render_mesh_body_image
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.time import now_dt
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = ['NavModelBodySimulated']

# Minimum limb-polyline vertices to emit a LIMB_ARC.  Matches BodyLimbNav's
# ``min_arc_vertices`` feasibility floor: a shorter arc cannot constrain the
# fit.  Imported from the catalog body model so the two emission gates cannot
# desync.
_MIN_LIMB_ARC_VERTICES: int = LIMB_ARC_MIN_VERTICES
# Minimum silhouette diameter to emit a LIMB_ARC.  The catalog body model gates
# the limb on its ellipsoid-fit uncertainty (``<= 3 px``) plus the shared
# vertex-count floor; the sim has no km scale, so it gates on resolution
# directly.  Below this the limb fit is imprecise and -- because it is an
# LM-refined distance-transform fit -- it injects cross-process jitter into the
# fused offset of any body scene, so only a well-resolved body emits a limb.
_MIN_LIMB_DIAMETER_PX: float = 100.0
# Per-vertex limb uncertainty for a simulated body.  The rendered silhouette edge
# is sharp and noise-free, so the predicted limb sits within ~1 px of the image
# edge; the tangent sigma reflects the one-pixel polyline sampling resolution.
_LIMB_SIGMA_NORMAL_PX: float = 1.0
_LIMB_SIGMA_TANGENT_PX: float = 0.5
# Above this phase the lit limb is a thin crescent and most of the silhouette
# boundary is terminator (a soft brightness gradient, not a sharp image edge), so
# a limb fit is unreliable -- the body is then navigated by the blob centroid.
# Gating emission here keeps LIMB_ARC off the high-phase scenes and matches the
# catalog body model's limb/blob handoff.
_LIMB_MAX_PHASE_DEG: float = 60.0


def _limb_polyline_from_mask(
    limb_mask: NDArrayBoolType,
    body_mask: NDArrayBoolType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Extract limb vertices and outward normals from a 1-pixel limb mask.

    Each ``True`` pixel of ``limb_mask`` (a body pixel adjacent to sky) becomes
    one vertex.  The outward normal points away from the body interior: it is the
    normalized sum of the unit directions toward each non-body 4-neighbour, so a
    vertex on the sunward limb gets a normal pointing into the sky.

    Parameters:
        limb_mask: Extfov-shape boolean limb mask (the silhouette boundary).
        body_mask: Extfov-shape boolean body silhouette mask.

    Returns:
        ``(vertices_vu, normals_vu)`` each shaped ``(N, 2)``; empty when the mask
        has no pixels.
    """
    if not limb_mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    vs, us = np.where(limb_mask)
    vertices_vu = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    rows, cols = body_mask.shape
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        dv = 0.0
        du = 0.0
        if v > 0 and not body_mask[v - 1, u]:
            dv -= 1.0
        if v < rows - 1 and not body_mask[v + 1, u]:
            dv += 1.0
        if u > 0 and not body_mask[v, u - 1]:
            du -= 1.0
        if u < cols - 1 and not body_mask[v, u + 1]:
            du += 1.0
        norm = float(np.hypot(dv, du)) or 1.0
        normals_vu[i, 0] = dv / norm
        normals_vu[i, 1] = du / norm
    return vertices_vu, normals_vu


def _nav_params(body_params: dict[str, Any]) -> dict[str, Any]:
    """Overlay a body's ``nav_override`` onto its params for the predicted body.

    Returns a shallow copy of ``body_params`` with the keys of its
    ``nav_override`` mapping applied on top and the ``nav_override`` key itself
    dropped.  Absent an override the params pass through unchanged, so the
    predicted geometry equals the rendered geometry (the default agreeing case).

    Parameters:
        body_params: A scene body parameter mapping (the render geometry).

    Returns:
        The parameter mapping the navigator predicts from (the navigation
        geometry).
    """
    override = body_params.get('nav_override')
    if not isinstance(override, dict):
        return dict(body_params)
    merged = {k: v for k, v in body_params.items() if k != 'nav_override'}
    merged.update(override)
    return merged


def _silhouette_diameter_px(body_mask: NDArrayBoolType) -> float:
    """Return the longer pixel extent of a rendered body silhouette.

    Measures the bounding-box span of the lit silhouette in each axis and
    returns the larger of the two.  Returns ``0.0`` for an empty mask.
    """
    if not body_mask.any():
        return 0.0
    rows = np.any(body_mask, axis=1)
    cols = np.any(body_mask, axis=0)
    v_indices = np.where(rows)[0]
    u_indices = np.where(cols)[0]
    v_extent = float(v_indices[-1] - v_indices[0] + 1)
    u_extent = float(u_indices[-1] - u_indices[0] + 1)
    return max(v_extent, u_extent)


def _tight_bbox_extfov(
    body_mask: NDArrayBoolType,
    *,
    ext_margin_vu: tuple[int, int],
    extfov_shape: tuple[int, int],
    diameter_px: float,
) -> tuple[int, int, int, int]:
    """Return a tight extfov-coord bbox around a data-coord body silhouette.

    The bbox is the silhouette's bounding box inflated by a slop margin
    (so the body stays inside it under a modest pointing error) and
    clamped to the extfov shape.  Returns the full extfov frame when the
    mask is empty (a degenerate render the caller will not emit features
    for).

    Parameters:
        body_mask: Data-shape boolean silhouette mask.
        ext_margin_vu: Extfov margins ``(v, u)`` to shift data coords into
            extfov coords.
        extfov_shape: Extfov array shape ``(h, w)`` to clamp against.
        diameter_px: Predicted silhouette diameter, used to size the slop.

    Returns:
        ``(v_min, u_min, v_max, u_max)`` half-open bbox in extfov coords.
    """
    ext_margin_v, ext_margin_u = ext_margin_vu
    h, w = extfov_shape
    if not body_mask.any():
        return (0, 0, int(h), int(w))
    rows = np.where(np.any(body_mask, axis=1))[0]
    cols = np.where(np.any(body_mask, axis=0))[0]
    slop = max(round(0.1 * diameter_px), 4)
    v_min = int(rows[0]) + ext_margin_v - slop
    v_max = int(rows[-1]) + ext_margin_v + slop + 1
    u_min = int(cols[0]) + ext_margin_u - slop
    u_max = int(cols[-1]) + ext_margin_u + slop + 1
    return (
        max(0, v_min),
        max(0, u_min),
        min(int(h), v_max),
        min(int(w), u_max),
    )


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
        self._predicted_diameter_px: float = 0.0
        self._km_per_pixel_at_limb: float = 0.0
        self._subject_range_km: float = float('inf')
        self._bbox_extfov_vu: tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def instances_for_obs(cls, obs: Observation) -> list[NavModel]:
        """Build one simulated body model per body of a simulated obs.

        Reads the per-body parameters the sim obs stashes on its snapshot
        (``obs.sim_params['bodies']``).  Returns an empty list for a real obs,
        so the SPICE-backed ``NavModelBody`` handles those instead.

        A body may carry an optional ``nav_override`` mapping.  The renderer
        ignores it (it draws the true shape), but the predicted body is built
        from the body params with ``nav_override`` overlaid -- the channel that
        makes the navigation geometry diverge from the render geometry.  A scene
        renders an irregular mesh at the true pose yet predicts an ellipsoid
        (shape mismatch, B7 scenario 2), or the same mesh at a different pose
        (chaotic-rotator pose disagreement, B7 scenario 3), without touching the
        rendered image.  The override never changes the centre, so the predicted
        body stays at the unshifted position the planted offset is measured from.

        Parameters:
            obs: Observation snapshot.

        Returns:
            One ``NavModelBodySimulated`` per body in the sim scene.
        """
        if not getattr(obs, 'is_simulated', False):
            return []
        sim_params = getattr(obs, 'sim_params', None)
        if not isinstance(sim_params, dict):
            return []
        out: list[NavModel] = []
        for body_params in sim_params.get('bodies', []) or []:
            if not isinstance(body_params, dict):
                continue
            body_name = str(body_params.get('name', 'SIM-BODY'))
            out.append(cls(f'body_sim:{body_name}', obs, body_name, _nav_params(body_params)))
        return out

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
        # The predicted shape is read from this model's own params, which need
        # not match what was rendered into the image: an irregular body can be
        # predicted as a mesh (matching pose), as an ellipsoid (shape mismatch),
        # or at a deliberately different pose (chaotic-rotator fixture).
        if str(p.get('shape_model', 'ellipsoid')) == 'polyhedral_mesh':
            sim_img = render_mesh_body_image(
                size=(data_size_v, data_size_u),
                center=(center_v, center_u),
                semi_axes_px=(axis1 / 2.0, axis2 / 2.0, axis3 / 2.0),
                spec=mesh_spec_from_params(p),
                illumination_angle=illumination_angle_rad,
                phase_angle=phase_angle_rad,
                anti_aliasing=1.0,
            )
        else:
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
        # Predicted disc diameter: the longer pixel extent of the rendered
        # silhouette.  Drives the BODY_BLOB emission gate and covariance.
        self._predicted_diameter_px = _silhouette_diameter_px(body_mask)
        # Tight extfov-coord bounding box around the body silhouette (plus
        # slop), matching the convention of the SPICE-backed ``NavModelBody``.
        # A whole-frame bbox would make ``BodyBlobNav`` integrate its observed
        # centroid over the entire frame, so scattered above-noise sky pixels
        # would swamp a small or high-phase lit region; a tight bbox keeps the
        # moment local to the body.  The slop lets the body stay inside the
        # bbox under a modest pointing error.
        self._bbox_extfov_vu = _tight_bbox_extfov(
            body_mask,
            ext_margin_vu=(ext_margin_v, ext_margin_u),
            extfov_shape=(model_img_full.shape[0], model_img_full.shape[1]),
            diameter_px=self._predicted_diameter_px,
        )
        # Physical scale at the limb.  The sim FOV is a dummy flat FOV with no
        # real angular scale, so km/pixel is only known when the scene states
        # it explicitly; absent that it stays 0, which makes the shared
        # phase-irregularity factor collapse to 0 (the regular-body case).
        self._km_per_pixel_at_limb = float(p.get('km_per_pixel', 0.0))
        self._metadata['phase_angle_deg'] = float(p.get('phase_angle', 0.0))

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the body's NavFeatures.

        Always emits a ``BODY_DISC`` carrying the rendered template (for the
        correlation technique).  Also emits a ``BODY_BLOB`` whenever the
        predicted silhouette is large enough -- the lit-weighted centroid is
        orientation-independent, so it is the technique that navigates small,
        high-phase, or irregular bodies that the disc correlation cannot.
        """
        if self._model_img is None or self._body_mask is None:
            return []
        v_min, u_min, v_max, u_max = self._bbox_extfov_vu
        template_img = self._model_img[v_min:v_max, u_min:u_max].copy()
        template_mask = self._body_mask[v_min:v_max, u_min:u_max].copy()
        features: list[NavFeature] = [
            NavFeature(
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
                template_img=template_img,
                template_mask=template_mask,
            )
        ]
        shape = load_body_shape(self._body_name, config=self._config)
        blob_min_px = max(BODY_BLOB_MIN_DIAMETER_PX, shape.min_blob_diameter_px)
        if self._predicted_diameter_px >= blob_min_px:
            features.append(self._build_blob_feature(shape, context=context))
        limb_feature = self._build_limb_arc_feature()
        if limb_feature is not None:
            features.append(limb_feature)
        return features

    def _build_limb_arc_feature(self) -> NavFeature | None:
        """Emit a LIMB_ARC from the rendered silhouette boundary, or ``None``.

        The predicted limb is the silhouette boundary of the rendered body; on a
        low-phase body it is essentially all lit, so BodyLimbNav's distance-
        transform fit aligns it to the image edge and recovers the offset.  The
        feature is emitted only when the boundary has enough vertices to
        constrain the fit (a tiny or barely-resolved body yields too short an
        arc).  The shadow-side vertices of a higher-phase body carry no sharp
        image edge and are down-weighted by the technique's robust fit rather
        than excluded here.
        """
        if self._limb_mask is None or self._body_mask is None:
            return None
        if self._predicted_diameter_px < _MIN_LIMB_DIAMETER_PX:
            return None
        if float(self._metadata.get('phase_angle_deg', 0.0)) > _LIMB_MAX_PHASE_DEG:
            return None
        vertices_vu, normals_vu = _limb_polyline_from_mask(self._limb_mask, self._body_mask)
        n = vertices_vu.shape[0]
        if n < _MIN_LIMB_ARC_VERTICES:
            return None
        sigma_normal = np.full(n, _LIMB_SIGMA_NORMAL_PX, dtype=np.float64)
        sigma_tangent = np.full(n, _LIMB_SIGMA_TANGENT_PX, dtype=np.float64)
        return NavFeature(
            feature_id=f'limb_arc:{self._body_name}',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model=self.name,
            geometry=LimbPolyline(
                vertices_vu=vertices_vu,
                normals_vu=normals_vu,
                sigma_normal_per_vertex_px=sigma_normal,
                sigma_tangent_per_vertex_px=sigma_tangent,
                bbox_extfov_vu=self._bbox_extfov_vu,
            ),
            subject_range_km=self._subject_range_km,
            position_cov_px=None,
            intensity_sigma_rel=0.0,
            preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
            reliability=1.0,
            reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
            usable_types=frozenset({NavFeatureType.LIMB_ARC}),
            flags=LimbArcFlags(body_name=self._body_name, visible_arc_fraction=1.0),
        )

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
