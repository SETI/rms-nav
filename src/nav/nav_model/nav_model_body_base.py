"""Shared base class for body navigation models.

``NavModelBodyBase`` carries the limb-mask helper and the body-label
annotation pipeline shared between every concrete body NavModel.  It is
abstract — registered subclasses (``NavModelBodySimulated`` today, plus
the real-scene body model when it lands) inherit the helpers and supply
the per-image rendering.

Anti-aliasing math lives separately in
``nav.nav_model.rings.ring_math.compute_antialiasing``; helpers here are
strictly observation-aware (image shape, font config, label-placement
heuristics).
"""

import math

import numpy as np
import scipy.ndimage as ndimage

from nav.annotation import (
    TEXTINFO_BOTTOM_ARROW,
    TEXTINFO_LEFT_ARROW,
    TEXTINFO_RIGHT_ARROW,
    TEXTINFO_TOP_ARROW,
    Annotation,
    Annotations,
    AnnotationTextInfo,
    TextLocInfo,
)
from nav.feature.feature import NavFeature, NavReliabilityBreakdown
from nav.feature.feature_type import NavFeatureType
from nav.feature.flags import BodyBlobFlags
from nav.feature.geometry import BodyBlobGeometry
from nav.nav_model.body_shape import BodyShape
from nav.support.filters import NavFilterKind, NavFilterSpec
from nav.support.image import shift_array
from nav.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model import NavModel

__all__ = ['BODY_BLOB_MIN_DIAMETER_PX', 'NavModelBodyBase']


BODY_BLOB_MIN_DIAMETER_PX: float = 8.0
"""Minimum predicted disc diameter (px) at which BODY_BLOB is emitted.

Below this diameter the predicted body silhouette is too small for a
brightness-weighted centroid to pin the body to better than ~1 px, so
the extractor emits no body feature for the image.  The per-body
shape table can override this floor for known irregular / gas-giant
bodies.
"""


_SUB_SOLAR_MIN_OFFSET_PX: float = 0.5
"""Minimum lit-centroid offset (px) for a meaningful sub-solar direction.

Below this the body is near full phase, the lit and geometric centroids
coincide to within pixel noise, and the direction toward the bright limb
is undefined; the BODY_BLOB feature then carries ``(0.0, 0.0)`` and
BodyBlobNav falls back to the filled-disc coarse template.
"""


class NavModelBodyBase(NavModel):
    """Base class for body navigation models.

    Provides shared helpers to compute a limb mask, build the BODY_BLOB
    feature, and create annotations consistent across every concrete body
    model.

    The BODY_BLOB construction (``_build_blob_feature`` and its
    ``_phase_irregularity_factor`` / ``_lit_weighted_centroid_vu``
    helpers) lives here so the SPICE-backed ``NavModelBody`` and the
    simulated ``NavModelBodySimulated`` share one implementation rather
    than two copies of the same phase-and-irregularity calibration math.
    Subclasses populate the attributes the helpers read: ``_model_img``,
    ``_body_mask``, ``_predicted_center_vu``, ``_predicted_diameter_px``,
    ``_km_per_pixel_at_limb``, ``_subject_range_km``, ``_bbox_extfov_vu``,
    and ``_metadata['phase_angle_deg']``.
    """

    _abstract = True

    _model_img: NDArrayFloatType | None
    _body_mask: NDArrayBoolType | None
    _predicted_center_vu: tuple[float, float]
    _predicted_diameter_px: float
    _km_per_pixel_at_limb: float
    _subject_range_km: float
    _bbox_extfov_vu: tuple[int, int, int, int]

    def _compute_limb_mask_from_body_mask(self, body_mask: NDArrayBoolType) -> NDArrayBoolType:
        """Compute limb mask as body pixels adjacent to at least one non-body neighbor."""
        neighbor = (
            shift_array(~body_mask, (-1, 0))
            | shift_array(~body_mask, (1, 0))
            | shift_array(~body_mask, (0, -1))
            | shift_array(~body_mask, (0, 1))
        )
        return body_mask & neighbor

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
          incidence-driven brightness) and collapses to the geometric
          center at phase 0 where the model is uniform.
        * **Inflated centroid covariance.** The lit-weighted centroid
          assumes a body of *known* rotational orientation; the same
          scene on an irregular body whose pose we do not know carries a
          residual centroid bias the ellipsoidal model cannot remove.
          The bias scales with the shape's RMS departure from an
          ellipsoid (in km) and with how much of the body the lit
          hemisphere fails to sample (``1 + 2 * sin^2(phase / 2)`` runs
          from 1 at full-phase to 3 at full crescent).  This
          irregularity sigma is added to the photon-noise-limited
          centroid sigma in quadrature so the joint-fit covariance
          reflects both terms.
        * **``phase_irregularity_factor`` on the flags.** Surfaces the
          dimensionless ``sin(phase/2) * residual / radius`` so the
          BLOB confidence formula can down-weight irregular high-phase
          blobs without the technique having to recompute it.
        """
        assert self._model_img is not None
        assert self._body_mask is not None
        # Estimate per-pixel SNR from the mean signal across the body.
        snr = float(self._model_img[self._body_mask].mean()) if self._body_mask.any() else 0.0
        sigma_centroid = self._predicted_diameter_px / (
            2.0 * math.sqrt(max(int(self._body_mask.sum()), 1)) * max(snr, 1e-6)
        )
        phase_angle_deg = float(self._metadata.get('phase_angle_deg', 0.0))
        if not math.isfinite(phase_angle_deg):
            phase_angle_deg = 0.0
        # Clamp to the BodyBlobFlags valid range; phase outside [0, 180]
        # is a corner-case artefact, never a physical value.
        phase_angle_deg = max(0.0, min(180.0, phase_angle_deg))
        phase_irregularity_factor = self._phase_irregularity_factor(shape, phase_angle_deg)
        sigma_irregular_px = phase_irregularity_factor * (self._predicted_diameter_px / 2.0)
        sigma_total_px = math.sqrt(sigma_centroid * sigma_centroid + sigma_irregular_px**2)
        cov = (sigma_total_px * sigma_total_px) * np.eye(2, dtype=np.float64)
        predicted_center_vu = self._lit_weighted_centroid_vu()
        sub_solar_dir_vu = self._sub_solar_dir_vu(predicted_center_vu)
        body_name = getattr(self, '_body_name', 'BODY')
        return NavFeature(
            feature_id=f'body_blob:{body_name}',
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
                body_name=body_name,
                predicted_diameter_px=self._predicted_diameter_px,
                phase_angle_deg=phase_angle_deg,
                phase_irregularity_factor=phase_irregularity_factor,
                sub_solar_dir_vu=sub_solar_dir_vu,
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
          hemisphere is lit and the factor is ``1`` -- we still do
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

    def _sub_solar_dir_vu(self, lit_centroid_vu: tuple[float, float]) -> tuple[float, float]:
        """Return the unit image-plane direction toward the bright limb.

        The brightness-weighted centroid of a partially-lit body sits between
        the geometric center and the bright limb, so the vector from the
        geometric center (``_predicted_center_vu``) to the lit centroid points
        along the projected body-to-Sun direction.  ``BodyBlobNav`` orients its
        phase-aware coarse template along this direction (a filled disc cannot
        match a high-phase crescent).  At low phase the two centroids nearly
        coincide and the direction is meaningless, so it collapses to
        ``(0.0, 0.0)``; the technique uses the disc template there and never
        consults the direction.

        Parameters:
            lit_centroid_vu: The brightness-weighted centroid (the value
                stored as the feature's predicted center), in the same extfov
                frame as ``_predicted_center_vu``.

        Returns:
            Unit ``(v, u)`` direction toward the bright limb, or
            ``(0.0, 0.0)`` when the lit centroid is within
            :data:`_SUB_SOLAR_MIN_OFFSET_PX` of the geometric center.
        """
        dv = lit_centroid_vu[0] - self._predicted_center_vu[0]
        du = lit_centroid_vu[1] - self._predicted_center_vu[1]
        norm = math.hypot(dv, du)
        if norm < _SUB_SOLAR_MIN_OFFSET_PX:
            return (0.0, 0.0)
        return (dv / norm, du / norm)

    def _create_annotations(
        self,
        u_center: int,
        v_center: int,
        model: NDArrayFloatType,
        limb_mask: NDArrayBoolType,
        body_mask: NDArrayBoolType,
    ) -> Annotations:
        """Creates annotation objects for labeling the body in visualizations.

        This is functionally equivalent to the implementation used by the
        normal body model, so annotation behavior is consistent across models.
        """
        obs = self._obs
        body_name = getattr(self, '_body_name', 'BODY')
        body_config = self._config.bodies

        text_loc: list[TextLocInfo] = []
        v_center_extfov = v_center + obs.extfov_margin_v
        u_center_extfov = u_center + obs.extfov_margin_u

        v_center_extfov_clipped = np.clip(v_center_extfov, 0, body_mask.shape[0] - 1)
        u_center_extfov_clipped = np.clip(u_center_extfov, 0, body_mask.shape[1] - 1)
        if not body_mask[v_center_extfov_clipped].any():
            body_mask_u_min = 0
            body_mask_u_max = body_mask.shape[1] - 1
        else:
            body_mask_u_min = int(np.argmax(body_mask[v_center_extfov_clipped]))
            body_mask_u_max = int(
                body_mask.shape[1] - np.argmax(body_mask[v_center_extfov_clipped, ::-1]) - 1
            )
        body_mask_v_min = int(np.argmax(body_mask[:, u_center_extfov_clipped]))
        body_mask_v_max = int(
            body_mask.shape[0] - np.argmax(body_mask[::-1, u_center_extfov_clipped]) - 1
        )
        body_mask_u_ctr = (body_mask_u_min + body_mask_u_max) // 2
        body_mask_v_ctr = (body_mask_v_min + body_mask_v_max) // 2

        # Scan around center to place labels on limb
        for orig_dist in range(0, max(body_mask_v_ctr - body_mask_v_min, body_config.label_scan_v)):
            for neg in [-1, 1]:
                dist = orig_dist * neg
                v = body_mask_v_ctr + dist
                if not 0 <= v < body_mask.shape[0]:
                    continue

                # Left side
                u = int(np.argmax(body_mask[v]))
                if u > 0:
                    angle = np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov)) % 360
                    if 135 < angle < 225:  # Left side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_LEFT_ARROW, v, u - body_config.label_horiz_gap)
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u)
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u)
                        )

                # Right side
                u = body_mask.shape[1] - int(np.argmax(body_mask[v, ::-1])) - 1
                if u < body_mask.shape[1] - 1:
                    angle = np.rad2deg(np.arctan2(v - v_center_extfov, u - u_center_extfov)) % 360
                    if angle > 315 or angle < 45:  # Right side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u + body_config.label_horiz_gap)
                        )
                    elif angle >= 225:  # Top side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_TOP_ARROW, v - body_config.label_vert_gap, u)
                        )
                    else:  # Bottom side
                        text_loc.append(
                            TextLocInfo(TEXTINFO_BOTTOM_ARROW, v + body_config.label_vert_gap, u)
                        )

                if orig_dist == 0:
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_TOP_ARROW,
                            body_mask_v_min - body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    text_loc.append(
                        TextLocInfo(
                            TEXTINFO_BOTTOM_ARROW,
                            body_mask_v_max + body_config.label_vert_gap,
                            body_mask_u_ctr,
                        )
                    )
                    break

        # Coarse scan for additional candidates
        for v_orig_dist in range(0, body_mask_v_ctr - body_mask_v_min, body_config.label_grid_v):
            for v_neg in [-1, 1]:
                v_dist = v_orig_dist * v_neg
                v = body_mask_v_ctr + v_dist
                if not 0 <= v < body_mask.shape[0]:
                    continue
                for u_orig_dist in range(
                    0, body_mask_u_ctr - body_mask_u_min, body_config.label_grid_u
                ):
                    for u_neg in [-1, 1]:
                        u_dist = u_orig_dist * u_neg
                        u = body_mask_u_ctr + u_dist
                        if not 0 <= u < body_mask.shape[1]:
                            continue
                        if not body_mask[v, u]:
                            continue
                        if u < model.shape[1] // 2:
                            text_loc.append(TextLocInfo(TEXTINFO_LEFT_ARROW, v, u))
                        else:
                            text_loc.append(TextLocInfo(TEXTINFO_RIGHT_ARROW, v, u))
                if v_orig_dist == 0:
                    break

        text_info = AnnotationTextInfo(
            body_name,
            text_loc=text_loc,
            ref_vu=None,
            font=body_config.label_font,
            font_size=body_config.label_font_size,
            color=body_config.label_font_color,
        )

        text_avoid_mask = ndimage.maximum_filter(body_mask, body_config.label_mask_enlarge)

        annotation = Annotation(
            obs,
            limb_mask,
            body_config.label_limb_color,
            thicken_overlay=body_config.outline_thicken,
            avoid_mask=text_avoid_mask,
            text_info=text_info,
            config=self._config,
        )

        annotations = Annotations()
        annotations.add_annotations(annotation)
        return annotations


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
