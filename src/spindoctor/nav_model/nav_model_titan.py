"""Titan NavModel -- the haze envelope of a body whose atmosphere hides its surface.

Titan's thick haze hides the ground, so its visible edge is the haze top:
wavelength-dependent, hundreds of kilometers above the solid body, and not
even circular at high phase.  Ellipsoid limb / terminator / disc navigation
is therefore systematically wrong on Titan rather than merely noisy, and the
shape-based body model skips it.  This model renders what a haze navigator
needs instead: the geometric disc center, the image-plane direction toward
the sub-solar point (the axis the haze is mirror-symmetric about), the solid
and envelope radii, and a mask of the pixels the fit must ignore.

Whenever Titan is inside the extended field of view the model emits exactly
one ``TITAN_LIMB`` feature.  Frame quality lives in that feature's
reliability rather than in a decline: an envelope that cannot fit inside the
extended frame wherever the true pointing puts it, one too heavily occluded,
or one too small to measure scores exactly zero, and the standard per-type
reliability gate then removes it, so a marginal Titan resolves through the
same statuses as any other marginal scene.

Titan's atmosphere is unique among the currently navigated bodies
(transparent at some wavelengths), so this handling is a deliberate special
case and does not generalize to other thick-atmosphere bodies such as Venus.

The observation-side half -- every ``oops`` and star-catalog query -- lives
in :mod:`spindoctor.nav_model.titan_geometry`; everything here is a pure
function of the :class:`~spindoctor.nav_model.titan_geometry.TitanGeometryInputs`
it returns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from oops import Observation

from spindoctor.annotation import Annotations
from spindoctor.config import DEFAULT_CONFIG, Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import TitanHazeFlags
from spindoctor.feature.geometry import TitanHazeGeometry
from spindoctor.nav_model.nav_model import NavModel
from spindoctor.nav_model.nav_model_body import TITAN_BODY_NAME, bodies_in_extfov
from spindoctor.nav_model.titan_geometry import TitanGeometryInputs, geometry_from_obs
from spindoctor.support.filters import NavFilterKind, NavFilterSpec

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from spindoctor.nav_orchestrator.nav_context import NavContext

__all__ = [
    'HIGH_PHASE_DEG',
    'NavModelTitan',
    'build_titan_feature',
    'titan_haze_reliability',
]


HIGH_PHASE_DEG: float = 150.0
"""Phase angle above which the sunward limb has shrunk toward a crescent.

Sets the ``high_phase`` flag on the emitted feature.  Above this phase the
arc sector the circle fit relies on carries its least support, so the flag
marks the frames whose along-track uncertainty deserves the most scrutiny.
"""


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _envelope_fits_in_frame(geometry: TitanGeometryInputs) -> bool:
    """Whether the envelope disc dilated by the search window is fully framed.

    Full visibility is a property of Titan's TRUE position, which may sit
    anywhere inside the pointing search window, so the disc is dilated by
    that window before the containment test.  A frame that only looks
    fully-framed at the predicted pointing would let the fit sample sky.
    """
    reach = geometry.r_env_px + geometry.window_px
    v0, u0 = geometry.predicted_center_vu
    rows, cols = geometry.extfov_shape_vu
    if rows <= 0 or cols <= 0:
        return False
    if v0 - reach < 0.0 or v0 + reach > rows - 1:
        return False
    return not (u0 - reach < 0.0 or u0 + reach > cols - 1)


def titan_haze_reliability(
    geometry: TitanGeometryInputs, *, config: Config
) -> tuple[float, NavReliabilityBreakdown]:
    """Score a haze feature's reliability and report the components.

    The score is ``sigmoid((D - midpoint) / scale) * (1 - occluded)`` with
    ``D`` the apparent envelope diameter in pixels, so it rises with
    resolution and falls with occlusion.  It is forced to exactly ``0.0``
    under three hard conditions, each of which makes the fit unusable rather
    than merely imprecise:

    - the envelope disc dilated by the search window does not fit inside the
      extended frame;
    - the occluded fraction exceeds ``max_occluded_fraction``;
    - the envelope diameter is below ``min_envelope_diameter_px``.

    A zero can never clear the per-type reliability gate, so a hard
    condition is exactly as strong as a refusal to emit -- but it travels
    through the standard gate machinery, leaving an attributable record
    instead of a silent absence.

    Parameters:
        geometry: The observation-derived haze geometry.
        config: Configuration supplying the ``titan.navigation`` thresholds
            and the reliability sigmoid's midpoint and scale.

    Returns:
        ``(reliability, breakdown)`` where ``reliability`` lies in
        ``[0, 1]`` and ``breakdown`` carries the envelope diameter and the
        occluded fraction that produced it.
    """
    nav_config = config.titan['navigation']
    diameter_px = 2.0 * geometry.r_env_px
    occluded = geometry.occluded_fraction
    breakdown = NavReliabilityBreakdown(
        titan_envelope_diameter_px=diameter_px,
        titan_occluded_fraction=occluded,
    )
    if not _envelope_fits_in_frame(geometry):
        return 0.0, breakdown
    if occluded > float(nav_config['max_occluded_fraction']):
        return 0.0, breakdown
    if diameter_px < float(nav_config['min_envelope_diameter_px']):
        return 0.0, breakdown
    midpoint = float(nav_config['reliability_diameter_midpoint_px'])
    scale = float(nav_config['reliability_diameter_scale_px'])
    size_term = _sigmoid((diameter_px - midpoint) / scale)
    return float(min(max(size_term * (1.0 - occluded), 0.0), 1.0)), breakdown


def build_titan_feature(
    geometry: TitanGeometryInputs, *, source_model: str, config: Config
) -> NavFeature:
    """Build the single ``TITAN_LIMB`` feature from an evaluated geometry.

    Pure: every observation-dependent quantity already lives on
    ``geometry``, so the emission rules and the reliability formula are
    exercisable without an observation.

    Parameters:
        geometry: The observation-derived haze geometry.
        source_model: Name of the emitting NavModel.
        config: Configuration supplying the ``titan.navigation`` thresholds
            and the surface-window filter list.

    Returns:
        One ``TITAN_LIMB`` :class:`~spindoctor.feature.feature.NavFeature`
        carrying the haze geometry, its flags, and its reliability
        breakdown.
    """
    reliability, breakdown = titan_haze_reliability(geometry, config=config)
    surface_window_filters = {
        str(name).upper() for name in config.titan['navigation']['surface_window_filters']
    }
    return NavFeature(
        feature_id=f'titan_limb:{TITAN_BODY_NAME}',
        feature_type=NavFeatureType.TITAN_LIMB,
        source_model=source_model,
        geometry=TitanHazeGeometry(
            predicted_center_vu=geometry.predicted_center_vu,
            sun_angle_rad=geometry.theta_rad,
            axis_degenerate=geometry.axis_degenerate,
            phase_deg=geometry.phase_deg,
            r_solid_px=geometry.r_solid_px,
            r_env_px=geometry.r_env_px,
            km_per_px=geometry.km_per_px,
            contaminant_mask=geometry.contaminant_mask,
            filters=geometry.filters,
            bbox_extfov_vu=geometry.bbox_extfov_vu,
        ),
        subject_range_km=geometry.subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=breakdown,
        usable_types=frozenset({NavFeatureType.TITAN_LIMB}),
        flags=TitanHazeFlags(
            body_name=TITAN_BODY_NAME,
            surface_window_filter=any(
                f.upper() in surface_window_filters for f in geometry.filters
            ),
            high_phase=geometry.phase_deg >= HIGH_PHASE_DEG,
        ),
    )


class NavModelTitan(NavModel):
    """Haze-envelope NavModel for Titan.

    Renders the geometry a haze navigator needs -- disc center, sub-solar
    symmetry axis, solid and envelope radii, contaminant mask -- and emits
    exactly one ``TITAN_LIMB`` feature whenever Titan is inside the extended
    field of view.  Marginal frames are expressed as low or zero
    reliability, not as a refusal to emit, so every outcome carries an
    attributable record.

    Parameters:
        name: Model instance name (``'titan:TITAN'``).
        obs: Observation snapshot.
        inventory: Optional pre-computed Titan inventory entry; pulled from
            ``obs.inventory`` on demand otherwise.
        siblings: ``(body_name, inventory_entry)`` for the other bodies in
            the FOV, used for occlusion and for the contaminant mask;
            enumerated from ``obs`` on demand otherwise.
        config: Optional ``Config`` override.
    """

    def __init__(
        self,
        name: str,
        obs: Observation,
        *,
        inventory: dict[str, Any] | None = None,
        siblings: list[tuple[str, dict[str, Any]]] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__(name, obs, config=config)
        self._inventory = inventory
        self._siblings = siblings
        self._geometry: TitanGeometryInputs | None = None

    @classmethod
    def instances_for_obs(cls, obs: Observation, *, config: Config | None = None) -> list[NavModel]:
        """Return one instance when Titan is inside the extfov, else none.

        Parameters:
            obs: Observation snapshot.
            config: Configuration whose satellite catalog decides whether
                Titan is in the mission set.  ``None`` uses ``DEFAULT_CONFIG``.

        Returns:
            A single ``NavModelTitan`` when Titan is present in the extfov,
            otherwise an empty list.
        """
        # Simulated obs drive model selection from operator parameters, not the
        # SPICE inventory; mirror NavModelBody and build nothing here.
        if getattr(obs, 'is_simulated', False):
            return []
        if config is None:
            config = DEFAULT_CONFIG
        in_extfov = bodies_in_extfov(obs, config=config)
        out: list[NavModel] = []
        for body_name, entry in in_extfov:
            if body_name.upper() != TITAN_BODY_NAME:
                continue
            siblings = [
                (other, other_entry)
                for other, other_entry in in_extfov
                if other.upper() != TITAN_BODY_NAME
            ]
            out.append(cls('titan:TITAN', obs, inventory=entry, siblings=siblings, config=config))
        return out

    @property
    def geometry_inputs(self) -> TitanGeometryInputs:
        """The evaluated haze geometry, computed on first access."""
        if self._geometry is None:
            self._geometry = geometry_from_obs(
                self.obs, self._config, inventory=self._inventory, siblings=self._siblings
            )
        return self._geometry

    def create_model(self) -> None:
        """Evaluate the haze geometry and record it in ``metadata``."""
        self._metadata.clear()
        self._metadata['body'] = TITAN_BODY_NAME
        with self._logger.open('TITAN MODEL'):
            geometry = self.geometry_inputs
            self._metadata['predicted_center_vu'] = list(geometry.predicted_center_vu)
            self._metadata['km_per_pixel'] = geometry.km_per_px
            self._metadata['envelope_diameter_px'] = 2.0 * geometry.r_env_px
            self._metadata['solid_diameter_px'] = 2.0 * geometry.r_solid_px
            self._metadata['phase_angle_deg'] = geometry.phase_deg
            self._metadata['sun_angle_deg'] = math.degrees(geometry.theta_rad)
            self._metadata['axis_degenerate'] = geometry.axis_degenerate
            self._metadata['occluded_fraction'] = geometry.occluded_fraction
            self._metadata['filters'] = list(geometry.filters)
            self._logger.info(
                'Predicted center (v, u) = (%.2f, %.2f); envelope diameter = %.2f px; '
                'km/px = %.4f; phase = %.2f deg',
                geometry.predicted_center_vu[0],
                geometry.predicted_center_vu[1],
                2.0 * geometry.r_env_px,
                geometry.km_per_px,
                geometry.phase_deg,
            )
            self._logger.info(
                'Symmetry axis = %.2f deg (degenerate = %s); occluded fraction = %.3f',
                math.degrees(geometry.theta_rad),
                geometry.axis_degenerate,
                geometry.occluded_fraction,
            )

    def to_features(self, context: NavContext) -> list[NavFeature]:
        """Emit the single ``TITAN_LIMB`` feature for this frame.

        Parameters:
            context: Per-image navigation context; unused because every
                quantity the feature carries is predicted geometry, not
                measured from pixels.

        Returns:
            A one-element list, always.  Frame quality is carried by the
            feature's reliability -- exactly zero when the envelope cannot
            be fully framed, is too heavily occluded, or is too small --
            rather than by an empty list.
        """
        del context
        feature = build_titan_feature(
            self.geometry_inputs, source_model=self.name, config=self._config
        )
        self._logger.info('Emitted TITAN_LIMB feature with reliability %.3f', feature.reliability)
        return [feature]

    def to_annotations(self, context: NavContext) -> Annotations:
        """Return an empty annotation collection.

        Parameters:
            context: Per-image navigation context; unused because the model
                renders no overlay.

        Returns:
            An empty ``Annotations`` collection, always.
        """
        del context
        return Annotations()
