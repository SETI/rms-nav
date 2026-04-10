"""Navigation model for planetary rings.

This module implements the orchestrator for the planetary ring navigation model.
It is a thin coordinator that:

1. Reads ring configuration from YAML (``rings.ring_features.<PLANET>``).
2. Constructs typed ``RingFeature`` objects via ``RingFeature.from_config()``,
   raising ``ValueError`` immediately on malformed config.
3. Validates no two features have overlapping date ranges over the same radial region.
4. Checks whether rings are visible in the current observation.
5. Filters features through the four-pass ``RingFeatureFilter`` pipeline.
6. Renders each surviving feature by calling ``feature.render(context)``.
7. Wraps each render result in a ``NavModelResult`` with annotations.

**Design principle**: This module contains no physics, no math, and no rendering
logic. All of that lives in the ``rings`` subpackage (``ring_feature``,
``ring_math``, ``ring_filter``). The orchestrator's only job is to wire together
configuration loading, backplane access, and ``NavModelResult`` construction.

**Config key names** (as of this version):

- ``epoch``: UTC epoch string for radial mode calculations (required).
- ``fade_width_pix``: Desired fade extent in pixels for single-edge features.
- ``min_allowed_fade_width_pix``: Minimum allowed fade after conflict reduction.
- ``min_feature_pixels``: Minimum resolvable feature width in pixels.
- ``features``: Dict of feature key -> feature dict (parsed via
  ``RingFeature.from_config``).
"""

import math
from typing import Any

import numpy as np
import oops

from nav.config import Config
from nav.support.time import now_dt, utc_to_et
from nav.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model_result import NavModelResult
from .nav_model_rings_base import NavModelRingsBase
from .rings import (
    RingFeature,
    RingFeatureFilter,
    RingsRenderContext,
    validate_no_date_overlaps,
)

# Config parameters with defaults
_DEFAULT_FADE_WIDTH_PIX = 100.0
_DEFAULT_MIN_ALLOWED_FADE_WIDTH_PIX = 3.0
_DEFAULT_MIN_FEATURE_PIXELS = 2.0


class NavModelRings(NavModelRingsBase):
    """Navigation model for planetary rings based on ephemeris data.

    Loads ring features from a YAML config file, filters them for the current
    observation (date, visibility, resolvability, fade conflicts), renders each
    surviving feature via ``RingFeature.render()``, and appends the results to
    ``self._models`` as ``NavModelResult`` instances.

    Each rendered edge becomes a separate ``NavModelResult`` so the navigator can
    independently offset-correct individual ring features.
    """

    def __init__(self, name: str, obs: oops.Observation, *, config: Config | None = None) -> None:
        """Create a navigation model for planetary rings.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """
        super().__init__(name, obs, config=config)

    def create_model(
        self,
        *,
        always_create_model: bool = False,
        never_create_model: bool = False,
        create_annotations: bool = True,
    ) -> None:
        """Create the internal model representation for planetary rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have
                useful contents.
            never_create_model: If True, only creates metadata without generating
                a model or annotations.
            create_annotations: If True, creates text annotations for the model.
        """
        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None

        self._metadata = metadata
        self._models.clear()

        with self._logger.open('CREATE RINGS MODEL'):
            self._create_model(
                always_create_model=always_create_model,
                never_create_model=never_create_model,
                create_annotations=create_annotations,
            )

        end_time = now_dt()
        metadata['end_time'] = end_time.isoformat()
        metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _create_empty_model_result(self) -> NavModelResult:
        """Return an empty NavModelResult with zeros image/mask and inf range."""
        obs = self.obs
        empty_img = obs.make_extfov_zeros()
        empty_mask = obs.make_extfov_false()
        empty_range = obs.make_extfov_zeros()
        empty_range[:, :] = math.inf
        return NavModelResult(
            model_img=empty_img,
            model_mask=empty_mask,
            weighted_mask=None,
            range=empty_range,
            blur_amount=None,
            uncertainty=0.0,
            confidence=1.0,
            stretch_regions=None,
            annotations=None,
        )

    def _create_model(
        self,
        always_create_model: bool,
        never_create_model: bool,
        create_annotations: bool,
    ) -> None:
        """Create the internal model for planetary rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have
                useful contents.
            never_create_model: If True, only creates metadata without rendering.
            create_annotations: If True, creates text annotations for the model.
        """
        obs = self.obs
        planet = obs.closest_planet
        if planet is None:
            self._logger.warning('No closest planet found -- cannot create ring model')
            return

        rings_config = self._config.rings
        if not hasattr(rings_config, 'ring_features'):
            self._logger.error('Configuration has no rings.ring_features section')
            return

        ring_features_dict = getattr(rings_config, 'ring_features', {})
        if planet not in ring_features_dict:
            self._logger.warning('No ring features configured for planet %s', planet)
            return

        planet_config = ring_features_dict[planet]

        # ------------------------------------------------------------------
        # Read planet-level config parameters
        # ------------------------------------------------------------------
        epoch_str: str | None = planet_config.get('epoch')
        if epoch_str is None:
            raise ValueError(f'No epoch configured for planet {planet}')
        epoch = utc_to_et(epoch_str)

        fade_width_pix = float(planet_config.get('fade_width_pix', _DEFAULT_FADE_WIDTH_PIX))
        if fade_width_pix <= 0:
            raise ValueError(f'Invalid fade_width_pix {fade_width_pix} for planet {planet}')

        min_allowed_fade_width_pix = float(
            planet_config.get('min_allowed_fade_width_pix', _DEFAULT_MIN_ALLOWED_FADE_WIDTH_PIX)
        )
        if min_allowed_fade_width_pix <= 0:
            raise ValueError(
                f'Invalid min_allowed_fade_width_pix {min_allowed_fade_width_pix} '
                f'for planet {planet}'
            )

        min_feature_pixels = float(
            planet_config.get('min_feature_pixels', _DEFAULT_MIN_FEATURE_PIXELS)
        )
        if min_feature_pixels <= 0:
            raise ValueError(f'Invalid min_feature_pixels {min_feature_pixels} for planet {planet}')

        self._logger.info(
            'Planet: %s, epoch: %s, fade_width_pix: %.1f',
            planet,
            epoch_str,
            fade_width_pix,
        )

        # ------------------------------------------------------------------
        # Load features from the 'features' sub-dict
        # ------------------------------------------------------------------
        features_dict: dict[str, Any] = planet_config.get('features', {})
        if not features_dict:
            self._logger.warning('No features found under rings.ring_features.%s.features', planet)
            if always_create_model:
                self._models.append(self._create_empty_model_result())
            return

        features: list[RingFeature] = []
        for key, data in features_dict.items():
            if not isinstance(data, dict):
                raise ValueError(
                    f'Ring config error: features.{key} is not a dict (got {type(data).__name__!r})'
                )
            features.append(RingFeature.from_config(key, data))

        validate_no_date_overlaps(features)
        self._logger.info('Loaded %d ring features for %s', len(features), planet)

        # ------------------------------------------------------------------
        # Check ring visibility
        # ------------------------------------------------------------------
        ring_target = f'{planet.lower()}:ring'
        bp_radii = obs.ext_bp.ring_radius(ring_target)
        if bp_radii.is_all_masked():
            self._logger.info('No rings visible in observation')
            if not always_create_model:
                return
            self._models.append(self._create_empty_model_result())
            return

        min_radius = float(bp_radii.min().vals)
        max_radius = float(bp_radii.max().vals)
        self._logger.info('Ring radii: min=%.2f km, max=%.2f km', min_radius, max_radius)

        # ------------------------------------------------------------------
        # Build resolutions backplane and resolution-at-radius lookup
        # ------------------------------------------------------------------
        resolutions: NDArrayFloatType = obs.ext_bp.ring_radial_resolution(ring_target).vals

        def min_res_at_radius(a: float) -> float | None:
            """Return the minimum radial resolution (km/pixel) at radius ``a``."""
            border_arr: NDArrayBoolType = (
                obs.ext_bp.border_atop(bp_radii.key, a).mvals.astype('bool').filled(False)
            )
            res_at_edge = resolutions[border_arr]
            if len(res_at_edge) == 0:
                return None
            if hasattr(res_at_edge, 'is_all_masked') and res_at_edge.is_all_masked():
                return None
            min_val = float(np.min(res_at_edge))
            return min_val if min_val > 0.0 else None

        # ------------------------------------------------------------------
        # Filter features
        # ------------------------------------------------------------------
        feature_filter = RingFeatureFilter(
            obs_time_et=obs.midtime,
            min_radius=min_radius,
            max_radius=max_radius,
            min_res_at_radius=min_res_at_radius,
            fade_width_pix=fade_width_pix,
            min_allowed_fade_width_pix=min_allowed_fade_width_pix,
            min_feature_pixels=min_feature_pixels,
        )
        surviving = feature_filter.filter(features)
        if not surviving:
            self._logger.warning('No ring features survived filtering')
            if always_create_model:
                self._models.append(self._create_empty_model_result())
            return

        self._logger.info('%d features survived filtering', len(surviving))

        # ------------------------------------------------------------------
        # Handle never_create_model
        # ------------------------------------------------------------------
        if never_create_model:
            self._metadata['planet'] = planet
            self._metadata['epoch'] = epoch_str
            self._metadata['feature_count'] = len(surviving)
            self._metadata['features'] = [
                {'name': f.name, 'type': f.feature_type.value} for f in surviving
            ]
            return

        # ------------------------------------------------------------------
        # Build all_edge_radii for fade-conflict width reduction in render
        # ------------------------------------------------------------------
        all_edge_radii: list[tuple[float, str]] = []
        for feat in surviving:
            all_edge_radii.extend(feat.all_base_radii())
        all_edge_radii.sort(key=lambda x: x[0])

        # ------------------------------------------------------------------
        # Distance backplane for range field in NavModelResult
        # ------------------------------------------------------------------
        bp_distance = obs.ext_bp.distance(ring_target, direction='dep')
        distance_arr = bp_distance.mvals.filled(math.inf)

        # ------------------------------------------------------------------
        # Render each surviving feature
        # ------------------------------------------------------------------
        for feature in surviving:
            context = RingsRenderContext(
                obs=obs,
                ring_target=ring_target,
                epoch=epoch,
                resolutions=resolutions,
                fade_width_pix=fade_width_pix,
                all_edge_radii=tuple(all_edge_radii),
            )
            render_results = feature.render(context)

            for render_result in render_results:
                feat_model = render_result.model_img
                feat_mask = render_result.model_mask

                range_arr = obs.make_extfov_zeros()
                range_arr[:, :] = distance_arr
                range_arr[~feat_mask] = math.inf

                annotations = None
                if create_annotations:
                    annotations = self._create_edge_annotations(
                        obs, render_result.edge_info_list, feat_mask
                    )

                self._models.append(
                    NavModelResult(
                        model_img=feat_model,
                        model_mask=feat_mask,
                        weighted_mask=None,
                        range=range_arr,
                        blur_amount=None,
                        uncertainty=render_result.uncertainty,
                        confidence=1.0,
                        stretch_regions=None,
                        annotations=annotations,
                    )
                )

        # ------------------------------------------------------------------
        # Update metadata
        # ------------------------------------------------------------------
        self._metadata['planet'] = planet
        self._metadata['epoch'] = epoch_str
        self._metadata['feature_count'] = len(surviving)
        self._metadata['features'] = [
            {'name': f.name, 'type': f.feature_type.value} for f in surviving
        ]

        n = len(self._models)
        self._logger.info('Model created: %d result%s', n, 's' if n != 1 else '')
