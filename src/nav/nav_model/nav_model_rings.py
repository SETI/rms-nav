"""Navigation model for planetary rings.

This module implements the orchestrator for the planetary ring navigation model.
It is a thin coordinator that:

1. Reads the planet block from merged config (``rings.ring_features.<PLANET>``),
   including required planet-level parameters (``epoch``, fade settings, etc.).
2. Requires a non-empty ``features`` dict under that block (each entry is
   validated when passed to ``RingFeature.from_config()``).
3. Checks ring visibility via the ``ring_radius`` backplane **before** calling
   ``RingFeature.from_config()``, so feature parsing is skipped when no ring
   radii appear in the FOV.
4. Constructs typed ``RingFeature`` objects via ``RingFeature.from_config()``,
   raising ``ValueError`` on malformed feature entries.
5. Validates no two features have overlapping date ranges over the same radial
   region (``validate_no_date_overlaps``).
6. Filters through the four-pass ``RingFeatureFilter`` pipeline.
7. Renders each surviving feature via ``feature.render(context)`` and wraps
   each result in a ``NavModelResult`` with annotations.

**Design principle**: This module contains no physics, no math, and no rendering
logic. All of that lives in the ``rings`` subpackage (``ring_feature``,
``ring_math``, ``ring_filter``). The orchestrator's only job is to wire together
configuration retrieval, backplane access, and ``NavModelResult`` construction.

**Planet ring block keys** (under ``rings.ring_features.<PLANET>``):

- ``general.log_level_model_rings``: Log level for the ``CREATE RINGS MODEL``
  ``PdsLogger.open()`` block. Ring filtering, feature rendering, and fade math log
  through the same ``PdsLogger`` as this model (``debug`` for internals,
  ``info`` for summaries on the orchestrator).
- ``epoch``: UTC epoch string for radial mode calculations (required).
- ``fade_width_pix``: Desired fade extent in pixels for single-edge features
  (required; no default).
- ``min_allowed_fade_width_pix``: Minimum allowed fade after conflict reduction
  (required; no default).
- ``min_feature_pixels``: Minimum resolvable feature width in pixels (required;
  no default).
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


def _require_positive_finite_planet_scalar(
    planet: str, key: str, planet_config: dict[str, Any]
) -> float:
    """Return a positive finite float from ``planet_config[key]``.

    Raises:
        ValueError: If ``key`` is missing or the value is not a positive finite float.
    """
    if key not in planet_config:
        raise ValueError(f'Missing required ring configuration key {key!r} for planet {planet}')
    raw: Any = planet_config[key]
    if isinstance(raw, bool):
        raise ValueError(
            f'Invalid {key} for planet {planet}: expected a finite numeric value, got bool'
        )
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid {key} for planet {planet}: expected a finite numeric value, got {raw!r}'
        ) from exc
    if not math.isfinite(v):
        raise ValueError(f'Invalid {key} {v} for planet {planet} (must be finite)')
    if v <= 0.0:
        raise ValueError(f'Invalid {key} {v} for planet {planet}')
    return v


class NavModelRings(NavModelRingsBase):
    """Navigation model for planetary rings based on ephemeris data.

    Retrieves ring feature definitions from the merged configuration, filters them
    for the current observation (date, visibility, resolvability, fade conflicts),
    renders each surviving feature via ``RingFeature.render()``, and appends the
    results to ``self._models`` as ``NavModelResult`` instances.

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

        log_level = self._config.general.get('log_level_model_rings')
        with self._logger.open('CREATE RINGS MODEL', level=log_level):
            self._create_model(
                always_create_model=always_create_model,
                never_create_model=never_create_model,
                create_annotations=create_annotations,
            )

        end_time = now_dt()
        metadata['end_time'] = end_time.isoformat()
        metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _create_empty_model_result(self) -> NavModelResult:
        """Build a placeholder ``NavModelResult`` when no ring content is rendered.

        Returns:
            ``NavModelResult`` with ``model_img`` and ``model_mask`` as extended-FOV
            zero arrays from the observation, ``range`` filled with ``math.inf``,
            ``uncertainty`` 0.0, ``confidence`` 1.0, and other optional fields unset
            (``weighted_mask``, ``blur_amount``, ``stretch_regions``, ``annotations``).
        """
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

        Returns:
            None. Appends ``NavModelResult`` entries to ``self._models`` when
            features are rendered; leaves ``self._models`` empty when returning
            early or when ``never_create_model`` is True (after updating
            ``self._metadata``).

        Raises:
            ValueError: If the planet block under ``rings.ring_features`` is not a
                mapping, ``features`` is missing or not a dict, ``epoch`` or a
                required numeric key is missing, numeric parameters are out of range,
                a feature entry is not a dict, or ``RingFeature.from_config`` /
                ``validate_no_date_overlaps`` rejects the configuration.
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
        if not isinstance(planet_config, dict):
            raise ValueError(
                f'Ring config error: rings.ring_features entry for planet {planet!r} must be '
                f'a dict (got {type(planet_config).__name__!r})'
            )

        # ------------------------------------------------------------------
        # Read planet-level config parameters (all required; no defaults)
        # ------------------------------------------------------------------
        epoch_str = planet_config.get('epoch')
        if epoch_str is None:
            raise ValueError(f'No epoch configured for planet {planet}')
        if not isinstance(epoch_str, str):
            raise ValueError(
                f'Ring config error: epoch for planet {planet!r} must be a string, '
                f'got {type(epoch_str).__name__!r}'
            )
        try:
            epoch = utc_to_et(epoch_str)
        except ValueError as exc:
            raise ValueError(
                f'Ring config error: epoch for planet {planet!r} is not a valid UTC '
                f'string ({epoch_str!r}): {exc}'
            ) from exc

        fade_width_pix = _require_positive_finite_planet_scalar(
            planet, 'fade_width_pix', planet_config
        )
        min_allowed_fade_width_pix = _require_positive_finite_planet_scalar(
            planet,
            'min_allowed_fade_width_pix',
            planet_config,
        )
        min_feature_pixels = _require_positive_finite_planet_scalar(
            planet, 'min_feature_pixels', planet_config
        )

        self._logger.debug(
            'Rings config: planet=%s epoch=%s fade_width_pix=%.1f '
            'min_allowed_fade_width_pix=%.1f min_feature_pixels=%.1f',
            planet,
            epoch_str,
            fade_width_pix,
            min_allowed_fade_width_pix,
            min_feature_pixels,
        )

        # ------------------------------------------------------------------
        # Feature map (validate before backplanes: must be a dict of feature dicts)
        # ------------------------------------------------------------------
        if 'features' not in planet_config:
            raise ValueError(
                f'Missing required ring configuration key "features" for planet {planet}'
            )
        features_raw: Any = planet_config['features']
        if not isinstance(features_raw, dict):
            raise ValueError(
                f'Ring config error: "features" for planet {planet!r} must be a dict '
                f'(got {type(features_raw).__name__!r})'
            )
        features_dict: dict[str, Any] = features_raw
        if not features_dict:
            self._logger.warning('No features found under rings.ring_features.%s.features', planet)
            if always_create_model:
                self._models.append(self._create_empty_model_result())
            return

        # -----------------------------------------------------------------------
        # Ring visibility (before building ``RingFeature`` instances from config)
        # -----------------------------------------------------------------------
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
        self._logger.info(
            'Ring radii in field of view: min=%.2f km, max=%.2f km',
            min_radius,
            max_radius,
        )

        # -----------------------------------------------------------------------
        # Retrieve RingFeature objects from the validated feature map
        # -----------------------------------------------------------------------
        features: list[RingFeature] = []
        for key, data in features_dict.items():
            if not isinstance(data, dict):
                raise ValueError(
                    f'Ring config error: planet {planet!r} features.{key!r} must be a dict '
                    f'(got {type(data).__name__!r})'
                )
            features.append(RingFeature.from_config(key, data))

        validate_no_date_overlaps(features)
        self._logger.info('Retrieved %d ring feature(s) for %s', len(features), planet)

        # -----------------------------------------------------------------------
        # Build resolutions backplane and resolution-at-radius lookup
        # -----------------------------------------------------------------------
        resolutions: NDArrayFloatType = obs.ext_bp.ring_radial_resolution(ring_target).vals

        def min_res_at_radius(a: float) -> float | None:
            """Minimum radial resolution (km/pixel) among pixels whose ring radius is ``a``.

            Uses ``obs.ext_bp.border_atop(bp_radii.key, a)`` to build a boolean mask of
            pixels whose nominal ring radius equals ``a`` (edge of the discrete radius
            sampling). ``resolutions`` is indexed by that mask; if no pixels are selected,
            the masked slice is empty, or it is entirely masked, returns ``None``.
            Otherwise returns the smallest positive finite value in the slice, or
            ``None`` if that minimum is not positive.
            """
            border_arr: NDArrayBoolType = (
                obs.ext_bp.border_atop(bp_radii.key, a).mvals.astype('bool').filled(False)
            )
            res_at_edge = resolutions[border_arr]
            res_ma = np.ma.asarray(res_at_edge)
            if res_ma.count() == 0:
                return None
            vals = np.asarray(res_ma.compressed(), dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            min_val = float(np.min(vals))
            return min_val if min_val > 0.0 else None

        # -----------------------------------------------------------------------
        # Filter features
        # -----------------------------------------------------------------------
        feature_filter = RingFeatureFilter(
            obs_time_et=obs.midtime,
            min_radius=min_radius,
            max_radius=max_radius,
            min_res_at_radius=min_res_at_radius,
            fade_width_pix=fade_width_pix,
            min_allowed_fade_width_pix=min_allowed_fade_width_pix,
            min_feature_pixels=min_feature_pixels,
            logger=self._logger,
        )
        surviving = feature_filter.filter(features)
        if not surviving:
            self._logger.info('No ring features passed filter for this observation')
            if always_create_model:
                self._models.append(self._create_empty_model_result())
            return

        self._logger.info('%d ring feature(s) passed filter', len(surviving))

        # -----------------------------------------------------------------------
        # Handle never_create_model
        # -----------------------------------------------------------------------
        if never_create_model:
            self._metadata['planet'] = planet
            self._metadata['epoch'] = epoch_str
            self._metadata['feature_count'] = len(surviving)
            self._metadata['features'] = [
                {'name': f.name, 'type': f.feature_type.value} for f in surviving
            ]
            return

        # -----------------------------------------------------------------------
        # Build all_edge_radii for fade-conflict width reduction in render
        # -----------------------------------------------------------------------
        all_edge_radii: list[tuple[float, str]] = []
        for feat in surviving:
            all_edge_radii.extend(feat.all_base_radii())
        all_edge_radii.sort(key=lambda x: x[0])

        # -----------------------------------------------------------------------
        # Distance backplane for range field in NavModelResult
        # -----------------------------------------------------------------------
        bp_distance = obs.ext_bp.distance(ring_target, direction='dep')
        distance_arr = bp_distance.mvals.filled(math.inf)

        # -----------------------------------------------------------------------
        # Render each surviving feature
        # -----------------------------------------------------------------------
        render_context = RingsRenderContext(
            obs=obs,
            ring_target=ring_target,
            epoch=epoch,
            resolutions=resolutions,
            fade_width_pix=fade_width_pix,
            all_edge_radii=tuple(all_edge_radii),
            logger=self._logger,
        )
        for feature in surviving:
            self._logger.debug(
                'Rendering ring feature %r type=%s',
                feature.key,
                feature.feature_type.value,
            )
            render_results = feature.render(render_context)
            self._logger.debug(
                'Ring feature %r produced %d render result(s)',
                feature.key,
                len(render_results),
            )

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
        self._logger.info('Ring model created: %d NavModelResult%s', n, 's' if n != 1 else '')
