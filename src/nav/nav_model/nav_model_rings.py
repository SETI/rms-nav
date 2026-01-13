"""Navigation model for planetary rings.

This module implements a navigation model system for planetary rings based on
YAML configuration files. The system renders known ring features (gaps and
ringlets) with proper anti-aliasing, handles single-edge features with
fading, creates text annotations, and supports date-based feature filtering.

The model uses ephemeris data to compute multi-mode ring edges, applies
anti-aliasing for smooth pixel transitions, and creates correlation-friendly
models for navigation offset determination.
"""

import math
from typing import Any, Optional

import numpy as np
import numpy.ma as ma
from scipy import ndimage

import oops
from oops.backplane import Backplane

from nav.annotation import (Annotation,
                            Annotations,
                            AnnotationTextInfo,
                            TextLocInfo,
                            TEXTINFO_LEFT_ARROW,
                            TEXTINFO_RIGHT_ARROW,
                            TEXTINFO_TOP_ARROW,
                            TEXTINFO_BOTTOM_ARROW)
from nav.config import Config
from nav.support.time import now_dt, utc_to_et
from nav.support.types import NDArrayBoolType, NDArrayFloatType

from .nav_model_rings_base import NavModelRingsBase


class NavModelRings(NavModelRingsBase):
    """Navigation model for planetary rings based on ephemeris data.

    This class creates navigation models for planetary rings by rendering
    known ring features (gaps and ringlets) from YAML configuration files.
    Features can have multiple orbital modes that create non-circular edges,
    and can be filtered by observation date.
    """

    def __init__(self,
                 name: str,
                 obs: oops.Observation,
                 *,
                 config: Optional[Config] = None) -> None:
        """Creates a navigation model for planetary rings.

        Parameters:
            name: The name of the model.
            obs: The Observation object containing image data.
            config: Configuration object to use. If None, uses DEFAULT_CONFIG.
        """

        super().__init__(name, obs, config=config)

    def create_model(self,
                     *,
                     always_create_model: bool = False,
                     never_create_model: bool = False,
                     create_annotations: bool = True) -> None:
        """Creates the internal model representation for planetary rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model or
                annotations.
            create_annotations: If True, creates text annotations for the model.
        """

        metadata: dict[str, Any] = {}
        start_time = now_dt()
        metadata['start_time'] = start_time.isoformat()
        metadata['end_time'] = None
        metadata['elapsed_time_sec'] = None

        self._metadata = metadata
        self._annotations = None
        self._uncertainty = 0.
        self._confidence = 1.0

        with self._logger.open('CREATE RINGS MODEL'):
            self._create_model(always_create_model=always_create_model,
                               never_create_model=never_create_model,
                               create_annotations=create_annotations)

        end_time = now_dt()
        metadata['end_time'] = end_time.isoformat()
        metadata['elapsed_time_sec'] = (end_time - start_time).total_seconds()

    def _create_model(self,
                      always_create_model: bool,
                      never_create_model: bool,
                      create_annotations: bool) -> None:
        """Creates the internal model representation for planetary rings.

        Parameters:
            always_create_model: If True, creates a model even if it won't have useful contents.
            never_create_model: If True, only creates metadata without generating a model.
            create_annotations: If True, creates text annotations for the model.
        """

        obs = self.obs
        planet = obs.closest_planet
        if planet is None:
            self._logger.warning('No closest planet found - cannot create ring model')
            return

        # Get ring features configuration
        rings_config = self._config.rings
        if not hasattr(rings_config, 'ring_features'):
            self._logger.error('Configuration has no rings.ring_features section')
            return

        ring_features_dict = getattr(rings_config, 'ring_features', {})
        if planet not in ring_features_dict:
            self._logger.warning(f'No ring features configured for planet {planet}')
            return

        planet_config = ring_features_dict[planet]

        # Get planet-specific configuration
        epoch_str = planet_config.get('epoch')  # only relevant for rings with multiple modes
        if epoch_str is None:
            raise ValueError('No epoch configured for planet {planet}')
        epoch = utc_to_et(epoch_str)

        feature_width_pix = planet_config.get('feature_width', 100)
        if feature_width_pix <= 0:
            raise ValueError(f'Invalid rings feature_width {feature_width_pix}')

        min_fade_width_multiplier = planet_config.get('min_fade_width_multiplier', 3.0)
        if min_fade_width_multiplier <= 0:
            raise ValueError(f'Invalid rings min_fade_width_multiplier {min_fade_width_multiplier}')

        self._logger.info(f'Planet: {planet}, Epoch for modes: {epoch_str}, '
                          f'Feature width: {feature_width_pix} pixels')

        # Determine ring target key
        ring_target = f'{planet.lower()}:ring'

        # Check if rings are visible in observation
        bp_radii = obs.ext_bp.ring_radius(ring_target)
        if bp_radii.is_all_masked():
            self._logger.info('No rings visible in observation')
            if not always_create_model:
                return
            self._model_img = obs.make_extfov_zeros()
            self._model_mask = obs.make_extfov_false()
            self._range = obs.make_extfov_zeros()
            self._range[:, :] = math.inf
            return

        min_radius = bp_radii.min().vals
        max_radius = bp_radii.max().vals
        self._logger.info(f'Ring radii: min={min_radius:.2f} km, max={max_radius:.2f} km')

        # Load and filter features by date
        features = self._load_ring_features(planet_config, obs.midtime,
                                            min_radius=min_radius, max_radius=max_radius)
        if not features:
            self._logger.warning('No ring features available')
            if not always_create_model:
                return
            self._model_img = obs.make_extfov_zeros()
            self._model_mask = obs.make_extfov_false()
            self._range = obs.make_extfov_zeros()
            self._range[:, :] = math.inf
            return

        self._logger.info(f'Loaded {len(features)} ring features')

        if never_create_model:
            self._metadata['planet'] = planet
            self._metadata['epoch'] = epoch_str
            self._metadata['feature_count'] = len(features)
            self._metadata['features'] = [
                {'name': f.get('name'), 'type': f.get('feature_type')}
                for f in features]
            return

        # Get backplanes
        radii_mvals = bp_radii.mvals
        resolutions = obs.ext_bp.ring_radial_resolution(ring_target).vals

        # Initialize model arrays
        model = obs.make_extfov_zeros()
        model_mask = obs.make_extfov_false()

        # Process features: gaps first, then ringlets
        feature_list_by_a: list[tuple[float, str]] = []
        for feature in features:
            feature_type = feature.get('feature_type')
            inner_data = feature.get('inner_data')
            outer_data = feature.get('outer_data')

            if inner_data:
                inner_a = self._get_base_radius(inner_data)
                if inner_a is not None:
                    feature_list_by_a.append((
                        inner_a,
                        'IEG' if feature_type == 'GAP' else 'IER'))
            if outer_data:
                outer_a = self._get_base_radius(outer_data)
                if outer_a is not None:
                    feature_list_by_a.append((
                        outer_a,
                        'OEG' if feature_type == 'GAP' else 'OER'))

        feature_list_by_a.sort(key=lambda x: x[0], reverse=True)

        # Render features
        for feature_type in ('GAP', 'RINGLET'):
            for feature in features:
                if feature.get('feature_type') != feature_type:
                    continue

                inner_data = feature.get('inner_data')
                outer_data = feature.get('outer_data')
                feature_name = feature.get('name')

                if (inner_data is not None and outer_data is not None and
                        feature_type == 'RINGLET'):
                    # Full ringlet - render both edges
                    self._render_full_ringlet(
                        obs, model, model_mask,
                        ring_target=ring_target, inner_data=inner_data,
                        outer_data=outer_data, epoch=epoch, radii_mvals=radii_mvals,
                        resolutions=resolutions, feature_name=feature_name)
                else:
                    # Single edge or gap
                    if inner_data is not None:
                        self._render_single_edge(
                            obs, model, model_mask,
                            ring_target=ring_target,
                            edge_data=inner_data,
                            epoch=epoch,
                            radii_mvals=radii_mvals,
                            resolutions=resolutions,
                            feature_width_pix=feature_width_pix,
                            min_fade_width_multiplier=min_fade_width_multiplier,
                            feature_list_by_a=feature_list_by_a,
                            feature_type=feature_type,
                            feature_name=feature_name,
                            edge_type='inner')
                    if outer_data is not None:
                        self._render_single_edge(
                            obs, model, model_mask,
                            ring_target=ring_target,
                            edge_data=outer_data,
                            epoch=epoch,
                            radii_mvals=radii_mvals,
                            resolutions=resolutions,
                            feature_width_pix=feature_width_pix,
                            min_fade_width_multiplier=min_fade_width_multiplier,
                            feature_list_by_a=feature_list_by_a,
                            feature_type=feature_type,
                            feature_name=feature_name,
                            edge_type='outer')

        # Compute range
        bp_distance = obs.ext_bp.distance(ring_target, direction='dep')
        distance_mvals = bp_distance.mvals
        self._range = obs.make_extfov_zeros()
        self._range[:, :] = distance_mvals.filled(math.inf)
        # Set range to inf where rings are not present
        self._range[~model_mask] = math.inf

        # Create annotations if requested
        if create_annotations:
            # Collect edge information for annotation
            edge_info_list: list[tuple[Backplane, float, str, str]] = []

            for feature in features:
                feature_type = feature.get('feature_type')
                feature_name = feature.get('name') or 'UNNAMED'
                inner_data = feature.get('inner_data')
                outer_data = feature.get('outer_data')

                if inner_data is not None:
                    inner_radii_bp = self._compute_edge_radii(
                        obs, ring_target, mode_data=inner_data, epoch=epoch)
                    inner_a = self._get_base_radius(inner_data)
                    if inner_radii_bp is not None and inner_a is not None:
                        edge_label = ('IEG' if feature_type == 'GAP' else 'IER')
                        label_text = f'{feature_name} {edge_label}'
                        # Compute edge mask from backplane
                        edge_mask = (obs.ext_bp.border_atop(inner_radii_bp.key, inner_a)
                                     .mvals.astype('bool').filled(False))
                        edge_info_list.append((edge_mask, label_text, edge_label))

                if outer_data is not None:
                    outer_radii_bp = self._compute_edge_radii(
                        obs, ring_target, mode_data=outer_data, epoch=epoch)
                    outer_a = self._get_base_radius(outer_data)
                    if outer_radii_bp is not None and outer_a is not None:
                        edge_label = ('OEG' if feature_type == 'GAP' else 'OER')
                        label_text = f'{feature_name} {edge_label}'
                        # Compute edge mask from backplane
                        edge_mask = (obs.ext_bp.border_atop(outer_radii_bp.key, outer_a)
                                     .mvals.astype('bool').filled(False))
                        edge_info_list.append((edge_mask, label_text, edge_label))

            self._annotations = self._create_edge_annotations(
                obs, edge_info_list, model_mask)

        self._model_img = model
        self._model_mask = model_mask

        # Update metadata
        self._metadata['planet'] = planet
        self._metadata['epoch'] = epoch_str
        self._metadata['feature_count'] = len(features)
        self._metadata['features'] = [
            {'name': f.get('name'), 'type': f.get('feature_type')}
            for f in features]

        self._logger.info('Model created')

    def _load_ring_features(self,
                            planet_config: dict[str, Any],
                            obs_time: float,
                            *,
                            min_radius: float,
                            max_radius: float) -> list[dict[str, Any]]:
        """Load and filter ring features from configuration by date.

        Parameters:
            planet_config: Dictionary containing planet-specific ring configuration including
                feature definitions.
            obs_time: Observation time in TDB seconds.

        Returns:
            List of feature dictionaries that match the observation date.
        """

        features: list[dict[str, Any]] = []
        feature_dict = {k: v for k, v in planet_config.items()  # TODO Not fond of this
                        if k not in ('epoch', 'feature_width', 'min_fade_width_multiplier')}

        for feature_key, feature_data in feature_dict.items():
            if not isinstance(feature_data, dict):  # TODO Too forgiving
                continue

            # Check date range
            start_date = feature_data.get('start_date')
            end_date = feature_data.get('end_date')

            if start_date is not None or end_date is not None:
                if start_date is not None:
                    try:
                        start_et = utc_to_et(start_date)
                    except Exception as e:
                        self._logger.warning(
                            f'Invalid start_date "{start_date}" for feature '
                            f'{feature_key}: {e}')
                        continue
                else:
                    start_et = None

                if end_date is not None:
                    try:
                        end_et = utc_to_et(end_date)
                    except Exception as e:
                        self._logger.warning(
                            f'Invalid end_date "{end_date}" for feature '
                            f'{feature_key}: {e}')
                        continue
                else:
                    end_et = None

                # Check if observation time is within range
                if start_et is not None and obs_time < start_et:
                    continue
                if end_et is not None and obs_time >= end_et:
                    continue

            # Validate feature structure
            feature_type = feature_data.get('feature_type')
            if feature_type not in ('GAP', 'RINGLET'):
                self._logger.warning(
                    f'Invalid feature_type "{feature_type}" for feature {feature_key}, skipping')
                continue

            inner_data = feature_data.get('inner_data')
            outer_data = feature_data.get('outer_data')

            if inner_data is None and outer_data is None:
                self._logger.warning(
                    f'Feature {feature_key} has neither inner_data nor outer_data, skipping')
                continue

            # Validate mode data structure
            if inner_data is not None:
                if not self._validate_mode_data(inner_data, feature_key=feature_key,
                                                edge_type='inner',
                                                min_radius=min_radius, max_radius=max_radius):
                    continue

            if outer_data is not None:
                if not self._validate_mode_data(outer_data, feature_key=feature_key,
                                                edge_type='outer',
                                                min_radius=min_radius, max_radius=max_radius):
                    continue

            features.append(feature_data)

        return features

    def _validate_mode_data(self,
                            mode_data: list[dict[str, Any]],
                            *,
                            feature_key: str,
                            edge_type: str,
                            min_radius: float,
                            max_radius: float) -> bool:
        """Validate mode data structure for a ring edge.

        Parameters:
            mode_data: List of mode dictionaries.
            feature_key: Name of the feature for error messages.
            edge_type: 'inner' or 'outer' for error messages.

        Returns:
            True if valid, False otherwise.
        """

        if not isinstance(mode_data, list) or len(mode_data) == 0:
            self._logger.warning(
                f'Feature {feature_key} {edge_type}_data is not a non-empty list, skipping')
            return False

        for i, mode in enumerate(mode_data):
            if not isinstance(mode, dict):
                self._logger.warning(
                    f'Feature {feature_key} {edge_type}_data[{i}] is not a dict, skipping')
                return False

            mode_num = mode.get('mode')
            if mode_num is None:
                self._logger.warning(
                    f'Feature {feature_key} {edge_type}_data[{i}] missing mode, skipping')
                return False

            # TODO
            # Mode 1 has different structure
            # required_fields = ['a', 'rms', 'ae', 'long_peri', 'rate_peri']
            # if mode_num != 1:
            #     required_fields = ['amplitude', 'phase', 'pattern_speed']
            # for field in required_fields:
            #     if field not in mode:
            #         self._logger.warning(
            #             f'Feature {feature_key} {edge_type}_data[{i}] '
            #             f'mode 1 missing {field}, skipping')
            #         return False
            # Validate a is positive
            if 'a' in mode:
                if not min_radius <= mode['a'] <= max_radius:
                    return False
                if mode['a'] <= 0:
                    self._logger.warning(
                        f'Feature {feature_key} {edge_type}_data[{i}] mode 1 '
                        f'has non-positive a={mode["a"]}, skipping')
                    return False
        return True

    def _get_base_radius(self,
                         mode_data: list[dict[str, Any]]) -> Optional[float]:
        """Get the base radius (semi-major axis) from mode data.

        Parameters:
            mode_data: List of mode dictionaries.

        Returns:
            Base radius in km, or None if not found.
        """

        if not mode_data:
            return None

        # Mode 1 contains the base radius
        for mode in mode_data:
            if mode['mode'] == 1:
                return float(mode['a'])

        return None

    def _parse_mode_data(self,
                         mode_data: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        """Parse mode data into format suitable for radial_mode computation.

        Parameters:
            mode_data: List of mode dictionaries from config.

        Returns:
            List of tuples: (mode, amplitude, phase_rad, speed_rad_per_sec)
                or (mode, a, ae, long_peri_rad, rate_peri_rad_per_sec) for mode 1.
        """

        parsed_modes: list[tuple[Any, ...]] = []

        for mode in mode_data:
            mode_num = mode['mode']

            # Skip inclination modes (mode > 90)
            if mode_num > 90:
                continue

            if mode_num == 1:
                # Mode 1: base radius with eccentricity
                a = mode['a']
                ae = mode['ae']
                long_peri = mode['long_peri']
                rate_peri = mode['rate_peri']

                # Convert long_peri from degrees to radians
                long_peri_rad = np.radians(long_peri)
                # Convert rate_peri from degrees/day to radians/second
                rate_peri_rad_per_sec = np.radians(rate_peri) / 86400.0

                parsed_modes.append((1, a, ae, long_peri_rad, rate_peri_rad_per_sec))
            else:
                # Other modes: amplitude, phase, pattern_speed
                amplitude = mode['amplitude']
                phase = mode['phase']
                pattern_speed = mode['pattern_speed']

                # Convert phase from degrees to radians
                phase_rad = np.radians(phase)
                # Convert pattern_speed from degrees/day to radians/second
                pattern_speed_rad_per_sec = np.radians(pattern_speed) / 86400.0

                parsed_modes.append((mode_num, amplitude, phase_rad, pattern_speed_rad_per_sec))

        return parsed_modes

    def _compute_edge_radii(self,
                            obs: oops.Observation,
                            ring_target: str,
                            *,
                            mode_data: list[dict[str, Any]],
                            epoch: float) -> Backplane:
        """Compute multi-mode edge radius using radial_mode.

        Parameters:
            obs: The observation object.
            ring_target: Ring target key (e.g., 'saturn:ring').
            mode_data: List of mode dictionaries from config.
            epoch: Epoch time in TDB seconds for mode calculations.

        Returns:
            Backplane containing the computed edge radii.
        """

        parsed_modes = self._parse_mode_data(mode_data)

        # Start with base ring radius
        radii_bp = obs.ext_bp.ring_radius(ring_target)

        # Apply modes sequentially
        for mode_info in parsed_modes:
            if mode_info[0] == 1:
                # Mode 1: special handling
                mode, a, ae, long_peri_rad, rate_peri_rad_per_sec = mode_info
                # For mode 1, we use the base radius 'a' and apply
                # eccentricity 'ae' as amplitude
                radii_bp = obs.ext_bp.radial_mode(
                    radii_bp.key,
                    mode,
                    epoch,
                    ae,  # amplitude
                    long_peri_rad,  # phase
                    rate_peri_rad_per_sec,  # speed
                    a0=a)  # reference semi-major axis
            else:
                # Other modes
                mode, amplitude, phase_rad, speed_rad_per_sec = mode_info
                radii_bp = obs.ext_bp.radial_mode(
                    radii_bp.key,
                    mode,
                    epoch,
                    amplitude,
                    phase_rad,
                    speed_rad_per_sec)

        return radii_bp

    def _find_resolutions_by_a(self,
                               obs: oops.Observation,
                               ring_target: str,
                               *,
                               a: float) -> tuple[float, float]:
        """Find the minimum and maximum resolutions at a given semi-major axis.

        Parameters:
            obs: The observation object.
            ring_target: Ring target key (e.g., 'saturn:ring').
            a: Semi-major axis in km.

        Returns:
            Tuple of (min_resolution, max_resolution) in km, or (0.0, 0.0) if not found.
        """

        resolutions = obs.ext_bp.ring_radial_resolution(ring_target)
        bp_radii = obs.ext_bp.ring_radius(ring_target)
        border = obs.ext_bp.border_atop(bp_radii.key, a).mvals.astype('bool').filled(False)
        res_set = resolutions[border]
        if len(res_set) == 0 or res_set.is_all_masked():
            return 0.0, 0.0

        min_val = res_set.min().vals
        max_val = res_set.max().vals
        return min_val, max_val

    def _compute_antialiasing(self,
                              *,
                              radii: np.ndarray,
                              edge_radius: float,
                              shade_above: bool,
                              resolutions: np.ndarray,
                              max_value: float = 1.0) -> np.ndarray:
        """Compute anti-aliasing shade at pixel boundaries.

        Creates smooth transitions at pixel boundaries where the ring edge crosses. The
        shade value represents the fraction of the pixel that is covered by the ring.

        Parameters:
            radii: Array of ring radii at pixel centers (km).
            edge_radius: Target edge radius (km).
            shade_above: If True, shade towards larger radii; if False, shade towards smaller radii.
            resolutions: Array of radial resolutions at each pixel (km).
            max_value: Maximum shade value (default 1.0).

        Returns:
            Array of shade values [0, max_value] for anti-aliasing.
        """

        if shade_above:
            shade_sign = 1.0
        else:
            shade_sign = -1.0

        # Compute shade based on distance from edge
        # When radii == edge_radius, shade should be 0.5 (pixel center at edge)
        # When edge is 0.5*resolution beyond pixel center, shade should be 1.0
        shade = 1.0 - shade_sign * (radii - edge_radius) / resolutions
        shade -= 0.5

        # Clip to valid range (note: old code had shade[shade > 1.] = 0.
        # which seems like a bug, but we'll match it for compatibility)
        shade[shade < 0.0] = 0.0
        shade[shade > 1.0] = 0.0
        shade *= max_value

        return np.asarray(shade, dtype=np.float64)

    def _compute_edge_fade(self,
                           *,
                           model: NDArrayFloatType,
                           radii: np.ndarray,
                           edge_radius: float,
                           shade_above: bool,
                           radius_width_km: float,
                           min_radius_width_km: float,
                           resolutions: np.ndarray,
                           feature_list_by_a: list[tuple[float, str]]
                           ) -> NDArrayFloatType | None:
        """Compute linear fade from a single edge.

        Creates a linear fade from the edge over the specified width. The fade provides a
        smooth gradient for correlation while avoiding false edges.

        Parameters:
            model: Current model array to add fade to.
            radii: Array of ring radii at pixel centers (km).
            edge_radius: Target edge radius (km).
            shade_above: If True, fade towards larger radii; if False, fade towards smaller radii.
            radius_width_km: Fade width in km.
            min_radius_width_km: Minimum fade width in km (feature will be skipped if width is
                smaller).
            resolutions: Array of radial resolutions at each pixel (km).
            feature_list_by_a: List of (radius, type) tuples for conflict checking.

        Returns:
            Updated model array with fade applied, or None if feature should be skipped.
        """

        if shade_above:
            shade_sign = 1
        else:
            shade_sign = -1

        # Check for conflicting features
        adjusted_width = radius_width_km
        for other_a, _other_type in feature_list_by_a:
            if 0 < shade_sign * (other_a - edge_radius) < radius_width_km:
                # Another feature is in the fade path - reduce width
                adjusted_width = abs(other_a - edge_radius) / 2
                self._logger.debug(
                    f'Adjusting fade width for feature at {edge_radius:.2f} '
                    f'vs {other_a:.2f}, new width {adjusted_width:.2f} km')

        if adjusted_width < min_radius_width_km:
            self._logger.debug(
                f'Skipping feature at {edge_radius:.2f} due to close proximity (width '
                f'{adjusted_width:.2f} < min {min_radius_width_km:.2f} km)')
            return None

        # Create fade array
        shade = np.zeros(radii.shape, dtype=np.float64)

        if shade_above:
            # Fade from edge_radius to edge_radius + width
            # Shade function: 1 - (a - a0) / w for a in [a0, a0+w]
            # Integral: Z = [(1+a0/w)*a - a^2/(2w)] / s
            def int_func(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
                """Integrate fade function for shade_above case."""
                result = (((1.0 + edge_radius / adjusted_width) * (a1 - a0) +
                          (a0**2 - a1**2) / (2.0 * adjusted_width)) /
                          resolutions)
                return np.asarray(result, dtype=np.float64)

            # Case analysis for pixel coverage
            pixel_lower = radii - resolutions / 2.0
            pixel_upper = radii + resolutions / 2.0

            # Case 1: Edge and fade end both within pixel
            eq2 = np.logical_and(pixel_lower <= edge_radius,
                                 edge_radius < pixel_upper)
            eq3 = np.logical_and(pixel_lower <= edge_radius + adjusted_width,
                                 edge_radius + adjusted_width < pixel_upper)
            eq_case1 = np.logical_and(eq2, eq3)
            case1 = int_func(np.full_like(radii, edge_radius),
                             np.full_like(radii, edge_radius + adjusted_width))
            shade[eq_case1] = case1[eq_case1]

            # Case 4: Edge before pixel, fade end after pixel (full coverage)
            eq_case4 = np.logical_and(edge_radius < pixel_lower,
                                      edge_radius + adjusted_width > pixel_upper)
            case4 = int_func(pixel_lower, pixel_upper)
            shade[eq_case4] = case4[eq_case4]

            # Case 2: Edge within pixel, fade end extends beyond
            eq_case2 = np.logical_and(eq2, np.logical_not(eq_case1))
            case2 = int_func(np.full_like(radii, edge_radius), pixel_upper)
            shade[eq_case2] = case2[eq_case2]

            # Case 3: Edge before pixel, fade end within pixel
            eq_case3 = np.logical_and(eq3, np.logical_not(eq_case1))
            case3 = int_func(pixel_lower,
                             np.full_like(radii, edge_radius + adjusted_width))
            shade[eq_case3] = case3[eq_case3]

        else:
            # Fade from edge_radius - width to edge_radius
            # Shade function: 1 - (a0 - a) / w for a in [a0-w, a0]
            # Integral: Z = [(1-a0/w)*a + a^2/(2w)] / s
            def int_func2(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
                """Integrate fade function for shade_below case."""
                result = (((1.0 - edge_radius / adjusted_width) * (a1 - a0) +
                          (a1**2 - a0**2) / (2.0 * adjusted_width)) /
                          resolutions)
                return np.asarray(result, dtype=np.float64)

            # Case analysis for pixel coverage
            pixel_lower = radii - resolutions / 2.0
            pixel_upper = radii + resolutions / 2.0

            # Case 1: Fade start and edge both within pixel
            eq2 = np.logical_and(pixel_lower < edge_radius,
                                 edge_radius <= pixel_upper)
            eq3 = np.logical_and(pixel_lower < edge_radius - adjusted_width,
                                 edge_radius - adjusted_width <= pixel_upper)
            eq_case1 = np.logical_and(eq2, eq3)
            case1 = int_func2(np.full_like(radii, edge_radius - adjusted_width),
                              np.full_like(radii, edge_radius))
            shade[eq_case1] = case1[eq_case1]

            # Case 4: Fade start before pixel, edge after pixel (full coverage)
            eq_case4 = np.logical_and(edge_radius > pixel_upper,
                                      edge_radius - adjusted_width < pixel_lower)
            case4 = int_func2(pixel_lower, pixel_upper)
            shade[eq_case4] = case4[eq_case4]

            # Case 2: Edge within pixel, fade start before pixel
            eq_case2 = np.logical_and(eq2, np.logical_not(eq_case1))
            case2 = int_func2(pixel_lower, np.full_like(radii, edge_radius))
            shade[eq_case2] = case2[eq_case2]

            # Case 3: Fade start within pixel, edge after pixel
            eq_case3 = np.logical_and(eq3, np.logical_not(eq_case1))
            case3 = int_func2(np.full_like(radii, edge_radius - adjusted_width),
                              pixel_upper)
            shade[eq_case3] = case3[eq_case3]

        # Clip shade to valid range and add to model
        shade = np.clip(shade, 0.0, 1.0)
        new_model = model + shade

        return new_model

    def _render_full_ringlet(self,
                             obs: oops.Observation,
                             model: NDArrayFloatType,
                             model_mask: NDArrayBoolType,
                             *,
                             ring_target: str,
                             inner_data: list[dict[str, Any]],
                             outer_data: list[dict[str, Any]],
                             epoch: float,
                             radii_mvals: ma.MaskedArray,
                             resolutions: np.ndarray,
                             feature_name: Optional[str]) -> None:
        """Render a complete ringlet with both inner and outer edges.

        Parameters:
            obs: The observation object.
            model: Model array to update.
            model_mask: Model mask array to update.
            ring_target: Ring target key.
            inner_data: Inner edge mode data.
            outer_data: Outer edge mode data.
            epoch: Epoch time for mode calculations.
            radii_mvals: Masked array of ring radii.
            resolutions: Array of radial resolutions.
            feature_name: Optional feature name for logging.
        """

        inner_radii_bp = self._compute_edge_radii(
            obs, ring_target, mode_data=inner_data, epoch=epoch)
        outer_radii_bp = self._compute_edge_radii(
            obs, ring_target, mode_data=outer_data, epoch=epoch)

        if inner_radii_bp is None or outer_radii_bp is None:
            self._logger.warning(f'Could not compute edge radii for ringlet {feature_name}')
            return

        inner_radii = inner_radii_bp.mvals
        outer_radii = outer_radii_bp.mvals

        inner_a = self._get_base_radius(inner_data)
        outer_a = self._get_base_radius(outer_data)

        if inner_a is None or outer_a is None:
            self._logger.warning(
                f'Could not get base radii for ringlet {feature_name}')
            return

        self._logger.debug(
            f'Rendering full ringlet {feature_name} from {inner_a:.2f} to '
            f'{outer_a:.2f} km')

        # Fill solid region between edges
        inner_above = (inner_radii - resolutions / 2.0 >= inner_a)
        outer_below = (outer_radii + resolutions / 2.0 <= outer_a)
        solid_ringlet = np.logical_and(inner_above, outer_below).filled(False)
        model[solid_ringlet] += 1.0
        model_mask[solid_ringlet] = True

        # Apply anti-aliasing at edges
        inner_radii_vals = inner_radii.filled(0.0)
        outer_radii_vals = outer_radii.filled(0.0)
        inner_mask = (~inner_radii.mask if hasattr(inner_radii, 'mask')
                      else np.ones_like(inner_radii_vals, dtype=bool))
        outer_mask = (~outer_radii.mask if hasattr(outer_radii, 'mask')
                      else np.ones_like(outer_radii_vals, dtype=bool))

        inner_shade = self._compute_antialiasing(
            radii=inner_radii_vals, edge_radius=inner_a, shade_above=False, resolutions=resolutions)
        outer_shade = self._compute_antialiasing(
            radii=outer_radii_vals, edge_radius=outer_a, shade_above=True, resolutions=resolutions)

        # Apply shades only where not already solid and where radii are valid
        inner_shade_mask = ~solid_ringlet & inner_mask
        outer_shade_mask = ~solid_ringlet & outer_mask

        model[inner_shade_mask] += inner_shade[inner_shade_mask]
        model[outer_shade_mask] += outer_shade[outer_shade_mask]

        # Update mask for shaded regions
        model_mask[inner_shade_mask] |= (inner_shade[inner_shade_mask] > 0.0)
        model_mask[outer_shade_mask] |= (outer_shade[outer_shade_mask] > 0.0)

    def _render_single_edge(self,
                            obs: oops.Observation,
                            model: NDArrayFloatType,
                            model_mask: NDArrayBoolType,
                            *,
                            ring_target: str,
                            edge_data: list[dict[str, Any]],
                            epoch: float,
                            radii_mvals: ma.MaskedArray,
                            resolutions: np.ndarray,
                            feature_width_pix: float,
                            min_fade_width_multiplier: float,
                            feature_list_by_a: list[tuple[float, str]],
                            feature_type: str,
                            feature_name: Optional[str],
                            edge_type: str) -> None:
        """Render a single edge with fading.

        Parameters:
            obs: The observation object.
            model: Model array to update.
            model_mask: Model mask array to update.
            ring_target: Ring target key.
            edge_data: Edge mode data.
            epoch: Epoch time for mode calculations.
            radii_mvals: Masked array of ring radii.
            resolutions: Array of radial resolutions.
            feature_width_pix: Feature width in pixels.
            min_fade_width_multiplier: Minimum fade width multiplier.
            feature_list_by_a: List of (radius, type) tuples for conflict
                checking.
            feature_type: 'GAP' or 'RINGLET'.
            feature_name: Optional feature name for logging.
            edge_type: 'inner' or 'outer'.
        """

        edge_radii_bp = self._compute_edge_radii(obs, ring_target, mode_data=edge_data, epoch=epoch)

        if edge_radii_bp is None:
            self._logger.warning(
                f'Could not compute edge radii for {edge_type} edge of '
                f'{feature_name}')
            return

        edge_radii = edge_radii_bp.mvals
        edge_a = self._get_base_radius(edge_data)

        if edge_a is None:
            self._logger.warning(
                f'Could not get base radius for {edge_type} edge of '
                f'{feature_name}')
            return

        # Determine fade direction
        if feature_type == 'RINGLET':
            # Ringlet inner edge: fade outward (shade_above=True)
            # Ringlet outer edge: fade inward (shade_above=False)
            shade_above = (edge_type == 'inner')
        else:
            # Gap inner edge: fade inward (shade_above=False)
            # Gap outer edge: fade outward (shade_above=True)
            shade_above = (edge_type == 'outer')

        # Get resolution at this radius
        min_res, max_res = self._find_resolutions_by_a(obs, ring_target, a=edge_a)
        if min_res == 0.0:
            self._logger.warning(
                f'Could not find resolution for edge at {edge_a:.2f} km')
            return

        # Calculate fade width
        radius_width_km = feature_width_pix * min_res
        min_radius_width_km = min_res * min_fade_width_multiplier

        self._logger.debug(
            f'Rendering {edge_type} edge of {feature_type} {feature_name} at '
            f'{edge_a:.2f} km, fade width {radius_width_km:.2f} km')

        # Apply fade
        new_model = self._compute_edge_fade(
            model=model,
            radii=edge_radii.filled(0.0),
            edge_radius=edge_a,
            shade_above=shade_above,
            radius_width_km=radius_width_km,
            min_radius_width_km=min_radius_width_km,
            resolutions=resolutions,
            feature_list_by_a=feature_list_by_a)

        if new_model is not None:
            # Update mask for faded region
            fade_mask = (new_model - model) > 0.0
            model[:, :] = new_model
            model_mask[fade_mask] = True
