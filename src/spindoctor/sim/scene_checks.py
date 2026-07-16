"""Field-type validators for the sim-scene schema.

The ``_check_*`` / ``_require_*`` helpers here enforce the per-field types of
every scene block: the per-object bodies / rings / stars entries, the optics
sub-blocks (PSF, smear, distortion, ghosts, stray light), the noise, detector,
artifacts, and spk_error blocks, plus the primitive scalar checks they all
share.  Each block's key inventory lives beside its checker (unknown keys fail
validation, so a typo cannot silently render an un-blurred or clean frame).

:func:`spindoctor.sim.scene.validate_sim_params` drives these helpers; every
violation raises :class:`spindoctor.sim.scene_schema.SimSceneValidationError`.
"""

from __future__ import annotations

import math
from typing import Any

from spindoctor.sim.scene_schema import _BODY_IDEALIZED_KEYS, SimSceneValidationError


def _check_body_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``bodies`` entry's field types."""
    label = f'bodies[{index}]'
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    shape_model = obj.get('shape_model')
    if shape_model is not None and shape_model not in ('ellipsoid', 'polyhedral_mesh'):
        raise SimSceneValidationError(
            f"{source}: {label}.shape_model must be 'ellipsoid' or 'polyhedral_mesh' "
            f'when present; got {shape_model!r}'
        )
    for key in (
        'center_v',
        'center_u',
        'axis1',
        'axis2',
        'axis3',
        'rotation_z',
        'rotation_tilt',
        'illumination_angle',
        'phase_angle',
        'range_km',
        'km_per_pixel',
        'mesh_lumpiness',
        'crater_fill',
        'crater_min_radius',
        'crater_max_radius',
        'crater_power_law_exponent',
        'crater_relief_scale',
        'anti_aliasing',
    ):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    for key in ('mesh_n_lat', 'mesh_n_lon', 'mesh_seed', 'seed'):
        if obj.get(key) is not None:
            _require_int(obj, key, source=f'{source}: {label}')
    pose = obj.get('pose_euler_deg')
    if pose is not None:
        if not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise SimSceneValidationError(
                f'{source}: {label}.pose_euler_deg must be a list of 3 angles when present'
            )
        for angle in pose:
            _check_optional_number(angle, f'{label}.pose_euler_deg[]', source=source)
    override = obj.get('nav_override')
    if override is not None:
        if not isinstance(override, dict):
            raise SimSceneValidationError(
                f'{source}: {label}.nav_override must be a mapping when present'
            )
        # The override expresses what the navigator BELIEVES about idealized
        # geometry, so only idealized body keys may appear in it.
        bad = set(override) - _BODY_IDEALIZED_KEYS
        if bad:
            raise SimSceneValidationError(
                f'{source}: {label}.nav_override may only override idealized body '
                f'keys; got {sorted(bad)}'
            )


def _check_ring_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``rings`` entry's field types."""
    label = f'rings[{index}]'
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    feature_type = obj.get('feature_type')
    if feature_type is not None and feature_type not in ('RINGLET', 'GAP'):
        raise SimSceneValidationError(
            f"{source}: {label}.feature_type must be 'RINGLET' or 'GAP' when present; "
            f'got {feature_type!r}'
        )
    for key in ('center_v', 'center_u', 'shading_distance', 'range', 'range_km'):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    for key in ('inner_data', 'outer_data'):
        _check_optional_mapping_list(obj.get(key), f'{label}.{key}', source=source)


def _check_star_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``stars`` entry's field types."""
    label = f'stars[{index}]'
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    _check_optional_str(obj.get('catalog_name'), f'{label}.catalog_name', source=source)
    _check_optional_str(obj.get('spectral_class'), f'{label}.spectral_class', source=source)
    for key in ('v', 'u', 'vmag', 'move_v', 'move_u'):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    _check_optional_positive_number(obj.get('psf_sigma'), f'{label}.psf_sigma', source=source)
    psf_size = obj.get('psf_size')
    if psf_size is not None:
        valid = isinstance(psf_size, (list, tuple)) and len(psf_size) == 2
        if not valid or any(isinstance(x, bool) or not isinstance(x, int) for x in psf_size):
            raise SimSceneValidationError(
                f'{source}: {label}.psf_size must be a list of 2 integers when present'
            )


# Allowed keys inside the scene-level optics block and its sub-blocks.  Unknown
# keys fail, so a typo does not silently render an un-blurred frame.
_OPTICS_KEYS: frozenset[str] = frozenset({'psf', 'smear', 'distortion', 'ghosts', 'stray_light'})
_PSF_KEYS: frozenset[str] = frozenset({'match_navigator', 'sigma_v', 'sigma_u', 'w', 'r0', 'n'})
_SMEAR_ENTRY_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'object_class'})
_SMEAR_OBJECT_CLASSES: frozenset[str] = frozenset({'all', 'stars', 'bodies', 'rings'})
_DISTORTION_KEYS: frozenset[str] = frozenset(
    {'k1', 'k2', 'center_v', 'center_u', 'nonradial_rms_px'}
)
_GHOST_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'amplitude', 'defocus_sigma'})
_SPK_ERROR_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'reference_range_km'})
_STRAY_LIGHT_KEYS: frozenset[str] = frozenset(
    {'amplitude', 'direction_deg', 'model', 'center_v', 'center_u'}
)


def _check_optics(value: Any, *, source: str) -> None:
    """Validate the scene-level ``optics`` block and every sub-block.

    Parameters:
        value: The ``optics`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid optics field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics must be a mapping when present')
    unknown = set(value) - _OPTICS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: optics: unknown keys: {sorted(unknown)}')
    _check_psf_block(value.get('psf'), source=source)
    _check_smear_list(value.get('smear'), source=source)
    _check_distortion_block(value.get('distortion'), source=source)
    _check_ghosts_list(value.get('ghosts'), source=source)
    _check_stray_light_block(value.get('stray_light'), source=source)


def _check_psf_block(value: Any, *, source: str) -> None:
    """Validate ``optics.psf`` (either match-navigator form or explicit params)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics.psf must be a mapping when present')
    unknown = set(value) - _PSF_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: optics.psf: unknown keys: {sorted(unknown)}')
    if 'match_navigator' in value:
        _check_optional_bool(
            value.get('match_navigator'), 'optics.psf.match_navigator', source=source
        )
        extra = set(value) - {'match_navigator'}
        if extra:
            raise SimSceneValidationError(
                f'{source}: optics.psf.match_navigator is exclusive; drop {sorted(extra)}'
            )
        return
    for key in ('sigma_v', 'sigma_u', 'r0'):
        _check_optional_positive_number(value.get(key), f'optics.psf.{key}', source=source)
    _check_optional_number(value.get('w'), 'optics.psf.w', source=source)
    _check_optional_number(value.get('n'), 'optics.psf.n', source=source)
    w = value.get('w')
    if w is not None and not 0.0 <= float(w) <= 1.0:
        raise SimSceneValidationError(f'{source}: optics.psf.w must lie in [0, 1]; got {w!r}')


def _check_smear_list(value: Any, *, source: str) -> None:
    """Validate ``optics.smear`` (a list of per-object-class motion entries)."""
    if value is None:
        return
    if not isinstance(value, list):
        raise SimSceneValidationError(f'{source}: optics.smear must be a list when present')
    for index, entry in enumerate(value):
        label = f'optics.smear[{index}]'
        if not isinstance(entry, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping')
        unknown = set(entry) - _SMEAR_ENTRY_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(entry.get('dv_px'), f'{label}.dv_px', source=source)
        _check_optional_number(entry.get('du_px'), f'{label}.du_px', source=source)
        object_class = entry.get('object_class', 'all')
        if object_class not in _SMEAR_OBJECT_CLASSES:
            raise SimSceneValidationError(
                f'{source}: {label}.object_class must be one of '
                f'{sorted(_SMEAR_OBJECT_CLASSES)}; got {object_class!r}'
            )


def _check_distortion_block(value: Any, *, source: str) -> None:
    """Validate ``optics.distortion`` (radial polynomial plus non-radial field)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: optics.distortion must be a mapping when present')
    unknown = set(value) - _DISTORTION_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: optics.distortion: unknown keys: {sorted(unknown)}'
        )
    for key in ('k1', 'k2', 'center_v', 'center_u'):
        _check_optional_number(value.get(key), f'optics.distortion.{key}', source=source)
    _check_optional_nonnegative_number(
        value.get('nonradial_rms_px'), 'optics.distortion.nonradial_rms_px', source=source
    )


def _check_ghosts_list(value: Any, *, source: str) -> None:
    """Validate ``optics.ghosts`` (a list of displaced defocused copies)."""
    if value is None:
        return
    if not isinstance(value, list):
        raise SimSceneValidationError(f'{source}: optics.ghosts must be a list when present')
    for index, entry in enumerate(value):
        label = f'optics.ghosts[{index}]'
        if not isinstance(entry, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping')
        unknown = set(entry) - _GHOST_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(entry.get('dv_px'), f'{label}.dv_px', source=source)
        _check_optional_number(entry.get('du_px'), f'{label}.du_px', source=source)
        _check_optional_number(entry.get('amplitude'), f'{label}.amplitude', source=source)
        _check_optional_nonnegative_number(
            entry.get('defocus_sigma'), f'{label}.defocus_sigma', source=source
        )


def _check_stray_light_block(value: Any, *, source: str) -> None:
    """Validate ``optics.stray_light`` (the smooth scattered-light ramp/bump)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: optics.stray_light must be a mapping when present'
        )
    unknown = set(value) - _STRAY_LIGHT_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: optics.stray_light: unknown keys: {sorted(unknown)}'
        )
    _check_optional_number(value.get('amplitude'), 'optics.stray_light.amplitude', source=source)
    _check_optional_number(
        value.get('direction_deg'), 'optics.stray_light.direction_deg', source=source
    )
    _check_optional_number(value.get('center_v'), 'optics.stray_light.center_v', source=source)
    _check_optional_number(value.get('center_u'), 'optics.stray_light.center_u', source=source)
    model = value.get('model')
    if model is not None and model not in ('linear', 'radial'):
        raise SimSceneValidationError(
            f"{source}: optics.stray_light.model must be 'linear' or 'radial'; got {model!r}"
        )


# The complete inventory of the truth-side noise block: the detector stage's
# stochastic / structured knobs plus the telemetry stage's missing-data rate.
# Unknown noise keys fail validation, so a typo cannot silently render the
# clean floor.  'poisson' is the only boolean; 'bloom_length' is an integer;
# 'vidicon' is a sub-mapping with its own inventory; everything else is a
# non-negative number.
_NOISE_BOOL_KEYS: frozenset[str] = frozenset({'poisson'})
_NOISE_INT_KEYS: frozenset[str] = frozenset({'bloom_length'})
_NOISE_NUMBER_KEYS: frozenset[str] = frozenset(
    {
        'read_noise_dn',
        'bias_dn',
        'cosmic_ray_rate_per_sec',
        'missing_data_rate',
        'signal_full_scale_frac',
        'pixel_area_cm2',
        'dark_current_e_per_sec',
        'hot_pixel_fraction',
        'hot_pixel_amplitude_e',
        'hot_pixel_column_factor',
        'banding_amplitude_e',
        'banding_period_px',
        'bias_pedestal_sigma_dn',
        'bias_row_gradient_dn',
        'bias_col_gradient_dn',
    }
)
_NOISE_KEYS: frozenset[str] = (
    _NOISE_BOOL_KEYS | _NOISE_INT_KEYS | _NOISE_NUMBER_KEYS | frozenset({'vidicon'})
)
_VIDICON_NOISE_KEYS: frozenset[str] = frozenset(
    {
        'read_noise_line_dn',
        'read_noise_pixel_dn',
        'coherent_amplitude_dn',
        'coherent_period_px',
    }
)

# The detector block selects the electron-chain gain state, the detector model
# (CCD electron chain or the Voyager vidicon DN path), the exposure the well
# fraction references, and the ADC quantization sub-mode.  Per-instrument
# defaults come from artifacts_catalog.py; scene keys override them.
_DETECTOR_KEYS: frozenset[str] = frozenset(
    {'gain_state', 'detector_model', 'exposure_ref_sec', 'quantization'}
)
_DETECTOR_MODELS: frozenset[str] = frozenset({'ccd', 'vidicon'})
# Quantization sub-modes: 'exact' rounds to integer DN (uniform bins); the ADC
# modes reproduce the documented histogram structure of each camera.
_QUANTIZATION_MODES: frozenset[str] = frozenset({'exact', 'uneven_12bit', '8bit', 'sqrt_lut'})
# The artifacts block: at this fidelity only the physical-chain opt-in switch
# (per-mode loss incidences and adversarial placement land with later phases).
_ARTIFACTS_KEYS: frozenset[str] = frozenset({'instrument_defaults'})


def _check_noise(value: Any, *, source: str) -> None:
    """Validate the scene-level ``noise`` block against its full key inventory.

    Parameters:
        value: The ``noise`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or mistyped noise field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: noise must be a mapping when present')
    unknown = set(value) - _NOISE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: noise: unknown keys: {sorted(unknown)}')
    for key in _NOISE_BOOL_KEYS:
        _check_optional_bool(value.get(key), f'noise.{key}', source=source)
    for key in _NOISE_INT_KEYS:
        _check_optional_nonnegative_int(value.get(key), f'noise.{key}', source=source)
    for key in _NOISE_NUMBER_KEYS:
        _check_optional_nonnegative_number(value.get(key), f'noise.{key}', source=source)
    vidicon = value.get('vidicon')
    if vidicon is None:
        return
    if not isinstance(vidicon, dict):
        raise SimSceneValidationError(f'{source}: noise.vidicon must be a mapping when present')
    unknown = set(vidicon) - _VIDICON_NOISE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: noise.vidicon: unknown keys: {sorted(unknown)}')
    for key in _VIDICON_NOISE_KEYS:
        _check_optional_nonnegative_number(vidicon.get(key), f'noise.vidicon.{key}', source=source)


def _check_detector(value: Any, *, instrument: str, source: str) -> None:
    """Validate the scene-level ``detector`` block's field types.

    Parameters:
        value: The ``detector`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name, used to
            check the selected gain state against the instrument's catalog.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid detector field,
            including a gain state the instrument does not catalogue.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: detector must be a mapping when present')
    unknown = set(value) - _DETECTOR_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: detector: unknown keys: {sorted(unknown)}')
    if value.get('gain_state') is not None:
        gain_state = _require_int(
            {'gain_state': value['gain_state']}, 'gain_state', source=f'{source}: detector'
        )
        # A gain state the instrument does not catalogue fails here, at
        # validation, with the catalogued alternatives in the message; the
        # render-time resolver keeps its own guard as a backstop for scenes
        # that bypass validation.
        from spindoctor.sim.forward.artifacts_catalog import resolve_detector_defaults

        table = resolve_detector_defaults(instrument).get('gain_e_per_dn_by_state') or {0: 1.0}
        if gain_state not in table:
            raise SimSceneValidationError(
                f'{source}: detector.gain_state {gain_state} is not catalogued for '
                f'instrument {instrument!r}; available states: {sorted(table)}'
            )
    model = value.get('detector_model')
    if model is not None and model not in _DETECTOR_MODELS:
        raise SimSceneValidationError(
            f'{source}: detector.detector_model must be one of {sorted(_DETECTOR_MODELS)}; '
            f'got {model!r}'
        )
    _check_optional_positive_number(
        value.get('exposure_ref_sec'), 'detector.exposure_ref_sec', source=source
    )
    quantization = value.get('quantization')
    if quantization is not None and quantization not in _QUANTIZATION_MODES:
        raise SimSceneValidationError(
            f'{source}: detector.quantization must be one of {sorted(_QUANTIZATION_MODES)}; '
            f'got {quantization!r}'
        )


def _check_artifacts(value: Any, *, source: str) -> None:
    """Validate the scene-level ``artifacts`` block's field types.

    Parameters:
        value: The ``artifacts`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid artifacts field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: artifacts must be a mapping when present')
    unknown = set(value) - _ARTIFACTS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: artifacts: unknown keys: {sorted(unknown)}')
    _check_optional_bool(
        value.get('instrument_defaults'), 'artifacts.instrument_defaults', source=source
    )


def _check_spk_error(value: Any, *, source: str) -> None:
    """Validate the scene-level ``spk_error`` block's field types."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: spk_error must be a mapping when present')
    unknown = set(value) - _SPK_ERROR_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: spk_error: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('dv_px'), 'spk_error.dv_px', source=source)
    _check_optional_number(value.get('du_px'), 'spk_error.du_px', source=source)
    _check_optional_positive_number(
        value.get('reference_range_km'), 'spk_error.reference_range_km', source=source
    )


def _require_ranges_for_spk_error(sim_params: dict[str, Any], *, source: str) -> None:
    """Every body and ring feature needs a physical ``range_km`` under spk_error.

    The parallax displacement scales as ``reference_range_km / range_km``, so a
    scene that plants spacecraft-ephemeris error must give the renderer a
    physical range for each object it displaces.

    Parameters:
        sim_params: The scene mapping (already type-checked).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If any body or ring lacks ``range_km``.
    """
    for block in ('bodies', 'rings'):
        for index, obj in enumerate(sim_params.get(block) or []):
            if obj.get('range_km') is None:
                raise SimSceneValidationError(
                    f'{source}: {block}[{index}] needs range_km when spk_error is present'
                )


def _require_str(raw: dict[str, Any], key: str, *, source: str) -> str:
    """Return ``raw[key]`` as a non-empty string, or fail validation."""
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SimSceneValidationError(f'{source}: {key} must be a non-empty string')
    return value


def _require_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    """Return ``raw[key]`` as an integer (bools rejected), or fail validation."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimSceneValidationError(f'{source}: {key} must be an integer')
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    """Return ``raw[key]`` as a positive integer, or fail validation."""
    value = _require_int(raw, key, source=source)
    if value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer')
    return value


def _check_optional_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a finite number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimSceneValidationError(f'{source}: {key} must be a number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{source}: {key} must be finite; got {value!r}')


def _check_optional_positive_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a positive number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive number when present')


def _check_optional_nonnegative_int(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a non-negative integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimSceneValidationError(
            f'{source}: {key} must be a non-negative integer when present'
        )


def _check_optional_positive_int(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a positive integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer when present')


def _check_optional_nonnegative_number(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a non-negative number."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SimSceneValidationError(f'{source}: {key} must be a non-negative number when present')


def _check_optional_str(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a string."""
    if value is None:
        return
    if not isinstance(value, str):
        raise SimSceneValidationError(f'{source}: {key} must be a string when present')


def _check_optional_bool(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a boolean."""
    if value is None:
        return
    if not isinstance(value, bool):
        raise SimSceneValidationError(f'{source}: {key} must be a boolean when present')


def _check_optional_mapping(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a mapping."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {key} must be a mapping when present')


def _check_optional_mapping_list(value: Any, key: str, *, source: str) -> None:
    """Fail validation unless ``value`` is None or a list of mappings."""
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimSceneValidationError(f'{source}: {key} must be a list of mappings')
