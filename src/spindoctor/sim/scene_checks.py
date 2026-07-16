"""Field-type validators for the sim-scene schema.

The ``_check_*`` / ``_require_*`` helpers here enforce the per-field types of
every scene block: the per-object rings / stars entries, the optics
sub-blocks (PSF, smear, distortion, ghosts, stray light), the noise, detector,
artifacts, and spk_error blocks, plus the primitive scalar checks they all
share.  Each block's key inventory lives beside its checker (unknown keys fail
validation, so a typo cannot silently render an un-blurred or clean frame).
The ``bodies``-entry checkers, the schema's largest block, live in the sibling
:mod:`spindoctor.sim.scene_checks_body` and build on the same primitives.

:func:`spindoctor.sim.scene.validate_sim_params` drives these helpers; every
violation raises :class:`spindoctor.sim.scene_schema.SimSceneValidationError`.
"""

from __future__ import annotations

import math
from typing import Any

from spindoctor.sim.forward.artifact_modes import (
    ARTIFACT_MODES,
    MODE_KEYS,
    ModeParam,
    mode_available,
    mode_unavailable_message,
)
from spindoctor.sim.scene_schema import SimSceneValidationError
from spindoctor.support.status_reason import NavStatusReason

# The scene-level ``expected`` block's allowed vocabularies.  These mirror the
# image-library sidecar taxonomy (status / confidence tier / status_reason) but
# are defined independently here: a sim scene is not a sidecar, so the sim path
# owns its own copy rather than importing the sidecar module.  ``confidence_tier``
# admits the five navigation ranks plus null (assert the status only).
_EXPECTED_KEYS: frozenset[str] = frozenset({'status', 'status_reason', 'confidence_tier'})
_EXPECTED_STATUSES: frozenset[str] = frozenset({'success', 'failed', 'conflicted'})
_EXPECTED_TIERS: frozenset[str] = frozenset({'high', 'medium', 'low', 'failed', 'conflicted'})
_EXPECTED_STATUS_REASONS: frozenset[str] = frozenset(reason.value for reason in NavStatusReason)


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
    for key in ('center_v', 'center_u', 'shading_distance', 'range_km'):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    for key in ('inner_data', 'outer_data'):
        _check_optional_mapping_list(obj.get(key), f'{label}.{key}', source=source)


# The ring_system block: shared projection geometry plus a list of radial
# optical-depth features.  Key inventories mirror the boundary classification
# in scene_schema (which asserts completeness over them at import).
_RING_SYSTEM_BLOCK_KEYS: frozenset[str] = frozenset(
    {'geometry', 'features', 'range_km', 'km_per_pixel', 'phase_deg', 'azimuthal', 'moonlets'}
)
_RING_SYSTEM_GEOMETRY_KEYS: frozenset[str] = frozenset(
    {'center_v', 'center_u', 'opening_deg_obs', 'opening_deg_sun', 'node_deg'}
)
_RING_FEATURE_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'kind',
        'width',
        'tau',
        'orbit',
        'side',
        'wavelength',
        'damping',
        'navigable',
        'declared_orbit_sigma',
        'orbit_error',
        'albedo',
        'phase_g',
    }
)
_RING_FEATURE_KINDS: frozenset[str] = frozenset({'ringlet', 'gap', 'edge', 'ramp', 'wave'})
_RING_FEATURE_ORBIT_KEYS: frozenset[str] = frozenset(
    {'a', 'ae', 'long_peri', 'rate_peri', 'modes', 'edge_wave'}
)
_RING_ORBIT_MODE_KEYS: frozenset[str] = frozenset({'m', 'amp', 'peri'})
_RING_EDGE_WAVE_KEYS: frozenset[str] = frozenset({'amp', 'wavelength', 'damp', 'lam0'})
# The planted per-feature ephemeris error (truth: render side only) and the
# uncertainty the navigator is entitled to know (idealized error bars).
_RING_ORBIT_ERROR_KEYS: frozenset[str] = frozenset(
    {'delta_a_px', 'delta_ae_px', 'delta_long_peri_deg'}
)
_RING_ORBIT_SIGMA_KEYS: frozenset[str] = frozenset(
    {'sigma_a_px', 'sigma_ae_px', 'sigma_long_peri_deg'}
)
# Which kinds take which shape keys: a stray key on a kind that ignores it
# would silently author a different feature than intended, so it fails.
_RING_KINDS_WITH_WIDTH: frozenset[str] = frozenset({'ringlet', 'gap', 'ramp'})
_RING_KINDS_WITH_SIDE: frozenset[str] = frozenset({'edge', 'ramp'})
_RING_FEATURE_SIDES: frozenset[str] = frozenset({'in', 'out'})
# Truth-side azimuthal structure (intensity only, never tau) and embedded
# moonlets (opaque discs at ring depth, with optional propeller lobes).
_RING_AZIMUTHAL_KEYS: frozenset[str] = frozenset({'modulation', 'spokes', 'shadow'})
_RING_MODULATION_KEYS: frozenset[str] = frozenset({'amplitude', 'm', 'phase_deg'})
_RING_SHADOW_KEYS: frozenset[str] = frozenset({'start_deg', 'extent_deg', 'darkness'})
_RING_SPOKES_KEYS: frozenset[str] = frozenset(
    {'count', 'r_inner', 'r_outer', 'contrast', 'width_deg'}
)
_RING_MOONLET_KEYS: frozenset[str] = frozenset(
    {'a', 'lam_deg', 'radius_px', 'amplitude', 'propeller'}
)
_RING_PROPELLER_KEYS: frozenset[str] = frozenset({'length_deg', 'width_px', 'contrast'})


def _check_ring_system(value: Any, *, source: str) -> None:
    """Validate the scene-level ``ring_system`` block.

    The block carries the shared projection geometry (opening angles, node,
    center), the system's physical range and pixel scale, the phase angle,
    and the list of radial optical-depth features.  The geometry block and
    both opening angles are required: an unstated opening angle has no
    sensible default (0 renders nothing and 90 silently degenerates to
    sky-plane circles).

    Parameters:
        value: The ``ring_system`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: ring_system must be a mapping when present')
    unknown = set(value) - _RING_SYSTEM_BLOCK_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: ring_system: unknown keys: {sorted(unknown)}')
    geometry = value.get('geometry')
    if not isinstance(geometry, dict):
        raise SimSceneValidationError(f'{source}: ring_system.geometry is required (a mapping)')
    unknown = set(geometry) - _RING_SYSTEM_GEOMETRY_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: ring_system.geometry: unknown keys: {sorted(unknown)}'
        )
    for key in ('center_v', 'center_u', 'node_deg'):
        _check_optional_number(geometry.get(key), f'ring_system.geometry.{key}', source=source)
    for key in ('opening_deg_obs', 'opening_deg_sun'):
        opening = geometry.get(key)
        label = f'ring_system.geometry.{key}'
        if opening is None:
            raise SimSceneValidationError(f'{source}: {label} is required')
        _check_optional_number(opening, label, source=source)
        if not -90.0 < float(opening) <= 90.0:
            raise SimSceneValidationError(
                f'{source}: {label} must lie in (-90, 90]; got {opening!r}'
            )
    _check_optional_positive_number(value.get('range_km'), 'ring_system.range_km', source=source)
    _check_optional_positive_number(
        value.get('km_per_pixel'), 'ring_system.km_per_pixel', source=source
    )
    phase = value.get('phase_deg')
    _check_optional_number(phase, 'ring_system.phase_deg', source=source)
    if phase is not None and not 0.0 <= float(phase) <= 180.0:
        raise SimSceneValidationError(
            f'{source}: ring_system.phase_deg must lie in [0, 180]; got {phase!r}'
        )
    features = value.get('features')
    _check_optional_mapping_list(features, 'ring_system.features', source=source)
    for index, feature in enumerate(features or []):
        _check_ring_feature(feature, index=index, source=source)
    _check_ring_azimuthal(value.get('azimuthal'), source=source)
    moonlets = value.get('moonlets')
    _check_optional_mapping_list(moonlets, 'ring_system.moonlets', source=source)
    for index, moonlet in enumerate(moonlets or []):
        _check_ring_moonlet(moonlet, index=index, source=source)


def _check_ring_azimuthal(value: Any, *, source: str) -> None:
    """Validate the truth-side ``ring_system.azimuthal`` structure block.

    ``modulation`` is a low-frequency brightness modulation
    (self-gravity-wake asymmetry), ``shadow`` a planet-shadow darkening
    wedge, and ``spokes`` a seeded field of azimuthally sharp, radially
    broad albedo wedges.  All three modulate the emitted intensity only --
    they are albedo/illumination structure, not optical depth.

    Parameters:
        value: The ``azimuthal`` mapping, or None when absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: ring_system.azimuthal must be a mapping when present'
        )
    unknown = set(value) - _RING_AZIMUTHAL_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: ring_system.azimuthal: unknown keys: {sorted(unknown)}'
        )
    modulation = value.get('modulation')
    if modulation is not None:
        label = 'ring_system.azimuthal.modulation'
        if not isinstance(modulation, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(modulation) - _RING_MODULATION_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        if modulation.get('amplitude') is None:
            raise SimSceneValidationError(f'{source}: {label}.amplitude is required')
        _check_optional_nonnegative_number(
            modulation.get('amplitude'), f'{label}.amplitude', source=source
        )
        m = modulation.get('m')
        if m is not None and (isinstance(m, bool) or not isinstance(m, int) or m < 1):
            raise SimSceneValidationError(
                f'{source}: {label}.m must be an integer >= 1 when present; got {m!r}'
            )
        _check_optional_number(modulation.get('phase_deg'), f'{label}.phase_deg', source=source)
    shadow = value.get('shadow')
    if shadow is not None:
        label = 'ring_system.azimuthal.shadow'
        if not isinstance(shadow, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(shadow) - _RING_SHADOW_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        if shadow.get('extent_deg') is None:
            raise SimSceneValidationError(f'{source}: {label}.extent_deg is required')
        _check_optional_positive_number(
            shadow.get('extent_deg'), f'{label}.extent_deg', source=source
        )
        _check_optional_number(shadow.get('start_deg'), f'{label}.start_deg', source=source)
        darkness = shadow.get('darkness')
        if darkness is None:
            raise SimSceneValidationError(f'{source}: {label}.darkness is required')
        _check_optional_number(darkness, f'{label}.darkness', source=source)
        if not 0.0 <= float(darkness) <= 1.0:
            raise SimSceneValidationError(
                f'{source}: {label}.darkness must lie in [0, 1]; got {darkness!r}'
            )
    spokes = value.get('spokes')
    if spokes is not None:
        label = 'ring_system.azimuthal.spokes'
        if not isinstance(spokes, dict):
            raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
        unknown = set(spokes) - _RING_SPOKES_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        count = spokes.get('count')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SimSceneValidationError(
                f'{source}: {label}.count must be an integer >= 1; got {count!r}'
            )
        for key in ('r_inner', 'r_outer', 'width_deg'):
            if spokes.get(key) is None:
                raise SimSceneValidationError(f'{source}: {label}.{key} is required')
            _check_optional_positive_number(spokes.get(key), f'{label}.{key}', source=source)
        if float(spokes['r_outer']) <= float(spokes['r_inner']):
            raise SimSceneValidationError(
                f'{source}: {label}.r_outer must exceed r_inner; got '
                f'{spokes["r_inner"]!r}..{spokes["r_outer"]!r}'
            )
        if spokes.get('contrast') is None:
            raise SimSceneValidationError(f'{source}: {label}.contrast is required')
        _check_optional_number(spokes.get('contrast'), f'{label}.contrast', source=source)


def _check_ring_moonlet(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``ring_system.moonlets`` entry.

    A moonlet is an opaque disc embedded in the ring plane at polar
    placement ``(a, lam_deg)``, emitting ``amplitude`` in normalized signal
    units, optionally with a stylized propeller disturbance (two tau lobes
    straddling it radially and azimuthally).

    Parameters:
        obj: The moonlet mapping.
        index: The moonlet's index in the list (for error messages).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    label = f'ring_system.moonlets[{index}]'
    unknown = set(obj) - _RING_MOONLET_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    if obj.get('a') is None:
        raise SimSceneValidationError(f'{source}: {label}.a is required')
    _check_optional_positive_number(obj.get('a'), f'{label}.a', source=source)
    if obj.get('lam_deg') is None:
        raise SimSceneValidationError(f'{source}: {label}.lam_deg is required')
    _check_optional_number(obj.get('lam_deg'), f'{label}.lam_deg', source=source)
    if obj.get('amplitude') is None:
        raise SimSceneValidationError(f'{source}: {label}.amplitude is required')
    _check_optional_nonnegative_number(obj.get('amplitude'), f'{label}.amplitude', source=source)
    _check_optional_positive_number(obj.get('radius_px'), f'{label}.radius_px', source=source)
    propeller = obj.get('propeller')
    if propeller is not None:
        prop_label = f'{label}.propeller'
        if not isinstance(propeller, dict):
            raise SimSceneValidationError(f'{source}: {prop_label} must be a mapping when present')
        unknown = set(propeller) - _RING_PROPELLER_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {prop_label}: unknown keys: {sorted(unknown)}'
            )
        for key in ('length_deg', 'width_px'):
            if propeller.get(key) is None:
                raise SimSceneValidationError(f'{source}: {prop_label}.{key} is required')
            _check_optional_positive_number(
                propeller.get(key), f'{prop_label}.{key}', source=source
            )
        if propeller.get('contrast') is None:
            raise SimSceneValidationError(f'{source}: {prop_label}.contrast is required')
        _check_optional_number(propeller.get('contrast'), f'{prop_label}.contrast', source=source)


def _check_ring_feature(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``ring_system.features`` entry.

    The shape keys are kind-specific: ``width`` is required by the banded
    kinds (ringlet / gap / ramp) and rejected elsewhere, ``side`` belongs to
    the one-sided kinds (edge / ramp), and ``wavelength`` / ``damping`` are
    the ``wave`` kind's radial train parameters.  A stray shape key on a
    kind that ignores it fails loudly rather than silently authoring a
    different feature.

    Parameters:
        obj: The feature mapping.
        index: The feature's index in the list (for error messages).
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    label = f'ring_system.features[{index}]'
    unknown = set(obj) - _RING_FEATURE_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    kind = obj.get('kind')
    if kind not in _RING_FEATURE_KINDS:
        raise SimSceneValidationError(
            f'{source}: {label}.kind must be one of {sorted(_RING_FEATURE_KINDS)}; got {kind!r}'
        )
    tau = obj.get('tau')
    if tau is None:
        raise SimSceneValidationError(f'{source}: {label}.tau is required')
    _check_optional_nonnegative_number(tau, f'{label}.tau', source=source)
    width = obj.get('width')
    if kind in _RING_KINDS_WITH_WIDTH:
        if width is None:
            raise SimSceneValidationError(f'{source}: {label}.width is required for kind {kind!r}')
        _check_optional_positive_number(width, f'{label}.width', source=source)
    elif width is not None:
        raise SimSceneValidationError(f'{source}: {label}.width is not allowed for kind {kind!r}')
    side = obj.get('side')
    if side is not None and kind not in _RING_KINDS_WITH_SIDE:
        raise SimSceneValidationError(f'{source}: {label}.side is not allowed for kind {kind!r}')
    if side is not None and side not in _RING_FEATURE_SIDES:
        raise SimSceneValidationError(f"{source}: {label}.side must be 'in' or 'out'; got {side!r}")
    for key in ('wavelength', 'damping'):
        value = obj.get(key)
        if kind == 'wave':
            if value is None:
                raise SimSceneValidationError(
                    f"{source}: {label}.{key} is required for kind 'wave'"
                )
            _check_optional_positive_number(value, f'{label}.{key}', source=source)
        elif value is not None:
            raise SimSceneValidationError(
                f'{source}: {label}.{key} is not allowed for kind {kind!r}'
            )
    _check_ring_feature_orbit(obj.get('orbit'), label=label, source=source)
    _check_optional_bool(obj.get('navigable'), f'{label}.navigable', source=source)
    orbit_error = obj.get('orbit_error')
    if orbit_error is not None:
        error_label = f'{label}.orbit_error'
        if not isinstance(orbit_error, dict):
            raise SimSceneValidationError(f'{source}: {error_label} must be a mapping when present')
        unknown = set(orbit_error) - _RING_ORBIT_ERROR_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {error_label}: unknown keys: {sorted(unknown)}'
            )
        for key in _RING_ORBIT_ERROR_KEYS:
            _check_optional_number(orbit_error.get(key), f'{error_label}.{key}', source=source)
    sigma = obj.get('declared_orbit_sigma')
    if sigma is not None:
        sigma_label = f'{label}.declared_orbit_sigma'
        if not isinstance(sigma, dict):
            raise SimSceneValidationError(f'{source}: {sigma_label} must be a mapping when present')
        unknown = set(sigma) - _RING_ORBIT_SIGMA_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {sigma_label}: unknown keys: {sorted(unknown)}'
            )
        for key in _RING_ORBIT_SIGMA_KEYS:
            _check_optional_nonnegative_number(
                sigma.get(key), f'{sigma_label}.{key}', source=source
            )
    _check_optional_nonnegative_number(obj.get('albedo'), f'{label}.albedo', source=source)
    phase_g = obj.get('phase_g')
    _check_optional_number(phase_g, f'{label}.phase_g', source=source)
    if phase_g is not None and not -1.0 < float(phase_g) < 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.phase_g must lie in (-1, 1); got {phase_g!r}'
        )


def _check_ring_feature_orbit(orbit: Any, *, label: str, source: str) -> None:
    """Validate one feature's ``orbit`` mapping (mode 1 + modes + edge wave).

    Every orbital angle (``long_peri``, mode ``peri``, edge-wave ``lam0``)
    is a ring-plane longitude in degrees measured from the ascending node;
    the edge-wave ``damp`` is in RADIANS of downstream longitude, and
    ``amp`` / ``wavelength`` are radial / arc-length pixel quantities.

    Parameters:
        orbit: The ``orbit`` value (must be a mapping).
        label: The feature label for error messages.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, missing, or invalid field.
    """
    if not isinstance(orbit, dict):
        raise SimSceneValidationError(f'{source}: {label}.orbit is required (a mapping)')
    unknown = set(orbit) - _RING_FEATURE_ORBIT_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}.orbit: unknown keys: {sorted(unknown)}')
    if orbit.get('a') is None:
        raise SimSceneValidationError(f'{source}: {label}.orbit.a is required')
    _check_optional_positive_number(orbit.get('a'), f'{label}.orbit.a', source=source)
    _check_optional_nonnegative_number(orbit.get('ae'), f'{label}.orbit.ae', source=source)
    _check_optional_number(orbit.get('long_peri'), f'{label}.orbit.long_peri', source=source)
    _check_optional_number(orbit.get('rate_peri'), f'{label}.orbit.rate_peri', source=source)
    modes = orbit.get('modes')
    _check_optional_mapping_list(modes, f'{label}.orbit.modes', source=source)
    for mode_index, mode in enumerate(modes or []):
        mode_label = f'{label}.orbit.modes[{mode_index}]'
        unknown = set(mode) - _RING_ORBIT_MODE_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {mode_label}: unknown keys: {sorted(unknown)}'
            )
        m = mode.get('m')
        if isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise SimSceneValidationError(
                f'{source}: {mode_label}.m must be an integer >= 2; got {m!r}'
            )
        _check_optional_nonnegative_number(mode.get('amp'), f'{mode_label}.amp', source=source)
        _check_optional_number(mode.get('peri'), f'{mode_label}.peri', source=source)
    wave = orbit.get('edge_wave')
    if wave is not None:
        wave_label = f'{label}.orbit.edge_wave'
        if not isinstance(wave, dict):
            raise SimSceneValidationError(f'{source}: {wave_label} must be a mapping when present')
        unknown = set(wave) - _RING_EDGE_WAVE_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {wave_label}: unknown keys: {sorted(unknown)}'
            )
        _check_optional_nonnegative_number(wave.get('amp'), f'{wave_label}.amp', source=source)
        for key in ('wavelength', 'damp'):
            value = wave.get(key)
            if value is None:
                raise SimSceneValidationError(f'{source}: {wave_label}.{key} is required')
            _check_optional_positive_number(value, f'{wave_label}.{key}', source=source)
        _check_optional_number(wave.get('lam0'), f'{wave_label}.lam0', source=source)


def _check_star_object(obj: dict[str, Any], *, index: int, source: str) -> None:
    """Validate one ``stars`` entry's field types."""
    label = f'stars[{index}]'
    _check_optional_str(obj.get('name'), f'{label}.name', source=source)
    _check_optional_str(obj.get('catalog_name'), f'{label}.catalog_name', source=source)
    _check_optional_str(obj.get('spectral_class'), f'{label}.spectral_class', source=source)
    for key in (
        'v',
        'u',
        'vmag',
        'move_v',
        'move_u',
        'catalog_error_v',
        'catalog_error_u',
        'delta_mag',
    ):
        _check_optional_number(obj.get(key), f'{label}.{key}', source=source)
    _check_optional_positive_number(obj.get('psf_sigma'), f'{label}.psf_sigma', source=source)
    _check_optional_bool(obj.get('navigable'), f'{label}.navigable', source=source)
    _check_star_companion(obj.get('companion'), label=label, source=source)
    psf_size = obj.get('psf_size')
    if psf_size is not None:
        valid = isinstance(psf_size, (list, tuple)) and len(psf_size) == 2
        if not valid or any(isinstance(x, bool) or not isinstance(x, int) for x in psf_size):
            raise SimSceneValidationError(
                f'{source}: {label}.psf_size must be a list of 2 integers when present'
            )


# An unresolved binary: a second point source at ``sep_px`` from the primary
# along ``angle_deg``, ``delta_mag`` fainter.  Its photocenter pull off the
# catalog position is a physical catalog error the navigator cannot know.
_COMPANION_KEYS: frozenset[str] = frozenset({'sep_px', 'delta_mag', 'angle_deg'})


def _check_star_companion(value: Any, *, label: str, source: str) -> None:
    """Validate one star's ``companion`` map (unresolved-binary parameters)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {label}.companion must be a mapping when present')
    unknown = set(value) - _COMPANION_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.companion: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('sep_px'), f'{label}.companion.sep_px', source=source
    )
    _check_optional_number(value.get('delta_mag'), f'{label}.companion.delta_mag', source=source)
    _check_optional_number(value.get('angle_deg'), f'{label}.companion.angle_deg', source=source)


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
_QUANTIZATION_MODES: frozenset[str] = frozenset(
    {'exact', 'uneven_12bit', '8bit', 'sqrt_lut', 'ls8b', 'contour_8bit'}
)
# The artifacts block: the two switches (the physical-chain opt-in and the
# adversarial-placement flag) plus one map per artifact mode, keyed by exactly
# the mode-key registry.  Unknown keys fail.
_ARTIFACTS_SWITCH_KEYS: frozenset[str] = frozenset({'instrument_defaults', 'adversarial'})
_ARTIFACTS_KEYS: frozenset[str] = _ARTIFACTS_SWITCH_KEYS | MODE_KEYS


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


def _check_artifacts(value: Any, *, instrument: str, source: str) -> None:
    """Validate the scene-level ``artifacts`` block against the mode registry.

    Beyond the two switch keys, the block is keyed by exactly the artifact-mode
    registry (unknown keys fail).  A mode that has no implementation yet fails
    as not-yet-implemented, and a mode unavailable on the scene's instrument
    fails with the registry's message (the LORRI hot-pixel case carries a
    bespoke one).  Each present mode's parameters are then type-checked against
    its schema.

    Parameters:
        value: The ``artifacts`` mapping, or None when the block is absent.
        instrument: The scene's (already validated) instrument name, used to
            check per-mode availability.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown, unimplemented, unavailable, or
            mistyped artifacts field.
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
    _check_optional_bool(value.get('adversarial'), 'artifacts.adversarial', source=source)
    for mode_name in set(value) & MODE_KEYS:
        _check_artifact_mode(mode_name, value[mode_name], instrument=instrument, source=source)


def _check_artifact_mode(mode_name: str, config: Any, *, instrument: str, source: str) -> None:
    """Validate one artifact-mode map: availability, implementation, and params."""
    label = f'artifacts.{mode_name}'
    if not isinstance(config, dict):
        raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
    mode = ARTIFACT_MODES[mode_name]
    if not mode.implemented:
        raise SimSceneValidationError(
            f'{source}: artifact mode {mode_name!r} is not yet implemented'
        )
    if not mode_available(mode_name, instrument):
        raise SimSceneValidationError(
            f'{source}: {mode_unavailable_message(mode_name, instrument)}'
        )
    param_map = mode.param_map
    unknown = set(config) - set(param_map)
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    for name, param in param_map.items():
        if name in config:
            _check_mode_param(config[name], param, key=f'{label}.{name}', source=source)


def _check_mode_param(value: Any, param: ModeParam, *, key: str, source: str) -> None:
    """Type-check one artifact-mode parameter value against its schema kind."""
    kind = param.kind
    if kind == 'bool':
        _check_optional_bool(value, key, source=source)
    elif kind == 'nonneg_number':
        _check_optional_nonnegative_number(value, key, source=source)
    elif kind == 'unit_interval':
        _check_optional_nonnegative_number(value, key, source=source)
        if value is not None and float(value) > 1.0:
            raise SimSceneValidationError(f'{source}: {key} must lie in [0, 1]; got {value!r}')
    elif kind == 'int':
        _check_optional_number(value, key, source=source)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise SimSceneValidationError(f'{source}: {key} must be an integer when present')
    elif kind == 'nonneg_int':
        _check_optional_nonnegative_int(value, key, source=source)
    elif kind == 'positive_int':
        _check_optional_positive_int(value, key, source=source)
    elif kind == 'enum':
        if param.choices is not None and value not in param.choices:
            raise SimSceneValidationError(
                f'{source}: {key} must be one of {list(param.choices)}; got {value!r}'
            )
    elif kind == 'int_list':
        _check_int_list(value, param.length, key=key, source=source)
    else:  # pragma: no cover - guards a registry authoring mistake
        raise SimSceneValidationError(f'{source}: {key} has unknown parameter kind {kind!r}')


def _check_int_list(value: Any, length: int | None, *, key: str, source: str) -> None:
    """Fail validation unless ``value`` is a list of ``length`` integers."""
    if value is None:
        return
    valid = isinstance(value, (list, tuple)) and (length is None or len(value) == length)
    if not valid or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise SimSceneValidationError(
            f'{source}: {key} must be a list of {length} integers when present'
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


# The background-sky star-count law: cumulative log10 N(<m) = a + b*m per square
# degree, a local-density multiplier, and an optional flat diffuse-sky floor.
# diffuse_e_per_px is detector-native despite the '_e_' in its name: electrons
# per pixel on a CCD, DN per pixel on the Voyager vidicon (which has no electron
# domain), matching the unit domain of the point-source plane it adds to.
_SKY_COUNTS_KEYS: frozenset[str] = frozenset({'a', 'b', 'density_factor', 'diffuse_e_per_px'})


def _check_sky_counts(value: Any, *, source: str) -> None:
    """Validate the scene-level ``sky_counts`` block's field types.

    Parameters:
        value: The ``sky_counts`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid sky_counts field.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: sky_counts must be a mapping when present')
    unknown = set(value) - _SKY_COUNTS_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: sky_counts: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('a'), 'sky_counts.a', source=source)
    _check_optional_number(value.get('b'), 'sky_counts.b', source=source)
    _check_optional_nonnegative_number(
        value.get('density_factor'), 'sky_counts.density_factor', source=source
    )
    _check_optional_nonnegative_number(
        value.get('diffuse_e_per_px'), 'sky_counts.diffuse_e_per_px', source=source
    )


def _check_star_catalog_scatter(value: Any, *, source: str) -> None:
    """Validate the scene-level ``star_catalog_scatter_px`` sigma.

    A non-negative per-star position-scatter sigma (detector pixels): every
    catalog star's RENDERED position is displaced by a seeded Gaussian draw of
    this sigma, added to any explicit per-star ``catalog_error_*``.

    Parameters:
        value: The ``star_catalog_scatter_px`` value, or None when absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: If present and not a non-negative number.
    """
    _check_optional_nonnegative_number(value, 'star_catalog_scatter_px', source=source)


def _check_expected(value: Any, *, source: str) -> None:
    """Validate the scene-level ``expected`` block (the expected outcome).

    The block declares what the navigator should produce for the scene, read by
    the sim integration suite's assertion machinery (not by the renderer or the
    navigator).  ``status`` is required; ``confidence_tier`` is required but may
    be null (assert the status only); ``status_reason`` is optional.  The
    cross-field rules mirror the image-library sidecar taxonomy: a ``failed`` or
    ``conflicted`` status pins the matching tier, and those tiers require the
    matching status, but only when the tier is asserted (non-null).

    Parameters:
        value: The ``expected`` mapping, or None when the block is absent.
        source: Label used in error messages.

    Raises:
        SimSceneValidationError: On any unknown or invalid expected field, or a
            status / confidence_tier inconsistency.
    """
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: expected must be a mapping when present')
    unknown = set(value) - _EXPECTED_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: expected: unknown keys: {sorted(unknown)}')
    status = value.get('status')
    if status not in _EXPECTED_STATUSES:
        raise SimSceneValidationError(
            f'{source}: expected.status must be one of {sorted(_EXPECTED_STATUSES)}; got {status!r}'
        )
    tier = value.get('confidence_tier')
    if tier is not None and tier not in _EXPECTED_TIERS:
        raise SimSceneValidationError(
            f'{source}: expected.confidence_tier must be one of {sorted(_EXPECTED_TIERS)} or '
            f'null; got {tier!r}'
        )
    if tier is not None:
        if status == 'failed' and tier != 'failed':
            raise SimSceneValidationError(
                f'{source}: expected.status=failed requires expected.confidence_tier=failed'
            )
        if status != 'failed' and tier == 'failed':
            raise SimSceneValidationError(
                f'{source}: expected.confidence_tier=failed requires expected.status=failed'
            )
        if status == 'conflicted' and tier != 'conflicted':
            raise SimSceneValidationError(
                f'{source}: expected.status=conflicted requires expected.confidence_tier=conflicted'
            )
        if status != 'conflicted' and tier == 'conflicted':
            raise SimSceneValidationError(
                f'{source}: expected.confidence_tier=conflicted requires expected.status=conflicted'
            )
    reason = value.get('status_reason')
    if reason is not None and reason not in _EXPECTED_STATUS_REASONS:
        raise SimSceneValidationError(
            f'{source}: expected.status_reason must be one of '
            f'{sorted(_EXPECTED_STATUS_REASONS)} when present; got {reason!r}'
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
    ring_system = sim_params.get('ring_system')
    if isinstance(ring_system, dict) and ring_system.get('range_km') is None:
        raise SimSceneValidationError(
            f'{source}: ring_system needs range_km when spk_error is present'
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
