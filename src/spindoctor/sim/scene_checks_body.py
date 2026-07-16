"""Body-entry field validators for the sim-scene schema.

One ``bodies`` entry carries the largest key surface in the schema: the
idealized geometry, the mesh and crater shape knobs, and the body-appearance
truth blocks (pose scatter, limb relief and photometry, albedo texture, disc
texture, transits).  The checkers for that entry live here, beside their key
inventories; the scalar primitives they share with the other block checkers
stay in :mod:`spindoctor.sim.scene_checks`, which also documents the overall
validation contract (unknown keys fail, every violation raises
:class:`spindoctor.sim.scene_schema.SimSceneValidationError`).

:func:`spindoctor.sim.scene.validate_sim_params` drives ``_check_body_object``
once per ``bodies`` entry.
"""

from __future__ import annotations

from typing import Any

from spindoctor.sim.forward.photometry import PHOTOMETRIC_LAWS
from spindoctor.sim.scene_checks import (
    _check_optional_mapping_list,
    _check_optional_nonnegative_int,
    _check_optional_nonnegative_number,
    _check_optional_number,
    _check_optional_positive_number,
    _check_optional_str,
    _require_int,
)
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
    _check_optional_nonnegative_int(
        obj.get('mesh_detail_octaves'), f'{label}.mesh_detail_octaves', source=source
    )
    shading = obj.get('shading')
    if shading is not None and shading not in ('flat', 'gouraud'):
        raise SimSceneValidationError(
            f"{source}: {label}.shading must be 'flat' or 'gouraud' when present; got {shading!r}"
        )
    _check_pose_scatter(obj.get('pose_scatter'), label=label, source=source)
    _check_body_relief_and_photometry(obj, label=label, source=source)
    _check_albedo_texture(obj.get('albedo_texture'), label=label, source=source)
    _check_disc_texture(obj.get('disc_texture'), label=label, source=source)
    _check_transits(obj.get('transits'), label=label, source=source)
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


# The opposition-surge map's key inventory (a simple normalized exponential
# surge: amplitude plus angular e-folding width in degrees of phase).  The
# photometric-law vocabulary is the renderer's own PHOTOMETRIC_LAWS.
_OPPOSITION_SURGE_KEYS: frozenset[str] = frozenset({'amplitude', 'width_deg'})

# The per-frame pose scatter: a seeded Gaussian perturbation (sigma per
# Euler axis, degrees) added to the RENDERED mesh pose only; the navigator
# predicts the catalog pose.
_POSE_SCATTER_KEYS: frozenset[str] = frozenset({'sigma_deg'})


def _check_pose_scatter(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``pose_scatter`` map (per-frame pose perturbation)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.pose_scatter must be a mapping when present'
        )
    unknown = set(value) - _POSE_SCATTER_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.pose_scatter: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('sigma_deg'), f'{label}.pose_scatter.sigma_deg', source=source
    )


def _check_body_relief_and_photometry(obj: dict[str, Any], *, label: str, source: str) -> None:
    """Validate one body's limb-relief and photometric truth keys."""
    _check_optional_nonnegative_number(
        obj.get('limb_relief_rms'), f'{label}.limb_relief_rms', source=source
    )
    _check_optional_positive_number(
        obj.get('limb_relief_corr_deg'), f'{label}.limb_relief_corr_deg', source=source
    )
    law = obj.get('photometric_law')
    if law is not None and law not in PHOTOMETRIC_LAWS:
        raise SimSceneValidationError(
            f'{source}: {label}.photometric_law must be one of '
            f'{sorted(PHOTOMETRIC_LAWS)}; got {law!r}'
        )
    _check_optional_positive_number(obj.get('minnaert_k'), f'{label}.minnaert_k', source=source)
    surge = obj.get('opposition_surge')
    if surge is None:
        return
    if not isinstance(surge, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.opposition_surge must be a mapping when present'
        )
    unknown = set(surge) - _OPPOSITION_SURGE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.opposition_surge: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        surge.get('amplitude'), f'{label}.opposition_surge.amplitude', source=source
    )
    _check_optional_positive_number(
        surge.get('width_deg'), f'{label}.opposition_surge.width_deg', source=source
    )


# The multiplicative albedo texture: a band-limited noise field (rms +
# correlation length in detector pixels on the disc) plus discrete circular
# spots in the body-polar surface frame (pole along axis1; longitude 90 deg
# is the sub-observer meridian).
_ALBEDO_TEXTURE_KEYS: frozenset[str] = frozenset({'rms', 'corr_px', 'spots'})
_SURFACE_SPOT_KEYS: frozenset[str] = frozenset(
    {'lat_deg', 'lon_deg', 'radius_deg', 'albedo_factor'}
)


def _check_surface_spot_list(value: Any, *, key: str, source: str) -> None:
    """Validate one spots/storms list (circular multiplicative albedo marks)."""
    if value is None:
        return
    _check_optional_mapping_list(value, key, source=source)
    for index, spot in enumerate(value):
        label = f'{key}[{index}]'
        unknown = set(spot) - _SURFACE_SPOT_KEYS
        if unknown:
            raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
        _check_optional_number(spot.get('lat_deg'), f'{label}.lat_deg', source=source)
        _check_optional_number(spot.get('lon_deg'), f'{label}.lon_deg', source=source)
        _check_optional_positive_number(
            spot.get('radius_deg'), f'{label}.radius_deg', source=source
        )
        _check_optional_nonnegative_number(
            spot.get('albedo_factor'), f'{label}.albedo_factor', source=source
        )


def _check_albedo_texture(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``albedo_texture`` map (noise field + spots)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.albedo_texture must be a mapping when present'
        )
    unknown = set(value) - _ALBEDO_TEXTURE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.albedo_texture: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('rms'), f'{label}.albedo_texture.rms', source=source
    )
    _check_optional_positive_number(
        value.get('corr_px'), f'{label}.albedo_texture.corr_px', source=source
    )
    _check_surface_spot_list(value.get('spots'), key=f'{label}.albedo_texture.spots', source=source)


# The giant-planet disc texture: a low-frequency latitude-banded
# multiplicative pattern (cosine in body-polar latitude) plus discrete
# storm ovals in the body-polar frame (the pole is the body's axis1
# direction, so bands rotate with the rotation_z pose).
_DISC_TEXTURE_KEYS: frozenset[str] = frozenset(
    {'band_amplitude', 'band_wavenumber', 'band_phase_deg', 'storms'}
)
# One transits entry: a transiting moon disc and/or its cast shadow disc,
# both texture on the rendered parent disc (offsets from the body center
# and radii in detector pixels).
_TRANSIT_ENTRY_KEYS: frozenset[str] = frozenset({'moon', 'shadow'})
_TRANSIT_MOON_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'radius_px', 'albedo_factor'})
_TRANSIT_SHADOW_KEYS: frozenset[str] = frozenset({'dv_px', 'du_px', 'radius_px', 'darkness'})


def _check_disc_texture(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``disc_texture`` map (bands + storm ovals)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(
            f'{source}: {label}.disc_texture must be a mapping when present'
        )
    unknown = set(value) - _DISC_TEXTURE_KEYS
    if unknown:
        raise SimSceneValidationError(
            f'{source}: {label}.disc_texture: unknown keys: {sorted(unknown)}'
        )
    _check_optional_nonnegative_number(
        value.get('band_amplitude'), f'{label}.disc_texture.band_amplitude', source=source
    )
    _check_optional_nonnegative_number(
        value.get('band_wavenumber'), f'{label}.disc_texture.band_wavenumber', source=source
    )
    _check_optional_number(
        value.get('band_phase_deg'), f'{label}.disc_texture.band_phase_deg', source=source
    )
    _check_surface_spot_list(value.get('storms'), key=f'{label}.disc_texture.storms', source=source)


def _check_transit_disc(value: Any, allowed: frozenset[str], *, label: str, source: str) -> None:
    """Validate one transit sub-map (the moon disc or the shadow disc)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {label} must be a mapping when present')
    unknown = set(value) - allowed
    if unknown:
        raise SimSceneValidationError(f'{source}: {label}: unknown keys: {sorted(unknown)}')
    _check_optional_number(value.get('dv_px'), f'{label}.dv_px', source=source)
    _check_optional_number(value.get('du_px'), f'{label}.du_px', source=source)
    _check_optional_positive_number(value.get('radius_px'), f'{label}.radius_px', source=source)
    _check_optional_nonnegative_number(
        value.get('albedo_factor'), f'{label}.albedo_factor', source=source
    )
    darkness = value.get('darkness')
    _check_optional_nonnegative_number(darkness, f'{label}.darkness', source=source)
    if darkness is not None and float(darkness) > 1.0:
        raise SimSceneValidationError(
            f'{source}: {label}.darkness must lie in [0, 1]; got {darkness!r}'
        )


def _check_transits(value: Any, *, label: str, source: str) -> None:
    """Validate one body's ``transits`` list (moon and/or shadow discs)."""
    if value is None:
        return
    _check_optional_mapping_list(value, f'{label}.transits', source=source)
    for index, entry in enumerate(value):
        entry_label = f'{label}.transits[{index}]'
        unknown = set(entry) - _TRANSIT_ENTRY_KEYS
        if unknown:
            raise SimSceneValidationError(
                f'{source}: {entry_label}: unknown keys: {sorted(unknown)}'
            )
        if entry.get('moon') is None and entry.get('shadow') is None:
            raise SimSceneValidationError(
                f'{source}: {entry_label} needs a moon and/or a shadow map'
            )
        _check_transit_disc(
            entry.get('moon'), _TRANSIT_MOON_KEYS, label=f'{entry_label}.moon', source=source
        )
        _check_transit_disc(
            entry.get('shadow'),
            _TRANSIT_SHADOW_KEYS,
            label=f'{entry_label}.shadow',
            source=source,
        )
