"""Scene-spec schema for the simulator scene catalog.

A sim scene is a YAML file describing a synthetic frame the navigator can be run
against: instrument, geometry, noise, stray light, and the planted offset the
navigator should recover.  The catalog is laid out as
``<scene_class>/<scene_name>.yaml`` (the directory is the registry).

The YAML fields are the flat runtime parameter names that the renderer
(:func:`spindoctor.sim.render.render_combined_model`),
:class:`spindoctor.obs.obs_inst_sim.ObsSim`, and the GUI consume, so a validated scene
file IS the ``sim_params`` mapping with no translation layer.  ``load_sim_scene``
parses and validates a file and returns that dict; ``save_sim_scene`` validates a
``sim_params`` dict and writes it (injecting ``schema_version`` and
``scene_name``); ``validate_sim_params`` validates an in-memory dict for
programmatic scene authors.  The validator is hand-rolled (no pydantic
dependency).

**The information boundary.**  Every key in the schema is classified as either
idealized (information the production pipeline could know from catalogs,
SPICE, labels, or config: exposed to the navigator through ``obs.nav_params``)
or truth (nature's values, planted errors, variance knobs, and contaminants:
readable only by the image-side renderer).  ``build_nav_params`` constructs
the filtered idealized view; :data:`TRUTH_KEYS` is the machine-readable truth
set the boundary test iterates.  A key added to the schema without a
classification fails the import-time completeness assertion, so every future
schema change must extend the boundary in the same change.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from spindoctor.sim.instruments import GENERIC_INSTRUMENT_ALIASES, SIM_INSTRUMENTS

# Scene classes for the sim catalog.  The structural test asserts every
# subdirectory under a catalog root is one of these so typos fail loudly.
DECLARED_SIM_SCENE_CLASSES: frozenset[str] = frozenset(
    {
        'phase_sweep_regular_body',
        'phase_sweep_irregular_body',
        'noise_sweep',
        'smear_sweep',
        'range_sweep',
        'multi_body_geometry',
        'algorithmic_invariants',
        'regression',
    }
)

# Instrument names a scene may name (the sim instruments plus the generic alias).
ALLOWED_INSTRUMENTS: frozenset[str] = frozenset(SIM_INSTRUMENTS) | GENERIC_INSTRUMENT_ALIASES

CURRENT_SCHEMA_VERSION: int = 2

# ---------------------------------------------------------------------------
# Key inventory and information-boundary classification.
#
# _ALLOWED_KEYS / _*_KEYS are the complete inventory (unknown keys fail
# validation so typos do not silently render the default scene).  The
# *_IDEALIZED_KEYS / *_TRUTH_KEYS sets classify every inventory key for the
# information boundary; the import-time assertion below keeps the
# classification complete and disjoint.
# ---------------------------------------------------------------------------

# Every top-level key a scene may carry.  These are the flat runtime sim_params
# names the renderer / ObsSim consume directly, plus the schema_version /
# scene_name metadata the renderer ignores.
_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'random_seed',
        'exposure_sec',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'shade_solid_rings',
        'oversample',
        'optics',
        'spk_error',
        'bodies',
        'rings',
        'stars',
        'background_stars_num',
        'background_stars_psf_sigma',
        'background_stars_distribution_exponent',
        'noise',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level idealized keys: frame identity, emulated-instrument configuration,
# and epoch/timing values the production pipeline reads from labels and
# published models.  'ring_epoch' is deliberately idealized: the precessing
# ring model's epoch is catalog knowledge the navigator-side ring model reads.
TOP_LEVEL_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'schema_version',
        'scene_name',
        'instrument',
        'size_v',
        'size_u',
        'exposure_sec',
        'midtime_utc',
        'closest_planet',
        'time',
        'ring_epoch',
        'bodies',
        'rings',
        'stars',
        'instrument_config',
        'fit_camera_rotation',
    }
)

# Top-level truth keys: the planted pointing error the navigator must recover,
# the RNG realization, and the contaminant / noise fields.  The renderer's
# appearance knob 'shade_solid_rings' is image-side only (the navigator's
# ring template is always solid-shaded by its own convention).
TOP_LEVEL_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'random_seed',
        'offset_v',
        'offset_u',
        'offset_rotation_deg',
        'shade_solid_rings',
        'oversample',
        'optics',
        'spk_error',
        'background_stars_num',
        'background_stars_psf_sigma',
        'background_stars_distribution_exponent',
        'noise',
    }
)

# Per-body idealized keys: the ellipsoid/mesh geometry, pose, lighting, and
# physical scale the production pipeline knows from SPICE and shape catalogs.
# The mesh keys are idealized because the published shape model of an
# irregular body is catalog knowledge; a scene plants shape error through
# 'nav_override', not by hiding the mesh.
_BODY_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'shape_model',
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
        'mesh_n_lat',
        'mesh_n_lon',
        'mesh_seed',
        'pose_euler_deg',
    }
)

# Per-body truth keys: surface texture (craters) is nature's terrain, 'seed'
# is its realization, and 'anti_aliasing' is an image-side rendering-fidelity
# knob (the navigator's template always renders at full anti-aliasing).
# 'nav_override' is special: its VALUES are what the navigator believes
# (idealized), so build_nav_params overlays them onto the body and drops the
# key; the underlying overridden true values never cross.
_BODY_TRUTH_KEYS: frozenset[str] = frozenset(
    {
        'crater_fill',
        'crater_min_radius',
        'crater_max_radius',
        'crater_power_law_exponent',
        'crater_relief_scale',
        'seed',
        'anti_aliasing',
        'nav_override',
    }
)

_BODY_KEYS: frozenset[str] = _BODY_IDEALIZED_KEYS | _BODY_TRUTH_KEYS

# Per-star idealized keys: catalog identity, position, magnitude, spectral
# class, the predicted smear vector (the pipeline computes it from attitude
# telemetry), and the PSF fitting-window size (instrument configuration).
_STAR_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'catalog_name',
        'v',
        'u',
        'vmag',
        'spectral_class',
        'move_v',
        'move_u',
        'psf_size',
    }
)

# Per-star truth keys: a per-star PSF width override is an anomaly of the
# rendered image (the navigator only knows the instrument's published PSF).
_STAR_TRUTH_KEYS: frozenset[str] = frozenset({'psf_sigma'})

_STAR_KEYS: frozenset[str] = _STAR_IDEALIZED_KEYS | _STAR_TRUTH_KEYS

# Per-ring keys (all idealized at present fidelity: the mode-1 orbits ARE the
# catalog orbits, with no planted per-feature error until the phase-F
# ring_system block lands).  'range' here is the z-order/depth hint of the
# legacy rings list, which phase F replaces wholesale.
_RING_IDEALIZED_KEYS: frozenset[str] = frozenset(
    {
        'name',
        'feature_type',
        'center_v',
        'center_u',
        'shading_distance',
        'inner_data',
        'outer_data',
        'range',
        'range_km',
    }
)

_RING_TRUTH_KEYS: frozenset[str] = frozenset()

_RING_KEYS: frozenset[str] = _RING_IDEALIZED_KEYS | _RING_TRUTH_KEYS

# The object blocks of the schema: block name -> (allowed, idealized, truth).
_OBJECT_BLOCKS: dict[str, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {
    'bodies': (_BODY_KEYS, _BODY_IDEALIZED_KEYS, _BODY_TRUTH_KEYS),
    'stars': (_STAR_KEYS, _STAR_IDEALIZED_KEYS, _STAR_TRUTH_KEYS),
    'rings': (_RING_KEYS, _RING_IDEALIZED_KEYS, _RING_TRUTH_KEYS),
}

# The machine-readable truth-key set the ObsSim boundary filter strips and
# the structural boundary test iterates.  Per-object-block entries use dotted
# '<block>.<key>' paths; top-level entries are bare key names.
TRUTH_KEYS: frozenset[str] = frozenset(TOP_LEVEL_TRUTH_KEYS) | frozenset(
    f'{block}.{key}'
    for block, (_allowed, _idealized, truth) in _OBJECT_BLOCKS.items()
    for key in truth
)


def _assert_boundary_classification_complete() -> None:
    """Every schema key must be classified idealized or truth, never both.

    Runs at import so a schema change that adds a key without classifying it
    fails everything loudly, not just one test.
    """
    overlap = TOP_LEVEL_IDEALIZED_KEYS & TOP_LEVEL_TRUTH_KEYS
    assert not overlap, f'top-level keys classified both idealized and truth: {sorted(overlap)}'
    unclassified = _ALLOWED_KEYS - (TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS)
    assert not unclassified, f'top-level keys with no boundary class: {sorted(unclassified)}'
    unknown = (TOP_LEVEL_IDEALIZED_KEYS | TOP_LEVEL_TRUTH_KEYS) - _ALLOWED_KEYS
    assert not unknown, f'classified top-level keys not in the inventory: {sorted(unknown)}'
    for block, (allowed, idealized, truth) in _OBJECT_BLOCKS.items():
        overlap = idealized & truth
        assert not overlap, f'{block} keys classified both idealized and truth: {sorted(overlap)}'
        unclassified = allowed - (idealized | truth)
        assert not unclassified, f'{block} keys with no boundary class: {sorted(unclassified)}'


_assert_boundary_classification_complete()


class SimSceneValidationError(ValueError):
    """Raised when a sim scene YAML is missing or malformed."""


def iter_scene_paths(root: Path) -> list[Path]:
    """Return every ``<class>/<name>.yaml`` scene path under ``root``, sorted."""
    return sorted(root.glob('*/*.yaml'))


def scene_class_for_path(path: Path) -> str:
    """The scene class is the immediate parent directory name."""
    return path.parent.name


def load_sim_scene(path: Path) -> dict[str, Any]:
    """Parse and validate a sim scene YAML into a flat ``sim_params`` dict.

    The returned mapping is exactly what
    :func:`spindoctor.sim.render.render_combined_model` and
    :class:`spindoctor.obs.obs_inst_sim.ObsSim` consume; the ``schema_version`` and
    ``scene_name`` keys are metadata the renderer ignores.

    Parameters:
        path: Path to a ``<scene_name>.yaml`` file.  The ``scene_name`` field
            must equal the filename stem.

    Returns:
        The validated flat ``sim_params`` mapping.

    Raises:
        SimSceneValidationError: On any missing/invalid field.
    """
    yaml = YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except Exception as exc:  # ruamel may raise several exception types
        raise SimSceneValidationError(f'{path}: cannot parse YAML: {exc}') from exc
    if not isinstance(raw, dict):
        raise SimSceneValidationError(f'{path}: top-level YAML must be a mapping')
    _require_int(raw, 'schema_version', source=str(path))
    scene_name = _require_str(raw, 'scene_name', source=str(path))
    if scene_name != path.stem:
        raise SimSceneValidationError(
            f'{path}: scene_name {scene_name!r} must match filename stem {path.stem!r}'
        )
    return validate_sim_params(raw, source=str(path))


def save_sim_scene(sim_params: dict[str, Any], path: Path) -> None:
    """Validate ``sim_params`` and write it to ``path`` as a flat YAML scene.

    The ``schema_version`` and ``scene_name`` (= the filename stem) keys are
    injected so the written file validates on reload.

    Parameters:
        sim_params: The flat GUI / render parameter mapping.
        path: Destination ``<scene_name>.yaml`` path; its stem is the scene name.
    """
    scene: dict[str, Any] = {
        **sim_params,
        'schema_version': CURRENT_SCHEMA_VERSION,
        'scene_name': path.stem,
    }
    validate_sim_params(scene, source=str(path))
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with path.open('w') as handle:
        yaml.dump(scene, handle)


def validate_sim_params(
    sim_params: dict[str, Any], *, source: str = 'sim_params'
) -> dict[str, Any]:
    """Validate a flat ``sim_params`` mapping against the schema inventory.

    This is the validation core shared by :func:`load_sim_scene`,
    :func:`save_sim_scene`, and programmatic scene authors (the calibration
    campaign generator, the doc-image galleries), which build dicts rather
    than files.  ``schema_version`` and ``scene_name`` are optional here (a
    dict author has no filename); when present, the version must be current.

    Parameters:
        sim_params: The flat scene parameter mapping.
        source: Label used in error messages (a path for file authors).

    Returns:
        ``sim_params``, unchanged, for call-chaining.

    Raises:
        SimSceneValidationError: On any unknown or invalid field.
    """
    unknown = set(sim_params) - _ALLOWED_KEYS
    if unknown:
        raise SimSceneValidationError(f'{source}: unknown scene keys: {sorted(unknown)}')

    if 'schema_version' in sim_params:
        schema_version = _require_int(sim_params, 'schema_version', source=source)
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise SimSceneValidationError(
                f'{source}: schema_version must be {CURRENT_SCHEMA_VERSION}, got {schema_version}'
            )
    if 'scene_name' in sim_params:
        _require_str(sim_params, 'scene_name', source=source)

    instrument = _require_str(sim_params, 'instrument', source=source)
    if instrument not in ALLOWED_INSTRUMENTS:
        raise SimSceneValidationError(
            f'{source}: instrument {instrument!r} is not one of {sorted(ALLOWED_INSTRUMENTS)}'
        )
    _require_positive_int(sim_params, 'size_v', source=source)
    _require_positive_int(sim_params, 'size_u', source=source)
    _require_int(sim_params, 'random_seed', source=source)

    _check_optional_positive_number(sim_params.get('exposure_sec'), 'exposure_sec', source=source)
    _check_optional_number(sim_params.get('offset_v'), 'offset_v', source=source)
    _check_optional_number(sim_params.get('offset_u'), 'offset_u', source=source)
    _check_optional_number(
        sim_params.get('offset_rotation_deg'), 'offset_rotation_deg', source=source
    )
    _check_optional_number(sim_params.get('time'), 'time', source=source)
    _check_optional_number(sim_params.get('ring_epoch'), 'ring_epoch', source=source)
    _check_optional_positive_number(
        sim_params.get('background_stars_psf_sigma'), 'background_stars_psf_sigma', source=source
    )
    _check_optional_number(
        sim_params.get('background_stars_distribution_exponent'),
        'background_stars_distribution_exponent',
        source=source,
    )
    _check_optional_nonnegative_int(
        sim_params.get('background_stars_num'), 'background_stars_num', source=source
    )
    _check_optional_str(sim_params.get('midtime_utc'), 'midtime_utc', source=source)
    _check_optional_str(sim_params.get('closest_planet'), 'closest_planet', source=source)
    _check_optional_bool(sim_params.get('shade_solid_rings'), 'shade_solid_rings', source=source)
    _check_optional_bool(
        sim_params.get('fit_camera_rotation'), 'fit_camera_rotation', source=source
    )
    _check_optional_mapping(sim_params.get('noise'), 'noise', source=source)
    _check_optional_mapping(sim_params.get('instrument_config'), 'instrument_config', source=source)
    _check_optional_positive_int(sim_params.get('oversample'), 'oversample', source=source)
    _check_optics(sim_params.get('optics'), source=source)
    _check_spk_error(sim_params.get('spk_error'), source=source)

    for block in ('bodies', 'rings', 'stars'):
        _check_optional_mapping_list(sim_params.get(block), block, source=source)
        allowed = _OBJECT_BLOCKS[block][0]
        for index, obj in enumerate(sim_params.get(block) or []):
            unknown = set(obj) - allowed
            if unknown:
                raise SimSceneValidationError(
                    f'{source}: {block}[{index}]: unknown keys: {sorted(unknown)}'
                )
            if block == 'bodies':
                _check_body_object(obj, index=index, source=source)
            elif block == 'rings':
                _check_ring_object(obj, index=index, source=source)
            else:
                _check_star_object(obj, index=index, source=source)

    if sim_params.get('spk_error') is not None:
        _require_ranges_for_spk_error(sim_params, source=source)
    _resolve_match_navigator_psf(sim_params)

    return sim_params


def build_nav_params(sim_params: dict[str, Any]) -> dict[str, Any]:
    """Build the navigator's filtered idealized view of a scene.

    This is the information boundary (the independence guarantee of the
    simulator-realism program): the returned mapping contains only keys
    classified idealized, with every :data:`TRUTH_KEYS` entry stripped.  For
    bodies, a ``nav_override`` mapping is overlaid first and the key dropped,
    so the navigator sees the geometry it *believes* without learning the
    true values underneath.  Objects marked non-navigable are dropped
    entirely (the ``navigable`` flag itself lands with later phases; the
    mechanism is in place).  All values are deep copies, so navigator-side
    code cannot mutate the renderer's scene.

    Parameters:
        sim_params: The full scene mapping (the renderer's input).

    Returns:
        The filtered ``nav_params`` mapping exposed as ``obs.nav_params``.
    """
    nav: dict[str, Any] = {}
    for key, value in sim_params.items():
        if key in _OBJECT_BLOCKS:
            continue  # handled below
        if key in TOP_LEVEL_IDEALIZED_KEYS:
            nav[key] = copy.deepcopy(value)
        # Anything else (truth keys, unknown keys) stays behind the boundary:
        # the filter is default-deny.
    for block, (_allowed, idealized, _truth) in _OBJECT_BLOCKS.items():
        if block not in sim_params:
            continue
        filtered_objects: list[dict[str, Any]] = []
        for obj in sim_params.get(block) or []:
            if not isinstance(obj, dict):
                continue
            if obj.get('navigable') is False:
                continue
            merged = dict(obj)
            if block == 'bodies':
                override = merged.pop('nav_override', None)
                if isinstance(override, dict):
                    merged.update(override)
            filtered_objects.append(
                {k: copy.deepcopy(v) for k, v in merged.items() if k in idealized}
            )
        nav[block] = filtered_objects
    return nav


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


def _resolve_match_navigator_psf(sim_params: dict[str, Any]) -> None:
    """Resolve ``optics.psf.match_navigator`` into a concrete Gaussian in place.

    The floor configuration (the self-consistency baseline of the sweeps) sets
    the image-side PSF equal to the navigator's own model: a pure Gaussian at
    the emulated instrument's configured ``star_psf_sigma``, with no Moffat
    wing and no field variation.  Resolving it to concrete numbers at
    validation time keeps the render cache key stable and makes the resolved
    kernel inspectable.

    Parameters:
        sim_params: The validated scene mapping; its ``optics.psf`` block is
            rewritten when it requests navigator matching.
    """
    optics = sim_params.get('optics')
    if not isinstance(optics, dict):
        return
    psf = optics.get('psf')
    if not isinstance(psf, dict) or not psf.get('match_navigator'):
        return
    # Import here to avoid a module-load cycle: the resolver needs the live
    # config only when a scene actually asks for navigator matching.
    from spindoctor.config import DEFAULT_CONFIG
    from spindoctor.sim.instruments import resolve_sim_inst_config

    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG, sim_params.get('instrument'), sim_params.get('instrument_config')
    )
    sigma = float(inst_config['star_psf_sigma'])
    optics['psf'] = {'sigma_v': sigma, 'sigma_u': sigma, 'w': 0.0, 'r0': 2.0, 'n': 3.0}


def _require_str(raw: dict[str, Any], key: str, *, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SimSceneValidationError(f'{source}: {key} must be a non-empty string')
    return value


def _require_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimSceneValidationError(f'{source}: {key} must be an integer')
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, source: str) -> int:
    value = _require_int(raw, key, source=source)
    if value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer')
    return value


def _check_optional_number(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimSceneValidationError(f'{source}: {key} must be a number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{source}: {key} must be finite; got {value!r}')


def _check_optional_positive_number(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SimSceneValidationError(f'{source}: {key} must be a positive number when present')


def _check_optional_nonnegative_int(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimSceneValidationError(
            f'{source}: {key} must be a non-negative integer when present'
        )


def _check_optional_positive_int(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SimSceneValidationError(f'{source}: {key} must be a positive integer when present')


def _check_optional_nonnegative_number(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SimSceneValidationError(f'{source}: {key} must be a non-negative number when present')


def _check_optional_str(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise SimSceneValidationError(f'{source}: {key} must be a string when present')


def _check_optional_bool(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        raise SimSceneValidationError(f'{source}: {key} must be a boolean when present')


def _check_optional_mapping(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{source}: {key} must be a mapping when present')


def _check_optional_mapping_list(value: Any, key: str, *, source: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimSceneValidationError(f'{source}: {key} must be a list of mappings')
