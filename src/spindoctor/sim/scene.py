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
``scene_name``).  The validator is hand-rolled (no pydantic dependency).
"""

from __future__ import annotations

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

CURRENT_SCHEMA_VERSION: int = 1

# Every top-level key a scene may carry.  These are the flat runtime sim_params
# names the renderer / ObsSim consume directly, plus the schema_version /
# scene_name metadata the renderer ignores.  An unknown key fails validation so
# typos do not silently render the default scene.
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
        'bodies',
        'rings',
        'stars',
        'background_stars_num',
        'background_stars_psf_sigma',
        'background_stars_distribution_exponent',
        'noise',
        'stray_light',
        'instrument_config',
        'fit_camera_rotation',
    }
)


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
    return _validate(raw, path=path)


def save_sim_scene(sim_params: dict[str, Any], path: Path) -> None:
    """Validate ``sim_params`` and write it to ``path`` as a flat YAML scene.

    The ``schema_version`` and ``scene_name`` (= the filename stem) keys are
    injected so the written file validates on reload.

    Parameters:
        sim_params: The flat GUI / render parameter mapping.
        path: Destination ``<scene_name>.yaml`` path; its stem is the scene name.
    """
    scene: dict[str, Any] = {
        'schema_version': CURRENT_SCHEMA_VERSION,
        'scene_name': path.stem,
        **sim_params,
    }
    _validate(scene, path=path)
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with path.open('w') as handle:
        yaml.dump(scene, handle)


def _validate(raw: dict[str, Any], *, path: Path) -> dict[str, Any]:
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise SimSceneValidationError(f'{path}: unknown scene keys: {sorted(unknown)}')

    schema_version = _require_int(raw, 'schema_version', path=path)
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise SimSceneValidationError(
            f'{path}: schema_version must be {CURRENT_SCHEMA_VERSION}, got {schema_version}'
        )
    scene_name = _require_str(raw, 'scene_name', path=path)
    if scene_name != path.stem:
        raise SimSceneValidationError(
            f'{path}: scene_name {scene_name!r} must match filename stem {path.stem!r}'
        )
    instrument = _require_str(raw, 'instrument', path=path)
    if instrument not in ALLOWED_INSTRUMENTS:
        raise SimSceneValidationError(
            f'{path}: instrument {instrument!r} is not one of {sorted(ALLOWED_INSTRUMENTS)}'
        )
    _require_positive_int(raw, 'size_v', path=path)
    _require_positive_int(raw, 'size_u', path=path)
    _require_int(raw, 'random_seed', path=path)

    _check_optional_positive_number(raw.get('exposure_sec'), 'exposure_sec', path=path)
    _check_optional_number(raw.get('offset_v'), 'offset_v', path=path)
    _check_optional_number(raw.get('offset_u'), 'offset_u', path=path)
    _check_optional_number(raw.get('offset_rotation_deg'), 'offset_rotation_deg', path=path)
    _check_optional_number(raw.get('time'), 'time', path=path)
    _check_optional_number(raw.get('ring_epoch'), 'ring_epoch', path=path)
    _check_optional_positive_number(
        raw.get('background_stars_psf_sigma'), 'background_stars_psf_sigma', path=path
    )
    _check_optional_number(
        raw.get('background_stars_distribution_exponent'),
        'background_stars_distribution_exponent',
        path=path,
    )
    _check_optional_nonnegative_int(
        raw.get('background_stars_num'), 'background_stars_num', path=path
    )
    _check_optional_str(raw.get('midtime_utc'), 'midtime_utc', path=path)
    _check_optional_str(raw.get('closest_planet'), 'closest_planet', path=path)
    _check_optional_bool(raw.get('shade_solid_rings'), 'shade_solid_rings', path=path)
    _check_optional_bool(raw.get('fit_camera_rotation'), 'fit_camera_rotation', path=path)
    _check_optional_mapping_list(raw.get('bodies'), 'bodies', path=path)
    _check_optional_mapping_list(raw.get('rings'), 'rings', path=path)
    _check_optional_mapping_list(raw.get('stars'), 'stars', path=path)
    _check_optional_mapping(raw.get('noise'), 'noise', path=path)
    _check_optional_mapping(raw.get('stray_light'), 'stray_light', path=path)
    _check_optional_mapping(raw.get('instrument_config'), 'instrument_config', path=path)

    return raw


def _require_str(raw: dict[str, Any], key: str, *, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SimSceneValidationError(f'{path}: {key} must be a non-empty string')
    return value


def _require_int(raw: dict[str, Any], key: str, *, path: Path) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SimSceneValidationError(f'{path}: {key} must be an integer')
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, path: Path) -> int:
    value = _require_int(raw, key, path=path)
    if value <= 0:
        raise SimSceneValidationError(f'{path}: {key} must be a positive integer')
    return value


def _check_optional_number(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimSceneValidationError(f'{path}: {key} must be a number when present')
    if not math.isfinite(float(value)):
        raise SimSceneValidationError(f'{path}: {key} must be finite; got {value!r}')


def _check_optional_positive_number(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SimSceneValidationError(f'{path}: {key} must be a positive number when present')


def _check_optional_nonnegative_int(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SimSceneValidationError(f'{path}: {key} must be a non-negative integer when present')


def _check_optional_str(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise SimSceneValidationError(f'{path}: {key} must be a string when present')


def _check_optional_bool(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if not isinstance(value, bool):
        raise SimSceneValidationError(f'{path}: {key} must be a boolean when present')


def _check_optional_mapping(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{path}: {key} must be a mapping when present')


def _check_optional_mapping_list(value: Any, key: str, *, path: Path) -> None:
    if value is None:
        return
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimSceneValidationError(f'{path}: {key} must be a list of mappings')
