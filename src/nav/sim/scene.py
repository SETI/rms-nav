"""Scene-spec schema for the simulator scene catalog.

A sim scene is a YAML file describing a synthetic frame the navigator can be run
against: instrument, geometry, noise, stray light, and the planted ground-truth
offset the navigator should recover.  The catalog is laid out as
``<scene_class>/<scene_name>.yaml`` (the directory is the registry).

This module is the canonical schema -- the importable peer of the YAML files and
the GUI.  ``load_sim_scene`` validates a file into a :class:`SimScene`;
``SimScene.to_sim_params`` maps it to the dict ``render_combined_model`` /
``ObsSim`` consume; ``scene_dict_from_sim_params`` is the inverse used by the GUI
to save the current scene back to YAML.  The validator is hand-rolled (no
pydantic dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from nav.sim.instruments import GENERIC_INSTRUMENT_ALIASES, SIM_INSTRUMENTS

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
    }
)

# Instrument names a scene may name (the sim instruments plus the generic alias).
ALLOWED_INSTRUMENTS: frozenset[str] = frozenset(SIM_INSTRUMENTS) | GENERIC_INSTRUMENT_ALIASES

CURRENT_SCHEMA_VERSION: int = 1


class SimSceneValidationError(ValueError):
    """Raised when a sim scene YAML is missing or malformed."""


@dataclass(frozen=True)
class GroundTruth:
    """The planted offset/rotation the navigator should recover."""

    planted_offset_dv_px: float = 0.0
    planted_offset_du_px: float = 0.0
    planted_rotation_deg: float = 0.0


@dataclass(frozen=True)
class SimScene:
    """A validated sim scene spec."""

    path: Path
    schema_version: int
    scene_name: str
    instrument: str
    image_size_vu: tuple[int, int]
    random_seed: int
    exposure_sec: float
    midtime_utc: str | None
    bodies: tuple[dict[str, Any], ...]
    rings: tuple[dict[str, Any], ...]
    stars: dict[str, Any] | None
    noise: dict[str, Any] | None
    stray_light: dict[str, Any] | None
    ground_truth: GroundTruth = field(default_factory=GroundTruth)

    @property
    def scene_class(self) -> str:
        """The scene class is the immediate parent directory name."""
        return self.path.parent.name

    def to_sim_params(self) -> dict[str, Any]:
        """Build the parameter dict ``render_combined_model`` / ``ObsSim`` use.

        The planted ground-truth offset becomes the rendered ``offset`` so a
        navigator predicting the unshifted geometry must recover it.

        Returns:
            A sim-params mapping.
        """
        params: dict[str, Any] = {
            'size_v': self.image_size_vu[0],
            'size_u': self.image_size_vu[1],
            'random_seed': self.random_seed,
            'instrument': self.instrument,
            'exposure_sec': self.exposure_sec,
            'offset_v': self.ground_truth.planted_offset_dv_px,
            'offset_u': self.ground_truth.planted_offset_du_px,
            'bodies': [dict(b) for b in self.bodies],
            'rings': [dict(r) for r in self.rings],
        }
        if self.noise is not None:
            params['noise'] = dict(self.noise)
        if self.stray_light is not None:
            params['stray_light'] = dict(self.stray_light)
        if self.stars is not None:
            if 'background_count' in self.stars:
                params['background_stars_num'] = int(self.stars['background_count'])
            if 'list' in self.stars:
                params['stars'] = [dict(s) for s in self.stars['list']]
        return params


def scene_dict_from_sim_params(sim_params: dict[str, Any], *, scene_name: str) -> dict[str, Any]:
    """Build a YAML-serialisable scene dict from GUI/render sim params.

    The inverse of :meth:`SimScene.to_sim_params`: the rendered offset becomes
    the planted ground truth, and the star background count / list fold back into
    a ``stars`` block.

    Parameters:
        sim_params: The GUI / render parameter mapping.
        scene_name: The scene name (must match the target filename stem).

    Returns:
        A mapping ready to validate via :func:`load_sim_scene` after writing.
    """
    scene: dict[str, Any] = {
        'schema_version': CURRENT_SCHEMA_VERSION,
        'scene_name': scene_name,
        'instrument': str(sim_params.get('instrument', 'generic')),
        'image_size_vu': [int(sim_params['size_v']), int(sim_params['size_u'])],
        'random_seed': int(sim_params.get('random_seed', 42)),
        'exposure_sec': float(sim_params.get('exposure_sec', 1.0)),
    }
    bodies = sim_params.get('bodies') or []
    if bodies:
        scene['bodies'] = [dict(b) for b in bodies]
    rings = sim_params.get('rings') or []
    if rings:
        scene['rings'] = [dict(r) for r in rings]
    stars_block: dict[str, Any] = {}
    if int(sim_params.get('background_stars_num', 0)):
        stars_block['background_count'] = int(sim_params['background_stars_num'])
    if sim_params.get('stars'):
        stars_block['list'] = [dict(s) for s in sim_params['stars']]
    if stars_block:
        scene['stars'] = stars_block
    if sim_params.get('noise') is not None:
        scene['noise'] = dict(sim_params['noise'])
    if sim_params.get('stray_light') is not None:
        scene['stray_light'] = dict(sim_params['stray_light'])
    offset_v = float(sim_params.get('offset_v', 0.0))
    offset_u = float(sim_params.get('offset_u', 0.0))
    if offset_v or offset_u:
        scene['ground_truth'] = {
            'planted_offset_dv_px': offset_v,
            'planted_offset_du_px': offset_u,
            'planted_rotation_deg': 0.0,
        }
    return scene


def save_sim_scene(sim_params: dict[str, Any], path: Path) -> None:
    """Write the current scene to ``path`` as YAML, derived from sim params.

    Parameters:
        sim_params: The GUI / render parameter mapping.
        path: Destination ``<scene_name>.yaml`` path; its stem is the scene name.
    """
    scene = scene_dict_from_sim_params(sim_params, scene_name=path.stem)
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    with path.open('w') as handle:
        yaml.dump(scene, handle)


def iter_scene_paths(root: Path) -> list[Path]:
    """Return every ``<class>/<name>.yaml`` scene path under ``root``, sorted."""
    return sorted(root.glob('*/*.yaml'))


def load_sim_scene(path: Path) -> SimScene:
    """Parse and validate a sim scene YAML.

    Parameters:
        path: Path to a ``<scene_name>.yaml`` file.

    Returns:
        A frozen :class:`SimScene`.

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


def _validate(raw: dict[str, Any], *, path: Path) -> SimScene:
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
    image_size_vu = _require_int_pair(raw, 'image_size_vu', path=path)
    random_seed = _require_int(raw, 'random_seed', path=path)
    exposure_sec = _optional_positive_float(raw.get('exposure_sec'), 'exposure_sec', path=path)
    midtime_utc = raw.get('midtime_utc')
    if midtime_utc is not None and not isinstance(midtime_utc, str):
        raise SimSceneValidationError(f'{path}: midtime_utc must be a string when present')

    bodies = _require_mapping_list(raw.get('bodies'), 'bodies', path=path)
    rings = _require_mapping_list(raw.get('rings'), 'rings', path=path)
    stars = _optional_mapping(raw.get('stars'), 'stars', path=path)
    noise = _optional_mapping(raw.get('noise'), 'noise', path=path)
    stray_light = _optional_mapping(raw.get('stray_light'), 'stray_light', path=path)
    ground_truth = _validate_ground_truth(raw.get('ground_truth'), path=path)

    return SimScene(
        path=path,
        schema_version=schema_version,
        scene_name=scene_name,
        instrument=instrument,
        image_size_vu=image_size_vu,
        random_seed=random_seed,
        exposure_sec=exposure_sec,
        midtime_utc=midtime_utc,
        bodies=tuple(bodies),
        rings=tuple(rings),
        stars=stars,
        noise=noise,
        stray_light=stray_light,
        ground_truth=ground_truth,
    )


def _validate_ground_truth(raw: Any, *, path: Path) -> GroundTruth:
    if raw is None:
        return GroundTruth()
    if not isinstance(raw, dict):
        raise SimSceneValidationError(f'{path}: ground_truth must be a mapping when present')
    return GroundTruth(
        planted_offset_dv_px=_optional_float(raw.get('planted_offset_dv_px'), path=path),
        planted_offset_du_px=_optional_float(raw.get('planted_offset_du_px'), path=path),
        planted_rotation_deg=_optional_float(raw.get('planted_rotation_deg'), path=path),
    )


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


def _require_int_pair(raw: dict[str, Any], key: str, *, path: Path) -> tuple[int, int]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value)
    ):
        raise SimSceneValidationError(f'{path}: {key} must be a [v, u] pair of positive integers')
    return (int(value[0]), int(value[1]))


def _optional_float(value: Any, *, path: Path) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimSceneValidationError(f'{path}: numeric field got {value!r}')
    coerced = float(value)
    if not math.isfinite(coerced):
        raise SimSceneValidationError(f'{path}: numeric field must be finite; got {value!r}')
    return coerced


def _optional_positive_float(value: Any, key: str, *, path: Path) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise SimSceneValidationError(f'{path}: {key} must be a positive number')
    return float(value)


def _optional_mapping(value: Any, key: str, *, path: Path) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SimSceneValidationError(f'{path}: {key} must be a mapping when present')
    return value


def _require_mapping_list(value: Any, key: str, *, path: Path) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SimSceneValidationError(f'{path}: {key} must be a list of mappings')
    return value
