"""Single-variable parameter-sweep harness for the sim scene catalog (Phase T3).

A sweep takes one catalog scene as a base, varies a single parameter (or a group
of parameters that move together, e.g. the three axes of a sphere) across a list
of values, and navigates each resulting frame.  The per-step row records the
recovered offset error, confidence, status, and primary technique, so a test can
assert how a navigation diagnostic *responds* to a controlled change -- the
verification layer a calibrated confidence formula relies on.

This module is the importable core (schema, loader, runner); ``sim_sweep_runner``
is the ``python -m`` entry point and ``test_sim_sweeps`` asserts the per-sweep
invariants.  The harness is in-process and needs no external holdings, but the
navigation it drives carries the usual sub-millipixel cross-process jitter, so
tests assert trends and bounds rather than exact values.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import NavOrchestrator
from nav.obs.obs_inst_sim import ObsSim
from nav.sim.scene import SimScene, load_sim_scene

_SCENES_ROOT = Path(__file__).parent / 'sim_scenes'


class SimSweepValidationError(ValueError):
    """Raised when a sweep spec YAML is missing or malformed."""


@dataclass(frozen=True)
class SweepSpec:
    """A validated single-variable sweep specification."""

    path: Path
    sweep_name: str
    base_scene: Path
    parameters: tuple[str, ...]
    values: tuple[float, ...]
    technique: str


@dataclass(frozen=True)
class SweepRow:
    """One navigated step of a sweep."""

    value: float
    status: str
    offset_error_px: float | None
    rotation_error_deg: float | None
    confidence: float
    primary_technique: str | None


def load_sweep(path: Path) -> SweepSpec:
    """Parse and validate a sweep spec YAML.

    Parameters:
        path: Path to a ``<sweep_name>.yaml`` spec.

    Returns:
        A frozen :class:`SweepSpec`.

    Raises:
        SimSweepValidationError: On any missing/invalid field.
    """
    yaml = YAML(typ='safe')
    try:
        raw = yaml.load(path.read_text())
    except Exception as exc:  # ruamel may raise several exception types
        raise SimSweepValidationError(f'{path}: cannot parse YAML: {exc}') from exc
    if not isinstance(raw, dict):
        raise SimSweepValidationError(f'{path}: top-level YAML must be a mapping')
    sweep_name = raw.get('sweep_name')
    if not isinstance(sweep_name, str) or sweep_name != path.stem:
        raise SimSweepValidationError(
            f'{path}: sweep_name {sweep_name!r} must match filename stem {path.stem!r}'
        )
    base_scene_raw = raw.get('base_scene')
    if not isinstance(base_scene_raw, str):
        raise SimSweepValidationError(f'{path}: base_scene must be a string path')
    base_scene = _SCENES_ROOT / base_scene_raw
    if not base_scene.is_file():
        raise SimSweepValidationError(f'{path}: base_scene {base_scene} does not exist')
    parameters = raw.get('parameters')
    if (
        not isinstance(parameters, list)
        or not parameters
        or any(not isinstance(p, str) for p in parameters)
    ):
        raise SimSweepValidationError(f'{path}: parameters must be a non-empty list of strings')
    values = raw.get('values')
    if (
        not isinstance(values, list)
        or len(values) < 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
    ):
        raise SimSweepValidationError(f'{path}: values must be a list of >= 2 numbers')
    technique = raw.get('technique', '*')
    if not isinstance(technique, str) or not technique:
        raise SimSweepValidationError(f'{path}: technique must be a non-empty string when present')
    return SweepSpec(
        path=path,
        sweep_name=sweep_name,
        base_scene=base_scene,
        parameters=tuple(parameters),
        values=tuple(float(v) for v in values),
        technique=technique,
    )


def iter_sweep_paths(root: Path) -> list[Path]:
    """Return every ``<sweep_name>.yaml`` spec under ``root``, sorted."""
    return sorted(root.glob('*.yaml'))


def _set_dotted(params: dict[str, Any], dotted: str, value: float) -> None:
    """Set ``value`` at a dotted path into a sim-params mapping.

    Supports mapping keys and integer list indices, e.g. ``bodies.0.phase_angle``
    or ``noise.read_noise_dn``.
    """
    keys = dotted.split('.')
    node: Any = params
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _primary_technique(result: Any) -> str | None:
    """Return the non-spurious technique with the highest confidence, or None."""
    candidates = [t for t in result.per_technique if not t.spurious]
    if not candidates:
        return None
    best = max(candidates, key=lambda t: t.confidence)
    return str(best.technique_name)


def _navigate_params(sim_params: dict[str, Any], only_techniques: str) -> Any:
    obs = ObsSim.from_file('/tmp/sweep.json', sim_params=sim_params)
    orchestrator = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques=only_techniques
    )
    return orchestrator.navigate(obs)


def _recovered_rotation_deg(result: Any) -> float | None:
    """Return the recovered camera roll in degrees, or ``None``.

    Prefers the fused result's rotation; when the fused status is below success
    (e.g. a clean star field whose placeholder-alpha confidence is low) it falls
    back to the highest-confidence non-spurious technique that reported a roll.
    """
    if result.rotation_rad is not None:
        return math.degrees(result.rotation_rad)
    candidates = [t for t in result.per_technique if not t.spurious and t.rotation_rad is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda t: t.confidence)
    return math.degrees(best.rotation_rad)


def _pinned_technique_result(result: Any, technique: str) -> Any | None:
    """Return the non-spurious per-technique result for ``technique``, or None."""
    for t in result.per_technique:
        if t.technique_name == technique and not t.spurious:
            return t
    return None


def run_sweep(spec: SweepSpec) -> list[SweepRow]:
    """Navigate every step of a sweep and return the per-step rows.

    The planted ground truth is read from the (post-override) sim params, so a
    sweep over the offset or the camera roll itself tracks correctly: the offset
    error is the Euclidean distance between the recovered and planted offset, and
    the rotation error is the absolute difference between the recovered and
    planted roll. Either is ``None`` when the navigator (or the relevant
    technique) produced no value.

    When ``spec.technique`` is a specific technique name (not ``'*'``) the sweep
    pins that technique and reads its *own* recovered offset/roll, so a technique
    can be characterised even when its clean-field confidence holds the fused
    status below success. With ``'*'`` the fused full-ensemble result is used.

    Parameters:
        spec: The sweep specification.

    Returns:
        One :class:`SweepRow` per value, in sweep order.
    """
    scene: SimScene = load_sim_scene(spec.base_scene)
    base_params = scene.to_sim_params()
    pinned = spec.technique != '*'
    rows: list[SweepRow] = []
    for value in spec.values:
        sim_params = copy.deepcopy(base_params)
        for parameter in spec.parameters:
            _set_dotted(sim_params, parameter, value)
        planted_v = float(sim_params.get('offset_v', 0.0))
        planted_u = float(sim_params.get('offset_u', 0.0))
        planted_rot = float(sim_params.get('offset_rotation_deg', 0.0))
        result = _navigate_params(sim_params, spec.technique)
        if pinned:
            pin = _pinned_technique_result(result, spec.technique)
            recovered_offset = pin.offset_px if pin is not None else None
            recovered_rot = (
                math.degrees(pin.rotation_rad)
                if pin is not None and pin.rotation_rad is not None
                else None
            )
        else:
            recovered_offset = result.offset_px
            recovered_rot = _recovered_rotation_deg(result)
        if recovered_offset is None:
            offset_error: float | None = None
        else:
            offset_error = math.hypot(
                recovered_offset[0] - planted_v, recovered_offset[1] - planted_u
            )
        rotation_error = None if recovered_rot is None else abs(recovered_rot - planted_rot)
        rows.append(
            SweepRow(
                value=value,
                status=str(result.status),
                offset_error_px=offset_error,
                rotation_error_deg=rotation_error,
                confidence=float(result.confidence),
                primary_technique=_primary_technique(result),
            )
        )
    return rows
