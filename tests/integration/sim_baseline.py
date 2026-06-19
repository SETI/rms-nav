"""Regression-baseline schema for the simulator scene catalog (Phase T2).

A sim baseline is a tiny JSON file at
``tests/integration/sim_baselines/<scene_name>.json`` recording the exact
rounded outcome the orchestrator produced for a catalog scene:

```json
{
  "scene_name": "planted_offset_disc",
  "status": "success",
  "offset_dv_px": 3.5,
  "offset_du_px": -1.7969,
  "confidence": 0.366
}
```

Unlike the real-image baseline (which always navigates), a sim scene may fail;
``status`` is recorded and the offsets are ``null`` for a non-success run.
``offset_*_px`` and ``confidence`` are rounded to 2 decimals so comparison
against a fresh navigation run is exact-equal on rounded values.  Navigation is
deterministic within a process, but the technique solvers carry sub-millipixel
floating-point jitter across processes (BLAS reordering under parallel load), so
the rounding is intentionally coarser than the real-image baselines' (which run
serially in the integration job); 0.01 px / 0.01 confidence still catches any
real regression while staying immune to that jitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nav.nav_model import build_models_for_obs
from nav.nav_orchestrator import NavOrchestrator
from nav.obs.obs_inst_sim import ObsSim

OFFSET_DECIMALS: int = 2
CONFIDENCE_DECIMALS: int = 2


@dataclass(frozen=True)
class SimBaseline:
    """An exact-rounded snapshot of one sim scene's navigation outcome."""

    scene_name: str
    status: str
    offset_dv_px: float | None
    offset_du_px: float | None
    confidence: float

    @classmethod
    def from_run(
        cls,
        *,
        scene_name: str,
        status: str,
        offset_px: tuple[float, float] | None,
        confidence: float,
    ) -> SimBaseline:
        """Build a baseline by rounding live navigation outputs."""
        dv = round(float(offset_px[0]), OFFSET_DECIMALS) if offset_px is not None else None
        du = round(float(offset_px[1]), OFFSET_DECIMALS) if offset_px is not None else None
        return cls(
            scene_name=scene_name,
            status=status,
            offset_dv_px=dv,
            offset_du_px=du,
            confidence=round(float(confidence), CONFIDENCE_DECIMALS),
        )

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, no NaN)."""
        return (
            json.dumps(
                {
                    'scene_name': self.scene_name,
                    'status': self.status,
                    'offset_dv_px': self.offset_dv_px,
                    'offset_du_px': self.offset_du_px,
                    'confidence': self.confidence,
                },
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + '\n'
        )


_REQUIRED_KEYS: tuple[str, ...] = (
    'scene_name',
    'status',
    'offset_dv_px',
    'offset_du_px',
    'confidence',
)


def load_sim_baseline(path: Path) -> SimBaseline:
    """Parse one sim baseline JSON file with strict schema validation."""
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: cannot parse JSON: {exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: top-level JSON must be a mapping, got {type(raw).__name__}')
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f'{path}: missing required key(s): {missing}')
    return SimBaseline(
        scene_name=_require_str(raw, 'scene_name', path),
        status=_require_str(raw, 'status', path),
        offset_dv_px=_optional_float(raw['offset_dv_px'], 'offset_dv_px', path),
        offset_du_px=_optional_float(raw['offset_du_px'], 'offset_du_px', path),
        confidence=_require_float(raw['confidence'], 'confidence', path),
    )


def baseline_for_scene(scene: dict[str, Any]) -> SimBaseline:
    """Render and navigate the scene ``sim_params``, returning its rounded baseline."""
    obs = ObsSim.from_file('/tmp/sim_baseline.yaml', sim_params=scene)
    result = NavOrchestrator(
        build_models_for_obs(obs), only_models='*', only_techniques='*'
    ).navigate(obs)
    return SimBaseline.from_run(
        scene_name=scene['scene_name'],
        status=str(result.status),
        offset_px=result.offset_px,
        confidence=float(result.confidence),
    )


def sim_baseline_path(baselines_dir: Path, scene_name: str) -> Path:
    """Return the canonical baseline path for a scene."""
    return baselines_dir / f'{scene_name}.json'


def discover_sim_baseline_paths(baselines_dir: Path) -> list[Path]:
    """List every ``<scene_name>.json`` under ``sim_baselines/`` (sorted)."""
    if not baselines_dir.is_dir():
        return []
    return sorted(baselines_dir.glob('*.json'))


def _require_str(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f'{path}: {key!r} must be a string, got {type(value).__name__}')
    return value


def _require_float(value: Any, key: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{path}: {key!r} must be a number, got {type(value).__name__}')
    return float(value)


def _optional_float(value: Any, key: str, path: Path) -> float | None:
    if value is None:
        return None
    return _require_float(value, key, path)
