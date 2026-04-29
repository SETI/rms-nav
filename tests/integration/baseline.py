"""Regression-baseline schema for the per-image library (Part 0 §17).

A baseline is a tiny JSON file at ``tests/integration/baselines/<image_id>.json``
recording the exact rounded outputs the orchestrator produced last time
the operator approved them:

```json
{
  "image_id": "C0061085400R",
  "offset_dv_px": 299.0010,
  "offset_du_px": -130.9985,
  "confidence": 0.871
}
```

``offset_*_px`` is rounded to 4 decimals; ``confidence`` to 3 decimals;
comparison against a fresh navigation run is exact-equal on rounded
values.  ``pipeline_run_iso8601`` is intentionally absent because it is
the only provenance field that is not byte-identical between identical
runs (Part 0 §11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

OFFSET_DECIMALS: int = 4
CONFIDENCE_DECIMALS: int = 3


@dataclass(frozen=True)
class Baseline:
    """An exact-rounded snapshot of one image's headline navigation result."""

    image_id: str
    offset_dv_px: float
    offset_du_px: float
    confidence: float

    @classmethod
    def from_run(
        cls,
        *,
        image_id: str,
        offset_px: tuple[float, float],
        confidence: float,
    ) -> Baseline:
        """Build a baseline by rounding live navigation outputs."""
        return cls(
            image_id=image_id,
            offset_dv_px=round(float(offset_px[0]), OFFSET_DECIMALS),
            offset_du_px=round(float(offset_px[1]), OFFSET_DECIMALS),
            confidence=round(float(confidence), CONFIDENCE_DECIMALS),
        )

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, no NaN)."""
        return (
            json.dumps(
                {
                    'image_id': self.image_id,
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


_REQUIRED_BASELINE_KEYS: tuple[str, ...] = (
    'image_id',
    'offset_dv_px',
    'offset_du_px',
    'confidence',
)


def load_baseline(path: Path) -> Baseline:
    """Parse one baseline JSON file with strict schema validation.

    Wraps JSON-decode errors, missing keys, and wrong-type fields in
    diagnostics that include the offending file path so a malformed
    baseline fails loudly with the file under suspicion identified.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: cannot parse JSON: {exc}') from exc
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: top-level JSON must be a mapping, got {type(raw).__name__}')
    missing = [k for k in _REQUIRED_BASELINE_KEYS if k not in raw]
    if missing:
        raise KeyError(f'{path}: missing required key(s): {missing}')
    image_id = raw['image_id']
    if not isinstance(image_id, str):
        raise TypeError(f'{path}: image_id must be a string, got {type(image_id).__name__}')
    floats: dict[str, float] = {}
    for key in ('offset_dv_px', 'offset_du_px', 'confidence'):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'{path}: {key!r} must be a number, got {type(value).__name__}')
        floats[key] = float(value)
    return Baseline(
        image_id=image_id,
        offset_dv_px=floats['offset_dv_px'],
        offset_du_px=floats['offset_du_px'],
        confidence=floats['confidence'],
    )


def baseline_path(baselines_dir: Path, image_id: str) -> Path:
    """Return the canonical baseline path for an ``image_id``."""
    return baselines_dir / f'{image_id}.json'


def discover_baseline_paths(baselines_dir: Path) -> list[Path]:
    """List every ``<image_id>.json`` under ``baselines/`` (sorted)."""
    if not baselines_dir.is_dir():
        return []
    return sorted(baselines_dir.glob('*.json'))
