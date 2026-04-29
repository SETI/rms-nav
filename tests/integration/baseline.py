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


def load_baseline(path: Path) -> Baseline:
    """Parse one baseline JSON file."""
    raw = json.loads(path.read_text())
    return Baseline(
        image_id=str(raw['image_id']),
        offset_dv_px=float(raw['offset_dv_px']),
        offset_du_px=float(raw['offset_du_px']),
        confidence=float(raw['confidence']),
    )


def baseline_path(baselines_dir: Path, image_id: str) -> Path:
    """Return the canonical baseline path for an ``image_id``."""
    return baselines_dir / f'{image_id}.json'


def discover_baseline_paths(baselines_dir: Path) -> list[Path]:
    """List every ``<image_id>.json`` under ``baselines/`` (sorted)."""
    if not baselines_dir.is_dir():
        return []
    return sorted(baselines_dir.glob('*.json'))
