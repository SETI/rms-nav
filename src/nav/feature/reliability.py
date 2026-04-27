"""Reliability gate — drops features whose self-assessed reliability is too low.

Each feature carries a [0, 1] reliability score it computed at extraction
time.  The orchestrator runs this gate between extraction and technique
invocation; gated-out features are recorded in the feature inventory with
their rejection reason but are not passed to any technique.

The thresholds live in ``config_520_features.yaml`` per feature type with
optional per-instrument overrides.  Default values defined here are used
when no override is supplied by the loader.
"""

import logging
from dataclasses import dataclass, field
from typing import Final

from nav.feature.feature import NavFeature
from nav.feature.feature_type import NavFeatureType

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    'DEFAULT_RELIABILITY_THRESHOLDS',
    'FeatureReliabilityGate',
    'GatedFeatureRecord',
]


DEFAULT_RELIABILITY_THRESHOLDS: Final[dict[NavFeatureType, float]] = {
    NavFeatureType.STAR: 0.20,
    NavFeatureType.LIMB_ARC: 0.30,
    NavFeatureType.TERMINATOR_ARC: 0.30,
    NavFeatureType.BODY_DISC: 0.30,
    NavFeatureType.BODY_BLOB: 0.20,
    NavFeatureType.RING_EDGE: 0.30,
    NavFeatureType.RING_ANNULUS: 0.30,
    NavFeatureType.CARTOGRAPHIC_MODEL: 0.30,
    NavFeatureType.TITAN_LIMB: 0.30,
}
"""Default per-type reliability thresholds.

Feature types not present in this map fall through with a 0.0 threshold (no
gate).  Values are placeholders subject to empirical calibration.
"""


@dataclass(frozen=True)
class GatedFeatureRecord:
    """Diagnostic record for a feature dropped by the gate.

    Parameters:
        feature: The feature that was dropped.
        reason: Stable string describing why; used by the curator and tests.
    """

    feature: NavFeature
    reason: str


@dataclass
class FeatureReliabilityGate:
    """Stateless gate filtering features by per-type reliability threshold.

    Parameters:
        thresholds: Mapping ``feature_type -> minimum reliability``.  Falls
            back to 0.0 (no gate) for missing keys.

    Raises:
        TypeError: if any key in ``thresholds`` is not a ``NavFeatureType``
            or any value is not numeric.
        ValueError: if any threshold is non-finite or outside ``[0, 1]``.
    """

    thresholds: dict[NavFeatureType, float] = field(
        default_factory=lambda: dict(DEFAULT_RELIABILITY_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        """Validate the thresholds mapping and every threshold value."""
        import math as _math
        from collections.abc import Mapping

        if not isinstance(self.thresholds, Mapping):
            raise TypeError(
                'FeatureReliabilityGate.thresholds must be a Mapping; '
                f'got {type(self.thresholds).__name__}'
            )
        for key, value in self.thresholds.items():
            if not isinstance(key, NavFeatureType):
                raise TypeError(f'thresholds keys must be NavFeatureType; got {type(key).__name__}')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f'thresholds[{key.name}] must be numeric; got {type(value).__name__}'
                )
            if not _math.isfinite(value):
                raise ValueError(f'thresholds[{key.name}] must be finite; got {value!r}')
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'thresholds[{key.name}] must lie in [0, 1]; got {value!r}')

    def apply(
        self, features: list[NavFeature]
    ) -> tuple[list[NavFeature], list[GatedFeatureRecord]]:
        """Split ``features`` into kept and gated lists.

        Parameters:
            features: All features emitted by extractors.

        Returns:
            Tuple ``(kept, gated)`` where ``gated`` carries rejection
            reasons.
        """
        kept: list[NavFeature] = []
        gated: list[GatedFeatureRecord] = []
        for feature in features:
            threshold = self.thresholds.get(feature.feature_type, 0.0)
            if feature.reliability < threshold:
                gated.append(
                    GatedFeatureRecord(
                        feature=feature,
                        reason=(
                            f'reliability_{feature.reliability:.3f}_below_threshold_{threshold:.3f}'
                        ),
                    )
                )
            else:
                kept.append(feature)
        return kept, gated
