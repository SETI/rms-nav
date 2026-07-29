"""NavFeatureSummary — one entry per emitted feature in NavResult.feature_inventory.

Carries enough information about each NavFeature for the curator to write a
post-mortem entry into the per-image JSON metadata, including the gate
decision (kept vs dropped, reason).  It does not carry the heavy bits of a
full NavFeature (templates, polylines, covariance) because those would bloat
the metadata.
"""

from dataclasses import dataclass, field

from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType

__all__ = ['NavFeatureSummary']


@dataclass(frozen=True)
class NavFeatureSummary:
    """Per-feature post-mortem entry consumed by the curator.

    Parameters:
        feature_id: Unique identifier matching ``NavFeature.feature_id``.
            Must be a non-empty string.
        feature_type: One of the ``NavFeatureType`` values.
        source_model: Name of the producing NavModel (``'stars'``,
            ``'body:MIMAS'``, ``'rings:SATURN'``).  Must be non-empty.
        reliability: Self-assessed reliability score in ``[0, 1]``.
        gated: True if the reliability gate dropped this feature.
        gate_reason: Human-readable reason when ``gated`` is True; ``None``
            otherwise.  When ``gated`` is True the reason must be
            non-empty.
        bbox_extfov_vu: Half-open ``(v_min, u_min, v_max, u_max)`` bounding
            box in extfov coordinates; a 4-tuple of ints.
        reliability_reasons: Per-component breakdown of ``reliability``,
            copied from the emitting feature.  The scalar score alone says a
            feature was dropped but not what dragged it down, so the
            breakdown is what makes a gate decision attributable from the
            per-image metadata rather than only from a debug log.  Defaults
            to an empty breakdown (every component ``None``).

    Raises:
        TypeError: if any field has the wrong type.
        ValueError: if invariants on field values are violated.
    """

    feature_id: str
    feature_type: NavFeatureType
    source_model: str
    reliability: float
    gated: bool
    gate_reason: str | None
    bbox_extfov_vu: tuple[int, int, int, int]
    reliability_reasons: NavReliabilityBreakdown = field(default_factory=NavReliabilityBreakdown)

    def __post_init__(self) -> None:
        """Validate types and invariants on every field."""
        if not isinstance(self.feature_id, str) or not self.feature_id:
            raise ValueError(f'feature_id must be a non-empty string; got {self.feature_id!r}')
        if not isinstance(self.feature_type, NavFeatureType):
            raise TypeError(
                f'feature_type must be NavFeatureType; got {type(self.feature_type).__name__}'
            )
        if not isinstance(self.source_model, str) or not self.source_model:
            raise ValueError(f'source_model must be a non-empty string; got {self.source_model!r}')
        if isinstance(self.reliability, bool) or not isinstance(self.reliability, (int, float)):
            raise TypeError(f'reliability must be numeric; got {type(self.reliability).__name__}')
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f'reliability must lie in [0, 1]; got {self.reliability!r}')
        if not isinstance(self.gated, bool):
            raise TypeError(f'gated must be bool; got {type(self.gated).__name__}')
        if self.gated and not (isinstance(self.gate_reason, str) and self.gate_reason):
            raise ValueError('gate_reason must be a non-empty string when gated is True')
        if len(self.bbox_extfov_vu) != 4 or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in self.bbox_extfov_vu
        ):
            raise TypeError(
                f'bbox_extfov_vu must be a 4-tuple of ints; got {self.bbox_extfov_vu!r}'
            )
        if not isinstance(self.reliability_reasons, NavReliabilityBreakdown):
            raise TypeError(
                'reliability_reasons must be NavReliabilityBreakdown; '
                f'got {type(self.reliability_reasons).__name__}'
            )
