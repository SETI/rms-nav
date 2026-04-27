"""NavFeatureSummary — one entry per emitted feature in NavResult.feature_inventory.

Carries enough information about each NavFeature for the curator to write a
post-mortem entry into the per-image JSON metadata, including the gate
decision (kept vs dropped, reason).  It does not carry the heavy bits of a
full NavFeature (templates, polylines, covariance) because those would bloat
the metadata.
"""

from dataclasses import dataclass

from nav.feature.feature_type import NavFeatureType

__all__ = ['NavFeatureSummary']


@dataclass(frozen=True)
class NavFeatureSummary:
    """Per-feature post-mortem entry consumed by the curator.

    Parameters:
        feature_id: Unique identifier matching ``NavFeature.feature_id``.
        feature_type: One of the ``NavFeatureType`` values.
        source_model: Name of the producing NavModel (``'stars'``,
            ``'body:MIMAS'``, ``'rings:SATURN'``).
        reliability: Self-assessed reliability score in ``[0, 1]``.
        gated: True if the reliability gate dropped this feature.
        gate_reason: Human-readable reason when ``gated`` is True; ``None``
            otherwise.  Examples: ``'predicted_snr_below_threshold'``,
            ``'in_body_silhouette'``.
        bbox_extfov_vu: Half-open ``(v_min, u_min, v_max, u_max)`` bounding
            box in extfov coordinates.
    """

    feature_id: str
    feature_type: NavFeatureType
    source_model: str
    reliability: float
    gated: bool
    gate_reason: str | None
    bbox_extfov_vu: tuple[int, int, int, int]
