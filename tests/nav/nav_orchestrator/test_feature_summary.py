"""Tests for ``nav.nav_orchestrator.feature_summary.NavFeatureSummary``."""

from nav.feature.feature_type import NavFeatureType
from nav.nav_orchestrator.feature_summary import NavFeatureSummary


def test_navfeaturesummary_kept_feature() -> None:
    """A non-gated feature has gate_reason None."""
    summary = NavFeatureSummary(
        feature_id='star:UCAC4:1',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        reliability=0.7,
        gated=False,
        gate_reason=None,
        bbox_extfov_vu=(0, 0, 16, 16),
    )
    assert summary.gated is False
    assert summary.gate_reason is None


def test_navfeaturesummary_gated_feature() -> None:
    """A gated feature carries the dropping reason."""
    summary = NavFeatureSummary(
        feature_id='limb_arc:MIMAS',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body:MIMAS',
        reliability=0.05,
        gated=True,
        gate_reason='visible_arc_fraction_below_threshold',
        bbox_extfov_vu=(100, 200, 300, 400),
    )
    assert summary.gated is True
    assert summary.gate_reason == 'visible_arc_fraction_below_threshold'
