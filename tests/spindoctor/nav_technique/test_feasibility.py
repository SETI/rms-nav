"""Tests for ``spindoctor.nav_technique.feasibility.NavFeasibilityReport``."""

from spindoctor.nav_technique.feasibility import NavFeasibilityReport


def test_navfeasibilityreport_feasible_default_count_zero() -> None:
    """A feasible report carries reason text and accepts default count zero."""
    report = NavFeasibilityReport(feasible=True, reason='3 STAR features available')
    assert report.feasible is True
    assert report.consumed_feature_count == 0


def test_navfeasibilityreport_infeasible_records_reason() -> None:
    """An infeasible report carries the rejection reason."""
    report = NavFeasibilityReport(feasible=False, reason='fewer_than_3_detected_sources')
    assert report.feasible is False
    assert report.reason == 'fewer_than_3_detected_sources'
