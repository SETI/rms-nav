"""Tests for ``STATUS_REASON_INFO_TEMPLATE`` covering every NavStatusReason."""

from spindoctor.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE
from spindoctor.support.status_reason import NavStatusReason


def test_every_status_reason_has_template() -> None:
    """Every NavStatusReason value has an entry in the template."""
    missing = set(NavStatusReason) - set(STATUS_REASON_INFO_TEMPLATE)
    assert missing == set()


def test_template_lines_non_empty() -> None:
    """Each template's line list is non-empty."""
    for reason, lines in STATUS_REASON_INFO_TEMPLATE.items():
        assert lines, f'{reason!r} has empty template'


def test_template_covers_full_taxonomy() -> None:
    """Template covers the full NavStatusReason taxonomy."""
    assert set(STATUS_REASON_INFO_TEMPLATE) == set(NavStatusReason)
