"""Tests for ``nav.feature.constants`` named-constant values."""

from nav.feature.constants import (
    AGREEMENT_FACTOR_CAP,
    COMBINED_CONFIDENCE_CAP,
    INCIDENCE_FACTOR_ANGLE_CAP_DEG,
    INCIDENCE_FACTOR_CLIP_DEG,
    JSON_INF_SENTINEL,
    MAX_INCIDENCE_FACTOR_CAP,
    MIN_ANISOTROPIC_SMEAR_PX,
)


def test_max_incidence_factor_cap_matches_80deg() -> None:
    """MAX_INCIDENCE_FACTOR_CAP equals the constant 4.76 used in source."""
    assert MAX_INCIDENCE_FACTOR_CAP == 4.76


def test_incidence_factor_caps_in_degrees() -> None:
    """Both incidence-factor cap angles are in degrees and ordered correctly."""
    assert INCIDENCE_FACTOR_ANGLE_CAP_DEG == 80.0
    assert INCIDENCE_FACTOR_CLIP_DEG > INCIDENCE_FACTOR_ANGLE_CAP_DEG


def test_agreement_factor_cap_value() -> None:
    """Agreement factor cap is 1.5."""
    assert AGREEMENT_FACTOR_CAP == 1.5


def test_combined_confidence_cap_value() -> None:
    """Combined confidence cap is 0.99 (never 1.0)."""
    assert COMBINED_CONFIDENCE_CAP == 0.99


def test_json_inf_sentinel_above_threshold() -> None:
    """JSON inf sentinel sits above 1e8 so consumers can detect it."""
    assert JSON_INF_SENTINEL >= 1.0e8


def test_min_anisotropic_smear_px_matches_plan() -> None:
    """Minimum anisotropic smear length is 0.5 px."""
    assert MIN_ANISOTROPIC_SMEAR_PX == 0.5
