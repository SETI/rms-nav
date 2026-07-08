"""Tests for ``spindoctor.feature.flags`` per-type flag dataclasses."""

from spindoctor.feature.flags import (
    BodyBlobFlags,
    BodyDiscFlags,
    CartographicModelFlags,
    LimbArcFlags,
    RingAnnulusFlags,
    RingEdgeFlags,
    StarFlags,
    TerminatorArcFlags,
)


def test_star_flags_defaults() -> None:
    """StarFlags has working defaults for every field."""
    flags = StarFlags()
    assert flags.saturated is False
    assert flags.smear_length_px == 0.0


def test_limb_arc_flags_carry_body_name() -> None:
    """LimbArcFlags stores the body_name string."""
    flags = LimbArcFlags(body_name='MIMAS', visible_arc_fraction=0.6)
    assert flags.body_name == 'MIMAS'
    assert flags.visible_arc_fraction == 0.6


def test_terminator_arc_flags_phase_angle() -> None:
    """TerminatorArcFlags carries phase_angle_factor."""
    flags = TerminatorArcFlags(
        body_name='ENCELADUS', visible_arc_fraction=0.4, phase_angle_factor=0.95
    )
    assert flags.phase_angle_factor == 0.95


def test_ring_edge_flags_polarity_predictable_default_false() -> None:
    """RingEdgeFlags defaults polarity_predictable to False (v1 invariant)."""
    flags = RingEdgeFlags()
    assert flags.polarity_predictable is False


def test_body_disc_flags_overflow_fraction() -> None:
    """BodyDiscFlags stores overflow_fov_fraction."""
    flags = BodyDiscFlags(body_name='RHEA', overflow_fov_fraction=0.1)
    assert flags.overflow_fov_fraction == 0.1


def test_body_blob_flags_predicted_diameter() -> None:
    """BodyBlobFlags stores predicted_diameter_px."""
    flags = BodyBlobFlags(body_name='PAN', predicted_diameter_px=4.0)
    assert flags.predicted_diameter_px == 4.0


def test_ring_annulus_flags_constituent_count() -> None:
    """RingAnnulusFlags counts constituent edges."""
    flags = RingAnnulusFlags(planet_name='SATURN', constituent_edge_count=5)
    assert flags.constituent_edge_count == 5


def test_cartographic_model_flags_mosaic_source() -> None:
    """CartographicModelFlags records the mosaic source identifier."""
    flags = CartographicModelFlags(body_name='MIMAS', mosaic_source='mimas_v1.npz')
    assert flags.mosaic_source == 'mimas_v1.npz'
