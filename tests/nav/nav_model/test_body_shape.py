"""Tests for ``nav.nav_model.body_shape``."""

from __future__ import annotations

from nav.nav_model.body_shape import (
    BODY_SHAPE_TABLE,
    DEFAULT_BODY_SHAPE,
    BodyShape,
    shape_for_body,
)


def test_body_shape_lookup_known_body() -> None:
    """A known body returns its specific entry."""
    assert shape_for_body('MIMAS') is BODY_SHAPE_TABLE['MIMAS']


def test_body_shape_lookup_case_insensitive() -> None:
    """Case-insensitive lookup matches an upper-case key."""
    assert shape_for_body('mimas') is BODY_SHAPE_TABLE['MIMAS']


def test_body_shape_lookup_unknown_body_falls_back_to_default() -> None:
    """Bodies absent from the table return ``DEFAULT_BODY_SHAPE``."""
    assert shape_for_body('NEW MOON') is DEFAULT_BODY_SHAPE


def test_default_body_shape_values() -> None:
    """The default body shape has its documented numeric defaults."""
    assert DEFAULT_BODY_SHAPE.ellipsoid_residual_km == 2.0
    assert DEFAULT_BODY_SHAPE.crater_scale_km == 5.0
    assert DEFAULT_BODY_SHAPE.albedo_variation == 0.15
    assert DEFAULT_BODY_SHAPE.spice_orbital_residual_km == 2.0
    assert DEFAULT_BODY_SHAPE.min_blob_diameter_px == 5.0


def test_body_shape_dataclass_is_frozen() -> None:
    """``BodyShape`` is frozen — assignment raises ``FrozenInstanceError``."""
    import dataclasses

    import pytest

    shape = BodyShape(
        ellipsoid_residual_km=1.0,
        crater_scale_km=2.0,
        albedo_variation=0.1,
        spice_orbital_residual_km=0.5,
    )
    with pytest.raises(dataclasses.FrozenInstanceError, match='ellipsoid_residual_km'):
        shape.ellipsoid_residual_km = 3.0  # type: ignore[misc]


def test_known_saturn_moons_share_shape_profile() -> None:
    """All major Saturn moons share the dedicated saturn-moon profile."""
    profile = BODY_SHAPE_TABLE['MIMAS']
    for body in ('MIMAS', 'ENCELADUS', 'TETHYS', 'DIONE', 'RHEA', 'IAPETUS', 'TITAN'):
        assert BODY_SHAPE_TABLE[body] is profile


def test_irregular_bodies_use_irregular_profile() -> None:
    """Hyperion and Phoebe share the irregular-shape profile."""
    assert BODY_SHAPE_TABLE['HYPERION'] is BODY_SHAPE_TABLE['PHOEBE']
    assert BODY_SHAPE_TABLE['HYPERION'].ellipsoid_residual_km == 10.0
