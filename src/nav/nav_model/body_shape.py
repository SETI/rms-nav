"""Per-body shape parameters used by the body NavModel.

The body extractor's covariance and emission gates consult the per-body
shape, albedo, and SPICE-residual quantities held in
``BODY_SHAPE_TABLE``.  Numeric values are conservative defaults grounded
in published moon-shape papers (Thomas 2010 for Cassini moons, Stooke
1994 for irregular shapes).

Each entry is a ``BodyShape`` frozen dataclass:

- ``ellipsoid_residual_km`` — RMS deviation of the body silhouette from
  the best-fit ellipsoid (the bulk shape error).
- ``crater_scale_km`` — characteristic per-image limb roughness
  contribution from craters and topography.
- ``albedo_variation`` — fractional brightness variation across the disc;
  drives ``photometric_model_error_km`` for terminator scoring.
- ``spice_orbital_residual_km`` — SPK ephemeris uncertainty projected to
  the limb plane; ~0.5 km for a major moon.
- ``min_blob_diameter_px`` — minimum predicted disc diameter at which the
  ``BODY_BLOB`` feature is preferred over an unresolved limb.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'BODY_SHAPE_TABLE',
    'DEFAULT_BODY_SHAPE',
    'BodyShape',
    'shape_for_body',
]


@dataclass(frozen=True)
class BodyShape:
    """Per-body shape and SPICE-residual quantities.

    Parameters:
        ellipsoid_residual_km: RMS shape residual from the best-fit
            ellipsoid (km).  Used as the primary contribution in the
            limb-arc normal-sigma quadrature sum.
        crater_scale_km: Characteristic crater / topographic scale (km),
            independent of ``ellipsoid_residual_km``.  Used in the
            limb-arc normal-sigma quadrature sum to model
            below-ellipsoid surface roughness.
        albedo_variation: Fractional disc brightness variation in
            ``[0, 1]``; drives terminator-arc reliability.
        spice_orbital_residual_km: SPK ephemeris uncertainty in km.
            ~0.5 for major moons; up to 5 for irregular satellites.
        min_blob_diameter_px: Predicted disc diameter (px) at which the
            extractor stops emitting LIMB_ARC and switches to BODY_BLOB.
    """

    ellipsoid_residual_km: float
    crater_scale_km: float
    albedo_variation: float
    spice_orbital_residual_km: float
    min_blob_diameter_px: float = 5.0


DEFAULT_BODY_SHAPE: BodyShape = BodyShape(
    ellipsoid_residual_km=2.0,
    crater_scale_km=5.0,
    albedo_variation=0.15,
    spice_orbital_residual_km=2.0,
    min_blob_diameter_px=8.0,
)
"""Fallback shape used when a body has no specific entry.

The numbers reflect a generic small icy moon: ~2 km bulk-shape residual,
~5 km crater scale, modest albedo variation, generous 2 km SPK residual.
``min_blob_diameter_px`` matches the Part 5 default (``body_blob_min_px``).
"""


_SATURN_MOON_SHAPE: BodyShape = BodyShape(
    ellipsoid_residual_km=1.0,
    crater_scale_km=2.0,
    albedo_variation=0.10,
    spice_orbital_residual_km=0.5,
    min_blob_diameter_px=8.0,
)
"""Profile for the major Saturn moons whose shape is well-measured."""


_IRREGULAR_MOON_SHAPE: BodyShape = BodyShape(
    ellipsoid_residual_km=10.0,
    crater_scale_km=5.0,
    albedo_variation=0.20,
    spice_orbital_residual_km=2.0,
    min_blob_diameter_px=8.0,
)
"""Profile for very irregular bodies (Hyperion, Phoebe, etc.)."""


_GAS_GIANT_SHAPE: BodyShape = BodyShape(
    ellipsoid_residual_km=50.0,
    crater_scale_km=0.0,
    albedo_variation=0.30,
    spice_orbital_residual_km=10.0,
    min_blob_diameter_px=20.0,
)
"""Profile for gas / ice giants whose limbs are atmospheric scattering."""


BODY_SHAPE_TABLE: dict[str, BodyShape] = {
    'SATURN': _GAS_GIANT_SHAPE,
    'JUPITER': _GAS_GIANT_SHAPE,
    'URANUS': _GAS_GIANT_SHAPE,
    'NEPTUNE': _GAS_GIANT_SHAPE,
    'MIMAS': _SATURN_MOON_SHAPE,
    'ENCELADUS': _SATURN_MOON_SHAPE,
    'TETHYS': _SATURN_MOON_SHAPE,
    'DIONE': _SATURN_MOON_SHAPE,
    'RHEA': _SATURN_MOON_SHAPE,
    'IAPETUS': _SATURN_MOON_SHAPE,
    'TITAN': _SATURN_MOON_SHAPE,
    'HYPERION': _IRREGULAR_MOON_SHAPE,
    'PHOEBE': _IRREGULAR_MOON_SHAPE,
    'EUROPA': _SATURN_MOON_SHAPE,
    'IO': _SATURN_MOON_SHAPE,
    'GANYMEDE': _SATURN_MOON_SHAPE,
    'CALLISTO': _SATURN_MOON_SHAPE,
}
"""Static lookup of per-body shape parameters.

Keys are upper-case SPICE body names.  Bodies absent from the table use
``DEFAULT_BODY_SHAPE``.
"""


def shape_for_body(body_name: str) -> BodyShape:
    """Return the shape entry for ``body_name`` (case-insensitive).

    Parameters:
        body_name: Body name in any case (``'mimas'`` / ``'MIMAS'``).

    Returns:
        The matching ``BodyShape`` or ``DEFAULT_BODY_SHAPE``.
    """
    return BODY_SHAPE_TABLE.get(body_name.upper(), DEFAULT_BODY_SHAPE)
