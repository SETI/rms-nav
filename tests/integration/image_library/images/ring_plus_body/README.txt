ring_plus_body
==============

Saturn ring edge or annulus combined with at least one moon. Tests
the ensemble combination of body-derived and ring-derived offsets.

Required:
- At least one ring-edge polyline (curved or flat) visible in FOV.
- At least one moon visible in FOV (limb arc fittable, partial-disk,
  or BLOB regime).
- Both feature kinds are independently navigable.

Excluded:
- Edge-on rings where the ring projects to a single line through the
  body (hard to characterize separately).
- Body fully occluded by ring shadow.
- No body in FOV (use ring_only_curved or ring_only_flat).

Typical sources:
- Cassini Saturn shots with ring-shepherd moons (Prometheus,
  Pandora) or mid-orbit moons in the same frame as the ring ansa.
