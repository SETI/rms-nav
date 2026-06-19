body_mostly_offscreen
=====================

A regular body mostly outside the FOV, leaving only a limb arc
fragment as the navigable feature.

Required:
- Single regular body.
- 50-90% of the disc off-frame (only 10-50% of the disc visible).
- A continuous limb arc spanning at least 10% of the full body
  circumference, fully inside the FOV.
- Phase angle below 90 degrees.

Excluded:
- 70-90% on-frame (use body_partial_overflow).
- No limb arc inside the FOV (use negative_cases).
- Crescent / phase greater than 90 degrees (use
  high_phase_terminator).

Typical sources:
- Cassini closest-approach NAC frames.
- Galileo close flybys (Io, Europa) at minimum range.
