high_phase_terminator
=====================

Body in crescent geometry with a usable terminator arc. Tests
BodyTerminatorNav as the primary technique.

Required:
- Single body in FOV (full or partial; can be regular or irregular).
- Phase angle greater than 90 degrees.
- Crescent must be wide enough that the terminator pixels are above
  noise (avoid one-pixel-thick crescents — the terminator arc must
  be fittable).

Excluded:
- Phase angle below 90 degrees (use body_full_fov,
  body_partial_overflow, or body_mostly_offscreen).
- Crescent so thin that no terminator pixels rise above noise (use
  negative_cases or pick a different frame).
- Multiple bodies (use multi_body — only one body in this class).

Typical sources:
- Cassini ISS approach phase (Saturn moons inbound).
- Galileo SSI Earth-departure crescent.
