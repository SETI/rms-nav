body_full_fov
=============

One regular ellipsoidal body filling most of the FOV with the full
limb in frame.

Required:
- Single regular body (Saturn moons, Galilean moons, Pluto, etc.;
  not a highly irregular shape).
- Body fills at least 70% of FOV area.
- Full limb visible — no edge of the disc clipped by the FOV
  boundary.
- Phase angle below 90 degrees.
- At least 30% of the visible disc is illuminated.

Excluded:
- Any portion of the limb cropped by the FOV (use
  body_partial_overflow or body_mostly_offscreen).
- Crescent geometry, phase greater than 90 degrees (use
  high_phase_terminator).
- Highly irregular body (use body_irregular).
- Body diameter under 15 px (use below_resolution_body).
- More than one body in the FOV (use multi_body).

Typical sources:
- Cassini ISS NAC mid-range satellites (Mimas, Enceladus, Tethys,
  Dione, Rhea).
- Galileo SSI Galilean-moon flybys.
- NHLORRI Pluto / Charon mid-approach.
