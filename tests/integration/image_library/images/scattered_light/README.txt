scattered_light
===============

Frames with a visible stray-light gradient that violates uniform-
illumination assumptions. Tests the per-instrument BANDPASS_DOG
source-image filter.

Required:
- A clearly visible stray-light gradient across the frame (one edge
  brighter than the other beyond what the actual scene contributes).
- Source camera is one whose optics are known to flare when the sun
  is near the FOV — Galileo SSI or Voyager ISS.
- Some other navigable feature is present (body, ring, or stars) so
  the test exercises the filter doing useful work.

Excluded:
- Frames with no visible stray-light gradient (the filter would
  no-op on these).
- Cassini ISS or NHLORRI frames (those instruments do not flare
  comparably; use one of the body / star classes).

Typical sources:
- Galileo SSI Earth-departure outer fields.
- Voyager ISS Saturn-encounter outer-leg frames.
