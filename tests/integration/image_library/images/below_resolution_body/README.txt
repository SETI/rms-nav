below_resolution_body
=====================

A body too small in the FOV for limb fitting; the brightness-
weighted centroid (BLOB) is the only navigable feature. Tests
BodyBlobNav and the resolution gate that forces it.

Required:
- Single body with diameter under 15 px in the FOV.
- Body bright enough that a brightness-weighted centroid is well
  above noise (resolution gate triggers BLOB rather than failing
  outright).
- No other navigable features in the FOV (no usable stars, no
  rings).

Excluded:
- Body diameter 15 px or larger (use one of the body_* classes).
- Body too dim for a meaningful centroid (use negative_cases).
- Stars or rings also navigable in the same frame (use the matching
  composite class).

Typical sources:
- Voyager / Cassini long-range satellite frames (distant moons
  before encounter).
