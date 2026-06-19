stars_plus_body
===============

One body and at least three usable catalog stars in the same FOV.
Tests the two-pass workflow (body offset on pass 1; star refine on
pass 2).

Required:
- Exactly one body in the FOV (any class except multi_body).
- At least 3 catalog stars predicted to land in the FOV with
  predicted SNR above the per-instrument minimum.
- Stars must not all fall on the body silhouette (the
  body-conflict-margin test would flag them).
- Body has a navigable feature (full limb, partial limb arc, BLOB).

Excluded:
- More than one body (use multi_body).
- Fewer than 3 usable stars (use one_bright_star_no_body or a body_*
  class).
- Rings in FOV (use ring_plus_body).

Typical sources:
- Cassini long-exposure background-stars frames.
- NHLORRI Pluto-system shots with field stars.
