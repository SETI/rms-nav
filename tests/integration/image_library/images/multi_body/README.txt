multi_body
==========

Two or more separable bodies in the FOV, exercising joint geometric
constraint.

Required:
- At least 2 distinct bodies visible.
- Bodies do not occlude each other (no overlapping silhouettes).
- The predicted disks do not touch — there is clear sky between them.
- At least one body has a navigable feature (full limb, partial limb
  arc, or BLOB regime).

Excluded:
- One body occulting another (occlusion is exercised separately).
- A single body plus stars (use stars_plus_body).
- A single body plus rings (use ring_plus_body).
- Bodies whose predicted disks merge or touch.

Typical sources:
- Cassini Saturn-system family portraits.
- Galileo Jupiter-system multi-moon shots.
