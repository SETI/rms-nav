body_partial_overflow
=====================

One regular body partly off-frame, with a substantial limb arc still
visible in the FOV.

Required:
- Single regular body.
- 70-90% of the disc visible inside the FOV (10-30% off one or two
  edges).
- A continuous limb arc spanning at least 30% of the visible portion
  of the limb.
- Phase angle below 90 degrees.

Excluded:
- Less than 50% of the disc visible (use body_mostly_offscreen).
- Full limb visible (use body_full_fov).
- Crescent / phase greater than 90 degrees (use
  high_phase_terminator).
- More than one body (use multi_body).

Typical sources:
- Cassini close encounters (Mimas, Enceladus inbound).
- Galileo SSI flybys mid-encounter.
