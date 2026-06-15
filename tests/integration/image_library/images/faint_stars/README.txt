faint_stars
===========

Frames where every catalog star in the FOV has predicted SNR below
the usable threshold. Tests that the reliability gate drops dim
stars cleanly without triggering false positives.

Required:
- At least one catalog star predicted in the FOV.
- Predicted SNR of every catalog star is below 3.0 (i.e. below the
  reliability-gate STAR threshold).
- The frame is otherwise navigable on a non-star feature (so
  expected.status is 'ok' or 'low' rather than 'failed') OR the
  scene is unnavigable and belongs in negative_cases instead.

Excluded:
- Any catalog star with predicted SNR above 3.0 (use
  star_dominated, one_bright_star_no_body, two_bright_stars_no_body,
  or stars_plus_body).
- Frames where no catalog star is even predicted in the FOV (no
  star reliability gate is exercised; pick a body / ring class).

Typical sources:
- Galileo SSI science frames (8-bit ADC, modest aperture).
- Voyager ISS outer-leg frames.
