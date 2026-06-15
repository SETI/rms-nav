star_dominated
==============

Scenes whose only navigable features are catalog stars.

Required:
- At least 3 catalog stars predicted to land in the extended FOV with
  predicted SNR above the per-instrument minimum.
- No body silhouette anywhere in the FOV.
- No ring edge or ring annulus visible.
- Smear length less than ~30 px (otherwise stars are unfittable).

Excluded:
- Any body in FOV (use stars_plus_body, body_*).
- Any ring feature in FOV (use ring_*).
- Saturation bloom across the whole frame.
- Frames where every catalog star has predicted SNR below threshold
  (use faint_stars).

Typical sources:
- Cassini ISS NAC star-calibration frames (Pleiades, Vega).
- NHLORRI cruise / approach star fields.
