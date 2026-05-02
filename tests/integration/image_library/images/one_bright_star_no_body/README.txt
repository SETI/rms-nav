one_bright_star_no_body
=======================

Catalog yields exactly one unambiguous star in the FOV. Tests the
1-star StarUniqueMatchNav path.

Required:
- Exactly 1 catalog star predicted in the FOV with SNR above
  threshold.
- The next-brightest catalog candidate is at least 1.5 magnitudes
  fainter, or absent from the FOV.
- No body silhouette in FOV.
- No ring annulus or ring edge in FOV.
- Star should not be saturated to the point of bloom dominating the
  frame.

Excluded:
- Two or more usable stars (use two_bright_stars_no_body or
  star_dominated).
- Next-brightest star within 1.5 mag (assignment ambiguous).
- Any body or ring in FOV.

Typical sources:
- Cassini ISS star-calibration frames (single named target).
- NHLORRI single-bright-star geometric-cal frames.
