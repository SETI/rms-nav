two_bright_stars_no_body
========================

Catalog yields exactly two unambiguous stars in the FOV. Tests the
2-star StarUniqueMatchNav path.

Required:
- Exactly 2 catalog stars predicted in the FOV with SNR above
  threshold.
- For each, the next-brightest competing candidate is at least 1.5
  magnitudes fainter or absent from the FOV.
- No body silhouette in FOV.
- No ring annulus or ring edge in FOV.
- Neither star is saturated such that its centroid is unrecoverable.

Excluded:
- Only one usable star (use one_bright_star_no_body).
- Three or more usable stars (use star_dominated).
- A bright pair where one or both have a comparably bright competitor
  within 1.5 mag.
- Any body or ring in FOV.

Typical sources:
- Cassini ISS / NHLORRI star-calibration frames covering two named
  targets.
