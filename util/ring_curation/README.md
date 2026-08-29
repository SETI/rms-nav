# Ring-curation campaign tooling

Generates and tracks the operator-curated ring truth set: images chosen so
manual navigation covers every axis that drives ring-navigation behavior.

Geometry metadata alone cannot tell whether ring features are *visible* in
a frame (occlusion by Saturn, foreshortening, exposure, smear), so the
shortlist is built in two stages -- a geometry pool, then a visual screen
of every pooled image's holdings preview:

- `make_shortlist.py pool` writes `pool.csv`: the top candidates per grid
  cell, several deep, from the RMS Node ring-summary metadata
  (`COISS_2999_ring_summary.tab`; no SPICE needed).  Edge-on frames are
  excluded outright -- a ring seen edge-on has no radial features to
  navigate against.
- Every pooled image's preview is inspected visually; verdicts land in
  `screen.csv` (columns `image_id,verdict,reason`, verdict one of
  good / marginal / bad).  Usable frames also get a scene-background call
  in `background.csv` (`image_id,background,reason`: sky / mixed /
  planet), because a ring seen against Saturn's disk is a different
  photometric regime from one on dark sky.
- `make_shortlist.py select` fills the grid from screened candidates only
  and writes `shortlist.csv`.  Primary grid: ring region x
  radial-resolution band (8 x 6, one primary per cell), so every part of
  the ring system -- every feature type and catalog-uncertainty class,
  0.08 to 10.18 km rms -- is exercised at every scale.  Secondary axes:
  lit/unlit face, ring opening regime, NAC/WAC balance, scene
  background.  Targeted frames are preferred over whole-system
  panoramas; a planet-dominated frame is chosen only when a cell has no
  sky or mixed alternative (thin rings seen from the unlit side often
  show only in transmission against the disk); no candidate's truth
  comes from the spokes bundle, whose published pointing is not trusted.
  A cell with no visually usable candidate stays empty and is reported.
- `shortlist.csv` is the committed candidate list (`role` column: do the
  `primary:*` rows first).
- `screen.csv` and `background.csv` are the committed visual-screening
  records, kept so rejected frames and the reasons stay on file and the
  selection is reproducible.
- `annotate_sidecars.py` stamps each saved campaign sidecar with its
  shortlist row and reports which grid cells still lack truth.

Workflow per image:

    sd_offset coiss_saturn <image_id> --manual

then navigate, and use "Save as Library Entry...", pointing the save dialog
at `tests/integration/image_library/campaigns/ring_2026/`.  See that
directory's README for why sidecars stage there instead of enrolling
directly into `images/`.
