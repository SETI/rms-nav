<!-- Frozen snapshot 2026-07-25: method analysis and concept comparison feeding
the #60 Titan-navigation decision. Not maintained; disposition and the
implementation of record live in plans/TITAN_NAV_PLAN.md and issue #60. -->

# Titan Navigation Concept

Decision-support document for program-plan open decision "Titan
navigation: implement haze-limb navigation or scope it out?" (#60,
Track D gate; WS-7 in the validation plan). It explains why Titan
defeats the shape-based pipeline, documents the two prior algorithms in
the predecessor codebase (`/seti/all_repos/rms-csmithing`), explains the
published method of Hanson, French, et al. (2025), proposes a
SpinDoctor-native design, compares the approaches, and ends with a
recommendation.

Written 2026-07-25. This is a concept document, not an implementation
plan; if the decision is "implement," the accepted approach gets an
engineering-plan entry and issues in the normal way.

## 1. Why Titan defeats shape-based navigation

Titan is the only body in the supported missions' image sets whose
visible boundary is an atmosphere rather than a surface:

- **The visible "limb" is the haze top, not the solid limb.** The solid
  radius is 2575 km, but detached and main haze layers extend hundreds
  of km above it. A limb fit against the SPICE ellipsoid is
  systematically wrong by up to ~300 km (many pixels at typical
  resolutions), not merely noisy.
- **The apparent radius depends on wavelength.** Short wavelengths (UV,
  violet) scatter off high haze and show the largest disc; near-IR
  continuum filters penetrate deeper and show a smaller one. Every
  filter combination effectively images a different sphere, so no
  single "haze radius" constant fixes the problem.
- **In the methane-window filters the surface shows through.** Near
  938 nm (Cassini CB3, and to a lesser degree CB1/CB2), surface albedo
  features are faintly visible under the haze. This is extra signal in
  principle, but it also means the disc interior is not a smooth
  photometric function of scattering geometry.
- **At high phase Titan is not even a circle.** Forward scattering
  through the haze extends the bright limb past the terminator; near
  180 degrees phase the haze forms a ring around the whole disc.
- **The disc has latitudinal structure.** The seasonal north-south
  albedo asymmetry (whose symmetry axis is itself tilted about 4
  degrees from the spin axis; Roman et al. 2009) means disc brightness
  is not purely a function of incidence/emission/phase.

The interim handling is in place today: `NavModelBody` hard-excludes
Titan, `NavModelTitan` records the decline, and a Titan-only frame
fails with `titan_unsupported`. The `TITAN_LIMB` feature type is
reserved (never emitted) with an ensemble weight of 0.30 already
configured in `config_540_orchestrator.yaml`.

## 2. Prior art in rms-csmithing

The predecessor repo contains two distinct Titan algorithms sharing a
common front end. Neither is live: the `titan_navigate` call in
`navigation/nav/offset.py` is commented out (`# XXX IMPLEMENT TITAN`),
and the profile database file the mature variant needs is not present
on disk.

### 2.1 Common front end (both variants)

1. **Enlarged geometry body.** A copy of the Titan `oops.Body` is
   registered as "TITAN+ATMOSPHERE" with the radius grown by a
   configured `atmosphere_height` (default 700 km, tuned empirically by
   `experiments/analyze_titan_photometry.py`). All inventory and
   backplane work uses this envelope so the full haze is inside the
   modeled disc.
2. **Visibility gating.** Full atmosphere visible in the (extended)
   FOV earns full size-confidence; solid-body-only visibility is
   heavily discounted; Titan partially off-frame aborts.
3. **Symmetry axis.** The projected Sun direction at Titan's center is
   derived from the incidence-angle backplane (minimum incidence pixel
   below 90 degrees phase, maximum above). Absent clouds and surface
   features, haze brightness is mirror-symmetric about the line through
   the body center and the sub-solar point.
4. **Cross-track offset by mirror correlation.** The subimage around
   Titan is rotated so the symmetry axis is horizontal, flipped, and
   correlated against itself. The correlation peak measures the
   perpendicular (cross-track) mis-centering directly. This pins one of
   the two offset axes without any photometric model.

The two variants differ in how they pin the remaining, along-Sun-axis
("along-track") coordinate, which is intrinsically the harder axis (the
legacy code assigns it roughly 4x the cross-track uncertainty).

### 2.2 Variant A: breakpoint-scatter minimization (`nav/titan.py`)

Training-free. For a trial center, radial brightness profiles are
sampled along rays fanning +/-60 degrees around the Sun direction; each
ray's "breakpoint" is the outermost sample exceeding 5x the local
background. If the trial center is correct, the breakpoint sits at the
same radial distance on every ray; the along-track distance minimizing
the RMS scatter of breakpoints across rays is the answer. Assigns
anisotropic uncertainty (~0.02 x radius in pixels along-track,
~0.005 x radius cross-track) and a deliberately modest confidence
("Titan is never a great choice", hard-coded 0.4).

### 2.3 Variant B: profile-database matching (historical `cb_titan.py`)

The mature, scientific variant, recoverable from git history. Adds:

- **A precomputed baseline library** (`titan-profiles.pickle`): median
  radial I/F profiles per filter (24 supported combinations) per
  5-degree phase bin, built offline by `create_titan_profiles.py` from
  a large corpus of calibrated Cassini WAC images (minimum 5 profiles
  per bin, bins widened to 10 degrees when short, per-bin mutual 1-D
  alignment before the median).
- **A filter-translation table** mapping NAC, polarizer, and
  methane-band filters onto the WAC filters that have baselines, each
  at a reduced confidence (e.g. UV3 -> VIO at 0.20).
- **A Lambert seed.** Because symmetry search assumes the search box is
  on Titan, a Lambert render of TITAN+ATMOSPHERE is correlated first to
  get a coarse offset (skipped above ~145 degrees phase, where Titan
  looks nothing like a Lambert sphere).
- **1-D profile matching.** The observed along-track profile
  (resampled to a 1 km grid) is cross-correlated against the baseline
  for that filter/phase bin; the best shift is the along-track offset.
  Confidence scales with the number of images behind the bin and the
  bin width; a missing filter/phase bin aborts the technique.

### 2.4 What the legacy experience teaches

- The symmetry axis + separate along-track treatment is sound and was
  developed through two full iterations; it should be kept.
- The along-track axis is where all the difficulty lives.
- The profile database was Cassini-WAC-specific, expensive to build,
  incomplete in filter/phase coverage even for Cassini, and needed
  hand-tuned translation tables for every other filter. It never
  reached "on by default" status.
- Known hazards recorded in the code: the black void masquerading as a
  symmetry axis when the search box exceeds the disc; occluding rings
  or moons; Titan hanging off the frame edge; the Lambert seed failing
  at high phase.

## 3. The published method (Hanson, French, et al. 2025)

"The Evolution of Titan's Cold South Polar Cloud" (GRL,
doi:10.1029/2024GL113415) navigated the Cassini ISS south-polar imagery
with a then-unpublished algorithm ("the French method") developed as
part of separate work to renavigate all Cassini ISS images to
approximately single-pixel precision. The paper describes it as
follows:

> [W]e take advantage of the fact that, absent variations due to
> clouds or visible surface features, a hazy atmosphere has a line of
> symmetry along the diameter determined by the body center and the
> sub-solar point. By finding the image offset perpendicular to this
> line that maximizes the observed symmetry, we constrain the image
> along this direction even in the presence of minor variations. Once
> that constraint is available, we use the fact that the region of the
> limb in the direction of the subsolar point is approximately
> circular to find the center of the limb's arc along the axis of
> symmetry, without knowing the actual altitude of the haze layer.
> Using these two constraints, we derive the final navigated offset.

Restated in this document's terms:

1. A hazy atmosphere, absent cloud or surface variations, has a line of
   symmetry along the diameter through the body center and the
   sub-solar point.
2. The image offset **perpendicular** to that line is found by
   maximizing the observed mirror symmetry (the cross-track step shared
   with the legacy code), tolerant of minor symmetry-breaking
   variations.
3. The **along-axis** offset comes from the limb region in the
   direction of the sub-solar point: that sunward limb arc is
   approximately circular, so fitting the arc locates its center along
   the symmetry axis **without knowing the haze altitude** -- the arc
   radius is a free parameter of the fit.

Step 3 is the decisive idea. It replaces both the breakpoint heuristic
and the profile library with a geometric constraint that is inherently
filter-independent: whatever altitude a given filter's haze presents,
the arc it forms is still (nearly) a circle, and only its center
matters. The method demonstrated science-grade registration on real
data, including frames containing the polar cloud itself -- i.e. it
tolerated a symmetry-breaking feature in practice.

## 4. Proposed SpinDoctor design

### 4.1 Phase 1: geometric technique (modernized French method)

Implement the paper's two-constraint geometry as a first-class model +
technique pair:

- `NavModelTitan` stops being decline-only: it computes the
  TITAN+ATMOSPHERE envelope geometry, the projected Sun angle, phase,
  per-filter metadata, and occlusion masks (other bodies and rings in
  front of Titan, reusing the existing occlusion machinery), and emits
  a single `TITAN_LIMB` feature carrying them. The recorded-decline
  path remains for infeasible frames.
- A new `NavTechniqueTitan` ("TitanHazeNav") consumes the feature:
  1. **Seed**: predicted SPICE center (pointing error bounds from the
     existing per-instrument config), with an optional coarse
     correlation against a radially symmetric envelope render when the
     predicted error exceeds a fraction of the disc radius. No Lambert
     photometry; a smooth radial ramp is enough for a seed.
  2. **Cross-track**: mirror correlation about the symmetry axis, with
     three upgrades over the legacy code: subpixel peak interpolation
     (the legacy code rounded to integer pixels); the correlation
     restricted to a limb annulus rather than the full disc, so the
     north-south albedo asymmetry and surface-window features (which
     live in the disc interior) cannot bias it; and the symmetry-axis
     angle optionally refined as a free parameter within a few degrees
     of the SPICE prediction (absorbing the known ~4 degree tilt of
     Titan's atmospheric symmetry axis).
  3. **Along-track**: robust circle-arc fit to the sunward limb. Trace
     the maximum-gradient contour (or a family of isophotes) through a
     sector around the sub-solar direction, then fit a circle with
     center constrained to the symmetry axis and radius free, using an
     M-estimator (Tukey biweight, as in the DT techniques) so cloud
     features and image artifacts are downweighted. Fitting several
     isophote levels gives internal consistency and an empirical
     along-track uncertainty.
  4. **Result**: `(dv, du)` with an anisotropic covariance (along-track
     sigma reported honestly larger), feeding the ensemble through the
     already-reserved `TITAN_LIMB` weight.
- Feasibility gates (recorded as status reasons, not silent skips):
  minimum apparent diameter; full haze envelope inside the extended
  FOV; no occluder crossing the sunward limb sector; background sanity
  (Titan against dark sky, not ring material).

This phase needs no training data, works for every filter of every
supported instrument (Cassini ISS, Voyager ISS -- which imaged Titan as
a featureless orange ball -- and any future mission), and matches the
method with demonstrated ~1 px performance on real Cassini data.

### 4.2 Phase 2 options (only if validation demands them)

- **Self-calibrated haze-radius table.** Accumulate the fitted arc
  radius per (instrument, filter, phase bin) from Phase 1's own
  high-confidence solutions. Once populated, the expected radius
  becomes a strong prior (or a fixed value), turning Titan into a
  known-radius circle: the along-track fit tightens, small/partial
  discs become navigable, and the fit can even be delegated to the
  existing distance-transform limb machinery with a circular model.
  This is the profile-database idea reduced to one scalar per bin,
  self-generated by the pipeline instead of by an offline campaign.
- **Surface-window cartographic correlation.** For methane-window
  filters where the surface shows through (CB3 above all), high-pass
  filter the disc interior and correlate against a Titan surface
  basemap via the existing `create_cartographic_model()` path. This is
  the only approach that can beat limb-based accuracy at high
  resolution, but it depends on an external basemap, haze blurring, and
  per-filter contrast, so it is strictly a refinement stage layered on
  a Phase 1 solution.
- **Full profile-database port (contingency only).** Regenerating
  `titan-profiles.pickle` and the filter-translation machinery is the
  fallback if, against expectation, the arc fit cannot deliver
  acceptable along-track accuracy at extreme phase. Not recommended
  otherwise (see comparison).

### 4.3 Simulation and validation

- The simulator already has an atmosphere module
  (`src/spindoctor/sim/forward/atmosphere.py`); extend it to render a
  mirror-symmetric haze disc with a configurable radial profile,
  phase-dependent limb extension, and optional symmetry-breaking cloud
  blobs. That gives ground-truth harness coverage for both fit axes,
  including the failure modes (off-edge, occluded, void-as-symmetry).
- Port the annotated 87-image legacy test list
  (`rms-csmithing/tests/titan_images.txt`) as the real-image cohort:
  it already flags occlusions, high phase, and known-bad frames.
- Acceptance: cross-track error at or below ~1 px and along-track at or
  below a stated multiple (target 2-4 px) against ground truth on
  simulated frames and against star-anchored or mosaic-consistency
  truth on real frames, across a filter/phase matrix; every infeasible
  frame fails with a specific status reason. This satisfies the WS-7
  acceptance criterion ("real Titan frames navigate within a stated
  bound").

## 5. Comparison

| Criterion | Profile database (legacy B) | Breakpoint scatter (legacy A) | Geometric arc fit (paper / Phase 1) |
|---|---|---|---|
| Training data | Large calibrated Cassini WAC corpus, offline campaign, per-bin minimums | None | None |
| Filter coverage | Only populated bins; translation table hacks for the rest | All | All (radius is free per image) |
| Other instruments | Effectively Cassini-only | Yes | Yes |
| Along-track accuracy | Good where bins are deep | Moderate; depends on 5x-background heuristic | ~1 px class demonstrated in publication |
| High phase | Lambert seed breaks >145 deg; profiles thin | Background/breakpoint fragile | Sunward arc is the bright crescent; degrades gracefully |
| Surface windows / clouds | Contaminate profiles | Contaminate breakpoints | Downweighted by robust arc fit; interior excluded from symmetry |
| Maintenance | Heavy (regeneration, storage, versioning) | Light | Light |
| Reuse of SpinDoctor machinery | Little | Little | Robust-fit idiom shared with DT techniques; ensemble hook exists |

The breakpoint variant is strictly dominated by the arc fit: both are
training-free, but the arc fit uses the whole sunward limb coherently
with a principled geometric model instead of a per-ray threshold
heuristic. The profile database can, at its best, encode phase-function
information the geometric method ignores -- but only for Cassini, only
in populated bins, and at a data-engineering cost the legacy project
itself never finished paying (the technique ended its life commented
out, with the pickle missing).

## 6. Recommendation

**Implement the geometric method (Phase 1) as the sole Titan technique;
do not port either legacy variant.** It is the better method for this
codebase on every axis that matters here: no training corpus, uniform
filter and instrument coverage, published validation at the accuracy
class we need, and a clean fit into the existing model/technique/
ensemble architecture (the `TITAN_LIMB` hooks already exist).

"Implement both" is not warranted as a starting position. The profile
database's residual advantage (deep-bin along-track accuracy on
Cassini) is speculative against a demonstrated ~1 px geometric result,
and its costs are certain. The right sequencing is: build Phase 1,
validate against the WS-7 criteria, and let the measured along-track
residuals decide whether any Phase 2 option is worth building -- the
self-calibrated radius table being the natural first escalation because
Phase 1 generates its training data as a by-product of normal
operation.

If the operator instead chooses to scope Titan out, nothing here is
wasted: this document plus the capability matrix entry satisfy the
"scoped out honestly" branch of WS-7.

## 7. Risks and open questions

- **Symmetry-breaking scenes.** Large polar clouds or strong seasonal
  hemispheric contrast could bias the mirror correlation despite the
  limb-annulus restriction; the published method's success on
  cloud-bearing frames is encouraging but our robustness margins need
  measuring (simulated cloud injection covers this).
- **Small discs.** Below some apparent diameter the arc sector has too
  few pixels for a stable fit; the minimum-size gate needs calibrating,
  and sub-threshold frames stay `titan_unsupported` (or graduate to the
  Phase 2 radius-table + blob-style centroid path).
- **Extreme phase (>160 deg).** The disc becomes a thin ring; the
  "sunward arc" is bright but short, and the symmetry correlation has
  little area. The legacy code simply aborted; we should measure where
  the arc fit actually fails before choosing gates.
- **Ensemble interaction.** A Titan solution with strongly anisotropic
  uncertainty is a new shape for the ensemble; the reconciliation step
  must respect per-axis confidence rather than a scalar (design detail
  for the implementation plan).
- **Voyager radiometry.** Voyager Titan images are fewer and noisier;
  the technique should work unchanged, but validation data is thin.

## 8. References

- Hanson, L. E., French, R. S., Waugh, D. W., Barth, E. L., and
  Anderson, C. M. (2025). The Evolution of Titan's Cold South Polar
  Cloud. Geophysical Research Letters, 52, e2024GL113415.
  <https://doi.org/10.1029/2024GL113415> (open access; source of the
  geometric navigation method, Section 3 above).
- Roman, M. T., et al. (2009). Determining a tilt in Titan's
  north-south albedo asymmetry from Cassini images. Icarus.
  <https://doi.org/10.1016/j.icarus.2009.04.021> (the ~4 degree
  atmospheric symmetry-axis tilt).
- Legacy implementation: `/seti/all_repos/rms-csmithing`, files
  `navigation/nav/titan.py` (breakpoint variant, current tree),
  `navigation/cb_titan.py` (profile variant, git history, e.g.
  `git show 468414f:navigation/cb_titan.py`),
  `utilities/create_titan_profiles.py`,
  `experiments/analyze_titan_photometry.py`,
  `tests/titan_images.txt`.
- Current interim handling:
  `src/spindoctor/nav_model/nav_model_titan.py`,
  `src/spindoctor/config_files/config_060_titan.yaml`,
  `TITAN_LIMB` in `src/spindoctor/feature/feature_type.py` and
  `config_540_orchestrator.yaml`.
