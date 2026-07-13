# SpinDoctor Engineering Plan (Tracks B-F)

*The implementation-detail companion to `plans/PROGRAM_PLAN.md` for
everything outside the validation and calibration science (Track A, which
has its own detail plan in `plans/VALIDATION_AND_CALIBRATION_PLAN.md`).
Each item below carries enough context — current behavior, files, fix
direction, constraints, acceptance — to be handed to a developer or an
implementing model with no other briefing beyond `/seti/newnav/CLAUDE.md`.
Issue numbers are the tracking source of truth; where this plan and an
issue disagree, update whichever is stale.*

Conventions that apply to every item here: project rules in
`CLAUDE.md` (line length, mypy strict, pdslogger-only logging, pytest
style, Conventional Commits); after any change that can move navigation
output, run the library cross-check
(`util/calibration/library_crosscheck.py`) and the sim integration suites
and account for every delta; never edit a library sidecar's `expected.*`
fields to match current behavior.

---

## Track B — Navigation correctness

Ordering within the track: #221 and #222 first (they are
confidently-wrong-answer defects and the agreement study will consume
ensemble output at scale), then the triage sessions (#237, #238), then
the investigation/design items (#179, #25, #128/#150), with the
smaller items (#24, #130, #132, #133, #180) as fill.

### #221 — Rank-1 ring result outvotes an absolute constraint

**Symptom** (found by the 2026-07-11 library cross-check): on flat-ring
frames carrying both a rank-1 `RingEdgeNav` result (constrains only the
ring-radial axis; the along-edge direction is unobservable) and an
absolute 2-D constraint (e.g. a `BodyBlobNav` moon fix), the fused result
follows the ring's unconstrained along-edge component and reports a high
confidence tier with 7-10 px of along-edge error.

**Where:** `src/spindoctor/nav_orchestrator/ensemble.py` — the agreement
grouping (Mahalanobis + pixel floor), the consensus-subset outlier
rejection, and the information-form combine. The rank-1 result's
along-edge variance is huge, so in information form it should contribute
almost nothing on that axis; the defect is in how agreement/consensus
treats the pair before the combine (a rank-1 result can "agree" with
anything along its null axis and so anchors a consensus it should not).

**Fix direction:** rank-aware agreement — when grouping, compare only on
the axes both results actually constrain (project the difference onto the
intersection of their observable subspaces); and/or a tier guard — a
fused result whose along-axis variance is dominated by a single rank-1
member cannot claim a tier better than that axis's honest sigma allows.

**Acceptance:** a regression test reconstructing the cross-check frame's
geometry (rank-1 ring + blob) asserts the fused offset tracks the
absolute constraint on the along-edge axis and the tier reflects the
honest per-axis sigma. Existing rank-1 tests
(`tests/spindoctor/nav_orchestrator/`, the `ring_only_flat` machinery)
stay green.

### #222 — Second-pass refinement votes as an independent opinion

**Symptom** (same cross-check): `StarRefineNav` in pass 2 runs from the
pass-1 consensus prior, lands near it (as it must — it is a local
refinement), and is then counted as independent corroboration. A wrong
pass-1 prior thereby gains confidence instead of being challenged; an
expected-failed frame reported success at high tier.

**Where:** the two-pass flow in
`src/spindoctor/nav_orchestrator/orchestrator.py` plus the ensemble's
membership rules in `ensemble.py`.

**Fix direction:** results seeded from a prior are correlated with that
prior; either exclude them from the agreement/consensus vote (use them
only to polish the winning group's estimate) or carry an explicit
provenance flag (`seeded_from_prior`) that the ensemble treats as
non-independent. The metadata should record the distinction either way.

**Acceptance:** a test where pass-1 produces a deliberately wrong prior
asserts the pass-2 refinement cannot raise the consensus confidence;
the expected-failed cross-check frame returns to failed.

### #237 — multi_body N17023890xx trio: all techniques spurious

One debugging session. Reproduce with the triage artifacts (metadata
JSONs and summary PNGs under the gitignored `_work/` triage output; the
frames re-run in ~35 s each with the local mounts per
`plans/COHORT_CURATION_PLAN.md` section 1). All techniques self-flagging
spurious on three consecutive multi-body frames suggests a shared cause:
prime suspects are inter-body occlusion handling and model conflict
marking (`src/spindoctor/nav_model/`), not three independent fit
failures. Outcome is either a fix plus regression test, or a documented
verdict that the frames are genuinely unnavigable (then they become
`negative_cases` candidates).

### #238 — Voyager scattered_light C00598xx quintet fails wholesale

One debugging session. Separate two hypotheses: (a) the Voyager
photometric path (stray-light gradient handling / DoG bandpass) breaks
navigable frames — look at the image-derivatives products and the star
model's gating on these frames; (b) the prescan admitted frames with no
navigable content — then the fix is the Stage-A criteria (the working
rule: stray-light gradient AND navigable content, prescan score >= 5 plus
>= 3 stars or a resolved ring/limb), and the verdict feeds
`util/cohort_curation/scan_stage_a.py`. Voyager runs need the
geometrically corrected (GEOMED) products, never raw.

### #179 — DT coarse-prior search vs competing edge populations

The coarse NCC search over the distance-transform image can lock onto the
wrong edge population (e.g. a ring edge when fitting a limb) and hand the
Levenberg-Marquardt refine an unrecoverable prior.
`src/spindoctor/nav_technique/dt_fitting.py` (`coarse_ncc_search`); the
minimum-support guard (#191) already landed, and the terminator technique
now has a second-minimum spurious gate (`find_secondary_dt_minimum`,
reusable) that closes its confident-wrong endpoint. What remains is a
calibration pass over the library: characterize when the coarse search's
top basins are ambiguous, and either widen the second-opinion gate to the
other DT techniques or add per-feature-type edge masking. Needs the
library cohort; coordinate with Track A so the fix is measured, not
guessed.

### #128 / #150 — Limb navigation redesign and the ~0.1 px systematic

The strategic pair behind several symptoms (the terminator
mis-convergence class, #187 chaotic rotators). #150 is the measured
~0.09-0.13 px limb bias: the
model predicts the geometric silhouette while the image's gradient ridge
sits ~0.1 px inside it (PSF), so `gradient_ridge_refine` is disabled for
the limb technique (`config_510_techniques.yaml`) while ring edges run
with it on. Candidate fixes (validation plan WS-10): forward-model the
PSF-inward offset in `src/spindoctor/nav_model/body/nav_model_body.py`,
or replace the integer-quantized DT with a continuous/interpolated one.
**Constraint:** do not touch until the real-image measurement exists
(Track A #225/#227 provide it) — the current partial cancellation is
accidental and a well-meaning "fix" can make real accuracy worse. #128 is
the fuller redesign (all body types and illuminations) and starts with a
design document, not code.

### Smaller Track B items

- **#25** — high-resolution bodies: the model renders sharper than the
  PSF-blurred image; investigate blur-matching the model
  (`nav_model_body.py`) at high resolution. Investigation first: measure
  whether it actually moves offsets on library close-flyby frames.
- **#24** — remove fuzzy/non-spherical bodies from techniques that assume
  a clean ellipsoid silhouette; the body-shape table
  (`config_220_body_shape.yaml`) now has real values to gate on.
- **#130** — per-instrument star limiting magnitudes measured from real
  star fields (a small campaign over the library's star classes;
  coordinate with #233's measured-SNR work — same frames, same tooling).
- **#132** — star-field rotation variance assumes isotropic residuals
  (up to 2x off); derive the anisotropic form in
  `nav_technique_star_field.py`.
- **#133** — star inlier matching is greedy/order-dependent; switch to
  optimal assignment (scipy `linear_sum_assignment`) if profiling says it
  is affordable.
- **#180** — wire `STATUS_REASON_INFO_TEMPLATE` through every
  `NavResult.failed` site so each failure carries a per-image reason
  (`src/spindoctor/nav_orchestrator/`); cheap, and it makes every other
  item in this track easier to debug — schedule it first among the small
  items.
- **#239** — implement whichever sub-5 px body policy the operator picks
  (see PROGRAM_PLAN decision gates).

## Track C — Statistics and QA

- **#240** — coverage-matrix invariant test: every scene class >= 2
  sidecars, every autonomous technique the expected primary somewhere.
  Lives with the library structural tests
  (`tests/integration/test_image_library.py`); runs in the deliberate
  tier, marked expected-incomplete until #235 fills `faint_stars` and
  `ring_only_flat`.
- **Standing practice** — after any calibration- or technique-affecting
  merge: `util/calibration/library_crosscheck.py` over the full library,
  every per-image delta accounted for; `sd_stats_ingest` /
  `sd_stats_report` over campaign outputs as the accuracy checkpoint
  (both from the statistics system).

## Track D — Capability completion

### PDS4 output bundles (required for all four instruments)

PDS4 *output* (bundle generation) and PDS4 *input* (reading
PDS4-archived data as a dataset source) are different things. Output
bundles are mandatory for every instrument. Input is
availability-contingent: no PDS4 archive of these datasets exists yet,
producing one is external development outside this project's control,
and input support (#34, `dataset_pds4.py`) is not required for project
completion — when an archive appears, implementing its `DataSetPDS4`
replaces the PDS3 source for that instrument.

Output current state: nothing works end to end yet. The Cassini path is
partially implemented — the per-dataset hook pattern (template dir,
LID/LIDVID builders, template variables) exists on
`DataSetPDS3CassiniISS` and the collection machinery runs — but it has
no final templates, zero tests (#242), and no schema validation, so its
output is unvalidated. The other three instruments additionally hit
`NotImplementedError` walls in their `pds4_*` DataSet hooks. The work
is therefore: finish and validate Cassini first (final templates,
tests, schema validation), then generalize — per-mission template trees
plus hook implementations, mechanical but voluminous.

Work items, in dependency order:

1. **#139** (Essential) — the global-index
   LID is malformed (missing `urn:nasa:pds:` prefix, wrong image part):
  `src/spindoctor/cli/pds4/collections.py`. Fix with a label-round-trip
   test.
2. **#69, #30** — backplane FITS description in data labels; backplane
   label design (couples to the #55 backplane-set decision).
3. **#79** — scrape PDS4 context products for targets (feeds #73).
4. **#71-#76, #47** — label/collection completeness items, each small:
   parameterized bundle name/version, target handling, ring geometry
   class fields, global-index labels, collection CSVs, ring incidence
   angle.
5. **#66** — integrity-checking pass over a generated bundle.
6. **#67** — cloud-aware bundle generation (with the Track D cloud
   audit).
7. Schema-validate generated `.lblx` against the PDS4 schemas in CI for
   all four instruments (acceptance for the whole family).

### Backplane family (decision: #28 scope)

Issues #55 (final backplane set) and #57 (FITS HDU content) are
decisions that gate #54 (cropping), #77 (optional args), and #63 (bodies far from
planets). The generator machinery exists
(`src/spindoctor/cli/backplanes/`); tests are Track E #241. End-product
value correctness is Track A #232.

### CK kernels (#188, prerequisite #50)

The headline "updated pointing" deliverable: write SPICE C-kernels
carrying the navigated attitude. #50 (switch to using the C matrix)
comes first and is small; #188 needs a design note (kernel granularity,
segment metadata, provenance linkage to `_metadata.json`) the operator
should see. Validation: round-trip a written kernel through oops and
confirm the reprojected geometry matches the navigated offset.

### Capability matrix (#231)

Generated/test-verified matrix; see the issue for the two-axis design
(feature support x validation status). Implementation home:
`docs/user_guide/` page generated from the registries
(`spindoctor.dataset`, `spindoctor.obs`, technique registry) plus a
static validation-status table that the WS reports update; a test
asserts the generated half matches the registries.

### Cloud and scale

- **#108** — audit every `sd_*` CLI for logging, cloud operation, and
  working `cloud_tasks` variants; fix what the audit finds.
- **#141 / #142** — dedup the CLI driver preamble and cloud-task loop;
  fix the dropped `extra_params` and the ImageFiles cardinality
  disagreement.
- **#236** — profiling + supported batch-parallel path (issue has the
  breakdown; respects #103/#134 thread-safety constraints — per-thread
  `Backplane` objects, no shared `obs`).
- **#126** — rotation-pyramid cost (~10 min on 1024^2); only bites
  rotation-fitting instruments (Galileo, Voyager); coordinate with
  Track F instrument enablement.
- **#118** — config validation system: schema per section, unknown-key
  rejection, type/range checks at load
  (`src/spindoctor/config/config.py`); pairs well with #176's
  constants-into-config completion.

### User-facing docs in Track D scope

- **#93** — the four instrument appendices (write Cassini's from the
  measured Track A results first; the other three land with Track F
  enablement).
- **#70** — supplemental-metadata file format documentation.

## Track E — Test and documentation debt

- **#241 / #242** — unit tests for `spindoctor.cli.backplanes` and
  `spindoctor.cli.pds4` (zero today). Mock the navigation inputs with
  fixture metadata; test arg parsing, product layout, error paths, LID
  construction. Do these before Track D's PDS4/backplane build-out.
- **#243** — direct tests for `nav_model/stars/conflicts.py`
  (`_check_one_star`, `mark_body_and_ring_conflicts`) with synthetic
  geometry; the existing per-pixel occlusion tests cover only the
  occlusion-fraction path.
- **#174** — regression baselines beyond the single frame: seed
  baselines for the full library (`python -m
  tests.integration.update_baselines --all`) and commit them with the
  per-image diff accounted for.
- **#177** — unit tests for `spindoctor.support.summary_png`.
- **#178** — missing dev-guide pages: filters, uncertainty (write
  after #230 makes sigmas load-bearing), troubleshooting.
- **#129** — drive Sphinx nitpicky warnings to zero, then add `-n` to
  the CI docs build.
- **#122** — verify the albedo/terminator-sharpness rationale in the
  body-terminator dev guide against the shipped implementation.
- **#244** — api_reference pages for the missing `mosaic_viewer`
  modules (follow the existing PyQt6-safe autodoc pattern).
- **#245** — self-contained language pass over `util/cohort_curation/`
  (drop the internal phase codenames; point at the committed docs).

## Track F — Instruments, features, hardening

### Instrument enablement (Phase-2 of the old roadmap)

Start after Track A's Cassini verdict; per instrument the pattern is:
fix ingest/navigation defects, add library frames (#235), extend the
calibration (#230) and the appendix (#93).

- **Voyager ISS:** #19 — star navigation broken; overlaps the distortion
  (#228) and limiting-magnitude (#130) work, so schedule together.
  Rotation fitting is currently off for cost (#126).
- **Galileo SSI:** #18 — star navigation broken (same cluster); #17 —
  REDO product handling in the dataset layer.
- **New Horizons LORRI:** #2 — PSF sigma calibration; #138 — decide and
  enforce the `_eng` product policy; #33 (deferred) — new instrument
  kernel.
- **Ring models:** #82 Jupiter, #81 Uranus, #83 Neptune — extend the
  per-planet ring catalogs (`config_3N0_*_rings.yaml`) and the ring
  model's edge selection; Voyager/Galileo frames exist in the archive
  scan for all three.
- **#181** — image-degradation classifier classes; taxonomy design
  first, patterns are largely Voyager/Galileo-specific.

### Features

In rough priority order: #27 BOTSIM (NAC/WAC simultaneous), #22 star
streaks, #107 backplane-reader companion repo, #34 PDS4 input (when
external archives exist — replaces the PDS3 source per instrument; not
required for project completion), #184
cartographic/bootstrap navigation (crater-mapping correlation of
overlapping navigated frames — explicitly far off; design record in the
issue), #183 polarity-aware ring matching, #187 chaotic-rotator
(Hyperion) pose handling, #186 manual-nav dialog redesign, #185
gated-feature styling on summary PNGs, #182 stop-after-features
flag, #23 body shape models (topographic meshes; also feeds Track A's sim
realism), sim polish: #84 ring edges/gaps overwrite, #78 CraterMaker
craters, #151 flux-correct star smear, #152 diffraction spikes, #157
line-based missing data, #158 smooth-shaded meshes.

### Hardening / cleanup tail

Mostly small, any time, no ordering: #13 SCET strings, #15 overlay
occlusion of background models, #21 metadata inventory cleanup, #38
filecache config, #39 AttrDict, #43 `--pds3-holdings-root`
placement, #65 exception class, #92 dependency groups, #96 dead code, #97 oversized
modules, #98 registry consolidation, #99 orphan report_profile, #100
root-path getters, #101 ArgumentParser.error, #102 CLI globals, #103
thread-unsafe caches, #104 broad excepts, #105 typed interop
boundaries, #109 safe-path helpers, #110 scalar validation helpers, #119 PNG
creation location, #135 from_file dedup, #137 dead validation
helper, #140 geometry-union access, #143 viewer cursor after pan, #144
QApplication lifetime, #147 confidence-context dedup, #155
display-scaling consolidation, #212 xdist worker nondeterminism
(software-only scope; the faulty CPU cores are permanently offlined and
nothing gates on hardware).
