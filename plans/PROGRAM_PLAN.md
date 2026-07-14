# SpinDoctor Program Plan

*The top-level plan of record for all remaining work. It is written to be
readable without knowledge of the code internals or the statistical
methodology; the detail lives in the three sub-plans it points to. Last
reconciled 2026-07-12.*

**Document map** (what to read for what):

| Document | Role |
|---|---|
| `plans/PROGRAM_PLAN.md` (this file) | What remains, why, in what order, and what can run in parallel. Every open issue is accounted for in the index at the end. |
| `plans/VALIDATION_AND_CALIBRATION_PLAN.md` | Full methodology for Track A (the science): what "validated" and "calibrated" mean, the statistical machinery, acceptance criteria per workstream. Binding where it overlaps anything else. |
| `plans/ENGINEERING_PLAN.md` | Full implementation detail for Tracks B-F: per-item context, file pointers, constraints, and acceptance criteria sufficient to hand any item to a developer (human or model) cold. |
| `plans/COHORT_CURATION_PLAN.md` | The operational playbook for growing the curated image library: metadata-driven discovery, operator review votes, sidecar generation. |
| `plans/archive/` | Superseded plans, kept as historical records with dates in their filenames. Nothing in there is current. |

---

## 1. The goal

One sentence: **an end user can take raw archival images from Cassini ISS,
Voyager ISS, Galileo SSI, or New Horizons LORRI and produce — locally or in
the cloud — precisely navigated pointing, reprojected mosaics, per-pixel
geometry backplanes, archive-quality PDS4 bundles, preview images and
metadata, and updated SPICE pointing kernels, with every accuracy and
confidence number backed by published evidence and every capability
documented.**

The end-of-project deliverable therefore has two halves:

1. **A finished pipeline** — every stage works for every supported
   instrument, runs at campaign scale, and is protected by tests.
2. **A defensible accuracy story** — the pointing errors and confidence
   values the pipeline reports are calibrated against evidence a reviewer
   can check, and a capability matrix states exactly what is supported and
   how well, per instrument.

The second half is not polish. This system's outputs are scientific
measurements; a navigation system that reports offsets without defensible
uncertainties is not finished, it is merely running.

## 2. Where we stand

The engineering core is built and healthy: the full navigation architecture
(nine autonomous techniques plus manual), reprojection, backplanes,
simulation, rank-1 (single-axis) ring support, the statistics system
(`sd_stats_ingest` / `sd_stats_report`), and strict quality gates (typing,
linting, ~2,600 tests) are in place. PDS4 bundle generation exists only
as partially implemented machinery (see Track D), though its generator
backend is now spec-tested (PR #257). An independent project review
(2026-07-08) rated the engineering strong and the self-assessment
honest.

What the same review identified as the single real deficit still stands,
half-addressed: **the validation program is designed but mostly not yet
run.** The first slice is in place — the confidence formulas are calibrated
against simulated scenes with planted, known-truth offsets, so confidence
values are no longer arbitrary defaults. But the simulation's realism is
itself unproven, which is why every output still carries a
`confidence_provisional` marker. Turning that provisional story into a
defensible one is Track A, the largest remaining block.

The curated image library — the raw material for both regression testing
and validation — stands at 69 operator-verified images (predominantly
Cassini, with a few Voyager, Galileo, and New Horizons), against a
first-stage budget of 47 spanning 17 scene classes and a final target of
at least 120 across all four instruments. Review batch 5 (PR #260, six
frames) merged and filled the last empty class, `ring_only_flat`; the
operator library review (branch `phase-d-reconciliation`, PR #262)
re-verified five frames, added a sixth, and applied the tier ratchet
across the library. Batch 5 also queued manual-nav frames toward the
per-class minima.

## 3. Why validation dominates the remainder (plain-language version)

No independent record exists of where these cameras were truly pointing —
not to the hundredth-of-a-pixel level the system reports. So accuracy
cannot be checked against an answer key. The program substitutes three
mutually reinforcing sources of evidence:

1. **Simulation with planted truth** — render a synthetic image where the
   correct answer is known by construction, and measure recovery error.
   Trustworthy only if the simulator is (a) independent of the navigator's
   own models (today they share rendering code, so good scores partly
   grade the navigator's homework with its own answer sheet) and (b)
   demonstrably realistic compared to real images. Fixing both is the
   heart of Track A.
2. **Agreement between independent methods on real images** — when a star
   field and a moon's edge independently yield the same pointing
   correction to within a fraction of a pixel, that agreement is evidence
   both are right to about that level. This is the only accuracy signal
   available from real archival frames, and extracting it honestly
   requires real statistical care (the methodology plan covers the traps).
3. **A large, deliberately diverse library of operator-verified images**
   — feeding both of the above and serving as the permanent regression
   safety net.

Everything else in the program — new instruments' quirks, PDS4 bundle
generalization, performance, documentation — is real work but
conventional engineering.
This is the part that makes the numbers mean something.

## 4. The tracks

Work is organized into six tracks. A track is a stream of related work
that can proceed largely independently of the others; the parallelism
notes say where they touch.

### Track A — Validation and calibration (the science half)

**Goal:** every accuracy and confidence number the pipeline emits is
backed by published evidence; the `confidence_provisional` marker is
retired.

**Why:** section 3. This is the critical path to the deliverable's second
half and the majority of the remaining effort.

**Shape of the work** (methodology and acceptance criteria in
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`; the workstream codes below are
its section names):

1. **Grow the image library** (#172 first stage, #235 growth; WS-3) —
   continuous background work: automated candidate discovery, operator
   votes in batches, sidecar generation. Feeds everything below.
2. **De-circularize the simulator and prove it realistic** (#227,
   with #223, #153, #84; WS-2) — the largest single item and the
   highest-risk one: separate the simulator's rendering from the
   navigator's models, then show statistically that simulated frames look
   like real ones. Has genuine design choices; the operator approves the
   design before build.
3. **Prove the agreement estimator** (#224; WS-0) — before trusting
   per-technique error numbers extracted from cross-technique agreement,
   prove on known-truth simulations that the extraction math works and
   map where it is even solvable.
4. **Validate camera distortion models** (#228; WS-17) — otherwise
   distortion masquerades as navigation error in the agreement study.
5. **The agreement study itself** (#225, corroborated by #226; WS-1,
   WS-1b) — the flagship: run the pipeline over hundreds to thousands of
   real frames with two or more independent fiducials and publish the
   agreement statistics. Its bulk (pairwise) layer needs only the library
   and distortion validation; it must not wait for items 2-3, which gate
   only the finer per-technique separation.
6. **Wire real images into CI** (#229; WS-4) — a small cached tier on
   every PR, the full suite on a schedule.
7. **Re-anchor confidence on real evidence** (#230; WS-5) — re-run the
   existing calibration tooling against the agreement study's
   measurements; retire the provisional marker.
8. **Close the accuracy tail** (#233 measured star SNR and constant
   sensitivity, WS-9; #150/#128 the known ~0.1 px limb bias, WS-10; #234
   realistic noise for calibrated images, WS-13; #232 end-product
   accuracy for backplanes/mosaics/PDS4 values, WS-18).

**Operator's role:** batch votes on library candidates (ongoing); approve
the simulator design (item 2); bless the realism verdict; approve
agreement-study frame selection; re-bless tiers after item 7. Everything
else is agent-executable.

**Parallelism:** items 1-4 all start immediately and run concurrently;
item 5's bulk layer starts as soon as 1 and 4 give it cohorts; 6 rides
alongside; 7-8 close out in sequence after 5.

### Track B — Navigation correctness (remaining)

**Goal:** no known case where the navigator returns a confidently wrong
answer or fails on a navigable scene.

**Why:** these defects poison the validation data (Track A consumes the
navigator's output at scale) and are exactly what a user hits first.
The known defects:

- **#221** — a one-axis ring measurement outvotes an absolute
  position constraint, reporting a wrong answer at high confidence. Top
  of the track: it is a tier-honesty defect.
- **#222** — second-pass star refinement corroborates its own first-pass
  input but votes as an independent opinion, inflating consensus
  confidence. Same class of defect. Two real-frame instances found in
  Phase D (N1686349893, N1572105349): the 1-star refine degrades an
  otherwise-correct body fix to ~1.8 px error while keeping a high tier.
- **#258** — an exact recovery is downgraded to `conflicted` by a
  low-confidence dissenter the consensus logic has already excluded
  (two stars_plus_body frames); the agreement-gap test still counts the
  excluded member. Same ensemble cluster as #221/#222.
- **#259** — a one-star match with an 18 px residual passes every gate
  on a negative-case Galileo frame (no residual gate on the one-star
  path; the #211 ambiguity gate is vacuous with no rival). Confidently
  wrong on an unnavigable scene.
- **#261** — the DT mis-convergence gate false-flags a correct
  RingEdgeNav fit spurious (per-edge median driven by one poor edge in a
  multi-edge fusion); the frame navigates but the pipeline discards its
  own correct high-confidence result. Concrete library datapoint for
  #179.
- **#263** — the single-inlier confidence cap (0.50) collides exactly
  with the high-tier confidence threshold (0.50), so a one-star, no-
  cross-check solution capped low *because it is weak* nonetheless earns
  the high tier. Tier-honesty defect in the same family as #221/#222.
- **#179** — the coarse search can lock onto the wrong edge population;
  needs a calibration pass against the library (feeds, and is fed by,
  #261).
- **#128 / #150** — the strategic limb-navigation redesign and the ~0.1 px
  limb systematic (shared with Track A's WS-10; design first, validate
  against real images before touching).
- **#25** — model blurring for very-high-resolution bodies (investigation).
- **#237 / #238** — two unexplained triage failures (a multi-body trio, a
  Voyager scattered-light quintet); each is one debugging session.
- **#239** — operator decision: how to treat bodies smaller than ~5 px.
- **#210** — the NCC techniques' covariances are orders of magnitude
  over-tight; the covariance-model review remains open even though the
  original coverage symptom is fixed.
- **#24, #130, #132, #133, #180** — smaller technique-quality and
  diagnosability items (#180 wires a per-image reason through every
  failure site — cheap and it makes debugging the rest of the track
  easier, so do it early).
- **#254** — a fully dark body emits a photometric BODY_BLOB feature
  where the body-model dev guide says it should emit nothing; likely
  harmless today (the reliability gate culls it) but it is a
  model-emission spec conflict to resolve.

**Parallelism:** fully parallel with Track A; #221/#222 should land
before the agreement study consumes ensemble output at scale.

### Track C — Statistics, QA, and the accuracy checkpoint

**Goal:** navigation quality is continuously measured, not assumed.

The statistics system (`sd_stats_ingest` / `sd_stats_report`) is built,
tested, and documented in the user guide. Remaining: the library
coverage-matrix invariant (#240), and the standing practice of re-running
the library cross-check after every calibration-affecting change. Small
track, mostly done, listed separately because it is the program's QA
instrument.

### Track D — Capability completion (decision gates first)

**Goal:** every capability the docs imply either works and is validated,
or is explicitly scoped out in the capability matrix.

Some items start with an operator decision, because each is a scope
commitment:

| Decision | Then the work is |
|---|---|
| **Titan navigation** (#60): implement atmospheric-limb navigation or scope it out? | A new haze-limb model and technique (hard, physics-heavy), or graceful degradation plus honest docs. |
| **CK kernels** (#188, prerequisite #50): ship updated-pointing SPICE kernels as a product? | The kernel writer and its validation — a headline deliverable either way. |
| **Backplane content** (#28 family): finalize the backplane set and formats | #55, #54, #57, #77, then the generator hardening (including the product-correctness defects #251, #252, #253 found by the #241 test suite). |

**PDS4 output bundles are required for all four instruments** — not a
scope decision — and **none of it works today**. The Cassini path is
partially implemented machinery with no final templates, no tests, and
no validation; Voyager, Galileo, and New Horizons additionally hit
not-implemented walls. The work is: finish and validate the Cassini
path (final templates — acceptance list recorded on #53; schema
validation; the interacting LID defects #139/#256 fixed in PR #264,
leaving two characterized #256 defects — swallowed `template.write`
errors and the dev-guide output-layout mismatch), then generalize —
per-mission label templates, LID builders, and collection machinery
(#53 with #66, #67, #69, #71-#76, #79, #47, #30, #63). Distinct from this, **PDS4 *input***
(#34) — reading PDS4-archived data instead of PDS3 — is treated like any
other future instrument: the archives do not exist yet, their creation
is external development outside our control, and input support is *not*
required for project completion; when an archive appears, its support
replaces the PDS3 source for that instrument.

Plus, not gated on decisions: the capability matrix itself (#231),
cloud-operation audit (#108, #67, #141, #142), performance and safe
parallelism for campaign scale (#236, #103, #134, #126), config
validation (#118), and the user-guide completion items (#93, #70).

**Parallelism:** decisions can be made any time; the resulting work is
independent of Tracks A-B except that #232 (end-product accuracy) wants
the backplane-content decisions settled and the PDS4 bundle
generalization done.

### Track E — Test and documentation debt

**Goal:** no shipped stage without tests; no doc a future maintainer
cannot follow.

- Zero-coverage stages: backplane CLI (#241), PDS4 CLI (#242).
- Untested star-conflict logic (#243); real-image baselines beyond one
  frame (#174).
- Summary-PNG unit tests (#177).
- Docs: missing dev-guide pages (#178), instrument appendices (#93, with
  Track D), API-reference gaps (#244), Sphinx nitpicky-clean CI (#129),
  terminator-doc verification (#122), curation-tooling language pass
  (#245).

**Parallelism:** entirely parallel with everything; good filler between
larger items. #241/#242 should precede any serious PDS4/backplane work in
Track D so that work lands on tested ground.

### Track F — Remaining instruments, features, and hardening

**Goal:** the other three instruments reach Cassini's proven level; the
enhancement backlog and code-quality tail are burned down.

- **Instruments** (after Track A proves the Cassini spine): Voyager star
  navigation (#19), Galileo star navigation and REDO handling (#18, #17),
  LORRI PSF and product policy (#2, #138, #33), outer-planet ring models
  (#82, #81, #83), rotation-pyramid cost (#126), degradation classifier
  (#181), per-instrument calibration extension (re-run #230 per
  instrument as library frames land).
- **Features:** BOTSIM (#27), star streaks (#22), backplane-reader repo
  (#107), PDS4 input when external archives exist (#34 — replaces the
  PDS3 source per instrument; not required for project completion),
  cartographic/bootstrap navigation
  (#184 — explicitly far off), polarity-aware ring matching (#183),
  chaotic-rotator poses (#187), manual-nav dialog redesign (#186),
  gated-feature PNG styling (#185), stop-after-features flag (#182),
  body shape models (#23), sim polish (#84, #78, #151, #152, #157, #158).
- **Hardening/cleanup** (any time, mostly small): #13, #15, #21, #38, #39, #43, #65, #92, #96-#105, #109, #110, #119, #135, #137, #139, #140, #143, #144, #147, #155, #212.

**Parallelism:** hardening is permanent filler. Instrument work waits for
Track A's Cassini verdict only in the sense that there is no point
calibrating three more instruments with an unproven method; the
star-navigation bug fixes (#19, #18) can start any time.

## 5. Suggested global order

1. **Now:** Track A items 1-4 start in parallel
   (library growth, simulator design proposal, estimator proof,
   distortion validation). Track B's #221/#222. The Track D decisions go
   to the operator as a batch — they cost nothing to decide early and
   unblock scoping.
2. **Next:** Track A item 5 (agreement study, bulk layer first), with
   Track E test-debt and Track B remainder as parallel fill.
3. **Then:** Track A items 6-8 (CI tiers, real-anchored recalibration,
   accuracy tail). This is the calibration finish line: confidence and
   uncertainty defensible against reality.
4. **Then:** Track D build-out per the decisions; Track F instruments,
   re-running the now-proven calibration per instrument.
5. **Last:** capability matrix finalized (#231), documentation
   completion, end-product accuracy (#232) — the deliverable's
   evidence package.

**Effort honesty:** Track A is multi-week at agent pace; its two largest
items (simulator de-circularization, agreement study) are serialized only
through the estimator proof between them. Tracks B+C+E are one to two
weeks of interleavable small/medium items. Track D depends on scope
decisions; Track F is another multi-week block dominated by per-instrument
calibration repeats. Operator hands-on time is dominated by library votes
and the five decision gates, not by any implementation.

## 6. Operator decision gates (collected)

1. Simulator de-circularization design approval (#227) — before build.
2. Titan: implement or scope out (#60).
3. CK kernels as a delivered product (#188).
4. Sub-5 px body policy (#239).
5. Recurring: library batch votes; realism verdict; agreement-study frame
   selection; tier re-blessing after #230.

## 7. Issue index

Every open issue, by track. **Bold** = created after the 2026-07-11
review (the 2026-07-12 reconciliation; #251-#254 and #256 by the
2026-07-13 backend test suites, PRs #255/#257; #258/#259/#261/#263 by the
2026-07-13 Phase D operator review).

| Track | Issues |
|---|---|
| A — validation & calibration | #84, #150, #153, #172, #174, #176, #223, **#224**, **#225**, **#226**, **#227**, **#228**, **#229**, **#230**, **#232**, **#233**, **#234**, **#235** |
| B — navigation correctness | #24, #25, #128, #130, #132, #133, #179, #180, #210, #221, #222, **#237**, **#238**, **#239**, **#254**, **#258**, **#259**, **#261**, **#263** |
| C — statistics & QA | **#240** (plus the standing cross-check and campaign-report practice) |
| D — capability completion | #28, #30, #47, #50, #53, #54, #55, #57, #60, #63, #66, #67, #69, #70, #71, #72, #73, #74, #75, #76, #77, #79, #93, #108, #118, #126, #139, #141, #142, #188, **#231**, **#236**, **#251**, **#252**, **#253**, **#256** |
| E — test & docs debt | #122, #129, #177, #178, **#241**, **#242**, **#243**, **#244**, **#245** |
| F — instruments, features, hardening | #2, #13, #15, #17, #18, #19, #21, #22, #23, #27, #33, #34, #38, #39, #43, #65, #78, #82, #81, #83, #92, #96, #97, #98, #99, #100, #101, #102, #103, #104, #105, #107, #109, #110, #119, #134, #135, #137, #138, #140, #143, #144, #147, #151, #152, #155, #157, #158, #181, #182, #183, #184, #185, #186, #187, #212 |

Cross-listed items (listed once above, noted here): #150/#128 serve both
Track A's limb-bias workstream and Track B's redesign; #103/#134/#126
serve both Track D performance and Track F hardening; #93 is written in
Track D, extended per instrument in Track F; #174 baselines are Track A
infrastructure delivered as Track E test work.

---

*History: this plan supersedes `plans/archive/ROADMAP_2026-07-12.md` (the
issue-ordered pipeline build-out) and
`plans/archive/FULL_PROGRAM_AFTER_MANUAL_WORK_2026-07-11.md` (the
six-track task inventory), consolidating both into one top-level view.
The validation methodology and the curation playbook were already single
documents and remain in place as the detail layer.*
