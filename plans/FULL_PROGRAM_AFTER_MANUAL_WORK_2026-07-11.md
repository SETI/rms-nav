# Complete program after the manual work — 2026-07-11

This is the full task inventory that remains after your five manual steps (merge
#208/#213/#214, vote `batch_retriage2`, sidecar reconciliation, ring_only_flat
curation, the two small leftovers). It supersedes the abbreviated "Part 5" of the
previous handoff, which listed only immediate follow-ons.

Sources: `plans/VALIDATION_AND_CALIBRATION_PLAN.md` (workstreams WS-0..WS-18 and
Milestones A–D), `plans/ROADMAP.md` (Phases 0–3 + hardening track),
`critiques/PROJECT_STATE_REVIEW_2026-07-08.md` (open dispositions), the 121 open
GitHub issues, and this week's session findings. Effort classes: **S** ≤ 1 day,
**M** = days, **L** = 1–2 weeks, **XL** = multi-week — assuming I do the work and
you review.

Bottom line up front: the remaining program is **six tracks**. Track A is the
science half (the calibration core you asked about). Tracks B–F are engineering,
capability, and debt work — much of it schedulable in parallel, none of it a
substitute for Track A.

---

## Track A — The validation & calibration program (the science half)

This is Milestones B and C of the validation plan. Status of every workstream:

| WS | What it is | Status | Effort left |
|---|---|---|---|
| WS-0a | Identifiability map + agreement-estimator math, proven on a truth-known sim | **Not started** | M |
| WS-0b | Bias-independence verdict for the estimator (needs the WS-2 sim) | **Not started** | M |
| WS-1 | Multi-object cross-technique agreement on real frames — the primary real-image accuracy instrument | **Not started** | XL |
| WS-1b | Reprojection consistency across overlapping frames (secondary corroboration; shares code with WS-18) | **Not started** | L |
| WS-2 | De-circularize the simulator AND prove it realistic against real images | **Not started** | XL |
| WS-3 | Real-image cohort: ≥120 sidecars, ≥20 per instrument | **In progress** (62; Cassini+Voyager only) | L (ongoing) |
| WS-4 | Real-image tests in CI | **Open** — CI still runs `-m "not integration"`; the library tier never runs automatically | M |
| WS-5 | Confidence calibration | **Sim-anchored interim done** (this week); full version re-anchors on WS-1's real error measurements | M (re-run once anchors exist) |
| WS-9 | Justify/derive/measure the remaining magic constants | **Partial** (many measured by the sim campaigns; remainder inventoried in #176) | M |
| WS-10 | Fix the limb ~0.09 px DT systematic (`#150`, related `#128`) | **Open** — still measurable in this week's refreshed report | M–L |
| WS-13 | Make the calibrated (I/F) sim path realistic and tested (noise model; currently noise-free) | **Open** | M |
| WS-17 | Geometric distortion model validation (Voyager/Galileo prerequisite for WS-1 off-Cassini) | **Not started** | M–L |

Already done and out of the list: WS-11 (degenerate-rotation reporting), WS-14
(provenance), WS-6's honesty half (docs no longer overclaim; the formal capability
matrix rides with Track D decisions).

**Execution order** (from the plan's dependency section, unchanged by this week):

1. **WS-3 continues** — your batch votes → my Stage-D sidecars → repeat with new
   cohort scans. Target: fill `faint_stars` and `ring_only_flat`, get every class
   to ≥2, add first Galileo SSI and New Horizons LORRI frames. WS-3 gates
   everything else.
2. **WS-2 and WS-0a start in parallel** immediately after your merges — neither
   waits on anything but WS-3's existing images.
   - WS-2 is the big one: separate the sim's rendering from the nav models'
     rendering (today they share functions in `src/spindoctor/sim/render.py` —
     the circularity), then prove realism statistically against real frames
     (noise spectra, PSF wings, limb profiles, ring-edge shapes, photometric
     levels). Issue #153 holds the deferred calibration-validation scene work
     that belongs here; #194/#195 (crater seed/illumination) rise in priority if
     they surface during realism comparison. **This has real architectural
     choices — the one Track-A item where I'll bring you a design to approve.**
   - WS-0a is analysis code: given N techniques' offsets+covariances on one
     frame, when is per-technique σ recoverable, and is the estimator unbiased.
3. **WS-17** — validate the Voyager/Galileo distortion models on star frames
   (Cassini's distortion is benign; this gates only the off-Cassini half of WS-1).
4. **WS-1** — the flagship: run the pipeline over a large multi-feature cohort
   (hundreds–thousands of frames, no hand-verified truth needed for the pairwise
   layer), measure cross-technique agreement at scale, extract per-technique σ in
   the WS-0-qualified bins, plus absolute attitude on the scarce star tie-points.
   Includes the star-technique real anchoring (the ~19 star-class queue failures
   are symptoms this study diagnoses). WS-1b rides alongside.
5. **WS-4** — wire the library + accuracy-regression tests into a scheduled CI
   tier (not per-PR; the plan is explicit about that).
6. **WS-5 full** — re-run this week's fit tooling against the WS-1/WS-2 anchors;
   re-derive tier boundaries; drop the sim-anchored caveats from the docs.
7. **WS-9, WS-10, WS-13** — close out with real anchors in hand.

**Your role in Track A** (much smaller than mine, but load-bearing): approve the
WS-2 de-circularization design; bless the realism verdict; approve WS-1 frame
selection; spot-verify outlier frames; re-bless tiers after WS-5-full. Plus the
ongoing batch voting that feeds WS-3.

---

## Track B — Navigation-correctness engineering (ROADMAP 1A + session findings)

These fix known wrongness or fragility in the navigator itself. Several directly
improve WS-1's data quality, so they should land before or during it.

| Item | Why | Effort |
|---|---|---|
| #123 — Mahalanobis grouping breaks on CRLB-tight covariances | Possibly largely fixed by this week's #210 sigma floors — needs verification and closure either way. Ensemble agreement quality feeds WS-1 directly. | S–M |
| #124 — No cross-technique outlier rejection (one disagreement forces `conflicted`) | Ensemble robustness; visible in the queue's `conflicted` frames. | M |
| #86 — Fix ring models (Saturn) | Essential per roadmap; ring-heavy frames are a large cohort fraction. | M–L |
| #125 — BodyTerminatorNav mis-convergence has no per-technique signal | Same class of defect as #209/#211: a technique that can't self-flag. | M |
| #179 / #191 — DT coarse-prior search robustness + minimum-support guard | Known false-lock vectors for limb/ring techniques. | M |
| #128 — Robust limb navigation redesign (all body types/illuminations) | The strategic fix behind #150/#125/#187. | XL (design first) |
| #150 — Limb floor is model-vs-image edge offset (the WS-10 systematic) | Accuracy floor for the limb technique. | M |
| #145 — Star-ring occlusion mis-classifies stars near ringlet edges | Star-technique data quality for WS-1. | S–M |
| #130 — Calibrate per-instrument star limiting magnitudes on real fields | Part of the star-gate real anchoring. | M |
| #146, #136, #12 — config-override selection bug, WAC ingest drop, label filespecs | Ingest/config correctness. | S each |
| #25 — Blurring for high-resolution bodies | Roadmap "important" for close flybys. | M |
| #202 — sd_offset exhausts >61 GB on N1646315051 (Helene) | Must be fixed before any large WS-1 campaign (it would kill batch runs). | M |
| Session: multi_body N17023890xx triage (3 frames, all techniques spurious) | Likely occlusion/model-conflict bug; one debugging session. | S–M |
| Session: Voyager scattered_light C00598xx quintet fails wholesale | Voyager photometric path or prescan criteria. | M |
| Session: sub-5 px body policy — expected-failure sidecars vs a relaxed-disc pathway | Your rescue proved the disc can lock 4–5 px bodies with loosened gates; decide and implement or curate as expected failures. **Needs your decision.** | S–M |
| #212 — closes after your CPU RMA (remove systemd unit + setup.sh taskset) | Hardware tracker. | S |

---

## Track C — Statistics, QA, and the accuracy checkpoint (ROADMAP 1C)

| Item | Why | Effort |
|---|---|---|
| #35 — Navigation statistics system (metadata → SQLite → report) | The roadmap's accuracy checkpoint: success/failure rates, technique usage, offset stats, per-frame disagreement, and "does confidence predict accuracy" — the standing QA check on calibration. Runs over any day/instrument. | L |
| `library_crosscheck.py` re-run after your sidecar reconciliation | Confirms tier agreement post-reconciliation; repeat after every calibration change. | S (recurring) |
| Coverage-matrix test (every class ≥2 sidecars, every technique exercised) wired into the deliberate tier | Turns Phase-10's deferred invariant on once Track-A/WS-3 fills the classes. | S |

---

## Track D — Capability gaps (Milestone D; decision gates first)

Each starts with a **scope-or-implement decision that is yours**:

| Item | The decision | If implemented | Effort |
|---|---|---|---|
| WS-7 / #60 — Titan navigation | Titan is a registered no-op placeholder. Implement atmospheric-limb navigation, or scope it out in the capability matrix? | New technique + model against haze-limb physics | XL |
| WS-8 / #53 family — PDS4 beyond Cassini | Generalize bundle generation or ship Cassini-only? Children: #71–#79, #66, #67, #69, #47, #54, #55, #57, #63, #70, #73, #74, #75, #76 | Label templates, LID builders, collection machinery per mission | XL |
| WS-12 / #93 — the four instrument user-guide appendices | Write them (needs per-instrument measured behavior from Track A/Phase 2) | — | M |
| WS-15 — performance & safe parallelism (#126 rotation pyramid ~10 min, #134 oops precision global mutation, #103 thread-unsafe caches) | Needed before cloud-scale campaigns; #134 also bites WS-1 batch runs | — | M–L |
| WS-18 — end-product geometric accuracy (backplanes, mosaics, PDS4 values) | The last validation layer; needs WS-1b + WS-2 + WS-8 | — | L |
| #188 / #50 — CK kernels with updated pointing as a delivered product | Roadmap 1H; a major deliverable decision | — | L |

---

## Track E — Test-coverage and documentation debt (from the active critique)

The critique's still-open addendum item 6 plus section-4 leftovers:

| Item | Why | Effort |
|---|---|---|
| Unit tests for `src/spindoctor/cli/backplanes/` — currently **zero** | A whole delivered stage with no tests. | M |
| Unit tests for `src/spindoctor/cli/pds4/` — currently **zero** | Same. | M |
| Tests for the real star-conflict logic (`nav_model/stars/conflicts.py`) | Untested and mocked-out where it would run; #145 lives here too. | S–M |
| #174 — real-image regression baselines beyond the single frame | 1 baseline vs 62 sidecars; the library test asserts behavior but only one frame pins exact numbers. | M |
| #177 — unit tests for `summary_png` | Filed, open. | S |
| api_reference: the 8 missing `mosaic_viewer` modules | Critique section 4 item 7. | S |
| #129 — zero Sphinx nitpicky warnings, gated in CI | Doc hygiene target. | M |
| #178 — missing dev-guide pages: filters, uncertainty, troubleshooting | The uncertainty page becomes important once WS-5-full makes sigmas load-bearing. | M |
| #122 — verify terminator albedo/sharpness doc rationale | Small doc-accuracy item. | S |

---

## Track F — Phase 2 instruments, Phase 3 features, and the hardening tail

**Phase 2 (after the Cassini spine is proven through Track A):**
- 2A Voyager: #19 (VGISS star navigation broken — overlaps the WS-17/WS-1 star work)
- 2B Galileo: #18 (GOSSI star nav), #17 (REDO handling)
- 2C LORRI: #2 (PSF calibration), #138 (`_eng` product policy)
- 2D Ring models: #82 Jupiter, #81 Uranus, #83 Neptune
- 2E Cross-instrument calibration: extend #173/#130/#93 per instrument as library
  frames land (this is WS-5-full run per-instrument)
- 2F Shared: #126 (rotation pyramid cost — bites Galileo/Voyager), #181
  (degradation classifier taxonomy)

**Phase 3 (after multi-instrument):** #27 BOTSIM, #22 star streaks, #107
backplane-reader repo, #34 Cassini PDS4 archive source, #84 sim ring edges/gaps,
#194–#198 sim polish (higher priority if WS-2 surfaces them), #184 CartographicNav
+ bootstrap (the crater-mapping design — explicitly far off), #183 polarity-aware
ring matching, #187 chaotic-rotator pose handling, #186 manual-nav dialog
redesign, #185 gated-feature PNG styling, #182 stop-after-features flag, #158/#157
sim rendering polish, #155 display-scaling consolidation.

**Hardening/cleanup (parallel, any time, mostly S each):** #65 exception class,
#104 broad-except control flow, #192 ensemble bare assert, #193 rotation-combine
weighting doc, #103 thread-unsafe caches, #98 registry consolidation, #97
oversized modules, #96 dead code, #135 from_file dedup, #143 viewer cursor bug,
#144 QApplication lifetime, #109/#110/#100 shared helpers, #101/#102 CLI cleanup,
#99 orphan module, #92 dependency groups, #105 Any→TypedDict boundaries, #140
geometry union access, #139 malformed global-index LID, #137 dead validation
helper, #141/#142 cloud-task dedup/cardinality, #118 config validation system,
#147 confidence-context dedup, #119 PNG creation location, #108 CLI
logging/cloud audit, #39 AttrDict, #77 backplane args, #55/#57 backplane content
decisions.

---

## Suggested global order (what I'd actually do)

1. **Immediately after your merges:** rebases; batch-vote sidecars (WS-3);
   #202 memory fix + #123 verification (they block campaign-scale runs);
   WS-2 design proposal to you; WS-0a estimator in parallel.
2. **While WS-2 builds:** Track B correctness items (#86, #124, #125, #179/#191,
   #145), #35 statistics system (it pays for itself immediately), Track E test
   debt (backplanes/pds4/star-conflict tests), WS-4 CI tier.
3. **Then:** WS-17 → WS-1 (+1b) → WS-5-full → WS-9/10/13. This is the calibration
   finish line: confidence and sigma defensible against reality.
4. **Then:** Track D decisions (Titan, PDS4 scope, CK kernels) and Phase 2
   instruments, with 2E re-running the now-proven calibration per instrument.
5. **Continuous:** hardening tail as filler; library growth every time you have
   review bandwidth.

**Effort honesty:** Track A alone is multi-week at agent pace (WS-2 and WS-1 are
the two XL items and they serialize through WS-0/WS-17 in between). Tracks B+C+E
are roughly 2–3 weeks of interleavable M/S items. Track D depends entirely on
your scope decisions. Phase 2 is another multi-week block per the same pattern as
Cassini but smaller. Nothing on this list besides your five manual steps and the
five decision gates marked "**yours**" requires your hands on a keyboard — the
rest needs your eyes (PR review, PNG verification, design sign-off) at the points
noted.
