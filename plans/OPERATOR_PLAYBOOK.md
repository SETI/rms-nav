# Operator Playbook: driving the agreement study and the calibration finish

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` as of 2026-08-04. Work through Section 0 first;
everything after it can be dispatched in parallel as agent sessions.
Environment for every command below: `source /seti/newnav/setup.sh` from
`/seti/newnav/rms-nav` (the venv is `venv/`).*

## 0. Right now (operator-only, minutes)

### 0.1 The pending decisions (comment on the issues)

Each is a scope commitment the downstream work waits on:

- **#316 ring orbit-uncertainty severity**: ship at the conservative
  default, ratchet `rings.orbit_radial_sigma_correlated_fraction`, or
  implement the wander decomposition. It demotes five operator-verified
  Keeler frames to low, and it must be settled before the #230
  recalibration reads their tiers.
- **#407 Titan haze navigation ratification**: the method is implemented and
  validated to a stated 1 px cross-track / 3 px along-track bound. What needs
  your decision is a bundle: five mid-implementation specification changes
  (each recorded with the measurement behind it), three acceptance bounds the
  evidence argues with (a unit-test noise bound, the planted-truth z-score
  band, and the >= 90% consistency-pair bound measured at 83.3%), and three
  staged curation artifacts — a twenty-frame overlay review batch under
  `util/titan_cohort/review_batch/` with every vote null, six library
  nominations with draft sidecars under `util/titan_cohort/nominations/`, and
  a recommendation to add a `titan_haze` scene class. The issue enumerates
  each one.
- **#338 highly-irregular terminator fit (N1853392805)**: accept the
  2-px-class ground truth, keep TERMINATOR_ARC for SPICE-known synchronous
  rotators, or wait for shape models (#23).

```bash
gh issue comment 316 --body "Decision: <ship default | ratchet fraction | wander decomposition>"
gh issue comment 407 --body "Decisions: <per the enumerated list>"
gh issue comment 338 --body "Decision: <accept 2px GT | keep TERMINATOR_ARC | shape models>"
```

### 0.2 Adopt the calibration's falsification criterion (#334)

The confidence calibration has no armed falsification criterion and its
real-frame regression gate is suspended. Edit
`util/calibration/CAMPAIGN_20260718.md`: change the "Transfer watch
(proposed)" heading to "Transfer watch (adopted YYYY-MM-DD)", adjusting the
thresholds if you disagree with the proposal. That gives the calibration a
criterion that can fail. Tracked by #334.

## 1. The deliberately-red library set

The full library suite
(`pytest tests/integration/test_autonomous_nav.py -m '' -n auto
--dist=loadfile`) leaves a small red set, each frame owned by an open
navigation issue. These are pins, not regressions; do not re-ratchet them
until the owning issue closes.

| frame(s) | owner |
|---|---|
| N1492091163, N1867601758, N1867602424 (wrong ring-feature locks) | #346 |
| N1853392805 (highly-irregular exclusion discards the terminator fit) | #338 |
| N1484593951, N1686349893 (resolved-body ~2 px offset misses) | #350 |
| N1487595731_1 (multi_body: expects BodyDiscCorrelateNav primary, gets BodyLimbNav) | #406 |
| N1633925572_1 (ring_plus_body: expects the medium tier, gets low) | #406 |

**Verify the pin set is exactly this after any navigation-affecting merge:**

```bash
pytest tests/integration/test_autonomous_nav.py -m '' -n auto --dist=loadfile
# Expect: red set = only the frames above; any other delta must be
# attributed in the merging PR.
```

## 2. Track A critical path (dispatch as agent sessions, in this order)

The estimator (WS-0, `util/agreement/`) and the distortion tool (WS-17,
`util/fov_distortion/`) are built and proven on sims; the study now needs
the cohorts and the bulk run. The findings below constrain how the study is
scoped — read them before 2.2.

**Estimator findings that gate the study:**

- **limb-DT and ring-DT are not bias-independent** through the shared
  preprocessing layer. That pair must be declared or excluded from joint
  solves; body+ring is the common Cassini composition, so this hits the
  main cohort.
- **limb-DT vs disc-NCC holds as an anchor against symmetric PSF error**: a
  symmetric rendered-PSF mismatch opens no shared edge bias (the disc-NCC is
  ~16x less PSF-sensitive than the limb-DT). Not a clean-on-all-PSF verdict
  — the asymmetric/coma kernel that most directly matches the mechanism is
  unrenderable by the sim (#359), and the disc's sub-pixel NCC resolution
  floors detectability (#361).
- **Multi-body frames are not two independent measurements** (#322):
  cross-body limb errors correlate at +0.72 and the naive solve
  misattributes the coupling onto disc.
- **A ~2 px inward bias on partial-arc limb fits** (#321) is a navigation
  finding in its own right.
- The reliability gate *filters* rather than *shifts*, so its common-mode
  effect is a survivorship selection, not a bias — it can only make
  agreement look better, never worse, and cannot manufacture cross-technique
  coupling. That result is conditional on a separable/monotonic admission
  model; the real score-vs-error coupling is deferred to #358 (the sim
  cannot supply it), and whether the solve needs a survivorship correction
  is #360.
- Estimator tests do not run in CI (#324).

### 2.1 WS-3 — library growth (continuous; your votes are the bottleneck)

Next concrete step is the batch-006 manual-nav pass (7 frames voted "m",
class changes recorded in `_work/cohort_curation/batch_006_followups.yaml`).

**Prompt:**

> Run the batch-006 manual-nav pass: the 7 frames in
> _work/cohort_curation/batch_006_followups.yaml. Apply the operator's
> recorded class changes (C3479608 -> two_bright_stars_no_body;
> C4337947/C4401900 -> one_bright_star_no_body). Use sd_offset with the
> manual technique for each frame; do not trust the triage offset for
> C0164400400R (bloom-biased). Produce sidecars for the frames that
> navigate, present the results for operator review before committing.
> One PR per the sidecar-batch convention.

Then resume normal batch generation (`util/cohort_curation/`) and vote as
batches arrive.

### 2.2 WS-1 — the agreement study (#225, #226); waits on 2.1 cohorts

Your role at the gate: approve the frame selection. Apply these scoping
gates from the estimator findings above:

- The **limb-ring pair is correlated**, so it may not carry per-technique
  covariance claims; declare it or exclude it.
- The **limb-disc pair holds as an anchor against symmetric PSF error**:
  carry a declared limb-disc covariance where precision demands (a mild
  intrinsic negative coupling is real but its sign is unreliable), and treat
  the asymmetric-PSF channel as still open (#359).
- **Multi-body cohorts** declare the limb-limb pair (#322) and should be cut
  by illumination geometry, since part of the coupling is illumination-locked.
- Blob and disc correlate at +0.83 on partial bodies; never share a solve
  there.
- Cohorts are already filtered by the reliability gate; its selection effect
  is bounded in-sim but its real-frame size is unknown until #358, so the
  study's covariances describe *navigable* frames rather than frames, and
  the report must say so.
- A healthy identifiability report is **not** evidence that independence
  holds; all-positive recovered variances are necessary but not sufficient.

**Prompt:**

> Execute the agreement study's bulk layer (WS-1, #225) per
> plans/VALIDATION_AND_CALIBRATION_PLAN.md: run the pipeline over the
> approved real-frame cohorts with two or more independent fiducials per
> frame, compute the pairwise agreement statistics with the WS-0-proven
> estimator, and produce the report. Do not start the per-technique
> separation layer (it waits on WS-0's solvability map saying where it is
> meaningful). Operator approves the frame selection before any bulk run.

### 2.3 The finish line (dispatch after 2.2 produces data)

- **#229 / WS-4** — real images in CI: "Wire a small cached real-image
  tier into every-PR CI and the full suite on a schedule, per WS-4." Related:
  the data-independent sim suites still never run in Actions (#336) and there
  is no canonical environment for the committed sim baselines (#335).
- **#230 / WS-5** — re-anchor confidence on real evidence. The correlated
  ring-witness fix (#317) is done, so the calibration no longer trains
  against rows where two ring techniques on one catalog were fused as
  independent witnesses. **Still settle #316 before reading the Keeler
  tiers:** the tooling fits tier boundaries from the fused confidence
  scalar, and the orbit-uncertainty severity call moves five
  operator-verified frames across a boundary. Then: "Re-run the
  calibration tooling against the agreement study's measurements per WS-5;
  retire the confidence_provisional marker where the evidence supports it;
  re-bless tiers with the operator." This is where the terminator's
  provisional label and the sim-anchored coefficients get their real-world
  upgrade.
- **Accuracy tail** — #233, #150/#128 (design first; see Section 3),
  plus #234 and #232.

## 3. Parallel fill (independent agent sessions, any order)

Copy the line as the session prompt, prepending: "Work in
/seti/newnav/rms-nav. Read CLAUDE.md and the named issue first.
Independent review before done; all CI gates; one PR."

- **Ring ensemble follow-ups**: #319 (no library coverage for
  opposed-ansae geometry, so the conditioning guard is unvalidated); #380
  (fit an explicit per-family cross-covariance instead of collapsing
  correlated witnesses to a representative — gated on real-frame rho
  measurements from #225). #316 is
  an operator decision (Section 0.1), reversible by config either way.
- **#150/#128 (photometric limb redesign)** *(Fable-required — see 3b;
  the physics is subtle enough that a wrong premise survives review)*:
  "Produce the DESIGN ONLY for the photometric-limb fit that removes the
  ~0.1 px limb-darkening bias, per the diagnosis on #150/#128. No
  implementation until the design is operator-approved; validation must be
  against real images per WS-10. Address whether the same model-vs-image
  bias applies to non-step (gradual / shouldered) ring edges, not only the
  limb."
- **#373 (coarse-lock calibration pass)**: "Make the RingEdgeNav coarse
  seed robust against competing edge populations per #373, folding in the
  wrong-lock datapoints from #346."
- **#130**: "Calibrate the star limiting-magnitude model against real
  fields per #130."
- **#394 (shape-lock veto residual)**: the veto is suppressed when a trusted
  star fix agrees with the geometric consensus, which leaves the corner
  where the star fix is itself wrong-locked — a safe `conflicted` becomes a
  confident-wrong `success`. Sequence with #230/WS-5.
- **CK kernels: the C-matrix consumers (#50)**: the kernels ship, so the
  reading half is dispatchable -- "Switch the backplane and reprojection
  readers from the metadata `offset` to the recorded `cmatrix` per #50; the
  round trip in `tests/integration/test_ck_round_trip.py` is the evidence
  the two agree on real frames." The kernel-side follow-ups (#433, #434,
  #437, #440/#444/#455, #446, #448, #452) are independent of it and of each
  other; #435/#436 are a pair and #436 waits on #435. Whether Cassini
  navigation should run from the predicted rather than the reconstructed
  kernels (#459) is a pending operator decision, not dispatchable work.
- **#430 (results index)**: implement `plans/RESULTS_DB_PLAN.md` -- the
  plan is the specification and its section 8 is the session protocol
  (per-phase implementer + adversarial reviewer, both Opus-class). Ready to
  dispatch as written.
- **Logging follow-ups** (small, independent, no sequencing): #424 (remove
  `sd_create_bundle_cloud_tasks` — it is unwired and leaks to the worker
  terminal); #418 (decide whether a mosaic cloud task's `status` should
  reflect its per-image failures, not only its counts — a policy question
  before it is a coding one); #423 (the GUI viewers print library log
  records to stdout; about ten tests capture that fallback and need their
  capture strategy changed first); #429 (give the `util/` tooling the same
  logging surface). #427 (reorganize the config namespace) is larger and
  should precede #118.
- **Sim realism residual (#227)**: the de-circularization is done and on
  main; #227 stays open only for the realism proof and closes at the
  operator's realism-verdict gate, itself gated on #309. #309
  (realism-configured multi-instrument campaign — biggest
  calibration-credibility win available; consumes the fidelity gaps #325,
  #329-#333, #341-#345, #290, #377) is the load-bearing step, with #310
  (structural boundary enforcement) and #311 (mirror-parity guard) hardening
  the partition. Each issue body is a prompt basis.
- **#355 (Voyager sim distortion per camera)**: re-measure and split the
  Voyager distortion defaults once the star-lock rate improves.

## 3b. Model-tier guidance (where a top-tier model is truly needed)

Reserve the top-tier (Fable-class) model for work where a
plausible-but-wrong answer survives review by looking right; a
mid-tier (Opus-class) implementer is the efficient default everywhere
else. Applied to the open items:

- **Top-tier required:** the #230/WS-5 calibration-fit adjudication and #309
  (calibration on messy evidence); the #358/#360 survivorship-correction
  math and the #359/#361 asymmetric-PSF coupling probes; the **#150/#128
  photometric-limb redesign** (both the design and its adjudication — the
  physics is subtle and a plausible-but-wrong premise rides straight through
  review: e.g. "rings are unaffected" holds only for sharp step-edges, but a
  gradual or shouldered ring edge carries the same model-vs-image photometric
  bias the limb does); and the independent-review pass on anything
  statistical, boundary-touching, or calibration-touching, regardless of who
  implemented it.
- **Mid-tier drafts, top-tier adjudicates:** #310 (the boundary
  restructuring — the guard tests catch mechanical regressions, the review
  catches new leak shapes).
- **Mid-tier or below suffices:** library growth, the agreement study's
  bulk execution (once WS-0 hands it a proven estimator), #229, #311, #373,
  #130, the logging follow-ups (#418, #423, #424, #429), and the
  documentation/engineering items.

## 3c. Tracking-issue register

Open issues grouped by theme so none is lost to a PR body; the sequencing
hooks reference the sections above. All carry A/B/Priority/Effort labels
with assignee rfrenchseti.

**Confident-wrong / ensemble honesty (sequence with #230/WS-5):**

- **#346** three library frames lock confidently onto the wrong ring feature
  (owns the N1492091163 / N1867601758 / N1867602424 red pins)
- **#394** shape-lock veto suppression trusts a star fix that could itself be
  wrong-locked, turning a safe `conflicted` into a confident-wrong `success`
- **#380** correlated-witness fusion collapses to a representative at rho=1;
  fit an explicit cross-covariance once #225 measures real-frame rho
- **#400** the ensemble merge and tier logic have never been exercised on the
  strongly anisotropic covariance the Titan haze fit reports

**Simulator fidelity gaps (feed #309 and the sim follow-ups in Section 3):**

- **#325** simulated stars shine through dark limbs; star-technique success is
  optimistic
- **#329** simulated calibrated products floor at 1 LSB; real products dither
  below it (WAC diverges 8x)
- **#330** instrument chains render cosmic-ray transients at zero
- **#331** simulated hot pixels are per-scene, not per-detector
- **#332** PSF catalog has one kernel per instrument; binned/summed readout
  modes are inexpressible
- **#333** four physical error axes are unmodeled
- **#290** body renderer exceeds the sim render-time budget on oversampled grids
- **#377** sim rings are single annuli; build realistic nested-ringlet scenes
  and tests
- **#341** the campaign's scene mixture is authored, unvalidated against real
  frames
- **#342** star_psf_sigma is a 3.0 placeholder on Galileo, Voyager, LORRI
- **#343** the tuned NAC PSF wing may be absorbing operator registration error
- **#344** haze brightness is a module constant
- **#345** a scene can echo truth-side noise into instrument_config with no
  validator warning

**Calibration governance / CI (gate WS-5 and the CI tier in Section 2.3):**

- **#334** calibration has no armed falsification criterion and its real-frame
  gate is suspended (owns the Section 0.2 transfer-watch step)
- **#335** no canonical environment for committed sim baselines (0.99 vs
  0.81-0.84 across machines)
- **#336** data-independent simulator integration suites never run in Actions
  (relates to #229/WS-4)
- **#340** library_crosscheck records only a yes/no primary-technique flag,
  not the winning technique
- **#426** a committed sim render is stale on `main` and the test that would
  say so is integration-marked, so nothing catches it per PR

**Titan haze refinements (the method ships; these are measured limits):**

- **#403** the arc ray reach is sized by the search window rather than by
  where the limb can be, costing rays on large well-framed frames
- **#404** the flat arc-residual cap behaves as a size-dependent gate, and
  the measurements say it must not simply be raised
- **#401** the extreme-phase (> 150 deg) edge of the working range is
  uncharacterized
- **#402** the main rings are masked opaque, refusing frames visible through
  the C ring or the gaps
- **#397** a self-calibrated haze-radius table would remove the dominant
  along-track error; **#398** CB3 cartographic refinement; **#399** a
  Voyager validation cohort; **#405** library growth through the standard
  curation pipeline

**Logging follow-ups (all small and independent; Section 3):**

- **#418** a mosaic cloud task reports success when every image in it failed
  (policy decision first)
- **#423** the GUI viewers print library log records to stdout through
  pdslogger's handler-less fallback
- **#424** remove `sd_create_bundle_cloud_tasks`
- **#427** the config namespace is organized on no stated axis; sequence
  before #118
- **#428** upstream registry-eviction request to `rms-pdslogger`
- **#429** give the `util/` tooling the same logging surface

**Library-frame reds and decisions (Section 1):**

- **#338** highly-irregular exclusion discards the ground-truth terminator fit
  on N1853392805 (decision)
- **#350** two resolved-body frames miss offset tolerance by ~2 px
  (N1484593951, N1686349893)
- **#406** two pre-existing reds (N1487595731_1, N1633925572_1) that fail
  identically on `main` and were not in the pinned table

**Agreement estimator real-frame follow-ups (sequence with #225/WS-1 and
#230/WS-5):**

- **#358** measure the real reliability-vs-error coupling and run the
  stratified estimator on the real #225 cohorts — the size of the gate's
  selection optimism the sim cannot supply (Important)
- **#360** decide, after #358, whether the agreement solve needs a
  selection-aware (survivorship) correction (a decision issue)
- **#359** probe limb-disc PSF coupling under the asymmetric/coma/field-varying
  PSF error the sim cannot render (Important)
- **#361** disc-NCC sub-pixel resolution floors the smallest limb-disc coupling
  detectable (Important)
- **#321** partial-arc limb fits carry an undiagnosed inward radial bias of
  about 2 px (navigation finding)
- **#322** cross-body limb errors correlate at +0.72; multi-body frames are
  not independent measurements
- **#324** agreement estimator tests do not run in CI

## 4. Standing practices for every session you dispatch

- Environment: `source /seti/newnav/setup.sh`.
- The controller pattern works: one session as controller, implementer
  subagents per phase/slice, an independent fresh-context review of every
  deliverable, fix rounds until the critique is clean, full CI
  (`./scripts/run-all-checks.sh -i`), then one PR. Ask for it explicitly in
  the prompt if you want it.
- CI expectations: `run-all-checks.sh -i` is the pre-merge gate; the library
  suite's red set must equal the documented pinned set (Section 1) or every
  delta must be attributed in the PR.
- Issues: every new issue carries A-type, B-location, Priority, Effort
  labels and assignee rfrenchseti.
- **Never leave future work, a deferred fix, a known limit, or a pending
  decision recorded only in a PR body, a comment, a campaign record, or a
  docstring — file a tracking issue and reference it from the prose.** PRs
  get merged and scroll away; an item that lives only in prose is an item
  that will be lost.
- Sidecar changes: one PR per review batch; per-frame dated notes in the
  sidecar, never only in gitignored files.
- Perf tests (`tests/integration/test_sim_perf.py`): serial only, never
  under a parallel battery.

## 5. Sequencing summary

```text
0.1 decisions (#316, #407, #338) -> 0.2 adopt transfer watch (#334)  (operator, minutes)
2.1 library growth (batch-006 + continued)   (agent session; your votes gate it)
2.2 agreement study bulk   (after 2.1 cohorts; you approve frames)
2.3 CI tier, re-anchor confidence, accuracy tail (after 2.2)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal named in Section 1 of the program plan.
