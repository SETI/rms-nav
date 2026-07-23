# Operator Playbook: driving the agreement study and the calibration finish

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` as of 2026-07-22. Work through Section 0 first;
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
- **#60 Titan**: implement haze-limb navigation or scope it out. The sim has
  the haze-limb substrate ready (phase-dependent apparent radius, ring of
  light), so "implement" is unblocked whenever you want it.
- **#188 CK kernels as a delivered product**: yes / no / defer.
- **#338 highly-irregular terminator fit (N1853392805)**: accept the
  2-px-class ground truth, keep TERMINATOR_ARC for SPICE-known synchronous
  rotators, or wait for shape models (#23).

```bash
gh issue comment 316 --body "Decision: <ship default | ratchet fraction | wander decomposition>"
gh issue comment 60  --body "Decision: <implement now | defer until X>"
gh issue comment 188 --body "Decision: <ship | defer>"
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
| N1572105349 (single-inlier refine offset-pull) | #222 |
| N1484593951, N1686349893 (resolved-body ~2 px offset misses) | #350 |
| N1806609736 (Iapetus shape-lock veto misfire vs a correct limb+star fix) | #392 |

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
- **#230 / WS-5** — re-anchor confidence on real evidence. **Handle #317
  first or explicitly:** the calibration tooling fits tier boundaries from
  the fused confidence scalar, and correlated ring witnesses emit
  high-confidence/large-error rows that push the high-tier boundary the
  wrong way; settle #316 before reading the Keeler tiers. Then: "Re-run the
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

- **Ring ensemble follow-ups**: #317 (correlated ring witnesses fused as
  independent — sequence before #230); #319 (no library coverage for
  opposed-ansae geometry, so the conditioning guard is unvalidated). #316 is
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
- **Star-matcher robustness**: #337 (triplet canonicalization seed lottery),
  #376 (widened saturation captures the wrong bright reference), #367
  (single-detectable-star wide offset). Each issue body is self-contained.
- **Confident-wrong vetoes** (sequence with #230/WS-5): #328 (haze crescent
  ~30 px wrong, Essential), #339 (scattered-light correlated disc/limb),
  #291 (disc locks at extreme shape mismatch), #326/#327 (body-body
  occlusion ignored by the disc template and the visible-arc report).
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
  #130, the star-matcher items (#337, #376, #367), and the
  documentation/engineering items.

## 3c. Tracking-issue register

Open issues grouped by theme so none is lost to a PR body; the sequencing
hooks reference the sections above. All carry A/B/Priority/Effort labels
with assignee rfrenchseti.

**Confident-wrong / ensemble honesty (sequence with #230/WS-5):**

- **#328** high-phase haze crescent returns a gate-passing success ~30 px
  wrong and nothing vetoes it (Essential)
- **#339** scattered-light correlated disc/limb errors fused as independent
  at the 0.99 confidence cap
- **#346** three library frames lock confidently onto the wrong ring feature
  (owns the N1492091163 / N1867601758 / N1867602424 red pins)
- **#317** ring techniques observing one catalog model are fused as
  independent witnesses (sequence before #230)
- **#291** BodyDiscCorrelateNav locks on confidently at extreme shape mismatch
- **#326** BODY_DISC correlation template ignores body-body occlusion at deep
  mutual-event overlap
- **#327** NavModelBody reports full visible_arc_fraction for limbs occluded
  by a nearer body

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

**Star navigation:**

- **#337** star-field matcher triplet canonicalization is a seed lottery on
  equal-brightness fields
- **#376** widened saturation match can capture the wrong bright reference in a
  crowded field
- **#367** autonomous star nav cannot lock a wide offset from a single
  detectable star

**Library-frame reds and decisions (Section 1):**

- **#338** highly-irregular exclusion discards the ground-truth terminator fit
  on N1853392805 (decision)
- **#350** two resolved-body frames miss offset tolerance by ~2 px
  (N1484593951, N1686349893)
- **#392** body-witness shape-lock veto misfires on Iapetus (N1806609736): a
  correct limb+star consensus is vetoed because the albedo-dichotomy-biased blob
  centroid disagrees; the geometric-side mirror of the #351 ensemble drop
- **#222** single-inlier pass-2 refine pulls the fused offset off a correct
  body fix (N1572105349)

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
0.1 decisions (#316, #60, #188, #338) -> 0.2 adopt transfer watch (#334)  (operator, minutes)
2.1 library growth (batch-006 + continued)   (agent session; your votes gate it)
2.2 agreement study bulk   (after 2.1 cohorts; you approve frames)
2.3 CI tier, re-anchor confidence, accuracy tail (after 2.2)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal named in Section 1 of the program plan.
