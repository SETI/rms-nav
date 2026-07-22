# Operator Playbook: from the sim-realism merge to the agreement study

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` as of 2026-07-19. Work through Section 0 in order;
everything after it can be dispatched in parallel as agent sessions.
Environment for every command below: `source /seti/newnav/setup.sh` from
`/seti/newnav/rms-nav` (the venv is `venv/`).*

## 0. Right now (operator-only, minutes)

### 0.1 Merge the sim-realism program -- DONE

Merged 2026-07-20: PR #313 squash-merged to `main` (f9c37f0). Commands
kept for the record:

```bash
gh pr view 313 --web      # final read if desired
gh pr merge 313 --squash
```

Evidence backing the merge: every phase PR (#289-#312) was independently
reviewed to a clean verdict and CI-gated; acceptance criteria verified at
the tip; the library-suite delta is 100% attributed in
`util/calibration/CAMPAIGN_20260718.md`. GitHub Actions does not run the
integration tier, so the green you rely on is the local battery recorded
in those PRs.

### 0.2 Clean up the investigation worktrees -- DONE

Done 2026-07-20: all investigation worktrees and their local branches were
removed (the operator directed a full worktree cleanup, so the baseline
worktree went too) and the checkout returned to `main`. The throwaway
attr288 packet was therefore not copied to `_work/attr288_packet`; its
failure taxonomy survives in the #288 issue body and is superseded by the
Section 1 re-ratchet, which reduces #288 to the deliberately-red pins.

### 0.3 Two cheap decisions (comment on the issues)

- **#60 Titan**: implement or scope out. The sim now has the haze-limb
  substrate ready (phase-dependent apparent radius, ring of light), so
  "implement" is unblocked whenever you want it.
- **#188 CK kernels as a delivered product**: yes/no/defer.

```bash
gh issue comment 60  --body "Decision: <implement now | defer until X>"
gh issue comment 188 --body "Decision: <ship | defer>"
```

## 1. The sidecar re-ratchet -- DONE, PR #353 (merged)

Done 2026-07-20; PR #353 was squash-merged to `main` on 2026-07-21
(97f4b41). The full library suite is `66 passed, 9 failed`
(`pytest tests/integration/test_autonomous_nav.py -m '' -n auto
--dist=loadfile`, 26 min on the canonical local machine), and the nine
failures are exactly the deliberately-red pins below. 36 sidecars were
re-ratcheted to measured behavior (23 tier flips under the 0.85 high
boundary and the 2.61 px limb covariance floor, 11 primary flips to
multi-star StarRefineNav and others, 3 conflicted->success recoveries)
plus the C1205021_GEOMED adjudication (medium->high, which resolves #347).
Ground-truth offsets were never touched. This clears the bulk of the #288
library regression.

Remaining red set, each owned by an open issue:

| frame(s) | owner |
|---|---|
| N1492091163, N1867601758, N1867602424 (wrong ring-feature locks) | #346 |
| N1853392805 (highly-irregular exclusion discards the terminator fit) | #338 |
| N1572105349 (single-inlier refine offset-pull) | #222 (reopened) |
| N1484593951, N1686349893 (resolved-body ~2 px offset misses) | #350 |
| N1530185128 (recalibration-induced spurious conflict) | #351 |
| W1444747627 (star gates spurious on a navigable small-offset field) | #352 |

Filed for this task: #350, #351, #352; #222 reopened; #347 resolved by the
adjudication. See the register in Section 3c. The original prompt and
verify block are kept below for the record.

The recalibration re-tiers real library frames; every flip is
pre-attributed. Your role is to review the attribution table once and
bless one PR.

**Read first:** `util/calibration/CAMPAIGN_20260718.md` — the per-frame
table and the "Transfer watch (proposed)" section.

**Prompt to give a session:**

> Re-ratchet the image-library sidecars to the post-recalibration
> pipeline, following util/calibration/CAMPAIGN_20260718.md as the
> attribution basis. Run the full library suite
> (pytest tests/integration/test_autonomous_nav.py -m '' -n auto
> --dist=loadfile) to capture current per-frame behavior; for every frame
> whose flip is attributed in the campaign record, update its sidecar
> expectations to measured behavior with a dated note naming the
> attribution; leave the deliberately-red pins (the frames owned by open
> issues) untouched and list them in the PR body. Adjudicate
> C1205021_GEOMED per its provenance notes (its medium pin contradicted
> the recorded rank=high from the start). One PR for the whole batch; do
> not merge it. Finish by re-running the suite and reporting the final
> red set with the issue that owns each remaining red frame.

**Still pending -- now the immediate next step, #353 having merged: adopt
the transfer watch** (tracked by #334): edit
`util/calibration/CAMPAIGN_20260718.md`, change the "Transfer watch
(proposed)" heading to "Transfer watch (adopted YYYY-MM-DD)", adjusting
thresholds if you disagree with the proposal. That gives the calibration
its falsification criterion. This is the one part of Section 1 not yet
done.

**Verify when done:**

```bash
pytest tests/integration/test_autonomous_nav.py -m '' -n auto --dist=loadfile
# Expect: red set = only the deliberately-pinned frames, each named in
# the re-ratchet PR body with its owning issue.
```

## 2. Track A critical path (dispatch as agent sessions, in this order)

### 2.1 WS-0 — prove the agreement estimator (#224) — EXECUTED, PR #314

Done 2026-07-19/20; PR #314 targets `rf_sim_realism` and is unmerged.
The estimator, campaign harness, identifiability map, and campaign record
live under `util/agreement/`; there are no `src/` changes.

Findings that change what comes after, so read these before 2.4:

- **limb-DT and ring-DT are not bias-independent** through the shared
  preprocessing layer. That pair must be declared or excluded from joint
  solves; body+ring is the common Cassini composition, so this hits the
  main cohort.
- **limb-DT vs disc-NCC showed no coupling** through the gradient/DT
  channel, and the shared-PSF-edge suspicion on this pair (the study's
  *base* equation) is now probed too (#320, PR #363, merged): a symmetric
  rendered-PSF mismatch opens no shared edge bias -- the disc-NCC is ~16x
  less PSF-sensitive than the limb-DT, which caps the coupling magnitude.
  So the base pair holds as an anchor against symmetric PSF error in the
  Cassini-NAC regime. Not a clean-on-all-PSF verdict: the asymmetric/coma
  kernel that most directly matches the mechanism is unrenderable by the
  sim (#359), and the disc's sub-pixel NCC resolution floors detectability
  (#361).
- **Multi-body frames are not two independent measurements** (#322):
  cross-body limb errors correlate at +0.72 and the naive solve is
  well-conditioned, reports everything identifiable, and misattributes
  the coupling onto disc.
- **A ~2 px inward bias on partial-arc limb fits** (#321) surfaced as a
  side effect. That is a navigation finding, not only a campaign one.
- Estimator tests do not run in CI (#324).

Stage 0b is complete: #320 (PSF-layer probe, PR #363) and #323
(reliability-gate selection effect, PR #362) both squash-merged to `main`
on 2026-07-21. #323's finding: the reliability gate *filters* rather than
*shifts*, so its common-mode effect is a survivorship selection, not a
bias -- and that selection can only make agreement look better, never
worse, and cannot manufacture cross-technique coupling; the result is
conditional on a separable/monotonic admission model, with the real
score-vs-error coupling deferred to #358 (the sim cannot supply it). #320's
finding is folded into the disc-NCC bullet above. New follow-ups filed:
#358, #359, #360, #361 (see the register in Section 3c). The original WS-0
prompt is kept below for reference.

**Prompt:**

> Execute WS-0 from plans/VALIDATION_AND_CALIBRATION_PLAN.md: prove the
> cross-technique agreement estimator on known-truth simulations. Read
> that plan section fully first, plus the capability-envelope section of
> docs/dev_guide/dev_guide_simulator.rst so you use the simulator inside
> its stated envelope. Build the known-truth campaigns with the existing
> scene machinery (planted offsets are ground truth by construction),
> derive where the pairwise/three-cornered-hat extraction is solvable and
> where it degenerates, validate the math against the planted truth at
> campaign scale, and deliver: the solvability map, the validated
> estimator implementation under util/ with tests, and a written report.
> Use independent review before you call it done, run all CI gates, and
> open one PR. Where the estimator needs error models the sim cannot
> honestly provide, say so explicitly rather than substituting sim
> optimism - the envelope doc lists what the sim cannot establish.

### 2.2 WS-17 — validate the camera distortion models (#228) -- DONE, PR #354

Done 2026-07-21; PR #354 squash-merged to `main` (a2227db). The
`experiments/fov_twist` one-off is rewritten into the maintained tool
`util/fov_distortion/` (a pure-numpy decompose/aggregate core with unit
tests, a star-navigation-backed per-frame measure step, and a process-pool
driver over per-instrument cohort YAMLs). Results are published as a
standalone chapter `docs/fov_distortion_report/` alongside the simulator
report, and the measured residual distortion now populates the sim
distortion defaults (`DISTORTION_RESIDUAL_PARAMS` in
`src/spindoctor/sim/forward/artifacts_catalog.py`), replacing the interim
single-amplitude estimates. `experiments/fov_twist/` is removed.

Measured per instrument (star-field residual after the navigator's
distortion model), with autonomous star nav locking only a fraction of the
off-Cassini cohorts:

- **Cassini ISS NAC/WAC** (50/50, 46/50): twist consistent at +/-0.011 deg
  (negligible), radial distortion at the noise floor -> rotation fitting
  stays off (matches the shipped setting).
- **Galileo SSI** (7/18): consistent -0.053 deg twist (static kernel
  candidate) plus a pincushion radial term reaching ~0.5 px at the corner.
- **New Horizons LORRI pre-KE** (16/48): clean static +0.191 deg twist
  (kernel candidate); post-KE epochs are outside pointing-kernel coverage.
- **Voyager 2 ISS NAC/WAC** (5/27, 14/45): frame-varying twist (0.28 px
  corner scatter, WAC mean +0.36 deg) -> per-frame rotation fitting
  required, and the largest residual distortion of any instrument. Timing
  the main pipeline over the 19 locked frames with rotation off vs on:
  median 4.61 s either way (-0.02 s, 0.99x), so the "too slow" reason for
  keeping `config_430_inst_vgiss.yaml` `fit_camera_rotation` off does not
  hold on star fields. Voyager 1 and the candidate ("possible") frame lists
  add nothing (VG1 locks 0; VG2 candidates lock the same handful).

The confidence calibration and the operator-curated library do not enable
instrument-default distortion, so the sim-default update leaves them
untouched. Follow-up #355 tracks re-measuring and splitting the Voyager sim
distortion per camera once the star lock rate improves. Independent
verification: the branch adds no new failures against `main`; core-library
line coverage is 91.3% on the default suite.

### 2.3 WS-3 — library growth (continuous; your votes are the bottleneck)

Next concrete step is the batch-006 manual-nav pass (7 frames voted "m",
class changes recorded in `_work/cohort_curation/batch_006_followups.yaml`).

**Prompt:**

> Run the batch-006 manual-nav pass: the 7 frames in
> _work/cohort_curation/batch_006_followups.yaml. Apply the operator's
> recorded class changes (C3479608 -> two_bright_stars_no_body;
> C4337947/C4401900 -> one_bright_star_no_body). Use sd_offset with the
> manual technique for each frame; do not trust the triage offset for
> C0164400400R (bloom-biased). Produce sidecars for the frames that
> navigate, present the results for operator review before committing,
> and fold the C1205021 adjudication in if Section 1's re-ratchet has not
> already resolved it. One PR per the sidecar-batch convention.

Then resume normal batch generation (`util/cohort_curation/`) and vote as
batches arrive.

### 2.4 WS-1 — the agreement study (#225, #226); 2.2 done, now waits on 2.3 cohorts

Your role at the gate: approve the frame selection. Then:

**Gates added by 2.1's results — apply these when scoping the study:**

- The **limb-ring pair is measured as correlated**, so it may not carry
  per-technique covariance claims; declare it or exclude it.
- The **limb-disc pair holds as an anchor against symmetric PSF error**
  (#320 ran, PR #363): the disc barely responds to a PSF mismatch, capping
  the coupling. Carry a declared limb-disc covariance where precision
  demands (a mild intrinsic negative coupling is real but its sign is
  unreliable), and treat the asymmetric-PSF channel as still open (#359).
- **Multi-body cohorts** declare the limb-limb pair (#322) and should be
  cut by illumination geometry, since part of the coupling is
  illumination-locked.
- Blob and disc correlate at +0.83 on partial bodies; never share a solve
  there.
- Cohorts are already filtered by the reliability gate; its selection
  effect is now bounded in-sim (#323, PR #362): the gate can
  only make agreement look better, never worse, and cannot manufacture
  coupling, so every per-technique sigma the study reports is a lower
  bound. The size of that optimism depends on the real score-vs-error
  coupling, which the sim cannot supply (#358) -- the study's covariances
  therefore describe navigable frames rather than frames, and the report
  should say so.
- General warning from the campaign: a healthy identifiability report is
  **not** evidence that independence holds, and all-positive recovered
  variances are necessary but not sufficient.

**Prompt:**

> Execute the agreement study's bulk layer (WS-1, #225) per
> plans/VALIDATION_AND_CALIBRATION_PLAN.md: run the pipeline over the
> approved real-frame cohorts with two or more independent fiducials per
> frame, compute the pairwise agreement statistics with the WS-0-proven
> estimator, and produce the report. Do not start the per-technique
> separation layer (it waits on WS-0's solvability map saying where it is
> meaningful). Operator approves the frame selection before any bulk run.

### 2.5 The finish line (dispatch after 2.4 produces data)

- **#229 / WS-4** — real images in CI: "Wire a small cached real-image
  tier into every-PR CI and the full suite on a schedule, per WS-4."
- **#230 / WS-5** — re-anchor confidence on real evidence. **Handle #317
  first or explicitly:** the calibration tooling fits tier boundaries from
  the fused confidence scalar, and correlated ring witnesses emit
  high-confidence/large-error rows that push the high-tier boundary the
  wrong way. Then: "Re-run the
  calibration tooling against the agreement study's measurements per
  WS-5; retire the confidence_provisional marker where the evidence
  supports it; re-bless tiers with the operator." This is where the
  terminator's provisional label and the sim-anchored coefficients get
  their real-world upgrade.
- **Accuracy tail** — #233, #150/#128 (design first; see Section 3),
  plus #234 and #232.

## 3. Parallel fill (independent agent sessions, any order)

Copy the line as the session prompt, prepending: "Work in
/seti/newnav/rms-nav. Read CLAUDE.md and the named issue first.
Independent review before done; all CI gates; one PR."

- **#301 + #291 (ensemble diagnostic channel)** — EXECUTED, PR #315
  (unmerged, targets `rf_sim_realism`). The channel, the convergence
  gate, and the fit-quality gates all landed; `orbit_error_ringlet`
  demotes high to medium and its error improves from 3.01 to 1.54 px.
  #291 persists bit-for-bit and is documented rather than absorbed.
  Follow-ups, in the order they matter:
  - **#318** — DONE, PR #356 squash-merged to `main` (366880c).
    RingAnnulusNav now consumes the channel: the ring models attach
    `orbit_normals_vu` and an effective `sigma_orbit_radial_px` to
    `RingAnnulusGeometry`, and the technique widens its NCC covariance
    from the same absorbed-translation sensitivity RingEdgeNav uses, so
    both ring techniques price an identical annulus geometry identically
    (a short visible arc widens one-for-one along its normal; a closed
    ring barely absorbs a uniform radial error). On `orbit_error_ringlet`
    RingAnnulusNav's radial sigma rises from 0.56 to ~2.56 px, the fused
    sigma widens to ~1.79 px (calibration ~1.3 sigma, was ~2.8), and the
    fused error bar now covers the residual bias; the tier stays medium,
    correctly, and the scene pin is re-measured to the honest post-channel
    behaviour (recovered error 2.31 px). The channel is now effective on
    both ring techniques rather than ~5%, and
    `rings.orbit_radial_sigma_correlated_fraction` scales both. Caveat:
    widening both members isotropically re-couples their weights and
    resurfaces the correlated-witness scalar (0.99 on this scene) -- that
    is #317, documented at the scene pin and in the ensemble guide, not
    masked.
  - **#316**: the fully-correlated severity is contested and demotes five
    operator-verified Keeler frames. Carries your decision; ratchet via
    `rings.orbit_radial_sigma_correlated_fraction` or implement the
    wander decomposition.
  - **#317**: correlated ring witnesses fused as independent — sequence
    before #230.
  - **#319**: no library coverage for opposed-ansae geometry, so the
    conditioning guard is unvalidated.
- **#150/#128 (photometric limb redesign)**: "Produce the DESIGN ONLY for
  the photometric-limb fit that removes the ~0.1 px limb-darkening bias,
  per the diagnosis on #150/#128. No implementation until the design is
  operator-approved; validation must be against real images per WS-10."
- **#179 (coarse-lock calibration pass)**: "Calibrate the coarse-search
  edge-population lock against the image library per #179, folding in
  the false-flag datapoint from #261."
- **#130**: "Calibrate the star limiting-magnitude model against real
  fields per #130."
- **#284 then #285**: "#284: fix UCAC4 bright-end photometry corrupting
  predicted star brightness. Then #285: wide-offset asterism matching for
  sparse fields (depends on #284)."
- **#277 residue (N1853392805)**: decide among the three recorded options
  (accept 2-px-class ground truth for resolved highly-irregular bodies /
  keep TERMINATOR_ARC for SPICE-known synchronous rotators / shape
  models per #23) and comment the decision on #277; then a session
  implements it.
- **Sim follow-ups**: #309 (realism-configured multi-instrument
  campaign — biggest calibration-credibility win available), #310
  (structural boundary enforcement), #311 (mirror-parity guard). Each
  issue body is self-contained as a prompt basis.
- **#287 (collect.py thread pinning)**: small fix; until it lands, every
  calibration campaign needs the shell-level exports below.

## 3b. Model-tier guidance (where a top-tier model is truly needed)

Reserve the top-tier (Fable-class) model for work where a
plausible-but-wrong answer survives review by looking right; a
mid-tier (Opus-class) implementer is the efficient default everywhere
else. Applied to the items above:

- **Top-tier required:** WS-0 (#224 — correlated-error estimator math and
  the solvability map; the one item not to delegate down even with a
  strong review); #230/WS-5 and #309 (calibration-fit adjudication on
  messy evidence); the design and adjudication of the #301/#291 ensemble
  diagnostic channel; and the independent-review pass on anything
  statistical, boundary-touching, or calibration-touching, regardless of
  who implemented it.
- **Mid-tier drafts, top-tier adjudicates:** the #150/#128
  photometric-limb design; #310 (the boundary restructuring — the guard
  tests catch mechanical regressions, the review catches new leak
  shapes).
- **Mid-tier or below suffices:** the sidecar re-ratchet, library
  growth, the agreement study's bulk execution (once WS-0 hands it a
  proven estimator), #229, #311, #284/#285, #130, #179, and the
  documentation/engineering items.

## 3c. Tracking-issue register (2026-07-20 audit + re-ratchet)

The 2026-07-20 sim-program audit filed #325-#347 and the Section 1
re-ratchet filed #350-#352 (and reopened #222). All are open and carry
A/B/Priority/Effort labels with assignee rfrenchseti. Listed here so none
is lost to a PR body; the sequencing hooks reference the sections above.

**Confident-wrong / ensemble honesty (sequence with #230/WS-5 and the
#301/#291 channel in Section 3):**

- **#328** high-phase haze crescent returns a gate-passing success ~30 px
  wrong and nothing vetoes it (Essential; the family that had no owner)
- **#339** scattered-light correlated disc/limb errors fused as independent
  at the 0.99 confidence cap
- **#346** three library frames lock confidently onto the wrong ring feature
  (owns the N1492091163 / N1867601758 / N1867602424 red pins)
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
- **#341** the campaign's scene mixture is authored, unvalidated against real
  frames
- **#342** star_psf_sigma is a 3.0 placeholder on Galileo, Voyager, LORRI
- **#343** the tuned NAC PSF wing may be absorbing operator registration error
- **#344** haze brightness is a module constant
- **#345** a scene can echo truth-side noise into instrument_config with no
  validator warning

**Calibration governance / CI (gate WS-5 and the CI tier in Section 2.5):**

- **#334** calibration has no armed falsification criterion and its real-frame
  gate is suspended (owns the Section 1 "adopt the transfer watch" step)
- **#335** no canonical environment for committed sim baselines (0.99 vs
  0.81-0.84 across machines)
- **#336** data-independent simulator integration suites never run in Actions
  (relates to #229/WS-4)
- **#340** library_crosscheck records only a yes/no primary-technique flag,
  not the winning technique

**Star navigation:**

- **#337** star-field matcher triplet canonicalization is a seed lottery on
  equal-brightness fields

**Library-frame reds and decisions (from the Section 1 re-ratchet):**

- **#338** highly-irregular exclusion discards the ground-truth terminator fit
  on N1853392805 (decision)
- **#347** C1205021_GEOMED medium-vs-high provenance mismatch -- resolved by
  the re-ratchet adjudication in PR #353
- **#350** two resolved-body frames miss offset tolerance by ~2 px
  (N1484593951, N1686349893)
- **#351** recalibration turns an operator-verified success into a spurious
  conflict (N1530185128)
- **#352** star gates self-flag spurious on a navigable small-offset WAC frame
  (W1444747627)
- **#222** (reopened) single-inlier pass-2 refine pulls the fused offset off a
  correct body fix (N1572105349)

**WS-0 Stage 0b follow-ups (2026-07-21; sequence with #225/WS-1 and
#230/WS-5):** #320 and #323 are DONE (PRs #363 and #362, both merged); these
are the pieces the sim could not close, filed from those PRs.

- **#358** measure the real reliability-vs-error coupling and run the stratified
  estimator on the real #225 cohorts -- the size of the gate's selection
  optimism the sim cannot supply (gates the #323 lower-bound claim on real
  frames; Important)
- **#360** decide, after #358, whether the agreement solve needs a
  selection-aware (survivorship) correction (Useful; a decision issue)
- **#359** probe limb-disc PSF coupling under the asymmetric/coma/field-varying
  PSF error the sim cannot render -- the one channel #320 left open on the base
  pair (Important)
- **#361** disc-NCC sub-pixel resolution floors the smallest limb-disc coupling
  #320 could detect (Important)

## 4. Standing practices for every session you dispatch

- Environment: `source /seti/newnav/setup.sh`. Calibration campaigns
  additionally need
  `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1` until #287 is fixed.
- The controller pattern that executed the sim plan works: one session as
  controller, implementer subagents per phase/slice, an independent
  fresh-context review of every deliverable, fix rounds until the
  critique is clean, full CI (`./scripts/run-all-checks.sh -i`), then one
  PR. Ask for it explicitly in the prompt if you want it.
- CI expectations: `run-all-checks.sh -i` is the pre-merge gate; the
  library suite's red set must equal the documented pinned set (after
  Section 1) or every delta must be attributed in the PR.
- Issues: every new issue carries A-type, B-location, Priority, Effort
  labels and assignee rfrenchseti.
- **Never leave future work, a deferred fix, a known limit, or a pending
  decision recorded only in a PR body, a comment, a campaign record, or a
  docstring — file a tracking issue and reference it from the prose.** A
  2026-07-20 audit of the sim-realism program found 23 such orphans
  (filed as #325-#347), including a confident-wrong family worth
  Essential priority that no issue owned. PRs get merged and scroll away;
  an item that lives only in prose is an item that will be lost.
- Sidecar changes: one PR per review batch; per-frame dated notes in the
  sidecar, never only in gitignored files.
- Perf tests (`tests/integration/test_sim_perf.py`): serial only, never
  under a parallel battery.

## 5. Sequencing summary

```text
0.1 merge #313 -> 0.2 cleanup -> 0.3 decisions   (operator, minutes)
1   sidecar re-ratchet + adopt transfer watch    (1 session + 1 review)
2.1 WS-0 estimator (exec, #314) + Stage 0b  -- DONE (PRs #362, #363 merged)
2.2 WS-17 distortion  -- DONE (PR #354)
2.3 batch-006 + growth     (parallel agent session; your votes gate it)
2.4 agreement study bulk   (after 2.3; 2.2 done; you approve frames)
2.5 CI tier, re-anchor confidence, accuracy tail (after 2.4)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal named in Section 1 of the program plan.
