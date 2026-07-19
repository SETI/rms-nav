# Operator Playbook: from the sim-realism merge to the agreement study

*Explicit operator instructions — commands to run, files to modify, and
prompts to hand to agent sessions — for every next step in
`plans/PROGRAM_PLAN.md` as of 2026-07-19. Work through Section 0 in order;
everything after it can be dispatched in parallel as agent sessions.
Environment for every command below: `source /seti/newnav/setup.sh` from
`/seti/newnav/rms-nav` (the venv is `venv/`).*

## 0. Right now (operator-only, minutes)

### 0.1 Merge the sim-realism program

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

### 0.2 Clean up the investigation worktrees

The #288 evidence packet lives in the baseline worktree; keep the packet,
drop the throwaway checkouts:

```bash
cp -r /seti/newnav/rms-nav-baseline/attr288 \
      /seti/newnav/rms-nav/_work/attr288_packet
cd /seti/newnav/rms-nav
for wt in rms-nav-attr288-main rms-nav-attr288-352014d \
          rms-nav-attr288-d694188 rms-nav-attr288-83583e9 \
          rms-nav-attr288-1ab0395 rms-nav-attr288-a0a3040 \
          rms-nav-attr288-27ada67 rms-nav-attr288-b3041dc \
          rms-nav-288fix; do
  git worktree remove --force /seti/newnav/$wt 2>/dev/null
done
git worktree prune
# rms-nav-baseline: keep or remove after confirming the packet copy.
```

### 0.3 Two cheap decisions (comment on the issues)

- **#60 Titan**: implement or scope out. The sim now has the haze-limb
  substrate ready (phase-dependent apparent radius, ring of light), so
  "implement" is unblocked whenever you want it.
- **#188 CK kernels as a delivered product**: yes/no/defer.

```bash
gh issue comment 60  --body "Decision: <implement now | defer until X>"
gh issue comment 188 --body "Decision: <ship | defer>"
```

## 1. The sidecar re-ratchet (first agent session after the merge)

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

**After the PR merges, adopt the transfer watch:** edit
`util/calibration/CAMPAIGN_20260718.md`, change the "Transfer watch
(proposed)" heading to "Transfer watch (adopted YYYY-MM-DD)", adjusting
thresholds if you disagree with the proposal. That gives the calibration
its falsification criterion.

**Verify when done:**

```bash
pytest tests/integration/test_autonomous_nav.py -m '' -n auto --dist=loadfile
# Expect: red set = only the deliberately-pinned frames, each named in
# the re-ratchet PR body with its owning issue.
```

## 2. Track A critical path (dispatch as agent sessions, in this order)

### 2.1 WS-0 — prove the agreement estimator (#224)

The item the sim work unblocks directly: known-truth simulation is now
trustworthy enough to prove the extraction math on. Multi-day agent work;
no operator input needed until the report.

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

### 2.2 WS-17 — validate the camera distortion models (#228)

**Prompt:**

> Execute WS-17 (#228): formalize experiments/fov_twist/find_fov_twist.py
> into the supported suite per the reframed issue #228. Star-field
> plate-solve residual maps per instrument/camera against the library and
> the star-calibration frame lists; documented results under docs/;
> measured residuals compared against the sim's residual-distortion
> defaults in artifacts_catalog.py (update the defaults with provenance
> if the measurements disagree). Tests, CI gates, independent review, one
> PR. The acceptance bar is in plans/VALIDATION_AND_CALIBRATION_PLAN.md.

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

### 2.4 WS-1 — the agreement study (#225, #226); starts when 2.2 + 2.3 give it cohorts

Your role at the gate: approve the frame selection. Then:

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
- **#230 / WS-5** — re-anchor confidence on real evidence: "Re-run the
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

- **#301 + #291 (ensemble diagnostic channel)**: "Design and implement
  the diagnostic channel that lets the ensemble distinguish radial model
  error from pointing error: consume declared_orbit_sigma in the ring
  covariance/ensemble path and add the coarse-peak-quality/convergence
  gates recommended in the #288 investigation records. Verify against
  orbit_error_ringlet (its 0.89/high confident-wrong should drop) and
  re-measure the #291 disc-correlation case."
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
- Sidecar changes: one PR per review batch; per-frame dated notes in the
  sidecar, never only in gitignored files.
- Perf tests (`tests/integration/test_sim_perf.py`): serial only, never
  under a parallel battery.

## 5. Sequencing summary

```text
0.1 merge #313 -> 0.2 cleanup -> 0.3 decisions   (operator, minutes)
1   sidecar re-ratchet + adopt transfer watch    (1 session + 1 review)
2.1 WS-0 estimator proof   \
2.2 WS-17 distortion        |  parallel agent sessions
2.3 batch-006 + growth     /
2.4 agreement study bulk   (after 2.2 + 2.3; you approve frames)
2.5 CI tier, re-anchor confidence, accuracy tail (after 2.4)
3   parallel fill items    (any time, independent)
```

The program's finish line for this arc: #230 retires the
`confidence_provisional` marker on real evidence, at which point every
confidence number the pipeline emits is backed by published, real-frame
measurements — the goal named in Section 1 of the program plan.
