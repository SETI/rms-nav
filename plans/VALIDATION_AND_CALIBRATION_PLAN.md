# RMS-NAV Validation and Calibration Plan

*A concrete, sequenced plan to close every issue raised in
`critiques/archive/SCIENTIST_REVIEW_CRITICAL_2026-06-19.md`. Each workstream states the problem, the
goal, the specific tasks (with the files involved), the acceptance criteria that
define "closed," dependencies, and risk. A traceability matrix at the end shows
each critical-report finding mapped to the workstream(s) that retire it.*

---

**Premise.** The core rewrite architecture is stable (AUTONAV phases 0-9 plus the
phase-10 hardening are merged); this plan is *calibration and validation*, not
architecture change, so the validation effort targets stable code. The open
phase-10 content — the curated image library, the confidence calibration, and the
body-shape table — is exactly the WS-3/WS-5 territory below, sequenced by `plans/PROGRAM_PLAN.md`
Track A. There is **no fixed accuracy specification** —
the goal is to characterize the best accuracy the system actually achieves,
honestly and with its uncertainty, not to clear a numeric bar.

## Guiding principles

1. **Validation is the highest-value work, and the hardest.** Absolute accuracy
 can only be characterized in a de-circularized, realism-validated simulation; on
 real data we can measure *consistency*, not accuracy. This is the core of the
 plan — but it does **not** block the independent honesty/ops fixes (docs,
 provenance, degenerate-rotation, performance), which run in parallel and depend
 on nothing here.
2. **Claim only what the statistics support.** Report pairwise *combined* error by
 default; claim *per-technique* accuracy only where the system is identifiable and
 the techniques' bias-independence has been *verified* (WS-0), never assumed. Every
 reported number carries an uncertainty interval. (Note on notation: the recovered
 error quantity is a 2x2 **covariance**, not a scalar — see WS-0/WS-1. Where this
 document writes "σ" it is shorthand for the relevant covariance element, e.g. the
 radial-radial element on the ring axis; it never implies an isotropic scalar.)
3. **No output field without a calibration behind it.** Anything that looks like a
 probability or a quality grade must either be calibrated against real data or
 be explicitly labelled provisional.
4. **One source of truth for "what works."** A capability matrix, enforced by
 tests, replaces scattered prose claims.
5. **CI must exercise what we ship.** If real-image navigation is the product,
 real-image navigation must run automatically.
6. **Scope honestly.** Where a capability won't be built soon (e.g. Titan), the
 claim is removed from the user-facing docs and demoted to a roadmap, not left
 as an implied feature.
7. **Validation can fail.** It may reveal accuracy worse than hoped. With no spec
 to pass/fail against, a poor result is not a gate — but it feeds the algorithm
 workstreams (e.g. WS-10), it is not merely reported and shrugged off.

---

## Relationship to `plans/PROGRAM_PLAN.md`

`plans/PROGRAM_PLAN.md` is the top-level plan of record across all remaining
work. This plan is its Track A detail layer: PROGRAM_PLAN says *when* the
validation and calibration work runs relative to everything else; this plan
says *how* and defines what "validated" and "calibrated" mean. Where the two
overlap, the methodology here is binding and the cross-track ordering there is
binding.

**Status note (2026-07-12).** Complete and out of the workstream list: WS-11
(degenerate-rotation reporting) and WS-14 (provenance) shipped in PR #200;
WS-5's interim half shipped in PR #208 — the confidence formulas and tier
boundaries are now *sim-anchored* (fitted against simulated planted-truth
recovery by the `util/calibration/` tooling), with `confidence_provisional`
still true pending the real-anchored pass (#230). WS-6's honesty half (docs no
longer overclaim) is done; the capability matrix remains (#231). Every
still-open workstream now has a tracking issue (cross-map below).

**Two shared decisions, declared once:**

- **Confidence-calibration methodology (binding for #230, formerly #173).** Confidence is
  calibrated per WS-5: per-technique reliability diagrams against measured error
  anchors (WS-1 per-technique covariance where identifiable, pairwise combined
  covariance elsewhere, WS-2 sim recovery error where no multi-object cohort
  exists), a monotonic calibration map, tiers defined by error percentiles, and
  the calibration basis recorded per emitted value. Operator-assigned sidecar
  tiers are plausibility cross-checks and regression expectations, never fit
  targets.
- **Library-size target (binding for #172 and WS-3).** One library, two stages:
  the 49-image curated library (#172, workflow `plans/COHORT_CURATION_PLAN.md`) is
  the first stage — it seeds regression baselines and the initial calibration
  cohort; WS-3 then grows the same library to >=20 images per instrument, >=120
  total, which is the size the WS-1 agreement study needs. The 49-image stage is
  a milestone inside the WS-3 target, not a competing number.

**Workstream-to-issue cross-map** (issues carry the trackable work; workstreams
carry the methodology and acceptance criteria):

| Workstream | GitHub issue(s) | Note |
|---|---|---|
| WS-0 | #224 | Estimator identifiability + bias-independence. |
| WS-1 | #225 | The agreement study. The merged statistics system (#35) reports metadata statistics and consumes WS-1's per-frame disagreement metric; it is not the agreement study. |
| WS-1b | #226 | Reprojection consistency. |
| WS-2 | #227 (+ #223, #153, #84) | Sim de-circularization + realism. |
| WS-17 | #228 | Distortion validation. |
| WS-3 | #172, #174, #235 | 49-image stage first (#172), then the >=120 growth target (#235); discovery/review workflow in `plans/COHORT_CURATION_PLAN.md`. |
| WS-4 | #229 | CI integration tiers. |
| WS-5 | #230 (sim-anchored half done via #173/PR #208), #176 | Real-anchored recalibration once WS-1 anchors exist. |
| WS-6 | #231 | Capability matrix (docs-honesty half already done). |
| WS-7 | #60 | Titan: implement or scope out. |
| WS-8 | #53, #67, #34 | PDS4 input + bundle generalization. |
| WS-9 | #233, #130, #176 | Measured star SNR + sensitivity tests (#233); constants inventory (#176); limiting magnitudes (#130). |
| WS-10 | #150, #128 | Limb bias root cause and redesign. |
| WS-11 | done (PR #200) | Degenerate-rotation reporting. |
| WS-12 | #93 | Instrument appendices. |
| WS-13 | #234 (+ #153) | Detector-noise model for the I/F render path. |
| WS-14 | done (PR #200) | Provenance block complete. |
| WS-15 | #236, #103, #134, #126 | Thread safety, profiling, batch-parallel throughput. |
| WS-18 | #232 (+ #28, #66 partial) | End-product accuracy checks. |

---

## Phase 0 — Establish what validation is actually possible

> **Premise (the hard truth).** Absolute, subpixel ground truth for the real
> pointing of an archival Cassini/Voyager/Galileo/NH frame **does not exist**. No
> independent instrument measured where that camera truly looked to a hundredth of
> a pixel, and the obvious proxies do not supply it:
>
> - **Star plate-solving** (Gaia/UCAC) recovers camera *attitude* only. Stars are
> at infinity, so they are blind to the spacecraft-relative-to-body geometry
> (SPK + target ephemeris) that places bodies and rings — i.e. blind to exactly
> the techniques that most need validating. It also fails on the bright-body
> frames where those techniques run (no usable stars), and Voyager/Galileo
> distortion turns the solve into a fight where the distortion model is the
> confound.
> - **Repeat/overlap "predict B from A"** does not work: inter-frame jitter means
> the two frames have *different* true pointings, so there is no shared known
> quantity. Only a *physical* reprojection-agreement variant survives (below),
> and only as a differential check.
> - **Legacy comparison** is same-lineage and the old subpixel layer was never
> trusted (it is why the rewrite exists). It catches pixel-scale regressions; it
> has no authority on subpixel truth.
>
> Therefore the strategy is: **measure absolute accuracy only in a simulation that
> is (a) independent of the navigator and (b) proven realistic against real
> images; and on real frames, measure *consistency*, not accuracy** — stating
> plainly the common-mode error that consistency cannot rule out. The honest
> deliverable is "recovery error on a realism-validated, de-circularized sim,
> corroborated by real-image consistency, with a stated residual common-mode
> bound" — not a single "X px" accuracy claim.

### WS-0: Validate the validator (identifiability + bias-independence)
**Closes:** the unproven pillars under WS-1 — that the variance system is solvable
in the form it is actually used (full 2x2 covariances, not scalars), and that the
compared techniques have independent biases through their *shared preprocessing* —
before any real-image per-technique number is trusted.

**Problem.** WS-1's per-technique covariance rests on (a) the variance-components
system being identifiable, (b) the compared techniques having uncorrelated errors,
and (c) the recovered quantity being a covariance, not a scalar σ. If any fails,
"per-technique σ" repeats the simulator's original sin: agreement masking a
*shared* bias. The estimator math can be checked on any truth-known sim now; the
*real-world* correlation verdict additionally needs the realistic sim (WS-2) and a
real-data cross-check (see WS-1 over-determination).

**Stage 0a — estimator + identifiability (needs only a truth-known sim; runs in
parallel with WS-2, on the existing sim).**
- **Recover-known-error test.** With each technique's true error known, run the
 WS-1 covariance-components solve and confirm it recovers per-technique
 *covariances* (full 2x2, see WS-1 fix below), for each cohort composition.
- **Identifiability map.** Enumerate, per composition, what is recoverable:
 limb+disc *alone* gives only the combined covariance (one matrix equation, two
 unknown matrices — **not** separable); **limb+disc+ring** recovers only the
 **radial-radial element** of each body technique's covariance (the ring is rank-1)
 and only after the limb's anisotropic covariance is projected onto the radial
 axis — which **rotates frame to frame**, so the bin must also be cut by
 limb-orientation-relative-to-radial, or the solve carried in full matrix form;
 **multi-body** frames add the orthogonal axis. This map governs where WS-1 may
 report per-technique covariance vs only combined pairwise covariance.

**Stage 0b — bias-independence (needs the realistic WS-2 sim).**
- **Inject a shared bias *through the shared preprocessing layer*.** The DT
 techniques consume common per-image products (`image_derivatives` gradient/edge
 distance-transform, the single noise-σ estimate, the reliability gate); a bias in
 that layer shifts limb, terminator, **and** ring-edge together. WS-0 must inject
 the bias *there*, not into each technique separately, or it will not reproduce
 the real coupling. Confirm the solve detects the induced correlation rather than
 hiding it.
- **Two pivotal pairs, not one.** `Var(limb−disc)` is the *base* equation, so
 **limb-DT vs disc-NCC** independence is as load-bearing as **limb-DT vs
 ring-DT**: if disc and limb share a PSF edge bias (an NCC disc-template peak also
 tracks the PSF-blurred edge), per-technique covariance is unrecoverable
 *everywhere*, not just on the ring leg. Measure cov for **both** pairs; treat
 *no* pair as "clean" until measured.
- **Caveat — this verdict is the sim's.** It is only as real as WS-2's realism;
 the independent real-data check is WS-1's over-determined-frame closure test.

**Acceptance criteria.**
- The covariance solve provably recovers known per-technique covariances on the sim.
- A published identifiability map states, per composition, what is recoverable
 (which covariance elements, on which axes) vs only combined pairwise covariance.
- cov is measured for limb-disc *and* limb-ring (and other joined pairs) by
 injection *through the shared preprocessing layer*; no pair is assumed clean;
 failing pairs are excluded from joint solves.

**Dependencies:** Stage 0a needs only a truth-known sim (parallel with WS-2);
Stage 0b needs WS-2 (realistic sim). Gates the per-technique claims in WS-1 and
WS-5. **Risk:** medium-high — the binding outcome is whether the
DT techniques are bias-independent *through the shared preprocessing*; if limb-disc
or limb-ring are correlated, WS-1 falls back to pairwise combined covariance for
most regimes (an honest outcome, not a failure).

### WS-1: Multi-object cross-technique agreement (the primary real-image accuracy test)
**Closes:** "no real-image accuracy assessment." This is the one method that
yields a real-image accuracy signal without external ground truth, which does not
exist for these frames.

**Per-instrument applicability (decides what is even possible).**
- **Cassini ISS:** body+ring common, resolved-moon frames common, and *dozens* of
 star+body / star+ring tie-points exist ⇒ both relative/algorithm σ at scale and
 an absolute-attitude anchor are achievable. The strong case.
- **NH LORRI:** star-frame count to be verified (WS-3 task). LORRI optics are
 low-distortion, so even if star tie-points are few, distortion is a minor
 worry; absolute-attitude anchoring depends on that count.
- **Voyager ISS / Galileo SSI:** effectively *zero* usable star frames. There is
 therefore **no real-data inertial reference**: only relative/algorithm agreement
 (intra-body, body+ring) is available, which is blind to common-mode CK/distortion.
 **Absolute attitude accuracy for these two is sim-only (WS-2)** and must be
 reported as such; distortion is taken from the literature (see WS-17).

**Core idea.** Find real frames navigable by two or more fiducials — a star field
**and** a moon, a moon **and** a ring edge, two moons, or the same moon by limb
**and** disc **and** terminator. Compute each technique's offset and compare. The
shared quantity is the pointing offset itself (the signal); a small disagreement is
evidence both are accurate to roughly that level. Two independence caveats keep
this honest, and both are real:

- **The techniques are not as independent as "different fiducials" suggests.** The
 DT techniques share preprocessing — the `image_derivatives` gradient/edge
 distance-transform, the single noise-σ estimate, the reliability gate — built once
 per `NavContext`. A bias in that shared layer moves limb, terminator, and
 ring-edge together. "Run each technique independently" (not just taking the
 orchestrator's fused pick) is necessary but **not sufficient**: the shared
 preprocessing is a common-mode coupling that WS-0 must probe and that the report
 must treat as the prime suspect for any agreement-masking-bias. Where feasible,
 recompute the agreement with independent preprocessing per technique to break the
 coupling for the measurement.
- **Agreement is evidence of accuracy only between *verified* bias-independent
 techniques** (WS-0); a shared PSF edge bias can move two techniques the same way.

This is why WS-1 reports plain pairwise disagreement everywhere, and per-technique
covariance only on the narrow footing WS-0 certifies.

**Two interpretations, depending on whether the SPICE geometry is shared:**

- **Same target, different technique** (limb vs disc vs terminator on one moon;
 two edges of one ring): SPK + body ephemeris are identical and *cancel* in the
 difference, isolating the pure **algorithm/fitting error** — the cleanest
 technique-accuracy measure.
- **Different fiducial class** (star vs moon, moon vs ring): the star correction
 touches attitude only while the moon correction also touches SPK + ephemeris, so
 their disagreement additionally contains the **ephemeris error**. This measures
 end-to-end "is the body really where we put it?" accuracy and *bounds SPK error*
 — but an SPK-driven disagreement must not be misread as a technique being
 inaccurate.

**Archive reality (drives the design).** In the Cassini set, **body + ring frames
are common; frames with stars alongside a body or ring are uncommon; star + body +
ring together is rare.** So per-frame triple collocation cannot be the workhorse,
and the absolute-attitude reference (stars) is the scarce resource. Per-technique
σ is therefore separated three ways, leaning on the common cohorts and spending
the scarce stellar frames only where nothing else will do:

- **Route 1 — intra-body (common, free of ephemeris).** A single resolved moon is
 navigable by *limb*, *disc*, and (at phase) *terminator* / *blob*. Same body ⇒
 identical SPK + ephemeris ⇒ they **cancel**, leaving pure algorithm error.
 Available on every resolved-moon frame, including the common body+ring ones. The
 candidate base pair is **disc (NCC) vs limb (DT)** — *if* WS-0 certifies them
 bias-independent (not assumed; an NCC disc peak can track the same PSF-blurred
 edge as the DT limb). limb vs terminator are correlated (shared DT/PSF bias) and
 never separate it. **Only valid on round, photometrically bland bodies** — see
 the body-shape caveat below.
- **Route 2 — body + ring pairwise (the bulk cohort).** Common ⇒ large N. Compare
 along the (usually rank-1) ring-radial axis. A moon and Saturn's rings do **not**
 share SPK, so the disagreement is σ_body_algo² + σ_ring_algo² **plus a per-moon
 ephemeris bias** — the ephemeris term is a fixed offset, not a variance (see
 below), and is handled as a nuisance parameter, not folded into a σ.
- **Route 3 — star + body / star + ring (scarce tie-points).** The only
 attitude-only, inertial reference, and it does two jobs. (1) It *anchors absolute
 attitude*: stars give true δ_CK. (2) On a star+body frame, `(body − star)`
 cancels δ_CK and so **directly measures that moon's ephemeris error**
 δ_SPK(moon) (up to the body-algorithm noise) — which is exactly the per-moon
 nuisance bias Route 2 needs. So the scarce star+body frames are the cleanest
 *ephemeris* probe, not only an attitude anchor. What a single such frame *cannot*
 do is cleanly split body-algorithm error from SPK without importing
 σ_body_algo from Route 1/WS-0; treat the per-moon ephemeris estimate as carrying
 the body-algorithm noise until that import is available.

**Estimation strategy — pairwise by default, per-technique only where earned.** The
default output is the *pairwise* disagreement of two techniques on one frame. It is
the least assumption-laden product — but **not** assumption-free: reporting one
combined covariance per bin still assumes within-bin stationarity (below), so
binning a continuous dependence inflates it. It bounds only the *combined* error
(Σ_A + Σ_B), not either alone. Separating per-technique error needs a
covariance-components / three-cornered-hat solve, reported **only where WS-0
certifies the cohort**:

- **The recovered quantity is a 2x2 covariance, not a scalar σ.** A limb fit's
 covariance is strongly *anisotropic* (tight perpendicular to the arc, loose along
 its tangent); a disc fit is near-isotropic. The identities therefore live in
 matrix form: `Cov(limb−disc) = Σ_limb + Σ_disc` (clean, same body), and the
 three-way solve recovers covariance *matrices*. Writing them as scalars is wrong.
- **Identifiability.** A resolved moon *alone* gives only `Σ_limb + Σ_disc` — one
 matrix equation, two unknown matrices — **not** separable. A **body+ring frame
 with a resolved moon** carries three estimators, but the ring is **rank-1
 (radial)**, so it constrains only the **radial-radial element** of each body
 technique's covariance. Worse, the limb's anisotropy axis rotates relative to the
 fixed radial direction *frame to frame*, so the projected radial variance is not a
 bin-constant: the bin must additionally be cut by **limb-orientation-relative-to-
 radial**, or the solve carried in full matrix form across frames. The clean
 scalar story (`Var(limb−ring)−Var(disc−ring)=σ_limb²−σ_disc²`) holds only as the
 radial-radial component, after that geometry is controlled. Both the
 moon-ephemeris term and Σ_ring cancel in that subtraction (verified for δ_SPK
 uncorrelated with the algorithm errors). The **orthogonal axis** needs a second
 body (multi-body) or a star.
- **Bias independence — two pivotal pairs.** `Cov(limb−disc)` is the *base*
 equation, so **limb-DT vs disc-NCC** independence is as load-bearing as **limb-DT
 vs ring-DT**: if disc and limb share a PSF edge bias, per-technique covariance is
 unrecoverable everywhere, not just on the ring leg. WS-0 measures cov for both
 pairs *through the shared preprocessing layer*; **no pair is assumed clean.**
 limb-DT/terminator-DT are correlated and never share a solve.

Within a qualifying bin, frames are pooled assuming each technique's error
covariance is stationary across the bin (binned by resolution / phase /
lit-fraction **and** limb-orientation-vs-radial). **Stationarity is assumed by the
pairwise product too**, not only the separation — so it is the broad limiter, not a
per-technique-only one; spot-check each bin (e.g. that its disagreements look like
one distribution, not a mix), and where it fails, shrink the bin or report the
disagreement distribution itself rather than a single covariance. Every estimate
carries an **uncertainty interval** reflecting the bin's sample size; cohorts grow
until those intervals are as tight as the available frames allow (no target
precision to size against — see Premise).

**Ephemeris error is a per-target bias, not noise.** A moon's SPK error at a given
epoch is a fixed offset, not a zero-mean random variable, so it cannot be a
variance term. In cross-class comparisons it is modelled as a **per-moon, per-epoch
bias** and either (a) estimated and removed using several frames of the same moon
over a short span where the bias is ~constant, before the variance solve, or
(b) carried as a separately-reported bias term — never silently absorbed into a
technique's σ.

**Body-shape caveat — different techniques measure different "centers."** The solve
assumes limb, disc, and terminator estimate the *same* offset. On a perfect uniform
sphere they do. On a real body with topography, limb-darkening, or albedo patches,
the geometric-limb center (limb fit) and the brightness-weighted template center
(disc NCC) converge to points that differ by a *real physical amount* — not an
error in either. That genuine difference lands in `Cov(limb−disc)` and is misread as
algorithm error, inflating the recovered covariances in a target-dependent way. So
the intra-body separation is valid **only on round, photometrically bland bodies**;
irregular or high-contrast bodies are excluded from per-technique separation (their
pairwise disagreement is still reported, flagged as shape-contaminated). The
qualification gate is **not a judgment call**: derive it from the existing static
body-shape table (`ellipsoid_rms_residual_km`, `crater_scale_km`, `albedo_variation`),
with explicit thresholds, so "round and bland enough" is a reproducible,
data-sourced criterion. This shrinks the qualifying cohort further. WS-2's
shape/photometric sweeps quantify the effect on the sim, but on a real frame it
cannot be separated from algorithm error.

**Over-determination is the one real-data check on all of the above.** A frame with
limb+disc+ring+second-body makes the covariance system *over-determined*. The
residual of that over-fit is the only test of bias-independence, stationarity, and
the ephemeris model that runs on **real photons** (WS-0's verdict is the sim's, and
inherits WS-2's realism). If the over-determined equations do not close within
their uncertainty intervals, an assumption is violated — and the report must say
which numbers are thereby suspect. Curate and exploit over-determined frames
specifically for this closure test; it is the real-data counterpart to WS-0.
**Caveat: this check may itself be data-starved.** Four-fiducial frames
(limb+disc+ring+second body) are rarer than the star triples already called rare,
and for Voyager/Galileo (no stars, perhaps few multi-moon+ring frames) there may be
too few — or none — to run it. Where the closure test cannot be populated, the
assumptions remain **sim-verified only** (WS-0), and the report must say so for that
instrument rather than implying a real-data check that did not happen.

**Robustness — variances are outlier- and failure-dominated.** Covariance estimates
are exquisitely sensitive to a few mis-navigated frames, and a technique that
*occasionally fails catastrophically* (returns garbage) produces a wildly different
"covariance" than one with consistent small error. Use robust covariance estimation
with explicit outlier rejection, and **report each technique's failure rate
separately from its error covariance** — a covariance without a companion failure
rate is misleading. A "failure" (no/garbage result) and a "large error" are
different products and are reported as such.

**Population, not per-frame.** WS-1 yields *bin-level* statistics pooled over many
frames; the navigator emits a *per-frame* covariance from each fit's information
matrix. So WS-1 validates the **average scaling** of the reported error bars over a
regime — does the per-frame covariance, aggregated over the bin, match the empirical
spread? — not whether any *single* image's error bar is correct. A scientist quoting
one image's uncertainty therefore gets population-level assurance for that regime,
not per-image verification. State this explicitly; it bounds what the whole method
can promise.

**The irreducible blind spot.** Even with bias independence verified, the method is
blind to error that shifts *all* compared features *identically* — a true
common-mode CK shift. Because body+ring agreement is purely *relative*, a
CK error common to a moon and a ring edge is invisible to it; only the scarce
inertial (star) frames and the sim catch that. (Distortion is **not** in this
blind spot — it acts *differently* at different field positions, so it shows up in
the disagreement and is handled as a contaminant in WS-17, not as a hidden term.)
**Net: relative / algorithm accuracy is characterized at scale; absolute attitude
accuracy rests on the small stellar tie-point set plus the de-circularized sim
(WS-2).** The report must say exactly this and not let the large body+ring N imply
absolute accuracy.

**Coverage caveat (state it).** This characterizes only regimes where ≥2
techniques are *simultaneously feasible*. A limb that fills the FOV and drowns the
stars, or a lone faint moon with no second fiducial, is never cross-checked; those
single-technique regimes remain characterized only by the sim (WS-2).

**Tasks.**
- Curate cohorts in `tests/integration/image_library/` (extends WS-3), each tagged
 by fiducial set, SPICE-sharing class, and per-technique bias mechanism:
 resolved-moon frames (Route 1, large), body+ring frames (Route 2, large — the
 ring-radial separation route), **multi-body frames (two+ resolved moons,
 common)** — which supply the *orthogonal* axis the rank-1 ring cannot and a
 direct probe of inter-moon ephemeris error (note two limbs share the limb bias,
 so they complete axes and measure relative ephemeris, they do not reveal a shared
 limb bias), and star+body / star+ring tie-points (Route 3, small but deliberately
 collected, for absolute attitude). The richest single frame is **two resolved
 moons + a ring edge**. Genuine star+body+ring triples are recorded when found but
 not depended on.
- Build `tests/integration/agreement/` harness: per frame, run every feasible
 technique independently (and, where feasible, on independent preprocessing to
 break the shared-`image_derivatives` coupling for the measurement) and record all
 pairwise disagreements as **2x2 covariances** with their SPICE-sharing class,
 body-shape class (round/bland vs irregular/high-contrast), and
 limb-orientation-vs-radial. Report **pairwise combined covariance everywhere**;
 run the covariance-components solve for **per-technique** covariance only on the
 bins WS-0 qualifies, excluding bias-dependent pairs and shape-contaminated
 bodies, with the per-moon ephemeris modelled separately. Use **robust** estimation
 with outlier rejection, and record each technique's **failure rate** separately.
 Run the **over-determined-frame closure test** as the real-data assumption check.
 Attach an uncertainty interval to every estimate.
- Produce `docs/agreement_report/agreement_report.rst`: per-instrument, per-regime
 pairwise disagreement distributions (always); per-technique covariance with
 uncertainty intervals only where identifiable and shape-clean; per-technique
 failure rates; the over-determination closure result; the relative-vs-absolute
 split; and the identifiability, anisotropy/projection, stationarity,
 bias-independence (per WS-0), body-shape, ephemeris-bias, and coverage caveats.
- **Wire it into the live product (not just offline):** surface per-frame
 cross-technique disagreement in `_metadata.json` as an empirical uncertainty / QA
 flag (within expected combined σ → trust; beyond it → flag for review), computed
 from the `NavResult.per_technique` results the orchestrator already produces.
 This per-frame signal is also the real-data anchor for WS-5.

**Acceptance criteria.**
- A published agreement report whose every number comes from real frames: pairwise
 combined covariance everywhere; per-technique covariance (with uncertainty
 intervals) only on WS-0-qualified, shape-clean bins; per-technique failure rates;
 an explicit relative-vs-absolute split.
- No per-technique covariance is reported for a non-identifiable bin, a
 bias-dependent pair, or a shape-contaminated body; ephemeris is reported as a
 separate bias, never inside a covariance.
- The over-determined-frame closure test passes (or the violated assumption and the
 numbers it taints are named).
- Absolute attitude accuracy is reported only from the stellar tie-points + sim,
 never implied from the body+ring cohort.
- Multi-object frames emit a per-frame cross-technique disagreement metric into the
 metadata, consumed by WS-5.

**Dependencies (split — this matters).** The **pairwise combined-covariance**
product — the bulk, most-trustworthy deliverable — needs only **WS-3** (cohort) and
**WS-17** (distortion correction); it does **not** depend on WS-0 or WS-2 and should
ship without waiting for them. Only the **per-technique separation** gates on
**WS-0** (identifiability + bias-independence) and, through it, the realistic
**WS-2** sim. Do not hold the pairwise product hostage to the riskiest workstreams.
Feeds WS-5. **Risk:** medium-high for the *per-technique* layer — the compounding
risks are that WS-0 finds the DT techniques bias-correlated through shared
preprocessing, that the shape-clean + geometry-controlled qualifying cohort is
small, and that star tie-points are too few to anchor absolute attitude; the
realistic end state may be "pairwise combined covariance at scale, per-technique
covariance only on a curated set of round uniform moons in favorable
ring/multi-body geometry." The pairwise layer carries low risk and no external-data
need.

### WS-1b: Reprojection consistency across overlapping frames (secondary corroboration)
**Closes:** the one regime WS-1 cannot reach — a body imaged repeatedly with **no
second in-frame fiducial** — so it is not redundant with WS-1.

For overlapping real images of one body, navigate each independently and check that
a fixed surface feature reprojects to the same body-fixed lat/lon (rings: same
radius/longitude). Inter-frame jitter is fine — each frame carries its own pointing;
we check surface-fixed agreement, never one frame's offset predicted from another.

**Unique value vs WS-1 and WS-18 (stated to avoid double-counting):** WS-1 needs ≥2
fiducials *in one frame*; WS-1b works across frames that each have only one body, so
it covers single-fiducial sequences WS-1 skips. It is the *navigation* counterpart
of WS-18's *mosaic-product* seam check — same arithmetic, different object: WS-1b
checks the navigated offsets, WS-18 checks the assembled mosaic. They share the
implementation (build it once, apply twice). *Blind spot:* error common to both
frames (shared SPK / timing) cancels undetected, so this corroborates WS-1 and never
substitutes for it.

**Acceptance criteria.**
- A reprojection-consistency harness exists that, given overlapping frames of one
 body (or ring region), navigates each frame independently and reports the
 body-fixed (lat/lon or radius/longitude) scatter of matched surface features,
 with an uncertainty interval per sequence.
- The published results cover at least one single-fiducial sequence per
 instrument where the archive provides one (documented as absent where it does
 not), binned by resolution, and state the shared-SPK/timing blind spot
 alongside every number.
- Sequences whose scatter exceeds the combined reported per-frame σ beyond the
 expected rate are flagged and fed into WS-1/WS-10 as inputs, not dropped.
- WS-18's mosaic seam check consumes the same implementation (asserted by a
 shared-module test), so the two checks cannot drift apart.

**Dependencies:** WS-3. Shares implementation with WS-18. **Risk:** medium.

### WS-2: De-circularize the simulator AND prove it realistic (primary accuracy instrument)
**Closes:** "the headline accuracy numbers are circular." This is now the *primary*
source of any absolute accuracy number, since real truth is unobtainable.

**Problem.** `sim/render.py` and `nav_model_*_simulated.py` share
`create_simulated_body` / `render_mesh_body_image` / `render_ring`, so model and
data have a common parent and the sub-pixel numbers are self-consistency, not
accuracy. Two things must both be true before sim recovery error is a credible
accuracy estimate: the image side must be **independent** of the navigator, and
the image side must be **realistic**.

**Approach.** The navigator must use its *best available* model — if a better PSF
or shape model exists, give it to the navigator; do **not** preserve a known-worse
model just to manufacture a gap. The image side is then made *more* realistic and
*independent in implementation*, and the residual recovery error is the genuine
model error the real pipeline would also incur. Report error as a function of the
*remaining* mismatch (the part the navigator cannot capture even at its best), and
separately validate that the simulated images are statistically indistinguishable
from real ones.

**Tasks.**
- **Independent, more-realistic image model.** Render the image with a forward
 model that is (a) implemented independently of the nav model (no shared rendering
 routine for any quantity under test) and (b) *more* realistic than the
 navigator's best model — higher oversampling, measured/empirical PSF (Airy +
 jitter + filter-dependent wings), topographic shape, real detector noise. The
 navigator uses its best model, not a deliberately handicapped one; the gap under
 test is the residual the navigator genuinely cannot model, not a hand-set delta.
- **Inject controlled model error** as sweep axes in `tests/integration/sim_sweep*`:
 PSF mismatch; shape mismatch (render with topography/high-res polyhedral
 `sim_body_polyhedral.py` + craters, navigate with the ellipsoid); photometric
 mismatch (render Lommel-Seeliger, navigate Lambert, etc.); ephemeris error
 (perturb planted body position to emulate SPK error); **full detector noise on
 every scene, including the I/F path** (WS-13).
- **Validate sim realism.** Tune the image forward model until simulated frames
 match real frames distributionally — noise statistics, PSF / encircled-energy,
 the gradient profile across a real limb, dynamic range, and artifact incidence —
 using the WS-3 real cohort as the reference. This match uses *any* real frames
 (it is a statistics comparison, not a pointing comparison), so it needs **no star
 frames** and is therefore achievable for Voyager / Galileo too — which matters,
 because for those instruments the sim is the *only* absolute-accuracy basis (see
 WS-1 applicability), making their realism match the load-bearing evidence. Where
 no accurate independent PSF/shape exists for an instrument (a real risk for
 Voyager/Galileo), say so and treat that instrument's sim accuracy as bounded by
 the unverified forward-model fidelity, not as measured.
- Recovery accuracy on the sim is only credible to the extent the realism match
 holds; report the match quality alongside the accuracy.
- **Rewrite `simulator_report.rst`** so each accuracy number is reported *as a
 function of model mismatch*, the zero-mismatch column is explicitly labelled
 "self-consistency floor (not accuracy)," and the realism-match evidence is
 presented as the precondition for reading the mismatch curves as accuracy.

**Acceptance criteria.**
- No quantity reported as "accuracy" is computed with image and model sharing a
 rendering function (asserted by a test on the import/call graph).
- The report presents error vs PSF/shape/photometric/ephemeris mismatch, the
 realism-match evidence, and the labelled self-consistency floor.
- The simulated-vs-real distributional match is *quantified and reported* per
 instrument for the features each technique consumes (a described figure of merit,
 not a pass/fail threshold — there is no spec; see Premise). Sim accuracy numbers
 are presented as credible only to the degree that reported match supports.

**Dependencies:** WS-3 (real frames to match against — the cohort, not the WS-1
analysis, so there is no cycle: WS-3 → WS-2 → WS-0 → WS-1). Gates WS-0. **Risk:** high — a credible independent + realistic image forward
model is the crux of the whole validation story.

### WS-17: Geometric distortion model validation (prerequisite for real-image agreement)
**Closes:** the confound that would otherwise inflate WS-1 disagreements (and any
plate solve), called out in the critical-report discussion of Voyager/Galileo
distortion.

**Problem.** WS-1 compares features at *different field positions*; ring/limb fits
span the frame. If the per-instrument geometric distortion model is wrong, two
*correct* techniques at different field positions disagree purely from distortion.
This is **not** a common-mode (hidden) term — distortion acts *differently* across
the field, so it shows up in the measured disagreement and *inflates apparent
technique error*. It must therefore be **corrected per feature-position before the
differences are formed** — not subtracted as a single aggregate "budget." Distortion
enters the key identity `Var(limb−ring)−Var(disc−ring)` as a per-frame vector
(distortion-at-the-limb-position minus distortion-at-the-ring-position); the clean
way to remove it is to apply the distortion model to each feature's pixel
coordinates first (same per-position rigor the plan applies to the ephemeris bias),
leaving only the model's *residual* uncertainty in the budget. Voyager and Galileo
are badly distorted; this must be pinned before their agreement numbers mean
anything.

**Tasks (split by whether star frames exist).**
- **Cassini (and NH LORRI if star frames exist):** validate/refine the distortion
 model from real star fields — a plate solve's residuals *as a function of field
 position* expose distortion directly (the one job star solves are good at, no
 spacecraft-position truth needed). Note the residual is not pure distortion: it
 also carries the star catalog's astrometric error (field-position-independent, so
 separable from distortion's smooth field pattern) **and the star centroiding error,
 which is *not* field-position-independent** — the PSF degrades toward the edges and
 corners, exactly where distortion is largest, so the two grow together and are
 *not* cleanly separable by their field dependence. Model the edge-dependent
 centroiding error explicitly (from the PSF map) and propagate its uncertainty
 rather than assuming it away. Then apply the distortion model per feature-position
 in WS-1 (above) and feed the field map into the WS-12 appendices.
- **Voyager / Galileo (zero star frames — the star route is impossible for exactly
 the most-distorted instruments):** there is no in-house way to validate
 distortion. **Adopt the published Voyager ISS / Galileo SSI geometric-distortion
 solutions from the calibration literature as given** (cited per the static-data
 policy), and use **body+ring agreement vs field position** as a coarse residual
 check only — if disagreement grows toward the frame edges, the adopted model is
 suspect. This check is entangled with other errors and is a sanity test, not a
 calibration.
- NH LORRI: low-distortion optics, so even absent star frames the distortion risk
 is small; document the adopted model and move on.

**Acceptance criteria.** Cassini (and LORRI if possible) distortion is quantified
from star residuals (with catalog error separated out and the edge-dependent
centroiding error modelled, not assumed flat) and applied **per feature-position**
in WS-1, leaving only its residual uncertainty in the budget. Voyager/Galileo
document the adopted literature distortion model with a field-position agreement
sanity check, and their agreement/accuracy claims are scoped to "literature
distortion assumed, unvalidated in-house."

**Dependencies:** WS-3 (star-field frames where they exist). **Risk:** high for Voyager/Galileo — distortion cannot be independently validated
there, which directly limits how far their real-image agreement numbers can be
trusted.

---

## Phase 1 — Make the safety net real

### WS-3: Expand the real-image regression cohort
**Closes:** "real-image regression rests on ~13 hand-blessed images."
**Tracked by:** #172 (the 49-image first stage; workflow
`plans/COHORT_CURATION_PLAN.md`) and #174 (integration tests + baselines);
#235 carries the growth beyond that stage.

**Tasks.**
- Complete the 49-image curated stage (#172), then grow
 `tests/integration/image_library/` to the target of **≥20 per instrument, ≥120
 total** — the size the WS-1 agreement study needs — spanning the geometry
 taxonomy already present (full-FOV body, partial overflow, below-resolution,
 high-phase terminator, multi-body, ring curved/flat, ring+body, star-dominated,
 faint stars, scattered light, negative cases).
- Expand `image_library/README.md` (currently a minimal schema/registry note)
 with the curation workflow, blessing/re-blessing procedure, and the
 ground-truth provenance for each entry.
- Add explicit **negative/failure cases** (unnavigable frames) and assert the
 system fails cleanly with the right status reason.

**Acceptance criteria.**
- Cohort size and per-category coverage meet the documented targets (49-image
 stage first, then ≥20 per instrument / ≥120 total).
- `README.md` documents schema + curation + blessing + provenance; every sidecar
 records its ground-truth source.

**Dependencies:** feeds WS-1, WS-7. **Risk:** low.

### WS-4: Run real-image tests in CI
**Closes:** "CI never runs integration tests."

**Problem.** `.github/workflows/run-tests.yml:94` runs `-m "not integration"`;
nothing real is exercised automatically.

**Tasks.**
- **Fast integration tier:** cache a small set of real images + the minimal SPICE
 subset (or stand up a read-through holdings cache) so a curated subset runs on
 every PR in a bounded time budget. Mark these `integration_fast`.
- **Nightly/weekly full tier:** run the complete integration + accuracy suite on a
 schedule (the repo already has a weekly cron) with the holdings/catalog env vars,
 uploading the accuracy report as an artifact and failing on regression beyond a
 tolerance band.
- Add an **accuracy-regression gate**: compare per-technique median/95th error
 against a committed baseline; fail if it degrades beyond threshold.

**Acceptance criteria.**
- Every PR runs at least the `integration_fast` tier with real images.
- A scheduled job runs the full accuracy suite and gates on regression.
- A documented, reproducible mechanism supplies images + kernels to CI.

**Library consumers and CI tiers (binding note).** The image library has four
distinct consumers, and only the smallest of them ever runs per PR:

| Consumer | What runs | When |
|---|---|---|
| Sim tier (baselines + sim navigation/invariant tests) | fast, no network | every default `pytest` and every PR |
| `integration_fast` real-image subset (~5-10 cached frames) | bounded minutes | every PR |
| Full library regression + accuracy suite | the whole library | nightly/weekly schedule only |
| Calibration (WS-5) and agreement study (WS-1) | offline analysis producing reports and coefficients | on demand, never CI-gated per PR (only the cheap calibration-drift check joins the scheduled tier) |

The arithmetic forbids anything else: at ~35 s/frame a 120-image library is
over an hour of compute before downloads. Do not wire the full library, the
calibration sweep, or the agreement analysis into per-PR CI; the per-PR gate
is the sim tier plus the small cached `integration_fast` subset.

**Dependencies:** WS-1, WS-3. **Risk:** medium — kernel/holdings
provisioning in CI is the hard part; a cached fixture bundle is the mitigation.

---

## Phase 2 — Stop shipping unbacked numbers

### WS-5: Calibrate (or quarantine) confidence and tiers
**Closes:** "ships a confidence it admits is meaningless."
**Tracked by:** #230 (the real-anchored calibration — this workstream defines
its methodology, binding per the relationship section above; the sim-anchored
interim landed via #173 / PR #208) and #176
(constants into config, which lands before calibration writes coefficients).
The curated library's operator-assigned `confidence_tier` labels serve as
plausibility cross-checks and regression expectations for the calibrated
output; they are never fit targets.

**Problem.** `confidence`/`confidence_tier` are emitted per image but the sigmoid
coefficients and tiers are uncalibrated defaults
(`nav_technique/confidence*.py`, `confidence_config.py`,
`config_510_techniques.yaml`).

**Real-data anchor.** There is no per-frame "achieved pixel error" on real images
(no external truth). The anchors are, in order of strength: per-technique covariance
from WS-1 *where WS-0 says it is identifiable, bias-independent, and shape-clean*;
pairwise combined covariance elsewhere; per-frame cross-technique disagreement; and
the de-circularized sim (WS-2) for single-technique regimes WS-1 cannot reach.

**Calibration is therefore patchy by regime — say so.** A *per-technique* confidence
can be calibrated against a real anchor only where per-technique covariance exists
(the narrow WS-0-qualified, shape-clean, geometry-controlled set). Everywhere else
the only real anchor is *combined* pairwise covariance, from which a per-technique
confidence is under-determined — there the calibration falls back to the sim (WS-2)
and is labelled sim-anchored, not real-anchored. The result is a confidence whose
*calibration basis* varies by regime; the metadata records which basis each value
used. **Avoid the feedback loop:** the per-frame disagreement used to calibrate must
come from independently-run techniques, and the calibrated confidence must not then
re-weight those same techniques' ensemble fusion in a way that feeds the calibration
target.

**Tasks.**
- Build per-technique **reliability diagrams** against the anchor: reported
 confidence (and reported σ) vs the WS-1 per-technique error variance *on
 identifiable bins*, the pairwise combined σ on non-identifiable bins, and sim
 recovery error (WS-2) where no multi-object cohort exists. Carry the anchor's own
 uncertainty interval into the diagram.
- Fit a **monotonic calibration map** (isotonic regression or temperature/Platt
 scaling on the existing sigmoid) so reported confidence tracks empirical
 reliability, and so each technique's reported σ is consistent with its WS-1/sim
 error estimate (coverage check: does cross-technique disagreement fall within the
 combined reported σ at the expected rate?).
- **Redefine tiers** by WS-1/sim error percentiles, not by hand.
- Add a **calibration regression test** that recomputes the reliability diagram and
 fails if calibration drifts.
- **Interim safeguard (ship immediately, before calibration lands):** mark the
 field `confidence_provisional` in `_metadata.json` and document it as
 uncalibrated, so no user mistakes it for a probability in the meantime.

**Acceptance criteria.**
- Confidence is calibrated against the WS-1 anchor where per-technique covariance
 exists and sim-anchored elsewhere, with a published reliability diagram carrying
 the anchor's uncertainty; per-frame cross-technique disagreement falls within
 combined covariance at the expected rate.
- Each emitted confidence records its **calibration basis** (real-anchored vs
 sim-anchored) in the metadata; the regime patchiness is documented, not hidden.
- Calibration does not consume a target that depends on the calibrated confidence
 (no feedback loop), and tier boundaries map to stated error percentiles.
- A test guards calibration drift.

**Dependencies:** WS-0 (which bins yield per-technique σ), WS-1 (anchor), WS-2
(single-technique regimes). **Risk:** medium — may reveal that some
techniques' covariances are mis-scaled (feeds WS-9).

---

## Phase 3 — Close the capability gaps (or scope them out honestly)

### WS-6: Reconcile claims with reality (capability matrix)
**Closes:** "unfinished rewrite wearing a finished manual," forward-looking
vapor.

**Tasks.**
- Author a single **capability matrix** (instrument × {navigate, backplanes,
 mosaic, PDS4 input, PDS4 bundle, manual}) with states {supported, partial,
 not-supported}, generated/verified from the registries and a test so it cannot
 drift from code. **A binary "navigate: supported" cell is not enough** — it would
 flatten the very nuance the rest of this plan establishes (Cassini's validated
 relative covariance + absolute attitude on dozens of frames vs Voyager/Galileo's
 sim-only absolute + literature distortion). Add a second **validation-status**
 axis per (instrument, technique): {relative-σ validated at scale, per-technique-σ
 on curated set, absolute-attitude anchored, sim-only}. The matrix must carry the
 accuracy-evidence tier, not just feature existence, and link to the WS-1/WS-2
 reports for the numbers.
- Sweep `docs/` for "placeholder / reserved for / pending / not yet implemented /
 future enhancement"; move each to a **roadmap** page or delete it. The user
 guide describes only what exists.
- Audit **reserved-but-unwired config keys** (curvature/roughness limb filters,
 ring fiducial promotion, `remove_body_shadows`, `min_emission_ring_body`,
 curvature reductions): either implement, or remove from shipped YAML and the
 docs so the config surface stops advertising absent features.

**Acceptance criteria.**
- A capability matrix exists, is test-verified, and is the single referenced
 source for "what works."
- No shipped config key is documented as functional unless the code consumes it.

**Dependencies:** light coupling to WS-7/8. **Risk:** low.

### WS-7: Titan / atmospheric-body navigation — implement or scope out
**Closes:** "Titan is a no-op."
**Tracked by:** #60 (Implement Titan navigation) for the "implement" branch.

**Problem.** `nav_model_titan.py:48` `to_features()` returns `[]`.

**Decision gate (do this first):** is haze-limb navigation in scope for this
release? Pick one:
- **Implement:** a haze-aware limb model — per-filter haze-top altitude
 profiles, a haze-limb feature type, and a DT/edge technique that fits the haze
 top instead of the solid limb, validated on real Titan frames via WS-1.
- **Scope out:** remove Titan from supported-target claims, mark it
 not-supported in the capability matrix, and ensure the orchestrator degrades
 gracefully (Titan in FOV falls back to rings/stars/other bodies, no silent
 empty-result confusion).

**Acceptance criteria.** Either real Titan frames navigate within a stated bound,
or Titan is unambiguously documented as not-supported and handled gracefully.

**Dependencies:** WS-1 if implementing. **Risk:** high if building (haze physics is genuinely hard).

### WS-8: PDS4 — finish input and generalize bundle generation
**Closes:** "PDS4 is largely fictional."
**Tracked by:** #53 (bundle generator parent), #67 (cloud-aware bundles), #34
(PDS4 Cassini input when the archive is available).

**Problem.** `dataset_pds4.py` raises "not yet implemented" for all methods (no
PDS4 input); `pds4_*` bundle hooks raise `NotImplementedError` for Voyager,
Galileo, NH.

**Tasks (scope per release decision):**
- **PDS4 input:** implement `DataSetPDS4` enumeration/reading, or remove the
 `*_pds4` dataset names from the user-facing list until implemented.
- **Bundle generalization:** implement `pds4_*` hooks (template dir, LID/LIDVID,
 template variables) for Voyager/Galileo/NH using `DataSetPDS3CassiniISS` as the
 reference, with per-mission template trees under `src/pds4/templates/`; or
 document bundle generation as Cassini-only in the capability matrix.
- Add bundle-validation tests (schema-validate generated `.lblx` against PDS4
 schemas) per supported instrument.

**Acceptance criteria.** Every instrument's PDS4 status in the capability matrix
matches reality; for each "supported" instrument, generated bundles pass PDS4
schema validation in CI.

**Dependencies:** WS-6. **Risk:** medium.

### WS-12: Write the instrument appendices
**Closes:** "the instrument appendices are empty."
**Tracked by:** #93.

**Tasks.** Replace the four placeholder appendices
(`user_guide_appendix_{coiss,gossi,nhlorri,vgiss}.rst`) with real content:
required kernels, plate scale/FOV, PSF characteristics, geometric distortion
(esp. Voyager), known artifacts (NAC haze/ghosts, LORRI), per-instrument config
knobs and their defaults, and instrument-specific gotchas. Source the per-
instrument numbers from the config files and the static-data citations.

**Acceptance criteria.** Each appendix documents kernels, optics, distortion,
artifacts, and the config knobs a user touches; none is a placeholder.

**Dependencies:** light coupling to WS-9 (constant justifications). **Risk:** low.

---

## Phase 4 — Make the numbers defensible

### WS-9: Justify, derive, or measure the magic constants
**Closes:** "the numbers a scientist would quote are built on hand-picked
constants," and the fabricated star SNR.
**Tracked by:** #176 (constants into config) and #130 (star limiting-magnitude
calibration).

**Tasks.**
- Inventory the load-bearing constants: `ROTATION_UNOBSERVABLE_VARIANCE = 1e15`
 (`nav_technique.py:46`), `DEFAULT_PINVH_RCOND = 1e-9` (`dt_fitting.py:94`),
 `SNR_REF = 8.0` / `SNR_FLOOR = 0.1` (`nav_model/stars/nav_model_stars.py:77-78`),
 blob noise thresholds, MAD factor, edge thresholds. For each: document its derivation,
 sensitivity, and the regime where it holds, next to its definition.
- **Measure star SNR from the image, not the magnitude.** Replace (or cross-check)
 the synthesized `snr_eff = SNR_REF * 2.512**(mag_limit - vmag)` with a
 photometrically measured per-star SNR from the actual frame for gating and
 covariance, keeping the magnitude estimate only as a *prior* where no pixels are
 available. Validate that recovered star covariances pass the WS-5 coverage test.
- Add a **sensitivity test** that perturbs each constant and asserts results are
 stable within the documented tolerance (flags hidden over-fitting to a constant).

**Acceptance criteria.** Every load-bearing constant has a documented derivation
and a sensitivity bound; star SNR used for covariance derives from measured
photometry; covariance coverage passes WS-5.

**Dependencies:** WS-1/WS-5 for coverage validation. **Risk:**
medium — measured SNR may change star gating behavior and need re-tuning.

### WS-10: Fix the limb systematic bias
**Closes:** "known systematic, no working fix."
**Tracked by:** #150 (model-vs-image edge offset) and #128 (limb-navigation
redesign).

**Problem.** Limb bias ~0.09–0.13 px (≤0.25 px two-axis); the implemented
gradient-ridge refine is disabled for the limb technique
(`config_510_techniques.yaml:237`) because it worsens limb fits there, while the
ring-edge technique runs with it enabled (`:354`). The limb's current partial
cancellation (integer DT quantization + Tukey) is accidental.

**Tasks.**
- Root-cause why gradient-ridge refine degrades the limb (the model-side limb-edge
 issue referenced as #150): the model predicts the geometric silhouette while the
 gradient ridge sits ~0.1 px inside it due to PSF. Two candidate fixes:
 1. **Model the PSF-inward offset** in `nav_model_body.py` so the predicted limb
 position matches where the gradient ridge actually falls (forward-model the
 bias out), then enable continuous gradient-ridge refine.
 2. **Sub-pixel DT** (replace integer-quantized distance transform with a
 continuous/interpolated one) so accuracy stops relying on a quantization
 accident.
- Re-measure limb accuracy on the WS-2 (mismatched-model) and WS-1 (real) cohorts;
 enable `gradient_ridge_refine` only if it demonstrably reduces real bias.

**Acceptance criteria.** Limb median bias reduced to a stated target (e.g.
≤0.03 px) on de-circularized synthetic and real cohorts, with the fix enabled by
default; no reliance on accidental cancellation documented as a feature.

**Dependencies:** WS-1, WS-2. **Risk:** medium-high — this is a
genuine algorithmic problem the team has already struggled with.

### WS-13: Make the calibrated (I/F) path realistic and tested
**Closes:** "the calibrated (I/F) path is essentially untested."

**Tasks.** Add a detector-noise model to the I/F render path in `sim/render.py`
(Poisson shot noise in electrons before conversion, read noise, full-well
saturation, bias pedestal, missing-data/CR markers), so I/F scenes exercise a
realistic noise regime. Add I/F frames to the WS-2 sweeps and the WS-1
consistency study (real calibrated products).

**Acceptance criteria.** I/F scenes carry a realistic detector model; the
accuracy and sweep reports include calibrated-path results comparable to the
raw-DN path.

**Dependencies:** WS-2. **Risk:** low-medium.

---

## Phase 5 — Operational hardening

### WS-11: Correctness of degenerate-rotation reporting
**Closes:** "small rolls are silently wrong."

**Tasks.** In `nav_technique_star_field.py` (and the ensemble), where the RANSAC
matcher returns roll below the ~0.75° separability floor, **report rotation as
unobservable** (use the `ROTATION_UNOBSERVABLE_VARIANCE` sentinel + an explicit
flag in diagnostics/metadata) instead of returning a spurious `0.0`. Document the
separability floor per instrument. Add a test that a sub-floor planted roll yields
an "unobservable" flag, never a confident zero.

**Acceptance criteria.** No frame reports a confident `0.0°` roll within the
non-separable regime; the metadata carries an explicit unobservable flag.
**Dependencies:** none. **Risk:** low.

### WS-14: Complete the provenance block in output metadata
**Closes:** the remaining gaps in "garbage in, garbage out, with no audit trail."

**Scope.** `nav_orchestrator/provenance.py` + the curator already pin the git
SHA, the loaded SPICE-kernel list, the static-data YAML hashes, and the run
timestamp into every `_metadata.json`. What remains:

**Tasks.** Add to the provenance block: a hash of the *resolved* config plus the
applied user/CLI overrides, and the star-catalog versions used. Document how to
reproduce a result from the block, and add a reproduce-from-metadata test.

**Acceptance criteria.** Any archived result can be traced to exact kernels,
config (including overrides), code version, and catalogs from its metadata
alone; a reproduce-from-metadata test passes.
**Dependencies:** none. **Risk:** low.

### WS-15: Performance and safe parallelism
**Closes:** "it is slow," "the obvious fix for slowness is mined."
**Tracked by:** #103 (thread-unsafe caches), #134 (oops precision mutated
process-globally), #126 (rotation-pyramid cost).

**Tasks.**
- **Thread-safety:** remove the global `oops` precision mutation in
 `reproj/rings.py` (`_reduced_oops_precision`) in favor of a per-instance/per-
 thread setting, and make `BodyMosaic.reproject` / `create_cartographic_model`
 build per-thread `Backplane` objects. Add a concurrency test that runs
 reprojection on shared geometry across threads and checks bitwise-identical
 results vs serial.
- **Profile** the ~35 s/1024-px navigation (`--profile` already exists); attack the
 dominant cost (likely backplane/model rendering and DT construction). Target a
 documented per-frame budget.
- **Star matching:** bound or improve the O(M³) triplet hash (cap `max_sources`
 with brightness pre-selection, or switch to a smarter matcher) and document the
 complexity.
- Provide a **supported batch-parallel path** (process- or task-level) with a
 documented throughput figure for a realistic campaign size.

**Acceptance criteria.** Reprojection is thread-safe with a proving test;
per-frame navigation cost is reduced against a documented baseline; a supported
parallel batch path exists with a published throughput number.
**Dependencies:** none. **Risk:** medium — `oops` global state may
be deep.

### WS-18: End-product geometric accuracy (backplanes, mosaics, PDS4)
**Closes:** a gap the critical report implied but no other workstream covers — the
deliverables are not the offset, they are the *products built on it*, and their
correctness is unvalidated.

**Problem.** WS-1/WS-2 validate the navigation *offset*. But the science outputs
are backplanes (per-pixel geometry), mosaics (reprojected grids), and PDS4 bundles.
A correct offset can still feed a wrong backplane angle or a mis-registered mosaic
seam through a downstream bug; none of that is tested for accuracy today.

**Tasks.**
- **Backplanes:** on the WS-2 sim, where per-pixel
 lon/lat/incidence/emission/phase/radius/resolution are known by construction,
 assert the generated backplanes match to a stated tolerance. **The sim-truth
 geometry must be computed independently of the production backplane generator**
 (do not let both come from the same `oops.Backplane` calls), or "backplane matches
 sim truth" is the same self-consistency circularity WS-2 exists to break, one layer
 down. On real frames, cross-check internal consistency (e.g. ring-radius backplane
 vs the navigated ring geometry).
- **Mosaics:** quantify seam registration where overlapping reprojections meet
 (this is WS-1b's reprojection-consistency check applied to the mosaic product),
 and confirm the BEST_RESOLUTION/coverage merge picks the right pixel.
- **PDS4:** beyond schema validation (WS-8), spot-check that label geometry values
 match the backplane metadata they are derived from.

**Acceptance criteria.** Backplane values match sim ground truth within a stated
tolerance and are internally consistent on real frames; mosaic seam registration is
quantified; PDS4 label geometry matches its source metadata.

**Dependencies:** WS-1b, WS-2, WS-8. **Risk:** low-medium.

---

## Sequencing and milestones

Dependency lists (each line: workstream — what it needs). No box-art, so it
survives reflow.

- **WS-0a** (estimator math + identifiability map) — a truth-known sim only; starts
 now, parallel with everything.
- **WS-3** (cohort) — nothing; starts now.
- **WS-2** (de-circularized + realism-validated sim) — WS-3.
- **WS-0b** (bias-independence verdict) — WS-2.
- **WS-17** (distortion) — WS-3 (star frames where they exist).
- **WS-1 pairwise layer** (combined covariance — the bulk product) — WS-3 + WS-17
 only. **Does not wait on WS-0/WS-2.**
- **WS-1 per-technique layer** (separation) — additionally WS-0 (+ WS-2 through it).
- **WS-1b** — WS-3; shares its implementation with WS-18.
- **WS-4** (CI) — WS-3 (+ WS-1 for the accuracy-regression gate).
- **WS-5** (confidence) — WS-0, WS-1, WS-2.
- **WS-6** (capability matrix) — light coupling to WS-7/8.
- **WS-7 / WS-8 / WS-12** (Titan / PDS4 / appendices) — decision gates first.
- **WS-9 / WS-10 / WS-13** (constants / limb bias / I/F) — WS-1 and/or WS-2.
- **WS-11 / WS-14 / WS-15** (rotation / provenance / performance) — independent;
 start anytime.
- **WS-18** (end-product accuracy) — WS-1b + WS-2 + WS-8.

WS-1 gates WS-5, WS-9, WS-10 (WS-18 gates on WS-1b, not WS-1 directly).

Note: WS-0 splits — **0a** (estimator code + identifiability map) needs only a
truth-known sim and starts immediately in parallel with WS-2; **0b** (the
bias-independence verdict) needs the realistic WS-2 sim and is cross-checked on real
data by WS-1's over-determination test.

Note: the chain is **WS-3 → WS-2 → WS-0 → WS-1** (no cycle). WS-2 (sim) carries the
absolute-accuracy number and is also where WS-0 proves the WS-1 estimator before any
real-image per-technique σ is believed. WS-1/WS-17 supply real-image *agreement*;
neither alone is "the" accuracy — sim accuracy + agreement + the stated
common-mode/identifiability/bias caveats together are the honest result.

- **Milestone A — "Honest" (fast wins, no new science):** WS-11, WS-14, the WS-5
 interim provisional label, and WS-6's doc reconciliation. The system stops
 over-claiming and stops emitting silently-wrong values. These are independent and
 parallelizable.
- **Milestone B — "Measured-as-far-as-possible":** WS-3 + WS-2 + WS-0 + WS-17 +
 WS-1 (+WS-1b) + WS-4. WS-0 first proves the agreement estimator on the sim and
 maps where per-technique σ is even recoverable; WS-1 then reports pairwise
 combined σ everywhere and per-technique σ only on the qualified bins, absolute
 attitude from the scarce star tie-points plus the realism-validated sim. CI
 exercises real frames. The honest output is "pairwise agreement at scale +
 per-technique σ where identifiable + absolute attitude on a small stellar sample
 + sim, with stated caveats," not a single absolute px figure.
- **Milestone C — "Calibrated & defensible":** WS-5 (full), WS-9, WS-10, WS-13.
 Confidence and covariance mean something; the limb bias is fixed; the I/F path is
 real.
- **Milestone D — "Complete & scalable":** WS-7, WS-8, WS-12, WS-15, WS-18.
 Capability gaps closed or honestly scoped; end products validated; performance and
 parallelism fit a campaign.

---

## Traceability: every critical-report finding → workstream

| # | Critical-report finding | Workstream(s) | "Closed" when |
|---|---|---|---|
| 1 | Validation is circular (shared forward model) | WS-2 | navigator uses its best model; image side independent + more-realistic; accuracy reported vs *residual* mismatch with realism-match evidence; no shared render fn under test |
| 2 | No real-image accuracy assessment | WS-1 (+WS-0, WS-1b, WS-17) | agreement report from real frames: pairwise combined σ everywhere, per-technique σ (with uncertainty intervals) only on WS-0-qualified identifiable, bias-independent bins; ephemeris carried as a separate bias; absolute attitude from scarce star tie-points + sim; explicit relative-vs-absolute split (external truth unobtainable) |
| 2′ | Per-technique σ unproven (identifiability + bias independence) | WS-0 | estimator recovers known sim σ; identifiability map published; bias independence established by injection test, not assumed |
| 3 | ~13-image cohort; CI skips integration | WS-3, WS-4 | cohort targets met; real images run in CI + nightly accuracy gate |
| 4 | Ships uncalibrated confidence/tiers | WS-5 | calibrated where per-technique covariance exists, sim-anchored elsewhere (basis recorded per value); reliability diagram; interim provisional label |
| 5 | Mid-rewrite; docs ≠ code; vapor; dead config | WS-6 | test-verified capability matrix; docs/config reconciled |
| 6a | Titan is a no-op | WS-7 | implemented+validated, or scoped-out + graceful |
| 6b | PDS4 input absent; bundles Cassini-only | WS-8 | matrix matches code; bundles schema-validate per supported inst |
| 6c | Empty instrument appendices | WS-12 | all four appendices written |
| 7a | Covariance shaped by magic constants | WS-9 | each constant derived + sensitivity-bounded |
| 7b | Star SNR fabricated from magnitude | WS-9 | measured-photometry SNR; covariance coverage passes |
| 8 | Limb bias, fix disabled | WS-10 | bias ≤ target on real+mismatched data, fix enabled |
| 9a | Slow (~35 s/frame), O(M³) matcher | WS-15 | per-frame budget cut; matcher bounded; throughput published |
| 9b | Reprojection thread-unsafe | WS-15 | per-thread state + proving concurrency test |
| 9c | Sub-0.75° roll returned as spurious zero | WS-11 | unobservable flag, never confident zero |
| 9d | I/F path noise-light/untested | WS-13 | realistic detector model + I/F in accuracy report |
| 9e | Provenance gaps (config-override hash, catalog versions) | WS-14 | provenance block carries resolved-config hash + catalog versions; reproduce-from-metadata test passes |
| 10 | End products (backplanes/mosaics/PDS4) accuracy untested | WS-18 | backplanes match sim truth; mosaic seams quantified; PDS4 geometry matches source |

---

## Assumptions and notes

- These workstreams assume continued access to PDS holdings, SPICE kernels, and star
 catalogs for the target instruments; provisioning these to CI (WS-4) is itself a
 task and a risk.
- **Absolute subpixel ground truth for real frames does not exist** (star solves
 measure attitude only and fail on bright-body frames; inter-frame jitter kills
 repeat-frame prediction; legacy code shares this lineage and its subpixel layer
 was never trusted). The plan therefore measures absolute accuracy only in a
 de-circularized, realism-validated sim (WS-2), and uses real-image *agreement*
 (WS-1, with distortion removed per WS-17) as corroboration, with the residual
 common-mode CK error stated as an explicit bound. Any "accuracy" claim that cannot
 be backed this way is downgraded in the capability matrix rather than invented.
- WS-2 is now the critical path and highest-risk item: the validity of every
 absolute accuracy number rests on the image-side forward model being both
 independent of the navigator and provably realistic against real frames. If
 realism cannot be demonstrated for an instrument, its accuracy numbers are
 reported as sim-only, mismatch-bounded estimates, not measured accuracy.
- Common-mode error — a CK shift that moves *all* compared features *identically* —
 is the irreducible blind spot of real-image agreement (it cancels in the
 difference). It is caught only by the scarce star frames and the sim. Note this is
 distinct from distortion, which acts *differently* across the field, shows up in
 the disagreement, and is *corrected per feature-position* (WS-17) rather than being
 a hidden term.

**Statistical assumptions the per-technique numbers depend on (each stated in the
report, none silently assumed):**

- **Bias independence** of any two techniques entering a joint solve — **including
 through their shared preprocessing** (the `image_derivatives` DT/gradient images,
 noise-σ, reliability gate built once per `NavContext`, which couple all DT
 techniques). Verified by WS-0's injection test *through that shared layer*, not
 assumed; *no* pair (not even disc-NCC vs limb-DT) is treated as clean until
 measured. The two load-bearing pairs are limb-disc (the base equation) and
 limb-ring.
- **The recovered quantity is a 2x2 covariance, not a scalar.** Limb error is
 anisotropic and its axis rotates relative to the fixed ring-radial direction
 frame to frame, so per-technique error is recovered only as covariance elements on
 controlled geometry (bin also by limb-orientation-vs-radial, or solve in matrix
 form). Scalar-σ framing is wrong (WS-0/WS-1).
- **Techniques estimate the *same* offset** — true only for round, photometrically
 bland bodies. On topographic/high-contrast bodies the limb-center and disc-center
 differ by a real physical amount that inflates the apparent disagreement, so such
 bodies are excluded from per-technique separation by a reproducible gate derived
 from the static body-shape table (`ellipsoid_rms_residual_km`, `crater_scale_km`,
 `albedo_variation`), not a judgment call (WS-1 body-shape caveat).
- **Identifiability** of the covariance system per bin (WS-0). The common limb+disc
 frame is *not* separable; per-technique covariance is reported only on qualified,
 shape-clean, geometry-controlled bins, combined pairwise covariance elsewhere.
- **Errors are well-behaved enough for variance estimation** — robust estimation
 with outlier rejection is used, and *failure rate* is reported separately from
 error covariance (a catastrophic-failure technique and a large-error technique are
 different products) (WS-1).
- **The assumptions are checkable on real data only via over-determined frames**
 (limb+disc+ring+second body); their closure is the one real-photon test of
 bias-independence/stationarity/ephemeris, since WS-0's verdict is the sim's (WS-1).
- **Within-bin stationarity** of each technique's error covariance (WS-1) — assumed
 by the *pairwise* product too, not only the separation, so it is the broad limiter.
 Inherited by WS-5's calibration and WS-2's realism binning. Bin on resolution /
 phase / lit-fraction / limb-orientation and spot-check; if it fails, report the
 disagreement distribution rather than a single covariance.
- **Moon ephemeris error is a per-target bias, not zero-mean noise** — modelled as a
 separate nuisance parameter, never folded into a technique's σ (WS-1).
- **An accurate, independent PSF/shape model exists** to drive WS-2's realistic
 image. Solid for Cassini, doubtful for Voyager/Galileo; where it does not, that
 instrument's sim accuracy is bounded by unverified forward-model fidelity, not
 measured.
- **Saturn's ephemeris error is negligible at subpixel scale** (used in the
 star+ring relation) — to be quantified, not assumed.
- **Star-catalog astrometric error is separable from distortion** in WS-17 (it is
 field-position-independent; distortion is a smooth field pattern). **Centroiding
 error is *not* separable that way** — it grows toward the edges with the PSF, where
 distortion is also largest — so it is modelled explicitly from the PSF map and its
 uncertainty propagated, not assumed flat. Distortion is then applied per
 feature-position in WS-1, leaving only its residual in the budget.
- **Realism validation needs only real frames, not star frames** (it is a
 statistics match), so WS-2 realism — hence the sole absolute-accuracy basis for
 Voyager/Galileo — is achievable for every instrument.
- **Validation reach is sharply instrument-dependent and must be reported per
 instrument, not pooled.** *Cassini ISS:* relative/algorithm σ at scale **plus**
 an absolute-attitude anchor from dozens of star tie-points — the only fully
 validatable instrument. *NH LORRI:* star-frame count to be verified (WS-3);
 low-distortion optics reduce the risk regardless. *Voyager ISS / Galileo SSI:*
 effectively zero star frames, so (a) absolute attitude accuracy is **sim-only**
 (no real-data inertial reference exists) and (b) geometric distortion cannot be
 validated in-house and is **adopted from the literature** (WS-17) — and these are
 the most-distorted instruments, so their real-image numbers carry the weakest
 backing. The agreement report must scope every number to its instrument's reach.
- (Within-bin stationarity, covered above, is the broad limiter — it constrains the
 *pairwise* fallback too, which is *less* assumption-laden than the separation but
 not assumption-free.)
- Scope decisions (Titan build vs scope-out; PDS4 breadth) should be made at the
 WS-6/WS-7/WS-8 decision gates before committing build effort; the plan supports
 either branch and the matrix keeps the docs honest regardless.
- **Validation can come back bad.** There is no fixed accuracy spec to pass (goal is
 best-achievable, honestly characterized), so a poor result is not a go/no-go gate
 — but it is not merely reported either: a worse-than-hoped σ routes into the
 algorithm workstreams (WS-10 for the limb; WS-9 for covariance scaling; new work
 if a technique is structurally weak). The plan does not assume validation merely
 confirms.
- Nothing here changes code or docs yet; this is the plan of record.
