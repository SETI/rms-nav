<!-- Frozen snapshot 2026-08-07: independent fresh-context adversarial review of
plans/CMATRIX_READERS_PLAN.md before implementation. Findings are to be
dispositioned in the plan and on issue #50; this document is not maintained. -->

# Adversarial Review: `plans/CMATRIX_READERS_PLAN.md`

Reviewed 2026-08-07 against the repo at commit `959ea1a` (branch
`rf_cmatrix_readers`). Every load-bearing claim was checked against source
(`src/spindoctor/support/cmatrix.py`, `src/spindoctor/cli/reproj/offsets.py`,
`src/spindoctor/cli/backplanes/backplanes.py`, `src/spindoctor/reproj/`, the
installed oops under `venv/`, `tests/integration/test_ck_round_trip.py`,
`docs/dev_guide/dev_guide_ck_kernels.rst`), and the checkable oops and numerical
claims were executed as probes with the project venv and local SPICE/holdings
data. Probes are quoted inline.

**Verdict: CONDITIONAL.** The core mechanism is right and was verified
end-to-end by execution on real frames: the conjugation algebra is correct, the
`R_hat` gate genuinely catches every dangerous drift/transpose/foreign-record
class, `oops.frame.Cmatrix(frame_id=None)` really registers nothing, and frame
replacement works through `Backplane` and `Event` on a real observation. But the
plan carries one false mathematical claim that its own named unit test will
expose (the bit-for-bit identity), a fallback rule that produces a product wrong
by roughly twice the offset in precisely the operational state the plan's own
follow-up #437 sets out to create, an acceptance phase that never runs the
mosaic consumers it switches, an agreement bound whose measurement metric is not
the metric its constants were measured in (verified numerically to matter at the
24% level on the WAC), and a stated mutation invariant that `from_file` already
violates. None of these blocks starting; all of them must change in the plan
before the corresponding phase is implemented.

---

## Findings

### MAJOR

**1. The gate-violation fallback double-corrects in the one foreseeable state
where the gate actually fires, and that state is the plan's own follow-up.**
Section 3.4 sends every `R_hat` gate failure to the offset path (`OffsetFOV`),
reasoning that "the degradation is to current behavior" and that pool drift "is
a defect condition, not an operational state to optimize for." But section 7's
first two follow-ups (#434 aside) plan for exactly the non-defect state: #437
"registers the corrected kernels for oops selection." The moment corrected
kernels are furnished at load time -- by #437, or by any operator who furnishes
the shipped kernels the writing half exists to produce -- `C_oops(mid)` becomes
`R . cmatrix`, so `R_hat = R . cmatrix . cmatrix_original^T` differs from
`R_expected` by the full correction angle (1e-5 rad scale, seven orders above
the 1e-9 gate) on **every kernel-covered image**. The reader then warns and
applies the offset **on top of an already-corrected observation**: a product
wrong by ~2x the offset, several pixels on every frame, built and shipped with
only a log line and a counter. Voyager is worse, not exempt: its frozen `ckgp`
snap (`venv/.../oops/hosts/voyager/iss.py:133`, `tol_ticks = 800 + texp/48`)
would freeze the *corrected* attitude during `from_file`, with the same result.
The condition is cheaply distinguishable from corruption: when the pool already
answers the recorded attitude, `C_oops(mid)` agrees with `R_expected . cmatrix`
to the same 1e-9. The ladder needs a distinct row -- detect
`pool_already_corrected`, apply **nothing** (the observation is already right),
count it under its own reason -- and section 3.4's "both paths are compromised"
parenthetical needs rewriting, because in this state the offset path is
compromised and the do-nothing path is exact. Without this row, merging #437
later silently converts every gate into a double-correction engine.

**2. The "bit-for-bit" identity claim is false as derived, and the named unit
test cannot pass with the specified algebra.**
Section 3.1: "a record whose correction is the identity (`cmatrix ==
cmatrix_original`) reproduces the observation's own midtime attitude
bit-for-bit," and Phase 1 pins it with
`test_a_zero_correction_reproduces_the_observation_frame` ("equals the
observation's bit-for-bit"). The specified computation is `C_oops_corr = (C_oops
. cmatrix_original^T) . cmatrix`; with `cmatrix == cmatrix_original == B` that
is `(C . B^T) . B`, two float64 matrix products, which does **not** reproduce
`C` exactly. Probe (venv python): 200 random rotation pairs,
`np.array_equal((C@B.T)@B, C)` failed **200/200**, typical max element deviation
1.1e-16. Harmless in magnitude, but the plan asserts exactness and a TDD
implementer will hit a failing test at step one. The writer half is bit-exact
only because `_spice_cmatrix` short-circuits an exactly-identity correction and
returns the original array (`src/spindoctor/support/cmatrix.py:369-370`); the
reader needs the mirrored short-circuit spelled out (`np.array_equal(cmatrix,
cmatrix_original)` -> build the frame from `C_oops(mid)` directly). Note also
that the stated rationale "`R_hat` is used rather than the table constant so
that identity holds exactly" is therefore wrong -- identity holds exactly only
via the short-circuit, under either choice; the real (and sufficient) reason to
compute `R_hat` is the gate.

**3. Phase 3 never exercises the mosaic consumers the PR switches, and section
3.2's sentinel claim is false.**
The switch lands in `sd_mosaic.py` and `sd_mosaic_cloud_tasks.py` (call sites at
`sd_mosaic.py:190-194`, `sd_mosaic_cloud_tasks.py:291-295`), whose downstream
code has structure the backplane stage does not: `RingMosaic.reproject` consumes
the cached `obs.ext_bp` (`src/spindoctor/reproj/rings.py:951`) and constructs
`oops.Event(obs.midtime, ..., obs.path, obs.frame)` **directly from the replaced
frame object** (`rings.py:920`), inside `_reduced_oops_precision`
(`src/spindoctor/reproj/_context_managers.py:16`); `BodyMosaic` goes through
`obs.uv_from_coords` (`src/spindoctor/reproj/bodies.py:852`). Phase 3's evidence
is a bare 17x17 LOS-grid comparison plus one backplane RA/dec run -- no
`RingReprojResult`/`BodyReprojResult` is ever produced through the cmatrix path.
Consequently section 3.2's claim that "the integration comparison would also
expose any consumer that quietly sampled the frame off-midtime" is false: the
LOS comparison computes `fov.los_from_uv` + `transform_at_time(midtime)` itself
and never runs a consumer, so it can expose nothing about consumer sampling;
only the one backplane run touches real consumer code, and no mosaic code at
all. (My probe confirmed `Event` accepts an unregistered `Cmatrix` frame and
that an identity-swap reproduces `Backplane` RA/dec to 0.0 on a real Cassini
frame -- the mechanism works -- but the plan's acceptance evidence must include
at least one ring and one body reprojection compared across the two paths, or
acceptance criterion 3's "per-pixel geometry from the two paths agrees" is a
statement about raw LOS, not about the products this PR changes.)

**4. The agreement bound's constants and its measurement are in different
metrics, and at the reference offset the measurement exceeds `K_inst` in both
metrics -- the 2x factor is not covering what the plan says it covers.**
Section 4 takes `K_inst` from the writer's **pixel-space** column (WAC 7.86e-2
px; `docs/dev_guide/dev_guide_ck_kernels.rst:177-180` lists tangent-plane
9.89e-2 px vs pixel-space 7.86e-2 px -- they differ by 26% on the WAC because of
distortion), while Phase 3 measures "the J2000 line of sight per pixel ...
converted to pixels at the frame's own measured scale" -- a constant-scale
angular conversion, i.e. the tangent-plane-like metric, not the pixel-space one.
Executed probe on the real `W1580760393_1_CALIB` with a (30, 40) px offset (50
px total), 17x17 grid, comparing `OffsetFOV`-on-original-frame against
replaced-frame paths: worst residual **0.0978 px** at constant scale and
**0.0908 px** in true pixel space (via `uv_from_los` inversion) -- both
**above** `K_WAC = 7.86e-2`, so `B(frame)` alone is not conservative even in the
K table's own metric; offset direction and grid placement push past the tabled
worst case. The test bound `2B + 0.005` still holds (0.162 px), and at
library-cohort offsets (~5 px WAC) the measured ~9.8e-3 px sits well inside
~2.07e-2 -- the *test* will pass -- but the derivation's narrative ("the factor
covers grid placement differing from the 17x17 measurement") is wrong: the
factor is also absorbing a metric mismatch and direction dependence, silently.
The plan must either state the measurement metric and pin `K` measured in that
same metric, or measure in pixel space via `uv_from_los` (my probe's method
(b)), so that "measured-plus-margin" means what it says. (Also verified in the
plan's favor: the doubling ratios >2 mean the linear `B` formula over-covers all
offsets below 50 px, which is the whole library cohort; and the near-center
probe residual 9.1e-4 px matches the plan's stated distorted-FOV zero-offset
floor.)

**5. The fallback ladder misses record classes and conflates two gates under one
reason.**
Checked against the writer's actual schema (`_curate_pointing` /
`_curate_times`, `src/spindoctor/nav_orchestrator/curator.py:237-283`):

- **`cmatrix` present, `cmatrix_original` absent or malformed.** `R_hat` cannot
  be built. The writer always emits `cmatrix_original` when `pointing` exists,
  but the ladder's rows key on `cmatrix` alone and the malformed row enumerates
  only matrix-value defects ("wrong shape, non-finite, bool/str elements, not a
  proper rotation"), not an absent companion field. A truncated or hand-edited
  record hits unspecified behavior.
- **`cmatrix` present, `times` absent or `midtime_et` malformed.** The midtime
  gate (section 3.4) cannot run. Not a ladder row. NaN in `midtime_et` defeats
  the `<= 1e-6 s` comparison exactly the way the project's malformed-input
  lesson documents; the plan's malformed-domain probes cover matrices only.
- **One reason string for two distinct gates.** `cmatrix_baseline_mismatch`
  labels both the `R_hat` failure and the foreign-midtime failure (section 3.3
  table row 5 vs section 3.4's two bullets). A record from the wrong image and a
  drifted pool are different diagnoses with different operator responses; the
  run-level tally the plan adds is only as useful as the reason names.
- **"Exactly as today" is two different todays.** `backplanes.py` today
  **raises** `ValueError` on a missing `offset` key (`backplanes.py:116-117`,
  the task errors) but degrades to (0,0) on a null offset
  (`backplanes.py:118-134`), while `offsets.py` warns-and-continues in every
  non-success case (`offsets.py:189-228`). A unified `select_pointing` cannot
  preserve both; the plan must say which behavior the missing-key case gets
  after unification.

**6. "Before any geometry is computed" is an invariant `from_file` itself
already breaks; the plan should specify `reset_all()` rather than restate a
false contract.**
Section 3.1: "mutating it after `from_file` and before any geometry is computed
is the same contract `apply_offset_to_obs` already relies on." Executed probe on
the real `N1461997416_1_CALIB`: **`obs._center_bp is not None` immediately after
`from_file`** -- `ObsSnapshot.__init__` runs the closest-planet scan
(`src/spindoctor/obs/obs_snapshot.py:93-102`), which calls `body_distance` ->
the cached `center_bp` property (`obs_snapshot.py:576-586`), building a
`Backplane` against the *uncorrected* frame before any caller can apply
anything. This is benign today only by accident of physics: the sole consumers
of that pre-swap cache (`closest_planet` at `backplanes_bodies.py:100` /
`backplanes_rings.py:45`, distances in the merge) are rotation-invariant.
Nothing pins that accident, and `rings.py:951`'s `obs.ext_bp` correctness
silently depends on first-access-after-swap ordering. `reset_all()`
(`obs_snapshot.py:339-363`) clears every cached Backplane and Meshgrid,
including `_center_bp`, and costs nothing at that point in the flow.
`apply_pointing_to_obs` should call it (for both mechanisms -- the offset path
has the identical latent hazard today), and the plan's contract sentence should
state the real invariant instead.

### MINOR

**7. Section 1's consumer inventory is wrong about `sd_mosaic_display`, and "the
three mosaic call sites" do not exist.**
`sd_mosaic_display.py` never loads or applies an offset -- it displays saved
reprojection/mosaic files and imports only `add_display_args` from `cli/reproj`
(`sd_mosaic_display.py:36-39`; grep for offset usage: zero hits). The only
offset call sites are `sd_mosaic.py:190` and `sd_mosaic_cloud_tasks.py:291`
(plus `backplanes.py:134`), matching Phase 2's file list -- section 1's
"(through the shared `cli/reproj` helpers) `sd_mosaic_display`" and the in-scope
"three mosaic call sites" should be corrected to two before someone hunts for
the third.

**8. The stated `apply_cmatrix_to_obs(obs, cmatrix, cmatrix_original)` signature
cannot run the midtime gate it is required to run.**
Section 3.4 puts the `|obs.midtime - times.midtime_et| <= 1e-6 s` gate "inside
`apply_cmatrix_to_obs`," but the section 3.1 signature has no
`times`/`midtime_et` parameter. One or the other must change; per the house
positional-grouping rule, `midtime_et` (or the times block) belongs in the
signature as part of the record being applied.

**9. The kernel-consumer agreement claim has a documented exception the plan
never mentions: BOTSIM losers (and every omitted result).**
Section 1's purpose -- products "agree with what a `furnsh` consumer computes"
-- is not true for a BOTSIM WAC: the kernel generator deliberately omits the
yielding camera's segment (`BOTSIM_YIELDING_CAMERA = 'WAC'`,
`src/spindoctor/cli/ck/images.py:49-60`; `botsim_losers` at `images.py:279`;
omission at `assignment.py:440-471`), so a `furnsh` consumer sees the
NAC-derived bus attitude for that WAC while the metadata reader applies the
WAC's own recorded `cmatrix`. Applying the WAC's own measurement is arguably the
better product, but the plan should state which is authoritative and scope the
agreement claim (the same divergence exists for any result omitted from kernels
for other reasons). Phase 4's docs are the natural place.

**10. The replacement frame has zero angular velocity where the original frame
had the spacecraft's.**
Probe on the real NAC frame: the original frame's transform at midtime carries
omega ~3.9e-5 rad/s; `Cmatrix`'s transform carries `Vector3(0,0,0)`
(`venv/.../oops/frame/cmatrix.py:48`, `Transform(cmatrix, Vector3.ZERO, ...)`).
No consumer in the switched paths reads it -- verified by the identity-swap
probe reproducing `Backplane` RA/dec to exactly 0.0, and event velocities pick
up path velocity, not frame omega, at the observer -- but any future
velocity-aware backplane (smear planes are #444/#455 territory) would silently
read zero. One sentence in section 3.2 turns a silent future wrong into a
documented boundary.

**11. The no-registration pin should cover both dicts, and the temporary-id
counter is the one piece of shared state the mechanism does touch.**
Executed probe: `Cmatrix(m)` with default `frame_id=None` leaves
`Frame.WAYFRAME_REGISTRY` and `Frame.FRAME_CACHE` both unchanged (`register`
returns early, `frame_.py:288-291`), `wayframe is self`, and `wrt(J2000)` works
unregistered (`frame_.py:446-453`) -- the plan's claims hold. Two refinements:
`test_frame_replacement_registers_nothing` pins only `FRAME_CACHE`; pin
`WAYFRAME_REGISTRY` too. And each construction increments the process-global
`Frame.TEMPORARY_FRAME_ID` counter (`frame_.py:407-416`; probe: delta 1 per
frame) -- harmless (the id is cosmetic when `wayframe is self`), but it is
technically new shared mutable state per image and deserves the one-line mention
next to the thread-safety claim.

---

## Right but fragile

- **The rationale "all Galileo today" on the `no_cmatrix_rotation_fitted` row is
  a data claim, not an invariant.** Manual-nav results flow through
  `orchestrator.with_pointing` (`nav_technique_manual.py:308`), so they normally
  carry a cmatrix -- good -- but any future rotation-fitting configuration on
  another instrument lands in that row and the parenthetical goes stale. Phrase
  the row by mechanism, not by mission.
- **`select_pointing` living in `cli/reproj/offsets.py` while serving
  `cli/backplanes`** is fine (the plan updates the CLAUDE.md note), but it makes
  `cli/reproj` the de-facto pointing-selection library for every downstream
  stage; if a third consumer appears, that is the moment it moves into the
  library package, and saying so now would prevent a shim later.

## Verified correct (spot-check record)

- **The inversion algebra.** Independent derivation: with an unchanged pool,
  `R_hat = C_oops . cmatrix_original^T = R` exactly (probe on the real WAC: max
  deviation from `diag(-1,-1,1)` was 2.2e-16), and `C_oops_corr = R_hat .
  cmatrix = M . C_oops` -- the observation's own attitude composed with the
  recorded correction, matching the writer's Step 3 (`cmatrix.py:344-375`,
  `dev_guide_ck_kernels.rst:204-218`) run backwards.
- **Gate completeness for the corruption classes.** Because `R_hat` mixes a
  read-time quantity with a recorded one, any read-vs-navigation pool difference
  above ~1e-9 rad, a transposed `cmatrix` or `cmatrix_original`, swapped fields,
  or a record pasted from another image (also caught by the midtime gate) all
  displace `R_hat` far beyond `_FLIP_TOL = 1e-9` (`cmatrix.py:112`). The one
  undetectable transpose case -- a symmetric (180-degree) rotation -- is
  self-inverse and therefore harmless.
- **`oops.frame.Cmatrix` mechanics.** Constructor signature and non-registration
  verified in source and by execution (finding 11); an unregistered frame
  composes through `wrt`, `Event` (`rings.py:920` pattern probed), and
  `Backplane` construction on a real observation; the identity swap reproduced
  RA/dec backplanes to exactly 0.0.
- **Midtime-only evaluation, the load-bearing simplification.** Holds. oops
  `Snapshot.uvt` returns `self._scalar_midtime` for every pixel
  (`venv/.../oops/observation/snapshot.py:103-128`), so every `Backplane` on a
  Snapshot is a midtime evaluation; `rings.py` passes `obs.midtime` at every
  geometry call (914, 920, 949, 1205-1311); `backplanes.py:111` hard-refuses
  non-Snapshot observations; `inventory` defaults to `tfrac=0.5`
  (snapshot.py:399). No consumer in the switched paths touches
  `obs.time[0]`/`[1]` or per-pixel times. (Section 3.2's citation "bodies.py
  passes `time=obs.midtime` throughout" is loose -- bodies.py relies on Snapshot
  semantics and stamps `obs.midtime` only into result metadata -- but the
  conclusion is right.)
- **The Voyager path.** The uniform formula handles it with no special case:
  `C_oops(mid)` read from the frozen frame, `R_expected = I`
  (`cmatrix.py:543-560, 624-629`), reproduced deterministically by the reader's
  own `from_file` under the fixed pool. No "4800 s cache" exists in the writer
  or the oops host; the frozen lookup's tolerance is `tol_ticks = 800 + texp/48`
  (`oops/hosts/voyager/iss.py:133`), a load-time snap, not a cache a reader
  could hit differently. (The corrected-kernel-furnished case is finding 1, not
  a Voyager special.)
- **Section 0/2 restatements.** The 1.5e-15 rad chain residual, 0.0029 px worst
  re-navigated axis, the 0.02/1e-12 pins, the three-process rationale, the flip
  table, the Galileo no-cmatrix behavior, and the metadata schema
  (`navigation_result.pointing.cmatrix`/`cmatrix_original` as nine row-major
  floats, `times.midtime_et`) all match `test_ck_round_trip.py:101-144,
  515-550`, `cmatrix.py:23-45, 101-108`, and `curator.py:237-283` exactly.
- **Fallback-row coverage of the known data classes.** Galileo (fitted rotation,
  `cmatrix` withheld -- `_build_pointing_solution`, `cmatrix.py:409-411`),
  simulated images (no `pointing`, `compute_pointing` returns None --
  `cmatrix.py:469-471`), pre-schema records, null-offset/non-success records:
  each maps to a specified row. Manual-nav results carry pointing
  (`nav_technique_manual.py:308`) and need no row of their own.
- **Concurrency.** No new registry or cache state (probed);
  `_reduced_oops_precision` and the per-thread-obs rule are untouched; the
  mechanism is consistent with CLAUDE.md's reproj thread-safety note, modulo the
  counter trivia in finding 11.
- **Process and scope.** One PR of four phases is coherent; nothing is a second
  PR in disguise; the `load_offset_if_any` -> `load_pointing_if_any` rename
  without a shim conforms to the no-shims rule; acceptance criteria are
  measurable (subject to finding 3's gap); Phase 4 covers user guide, dev guide,
  CLAUDE.md and plan reconciliation, with the metadata-format chapter explicitly
  deferred to #431 and listed as a follow-up; logging follows the
  both-logs/counted pattern and tests use `capsys`. `cmatrix.py` is 783 lines,
  so the addition plausibly stays under the 1000-line cap, and the plan
  pre-declares the package split if not.
- **Bound linearity direction.** The measured doubling ratios (2.03-2.13,
  `dev_guide_ck_kernels.rst:194-196`) mean the linear `B(frame)` over-covers
  every offset below the 50 px reference, which covers the whole library cohort;
  the plan's per-frame arithmetic (~6e-3 px NAC at 19 px, ~2e-2 px WAC at 5 px)
  checks out.
