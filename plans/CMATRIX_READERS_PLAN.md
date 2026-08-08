# SpinDoctor C-Matrix Readers Plan

*The design of record for the reading half of #50: the backplane and
reprojection stages consume the recorded C-matrix instead of the pixel
offset. The writing half -- recording `cmatrix` / `cmatrix_original` and
`times` in every navigated image's metadata, and generating corrected
C-kernels from them -- is delivered and documented in
`plans/CK_KERNEL_PLAN.md`; this plan changes only the consumers. One PR.*

---

## 0. Status

Design complete; not yet implemented. Adversarially reviewed 2026-08-07
(`critiques/CMATRIX_READERS_PLAN_CRITIQUE_2026-08-07.md`, verdict
CONDITIONAL with the mechanism verified correct by execution); every
finding is folded into the sections below, in particular the
`pool_already_corrected` ladder row, the identity short-circuit, the
consumer-level acceptance evidence, the in-metric bound, the closed
ladder gaps, and the `reset_all()` contract.

The gate the playbook set for this work is met: the round trip in
`tests/integration/test_ck_round_trip.py` shows, per instrument, that
the recorded `cmatrix` means what the offset means -- the pointing chain
reproduces the recorded attitude to at most 1.5e-15 rad, and a
re-navigation against the corrected kernel lands within 0.0029 px per
axis of zero across the cohort.

---

## 1. Purpose and scope

The metadata readers today rebuild the navigated pointing as a uniform
pixel shift, at exactly three call sites:

- `src/spindoctor/cli/backplanes/backplanes.py` (line ~134) reads the
  top-level `offset` and wraps `snapshot.fov` in `oops.fov.OffsetFOV`.
- `src/spindoctor/cli/reproj/offsets.py` does the same via
  `load_offset_if_any` / `apply_offset_to_obs` for `sd_mosaic` and
  `sd_mosaic_cloud_tasks`. (`sd_mosaic_display` displays saved
  reprojection products and loads no offsets; it imports only the
  argument helpers from `cli/reproj` and is untouched.)

The recorded `cmatrix` is the same measurement expressed as the attitude
the camera actually had, and it is the *senior* form: an exact rigid
rotation where the shift is its first-order approximation (the two
differ at second order in field angle, section 4), and the form a SPICE
consumer of the corrected kernels sees. Switching the readers makes
SpinDoctor's own products agree with what a `furnsh` consumer computes
for every image whose segment was written. The agreement claim is scoped
to those images: for a result the generator deliberately omitted from
the kernels -- a BOTSIM-yielding WAC, or any other omission -- the
reader still applies that image's *own* recorded measurement, which is
the better product for that image, while a kernel consumer sees the
bus attitude the winning segment implies. The metadata reader is
authoritative for SpinDoctor products; Phase 4 documents the divergence.

**In scope:** one library entry point that turns a recorded C-matrix
pair into corrected oops geometry; the metadata selection and fallback
ladder shared by both readers; the switch at the three call sites;
logging and run-level accounting; unit tests pinned by mutation; a
consumer-level integration comparison on real library frames;
documentation and plan reconciliation.

**Out of scope, deliberately (non-goals):**

- Reading pointing from the corrected kernel *files*, or registering
  them in the SPICE database oops selects from (#437). The kernels are
  not in that database, a per-image `furnsh` inside a batch process
  fights oops frame caching (the round trip needs three processes for
  exactly that reason), and Voyager cannot see a kernel furnished after
  `from_file` at all. The metadata is already in hand at both call
  sites. (The ladder is nonetheless built so that a pool which already
  answers the corrected attitude -- the state #437 creates -- is
  recognized and left alone rather than double-corrected; section 3.4.)
- Per-epoch attitude interpolation across the exposure, smear-aware
  consumers, and the interior-epoch error budget (#440, #444, #455).
- A corrected attitude for fitted-rotation results (#434). Until the
  pivot is recorded, those results carry no `cmatrix` and the offset
  path remains their documented behavior.
- Any mosaic-display change: `sd_mosaic_display` consumes saved
  products, not observations.
- The metadata schema, the navigator, the manual-nav driver, and the
  kernel writer: unchanged. The `offset` field stays recorded and
  reported exactly as it is; only which field the readers *apply*
  changes.
- The results-index ingest (#430) and the metadata format chapter
  (#431).

---

## 2. What the writer half already guarantees

Everything the readers rely on is established and tested on the writing
side; it is restated here so this plan stands alone.

- `navigation_result.pointing` carries `cmatrix_original` (always, when
  a `NavResult` exists) and `cmatrix` (only when the navigation produced
  an offset and fitted no camera rotation), both as nine row-major
  floats: J2000-to-camera rotations in the **SPICE camera frame
  convention** at the exposure **midtime**. `navigation_result.times`
  carries `start_et`, `stop_et`, `midtime_et`, `exposure_s` and the
  clock strings.
- Both matrices were validated at write time: proper orthonormal
  rotations to 1e-9, every element finite, serialized unrounded.
- The oops observation frame relates to the SPICE camera frame by a
  constant per-instrument flip `R`, `C_oops = R . C_spice`: Cassini ISS
  `diag(-1, -1, +1)`, LORRI `diag(+1, -1, -1)`, Galileo and Voyager the
  identity. The writer measures `R` at runtime and refuses a mismatch.
- Voyager's observation frame is frozen from a tolerance-snapped `ckgp`
  lookup; its `cmatrix_original` *is* that frozen attitude and `R = I`
  by construction, so no reader special case is needed.
- A fitted-rotation result (all Galileo today) records
  `cmatrix_original` only. A simulated image records no `pointing`
  block at all.
- The kernel generator omits some recorded measurements from the
  written kernels -- the BOTSIM-yielding WAC, and every image with an
  omission reason -- so a kernel consumer does not see those images'
  own corrections even though the metadata carries them.
- The round trip proved on real frames of all four kernel-eligible
  instruments that the recorded `cmatrix`, written into a kernel and
  read back, moves the pointing by exactly the navigated offset (chain
  residual at most 1.5e-15 rad; re-navigated offset at most 0.0029 px
  per axis, a property of technique re-measurement, not of the chain).

---

## 3. Design decisions

### 3.1 Mechanism: replace the observation frame, behind one entry point

The corrected frame is, by the writing half's own derivation, *the
frame in which the unmodified FOV holds*. So the reader inverts Step 3
of that derivation rather than re-deriving anything:

```text
if cmatrix == cmatrix_original (np.array_equal):
    C_oops_corr = C_oops(mid)                       # short-circuit
else:
    R_hat       = C_oops(mid) . cmatrix_original^T  # measured, gated
    C_oops_corr = R_hat . cmatrix
obs.frame = oops.frame.Cmatrix(C_oops_corr)         # frame_id=None
```

with `C_oops(mid)` read from `obs.frame.wrt(J2000)` at `obs.midtime`,
exactly as `_observation_attitude` already does. The FOV is left alone.
Algebraically `C_oops_corr = C_oops . (cmatrix_original^T . cmatrix)`:
the observation's own attitude composed with the recorded correction.
The `array_equal` short-circuit mirrors `_spice_cmatrix`'s on the
writing side and is what makes an identity correction reproduce the
observation's own midtime attitude exactly -- two float64 matrix
products do not cancel to bit precision, so without it "no correction
means no change" would be false at the 1e-16 level. `R_hat` is measured
for one reason only: the gate (section 3.4), which in a single
inequality catches a drifted pool, a transposed or corrupted record,
and a changed host convention. When the gate passes, `R_hat` and the
table constant agree to 1e-9 and either serves the composition.

**Where it lives.** One public function beside `compute_pointing` in
`spindoctor/support/cmatrix.py` -- the module whose docstring already
declares itself the one place the conventions live:

```python
apply_cmatrix_to_obs(obs, cmatrix, cmatrix_original, midtime_et)
```

The four positionals are one logical group -- the observation and the
record being applied to it; `midtime_et` is in the signature because
the midtime gate runs inside this function. It mutates `obs.frame`, is
added to `__all__`, and raises `NavPointingError` for every expected
failure (malformed record, unknown host, gate violation) plus a
distinguished outcome for a pool that already answers the corrected
attitude (section 3.4), so callers absorb exactly one exception,
mirroring `compute_pointing`'s contract. Nothing in `spindoctor.cli`
computes the conjugation itself. If the addition pushes `cmatrix.py`
past the 1000-line cap, the module becomes a package with its interface
unchanged.

**Cache hygiene.** "Apply before any geometry is computed" is not an
invariant `from_file` leaves available: `ObsSnapshot.__init__` runs the
closest-planet scan, which builds and caches a `Backplane`
(`_center_bp`) against the *uncorrected* frame before any caller can
apply anything. That is benign today only because the consumers of that
particular cache are rotation-invariant, and nothing pins the accident.
So `apply_pointing_to_obs` (section 3.3) calls `obs.reset_all()` after
applying **either** mechanism -- the offset path has the identical
latent hazard -- clearing every cached `Backplane` and `Meshgrid` so
all downstream geometry, including `obs.ext_bp` consumed by
`RingMosaic.reproject`, is built on the corrected observation.

**Mechanisms weighed and rejected:**

- *Derive `(dv, du)` back from the cmatrix and keep `OffsetFOV`.* This
  consumes the field in name only: it re-applies the shift
  approximation, discards the exact rotation, and adds an inversion the
  writing half never needed. Rejected.
- *Furnish the corrected kernel.* See non-goals; the per-image
  metadata-driven mechanism is the natural fit until #437 lands.
- *Replace the FOV with a rotated one.* There is no oops FOV that
  expresses a rigid rotation; the frame is where a rotation belongs,
  and `oops.frame.Cmatrix` exists for exactly this ("rotates J2000
  coordinates into the frame of a camera").

**oops mechanics, verified against the installed oops:** `Snapshot`
stores `self.frame` as a plain attribute and (post `reset_all()`)
builds its events from it lazily, the same contract
`apply_offset_to_obs` relies on for `obs.fov`. `Cmatrix(...,
frame_id=None)` takes a temporary id and is **not registered**
(`Frame.register` returns before touching either registry for a None
id; an unregistered frame composes through `wrt`, `Event` and
`Backplane`), so a 50,000-image mosaic loop pollutes no global frame
state and the reproj thread-safety rule (each thread its own `obs`) is
undisturbed. The one piece of shared state the mechanism does touch is
the process-global `Frame.TEMPORARY_FRAME_ID` counter, incremented once
per construction -- benign, since the id is cosmetic when the wayframe
is the frame itself. A unit test pins that both `Frame.FRAME_CACHE` and
`Frame.WAYFRAME_REGISTRY` are unchanged by an application.

### 3.2 The exposure-time question

The recorded `cmatrix` is a midtime attitude, so the replacement frame
is constant across the exposure, where the offset path kept the
time-varying uncorrected frame under a constant shift. Nothing changes
for these consumers, because every one of them is a midtime
evaluation: oops `Snapshot.uvt` returns the scalar midtime for every
pixel, so every `Backplane` built on a Snapshot -- which the backplane
stage hard-requires -- is evaluated there, and `RingMosaic.reproject`
constructs its `Event`s at `obs.midtime` explicitly. At the midtime the
constant frame equals the corrected time-varying chain exactly. Off the
midtime the two mechanisms differ by the attitude rate times the offset
from midtime -- a quantity only a smear-integrating consumer would see,
which is #444 / #455 territory and a non-goal here. The consumer-level
comparisons of Phase 3 (ring and body reprojection and backplanes run
through both paths) are what would expose a consumer that quietly
sampled the frame off-midtime, since the offset path retains the
time-varying frame; the LOS-grid comparison alone could not, as it
never runs consumer code.

**A documented boundary:** the replacement frame's transform carries
zero angular velocity where the original frame carried the
spacecraft's (~4e-5 rad/s on a Cassini frame). No consumer in the
switched paths reads frame omega -- verified by an identity-swap probe
reproducing `Backplane` RA/dec to exactly 0.0 -- but a future
velocity-aware backplane (smear planes, #444/#455) must not consume the
replaced frame's omega. The dev guide states this beside the mechanism.

### 3.3 Fallback policy, exactly

The offset path is not removed. It is the documented behavior for every
record that carries no usable `cmatrix` -- a statement about current
data classes, not a compatibility shim. The ladder, applied per image
(rows keyed by mechanism, not by mission):

| Record class | Applied | Logged |
|---|---|---|
| `pointing.cmatrix` present and valid, gates pass | C-matrix (frame replacement) | image log: debug/info one-liner naming the source |
| gates find the pool already answering the corrected attitude (section 3.4) | **nothing** -- the observation is already correct | image log one-liner; counted at run level, reason `pool_already_corrected` |
| `cmatrix` absent, offset present (a fitted-rotation result -- the mechanism; Galileo is the only instance today) | offset via `OffsetFOV`, as today | image log: reason `no_cmatrix_rotation_fitted`; counted per reason at run level |
| `pointing` block absent, offset present (simulated images; any pre-`pointing`-schema record) | offset via `OffsetFOV` | image log: reason `no_pointing_block`; counted |
| `pointing` block unusable: `cmatrix` malformed (wrong shape, non-finite, bool/str elements, not a proper rotation), or `cmatrix_original` absent or malformed, or `times` / `midtime_et` absent or non-finite (the gates cannot run) | offset via `OffsetFOV` | **warning to image log and one line to the run log**, reason `malformed_pointing` |
| midtime gate fails (record belongs to another observation) | offset via `OffsetFOV` | **warning to both logs**, reason `cmatrix_foreign_midtime` |
| `R_hat` gate fails and the pool is not the corrected one | offset via `OffsetFOV` | **warning to both logs**, reason `cmatrix_baseline_mismatch` |
| no offset either (`null_offset`, non-success status, missing/unreadable metadata) | uncorrected pointing / skip, exactly as today | unchanged: the existing both-logs warning and `uncorrected_pointing` / skip accounting |

The malformed rows are probed as their own test domain (NaN defeats
every comparison -- including a NaN `midtime_et` against the midtime
gate -- `reshape` accepts wrong ranks, bools convert to floats);
parsing refuses rather than coerces, in the style of
`spindoctor.cli.ck.pointing`.

Selection is shared, not duplicated: `cli/reproj/offsets.py` grows
`select_pointing(nav_metadata) -> PointingSelection` (a frozen
dataclass: which mechanism, the values including `midtime_et`, the
reason when degraded) operating on an already-parsed metadata dict,
plus `apply_pointing_to_obs(obs, selection)` dispatching to
`apply_cmatrix_to_obs` or the existing `OffsetFOV` application, calling
`obs.reset_all()` after either (section 3.1), and returning what it
applied. `load_offset_if_any` becomes `load_pointing_if_any` (renamed
outright -- no shim), still owning the hardened path resolution and
file/JSON error ladder it has today. `backplanes.py` calls
`select_pointing` / `apply_pointing_to_obs` on the dict it already
reads (its skip-on-status logic is untouched), and its result dict
gains `pointing_source: 'cmatrix' | 'pool' | 'offset' | 'none'` beside
the existing `uncorrected_pointing`, so a cloud task reports which
mechanism a product got; the mosaic cloud-task tally gains the new
reasons.

**One deliberate per-caller divergence, stated rather than papered
over.** Today `backplanes.py` *raises* on a success-status record with
no `offset` key at all (a defect-shaped record fails the single-image
task) while degrading on a null offset, and `offsets.py`
warns-and-continues on everything (a batch pass must survive one bad
record). Unification keeps that split: `select_pointing` classifies the
record; the backplane caller continues to raise on the
missing-key-with-success-status class, the mosaic callers continue to
warn and count. The severity policy belongs to the caller; the
classification is shared.

The module docstrings and CLAUDE.md's `cli/reproj` note record that
`offsets.py` now serves the backplane stage too. If a third consumer of
`select_pointing` ever appears, that is the moment the selection moves
into the library package proper -- said now so it is a planned move,
not a shim.

### 3.4 Precedence and honesty

**When both `cmatrix` and offset exist, the cmatrix wins.** That is the
point of the field: it is the exact form, and it is what a kernel
consumer sees for every image whose segment was written.

**Disagreement is detected, classified, and never blindly applied.**
Inside `apply_cmatrix_to_obs`, in order:

1. Record validation: both matrices proper rotations, `midtime_et`
   finite (else `malformed_pointing`, resolved by the caller before
   this function is reached via `select_pointing`).
2. Midtime gate: `|obs.midtime - midtime_et| <= 1e-6 s` -- the
   attitude belongs to that epoch; a mismatch means the record is not
   this observation's. Failure: `cmatrix_foreign_midtime`, offset
   fallback with both-logs warning.
3. Flip gate: `max|R_hat - R_expected| <= 1e-9` (the writer's own
   `_FLIP_TOL`), with `R_expected` from the instrument's
   `_FrameIdentity`. Since `R_hat` mixes the observation's *current*
   attitude with the *recorded* baseline, this one inequality fails on
   a changed kernel pool, a transposed or swapped matrix (a transposed
   rotation is still a proper rotation, so validation alone cannot
   catch it), or a changed host convention.
4. On flip-gate failure, one more cheap probe before concluding
   corruption: `max|C_oops(mid) - R_expected . cmatrix| <= 1e-9`. If
   it holds, the furnished pool **already answers the corrected
   attitude** -- corrected kernels furnished at load time, which is
   exactly the state #437 exists to create, and which Voyager's frozen
   `from_file` snap would bake in even earlier. The correct action is
   to apply **nothing**: the observation is already right, and either
   fallback would corrupt it -- the offset path would double-correct
   by ~2x the offset. Outcome `pool_already_corrected`, counted under
   its own reason; without this row, landing #437 later would convert
   the gate into a double-correction engine on every kernel-covered
   image.
5. Only when neither explanation fits is it `cmatrix_baseline_mismatch`:
   warn to both logs, apply the offset path.

Rationale for the mismatch fallback: an *unexplained* gate failure's
overwhelmingly likely cause is a corrupted record, a reader defect, or
an environment mismatch, and the one lesson of the writing half is that
a conjugation error survives every hermetic check while producing a
proper rotation pointing the wrong way -- applying the cmatrix anyway
would be precisely the silently wrong science product this plan exists
to prevent. The offset path under those conditions reproduces today's
product exactly, so the degradation is to current behavior, visibly
counted, never to something new. The known non-defect state in which a
gate fires -- the already-corrected pool -- is not sent there, because
there the offset path is the compromised one and doing nothing is
exact; that is what step 4 distinguishes.

There is no per-image runtime re-derivation of the offset from the
cmatrix: the two were computed from one measurement in one process by
`compute_pointing`, their agreement is pinned by the unit equivalence
test and measured per instrument by the Phase 3 comparison, and a
runtime recomputation through the same code could only agree by
construction.

---

## 4. The agreement bound, derived

The acceptance question is: per pixel, how far may cmatrix-path
geometry sit from offset-path geometry on a real frame before it is a
defect?

- **The metric is pixel space**, measured by inverting each path's
  J2000 line of sight back through the *other* path's mapping via
  `uv_from_los`-style inversion -- not a constant-scale angular
  conversion. The two metrics differ by 26% on the distorted WAC
  (writer's table: 9.89e-2 px tangent-plane vs 7.86e-2 px pixel space
  at the same 50 px), so a bound pinned in one and measured in the
  other spends its margin on the metric mismatch instead of on real
  variation.
- **At the boresight: zero, by construction.** `compute_pointing` built
  the correction as the minimal rotation carrying the `OffsetFOV`
  boresight line of sight onto the unmodified one, so the two paths
  agree there to floating point. Pinned at 1e-3 px of slack.
- **Away from the boresight: second order in field angle, linear in
  the offset** (measured doubling ratios 2.03-2.13, so the linear form
  over-covers every offset below the reference -- the whole library
  cohort). The writer's tabled constants are worst-case over eight
  offset directions on a 17x17 grid at 50 px total displacement, but a
  review probe on the real `W1580760393_1_CALIB` at a single (30, 40)
  px offset measured **0.0908 px in pixel space** (0.0978 px at
  constant scale) -- above the tabled 7.86e-2 -- so direction and grid
  placement can exceed the table and the constants must be
  **re-measured in the test's own metric**. Phase 3 therefore first
  measures `K_inst` per cohort frame: worst pixel-space residual over
  the 17x17 grid, swept over eight offset directions at 50 px total
  displacement, on that frame's own FOV. The per-frame expected bound
  is then

  ```text
  B(frame) = K_inst * (|offset|_total / 50 px)
  ```

- **Test bound:** worst grid pixel residual `<= 2 * B(frame) + 0.005
  px`. The factor of two covers what genuinely remains after the
  in-metric, direction-swept `K`: the navigated offset's direction
  falling between swept directions and grid placement -- not a metric
  mismatch, which the re-measurement removes. The floor covers
  numerical noise and the distorted-FOV zero-offset structure
  (measured at 9.1e-4 px near center on the WAC). On the library
  cohort the bound stays small -- a ~5 px WAC frame allows ~2e-2 px
  against ~1e-2 px measured -- while every directional error this plan
  guards against (sign, transpose, skipped conjugation) displaces the
  geometry by roughly *twice the offset*, several pixels on every
  frame.

House decision rule, unchanged from the CK plan: when the measured
residual is at or below the derived bound, pin the test at
measured-plus-margin; when it is above, **stop and diagnose** -- a
larger residual is a defect, never a tolerance to raise.

Note the asymmetry the bound describes: away from the boresight the two
paths differ because the *offset* path is the approximation. The bound
is an agreement statement between old and new products, not an error
budget of the new path.

---

## 5. Execution

Four phases, one PR. Every phase lands with its tests; TDD throughout.

### Phase 1 -- the mechanism

`apply_cmatrix_to_obs(obs, cmatrix, cmatrix_original, midtime_et)` in
`spindoctor/support/cmatrix.py`: the identity short-circuit, the gate
sequence of section 3.4 including the `pool_already_corrected` probe,
input validation reusing `_as_readonly_3x3` / `_validate_rotation`, and
the temporary-frame replacement.

Hermetic unit tests (`tests/spindoctor/support/`), each named for the
mutation it pins -- a change that flips a sign, transposes a matrix, or
skips the conjugation must fail the named test:

- `test_the_reader_reproduces_the_offset_boresight`: plant an offset on
  a synthetic FOV/frame pair, run `compute_pointing`, apply the
  recorded pair through the reader, and assert the corrected frame's
  boresight line of sight equals the `OffsetFOV` line of sight to
  floating point. Fails under any sign flip or reversed composition.
- `test_skipping_the_conjugation_points_the_wrong_way`: a synthetic
  non-involutory flip (quarter turn about Z, as Phase A of the CK plan
  used) between the "oops" and "SPICE" frames; asserts the applied
  correction moves the boresight the recorded way, which composing
  `cmatrix` without `R_hat` (or with its transpose) does not. Every
  real `R` is diagonal and self-inverse, so only a synthetic frame can
  pin the direction.
- `test_a_zero_correction_reproduces_the_observation_frame`: `cmatrix`
  and `cmatrix_original` equal as arrays yield a frame whose midtime
  attitude equals the observation's **exactly, via the short-circuit**;
  the test fails if the short-circuit is removed, because the composed
  products differ at the 1e-16 level.
- `test_a_transposed_record_fails_the_flip_gate`: transposing either
  recorded matrix (still a proper rotation) trips the `R_hat` gate,
  raising `NavPointingError` with the mismatch in the message.
- `test_a_drifted_baseline_fails_the_gate`: perturb the observation
  frame by 1e-5 rad; the gate refuses with the mismatch reason.
- `test_an_already_corrected_pool_is_left_alone`: build the observation
  frame as `R . cmatrix` (the pool already answering the corrected
  attitude); the outcome is the distinguished
  `pool_already_corrected`, the frame is not replaced, and no offset
  is applied.
- `test_a_foreign_midtime_is_refused`: `midtime_et` off by 1 s.
- Malformed-domain probes: NaN element, wrong shape/rank, non-rotation,
  bool elements, absent `cmatrix_original`, absent or NaN `midtime_et`
  -- each refused, message asserted; the NaN `midtime_et` case pins
  that the midtime gate cannot be silently defeated by a comparison
  that is false both ways.
- `test_frame_replacement_registers_nothing`: both
  `Frame.FRAME_CACHE` and `Frame.WAYFRAME_REGISTRY` unchanged across
  two applications (the `TEMPORARY_FRAME_ID` counter increment is the
  documented exception), so batch loops cannot leak.

### Phase 2 -- selection and the reader switch

`PointingSelection` / `select_pointing` / `apply_pointing_to_obs` (with
its `reset_all()` call) in `cli/reproj/offsets.py`; `load_offset_if_any`
renamed to `load_pointing_if_any`; the switch in `backplanes.py`,
`sd_mosaic.py` and `sd_mosaic_cloud_tasks.py`; the logging ladder and
counters of section 3.3, including `pointing_source` in the backplane
result, the extended cloud-task tally, and the per-caller severity
split for the missing-offset-key class.

Unit tests: one per ladder row (selection outcome, applied mechanism,
reason string, and -- via `capsys` -- the warning text and which log
got it), including `pool_already_corrected` applying nothing;
precedence (both fields present, the frame is replaced and the FOV
untouched); the mismatch path ends with `OffsetFOV` applied and
both-logs warnings; a cached-`Backplane` test pinning that geometry
computed after `apply_pointing_to_obs` is built on the corrected frame
(fails if `reset_all()` is dropped, using a synthetic frame whose swap
is *not* rotation-invariant for the probed quantity); the backplane
raise-vs-warn divergence, one test per caller behavior; existing
offsets tests updated for the rename.

### Phase 3 -- integration comparison (the acceptance evidence)

`tests/integration/test_cmatrix_readers.py`, marked `integration`, on
the round-trip library cohort (Cassini NAC `N1461997416_1_CALIB`, WAC
`W1580760393_1_CALIB`, Voyager `C1205021_GEOMED`, LORRI
`lor_0030713591_0x633_sci`, plus the body-navigated WAC
`W1637520502_1_CALIB`) and the Galileo frame `C0059894800R`:

- **Measure `K_inst` in-metric first** (section 4): per cohort frame,
  worst pixel-space residual, 17x17 grid, eight directions, 50 px
  total displacement; record the values in the PR beside the writer's
  tabled ones.
- **LOS grid:** navigate the frame, load the observation twice, apply
  the metadata once through each path, compare per-pixel geometry in
  pixel space over a 17x17 grid. Assert the worst pixel within the
  section 4 bound and the boresight within 1e-3 px.
- **The consumers the PR actually switches**, compared across the two
  paths -- this, not the LOS grid, is what exercises `obs.ext_bp`, the
  `Event`-from-replaced-frame construction inside
  `_reduced_oops_precision`, and `uv_from_coords`:
  - one `RingMosaic.reproject` on a library frame with resolved ring
    content: matching populated cells agree within the section 4 bound
    converted through the projection's local scale;
  - one `BodyMosaic.reproject` on `W1637520502_1_CALIB`: same
    assertion on the body grid;
  - one end-to-end `generate_backplanes_image_files` run on the NAC
    frame: RA/dec planes agree within the bound (RA scaled by cos
    dec).
- The Galileo frame selects the offset path with reason
  `no_cmatrix_rotation_fitted` and produces a product identical to
  today's.

Measured residuals are recorded in the PR and the pins set at
measured-plus-margin per the section 4 decision rule.

### Phase 4 -- documentation and reconciliation

- User guide: the backplane and mosaic chapters state which pointing a
  product is built on, the full fallback ladder including
  `pool_already_corrected`, and how the run reports it; the scoped
  kernel-agreement statement (BOTSIM losers and other omitted results
  keep their own recorded measurement in SpinDoctor products, which is
  authoritative for them, while kernel consumers see the winning
  segment's attitude).
- Dev guide: `dev_guide_ck_kernels.rst` gains a short "The readers"
  section -- the inversion formula with its short-circuit, the gate
  sequence and the already-corrected-pool row, the fallback ladder,
  the zero-omega boundary of the replacement frame, and the agreement
  bound with its in-metric derivation.
- CLAUDE.md: the `cli/reproj` note reflects `offsets.py` serving the
  backplane stage.
- Plans: this plan's status section updated to as-built; the
  `OPERATOR_PLAYBOOK.md` #50 dispatch entry and `CK_KERNEL_PLAN.md`
  section 7's first follow-up reconciled as delivered; the critique's
  findings dispositioned on #50. #50 closes with the PR.

---

## 6. Acceptance criteria

1. Both readers apply the recorded `cmatrix` whenever it is present and
   passes the gates; the FOV is never wrapped in `OffsetFOV` on that
   path; the conjugation exists in exactly one library function; and
   `apply_pointing_to_obs` resets the observation's cached geometry so
   no product is built on a pre-application `Backplane`.
2. Every fallback row of section 3.3 behaves and logs as specified --
   including `pool_already_corrected` applying nothing -- with the
   offset path producing byte-identical geometry to today's and the
   backplane/mosaic severity split preserved.
3. On real library frames of all four kernel-eligible instruments plus
   the body-navigated WAC frame, the two paths agree within the
   section 4 in-metric bound at the LOS grid **and** through the
   switched consumers themselves (one ring reprojection, one body
   reprojection, one backplane run), boresight within 1e-3 px, pinned
   at measured-plus-margin.
4. Each directional mutation -- sign flip, transposed record, skipped
   or reversed conjugation, drifted baseline, removed identity
   short-circuit, removed `reset_all()` -- fails its named unit test.
5. A gate violation degrades to the offset path with warnings in both
   logs and a per-reason count in the run summary / task result, with
   `cmatrix_foreign_midtime` and `cmatrix_baseline_mismatch` reported
   as distinct reasons; a pool that already answers the corrected
   attitude is left alone and counted; no product is built on a
   cmatrix that failed a gate.
6. `ruff check`, `ruff format --check`, `mypy --strict`,
   `sphinx-build -W` and `pymarkdown scan` pass; suite coverage does
   not drop; the suite passes at `-n 4`.

---

## 7. Follow-ups

- **Fitted-rotation results join the cmatrix path** when #434 records
  the pivot and lifts the `rotation_unsupported` omission; the fallback
  row for `no_cmatrix_rotation_fitted` then empties on its own.
- **Kernel-file-driven reading** becomes possible if #437 registers the
  corrected kernels for oops selection. The `pool_already_corrected`
  row is the readers' forward compatibility with that state: on a
  kernel-covered image the reader detects the corrected pool and
  applies nothing, so #437 can land without touching the readers; the
  metadata path stays as the mechanism for results not written into
  kernels.
- **Smear-aware consumers** needing attitude (or frame angular
  velocity, which the replacement frame carries as zero) across the
  exposure depend on the interior-epoch work (#440, #444, #455) and
  would consume the kernel, not the midtime matrix.
- **The metadata format chapter** (#431) should document the readers'
  precedence and fallback ladder when it is written.
