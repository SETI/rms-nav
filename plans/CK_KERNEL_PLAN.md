# SpinDoctor Corrected-Pointing C-Kernel Plan

*Implementation plan for recording each navigated image's corrected camera
attitude as a C-matrix in its metadata, and for generating SPICE C-kernels
from those matrices so that any SPICE-based tool can load navigated pointing
with a `furnsh`. Written to be executed by an implementing model with no
briefing beyond `/seti/newnav/CLAUDE.md` and the repository itself.
Conventions from `CLAUDE.md` and `.cursor/rules/` apply throughout: line
length 100, mypy strict, pdslogger-only logging, Google-style docstrings
with `Parameters:`, Conventional Commits, one logical change per commit,
modules under 1000 lines, no issue numbers in docstrings or `.rst` files.*

Integration branch: `rf_ck_kernels`, cut from `main`. Each phase below lands
as its own pull request targeting that branch.

---

## 1. Purpose and scope

Navigation measures where a camera was actually pointing. Today that
measurement leaves the pipeline as a pixel offset in a JSON file, which only
SpinDoctor knows how to apply. Expressed instead as a C-matrix and written
into a C-kernel, the same measurement loads into oops, ISIS, or a plain
`spiceypy` script with one `furnsh`, and every geometry computation those
tools perform is then correct without anyone touching an offset.

This plan does two things:

1. **Records the corrected attitude in the metadata**, as a C-matrix
   computed at navigation time. The pixel offset stays exactly as it is; the
   C-matrix is recorded beside it.
2. **Generates C-kernels from those C-matrices**, as an overlay: one type-3
   segment per navigated image, covering only that image's exposure, in
   files that mirror the coverage of the original kernels they correct.

The kernel writer consumes the recorded C-matrix. It performs no
offset-to-rotation conversion of its own, and it does not import oops.

**In scope:** the C-matrix computation and its metadata fields; the exposure
and clock fields the writer needs; the `sd_create_ck` program and its writer
core; file mirroring and naming; the per-mission frame handling; the BOTSIM
and Voyager rules; the comment area, meta-kernel, and CSV report; and the
round-trip validation that closes the loop from navigation through kernel
and back.

**Out of scope, deliberately:**

- Replacing the pixel offset. Every existing consumer of `offset` continues
  to work unchanged.
- Correcting images that did not navigate. They get no segment and fall
  through to the original kernels.
- New camera frames. The correction is written at the bus or platform level
  the existing kernels already describe.
- **Fitted twist.** The per-image `rotation_deg` a technique fits is a
  rotation about a per-technique pivot (a feature centroid), and that pivot
  is not recorded anywhere in the result or the metadata; a twist about the
  boresight and a twist about an off-center pivot differ by a translation
  of order several pixels. The correction is therefore not expressible from
  the recorded data. When a result carries a fitted rotation, no `cmatrix`
  is recorded and the generator omits the image with a reason (sections
  2.2, 3.2). Only Galileo SSI fits rotation today
  (`fit_camera_rotation: true` appears only in `config_410_inst_gossi.yaml`;
  pin behavior to the config key, not to prose). Twist support is a
  follow-up (section 7).
- Registration of the corrected kernels in the SPICE database oops selects
  from. That database is scheduled for replacement; a plain `furnsh` of the
  meta-kernel is the supported loading path, and registration is a
  follow-up (section 7).
- Ingesting metadata written by an older schema. The generator reads the
  current schema only; results are regenerated before kernel production.

---

## 2. The C-matrix in the metadata

### 2.1 Definition, and the frame trap this section exists to avoid

A C-matrix is the rotation taking a vector expressed in J2000 to the same
vector expressed in a frame:

```text
v_frame = C . v_J2000
```

which is what `cspyce.pxform('J2000', frame_name, et)` returns.

Two are recorded per navigated image, both in the **SPICE camera frame
convention** and both at the exposure **midtime**:

- `cmatrix_original` -- the attitude the furnished kernels gave at
  navigation time, before any correction.
- `cmatrix` -- the corrected attitude, the one a kernel should carry.

Recording both makes the correction self-contained: their difference is the
correction, and `cmatrix_original` is what lets the writer verify the
baseline kernels have not changed since navigation ran.

**The trap: the oops observation frame is not the SPICE camera frame.** The
navigated offset lives in the oops observation frame, and for three of the
four instruments oops builds that frame with a constant flip on top of the
SPICE frame:

| Instrument | oops frame vs SPICE frame | Where |
|---|---|---|
| Cassini ISS (NAC, WAC) | `R = diag(-1, -1, +1)` (a 180 degree flip) | `oops/hosts/cassini/iss.py` (`rot180`) |
| New Horizons LORRI | `R = diag(+1, -1, -1)` (SPICE boresight is -Z) | `oops/hosts/newhorizons/lorri.py` |
| Galileo SSI | `R = I` (oops uses `GLL_SCAN_PLATFORM` directly) | `oops/hosts/galileo/ssi.py` |
| Voyager ISS | no SPICE-frame relation at runtime: oops freezes a `Cmatrix` built from a tolerance-snapped `ckgp` (see 2.2) | `oops/hosts/voyager/iss.py` |

where `R` relates the two J2000-referenced attitudes as
`C_oops = R . C_spice`. A correction rotation built in the oops frame and
composed onto a `pxform`-derived matrix without conjugating through `R` is a
proper rotation of the right magnitude pointing the **wrong way** (for
Cassini, both tangent-plane components negated) -- and every hermetic test
that never touches SPICE passes anyway. `R` is therefore computed at
runtime, never assumed, and asserted against this table (section 2.2).

### 2.2 How the corrected matrix is derived

This is the part a future oops call will replace, so it lives in one module
behind one function and nothing else computes it.

**Step 1 -- the corrected boresight in the oops frame.** The navigated
offset is applied downstream as `oops.fov.OffsetFOV(fov, uv_offset=(du,
dv))` (note the order: the metadata `offset` is `[dv, du]`, and both
existing consumers construct the FOV with `(du, dv)`). That FOV maps pixels
to camera tangent-plane coordinates as

```text
xy_from_uv(uv) = fov.xy_from_uv(uv) - xy_offset
xy_offset      = fov.xy_from_uv(fov.uv_los + (du, dv))
```

Under the corrected pointing, the true direction seen by pixel `uv` in the
original oops frame is `fov.los_from_xy(fov.xy_from_uv(uv) - xy_offset)`.
The corrected frame is the one in which the *unmodified* FOV holds, so the
rotation `M` (oops-frame coordinates, original to corrected) satisfies

```text
M . los_from_xy(xy - xy_offset) = los_from_xy(xy)     for all xy
```

Evaluating at the boresight (`xy = xy_from_uv(uv_los)`, which need not be
zero for a subarray or distorted FOV) gives the constraint

```text
d = los_from_xy(xy_from_uv(uv_los) - xy_offset)
M . d = los_from_xy(xy_from_uv(uv_los))
```

**Step 2 -- construct `M`.** `M` is the minimal rotation taking `d` to the
boresight direction `b = los_from_xy(xy_from_uv(uv_los))`: axis
`d x b` normalized, angle `arccos(d . b)`, realized with
`cspyce.axisar(axis, angle)` -- which is the **active** vector rotation
(`M . d = b`); `cspyce.rotate` is the frame rotation and gives the
transpose. Guard: when `|d x b| < 1e-12` (a zero or sub-nanoradian offset),
`M = I`. `M` is exact by construction; nothing is orthonormalized. A fitted
`rotation_deg` on the result means no `cmatrix` is computed at all (section
1); there is no twist term in `M`.

An exact rigid rotation is not exactly a uniform tangent-plane shift; the
difference is second order in field angle -- about 1e-9 radians on a
Cassini NAC and at most ~0.04 px at the corner of a WAC for a 50 px offset,
measured. This bounds what the round trip in Phase D can be expected to
recover and is part of its error budget.

**Step 3 -- express both attitudes in the SPICE convention.** With
`C_oops` the observation frame's J2000-to-camera matrix at midtime
(evaluated from the observation's own frame object) and, for Cassini,
Galileo and LORRI,

```text
cmatrix_original = pxform('J2000', camera_frame, et_mid)
R                = C_oops . cmatrix_original^T
cmatrix          = (R^T . M . R) . cmatrix_original
```

`R` is asserted equal (to 1e-9) to the section 2.1 constant for the
instrument, and asserted epoch-independent by recomputing at `start_et` and
`stop_et`; a violation raises rather than being absorbed. For **Voyager**,
oops builds the observation frame as
`P . ckgp(ck_id, sce2c(scid, et_mid), 800 + texp/48, 'J2000')` with
`P = pxform('VGn_SCAN_PLATFORM', camera_frame, 0)` -- frozen,
time-independent, and tolerance-snapped, so `pxform` at midtime does not
reproduce it. For Voyager, `cmatrix_original = C_oops` itself (already the
SPICE convention built on the snapped platform attitude), `R = I` by
construction, and the writer must reproduce the baseline with the same
snapped `ckgp` call (section 3.3) -- which is why `exposure_s` is a
recorded field.

Sanity check on the result: `|det(cmatrix) - 1| < 1e-9` and
`max|cmatrix . cmatrix^T - I| < 1e-9`, both raising on violation -- a
non-proper rotation here is a defect, not something to repair.

### 2.3 Metadata fields

Added under `navigation_result` (so they exist exactly when a `NavResult`
exists; a load-error document has no `navigation_result` and none of
these):

```text
pointing:
  cmatrix:            [9 floats, row-major]   # corrected, J2000 -> camera, SPICE convention;
                                              # absent when rotation_deg was fitted
  cmatrix_original:   [9 floats, row-major]   # uncorrected, same convention
  camera_frame:       "CASSINI_ISS_NAC"       # SPICE frame name
  camera_frame_id:    -82360
  ck_frame_id:        -82000                  # the object a CK targets
times:
  start_et:           float                   # TDB seconds past J2000
  stop_et:            float
  midtime_et:         float
  exposure_s:         float
  sclk_start:         "1/1484573293.055"      # sce2s string forms
  sclk_midtime:       "1/1484573295.118"
  sclk_stop:          "1/1484573297.181"
```

and, under `observation`, `shutter_mode` (the instrument's shutter mode
string when the host exposes one, e.g. Cassini's `BOTSIM`; the BOTSIM rule
in section 3.2 is undetectable without it).

`cmatrix_original` and the `times` and frame fields are written whenever a
`NavResult` exists; `cmatrix` additionally requires an offset and no fitted
rotation. `ck_frame_id` is derived at navigation time from the observation's
host identity -- for Voyager, from the spacecraft number
(`scid * 1000 - 100`) -- and recorded per image so the writer never infers
it. It does not belong in the instrument config files: the config is keyed
per instrument, and Voyager needs a different id per spacecraft under one
key.

These field names and shapes are also the ones the results-index schema
declares in advance (`plans/RESULTS_DB_PLAN.md` section 2.3); changing them
here means changing them there.

### 2.4 Where it is computed

The computation lives in `spindoctor/support/cmatrix.py`, a single entry
point taking the observation, the offset, and the fitted-rotation flag, and
returning a frozen dataclass carrying both matrices, the frame identities,
and the times block. It imports oops -- it must, to read the observation's
frame and FOV -- which is fine on the pipeline side; the constraint that
bans oops applies to the kernel writer (section 3.6). When oops gains its
own corrected-attitude API, this module's body is replaced and its
interface stays. A `#` comment records that intent.

`NavResult` is constructed inside the ensemble, which never sees the
observation, and it is a frozen dataclass -- so the wiring is: `NavResult`
gains an optional `pointing` field (default None), populated in
`_navigate_pipeline` in `nav_orchestrator/orchestrator.py` via
`dataclasses.replace` at the point where `context.obs` is in hand (the same
neighborhood that builds provenance from `obs.midtime`), and
`curator.build_metadata_dict` serializes it. Phase A states this as the
insertion point so the implementer does not go looking for an `obs`
argument inside `ensemble()` that does not exist.

---

## 3. The kernel generator

### 3.1 Frames, per mission

The correction is measured at the camera and written at the object the
existing kernels describe, with reference frame J2000:

| Mission | CK object | Camera-to-CK-object relation |
|---|---|---|
| Cassini ISS | -82000 (bus) | fixed FK rotation, **not** small |
| Voyager ISS | -31100 / -32100 (scan platform) | fixed FK rotation (the platform-to-camera `pxform` oops applies; not identity) |
| Galileo SSI | -77001 (scan platform) | identity (SSI reads the platform frame) |
| New Horizons LORRI | -98000 (spacecraft) | fixed FK rotation |

`F = pxform(ck_frame, camera_frame, et_mid)` is always computed, never
assumed -- Cassini's `F` is a permutation-like matrix nowhere near
identity. The corrected attitude of the CK object and the correction as a
rotation in the CK frame's own coordinates are

```text
C_ck_corrected(mid) = F^-1 . cmatrix
delta               = C_ck_corrected(mid) . C_ck_original(mid)^T
                    = F^-1 . (R^T M R) . F        (the oops-frame M conjugated into the CK frame)
```

with `C_ck_original(mid)` read from the original kernels. Across the
exposure the correction is held **body-fixed**:

```text
C_ck_corrected(t) = delta . C_ck_original(t)      for t in [start, stop]
```

That is the physical model -- the spacecraft is pointed slightly wrong and
the error turns with it -- so attitude still varies correctly within the
exposure and smear geometry stays right. **Exception: Voyager.** The
navigated model assumed a constant, tolerance-snapped attitude (section
2.2), so a Voyager segment carries that single corrected attitude,
constant across its window; writing time-varying pointing there would
disagree with what was navigated.

### 3.2 What gets a segment

- **Eligible:** any image whose status is `success` or `conflicted` and
  which has a recorded `cmatrix`. No confidence or rank threshold. The
  consequence is that the report and the segment comments must carry
  status, status reason, confidence, and rank, since they are then the only
  way a consumer can filter low-confidence or conflicted pointing.
- **Not eligible:** every other image -- including a result with a fitted
  `rotation_deg`, which has no `cmatrix` (section 1) and is reported as
  `rotation_unsupported`. No segment; pointing falls through to the
  originals. No uncorrected copies are written.
- **BOTSIM pairs** (simultaneous Cassini NAC and WAC exposures) would ask
  one bus attitude to honor two corrections at once. Pairing predicate: two
  eligible Cassini images with `observation.shutter_mode == 'BOTSIM'`,
  opposite cameras, and `|start_et_a - start_et_b| <= 1.0 s`. One
  correction per pair: NAC wins; WAC is used only when the NAC member is
  not eligible. The loser is reported as `botsim_loser`.
- **Voyager** corrected files pair only with the per-encounter SEDR
  platform kernels (the platform object is the only one Voyager ISS
  pointing reads; the decades-spanning bus kernels cover a different object
  and an image served only by them cannot have navigated in the first
  place). An eligible image whose baseline no candidate reproduces --
  Voyager or otherwise -- is reported as `no_reproducing_baseline`
  (section 3.3).

### 3.3 Segments and files

One **type 3** segment per eligible image, interpolation interval exactly
`[start_et, stop_et]`.

**Records.** At exposure start, midtime, and stop. When `exposure_s`
exceeds 10 s, additional records at a 1 s cadence. (SPICE offers no API to
enumerate a type-3 segment's interior records, so the earlier idea of
copying the original's record times is dropped; at typical exposure lengths
the window contains no interior records anyway.) Time tags must be
**strictly increasing in encoded SCLK**; an exposure so short that start,
mid and stop collapse to one tick produces a single-record segment at
midtime, which type 3 permits and Phase B tests. Records are quaternions
from `cspyce.m2q` with **sign continuity enforced**: `m2q` fixes the scalar
component non-negative, which can flip sign between adjacent records and
corrupt the interpolation; each record is negated as needed to keep a
non-negative dot product with its predecessor.

**Clocks.** Encoded SCLK comes from `cspyce.sce2c(sclk_id, et)` where
`sclk_id = cspyce.ckmeta(ck_frame_id, 'SCLK')` -- the spacecraft clock ID
(-82, -31, -77, -98), which is **not** derivable from the CK object id by
integer division (`-31100 // 1000` is -32 in Python: wrong spacecraft,
silently). The SCLK kernel furnished must be the one navigation used,
resolved from `provenance.spice_kernels`; a different SCLK is a silent
time-tag error.

**Angular velocity: copied unchanged, never rotated.** CK angular velocity
is expressed in the segment's **base reference frame** (J2000), per the CK
Required Reading, and the corrected frame differs from the original by a
constant body-fixed rotation -- two frames rigidly attached to each other
have identical angular velocity in the base frame. So when the original
segment carries AV (`ckgpav` succeeds), the corrected records carry the
same vectors bit-identically and `avflag = 1`; when the original has none
(`ckgpav` raises for want of AV -- Voyager and Galileo type 1), the
corrected segment writes `avflag = 0` and queries fall back to `ckgp`.
Rotating AV through `delta` -- superficially the "thorough" treatment -- is
the wrong-frame error, and Phase B's test must fail if it is introduced.

**Files mirror the originals.** Each output `.bc` corresponds to exactly
one original CK file and carries the segments of the images whose corrected
attitude that original supplied. Output size stays proportional to the
originals, the baseline pairing is legible from the filename, and
regeneration is per original file. Naming: the original basename with
`_nav` before the extension -- `03236_04002ra.bc` becomes
`03236_04002ra_nav.bc` (a real reconstructed-kernel name; use real names in
tests). No label files are written.

**Assignment is by reproduction, not by guessing.** `provenance.
spice_kernels` records sorted basenames only (deliberately, for hash
determinism), it does not preserve load order, and in a batch process it
accumulates kernels from earlier images -- a superset. So:

1. **Pre-index once per run:** every CK under the mission's kernel
   directories is scanned with `ckobj` and `ckcov` (SEGMENT level, TDB), so
   candidates are filtered by CK object and midtime coverage before any
   furnishing. Basenames from provenance resolve against this index; a
   basename found in more than one directory contributes each file as a
   candidate.
2. **Per image** (grouped by candidate set so the kernel pool is switched
   per group, not per image): furnish the supporting kernels (LSK, SCLK,
   FK) plus one candidate CK at a time, evaluate the original attitude at
   midtime -- for Voyager via the snapped
   `ckgp(ck_id, sce2c(scid, mid), 800 + exposure_s/48, 'J2000')`, composed
   with `P`, matching section 2.2; otherwise via `pxform` -- and keep
   candidates that reproduce the recorded `cmatrix_original` to within
   **1e-9 radians** (an angular tolerance on the rotation between the two
   matrices; this is a reproduction bound, far tighter than any navigation
   tolerance).
3. **Tie-break** when several reproduce (expected: the holdings carry
   reconstructed, gapfill and predicted sets with overlapping coverage, and
   oops loads gapfill by default): prefer by kernel class from the
   directory name -- reconstructed over gapfill over predicted -- then the
   lexicographically greatest basename. The reproducing candidates agree on
   the attitude by construction, so the choice affects only which output
   file carries the segment; it must merely be deterministic.
4. An eligible image **no** candidate reproduces gets no segment, reason
   `no_reproducing_baseline`. This is also the baseline-drift detector: if
   the kernel set changed since navigation, reproduction fails and the
   image is refused rather than corrected against a baseline that no
   longer exists.

**Omission reasons**, the complete set: `not_eligible`, `botsim_loser`,
`rotation_unsupported`, `no_reproducing_baseline`, `degenerate_exposure`
(reserved for an exposure the single-record path cannot express, should
that occur).

### 3.4 Companion outputs

- **Comment area**, per file: generator version, configuration hash, the
  baseline kernel basenames and the SCLK kernel used, and one line per
  image (name, time, offset, sigma, confidence, rank, status, status
  reason). Mechanics: segments are written through `ckopn` (with `ncomch`
  reserved generously up front -- estimate the comment size and reserve
  with slack) / `ckw03` / `ckcls`; the comment text then goes in via
  `dafopw` / `dafac` / `dafcls`. Reading it back (acceptance) uses
  `dafopr` / `dafec`.
- **Meta-kernel** per file set, furnishing originals first and corrections
  after, so precedence is explicit rather than an ordering a user has to
  know.
- **CSV report, one per mission.** Columns: `image_name`, `utc`, `et`,
  `sclk`, `offset_dv`, `offset_du`, `sigma_dv`, `sigma_du`, `confidence`,
  `confidence_rank`, `status`, `status_reason`, `source_bc`,
  `omission_reason`. Every image considered appears exactly once:
  `source_bc` names the file carrying its segment, or is empty with
  `omission_reason` set. Sources, precisely: `offset` (top level, `[dv,
  du]`, unrounded) for the offset pair; `sigma_px` from
  `navigation_result` (rounded there, and reported as recorded);
  `confidence` top level; `confidence_rank` and `status_reason` from
  `navigation_result`; `status` top level; `sclk` from
  `times.sclk_midtime`. The column set is expected to evolve; treat it as
  version 1.

### 3.5 The program

`sd_create_ck`, a new dispatch module in `src/spindoctor/cli/`, entry point
in `pyproject.toml`, arguments: mission, time range, input roots, output
directory, and the shared logging surface. Add `SD_CREATE_CK =
'sd_create_ck'` to `spindoctor/config/program_names.py` and
`PROGRAM_NAMES`; declare `PROGRAM_NAME`; call `add_logging_arguments(
parser)` (image flags on -- it processes images individually); wrap the
configuration load in `reporting_logging_errors()` with the literal
adjacency the logging test suite asserts. No entry is needed under
`logging.programs` in `config_015_logging.yaml` (it ships empty and keys
are optional). Anything that degrades or omits a result goes to both logs:
per-image detail to the image log, one line plus a count to the run log.

### 3.6 The writer's imports, and cspyce

The writer core lives in `spindoctor/cli/ck/` and imports `cspyce` and
nothing from oops -- and nothing from `spindoctor.support`, which is where
the oops-importing `cmatrix` module lives; one careless helper import there
would drag oops in transitively. The guarantee is asserted on `sys.modules`
after importing the writer package in a fresh interpreter, not by scanning
source text.

The `cspyce` surface the writer needs, all present in the installed 2.3.6:
`furnsh`, `unload`, `kclear`, `pxform`, `frmnam`, `namfrm`, `ckmeta`,
`sce2c`, `sce2s`, `ckobj`, `ckcov`, `ckgp`, `ckgpav`, `m2q`, `ckopn`,
`ckw03`, `ckcls`, `dafopw`, `dafac`, `dafcls`, `dafopr`, `dafec`. One
global to respect: `cspyce.use_errors()` / `use_flags()` is process-wide
and shared with oops; the writer assumes the exceptions regime
(`use_errors`, the package default) and must never flip it.

---

## 4. Implementation phases

### Phase A — C-matrix in the metadata

`spindoctor/support/cmatrix.py` per section 2.2; the `NavResult.pointing`
field and the `_navigate_pipeline` wiring per section 2.4; curator
serialization of `pointing`, `times`, and `observation.shutter_mode` per
section 2.3.

Hermetic tests (synthetic FOV and frames):

- A planted offset produces a `cmatrix` whose correction, inverted,
  recovers that offset; the test fails if the sign of `xy_offset` flips.
- A zero offset produces `cmatrix == cmatrix_original` exactly, through
  the `M = I` guard.
- The Cassini-style flip: with a synthetic `R = diag(-1,-1,1)` between the
  "oops" and "SPICE" frames, the recorded `cmatrix` reproduces the offset
  in the SPICE convention; the test fails if the `R` conjugation is
  dropped (the un-conjugated result negates both tangent components).
- A result carrying a fitted rotation records `cmatrix_original` but no
  `cmatrix`.
- Determinant/orthonormality violations raise.

Integration tests (real frames, marked `integration`): for one Cassini
NAC, one Cassini WAC, one LORRI, and one Galileo frame, the measured `R`
equals the section 2.1 constant to 1e-9 and is identical at start, mid and
stop; for one Voyager frame, `cmatrix_original` equals the frozen oops
attitude.

### Phase B — Writer core, one image

The type-3 segment writer for a single image: `F` and `delta` per section
3.1, records, SCLK, quaternion sign continuity, AV policy, the
single-record degenerate path.

Tests are hermetic: the suite writes its own minimal LSK and SCLK text
kernels (small text files `furnsh` accepts), plus an original CK produced
by the writer's own primitives, so no holdings are needed. Write a kernel
from a constructed attitude history and correction, furnish, query back
with `ckgpav` at `tol = 0`:

- Attitude at start, mid, stop and an interior time matches the composed
  truth within 1e-9 radians.
- AV is bit-identical to the original's; the test **fails if AV is rotated
  through `delta`**.
- An AV-less original yields `avflag = 0` and a working `ckgp` fallback.
- A sign-discontinuous quaternion sequence is repaired; interpolated
  attitude mid-record stays continuous.
- A sub-tick exposure produces a valid single-record segment.
- Sub-spacecraft-clock ids come from `ckmeta`, asserted for all four CK
  objects.

### Phase C — Grouping, mirroring, naming, and the coverage rules

The candidate pre-index (`ckobj`/`ckcov`), assignment-by-reproduction with
its tolerance and tie-break, basename resolution, output naming, the
BOTSIM predicate, the Voyager snapped-baseline reproduction, and the
omission-reason set.

Tests: assignment keeps exactly the reproducing candidate and refuses when
none reproduces; the tie-break is deterministic and class-ordered; a BOTSIM
pair yields one segment and a `botsim_loser` row; a fitted-rotation result
yields `rotation_unsupported`; filenames derive from real-shaped original
names.

### Phase D — Round-trip validation

The acceptance test of the whole plan, as an integration test, run as
**three subprocesses** because oops caches frames and manages its own
kernel pool, and a mid-process `furnsh` is not guaranteed to take effect:

1. *Navigate* a real image normally; record offset and `cmatrix`.
2. *Generate* the corrected kernel from that result.
3. *Re-navigate* in a fresh process in which the corrected kernel is
   furnished immediately **after** the host's `from_file` returns (after
   its lazy CK furnishing) and before any geometry is computed. In that
   process, first assert the pointing actually changed: `ckgp` /
   `pxform` at midtime differs from the uncorrected value by the expected
   correction. This distinguishes "kernel took effect" from "kernel was
   silently buried", which the offset alone cannot.
4. Assert the re-navigation's offset is approximately `(0, 0)` and its
   `cmatrix` matches run 1's as a rotation. The comparison is valid only
   when both runs commit the same winning technique set; the test records
   both sets and fails as *inconclusive-mismatch* (not as a pass) when
   they differ.

Tolerances: the target is at or below **0.1 px per axis**, and the
C-matrix target is that offset's angular equivalent at the instrument's
pixel scale. Decision rule: when the measured residual is at or below
0.1 px, pin the test at measured-plus-margin; when it is above, **stop and
diagnose** -- a larger residual is a defect (most likely a section 2
convention error), never a tolerance to raise. The section 2.2
rotation-vs-shift bound (~0.04 px worst case, WAC corner) is part of the
budget.

Cohort: one star-navigated Cassini NAC frame (best-constrained truth), one
Cassini WAC frame, and one frame from each other instrument that has a
navigated library frame. This is where acceptance criterion 2's
per-instrument claim is earned. Local-only (needs local binary kernels);
marked `integration`.

### Phase E — Comment area, meta-kernel, report, driver

Section 3.4 and the `sd_create_ck` driver per section 3.5.

Tests: every considered image appears exactly once in the CSV, with either
`source_bc` or `omission_reason`; the comment area reads back via
`dafopr`/`dafec` and names the baseline and SCLK kernels; the meta-kernel
furnishes originals before corrections; the driver passes the shared
logging-surface assertions.

### Phase F — Documentation and reconciliation

A user-guide page: what the kernels are, what they claim and do not claim
(corrected only where an image was navigated; the originals remain
required -- prominently, not as a footnote); loading with and without the
meta-kernel; the naming convention; the report columns. A dev-guide page:
the frame relations of section 2.1 including the oops-flip table, the
derivation, the AV rationale, and the writer's structure. Sphinx toctrees
updated; plan files reconciled.

---

## 5. Acceptance criteria

1. Every navigated image with an offset and no fitted rotation carries
   `cmatrix`, `cmatrix_original`, the frame identities and the `times`
   block; both matrices are proper rotations; a fitted-rotation result
   carries `cmatrix_original` only.
2. The recorded `cmatrix` reproduces the navigated offset (inverting the
   correction recovers `(dv, du)` to a stated sub-pixel tolerance) on real
   frames from each instrument that has a navigated library frame, and the
   measured oops-vs-SPICE `R` matches the section 2.1 table on each.
3. Round-trip: navigate, generate, furnish, re-navigate yields an offset
   at or below the pinned tolerance with a matching C-matrix, the
   pointing-actually-changed assertion passing, and matching technique
   sets between runs.
4. A kernel written by this tool loads in a plain `furnsh` session with no
   SpinDoctor code present; `ckgp` returns the expected attitude inside a
   navigated exposure and falls through to the original outside it;
   `ckgpav` additionally returns the original's angular velocity wherever
   `avflag = 1`, and the report records which files carry AV.
5. Every image considered appears exactly once in its mission's CSV, with
   either a `source_bc` or an `omission_reason` from the section 3.3 set.
6. No image whose recorded baseline no candidate kernel reproduces
   receives a segment.
7. Importing the writer package in a fresh interpreter loads no oops
   module and nothing from `spindoctor.support`, asserted on
   `sys.modules`.
8. `ruff check`, `ruff format --check`, `mypy --strict`, `sphinx-build -W`
   and `pymarkdown scan` all pass; suite coverage stays at or above 90%,
   with the writer core covered by the hermetic self-written-kernel tests
   of Phase B, not only by integration runs.

---

## 6. Risks and constraints

**Frame-convention errors are the dominant risk, and hermetic tests cannot
see the worst one.** The oops-vs-SPICE flip (section 2.1) only bites when
real host frames meet real kernels, which is exactly what unit tests mock
away. That is why Phase A carries per-instrument integration assertions on
`R` and why Phase D exists. The sign conventions that *are* hermetically
testable (`xy_offset`, `axisar` vs `rotate`, composition order) each have
a test that fails when flipped.

**The AV trap points the other way than intuition.** The plausible-looking
"thorough" treatment -- rotating AV through the correction -- is the
wrong-frame error, because CK AV lives in the base frame and the
correction is a constant rigid rotation of the body. The correct treatment
is verbatim copy. Phase B pins this in the correct direction.

**One bus attitude cannot serve two cameras.** The BOTSIM rule manages the
case that exists today. If per-camera conflicts matter more broadly, the
answer is corrected camera frames in a supplemental FK -- a larger change
with an adoption cost, not in this plan.

**Segment count per file.** Thousands of segments make CK lookups a linear
scan. Existing NH kernels carry hundreds; measure the largest Cassini file
produced and report the number in the PR.

**Consumers need the originals.** An overlay claims pointing only inside
navigated exposures; a user loading only the corrections gets nothing
between images. Prominent in the user guide.

**The correction is a midtime snapshot held body-fixed across the
exposure** (constant outright for Voyager, matching what navigation
assumed). It is an approximation for an exposure during which the pointing
error itself changed; nothing available today measures that.

---

## 7. Follow-ups

File as tracking issues alongside the implementation issue:

- **Consumers switch from the offset to the C-matrix.** Once the Phase D
  round trip has shown, per instrument, that the recorded `cmatrix` means
  what the offset means, every main program that today reads the metadata
  `offset` and builds an `OffsetFOV` -- backplanes, reprojection/mosaics,
  and any later consumer -- moves to consuming `cmatrix` instead, and the
  offset becomes a derived report value rather than the applied one. This
  is the reading half of #50 (the plan itself delivers the writing half),
  and it must not start before the round-trip evidence exists: the round
  trip is the only end-to-end check that the two representations agree.
- **Replace the C-matrix derivation with the oops API** when oops gains
  one: `spindoctor/support/cmatrix.py` keeps its interface, its body goes.
- **Fitted-twist support**: record the rotation pivot
  (`rotation_pivot_vu`) on the technique result and through the ensemble,
  define the boresight-referenced conversion -- including the sign flip
  from the `(v, u)`-to-`(x, y)` axis swap, under which a positive
  image-plane rotation is a negative rotation about camera +Z -- and lift
  the `rotation_unsupported` omission. Costs only Galileo today.
- **Static per-instrument twist** belongs in an FK/IK correction rather
  than per-image CK records; the measured LORRI and SSI values are
  candidates.
- **Adapt the CK writer when corrected instrument kernels exist.** If new
  FK/IK kernels encoding the measured static rotations are produced (the
  previous item), the writer's world changes: `F` and the camera frame
  definitions come from the new kernels, corrections shrink by the static
  twist the FK now carries, and every previously-generated overlay kernel
  is stale against the new baseline. The issue covers deciding whether the
  adaptation is feasible per mission (an FK change alters what
  `cmatrix_original` reproduction means, so recorded metadata may need
  regeneration first), and regenerating the overlay set against the new
  kernels.
- **SPICE database registration** for oops kernel selection, if wanted
  before that database is replaced: rows and load priority for the
  corrected kernels, plus the per-mission furnish-policy story. The
  meta-kernel path works without it.

Filed already, because it is broader than this plan and blocks nothing:
the documentation chapter specifying the metadata JSON format -- every
key, its meaning, presence rules, and examples, including the `pointing`
and `times` blocks this plan adds (#431).

---

## 8. Execution protocol

1. Branch `rf_ck_kernels` off current `main`; one commit series per phase.
2. Per phase: dispatch an **implementer subagent** (Opus-class) whose
   prompt embeds that phase's section of this plan verbatim plus sections
   1-3, so the subagent needs no other briefing and does not have to
   locate this file. Then dispatch an **independent, fresh-context
   adversarial reviewer** (also Opus-class) with the diff, the same plan
   sections, and instructions to (a) verify each normative statement of
   sections 2 and 3 against the code line by line, (b) run the phase's
   tests plus `ruff check src tests`, `ruff format --check src tests`, and
   `mypy src tests`, (c) attack the sign and frame conventions
   specifically -- including re-deriving `R`, `M`, `delta` and the AV
   rule independently rather than trusting the plan's statement of them,
   and (d) hunt for convention violations and unstated deviations. Fix
   rounds until the review is clean; the controller, not the implementer,
   judges cleanliness.
3. The reviewer must verify each guarantee by **breaking the source and
   confirming a test fails**. For this plan that means, at minimum:
   flipping the `xy_offset` sign, swapping `axisar` for `rotate`, dropping
   the `R` conjugation, reversing the `delta` composition, and rotating AV
   through `delta` -- each in turn, each caught by a named test. A test
   that passes against a deliberately broken implementation is a defect in
   the test, and is reported as one.
4. Deviations discovered mid-phase are recorded in the phase commit
   message and reconciled into this plan file in the same commit, so the
   document the next reviewer holds is never stale. Scope changes go to
   the operator instead.
5. Final sweep before the pull request to `main`:
   `./scripts/run-all-checks.sh -i`, plus the Phase D round trip re-run on
   the final revision with its measured residuals reported in the pull
   request.
6. One pull request to `main`: summary, phase map, evidence, `Closes` per
   issue, and the plan and guide reconciliation included.
