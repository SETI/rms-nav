# SPICE C-Kernel Generation: Design Options (#188)

Status: pre-decision report, 2026-07-30. No code has been changed. This is the design note called for in ENGINEERING_PLAN.md ("CK kernels (#188, prerequisite #50)"). Sources: the current pipeline (`src/spindoctor/`), the old generator (`csmithing/navigation/nav_main_csmith.py`), the kernel holdings under `$OOPS_RESOURCES/SPICE`, and the spicedb/oops loading code in the venv.

## 1. Facts the design rests on

**What the pipeline records.** One JSON per image at `<nav_results_root>/<volume>/<filespec>_metadata.json`. The authoritative fields: top-level `status` (`success` / `failed` / `conflicted` / `error`), top-level `offset` = `[dv, du]` pixels at full precision (convention: predicted `(v, u)` means actual `(v + dv, u + du)`), `confidence`, `navigation_result.confidence_rank`, `navigation_result.rotation_deg` + `sigma_rotation_deg` (present only when twist was fitted -- currently Galileo SSI only), and `navigation_result.provenance.image_et` (midtime, TDB) plus `provenance.spice_kernels` (basenames of every kernel furnished at nav time). **Not** recorded: start/stop times, exposure, SCLK, camera frame, FOV model. About 90% of the live results tree is an older schema (including one obsolete `status: 'ok'` file); the `sd_stats` SQLite DB is a convenient per-image index (offset, status, et, camera).

**Offset semantics.** Downstream tools apply the offset as `oops.fov.OffsetFOV(fov, uv_offset=(du, dv))` -- a constant shift in the undistorted camera tangent plane `(x, y)`. The attitude-space equivalent is a small rotation about the camera X/Y axes (plus Z for twist).

**Metadata transition (load-bearing for this design).** oops will soon support setting and retrieving a C-matrix for a nav result, and the pipeline metadata will eventually record that corrected C-matrix in place of the pixel offset. The generator must support **both eras**: a thin input-adapter layer produces the per-image correction rotation (camera frame) either from today's `offset [dv, du]` (plus `rotation_deg` when present) or, later, from the recorded C-matrix taken as a delta against the original attitude at midtime. The offset conversion needs only the camera's boresight pixel scale, which the IK reports (`getfov` or IK constants), so both paths stay pure `cspyce` -- no oops import anywhere: furnish LSK/SCLK/FK/IK/original CKs, `sce2c` for time tags, `pxform` for the camera-to-CK-frame conjugation, `ckw03`/`dafac` for writing. Caveat of the offset path: a boresight-pixel-scale conversion matches the pipeline's `OffsetFOV` semantics exactly at the boresight and to first order elsewhere; the round-trip validation quantifies the residual. The writer core is identical in both eras -- only the adapter changes.

**Which frame a corrected CK must target** (the same object the existing CKs use, reference J2000):

| Mission | CK object | Existing CKs | Camera relation |
|---|---|---|---|
| Cassini ISS | -82000 (bus) | type 3 + AV | NAC/WAC are fixed FK offsets from the bus |
| Voyager ISS | -31100 / -32100 (scan platform) | type 1, no AV | ISS-NA coincident with platform |
| Galileo SSI | -77001 (scan platform) | type 1 + type-3 predicts, B1950 | SSI reads the platform frame directly |
| NH LORRI | -98000 (spacecraft) | type 2 + 3 + AV | LORRI fixed FK offset |

A correction measured in the camera frame must be conjugated through the fixed FK rotation into the CK frame before writing.

**CK types.** Type 1: discrete instances, consumer supplies a lookup tolerance (`ckw01`). Type 2: piecewise constant-rate intervals (`ckw02`). Type 3: discrete records, linearly interpolated inside declared interpolation intervals -- the community standard for reconstructed pointing (`ckw03`). Type 5: polynomial windows (`ckw05`). Types 4 and 6 have no public writers. The venv's `cspyce` 2.3.6 exposes all four writers plus `dafac` for comment areas, so the external `msopck` utility the old tool shelled out to is not needed.

**How kernels get loaded.** oops selects kernels from `$OOPS_RESOURCES/SPICE/SPICE.db` (SQLite; `LOAD_PRIORITY` orders the furnish; last-furnished wins in SPICE, and lookups fall through to earlier-loaded kernels wherever a later one has no coverage). Mission quirks: Cassini furnishes CKs lazily month-by-month during `from_file`; Voyager furnishes everything eagerly at import; Galileo bypasses the DB with a hard-coded file list inside oops; New Horizons furnishes by time window. `spicedb` has no write API, so registering new kernels means direct SQLite inserts. Users outside oops just `furnsh` the corrected CK after the originals.

**The old tool.** `nav_main_csmith.py` was Cassini-only. It rewrote each *predicted* CK wholesale as a type-3 kernel via `msopck`: boresight-only correction (via apparent RA/Dec, applied to the bus frame), three records per navigated exposure (start/mid/end, angular velocity zeroed), and a linear ramp of the correction across the gaps between images; failed images were skipped. Known defects worth not repeating: AV copied unrotated when quaternions were rotated, fabricated ("ramped") corrections at times with no measurement, hard aborts at kernel boundaries, a placeholder comment area, and enough bit-rot that it cannot run today.

## 2. How a user would want to use the kernels

- Load navigated pointing into **any** SPICE-based tool (oops, ISIS, plain cspice/spiceypy scripts) with a `furnsh`, no SpinDoctor and no JSON files -- geometry, backplanes, and mosaics just work.
- Run SpinDoctor's own reprojection/backplanes without `--nav-results-root`.
- Publish as a community or PDS deliverable (the "c-smithed kernels" product), with provenance a reviewer can audit.
- Choose scope: one volume, one mission, one time range; a meta-kernel that loads originals + corrections in the right order.
- Distinguish corrected from uncorrected time: know whether pointing at time *t* came from navigation or from the original kernel (segment IDs / coverage tell them).
- Skim a lightweight per-image CSV (image name, time, SCLK, uncertainty, confidence) to judge coverage and quality without downloading the kernel bundle.
- For images that did not navigate: either transparently fall back to original pointing, or opt into explicit uncorrected copies so a single kernel covers the whole image set.

## 3. Cross-cutting problems (every design inherits these)

1. **Two metadata eras must both work.** The generator needs an input adapter: offset-based today (pixel offset to a small camera-axis rotation via the IK boresight pixel scale), C-matrix-based later (recorded C-matrix vs original attitude at midtime). The round-trip validation (write CK, reload, reproject with no offset, compare against the navigated geometry) is the acceptance gate for both paths and must define the acceptable residual. In either era the recorded correction is a midtime snapshot; attitude at exposure start/end comes from composing the midtime delta onto the original CK's attitude, so the generator reads the original CKs regardless.
2. **Camera frame vs CK frame.** The correction is measured per camera but written at bus/platform level. Cassini BOTSIM pairs (simultaneous NAC + WAC) expose the flaw: one bus attitude cannot honor two different corrections at the same instant. Decided policy: **skip the loser** -- one correction per BOTSIM pair, with winner selection following the old tool's rule (NAC preferred, WAC only when the NAC solution is unusable); the losing image's exposure window gets no segment and falls through to original pointing.
3. **Twist.** Only Galileo currently fits per-image rotation. The FOV-distortion report's static twists (LORRI +0.191 deg, SSI -0.053 deg) belong in an FK/IK fix, not in per-image CK records; Voyager's twist is frame-varying and only capturable if rotation fitting is turned on there.
4. **Metadata gaps.** No start/stop/exposure/SCLK in the JSON. Since the schema is already changing for the C-matrix, extend it in the same pass with start/stop ET, SCLK, and frame identity -- that makes CK generation a fast metadata-only pass with no holdings access and no oops import. (The fallback -- re-opening every image -- is slow and drags oops back in.)
5. **Baseline drift.** The delta is relative to whatever attitude the original kernels gave at nav time. If the originals ever change, the deltas are stale. Mitigation: stamp the baseline kernel basenames (already in `provenance.spice_kernels`) into the segment comments and regenerate on kernel-set change. The local kernel set is frozen, so the risk is mainly for external redistribution.
6. **Making oops prefer the corrected kernels is mission-specific.** Cassini's lazy monthly furnishing can bury an overlay furnished early; Galileo ignores the DB entirely. The chosen design needs a companion loading story: SPICE.db rows at higher `LOAD_PRIORITY` under a name oops actually queries, and/or an explicit post-`from_file` furnish hook, and/or small oops host changes.
7. **Legacy metadata (decided).** The existing results files are ignored entirely: the tool reads the current schema only, and results will be regenerated before kernel production. No dual-schema support.
8. **SCLK choice.** Encoded-SCLK time tags must use the same SCLK kernels oops furnishes (`cas00172`, `vg100019`/`vg200022`, `mk00062a`, `new-horizons_1280`); newer SCLKs sit on disk unregistered -- pick deliberately and record the choice in the comment area.

## 4. The designs, best to worst

### 1. Overlay CK: per-image type-3 segments covering only each exposure (ADOPTED 2026-07-31)

A standalone `sd_create_ck` reads the metadata (or the stats DB) for successful images. For each image it samples the **original** attitude at exposure start/mid/end (plus any original CK records falling inside the window), composes the correction delta (recorded C-matrix vs original at midtime, conjugated into the CK frame), and writes one type-3 segment whose interpolation interval covers exactly `[start, stop]`. Output files **mirror the original CK files' coverage ranges**: one corrected `.bc` per original CK it overlaps (Cassini: per `YYDDD_YYDDD` reconstructed file; Galileo: per orbit CK; NH: per merged monthly/yearly file; Voyager: per SEDR encounter kernel), named after the original with a suffix. This keeps file sizes proportional to the originals, makes the baseline pairing explicit in the name, and regenerates incrementally one original file at a time -- and it avoids any volume-based split, which has no meaning in PDS4 (a flat directory). The originals stay loaded beneath; SPICE falls through to them at all other times. Failed and unnavigated images get no segments at all (decided): their pointing falls through to the original kernels, and the earlier optional copy-unnavigated mode is dropped from scope. Comment area carries generator version, config hash, per-image offsets/confidence, and baseline kernel names. Companion outputs: a meta-kernel per file set, SPICE.db registration rows, and an **optional CSV report** -- one row per navigated image (image name, UTC time and ET, SCLK, offset, sigma, confidence, confidence rank, and which `.bc` file carries it) -- so users can assess coverage and quality without downloading the kernel set.

*Why best:* honest coverage (claims correction only where one was measured); type 3 is the standard consumers expect; attitude varies correctly within the exposure (smear geometry right); tiny outputs (order hundreds of bytes/image, so ~100 MB for all of Cassini); regenerates incrementally per volume; one architecture serves all four missions and both oops and non-oops users.

*Problems:* all of section 3, most sharply the BOTSIM conflict (#2) and the oops precedence story (#6). Overlapping NAC/WAC exposure windows produce overlapping segments where last-loaded silently wins. A user who loads only the overlay gets no pointing between images -- must document that originals remain required. Angular velocity (decided: include it, `avflag = 1`): the original AV must be rotated through the correction delta correctly -- copying it unrotated was exactly the old tool's bug. Thousands of segments per file make CK lookups a linear scan -- likely fine (NH files already carry 374 segments) but should be measured. Mirroring assumes each mission has a sensible original file to pair with; for Voyager the decades-spanning bus "super" CKs are the wrong partner (ISS pointing reads the per-encounter SEDR platform kernels, so mirror those), and a rule is needed for images whose exposure only the bus kernels cover. Needs the schema extension (#4) for exposure windows, or falls back to re-opening images.

### 2. Discrete type-1 records at exposure midtimes

Same architecture (including the mirrored file ranges), but a single corrected-attitude record per image at midtime in a type-1 segment. Precedent: the Voyager ISS SEDR CKs are exactly this.

*Pros:* the simplest possible writer; trivially verifiable; tiny; a natural first milestone.

*Problems:* type 1 makes the *consumer's* `ckgp` tolerance part of the contract -- too small and exact-time lookups fail on encoded-SCLK float mismatches, too large and a query near-but-outside an exposure silently snaps to that image's pointing. Attitude is frozen across the exposure, which is worse than the original kernel for long exposures or slews. oops behaves differently per mission (Voyager's `SpiceType1Frame` has a baked-in tolerance; the Cassini path expects continuous type 3). No AV. Good as a prototype stage inside Design 1; weak as the shipped product.

### 3. Opt-in corrected camera frames (new FK + delta CK)

Define new CK-based frames (e.g. `CASSINI_ISS_NAC_NAV`, new IDs) in a supplemental FK, with a CK giving the corrected camera attitude. Originals are untouched; consumers opt in by naming the new frame.

*Pros:* the only design that is *correct at camera level* -- BOTSIM pairs get independent per-camera corrections, and per-image twist lands where it physically belongs; zero risk of contaminating the standard frames; the correction data is pure and small.

*Problems:* nobody benefits until consumers change frame names -- oops hosts hardwire `CASSINI_ISS_NAC` etc., so oops (or spindoctor wrapper) changes are required, and third-party tools expecting standard frames get nothing at all; unofficial NAIF ID allocation; a permanent documentation/support burden; and it still needs all of Design 1's timing/SCLK machinery. High adoption cost for the same math. Worth revisiting only if camera-level conflicts prove to matter in practice.

### 4. One mini-CK file per image, aggregated by meta-kernels

Same math as Design 1, but one `.bc` per image; meta-kernels define collections.

*Pros:* perfectly incremental; per-image provenance is trivial.

*Problems:* Cassini alone is ~400k images -- CSPICE's total-loaded-kernel limit (~5000 DAF handles) makes "load a mission" literally impossible without a consolidation step, at which point this collapses into Design 1 with extra filesystem pain; SPICE.db gains a row per file; meta-kernels hit `KERNELS_TO_LOAD` path-length awkwardness. Only sensible as a transient intermediate artifact.

### 5. Corrected copies of the original CK files (the old csmithing approach, modernized)

Rewrite every original CK that overlaps the navigated set, rotating each record; ramp or hold the correction between images.

*Pros:* one self-contained drop-in kernel set; failed images keep original pointing automatically (the "copy original pointing" requirement falls out for free); familiar to the operator.

*Problems:* fabricates corrections at times with no measurement -- ramping across inter-image slews is unphysical, and the product silently claims navigated quality everywhere (the old tool's worst trait); enormous outputs (the Cassini reconstructed set alone is 6.7 GB) regenerated wholesale whenever navigation improves; must faithfully rotate AV (the old bug) and reproduce each mission's structural quirks (Galileo type-1 files, Voyager ECLIPB1950 bus supers vs J2000 platform CKs, NH type-2 segments); pins hard to exact original kernel versions; per-image provenance is hard to express. Highest effort, highest risk, and a misleading product.

### 6. No kernels: runtime corrected-pointing service

Keep the JSON/stats DB authoritative; a library/CLI applies `OffsetFOV` (or an in-memory `oops.frame.Navigation`) on demand, optionally emitting throwaway per-run CKs.

*Pros:* zero kernel management; always consistent with the latest results.

*Problems:* fails the actual requirement -- SPICE users outside SpinDoctor get nothing, and there is no shippable product. Listed to mark the boundary of the design space (and because pieces of it -- ephemeral CK emission -- are useful for Design 1's round-trip validation harness).

## 5. Ranking summary and recommendation

| Rank | Design | One line |
|---|---|---|
| 1 | Overlay type-3, exposure-window segments | Honest, standard, small, incremental; needs BOTSIM + loading policy |
| 2 | Discrete type-1 midtime records | Great milestone, weak product (tolerance semantics, frozen attitude) |
| 3 | New corrected camera frames | Physically cleanest; adoption cost kills it for now |
| 4 | One file per image + meta-kernels | Collapses into Design 1 once consolidation is forced |
| 5 | Rewrite originals (csmithing classic) | Fabricated coverage, huge, fragile; do not repeat |
| 6 | Runtime service, no kernels | Doesn't meet the requirement |

**Decision (2026-07-31): Design 1 is adopted.** Staging: (a) extend `_metadata.json` with start/stop ET, SCLK, and frame identity now, without waiting for the C-matrix change -- the same fields carry over when the C-matrix later replaces the offset (cross-cut #4); (b) build the generator on `cspyce` alone -- no oops import -- with the offset-based input adapter first, the C-matrix adapter when the pipeline change lands, and the round-trip test as the acceptance gate for both; (c) ship the Design 2 midtime writer as the first validated milestone, then widen records to exposure windows and switch to type 3; (d) land the loading story (SPICE.db registration + per-mission furnish policy) and the CSV report in the same release as the first kernels, since kernels nobody can load correctly are worse than none.

## 6. Decision record and remaining open items

Decided (2026-07-31):

- **Design 1 adopted** (overlay type-3 CK, exposure-window segments, mirrored file ranges, cspyce-only writer, dual-era input adapter).
- **Inclusion gate: `success` and `conflicted`, all confidence levels** -- a segment is written whenever a status-eligible image has an offset (or, later, a C-matrix). No confidence or rank threshold. Consequence: the CSV report and segment comments must carry status, confidence, and rank, since they are now the only way a consumer can filter out low-confidence or conflicted pointing.
- **BOTSIM: skip the loser.** One correction per pair (NAC preferred, per the old tool's rule); the loser's window falls through to original pointing.
- **Failed/unnavigated images: fall through to the originals.** No uncorrected-copy segments; the copy-unnavigated mode is dropped.
- **Angular velocity: included** (`avflag = 1`), with the original AV rotated through the correction delta.
- **Legacy metadata: ignored.** Current schema only; results will be regenerated.

Still open:

- Naming convention for the mirrored files (suffix on the original name), and the Voyager pairing rule (SEDR encounter kernels vs the bus supers).
- CSV report: exact column set (name, UTC/ET, SCLK, offset, sigma, confidence, rank, status, source `.bc`?) and whether one CSV per mission or per kernel file.
- Loading story per mission (SPICE.db name/priority vs oops host changes) -- affects kernel naming, so decide early.
