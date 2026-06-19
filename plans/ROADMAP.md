# RMS-NAV Roadmap to a Production Pipeline

This document collates the open GitHub issues
([SETI/rms-nav](https://github.com/SETI/rms-nav/issues)) into an ordered plan.

**Strategy: one instrument, end to end, first.** Rather than build every
capability across all four missions at once, the target is a *complete, working,
cloud-capable pipeline for Cassini ISS* -- navigation -> reprojection -> backplanes
-> PDS4 bundles -> summary/preview images + metadata -> updated SPICE CK kernels,
with the docs and tests an end user needs. Once that vertical slice ships and is
calibrated, we add the other instruments (Voyager ISS, Galileo SSI, New Horizons
LORRI) and the remaining features on top of a proven spine.

The eventual goal is unchanged: full production processing of **Cassini, Voyager,
Galileo, and New Horizons** into quality bundles (backplanes, metadata, summary
images, preview images) plus a full set of new **SPICE CK kernels with updated
pointing**.

**Scope filter.** Lists issues at priority Critical / Essential / Important /
Useful that lie on the path. **Priority Defer** and most **Priority 5 Minor**
items are excluded and listed at the end. Issue links use the form `#NNN`.

---

## Phase 0 -- Foundation (unblocks everything)

| Issue | Title | Why first |
|---|---|---|
| [#189](https://github.com/SETI/rms-nav/issues/189) | Rename the package to **SpinDoctor** | Pervasive rename; do it before broad new work to minimize churn. Coordinate with #95. |
| [#95](https://github.com/SETI/rms-nav/issues/95) | Packaging/typing gap: missing `py.typed`, package-name collisions | **Priority 1 Critical.** Blocks clean installs / typed downstream use. |
| [#176](https://github.com/SETI/rms-nav/issues/176) | Move all tuning constants into config YAMLs (placeholders first), then tune | Calibration writes into config; the files must exist first. |
| [#118](https://github.com/SETI/rms-nav/issues/118) | Comprehensive config validation system | Guards the new config surface against silent misconfiguration. |

---

## Phase 1 -- Cassini ISS, end to end (including cloud)

The deliverable: an end user can take Cassini ISS images and produce calibrated
navigation, reprojections, backplanes, and PDS4 bundles (with summary/preview
images, metadata, and updated CK kernels), running locally or in the cloud.

Most of the machinery below is mission-agnostic -- the point of Phase 1 is to make
it provably *work, end to end, for Cassini*. Sub-streams 1A->1C are sequential
(navigation -> calibration -> **accuracy checkpoint**); only once the accuracy
checkpoint (1C) shows the navigation is sound do we invest in the downstream
products (1D-1H). 1D-1J can then largely proceed in parallel.

### 1A. Navigation correctness (Cassini scenes: Saturn, rings, icy moons, stars)

| Issue | Title | Pri |
|---|---|---|
| [#123](https://github.com/SETI/rms-nav/issues/123) | Mahalanobis agreement grouping breaks (CRLB-tight covariances) | Essential |
| [#86](https://github.com/SETI/rms-nav/issues/86) | Fix ring models (Saturn) | Essential |
| [#126](https://github.com/SETI/rms-nav/issues/126) | BodyDiscCorrelateNav rotation pyramid is ~10 min on 1024x1024 | Essential |
| [#124](https://github.com/SETI/rms-nav/issues/124) | Ensemble has no cross-technique outlier rejection | Important |
| [#125](https://github.com/SETI/rms-nav/issues/125) | BodyTerminatorNav mis-convergence has no per-technique signal | Important |
| [#128](https://github.com/SETI/rms-nav/issues/128) | Architectural redesign: robust limb navigation across body types | Important |
| [#179](https://github.com/SETI/rms-nav/issues/179) | Make the DT coarse-prior search robust against competing edges | Important |
| [#145](https://github.com/SETI/rms-nav/issues/145) | Star-ring occlusion mis-classifies stars near ringlet edges/gaps | Important |
| [#25](https://github.com/SETI/rms-nav/issues/25) | Implement blurring for high-resolution bodies | Important |
| [#150](https://github.com/SETI/rms-nav/issues/150) | BodyLimbNav floor is a model-vs-image edge offset | -- |
| [#146](https://github.com/SETI/rms-nav/issues/146) | `instances_for_obs` ignores a per-run config override | Useful |
| [#130](https://github.com/SETI/rms-nav/issues/130) | Calibrate per-instrument star limiting magnitudes *(Cassini first)* | Useful |
| [#136](https://github.com/SETI/rms-nav/issues/136) | `--last-image-num` can drop later WAC images *(Cassini ingest)* | Important |
| [#12](https://github.com/SETI/rms-nav/issues/12) | Update DataSetPDS3 to properly use label filespecs | Useful |

### 1B. Calibration and the test library ("calibrated"), Cassini-scoped

Strictly ordered; the library and calibration start Cassini-only and grow with
later phases.

1. [#172](https://github.com/SETI/rms-nav/issues/172) -- Build the curated test library with ground-truth offsets (seed with Cassini scenes). *(Playbook: `plans/PHASE10_CURATION.md`.)*
2. [#175](https://github.com/SETI/rms-nav/issues/175) -- Source body ellipsoid from SPICE/oops; populate per-body albedo config (Saturn system first).
3. [#173](https://github.com/SETI/rms-nav/issues/173) -- Calibrate confidence-formula alpha coefficients against the library. *(depends on #172, #176)*
4. [#174](https://github.com/SETI/rms-nav/issues/174) -- Autonomous-nav integration tests + per-image regression baselines.

### 1C. Navigation statistics & accuracy checkpoint

Run this **as soon as basic navigation is producing metadata, before investing in
the downstream products (1D-1H)**, so pipeline accuracy is measured early and
problems are fixed cheaply rather than after building on top of them.

| Issue | Title | Pri |
|---|---|---|
| [#35](https://github.com/SETI/rms-nav/issues/35) | Navigation statistics system: ingest metadata to SQLite + report with stats and charts | Important |

> #35 ingests the per-image metadata into SQLite and renders a deterministic
> text+figure report (success/failure + reasons, technique/model usage, V/U
> offset stats, body/ring usage, cross-technique agreement, and how well the
> confidence levels predict accuracy -- a direct QA check on the #173
> calibration). It runs for any partial/full day and any instrument, so it keeps
> serving every later phase. Cloud-sourced metadata aligns with #108; an early
> local run does not need to wait for that.

### 1D. Reprojection

| Issue | Title |
|---|---|
| [#134](https://github.com/SETI/rms-nav/issues/134) | RingMosaic reprojection mutates oops precision process-globally (concurrency hazard) |

### 1E. Backplane generation

| Issue | Title | Pri |
|---|---|---|
| [#28](https://github.com/SETI/rms-nav/issues/28) | Implement the backplane generator (parent) | TBD |
| [#55](https://github.com/SETI/rms-nav/issues/55) | Determine final set of backplanes to include | Useful |
| [#63](https://github.com/SETI/rms-nav/issues/63) | `create_body_backplanes` only handles bodies near planets | TBD |
| [#54](https://github.com/SETI/rms-nav/issues/54) | Implement backplane cropping | Useful |
| [#57](https://github.com/SETI/rms-nav/issues/57) | Figure out what to put in the FITS backplane HDUs | Useful |
| [#77](https://github.com/SETI/rms-nav/issues/77) | Allow optional arguments for backplane creation | Useful |

### 1F. PDS4 bundle generation (Cassini bundle)

| Issue | Title | Pri |
|---|---|---|
| [#53](https://github.com/SETI/rms-nav/issues/53) | Implement PDS4 bundle generator (parent) | TBD |
| [#139](https://github.com/SETI/rms-nav/issues/139) | Global-index LID is malformed | Essential |
| [#69](https://github.com/SETI/rms-nav/issues/69) | Add FITS backplane file description to data labels | Important |
| [#79](https://github.com/SETI/rms-nav/issues/79) | Scrape PDS4 context products for targets | Important |
| [#30](https://github.com/SETI/rms-nav/issues/30) | Design the PDS4 labels for the backplane files | Useful |
| [#66](https://github.com/SETI/rms-nav/issues/66) | Add integrity-checking pass to bundle generation | Useful |
| [#71](https://github.com/SETI/rms-nav/issues/71) | Parameterize bundle name and version in labels | Useful |
| [#73](https://github.com/SETI/rms-nav/issues/73) | Handle targets in data labels | Useful |
| [#75](https://github.com/SETI/rms-nav/issues/75) | Add ring geometry class fields to data.lblx | Useful |
| [#76](https://github.com/SETI/rms-nav/issues/76) | Create labels for global index files | Useful |
| [#47](https://github.com/SETI/rms-nav/issues/47) | Include ring incidence angle in PDS4 label | Useful |

### 1G. Output products: summary / preview images + metadata

| Issue | Title |
|---|---|
| [#177](https://github.com/SETI/rms-nav/issues/177) | Add unit tests for `nav.support.summary_png` *(renderer already shipped in #131)* |
| [#185](https://github.com/SETI/rms-nav/issues/185) | Style gated-out features distinctly on the summary PNG |
| [#119](https://github.com/SETI/rms-nav/issues/119) | Move creation of PNG files out of `navigate_image_files.py` |
| [#21](https://github.com/SETI/rms-nav/issues/21) | Clean up inventory in public metadata |
| [#15](https://github.com/SETI/rms-nav/issues/15) | Overlay for overlapping models does not hide background model |

### 1H. SPICE CK kernels with updated pointing

| Issue | Title | Pri |
|---|---|---|
| [#50](https://github.com/SETI/rms-nav/issues/50) | Switch to using C Matrix (prerequisite) | Useful |
| [#188](https://github.com/SETI/rms-nav/issues/188) | Generate SPICE CK kernels with updated pointing as a delivered product | Essential |

### 1I. Cloud support and production operation

Required for Phase 1: the Cassini pipeline must run as cloud batch jobs.

| Issue | Title | Pri |
|---|---|---|
| [#108](https://github.com/SETI/rms-nav/issues/108) | Check all CLI programs for logging, cloud operation, `cloud_tasks` | Essential |
| [#67](https://github.com/SETI/rms-nav/issues/67) | Make PDS4 bundle generation fully cloud aware | Important |
| [#141](https://github.com/SETI/rms-nav/issues/141) | Dedup CLI driver preamble + cloud_tasks loop; fix dropped `extra_params` | Useful |
| [#180](https://github.com/SETI/rms-nav/issues/180) | Wire `STATUS_REASON_INFO_TEMPLATE` through every `NavResult.failed` site | Useful |
| [#181](https://github.com/SETI/rms-nav/issues/181) | Image-degradation classifier classes *(taxonomy needs design first)* | Useful |

> The statistics/report system (#35) lands earlier, at the 1C accuracy
> checkpoint; cloud-sourced metadata for it aligns with #108 here.

### 1J. Documentation and tests (Cassini)

| Issue | Title |
|---|---|
| [#93](https://github.com/SETI/rms-nav/issues/93) | Fill in instrument-specific user-guide appendices *(Cassini/COISS portion; currently stubs)* |
| [#178](https://github.com/SETI/rms-nav/issues/178) | Write missing dev-guide pages: filters, uncertainty, troubleshooting |
| [#94](https://github.com/SETI/rms-nav/issues/94) | Fill in developer-guide navigation-model pages |
| [#70](https://github.com/SETI/rms-nav/issues/70) | Describe the supplemental-metadata file format in the User Guide |
| [#122](https://github.com/SETI/rms-nav/issues/122) | Verify albedo / terminator-sharpness rationale in body-terminator docs |
| [#129](https://github.com/SETI/rms-nav/issues/129) | Reach zero Sphinx nitpicky warnings and gate in CI |

**Phase 1 exit criteria:** an end user can process a real Cassini ISS data set
end to end -- locally and in the cloud -- and get calibrated offsets, backplanes,
a valid PDS4 bundle (with summary + preview images and metadata), and a CK kernel
carrying the updated pointing, with passing integration tests and user docs that
explain how.

---

## Phase 2 -- Add the remaining instruments

With the Cassini spine proven, extend ingest, navigation, and per-instrument
calibration to the other three missions. The downstream machinery (reproj,
backplanes, bundles, CK) should already generalize; the work here is
instrument-specific.

### 2A. Voyager ISS

| Issue | Title |
|---|---|
| [#19](https://github.com/SETI/rms-nav/issues/19) | VGISS star navigation doesn't work |

### 2B. Galileo SSI

| Issue | Title |
|---|---|
| [#18](https://github.com/SETI/rms-nav/issues/18) | GOSSI star navigation doesn't work |
| [#17](https://github.com/SETI/rms-nav/issues/17) | GOSSI does not handle REDO properly |

### 2C. New Horizons LORRI

| Issue | Title |
|---|---|
| [#2](https://github.com/SETI/rms-nav/issues/2) | Research and calibrate New Horizons LORRI PSF sigma |
| [#138](https://github.com/SETI/rms-nav/issues/138) | LORRI accepts both `_sci` and `_eng`; confirm ENG handling |

### 2D. Outer-planet ring models (Voyager / Galileo systems)

| Issue | Title |
|---|---|
| [#82](https://github.com/SETI/rms-nav/issues/82) | Implement ring models for Jupiter |
| [#81](https://github.com/SETI/rms-nav/issues/81) | Implement ring models for Uranus |
| [#83](https://github.com/SETI/rms-nav/issues/83) | Implement ring models for Neptune |

### 2E. Cross-instrument calibration

- Extend [#173](https://github.com/SETI/rms-nav/issues/173) with per-instrument
  alpha vectors and a per-mission residual audit once each instrument's library
  images are in (the calibration design already anticipates this).
- Extend [#130](https://github.com/SETI/rms-nav/issues/130) (limiting magnitudes)
  and [#93](https://github.com/SETI/rms-nav/issues/93) (the remaining
  instrument appendices) to the other three missions.

---

## Phase 3 -- Additional features and enhancements

Schedule after the multi-instrument pipeline is solid.

| Issue | Title |
|---|---|
| [#27](https://github.com/SETI/rms-nav/issues/27) | Implement BOTSIM navigation (Cassini NAC/WAC simultaneous) |
| [#22](https://github.com/SETI/rms-nav/issues/22) | Implement star streaks |
| [#107](https://github.com/SETI/rms-nav/issues/107) | Repo with a backplane reader / example programs |
| [#34](https://github.com/SETI/rms-nav/issues/34) | Support the PDS4 version of Cassini ISS (when archive is available) |
| [#84](https://github.com/SETI/rms-nav/issues/84) | Fix simulated ring edges and gaps |
| [#40](https://github.com/SETI/rms-nav/issues/40) | Add features to simulated images |

---

## Hardening and cleanup (parallel track, any phase)

Quality work that improves robustness; can proceed alongside the phases above.

| Issue | Title |
|---|---|
| [#65](https://github.com/SETI/rms-nav/issues/65) | Harden code and implement new exception class |
| [#104](https://github.com/SETI/rms-nav/issues/104) | Replace broad `except Exception` control-flow |
| [#103](https://github.com/SETI/rms-nav/issues/103) | Guard/document thread-unsafe module-level caches |
| [#98](https://github.com/SETI/rms-nav/issues/98) | Consolidate parallel instrument registries |
| [#97](https://github.com/SETI/rms-nav/issues/97) | Split oversized modules exceeding the 1000-line limit |
| [#96](https://github.com/SETI/rms-nav/issues/96) | Prune dead code (flux.py, commented blocks) |
| [#135](https://github.com/SETI/rms-nav/issues/135) | Dedup the five `from_file` extfov-margin blocks |
| [#143](https://github.com/SETI/rms-nav/issues/143) | nav_backplane_viewer cursor read-out wrong after pan at zoom != 1 |
| [#109](https://github.com/SETI/rms-nav/issues/109) | Shared helpers for safe paths under a root |
| [#110](https://github.com/SETI/rms-nav/issues/110) | Shared scalar validation helpers in nav.support |
| [#100](https://github.com/SETI/rms-nav/issues/100) | Collapse three root-path getters in config_helper.py |
| [#101](https://github.com/SETI/rms-nav/issues/101) | Replace print()/sys.exit with ArgumentParser.error |
| [#102](https://github.com/SETI/rms-nav/issues/102) | Eliminate module-level mutable globals in CLI drivers |
| [#99](https://github.com/SETI/rms-nav/issues/99) | Wire up or delete orphan report_profile.py |
| [#39](https://github.com/SETI/rms-nav/issues/39) | Improve AttrDict to allow missing attributes |
| [#92](https://github.com/SETI/rms-nav/issues/92) | Break up requirements into optional dependency groups |
| [#24](https://github.com/SETI/rms-nav/issues/24) | Remove fuzzy and non-spherical bodies from navigation *(needs rescoping to the current architecture)* |

---

## Excluded: deferred, distant-future, and minor (off the critical path)

Tracked but intentionally **not** scheduled toward the production goal:

- **Deferred / distant-future:** #23 (body shape models -- genuinely waits on
  oops gaining non-ellipsoidal/DSK support, which is not expected; the sim's
  polyhedral shapes exist only to measure how much navigation degrades on
  non-ellipsoidal bodies, not as a navigation route), #33 (NHLORRI SPICE
  kernel), #151, #152, #153 (sim calibration layers), #184 (CartographicNav),
  #187 (Hyperion chaotic-rotator pose, depends on #23), #60 (Titan navigation).
- **Minor (Priority 5):** #13, #38, #43, #72, #74, #105, #132, #133, #137, #140,
  #142, #144, #147, #155, #157, #158, #182 (stop-after-features inspection),
  #183 (polarity-aware RingEdgeNav).
- **Simulator follow-ups:** #78 (CraterMaker), plus the sim items above.
- **Closed during triage as out of date** (referenced the removed
  `NavTechniqueCorrelateAll`): #20, #87, #88.

---

*Generated 2026-06-19 from the open SETI/rms-nav issue set; structured around a
Cassini-first end-to-end pipeline. Pre-rewrite `B-NavTechniqueCorrelateAll`
issues were triaged (#20, #87, #88 closed; #24 left open for rescoping; #86
remains valid). The SPICE CK-generation gap was filed as #188.*
