# Issue cross-reference for CODE_CRITIQUE.md

Cross-references the open GitHub issues (SETI/rms-nav) against the findings in
`critiques/CODE_CRITIQUE.md`. Each matched finding carries an inline
`> **Tracked by:** #NN` blockquote immediately after its heading in the critique.
Matching was deliberately conservative: an issue is listed only when it clearly
describes the same problem (same bug/cleanup/topic in the same area) as the
finding.

Scope notes:
- 80 open issues and ~190 findings were reviewed in full.
- Several findings are **already remediated** (see the critique's "## Remediation
  status" table) and were intentionally NOT annotated: CODE-NAV-001, CODE-NAV-002
  (both FIXED), CODE-NAV-003 / CODE-CFG-001-as-status (REDIRECTED), the Part-3
  CODE-BACKPLANE-001 + Part-3 CODE-PDS4-001 status-literal pair (FIXED), and the
  Part-7 CODE-MAIN-001 async entry point (FIXED, but its tracker #108 is still
  recorded for the finding).
- ID collisions are handled per-occurrence: CODE-MAIN-001 labels both a Part-1
  module-size finding (tracked by #97) and the Part-7 async-entry-point finding
  (tracked by #108); CODE-PDS4-001 labels both a Part-1 bare-except cluster
  (tracked by #104) and the Part-3 status-literal finding (FIXED, untracked here).

## Issue -> finding mapping (matched open issues)

| Issue | Title | Matched finding IDs |
|---|---|---|
| #13 | Clean up handing of SCET strings | CODE-OBS-008 |
| #17 | GOSSI does not handle REDO properly | CODE-DS-015 |
| #60 | Implement Titan navigation | CODE-MODEL-001 |
| #84 | Fix simulated ring edges and gaps | CODE-SIM-4 |
| #96 | Prune dead code (flux.py, correlate_old.py, commented blocks) | CODE-SUP-002, CODE-SUPPORT-001, CODE-SUPPORT-018, CODE-DS-009 |
| #97 | Split oversized modules exceeding the 1000-line limit | CODE-NAV-020, CODE-NAV-MODEL-005, CODE-REPROJ-003, CODE-SUP-002, CODE-SUPPORT-001, CODE-DATASET-001, CODE-MAIN-001 (Part 1, sim module size), CODE-STYLE-002, CODE-MODEL-010 |
| #98 | Consolidate parallel instrument registries | CODE-DS-024 |
| #99 | Wire up or delete orphan report_profile.py | CODE-UTIL-1 |
| #104 | Replace broad `except Exception` control-flow | CODE-ORCH-002, CODE-NAV-MODEL-004, CODE-SUP-001, CODE-PDS4-001 (Part 1, collections.py), CODE-OBS-003, CODE-BACKPLANE-004, CODE-BACKPLANE-005, CODE-EXP-1, CODE-SUPPORT-009, CODE-MAIN-003 |
| #105 | Replace pervasive Any / dict[str, Any] with TypedDicts/Protocols | CODE-OBS-002, CODE-REPROJ-006, CODE-SUPPORT-008, CODE-TECH-004, CODE-TECH-010 |
| #108 | Check CLI programs for logging/cloud and that cloud_tasks works | CODE-MAIN-001 (Part 7, async entry point), CODE-MAIN-004 |
| #118 | Comprehensive config validation system | CODE-CFG-001, CODE-CFG-002, CODE-NAV-MODEL-002, CODE-ORCH-005, CODE-MODEL-008, CODE-TECH-001 |
| #123 | Mahalanobis grouping breaks (CRLB-tight covariances) | CODE-ORCH-001, CODE-NAV-004, CODE-NAV-005, CODE-NAV-010 |
| #125 | BodyTerminatorNav mis-convergence has no per-technique signal | CODE-NAV-006 |
| #128 | Architectural redesign: robust limb navigation | CODE-NAV-006, CODE-NAV-007 |

Total findings annotated as tracked: **44** distinct finding occurrences
(some cite two issues; #104 and #97 each cover the largest sets).

## Untracked findings (candidates for new issues)

These findings have no clearly-matching open issue. Grouped by subsystem, leading
with the highest severities. (Critical/High are named explicitly; the long Medium/
Low tail is summarized.)

### Critical / High — untracked

- **CODE-NAV-008 (High)** — ensemble `_combine_precision_weighted` averages an
  angle (rotation_rad) with no wrap handling. (Related to ensemble issues #123/#124
  but not the same problem; #124 is outlier rejection, #123 is covariance scale.)
- **CODE-NAV-009 (High)** — `_mahalanobis_distance` null-space test uses a fixed
  absolute tolerance `1e-6` against an un-normalized residual.
- **CODE-ORCH-003 (High)** — NaN missing-data markers crash `navigate()` for
  `calibrated_if` images (uncaught `ValueError`, violates no-raise contract;
  classifier missing/blank detection dead for calibrated images).
- **CODE-CFG-1 (High)** — `update_config` shallow (depth-1) merge; nested user
  overrides clobber sibling defaults.
- **CODE-MAIN-002 (High)** — simulated-image GUI drops `shade_solid_rings` on load
  and can crash on `closest_planet=None` via `QComboBox.findText(None)`.
- **CODE-OBS-001 (High)** — `ObsInst.star_psf_size` loop-variable-leak default with
  a mistyped/unchecked return; `UnboundLocalError` on empty config.
- **CODE-OBS-011 (High)** — Voyager spacecraft / I-over-F correction keyed off an
  unvalidated single label character `LAB02[4]` (drives a 3.345x pixel rescale).
- **CODE-DS-001 (High)** — `vol_start_idx`/`vol_end_idx` possibly-unbound under
  mypy strict in `_yield_image_files_index`.
- **CODE-DS-002 (High)** — `choose_random_images` is biased and can livelock /
  under-yield under active filters.
- **CODE-DS-010 (High)** — Cassini BOTSIM grouping mis-pairs and silently drops
  unpaired frames; time-slop check misuses the image number as seconds.
  (Issue #27 is "implement BOTSIM navigation," a different concern — this is a
  grouping bug.)

### Medium — untracked (by subsystem, summarized)

- **nav_technique / core math:** CODE-NAV-011 (Tikhonov/covariance inconsistency),
  CODE-NAV-012 (brightness_margin SNR-vs-flux), CODE-NAV-013 (`_combine_confidence`
  trace mixing), CODE-NAV-014 (rotation-variance anisotropy), CODE-NAV-015
  (polarity_filter clamps OOB vertices), CODE-NAV-016 (`_build_polyline_mask`
  duplicated x3 — partly overlaps CODE-TECH-005), CODE-NAV-018 (greedy inlier
  matching).
- **nav_model:** CODE-NAV-MODEL-001 (hard-coded photometric `(softness*0.5)^2`),
  CODE-NAV-MODEL-003 (`visible_lit_fraction` denominator naming/phase coupling),
  CODE-MODEL-007 (ring occlusion uses `bp_radii.median()` over a masked window).
- **ensemble / orchestrator:** CODE-ORCH-003 (sigma-gate reason logging),
  CODE-ORCH-004 (body names parsed from feature_ids by string prefix; partly
  overlaps CODE-ORCH-009), CODE-ORCH-004 (Saturation/classifier stats on DC-removed
  image — Part 6), CODE-ORCH-006 (provenance git/hash re-run per image).
- **reproj / backplanes / pds4:** CODE-PDS4-002 (malformed LID in global index .tab),
  CODE-BACKPLANE-002 (stats units disagree with FITS array units),
  CODE-BACKPLANE-003 (occlusion uses body center distance not per-pixel),
  CODE-REPROJ-001 (global ring antimask assumes grid-aligned longitude_start),
  CODE-REPROJ-001/002 (reproj stdlib `logging` — consistency only).
- **obs / dataset:** CODE-OBS-004 (`_ra_dec_limits` dec wrap dead/incorrect),
  CODE-OBS-007 (dead WAC branch), CODE-OBS-012 (Voyager LABEL3 brittle parse),
  CODE-OBS-013 (mixed `%`/f-string logging), CODE-OBS-015 (GOSSI/NHLORRI
  uncalibrated data reported as I/F — partly relates to #2 PSF but distinct),
  CODE-OBS-017 (`extfov_margin_vu[shape]` KeyError on non-standard heights),
  CODE-DS-003 (`done` early-exit assumes monotonic image numbers),
  CODE-DS-004 (CSV/file-list use stdlib `open`, no URL support),
  CODE-DS-005 (CSV ragged-row IndexError), CODE-DS-006 (loop-variable shadowing),
  CODE-DS-011 (`pds4_bundle_path_for_image` returns '' footgun),
  CODE-DS-012 (`_get_img_name_from_label_filespec` strips sub-frame suffix),
  CODE-DS-017 (VGISS `_img_name_valid` rejects product filenames),
  CODE-DS-019 (NHLORRI `_eng`/`_sci` mixed), CODE-DS-021 (ImageFile path caches
  not thread-safe).
- **support:** CODE-SUPPORT-002 (`clean_obj` mutates caller in place),
  CODE-SUPPORT-004 (shift/pad/unpad aliasing inconsistency),
  CODE-SUPPORT-005 (null-sigma short-circuit wrong for GRADIENT/MORPH),
  CODE-SUPPORT-012 (`evaluate_candidate` crop_center raises on small model).
- **sim:** CODE-SIM-1 (shared crater seed), CODE-SIM-2 (craters disable limb AA),
  CODE-SIM-3 (two divergent illumination conventions).
- **technique coverage:** CODE-TECH-007 (vertex count compared against
  `min_arc_px` pixel-length threshold).

### Low — untracked (counts by subsystem)

A large Low-severity tail is untracked. Representative groups:
- nav math/Low: CODE-NAV-019, CODE-NAV-021, CODE-NAV-022; CODE-NAV-MODEL-006.
- ensemble/orch Low: CODE-ORCH-007..014, CODE-DERIV-001, CODE-DERIV-002.
- reproj/pds4 Low: CODE-REPROJ-003..006, CODE-PDS4-003, CODE-PDS4-004.
- feature/anno/sim/config Low: CODE-CFG-2, CODE-CFG-3, CODE-CFG-4, CODE-SIM-5..9,
  CODE-FEAT-1, CODE-FEAT-2, CODE-ANNO-1, CODE-ANNO-2, CODE-ANNO-3, CODE-EXP-2.
- support Low: CODE-SUPPORT-003, 006, 007, 010, 011, 013, 014, 015, 016, 017.
- obs/dataset Low: CODE-OBS-005, 006, 009, 010, 014, 016, 018, 019, 020;
  CODE-DS-007, 008, 013, 014, 016, 018, 020, 022, 023, 025, 026, 027;
  CODE-STYLE-001, CODE-EXP-001.
- CLI/UI Low: CODE-MAIN-005, 006, 007; CODE-UI-001..009.
- model/technique Low: CODE-MODEL-002, 003, 004, 005, 006, 009;
  CODE-TECH-002, 003, 005, 006, 008, 009, 011.

These are candidates for batch cleanup issues (e.g. a "stdlib logging ->
pdslogger consistency" sweep for CODE-REPROJ-002 / CODE-MAIN-002 / CODE-UI-006 /
CODE-CFG-4, a "narrow/aliasing helpers" sweep, a "duplicated technique helpers"
sweep for CODE-NAV-016 / CODE-TECH-005, etc.).

## Open issues that matched no finding (out of critique scope)

These open issues are forward-looking feature work, infrastructure, or domain
calibration not surfaced as a code-critique finding:

- #2 — Research and calibrate New Horizons LORRI PSF sigma value
- #12 — Update DataSetPDS3 to properly use label filespecs
- #15 — Overlay for overlapping models does not hide background model
- #18 — GOSSI star navigation doesn't work
- #19 — VGISS star navigation doesn't work
- #20 — Determine confidence of PSF fit in stellar refinement
- #21 — Clean up inventory in public metadata
- #22 — Implement star streaks
- #23 — Support body shape models
- #24 — Remove fuzzy and non-spherical bodies from navigation
- #25 — Implement blurring for high-resolution bodies
- #27 — Implement BOTSIM navigation
- #28 — Implement the backplane generator (parent)
- #30 — Design the PDS4 labels for the backplane files
- #33 — Create a new SPICE instrument kernel for NHLORRI
- #34 — Support the PDS4 version of Cassini ISS
- #35 — CLI program to summarize/statistically analyze offset results
- #38 — Config options to determine filecache locations
- #39 — Improve AttrDict to allow missing attributes
- #40 — Add features to simulated images
- #43 — Move `--pds3-holdings-root` argument to DataSetPDS3
- #47 — Include ring incidence angle in PDS4 label
- #50 — Switch to using C Matrix
- #53 — Implement PDS4 bundle generator (parent)
- #54 — Implement backplane cropping
- #55 — Determine final set of backplanes to include
- #56 — Explore masking in the new correlation code
- #57 — Figure out what to put in the FITS backplane HDUs
- #63 — `create_body_backplanes` only handles bodies near planets
- #65 — Harden code and implement new exception class (general; subsumed by #104/#118)
- #66 — Add integrity-checking pass to PDS4 bundle generation
- #67 — Make PDS4 bundle generation fully cloud aware
- #69 — Add FITS backplane file description to data labels
- #70 — Describe supplemental metadata file format in User Guide
- #71 — Parameterize bundle name/version in PDS4 labels
- #72 — Create PDS4 collection_context CSV/LBLX files
- #73 — Handle targets in data labels
- #74 — Create collection_document.csv and lblx files
- #75 — Add ring geometry class fields to data.lblx
- #76 — Create labels for global index files
- #77 — Allow optional arguments for backplane creation
- #78 — Add fancier craters using CraterMaker
- #79 — Scrape PDS4 context products for targets
- #81 — Implement ring models for Uranus
- #82 — Implement ring models for Jupiter
- #83 — Implement ring models for Neptune
- #86 — Fix ring models (model edges too close to features)
- #87 — Correlate on only some ring models
- #88 — Why N1532373096 doesn't navigate well
- #92 — Break up requirements into optional dependency groups
- #93 — Fill in instrument-specific user guide appendices (docs)
- #94 — Fill in developer guide navigation model pages (docs)
- #95 — packaging/typing gap (py.typed marker, package name collisions)
- #100 — Collapse three root-path getters in config_helper.py
- #101 — Replace print()/sys.exit error pattern with ArgumentParser.error
- #102 — Eliminate module-level mutable globals in CLI drivers
- #103 — Guard/document thread-unsafe module-level caches (nav_model_stars, misc, image)
- #107 — Repo with a backplane reader / example programs
- #109 — Shared helpers for safe paths under a root
- #110 — Shared scalar validation helpers in nav.support
- #119 — Move PNG creation out of navigate_image_files.py
- #122 — Verify albedo/terminator-sharpness rationale in docs (docs)
- #124 — Ensemble cross-technique outlier rejection (RANSAC)
- #126 — BodyDiscCorrelateNav rotation pyramid runtime
- #129 — Reach zero Sphinx nitpicky warnings (docs/CI)

Notable near-misses left unmatched (conservative):
- #100/#101/#102 (config_helper getters, print/sys.exit, CLI globals) — the
  critique's CLI findings (CODE-MAIN-005/006/007) describe *different* specific
  duplication/structure problems, not these exact patterns.
- #103 (thread-unsafe caches in nav_model_stars/misc/image) — the closest
  critique finding, CODE-DS-021, is about `ImageFile` path caches in a different
  module; not annotated to avoid a weak cross-module match.
- #95 (py.typed / package collisions) — no critique finding covers packaging.
- #124 (RANSAC outlier rejection) and #126 (disc-pyramid runtime) — no finding
  describes these specific algorithm/performance problems.
