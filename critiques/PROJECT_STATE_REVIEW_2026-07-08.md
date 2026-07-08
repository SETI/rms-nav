# RMS-NAV Project State Review

**Date:** 2026-07-08
**Baseline:** `main` @ 77a90f4 ("Navigation core rewrite (Phases 0-10)"), working tree clean.
**Scope:** A meta-review of the complete software state: (1) accuracy of every document in
`critiques/`, (2) completeness of every document in `plans/`, (3) documentation accuracy,
and (4) an overall verdict on design and implementation quality, grounded in freshly run
quality gates and independent verification of a large sample of prior findings against the
current source.

**Method.** All quality gates were re-run on this date. Every critique and plan document was
read and its checkable claims verified against the source tree by seven independent
verification passes (code-critique sampling with math re-derivation, test-suite audit,
scientist-review verification, plan traceability analysis, AUTONAV/SIM deliverable
spot-checks, documentation audit, and a direct run of the test/lint/type gates). Sampled
claims were confirmed, refuted, or marked stale with file evidence; nothing below is taken
from the reviewed documents on faith.

---

## 1. Measured quality gates (this date, this machine)

| Gate | Result |
|---|---|
| `ruff check src tests` | Clean |
| `ruff format --check src tests` | Clean (328 files) |
| `mypy src tests` (strict) | Clean (329 source files) |
| `pytest -n auto --dist=loadfile` (default, no holdings env) | 34 failed, 1672 passed, 6 skipped, 11 errors |
| Same failing files re-run with holdings env vars set | 73/73 passed |

The default-suite failures are entirely environmental: `tests/nav/obs/test_obs_snapshot.py`,
`tests/nav/dataset/test_dataset_pds3_*.py`, and `tests/nav/inst/test_inst_*.py` reach
`OOPS_RESOURCES` / PDS3 holdings URLs but carry **no `integration` marker**, so a plain
`pytest` on a machine without the env vars fails 45 tests. With the env vars set (as CI sets
them), the full suite passes. This is a real hygiene defect (see finding H-1) but not a code
regression: the suite is green under its intended environment.

**Repository size:** 185 source files / 64,283 lines under `src/`; 155 test files /
33,502 lines / 1,591 test functions under `tests/`.

---

## 2. Are the critiques accurate?

Overall: **yes, to an unusual degree.** The critique corpus is factually reliable, and the
project has demonstrably acted on it. The main caveat is staleness: all six documents are
dated 2026-06-19 and were merged in the same commit as the code they review, so several
claims describe pre-merge states that the merge itself remediated.

### 2.1 CODE_CRITIQUE.md (190 findings) — ACCURATE (verified 37/38 sampled)

All 6 Critical findings, 18 of 22 High findings, and 14 Medium/Low findings were re-verified
against HEAD, including re-derivation of the mathematical claims (weighted-mean variance
powers, Tukey-weighted-RMS blindness to outliers, the dimensional analysis of the rotation
sigma, angle wrap, the zero-RMS degenerate path). Results:

- **37 of 38 sampled findings fully accurate.** Zero refuted, zero unverifiable. One status
  note is stale: Part-1 CODE-PDS4-001 (five bare excepts in `collections.py`) is marked
  "remains open" but is ~80% resolved at HEAD (one remains, at `collections.py:166`, and it
  now logs via `logger.exception`).
- **The document is a remediation ledger, not a description of HEAD.** Roughly 72% of
  findings carry inline FIXED/IGNORED/DEFERRED status, and every sampled "FIXED" claim
  verifiably landed (e.g. degenerate LM path at `dt_fitting.py:1372-1389` with its
  regression test at `test_dt_fitting.py:525`; deep config merge at `config.py:43`;
  structured `source_bodies` at `ensemble.py:179-189`). Every sampled "still open" claim
  verifiably still exists (e.g. the four bare `except Exception` sandboxes at
  `orchestrator.py:535,624,647,741`; the malformed LID at `collections.py:182`; placeholder
  coefficients in `config_400_inst_coiss.yaml`).
- **Severity calibration is sound.** The six Criticals are genuine accept-wrong-answer or
  whole-stage-inert defects. Minor quibbles: CODE-NAV-008 (rotation wrap) and CODE-OBS-003 /
  CODE-SUP-001 lean generous at High.
- **Known weaknesses:** finding-ID collisions beyond the two acknowledged ones
  (CODE-ORCH-003, CODE-ORCH-004, CODE-OBS-001, CODE-MAIN-002, and CODE-CFG-1 vs
  CODE-CFG-001 each label two distinct findings); line-number citations for still-open
  findings have drifted 10-20 lines.

**Residual issues the verification pass found that CODE_CRITIQUE misses** (candidates for
new findings/issues):

1. **`coarse_ncc_search` overlap fraction has no minimum-support guard.** The CODE-NAV-007
   fix scores `overlap / N_inbounds` (`dt_fitting.py:264`); a shift that pushes nearly the
   whole polyline off-frame but leaves a handful of vertices on edge pixels scores 1.0 and
   beats a 450/500 match, since the small-shift tie-break applies only on exact score
   equality. Also, the fix note's claim that overlap/N is "the true binary-NCC argmax" is
   overstated (binary NCC normalizes by sqrt(N), not N).
2. **Bare `assert` enforcing the small-angle bound** at `ensemble.py:410` is stripped under
   `python -O`, and when it does fire it is swallowed by the orchestrator's bare-except
   sandboxes (CODE-ORCH-002), turning an intended contract violation into a silent
   technique drop. The two findings interact; neither document flags it.
3. **Inconsistent estimators across the ensemble's two blocks:** the circular-mean rotation
   combine weights by `info[2,2]` only, discarding rotation-translation cross-information
   that the linear information-form combine retains for translation. Harmless under the
   small-angle bound, but worth a documented rationale.

### 2.2 TESTS_CRITIQUE.md — PARTIALLY ACCURATE (core confirmed; sim claims stale)

The structural findings all hold at HEAD: the SPICE-backed `create_model` / `_render`
projection path has no unit tests (only holdings-gated integration coverage); the real
star-conflict logic (`_check_one_star`, `mark_body_and_ring_conflicts` in
`src/nav/nav_model/stars/conflicts.py`) is untested and is mocked out where it would run;
`src/backplanes/` and `src/pds4/` ship with **zero tests**; real-image regression is one
baseline deep (`tests/integration/baselines/N1597846115_2_CALIB.json`) against 11 sidecars,
with the offset check gated on `expected.status == 'success'` and one sidecar's status
pinned to acknowledged-wrong behavior; `test_ring_filter.py` is the lone caplog/stdlib-
logging violator; 23 of 224 `pytest.raises` lack `match=`.

But a significant minority of the critique is stale because the phase-10 branch added a
large simulated-image test tier after it was written: `tests/nav/sim/` (10 unit files), 29
sim baselines under `tests/integration/sim_baselines/`, and unmarked in-process sim
navigation/regression/invariant suites that run in the default suite. The critique's
cross-cutting theme "the default run protects nothing end-to-end" is substantially rebutted
for synthetic scenes; it remains true only for real images. Its coverage percentages and
the old baseline offset values are obsolete.

The critique also misses: the misleading "integration" naming of three unit-test files
(`test_nav_model_body_integration.py` etc., which are unmarked shim-based unit tests); the
untested CLI dispatcher layer in `src/main/` (only `nav_create_simulated_image` has tests);
and the unmarked-but-network-dependent tests found in section 1.

### 2.3 SCIENTIST_REVIEW.md / SCIENTIST_REVIEW_CRITICAL.md — SOUND / MOSTLY SOUND

The two reviews are a matched pair examining the same verified facts through generous and
hostile lenses; neither invents anything material. Every hard anchor checks out at HEAD:
the shared sim/nav-model renderer (`src/nav/sim/render.py` functions called by
`nav_model_body_simulated.py:306,316`), Titan's `return []`
(`nav_model_titan.py:48`), the limb `gradient_ridge_refine: 0`
(`config_510_techniques.yaml:237`), the uncalibrated-confidence admissions in
`docs/simulator_report/simulator_report.rst:36,337,848`, the 1e15/1e-9/SNR_REF constants,
the four placeholder instrument appendices, the PDS4 `NotImplementedError` walls, the CI
`-m "not integration"` exclusion (`.github/workflows/run-tests.yml:94`), and the 13-sidecar
library.

The critical review's central charge is structurally correct and remains the project's most
important open scientific liability: **the only quantitative validation pits the navigator
against images generated by its own forward model**, so the marquee sub-pixel numbers
(0.005-0.13 px) bound numerics, not real-world accuracy. The simulator report's own
"algorithmic-invariant" framing partially blunts, but does not answer, this.

However, five of the hostile review's claims are stale or overstated at HEAD and should not
be acted on as written:

1. **"No kernel/config provenance recorded" is false.** `src/nav/nav_orchestrator/provenance.py`
   plus `curator.py:169-181` pin the git SHA, the sorted loaded-SPICE-kernel list, and
   static-data YAML hashes into every per-image `_metadata.json`. (Remaining gap: resolved
   config-override text and star-catalog versions are not pinned.)
2. "Hundreds" of doc placeholders is a 2-5x exaggeration (95 hits on the broad pattern, 45
   on the review's own phrase list).
3. The mid-rewrite/CLAUDE.md-contradiction framing is obsolete: the branch is `main`, and
   CLAUDE.md now correctly states the star techniques are implemented.
4. Ring-edge `gradient_ridge_refine` is now **enabled** (`config_510_techniques.yaml:354`
   with rationale); only the limb half of the "fix shipped disabled" claim stands.
5. "Small rolls silently wrong, not flagged" overstates: the simulator report itself
   documents the spurious-zero behavior, and a rotation-unobservable covariance path exists
   (`nav_technique_star_field.py:1041-1059`). The failure mode is real; "unflagged" is not.

### 2.4 ISSUE_CROSSREF.md — INTERNALLY CONSISTENT, mildly stale

All 17 spot-checked finding IDs exist; per-issue tallies match the critique's Tracked-by
annotations; remediation exclusions are honored. It is now behind the critique it indexes:
issues #132-#139 were filed for its own "untracked candidates" after it was written
(follow-through, not contradiction). One wart: `CODE-REPROJ-001` appears twice in the
Medium-untracked list with two different descriptions (the second is almost certainly
CODE-REPROJ-002).

### 2.5 SIM_REWRITE_FINDINGS.md — SOUND; roughly half addressed

The simulator rewrite it anticipated landed in the same merge. At HEAD: SIM-2 (crater limb
anti-aliasing), SIM-5 (partially; two caches now maxsize=30, four remain maxsize=1), and
SIM-8 (comment) are addressed. Still applicable: SIM-1 (shared crater seed / shape-cache
key omits body identity, `render.py:327`), SIM-3 (crater lighting ignores `rotation_z`,
`sim_body.py:440-483`), SIM-4 (GAP/RINGLET overwrites scene, `render.py:1051,1069`, issue
#84), SIM-6 (dead public API `render_stars`/`render_bodies`), SIM-7 (bbox uses max axis for
both dimensions, `render.py:366-370`), SIM-9 (silent eccentricity clamp,
`sim_ring.py:96-97,137-138`).

---

## 3. Are the plans complete?

Overall: **the plan set is coherent, mutually consistent in substance, and unusually
disciplined — but it needs a reconciliation pass.** Two documents partition overlapping
work along different axes without referencing each other, and several premises are already
stale against the code that shipped alongside them.

### 3.1 AUTONAV_PLAN.md — complete as a design record; correctly retired

This is the design and phase plan for exactly the rewrite 77a90f4 merged. Phases 0-9 verify
against source: NavOrchestrator/ensemble/curator, the `NavFeature` algebra, all nine
autonomous techniques plus manual, shared DT machinery and image derivatives, rotation
fields, config renumbering, provenance, and the summary-PNG renderer all exist as planned
(deliverable spot-checks pass). The plan has explicitly handed the live plan of record to
`plans/ROADMAP.md` and GitHub issues.

**The commit message's "Phases 0-10" overstates.** Phase 10's *infrastructure* shipped, but
its substance is open: the library holds 13 images against a 49-image budget, there is one
real-image baseline, the confidence alphas in `config_510_techniques.yaml` are uncalibrated
defaults, and `config_220_body_shape.yaml` carries 51 PLACEHOLDER markers. Phase 11 is
pending/partially dropped. Known deviations: the Part-1/Part-8 `NavFeatureExtractor` plugin
design was superseded (extraction folded into `NavModel.to_features`) without the prose
being rewritten; `config_520/530/540` YAMLs were never created (ensemble tunables are
module constants, tracked as #176); VGISS rotation fitting shipped off ("too slow") where
the plan says on; the per-phase `phase_NN_review/` folders its Definition of Done requires
are not on `main`. The plan's method of retaining superseded prose under precedence rules
makes it safe to read only with the rules in hand.

### 3.2 SIM_IMPROVEMENT_PLAN.md — effectively complete

Status board verified against the tree: B0-B2/B4-B7, G0-G2/G4-G8, T1-T4 done with the
claimed files present; B3/G3 (smear, #151), B8 (diffraction spikes, #152), and T5-T7
(calibration validation, #153) deliberately deferred with issue numbers, and the deferred
files are correctly absent. Minor drift only (scene loader landed at `src/nav/sim/scene.py`
rather than the manifest's path; heading `[done]` markers inconsistent — section 0.1 is
authoritative). As a plan it is fully actionable and its section 0 is an accurate handoff.

### 3.3 REMEDIATION_PLAN.md — strongest plan; near-complete; needs a refresh pass

Traceability to SCIENTIST_REVIEW_CRITICAL is genuinely complete: every finding maps to a
workstream (WS-0..WS-18), and the matrix is honest about WS-18 exceeding the critique.
Dependencies are coherent (WS-3 -> WS-2 -> WS-0b -> WS-1 per-technique, with the pairwise
layer correctly exempted); acceptance criteria are mostly concrete and testable, with
deliberately soft ones justified by the no-fixed-spec premise. Defects:

- **WS-1b has no acceptance-criteria block** — the only workstream without one.
- **Stale premises** (overtaken by the merged code): WS-14 provenance is largely
  implemented (narrow the workstream to config-override hash + star-catalog versions);
  the WS-6 CLAUDE.md contradiction is already fixed; WS-10's ring-edge half is enabled;
  WS-3's "empty README" is now a populated `README.md` (though still missing the required
  curation/blessing/provenance content).
- Small errors: WS-1 cites config field `ellipsoid_residual_km`; the actual field is
  `ellipsoid_rms_residual_km` (`config_220_body_shape.yaml:24`). WS-7's "implement" branch
  is thin relative to its stated risk. Its premise sentence "the core rewrite is complete
  as of phase 10" over-credits Phase 10 (see 3.1).

### 3.4 ROADMAP.md — coherent, but unreconciled with REMEDIATION_PLAN

Internal ordering is dependency-sound (constants-to-config before calibration; library
before calibration; the 1C accuracy checkpoint before downstream investment; rotation-
pyramid performance correctly deferred past Cassini). The problem is that ROADMAP and
REMEDIATION_PLAN never reference each other and partition overlapping territory along
different axes (issues vs workstreams), with no cross-mapping. Substantive collisions:

1. **Two competing confidence-calibration methodologies:** ROADMAP 1B (#173, tier-midpoint
   calibration per PHASE10_CURATION) vs REMEDIATION WS-5 (reliability-diagram calibration
   anchored to WS-0/1/2 accuracy studies), which treats the former approach as the problem
   to fix. Which is the plan of record is genuinely ambiguous.
2. **Conflicting library-size targets:** WS-3 ">=20 per instrument, >=120 total" vs the
   49-image AUTONAV/PHASE10 budget vs no number in ROADMAP #172.
3. Nothing in ROADMAP covers REMEDIATION's core Phase 0 (the de-circularized validation
   program WS-0/1/2); its "accuracy checkpoint" (#35) is metadata statistics, not the
   agreement study.

### 3.5 PHASE10_CURATION.md — accurate, actionable, ~20% executed

Its status snapshot exactly matches the repo (11 curated sidecars in 9 of 17 scene classes;
1 baseline; no calibration sweep; alphas at hand-set defaults). All referenced
infrastructure exists. Stale details: it points to "ROADMAP.md Milestone 2" (ROADMAP has no
milestones), says `config_510_techniques.yaml` contains PLACEHOLDER strings (it no longer
does; the 51 markers live in `config_220_body_shape.yaml`), and counts 11 bodies in
config_220 where there are 10. Its self-declared deletion condition is far from met.

---

## 4. Documentation state — GOOD, materially improved since the 2026-04-28 audit

All five findings of the prior documentation audit are resolved, one by migration:
`user_guide_navigation.rst` now lists the correct nine-technique registry, but
`docs/introduction_configuration.rst:186` still tells users `correlate_all` is a valid
`--nav-techniques` value — the same user-facing harm in a different file. CLAUDE.md is the
best-maintained document in scope (every command, config section, pipeline claim, and
architecture note verified). There is **no capability overclaiming anywhere**: Titan is
honestly documented as a placeholder; accuracy claims cite their measurements.

Remaining defects, in priority order:

1. `docs/introduction_configuration.rst:186` — invalid `correlate_all`/`manual`
   `--nav-techniques` advice (actively misleading).
2. `docs/introduction_configuration.rst:16-36,213-229` — config layering described wrongly:
   `--config-file` *replaces* `nav_default_config.yaml` (`config_helper.py:120-140`), it
   does not layer on top of it; the worked example is right by coincidence.
3. `README.md:181,188,196` — nonexistent `nav_mosaic_rings_cloud_tasks` /
   `nav_mosaic_body_cloud_tasks` command names (the real entry point is the single
   `nav_mosaic_cloud_tasks`); propagates into the Sphinx build via `index.rst`.
4. Stale symbols in dev_guide: `shape_for_body` (8 sites; actual `load_body_shape`,
   `body_shape.py:167`) and `StarFlags.psf_size` (1 site).
5. CONTRIBUTING.md drift: cites a nonexistent CODE_OF_CONDUCT.md, instructs `pre-commit
   install` with no `.pre-commit-config.yaml`, says "Python 3.10+" against the 3.11 floor,
   and its example docstring uses `(u, v)` against the project-wide `(v, u)` convention.
6. `user_guide_navigation.rst:672,698,757,786,800` — stale two-digit config filename
   prefixes (`config_03_stars.yaml` etc.).
7. Lower priority: api_reference omissions (notably 8 of 11 `mosaic_viewer` modules), the
   four placeholder instrument appendices (honest but empty), and the dead
   `log_level_nav_correlate_all` config key with phrasing that violates the project's
   documentation conventions.

---

## 5. Overall verdict

### Well-designed and well-implemented at the engineering level

The codebase earns a strong engineering grade. The architecture is clean and consistently
layered (DataSet -> Obs -> NavOrchestrator -> NavModel -> NavTechnique -> ensemble, with a
parallel reproj/backplanes/PDS4 tail), registries and factories are uniform, the typed
NavFeature algebra and RORO conventions are followed, and the strictest gates the project
sets for itself — ruff, ruff format, mypy `strict` across 329 files, 1,678 passing tests —
are all green. The mathematics at the core (robust DT fitting, information-form ensemble
combination, per-technique covariance) has survived a hostile 190-finding review plus
independent re-derivation, and the review-fix-verify loop demonstrably works: every sampled
Critical/High fix landed with a regression test, and the critique corpus doubles as an
honest remediation ledger. Provenance (git SHA, SPICE kernels, static-data hashes) is
pinned into every output. This is a genuinely healthy process, rare at this scale.

### Not yet scientifically validated — the honest gap

The system's engineering quality outruns its evidentiary basis. The still-valid core of the
critical review stands at HEAD:

- **No real-image accuracy study exists.** The only quantitative validation is
  self-referential: the simulator and the navigation models share renderer functions, so
  the published sub-pixel numbers bound numerics, not accuracy on archival frames.
- **Confidence is uncalibrated** and shipped with placeholder alphas; the calibration
  cohort (11 curated sidecars, 1 baseline) is ~20% of its own budget and is excluded from
  CI (`-m "not integration"`).
- **Known accuracy items are open:** the ~0.1 px limb systematic with its fix held off for
  the limb technique; body-shape table 51 placeholders deep; the sub-0.75-degree roll
  degeneracy.
- **Capability edges:** Titan is a no-op placeholder, PDS4 output is Cassini-only with
  `NotImplementedError` walls for the other three instruments, and `src/backplanes/` and
  `src/pds4/` have zero tests.

None of this is hidden — the docs admit it, the plans target it — but until REMEDIATION
Phase 0-2 (identifiability study, cross-technique agreement on real images, de-circularized
simulator validation) and the Phase-10 curation/calibration are executed, the headline
accuracy numbers should not be quoted as real-world performance.

### Highest-value actions

1. **Execute the validation program.** Finish the image library (PHASE10_CURATION steps
   C-G), then run REMEDIATION WS-0/WS-1/WS-2. Everything scientific hangs on this.
2. **Reconcile the plans.** One page declaring: which confidence-calibration methodology is
   the plan of record (WS-5 vs #173), the library-size target (120 vs 49), and a
   WS-to-issue cross-map; add acceptance criteria to WS-1b; refresh the stale premises
   (WS-14, WS-6, WS-10 ring half).
3. **Mark or gate the network-dependent unit tests** (`tests/nav/obs/test_obs_snapshot.py`,
   `tests/nav/dataset/test_dataset_pds3_*.py`, `tests/nav/inst/test_inst_*.py`) so a plain
   `pytest` passes without holdings access — 45 tests currently fail out of the box.
4. **File the three residual code findings** from section 2.1 (coarse-search
   minimum-support guard; the `ensemble.py:410` bare assert swallowed by orchestrator
   sandboxes; the rotation-combine weighting rationale), and the six still-open SIM
   findings if the simulator remains load-bearing for validation.
5. **Fix the top documentation defects** (section 4, items 1-5) — a half-day of work that
   removes every actively-misleading statement from the user-facing docs.
6. **Close the test-coverage holes that gate release confidence:** unit tests for
   `src/backplanes/` and `src/pds4/`, the star-conflict logic, and real-image baselines
   beyond the single frame (including retiring the sidecar whose asserted status is
   documented as wrong).

### One-line summary

An unusually well-engineered and honestly self-critical codebase whose critiques are
accurate, whose plans are strong but need reconciliation, and whose single real deficit is
that its validation program — correctly designed and fully planned — has not yet been run.
