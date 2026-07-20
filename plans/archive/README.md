# Archived plans

Historical planning records, kept for reference. The live plans are in the
parent `plans/` directory: `PROGRAM_PLAN.md` is the top-level plan of record;
`VALIDATION_AND_CALIBRATION_PLAN.md` (Track A science detail),
`ENGINEERING_PLAN.md` (Tracks B-F implementation detail), and
`COHORT_CURATION_PLAN.md` (image-library growth playbook) are its detail
layers. Archived documents may reference each other and pre-archive paths
(e.g. `plans/ROADMAP.md`, `plans/PHASE10_CURATION.md`); resolve such
references against the dated filenames in this directory.

- `AUTONAV_PLAN_2026-06-19.md` — design record and phase plan for the
  autonomous-navigation core rewrite (merged as of commit 77a90f4). Phases 0-9
  shipped as specified; the remaining work migrated to GitHub issues. Parts
  1-16 remain the design reference for the shipped architecture (read with the
  document's own precedence rules: the Part 0 header overrides stale body
  text).
- `SIM_IMPROVEMENT_PLAN_2026-06-19.md` — simulator improvement plan, executed
  except for items deferred with issue numbers (#151, #152, #153). Section 0.1
  is the authoritative status board.
- `SIM_REALISM_PLAN_2026-07-18.md` — the simulator realism and
  de-circularization design (rev 2.1), fully executed 2026-07-16 to
  2026-07-18: all ten phases (A-J) plus the final sweep merged through the
  `rf_sim_realism` integration branch, all acceptance criteria (its
  Section 11) verified met. The as-built system is documented in
  `docs/dev_guide/dev_guide_simulator.rst` (including the capability
  envelope), the calibration in `util/calibration/CAMPAIGN_20260718.md`,
  and the independent assessment in
  `critiques/SIM_REALISM_CRITIQUE_2026-07-18.md`; follow-up work moved to
  GitHub issues (#301, #309, #310, #311).
- `ROADMAP_2026-07-12.md` — the issue-ordered, Cassini-first pipeline
  build-out that served as the plan of record until 2026-07-12; consolidated
  into `plans/PROGRAM_PLAN.md` together with the post-stack task inventory.
- `FULL_PROGRAM_AFTER_MANUAL_WORK_2026-07-11.md` — the six-track task
  inventory written after the 2026-07 PR stack (#208-#220) was prepared;
  consolidated into `plans/PROGRAM_PLAN.md` the following day. Its track
  letters (A-F) carried over to the live plan.
- `PHASE10_CURATION_2026-07-12.md` — the original operator playbook for the
  first-stage (49-image) library build. Its durable content moved to
  `docs/dev_guide/dev_guide_image_library.rst` (sidecar schema, tier rubric,
  baselines) and the appendix of `plans/COHORT_CURATION_PLAN.md` (scene-class
  budget, selection guide, mission hints); the automated Stage A-E workflow in
  that plan superseded its manual front half.
