# Archived critiques

Point-in-time review documents (dated in each filename; all reviewing the code merged as commit
77a90f4), kept for reference. Each is a snapshot of what its author found on its
date, frozen as written: none is maintained, and none carries a running status
section. Internal file paths and line numbers reflect the tree as reviewed and
have drifted since; the paths predate the SpinDoctor rename, so `src/nav/` is now
`src/spindoctor/`, `tests/nav/` is `tests/spindoctor/`, and the CLI entry points
are `sd_*`.

For what is actually open, read the live plans and the GitHub issues, not these
records: `plans/PROGRAM_PLAN.md` is the plan of record, and every finding below
that remains open is tracked there or in an issue.

- `CODE_CRITIQUE_2026-06-10.md` — 190-finding source review with inline remediation
  status; the majority of findings are fixed, and every still-open
  Critical/High finding is tracked by a GitHub issue (see `ISSUE_CROSSREF_2026-06-19.md`
  and issues #132-#139).
- `TESTS_CRITIQUE_2026-06-10.md` — test-suite review. Much of what it reports has
  since been closed: the star-conflict logic gained direct coverage (#216), and the
  backplanes and PDS4 backends gained suites (#257). Regression baselines beyond the
  single real-image frame remain open as #174.
- `SCIENTIST_REVIEW_2026-06-19.md` / `SCIENTIST_REVIEW_CRITICAL_2026-06-19.md` — paired scientific
  reviews; every finding of the critical review is mapped to a workstream in
  `plans/VALIDATION_AND_CALIBRATION_PLAN.md` (see its traceability matrix).
- `ISSUE_CROSSREF_2026-06-19.md` — mapping of open GitHub issues to `CODE_CRITIQUE_2026-06-10.md`
  finding IDs; its "untracked candidates" were subsequently filed as issues
  #132-#139.
- `SIM_REWRITE_FINDINGS_2026-06-19.md` — simulator findings; each is either addressed in
  the merged simulator or tracked as a GitHub issue (#84, #194-#198).
- `PROJECT_STATE_REVIEW_2026-07-08.md` — meta-review of the whole software state:
  it audits the accuracy of the five reviews above, the completeness of the plan
  set as it stood, and the documentation, against freshly run quality gates. Its
  six recommendations were largely executed by PR #200 the same day it was written
  and the rest have since been folded into the live plans; its enduring value is
  the verification record (37 of 38 sampled code findings independently confirmed)
  and its verdict that the engineering quality outruns the evidentiary basis until
  the validation program runs.
