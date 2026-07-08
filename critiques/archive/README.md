# Archived critiques

Point-in-time review documents (dated in each filename; all reviewing the code merged as commit
77a90f4), kept for reference. The current assessment of the software state is
in the parent `critiques/` directory. Internal file paths and line numbers in
these records reflect the tree as reviewed and have drifted since.

- `CODE_CRITIQUE_2026-06-10.md` — 190-finding source review with inline remediation
  status; the majority of findings are fixed, and every still-open
  Critical/High finding is tracked by a GitHub issue (see `ISSUE_CROSSREF_2026-06-19.md`
  and issues #132-#139).
- `TESTS_CRITIQUE_2026-06-10.md` — test-suite review; its still-open items (no unit tests
  for the SPICE-backed render path, star-conflict logic, `src/backplanes/` and
  `src/pds4/`; single real-image baseline) are summarized in the parent
  directory's project state review.
- `SCIENTIST_REVIEW_2026-06-19.md` / `SCIENTIST_REVIEW_CRITICAL_2026-06-19.md` — paired scientific
  reviews; every finding of the critical review is mapped to a workstream in
  `plans/VALIDATION_AND_CALIBRATION_PLAN.md` (see its traceability matrix).
- `ISSUE_CROSSREF_2026-06-19.md` — mapping of open GitHub issues to `CODE_CRITIQUE_2026-06-10.md`
  finding IDs; its "untracked candidates" were subsequently filed as issues
  #132-#139.
- `SIM_REWRITE_FINDINGS_2026-06-19.md` — simulator findings; each is either addressed in
  the merged simulator or tracked as a GitHub issue (#84, #194-#198).
