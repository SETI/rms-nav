# Phase 9 — Documentation critique

Reviewed against `.cursor/rules/documentation.mdc`.

## Module-level docstrings

- `nav.nav_orchestrator.nav_context` — module docstring unchanged; still accurate. The new `fit_camera_rotation` / `max_rotation_deg` fields are documented in the dataclass docstring per the project's Google-style convention.
- `nav.nav_technique.nav_technique` — module docstring unchanged; the new helpers (`ROTATION_UNOBSERVABLE_VARIANCE`, `embed_rotation_unobservable`, `rotation_pivot_distance_px`, `rotation_unobservable_sigma_rad`) each carry a one-paragraph docstring with units, intent, and rationale.
- All technique-specific module docstrings retain their behavioural summary; rotation handling is added inside the `navigate` docstring's "Returns:" paragraph where it differs from the 2-DoF case (e.g. "or 3x3 with rotation reported as unobservable when ``context.fit_camera_rotation`` is True").

## Function and class docstrings

- `NavContext.with_prior` — documented to accept either 2x2 or 3x3 covariance; explicitly calls out that only the 2x2 translation block is propagated to pass-2.
- `_combine_precision_weighted` — docstring updated to describe the new `_CombinedEstimate` return, the rotation handling, and the mixed-DoF rejection. The `Raises:` block lists every newly-added error condition.
- `embed_rotation_unobservable` — full Google-style docstring with `Parameters:` and `Returns:` plus a paragraph explaining why the sentinel is used instead of `+inf`.
- `_result_param_vector` — internal helper in `ensemble.py`; carries a docstring describing both 2-DoF and 3-DoF return shapes plus the consistency check between covariance shape and `rotation_rad` field.

## RST / docs build

- `docs/_build` was rebuilt successfully under `-W` after fixing a docstring substitution conflict (`|rotation|` was being interpreted as a docutils substitution).
- `pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md` is clean.
- No new RST file was added; the rotation work is internal to existing modules whose API pages auto-generate via `automodule`.

## Code comments

- Comments inside the technique bodies explain the why of rank-deficient 3x3 covariance for centroid / template-NCC techniques (rotation-invariant by construction). One short comment per call-site, no multi-paragraph blocks.
- The interim simplification deviations from the plan body are documented in `CRITIQUE_PYTHON.md` and `CRITIQUE_SUMMARY.md` rather than in code comments, per the project convention to keep code comments focused on hidden invariants.

## CLAUDE.md / README

- `CLAUDE.md` was not updated by this phase. The rotation-fit knob is exposed as a per-instrument config flag and surfaces automatically in JSON outputs via the curator; no developer-facing workflow changes.
- `README.md` likewise unchanged.

## Recommendation

No Critical or High findings. One Low-severity item: a paragraph could be added to `docs/architecture.rst` (or a new `developer_guide_rotation.rst`) explaining the 3-DoF combine and the rank-deficient pattern. Not phase-blocking.
