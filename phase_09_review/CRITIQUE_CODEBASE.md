# Phase 9 — Codebase analysis critique

Reviewed against `.cursor/skills/python-codebase-analysis/SKILL.md`.

## Architectural coherence

- The Phase 9 work is a small surface-area extension: one boolean flag on `NavContext`, one new dataclass return type on the ensemble combine, three new helpers, and per-technique covariance promotion. No subsystem boundary is moved; no new modules are introduced.
- The plumbing flow (`InstrumentSettings.fit_camera_rotation` -> `NavContext.fit_camera_rotation` -> per-technique `navigate`) is one-way; no technique mutates the context. This matches Cardinal Principle 7 (no cross-image state, no mid-image mutation).
- The ensemble's new `_CombinedEstimate` dataclass replaces a previous tuple return — slightly improves call-site readability and lets the rotation field be optional.

## Module size

- `nav_technique.py` grew from 271 -> 360 lines (under the 1000-line cap).
- `ensemble.py` grew from 505 -> 560 lines (under the cap).
- `nav_context.py`, `orchestrator.py`, `nav_result.py`, `curator.py` are unchanged in size beyond a few lines.
- Every per-technique module gained ~15-30 lines.

No module is approaching the 1000-line cap.

## Subsystem coupling

- The orchestrator's only new dependency on rotation knobs is via `instrument_settings_from_obs(obs).fit_camera_rotation` — already a public helper. No new attribute reads on `obs`.
- The ensemble's only new dependency is on `NavTechniqueResult.rotation_rad` — already declared on the dataclass before this phase.
- Every per-technique change reads `context.fit_camera_rotation` and (where relevant) `context.max_rotation_deg`. No technique reads from the obs directly to discover the flag — strong containment.

## Public API impact

- `NavContext` gained two fields, both with safe defaults (`fit_camera_rotation=False`, `max_rotation_deg=5.0`). Existing constructors that omit them keep the 2-DoF behaviour.
- `NavResult.ok(...)` already accepted `rotation_rad` / `sigma_rotation_rad`; this phase populates them when the ensemble produces them.
- `NavTechniqueResult` dataclass is unchanged (rotation fields existed pre-Phase 9).
- `_combine_precision_weighted` return shape changed from a tuple to `_CombinedEstimate`; this is an internal helper (leading underscore) so no consumers outside `ensemble()` and `tests/nav/nav_orchestrator/test_ensemble.py` were affected.
- The new helpers (`ROTATION_UNOBSERVABLE_VARIANCE`, `embed_rotation_unobservable`, `rotation_pivot_distance_px`, `rotation_unobservable_sigma_rad`) are added to `nav.nav_technique.nav_technique.__all__`.

## Code-style consistency

- Error messages are full sentences with ample context (technique name, expected shape, actual shape) per project convention.
- Defensive shape checks (`if covariance.shape != (3, 3): raise RuntimeError(...)`) match the existing pattern in BodyLimbNav.
- Frozen dataclass + `__post_init__` discipline preserved throughout (`_CombinedEstimate` is frozen).

## Static-data citation discipline

- The two YAML edits flip an existing boolean from `false` to `true`. No numeric value was added that requires a `_sources` entry. Citation requirement (Part 0 §74) is preserved: nothing new ships without a citation.

## Recommendation

No Critical or High findings. The architecture remains coherent; the surface-area extension is minimal and well-contained.
