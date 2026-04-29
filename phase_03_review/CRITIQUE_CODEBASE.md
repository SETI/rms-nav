# Phase 3 Code Review — Codebase Analysis

Review using `.cursor/skills/python-codebase-analysis/SKILL.md`.

## Module-tree health

- **`nav.nav_orchestrator.instrument_config`** — new module, ~200 lines, single-responsibility (translate per-camera YAML block into `InstrumentSettings`).
- **`nav.nav_orchestrator.provenance`** — extended from a single dataclass to also expose `ProvenanceMetadata` and the `collect_provenance_metadata` helper. Total ~190 lines, still well under the 1000-line module ceiling.
- **`nav.nav_orchestrator.orchestrator`** — gained `_apply_source_image_filter`, `_build_saturation_mask`, `_log_status_reason`, `_log_classifier_verdict`, `_log_final_result`, `_fail`. Footprint grew but stayed below 700 lines; the file is still read-once-and-understand size.
- **`nav.nav_technique.nav_technique`** — gained `confidence_spec` / `confidence_attributes` class attributes plus `validate_registered_confidence_specs` and `log_confidence_breakdown` helpers. Same footprint growth; still well-bounded.
- **`nav.nav_technique.confidence`** — gained `ConfidenceTermContribution` and `ConfidenceBreakdown` dataclasses plus the `return_breakdown=True` mode of `evaluate_sigmoid_combination`. Module stays focused; no orchestration logic crept in.
- **`nav.nav_orchestrator.ensemble`** — added per-failure-path INFO logging (combined-confidence, conflicted gap, all-spurious technique-name list, unobservable-offset input count, no-tier-earned sigma values). No structural change to the algorithm itself.

## API stability

- `Provenance` gained no new fields; `collect_provenance_metadata` is a new free function the orchestrator's `_make_provenance` consumes. External callers using `Provenance(...)` directly are unaffected.
- `InstrumentSettings` is new; no external API obligation yet.
- `NavTechnique` gained two class attributes with conservative defaults (`None` / empty frozenset); existing third-party subclasses keep working.
- `evaluate_sigmoid_combination` defaults `return_breakdown=False`, preserving the legacy `float`-returning shape; any external caller still gets the same value.

## Cross-cutting concerns

- **No `import logging` introduced** anywhere in the new core code.
- **No backwards-compat shims.** Every renamed file moves cleanly via `git mv`; legacy filename references in docs are updated in the same PR.
- **Frozen dataclasses with `__post_init__` validation** continue to be the project pattern; both new dataclasses follow it.
- **No wrapper helpers.** `psf_sigma_px` is the single source of truth for PSF sigma extraction; the deleted `_psf_sigma` and `_star_psf_sigma` wrappers used to mask the per-axis-vs-single-`sigma` interface bug.

## Open items

_None._
