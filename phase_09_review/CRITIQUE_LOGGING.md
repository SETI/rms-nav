# Phase 9 — Logging critique

Reviewed against `.cursor/rules/logging_best_practices.mdc`.

## Scope

The Phase 9 work threaded `fit_camera_rotation` through orchestrator -> ensemble -> per-technique fit. Every modified module already uses `pdslogger` via `NavBase.logger` / `nav.config.IMAGE_LOGGER`.

## Findings

- **No `import logging`** anywhere in the new core code.
- **No bare `print()`** in any modified file.
- **`with self.logger.open(f'TECHNIQUE: {self.name}'):`** wraps every `navigate` body — preserved in BodyLimbNav, BodyTerminatorNav, RingEdgeNav, BodyDiscCorrelateNav, RingAnnulusNav, BodyBlobNav, StarFieldFromCatalogNav, StarUniqueMatchNav, StarRefineNav. The 3-DoF path enters the same logger section.
- **Per-technique INFO lines** — the existing offset / confidence / spurious / at_edge logging was preserved; rotation is not yet surfaced in INFO output. This is a low-severity gap: an operator running at INFO sees a rotation-fit result no differently from a 2-DoF result. Future enhancement: emit a single INFO line per 3-DoF technique with the converged `rotation_rad` and the rotation `at_edge` flag.
- **No exception swallowing** — the orchestrator's plugin-sandbox catches in `_extract_features`, `_run_pass`, `_collect_annotations`, and `_build_models` are unchanged; new code does not introduce additional `except Exception:` sites.

## Recommendation

One Low-severity finding: log the converged rotation per technique at INFO when `fit_camera_rotation` is on.

```text
Suggested fix prompt:
  Each DT-based technique (BodyLimbNav, BodyTerminatorNav, RingEdgeNav)
  emits an INFO line "Converged at offset (...) px, RMS ... px, ...,
  confidence ...".  When ``context.fit_camera_rotation`` is True, append
  ", rotation = +X.XXX deg (sigma Y.YYY)" to the same line so an operator
  scanning logs sees the rotation outcome alongside the translation.
  Also log the rotation_at_edge flag when set.
```

Not phase-blocking.
