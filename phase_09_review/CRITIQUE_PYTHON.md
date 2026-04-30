# Phase 9 — Python best-practices critique

Scope: every file touched by the Phase 9 camera-rotation work.

## Files modified

- `src/nav/nav_orchestrator/nav_context.py` — gained `fit_camera_rotation`, `max_rotation_deg`; `with_prior` now accepts a 3x3 input.
- `src/nav/nav_orchestrator/orchestrator.py` — populates the new context fields from `InstrumentSettings`.
- `src/nav/nav_orchestrator/ensemble.py` — 3-DoF parameter-vector handling, structured `_CombinedEstimate` return, mixed-DoF rejection, rotation propagation onto `NavResult.ok`.
- `src/nav/nav_technique/nav_technique.py` — added `ROTATION_UNOBSERVABLE_VARIANCE`, `embed_rotation_unobservable`, `rotation_pivot_distance_px`, `rotation_unobservable_sigma_rad`.
- `src/nav/nav_technique/nav_technique_body_limb.py` / `_body_terminator.py` / `_ring_edge.py` — pass `fit_rotation` to LM, use a vertex-centroid pivot, drive `at_edge` from the rotation cap, populate `rotation_rad` / `sigma_rotation_rad`.
- `src/nav/nav_technique/nav_technique_body_blob.py` / `_body_disc.py` / `_ring_annulus.py` — emit a rank-deficient 3x3 covariance with rotation unobservable when the flag is on.
- `src/nav/nav_technique/nav_technique_star_unique_match.py` / `_star_refine.py` / `_star_field.py` — same rank-deficient 3x3 promotion when the flag is on.
- `src/nav/config_files/config_410_inst_gossi.yaml` / `config_430_inst_vgiss.yaml` — `fit_camera_rotation: true`.

## Findings

### Style and structure

- **PEP 8 / 100-column lines** — every diff respects the project line cap; `ruff format --check` was clean before this critique.
- **Imports** — three alphabetical groups in every modified file; no inline imports added.
- **Docstrings** — every new public symbol carries a Google-style docstring with `Parameters:` and `Returns:` blocks per the project convention.
- **`__all__` ordering** — the rotation helpers were added to `nav.nav_technique.nav_technique.__all__`; `ruff` re-sorted into the project standard ordering automatically.
- **Magic constants** — `ROTATION_UNOBSERVABLE_VARIANCE` is a module-level `ALL_CAPS` constant with a docstring stating units and intent; no inline literal `1.0e15` appears in technique bodies.
- **No backwards-compat shims** — `NavContext.with_prior` previously rejected non-2x2 covariance; the new code accepts 3x3, projects to the 2x2 translation block, and proceeds. There is no shim emitting a deprecation warning, matching Cardinal Principle 1.

### Function signatures

- New helpers (`embed_rotation_unobservable`, `rotation_pivot_distance_px`, `rotation_unobservable_sigma_rad`) take 1 or 2 positional parameters and no keyword-only parameters; well within the 3-positional cap.
- The `_CombinedEstimate` dataclass is RORO-shaped: replaces the previous tuple return of `_combine_precision_weighted` with a typed dataclass. `combine` callers gain field-name-based access at no extra cost.

### Defensive programming

- `_combine_precision_weighted` now raises on mixed-DoF inputs and on zero total weight — both with messages that name the offending technique. Non-emptiness assertion is preserved.
- `_result_param_vector` raises `ValueError` when a 3x3 covariance is paired with `rotation_rad=None`; this mismatch represents a programmer error in a technique implementation, so failing fast is correct.
- Every per-technique 3-DoF code path raises `RuntimeError` when the LM result's covariance shape doesn't match the requested DoF — a defensive shield against future refactors of `lm_subpixel_refine`.

### Logging

- All new code paths reuse `self.logger` / `IMAGE_LOGGER` — no `import logging`.
- The orchestrator and ensemble paths emit no new INFO log line specific to rotation; the existing per-technique INFO lines surface offset and confidence, and `LMRefineResult.rotation_rad` is included in the final NavResult propagation. A future enhancement could log the converged rotation per-technique at INFO when `fit_camera_rotation` is on.

### Testing

- Three new ensemble tests (`test_ensemble_3dof_combines_translation_and_rotation`, `test_ensemble_3dof_with_rotation_unobservable_input`, `test_ensemble_rejects_mixed_dof_inputs`) cover the new combine paths.
- Three new context tests (`test_navcontext_rotation_fields_default_off`, `test_navcontext_rotation_fields_propagate`, `test_navcontext_with_prior_accepts_3x3_covariance`) cover the dataclass changes.
- Two new BodyLimbNav tests (`test_body_limb_nav_3dof_emits_3x3_covariance`, `test_body_limb_nav_3dof_at_edge_when_rotation_saturates`) cover the LM 3-DoF path and the rotation `at_edge` rule.
- A new BodyBlobNav test (`test_body_blob_3dof_rotation_unobservable`) covers the rank-deficient 3x3 covariance path.
- The `make_nav_context` factory grew `fit_camera_rotation` and `max_rotation_deg` kwargs; existing tests retain default off.

### Known deviations from the plan body

- The plan calls for `BodyDiscCorrelateNav` to run a 3-D NCC pyramid (rotation-sample schedule per Part 5b). This phase implements a simpler interim path: when `fit_camera_rotation` is on, the technique emits a rank-deficient 3x3 covariance with the rotation slot unobservable. The translation NCC pyramid still produces the 2-D peak; the rotation contribution is "no information" rather than "fitted". Cassini ISS / NHLORRI carry `fit_camera_rotation=False` so this code path does not run for the missions BodyDiscCorrelateNav was designed for; VGISS / GOSSI typically rely on DT techniques (limb / terminator / ring edge) where rotation is fully fit. The 3-D NCC pyramid is tracked as deferred work for Phase 12+ (or Phase 10 calibration if rotation residuals warrant it).
- The plan calls for `StarFieldFromCatalogNav` to fit rotation via Procrustes / Kabsch, and for the 2-star path of `StarUniqueMatchNav` to fit rotation. This phase emits rank-deficient 3x3 covariance for every star technique. The translation constraint propagates correctly through the ensemble; rotation is unobservable from any star path. Same rationale: the rotation-needing instruments (VGISS / GOSSI) typically have body / ring features for rotation, and the star paths' translation contributions still combine correctly via `pinvh`. Tracked as deferred work.

### Recommendation

No Critical or High findings. The interim simplifications above are documented in CRITIQUE_SUMMARY.md and tracked for follow-up.
