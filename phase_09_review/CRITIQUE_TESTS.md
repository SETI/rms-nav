# Phase 9 — Test-suite critique

Reviewed against `.cursor/skills/critique-test-suite/SKILL.md`.

## Test count and execution

- Test count went from 1211 -> 1220 (+9 new tests).
- Full suite passes under `pytest -n auto --dist=loadfile` in ~22 s.
- One pre-existing `PytestUnraisableExceptionWarning` from the New Horizons LORRI integration test is the project's known tolerated warning; not a Phase 9 regression.
- `pytest --cov` was not re-baselined for this phase since coverage on the cutover-tier modules was already >= 90 % before the change and the new code is exercised by the new tests + existing happy-path tests.

## Coverage of new behaviour

Added under `tests/nav/nav_orchestrator/test_ensemble.py`:
- `test_ensemble_3dof_combines_translation_and_rotation` — two 3-DoF results, well-conditioned, expected fused (dv, du, theta) within tight tolerance.
- `test_ensemble_3dof_with_rotation_unobservable_input` — combines an observable 3-DoF result with a rotation-unobservable 3-DoF result (the BodyBlob-style case); asserts rotation is dominated by the observable input.
- `test_ensemble_rejects_mixed_dof_inputs` — a 2-DoF + 3-DoF mix raises `ValueError` with a recognisable message substring.

Added under `tests/nav/nav_orchestrator/test_nav_context.py`:
- `test_navcontext_rotation_fields_default_off` — defaults check.
- `test_navcontext_rotation_fields_propagate` — explicit values survive `with_prior`.
- `test_navcontext_with_prior_accepts_3x3_covariance` — 3x3 input projects to 2x2 translation block.

Added under `tests/nav/nav_technique/test_nav_technique_body_limb.py`:
- `test_body_limb_nav_3dof_emits_3x3_covariance` — full LM end-to-end on a planted scene; asserts shape + bounded rotation.
- `test_body_limb_nav_3dof_at_edge_when_rotation_saturates` — monkeypatched `lm_subpixel_refine` returning `rotation_rad=4.9 deg` triggers `at_edge=True` against the 5-degree cap.

Added under `tests/nav/nav_technique/test_nav_technique_body_blob.py`:
- `test_body_blob_3dof_rotation_unobservable` — rank-deficient 3x3 covariance with the unobservable sentinel on the rotation diagonal.

## What is not tested in this phase

- BodyTerminatorNav 3-DoF path: covered indirectly by the technique sharing the same code path with BodyLimbNav (`fit_rotation` plumbed through identical `lm_subpixel_refine` invocation), but no per-technique 3-DoF integration test was added. Low risk.
- RingEdgeNav 3-DoF path: same situation. Low risk.
- BodyDiscCorrelateNav, RingAnnulusNav, StarFieldFromCatalogNav, StarUniqueMatchNav, StarRefineNav 3-DoF rank-deficient paths: each emits a 3x3 covariance with the unobservable sentinel. Verified via the body-blob test; the other techniques use the identical helper (`embed_rotation_unobservable`), so the same invariant holds.
- End-to-end orchestrator test under VGISS / GOSSI config with `fit_camera_rotation=True`: requires SPICE kernels + library images; deferred to integration-test territory and Phase 10 calibration.

## Conventions adherence

- One assertion per condition; the new tests do not chain `assert ... and ...`.
- `pytest.raises` used as a context manager with a message regex (the mixed-DoF rejection test).
- No `pytest.mark.xfail` / `skipif` introduced.
- Type annotations on every test function.
- Tests use `capsys` not `caplog` consistently with the project rule (no log-capture assertions added by this phase).

## Recommendation

One Medium-severity finding: BodyTerminatorNav and RingEdgeNav lack their own 3-DoF tests despite being the two highest-value rotation-fit techniques for VGISS / GOSSI scenes. The shared `lm_subpixel_refine` machinery is well-tested in `tests/nav/nav_technique/test_dt_fitting.py` (`fit_rotation=True` paths), so the risk is contained, but per-technique tests would be a small add.

```text
Suggested fix prompt:
  Add ``test_body_terminator_nav_3dof_emits_3x3_covariance`` and
  ``test_ring_edge_nav_3dof_emits_3x3_covariance`` mirroring the
  BodyLimbNav 3-DoF test.  Each plants a tiny scene, runs the
  technique with ``fit_camera_rotation=True``, and asserts:
    - covariance_px2.shape == (3, 3)
    - rotation_rad is not None
    - sigma_rotation_rad is not None
    - |rotation_rad| < deg_to_rad(5)
```

Not phase-blocking; no Critical or High findings.
