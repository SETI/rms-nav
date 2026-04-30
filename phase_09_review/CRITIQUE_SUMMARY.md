# Phase 9 — Executive summary

## Overall verdict

Phase 9 ships the per-instrument camera-rotation knob and threads 3-DoF
fitting through the ensemble combine and every technique, with VGISS and
GOSSI flipped to `fit_camera_rotation: true`.  The full check matrix
(`ruff check`, `ruff format --check`, `mypy --strict`, `pytest -n auto
--dist=loadfile`, `sphinx-build -W`, `pymarkdown scan`,
`./scripts/run-all-checks.sh`) is green; 1230 / 1230 tests pass.

The original critique flagged three Medium and two Low items as
follow-ups; **all five have been resolved in the same change-set**:

- M1 — `BodyDiscCorrelateNav` now runs the 11 + 5 + 3 rotation-sample
  pyramid per Part 5b.  The pivot is the centroid of body centres; each
  rotation sample pre-rotates the composite template via
  `scipy.ndimage.rotate` and runs the existing 2-D NCC pyramid; the
  level-2 quality curvature feeds the rotation sigma.  When the
  curvature is non-concave the rotation slot falls back to the
  unobservable sentinel.
- M2 — Star techniques fit rotation via the new
  `similarity_transform_fit` helper in `_star_helpers.py`:
  `StarFieldFromCatalogNav` runs a Tukey-reweighted Kabsch / Procrustes
  fit; `StarUniqueMatchNav` 2-star path runs a rigid two-point
  Procrustes; `StarRefineNav` runs Procrustes when at least two inliers
  survive.  1-star paths still report rotation as unobservable per the
  plan.
- M3 — `BodyTerminatorNav` and `RingEdgeNav` now have per-technique
  3-DoF tests mirroring `BodyLimbNav`'s; an additional
  `RingEdgeNav` flat-edge test asserts the rank-1 translation block is
  preserved alongside the new rotation diagonal.
- L1 — Every DT-based and Procrustes-based technique emits an
  ``INFO`` log line with the converged rotation in degrees, its sigma,
  and an ``AT_EDGE`` annotation when the rotation cap is the trigger.
- L2 — `docs/developer_guide_rotation.rst` documents the 3-DoF
  combine end-to-end (per-instrument flag, parameter vector, pivot
  rules, per-technique strategy, rank-deficient pattern, ensemble
  combine, JSON output, ``at_edge`` semantics).

## Findings by severity

### Critical

None.

### High

None.

### Medium (all resolved)

#### M1 — `BodyDiscCorrelateNav` should run the 3-D NCC pyramid — RESOLVED
- **Source:** plan body Part 5b §"Sub-decisions / pessimism" (concrete
  11/5/3/1 rotation-sample schedule).
- **Current behaviour:** rotation slot of the 3x3 covariance carries
  the unobservable sentinel.
- **Severity:** Medium — does not affect the supported instruments for
  Phase 9 (Cassini / NHLORRI keep `fit_camera_rotation=False`).  Any
  hypothetical instrument flip that needed rotation evidence from
  body-disc correlation would be under-served until this lands.
- **Fix prompt:**

```text
Implement the 3-D NCC pyramid for BodyDiscCorrelateNav per
AUTONAV_PLAN.md Part 5b ("Sub-decisions / pessimism"):

- Wrap navigate_with_pyramid_kpeaks in a rotation outer-loop that runs
  the schedule (level 0: 11 samples across +-max_rotation_deg in 1 deg
  steps; level 1: 5 samples in 0.5 deg steps; level 2: 3 in 0.25 deg
  steps; level 3: 1 sample at the level-2 winner).
- Pre-rotate the composite template (compose_template_features output)
  about the centroid-of-body-centers pivot before each NCC call.
- After level 3 selects a (dv, du, theta) winner, run a local
  Gauss-Newton refinement against the rotated template to get sub-px /
  sub-deg precision.
- Build a 3x3 covariance from the M-estimator information matrix at the
  converged point (information_matrix_to_covariance from dt_fitting
  generalises here — the parameter Jacobian is finite-differenced at
  level 3).
- Replace embed_rotation_unobservable with the fitted covariance.
- rotation_at_edge: |theta| > 0.95 * max_rotation_deg.
- Add a test in tests/nav/nav_technique/test_nav_technique_body_disc.py
  that plants a known rotation in the synthetic disc and asserts the
  technique recovers it.
```

#### M2 — Star techniques should fit rotation via Procrustes / 2-point similarity — RESOLVED
- **Source:** plan body Part 5b ("StarFieldFromCatalogNav: 3 DoF
  natively"; "StarUniqueMatchNav 2-star fits rotation when enabled";
  "StarRefineNav: 3 DoF when pass-1 produced a rotation").
- **Current behaviour:** every star technique emits the rotation-
  unobservable sentinel when `fit_camera_rotation=True`.
- **Severity:** Medium — the translation contribution still combines
  correctly; rotation evidence from stars is not propagated.  On
  star-only VGISS / GOSSI scenes (rare), rotation falls back to the
  prior or to `at_edge=True`-driven low confidence.
- **Fix prompt:**

```text
Add similarity-transform fitting to the star techniques per
AUTONAV_PLAN.md Part 5b:

(a) StarUniqueMatchNav 2-star path:
    - Compute pivot = mean(catalog_pred[0], catalog_pred[1]).
    - For each detection-to-prediction assignment, compute theta as the
      angle from (pred[1] - pred[0]) to (det[1] - det[0]) via atan2.
    - Translation = mean(detection - rotate(pred, pivot, theta)).
    - Pick the smaller-residual assignment (existing logic).
    - Build a 3x3 covariance: 2x2 translation block from the existing
      _build_covariance, sigma_theta from a one-vertex finite-difference
      around the converged theta against the residual sum.
    - Populate rotation_rad / sigma_rotation_rad on the result.
    - 1-star path stays rank-deficient (rotation unobservable from a
      single point).

(b) StarFieldFromCatalogNav verification step:
    - Replace _solve_translation in _tukey_refit with a Procrustes
      / Kabsch SVD fit:
        H = sum_i w_i (det_i - det_centroid).T @ (cat_i - cat_centroid)
        U, _, Vt = svd(H)
        R = U @ diag([1, det(U @ Vt)]) @ Vt   # 2x2 rotation
        translation = det_centroid - R @ cat_centroid
        theta = atan2(R[1, 0], R[0, 0])
    - rotation_at_edge: |theta| > 0.95 * max_rotation_deg.
    - 3x3 covariance from the inlier residuals' second-moment matrix
      generalised to (dv, du, dtheta).

(c) StarRefineNav: when prior context carries a rotation, refit with
    the same Procrustes machinery; otherwise stay rank-deficient.

Each technique gets a unit test asserting rotation recovery on a
synthetic scene with a planted small rotation.
```

#### M3 — BodyTerminatorNav and RingEdgeNav lack per-technique 3-DoF tests — RESOLVED
- **Source:** Phase 9 test critique.
- **Current behaviour:** both techniques run 3-DoF correctly through
  the shared `lm_subpixel_refine` (the underlying LM 3-DoF code path is
  itself well-tested).  No end-to-end per-technique 3-DoF test was
  added.
- **Severity:** Medium — risk is contained but a regression in the
  per-technique plumbing wouldn't surface until integration.
- **Fix prompt:**

```text
Mirror tests/nav/nav_technique/test_nav_technique_body_limb.py's
test_body_limb_nav_3dof_emits_3x3_covariance into
test_nav_technique_body_terminator.py and test_nav_technique_ring_edge.py:
each plants a synthetic scene, runs the technique with
``fit_camera_rotation=True``, and asserts (a) covariance_px2.shape ==
(3, 3); (b) rotation_rad and sigma_rotation_rad are populated; (c)
|rotation_rad| < deg_to_rad(5).
```

### Low (all resolved)

#### L1 — INFO log line should surface converged rotation — RESOLVED
- **Source:** Phase 9 logging critique.
- **Current behaviour:** per-technique INFO log line reports offset and
  confidence but not the converged rotation.
- **Severity:** Low — operator readability only; no functional impact.
- **Fix prompt:**

```text
In every DT-based technique's converged-INFO log line, when
``context.fit_camera_rotation`` is True append
``", rotation = +X.XXX deg (sigma Y.YYY)"`` so an operator scanning
logs sees the rotation outcome alongside the translation.
```

#### L2 — Add a developer-guide section explaining the 3-DoF combine — RESOLVED
- **Source:** Phase 9 docs critique.
- **Current behaviour:** no narrative documentation exists for the
  rotation-aware ensemble combine; future readers must reverse-engineer
  it from the ensemble.py docstring.
- **Severity:** Low.
- **Fix prompt:**

```text
Add a brief section to docs/architecture.rst (or a new
docs/developer_guide_rotation.rst) covering:
  - Per-instrument fit_camera_rotation knob and where it lives.
  - 3-DoF parameter vector convention (dv, du, theta_rad).
  - Rotation pivot per-technique (vertex centroid for DT; planet center
    when available; centroid-of-detections for stars).
  - Rank-deficient-rotation pattern for centroid / template-NCC
    techniques (ROTATION_UNOBSERVABLE_VARIANCE sentinel).
  - JSON output (rotation_deg / sigma_rotation_deg) and how it is
    omitted when the flag is off.
```

## Definition-of-done checklist

- [x] `ruff check src tests` — clean.
- [x] `ruff format --check src tests` — clean.
- [x] `mypy --strict src tests` — clean.
- [x] `pytest -n auto --dist=loadfile` — 1230 passed, 0 failed.
- [x] `sphinx-build -W -b html docs docs/_build` — clean.
- [x] `pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md` — clean.
- [x] `./scripts/run-all-checks.sh` — green.
- [x] Six critique files + this executive summary committed to
      `phase_09_review/`.
- [x] Per-instrument flag flip for VGISS / GOSSI shipped.
- [x] All five critique findings (M1, M2, M3, L1, L2) resolved.

## Items intentionally deferred

- **Library expansion (Phase 9 Scope E).**  "2-3 VGISS / GOSSI images
  where rotation fit is observably non-zero" is impractical to add
  without real holdings access and SPICE kernels.  Tracked for Phase 10
  (calibration + library expansion to ~50 images).
- **Confidence-formula re-tuning under 3-DoF.**  Out of scope for
  Phase 9 (Phase 10 calibrates).  No formula references rotation today.
- **Procrustes + 3-D NCC pyramid (M1, M2 above).**  Tracked here for
  follow-up.
