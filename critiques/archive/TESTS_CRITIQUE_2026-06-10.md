# RMS-NAV Test Suite Critique

Branch: `core_rewrite_phase10` · Scope: every file under `tests/` (~26k LOC)
· Generated 2026-06-10

## Executive summary

The unit test suite for the navigation **math** (models, techniques, and the
shared distance-transform / Levenberg-Marquardt / ensemble infrastructure) is,
on the whole, genuinely good: the technique tests plant known offsets and
rotations into synthetic disc/step/star images and assert recovery to
0.05-1.0 px tolerances against analytically-derived expected values; the
`dt_fitting` tests check the Holland-Welsch Tukey formula against closed form,
verify rank-deficient covariance, trust-region clamping, and damping
saturation; failure modes (spurious, at-edge, polarity rejection) are forced
deterministically via `monkeypatch`. This is real verification, not smoke
testing.

That good work sits on top of two serious problems:

1. **The model *rendering* algorithms are not unit-tested at all.** Every
   `NavModel*.create_model` path (the code that projects body limbs,
   terminators, ring edges, and stars into image coordinates and builds
   `NavFeature`s) is exercised *only* by integration tests that require live
   `oops`/SPICE/holdings and the network. The unit coverage of
   `nav_model_body.py` is 49%, `nav_model_body_base.py` 8%, the `*_simulated`
   models 20-24%, and `nav/sim/` 3-8%. The default suite never executes the
   most navigation-critical math.
2. **The regression/baseline layer is near-vacuous and partly snapshots known-
   wrong output.** There is exactly **one** baseline JSON for **11** curated
   sidecars, it is gated behind `integration` + `PDS3_HOLDINGS_DIR`, and the
   single baseline it does pin (`N1597846115`) records an offset the sidecar's
   own notes describe as the *wrong* status decision.

I ran the full unit suite via
`pytest --ignore=tests/integration --cov=nav --cov-report=term-missing` and
targeted technique/integration subsets. I did **not** run the
`integration`-marked regression body (no holdings/network in this
environment); those findings are by reading.

### Severity-tiered table of contents

| Tier | IDs | Theme |
|------|-----|-------|
| High | TEST-MODEL-001, TEST-INT-002, TEST-INT-003, TEST-STAGE-001, TEST-STAR-001 | Untested model rendering, vacuous regression layer, untested downstream stages, untested star-conflict resolution |
| Medium | TEST-INFRA-001, TEST-CONV-001, TEST-CONV-002, TEST-MODEL-002, TEST-INT-004 | Untested dataclass validation guards, caplog/stdlib-logging divergence, bare `pytest.raises`, simulated-model gaps, baseline-as-snapshot |
| Low | TEST-CONV-003, TEST-MODEL-003, TEST-MISC-001 | Multi-condition assert, monotonicity-only reliability tests, minor hygiene |

---

## Coverage matrix

Coverage % is from the unit suite (`--ignore=tests/integration`). "Integration
only" means the real algorithm runs only under the holdings-gated suite.

### Navigation models

| Model | Unit cov | Real algorithm tested? | Planted ground-truth? | Gap / risk |
|-------|---------|------------------------|-----------------------|------------|
| `NavModelStars` | 88% | Mostly (detection, catalog, SNR, PSF unit-tested) | Detection recovery yes | `create_model` star-projection path partly integration-only. **Medium** |
| `NavModelBody` | 49% | **No** — `create_model` integration-only (per file docstring) | Helpers only (`_incidence_factor`, sigmoids, sigma) | Limb/terminator/disc/blob feature *extraction* never unit-tested. **High** |
| `NavModelBody` base | 8% | No | No | Base class wiring untested. **High** |
| `NavModelBodySimulated` | 24% | No | No | Depends on `sim/`; integration-only. **Medium** |
| `NavModelRings` | 61% | Partial (ring_math/ring_feature/ring_filter helpers strong) | Math helpers yes | `create_model` ring-edge projection integration-only. **High** |
| `NavModelRings` base | 10% | No | No | **High** |
| `NavModelRingsSimulated` | 20% | No | No | **Medium** |
| `NavModelTitan` | (small file) | Thin (40-line test file) | No | Photometric Titan model barely covered. **Medium** |

### Navigation techniques

| Technique | Unit cov | Planted-offset recovery | Failure paths (spurious/at-edge/conflict) | Risk |
|-----------|---------|-------------------------|-------------------------------------------|------|
| `BodyLimbNav` | 90% | Yes, 0.05 px; partial-arc; multi-body; polarity exact-count | spurious (inlier-frac, LM-walk), at-edge, 3-DoF, derivative-missing | **Low** (exemplary) |
| `BodyTerminatorNav` | 90% | Yes (has recovery + monkeypatch) | Yes | **Low** |
| `BodyDiscCorrelateNav` | 91% | Yes (z-buffer multi-body, peak-ratio, 3-DoF) | at-edge, no-template | **Low** |
| `BodyBlobNav` | 98% | Yes | Yes | **Low** |
| `RingEdgeNav` | 90% | Yes | Yes | **Low** |
| `RingAnnulusNav` | 93% | Yes (16 tests) | Yes | **Low** |
| `StarFieldFromCatalogNav` | 93% | Yes (22 tests) | Yes | **Low** |
| `StarUniqueMatchNav` | 92% | Yes (1-star, 2-star Procrustes rotation, margin gate) | spurious, at-edge, rotation-unobservable | **Low** |
| `StarRefineNav` | 92% | Yes | Yes | **Low** |
| `NavTechniqueManual` | 92% | UI-driven; n/a | Yes | **Low** |

### Shared infrastructure

| Module | Unit cov | Tested? | Gap / risk |
|--------|---------|---------|------------|
| `dt_fitting` | 90% | Yes — Tukey closed-form, NCC integer recovery, LM subpixel + rotation closed-form, rank-1 covariance, trust-region, damping-saturation, full input-validation matrix | Strong. **Low** |
| `ensemble` | 93% | Yes — precision-weighted mean (analytic), conflicted, all-spurious, derive_confidence_rank | **Low** |
| `confidence` | 83% | Partial — `__post_init__` validation branches (TypeError/ValueError on bad config) untested | **Medium** |
| `confidence_config` | 85% | Partial — same validation-guard gap | **Medium** |
| `feasibility` | 64% | Happy path only — every `__post_init__` raise branch untested | **Medium** |
| `image_derivatives` | (high) | Yes (gradient + edge-DT used throughout) | **Low** |
| `nav.nav_model.stars.conflicts` | 49% | **No** — only `parse_ring_occlusion_annuli` + `_conflict_body_list` tested; the actual occlusion/conflict-detection logic (body silhouette, ring annuli) untested | **High** |

### Downstream stages (no test directory at all)

| Stage | Src LOC | Unit cov | Risk |
|-------|---------|---------|------|
| `src/backplanes/` | ~1130 | none | **High** — no `tests/backplanes/` |
| `src/pds4/` | (bundle_data, collections) | none | **High** — no `tests/pds4/` |
| `src/nav/sim/` | ~1765 | 3-8% | **High** — no `tests/nav/sim/`; underpins the simulated models |

---

## Findings

### High

#### TEST-MODEL-001 — Model feature-extraction (`create_model`) has no unit test
- **Files:** `tests/nav/nav_model/test_nav_model_body.py` (docstring explicitly defers `create_model` to integration), `test_nav_model_rings.py`, and the `*_integration.py` siblings.
- **Severity:** High.
- **Description:** The functions that turn a navigated `obs` into `NavFeature` limb/terminator/disc/ring-edge/star geometries — the heart of what the techniques consume — are tested only through the holdings-gated integration path. Unit coverage: `nav_model_body.py` 49% (lines 233-252, 303-391, 423-599 unexecuted), `nav_model_body_base.py` 8%, `nav_model_rings.py` 61%, bases ~10%. The pure helpers around them are well tested, but the projection/sampling/assembly code is not.
- **Why it matters:** A sign error or off-by-one in limb sampling, normal orientation, bbox computation, or sigma propagation would not be caught by any test runnable without SPICE + network. The technique tests use a *hand-built* conftest polyline, so they cannot catch a bug in how the real model builds that polyline.
- **Impact:** The most navigation-critical geometry code is unverified in the default suite. Mitigation: build `create_model` tests against the `tests/shims/` fake backplane (which already exists) so a synthetic geometry can be projected and the resulting `NavFeature` asserted without holdings.

#### TEST-INT-002 — Regression baseline layer is near-vacuous
- **Files:** `tests/integration/test_baselines.py::test_regression_baseline_exact_match`; `tests/integration/baselines/` (one file: `N1597846115_2_CALIB.json`).
- **Severity:** High.
- **Description:** 11 curated sidecars exist but only **1** baseline. The exact-match regression test parametrizes over `discover_baseline_paths`, so it guards exactly one image, and only when `PDS3_HOLDINGS_DIR` is set and the `integration` marker is selected. In the default suite it contributes nothing.
- **Why it matters:** The "load-bearing regression" described in the docstrings is one image deep. A regression in any technique on any other scene class would pass the suite silently.
- **Impact:** No meaningful end-to-end regression protection in practice. Populate baselines for all curated sidecars (the `update_baselines` harness exists for this).

#### TEST-INT-003 — `test_autonomous_nav` (the real ground-truth harness) never runs in the default suite and has loophole tolerances
- **Files/tests:** `tests/integration/test_autonomous_nav.py::test_one_library_image`.
- **Severity:** High.
- **Description:** This is the only test that scores real nav output against operator ground truth (status, confidence tier, offset-within-uncertainty, primary technique, must-run/must-skip). It is entirely `integration`-gated and holdings-gated. Two structural loopholes: (a) the offset-accuracy assertion (block c) runs **only** when `expected.status == 'ok'`, so any sidecar marked `conflicted`/`failed` (e.g. `high_phase_terminator`, whose offset is acknowledged-questionable) never has its offset checked; (b) the tolerance is `offset_uncertainty_px + 0.5`, set per-sidecar — a loosely-curated sidecar can assert almost nothing.
- **Why it matters:** The suite's strongest correctness signal is invisible to the default run and can be weakened per-sidecar.
- **Impact:** Real navigation accuracy is unverified in CI's default job. Add a CI lane that sets the env vars and runs `-m integration`; consider also checking offset for `conflicted` images where a ground-truth offset is recorded.

#### TEST-STAGE-001 — Entire downstream stages have zero tests
- **Files:** No `tests/backplanes/`, no `tests/pds4/`, no `tests/nav/sim/` directory exists.
- **Severity:** High.
- **Description:** `src/backplanes/` (~1130 LOC: `backplanes_bodies.py`, `backplanes_rings.py`, `merge.py`, `writer.py`), `src/pds4/` (`bundle_data.py`, `collections.py`), and `src/nav/sim/` (~1765 LOC: `render.py`, `sim_body.py`, `sim_ring.py`) have no unit tests. `sim` measures 3-8% coverage from incidental imports.
- **Why it matters:** Backplane generation and PDS4 bundle assembly are shipped CLI entry points (`nav_backplanes`, `nav_create_bundle`); `sim` underpins `NavModelBodySimulated`/`NavModelRingsSimulated` and `nav_create_simulated_image`. None is verified.
- **Impact:** Three production subsystems can regress undetected. The 90% line-coverage target in CLAUDE.md is unattainable while these are untested.

#### TEST-STAR-001 — Star-conflict resolution logic is untested
- **File/test:** `tests/nav/nav_model/stars/test_conflicts.py` covers only `parse_ring_occlusion_annuli` and `_conflict_body_list`.
- **Severity:** High.
- **Description:** `src/nav/nav_model/stars/conflicts.py` is 49% covered; lines 154-178, 210-218, 244-261 (the actual conflict-detection: stars occluded by body silhouettes and ring annuli) are never executed.
- **Why it matters:** This logic decides which catalog stars are usable for the star techniques. A bug here silently corrupts the star-nav input set (and the `in_body_silhouette` flag the techniques key off).
- **Impact:** The occlusion gate that protects every star technique is unverified. Add tests that plant a body silhouette and a ring annulus and assert which star ids are flagged conflicting.

### Medium

#### TEST-INFRA-001 — Dataclass `__post_init__` validation guards are untested across feasibility/confidence/confidence_config
- **Files/tests:** `tests/nav/nav_technique/test_feasibility.py` (2 tests, happy-path only — `NavFeasibilityReport.__post_init__` raises at lines 39/43/49/54/59 untested); `tests/nav/nav_technique/test_confidence.py` (`ConfidenceTerm.__post_init__` TypeError/ValueError branches untested, src lines 62-173); `test_confidence_config.py` (same pattern).
- **Severity:** Medium.
- **Description:** These dataclasses carry rich validation (TypeError on non-bool/non-str/non-numeric, ValueError on empty reason, negative count, zero divisor, out-of-range cap). The tests only construct valid instances.
- **Why it matters:** The validation exists precisely to fail loudly on bad config; if a guard regressed (e.g. someone removes the `divisor == 0` check), no test would notice, and a silent divide-by-zero would surface as `inf` confidence downstream.
- **Impact:** ~30+ validation statements unverified. Add `pytest.raises(...)`-with-`match` cases per guard.

#### TEST-CONV-001 — `test_ring_filter.py` uses `caplog` + an injected stdlib logger, violating the capsys/pdslogger convention
- **File/tests:** `tests/nav/nav_model/test_ring_filter.py` lines 13, 115, 192-268 (`test_date_exclusion_logged_at_debug`, `test_radius_exclusion_logged_at_debug`, etc.).
- **Severity:** Medium.
- **Description:** CLAUDE.md mandates pdslogger output be captured via `capsys`, not `caplog`. These tests instead inject a **stdlib** `logging.Logger` into `RingFeatureFilter` (the helper docstring at src `ring_filter.py:114-115` admits it exists "so caplog-based tests can enable DEBUG"). This both uses `caplog` and routes a core nav module through stdlib logging — the exact pattern CLAUDE.md prohibits in core code.
- **Why it matters:** Convention divergence that other modules will copy; also `test_radius_exclusion_logged_at_debug` (line 262) contains a multi-condition assert (`'partial' in r.message and 'outer' in r.message`).
- **Impact:** Test-only stdlib-logging seam in `nav.nav_model`. Convert to pdslogger + `capsys`, or justify the seam explicitly.

#### TEST-CONV-002 — Bare `pytest.raises(...)` without message assertion
- **Files/tests:** `tests/nav/support/test_image.py` lines 17, 69, 94, 129 (`test_shift_array`, `test_pad_array`, ...); `tests/nav/reproj/test_cartographic_model.py:342` (`pytest.raises(AttributeError)`); others in the 23 `pytest.raises` calls lacking `match=` (some legitimately assert on `exc_info` afterward; these do not).
- **Severity:** Medium.
- **Description:** CLAUDE.md requires `pytest.raises` as a context manager *asserting on message content*. Several bare `pytest.raises(ValueError)` blocks assert only the exception *type*.
- **Why it matters:** A bare type assertion passes even if the wrong code path raised the right type for the wrong reason.
- **Impact:** Weak negative tests. Add `match=` substrings.

#### TEST-MODEL-002 — Simulated models verified only at 20-24%
- **Files:** `tests/nav/nav_model/test_nav_model_body_integration.py`, `test_nav_model_rings_integration.py` (integration-gated); no unit path.
- **Severity:** Medium.
- **Description:** `NavModelBodySimulated` (24%) and `NavModelRingsSimulated` (20%) render via `src/nav/sim/` (3-8%). Their render-and-correlate path is integration-only.
- **Why it matters:** Simulated-image navigation (cartographic correlation) is a distinct algorithm from the DT techniques and is essentially unverified offline.
- **Impact:** Couples to TEST-STAGE-001 (sim untested). Build a `sim`-only unit test that renders a synthetic body/ring and asserts pixel-level properties, then a simulated-model test on top.

#### TEST-INT-004 — The single baseline snapshots acknowledged-wrong output
- **File:** `tests/integration/baselines/N1597846115_2_CALIB.json` vs `.../high_phase_terminator/N1597846115_2_CALIB.yaml`.
- **Severity:** Medium.
- **Description:** The baseline pins `(dv, du, conf) = (6.0871, 1.1922, 0.167)`. The sidecar's own `notes` state the ensemble *wrongly* flags this `conflicted` and that the status decision "is wrong" pending Phase-10 calibration. So the regression test locks in current (operator-acknowledged-incorrect) behavior, and the sidecar ground truth `(5.1857, 1.3026)` differs from the baseline by ~0.9 px in dv.
- **Why it matters:** A "regression baseline" that encodes a known bug will flag the *fix* as a regression, creating pressure to keep the bug.
- **Impact:** Baseline semantics conflated with ground truth. Document that baselines are behavioral snapshots distinct from sidecar ground truth, and refresh on intended changes.

### Low

#### TEST-CONV-003 — Isolated multi-condition assert
- **File/test:** `tests/nav/nav_model/test_ring_filter.py:262` — `assert any('partial' in r.message and 'outer' in r.message ...)`.
- **Severity:** Low. Split into two checks (also see TEST-CONV-001).

#### TEST-MODEL-003 — Reliability tests assert only monotonicity / sign, not values
- **File/tests:** `tests/nav/nav_model/test_nav_model_body.py` (`test_limb_reliability_increases_*`, `test_disc_reliability_increases_with_diameter`, etc.).
- **Severity:** Low.
- **Description:** Many reliability tests assert `high > low` or `> 0.5` without pinning the sigmoid output to a value. A miscalibrated-but-monotone sigmoid passes. (The `> 0.30 gate` test is closer to load-bearing.)
- **Impact:** Reliability calibration drift is partly invisible. Add at least one closed-form value assertion per reliability function.

#### TEST-MISC-001 — `tests/support`/hygiene notes
- **Severity:** Low.
- `test_image.py::test_shift_array` performs two unrelated operations (the raise check *and* a subsequent value check) in one function — split per the one-assert/one-behavior convention. Minor.

---

## Cross-cutting themes

1. **Math is well-tested where it is hand-fed synthetic input; untested where
   the code builds that input.** The technique + dt_fitting + ensemble layers
   are exemplary. The model `create_model` layer that *produces* the polylines
   those techniques consume is integration-only. The suite cannot catch a bug
   in the seam between them.
2. **The integration tier carries the suite's only real end-to-end and
   regression signal, but is invisible to the default run.** Anything gated behind `integration` + holdings is, for
   day-to-day development and the default CI job, not protecting anything.
3. **Validation-guard branches (`__post_init__`, input validators) are
   routinely left untested** even where the happy path is covered (feasibility,
   confidence, confidence_config). These guards exist to fail loudly; nothing
   verifies they still do.
4. **Whole shipped subsystems (backplanes, pds4, sim) have no test directory.**
   The stated 90% coverage target is structurally unreachable.
5. **Convention adherence is high but not uniform:** one module reaches for
   stdlib logging + caplog as a test seam; a handful of negative tests assert
   only exception type. RNG is consistently seeded (good); type annotations on
   test functions are present (good).

---

## Fix Prompts

Each prompt is self-contained and references its finding ID.

### FP-1 (TEST-MODEL-001) — Unit-test model feature extraction against the fake backplane
There is a fake backplane shim at `tests/shims/backplane.py` and obs shim at
`tests/shims/obs.py`. Use them to unit-test the `create_model` paths that are
currently integration-only. Add `tests/nav/nav_model/test_nav_model_body_create.py`
and `test_nav_model_rings_create.py`. For the body model: construct a fake obs
whose backplane returns a known spherical body limb (a circle of known center
and radius in image pixels, with known incidence angles), call
`NavModelBody(...).create_model(...)`, and assert the produced `LIMB_ARC`
`NavFeature` has: vertices lying on the expected circle to < 0.5 px, outward
normals pointing radially outward (dot with radial direction > 0.99),
`bbox_extfov_vu` enclosing all vertices, and `sigma_normal_per_vertex` matching
the `_sigma_normal_per_vertex` quadrature formula already tested in
`test_nav_model_body.py`. Do the same for a terminator arc (planted phase
geometry) and for a ring edge in `test_nav_model_rings_create.py` (known radius
projected to a known image curve). Each assertion in its own test. Verify:
`pytest tests/nav/nav_model/test_nav_model_body_create.py -q` passes and
`pytest --cov=nav.nav_model.nav_model_body --cov-report=term-missing` shows
`create_model` lines (currently 303-599) executed.

### FP-2 (TEST-INT-002, TEST-INT-003, TEST-INT-004) — Make the regression/ground-truth tier load-bearing
(a) Populate baselines for all curated sidecars: run the existing
`tests/integration/update_baselines.py` harness with `PDS3_HOLDINGS_DIR` (and
the other env vars in CLAUDE.md) set, producing one JSON under
`tests/integration/baselines/` per sidecar in
`tests/integration/image_library/images/`. Commit them. (b) In
`tests/integration/test_autonomous_nav.py::test_one_library_image`, extend block
(c) so that when a sidecar records a numeric `ground_truth.offset_dv_px/du_px`
*and* `expected.status` is `conflicted` (not just `ok`), the offset is still
checked against `offset_uncertainty_px + 0.5` — gate only the `failed` case out.
(c) Add a comment block (and a README note in
`tests/integration/image_library/README.md`) clarifying that
`baselines/*.json` are behavioral snapshots, distinct from sidecar ground
truth, and must be refreshed via `update_baselines.py` on any intended change.
(d) Add a CI job in `.github/workflows/run-tests.yml` that exports the holdings
env vars and runs `pytest -m integration -n auto --dist=loadfile`. Verify:
with env vars set, `pytest -m integration -q` runs one case per sidecar and one
per baseline and passes.

### FP-3 (TEST-STAGE-001) — Create tests for backplanes, pds4, and sim
Create `tests/backplanes/`, `tests/pds4/`, and `tests/nav/sim/` with `__init__.py`.
For `tests/nav/sim/test_render.py`: render a synthetic disc/sphere via
`src/nav/sim/render.py` / `sim_body.py` for a known geometry and assert image
properties (peak location at expected center, flux monotonic with radius,
terminator on the expected side) — no holdings needed if the sim accepts a
fake obs/geometry; if it requires `oops`, use `tests/shims/`. For
`tests/backplanes/test_backplanes_bodies.py` and `test_backplanes_rings.py`:
feed a fake-backplane obs and assert the generated per-pixel arrays (lat, lon,
incidence) match closed-form values for a sphere/flat-ring at a known geometry.
For `tests/pds4/test_bundle_data.py`: assemble a bundle from a fixture and
assert the LID/LIDVID and label fields. Aim each at >80% coverage of its
target module. Verify: `pytest tests/nav/sim tests/backplanes tests/pds4 -q`
passes and `--cov` shows the target modules well above their current 0-8%.

### FP-4 (TEST-STAR-001) — Test star-conflict resolution
In `tests/nav/nav_model/stars/test_conflicts.py`, add tests for the conflict-
detection functions covering `src/nav/nav_model/stars/conflicts.py` lines
154-178, 210-218, 244-261. Construct a fake obs/backplane (use `tests/shims/`)
with a body silhouette mask and a set of catalog star predictions: assert that
stars whose predicted `(v,u)` fall inside the silhouette are flagged conflicting
(and their `in_body_silhouette` reason set), and stars outside are not. Add a
ring-annulus case using `parse_ring_occlusion_annuli` output: a star whose
radius falls in an occluding annulus is flagged, one outside is not. One assert
per condition. Verify: `pytest tests/nav/nav_model/stars/test_conflicts.py -q`
passes and `--cov=nav.nav_model.stars.conflicts` reaches >85%.

### FP-5 (TEST-INFRA-001) — Test the validation guards
Add negative tests with `pytest.raises(..., match=...)`:
- `tests/nav/nav_technique/test_feasibility.py`: one test per `__post_init__`
  raise in `NavFeasibilityReport` — non-bool `feasible` (TypeError), non-str
  `reason` (TypeError), bool/non-int `consumed_feature_count` (TypeError),
  negative count (ValueError), `feasible=False` with empty `reason` (ValueError).
- `tests/nav/nav_technique/test_confidence.py`: `ConfidenceTerm` guards — non-str
  feature, empty feature, non-numeric/non-finite `alpha`/`offset`/`divisor`,
  `divisor == 0`, `cap_at` non-numeric/non-finite/out-of-[0,1].
- `tests/nav/nav_technique/test_confidence_config.py`: the analogous guards in
  `ConfidenceSpec`/config loaders.
Assert on the exact message substrings already in the source. Verify each
module's coverage rises (feasibility 64%→100%, confidence 83%→~95%).

### FP-6 (TEST-CONV-001, TEST-CONV-002, TEST-CONV-003) — Convention cleanup
(a) In `tests/nav/nav_model/test_ring_filter.py`, convert the `caplog`-based
DEBUG-logging tests (lines 192-268) to capture pdslogger output via `capsys`,
and remove the injected stdlib-logger seam from the test (and, if the only
consumer was tests, from `src/nav/nav_model/rings/ring_filter.py` per CLAUDE.md's
"never import stdlib logging in core nav code"); if the seam must stay,
document why. Split the multi-condition assert at line 262 into two asserts.
(b) Add `match=` to bare `pytest.raises` blocks in
`tests/nav/support/test_image.py` (lines 17, 69, 94, 129) and
`tests/nav/reproj/test_cartographic_model.py:342`, asserting on the real
message substring. (c) Split `test_shift_array`/`test_pad_array` so the raise
check and the value check are separate test functions. Verify:
`ruff check tests` clean and the affected files pass.

### FP-7 (TEST-MODEL-003) — Pin reliability values, not just monotonicity
In `tests/nav/nav_model/test_nav_model_body.py`, augment the reliability tests
(`_limb_reliability`, `_disc_reliability`, `_terminator_reliability`,
`_blob_reliability`): in addition to the existing monotonicity assertions, add
one closed-form value assertion per function for a representative input (compute
the expected sigmoid output from the documented coefficients and assert
`== pytest.approx(expected, rel=1e-9)`). This makes a calibration change to any
coefficient fail a test. Verify: `pytest tests/nav/nav_model/test_nav_model_body.py -q` passes.
