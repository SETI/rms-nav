# RMS-NAV Source Code Critique

A full-depth correctness and quality critique of every Python source file under
`src/` in the rms-nav repository (branch `core_rewrite_phase10`). No file and no
line was skipped. Special attention was paid, as required, to the correctness of
the navigation models and navigation techniques: all mathematics and optimization
machinery was re-derived and checked against reference formulations for
correctness, thoroughness, configurability, and appropriateness to the problem.

## How this critique was produced

The review was partitioned across eight independent passes so every subsystem was
read in full:

- A deep mathematics pass that read every math-critical model, technique, and
  shared-fitting file in full (`dt_fitting`, `ensemble`, `confidence`,
  `predicted_snr`, `smeared_psf`, `_star_helpers`, all body/ring/star techniques,
  `nav_model_body`, `ring_math`, `distance_transform`, `correlate`) and re-derived
  each formula.
- Seven subsystem passes, each reading every file in its scope at full depth:
  image-reading (`obs`) and dataset enumeration; reprojection, backplanes, and
  PDS4; feature, annotation, simulator, config, util, and experiments; the
  `support` shared-infrastructure helpers; the navigation orchestrator; the CLI
  drivers and PyQt6 UI; and a non-math coverage sweep of the model and technique
  subpackages.

Findings carry stable IDs by subsystem (`CODE-NAV-*`, `CODE-OBS-*`, `CODE-DS-*`,
`CODE-REPROJ-*`, `CODE-BACKPLANE-*`, `CODE-PDS4-*`, `CODE-ORCH-*`, `CODE-DERIV-*`,
`CODE-SUPPORT-*`, `CODE-CFG-*`, `CODE-SIM-*`, `CODE-FEAT-*`, `CODE-ANNO-*`,
`CODE-MAIN-*`, `CODE-UI-*`, `CODE-MODEL-*`, `CODE-TECH-*`, `CODE-EXP-*`).

## How to read this document

Part 1 is the deep mathematics and core critique; it also contains a brief
preliminary scan of the peripheral subsystems, which Parts 2 through 8 then
supersede and expand with the exhaustive per-subsystem passes. All findings come
first (Parts 1 through 8). Every fix prompt is collected at the end under
**Consolidated Fix Prompts**, ordered to match the parts; each prompt is
self-contained and references its finding ID.

## Severity rollup

| Part | Subsystem | Findings | Critical | High |
|---|---|---:|---:|---:|
| 1 | Navigation models, techniques, and core math (+ preliminary peripheral scan) | 41 | 3 | 11 |
| 2 | Image reading (`obs`) and dataset enumeration | 47 | 0 | 7 |
| 3 | Reprojection, backplanes, and PDS4 | 11 | 2 | 1 |
| 4 | Feature, annotation, simulator, config, util, experiments | 21 | 0 | 1 |
| 5 | Support (shared infrastructure) | 18 | 0 | 0 |
| 6 | Navigation orchestrator | 15 | 0 | 1 |
| 7 | CLI drivers and PyQt6 UI | 16 | 1 | 1 |
| 8 | Model/technique non-math coverage gaps | 21 | 0 | 0 |
| **Total** | | **190** | **6** | **22** |

The remaining 162 findings are Medium and Low severity, detailed in their parts.

## Critical findings (all six)

1. **CODE-NAV-001** (`dt_fitting.lm_subpixel_refine`) — On total non-convergence
   (all robust weights driven to zero) the refiner can report `rms_px == 0.0`.
   Every DT technique's spurious-rejection gate reads a zero RMS as a perfect fit,
   so a completely failed fit is accepted as a clean navigation. False-positive
   offsets with high confidence.
2. **CODE-NAV-002** (body limb/terminator `_build_polyline_sampler`) — Vertex
   normals are derived from the one-pixel ridge mask rather than the continuous
   backplane, so the polarity-filter sign that decides in/out is effectively
   arbitrary. This is the plausible root cause of the documented limb
   mis-convergence that the trust-region and Tikhonov patches work around.
3. **CODE-NAV-003** (star `predicted_snr`) — The Cassini predicted-SNR path
   multiplies by `signal_dn_to_image_unit_scale`, whose value is a `# PLACEHOLDER`.
   That single uncalibrated constant linearly gates whether any star feature
   survives across the entire calibrated holdings set.
4. **CODE-BACKPLANE-001 / CODE-REPROJ** (`src/backplanes/backplanes.py`) — The
   backplane driver skips any image whose nav metadata `status != 'success'`, but
   `NavResult.status` is `'ok'`. Backplane generation therefore skips every
   successfully navigated image; the downstream stage is inert on real data and no
   test covers it.
5. **CODE-PDS4-001** (`src/pds4/bundle_data.py`) — The same `status != 'success'`
   gate (correct value is `'ok'`) makes PDS4 bundle assembly skip every
   successfully navigated image.
6. **CODE-MAIN-001** (`nav_backplanes_cloud_tasks`, `nav_create_bundle_cloud_tasks`)
   — Both register a bare `async def main` as the setuptools console-script entry
   point, so the installed commands return an un-awaited coroutine and exit
   immediately without ever starting the worker. Backplane and PDS4 cloud batch
   processing are non-functional through the installed entry points.

## Selected High-severity findings

- **CODE-NAV-004 / CODE-NAV-005** — StarField and BodyBlob covariances use
  `Sum(w*r^2) / (Sum w)^2` (wrong power of the weight sum, no degrees-of-freedom
  factor), mis-scaling the sigmas the ensemble and confidence tiers consume.
- **CODE-NAV-006** — DT spurious-gates test the Tukey-weighted RMS, which by
  construction suppresses the outliers the gate exists to detect; only
  `RingEdgeNav` added a raw-residual check.
- **CODE-NAV-007** — `coarse_ncc_search` is a raw overlap count, not a normalized
  cross-correlation; its argmax is not stable under edge-density variation or
  boundary clipping, biasing the seed the LM stage anchors to.
- **CODE-NAV-010** — `BodyDiscCorrelateNav` rotation sigma is dimensionally
  `rad^2 / quality^2`, not `rad^2`; rotation uncertainty is on an arbitrary scale.
- **CODE-ORCH-003** — Calibrated-instrument `NaN` missing-data markers flow
  uncleaned into the finite-only derivative computation, raising an uncaught
  `ValueError` out of `navigate()` and violating the no-raise contract, while the
  classifier's missing/blank detection is dead-coded for calibrated images.
- **CODE-CFG-1** — `update_config` shallow-merges, so a user override of one nested
  key clobbers the sibling defaults under the same section.
- **CODE-MAIN-002** — The simulated-image GUI drops `shade_solid_rings` on load and
  can crash on a missing or null `closest_planet` via `QComboBox.findText(None)`.
- **obs / dataset Highs** — `star_psf_size` loop-variable leak with a mistyped
  return; `closest_planet` SPICE failures raised at read time; Voyager
  spacecraft / I-over-F correction keyed off an unvalidated single label character
  that rescales pixels by 3.345x; `choose_random_images` biased and prone to
  livelock under filters; a possibly-unbound volume index; and Cassini BOTSIM
  grouping that silently drops unpaired frames and mis-pairs via a bogus
  three-second interpretation of the image number.

## Cross-cutting themes

1. **Covariances are Cramer-Rao lower bounds presented as 1-sigma.** This is the
   highest-leverage fix: the per-technique covariances are optimistically tight,
   which forces the ensemble's 5 px grouping floor (discarding directional
   information) and distorts every confidence tier downstream.
2. **Robust statistics misused as failure detectors.** Tukey-weighted residuals
   are used in the very gates meant to catch outliers, so the gates are blind to
   the failures they target.
3. **Geometry derived from discrete masks instead of the continuous backplane.**
   Limb and terminator normals come from a one-pixel ridge mask, making sign and
   sub-pixel placement unreliable.
4. **Status-string contract drift.** Navigation emits `ok` / `failed` /
   `conflicted`, but the backplane and PDS4 stages gate on `success`, silently
   disabling the entire downstream pipeline. The same drift class appears in the
   test suite's Voyager `raw_dn` vs `calibrated_if` config mismatch.
5. **Broad `except Exception` in correctness-critical paths.** Orchestrator, obs,
   and CLI drivers swallow both the deliberate `RuntimeError` contract guards and
   genuine bugs, turning hard failures into silent wrong answers.
6. **Uncalibrated placeholder coefficients in live math.** The star signal scale
   and several ring-reliability constants ship as placeholders yet drive
   feature-survival and reliability decisions.
7. **Convention deviations.** stdlib `logging` instead of pdslogger across the
   reprojection modules; oversized modules and large dead-code blocks
   (`flux.py` is 1174 lines of almost entirely commented-out code;
   `nav_create_simulated_image.py` is roughly 2500 LOC).
8. **Heavy duplication.** Per-instrument `from_file` / `__init__` / PDS4 hooks, the
   CLI driver preamble copied five times, and the zoom/slider/alpha-blend helpers
   duplicated across UI windows.
9. **Unguarded angle arithmetic.** Rotation is averaged as a Euclidean coordinate
   without wrap, and the ensemble's Mahalanobis null-space test uses a fixed
   absolute tolerance that is brittle to offset scale.

---

## Remediation status

Tracks fixes applied against this critique. "FIXED" items are on the
`core_rewrite_phase10` branch, verified by the noted tests. "PARTIAL" means the
core defect is fixed but some follow-up (noted) is deferred.

| Finding | Title | Status (2026-06-10) | Notes |
|---|---|---|---|
| CODE-NAV-001 | `lm_subpixel_refine` reports a clean fit on total non-convergence | FIXED | `LMRefineResult.degenerate` added; `rms_px=inf` + all-inf covariance when every weight is rejected; limb / terminator / ring-edge spurious gates treat `degenerate` as spurious. 238 technique tests pass. Covariance documented as data-only (excludes the Tikhonov anchor). |
| CODE-NAV-002 | Limb/terminator normals from the ridge mask, arbitrary sign | FIXED | Normals now computed from the silhouette mask (limb) / lit mask (terminator); outward-normal regression test added. |
| CODE-MAIN-001 (Part 7, async entry point) | `nav_backplanes_cloud_tasks` / `nav_create_bundle_cloud_tasks` register `async def main` | FIXED | Both now expose a sync `main()` that runs `asyncio.run(async_main())`; verified `main` is no longer a coroutine. |
| CODE-BACKPLANE-001 + CODE-PDS4-001 | Backplane / PDS4 stages skip every navigated image (status literal) | FIXED | `NavResult.status` standardized on `'success'` (the value the two stages already check). Producer, consumers, sidecar validator, 8 sidecars, and 8 test files updated; unrelated `reason='ok'` / `StatusReason.OK` / cloud-tasks worker protocol left intact. |
| CODE-NAV-003 | Star predicted-SNR depends on a placeholder DN-to-I/F scale | FIXED | Replaced the DN-based SNR gate with a magnitude gate on the existing `obs.star_max_usable_vmag()`; covariance/reliability now use a magnitude-margin pseudo-SNR (`SNR_REF=8.0`, `SNR_FLOOR=0.1`); `signal_dn_to_image_unit_scale` removed from `InstrumentSettings`/`NavContext`/configs (placeholders gone); `predicted_snr.py` retained only as a raw-DN diagnostic. 503 nav tests pass. Follow-up: full CISSCAL photometry + non-Cassini limiting magnitudes. The general placeholder-scanning guard (CODE-CFG-001) remains tracked by issue #118. |
| CODE-CFG-1 | `update_config` shallow merge clobbers nested sibling overrides | FIXED | Added `_deep_merge`; nested user overrides combine key-by-key while preserving sibling defaults. 4 new tests pass. |
| CODE-MAIN-002 | Sim-GUI drops `shade_solid_rings`, crashes on null `closest_planet` | FIXED | `_load_parameters` now preserves `shade_solid_rings` and re-syncs the checkbox; `closest_planet` falls back to `'SATURN'` before the combo update. 6 GUI tests pass headless. |
| CODE-OBS-001 | `star_psf_size` loop-variable leak / mistyped fallthrough | FIXED | Rewritten to sort thresholds, use an explicit `default_mag`, raise `ValueError` on empty, and validate/coerce a 2-tuple. 5 new tests. |
| CODE-OBS-011 | Voyager I/F correction keyed off one unvalidated label char | FIXED | `_voyager_spacecraft_digit` validates `LAB02` (str, len>=5, char='1'/'2'); `_voyager_if_factor` guards the `LABEL3` parse. Formats confirmed against real holdings and `oops`; 10 new tests. (CODE-OBS-012 fixed in the same change.) |
| CODE-DS-001 / 002 / 010 | Unbound idx; biased random sampling; BOTSIM mis-pair / frame loss | FIXED | DS-001: indices pre-initialized for mypy. DS-002: random path samples a filtered pool per volume, bounded by the volume set (no livelock/bias). DS-010: never drops a frame; pairs only on opposite camera + same `OBSERVATION_ID` + `IMAGE_TIME` within 2.0 s (not the image-number-as-seconds heuristic). 4 new tests. |
| CODE-NAV-008 / 009 | Ensemble angle-wrap; absolute null-space tolerance | FIXED | NAV-008: rotation combined as a circular mean with a `max_allowed_rotation_deg=5.0` config field + small-angle assertion (translation merge unchanged). NAV-009: null-space test now relative (`rel_tol * max(‖delta‖, eps)`). 29 ensemble tests pass. |
| CODE-ORCH-003 | NaN markers crash `navigate()` for calibrated images | FIXED | `_make_context` sanitizes NaN/marker pixels to a finite fill before the derivative kernels and threads the true `missing_frac` into the classifier (`np.isnan`/`np.nanmax`); calibrated frames now fail gracefully and missing-data detection works. 5 new tests. |
| CODE-NAV-004 / 005 | Weighted-mean covariance: wrong power of `Σw`, no DOF factor | FIXED | `StarFieldFromCatalogNav._build_covariance`/`_build_covariance_3dof` and `BodyBlobNav._joint_covariance` now use the reduced-chi-square form `Var(axis)=max(χ²_ν/Σw, 1/Σw)` with `χ²_ν=Σ(w·r²)/max(N−p,1)` (floor is `1/Σw`, not `1/(Σw)²`). Single-blob collapses to the inverse-precision floor rather than over-confidence. Analytic-value tests added. |
| CODE-NAV-010 | `BodyDiscCorrelateNav` rotation sigma has wrong dimensions | FIXED | `_rotation_sigma_from_quality` returns `None` (rotation-unobservable) rather than a dimensionally-wrong value; the caller sets `cov[2,2]=ROTATION_UNOBSERVABLE_VARIANCE` and the ensemble maps it to ~zero rotation information. |
| CODE-ORCH-001 | Ensemble 5-px floor compensates for over-tight covariances | PARTIAL | Root cause (over-confident covariances) fixed via NAV-004/005. Added a `model_error_floor_px` config knob (default 0.0) on `StarFieldFromCatalogNav`/`BodyBlobNav` so covariance is CRLB ⊕ model-error. Deferred (needs the integration library to calibrate/verify): the same hook on limb/ring-edge/disc (shared fitting modules) and shrinking the ensemble `agreement_pixel_floor`. |
| CODE-NAV-006 | DT spurious gate uses Tukey-weighted RMS (hides outliers) | FIXED | Added `LMRefineResult.raw_rms_px` (unweighted `sqrt(mean(r²))` over all vertices); limb and terminator spurious gates now OR-in `raw_rms_px > max(spurious_dt_floor_px, spurious_dt_rms_factor·sigma_min)`, mirroring RingEdgeNav. A Tukey-masked bad arc is now flagged spurious even when the weighted RMS is small (mutation-verified test). |
| CODE-NAV-007 | `coarse_ncc_search` is a raw overlap count, not an NCC | FIXED | The coarse seed now divides each shift's overlap by the in-bounds vertex count (`sv.size`), so the argmax is the true binary-NCC argmax (`NCC=sqrt(overlap/N_inbounds)`) instead of favouring shifts that keep more vertices in bounds or cover a denser edge region. Docstring corrected (dropped the false "count argmax == NCC argmax" claim); regression test on a boundary-clipping fixture where raw-count argmax `(-1,0)` differs from NCC argmax `(-2,0)`. |

ID-collision caveat: the labels `CODE-MAIN-001` and `CODE-PDS4-001` also tag
*different* findings in Part 1's preliminary peripheral scan (an oversized module
and a bare-except cluster, respectively); those remain open. The rows above refer
to the detailed-pass findings in Parts 3 and 7.

---

# Findings


---

# Part 1 — Navigation Models, Techniques & Core Math (deep dive + preliminary peripheral scan)

## nav_technique

The math-correctness findings lead, per the request.

### Critical

#### CODE-NAV-001 — LM trial-acceptance freezes Tukey weights but the *final* covariance/RMS recompute does not match the accepted step's weights consistently
**File:** `src/nav/nav_technique/dt_fitting.py` · `lm_subpixel_refine`
**Severity:** Critical (subtle; needs confirmation by test)

The inner LM loop correctly freezes the Tukey weights across a single
Gauss-Newton trial (lines 764-778) — that is the standard IRLS/LM separation and
is right. However, two interacting issues remain:

1. The **information matrix used for the reported covariance** (lines 853-877)
   is built from `state.jacobian` and `state.weights`, which on an accepted
   final step are recomputed at the accepted pose (lines 803-819) — good — but
   on a *rejected* final iteration (loop exits via `max_iterations` after a
   rejection) `state.jacobian/weights` reflect the *pre-trial* pose evaluated at
   the *start* of that iteration (lines 685-704), which is the last accepted
   pose. That is consistent. The genuine hazard is the `best_cost == inf`
   fallback path (lines 828-852): it re-evaluates residuals from
   `state.raw_residuals` but only recomputes the Jacobian, never re-checking that
   `state.weights` corresponds to the same residuals — if the very first
   iteration produced all-zero weights and broke at line 705-706, the covariance
   branch at line 867 correctly returns `inf`, but the `rms_px` at line 856-859
   is computed from `final_weights` that may be all-zero, returning `0.0` — a
   *zero* RMS for a fit that never converged. A zero RMS is then read by the
   technique spurious-gates (`result.rms_px > floor`) as a *good* fit.

2. The Tikhonov term is added to `cost_before`/`trial_cost` (lines 724-726,
   779-784) but **not** to the information matrix used for the final covariance.
   The anchored LM minimizes `C + α·Σw·||Δ||²`, so the parameter Hessian at the
   optimum is `JᵀWJ + α·Σw·I` on the translation block; reporting
   `pinvh(JᵀWJ)` (without the Tikhonov diagonal) *over-states* the translation
   uncertainty relative to what the anchored objective actually constrains. This
   is arguably defensible (you want the data-only covariance) but it is
   undocumented and inconsistent with the cost that was minimized.

**Why it is risky:** a non-converged or all-rejected fit can report `rms_px ==
0.0`, which every DT technique (`BodyLimbNav`, `BodyTerminatorNav`,
`RingEdgeNav`) treats as a *clean* fit in its `spurious` test
(`result.rms_px > max(floor, factor·sigma_min)`), so a total failure can pass
through as a high-confidence offset. **Impact:** false-positive navigation on
degenerate inputs. **Confirm by** adding a unit test that feeds a polyline with
all vertices polarity-rejected and asserting the result is flagged spurious and
covariance is `inf` (it is) *and* `rms_px` is not `0.0` (it currently is).

#### CODE-NAV-002 — Body limb/terminator polyline normals are computed from the *ridge mask*, not the body interior; polarity sign is effectively arbitrary
**File:** `src/nav/nav_model/nav_model_body.py` · `_build_polyline_sampler` (lines 993-1009)
**Severity:** Critical

The "outward normal" at each limb vertex is derived by checking whether the
neighbouring pixel **of the 1-pixel-wide limb ridge mask** (`local_mask`) is
False:

```python
if v > 0 and not local_mask[v - 1, u]:
    v_dir = -1.0
elif v < rows - 1 and not local_mask[v + 1, u]:
    v_dir = 1.0
```

For a thin diagonal ridge, *both* vertical neighbours are off-ridge (False), so
`v_dir` is set to `-1.0` purely because the `-1` branch is tested first — the
sign does **not** track which side is the body interior versus space. The normal
should be the gradient of the **body silhouette mask** (`body_mask_valid` /
`is_lit`), where the body-side neighbour is True and the space-side neighbour is
False, giving a genuine inside→outside direction.

The DT techniques (`BodyLimbNav`, `BodyTerminatorNav`) then **negate** these
normals (`-feat.geometry.normals_vu`) and feed them to `polarity_filter`, which
keeps a vertex only when `dot(model_normal, image_gradient) > 0`. If the model
normal sign is arbitrary, the polarity filter rejects roughly half the limb
vertices for no physical reason — and worse, can systematically reject the
*correct* lit-limb vertices and keep the wrong ones, biasing the LM toward
crater rims / the terminator (exactly the failure mode the trust-region and
Tikhonov terms were bolted on to suppress, see CODE-NAV-001 / CODE-NAV-006).

**Why it is wrong:** the normal must point from inside the body to outside
(space); only then does `dot(-normal, gradient) > 0` correspond to "image is
brighter inside the limb." Using the ridge mask makes the direction depend on
the diagonal orientation of the discrete ridge, not the geometry. **Impact:**
weakened or inverted polarity gating on every body limb/terminator fit; this is
plausibly the root cause of the documented mis-convergence cases
(N1574928113, N1572471790) that the trust-region / lit-side filtering patches
work around. **Confirm by** rendering a synthetic disc, extracting the limb
polyline, and asserting `dot(normal_i, (vertex_i - body_center)) > 0` for every
vertex — it will fail for a meaningful fraction today.

#### CODE-NAV-003 — Star predicted-SNR uses an admitted PLACEHOLDER scale that gates whether *any* star feature survives on calibrated Cassini images
**Files:** `src/nav/nav_model/stars/predicted_snr.py` · `predicted_snr`; `src/nav/config_files/config_400_inst_coiss.yaml` (lines 116, 151)
**Severity:** Critical (correctness of reliability gate; config)

`predicted_snr` converts image noise to DN via
`image_noise_sigma_dn = image_noise_sigma / signal_dn_to_image_unit_scale`. For
calibrated Cassini NAC/WAC the config value is
`signal_dn_to_image_unit_scale: 5.0e-7  # PLACEHOLDER — calibrate in Phase 10`.
The branch is in the current branch name (`phase10`), so this is the moment it
should be calibrated, and it is not. Because this scale enters as
`sigma_dn = sigma_IF / 5e-7 = sigma_IF · 2e6`, an order-of-magnitude error in
the placeholder moves the predicted SNR (and therefore the
`predicted_snr`-driven reliability gate, the brightness-margin floor in
`StarUniqueMatchNav`, and the per-feature position covariance) by the same order
of magnitude. The module docstring even states "Without this conversion the SNR
for every catalog star collapses to `sqrt(signal_dn)` ... and the reliability
gate drops them all" — i.e. the gate's behaviour is dominated by this one
uncalibrated number.

**Why it is wrong/risky:** an uncalibrated coefficient that linearly controls a
hard feasibility gate is a correctness defect, not a tuning nicety. With a wrong
scale, star techniques are silently enabled/disabled on the entire Cassini
calibrated holdings. **Impact:** star-based navigation may be globally
suppressed or globally over-trusted on the primary mission dataset. **Confirm by**
deriving the true DN→I/F factor from the Cassini calibration pipeline (RADIANCE
/ I-over-F conversion factor per camera/filter/gain) and replacing the
placeholder; add a config-validation test that *fails* if any
`signal_dn_to_image_unit_scale` value still carries a `PLACEHOLDER` marker.

### High

#### CODE-NAV-004 — `StarFieldFromCatalogNav._build_covariance`: wrong power of `Σw` in the translation-mean covariance
> **Tracked by:** #123 — Mahalanobis agreement grouping breaks because per-technique covariances are CRLB-tight
**File:** `src/nav/nav_technique/nav_technique_star_field.py` · `_build_covariance` (lines 881-901)
**Severity:** High

```python
var_v = sum(w * r_v**2) / total        # weighted residual variance
cov_v = max(var_v / total, floor)      # divides by total AGAIN
```

So `cov_v = Σ(w·r²) / (Σw)²`. For the variance of a weighted mean with
inverse-variance weights `w_i = 1/σ_i²`, the correct estimator is
`Var(mean) = 1/Σw` scaled by the reduced chi-square
`χ²_ν = Σ(w·r²)/(N − p)`, i.e. `Var ≈ χ²_ν / Σw = [Σ(w·r²)/(N−1)] / Σw`. The
code uses `Σ(w·r²)/(Σw)²`, which for uniform weights `w=1` gives `Σr²/N²` =
`Var/N` (coincidentally correct for the unit-weight mean), but for genuinely
non-uniform inverse-variance weights it is **not** `χ²_ν/Σw` — it lacks the
`1/(N−p)` degrees-of-freedom factor and double-counts `Σw`. The same pattern is
repeated in the 3-DoF variant (`_build_covariance_3dof`, lines 938-944) and in
`BodyBlobNav._joint_covariance` (lines 287-289, `total_weight_sq`).

**Why it is wrong:** the reported per-axis sigma is mis-scaled by a factor that
depends on the weight distribution and the inlier count. **Impact:** the
ensemble's information-form merge (`_combine_precision_weighted`) and the
confidence tiers (`derive_confidence_rank`, `max_sigma_px`) consume these sigmas
directly; a mis-scaled sigma changes which results group, which group wins, and
the final confidence tier. **Confirm by** a unit test with two stars at known
σ and a known residual, comparing the reported sigma against the analytic
weighted-mean sigma.

#### CODE-NAV-005 — `BodyBlobNav._joint_covariance`: same `(Σw)²` mis-normalization plus a hidden single-blob over-confidence
> **Tracked by:** #123 — Mahalanobis agreement grouping breaks because per-technique covariances are CRLB-tight
**File:** `src/nav/nav_technique/nav_technique_body_blob.py` · `_joint_covariance` (lines 257-290)
**Severity:** High

Two issues:
1. `var_v = Σ(w·r²)/(Σw)²` — same defect as CODE-NAV-004.
2. The single-blob branch (`offsets_v.size <= 1`) returns `floor·I` where
   `floor = 1/Σw`, and `w = N_lit·SNR²/R²`. For a bright, large body this `Σw`
   can be enormous, so the reported sigma is sub-milli-pixel. A
   brightness-weighted centroid on a real body is good to ~0.1–1 px at best
   (limb softness, phase bias, albedo). Advertising sub-mpx certainty makes the
   blob result dominate the ensemble information-form merge even though its
   `confidence` is capped at 0.4 — confidence and covariance are combined
   independently downstream, so a tiny covariance wins the precision-weighted
   average regardless of the 0.4 confidence cap.

**Why it is risky:** the CRLB weight is a *lower bound* on variance, not a
realistic 1σ; using `1/Σw` as the actual covariance under-states uncertainty by
orders of magnitude — the exact problem the ensemble's `agreement_pixel_floor`
exists to paper over (see CODE-ORCH-001). **Impact:** blob results can hijack the
precision-weighted merge. **Confirm by** logging `combined.covariance` for a
scene where a blob and a star field both fire; the blob's tiny covariance will
dominate `mu_combined`.

#### CODE-NAV-006 — DT-fit "spurious" gates can be defeated by Tukey weight collapse; `rms_px` is Tukey-weighted, not raw
> **Tracked by:** #125 — BodyTerminatorNav mis-convergence has no per-technique signal, #128 — Architectural redesign: robust limb navigation across all body types and illuminations
**Files:** `nav_technique_body_limb.py` (lines 335-344), `nav_technique_ring_edge.py` (lines 304-316), `nav_technique_body_terminator.py`
**Severity:** High

`result.rms_px` is the **Tukey-weighted** RMS (`dt_fitting.py` line 857:
`sqrt(Σ(w·r²)/Σw)`). When the LM converges to a pose where one sub-arc fits
cleanly and the rest are grossly misaligned, Tukey zeroes the misaligned
vertices and `rms_px` collapses toward zero — so the primary spurious test
`rms_px > floor` *passes* a bad fit. `RingEdgeNav` already recognises this and
adds a raw `per_edge_rms_summed` check (lines 287-316); `BodyLimbNav` and
`BodyTerminatorNav` do **not** — they rely on `inlier_count`,
`inlier_fraction`, and `lm_displacement` only, none of which catches "one
clean arc, everything else rejected, low displacement." 

**Why it is wrong:** the robust RMS is the wrong statistic for a *mis-convergence*
detector precisely because robustness hides the outliers you are trying to
detect. **Impact:** body-limb mis-convergence onto a partial arc can pass as a
confident fit. **Fix:** mirror the `RingEdgeNav` raw per-feature RMS check in the
limb/terminator techniques, or report a raw (unweighted) RMS alongside the
robust one and gate on the raw value.

#### CODE-NAV-007 — `coarse_ncc_search` is documented as NCC but is a raw count; the "argmax unchanged" claim is false in general
> **Tracked by:** #128 — Architectural redesign: robust limb navigation across all body types and illuminations
**File:** `src/nav/nav_technique/dt_fitting.py` · `coarse_ncc_search` (lines 106-203)
**Severity:** High (correctness of the seed that the whole LM trusts)

The function computes `Σ polyline_mask · edge_mask[shifted]` — a raw overlap
count — and the docstring argues this equals the NCC argmax because the
normalizer `sqrt(|polyline|·|edge|)` is "constant in polyline and varies only
mildly with edge over the small window." That is not true when the model
polyline shifts off the image boundary (in-bounds vertex count drops, lines
182-191 explicitly drop out-of-bounds vertices) or when the image edge density
varies across the search window (it does, near a bright limb). A raw count
**systematically prefers shifts that place the polyline over the densest edge
region**, which is not the same as best alignment. Because the LM is seeded from
this integer offset and the trust-region (CODE-NAV-001) then *anchors* the LM to
this seed, a biased seed becomes a biased final answer.

**Why it is risky:** the entire DT pipeline's robustness rests on the coarse seed
being within a pixel of the true alignment; a count-vs-NCC discrepancy of a few
pixels defeats the trust region. **Impact:** systematic offset bias on textured
or partially-out-of-frame bodies. **Confirm by** comparing the count-argmax
against a true per-shift NCC argmax on a fixture where edge density is
non-uniform; they will differ. **Fix:** either rename the function and document
it honestly as a chamfer-count seed (and widen the trust region accordingly), or
divide each shift's score by the in-bounds vertex count so density bias cancels.

#### CODE-NAV-008 — `_combine_precision_weighted` averages parameter vectors that include an angle without any wrap handling
**File:** `src/nav/nav_orchestrator/ensemble.py` · `_combine_precision_weighted` (lines 342-414); `_result_param_vector`
**Severity:** High

For 3-DoF results the parameter vector is `(dv, du, rotation_rad)` and the
combine does a linear information-weighted average `Σµ`. Rotation is an angle;
two results near `+π` and `−π` average to ~0 instead of `±π`. The `at_edge`
fractions clamp rotation to `±max_rotation_deg` (typically ±5°), so wrap is rare
in practice — but the Mahalanobis grouping (`_mahalanobis_distance`) and the
combine both treat the angle as a Euclidean coordinate, and nothing documents or
enforces the small-angle assumption. For VGISS/GOSSI where rotation is fit, a
20° pointing twist (within some `max_rotation_deg` configs) is enough for the
linear average to bias the combined angle.

**Why it is risky:** silent angle-wrap bias when fitting camera rotation.
**Impact:** wrong combined rotation for multi-technique 3-DoF scenes. **Fix:** at
minimum assert `|rotation_rad| < max_rotation_deg` on entry; properly, combine
rotations on the circle (atan2 of weighted sin/cos) or document and enforce the
small-angle regime.

#### CODE-NAV-009 — `_mahalanobis_distance` null-space test uses a fixed absolute tolerance `1e-6` against an un-normalized residual
**File:** `src/nav/nav_orchestrator/ensemble.py` · `_mahalanobis_distance` (lines 110-138)
**Severity:** High

The null-space rejection is
`if norm(delta − cov_sum·pinv·delta) > 1e-6: return inf`. `delta` is a pixel (or
radian) displacement; `1e-6` is an absolute pixel threshold with no relation to
the magnitude of `delta` or the scale of `cov_sum`. When two rank-1
(flat-ring-edge) results disagree by, say, 3 px along the unobservable axis, the
residual norm is ~3, correctly returning `inf`; but when two well-conditioned
results have a `cov_sum` with one eigenvalue at `~1e-7` (near but not exactly
rank-deficient — common after `pinvh` with `rtol=1e-9`), the projection residual
can be `~1e-5` and the pair is declared infinitely distant *spuriously*,
breaking a grouping that should have succeeded. The threshold should be relative
to `norm(delta)`.

**Why it is risky:** grouping is the pivot of the whole ensemble; a brittle
absolute tolerance makes grouping depend on the absolute pixel scale of the
offset. **Impact:** intermittent failure to group agreeing results → spurious
"conflicted" verdicts. **Fix:** use `norm(null_proj) > rel_tol · max(norm(delta),
eps)`.

#### CODE-NAV-010 — `_rotation_sigma_from_quality` covariance has wrong dimensions
> **Tracked by:** #123 — Mahalanobis agreement grouping breaks because per-technique covariances are CRLB-tight
**File:** `src/nav/nav_technique/nav_technique_body_disc.py` · `_rotation_sigma_from_quality` (lines 692-751)
**Severity:** High

`sigma_sq = 1.0 / (-second_deriv · q_centre)`. `second_deriv` has units
`quality / rad²`; multiplying by `q_centre` (units `quality`) gives
`quality²/rad²`, so `sigma_sq` has units `rad²/quality²` — **not** `rad²`. The
standard curvature→variance relation for a log-likelihood-like quality surface is
`σ² = 1/(−d²ℓ/dθ²)`, i.e. just `1/(-second_deriv)` (units `rad²` only if quality
is the log-likelihood). Dividing additionally by `q_centre` is dimensionally
inconsistent and makes the reported rotation sigma depend on the absolute NCC
quality scale in a way that has no statistical meaning. The doc even admits
"Quality is in the same scale as PSR/PMR" — PSR/PMR are dimensionless ratios, not
a log-likelihood, so the curvature→variance identity does not apply at all
without a calibration factor.

**Why it is risky:** the rotation covariance entering the 3-DoF ensemble is on an
arbitrary scale. **Impact:** rotation uncertainty for `BodyDiscCorrelateNav` is
not trustworthy; it can over- or under-constrain the combined rotation. **Fix:**
derive σ_θ from a proper model (e.g. relate NCC-peak curvature to the
Cramér-Rao bound via the per-pixel noise and template energy) or treat rotation
as unobservable for the correlation technique until calibrated.

### Medium

#### CODE-NAV-011 — Tikhonov/covariance inconsistency (see CODE-NAV-001 item 2), promoted as its own finding
**File:** `dt_fitting.py` · `lm_subpixel_refine`
The reported translation covariance excludes the Tikhonov diagonal that was part
of the minimized objective. Decide and document whether the covariance is the
data-only or the anchored-objective covariance, and make the code match the
stated choice.

#### CODE-NAV-012 — `brightness_margin_mag` assumes background-limited (linear) SNR but `predicted_snr` is shot+read (sub-linear)
**File:** `src/nav/nav_technique/_star_helpers.py` · `brightness_margin_mag` (lines 90-114)
The `Δmag = 2.5·log10(s1/s2)` identity holds only when SNR ∝ flux
(background-limited). `predicted_snr` (CODE-NAV-003) includes a `+ total_signal`
shot term, so for bright stars SNR ∝ √flux and the implied Δmag is
double-counted/under-counted. For the uniqueness gate this biases the
brightness-margin in the bright regime. Medium because the gate has slack, but
the formula should consume `integrated_signal_dn` directly (true flux ratio)
rather than the SNR ratio.

#### CODE-NAV-013 — `_combine_confidence` agreement boost `1 + 0.5·log2(n)` is unbounded-in-spirit and mixes confidence with precision weighting opaquely
**File:** `src/nav/nav_orchestrator/ensemble.py` · `_combine_confidence` (lines 417-467)
The "number of significant contributors" boost uses `trace(pinvh(Σ))` as the
significance weight — but `trace` of the information matrix mixes the v, u, and θ
precisions on different scales (px⁻² and rad⁻²), so a 3-DoF result's trace is
dominated by whichever axis happens to have the smallest variance. The
significance count `n_significant` and the weighted confidence average are
therefore both skewed by the rotation axis when present. Use per-axis-normalized
weights or `det(info)^(1/p)` instead of raw trace.

#### CODE-NAV-014 — `similarity_transform_fit` weighted-centroid residual `var_residual = 0.5·(var_v+var_u)` ignores anisotropy in the rotation-variance formula
**File:** `nav_technique_star_field.py` · `_build_covariance_3dof` (lines 938-944) and `_star_helpers.similarity_transform_fit`
The rotation variance `σ_θ² = var_residual / spread` assumes isotropic residuals;
for anisotropic per-axis residuals it can be off by up to 2×. Minor for
near-isotropic star centroids, but undocumented.

#### CODE-NAV-015 — `polarity_filter` clamps out-of-bounds vertices to the boundary pixel rather than dropping them
**File:** `dt_fitting.py` · `polarity_filter` (lines 252-259)
A vertex outside the image samples the *nearest boundary pixel's* gradient and
is then kept/dropped on that basis. The docstring rationalizes this ("rarely a
real edge"), but a strong boundary gradient (the image frame edge, common after
zero-padding into extfov) will spuriously *accept* off-image vertices. Better to
mark out-of-bounds vertices as rejected explicitly.

#### CODE-NAV-016 — `_build_polyline_mask` (3 copies) rounds vertices to nearest int and silently drops out-of-bounds; identical code duplicated across three techniques
**Files:** `nav_technique_body_limb.py`, `nav_technique_body_terminator.py`, `nav_technique_ring_edge.py` (each ~lines 51-70)
Pure duplication of a non-trivial helper that belongs in `dt_fitting.py` or
`distance_transform.py`. Also: rounding the *seed* polyline mask to int loses the
sub-pixel structure that `coarse_ncc_search` could exploit. Consolidate.

#### CODE-NAV-017 — `_seed_from_image_et` computes a RANSAC seed that is admittedly never used
**File:** `nav_technique_star_field.py` · `_seed_from_image_et` (lines 428-443)
The matcher iterates deterministically; the seed is "informational." Dead
computation logged at debug. Remove or wire a real RNG, but do not keep
load-bearing-looking dead code in a correctness-critical path.

#### CODE-NAV-018 — `StarFieldFromCatalogNav` greedy inlier matching is order-dependent and not globally optimal
**File:** `nav_technique_star_field.py` · `_greedy_inlier_count` (lines 325-371)
Greedy nearest-neighbour in detection-index order can mis-assign when two
detections compete for one catalog star; a Hungarian/optimal assignment within
the tolerance ball would be more robust and is cheap at N≤30. Medium because
RANSAC re-scores many candidates, partially masking the issue.

### Low

- **CODE-NAV-019** — `_INFINITY_DT_PENALTY_PX = 1e6` is a magic sentinel in
  `dt_fitting.py`; it interacts with `sigmas` (scaled = 1e6/σ) and Tukey's
  finite cutoff fine, but if any σ ≥ 1e6/4.685 the rejected vertex would *not* be
  zeroed. Add an assertion that σ ≪ that bound.
- **CODE-NAV-020** — `nav_technique_star_field.py` is 993 lines, approaching the
  1000-line module cap; splitting the triplet-hash matcher into a helper module
  would respect the convention and improve testability.
  *Tracked by:* #97 — split oversized modules exceeding 1000-line rulebook limit.
- **CODE-NAV-021** — Several techniques use `1e6 * np.eye(2)` as a "no info"
  covariance for spurious results instead of `inf`; the ensemble drops spurious
  results anyway, but a finite huge covariance would *not* be dropped if the
  spurious flag were ever cleared. Prefer `inf` for honesty.
- **CODE-NAV-022** — `_peak_to_runner_up_ratio` divides by `1e-9` when the
  runner-up quality is non-positive, producing ratios up to `~1e9` that then feed
  a sigmoid confidence term; cap the ratio.

---

## nav_model

### High (already listed: CODE-NAV-002, CODE-NAV-003)

### Medium

#### CODE-NAV-MODEL-001 — `_sigma_normal_per_vertex` hard-codes the photometric term `(limb_softness·0.5)²`
**File:** `nav_model_body.py` · `_sigma_normal_per_vertex` (lines 1030-1059)
The `0.5` photometric-softness coefficient and the albedo term are baked in with
no config knob and no calibration provenance, yet they directly set the
per-vertex LM weights (the inverse-variance prior). The `nan_to_num` fallback
substitutes `LIMB_ARC_MAX_UNCERTAINTY_PX` for NaN and `1e3` for inf — a vertex
with zero km/px (off-body) silently gets a 3-px sigma rather than being dropped.
Hoist the coefficients to config and drop zero-resolution vertices.

#### CODE-NAV-MODEL-002 — Body emission thresholds are module-level constants documented as "config default pending calibration"
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
**File:** `nav_model_body.py` (lines 100-156): `LIMB_ARC_MAX_UNCERTAINTY_PX`,
`BODY_BLOB_MIN_DIAMETER_PX`, `BODY_DISC_MIN_VISIBLE_LIT_FRACTION`,
`BODY_DISC_MAX_OVERFLOW_FRACTION`, `TERMINATOR_MIN_PHASE_FACTOR`, etc.
These are feature-gating thresholds (they decide which technique runs) but live
as Python constants, not in `config_files/`. The docstrings explicitly say
"numeric value is a config default pending calibration." Per project conventions
("magic numbers ... should be in config YAML") these belong in the bodies config
section. Several are exported in `__all__`, implying other modules import them —
double-check none are overridden inconsistently.

#### CODE-NAV-MODEL-003 — `visible_lit_fraction` denominator uses `body_total` (lit+dark) but the name says "lit"
**File:** `nav_model_body.py` · `_build_backplane_model` (lines 576-585)
`visible_lit_fraction = lit_visible_in_fov / body_total`. The numerator is
lit-and-in-FOV pixels; the denominator is the *whole* disc. So a fully-in-frame
body at high phase (small lit fraction) scores low even when its entire lit
hemisphere is visible — which is then compared against
`BODY_DISC_MIN_VISIBLE_LIT_FRACTION = 0.4`. The code comment defends this as
intentional (to keep discriminating power), but the *name* is misleading and the
0.4 threshold then conflates phase with framing. Rename to
`visible_disc_lit_fraction` and document the phase coupling, or split phase out.

#### CODE-NAV-MODEL-004 — Rings model swallows shadow-computation failures with bare `except Exception`
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
**File:** `nav_model_rings.py` (lines 430-438)
A broad `except Exception` around `where_inside_shadow` logs a warning and
proceeds with no shadow removal. This can silently produce a ring model with the
planet shadow painted as ring signal, biasing the ring-edge fit. Narrow the
except to the specific oops/SPICE exceptions and consider failing the model
rather than navigating on a contaminated template.

### Low

- **CODE-NAV-MODEL-005** — `nav_model_body.py` (1118 lines) and
  `nav_model_rings.py` (949 lines) are at/over the module-size guideline; the
  body file mixes rendering, feature emission, reliability sigmoids, and sigma
  math that could be split.
  *Tracked by:* #97 — split oversized modules exceeding 1000-line rulebook limit.
- **CODE-NAV-MODEL-006** — `_visible_arc_fraction` always returns 1.0 when any
  vertex survives (lines 1062-1071) — a placeholder. It feeds reliability and
  confidence, so it is a constant-1 input to several formulas. Document as a
  known stub or implement `survivors/total`.

---

## ensemble / orchestrator

### High (already listed: CODE-NAV-008, CODE-NAV-009)

#### CODE-ORCH-001 — The ensemble's grouping depends on a 5-pixel floor *because* the covariances it is supposed to use are wrong
> **Tracked by:** #123 — Mahalanobis agreement grouping breaks because per-technique covariances are CRLB-tight
**File:** `src/nav/nav_orchestrator/ensemble.py` · `EnsembleConfig.agreement_pixel_floor`, `_agreement_groups`
**Severity:** High

The config and code both candidly explain that per-technique covariances are
"CRLB-tight ... well below the actual position uncertainty driven by model error
and pointing residuals," so a `5.0 px` Euclidean floor is OR-ed into the
Mahalanobis grouping test. This is a band-aid over CODE-NAV-004/005/010: the
*right* fix is to make each technique report a realistic covariance (CRLB *plus*
model-error and pointing-residual variance), after which the Mahalanobis test
alone would group correctly and the pixel floor could be removed. As written, the
ensemble is effectively grouping on raw pixel distance for any pair within 5 px,
which discards the directional (rank-1) information that the covariances carry —
two rank-1 ring-edge results that agree along their *observable* axes but are 4 px
apart along an *unobservable* axis will be grouped and then mass-averaged,
producing a spurious 2-DoF "fix." **Fix:** add a model-error covariance floor at
the *technique* level (so each result's Σ is honest) and shrink/remove the
ensemble pixel floor; at minimum, apply the pixel floor only on the *observable*
subspace, not raw Euclidean distance.

#### CODE-ORCH-002 — Orchestrator wraps plugin technique calls in bare `except Exception` four times, hiding programming errors
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
**File:** `src/nav/nav_orchestrator/orchestrator.py` (lines 551, 640, 663, 755)
**Severity:** High

Four `except Exception:` "plugin sandbox" handlers swallow *all* exceptions from
feature extraction and technique execution. This converts genuine bugs
(shape mismatches, None dereferences, the `RuntimeError`s the techniques
deliberately raise on zero-vertex inputs) into silent technique drops, so a
systematic regression manifests only as "fewer techniques fired" with no
traceback. Catch a narrow set (or at least re-raise on
`AssertionError`/`TypeError`/`AttributeError`, which are always bugs) and always
log the traceback at ERROR. The DT techniques' own
`raise RuntimeError('... despite is_feasible reporting feasibility ...')` guard
(e.g. `BodyLimbNav` line 238) is *designed* to surface a contract violation —
the orchestrator's blanket catch defeats it.

### Medium

#### CODE-ORCH-003 — `derive_confidence_rank` treats `sigma_px=None` as passing the sigma gate for tiers with `max_sigma_px=None` only, but `high`/`medium` require a sigma
**File:** `ensemble.py` · `derive_confidence_rank` (lines 470-507)
When `sigma_px is None` and a tier has a numeric `max_sigma_px`, the tier is
skipped (correct), but a result with unknown sigma can still land in `low`
(`max_sigma_px=None`). A technique that fails to populate covariance therefore
silently caps at `low` rather than being flagged. Acceptable, but should be
logged so a missing covariance is visible.

#### CODE-ORCH-004 — `_drop_superseded_fallbacks` parses body names out of `feature_ids` by string prefix
**File:** `ensemble.py` · `_source_bodies` (lines 174-190)
Body identity is recovered by `fid.startswith('body_disc:')` etc. and splitting
on `:`. This couples the ensemble to a stringly-typed feature-id convention; a
feature-id format change silently disables fallback suppression. Carry the body
name as a structured field on `NavTechniqueResult` instead.

---

## reproj / backplanes / pds4

### Medium

#### CODE-REPROJ-001 — `_reduced_oops_precision` mutates oops global config; not thread-safe, as documented, but `RingMosaic.reproject` is reachable from multi-threaded mosaic drivers
**Files:** `src/nav/reproj/_context_managers.py`, `src/nav/reproj/rings.py` (line 1093)
The context manager flips `oops.config.PATH_PHOTONS.dlt_precision` globally.
CLAUDE.md's gotchas confirm `RingMosaic.reproject` is not safe for concurrent use
on the same obs, but the global precision mutation is unsafe across *all* obs in
the process — a second thread doing a full-precision backplane computation while
the first holds the reduced-precision block gets silently degraded geometry. The
mosaic CLIs should serialize reprojection or the precision should be a
thread-local / per-call argument rather than global mutation.

#### CODE-REPROJ-002 — `nav.reproj.{rings,bodies,cartographic_model,_serialization}` import stdlib `logging`
**Files:** `src/nav/reproj/rings.py:12`, `bodies.py:8`, `cartographic_model.py:11`, `_serialization.py:42`
CLAUDE.md forbids stdlib `logging` only in the listed core namespaces (feature,
nav_model, nav_orchestrator, nav_technique, support), and `reproj` is not listed
— so this is *not* a rule violation, but it is inconsistent with the rest of the
nav package which uses `pdslogger`. Flagging as a consistency item, not a
breach.

#### CODE-REPROJ-003 — `bodies.py` (1980 lines) and `rings.py` (1921 lines) far exceed the 1000-line module cap
> **Tracked by:** #97 — split oversized modules exceeding 1000-line rulebook limit
These are the two largest non-experiment, non-UI source files. The 1000-line
guideline is explicit; both warrant splitting (e.g. reproject vs. accumulate vs.
serialization).

#### CODE-PDS4-001 — `pds4/collections.py` uses bare `except Exception` in five places
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
**File:** `src/pds4/collections.py` (lines 84, 118, 180, 301, 316)
Bundle assembly swallows broad exceptions; a malformed label or missing product
becomes a silent skip. Narrow to the specific I/O / template exceptions so
bundle-generation failures are visible.

---

## support / obs / dataset / config

### High

#### CODE-SUP-001 — `mad_std` / robust-noise helpers wrapped in bare `except Exception`
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
**File:** `src/nav/support/misc.py` (lines 142, 173)
The robust noise estimate underpins every detection threshold
(`detection_sigma · image_noise_sigma`), the star SNR (CODE-NAV-003), and the
blob signal gate. A broad except that returns a fallback silently can mask a real
failure and substitute a noise estimate that mis-scales every downstream
threshold. Narrow the handler and surface failures.

### High (already listed: CODE-CFG-001 = CODE-NAV-003 config side)

#### CODE-CFG-001 — Placeholder/uncalibrated coefficients shipped in instrument configs
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
**File:** `src/nav/config_files/config_400_inst_coiss.yaml` (lines 116, 151)
See CODE-NAV-003. Add a startup validation pass that scans loaded config for the
literal `PLACEHOLDER` marker (or a dedicated `calibrated: false` flag) and either
fails or emits a prominent WARNING so uncalibrated coefficients cannot silently
ship.

### Medium

#### CODE-OBS-001 — `compute_smear_vector_px` differences two boresight projections but assumes linear pixel motion across the exposure
**File:** `src/nav/nav_model/stars/smeared_psf.py` · `compute_smear_vector_px` (lines 115-149)
The smear vector is `uv(tfrac=1) − uv(tfrac=0)` at the FOV centre. For a fast
slew the per-pixel motion is not uniform across the frame (it varies with field
position) and not linear in time; using the centre-FOV chord under-/over-states
smear at the corners. Adequate for narrow-angle cameras; document the
small-field assumption. Also: the function silently assumes `obs.time` brackets
the exposure — no validation that `time[1] > time[0]`.

#### CODE-SUP-002 — `flux.py` (1174 lines) and `image.py` (971 lines) are large; `flux.py` opens with a commented-out `# import logging`
> **Tracked by:** #96 — prune dead code (flux.py, correlate_old.py, commented-out blocks), #97 — split oversized modules exceeding 1000-line rulebook limit
**File:** `src/nav/support/flux.py:1`
Dead commented import; large module. Low-risk but worth cleanup. `image.py` mixes
many unrelated array utilities.

#### CODE-CFG-002 — `BodyDiscCorrelateNav._upsample_factor` falls back to a hard-coded `128` when `config.offset` is absent
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
**File:** `nav_technique_body_disc.py` (lines 753-758)
The technique reads `correlation_fft_upsample_factor` defensively with a magic
`128` default in two places. If the config block is genuinely missing that is a
config error that should fail loudly (consistent with the "missing key is a
KeyError so a typo fails fast" policy stated in the other techniques), not
silently default.

### Low

- **CODE-DATASET-001** — `dataset_pds3.py` (895 lines) is large; the per-mission
  subclasses duplicate `yield_image_files_*` scaffolding that could be lifted to
  the base.
  *Tracked by:* #97 — split oversized modules exceeding 1000-line rulebook limit.
- **CODE-STYLE-001** — Many techniques repeat the identical `_fail(...)` /
  spurious-result construction (`1e6*eye`, embed-rotation, zero offset). A shared
  `NavTechnique._spurious_result(...)` helper would remove ~5 copies.

---

## main / ui / experiments

### Medium

#### CODE-MAIN-001 — `nav_create_simulated_image.py` is 2509 lines — the single largest source file
> **Tracked by:** #97 — split oversized modules exceeding 1000-line rulebook limit
**File:** `src/main/nav_create_simulated_image.py`
2.5× the module-size guideline. CLI dispatch, rendering orchestration, and arg
parsing are intermixed; the simulation logic that is reused (`nav.sim`) should
absorb the non-CLI parts so the driver is thin.

#### CODE-MAIN-002 — `main/nav_mosaic.py` imports stdlib `logging`
**File:** `src/main/nav_mosaic.py:28`
`main/` scripts are outside the forbidden namespaces, so not a rule breach, but
inconsistent with the pdslogger-everywhere posture.

### Low

- **CODE-STYLE-002** — UI mosaic-viewer modules (`tiled_image_widget.py` 1789,
  `ring_window.py` 1752, `body_window.py` 1324) far exceed 1000 lines. These are
  the scoped-mypy-override PyQt6 files, so lower priority, but they concentrate a
  lot of untested logic.
  *Tracked by:* #97 — split oversized modules exceeding 1000-line rulebook limit.
- **CODE-EXP-001** — `src/experiments/` (2901 lines) and
  `src/nav/support/correlate_old.py` are excluded from lint/type per CLAUDE.md.
  `experiments/compare_mosaics.py` (1151 lines) duplicates substantial chunks of
  `reproj` mosaic logic; if any of it is becoming load-bearing it should graduate
  into `nav` with tests, otherwise it risks drifting out of sync with the real
  pipeline (the git status shows it was recently edited).

---

## Cross-cutting themes

1. **Covariances are CRLB lower bounds masquerading as 1σ.** Across
   `StarField`, `BodyBlob`, `BodyDiscCorrelateNav` rotation, and the per-feature
   `position_cov_px`, the reported covariance is a noise-floor / CRLB quantity
   with the wrong `Σw` normalization in several cases (CODE-NAV-004/005/010) and
   *no* model-error term. The ensemble then needs an ad-hoc 5 px pixel floor
   (CODE-ORCH-001) to function at all. The single highest-leverage fix in the
   whole codebase is to make every technique report a *realistic* covariance
   (CRLB ⊕ model-error ⊕ pointing-residual), after which the ensemble's
   Mahalanobis machinery would work as designed and the pixel-floor hack could be
   retired.

2. **Robust statistics used as failure detectors.** The Tukey-weighted RMS is the
   reported fit quality *and* the spurious-detector input in the DT techniques;
   because the biweight zeroes outliers, a robust RMS cannot detect the
   "one-clean-arc" mis-convergence it is meant to catch (CODE-NAV-006). Only
   `RingEdgeNav` added a raw per-feature check. Standardize a raw RMS alongside
   the robust one everywhere.

3. **Geometry derived from discrete ridge masks rather than the underlying
   continuous field.** The limb-normal bug (CODE-NAV-002) and the integer
   polyline-mask rounding (CODE-NAV-016) both throw away the sub-pixel /
   inside-outside information available from the backplane in favour of discrete
   neighbour tests. Compute normals from the silhouette/incidence backplane
   gradient, not the 1-px ridge.

4. **Broad `except Exception` in correctness-critical paths.** The orchestrator
   plugin sandbox (CODE-ORCH-002), rings shadow removal (CODE-NAV-MODEL-004),
   robust noise (CODE-SUP-001), and pds4 collections (CODE-PDS4-001) all swallow
   everything. Several of these mask the deliberate `RuntimeError` contract
   guards the techniques raise. Narrow them and always log tracebacks.

5. **Uncalibrated coefficients shipped as if final.** `PLACEHOLDER`
   `signal_dn_to_image_unit_scale` (CODE-NAV-003), the hard-coded `0.5`
   photometric sigma term (CODE-NAV-MODEL-001), and the body emission thresholds
   (CODE-NAV-MODEL-002) all affect *which* features and techniques run, yet none
   has calibration provenance. Add a config-validation gate that refuses to run
   with placeholder-marked coefficients.

6. **Module-size and duplication.** Five non-UI files exceed 1000 lines
   (`nav_create_simulated_image.py` 2509, `bodies.py` 1980, `rings.py` 1921,
   `flux.py` 1174, `nav_model_body.py` 1118). The `_build_polyline_mask` helper
   and the spurious-result construction are copy-pasted across techniques.

7. **Angle arithmetic without wrap handling.** Rotation enters the
   information-form combine and the Mahalanobis distance as a plain Euclidean
   coordinate (CODE-NAV-008). Safe only inside the small `max_rotation_deg`
   envelope; nothing enforces it.

**Testability assessment.** The math primitives (`dt_fitting`,
`_star_helpers.similarity_transform_fit`, `ring_math`, `predicted_snr`,
`distance_transform`, the ensemble) are pure functions over numpy arrays with
explicit validation and dataclass I/O — excellent for unit testing, and the
project clearly already exercises them. The main testability gaps are (a) the
covariance-normalization bugs are exactly the kind a focused unit test would
catch but evidently does not, suggesting missing analytic-fixture tests; (b) the
orchestrator's broad excepts make it hard to test that contract-violation
`RuntimeError`s actually propagate; (c) the placeholder-config issue has no
guard test.

---



---

# Part 2 — Image Reading (obs) & Dataset Enumeration

# Obs + DataSet subsystem critique

Files reviewed (19): src/nav/obs/__init__.py, src/nav/obs/obs.py,
src/nav/obs/obs_inst.py, src/nav/obs/obs_snapshot_inst.py,
src/nav/obs/obs_snapshot.py, src/nav/obs/obs_inst_cassini_iss.py,
src/nav/obs/obs_inst_voyager_iss.py, src/nav/obs/obs_inst_galileo_ssi.py,
src/nav/obs/obs_inst_newhorizons_lorri.py, src/nav/obs/obs_inst_sim.py,
src/nav/dataset/__init__.py, src/nav/dataset/dataset.py,
src/nav/dataset/dataset_pds3.py, src/nav/dataset/dataset_pds3_cassini_iss.py,
src/nav/dataset/dataset_pds3_galileo_ssi.py,
src/nav/dataset/dataset_pds3_voyager_iss.py,
src/nav/dataset/dataset_pds3_newhorizons_lorri.py,
src/nav/dataset/dataset_pds4.py, src/nav/dataset/dataset_sim.py.

---

## src/nav/obs/obs_inst.py

### CODE-OBS-001 — High — `ObsInst.star_psf_size` returns plain `dict` when no threshold matches; default path is fragile and mistyped
`star_psf_size` (lines 72-92). The loop iterates `for mag in sorted(star_psf_sizes)`
and returns `tuple(star_psf_sizes[mag])` on the first `star.vmag < mag`. The fallthrough
`return tuple(star_psf_sizes[mag])` reuses the loop variable `mag` left over from the last
iteration to mean "largest threshold". This works only by accident of Python's loop-variable
leak; ruff/B023 flags this pattern, and a future refactor that wraps the body in a
comprehension or generator will silently break the default. If `star_psf_sizes` is ever empty
the function raises `UnboundLocalError` (`mag` never bound) instead of a clear error.
WHY wrong/risky: depends on loop-variable leakage for correctness; no guard for the empty-config
case; the return type is annotated `tuple[int, int]` but `tuple(star_psf_sizes[mag])` produces a
tuple of arbitrary length from the YAML list (no length check), so a malformed config silently
yields a 3-tuple. Impact: wrong PSF stamp size or a confusing crash on bad config.

### CODE-OBS-002 — Low — `inst_config` typed `dict[str, Any] | None` but populated with `AttrDict`; `star_psf`/`star_psf_size` index with `['...']`
> **Tracked by:** #105 — replace pervasive Any / dict[str, Any] with TypedDicts and Protocols at interop boundaries
`_inst_config: dict[str, Any] | None` (line 24). Subclasses assign `new_obs._inst_config = inst_config`
where `inst_config` is an `AttrDict` (Cassini) or a nested plain `dict`. Functionally fine (AttrDict
is a `dict` subclass), but the `inst_config` property advertises `dict[str, Any] | None` while callers
also rely on attribute access elsewhere. Minor type-clarity issue; confirm by running `mypy src`.

---

## src/nav/obs/obs_snapshot.py

### CODE-OBS-003 — High — `closest_planet` computation in `__init__` is silently swallowed by `hasattr` guard and `body_distance` failures
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
`ObsSnapshot.__init__` lines 93-102. The closest-planet search runs only
`if not hasattr(self, '_closest_planet')`. Because `self.__dict__ = snapshot.__dict__`
(line 58) copies all snapshot attributes, any snapshot that happens to carry a
`_closest_planet` attribute (e.g. ObsSim sets `snapshot._closest_planet` deliberately) skips
the computation — that's intended for sim, but for a real instrument a stale `_closest_planet`
leaking in from oops would silently suppress the search. More importantly, `self.body_distance(planet)`
(line 97) builds a `center_bp` Backplane and calls SPICE for every planet in `config.planets`
during construction; if SPICE data is missing for any planet this raises during `from_file`
rather than at navigation time, and there is no error context identifying which planet/image failed.
WHY risky: construction-time SPICE dependency with no try/except or logging of which body failed;
`hasattr` guard couples sim and real paths. Impact: opaque failures reading images; hard to debug.
Confirm: read an image with one planet's SPICE kernel absent.

### CODE-OBS-004 — Medium — `_ra_dec_limits` declination wrap test uses `np.pi` threshold (should be `np.pi/2` range / different logic)
`_ra_dec_limits` lines 506-509. RA wrap-around at `> np.pi` is correct for RA in [0, 2pi).
But declination ranges only over [-pi/2, +pi/2], so `dec_max - dec_min > np.pi` can never be
true (max span is pi), and the wrap branch `dec[np.where(dec > np.pi)]` selects on `dec > np.pi`
which is never satisfied for valid declinations — it would produce an empty array and `.min()`
would raise. WHY wrong: declination does not wrap; this branch is dead and, if it ever did
trigger, would crash on an empty array. Impact: dead/incorrect code; latent crash. Confirm: no
real dec can satisfy the condition, so it is dead, but it signals a copy-paste error from the RA
block and should be removed.

### CODE-OBS-005 — Low — `extfov_margin_vu == (0, 0)` short-circuits make `ext_bp`/`ext_corner_bp` alias the non-ext Backplane
`ext_bp` (lines 370-372) and `ext_corner_bp` (lines 405-406) assign `self._ext_bp = self.bp`
when margin is zero. Correct, but it means the extended and non-extended caches share one
Backplane object; callers that mutate or `reset` one expecting independence could be surprised.
Low risk given current usage. Note only.

### CODE-OBS-006 — Low — `unpad_array_to_extfov` docstring/behavior mismatch; missing `Parameters:` section
`unpad_array_to_extfov` (lines 153-161): the method slices to `extdata_shape_vu` but the name
says "unpad ... to extfov" while it actually crops a (possibly larger, e.g. unpackbits-rounded)
array down to extdata shape. Docstring lacks the `Parameters:` block required by project
convention. Cosmetic / doc-convention.

---

## src/nav/obs/obs_inst_cassini_iss.py

### CODE-OBS-007 — Medium — `star_min_usable_vmag` has dead WAC branch returning identical value
`star_min_usable_vmag` lines 99-109: `if self.detector == 'WAC': return 0.0` then
`return 0.0`. Both branches return the same value, so the conditional is dead code that
implies an intended-but-unwritten WAC-specific value. WHY: either a TODO or leftover; misleads
readers into thinking WAC/NAC differ. Impact: confusion; flag the intended value or delete the branch.

### CODE-OBS-008 — Medium — `get_public_metadata` reads `SPACECRAFT_CLOCK_START_COUNT` as `float()` and assumes label keys exist
> **Tracked by:** #13 — Clean up handing of SCET strings
`get_public_metadata` lines 136-137. `float(self.dict['SPACECRAFT_CLOCK_START_COUNT'])` will
`KeyError` if the label lacks the key and `ValueError` if the SCLK is a partitioned string like
`"1/1234567890.123"` (Cassini SCLK counts are commonly `partition/count`). WHY risky: Cassini
SCLK strings are not always plain floats; this is exactly the format stored in
`SPACECRAFT_CLOCK_COUNT_PARTITION` + count. Impact: `get_public_metadata` can crash on otherwise
valid images. Confirm: inspect a COISS index row's `SPACECRAFT_CLOCK_START_COUNT` value format.

### CODE-OBS-009 — Low — `instrument_lid` builder assumes `self.detector[0]` is N/W; no validation
Line 143: `iss{self.detector[0].lower()}a.co`. For detector 'NAC' -> 'issna.co', 'WAC' ->
'isswa.co'. Correct for the two real cameras but silently produces a malformed LID for any
unexpected detector string. Low risk. Note.

### CODE-OBS-010 — Low — `star_max_usable_vmag` uses `np.log(self.texp)` without guarding `texp <= 0`
Lines 122/127: `np.log(self.texp / 26)` and `np.log(self.texp)`. A zero or negative exposure
time yields `-inf`/`nan`. Unlikely but unguarded. Note.

---

## src/nav/obs/obs_inst_voyager_iss.py

### CODE-OBS-011 — High — Spacecraft and I/F-correction selection keys off a single character `obs.dict['LAB02'][4]` with no validation
Lines 66 and 111. `obs.dict['LAB02'][4]` is used both to choose the V1-Saturn I/F correction and
to build the instrument LID (`vg{spacecraft}`). If `LAB02` is missing, shorter than 5 chars, or
formatted differently across volumes, this raises `KeyError`/`IndexError`, or worse silently picks
the wrong spacecraft and applies (or fails to apply) the 3.345x correction. WHY risky: a hard-coded
magic index into an unvalidated label string drives a photometric correction that changes pixel
values by 3.345x; a wrong character mis-calibrates the whole image. Impact: silently wrong I/F
scaling -> wrong limb/terminator/ring fits. Confirm: check `LAB02` format across VGISS volumes;
add an explicit assert that the extracted char is '1' or '2'.

### CODE-OBS-012 — Medium — `label3` factor parse via brittle string `.replace` of a fixed phrase; no error handling
Lines 62-64. `obs.dict['LABEL3'].replace('FOR (I/F)*10000., MULTIPLY DN VALUE BY', '')` then
`float(...)`. If `LABEL3` text differs by punctuation/spacing/case in any volume, the replace
leaves the phrase intact and `float()` raises `ValueError` with no context. WHY risky: silent
dependence on exact label wording; a single differing volume breaks all reads there. Impact:
crash on read for any non-conforming label. Confirm: diff LABEL3 across VGISS volumes; wrap in a
guarded parse with a clear error.

### CODE-OBS-013 — Medium — `pdslogger` percent-style logging mixed with f-strings; one call uses `%` args
Line 68-69: `logger.debug('  Applied Voyager 1 @ Saturn I/F correction: %.4fx', _V1_SATURN_IF_CORRECTION)`
uses printf-style args while every other debug call in this file uses f-strings. pdslogger
supports both, but the inconsistency is a style violation and risks a future reader copy-pasting
the `%`-form without args. Low/Medium. Align with f-string style.

### CODE-OBS-014 — Low — `get_public_metadata` `camera`/`filters` rely on `self.detector`/`self.filter` from oops with no validation
Lines 132-134. Consistent with other instruments; just noting the same unvalidated-attribute
pattern. `filters` is `[self.filter]` (single) vs Cassini's two-element list — confirm downstream
PDS4 templates handle the variable-length list.

---

## src/nav/obs/obs_inst_galileo_ssi.py / obs_inst_newhorizons_lorri.py

### CODE-OBS-015 — Medium — NH LORRI and Galileo read uncalibrated data (`calibration=False` / no calibration) but report I/F-style metadata downstream
LORRI `from_file` line 50 passes `calibration=False` with a TODO; Galileo applies no calibration
(`# TODO Calibrate once oops.hosts is fixed`). The resulting `obs.data` is in raw DN, not I/F,
yet the rest of the pipeline (noise thresholds, model brightness comparison) generally assumes a
consistent photometric scale per instrument. WHY risky: silent unit inconsistency between
instruments; any model/data brightness comparison tuned for I/F will be wrong for these two.
Impact: degraded navigation for GOSSI/NHLORRI until calibration is wired. This is a known TODO but
worth flagging as a correctness gap, not just a nicety.

### CODE-OBS-016 — Low — Heavy duplication across all five `from_file` implementations
The extfov-margin resolution block
(`if extfov_margin_vu is None: if isinstance(...dict): [...] else: [...]`) and the
`fc_path`/`abspath`/`image_url`/logging boilerplate are copy-pasted verbatim in cassini, voyager,
galileo, nhlorri, and sim. WHY: five copies drift independently (Cassini's dict lookup keys on
`obs.data.shape[0]` exactly like the others but has an extra `is_calibrated` branch; any fix to the
margin-resolution logic must be applied five times). Impact: maintenance hazard. Recommend a shared
helper on `ObsSnapshotInst` (e.g. `_resolve_extfov_margin(inst_config, data_shape, override)`).

### CODE-OBS-017 — Medium — `extfov_margin_vu_entry[obs.data.shape[0]]` raises `KeyError` for non-standard image heights
All five instruments do `extfov_margin_vu_entry[obs.data.shape[0]]` (Cassini line 87, Voyager 74,
Galileo 57, NHLorri 60, Sim 111) when the config entry is a dict keyed by image height
(256/512/1024 for Cassini). A windowed/subframed/summed image with a height not in the dict raises
a bare `KeyError` with no context. WHY risky: Cassini and others do produce non-standard sizes
(line/sample subframes). Impact: opaque crash on legitimate images. Add a clear error or a nearest/
default fallback.

---

## src/nav/obs/obs_inst_sim.py

### CODE-OBS-018 — Low — `ObsSim.from_file` attaches fake SPICE kernels and `_closest_planet` directly to the snapshot
Lines 115-119. `snapshot._closest_planet = sim_params.get('closest_planet')` and
`new_obs.spice_kernels = [...]` are set as bare attributes. The `_closest_planet` then defeats the
base-class computation (intended). Fine for tests but the `spice_kernels` magic literal
`['fake_kernel1.txt', 'fake_kernel2.txt']` is undocumented and could leak into PDS4 provenance.
Note only.

### CODE-OBS-019 — Low — `ObsSim.from_file` redundant `__init__` override
Lines 18-19: `__init__` just calls `super().__init__(snapshot, **kwargs)` with no added behavior;
it can be deleted (dead override). Minor.

---

## src/nav/obs/__init__.py

### CODE-OBS-020 — Low — `inst_name_to_obs_class` raises bare `KeyError` for unknown names while the dataset twin documents `Raises: KeyError`
Line 33: `return _INST_NAME_TO_OBS_CLASS_MAPPING[name.lower()]`. Unknown instrument name yields a
bare `KeyError` with no list of valid names, unlike `dataset_name_to_class` which at least documents
the raise. Minor UX/consistency: catch and re-raise with `inst_names()` in the message.

---

## src/nav/dataset/dataset_pds3.py

### CODE-DS-001 — High — `vol_start_idx`/`vol_end_idx` may be referenced unbound (mypy strict should catch; confirm)
`_yield_image_files_index` lines 569-586. `vol_start_idx` is assigned only inside
`if vol_start is not None:` and `vol_end_idx` only inside `if vol_end is not None:`. The later
comprehension (lines 579-586) references them guarded by the same `is not None` checks, so runtime
is safe, but the cross-check at line 577 (`if vol_start is not None and vol_end is not None and
vol_start_idx > vol_end_idx`) is also guarded — OK. The real risk: under mypy strict these are
"possibly-undefined" names. WHY: relies on control-flow narrowing mypy may not track across the
comprehension. Confirm by running `mypy src/nav/dataset/dataset_pds3.py`; if it passes, downgrade to
Low (readability), else it is a type error to fix by initializing both indices to `None`/sentinel.

### CODE-DS-002 — High — `choose_random_images` logic is biased and can loop effectively forever / under-yield
`_yield_image_files_index` lines 686-851. When `choose_random_images` is set:
(a) it picks one random volume, reads its full index, picks **one random row** (line 727), then
breaks after at most one yield per outer `while True` iteration; (b) the chosen random row may fail
every filter (img name list, number range, camera, additional criteria) in which case the inner
`for row in rows` loop has only that single row and yields nothing, then `break`s out and loops
again — so a request for N random images can spin many iterations producing zero progress, and the
sampling is biased toward volumes/rows that pass filters. (c) There is no bound on the `while True`
loop other than `num_yields >= limit_yields`; if every random pick is filtered out (e.g. a camera
filter excludes the sampled row), this is an unbounded busy loop. WHY wrong: random sampling that
re-reads the whole index per single sample is O(N) per image and can livelock under any active
filter. Impact: hangs or extreme slowness for `--choose-random-images N` combined with
`--camera`/name filters. Confirm: run with `--choose-random-images 5 --camera wac` against a NAC-heavy
volume.

### CODE-DS-003 — Medium — `done` early-exit assumes monotonic image numbers but BOTSIM/NAC+WAC interleave breaks it
Lines 771-777: `if img_end_num is not None and img_num > img_end_num: ... done = True; break`
relies on "Images are in monotonically increasing order". COISS index rows are ordered by
FILE_SPECIFICATION_NAME, and NAC ('N...') and WAC ('W...') share the same numeric counter but sort
separately by leading letter; within one camera the numbers are monotonic, but the index mixes both
cameras. If the index is sorted by filespec (N* then W*), the numeric sequence is **not** globally
monotonic, so `done=True` can terminate the scan early and **drop later WAC images** within range.
WHY risky: silent under-selection of images at the upper end of a number range. Impact: missing
images in batch runs that use `--last-image-num`. Confirm: inspect ordering of a COISS_2xxx index
and test `--last-image-num` near a NAC/WAC boundary.

### CODE-DS-004 — Medium — `open(filename, ...)` for CSV/file-list uses stdlib `open`, bypassing FileCache; no URL support
`yield_image_files_from_arguments` lines 356 and 376. The `--image-filespec-csv` and
`--image-file-list` inputs are read with builtin `open`, so they cannot be remote URLs even though
the rest of the dataset layer is FCPath/FileCache-based and `pds3_holdings_root` may be a URL. WHY:
inconsistent I/O model; a user pointing these at an http(s) path gets a confusing `FileNotFoundError`.
Impact: feature gap / surprising error. Use FCPath.

### CODE-DS-005 — Medium — `image_filespec_csv` column-detection leaves `colnum` referencing the last column on a malformed (but header-present) file
Lines 359-371. The `for colnum in range(len(header))` with `else: raise` correctly raises when no
matching header is found. But if a matching header **is** found, `colnum` is the matched index — fine.
The subtle bug: `row[colnum]` (line 370) will `IndexError` on any data row shorter than `colnum+1`
columns with no per-row guard, aborting the whole run. WHY: ragged CSVs are common; one short row
kills the batch. Impact: brittle. Wrap row access with a length check and skip/log bad rows.

### CODE-DS-006 — Medium — `--image-file-list` reuses the loop variable `filename` for both the outer file and the parsed token
Lines 375-385: outer `for filename in arguments.image_file_list:` then inside the loop
`filename = line.split(' ')[0]`. The outer `filename` (the list-file path) is overwritten by the
per-line token, so after the first line the original filename is lost — harmless today because it is
not used again in the loop, but it is a latent bug if error messages later reference `filename`
expecting the list-file path (the `ValueError` on line 384 already prints the token, not the source
file, which is the wrong context for the user). WHY: confusing variable shadowing; wrong error
context. Impact: poor diagnostics. Rename inner variable.

### CODE-DS-007 — Low — `_validate_selection_arguments` is dead code (admitted in a TODO)
Lines 318-329, with `# TODO This method is currently unused and should be used`. Either wire it into
`yield_image_files_from_arguments` or remove it. Dead code per conventions.

### CODE-DS-008 — Low — `lru_cache` on a bound method (`_read_pds_table`) keyed by `(self, fn, columns)` holds dataset instances alive
Line 451-452 (`@lru_cache(maxsize=3)` with `# noqa: B019`). The noqa acknowledges it, and the comment
says instances are long-lived; acceptable, but worth noting that the cache key includes `self`, so
three distinct *tables* are cached per instance, not three globally — the maxsize=3 may thrash for a
dataset spanning many volumes. Note only.

### CODE-DS-009 — Low — Large commented-out blocks throughout (`force_has_offset_file`, BOTSIM combine in base, planet validation)
> **Tracked by:** #96 — prune dead code (flux.py, correlate_old.py, commented-out blocks)
Lines ~221-316, ~395-449, ~636-832. Substantial dead/commented code documents future intent but
violates the "no dead code" convention and makes the 895-line module harder to read. Recommend moving
the plan to an issue/doc and deleting.

---

## src/nav/dataset/dataset_pds3_cassini_iss.py

### CODE-DS-010 — High — BOTSIM grouping mis-pairs when two consecutive non-paired BOTSIM images appear, and time-slop check uses image number as seconds
`yield_image_files_index` lines 287-316. Two issues:
(1) The pairing assumes alternating N/W; if two consecutive rows are both BOTSIM but the second is
not the partner (e.g. N then N), the `abs(img_num - last_img_num) <= 3` test (line 301) treats the
**IMAGE_NUMBER** difference as "3 seconds slop", but IMAGE_NUMBER is the SCLK-derived image counter,
not seconds — for Cassini the counter increments by ~1 per SCLK second only approximately, and back-
to-back BOTSIM frames can differ by more than 3 in the counter while still being a true pair, or two
unrelated NAC frames can differ by <=3 and be wrongly paired. (2) When the `<=3` test fails, the code
sets `last_imagefile = imagefile` (line 311) and **discards `last_imagefile` (the previous one)
without yielding it**, silently dropping an image from the output. WHY wrong: silent image loss plus
a physically wrong "seconds" interpretation of an image counter. Impact: dropped images and incorrect
NAC/WAC pairings in `--group botsim` runs. Confirm: feed a sequence with a lone BOTSIM frame between
pairs and assert all frames are yielded.

### CODE-DS-011 — Medium — `pds4_bundle_path_for_image` returns `''` for short names but callers concatenate it into a path
Lines 365-384: `if len(image_name) < 11: return ''`. `pds4_path_stub` (line 397) handles the empty
string, but other call sites that do `bundle_path + something` would produce a malformed leading path.
WHY: sentinel empty-string return is an easy footgun. Impact: silently wrong bundle paths for unexpected
names. Prefer raising `ValueError` for an invalid name (it should never legitimately be <11 here).

### CODE-DS-012 — Medium — `_get_img_name_from_label_filespec` strips at first `_`, collapsing the BOTSIM sub-frame suffix and `_CALIB`
Line 95: `img_name.rsplit('.')[0].rsplit('_')[0]`. For a CALIB filespec `N1234567890_1_CALIB.IMG`
this returns `N1234567890` (drops `_1`), which is the intended image **name**. But `_img_name_valid`
explicitly supports the `[NW]dddddddddd_d[d]` sub-frame form (lines 121-129), so the two functions
disagree about whether the `_1` sub-frame index is part of the image name. Downstream filtering
(`img_name in img_name_filter_list`) compares the collapsed name against filespec-derived names, which
is internally consistent here, but the inconsistency between "valid names may carry `_d`" and
"extracted names never do" is a latent matching bug if a filter list contains sub-frame-qualified
names. Confirm: filter with `N1234567890_1` and verify it matches.

### CODE-DS-013 — Low — `_check_additional_image_selection_criteria` requires `arguments.camera` to exist; raises AttributeError if a different parser is used
Line 204: `if arguments is None or arguments.camera is None`. If `yield_image_files_index` is called
programmatically with an `arguments` Namespace that lacks `camera`, this raises `AttributeError`
instead of treating it as "no filter". WHY: tight coupling to the CLI namespace shape. Impact:
brittle for non-CLI callers. Use `getattr(arguments, 'camera', None)`.

### CODE-DS-014 — Low — `_volset_and_volume`/`_volume_to_index` index `volume[6]` assuming exactly `COISS_Nxxx`
Lines 159, 169. `volume[6]` extracts the thousands digit; correct for `COISS_2001` -> '2'. Fragile
to any volume-name format change; no validation. Note (same pattern in VGISS `volume[6]`).

---

## src/nav/dataset/dataset_pds3_galileo_ssi.py

### CODE-DS-015 — Medium — Hard-coded orbit/target directory whitelist will silently reject any new/renamed directory
> **Tracked by:** #17 — GOSSI does not handle REDO properly
`_get_img_name_from_label_filespec` lines 79-125. Two big hard-coded tuples enumerate every Galileo
encounter directory (`RAW_CAL`, `VENUS`, ..., and `C3`..`J0`). Any directory not in these lists raises
`ValueError('bad target directory')`, which in the index loop is logged-and-skipped
(dataset_pds3.py line 737), so an unrecognized directory silently drops all its images. WHY risky:
config-vs-hardcoded — the set of Galileo directories is data, not code; a holdings update adds images
that vanish without warning. Impact: silent data loss. Move to config or derive structurally. Confirm:
add a directory not in the list and verify images are skipped.

### CODE-DS-016 — Low — `_volset_and_volume` hard-codes `GO_0xxx` for all volumes incl. `GO_0002`..`GO_0023`
Line 185: `return f'GO_0xxx/{volume}'`. Correct only because all Galileo volumes are in the `GO_0xxx`
volset; brittle if a `GO_1xxx` ever appears. Note.

---

## src/nav/dataset/dataset_pds3_voyager_iss.py

### CODE-DS-017 — Medium — `_img_name_valid` accepts only the 8-char `Cddddddd` form, but VGISS images are also referenced as `C1234567_GEOMED`
`_img_name_valid` lines 100-120 require `len == 8`. The `--image-file-list` path
(dataset_pds3.py line 383) validates each list entry with `_img_name_valid`, so a user listing
`C1234567_GEOMED` or `C1234567_CALIB` (the actual on-disk product names) is rejected with
"Bad filename". WHY: mismatch between the canonical short name and the product filenames users
naturally have. Impact: usability — valid file lists rejected. Confirm: put `C1234567_GEOMED.IMG`
in an `--image-file-list` and observe the ValueError.

### CODE-DS-018 — Low — `_get_img_name_from_label_filespec` only accepts `_GEOMED.LBL`, silently returning None for `_CALIB`/`_RAW`
Lines 95-97. Non-GEOMED products return `None` (skip). That is a deliberate "only navigate GEOMED"
choice, but it is undocumented in the method and means a CALIB-only volume yields nothing with no log.
Note / add a comment.

---

## src/nav/dataset/dataset_pds3_newhorizons_lorri.py

### CODE-DS-019 — Medium — `_get_label_filespec_from_index` and `_get_img_name_from_label_filespec` mix `_eng` and `_sci` but only one image type should be navigated
Lines 48 and 90 accept both `_sci.lbl` and `_eng.lbl`. ENG (engineering) LORRI products are
typically not science-calibrated frames for navigation; accepting both means the index loop will
yield engineering frames too. WHY: likely unintended inclusion of non-science frames. Impact: ENG
frames navigated/processed unexpectedly. Confirm intent; if only `_sci` is wanted, drop `_eng`.

### CODE-DS-020 — Low — `range_dir` length check `== 15` and `[8] != '_'` is an unexplained magic format
Lines 85-88: `if len(range_dir) != 15 or range_dir[8] != '_'`. The expected format
`ddddddd_ddddddd` is 15 chars; correct but undocumented magic constants. Note.

---

## src/nav/dataset/dataset.py

### CODE-DS-021 — Medium — `ImageFile.image_file_path`/`label_file_path` cache `get_local_path()` results with no thread safety
Lines 44-56. The lazy properties memoize `_image_file_path`/`_label_file_path` without locking. If a
single `ImageFile` is shared across threads (the architecture note warns obs is not thread-safe, but
ImageFiles can be enumerated then dispatched), two threads can both see `None` and race the download.
FileCache itself may be MP-safe, but the dataclass field assignment is not synchronized. WHY: data
race on first access. Impact: redundant downloads or inconsistent path. Low-Medium; document single-
thread-per-ImageFile or add a lock.

### CODE-DS-022 — Low — `ImageFiles.__getitem__` typed `idx: int` but does not support slices
Lines 77-78. `__getitem__(self, idx: int)` returns a single `ImageFile`; slicing
`image_files[1:3]` would return a `list[ImageFile]` at runtime but is untyped/unsupported by the
annotation. Minor API gap. Note.

### CODE-DS-023 — Low — Seven `pds4_*` base methods raise `NotImplementedError` instead of being abstract, allowing silent partial implementations
Lines 153-281. The comment explains the deliberate non-abstract choice (datasets may not support
PDS4). Reasonable, but it means a subclass that implements *some* pds4 methods and forgets others
fails only at runtime when that path is hit. Acceptable tradeoff; noting the design.

---

## src/nav/dataset/__init__.py

### CODE-DS-024 — Low — Module-level `assert` enforces registry consistency; stripped under `python -O`
> **Tracked by:** #98 — consolidate parallel instrument registries into a single registry
Lines 46-48: `assert sorted(...) == sorted(...), 'Dataset names are inconsistent'`. Running under
`-O` removes the assert, so a future edit that desyncs the two mappings would ship silently. Replace
with a real check raising at import, or a unit test. Note.

### CODE-DS-025 — Low — `dataset_name_to_class`/`dataset_name_to_inst_name` raise bare `KeyError`; messages lack valid-name list
Lines 68, 83. Same as CODE-OBS-020. Minor UX.

---

## src/nav/dataset/dataset_sim.py

### CODE-DS-026 — Low — `pds4_template_variables` signature omits the keyword-only `*` present in the base/other subclasses
Lines 87-94: `def pds4_template_variables(self, image_file, nav_metadata, backplane_metadata)` —
positional, unlike the base (`*, image_file, ...`, dataset.py line 259) and Cassini. This violates
the "3 positional max, rest keyword-only" convention and is an LSP mismatch with the base signature
(mypy strict may flag the override). Confirm with `mypy`; align to keyword-only.

---

## src/nav/dataset/dataset_pds4.py

### CODE-DS-027 — Low — Entire class is `NotImplementedError` stubs; `_img_name_valid` is declared `@staticmethod` but base wants it abstract
Lines 9-37. All methods raise. Fine as a placeholder, but it is registered nowhere
(`_DATASET_NAME_TO_CLASS_MAPPING` has no PDS4 entry), so it is currently unreachable dead scaffolding.
Note.

---

## Cross-cutting observations

- **Duplication (CODE-OBS-016 expanded):** the five `from_file` bodies, the five
  near-identical `pds4_*` `NotImplementedError` stubs (galileo/voyager/nhlorri), and the per-dataset
  `__init__` that only forwards to `super().__init__` are large-scale copy-paste. The forwarding
  `__init__`s (Cassini lines 214-237, Voyager 169-192, Galileo 215-238, NHLorri 169-192) add nothing
  over `DataSetPDS3.__init__` and can be deleted.
- **`pdslogger` usage:** all obs `from_file` methods grab `logger = IMAGE_LOGGER` directly rather than
  via `NavBase.logger`, because they are `@staticmethod` and have no `self`. That is acceptable given
  the static factory design, but it means these read paths cannot honor a per-instance logger. Note.
- **No bare `print` / no stdlib `logging` import found** in scope — convention upheld.
- **Module size:** `dataset_pds3.py` (895) and `dataset_pds3_cassini_iss.py` (694) are under 1000 but
  the Cassini `pds4_template_variables` (lines 454-636) is a 180-line literal dict that should be data
  (YAML/JSON), not code.

---



---

# Part 3 — Reprojection, Backplanes & PDS4

# Critique: reproj / backplanes / pds4 subsystems

Files reviewed (23): src/nav/reproj/__init__.py, src/nav/reproj/_context_managers.py, src/nav/reproj/_serialization.py, src/nav/reproj/bodies.py, src/nav/reproj/cartographic_model.py, src/nav/reproj/photometric_model.py, src/nav/reproj/ring_orbit_model.py, src/nav/reproj/rings.py, src/reproj_cli/__init__.py, src/reproj_cli/args.py, src/reproj_cli/factories.py, src/reproj_cli/offsets.py, src/reproj_cli/paths.py, src/reproj_cli/reproject.py, src/backplanes/__init__.py, src/backplanes/backplanes.py, src/backplanes/backplanes_bodies.py, src/backplanes/backplanes_rings.py, src/backplanes/merge.py, src/backplanes/writer.py, src/pds4/__init__.py, src/pds4/bundle_data.py, src/pds4/collections.py.

I read every file in scope in full. A separate review covers nav_model/nav_technique math; this review focuses on reprojection geometry/accumulation, thread safety, backplane per-pixel correctness, PDS4 label/LID correctness, I/O and error paths, type-safety, efficiency, duplication, and conventions.

---

## Findings

### CODE-BACKPLANE-001 — Backplane generation skips ALL navigated images (wrong status literal)
- Severity: **Critical**
- File/symbol: `src/backplanes/backplanes.py`, `generate_backplanes_image_files`, line 53-61.
- Description: The function reads the nav `_metadata.json` and tests `status = nav_metadata.get('status'); if status != 'success': ... return`. But `nav` writes `NavResult.status`, which is the `Literal['ok', 'failed', 'conflicted']` value `'ok'` for successful navigations (`src/nav/nav_orchestrator/nav_result.py` line 25, 175; written verbatim in `src/nav/navigate_image_files.py` line 192 as `'status': result.status`). The string `'success'` is never written to a nav metadata file (it only appears as a *cloud-task* runner status in `src/main/nav_*_cloud_tasks.py`, a different dict).
- WHY: A successful navigation produces `status == 'ok'`, which is `!= 'success'`, so the guard always fires and `return`s after logging "Skipping". `src/reproj_cli/offsets.py` (line 151) correctly checks `status != 'ok'` for the *same* files, proving the intended value is `'ok'`.
- Impact: `nav_backplanes` produces NO backplane FITS for any successfully navigated image (only logs "Skipping ... status=ok"). The entire downstream backplane stage is dead for real data. No test exercises this path, so it went uncaught.

### CODE-PDS4-001 — PDS4 bundle generation skips ALL navigated images (wrong status literal)
- Severity: **Critical**
- File/symbol: `src/pds4/bundle_data.py`, `generate_bundle_data_files`, line 53-62.
- Description: Same defect as CODE-BACKPLANE-001: `if status != 'success': ... return`. nav writes `'ok'`.
- WHY: Successful nav metadata has `status == 'ok'`; the guard skips every such image.
- Impact: `nav_create_bundle` generates no data/label/supplemental/browse files for any real navigated image. Combined with CODE-BACKPLANE-001, the entire backplane→PDS4 pipeline is inert on real data.

### CODE-PDS4-002 — Malformed LID in global index .tab files (missing `urn:nasa:pds:` prefix, wrong image part)
- Severity: **High**
- File/symbol: `src/pds4/collections.py`, `generate_global_index_files`, line 196: `lid = f'{pds4_bundle_name}:data:{image_name}'`.
- Description: The body/ring global-index rows write a hand-built LID `'{bundle}:data:{image_name}'`. The canonical data LID (see `DataSet.pds4_image_name_to_data_lid`, e.g. `src/nav/dataset/dataset_pds3_cassini_iss.py` line 428-439) is `urn:nasa:pds:{bundle}:data:{image_lid_part}` where `image_lid_part = image_name.split('_',1)[0].split('.',1)[0]; image_lid_part = image_name[1:] + image_name[0].lower()` (e.g. `N1234567890` -> `1234567890n`).
- WHY: The collections code bypasses the dataset's LID builder, omitting the `urn:nasa:pds:` namespace prefix and using the raw `image_name` instead of the transformed `image_lid_part`. Both differ from every other LID emitted in the bundle (collection .tab files use `dataset.pds4_image_name_to_data_lidvid`; labels use `DATA_LID`).
- Impact: The `global_index_bodies.tab` / `global_index_rings.tab` LID columns reference products by an identifier that does not match the actual product LIDs, breaking PDS4 cross-references / validation for the supplemental index.

### CODE-BACKPLANE-002 — Backplane statistics units disagree with stored FITS array units
- Severity: **Medium**
- File/symbol: `src/backplanes/backplanes_bodies.py` `create_body_backplanes` lines 178-183; `src/backplanes/backplanes_rings.py` lines 93-98; consumed by `src/backplanes/writer.py` (BUNIT) and `src/pds4/collections.py` (index columns).
- Description: When a backplane's configured `units == 'rad'`, the min/max statistics are converted to degrees (`valid_values = np.degrees(valid_values)`) before being stored in `*_stats`, but the array written to FITS (`per_type_arrays[bp_name]` / `result['arrays'][bp_name]`) remains in radians, and `writer.py` sets `BUNIT` to the *declared* unit (`'rad'`). The degrees-valued stats then flow into the supplemental JSON and into `global_index_*.tab` min/max columns.
- WHY: The angular FITS pixel data is in radians while the metadata min/max for the same quantity is in degrees; nothing records that the stats were converted.
- Impact: The PDS4 global index reports angular ranges in degrees while the backplane data and BUNIT say radians — a unit mismatch in the delivered products. Confirm intended units for the index; if degrees are wanted in the index, the FITS/BUNIT should match or the conversion should be documented and the column labels set to degrees.

### CODE-BACKPLANE-003 — Body/ring occlusion in merge uses body *center* distance, not per-pixel distance
- Severity: **Medium**
- File/symbol: `src/backplanes/merge.py`, `merge_sources_into_master`, lines 34-46, 73-77, 126.
- Description: Each body contributes a single scalar distance (`float(entry['distance'])` = inventory center `range`) broadcast across its whole mask. `nearest_body_distance` and the ring occlusion test `occluded = body_presence & (nearest_body_distance < ring_distance)` therefore compare the body's *center* range against the per-pixel ring distance.
- WHY: A body spans a range of distances across its disk (near limb vs. sub-observer point). Using the center range mis-orders body-vs-ring occlusion near the limb and for bodies that straddle the ring plane.
- Impact: Rings may be incorrectly revealed or occluded by up to roughly the body radius in distance near the body limb, corrupting merged ring backplane pixels there. Inter-body ordering has the same limitation. Confirm whether per-pixel body distance backplanes are available to replace the scalar.

### CODE-REPROJ-001 — Global ring antimask placement assumes grid-aligned `longitude_start`
- Severity: **Medium**
- File/symbol: `src/nav/reproj/rings.py`, `_reproject_inner`, lines 1210-1212, 1221-1225, 1255-1257, 1402-1403.
- Description: Local longitude bins are computed relative to `longitude_start` (`bp_lon_binned = floor((lon - longitude_start)/res)`), and the per-column actual longitude is reconstructed as `lon_bins_restr * res + longitude_start` (correct). But the *global* antimask is filled with `new_antimask[lon_bins_restr[...] + full_min_lon_bin] = True` where `full_min_lon_bin = floor(longitude_start/res)`. The global mosaic's bin→longitude convention is `global_bin * res` (no `longitude_start` offset; see `RingMosaic.bounds` / `to_bounded` which use `bin * lon_resolution`).
- WHY: For relative bin `b`, the data's actual longitude is `b*res + longitude_start`, but it is placed at global bin `b + floor(longitude_start/res)` whose mosaic longitude is `(b + floor(longitude_start/res))*res = b*res + floor(longitude_start/res)*res`. These differ by the fractional remainder `longitude_start - floor(longitude_start/res)*res`. With the default `longitude_start = 0` (and any grid-aligned custom start) the remainder is 0 and everything agrees; with a non-grid-aligned `--longitude-range` start the column is misregistered by up to one bin relative to its true longitude.
- Impact: A custom `longitude_range` whose start is not a multiple of `longitude_resolution` shifts reprojected ring columns by a sub-bin/one-bin offset in the mosaic longitude grid, and can mis-merge against columns from other images that used a different (e.g. default) start. To confirm: build two reprojections of the same scene with `longitude_range` starts differing by a non-multiple of `lon_resolution` and check that identical features land in the same global bins.

### CODE-REPROJ-002 — reproj modules use stdlib `logging` instead of pdslogger
- Severity: **Medium**
- File/symbol: `src/nav/reproj/bodies.py` (import line 9, use line 1016), `src/nav/reproj/rings.py` (import line 12, use line 1038, type hint `logger: logging.Logger` line 1130), `src/nav/reproj/cartographic_model.py` (import line 11, use line 77), `src/nav/reproj/_serialization.py` (import line 42, `_logger` line 57).
- Description: All four `nav.reproj` modules import the stdlib `logging` module and call `logging.getLogger(...)`. CLAUDE.md states core nav code must use `pdslogger` via `NavBase.logger` / the `IMAGE_LOGGER`, and "Never import the stdlib `logging` module in core code". The rest of `nav` (and the backplane/pds4 packages, which take `logger: PdsLogger`) follow this.
- WHY: `nav.reproj` is not in the explicit core list in CLAUDE.md, but it is library code under `nav`, and the convention/rule is repo-wide. These `getLogger` loggers are unconfigured and bypass the pdslogger stream handler that tests rely on (tests capture via `capsys`), so reprojection log output is not captured/structured consistently with the rest of nav.
- Impact: Inconsistent logging; per-image reprojection diagnostics do not flow through `IMAGE_LOGGER` sections. Confirm with the maintainer whether `nav.reproj` is intended to be exempt; if not, switch to pdslogger.

### CODE-BACKPLANE-004 — Broad `except Exception` in simulated body-mask resolution
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
- Severity: **Medium**
- File/symbol: `src/backplanes/backplanes_bodies.py`, `_create_simulated_body_backplane`, lines 50-58.
- Description: A bare `except Exception:` wraps name lookup + index-map slicing and, on any failure, falls back to filling the entire rectangle as the body mask.
- WHY: A genuine programming error (wrong attribute, shape mismatch, bad index) is silently swallowed and replaced with a full-rectangle mask, producing wrong simulated backplanes rather than failing loudly. Ruff's `BLE001`/`B` category targets exactly this.
- Impact: Simulated backplane masks can be silently corrupted (whole-rectangle fill) masking real bugs in the sim path. Narrow to the expected exceptions (`ValueError`, `KeyError`, `IndexError`) or remove the fallback.

### CODE-BACKPLANE-005 — Broad `except Exception` around NAIF ID lookup; non-deterministic fake IDs
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
- Severity: **Low**
- File/symbol: `src/backplanes/merge.py`, lines 47-54 and line 52.
- Description: `int(cspyce.bodn2c(body_name))` is wrapped in `except Exception`. For simulated data the fallback is `10000 + (abs(hash(body_name)) % 20000)`. `hash()` of a `str` is process-randomized (PYTHONHASHSEED), so the fake NAIF ID for a given body differs run-to-run and can collide across bodies.
- WHY: Broad except + non-deterministic, collision-prone IDs. The same non-determinism appears in `_create_simulated_body_backplane` seed (`abs(hash((body_name, backplane_name)))`, line 40).
- Impact: Sim-only `BODY_ID_MAP` values (and sim backplane fill values) are not reproducible across runs and two bodies can map to the same id. Use a stable hash (e.g. `hashlib`) or an explicit per-name counter, and narrow the except to the real cspyce error.

### CODE-REPROJ-003 — Off-by-one / wording mismatch on the uint16 image-count cap
- Severity: **Low**
- File/symbol: `src/nav/reproj/bodies.py` `add` lines 1409-1413 (and docstring lines 634-636, 1404-1405); `src/nav/reproj/rings.py` `add` lines 1588-1592 (and docstring lines 719-723, 1546-1548).
- Description: The guard is `if self._image_count > np.iinfo(np.uint16).max:` (i.e. `> 65535`). Since `_image_count` is incremented after each add and is used as the `image_number` *before* incrementing, image numbers 0..65535 are valid and the guard only raises on the 65537th add. The bodies docstring says "capping a single mosaic at 65 535 contributing images" and "raises OverflowError if that limit is exceeded" which is inconsistent with the actual 65536-image capacity; the rings docstring correctly says 65,536.
- WHY: Documentation/implementation mismatch (bodies) and the comparison `>` vs `>=` makes the cap one larger than the round "65535" figure cited.
- Impact: Cosmetic; no overflow can occur (the 65536th image uses image_number 65535 = uint16 max, the 65537th raises before writing). Align the bodies docstring with the rings docstring and the actual `> max` behaviour.

### CODE-REPROJ-004 — `RingMosaic` docstring claims `reproject()` mutates `obs.fov`; it does not
- Severity: **Low**
- File/symbol: `src/nav/reproj/rings.py` module docstring lines 6-8; class `RingMosaic` Notes lines 714-716; `reproject` docstring.
- Description: The thread-safety notes state reproject "temporarily mutate[s] obs.fov and oops global precision settings." `_reproject_inner` only builds a `Meshgrid`/`Backplane` and reduces oops global precision via `_reduced_oops_precision`; it never assigns `obs.fov`.
- WHY: Stale/incorrect hazard documentation. The real (and correctly documented elsewhere) hazard is the shared oops global precision mutation in `_reduced_oops_precision` and Backplane construction on a shared `obs`.
- Impact: Misleading thread-safety guidance; could lead callers to over- or under-protect. Drop the `obs.fov` claim; keep the oops-global-precision + shared-`obs`-Backplane hazard.

### CODE-REPROJ-005 — `radius_at_longitude` method docstring drops the `/86400` day conversion
- Severity: **Low**
- File/symbol: `src/nav/reproj/ring_orbit_model.py`, `radius_at_longitude` docstring lines 91-93 ("pericenter direction at time `et` is `w0 + dw * et`").
- Description: The method docstring says the pericenter direction is `w0 + dw * et`, but the implementation (line 102) and the class docstring (line 38) correctly use `w0 + dw * et / 86400` because `dw` is rad/day and `et` is seconds.
- WHY: The abbreviated method docstring omits the seconds→days conversion that the code performs.
- Impact: Documentation only; the code is correct. Fix the method docstring to read `w0 + dw * et / 86400` (or "per day").

### CODE-PDS4-003 — `pds4` collection-label exception handling: `logger.exception` then `raise` (redundant) and inconsistent template.write call styles
- Severity: **Low**
- File/symbol: `src/pds4/collections.py` lines 82-89, 116-123, 299-303, 314-318; vs `src/pds4/bundle_data.py` line 112-113.
- Description: Two stylistic inconsistencies. (1) Collection/global-index labels wrap `template.write(...)` in `try/except Exception: logger.exception(...); raise`, which logs a full traceback then re-raises the same exception (it will be logged again by the caller / top-level handler). (2) `template.write` is sometimes passed a local `str(...get_local_path())` (lines 83, 117) and sometimes an `FCPath` directly (lines 300, 315; and `bundle_data.py` line 113). `pdstemplate.PdsTemplate.write` normalizes its argument to `FCPath` and uses `write_bytes` (which uploads), so both styles work and the explicit `.upload()` calls after the local-path variants (e.g. collections.py 90, 124) are redundant — but the mixed styles obscure that and invite the (incorrect) assumption that the FCPath variants do not upload.
- WHY: Double-logging tracebacks and two different invocation conventions for the same API reduce clarity; the broad `except Exception` also violates the ruff `BLE` guidance even though it re-raises.
- Impact: Cosmetic / maintainability. Pick one `template.write` convention (pass the `FCPath`, drop the redundant `get_local_path()`+`upload()`), and either remove the catch-log-reraise or narrow it.

### CODE-PDS4-004 — Redundant/inconsistent parent-dir creation for global index .tab files
- Severity: **Low**
- File/symbol: `src/pds4/collections.py`, `generate_global_index_files`, lines 233-234 (bodies_tab, no mkdir) vs 261-262 (rings_tab, explicit mkdir).
- Description: `rings_tab_local.parent.mkdir(parents=True, exist_ok=True)` is called for the rings index but not for the bodies index in the same `supplemental_dir`. This is benign because `FCPath.get_local_path()` defaults to `create_parents=True` (verified), so both parents are already created — making the rings `mkdir` redundant and the asymmetry confusing.
- WHY: Inconsistent and redundant directory handling; a future change to `get_local_path` defaults could break the bodies path while leaving rings working.
- Impact: None today; cosmetic/robustness. Either remove the redundant `mkdir` or add it symmetrically and rely on it.

### CODE-REPROJ-006 — `cartographic_model.py` uses `Any` for `obs` and rebuilds Backplane unconditionally
> **Tracked by:** #105 — replace pervasive Any / dict[str, Any] with TypedDicts and Protocols at interop boundaries
- Severity: **Low**
- File/symbol: `src/nav/reproj/cartographic_model.py`, `create_cartographic_model` (param `obs: Any` line 45; `bp = oops.backplane.Backplane(obs)` line 95).
- Description: `obs` is typed `Any`, and the function always constructs a fresh `Backplane(obs)` (consistent with the documented thread-safety hazard). Other reproj entry points accept an `override_backplane` to avoid redundant Backplane construction; this one does not, so callers that already hold a Backplane for `obs` pay to rebuild it (latitude/longitude/center_resolution all recomputed).
- WHY: Redundant Backplane construction is the dominant cost; `bodies.reproject` already supports `override_backplane` for exactly this reason. `Any` typing also weakens mypy-strict coverage at the boundary.
- Impact: Efficiency only (extra Backplane build per cartographic-model call) and weaker typing. Consider an optional `override_backplane` parameter mirroring `BodyMosaic.reproject`.

---



---

# Part 4 — Feature, Annotation, Simulator, Config, Util & Experiments

# Critique: feature / annotation / sim / config / util / experiments

Files reviewed (31): `src/nav/feature/__init__.py`, `src/nav/feature/feature.py`,
`src/nav/feature/feature_type.py`, `src/nav/feature/geometry.py`,
`src/nav/feature/flags.py`, `src/nav/feature/constants.py`,
`src/nav/feature/composition.py`, `src/nav/feature/reliability.py`,
`src/nav/config/__init__.py`, `src/nav/config/config.py`,
`src/nav/config/config_helper.py`, `src/nav/config/logger.py`,
`src/nav/sim/__init__.py`, `src/nav/sim/render.py`, `src/nav/sim/sim_body.py`,
`src/nav/sim/sim_ring.py`, `src/nav/annotation/__init__.py`,
`src/nav/annotation/annotation.py`, `src/nav/annotation/annotation_text_info.py`,
`src/nav/annotation/annotations.py`, `src/util/report_profile.py`,
`src/experiments/backplanes/smoke_backplanes_sim.py`,
`src/experiments/compare_mosaics.py`,
`src/experiments/correlation/check_corr_offset.py`,
`src/experiments/correlation/sweep_upsample_factor.py`,
`src/experiments/correlation/upsampled_dft.py`,
`src/experiments/fov_twist/find_fov_twist.py`, `src/experiments/nav_master1.py`,
`src/experiments/nav_model_body1.py`, `src/experiments/nav_model_stars1.py`,
`src/experiments/offset_sensitivity/analyze_offset_results.py`,
`src/experiments/offset_sensitivity/generate_offset_tasks.py` (32 paths; the two
`offset_sensitivity` files plus the others make the in-scope total — all were read
in full).

Determinism summary for the simulator: there is **no time-based seeding and no
module-level RNG mutation** anywhere in `src/nav/sim`. All randomness flows through
local `np.random.RandomState(seed)` instances seeded from an explicit
`random_seed` (default 42) or a deterministic `hash((axes, center))` fallback
(`hash` randomization only affects str/bytes, not float tuples, so this is stable).
Seeding is correct. The findings below concern *quality / self-consistency* of the
noise/PSF/crater model and per-scene seed sharing, not seed nondeterminism.

---

## Findings

### CODE-CFG-1 — `update_config` does a shallow (depth-1) merge; nested user overrides clobber sibling keys
- Severity: High
- File: `src/nav/config/config.py`, `Config.update_config` (lines 235-239)
- Description: The merge loop is
  `self._config_dict[key].update(new_config[key])`. `.update()` is a shallow dict
  merge: it replaces every *second-level* value wholesale. A user
  `nav_default_config.yaml` that sets, e.g., `bodies: {foo: {a: 1}}` to tweak a
  single sub-key will replace the entire `bodies.foo` sub-dict, dropping any
  default keys under `foo`. The CLAUDE.md "user-level overrides" contract implies
  a deep merge.
- WHY: Users reasonably expect to override one nested tunable without re-stating
  the whole sub-block; the current behavior silently drops the rest of the
  defaults for that sub-block.
- Impact: Silent loss of default config values on partial nested override —
  hard-to-diagnose runtime misconfiguration (e.g. a body losing its radii because
  the user only meant to change its albedo).

### CODE-CFG-2 — `read_config(reread=True)` after the no-path branch leaves stale state / does not re-init `_config_dict`
- Severity: Medium
- File: `src/nav/config/config.py`, `Config.read_config` (lines 158-170)
- Description: When `config_path is None` and `reread=True`, the method does NOT
  clear `self._config_dict` before re-globbing; it calls
  `update_config(filename, read_default=False)` for each file, which *merges into*
  the already-populated dict. A reread therefore unions old and new keys rather
  than producing a clean reload. Keys removed from the YAML files between reads
  survive. (The early-return guard `if not reread and self._config_dict` is also
  the only place `reread` is honored; the path!=None branch ignores `reread`
  entirely and always reassigns, which is inconsistent.)
- WHY: "reread" implies a fresh load; the no-path branch instead does an additive
  merge over stale state.
- Impact: Tests or long-running processes that mutate config files and call
  `read_config(reread=True)` can observe leftover keys. Confirm by setting a key,
  removing it from a config file, and rereading; the key persists.

### CODE-CFG-3 — `category()` and several section properties return fresh `AttrDict` copies, so writes silently no-op; `category()` is not cached
- Severity: Low
- File: `src/nav/config/config.py`, `Config.category` (lines 244-248) and
  `planets`/`satellites`/`fuzzy_satellites`/`ring_satellites`
- Description: `category()` builds a brand-new `AttrDict` on every call from
  `self._config_dict.get(category, {})`, while the named properties
  (`general`, `offset`, ...) return cached `AttrDict`s built once in
  `_update_attrdicts`. Two access styles with different identity/caching semantics
  for the same data. Mutating the object returned by `category()` does nothing.
- WHY: Inconsistent and a latent foot-gun; duplicated access paths.
- Impact: Low (config is treated read-only in practice), but the inconsistency
  invites bugs and wastes allocations on hot paths.

### CODE-CFG-4 — `logger.py` imports the stdlib `logging` module inside the nav tree
- Severity: Low
- File: `src/nav/config/logger.py` (line 17), also used at lines 124, 149
- Description: CLAUDE.md states the stdlib `logging` module must never be imported
  in core code (`nav.feature`, `nav.nav_model`, `nav.nav_orchestrator`,
  `nav.nav_technique`, `nav.support`). `nav.config` is not on that explicit list,
  and the import is only used for the `logging.Handler` / `logging.FileHandler`
  *type annotations* on the pdslogger-backed handlers — a legitimate need. Flag
  only so the reviewer can confirm `nav.config` is intentionally exempt; if so,
  no change. If the intent is zero stdlib-`logging` imports tree-wide, replace the
  annotations with `pdslogger` types or `Any`.
- WHY: Convention scope is ambiguous for `nav.config`.
- Impact: Cosmetic / convention only.

### CODE-SIM-1 — All bodies in a combined scene share one crater seed, producing identical crater fields
- Severity: Medium
- File: `src/nav/sim/render.py`, `_render_combined_model_cached` (line 766,
  `seed=random_seed`) and `_render_single_body` (line 384)
- Description: `_render_single_body` is called with `seed=random_seed` for every
  body, and `_render_body_shape_cached` is also keyed by that seed. Two bodies with
  the same axes/shape parameters but distinct identities will therefore (a) get the
  exact same crater pattern, and (b) collide in the shape cache (same key →
  returns the same array). The same is true in `_render_bodies_positioned_cached`
  (line 275: `body_seed = seed if seed is not None else params.get('seed')`).
  Per-body seed (`params['seed']`) is only honored when the global seed is `None`.
- WHY: A self-consistent synthetic scene wants independent surface texture per
  body; sharing the seed defeats that and makes the crater model degenerate for
  multi-body fixtures.
- Impact: Reduced realism / correlated textures in multi-body simulated images;
  potential to bias correlation-based nav tests. Recommend mixing the body
  name/index into the seed (e.g. `seed ^ hash(body_name)`).

### CODE-SIM-2 — Craters disable limb anti-aliasing; AA edge is forcibly zeroed
- Severity: Medium
- File: `src/nav/sim/sim_body.py`, `_add_craters_and_shading` (line 479:
  `intensity_out[~ellipse_mask_nz] = 0.0`) vs `create_simulated_body` AA path
  (lines 135-138)
- Description: When `crater_fill > 0`, the returned intensity is hard-masked to
  `ellipse_dist_sq < 1.0` (strict), zeroing the soft AA rim that the non-crater
  path (`_lambertian_shading`) preserves. So `anti_aliasing` has effect only when
  there are no craters; with craters the limb becomes a hard step even at high
  `aa_scale`. The supersampling box-filter downsample (line 195) still runs, giving
  partial mitigation, but the intended smooth-rim model is lost. The two shading
  paths are also near-duplicates (illumination vector + Lambert clip computed twice
  with slightly different sign conventions — see CODE-SIM-3).
- WHY: The AA contract in the docstring ("Only affects the edge") is silently
  violated for the crater path.
- Impact: Inconsistent limb sharpness between cratered and non-cratered bodies;
  the DT/limb techniques that consume simulated limbs see a different edge profile
  depending on an unrelated parameter.

### CODE-SIM-3 — Two divergent illumination conventions between the crater and no-crater shaders
- Severity: Medium
- File: `src/nav/sim/sim_body.py`, `_lambertian_shading` (lines 249-250) vs
  `_add_craters_and_shading` (lines 459-460)
- Description: The in-plane illumination unit vector is computed with *swapped*
  axis assignments:
  - `_lambertian_shading`: `illum_v_2d = -cos(angle)`, `illum_u_2d = sin(angle)`,
    then the normal is rotated back through `cos_rz/sin_rz`.
  - `_add_craters_and_shading`: `lx_2d = sin(angle)` (u), `ly_2d = -cos(angle)`
    (v), and the surface normal comes from a height-field gradient with **no**
    `rotation_z` back-rotation applied to the lighting.
  The crater path never rotates the lighting (or the gradient frame) by
  `rotation_z`, so for a body with non-zero `rotation_z` the lit hemisphere of a
  cratered body will not match the lit hemisphere of the same body rendered
  without craters. The phase/terminator line will be placed differently.
- WHY: Two code paths intended to model the same physics use different, partly
  incompatible conventions.
- Impact: For `rotation_z != 0` cratered bodies the terminator is wrong relative
  to the smooth model; any test comparing simulated body against a NavModel limb
  could mis-locate. Confirm by rendering one body with `crater_fill=0` and one with
  `crater_fill>0` at `rotation_z=pi/2`, same illumination — the bright sides
  differ.

### CODE-SIM-4 — GAP/RINGLET composition overwrites background with shaded value instead of compositing
> **Tracked by:** #84 — Fix simulated ring edges and gaps
- Severity: Medium
- File: `src/nav/sim/render.py`, `_render_combined_model_cached` GAP branch
  (lines 741-758) and RINGLET branch (lines 724-740)
- Description: For range-ordered composition the code renders a ring into a scratch
  buffer and then does `img[ring_mask] = ring_img[ring_mask]` (RINGLET) or
  `img[ring_mask] = temp_bg[ring_mask]` (GAP, where `temp_bg` started as all-ones).
  This *replaces* whatever was already in `img` at those pixels (background noise,
  stars, farther rings) with the ring/gap coverage value. The GAP path is worse: it
  writes `1.0 - gap_coverage` over the real scene, so a partial gap leaves a near-
  white patch rather than darkening the existing background. The single-ring
  `render_ring` (sim_ring.py lines 413, 440) correctly *adds*/*subtracts*; the
  combined compositor does not use that additive path for range ordering.
- WHY: The compositor conflates "ring coverage" with "final pixel value" and
  discards the existing scene under the ring footprint.
- Impact: Background and underlying features vanish under rings; gaps brighten
  instead of darken. Affects multi-layer simulated scenes (rings + noise + stars).

### CODE-SIM-5 — `_render_*` use `lru_cache(maxsize=1)`, so any parameter change is a full recompute and nested caches thrash
- Severity: Low
- File: `src/nav/sim/render.py` (`_render_stars_cached` maxsize=1,
  `_render_bodies_positioned_cached` maxsize=1, `_render_background_noise_cached`
  maxsize=1, `_render_background_stars_cached` maxsize=1,
  `_render_combined_model_cached` maxsize=1)
- Description: Every one of the inner caches holds a single entry. The combined
  renderer calls the star/noise/body sub-renderers; alternating between two scenes
  (e.g. a GUI toggling `ignore_offset`, or a sweep over offsets) evicts on every
  call, so the caches provide no benefit and add JSON-serialization overhead
  (`json.dumps(..., sort_keys=True)` on every call) on the hot path.
- WHY: Caches sized 1 across a multi-stage pipeline rarely hit.
- Impact: Wasted serialization + recompute; not a correctness issue. Consider
  larger maxsize or removing the inner caches.

### CODE-SIM-6 — `render_stars` / `render_bodies` are dead public API and leak cached mutable objects
- Severity: Low
- File: `src/nav/sim/render.py`, `render_stars` (138-152), `render_bodies`
  (435-491)
- Description: Grep shows only `render_combined_model` is imported outside
  `render.py` (`obs_inst_sim.py`, `nav_create_simulated_image.py`). `render_stars`
  returns `cached_star_list` directly from the `lru_cache`d
  `_render_stars_cached` *without copying*, so any external caller would mutate
  shared cached `MutableStar` objects. `render_bodies` is similarly unused. They
  duplicate logic already inlined in `_render_combined_model_cached`.
- WHY: Dead/duplicated code with an aliasing hazard if revived.
- Impact: Maintenance burden and a latent aliasing bug; safe to delete or route
  the combined path through them.

### CODE-SIM-7 — Body inventory bbox uses `max(axis1,axis2,axis3)/2` for both axes and ignores tilt
- Severity: Low
- File: `src/nav/sim/render.py`, `_render_single_body` (lines 417-426) and
  `_render_bodies_positioned_cached` (lines 314-323)
- Description: `max_dim = max(axis1, axis2, axis3) / 2.0` is used as the half-extent
  for *both* v and u (`v_pixel_size = u_pixel_size = 2*max_dim`). For a non-circular
  or tilted ellipsoid this over- (or under-, after tilt compression) states the
  on-image bbox. The reported inventory extent will not match the rendered
  silhouette.
- WHY: A single scalar cannot represent an anisotropic, tilted projection.
- Impact: Downstream consumers of `inventory` (hit-testing, fixtures) get a coarse
  bbox. Low because inventory is diagnostic.

### CODE-SIM-8 — `_render_stars_cached` star-flux scaling reads as inverted / mislabeled
- Severity: Low
- File: `src/nav/sim/render.py`, lines 59, 116
- Description: `star.dn = 2.512 ** -(star.vmag - 4.0)`. For brighter stars (smaller
  vmag) `dn` is larger — correct direction. But the comment at line 114 says
  "vmag=0 -> peak=1" while `scale_factor = star.dn / (2.512**4.0)` gives
  `2.512**-(vmag-4) / 2.512**4 = 2.512**-vmag`, so peak=1 occurs at vmag=0
  (consistent) — yet a vmag=4 star (the documented zero-point) gets peak
  `2.512**-4 ≈ 0.025`. The magnitudes are then clipped into `[0,1]` by
  `render_stars`. The model is internally consistent but the "vmag=4" zero-point in
  the comment and `vmag` default (8.0) yield essentially invisible default stars
  (peak ≈ 0.0006). Worth confirming this is intended.
- WHY: Comment/zero-point mismatch; default star is below noise.
- Impact: Confusing photometry; default `vmag=8` stars render near-black. Confirm
  intended dynamic range.

### CODE-SIM-9 — `e >= 1.0 -> e = 0.99` silently corrupts ring geometry
- Severity: Low
- File: `src/nav/sim/sim_ring.py`, `compute_edge_radius_at_angle` (lines 96-97),
  `_compute_edge_radii_array` (lines 137-138)
- Description: When `ae/a >= 1` the eccentricity is silently clamped to 0.99 with
  no warning. A caller that mis-specifies `ae`/`a` gets a plausible-but-wrong
  ellipse instead of an error. There is no pdslogger here (sim is allowed bare
  computation) but a silent clamp of a physical impossibility is a footgun.
- WHY: Silent correction of invalid input.
- Impact: Wrong simulated ring radius with no diagnostic. Low (sim-only).

### CODE-FEAT-1 — `composition.py` reaches into `geometry.bbox_extfov_vu` / `vertices_vu` via `getattr`/attribute access without exhaustively covering the sum type
- Severity: Low
- File: `src/nav/feature/composition.py`,
  `compose_template_features` (line 74, `feature.geometry.bbox_extfov_vu`) and
  `compose_dialog_overlay` (line 143, `getattr(feature.geometry, 'vertices_vu')`)
- Description: `compose_template_features` assumes every template-bearing feature's
  geometry has `bbox_extfov_vu`. That holds for the three template geometries
  (BODY_DISC, RING_ANNULUS, CARTOGRAPHIC_MODEL) but is enforced only by the
  `template_img is not None` filter, not by the type system — a future template
  geometry without a bbox would `AttributeError` at runtime. The `getattr(...,
  'vertices_vu', None)` duck-typing in `compose_dialog_overlay` similarly bypasses
  the typed sum-type and would silently skip a renamed field. Under mypy-strict the
  attribute access on the union `NavFeatureGeometry` is only sound because all
  template variants happen to share the field; this is fragile.
- WHY: The carefully-typed geometry sum type is consumed via stringly-typed
  attribute lookups, defeating the static guarantees the package was built for.
- Impact: Latent `AttributeError` / silent-skip if the geometry set grows.
  Recommend `isinstance` dispatch or a shared protocol/base with `bbox_extfov_vu`.

### CODE-FEAT-2 — `NavReliabilityBreakdown` / flag dataclasses validate ranges, but the reliability score itself is not cross-checked against the breakdown
- Severity: Low
- File: `src/nav/feature/feature.py` (`NavReliabilityBreakdown`, lines 28-70),
  `reliability.py`
- Description: This is a design observation, not a bug: `reliability` is validated
  `[0,1]` but is wholly decoupled from `reliability_reasons`; nothing ensures the
  reported components are consistent with the scalar score. Given that the gate
  (`FeatureReliabilityGate.apply`) keys solely on the scalar, an extractor bug that
  produces a high scalar with a contradictory breakdown is undetectable here.
- WHY: The breakdown exists for curation but has no validating relationship to the
  number that actually gates.
- Impact: None today; flag for the math review that owns the scoring formulas.

### CODE-ANNO-1 — `draw_rect` (used by the star-marker path) does not clip; mitigated only by manual half-width clamp
- Severity: Low
- File: `src/nav/feature/composition.py`, `_paint_star_marker` (lines 205-216) +
  `src/nav/support/image.py`, `draw_rect` (805-847)
- Description: `draw_rect` uses raw numpy slicing with no negative-index guard, so a
  negative coordinate would wrap around to the far edge and paint a spurious
  rectangle. `_paint_star_marker` guards this by clamping `v_half/u_half` to
  `min(..., v_int, h-1-v_int)` and bailing to a single pixel when edge-tight — so
  the *current* caller is safe. But the safety lives entirely in the caller; any
  other use of `draw_rect` with an off-image center is a latent corruption bug.
- WHY: An unclipped low-level primitive whose safety is delegated to every caller.
- Impact: Low (current callers guard it). Consider clipping inside `draw_rect`
  itself, as `draw_circle` already does.

### CODE-ANNO-2 — `annotation_text_info.py` has a `# TODO Add error handling` font loader that will raise an unhelpful error on a missing font
- Severity: Low
- File: `src/nav/annotation/annotation_text_info.py`, `_load_font` (lines 30-43)
- Description: `ImageFont.truetype(path, size)` is called with no error handling
  (the TODO acknowledges it). A missing/invalid `truetype_font_dir` surfaces as a
  raw PIL `OSError` deep inside the annotation loop rather than a clear config
  error pointing at `general.truetype_font_dir`.
- WHY: Operator-facing failure mode is opaque.
- Impact: Low; cosmetic error quality.

### CODE-ANNO-3 — `AnnotationTextInfo` stores `self._config = DEFAULT_CONFIG` but never uses it
- Severity: Low
- File: `src/nav/annotation/annotation_text_info.py` (line 68)
- Description: `self._config` is assigned the global singleton and never read (the
  font dir is passed in via `tt_dir` from `annotations.py`). Dead field; also
  hard-wires the singleton instead of accepting an injected `Config` like the rest
  of the package.
- WHY: Dead code / inconsistent config injection.
- Impact: None functional; cleanup.

### CODE-UTIL-1 — `report_profile.py` hardcodes `./prof/combined.prof`, no docstrings, no arg
> **Tracked by:** #99 — wire up or delete orphan src/util/report_profile.py
- Severity: Low
- File: `src/util/report_profile.py` (whole file)
- Description: A 12-line throwaway: hardcoded relative path, no module/function
  docstring (violates the Google-docstring "every module/function gets one" rule),
  no argument to choose the profile file, dead commented `print_stats()`. It is a
  shipped `src/util` module (not under the lint-excluded `experiments/`), so it is
  in scope for ruff/mypy and the docstring convention.
- WHY: Convention violation (missing docstrings) in a non-excluded path; not
  parameterized.
- Impact: Low; either add docstrings + an argparse path arg, or move it under
  `experiments/`.

### CODE-EXP-1 — Broad `except Exception` in experiment scripts
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
- Severity: Low (experiments excluded from lint/mypy per CLAUDE.md)
- File: `src/experiments/fov_twist/find_fov_twist.py` (line 411),
  `src/experiments/offset_sensitivity/analyze_offset_results.py` (lines 78, 132),
  `src/experiments/compare_mosaics.py` (lines 1142, 1145)
- Description: Several `except Exception`/broad tuples swallow errors and continue.
  Acceptable for throwaway analysis scripts but would fail the project's `B`/`BLE`
  posture if these graduated out of `experiments/`.
- WHY: Broad excepts hide failures.
- Impact: Low — experiments only.

### CODE-EXP-2 — Experiment scripts duplicate large commented URL/body blocks and a duplicated `gaussian_patch`
- Severity: Low (experiments)
- File: `src/experiments/nav_master1.py`, `nav_model_body1.py`,
  `nav_model_stars1.py` (near-identical commented dataset menus);
  `src/experiments/correlation/upsampled_dft.py` &
  `sweep_upsample_factor.py` (both define `gaussian_patch`)
- Description: Heavy copy-paste of commented dataset URLs and a duplicated
  `gaussian_patch` helper across correlation experiments.
- WHY: Duplication / dead commented code.
- Impact: Low — experiments only; cleanup if any graduate to the test suite.

---



---

# Part 5 — Support (Shared Infrastructure)

# Code critique: `src/nav/support/`

Files reviewed (18): `__init__.py`, `attrdict.py`, `constants.py`, `correlate.py`,
`distance_transform.py`, `file.py`, `filter_combo.py`, `filters.py`, `flux.py`,
`image.py`, `image_quality.py`, `misc.py`, `nav_base.py`, `noise_estimate.py`,
`status_reason.py`, `summary_png.py`, `time.py`, `types.py`.

Per the task brief, the core *math* of `correlate.py` and `distance_transform.py`
was deeply reviewed previously; findings below for those files are restricted to
NEW issues in error handling, types, conventions, and dead/structural problems.

---

## Findings

### CODE-SUPPORT-001 — `flux.py` is 1174 lines, ~99.9% dead commented code, over the module-size limit
> **Tracked by:** #96 — prune dead code (flux.py, correlate_old.py, commented-out blocks), #97 — split oversized modules exceeding 1000-line rulebook limit
Severity: **Medium**
File: `src/nav/support/flux.py` (entire file; only live symbol is `clean_sclass`, lines 561-568).

Description: The whole module is commented-out CISSCAL/flux/star-photometry code
except a single 8-line function `clean_sclass`. The file has zero live imports and
is 1174 lines, violating the "modules under 1000 lines" convention. The module
docstring in `__init__.py` (lines 34-36) even advertises it as "Legacy flux and
filter-convolution experiments; most of the implementation is commented out but
kept for reference."

WHY: The project conventions cap modules at 1000 lines and forbid backwards-compat
/ reference shims unless explicitly requested. 1166 lines of commented IDL-port
code is pure dead weight that every grep, import-scan, and reader pays for.

Impact: Maintenance noise; misleading line counts; the lone live function should
live with the other star helpers, not be buried under 560 lines of comments.

---

### CODE-SUPPORT-002 — `clean_obj` mutates its caller's dict/list in place
Severity: **Medium**
File: `src/nav/support/file.py`, `_clean_dict` (lines 49-61), `_clean_list` (lines 64-78), via `clean_obj` (lines 9-25), reached from `dump_yaml` and `json_as_string`.

Description: `_clean_dict` writes back into the *same* dict (`obj[k] = clean_obj(v)`)
and `_clean_list` rebuilds a list but `clean_obj` is documented as a pure converter
("Returns: The object with all NumPy types converted"). For a dict argument the
original caller object is mutated: every NumPy scalar in the caller's structure is
replaced by a Python native in place. `json_as_string(metadata)` and
`dump_yaml(data)` therefore silently rewrite the metadata dict the caller still
holds (see `navigate_image_files.py:129,142`).

WHY: A "clean/serialize" helper that advertises a return value but also mutates its
input is a latent aliasing bug: a caller that serializes a dict and then inspects
the same dict afterward gets different (already-converted) values, and any later
re-serialization or equality check can be affected. It also makes `clean_obj` unsafe
to call on shared/cached structures.

Impact: Currently benign because callers discard the dict after serializing, but it
is a footgun. Confirm by `d = {'x': np.int64(3)}; clean_obj(d); type(d['x'])` →
`int`, proving in-place mutation.

---

### CODE-SUPPORT-003 — `next_power_of_2(0)` returns 2 and `next_power_of_2` accepts/ignores negatives
Severity: **Low**
File: `src/nav/support/image.py`, `next_power_of_2` (lines 173-186).

Description: For `n == 0`, `bin(0)[2:]` is `'0'`, `count('1') == 0` (not 1), so the
function returns `1 << len('0')` = `1 << 1` = `2`, which is not "the smallest power
of 2 >= 0" (that would be 1). For negative `n`, `bin(-4)` is `'-0b100'`, slicing
`[2:]` yields `'b100'`, producing nonsense. There is no input validation.

WHY: `pad_array_to_power_of_2` calls this on `data.shape` entries; a zero-length
axis (degenerate array) would pad to 2 silently rather than erroring.

Impact: Low — shapes of 0 are unusual — but the function's stated contract is
violated and the negative case is undefined behavior. Confirm: `next_power_of_2(0)`
returns `2`.

---

### CODE-SUPPORT-004 — `shift_array` / `pad_array` / `unpad_array` return the input array unchanged on no-op (aliasing inconsistency)
Severity: **Medium**
File: `src/nav/support/image.py`, `shift_array` (lines 43-44), `pad_array` (lines 85-86), `unpad_array` (lines 115-116).

Description: Each function has an early `if all(x == 0 ...): return array` that
returns the *same* object the caller passed in, whereas the non-trivial path returns
a fresh array (`array.copy()` / `np.pad` / a new slice-view). The contract is
therefore inconsistent: sometimes the result aliases the input, sometimes not.
`shift_array`'s docstring says "Returns: The array shifted by the given amount"
without noting the no-copy fast path. `unpad_array` with non-zero margin returns a
*view* (slice) of the input, while `shift_array` returns a copy — so even the
non-trivial paths disagree on aliasing.

WHY: Callers that mutate the returned array (these are drawing/padding helpers used
in image manipulation) will silently corrupt their source on the zero-offset path
but not otherwise, making bugs scene-dependent and hard to reproduce.

Impact: Latent in-place corruption / order-dependent behavior. Either always copy or
document the aliasing precisely. Confirm by `a = np.zeros((4,4)); shift_array(a, [0,0]) is a`
→ `True` vs `shift_array(a, [1,0]) is a` → `False`.

---

### CODE-SUPPORT-005 — `apply_filter` null-sigma short-circuit returns raw intensity for `GRADIENT_OF_GAUSSIAN` and `MORPH_DILATE`, which is semantically wrong
Severity: **Medium**
File: `src/nav/support/filters.py`, `apply_filter` (lines 269-273), `_largest_sigma` (lines 109-131).

Description: The universal short-circuit treats any spec whose largest sigma is below
`null_filter_threshold_sigma` (default 0.4) as identity and returns `arr` unchanged.
For `ISOTROPIC_GAUSSIAN`/`ANISOTROPIC_GAUSSIAN`/`BANDPASS_DOG` returning the input is
defensible (tiny blur ~ identity). But for `GRADIENT_OF_GAUSSIAN` the operation is a
*gradient magnitude* — identity returns the raw intensity image, not a near-zero
gradient. For `MORPH_DILATE`, identity returns the un-dilated array (the dilate path
itself already returns `arr` when `half_width <= 0`, but a sub-threshold sigma like
0.3 short-circuits before reaching `_apply_morph_dilate`).

WHY: A technique that requests `GRADIENT_OF_GAUSSIAN` with a small sigma silently
receives raw intensities instead of an edge image, changing the meaning of the
downstream matching metric without any error. The short-circuit conflates "blur is
negligible" with "operation is identity," which only holds for the blur kinds.

Impact: Wrong filtered output for two of seven kinds when sigma is small. Restrict
the null-sigma short-circuit to the blur-family kinds (mirroring how
`DISTANCE_TRANSFORM` is already excluded on line 270).

---

### CODE-SUPPORT-006 — `_apply_anisotropic_gaussian` crops only top-left after rotate, losing centering and risking shape mismatch
Severity: **Low**
File: `src/nav/support/filters.py`, `_apply_anisotropic_gaussian` (lines 158-169).

Description: After `rotate(..., reshape=False)` twice, the comment claims rotate "may
shift the array by sub-pixel; trim/pad back to original shape" and does
`out = out[: arr.shape[0], : arr.shape[1]]`. With `reshape=False`, scipy's `rotate`
already returns the same shape, so this slice is a no-op for shape but does nothing
for the acknowledged sub-pixel shift, and if a future scipy ever returned a larger
array the top-left crop would discard a centered result asymmetrically.

WHY: The code documents a centering concern then applies a crop that does not address
it; the slice is dead for its stated purpose and misleading.

Impact: Low (currently a no-op). Either remove the slice or replace it with a true
center-crop/pad to make intent and effect agree. Confirm scipy `rotate(reshape=False)`
preserves shape.

---

### CODE-SUPPORT-007 — `mad_std` returns NaN on empty input and has no guard
Severity: **Low**
File: `src/nav/support/misc.py`, `mad_std` (lines 115-119); reached by `noise_estimate.estimate_image_noise_sigma` and `correlate.evaluate_candidate`.

Description: `mad_std([])` computes `np.median([])` → NaN (with a RuntimeWarning) and
returns NaN. `estimate_image_noise_sigma` guards against an empty *masked* selection
but a caller passing an all-NaN or empty array to `mad_std` directly gets a silent
NaN. In `evaluate_candidate`, `sigma_n = mad_std(resid)` feeds `fisher_covariance`;
a NaN sigma propagates into the covariance and `sigma_xy`.

WHY: Robust-noise estimation is a shared, high-leverage helper; a NaN leaking through
poisons every downstream confidence/covariance computation with no error.

Impact: Low-to-Medium depending on caller; `mad_std` should reject empty/all-NaN
input or document the NaN return. Confirm: `mad_std([])` returns `nan` with a warning.

---

### CODE-SUPPORT-008 — `array_zoom` annotates `result: np.ndarray` (bare generic) under mypy strict
> **Tracked by:** #105 — replace pervasive Any / dict[str, Any] with TypedDicts and Protocols at interop boundaries
Severity: **Low**
File: `src/nav/support/image.py`, `array_zoom` (line 268).

Description: `result: np.ndarray = np.asarray(a)` uses the unparameterized
`np.ndarray`. Everywhere else the module uses `NDArrayType[NPType]`. Under
`mypy --strict`, bare `np.ndarray` is `ndarray[Any, Any]` and weakens type checking;
it is inconsistent with the file's own conventions.

WHY: Project mypy is strict; the rest of the module is carefully parameterized.

Impact: Type-safety erosion; cosmetic but against conventions. Confirm via
`mypy src/nav/support/image.py` (may currently pass because `np.ndarray` is allowed,
but it defeats the generic propagation the signature promises).

---

### CODE-SUPPORT-009 — `current_git_version` / `get_local_host_name` swallow all exceptions with bare `except Exception`
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
Severity: **Low**
File: `src/nav/support/misc.py`, `current_git_version` (lines 137-143), `get_local_host_name` (lines 170-174).

Description: Both wrap their body in `except Exception:` and cache a sentinel string.
This is broad-except (ruff `BLE`/style), and it also caches the *failure* permanently
for the process — a transient `subprocess`/`getfqdn` hiccup poisons every later call
with `'GIT DESCRIBE FAILED'` / `'LOCAL HOST NAME FAILED'`.

WHY: The project bans broad excepts in core code; and caching a transient failure
forever is surprising. `subprocess.check_output` can raise `OSError`,
`CalledProcessError`, `FileNotFoundError` — catching exactly those is clearer.

Impact: Low — these are diagnostic-logging helpers — but the convention violation and
sticky-failure caching are worth narrowing.

---

### CODE-SUPPORT-010 — `draw_rect` docstring describes `yhalfwidth` incorrectly
Severity: **Low**
File: `src/nav/support/image.py`, `draw_rect` (lines 805-826).

Description: The docstring says `yhalfwidth: This is the inner border of the
rectangle.` while `xhalfwidth: The width of the rectangle on each side of the center.`
By symmetry `yhalfwidth` is plainly the vertical half-width; the docstring is a
copy-paste error. Several parameters are also documented out of signature order.

WHY: Misleading docstring on a public drawing helper.

Impact: Documentation-only; trivial fix.

---

### CODE-SUPPORT-011 — `summary_png.grayscale_to_rgb_with_quantile_stretch`: `clip_quantile` can go to 0 or negative for tiny images
Severity: **Low**
File: `src/nav/support/summary_png.py`, `grayscale_to_rgb_with_quantile_stretch` (lines 121-126).

Description: `clip_count = min(default_clip_count, max(1, n_bright // 20))` then
`clip_quantile = 1.0 - clip_count / n_finite`. For a 1-pixel finite image
(`n_finite == 1`), `clip_count` is at least 1, so `clip_quantile == 0.0`, making
`white = quantile(values, 0.0) = min == black`; the subsequent `white <= black`
guard (line 125) saves it. But if `clip_count > n_finite` were ever reachable,
`clip_quantile` would be negative and `np.quantile` raises. `default_clip_count`
is `max(1, round(n_finite*0.001))` so it is bounded by ~n_finite, but the
`n_bright // 20` branch is independently bounded by the count of bright pixels
(`<= n_finite`), so `clip_count <= n_finite` always holds — the negative case is
not actually reachable. The black==white degenerate case is the only live edge and
is handled.

WHY: Worth recording that the `white <= black` guard is load-bearing for tiny/flat
images; remove it and small-image rendering breaks.

Impact: Very low (currently safe). Flag as a brittle invariant; a defensive
`clip_quantile = max(clip_quantile, small)` or a comment documenting the bound would
harden it. Confirm by feeding a 1x1 finite image.

---

### CODE-SUPPORT-012 — `evaluate_candidate` assumes model fits inside image; `crop_center` raises if model smaller than image
Severity: **Medium**
File: `src/nav/support/correlate.py`, `evaluate_candidate` (lines 449-454), uses `crop_center` (`image.py:148-170`).

Description: `evaluate_candidate` does
`model_shift = fourier_shift(model_pad[:model_h, :model_w], dy, dx)` then
`crop_center(model_shift, (image_h, image_w))`. `crop_center` raises
`ValueError("Output shape ... cannot be larger than image shape ...")` when
`image_h > model_h` or `image_w > model_w`. The pyramid driver
(`navigate_with_pyramid_kpeaks`) explicitly supports a model that is *padded larger*
than the image (docstring lines 675-678: "It does not need to be the same size as
the image"), but the inverse — a model *smaller* than the image — is silently
unsupported and will throw deep inside refinement rather than at the API boundary.

WHY: The public function advertises arbitrary model/image size relationships; one
direction crashes with an opaque error from a helper two calls deep instead of a
validated, intentional failure.

Impact: Medium — a mis-sized model produces a confusing `ValueError` from
`crop_center` rather than a clear contract error. Confirm by calling
`navigate_single_scale_kpeaks` with `model` smaller than `image` in both dims.

---

### CODE-SUPPORT-013 — `navigate_with_pyramid_kpeaks` crashes with IndexError when `pyramid_levels <= 0`
Severity: **Low**
File: `src/nav/support/correlate.py`, `navigate_with_pyramid_kpeaks` (lines 838-919).

Description: With `pyramid_levels == 0`, the `for lvl in range(pyramid_levels, 0, -1)`
loop never runs, `level_shifts` stays empty, and `shifts_arr[-1]` (line 918) raises
`IndexError` on an empty array. No validation rejects a non-positive level count.

WHY: A shared correlation entry point should fail with a clear message on invalid
configuration, not an opaque numpy IndexError.

Impact: Low (callers pass >=1), but a config typo (`pyramid_levels: 0`) yields a
cryptic crash. Add an explicit `pyramid_levels >= 1` check.

---

### CODE-SUPPORT-014 — Hard-coded magic numbers throughout `correlate.py` that the rest of the system makes configurable
Severity: **Low**
File: `src/nav/support/correlate.py` — e.g. `_NCC_BIDIR_W_FRAC_MIN=0.3`, `_NCC_BIDIR_VAR_FRAC_MIN=0.1` (lines 109-110); the `at_edge` 2.0-pixel margin (lines 948-951); the `1e6`/`1e3` degenerate-uncertainty sentinels (lines 379, 627-628); the `quality_thresh=6.0`, `consistency_tol=2.0`, `prior_weight_final=0.25` defaults.

Description: These thresholds govern peak rejection and spurious-flagging and are
embedded as module constants / default args rather than sourced from `Config`
(`config.offset` / a correlation section). The functions are free functions with no
`NavBase`/`config` access, so they cannot read project config.

WHY: The project convention favors config-driven tuning over hard-coded constants,
and these specific values (overlap/variance floors, edge margin, quality threshold)
are exactly the kind a navigator wants to tune per mission.

Impact: Low — they have reasonable defaults — but they are invisible to the config
layer; callers in `nav_technique/*` cannot override them without editing module
constants. Consider threading them through as parameters fed from config.

---

### CODE-SUPPORT-015 — `filters._apply_morph_dilate` ignores per-axis sigma; uses only the max as a square structuring element
Severity: **Low**
File: `src/nav/support/filters.py`, `_apply_morph_dilate` (lines 204-214).

Description: `half_width = int(np.ceil(max(spec.sigma_xy)))` collapses the
`(sigma_v, sigma_u)` pair to a single isotropic square element, discarding any
anisotropy the caller encoded in `sigma_xy`. The docstring acknowledges "square
structuring element" but the spec type advertises per-axis `sigma_xy`.

WHY: A caller setting `sigma_xy=(1.0, 5.0)` to dilate an elongated search margin
silently gets a 5x5 square, over-dilating one axis.

Impact: Low — current usages may pass isotropic values — but the API promises
per-axis control it does not honor. Confirm by checking callers' `sigma_xy` for
MORPH_DILATE specs.

---

### CODE-SUPPORT-016 — `time.utc_to_et` docstring example is not parseable by `julian.tai_from_iso` reliably; space-separated form unverified
Severity: **Low**
File: `src/nav/support/time.py`, `utc_to_et` (lines 55-67).

Description: The docstring claims both `"2008-01-01 12:00:00"` (space) and
`"2008-01-01T12:00:00"` (T) are accepted, but the code delegates straight to
`julian.tai_from_iso(utc)` with no normalization. Whether the space form parses
depends entirely on the `julian` library's tolerance; the example is asserted, not
guaranteed by this code.

WHY: A docstring promising an input format the function does not itself enforce is a
latent contract gap; if `julian` rejects the space form, callers relying on the
docstring break.

Impact: Low — needs confirmation against the installed `julian` version. Confirm:
`utc_to_et('2008-01-01 12:00:00')` vs the `T` form; if the space form raises, fix the
docstring or normalize the separator.

---

### CODE-SUPPORT-017 — `correlate.py` retains TODO/commentary blocks masquerading as code logic
Severity: **Low**
File: `src/nav/support/correlate.py` — `evaluate_candidate` (lines 456-463, 477-484), `navigate_single_scale_kpeaks` (lines 617-622), `navigate_with_pyramid_kpeaks` docstring opens with `"""TODO Clean this up` (line 668).

Description: Several multi-line comments are review-note prose embedded in the
function body (e.g. "Consider logging this condition", "Consider returning None or
raising an exception", "the units/scale of quality ... may not be comparable"). The
public docstring of the pyramid driver literally begins with "TODO Clean this up".

WHY: These are unresolved design notes left in production code; they describe known
correctness gaps (prior penalty unit mismatch; silent fallback sigma; `-np.inf`
quality sentinel that downstream may not check) without resolving them.

Impact: Low directly, but they flag real latent issues (the `-np.inf` quality on the
no-candidate path, the prior-penalty scale mismatch) that deserve tracked follow-up
rather than inline TODOs.

---

### CODE-SUPPORT-018 — `__init__.py` module docstring mischaracterizes `flux` and uses time-anchored "Legacy" phrasing
> **Tracked by:** #96 — prune dead code (flux.py, correlate_old.py, commented-out blocks)
Severity: **Low**
File: `src/nav/support/__init__.py` (lines 34-36).

Description: The `flux` entry reads "Legacy flux and filter-convolution experiments;
most of the implementation is commented out but kept for reference." This both
advertises dead code as an API surface and uses the time-anchored "Legacy" framing
that the project's documentation guidance avoids (describe current state, not
migration history).

WHY: Documentation convention; and pointing readers at a module that is 99% comments
is actively misleading.

Impact: Low; resolves naturally if CODE-SUPPORT-001 is addressed.

---

## Notes on items checked and found OK

- `distance_transform.sample_dt_bilinear` / `apply_translation`: shape validation,
  clamping, and bilinear weights are correct; no new issues.
- `masked_ncc` / `_masked_ncc_bidir`: divide-by-zero guards (`_NCC_EPS`, `safe_w`,
  `np.maximum(denom, _NCC_EPS)`), negative-variance clamps, and `-inf` sentinel
  handling are sound; PSR/PMR/PER metrics correctly strip non-finite values before
  reducing. (Math correctness deferred to prior review.)
- `nav_base.NavBase`: config/logger plumbing is correct; uses `IMAGE_LOGGER` per
  convention, no stdlib `logging` import anywhere in the package (verified by grep).
- `attrdict.AttrDict`: the `__dict__ = self` trick is intentional and documented.
- `image_quality` / `noise_estimate`: 2-D guards, mask-shape checks, and the
  `image_noise_sigma <= 0` guard are all present and correct.
- `filter_combo.canonicalize`, `status_reason`, `constants`, `types`: clean.

---



---

# Part 6 — Navigation Orchestrator

# nav_orchestrator — full-depth code critique

Files reviewed (13): `__init__.py`, `orchestrator.py`, `nav_context.py`, `nav_result.py`,
`ensemble.py`, `image_classifier.py`, `image_classifier_result.py`, `instrument_config.py`,
`provenance.py`, `curator.py`, `feature_summary.py`, `status_reason_info.py`,
`image_derivatives.py`.

Prior review covered ensemble's core math (CODE-ORCH-001/002 plus the rotation-wrap and
null-space-tolerance notes) and `confidence.py`; those are not re-derived here. New findings
continue at CODE-ORCH-003 / CODE-DERIV-001.

---

## Findings

### CODE-ORCH-003 — NaN missing-data markers crash `navigate()` for `calibrated_if` images
**Severity: High**
**File/symbol:** `orchestrator.py::NavOrchestrator._make_context` (lines 762-825); interacts with
`image_classifier.py::NavImageClassifier.classify` (lines 121-127),
`image_derivatives.py::_smooth_and_compute_gradients` (lines 152-159), and
`instrument_config.py::_coerce_marker_value` (CISS CALIB sets `marker_value: NaN`,
`config_400_inst_coiss.yaml` line 115).

**Description:** For `calibrated_if` instruments the missing-data sentinel is literally `NaN`
(CISS CALIB `marker_value: NaN`). `_make_context` does `raw_image = obs.extdata.astype('float64')`
with no NaN cleaning, then unconditionally calls `compute_all_image_derivatives(image, noise_sigma, ...)`.
`_smooth_and_compute_gradients` raises `ValueError('image_ext must contain only finite values ...')`
the moment any pixel is NaN. The exception is **not** caught anywhere in `navigate()` — the
classifier/derivative block sits before the technique sandboxes — so it propagates straight out of
`NavOrchestrator.navigate`, violating the documented "the orchestrator must never raise through to
its caller" contract that the four `except Exception` sandboxes exist to uphold.

Compounding correctness problems on the same NaN path even before the crash:
- `NavImageClassifier.classify` computes `miss_mask = sensor == self.thresholds.missing_data_marker_dn`.
  With a NaN marker `NaN == NaN` is always `False`, so `missing_frac` is permanently `0.0` and the
  `mostly_missing_data` / `partial_dropout` paths can never fire for calibrated images.
- `max_dn = float(np.max(sensor))` returns `NaN` when any pixel is NaN, so the `max_dn < blank_max_dn`
  blank short-circuit silently never triggers (NaN comparisons are `False`).
- `estimate_image_noise_sigma` (MAD over the NaN-bearing array) returns `NaN`, which then poisons
  `cr_noise_sigma`, `noise_sigma`, and the gradient threshold.

**WHY:** The `calibrated_if` path was designed knowing the marker is NaN
(`_coerce_marker_value` explicitly maps `"NaN"`/`None` → `float('nan')`), but no stage substitutes
the NaN before the finite-only derivative kernels run, and the classifier's `==` test cannot match
NaN.

**Impact:** Any CISS CALIB frame that actually contains dropout pixels aborts the whole navigation
with an uncaught `ValueError` instead of producing a `NavResult.failed`. Frames with no dropouts work
by luck. Missing-data classification is silently dead for all calibrated instruments.
**Confirm:** Feed `_make_context` an `extdata` containing a single `np.nan` with a CISS-CALIB-style
`inst_config` (`data_units='calibrated_if'`, `marker_value: NaN`) and observe the `ValueError`
escaping `navigate()`; check `classifier.missing_frac == 0.0` on the same input.

---

### CODE-ORCH-004 — Saturation mask + classifier statistics computed on the DC-removed filtered image
**Severity: Medium**
**File/symbol:** `orchestrator.py::_make_context` lines 774-787 (`image, pre_filter = self._apply_source_image_filter(...)`,
then `classifier.classify(image, ...)` and `_build_saturation_mask(image, ...)`).

**Description:** `_apply_source_image_filter` can apply a `BANDPASS_DOG` (difference-of-Gaussians)
filter that removes the DC component of the image. The orchestrator then runs the classifier *and*
`_build_saturation_mask` on that filtered `image`. The saturation mask is `image >= saturation_dn`
(an absolute full-well DN threshold), and the classifier's `blank_max_dn` / `saturation_threshold_dn`
are likewise absolute-DN gates. After a DC-removing bandpass these absolute thresholds are
meaningless — a saturated raw region near full-well DN comes out near zero post-DoG, so saturated
pixels are never flagged, and a bright uniform frame can read as "blank".

The inline comment justifies post-filter classification "so blank/saturation/missing fractions match
what downstream extractors will see," which is reasonable for *missing* fraction but wrong for the
absolute-DN saturation / blank gates.

**WHY:** A single `image` variable is reused for three purposes (derivatives, classifier, saturation)
when only the derivative inputs should see the bandpass; the absolute-DN gates need `raw_image`.

**Impact:** Latent — every shipped `source_image_filter` block is currently `enabled: false`
(`config_400_inst_coiss.yaml`), so no live instrument hits this today. The moment a DoG pre-filter is
enabled for a `raw_dn` camera, saturation flags and the blank/overexposed short-circuits silently
break. **Confirm:** Enable `source_image_filter.kind: BANDPASS_DOG` for COISS NAC and check that a
saturated test frame yields `saturation_frac ≈ 0`.

---

### CODE-ORCH-005 — `signal_dn_to_image_unit_scale` is an uncalibrated placeholder used in live SNR math
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
**Severity: Medium**
**File/symbol:** `instrument_config.py::instrument_settings_from_obs` / `_read_signal_scale`
(lines 244-286); config `config_400_inst_coiss.yaml` line 116
(`signal_dn_to_image_unit_scale: 5.0e-7  # PLACEHOLDER`).

**Description:** For `calibrated_if` the code *requires* `noise.signal_dn_to_image_unit_scale` and
threads it onto `NavContext.signal_dn_to_image_unit_scale`, where (per the field docstring) the STAR
`predicted_snr` formula multiplies a DN-keyed catalog signal by it to compare against an I/F-keyed
noise sigma. The only shipped value for CISS CALIB is `5.0e-7` explicitly marked `PLACEHOLDER —
calibrate in Phase 10`. Worse, the other three "calibrated_if" configs (`gossi`, `vgiss`, `nhlorri`)
set `signal_dn_to_image_unit_scale: 1.0` with the comment "raw_dn — image is already in DN", which
contradicts their own `data_units: calibrated_if` declaration.

**WHY:** The validation in `_read_signal_scale` only checks `> 0` and finiteness; it cannot detect a
wrong order-of-magnitude calibration constant, and the `data_units` vs comment contradiction shows
the configs were copied without recalibration.

**Impact:** Star-based navigation SNR on calibrated images is scaled by an unverified constant; a
wrong scale silently mis-ranks star detectability (too many or too few stars survive), corrupting
every downstream star technique without any error. The contradictory `1.0` values for gossi/vgiss/
nhlorri mean those calibrated cameras are effectively running the raw-DN scale. **Confirm:** Compare
`predicted_snr` output for a known star on a CISS CALIB frame against a hand-calibrated DN→I/F factor.

---

### CODE-ORCH-006 — `provenance` git-subprocess + full config hashing re-run on every image
**Severity: Medium**
**File/symbol:** `provenance.py::collect_provenance_metadata` /
`_resolve_git_sha` / `_resolve_static_data_hashes`; called by `orchestrator.py::_make_provenance`
(line 893), invoked once per `navigate()` and once per `prepare()`.

**Description:** Every single image navigation spawns two `git` subprocesses (`rev-parse` +
`status --porcelain`, each with a 5 s timeout) and sha256-hashes every `config_220_*/config_3*/
config_4*` YAML in `config_files/`. None of this is cached. The git SHA and the static-data hashes
are process-invariant (the repo and shipped config files do not change mid-run); only the SPICE
kernel list legitimately varies per image (different kernels can be loaded for different ETs).

**WHY:** `collect_provenance_metadata` bundles the invariant lookups (git, file hashes) with the
genuinely per-image lookup (kernels) and is called unconditionally per image with no memoization.

**Impact:** Batch runs over thousands of images pay thousands of redundant `git` subprocess
fork/exec pairs and redundant file-hash passes — a measurable per-image latency tax and a hard
dependency on `git` being on `PATH` for every image. **Confirm:** Profile `navigate()` over N images;
git/hash cost scales linearly with N. **Fix direction:** memoize `_resolve_git_sha()` and
`_resolve_static_data_hashes()` at module/process scope; keep `_resolve_spice_kernels()` per image.

---

### CODE-ORCH-007 — `with_prior` raises `ValueError`/`TypeError` swapped vs documented contract
**Severity: Low**
**File/symbol:** `nav_context.py::NavContext.with_prior` lines 128-143.

**Description:** The docstring's `Raises:` says `TypeError: if offset_px is not a length-2 sequence of
numbers`. The code raises **`ValueError`** for the length check (`if len(offset_px) != 2: raise
ValueError(...)`, line 137) and `TypeError` only for the numeric-coercion failure. A caller passing a
3-tuple gets `ValueError`, not the documented `TypeError`.

**WHY:** The length guard was added with a `ValueError` message while the docstring kept the original
`TypeError` classification.

**Impact:** Callers catching `TypeError` to handle malformed priors will miss the length case and let
a `ValueError` escape. Minor but a real contract mismatch. **Confirm:** `with_prior(offset_px=(1,2,3),
covariance_px2=np.eye(2))` raises `ValueError`, not `TypeError`.

---

### CODE-ORCH-008 — Conflicted pass-1 ensemble silently drives the pass-2 prior
**Severity: Low**
**File/symbol:** `orchestrator.py::navigate` lines 450-467.

**Description:** After the pass-1 ensemble, only `status == 'failed'` short-circuits. A
`status == 'conflicted'` pass-1 result (best-vs-runner-up gap below threshold → an explicitly
untrustworthy "best group" offset) falls through and, because it still has `offset_px` and
`covariance_px2`, is installed verbatim as the pass-2 prior via `context.with_prior(...)`. Pass-2
prior-required techniques then refine against a prior the ensemble itself flagged as conflicted.

**WHY:** The branch checks `offset_px is not None and covariance_px2 is not None` (both true for
conflicted) without checking `status`.

**Impact:** When pass-1 is genuinely ambiguous, pass-2 techniques are seeded toward one arbitrary
mode, potentially locking the final ensemble onto the wrong cluster instead of letting pass-2 evidence
break the tie. Behavioral, not a crash. **Confirm:** Construct two well-separated pass-1 groups with
a sub-`agreement_gap` confidence gap and verify the conflicted offset is passed as the pass-2 prior.

---

### CODE-ORCH-009 — Fallback-supersession logic duplicated between orchestrator and ensemble
**Severity: Low**
**File/symbol:** `orchestrator.py::_BODY_FEATURE_PREFIXES` / `_feature_source_bodies` /
`_bodies_with_non_spurious_primary` (lines 210-262) versus
`ensemble.py::_BODY_FEATURE_PREFIXES` / `_source_bodies` / `_technique_tier` /
`_drop_superseded_fallbacks` (lines 162-244).

**Description:** Both modules independently define the identical `_BODY_FEATURE_PREFIXES` tuple and
re-implement "parse `<prefix>:<body>` out of a feature_id" and "which bodies have a non-spurious
primary". The orchestrator pre-filters fallback features by covered body (`_run_pass(excluded_bodies=...)`),
then the ensemble re-derives the same body coverage and drops superseded fallbacks again. The
orchestrator's own docstrings even call the ensemble pass "the redundant downstream gate."

**WHY:** The supersession rule lives in two places with two copies of the prefix list and parsing,
kept in sync only by comment cross-references (`orchestrator.py` line 218 points at
`ensemble._BODY_FEATURE_PREFIXES`).

**Impact:** Maintenance hazard — adding a new body-feature prefix (e.g. a new arc kind) requires
editing both tuples and both parsers, and the two will drift silently. Also a small efficiency cost:
the coverage set is computed twice per image, each with an O(results × registry) nested scan
(`_bodies_with_non_spurious_primary` loops `NavTechnique._registry` per result;
`_technique_tier` does the same per call). **Fix direction:** hoist the prefix tuple + a single
`source_bodies(feature_id)` / `tier_of(name)` helper into a shared module and have both call it.

---

### CODE-ORCH-010 — `NavResult.__post_init__` does not enforce the `conflicted` rank/status invariant
**Severity: Low**
**File/symbol:** `nav_result.py::NavResult.__post_init__` lines 84-99.

**Description:** `__post_init__` enforces `failed ⇒ offset_px is None`, `ok ⇒ offset_px is not None`,
and `confidence_rank=='failed' ⇒ status=='failed'`, but never enforces the symmetric
`confidence_rank=='conflicted' ⇔ status=='conflicted'`. A directly-constructed `NavResult` with
`status='ok', confidence_rank='conflicted'` (or `status='conflicted'` with rank `'high'`) passes
validation. The class docstring advertises direct instantiation as "also supported," so the guard is
the only protection.

**WHY:** The invariant set was written for the `failed` rank only; the `conflicted` rank received no
matching check.

**Impact:** A mis-constructed result that downstream consumers treat as trustworthy ("ok") while the
rank says "conflicted" (or vice-versa) slips through. The canonical `.conflicted()` constructor is
self-consistent, so impact is confined to direct construction / future refactors. **Confirm:**
`NavResult(status='ok', confidence_rank='conflicted', offset_px=(0,0), ...)` constructs without error.

---

### CODE-ORCH-011 — `derive_confidence_rank` collapses "low confidence" and "sigma too large" into one reason
**Severity: Low**
**File/symbol:** `ensemble.py::derive_confidence_rank` (lines 470-507) and its consumer
`ensemble` lines 659-683.

**Description:** When no tier matches, `derive_confidence_rank` returns `'failed'` regardless of
whether the cause was `confidence < min_confidence` or `max_sigma > max_sigma_px`. The ensemble then
maps that to `NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD` even when the confidence was perfectly
adequate and only the sigma constraint failed. The INFO log does print both values, but the persisted
`status_reason` (the machine-readable field downstream code branches on) is mislabeled.

**WHY:** A single `'failed'` sentinel conflates two distinct rejection causes, and the only available
status reason names the confidence cause.

**Impact:** Operators / downstream filters keying on `status_reason` cannot distinguish "the answer was
confident but imprecise" from "low confidence." Diagnostic accuracy only — no numerical error.
**Confirm:** Build a combined estimate with `confidence=0.9` but `sigma=10px`; observe
`status_reason == FINAL_CONFIDENCE_BELOW_THRESHOLD`.

---

### CODE-ORCH-012 — Curator silently maps NaN → 0.0, hiding non-finite covariance/offset entries
**Severity: Low**
**File/symbol:** `curator.py::_round_float` lines 38-44 (used by `_round_pair`, `_round_2x2`,
`_curate_*`).

**Description:** `_round_float` returns `0.0` for any NaN input. NaN offsets are excluded by
`NavResult` invariants, but `covariance_px2` is **not** checked for NaN in `NavResult.__post_init__`
(only finiteness of `with_prior`'s prior is checked, and only squareness here). A NaN slipping into a
combined covariance (e.g. from a degenerate `pinvh`) is rendered as a clean `0.0` in the JSON, which
reads as a zero-variance (infinitely confident) entry — the opposite of the truth.

**WHY:** The NaN→0.0 rule was chosen for "stable byte-identical JSON" but applies uniformly, including
to variance terms where 0.0 is a dangerously misleading value.

**Impact:** A pathological covariance is silently laundered into a JSON that claims perfect certainty.
Low likelihood (requires upstream NaN) but high consequence if it occurs. **Confirm:** Pass a
`NavResult` whose `covariance_px2` contains a NaN through `build_metadata_dict`; the JSON shows
`0.0` with no flag.

---

### CODE-ORCH-013 — `_round_2x2` is misnamed; it serializes any NxN matrix
**Severity: Low (cosmetic)**
**File/symbol:** `curator.py::_round_2x2` (lines 53-62) and its use at lines 115, 209.

**Description:** The function iterates `matrix.shape[0] × shape[1]`, so it correctly serializes the
3x3 rotation-aware covariance, yet it is named `_round_2x2` and the JSON key is hard-coded
`covariance_px2` even when the matrix is 3x3 (rotation block included). The docstring says "Round a
2x2 covariance."

**WHY:** Name/docstring predate the 3-DoF covariance support that the same function now handles.

**Impact:** None functionally; misleading name and an under-described `covariance_px2` key that may
silently be 3x3. **Confirm:** Inspect curated JSON for a `fit_camera_rotation` result — `covariance_px2`
is a 3x3 list under a 2x2-named key.

---

### CODE-DERIV-001 — Edge-DT input mask uses `>` strictly, dropping pixels exactly at threshold; threshold-equality inconsistent with classifier
**Severity: Low**
**File/symbol:** `image_derivatives.py::_directional_nms` line 319
(`np.where(keep & (gradient_magnitude > threshold), 1.0, 0.0)`).

**Description:** The edge mask keeps pixels with `gradient_magnitude > threshold` (strict). This is
internally fine, but note two things worth confirming: (1) when `image_noise_sigma == 0.0`
(allowed — `build_image_edge_dt`/`compute_all_image_derivatives` validate `>= 0`, not `> 0`), the
threshold collapses to `0.0` and **every** non-zero-gradient pixel becomes an edge candidate, so the
NMS-thinned mask can be enormous and the DT degenerates toward all-zero. The docstring claims the DT
"always produces a fully-defined array" via the saturation fallback for an *empty* mask, but the
opposite degenerate case (zero threshold → near-full mask) is not guarded. (2) The directional-NMS
`center >= neighbour` comparisons use `>=`, so on a flat plateau of equal gradient magnitude *both*
plateau pixels are retained (mutual `>=`), widening edges beyond one pixel — acceptable for DT input
but contradicts the "one-pixel-wide edge map" claim in the docstring.

**WHY:** `image_noise_sigma >= 0` is accepted but a literal `0.0` makes the threshold vanish; the
plateau `>=` tie-break keeps both sides.

**Impact:** Mostly cosmetic / robustness. A genuinely zero noise sigma (synthetic / saturated-flat
inputs) yields a meaningless DT that still feeds every DT technique's LM step. **Confirm:** Call
`compute_all_image_derivatives(image, image_noise_sigma=0.0)` on a gradient-rich image and inspect the
near-saturated-to-zero DT. **Fix direction:** clamp the threshold to `max(k*sigma, eps)` or require
`image_noise_sigma > 0` consistently with `_make_context`'s `cr_noise_sigma` clamp (which already does
`max(..., 1e-6)`).

---

### CODE-DERIV-002 — `noise_sigma` fed to derivatives is in image-native units, but the DT threshold mixes scales for `calibrated_if`
**Severity: Low**
**File/symbol:** `image_derivatives.py` docstring (lines 22-27, "DN units") and `build_image_edge_dt`
/ `compute_all_image_derivatives` parameter docs ("MAD-derived noise sigma (DN units)"); fed from
`orchestrator.py::_make_context` line 794 where `noise_sigma` is the classifier's I/F-unit sigma for
calibrated instruments.

**Description:** The derivative module's docstrings repeatedly state the threshold scaling uses noise
sigma in **DN units**, but for `calibrated_if` instruments the orchestrator passes an I/F-unit sigma
(`NavContext.image_noise_sigma` is documented as native units — DN or I/F). The threshold
`edge_threshold_k_sigma * image_noise_sigma` is then applied to the gradient of an I/F image, which is
self-consistent (both gradient and sigma are in I/F), so this is **not** a numerical bug — it is a
documentation defect that asserts "DN" where the value may be I/F.

**WHY:** The derivative module was written DN-first and the docstrings were never generalized when
`calibrated_if` support landed.

**Impact:** Documentation-only; a maintainer trusting "DN units" could wrongly rescale. **Confirm:**
Read `build_image_edge_dt` / `compute_all_image_derivatives` param docs vs `NavContext.image_noise_sigma`
field doc.

---

### CODE-ORCH-014 — `_run_pass` recomputes the available-feature-type set inside the technique loop
**Severity: Low (efficiency)**
**File/symbol:** `orchestrator.py::_run_pass` line 723 (`available_types = {f.feature_type for f in features}`
inside the per-class loop).

**Description:** `available_types` is rebuilt from the full `features` list on every iteration of the
registry loop, even though `features` is loop-invariant. The `kept_names`/`names` pre-pass is also
computed by iterating the registry once, then the body re-iterates the registry with the same
`requires_prior`/`tier_filter` predicates — duplicated filtering.

**WHY:** Set comprehension placed inside the loop instead of hoisted; two registry passes that could
be one.

**Impact:** Negligible per image (registry < ~10 classes, features small) but it is dead recompute and
slightly obscures the control flow. **Fix direction:** hoist `available_types` above the loop; drop
the redundant `names`/`kept_names` pre-pass or reuse it as the loop driver.

---



---

# Part 7 — CLI Drivers & PyQt6 UI

# Full-depth critique: CLI drivers (`src/main/`) and PyQt6 UI (`src/nav/ui/`)

Files reviewed (28): src/main/__init__.py, src/main/nav_backplane_viewer.py, src/main/nav_backplanes.py, src/main/nav_backplanes_cloud_tasks.py, src/main/nav_consolidate_metadata.py, src/main/nav_create_bundle.py, src/main/nav_create_bundle_cloud_tasks.py, src/main/nav_create_simulated_image.py, src/main/nav_mosaic.py, src/main/nav_mosaic_cloud_tasks.py, src/main/nav_mosaic_display.py, src/main/nav_offset.py, src/main/nav_offset_cloud_tasks.py, src/nav/ui/__init__.py, src/nav/ui/common.py, src/nav/ui/library_entry.py, src/nav/ui/manual_nav_dialog.py, src/nav/ui/mosaic_viewer/__init__.py, src/nav/ui/mosaic_viewer/body_window.py, src/nav/ui/mosaic_viewer/common.py, src/nav/ui/mosaic_viewer/graticule.py, src/nav/ui/mosaic_viewer/histogram_stretch.py, src/nav/ui/mosaic_viewer/matplotlib_qt.py, src/nav/ui/mosaic_viewer/photometric_display.py, src/nav/ui/mosaic_viewer/projections.py, src/nav/ui/mosaic_viewer/ring_window.py, src/nav/ui/mosaic_viewer/sphere_render.py, src/nav/ui/mosaic_viewer/tiled_image_widget.py

Every file was read in full (large files in pages).

---

## Findings

### CODE-MAIN-001 — `nav_backplanes_cloud_tasks` / `nav_create_bundle_cloud_tasks` register an `async def main` as a console-script entry point (never runs)
> **Tracked by:** #108 — Check all CLI programs for proper logging, cloud operation, and that `cloud_tasks` works
Severity: **Critical**
Files/symbols:
- `src/main/nav_backplanes_cloud_tasks.py:101` `async def main()`
- `src/main/nav_create_bundle_cloud_tasks.py:109` `async def main()`
- `pyproject.toml:214` `nav_backplanes_cloud_tasks = "main.nav_backplanes_cloud_tasks:main"`
- `pyproject.toml:217` `nav_create_bundle_cloud_tasks = "main.nav_create_bundle_cloud_tasks:main"`

Description: Both modules declare `async def main()` and register it directly as the setuptools console-script target. A setuptools console script calls the target synchronously: `sys.exit(main())`. Calling an `async def` returns a coroutine object that is never awaited; the worker never starts, no event loop runs, and Python emits `RuntimeWarning: coroutine 'main' was never awaited` and exits 0. The sibling drivers do this correctly: `nav_offset_cloud_tasks.py:147` and `nav_mosaic_cloud_tasks.py:340` define a *sync* `main()` that calls `asyncio.run(async_main())`. Note also that running the module as `python nav_backplanes_cloud_tasks.py` works because the `__main__` guard uses `asyncio.run(main())` (line 132/146) — so the bug only manifests through the installed entry point, which is the production path.

WHY: An `async def` cannot be the body of a setuptools entry point because entry points are invoked as plain synchronous calls. The two working cloud_tasks drivers prove the intended pattern (sync `main()` wrapping `asyncio.run`).

Impact: The `nav_backplanes_cloud_tasks` and `nav_create_bundle_cloud_tasks` console commands are completely non-functional — they exit immediately without consuming any queue. Backplane and PDS4-bundle cloud batch processing cannot run via the published CLI.

---

### CODE-MAIN-002 — `nav_create_simulated_image` JSON load drops `shade_solid_rings` and can crash on `closest_planet=None`
Severity: **High**
File/symbol: `src/main/nav_create_simulated_image.py:2386-2477` `_load_parameters`

Description: Two defects in the JSON-load path:
1. `shade_solid_rings` is silently lost. The reconstructed `self.sim_params` dict (lines 2399-2419) omits the `shade_solid_rings` key entirely, and the `_shade_solid_rings_check` checkbox is never re-synced. After loading a file that had `shade_solid_rings: true`, the in-memory dict has no such key (so `_update_image`/render sees the default) and the checkbox still shows its previous state — UI and data diverge, and a subsequent save writes back without the key.
2. `closest_planet` can become `None` and crash the combo update. Line 2415 stores `params.get('closest_planet')` with no default, so when the key is absent the value is `None` (and the key now exists). Line 2430 then reads `self.sim_params.get('closest_planet', 'SATURN')`, but because the key exists with value `None`, the default does not apply and `closest_planet` is `None`. `QComboBox.findText(None)` / `setCurrentText(None)` then raise `TypeError`. The whole load is wrapped in a broad `except Exception` (line 2476) that pops an error dialog, so a partially-applied load leaves the UI in an inconsistent state with no clear cause.

WHY: The save path (`_save_parameters`, line 2372) writes the full `sim_params` including `shade_solid_rings`, so a round-trip is expected to preserve it. The general-tab init (line 279) correctly guards with `if closest_planet:`; the load path does not.

Impact: Save→load round-trip is lossy for ring shading, and any params file lacking `closest_planet` (or with `closest_planet: null`) throws during load, aborting the load mid-way.

---

### CODE-MAIN-003 — Broad `except Exception` swallows rendering and I/O errors in the simulated-image GUI
> **Tracked by:** #104 — replace broad except Exception control-flow in obs, nav_master, misc, and nav_mosaic
Severity: **Medium**
File/symbol: `src/main/nav_create_simulated_image.py:2145, 2369, 2383, 2476` (and `nav_backplane_viewer.py:817`)

Description: `_update_image` (2145), `_save_image` (2369), `_save_parameters` (2383), and `_load_parameters` (2476) each catch bare `Exception` and show a `QMessageBox`. While a top-level GUI handler is defensible, these are so broad they hide programming errors (e.g. the `TypeError` from CODE-MAIN-002) behind a generic "Failed to ..." dialog, defeating diagnosis. `nav_backplane_viewer.py:814-818` has a bare `except Exception: pass` around building a default filename that can mask real bugs.

WHY: Project conventions forbid swallowing errors without logging; none of these log through `pdslogger` or even `print` the traceback. The mosaic-viewer windows by contrast use `logger.exception(...)` on load failure.

Impact: Failures are reduced to a generic dialog with no logged traceback, making field debugging hard. Lower severity because it is GUI-local, but it directly hampers diagnosing CODE-MAIN-002.

---

### CODE-MAIN-004 — `nav_mosaic.py` flips the stdlib root logger level, contradicting the pdslogger convention
> **Tracked by:** #108 — Check all CLI programs for proper logging, cloud operation, and that `cloud_tasks` works
Severity: **Medium**
File/symbol: `src/main/nav_mosaic.py:28, 553-558`

Description: `nav_mosaic.py` imports the stdlib `logging` module and, when `--log-level` is given, does `logging.getLogger().setLevel(numeric)` plus `MAIN_LOGGER.setLevel(numeric)`. The rest of the pipeline routes logging through `pdslogger` (`MAIN_LOGGER` / `IMAGE_LOGGER`), which has its own handlers and is not the stdlib root logger. Setting the *root* stdlib logger level has no effect on pdslogger output and may surprise by changing third-party library verbosity instead. `nav_offset.py` and `nav_consolidate_metadata.py` correctly drive verbosity through `setup_logging(...)` + the `--log-level-*` flags and never touch stdlib `logging`. `main/*.py` is outside the CLAUDE.md stdlib-logging ban (which lists only core packages), so this is a consistency/correctness issue, not a rule violation.

WHY: Two different mechanisms (`--log-level` via stdlib root logger here vs. `--log-level-main-*` via `setup_logging` elsewhere) for the same goal; the stdlib-root path does not influence the pdslogger handlers it intends to.

Impact: `--log-level` on `nav_mosaic` does not reliably change the program's own log verbosity; behavior diverges from the other drivers. Confirm by running `nav_mosaic rings <ds> --log-level DEBUG` and checking that MAIN_LOGGER DEBUG lines actually appear (they will, only because of the explicit `MAIN_LOGGER.setLevel`; the `logging.getLogger()` call is dead/misleading).

---

### CODE-MAIN-005 — `nav_create_bundle summary` and `nav_backplane_viewer` reimplement config loading instead of `load_default_and_user_config`
Severity: **Low**
Files/symbols:
- `src/main/nav_create_bundle.py:208-216` (`main_summary`)
- `src/main/nav_backplane_viewer.py:1493-1501` (`main`)

Description: Every other driver calls `load_default_and_user_config(arguments, DEFAULT_CONFIG)`. These two open-code the same logic (`DEFAULT_CONFIG.read_config()`; loop over `--config-file`; else try `nav_default_config.yaml` swallowing `FileNotFoundError`). The duplication risks drift if the canonical loader changes (e.g. adds env-var precedence or validation), and `main_summary` notably does NOT set `pdstemplate` logger / env precedence the same way `main_labels` does for the shared helper.

WHY: Single source of truth for config/CLI/env precedence is the stated architecture (`load_default_and_user_config`). Hand-rolled copies bypass it.

Impact: Subtle precedence divergence between subcommands/drivers; maintenance hazard. Low because current behavior matches by coincidence.

---

### CODE-MAIN-006 — High structural duplication across the six dataset-driven drivers and the cloud_tasks file-list parsing
Severity: **Low**
Files/symbols: `nav_offset.py:54-218`, `nav_backplanes.py:38-132`, `nav_consolidate_metadata.py:49-185`, `nav_create_bundle.py:86-126`, `nav_backplane_viewer.py:1431-1485` (parse_args + unknown-dataset block); `nav_offset_cloud_tasks.py:87-107`, `nav_backplanes_cloud_tasks.py:70-88`, `nav_create_bundle_cloud_tasks.py:77-95`, `nav_mosaic_cloud_tasks.py:199-226` (per-file `ImageFile` construction + missing-field error returns)

Description: The "validate dataset name → print usage → instantiate DataSet → add `--config-file`/`--pds3-holdings-root`/`--nav-results-root` arg group" preamble is copy-pasted across five drivers with near-identical text and three identical `sys.exit(1)` branches. Separately, each cloud_tasks `process_task` re-implements the same loop that pulls `image_file_url`/`label_file_url`/`results_path_stub`/`index_file_row` from each file dict and returns `{'status':'error','status_error':'no_image_file_url'}` etc. `nav_offset_cloud_tasks` additionally reads `extra_params` (line 93) while `nav_backplanes_cloud_tasks` and `nav_create_bundle_cloud_tasks` do not — so an `extra_params` carried in a task is silently dropped for backplanes/bundle, a latent inconsistency.

WHY: Same logic, four+ copies, already drifting (extra_params handled in one, not the others).

Impact: Maintenance burden and inconsistency risk (the extra_params drop is a real behavioral gap if any dataset needs extra_params during backplane/bundle generation). Low severity since the common datasets do not currently rely on extra_params downstream.

---

### CODE-MAIN-007 — `nav_offset --output-cloud-tasks-file` iterates the dataset but assumes a single file for the task_id while the body handles multi-file batches
Severity: **Low**
File/symbol: `src/main/nav_offset.py:393-427`

Description: When building cloud-task JSON, `task_id` is derived from `imagefiles.image_files[0].label_file_name` (line 402) and the loop collects all `image_files` into `task_files`. But the main processing loop (line 430) asserts `len(imagefiles.image_files) == 1`. So the cloud-task writer is written to tolerate multi-file batches that the local processing path forbids. If a dataset ever yields multi-file groups, the local run asserts (crash) while the cloud-task export silently produces a task whose `task_id` is named only after the first file. The two code paths disagree on the batch-size contract.

WHY: Inconsistent assumptions about `ImageFiles` cardinality between the cloud-export and local-run paths of the same driver.

Impact: Either the assert is dead (datasets are always single-file, in which case the multi-file export code is dead) or a real divergence exists. Confirm by checking whether any registered `DataSet.yield_image_files_from_arguments` can yield groups with `len(image_files) > 1` for nav_offset.

---

### CODE-UI-001 — `nav_backplane_viewer._compose_and_display` duplicates ~70 lines of alpha-blend that `_alpha_blend_layer` already encapsulates
Severity: **Medium**
File/symbol: `src/main/nav_backplane_viewer.py:1207-1274` vs helper `_alpha_blend_layer` (164-183) and `_render_full_rgba` (827-912)

Description: `_render_full_rgba` correctly composites the summary overlay and BODY_ID overlay by calling the `_alpha_blend_layer` helper. `_compose_and_display` (the on-screen path) instead inlines the identical premultiplied-alpha math twice (summary at 1207-1225, BODY_ID at 1227-1274), including a duplicated colormap-resolution fallback (1247-1257) that `_load_colormap` already provides. The two paths can drift (the on-screen BODY_ID path uses a `cm.get_cmap` fallback the helper-based path lacks). This is also a correctness risk: any fix to the blend (e.g. NaN handling) must be made in three places.

WHY: One blend routine exists (`_alpha_blend_layer`) and is used by the save path; the live-display path re-implements it inline, defeating the helper.

Impact: Maintenance hazard and latent on-screen/saved-PNG divergence. Medium because the viewer already has a save path that does it correctly, so the inline copies are pure liability.

---

### CODE-UI-002 — `nav_backplane_viewer` cursor read-out assumes no pan offset, so values are wrong after panning at zoom != 1
Severity: **Medium**
File/symbol: `src/main/nav_backplane_viewer.py:1349-1367` `_update_cursor_status`

Description: `_update_cursor_status` maps the mouse position to image coords with `u = pos.x() / zoom`, `v = pos.y() / zoom` and a comment admitting "Since we didn't actually change widget offset, map pos to image by dividing by zoom". The label is inside a `QScrollArea`; when zoomed in and scrolled, `pos` is relative to the (large) label, which the comment claims has no offset — but the displayed pixmap is scaled and the scroll position shifts what is visible. The reported V,U and the sampled `self._img_float[v,u]` will not correspond to the pixel actually under the cursor once the user pans. Compare `manual_nav_dialog._update_status_from_mouse` (line 1121) which uses the same label-relative assumption but there the label resizes to the scaled pixmap so label coords == scaled image coords; the backplane viewer's `_compose_and_display` (1339-1343) also resizes the label to the pixmap, so the divide-by-zoom is approximately right ONLY because label coords are pixmap coords — but the status code ignores the `ZoomPanController` scroll model entirely, and the inline comment signals the author was unsure.

WHY: The cursor sampling path does not go through the same coordinate transform the rendering path uses; it relies on an implicit "label == scaled image" invariant that is fragile and undocumented.

Impact: Possible incorrect V,U / value / BODY_ID readouts after pan/zoom. Mark uncertain: confirm by zooming the backplane viewer to e.g. 4x, panning, and checking the reported pixel value against a known feature. If wrong, route through `self._label` mapping + scroll offsets like `ZoomPanController`.

---

### CODE-UI-003 — `_zoom_at_point` in three widgets computes `new_zoom` and early-returns, then re-derives it in the controller (dead computation / drift risk)
Severity: **Low**
Files/symbols:
- `src/main/nav_create_simulated_image.py:555-570` `_zoom_at_point`
- `src/main/nav_backplane_viewer.py:1172-1187` `_zoom_at_point`
- `src/nav/ui/manual_nav_dialog.py:1006-1016` `_zoom_at_point`

Description: Each of these methods computes `old_zoom`, `new_zoom = clip(old*factor, 0.1, 50.0)`, early-returns if unchanged, then delegates to `self._zoom_ctl.zoom_at_point(factor, ...)` which independently re-clips with the SAME constants (`common.py:156`). The local clamp constants `(0.1, 50.0)` are duplicated from `ZoomPanController._zoom_at_point` and can silently diverge from the controller's real limits. The local `new_zoom` is otherwise unused (only the early-return depends on it). In `nav_backplane_viewer` and `manual_nav_dialog` the `_zoom_in/_zoom_out` wrappers also duplicate the controller's `zoom_in_center`/`zoom_out_center` logic that already exists in `common.py:101-127`.

WHY: The controller is the single source of zoom-clamp truth; these wrappers re-encode the limits and the centering math.

Impact: If the controller's clamp range changes, these wrappers will mis-predict the no-op case and the duplicated `_zoom_in/_zoom_out` will drift. Low — currently consistent.

---

### CODE-UI-004 — `nav_create_simulated_image` `_zoom_in/_zoom_out` ignore the scroll-offset convention used elsewhere
Severity: **Low**
File/symbol: `src/main/nav_create_simulated_image.py:525-553`

Description: `_zoom_in`/`_zoom_out` compute `scaled_x = center_x + scrollbar_h.value()` and pass `(center_x, center_y)` as the viewport anchor. This is the same pattern as `nav_backplane_viewer._zoom_in` (1135). It works, but it is a fourth copy of the "viewport-centre anchored zoom" that `ZoomPanController.zoom_in_center`/`zoom_out_center` (common.py:101-127) already implements identically. Pure duplication; flagged per the high-duplication directive.

WHY: Same centre-anchored zoom logic exists in the shared controller and is re-implemented per window.

Impact: Maintenance only.

---

### CODE-UI-005 — `ring_window` redefines an inner `_ZoomSync` class while `body_window` hoists it to module scope
Severity: **Low**
Files/symbols:
- `src/nav/ui/mosaic_viewer/ring_window.py:837-842` (inner class inside `_make_zoom_sync`)
- `src/nav/ui/mosaic_viewer/body_window.py:219-234` (module-level `_ZoomSync`)

Description: `ring_window._make_zoom_sync` defines a fresh `_ZoomSync(_SyncedSlider)` subclass on every call (a new class object per zoom row), whereas `body_window` defines the identical class once at module scope. Both `_SyncedSlider` classes (ring_window:200, body_window:133) are themselves byte-for-byte duplicates across the two windows. This is exactly the cross-driver duplication the directive calls out.

WHY: Two copies of `_SyncedSlider` and two different definition strategies for `_ZoomSync` for identical behavior.

Impact: Defining a class inside a method is wasteful (a new type per call) and the duplicated `_SyncedSlider` doubles the maintenance surface. Low. Should live in `nav/ui/common.py` once.

---

### CODE-UI-006 — `mosaic_viewer` modules use stdlib `logging.getLogger(__name__)` instead of pdslogger
Severity: **Low**
Files/symbols: `src/nav/ui/mosaic_viewer/common.py:9,25`; `body_window.py:3,50`; `ring_window.py:8,57,425`

Description: These modules import the stdlib `logging` module and create module loggers. CLAUDE.md restricts the stdlib-logging ban to `nav.feature`, `nav.nav_model`, `nav.nav_orchestrator`, `nav.nav_technique`, `nav.support` — `nav.ui.*` is not on that list, so this is technically permitted. However, it is the only place in `nav/` that uses stdlib logging, diverging from the project-wide `pdslogger` convention, and these logger calls (`logger.exception`, `logger.debug`) will not appear in the per-image / main pdslogger streams the rest of the system writes.

WHY: Project-wide convention is pdslogger; these are the lone holdouts and their output is invisible to the normal log routing.

Impact: Load-failure tracebacks and EW/radial draw warnings land in the stdlib root handler (often nowhere) rather than the nav log files. Low; confirm by triggering a `load_ring_file` failure and checking whether the traceback reaches the configured nav log.

---

### CODE-UI-007 — `__all__ = []` not annotated; `tiled_image_widget` reaches into private `_slider_to_zoom`/`_zoom_to_slider`
Severity: **Low**
Files/symbols: `src/nav/ui/__init__.py:20`, `src/nav/ui/mosaic_viewer/__init__.py:11`; `body_window.py:44-48`, `ring_window.py:51-55`

Description: Minor: `__all__ = []` should be `__all__: list[str] = []` for mypy-strict cleanliness. More notably, `body_window` and `ring_window` import the underscore-prefixed `_slider_to_zoom` / `_zoom_to_slider` from `tiled_image_widget`. Importing names marked private across modules is a smell; these zoom-mapping helpers are clearly part of a shared contract and should be public (or moved to `common.py`).

WHY: Private (`_`-prefixed) symbols imported across module boundaries indicate a missing public API.

Impact: Cosmetic / API-hygiene. Low.

---

### CODE-UI-008 — `manual_nav_dialog.run_modal` calls `app.quit()` on a freshly created QApplication but never deletes it, and other dialogs leave the same dangling-app pattern
Severity: **Low**
Files/symbols: `src/nav/ui/manual_nav_dialog.py:1145-1159`; compare `nav_backplane_viewer.py:1519-1538`

Description: `run_modal` creates a `QApplication([])` if none exists, runs `self.exec()`, then on the created path calls `app.quit()`. `quit()` only requests the event loop to exit (the loop is already done after `exec()` returns), it does not release the `QApplication` singleton; a subsequent `run_manual_nav` in the same process will see the stale instance via `QApplication.instance()`. The same is true of `nav_backplane_viewer.main` (creates app, `app.quit()` after `exec()`). For the `--manual` single-shot CLI path this is benign, but if `run_manual_nav` is ever invoked twice in one process (e.g. tests, or a future batch manual mode) the leftover app/state can cause subtle issues.

WHY: `QApplication.quit()` after `exec()` returns is a no-op for cleanup; the singleton persists.

Impact: No effect on the current single-image `--manual` flow; latent issue for repeated invocation. Low. Confirm by calling `run_manual_nav` twice in one pytest process.

---

### CODE-UI-009 — Histogram-stretch `mousePressEvent` tie-break can pick the white indicator when black is exactly as close
Severity: **Low**
File/symbol: `src/nav/ui/mosaic_viewer/histogram_stretch.py:260-265`

Description: When black and white indicators coincide (or are within `_PICK_THRESHOLD_PX` and equidistant), the pick logic `if d_black <= d_white ...: black elif d_white < d_black ...: white` correctly prefers black on a tie. But when black and white sit on the *same pixel* (possible after `set_values` with `white == black`, which `set_data`/`set_range` permit before the +1e-6 nudge in `set_data` only), the user can never grab the white indicator to separate them — both map to the same `x` and black always wins. Minor edge case.

WHY: Coincident indicators are reachable (e.g. degenerate image where percentile black==white before the epsilon bump) and the tie-break makes white unreachable.

Impact: Rare UI dead-end (white indicator unselectable) for flat images. Low.

---



---

# Part 8 — Model/Technique Non-Math Coverage Gaps

# nav_model / nav_technique coverage-gap critique

Files reviewed (41): nav_model/__init__.py, body_shape.py, nav_model.py,
nav_model_body.py, nav_model_body_base.py, nav_model_body_simulated.py,
nav_model_rings.py, nav_model_rings_base.py, nav_model_rings_simulated.py,
nav_model_titan.py, rings/__init__.py, rings/ring_feature.py,
rings/ring_filter.py, rings/ring_math.py, rings/ring_render_context.py,
rings/ring_render_result.py, rings/ring_types.py, stars/__init__.py,
stars/catalog.py, stars/conflicts.py, stars/detection.py,
stars/nav_model_stars.py, stars/predicted_snr.py, stars/smeared_psf.py,
nav_technique/__init__.py, _star_helpers.py, confidence.py,
confidence_config.py, diagnostics.py, dt_fitting.py, feasibility.py,
nav_technique.py, nav_technique_body_blob.py, nav_technique_body_disc.py,
nav_technique_body_limb.py, nav_technique_body_terminator.py,
nav_technique_manual.py, nav_technique_ring_annulus.py,
nav_technique_ring_edge.py, nav_technique_star_field.py,
nav_technique_star_refine.py, nav_technique_star_unique_match.py,
technique_result.py.

(The reviewer also read predicted_snr.py and smeared_psf.py headers /
ring_math.py signatures for non-math issues; no new non-math defects were
found in those three, so they are folded into the count above implicitly
via their package __init__ files.)

Scope note: the prior math review's IDs are CODE-NAV-*; everything below
uses CODE-MODEL-* (nav_model) or CODE-TECH-* (nav_technique) to avoid
collisions. Math already covered by the prior review is not re-derived.

---

## Findings

### CODE-MODEL-001 — `NavModelTitan` is registered but permanently inert (dead registration)
> **Tracked by:** #60 — Implement Titan navigation
Severity: Medium
File: `src/nav/nav_model/nav_model_titan.py` (whole class);
interacts with `src/nav/nav_model/nav_model.py::build_models_for_obs`.

`NavModelTitan` is a concrete `NavModel` subclass that does **not** set
`_abstract = True` in its body, so `NavModel.__init_subclass__` appends it
to `NavModel._registry`. But it does **not** override `instances_for_obs`,
so it inherits the base default that returns `[]`. The net effect:
`build_models_for_obs` iterates the registry, calls
`NavModelTitan.instances_for_obs(obs)`, gets `[]` every time, and the
class is never instantiated. It is registry overhead with zero behavior.

WHY: The class docstring says "This module exists as a registered
placeholder", but registration buys nothing here — a registered subclass
whose `instances_for_obs` is the no-op default is indistinguishable at
runtime from an unregistered one, except it shows up in
`NavModel._registry` and any test/diagnostic that enumerates registered
models (misleading inventory). If the intent was "construct a Titan model
when Titan is in the FOV and have it emit nothing", that path does not
exist; if the intent was "do nothing", the registration is dead code.

Impact: Misleading model inventory; a future reader may assume Titan
images get a (no-op) Titan model when in fact `NavModelBody` handles
Titan as an ordinary body (TITAN is in `BODY_SHAPE_TABLE`). Confirm by
running `build_models_for_obs` on a Titan-in-FOV obs and checking no
`NavModelTitan` instance appears. Either mark `_abstract = True` (drop it
from the registry until the haze-aware algorithm lands) or give it a real
`instances_for_obs`.

---

### CODE-MODEL-002 — Simulated rings `constituent_edge_count` is computed with a confusing `1 + a + b - 1` expression that double-counts intent
Severity: Low
File: `src/nav/nav_model/nav_model_rings_simulated.py`, `to_features`
lines 174-180.

```python
flags=RingAnnulusFlags(
    planet_name=self._ring_name,
    constituent_edge_count=1
    + int(self._ring_feature.outer_edge is not None)
    + int(self._ring_feature.inner_edge is not None)
    - 1,
),
```

`1 + a + b - 1` is algebraically `a + b` (count of present edges), so the
leading `+ 1 ... - 1` is dead arithmetic. More importantly the catalog
`NavModelRings` path stores `constituent_edge_count` = number of fused
*rings* (annulus constituents), whereas here it stores the number of
*edges* of one ring. The two producers disagree on what the field means.

WHY: A downstream consumer (`RingAnnulusFlags.constituent_edge_count`,
read by the ring-annulus confidence path and curator) cannot tell whether
the number is "edges of a single ringlet" (1 or 2) or "rings fused into a
composite" (catalog path). The simulated value will almost always be 2,
which is a plausible-but-wrong constituent count for confidence scaling.

Impact: Inconsistent semantics for one diagnostic/flag field across the
real vs simulated ring models; potential mis-weighting if the simulated
annulus ever flows through the same confidence formula. Replace the
expression with `int(inner is not None) + int(outer is not None)` and add
a one-line comment reconciling the meaning with the catalog path (or
rename so the two are not confused).

---

### CODE-MODEL-003 — `shape_for_body` is an explicit "backward-compatible alias", violating the no-backwards-compat-shims convention
Severity: Low
File: `src/nav/nav_model/body_shape.py`, `shape_for_body` lines 210-225;
caller `src/nav/nav_model/nav_model_body.py` imports `shape_for_body`
(line 72) and uses it at line 608.

The docstring states: "Backward-compatible alias for
:func:`load_body_shape`. Earlier callers used `shape_for_body(name)` ...".
Project convention (`CLAUDE.md`): "No backwards-compat shims unless
explicitly requested." `shape_for_body` is exported in `__all__` and is
the *only* name `nav_model_body.py` actually calls, while `load_body_shape`
is the "real" function — so there are two public names for one behavior.

WHY: The convention exists to avoid exactly this kind of name duplication.
Keeping both forces every reader to learn that they are identical and
keeps an alias alive whose stated justification ("earlier callers") is a
migration artifact, not a current requirement.

Impact: API-surface clutter and a documented convention violation.
Collapse to a single name: either rename `load_body_shape` ->
`shape_for_body` (the name actually used) and delete the alias, or switch
`nav_model_body.py` to `load_body_shape` and delete `shape_for_body`.
Verify by grepping `shape_for_body` across `src/` and `tests/` and
updating the single call site + tests.

---

### CODE-MODEL-004 — `_yaml_entry_for` swallows all exceptions with a bare `except Exception`
Severity: Low
File: `src/nav/nav_model/body_shape.py`, `_yaml_entry_for` lines 245-248.

```python
try:
    body_shape_section = cfg.body_shape
except Exception:
    return None
```

The catch is intended to handle "config.body_shape not yet loaded during
early bootstrapping", but it catches *everything* — an `AttributeError`
from a typo in `Config`, a `KeyError` from a malformed section, a
genuine programming error in the property — all become a silent
"no YAML overrides", which then silently falls back to the hard-coded
`BODY_SHAPE_TABLE`. There is no log line.

WHY: A misconfigured or broken `Config.body_shape` should surface, not be
masked into "use the hardcoded fallback". The same module elsewhere
(ring config) deliberately raises on bad config; this path deliberately
hides it. Per the project's "fail fast at process startup" posture for
config (see `nav_model_rings._require_positive_finite_planet_scalar`), a
broad swallow here is inconsistent.

Impact: A real Config bug (wrong attribute, bad YAML) silently degrades
every body's shape to the table fallback with no diagnostic. Narrow the
except to the specific exception the bootstrap race actually raises
(likely `AttributeError`), or at minimum log at DEBUG before returning
`None`. Confirm by reading what `Config.body_shape` raises before the
loader runs.

---

### CODE-MODEL-005 — `_merge_catalogs` upgrades `pretty_name` to the later catalog's `pretty_name`, but the docstring/guard says `name`
Severity: Low
File: `src/nav/nav_model/stars/catalog.py`, `_merge_catalogs` lines
559-560.

```python
if (not prev.name) and star.name:
    prev.pretty_name = star.pretty_name
```

The guard checks `star.name` (the raw catalog name string) but assigns
`star.pretty_name`. `pretty_name` is derived earlier in
`_find_stars_in_one_catalog` as `str(star.unique_number)` and only set to
the cleaned `name` when `star.name.strip()` is truthy (lines 270-275). In
the normal case where `star.name` is non-empty these coincide, but if a
later-catalog star has `name` set yet `pretty_name` was left as the
unique-number string (e.g. a whitespace-only name that failed the
`.strip()` test on the producing path differently), the earlier star's
`pretty_name` is "upgraded" to a numeric id — a downgrade, not an upgrade.

WHY: The guard predicate and the assignment source are different fields,
so the "nicer name wins" intent is not actually enforced — it depends on
`pretty_name` already mirroring `name`, which is a separate code path's
invariant.

Impact: In rare records the dedup name-upgrade produces a worse label.
Low because the two fields almost always agree. Fix by asserting the
intent directly: `prev.pretty_name = star.pretty_name` only when
`star.pretty_name` is non-numeric/nicer, or guard on
`star.pretty_name` rather than `star.name`. Confirm against a catalog
record whose `name` is set but `pretty_name` fell back to
`unique_number`.

---

### CODE-MODEL-006 — `NavModelStars._emit_features` calls `star.conflicts.startswith(...)` without the `or ''` guard the rest of the file uses
Severity: Low
File: `src/nav/nav_model/stars/nav_model_stars.py`, lines 237-238.

```python
in_body = bool(star.conflicts.startswith('BODY'))
in_ring = bool(star.conflicts.startswith('RING'))
```

`create_model` (lines 132-133) defensively writes
`(s.conflicts or '').startswith(...)`, and `_star_short_info`/`_star_summary`
tolerate `None`. Here the access is unguarded. The catalog reduction sets
`star.conflicts = ''` (catalog.py line 276), so in the normal pipeline
`conflicts` is never `None` at this point — but the inconsistency means a
star record that reaches `_emit_features` with `conflicts=None` (a
simulated/stub star, or a future code path that skips the catalog
normalization) raises `AttributeError` mid-emit.

WHY: The codebase already established the `(x or '')` guard as the safe
idiom for this exact field two methods up; this site diverges. The
type contract on `MutableStar.conflicts` is not enforced to be non-None.

Impact: Latent `AttributeError` on a non-catalog star path; brittle.
Apply the same `(star.conflicts or '')` guard for symmetry, or assert /
type-narrow `conflicts` to `str` once after reduction.

---

### CODE-MODEL-007 — `mark_body_and_ring_conflicts` ring check uses `bp_radii.median()` over a partially-masked window, biasing the occlusion test
Severity: Medium
File: `src/nav/nav_model/stars/conflicts.py`, `_check_one_star`
lines 167-177.

The star's small conflict meshgrid spans `±body_conflict_margin` pixels.
For the ring test it computes `radius_km = float(bp_radii.median().vals)`
and checks whether *that single median radius* falls inside any opaque
annulus. If only part of the window intersects an opaque annulus (the
star sits on the inner/outer edge of a ringlet, or the window straddles a
gap boundary), the median radius can land in a gap while the star pixel
itself is on the ring, or vice-versa. The body check, by contrast,
correctly uses `where_intercepted(...).any()` — "any pixel in the window
hits the body".

WHY: The two occlusion checks use inconsistent reductions. `any()` is the
right semantics for "is the predicted pixel occluded"; `median()` over a
multi-pixel window is a smoothing reduction that can both miss and
fabricate edge-of-annulus occlusions. The margin exists to tolerate SPICE
pointing error, which makes the window deliberately larger than one
pixel, amplifying the discrepancy.

Impact: Edge-of-ring stars may be incorrectly tagged (or not tagged)
`RING:` occluded, which propagates into the STAR feature's
`in_body_silhouette` reliability hard-zero. Low-to-medium prevalence but a
real correctness asymmetry. Confirm by constructing a window straddling an
annulus boundary; consider testing "any window pixel inside any annulus"
to match the body semantics, or sampling the radius at the star's own
pixel rather than the window median.

---

### CODE-MODEL-008 — Ring-edge / ring-annulus reliability use uncalibrated PLACEHOLDER coefficients with no config hook
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
Severity: Medium
File: `src/nav/nav_model/nav_model_rings.py`, `_ring_edge_reliability`
lines 907-925 and `_ring_annulus_reliability` lines 928-949;
constants `RING_EDGE_DEFAULT_RELIABILITY` (0.7),
`RING_EDGE_SIGMA_ALONG_PX` (0.5).

`_ring_edge_reliability` openly approximates "a sigmoid of the (yet-
uncalibrated) emission-angle factor ... by a constant in this
implementation pending Phase-5 calibration", and applies a hardcoded
`* 0.7` straight-line multiplier. `_ring_annulus_reliability` substitutes
`min(1, constituent_count/5) * RING_EDGE_DEFAULT_RELIABILITY` and a
hardcoded `radial_extent_px / 50 - 1` sigmoid because "per-edge
constituent reliabilities are not tracked yet". The `50.0` and `0.7` and
`5.0` are all module-level magic numbers, not config keys — unlike the
ring-annulus *emission* gate which is fully config-driven
(`_ring_annulus_emission_params`).

WHY: The design's reliability formulas are explicitly placeholders, but
the magic constants live inline rather than in
`config_510_techniques.yaml` (or a rings reliability block), so they
cannot be tuned per-planet the way the rest of the rings pipeline is.
This is a config-vs-hardcoded inconsistency within the same file.

Impact: Ring reliability cannot be calibrated without code edits; the
0.7/5.0/50.0 values silently set the gate that decides whether ring
features survive the reliability threshold (0.30). Surface these as config
keys (mirroring `feature_emission.ring_annulus`) and mark each clearly as
PLACEHOLDER until Phase-5 calibration. Confirm by grepping config for
`ring_edge` reliability keys (none exist today).

---

### CODE-MODEL-009 — `NavModelRings.instances_for_obs` / `NavModelBody.instances_for_obs` read `DEFAULT_CONFIG`, ignoring any per-run config override
Severity: Low
File: `src/nav/nav_model/nav_model_rings.py` lines 230-234;
`src/nav/nav_model/nav_model_body.py` lines 229-233.

Both classmethods hardcode `DEFAULT_CONFIG` to decide which bodies / ring
systems to instantiate (`rings_config = DEFAULT_CONFIG.rings`,
`config = DEFAULT_CONFIG`). The constructed instances later accept a
per-instance `config`, but the *selection* of which instances to build is
always against the global default. `build_models_for_obs(obs)` takes no
config argument, so there is no way to thread a non-default config into
the registry walk.

WHY: A caller that builds a `Config` override (custom satellite list,
custom `rings.ring_features`) and expects it to govern model construction
gets the default selection instead. The per-instance `config` parameter on
the constructors is therefore half-wired: it affects rendering but not
which models exist.

Impact: Config overrides that change the body/ring inventory are silently
ignored at the model-construction stage. Low because production uses the
loaded default singleton, but it is a latent surprise for tests and tools.
Consider threading `config` through `build_models_for_obs` and
`instances_for_obs`, or document that instance selection is always
default-config-driven.

---

### CODE-MODEL-010 — `nav_model_body.py` exceeds the 1000-line module convention (1118 lines)
> **Tracked by:** #97 — split oversized modules exceeding 1000-line rulebook limit
Severity: Low
File: `src/nav/nav_model/nav_model_body.py` (1118 lines).

CLAUDE.md: "Modules: keep under 1000 lines; split into a package if
larger." `nav_model_body.py` is 1118 lines. The body model already has a
companion base (`nav_model_body_base.py`) and a shape module
(`body_shape.py`), so the silhouette extraction / polyline building /
per-feature emission helpers are natural split candidates (e.g. a
`body/` subpackage paralleling `rings/` and `stars/`).

WHY: A documented hard convention; the rings and stars models were already
split into subpackages, so the body model is the outlier.

Impact: Maintainability only; no runtime effect. Confirm with `wc -l`.
Split the free helpers (limb/terminator polyline extraction, emission
builders `_build_*_feature`) into a sibling module.

---

### CODE-TECH-001 — `BodyDiscCorrelateNav._upsample_factor` has no validation, unlike the near-identical `RingAnnulusNav._upsample_factor`
> **Tracked by:** #118 — Design and implement a comprehensive config validation system
Severity: Medium
File: `src/nav/nav_technique/nav_technique_body_disc.py`
`_upsample_factor` lines 753-758; compare
`src/nav/nav_technique/nav_technique_ring_annulus.py`
`_upsample_factor` lines 280-305.

`RingAnnulusNav._upsample_factor` validates the config value: rejects
non-real / bool, coerces to `int >= 1`, and caps at
`_MAX_UPSAMPLE_FACTOR` (1e6) "so a malformed config cannot push the FFT
into a multi-gigabyte allocation". `BodyDiscCorrelateNav._upsample_factor`
is the duplicate-with-the-guard-removed:

```python
return int(getattr(offset_block, 'correlation_fft_upsample_factor', 128))
```

No bounds, no type guard. The two techniques read the *same* config key
(`config.offset.correlation_fft_upsample_factor`) with different safety.

WHY: The exact failure mode the ring-annulus version documents (a
misconfigured huge upsample factor hangs the process with a multi-GB FFT
allocation) is unguarded in the disc technique, which runs the same
`navigate_with_pyramid_kpeaks` FFT. A `correlation_fft_upsample_factor`
of, say, `1e9` or a string would either OOM or raise an opaque
`int(...)`/numpy error mid-navigate instead of a clean config-time
`ValueError`.

Impact: Inconsistent robustness for one shared config key; a config typo
that the ring path catches cleanly will hang/crash the body-disc path.
De-duplicate: extract one shared validated `_upsample_factor` helper
(e.g. in `nav_technique.py` or a small config helper) and call it from
both techniques. Verify the body-disc path now raises the same
`ValueError` on out-of-range input.

---

### CODE-TECH-002 — `ManualNavDiagnostics` is missing from `nav_technique/__init__.py` exports
Severity: Low
File: `src/nav/nav_technique/__init__.py` (import block lines 27-38 and
`__all__` lines 53-82); the class is defined and exported in
`src/nav/nav_technique/diagnostics.py::__all__`.

`diagnostics.py` lists `ManualNavDiagnostics` in its `__all__`, and
`nav_technique_manual.py` imports it directly from
`nav.nav_technique.diagnostics`. But the package `__init__.py` re-exports
every *other* diagnostics dataclass (`BodyBlobDiagnostics`,
`BodyDiscDiagnostics`, ... `StarUniqueMatchDiagnostics`,
`NavTechniqueDiagnostics`) and omits `ManualNavDiagnostics`.

WHY: Asymmetric public surface — `from nav.nav_technique import
StarFieldDiagnostics` works but `from nav.nav_technique import
ManualNavDiagnostics` raises `ImportError`, even though both are
first-class per-technique diagnostics and `ManualNavDiagnostics` is part
of the `NavTechniqueDiagnostics` union.

Impact: A consumer enumerating diagnostics types via the package root
silently misses the manual one (e.g. the curator's per-technique field
walk, or a test asserting the union is fully re-exported). Add
`ManualNavDiagnostics` to the `__init__.py` import + `__all__`.

---

### CODE-TECH-003 — `NavTechniqueManual.__init__` documents an "optional for backwards compatibility" parameter, violating the no-backwards-compat convention
Severity: Low
File: `src/nav/nav_technique/nav_technique_manual.py`, `__init__`
lines 86-89.

```python
# ``annotations`` is the merged-per-NavModel ``Annotations`` the
# dialog uses ...  Optional for backwards compatibility / tests
# that only need the offset-pick path; ``run_manual_nav``
# always populates it.
```

CLAUDE.md forbids backwards-compat shims unless explicitly requested.
"Optional for backwards compatibility" is exactly the anchoring-to-
migration-history phrasing the project conventions (and the user's
documented doc-phrasing preference) prohibit.

WHY: Convention + documented user preference: describe current state, not
migration history. The parameter being optional is fine; justifying it as
"backwards compatibility" is the violation.

Impact: Documentation-only, but it is a direct convention/phrasing
violation. Reword to describe the current contract (e.g. "Optional; only
the offset-pick path needs it. `run_manual_nav` always supplies it.")
without the "backwards compatibility" framing.

---

### CODE-TECH-004 — `search_window_for_obs` reads `obs.extfov_margin_vu` via a module-level `# type: ignore[attr-defined]`
> **Tracked by:** #105 — replace pervasive Any / dict[str, Any] with TypedDicts and Protocols at interop boundaries
Severity: Low
File: `src/nav/nav_technique/nav_technique.py`, `search_window_for_obs`
line 128.

```python
margin = context.obs.extfov_margin_vu  # type: ignore[attr-defined]
```

The `# type: ignore[attr-defined]` papers over the fact that the `obs`
attribute type visible to mypy (`ObsSnapshot`) does not declare
`extfov_margin_vu`. The docstring even relies on the runtime
`AttributeError` as a feature ("a test obs stand-in that omits the
attribute surfaces an `AttributeError`"). Every DT/correlation technique
funnels through this helper.

WHY: A type-ignore on a load-bearing attribute access is a type-safety
gap under `strict = true`. If the real obs type *does* expose
`extfov_margin_vu` (it appears to, since the whole pipeline depends on
it), the ignore hides a missing protocol/attribute declaration rather
than fixing it; if it does not, the access is genuinely unsafe.

Impact: mypy-strict suppression that hides a missing type declaration on a
universally-used obs attribute. Confirm whether `ObsSnapshot` (or the
`NavContext.obs` annotation) declares `extfov_margin_vu`; if it should,
add it to the type and drop the ignore. The CLAUDE.md rule is "No
module-level `# type: ignore` without a specific error code" — this one
has a code, so it is borderline, but the better fix is the declaration.

---

### CODE-TECH-005 — Four-way duplicated `_build_polyline_mask` and two-way duplicated `_peak_to_runner_up_ratio` / `_TukeyConfidenceContext` adapters
Severity: Low
Files: `_build_polyline_mask` is byte-identical in
`nav_technique_body_limb.py` (51-61),
`nav_technique_body_terminator.py` (56-66),
`nav_technique_ring_edge.py` (60-70). `_peak_to_runner_up_ratio` is
byte-identical in `nav_technique_body_disc.py` (204-224) and
`nav_technique_ring_annulus.py` (82-102). Each technique also defines its
own near-identical `_XxxConfidenceContext` adapter.

WHY: Three exact copies of `_build_polyline_mask` and two exact copies of
`_peak_to_runner_up_ratio` are pure duplication — a bug fix or behavior
change must be applied in 2-3 places and will drift. `dt_fitting.py` is
the natural home for the polyline-mask helper (all three callers are DT
techniques); a shared `nav_technique.py` helper fits
`_peak_to_runner_up_ratio` (both callers are NCC techniques).

Impact: Maintenance hazard / drift risk; no current functional bug.
Hoist `_build_polyline_mask` into `dt_fitting` (or a shared technique
helper) and `_peak_to_runner_up_ratio` into `nav_technique.py`; import
from the single source. Verify all call sites unchanged behavior.

---

### CODE-TECH-006 — `RingEdgeNav` silently slices a non-(2,2) covariance to 2x2 while `BodyLimbNav`/`BodyTerminatorNav` log a WARNING for the same case
Severity: Low
File: `src/nav/nav_technique/nav_technique_ring_edge.py` lines 327-329;
compare `nav_technique_body_limb.py` lines 298-304 and
`nav_technique_body_terminator.py` lines 322-329.

In the `fit_rotation=False` branch, `BodyLimbNav` and
`BodyTerminatorNav` both do: if the LM returned a non-`(2,2)` covariance,
log a WARNING ("returned %s covariance with fit_rotation=False;
truncating") then slice `[:2, :2]`. `RingEdgeNav` does the slice with no
log:

```python
else:
    if covariance.shape != (2, 2):
        covariance = covariance[:2, :2]
    rotation_rad = None
```

WHY: The three DT techniques share `lm_subpixel_refine`; an unexpected
covariance shape from the shared fitter is a programmer-error signal the
two body techniques surface but the ring technique hides. Inconsistent
diagnostics for an identical anomaly.

Impact: A bug in `lm_subpixel_refine`'s covariance shaping would be
visible in body logs but silent in ring logs. Add the same WARNING (or,
better, fold the truncate-and-warn into a shared helper alongside
CODE-TECH-005). Trivial fix.

---

### CODE-TECH-007 — `BodyLimbNav.is_feasible` / `navigate` compare a *vertex count* against `min_arc_px` (a pixel-length threshold)
Severity: Medium
File: `src/nav/nav_technique/nav_technique_body_limb.py` lines 167-172 and
223-228; same pattern in `nav_technique_body_terminator.py` lines 191-196
and 243-248.

```python
and f.geometry.vertices_vu.shape[0] >= self._min_arc_px
```

The threshold is named `min_arc_px` (and the feasibility reason is
`no_limb_arc_features_with_sufficient_visible_arc`), but it is compared
against `vertices_vu.shape[0]`, the *number of polyline vertices*, not the
arc length in pixels. These coincide only when the polyline samples one
vertex per pixel of arc. If the model ever sub-samples the polyline (every
2nd pixel) or super-samples it, the gate's effective arc-length threshold
silently changes by that factor.

WHY: The name and the docstring assert a pixel-arc-length semantics that
the code does not enforce — it enforces a vertex-count semantics. The
diagnostic `visible_arc_px` later set to `float(vertices.shape[0])`
(line 348) reinforces the conflation: arc length is *assumed* equal to
vertex count everywhere.

Impact: Correctness depends on the unstated "one vertex == one pixel of
arc" invariant from the model. If the ring/limb extractor's vertex spacing
ever differs from 1 px, the feasibility gate and the `visible_arc_px`
diagnostic both silently mis-scale. Either rename the tunable to
`min_arc_vertices` and the diagnostic accordingly, or compute true arc
length from consecutive-vertex distances. Confirm the model's vertex
spacing in `nav_model_body._extract_limb_polyline` (1 px per the limb-mask
construction) and document the invariant explicitly.

---

### CODE-TECH-008 — `NavTechniqueResult.__post_init__` validates `confidence ∈ [0,1]` but not `offset_px` finiteness or `feature_ids` element types
Severity: Low
File: `src/nav/nav_technique/technique_result.py`, `__post_init__`
lines 50-72.

The validator carefully checks covariance shape/symmetry/PSD and
`confidence` range, and coerces `feature_ids` to a tuple, but does not
check that `offset_px` is a length-2 finite tuple, nor that
`feature_ids` elements are strings, nor that `rotation_rad` /
`sigma_rotation_rad` (when not None) are finite. A NaN offset or a
non-finite rotation flows straight into the ensemble combine.

WHY: This is the one dataclass every technique funnels through, and it is
the natural fail-fast boundary for "a technique produced garbage". The
covariance is validated to be finite (via `eigvalsh`), but the offset that
the covariance describes is not — an inconsistency. A NaN offset will
silently poison the precision-weighted merge downstream rather than
raising at the producing technique's boundary.

Impact: A buggy technique returning a NaN/inf offset is not caught at the
result boundary; the failure surfaces later in the ensemble with a less
localizable message. Add a finite-length-2 check on `offset_px` and a
finite check on the optional rotation fields. Verify with a unit test
passing `offset_px=(float('nan'), 0.0)`.

---

### CODE-TECH-009 — `BodyDiscCorrelateNav` slices the runner-up ratio helper's "negative-quality floored at 1e-9" path identically to the ring version, but the documented behavior is mathematically odd
Severity: Low
File: `nav_technique_body_disc.py` `_peak_to_runner_up_ratio`
lines 204-224 (and the ring-annulus twin). Flagging as a NEW non-math
observation on a duplicated helper.

When `runner_q <= 1e-9` the helper returns
`max(winner_q, 0.0) / 1e-9` — i.e. it divides by the hardcoded `1e-9`
sentinel, producing a ratio on the order of `winner_q * 1e9`. The
docstring frames this as "floored at a small positive value so the ratio
stays well-defined", but the resulting number is not a ratio against the
*runner-up* at all; it is `winner_q` scaled by `1e9`, which then feeds the
`peak_to_runner_up_ratio` confidence term as if it were a genuine
separation ratio. A clean single-peak case correctly returns `1.0`
(handled earlier), so this branch only fires when there *is* a second peak
but its quality collapsed to ~0 — exactly the "extremely unambiguous"
case, yet it produces an enormous, scale-dependent value rather than a
saturating one.

WHY: The confidence sigmoid term consuming `peak_to_runner_up_ratio`
presumably saturates for large inputs, so the `1e9`-scale output is
probably harmless in practice — but it is an undocumented magic
denominator that makes the term's magnitude depend on the absolute NCC
quality scale rather than on the peak/runner-up separation. This is a
shared helper (see CODE-TECH-005), so the oddity is duplicated.

Impact: Low (likely sigmoid-saturated downstream), but the value is
scale-dependent and the `1e-9` is a magic number in two files. When
consolidating per CODE-TECH-005, replace the divide-by-sentinel with an
explicit saturating cap (e.g. return a large constant) and document it.
Confirm by checking the confidence-spec divisor/cap_at for
`peak_to_runner_up_ratio` in `config_510_techniques.yaml`.

---

### CODE-TECH-010 — `log_confidence_breakdown` types its `logger` parameter as `Any`
> **Tracked by:** #105 — replace pervasive Any / dict[str, Any] with TypedDicts and Protocols at interop boundaries
Severity: Low
File: `src/nav/nav_technique/nav_technique.py`, `log_confidence_breakdown`
lines 155-157 (`logger: Any`).

Every technique passes `self.logger` (a `pdslogger.PdsLogger`) to this
helper, but the parameter is annotated `Any`, defeating mypy checking of
the `.info(...)`/`.debug(...)` call shapes inside. Under `strict = true`
this is an avoidable `Any` on a core helper.

WHY: The project's logging convention is specifically `PdsLogger` via
`NavBase.logger`; typing the parameter as `PdsLogger` would both document
the contract and let mypy verify the format-string call sites. The `Any`
also weakens the type story for the whole confidence-logging path.

Impact: Lost type checking on a shared helper. Annotate as
`pdslogger.PdsLogger` (already importable; `nav_technique_body_blob.py`
imports `PdsLogger` directly). Verify mypy still passes.

---

### CODE-TECH-011 — `NavTechniqueManual` is registered out of the autonomous registry via `_abstract = True`, conflating "abstract" with "opt-out of discovery"
Severity: Low
File: `src/nav/nav_technique/nav_technique_manual.py` line 71
(`_abstract = True`), with a concrete `name`,
`accepts_feature_types`, and full `is_feasible`/`navigate`
implementations.

`NavTechniqueManual` is fully concrete (it is instantiated by
`run_manual_nav`) yet sets `_abstract = True` purely to stay out of
`NavTechnique._registry`. The `_abstract` flag's documented meaning on the
base is "shared base / not directly instantiable"; here it is overloaded
to mean "do not auto-discover". The class is constructed directly
(`NavTechniqueManual(config=..., annotations=...)` at line 252), so it is
not abstract in any normal sense.

WHY: Overloading `_abstract` for two distinct concepts (true ABC bases
like the `NavTechnique` base vs. "concrete but not auto-run") makes the
registry semantics ambiguous and means a reader cannot tell from
`_abstract = True` whether a class is instantiable. A future maintainer
adding a second interactive/concrete-but-not-discovered technique has no
distinct flag to reach for.

Impact: Semantic muddiness in the registry contract; no runtime bug.
Consider a separate `_auto_register = False` (or `discoverable`) flag
distinct from abstractness, so "concrete, instantiated directly, not in
the autonomous registry" is expressed without claiming the class is
abstract. Low priority.

---



---

# Consolidated Fix Prompts

Every fix prompt below is self-contained and references its finding ID.


---

## Fix Prompts — Part 1 — Navigation Models, Techniques & Core Math (deep dive + preliminary peripheral scan)


Each prompt is self-contained and actionable by an AI with no prior context.

### FIX CODE-NAV-001 — Make `lm_subpixel_refine` report a non-zero RMS and honest covariance on non-convergence

> **STATUS: FIXED (uncommitted, 2026-06-10).** Implemented as specified: `degenerate` field on `LMRefineResult`, `rms_px=inf` and all-inf covariance on full rejection, spurious gates updated in limb/terminator/ring-edge. Covariance documented as data-only.
In `src/nav/nav_technique/dt_fitting.py`, function `lm_subpixel_refine`: after the
main loop, the final RMS is `sqrt(Σ(w·r²)/Σw)` computed from `final_weights`. When
every Tukey weight is zero (all vertices rejected) the loop breaks at the
`if not np.any(weights > 0): break` guard and `rms_px` is set to `0.0`, which
downstream spurious-gates read as a *good* fit. Change: when
`inlier_count == 0 or final_weights.sum() == 0.0`, set `rms_px = float('inf')`
(not `0.0`) so the DT techniques' `result.rms_px > floor` test fires. Also add a
new boolean field to `LMRefineResult`, `degenerate: bool`, set True in this case,
and have `BodyLimbNav`/`BodyTerminatorNav`/`RingEdgeNav` treat `degenerate=True`
as `spurious=True`. Separately, decide whether the reported `covariance` should
include the Tikhonov diagonal: if the covariance is meant to reflect the
*minimized* objective, add `tikhonov_lambda` to `hessian[0,0]` and `hessian[1,1]`
before calling `information_matrix_to_covariance`; otherwise add a one-line
docstring note that the covariance is data-only and intentionally excludes the
anchor. Verify with a new test `tests/nav/nav_technique/test_dt_fitting.py`:
feed a polyline whose vertices all fail polarity (gradient image of zeros with a
non-zero normal) and assert `result.rms_px == inf`, `result.degenerate is True`,
and `np.isinf(result.covariance).all()`.

### FIX CODE-NAV-002 — Compute limb/terminator vertex normals from the silhouette backplane, not the ridge mask

> **STATUS: FIXED (uncommitted, 2026-06-10).** `_build_polyline_sampler` now takes a `region_mask` and computes the outward normal from the silhouette (limb) / lit (terminator) mask; outward-normal regression test added; existing negation in the techniques preserved.
In `src/nav/nav_model/nav_model_body.py`, function `_build_polyline_sampler`: the
per-vertex normal is currently derived from `local_mask` (the 1-px ridge), making
its sign depend on the diagonal orientation of the ridge rather than on which side
is the body interior. Pass the body silhouette mask (`body_mask_valid`, or for the
terminator the lit mask `is_lit`) into `_build_polyline_sampler` and compute the
outward normal as the discrete gradient of *that* mask: for each vertex `(v,u)`,
`n_v = (mask[v-1,u] - mask[v+1,u])`, `n_u = (mask[v,u-1] - mask[v,u+1])` using the
body-side=True / space-side=False convention so the vector points from inside to
outside; normalize. (Better still, sample the incidence-angle or model-image
gradient at the vertex and use its negative as the inside-pointing direction.)
Keep the existing negation in `_aggregate_limb_features` so
`dot(-normal, image_gradient) > 0` means "brighter inside the limb." Verify with a
test that renders a synthetic lit disc, extracts the limb polyline, and asserts
`dot(normal_i, vertex_i - predicted_center_vu) > 0` for ≥ 95% of vertices.

### FIX CODE-NAV-003 / CODE-CFG-001 — Calibrate `signal_dn_to_image_unit_scale` and add a placeholder guard
In `src/nav/config_files/config_400_inst_coiss.yaml` (lines 116, 151) replace the
`5.0e-7  # PLACEHOLDER` values with the true Cassini DN→I/F conversion factor
(derive from the ISS calibration pipeline: I/F = DN · RADIANCE_factor / solar
flux at the relevant heliocentric distance, per camera/filter/gain/summation —
consult the CISSCAL / `calib` documentation; the value is camera- and
mode-specific). Add a config-validation step in
`src/nav/config/config.py` (or a `read_config` post-pass) that scans every loaded
string/comment value for the token `PLACEHOLDER` and raises a `ValueError` (or
logs a prominent WARNING gated by a `strict_config` flag) listing the offending
keys. Verify with `tests/nav/config/test_config.py::test_no_placeholder_coeffs`
asserting no shipped config retains a placeholder marker, and a `predicted_snr`
test confirming a known Cassini star yields a plausible SNR (order 10–1000, not
`sqrt(signal_dn)`).

### FIX CODE-NAV-004 / CODE-NAV-005 / CODE-NAV-014 — Correct the weighted-mean covariance normalization
In `src/nav/nav_technique/nav_technique_star_field.py` (`_build_covariance`,
`_build_covariance_3dof`) and `src/nav/nav_technique/nav_technique_body_blob.py`
(`_joint_covariance`): the parameter covariance is computed as
`Σ(w·r²)/(Σw)²`. Replace with the reduced-chi-square form: compute
`chi2_nu = Σ(w·r²) / max(N − p, 1)` (p = number of fitted parameters: 2 for
translation, 3 with rotation) and `Var(axis) = chi2_nu_axis / Σw`, where
`chi2_nu_axis = Σ(w·r_axis²)/max(N−p,1)`. Keep the existing positive-definite
floor but base it on `1/Σw` (the pure inverse-precision), not `1/Σw²`. For the
3-DoF rotation variance, use `σ_θ² = chi2_nu_residual / Σ(w·|cat−cc|²)` with
`chi2_nu_residual = 0.5·(Σw·r_v² + Σw·r_u²)/max(N−p,1)`. Add a model-error floor
(see FIX CODE-ORCH-001). Verify with unit tests using two-point and N-point
fixtures of known σ and known residuals, comparing reported per-axis sigma to the
analytic weighted-mean sigma within 1e-9.

### FIX CODE-NAV-006 — Add a raw (unweighted) RMS spurious check to the body limb/terminator techniques
In `src/nav/nav_technique/dt_fitting.py`, add a `raw_rms_px` field to
`LMRefineResult` computed as `sqrt(mean(residuals_px²))` over *all* vertices
(no weights). In `nav_technique_body_limb.py` and
`nav_technique_body_terminator.py`, extend the `spurious` predicate to include
`result.raw_rms_px > max(spurious_dt_floor_px, spurious_dt_rms_factor*sigma_min)`
(mirroring the per-edge check `RingEdgeNav` already does). Add the corresponding
tuning keys if absent. Verify with a fixture where one limb arc aligns and a
second is offset by 10 px: assert the result is flagged spurious even though the
Tukey-weighted `rms_px` is small.

### FIX CODE-NAV-007 — Normalize the coarse seed by in-bounds vertex count (or rename and widen the trust region)
In `src/nav/nav_technique/dt_fitting.py`, `coarse_ncc_search`: divide each shift's
overlap score by the number of in-bounds polyline vertices at that shift
(`valid.sum()`), so a shift that places more of the polyline over dense edges does
not win purely on density / count. Alternatively, if the raw-count behaviour is
intended, rename the function to `coarse_chamfer_seed`, update its docstring to
drop the (false) "NCC argmax unchanged" claim, and ensure the DT techniques'
`lm_trust_region_px` is wide enough to absorb the resulting seed bias. Verify with
a fixture whose image edge density is non-uniform across the search window:
assert the normalized seed matches the true per-shift NCC argmax.

### FIX CODE-NAV-008 — Handle rotation as an angle in the ensemble combine and grouping
In `src/nav/nav_orchestrator/ensemble.py`: at the top of `_combine_precision_weighted`
and `_agreement_groups`, when any result is 3-DoF, assert
`abs(rotation_rad) < math.radians(max_allowed_rotation_deg)` (thread the value
through, e.g. from the ensemble config or provenance) so the small-angle
assumption is enforced and documented. Properly, combine the rotation component on
the circle: accumulate `Σ w·sin θ` and `Σ w·cos θ` and report
`atan2(Σw sinθ, Σw cosθ)` for the rotation parameter while keeping the
information-form merge for translation. Verify with a test combining two 3-DoF
results at `+179°`-equivalent and `−179°`-equivalent small-angle stand-ins and
asserting the combined angle does not collapse to 0.

### FIX CODE-NAV-009 — Make the Mahalanobis null-space test relative
In `src/nav/nav_orchestrator/ensemble.py`, `_mahalanobis_distance`: change
`if np.linalg.norm(null_proj) > 1e-6:` to
`if np.linalg.norm(null_proj) > rel_tol * max(np.linalg.norm(delta), eps):` with
`rel_tol = 1e-6` and `eps = np.finfo(float).eps`. This makes the
"disagreement-along-the-null-direction" test scale-invariant. Verify with a test
that two near-rank-deficient covariances (one eigenvalue ~1e-7) with a small but
genuine agreement still group, while two truly disagreeing rank-1 results
(displacement ~3 px along the null axis) return `inf`.

### FIX CODE-NAV-010 — Derive `BodyDiscCorrelateNav` rotation sigma from a calibrated model or mark it unobservable
In `src/nav/nav_technique/nav_technique_body_disc.py`,
`_rotation_sigma_from_quality`: the current `sigma_sq = 1/(-H·q_centre)` is
dimensionally `rad²/quality²`. Replace with `sigma_sq = 1/(-H)` *only if* the NCC
quality is on a log-likelihood scale; since PSR/PMR are not, instead either (a)
return `None` (→ rotation-unobservable sentinel) until a calibrated mapping from
NCC-peak curvature to angular variance is derived from the per-pixel noise and
template energy, or (b) compute the CRB directly: `σ_θ² = noise_var / (Σ |∂T/∂θ|²)`
using the template's rotational derivative energy. Document the chosen approach.
Verify with a synthetic rotated-template fixture of known noise: assert the
reported σ_θ tracks the injected noise level (doubling noise ⇒ doubling σ_θ).

### FIX CODE-NAV-012 — Compute the brightness margin from the flux ratio, not the SNR ratio
In `src/nav/nav_technique/_star_helpers.py`, `brightness_margin_mag` is called
with predicted-SNR values; change `StarUniqueMatchNav` to pass
`integrated_signal_dn` (the true in-band flux) for the brightest and runner-up,
and keep `Δmag = 2.5·log10(flux1/flux2)`. This removes the shot-vs-background
SNR-linearity assumption. Verify with two stars of known V magnitudes: assert the
returned margin equals their catalog Δmag (plus the per-instrument mag_offset
difference, which is zero for same-band stars).

### FIX CODE-ORCH-001 — Add a per-technique model-error covariance floor and shrink the ensemble pixel floor
At the point each technique builds its covariance (the star, blob, limb, ring,
disc techniques), add a configurable model-error variance floor
`sigma_model_px²·I` (e.g. a `tuning.model_error_floor_px` per technique,
reflecting SPICE pointing residual + body-shape model error) so the reported
covariance is `CRLB ⊕ model_error`. Then in
`src/nav/nav_orchestrator/ensemble.py` reduce `agreement_pixel_floor` toward 0 and
rely on the (now-honest) Mahalanobis grouping; if a floor is still wanted, apply
it only on the *observable* subspace by projecting the pixel distance through the
summed-covariance pseudoinverse rather than using raw Euclidean distance. Verify
with the integration image library that the previously floor-dependent groupings
still form, and add a unit test that two rank-1 ring-edge results disagreeing
along their unobservable axis are *not* grouped.

### FIX CODE-ORCH-002 — Narrow the orchestrator plugin-sandbox excepts and always log tracebacks
In `src/nav/nav_orchestrator/orchestrator.py` (lines ~551, 640, 663, 755):
replace `except Exception:` with a handler that re-raises programmer-error types
(`AssertionError`, `TypeError`, `AttributeError`, `KeyError`, `NameError`) and
only swallows a defined set of "expected plugin failure" exceptions, logging the
full traceback at ERROR (`self.logger.exception(...)`) in every caught case. This
ensures the techniques' deliberate `RuntimeError` contract guards surface. Verify
with a test that injects a technique stub raising `AttributeError` and asserts it
propagates (or is logged at ERROR with a traceback), not silently dropped.

### FIX CODE-NAV-016 — Deduplicate `_build_polyline_mask`
Move the identical `_build_polyline_mask` from `nav_technique_body_limb.py`,
`nav_technique_body_terminator.py`, and `nav_technique_ring_edge.py` into
`src/nav/nav_technique/dt_fitting.py` (or `support/distance_transform.py`) as a
single public helper and import it in all three. No behaviour change; verify the
existing technique tests still pass.

### FIX CODE-NAV-MODEL-002 / CODE-NAV-MODEL-001 — Move body emission thresholds and the photometric sigma coefficient into config
Move `LIMB_ARC_MAX_UNCERTAINTY_PX`, `BODY_BLOB_MIN_DIAMETER_PX`,
`BODY_DISC_MIN_VISIBLE_LIT_FRACTION`, `BODY_DISC_MAX_OVERFLOW_FRACTION`,
`TERMINATOR_MIN_VERTICES`, `TERMINATOR_MIN_PHASE_FACTOR`, and the `0.5`
photometric-softness coefficient in `_sigma_normal_per_vertex` into the
`config.bodies` YAML section, reading them through `self._config.bodies` with
fail-fast `KeyError` on missing keys (consistent with the techniques' policy).
Keep the module constants only as documented defaults if the rest of the codebase
imports them. Verify config round-trips and that body feature emission is
unchanged at the default values.

### FIX CODE-NAV-MODEL-004 / CODE-SUP-001 / CODE-PDS4-001 — Narrow broad excepts in model/support/pds4
For `nav_model_rings.py` shadow removal (line 433), `support/misc.py` robust-noise
helpers (lines 142, 173), and `pds4/collections.py` (lines 84, 118, 180, 301,
316): replace `except Exception:` with the specific exception types actually
expected (oops/SPICE geometry errors, numpy errors, I/O / template errors
respectively), log the traceback, and re-raise anything unexpected. For the noise
estimator specifically, prefer failing the navigation over silently substituting
a fallback noise value that mis-scales every downstream threshold. Verify with
tests that inject an unexpected exception and assert it propagates.

### FIX CODE-REPROJ-001 — Make oops precision reduction safe under concurrency
In `src/nav/reproj/_context_managers.py` and the mosaic CLIs: either serialize all
`RingMosaic.reproject` calls behind a process-wide lock around the
`_reduced_oops_precision` block, or thread the precision setting as a per-call
parameter into the oops calls rather than mutating `oops.config` globals. Document
the chosen contract in CLAUDE.md's gotchas. Verify with a test that runs two
reprojections concurrently and asserts the second's geometry precision is
unaffected by the first's reduced-precision block.

### FIX CODE-CFG-002 — Fail fast on missing correlation upsample config
In `src/nav/nav_technique/nav_technique_body_disc.py`, `_upsample_factor`: replace
the `getattr(..., 128)` defensive defaults with a direct
`int(self.config.offset.correlation_fft_upsample_factor)` that raises on a missing
block/key, consistent with the other techniques' fail-fast-on-config-typo policy.
Verify a missing key raises at `__init__`/navigate rather than silently using 128.

### FIX CODE-NAV-020 / module-size — Split oversized modules
Split `src/main/nav_create_simulated_image.py` (2509 lines),
`src/nav/reproj/bodies.py` (1980), `src/nav/reproj/rings.py` (1921),
`src/nav/support/flux.py` (1174), `src/nav/nav_model/nav_model_body.py` (1118),
and `src/nav/nav_technique/nav_technique_star_field.py` (993, approaching the cap)
into packages under the 1000-line guideline, extracting pure-function helpers into
testable submodules. No behaviour change; verify full suite and `sphinx-build -W`
still pass.


---

## Fix Prompts — Part 2 — Image Reading (obs) & Dataset Enumeration


### FIX CODE-OBS-001
In `src/nav/obs/obs_inst.py` `ObsInst.star_psf_size`, rewrite the lookup so it does not rely on the
leaked loop variable. Sort the thresholds, iterate, and capture an explicit `default_mag = max(keys)`
(raise `ValueError('star_psf_sizes is empty')` if no keys). Validate that each `star_psf_sizes[mag]`
is a 2-element sequence and return `tuple(int(x) for x in ...)`; assert `len == 2`. Verify with a unit
test passing a star fainter than every threshold (expect the largest size) and an empty `star_psf_sizes`
(expect ValueError, not UnboundLocalError).

### FIX CODE-OBS-003
In `src/nav/obs/obs_snapshot.py` `ObsSnapshot.__init__`, wrap the closest-planet loop so a SPICE
failure in `body_distance(planet)` is caught per-planet, logged via the instance logger with the
planet name and image identity, and excluded from the min search; only raise if *no* planet resolves.
Document that ObsSim intentionally pre-sets `_closest_planet`. Verify by constructing an ObsSnapshot
with one planet's kernels absent and asserting it still picks the nearest resolvable planet (add a test
that monkeypatches `body_distance` to raise for one planet).

### FIX CODE-OBS-004
In `src/nav/obs/obs_snapshot.py` `_ra_dec_limits`, delete the declination wrap-around block
(lines 506-509) — declination cannot wrap. Keep the RA wrap. Add a comment explaining dec is bounded
to [-pi/2, +pi/2]. Verify existing `ra_dec_limits` tests still pass.

### FIX CODE-OBS-007
In `src/nav/obs/obs_inst_cassini_iss.py` `star_min_usable_vmag`, remove the dead `if self.detector ==
'WAC'` branch (both return 0.0) or replace it with the intended WAC-specific minimum and a sourcing
comment. Verify no test asserts the old branch.

### FIX CODE-OBS-008
In `src/nav/obs/obs_inst_cassini_iss.py` `get_public_metadata`, parse `SPACECRAFT_CLOCK_START_COUNT`
/ `STOP_COUNT` defensively: handle the `partition/count` form (split on `/`, take the count), use
`self.dict.get(...)` with a clear error if absent, and wrap `float()` in a try that raises a contextual
`ValueError` naming the image. Verify with a test feeding both a plain numeric SCLK and a
`"1/1234567890.123"` form.

### FIX CODE-OBS-011 / CODE-OBS-012
In `src/nav/obs/obs_inst_voyager_iss.py` `from_file` and `get_public_metadata`, extract the spacecraft
id once with validation: read `obs.dict.get('LAB02')`, assert it is a str of length >= 5, take char [4],
and assert it is `'1'` or `'2'` (raise a clear ValueError otherwise). Reuse that validated value in both
the I/F-correction branch and the LID builder. For the I/F factor, replace the bare `.replace(...)` +
`float(...)` with a guarded parse: if the fixed phrase is absent or the remainder is not numeric, raise
`ValueError(f'Unexpected Voyager LABEL3 format: {label!r}')`. Verify with tests for V1@Saturn (correction
applied), V2 (not applied), and a malformed LABEL3/LAB02 (clear error).

### FIX CODE-OBS-013
In `src/nav/obs/obs_inst_voyager_iss.py`, change the `%`-style `logger.debug(...)` call to an f-string
to match the file's other debug calls. No functional change; verify ruff/format clean.

### FIX CODE-OBS-016 / CODE-DS (forwarding __init__)
Add `ObsSnapshotInst._resolve_extfov_margin(inst_config, data_shape_v, override) -> tuple[int,int]`
that encapsulates the dict-vs-scalar resolution (with a clear KeyError message per CODE-OBS-017) and call
it from all five `from_file`s. Separately, delete the no-op forwarding `__init__`s in the four PDS3
dataset subclasses (they only call super). Verify mypy + existing obs/dataset tests pass.

### FIX CODE-OBS-017
In the shared `_resolve_extfov_margin` (or each `from_file` if not refactored), when the margin entry is
a dict, use `entry.get(data_shape_v)` and raise `ValueError(f'No extfov_margin_vu configured for image
height {data_shape_v}; configured heights: {sorted(entry)}')` instead of a bare KeyError. Verify with a
test passing a synthetic image height absent from the config.

### FIX CODE-OBS-019
Delete the redundant `ObsSim.__init__` override in `src/nav/obs/obs_inst_sim.py`. Verify sim tests pass.

### FIX CODE-OBS-020 / CODE-DS-025
In `src/nav/obs/__init__.py` `inst_name_to_obs_class` and `src/nav/dataset/__init__.py`
`dataset_name_to_class`/`dataset_name_to_inst_name`, catch `KeyError` and re-raise
`ValueError(f'Unknown name {name!r}; valid: {inst_names()}')`. Verify with a test asserting the message.

### FIX CODE-DS-001
In `src/nav/dataset/dataset_pds3.py` `_yield_image_files_index`, initialize
`vol_start_idx: int | None = None` and `vol_end_idx: int | None = None` before the guards so mypy and
readers see them defined; keep the existing `is not None` guards. Run `mypy src/nav/dataset` to confirm.

### FIX CODE-DS-002
In `src/nav/dataset/dataset_pds3.py`, rework `choose_random_images`: read each chosen volume's index
once, collect *all* rows that pass every filter, then `random.sample` up to the remaining needed count
from that pool; track volumes already exhausted and stop when no eligible rows remain anywhere (bound the
loop by the set of volumes, not `while True`). Verify with `--choose-random-images 5 --camera wac` on a
NAC-heavy volume (must terminate and return WAC images only).

### FIX CODE-DS-003 / CODE-DS-010
In `src/nav/dataset/dataset_pds3.py`, do not use the `done=True` monotonic early-exit when the index can
interleave cameras (or sort rows by image number before scanning). In
`src/nav/dataset/dataset_pds3_cassini_iss.py` `yield_image_files_index` BOTSIM grouping: (a) never drop
`last_imagefile` — always yield it when it cannot be paired; (b) base the pairing on
`IMAGE_NUMBER` equality of the two frames (BOTSIM N/W share the counter) plus opposite camera letters,
not an `abs(diff) <= 3` "seconds" heuristic; if you must keep a tolerance, comment that IMAGE_NUMBER is a
counter, not seconds. Add a test with a lone BOTSIM frame between two real pairs asserting every frame is
yielded exactly once and pairs are (N,W).

### FIX CODE-DS-004
In `src/nav/dataset/dataset_pds3.py` `yield_image_files_from_arguments`, read
`--image-filespec-csv` and `--image-file-list` via `FCPath(filename).open()` (or `.read_text()`) so they
may be URLs, consistent with the rest of the layer. Verify a local file still works and a `file://`/http
path is accepted.

### FIX CODE-DS-005
In the CSV reader, after locating `colnum`, guard each row: `if colnum >= len(row): log.warning(...);
continue`. Verify with a CSV containing one short row that the run skips it and continues.

### FIX CODE-DS-006
In `src/nav/dataset/dataset_pds3.py` `--image-file-list` loop, rename the inner token variable (e.g.
`token = line.split(' ')[0]`) so the outer `filename` (the list-file path) survives, and include both the
list-file path and the bad token in the `ValueError` message. Verify the error names the source file.

### FIX CODE-DS-007 / CODE-DS-009
Either wire `_validate_selection_arguments` into `yield_image_files_from_arguments` (call it first) or
delete it; and delete the large commented-out blocks (BOTSIM-in-base, force_has_* selection, planet
validation), moving any still-wanted plan into an issue. Verify ruff clean and tests pass.

### FIX CODE-DS-011
In `src/nav/dataset/dataset_pds3_cassini_iss.py` `pds4_bundle_path_for_image`, raise
`ValueError(f'Invalid Cassini image name {image_name!r}')` for names shorter than 11 chars instead of
returning `''`, and update `pds4_path_stub` accordingly. Verify with a test passing a too-short name.

### FIX CODE-DS-013
In `src/nav/dataset/dataset_pds3_cassini_iss.py` `_check_additional_image_selection_criteria`, use
`getattr(arguments, 'camera', None)` so programmatic callers without a `camera` attribute are treated as
"no filter". Verify with a Namespace lacking `camera`.

### FIX CODE-DS-015
In `src/nav/dataset/dataset_pds3_galileo_ssi.py` `_get_img_name_from_label_filespec`, replace the two
hard-coded directory whitelists with structural logic (e.g. the image name is the last path component
ending in `.LBL`; derive depth from the filespec) or move the directory set into config so new encounter
directories do not silently drop images. At minimum, log a warning when an unrecognized directory is
skipped. Verify by passing a filespec with a new directory and asserting a warning + correct name
extraction.

### FIX CODE-DS-017
In `src/nav/dataset/dataset_pds3_voyager_iss.py` `_img_name_valid`, accept the on-disk product forms
(`Cddddddd`, `Cddddddd_GEOMED`, `Cddddddd_CALIB`, with optional `.IMG/.LBL`) by stripping a known
suffix/extension before the length check, mirroring how Cassini strips `_CALIB`. Verify an
`--image-file-list` containing `C1234567_GEOMED.IMG` is accepted.

### FIX CODE-DS-019
Confirm the intended NH LORRI product scope. If only science frames should be navigated, remove `_eng`
from both `_get_label_filespec_from_index` and `_get_img_name_from_label_filespec` (and the `_img_name_valid`
path), leaving only `_sci`. If ENG is intentional, add a comment documenting why. Verify with an index row
referencing an `_eng.lbl` product.

### FIX CODE-DS-024
In `src/nav/dataset/__init__.py`, replace the module-level `assert` with an explicit
`if sorted(...) != sorted(...): raise RuntimeError('Dataset name registries are inconsistent')` so it is
not stripped under `-O`, or move the check into a unit test. Verify by desyncing the maps and importing.

### FIX CODE-DS-026
In `src/nav/dataset/dataset_sim.py` `pds4_template_variables`, add the keyword-only `*` to match the base
signature: `def pds4_template_variables(self, *, image_file, nav_metadata, backplane_metadata)`. Run mypy
to confirm the override now matches.


---

## Fix Prompts — Part 3 — Reprojection, Backplanes & PDS4


### Fix CODE-BACKPLANE-001
In `src/backplanes/backplanes.py`, function `generate_backplanes_image_files` (around line 53-54), the nav status guard compares against the wrong literal. Nav writes `NavResult.status` whose success value is `'ok'` (`Literal['ok','failed','conflicted']`, see `src/nav/nav_orchestrator/nav_result.py` and `src/nav/navigate_image_files.py` line 192). Change `if status != 'success':` to `if status != 'ok':`. Keep the existing warning/return behavior for non-`'ok'` statuses. Verify by: writing a metadata JSON with `{"status":"ok","offset":[1.0,2.0]}`, calling the function, and confirming it proceeds to build backplanes (does not log "Skipping"); add/adjust a unit test asserting that `status=='ok'` is processed and `status in {'failed','conflicted','error'}` is skipped.

### Fix CODE-PDS4-001
In `src/pds4/bundle_data.py`, function `generate_bundle_data_files` (around line 53-54), change the status guard `if status != 'success':` to `if status != 'ok':` (nav writes `'ok'` for success; see CODE-BACKPLANE-001). Keep the warning/return for other statuses. Verify by: supplying a nav metadata JSON with `status='ok'` plus a matching backplane metadata JSON, running the function, and confirming the supplemental file, data label, and browse label are written; add a unit test covering `'ok'` (processed) vs `'failed'` (skipped).

### Fix CODE-PDS4-002
In `src/pds4/collections.py`, function `generate_global_index_files`, replace the hand-built LID at line 196 `lid = f'{pds4_bundle_name}:data:{image_name}'` with the dataset's canonical builder. Use the LID (not LIDVID) form: add `pds4_image_name_to_data_lid` to the `DataSet` base API if not already public (it exists on the Cassini subclass, `src/nav/dataset/dataset_pds3_cassini_iss.py` line 428) and call `lid = dataset.pds4_image_name_to_data_lid(image_name)`. Ensure `image_name` passed in matches what that method expects (it internally strips `_`/`.` suffixes and reorders the first character). Verify by: generating the global index for a known image (e.g. `N1234567890_1`) and asserting the `LID` column equals `urn:nasa:pds:{bundle}:data:1234567890n` (matching `pds4_image_name_to_data_lid`), and that it equals the LID portion of the value `collection_data.tab` writes via `pds4_image_name_to_data_lidvid`.

### Fix CODE-BACKPLANE-002
Decide the intended unit for backplane statistics and make FITS data, BUNIT, supplemental JSON, and global-index columns consistent. In `src/backplanes/backplanes_bodies.py` (lines 178-183) and `src/backplanes/backplanes_rings.py` (lines 93-98), the stats are converted to degrees for `units=='rad'` while the FITS arrays in `writer.py` stay in radians with `BUNIT='rad'`. Either (a) stop converting the stats (store radians, matching the FITS/BUNIT), or (b) keep degrees in the stats but record the stats unit explicitly (e.g. emit a `units` field alongside min/max and set the global-index column header/label to degrees). Update `src/pds4/collections.py` index headers and any PDS4 templates to match the chosen unit. Verify by: dumping the FITS BUNIT and the supplemental JSON for an angular backplane (e.g. `body_incidence_angle`) and confirming the numeric ranges and declared units are mutually consistent.

### Fix CODE-BACKPLANE-003
In `src/backplanes/merge.py`, `merge_sources_into_master`, replace the scalar per-body distance (`float(entry['distance'])` broadcast over the mask, lines 34-46) with a per-pixel body distance backplane when available. If `create_body_backplanes` can emit a per-pixel `distance` array per body (mirroring `create_ring_backplanes` `result['distance']`), use it to build `body_dist_stack` so `nearest_body_idx`, `nearest_body_distance`, and the ring occlusion test (line 126) compare like-for-like per-pixel distances. If per-pixel body distance is not yet computed, add it to `create_body_backplanes`. Verify by: a synthetic scene with a body limb crossing the ring plane and confirming ring pixels just outside the body limb are no longer occluded by the body-center distance; add a regression test on the occlusion mask.

### Fix CODE-REPROJ-001
In `src/nav/reproj/rings.py`, `_reproject_inner`, make the global antimask placement consistent with the global mosaic's `bin*res` longitude convention. The per-column actual longitude is `lon_bins_restr * res + longitude_start` (line 1256). For the global antimask (lines 1402-1403) compute the global bin from the actual longitude rather than `lon_bins_restr + full_min_lon_bin`: e.g. `global_bins = np.round((lon_bins_restr[good_lon_antimask] * self._lon_resolution + longitude_start) / self._lon_resolution).astype(int) % self._n_full_lon`, and likewise ensure the per-column longitudes used for sampling agree with the bin centers the mosaic will assign. Alternatively, require/snap `longitude_start` to a multiple of `lon_resolution` and document it. Verify by: reproject the same simulated ring with `longitude_range` starts differing by a non-multiple of `lon_resolution`, accumulate both into one mosaic, and assert identical features map to the same global longitude bins (no sub-bin drift); add a unit test on `new_antimask` indices for a non-grid-aligned start.

### Fix CODE-REPROJ-002
In `src/nav/reproj/bodies.py`, `src/nav/reproj/rings.py`, `src/nav/reproj/cartographic_model.py`, and `src/nav/reproj/_serialization.py`, replace stdlib `logging` with the project pdslogger. Remove `import logging`; obtain the logger from `nav.config` (e.g. `from nav.config import IMAGE_LOGGER`) consistent with `src/backplanes/*`, and change the `logging.Logger` type hint in `rings._reproject_inner` (line 1130) to `pdslogger.PdsLogger`. If the maintainer intends `nav.reproj` to be exempt (it is not in CLAUDE.md's explicit core list), document that exemption instead. Verify by: `grep -rn "import logging" src/nav/reproj` returning nothing (or only the documented exemption), `ruff check`/`mypy src` passing, and a reprojection test capturing log output via `capsys`.

### Fix CODE-BACKPLANE-004
In `src/backplanes/backplanes_bodies.py`, `_create_simulated_body_backplane` (lines 50-58), narrow the bare `except Exception:` to the specific expected failures of the name-lookup + index-map slice (`ValueError`, `KeyError`, `IndexError`). Prefer letting unexpected exceptions propagate rather than silently falling back to a full-rectangle mask; if a fallback is genuinely needed for sim robustness, log it at warning level with the body name and reason. Verify by: a sim test where the body is present in `sim_body_order_near_to_far` produces a correct sub-mask, and a test where the name is genuinely missing raises (or warns) rather than silently filling the whole rectangle.

### Fix CODE-BACKPLANE-005
In `src/backplanes/merge.py` (lines 47-54) narrow `except Exception` around `cspyce.bodn2c` to the actual cspyce "name not found" error type. Replace the non-deterministic fake NAIF id `10000 + (abs(hash(body_name)) % 20000)` with a stable mapping (e.g. `int.from_bytes(hashlib.sha1(body_name.encode()).digest()[:4], 'big')` reduced into a reserved int32 range, or an explicit per-run name→id registry that guarantees uniqueness). Apply the same stable-hash fix to the seed in `_create_simulated_body_backplane` (`src/backplanes/backplanes_bodies.py` line 40). Verify by: running sim backplane generation twice in separate processes and asserting identical `BODY_ID_MAP` values and identical fill values; add a test that two distinct sim body names do not collide.

### Fix CODE-REPROJ-003
Align the uint16 cap documentation with the implementation. In `src/nav/reproj/bodies.py`, update the constructor `Note` (lines 634-636) and `add` docstring (lines 1404-1405) to state the mosaic holds up to 65,536 contributing images (image_number 0..65535) and that `add()` raises `OverflowError` on the 65,537th image, matching the rings docstring (lines 719-723) and the `self._image_count > np.iinfo(np.uint16).max` guard. Optionally make the bodies and rings guards textually identical. Verify by: confirming the docstrings match the `>`-comparison behavior; no code change to the guard is required.

### Fix CODE-REPROJ-004
In `src/nav/reproj/rings.py`, remove the inaccurate `obs.fov` mutation claim from the module docstring (lines 6-8), the `RingMosaic` class Notes (lines 714-716), and the `reproject` docstring. Replace with the accurate hazard: reproject mutates oops global light-travel precision via `_reduced_oops_precision` and builds a `Backplane` on the shared `obs`, so concurrent calls on the same `obs` (or in the same process) are unsafe. Verify by: `grep -n "obs.fov" src/nav/reproj/rings.py` showing no remaining mutation claim, and `_reproject_inner` not assigning `obs.fov`.

### Fix CODE-REPROJ-005
In `src/nav/reproj/ring_orbit_model.py`, fix the `radius_at_longitude` method docstring (lines 91-93) so the pericenter expression reads `w0 + dw * et / 86400` (or "w0 plus dw per day times elapsed days"), matching the implementation (line 102) and the class docstring (line 38). No code change. Verify by reading the corrected docstring; optionally add a doctest-style example checking `radius_at_longitude` at a known et.

### Fix CODE-PDS4-003
In `src/pds4/collections.py`, standardize the `template.write` invocations and exception handling. Pass the `FCPath` label path directly to `template.write(...)` everywhere (drop the `cast(Path, ...get_local_path())` + trailing `.upload()` pairs at lines 75-90 and 111-124, since `pdstemplate.PdsTemplate.write` normalizes to FCPath and uploads via `write_bytes`). Remove the `try/except Exception: logger.exception(...); raise` wrappers (lines 82-89, 116-123, 299-303, 314-318) or narrow them to the specific pdstemplate error and avoid double-logging (do not both `logger.exception` and re-raise). Mirror the cleaner pattern already used in `src/pds4/bundle_data.py` line 113. Verify by: generating collection and global-index labels and confirming the .lblx files appear at the remote/local destination exactly as before, with a single traceback (not two) on a forced template error.

### Fix CODE-PDS4-004
In `src/pds4/collections.py`, `generate_global_index_files`, make the parent-directory handling for `bodies_tab` and `rings_tab` consistent. Since `FCPath.get_local_path()` already creates parents by default, remove the redundant `rings_tab_local.parent.mkdir(parents=True, exist_ok=True)` at line 262 (and do not add one for bodies), OR add the same `mkdir` before the bodies write (line 234) and keep both. Choose one. Verify by: generating both index .tab files into a fresh (nonexistent) `document/supplemental` directory and confirming both are written without error.

### Fix CODE-REPROJ-006
In `src/nav/reproj/cartographic_model.py`, `create_cartographic_model`, optionally add an `override_backplane: Any = None` keyword parameter mirroring `BodyMosaic.reproject`, and use it instead of building `oops.backplane.Backplane(obs)` at line 95 when provided (still falling back to constructing one when `None`). Tighten the `obs` type from `Any` toward the project Observation/oops type if a suitable alias exists. Verify by: passing a pre-built Backplane and confirming `latitude`, `longitude`, and `center_resolution` are read from it (no second Backplane construction), and that omitting it preserves current behavior; keep the documented thread-safety note about constructing a Backplane from `obs`.


---

## Fix Prompts — Part 4 — Feature, Annotation, Simulator, Config, Util & Experiments


### Fix CODE-CFG-1
In `src/nav/config/config.py`, `Config.update_config`, replace the shallow per-key
`self._config_dict[key].update(new_config[key])` merge with a recursive deep-merge
helper so nested mappings combine key-by-key (user value wins at the leaf; sibling
default keys under the same sub-block are preserved). Add a module-level
`_deep_merge(base: dict, overlay: dict) -> dict` that recurses when both values are
`dict`, otherwise overwrites. Leave list/scalar override semantics as "overlay
replaces". Verify: write a default with `bodies: {MIMAS: {radii_km: [1], albedo: 2}}`
and a user file with `bodies: {MIMAS: {albedo: 9}}`; after `update_config` confirm
`config.bodies['MIMAS']` still has `radii_km == [1]` and `albedo == 9`. Add a unit
test in `tests/nav/config/`.

### Fix CODE-CFG-2
In `src/nav/config/config.py`, `Config.read_config`, make the `config_path is None`
+ `reread=True` path do a clean reload: clear `self._config_dict = {}` (and reset
the AttrDicts) before re-globbing `config_files/*.yaml`. Ensure the
`path is not None` branch also honors `reread` consistently (it already reassigns;
document that it is always a full replace). Verify: load config, then point the
loader at a temp dir missing a previously-present key and `read_config(reread=True)`;
assert the removed key is gone.

### Fix CODE-CFG-3
In `src/nav/config/config.py`, either (a) cache `category()` results the same way
`_update_attrdicts` caches the named sections, or (b) document that `category()` and
the section properties return read-only snapshots and stop allocating a new AttrDict
per call by returning the cached section dict. Pick one identity/caching model and
apply it uniformly across `category`, `planets`, `satellites`, `fuzzy_satellites`,
`ring_satellites`. Verify mypy + existing config tests pass.

### Fix CODE-CFG-4
Decide whether `nav.config` is exempt from the no-stdlib-`logging` rule. If not
exempt, in `src/nav/config/logger.py` drop `import logging` and replace the
`logging.Handler` / `logging.FileHandler` annotations with the corresponding
`pdslogger` handler types (or `Any` with a comment). If exempt, add a one-line
comment at the import explaining the exemption. Verify `ruff check` and `mypy src`
stay clean.

### Fix CODE-SIM-1
In `src/nav/sim/render.py`, when seeding each body in `_render_combined_model_cached`
and `_render_bodies_positioned_cached`, derive a per-body seed by mixing the body
name/index into the scene seed, e.g. `body_seed = (random_seed ^ (hash(body_name)
& 0x7FFFFFFF)) & 0x7FFFFFFF`, and pass that as `seed=` to `_render_single_body` /
`_render_body_shape_cached`. Keep `random_seed` as the scene-level seed for
noise/stars. Verify: render a 2-body scene with two identically-shaped bodies and
assert their crater patterns differ (`not np.array_equal` of the two masks' interior
intensities) while a re-render with the same scene seed is bit-identical.

### Fix CODE-SIM-2
In `src/nav/sim/sim_body.py`, `_add_craters_and_shading`, stop hard-zeroing the AA
rim: instead of `intensity_out[~ellipse_mask_nz] = 0.0`, multiply by the soft
`ellipse_mask` (which already encodes the AA rim) and only zero pixels with
`ellipse_mask == 0`. Confirm the crater path now honors `anti_aliasing` by rendering
the same body with `crater_fill>0` at `anti_aliasing=0` vs `1` and asserting the
limb-row intensity profile is softer in the AA case. Add a regression test.

### Fix CODE-SIM-3
Unify the illumination convention. Extract a single helper that, given
`illumination_angle`, `phase_angle`, and `rotation_z`, returns the normalized 3-D
light vector in image coordinates, and call it from BOTH `_lambertian_shading` and
`_add_craters_and_shading`. In the crater path, apply the same `rotation_z`
treatment to the lighting/gradient frame that the smooth path applies to the normal.
Verify: render one body with `crater_fill=0` and one with `crater_fill>0` at
`rotation_z=pi/2` and matched illumination; assert the lit-hemisphere centroid
(intensity-weighted) agrees within a pixel.

### Fix CODE-SIM-4
In `src/nav/sim/render.py`, `_render_combined_model_cached`, composite rings/gaps
*additively* over the existing `img` instead of overwriting. For RINGLET call the
additive `render_ring` path directly on `img` (or add `ring_coverage` only where it
exceeds the current pixel within the range order); for GAP subtract `gap_coverage`
from `img` (`img[mask] = clip(img[mask] - gap_coverage[mask], 0, 1)`) rather than
writing `temp_bg`. Reuse `sim_ring.render_ring`'s add/subtract logic. Verify: render
noise+star background, then a partial gap; assert the gap pixels are *darker* than
the surrounding background, not near-white, and that background outside the ring is
unchanged.

### Fix CODE-SIM-5
In `src/nav/sim/render.py`, raise the `maxsize` on the inner caches (or drop them in
favor of the single combined cache) so a two-scene alternation hits. At minimum bump
`_render_combined_model_cached` to `maxsize=8` and the body-shape cache stays at 30.
Verify no behavior change via existing sim tests and a quick timing check that a
repeated render of two scenes no longer re-serializes every call.

### Fix CODE-SIM-6
Remove `render_stars` and `render_bodies` from `src/nav/sim/render.py` if confirmed
unused (grep shows only `render_combined_model` is imported externally), or, if kept,
return `copy.deepcopy` of the cached star list / body dicts so callers cannot mutate
cache state. Verify the full test suite and `nav_create_simulated_image` still run.

### Fix CODE-SIM-7
In `src/nav/sim/render.py` inventory construction, compute separate v/u half-extents
from the projected ellipse (account for `rotation_z`/`rotation_tilt`) instead of a
single `max(axis1,axis2,axis3)/2`. At minimum use `axis1/2` for one axis and
`axis2/2` for the other before tilt. Verify the inventory bbox encloses the rendered
`body_mask` (assert `mask` is within `[v_min_unclipped, v_max_unclipped] x [...]`).

### Fix CODE-SIM-8
In `src/nav/sim/render.py`, reconcile the star flux comment and zero-point: either
change the comment at lines 113-116 to state the true zero-point (peak=1 at vmag=0)
or rescale so the documented vmag=4 maps to a sensible peak, and raise the default
`vmag` only if intended. Document the clip-to-[0,1] in `render_stars`. Verify a
`vmag=0` star peaks at ~1.0 and a default star is visible above the configured noise.

### Fix CODE-SIM-9
In `src/nav/sim/sim_ring.py`, `compute_edge_radius_at_angle` and
`_compute_edge_radii_array`, replace the silent `e >= 1.0 -> 0.99` clamp with a
raised `ValueError` (sim code may raise) naming the offending `a`/`ae`. Verify a
test that passes `ae > a` now raises with a message containing both values.

### Fix CODE-FEAT-1
In `src/nav/feature/composition.py`, replace attribute-by-name access on the geometry
union with explicit `isinstance` dispatch (or define a `Protocol`/base carrying
`bbox_extfov_vu`). In `compose_template_features` filter to the known template
geometries by `isinstance` before reading `bbox_extfov_vu`; in `compose_dialog_overlay`
dispatch polyline rendering on `isinstance(geometry, (LimbPolyline, TerminatorPolyline,
RingEdgePolyline))` instead of `getattr(..., 'vertices_vu', None)`. Verify mypy-strict
passes and existing composition tests still render identical output.

### Fix CODE-FEAT-2
No code change required; record for the scoring/math review that `reliability` and
`reliability_reasons` are decoupled. If desired, add an optional debug assertion in
the extractors (not in `NavFeature.__post_init__`) that recomputes the scalar from the
breakdown and warns via pdslogger on large divergence. Verify it does not run in the
hot path.

### Fix CODE-ANNO-1
In `src/nav/support/image.py`, `draw_rect`, clip all four edge slices to
`[0, shape)` (mirror the bounds checks already in `draw_circle`'s inner loop) so a
negative or out-of-range center cannot wrap-around. Then simplify
`_paint_star_marker` to rely on the primitive's clipping. Verify a star marker with a
center one pixel inside the FOV edge does not paint a stray rectangle on the opposite
edge; add a unit test.

### Fix CODE-ANNO-2
In `src/nav/annotation/annotation_text_info.py`, `_load_font`, wrap
`ImageFont.truetype` in a try/except that re-raises a clear error mentioning the
resolved path and the `general.truetype_font_dir` config key. Remove the
`# TODO Add error handling`. Verify by pointing the font dir at a nonexistent path
and asserting the raised message names the config key.

### Fix CODE-ANNO-3
In `src/nav/annotation/annotation_text_info.py`, remove the unused
`self._config = DEFAULT_CONFIG` assignment (and the `DEFAULT_CONFIG` import if it
becomes unused), or thread a real injected `Config` through the constructor to match
the rest of the annotation package. Verify ruff (unused import / attribute) and the
annotation tests pass.

### Fix CODE-UTIL-1
In `src/util/report_profile.py`, add a module docstring and a `main` docstring, accept
the profile path via `argparse` (default `./prof/combined.prof`), and delete the dead
commented `print_stats()`. Alternatively relocate the file under `src/experiments/`.
Verify `ruff check src/util` and `mypy src/util` are clean.

### Fix CODE-EXP-1 / CODE-EXP-2
Low priority (experiments are lint/mypy-excluded). If any of these scripts is promoted
toward the test suite, narrow the broad `except Exception` clauses to specific
exception types and de-duplicate the shared `gaussian_patch` helper and commented
dataset menus into a single shared module. No action required while they remain under
`src/experiments/`.


---

## Fix Prompts — Part 5 — Support (Shared Infrastructure)


### Fix CODE-SUPPORT-001
In `src/nav/support/flux.py`, the only live code is `clean_sclass` (lines 561-568).
Delete the ~1166 lines of commented-out CISSCAL/flux/star-photometry code and move
`clean_sclass` to the module that already owns star-spectral-class helpers (search
for other `sclass`/`spectral_class` utilities under `src/nav/`, likely a stars
support module). Update all imports of `nav.support.flux.clean_sclass` accordingly,
remove `'flux'` from `__all__` and the `flux` entry in `src/nav/support/__init__.py`
docstring, and delete `flux.py`. Verify: `grep -rn "support.flux\|support import flux"
src tests` returns nothing; `ruff check src` and `mypy src` pass; `pytest -m ""`
still green.

### Fix CODE-SUPPORT-002
In `src/nav/support/file.py`, make `clean_obj` non-mutating. In `_clean_dict` build
and return a new dict (`return {k: clean_obj(v) for k, v in obj.items()}`) instead of
writing back into `obj`; `_clean_list` already builds a new list (keep it). Confirm
with a unit test: `d = {'x': np.int64(3)}; out = clean_obj(d); assert d['x'] is ...`
(still `np.int64`) and `type(out['x']) is int`. Verify `dump_yaml`/`json_as_string`
output is unchanged and `pytest tests` for file helpers passes.

### Fix CODE-SUPPORT-003
In `src/nav/support/image.py` `next_power_of_2` (lines 173-186), add a guard:
raise `ValueError` for `n < 0`, and return `1` for `n == 0` (or raise, matching how
callers expect it). Keep the existing power-of-2 fast path. Add tests for `n in
{0, 1, 2, 3}`. Verify `next_power_of_2(0) == 1` and `pad_array_to_power_of_2` still
behaves for normal shapes.

### Fix CODE-SUPPORT-004
In `src/nav/support/image.py`, make `shift_array`, `pad_array`, and `unpad_array`
consistent about aliasing. Recommended: always return a fresh array — change the
zero-offset/zero-margin fast paths to `return array.copy()` (and for `unpad_array`
return `array[reversed_padding].copy()` so it is never a view). Alternatively, if
returning the input unchanged is intended for performance, document it explicitly in
each docstring's Returns section. Add tests asserting the chosen aliasing contract
for both the no-op and non-trivial paths. Verify drawing/padding callers do not rely
on the previous view behavior.

### Fix CODE-SUPPORT-005
In `src/nav/support/filters.py` `apply_filter` (lines 269-273), restrict the
null-sigma short-circuit to the blur-family kinds only. Change the condition so the
identity return fires when
`spec.kind in (ISOTROPIC_GAUSSIAN, ANISOTROPIC_GAUSSIAN, BANDPASS_DOG)` AND
`_largest_sigma(spec) < spec.null_filter_threshold_sigma`. Do NOT short-circuit
`GRADIENT_OF_GAUSSIAN`, `MORPH_DILATE`, or `DISTANCE_TRANSFORM` (let them run their
operation, which already self-handles trivial sizes). Add tests: a
`GRADIENT_OF_GAUSSIAN` spec with tiny sigma must return an edge-magnitude image (not
the raw array). Verify `pytest tests` for filters passes.

### Fix CODE-SUPPORT-006
In `src/nav/support/filters.py` `_apply_anisotropic_gaussian` (lines 166-168), either
remove the dead `out = out[: arr.shape[0], : arr.shape[1]]` slice (scipy
`rotate(reshape=False)` preserves shape) or replace it with a genuine center-crop/pad
to `arr.shape`. Add a test that the output shape equals the input shape for an
`align_axis` spec. Verify `pytest tests` for filters passes.

### Fix CODE-SUPPORT-007
In `src/nav/support/misc.py` `mad_std` (lines 115-119), guard empty/all-NaN input:
after `a_array = np.asarray(a)`, if `a_array.size == 0` raise `ValueError('mad_std
requires a non-empty array')`. Optionally compute the median over finite values only.
Update `estimate_image_noise_sigma` callers if needed. Add a test asserting
`mad_std([])` raises. Verify `pytest tests` (misc + noise_estimate) passes with no
RuntimeWarning.

### Fix CODE-SUPPORT-008
In `src/nav/support/image.py` `array_zoom` (line 268), change the annotation from
`result: np.ndarray` to the parameterized `NDArrayType[NPType]` already imported, so
the generic propagates from input to output. Verify `mypy src/nav/support/image.py`
passes and the function still returns the correct dtype.

### Fix CODE-SUPPORT-009
In `src/nav/support/misc.py`, narrow the excepts: in `current_git_version` catch
`(OSError, subprocess.CalledProcessError)`; in `get_local_host_name` catch `OSError`.
Consider not caching the failure sentinel permanently (only cache success), so a
transient failure can be retried. Keep the same return strings on failure. Verify
`ruff check src/nav/support/misc.py` (no broad-except warning) and existing tests
pass.

### Fix CODE-SUPPORT-010
In `src/nav/support/image.py` `draw_rect` docstring (lines 817-825), correct the
`yhalfwidth` description to "The height of the rectangle on each side of the center."
and reorder the parameter docs to match the signature order
(`color, xctr, yctr, xhalfwidth, yhalfwidth, thickness, dot_spacing`). No code
change. Verify `sphinx-build -W` still succeeds.

### Fix CODE-SUPPORT-011
In `src/nav/support/summary_png.py` `grayscale_to_rgb_with_quantile_stretch`
(lines 121-126), add a defensive clamp `clip_quantile = min(max(clip_quantile, 0.0),
1.0)` and a comment noting the `white <= black` guard (line 125) is required for
flat/tiny images. Add a test feeding a 1x1 and a constant-valued image and assert the
function returns a valid uint8 RGB array without raising. Verify `pytest tests` for
summary_png passes.

### Fix CODE-SUPPORT-012
In `src/nav/support/correlate.py` `navigate_single_scale_kpeaks` (near lines
558-564), validate the model/image size relationship up front: if the model
(post-gradient) is smaller than the image in either dimension, raise a clear
`ValueError` explaining that the model must be at least the image size in each axis
(matching the documented "padded larger than the image" contract), rather than
letting `crop_center` raise from inside `evaluate_candidate`. Add a test exercising a
too-small model. Verify `pytest tests/nav/.../correlate*` passes.

### Fix CODE-SUPPORT-013
In `src/nav/support/correlate.py` `navigate_with_pyramid_kpeaks` (after the docstring,
before the loop ~line 738), add `if pyramid_levels < 1: raise ValueError(...)`. Add a
test asserting `pyramid_levels=0` raises a clear error rather than IndexError. Verify
existing correlation tests still pass.

### Fix CODE-SUPPORT-014
In `src/nav/support/correlate.py`, promote the hard-coded thresholds
(`_NCC_BIDIR_W_FRAC_MIN`, `_NCC_BIDIR_VAR_FRAC_MIN`, the `at_edge` 2.0-pixel margin,
the degenerate `1e6`/`1e3` sentinels) to function parameters with their current
values as defaults, and have the `nav_technique/*` callers source them from `Config`
(add a correlation/offset config section if none fits). Keep behavior identical when
callers do not override. Verify all callers compile, `mypy src` passes, and
correlation tests are unchanged with defaults.

### Fix CODE-SUPPORT-015
In `src/nav/support/filters.py` `_apply_morph_dilate` (lines 204-214), honor per-axis
sigma: derive `half_v = int(np.ceil(spec.sigma_xy[0]))` and
`half_u = int(np.ceil(spec.sigma_xy[1]))`, build a `(2*half_v+1, 2*half_u+1)`
structuring size, and return `grey_dilation(arr, size=(size_v, size_u))` (early-return
`arr` when both half-widths are <= 0). Update the docstring to "rectangular
structuring element". Add a test with anisotropic `sigma_xy`. Verify filters tests
pass.

### Fix CODE-SUPPORT-016
In `src/nav/support/time.py` `utc_to_et` (lines 55-67), confirm against the installed
`julian` version whether the space-separated form parses. If it does not, either
normalize (`utc = utc.replace(' ', 'T', 1)` only between date and time) or fix the
docstring to list only the supported form. Add a test for each format string in the
docstring. Verify `pytest tests` for time helpers passes.

### Fix CODE-SUPPORT-017
In `src/nav/support/correlate.py`, resolve or relocate the embedded design-note
comments: (a) remove "TODO Clean this up" from the `navigate_with_pyramid_kpeaks`
docstring (line 668) and rewrite the docstring summary; (b) for the no-candidate path
returning `quality: -np.inf` (lines 617-631), either document that callers must treat
`-inf` quality / `1e6` covariance as "no result" or change the contract to a typed
sentinel; (c) decide on the prior-penalty unit concern (lines 477-484) and either fix
the scaling or move the note to an issue tracker. At minimum, convert inline prose
into concise docstring caveats. Verify `ruff`/`mypy`/tests pass.

### Fix CODE-SUPPORT-018
In `src/nav/support/__init__.py` (lines 34-36), once CODE-SUPPORT-001 lands, remove
the `flux` docstring entry and `'flux'` from `__all__`. If `flux.py` is retained for
any reason, rewrite the entry to describe current state without the time-anchored
"Legacy"/"kept for reference" framing per the project's documentation phrasing rule.
Verify `sphinx-build -W` and `pymarkdown`/doc checks pass.


---

## Fix Prompts — Part 6 — Navigation Orchestrator


### Fix CODE-ORCH-003
In `src/nav/nav_orchestrator/orchestrator.py::NavOrchestrator._make_context`, after
`raw_image = obs.extdata.astype('float64')`, sanitize missing-data markers before any finite-only
computation. Read `settings.marker_value`; if it is NaN, build `missing_mask = np.isnan(raw_image)`
and replace those pixels with a finite fill (e.g. `0.0`) for the *derivative/classifier* inputs while
recording `missing_frac` from `missing_mask` directly. Specifically: (a) compute `missing_mask`
correctly for both the `==marker` and `isnan(marker)` cases and thread the true missing fraction into
the classifier instead of relying on `sensor == marker`; (b) replace NaN pixels with `0.0` in the
array passed to `compute_all_image_derivatives` and `estimate_image_noise_sigma` so they never raise.
Mirror the fix in `image_classifier.py::NavImageClassifier.classify`: compute `miss_mask` as
`np.isnan(sensor)` when `missing_data_marker_dn` is NaN, and compute `max_dn` with `np.nanmax`
(guarding the all-NaN case). Verify: a CISS-CALIB-style `inst_config` (`data_units='calibrated_if'`,
`marker_value: NaN`) with an `extdata` containing `np.nan` pixels runs `navigate()` end-to-end and
returns a `NavResult` (never raises); `missing_frac` reflects the NaN fraction; the
`mostly_missing_data` short-circuit fires when NaN fraction exceeds `max_missing_frac_clean`. Add a
unit test feeding NaN pixels through both `classify` and `_make_context`.

### Fix CODE-ORCH-004
In `src/nav/nav_orchestrator/orchestrator.py::_make_context`, compute the saturation mask and the
absolute-DN classifier gates (`blank_max_dn`, `saturation_threshold_dn`) from `raw_image` (pre-filter),
not the post-filter `image`. Keep the *missing-data* fraction and the derivative inputs on the
post-filter `image`. Concretely: pass `raw_image` to `_build_saturation_mask`, and either run the
classifier twice-scoped (absolute-DN gates on `raw_image`, missing/derivative-aligned stats on
filtered `image`) or pass both arrays into a revised `classify` signature. Update the inline comment
that currently claims post-filter classification is correct for saturation. Verify: with
`source_image_filter.kind: BANDPASS_DOG` enabled for COISS NAC, a saturated test frame yields
`saturation_frac` matching the raw-DN saturation count (not ~0), and a bright-uniform frame is not
mis-classified as `blank`.

### Fix CODE-ORCH-005
This is primarily a config-calibration task surfaced by code. In
`src/nav/config_files/config_400_inst_coiss.yaml` replace the placeholder
`signal_dn_to_image_unit_scale: 5.0e-7` with the measured DN→I/F factor for CISS CALIB, and fix the
contradictory `signal_dn_to_image_unit_scale: 1.0  # raw_dn` entries under `data_units: calibrated_if`
in `config_410_inst_gossi.yaml`, `config_430_inst_vgiss.yaml`, `config_420_inst_nhlorri.yaml`
(either correct the value or the `data_units`). In
`src/nav/nav_orchestrator/instrument_config.py::_read_signal_scale`, add an upper-bound sanity check
or a `WARNING` log when a `calibrated_if` instrument's scale is exactly `1.0` (the raw-DN value),
since that is almost certainly an un-recalibrated copy. Verify: `predicted_snr` for a known star on a
CISS CALIB frame matches a hand-computed DN→I/F SNR; the warning fires for any calibrated camera left
at `1.0`.

### Fix CODE-ORCH-006
In `src/nav/nav_orchestrator/provenance.py`, memoize the process-invariant lookups: wrap
`_resolve_git_sha` and `_resolve_static_data_hashes` in `functools.lru_cache(maxsize=1)` (or cache
their results in module-level globals computed on first use). Leave `_resolve_spice_kernels` uncached
because the loaded-kernel set legitimately varies per image. `collect_provenance_metadata` then pays
the git subprocess + file-hash cost once per process and only the kernel scan per image. Verify:
profiling `navigate()` over N images shows `git` is invoked once total (not N times); the static-data
hash dict is identical across images; the SPICE kernel tuple still reflects per-image kernel changes.
Add a test asserting `_resolve_git_sha` runs the subprocess only once across repeated calls.

### Fix CODE-ORCH-007
In `src/nav/nav_orchestrator/nav_context.py::NavContext.with_prior`, make the exception classes match
the docstring: change the `len(offset_px) != 2` guard (currently `raise ValueError`) to `raise
TypeError`, OR update the docstring `Raises:` block to state `ValueError` for the length case. Prefer
aligning code to the documented `TypeError` for "not a length-2 sequence." Verify:
`with_prior(offset_px=(1, 2, 3), covariance_px2=np.eye(2))` raises the documented exception type; add a
`pytest.raises` test asserting the type and message.

### Fix CODE-ORCH-008
In `src/nav/nav_orchestrator/orchestrator.py::navigate`, gate the pass-2 prior installation on pass-1
status as well as offset presence. Change the condition at lines 461-465 to also require
`pass1_ensemble.status == 'ok'` before calling `context.with_prior(...)`; when the pass-1 ensemble is
`conflicted`, run pass-2 prior-free of the conflicted offset (use bare `context`) or log and skip
pass-2 entirely. Verify: with two well-separated pass-1 groups whose summed-confidence gap is below
`agreement_gap`, pass-2 receives no prior derived from the conflicted offset; add a test asserting the
pass-2 `NavContext.prior_offset_px is None` in that scenario.

### Fix CODE-ORCH-009
Create a single shared helper module (e.g. `src/nav/nav_technique/body_feature_ids.py` or extend an
existing support module) exposing `BODY_FEATURE_PREFIXES`, `source_bodies(feature_id) -> frozenset[str]`,
and `tier_of(technique_name) -> str`. Replace the duplicate `_BODY_FEATURE_PREFIXES` tuples and the
`_feature_source_bodies` / `_bodies_with_non_spurious_primary` parsing in `orchestrator.py` and the
`_BODY_FEATURE_PREFIXES` / `_source_bodies` / `_technique_tier` parsing in `ensemble.py` with calls to
the shared helper. Keep one O(name→tier) lookup (a dict built once from `NavTechnique._registry`)
instead of the per-call linear scans. Verify: adding a new body-feature prefix requires editing exactly
one tuple; existing orchestrator + ensemble tests still pass; the coverage set is computed via the
shared helper in both call sites.

### Fix CODE-ORCH-010
In `src/nav/nav_orchestrator/nav_result.py::NavResult.__post_init__`, add the symmetric conflicted
invariant: raise `ValueError` if `confidence_rank == 'conflicted'` and `status != 'conflicted'`, and
if `status == 'conflicted'` and `confidence_rank != 'conflicted'`. Verify:
`NavResult(status='ok', confidence_rank='conflicted', ...)` and
`NavResult(status='conflicted', confidence_rank='high', ...)` both raise; the canonical `.conflicted()`
constructor still succeeds. Add `pytest.raises` tests for both directions.

### Fix CODE-ORCH-011
In `src/nav/nav_orchestrator/ensemble.py`, distinguish the two rejection causes. Either change
`derive_confidence_rank` to return a richer result (e.g. an enum/`tuple[rank, reason]`) or, in
`ensemble` after `rank == 'failed'`, recompute whether the failure was confidence-driven
(`combined_confidence < lowest tier min_confidence`) versus sigma-driven and select a distinct
`NavStatusReason` for the sigma case (add e.g. `SIGMA_ABOVE_TIER_LIMIT` to `NavStatusReason` and a
matching `STATUS_REASON_INFO_TEMPLATE` entry in `status_reason_info.py`). Verify: a combined estimate
with high confidence but large sigma yields the sigma-specific reason, while genuinely low confidence
still yields `FINAL_CONFIDENCE_BELOW_THRESHOLD`; add tests covering both.

### Fix CODE-ORCH-012
In `src/nav/nav_orchestrator/nav_result.py::NavResult.__post_init__`, reject non-finite covariances:
after the squareness check, raise `ValueError` if `not np.isfinite(cov).all()` (matching the
finiteness guard already in `with_prior`). This makes a NaN covariance fail loudly upstream rather
than be laundered to `0.0` by the curator. Optionally, in `curator.py::_round_float`, log a WARNING
when collapsing NaN→0.0 instead of doing it silently. Verify: constructing a `NavResult` (or `.ok`)
with a NaN in `covariance_px2` raises; existing finite-covariance tests still pass.

### Fix CODE-ORCH-013
In `src/nav/nav_orchestrator/curator.py`, rename `_round_2x2` to `_round_matrix` (or
`_round_covariance`) and update its docstring to "Round an NxN covariance into a JSON-friendly nested
list (2x2 translation-only, 3x3 with rotation)." Optionally emit the rotation-aware matrix under a
clearly-dimensioned key or document in the schema that `covariance_px2` is 3x3 when `rotation_deg` is
present. Verify: curated JSON for a `fit_camera_rotation` result still contains the full 3x3 matrix;
no behavior change, only naming/docs. Run the curator tests.

### Fix CODE-DERIV-001
In `src/nav/nav_orchestrator/image_derivatives.py`, make the threshold robust to a zero noise sigma:
in `_build_edge_dt_from_gradients` clamp `threshold = edge_threshold_k_sigma * max(image_noise_sigma,
eps)` (or require `image_noise_sigma > 0` in `build_image_edge_dt` /
`compute_all_image_derivatives`, consistent with `_make_context`'s `cr_noise_sigma` clamp). Update the
module docstring's "always produces a fully-defined array" note to cover the zero-threshold degenerate
case, and soften the "one-pixel-wide edge map" claim given the `>=` plateau tie-break (or switch the
plateau comparison to a strict `>` on one side to enforce single-pixel ridges). Verify:
`compute_all_image_derivatives(image, image_noise_sigma=0.0)` no longer produces an all-candidate mask;
add a test with `image_noise_sigma=0.0` asserting a bounded edge count.

### Fix CODE-DERIV-002
In `src/nav/nav_orchestrator/image_derivatives.py`, correct the docstrings of the module header,
`build_image_edge_dt`, and `compute_all_image_derivatives` to say the noise sigma and gradient
threshold are in the **image's native units (DN for `raw_dn`, I/F for `calibrated_if`)**, matching the
`NavContext.image_noise_sigma` field documentation, rather than asserting "DN units." No code change.
Verify: doc text no longer claims DN where I/F is possible; `sphinx-build -W` still passes.

### Fix CODE-ORCH-014
In `src/nav/nav_orchestrator/orchestrator.py::_run_pass`, hoist
`available_types = {f.feature_type for f in features}` above the registry loop (it is loop-invariant).
Collapse the duplicated registry filtering: drive the loop from the already-computed `kept_names`/
predicate pass instead of re-checking `requires_prior`/`tier_filter`/`kept_names` inside the body.
Verify: behavior is identical (same techniques run on the same inputs) under existing `_run_pass`
tests; the per-image feature-type set is computed once.


---

## Fix Prompts — Part 7 — CLI Drivers & PyQt6 UI


### Fix CODE-MAIN-001 (async entry point)

> **STATUS: FIXED (uncommitted, 2026-06-10).** Both `_cloud_tasks` drivers now expose a synchronous `main()` that calls `asyncio.run(async_main())`; the async body was renamed to `async_main`. Verified `main` is no longer a coroutine function.

In `src/main/nav_backplanes_cloud_tasks.py`, rename the existing `async def main()` (line 101) to `async def async_main()`, and add a synchronous wrapper:
```python
def main() -> None:  # Required for setuptools entry points
    asyncio.run(async_main())
```
Update the `if __name__ == '__main__':` guard (line 132) to call `main()` instead of `asyncio.run(main())`. Apply the identical change to `src/main/nav_create_bundle_cloud_tasks.py` (rename `async def main` at line 109 to `async_main`, add sync `main`, fix the guard at line 146). Verify: `python -c "import main.nav_backplanes_cloud_tasks as m; import inspect; assert not inspect.iscoroutinefunction(m.main)"` for both modules, and confirm the pattern now matches `nav_offset_cloud_tasks.py:147` / `nav_mosaic_cloud_tasks.py:340`.

### Fix CODE-MAIN-002
In `src/main/nav_create_simulated_image.py` `_load_parameters` (lines 2399-2419): (a) add `'shade_solid_rings': bool(params.get('shade_solid_rings', False)),` to the reconstructed `self.sim_params` dict; (b) change `'closest_planet': params.get('closest_planet'),` to `'closest_planet': params.get('closest_planet') or 'SATURN',` (or keep `None` but guard the combo update). After the dict is built, sync the checkbox:
```python
self._shade_solid_rings_check.blockSignals(True)
self._shade_solid_rings_check.setChecked(bool(self.sim_params['shade_solid_rings']))
self._shade_solid_rings_check.blockSignals(False)
```
At the closest-planet combo update (lines 2430-2435), guard against `None`:
```python
closest_planet = self.sim_params.get('closest_planet') or 'SATURN'
index = self._closest_planet_combo.findText(closest_planet)
```
Verify: save a model with `shade_solid_rings: true`, load it, confirm the checkbox is checked and `self.sim_params['shade_solid_rings']` is `True`; load a JSON with no `closest_planet` key and confirm no exception/dialog.

### Fix CODE-MAIN-003
In `src/main/nav_create_simulated_image.py`, narrow the broad excepts: `_update_image` (2145) — catch the specific render exceptions `render_combined_model` can raise and log via a logger before the dialog; `_save_image`/`_save_parameters` — catch `(OSError, ValueError)`; `_load_parameters` — catch `(OSError, ValueError, json.JSONDecodeError, TypeError, KeyError)` and include `repr(e)`/traceback in the logged message. In `src/main/nav_backplane_viewer.py:814-818`, replace `except Exception: pass` with a guarded build of `default_name` that does not need a try/except (the attribute is always set). Verify: a malformed params JSON shows a dialog AND logs a traceback; ruff `B` does not flag a bare broad except.

### Fix CODE-MAIN-004
In `src/main/nav_mosaic.py:553-558`, remove the stdlib-`logging`-root manipulation. Route `--log-level` through the same `setup_logging` / config path the other drivers use, or at minimum drop `logging.getLogger().setLevel(...)` (which affects only the stdlib root, not pdslogger) and set only the pdslogger levels via the documented `general.log_level_main_*` config keys. If `--log-level` must stay, rename it to the `--log-level-main-*` family for consistency with `nav_offset.py`. Remove the now-unused `import logging` if no other use remains (the `logging.Handler` type hint at line 77 still needs it — keep the import but drop the root-logger calls). Verify: `nav_mosaic rings <ds> --log-level DEBUG` changes MAIN_LOGGER verbosity through the documented mechanism, and the stdlib root logger level is untouched.

### Fix CODE-MAIN-005
In `src/main/nav_create_bundle.py:208-216` (`main_summary`) and `src/main/nav_backplane_viewer.py:1493-1501` (`main`), replace the hand-rolled `DEFAULT_CONFIG.read_config()` + config-file loop + `nav_default_config.yaml` fallback with a single call to `load_default_and_user_config(arguments, DEFAULT_CONFIG)` (already imported elsewhere in the package). Verify: both still honor `--config-file` and fall back to `./nav_default_config.yaml`, and behavior matches `main_labels`.

### Fix CODE-MAIN-006
Extract the shared driver preamble into a helper in a small `src/main/_driver_common.py` (or reuse an existing support module): a function that validates `dataset_name`, prints usage + valid datasets on error with `sys.exit(1)`, instantiates the `DataSet`, and adds the common `--config-file` / `--pds3-holdings-root` / `--nav-results-root` arg group. Call it from `nav_offset`, `nav_backplanes`, `nav_consolidate_metadata`, `nav_create_bundle`, `nav_backplane_viewer`. Separately, extract the cloud_tasks per-file `ImageFile` builder + missing-field error returns into one helper used by all four `process_task` functions, and decide whether `extra_params` should be threaded through `nav_backplanes_cloud_tasks` / `nav_create_bundle_cloud_tasks` (it currently is in `nav_offset_cloud_tasks` only — make it consistent). Verify: all drivers still parse args identically; add a unit test that the shared file-builder rejects a task missing `image_file_url`.

### Fix CODE-MAIN-007
Decide the batch-size contract for `nav_offset`. If datasets are always single-file for offset, drop the multi-file `task_files` loop (lines 404-412) in favor of the single file and assert `len(imagefiles.image_files) == 1` in the cloud-export path too, matching the local-run assert (line 430). If multi-file batches are intended, remove the local-run assert and make `task_id` deterministic across all files in the batch (e.g. include a stable hash of all label names), not just `[0]`. Verify: the cloud-export and local-run paths agree on cardinality; add a test asserting the chosen contract.

### Fix CODE-UI-001
In `src/main/nav_backplane_viewer.py`, replace the inlined summary-overlay blend (1207-1225) and BODY_ID blend (1227-1274) in `_compose_and_display` with calls to the existing `_alpha_blend_layer` helper and `_load_colormap`, exactly as `_render_full_rgba` (827-912) already does — ideally factor the common compositing of (image base, summary, BODY_ID, body, ring) into one method called by both `_compose_and_display` and `_render_full_rgba`. Verify: on-screen composite and saved PNG are pixel-identical for the same controls (compare `_render_full_rgba()` output to a grab of the displayed `_last_rgba`).

### Fix CODE-UI-002
In `src/main/nav_backplane_viewer.py:1349-1367` `_update_cursor_status`, map the cursor through the same coordinate model the renderer uses. Because `_compose_and_display` resizes `self._label` to the scaled pixmap, `pos` is already in scaled-pixmap space, so `u = pos.x()/zoom`, `v = pos.y()/zoom` is correct ONLY if `pos` is label-relative (it is, since the event comes from `_ImageLabel`). Add an explicit assertion/comment that `pos` is label-relative, OR (preferred) verify-then-fix: zoom to 4x, pan, and confirm the reported value matches a known pixel; if it is off by the scroll offset, subtract the label's position within the viewport. Verify: cursor readout tracks the correct pixel at 4x after panning to each corner.

### Fix CODE-UI-003 / CODE-UI-004
Delete the redundant local `new_zoom` clamp + early-return in `_zoom_at_point` of `nav_create_simulated_image.py` (555-570), `nav_backplane_viewer.py` (1172-1187), and `manual_nav_dialog.py` (1006-1016) — let `ZoomPanController.zoom_at_point` own the clamp (have it return whether the zoom changed if the early-return is needed). Replace the per-window `_zoom_in`/`_zoom_out` centre-anchored implementations with calls to `ZoomPanController.zoom_in_center()` / `zoom_out_center()` (common.py:101-127). Centralize the clamp constants `(0.1, 50.0)` in `common.py` as module constants. Verify: wheel and button zoom still anchor correctly; the clamp limit lives in exactly one place.

### Fix CODE-UI-005
Move `_SyncedSlider` and `_ZoomSync` into `src/nav/ui/common.py` (or `mosaic_viewer/common.py`) as a single shared definition. Import them in both `body_window.py` and `ring_window.py`; delete the duplicated `_SyncedSlider` (ring_window:200, body_window:133) and the inner-class `_ZoomSync` in `ring_window._make_zoom_sync` (837-842). Verify: both windows' zoom sliders behave identically; no per-call class creation remains (`_make_zoom_sync` references the module-level `_ZoomSync`).

### Fix CODE-UI-006
Replace `logging.getLogger(__name__)` in `mosaic_viewer/common.py`, `body_window.py`, and `ring_window.py` with the project `pdslogger` (e.g. `from nav.config.logger import IMAGE_LOGGER` or the appropriate `MAIN_LOGGER`), matching how the rest of `nav/` logs. Update the `logger.exception(...)`/`logger.debug(...)` call sites accordingly. Verify: a `load_ring_file`/`load_body_file` failure traceback appears in the configured nav log stream, and no stdlib `logging` import remains in these three modules.

### Fix CODE-UI-007
Annotate `__all__: list[str] = []` in `src/nav/ui/__init__.py:20` and `src/nav/ui/mosaic_viewer/__init__.py:11`. Promote `_slider_to_zoom` / `_zoom_to_slider` in `tiled_image_widget.py` to public names (`slider_to_zoom` / `zoom_to_slider`) or move them to `mosaic_viewer/common.py`, and update the imports in `body_window.py:44-48` and `ring_window.py:51-55`. Verify: mypy-strict passes; no cross-module import of a `_`-prefixed name remains.

### Fix CODE-UI-008
In `src/nav/ui/manual_nav_dialog.py:1145-1159` and `src/main/nav_backplane_viewer.py:1519-1538`, do not call `app.quit()` after `exec()` returns (it is a no-op for cleanup). If the goal is to release the app for repeated in-process use, either keep the app and reuse it, or schedule deletion explicitly. At minimum, document that `run_modal` reuses an existing `QApplication.instance()` and only the outermost caller owns lifecycle. Verify: call `run_manual_nav` twice in one pytest process (with `pytest -p no:cacheprovider`) and confirm no crash and a single shared `QApplication.instance()`.

### Fix CODE-UI-009
In `src/nav/ui/mosaic_viewer/histogram_stretch.py:260-265`, when both indicators map to the same `x` (coincident), allow grabbing white via a modifier or by alternating on repeated clicks, or have `set_values`/`set_range` enforce a minimum separation (`white >= black + epsilon`) the same way `set_data` (line 117) already does. Verify: load a flat (constant-valued) image, confirm both indicators are independently draggable.


---

## Fix Prompts — Part 8 — Model/Technique Non-Math Coverage Gaps


### Fix CODE-MODEL-001
In `src/nav/nav_model/nav_model_titan.py`, decide the intended behavior
for the Titan stub. If Titan navigation is genuinely deferred and the
class should not appear in the model inventory, add `_abstract = True` to
the `NavModelTitan` class body so `NavModel.__init_subclass__` skips
registration, and update the module docstring to say "not registered until
the haze-aware algorithm lands". If instead a no-op Titan model *should*
be constructed when Titan is in the FOV, implement
`@classmethod def instances_for_obs(cls, obs)` returning
`[cls('titan', obs)]` when Titan is present (mirroring
`NavModelStars.instances_for_obs`). Verify: run `build_models_for_obs` on
a Titan-in-FOV obs fixture and assert the resulting list either contains
no `NavModelTitan` (abstract route) or exactly one (instances route), per
the chosen design. Run `pytest tests/nav/nav_model -k titan`.

### Fix CODE-MODEL-002
In `src/nav/nav_model/nav_model_rings_simulated.py` `to_features`
(lines 174-180), replace the `1 + ... - 1` arithmetic with the plain edge
count and reconcile semantics with the catalog path. Change
`constituent_edge_count` to
`int(self._ring_feature.inner_edge is not None) + int(self._ring_feature.outer_edge is not None)`
and add a comment noting that the simulated path counts *edges of one
ring* whereas `NavModelRings` counts *fused rings*; if the downstream
confidence formula requires the latter meaning, set it to `1` here
instead. Verify the value matches the documented meaning of
`RingAnnulusFlags.constituent_edge_count` and run
`pytest tests/nav/nav_model -k simulated_ring`.

### Fix CODE-MODEL-003
Remove the backwards-compat alias in `src/nav/nav_model/body_shape.py`.
Pick one canonical name: rename `load_body_shape` to `shape_for_body`
(the name actually called by `nav_model_body.py`) and delete the separate
`shape_for_body` wrapper, OR keep `load_body_shape` and update
`src/nav/nav_model/nav_model_body.py` (import at line 72, call at line 608)
to use it, then delete `shape_for_body`. Update `__all__` and grep
`shape_for_body` / `load_body_shape` across `src/` and `tests/` to fix
every reference. Verify `mypy src` and `pytest tests/nav/nav_model -k
body_shape`.

### Fix CODE-MODEL-004
In `src/nav/nav_model/body_shape.py` `_yaml_entry_for` (lines 245-248),
narrow `except Exception:` to the specific exception that
`Config.body_shape` raises during early bootstrap (inspect the
`Config.body_shape` property; likely `AttributeError`). Replace
`except Exception: return None` with `except AttributeError: return None`,
or, if a broader catch is genuinely needed, log at DEBUG before returning
`None` so a real Config error is not silently masked. Verify `pytest
tests/nav/nav_model -k body_shape` and that a deliberately broken
`config.body_shape` now raises rather than silently falling back.

### Fix CODE-MODEL-005
In `src/nav/nav_model/stars/catalog.py` `_merge_catalogs` (lines 559-560),
make the name-upgrade guard and assignment consistent. Change the
condition to test the field actually assigned, e.g.
`if (not prev.name) and star.name and star.pretty_name:` and assign
`prev.pretty_name = star.pretty_name`; or better, only upgrade when
`star.pretty_name` is not the bare unique-number fallback. Update the
docstring to match. Verify with a unit test where the earlier star has an
empty `name` and the later star has both `name` and a human-readable
`pretty_name`; run `pytest tests/nav/nav_model -k catalog`.

### Fix CODE-MODEL-006
In `src/nav/nav_model/stars/nav_model_stars.py` `_emit_features`
(lines 237-238), guard the `conflicts` access like `create_model` does:
`conflicts = star.conflicts or ''; in_body = conflicts.startswith('BODY');
in_ring = conflicts.startswith('RING')`. Alternatively, narrow
`MutableStar.conflicts` to a non-optional `str` and assert it after
`reduce_catalogs`. Verify `mypy src` and `pytest tests/nav/nav_model -k
stars`.

### Fix CODE-MODEL-007
In `src/nav/nav_model/stars/conflicts.py` `_check_one_star`
(lines 167-177), replace the median-radius ring-occlusion test with an
any-pixel-inside-annulus test to match the body `where_intercepted(...)`
semantics. Compute the per-pixel `ring_radius` array over the window
(`bp_radii.mvals`), and for each annulus mark a conflict when any unmasked
window pixel's radius falls in `[inner_km, outer_km]` (e.g.
`np.any((radii >= inner) & (radii <= outer))` over unmasked entries).
Keep the body-intercept short-circuit. Verify with a window straddling an
annulus boundary; run `pytest tests/nav/nav_model -k conflict`.

### Fix CODE-MODEL-008
In `src/nav/nav_model/nav_model_rings.py`, move the reliability magic
numbers (`RING_EDGE_DEFAULT_RELIABILITY` 0.7, the straight-line `0.7`
multiplier, `RING_EDGE_SIGMA_ALONG_PX` 0.5, and the `5.0` / `50.0`
constants in `_ring_annulus_reliability`) into config (e.g. a
`rings.reliability` block or `feature_emission.ring_*` keys in
`config_510_techniques.yaml`), read them through `self._config` in
`to_features`, and pass them into the `_ring_edge_reliability` /
`_ring_annulus_reliability` helpers as parameters. Keep the existing
values as the shipped defaults and mark each as PLACEHOLDER pending
Phase-5 calibration in the YAML comments. Verify config loads and
`pytest tests/nav/nav_model -k rings`.

### Fix CODE-MODEL-009
Decide whether per-run config should govern model selection. If yes, add a
`config: Config | None = None` parameter to `build_models_for_obs`
(`src/nav/nav_model/nav_model.py`) and to each
`instances_for_obs(cls, obs, *, config=None)`, defaulting to
`DEFAULT_CONFIG`, and thread it through `NavModelRings.instances_for_obs`
(lines 230-234) and `NavModelBody.instances_for_obs` (lines 229-233).
Update all `build_models_for_obs` call sites. If no, add a one-line
docstring note on both `instances_for_obs` methods stating that model
*selection* always uses `DEFAULT_CONFIG` and only rendering honors a
per-instance override. Verify `pytest tests/nav/nav_model`.

### Fix CODE-MODEL-010
Split `src/nav/nav_model/nav_model_body.py` (1118 lines) under the
1000-line cap. Create a `src/nav/nav_model/body/` subpackage (paralleling
`rings/` and `stars/`) and move the free helper functions (limb/terminator
polyline extraction, the `_build_*_feature` emission builders, and any
pure-geometry helpers) into a `body/extraction.py` / `body/emission.py`
module, re-exporting from `nav_model_body.py` or importing them back. Keep
the `NavModelBody` class and its registry hook in place. Verify
`wc -l src/nav/nav_model/body/*.py src/nav/nav_model/nav_model_body.py`
all under 1000, then `mypy src` and `pytest tests/nav/nav_model -k body`.

### Fix CODE-TECH-001
Extract a single validated upsample-factor helper and use it in both NCC
techniques. Add (e.g.) `parse_upsample_factor(config) -> int` to
`src/nav/nav_technique/nav_technique.py` containing the validation
currently in `RingAnnulusNav._upsample_factor`
(`src/nav/nav_technique/nav_technique_ring_annulus.py` lines 280-305:
reject non-real/bool, coerce to int, require `1 <= x <= _MAX_UPSAMPLE_FACTOR`).
Replace `BodyDiscCorrelateNav._upsample_factor`
(`src/nav/nav_technique/nav_technique_body_disc.py` lines 753-758) and
`RingAnnulusNav._upsample_factor` to delegate to it. Verify the body-disc
path raises `ValueError` on `correlation_fft_upsample_factor=2_000_000` or
a non-numeric value; run
`pytest tests/nav/nav_technique -k upsample`.

### Fix CODE-TECH-002
In `src/nav/nav_technique/__init__.py`, add `ManualNavDiagnostics` to the
`from nav.nav_technique.diagnostics import (...)` block (lines 27-38) and
to `__all__` (lines 53-82), keeping alphabetical order. Verify
`python -c "from nav.nav_technique import ManualNavDiagnostics"` and
`ruff check src/nav/nav_technique/__init__.py`.

### Fix CODE-TECH-003
In `src/nav/nav_technique/nav_technique_manual.py` `__init__` (lines
86-89), reword the `annotations` comment to describe the current contract
without "backwards compatibility". For example: "Optional; only the
offset-pick path needs it. `run_manual_nav` always supplies the
merged-per-NavModel `Annotations` used for the labelled summary PNG."
Verify `pymarkdown`/`ruff` clean and no test references the old wording.

### Fix CODE-TECH-004
In `src/nav/nav_technique/nav_technique.py` `search_window_for_obs`
(line 128), determine whether `ObsSnapshot` (or the `NavContext.obs`
annotation) declares `extfov_margin_vu`. If it should, add the attribute
to the obs type/protocol and remove the `# type: ignore[attr-defined]`.
If the attribute is genuinely optional, type `context.obs` against a
protocol that declares it `tuple[int, int]`. Keep the documented
AttributeError-on-stub behavior. Verify `mypy src tests` passes with the
ignore removed.

### Fix CODE-TECH-005
De-duplicate the copied helpers. Move `_build_polyline_mask` (identical in
`nav_technique_body_limb.py` 51-61, `nav_technique_body_terminator.py`
56-66, `nav_technique_ring_edge.py` 60-70) into
`src/nav/nav_technique/dt_fitting.py` as a public `build_polyline_mask`
and import it in all three techniques. Move `_peak_to_runner_up_ratio`
(identical in `nav_technique_body_disc.py` 204-224 and
`nav_technique_ring_annulus.py` 82-102) into
`src/nav/nav_technique/nav_technique.py` and import it in both. Delete the
local copies. Verify `pytest tests/nav/nav_technique` and `mypy src`.

### Fix CODE-TECH-006
In `src/nav/nav_technique/nav_technique_ring_edge.py` (lines 327-329), add
the same WARNING the body techniques emit before truncating a non-(2,2)
covariance: `self.logger.warning('RingEdgeNav: lm_subpixel_refine returned
%s covariance with fit_rotation=False; truncating to (2, 2)',
covariance.shape)`. Better, fold the truncate-and-warn into a shared
helper introduced in CODE-TECH-005 and call it from all three DT
techniques. Verify `pytest tests/nav/nav_technique -k ring_edge`.

### Fix CODE-TECH-007
Resolve the vertex-count-vs-arc-length conflation in
`src/nav/nav_technique/nav_technique_body_limb.py` and
`nav_technique_body_terminator.py`. Either (a) rename the tunable
`min_arc_px` -> `min_arc_vertices` in `config_510_techniques.yaml` and in
both techniques' `__init__`/`is_feasible`/`navigate`, and rename the
diagnostic `visible_arc_px` accordingly, documenting the "1 vertex per
pixel" model invariant; or (b) compute true arc length from
`np.sum(np.hypot(diff(vertices_vu, axis=0)))` and gate on that. Confirm
the limb-mask vertex spacing in
`nav_model_body` is 1 px/vertex before choosing (a). Verify
`pytest tests/nav/nav_technique -k "limb or terminator"`.

### Fix CODE-TECH-008
In `src/nav/nav_technique/technique_result.py` `__post_init__`, add
finiteness validation: check `offset_px` is a length-2 tuple of finite
floats (`raise ValueError` otherwise), and that `rotation_rad` /
`sigma_rotation_rad`, when not `None`, are finite. Place the checks
alongside the existing confidence-range check (lines 64-65). Add a unit
test passing `offset_px=(float('nan'), 0.0)` and asserting `ValueError`.
Verify `pytest tests/nav/nav_technique -k technique_result`.

### Fix CODE-TECH-009
When consolidating `_peak_to_runner_up_ratio` per CODE-TECH-005, replace
the `runner_q <= 1e-9` branch's `max(winner_q, 0.0) / 1e-9` divide-by-
sentinel (in `nav_technique_body_disc.py` 222-223 and the ring-annulus
twin) with an explicit saturating return (e.g. a named
`_UNAMBIGUOUS_PEAK_RATIO = 1.0e6` constant) so the value is scale-
independent and not driven by absolute NCC quality. Document it. Cross-
check the `peak_to_runner_up_ratio` term's `divisor`/`cap_at` in
`config_510_techniques.yaml` to confirm the downstream sigmoid saturates.
Verify `pytest tests/nav/nav_technique -k runner_up`.

### Fix CODE-TECH-010
In `src/nav/nav_technique/nav_technique.py`, change
`log_confidence_breakdown`'s `logger: Any` parameter (line 156) to
`logger: pdslogger.PdsLogger` (add `import pdslogger` or
`from pdslogger import PdsLogger` to the import group). Verify `mypy src`
and `pytest tests/nav/nav_technique -k confidence`.

### Fix CODE-TECH-011
In `src/nav/nav_technique/nav_technique.py`, introduce a dedicated
registry opt-out flag distinct from abstractness: add
`_auto_register: ClassVar[bool] = True` on the `NavTechnique` base and
change `__init_subclass__` to append only when
`cls.__dict__.get('_auto_register', True)` (keep `_abstract` meaning
"true ABC base, not instantiable"). In
`src/nav/nav_technique/nav_technique_manual.py`, replace
`_abstract = True` with `_auto_register = False`. Verify the registry no
longer lists `NavTechniqueManual` and `pytest tests/nav/nav_technique`
passes. (Low priority / optional refactor.)
