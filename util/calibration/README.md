# Confidence calibration tooling (sim-anchored)

Offline tooling that fits the per-technique confidence formulas
(`config_510_techniques.yaml`) and the orchestrator acceptance gates
(`config_540_orchestrator.yaml`).  The current fit is **sim-anchored**:
the anchors are recovery errors against planted truth on randomized
simulated scenes, because no real-image anchor set (operator-verified
offsets with measured per-technique errors) exists yet.  When such a set
exists, the same fit reruns against it and the sim-anchored values
become the fallback for regimes real data cannot reach.  (Historical
background and sequencing: `plans/VALIDATION_AND_CALIBRATION_PLAN.md`.)

Everything here is a repo-checkout script (not part of the distributed
package); generated artifacts go under `_work/calibration/` (gitignored).

## Pipeline

1. **`scene_gen.py`** — seeded randomized sim_params per technique family
   (disc / limb / terminator / blob / ring / star_field / star_unique),
   spanning each technique's regime from clean through the failure cliff:
   noise, feature size/count/brightness, and the controlled model-error
   axes (mesh-vs-ellipsoid shape mismatch, predicted-pose error).
2. **`collect.py`** — navigates every scene with the full autonomous
   ensemble in-process (no external holdings needed) and writes one JSONL
   row per frame: planted truth, the fused result, and every
   per-technique result with its full diagnostics vector.

   ```bash
   venv/bin/python util/calibration/collect.py \
       --per-family 600 --workers 14 --out _work/calibration/rows_v1.jsonl
   ```

3. **`fit.py`** — refits each technique's sigmoid alphas
   (`config_510_techniques.yaml`) as an L2-regularized logistic regression
   of "offset error <= 1 px" on the YAML-normalized diagnostics terms
   (Platt-scaling variant; term offset/divisor/cap transforms unchanged).
   Rows where a hard gate fired (`spurious`/`at_edge`) are excluded — the
   gates zero confidence regardless of alphas.  Emits a JSON proposal and
   a Markdown report (reliability tables, AUC/Brier before vs after).

   ```bash
   venv/bin/python util/calibration/fit.py _work/calibration/rows_v1.jsonl \
       --out-json _work/calibration/fit_v1.json \
       --out-report _work/calibration/fit_v1.md
   ```

4. Write the fitted alphas into `config_510_techniques.yaml` (by hand, so
   the YAML comments stay curated), then **re-collect** — fused
   confidences depend on the per-technique alphas.

   What a re-collect can and cannot verify: scene draws are
   seed-deterministic and a pass-1 technique's diagnostics and errors do
   not depend on any confidence formula, so pass-over-pass alpha
   reproduction is *structural* for the pass-1 techniques — reproducing
   them confirms only that the pipeline is deterministic.  The
   substantive convergence content of a re-collect is the
   prior-dependent pass-2 technique (`StarRefineNav`: the pass-1
   formulas change which priors reach pass 2, so its cohort genuinely
   re-forms) and the fused quantities the next two steps check (the
   floors re-solving to ~0 additional, the gate curves re-deriving to
   the shipped boundaries).
5. **`fit_floors.py`** — solves each technique's `model_error_floor_px`
   tuning value: the quadrature floor that brings the 2-sigma
   coverage of `sqrt(sigma_reported^2 + floor^2)` to the 2D-Gaussian
   expectation (0.865) against planted truth.  Run it on a collection
   pass made with the floors at their current values: a converged
   configuration solves to ~0 additional floor for every technique.

   ```bash
   venv/bin/python util/calibration/fit_floors.py _work/calibration/rows_v5.jsonl
   ```

6. **`fit_gates.py`** — derives the orchestrator acceptance parameters
   (`config_540_orchestrator.yaml`) from the pass-2 fused rows: tier
   `min_confidence` boundaries (the smallest confidence at which each
   tier's sigma-gated subset achieves a 0.9 success rate against the
   tier's error budget) and the final `min_confidence` gate, plus the
   per-technique sigma coverage check.

   ```bash
   venv/bin/python util/calibration/fit_gates.py _work/calibration/rows_v2.jsonl \
       --out-report _work/calibration/gates_v2.md
   ```

7. **`library_crosscheck.py`** — plausibility cross-check:
   runs the calibrated pipeline over every operator-curated sidecar
   (needs the local-holdings environment) and reports status / tier /
   offset / primary-technique agreement independently per image, plus a
   tier confusion table.  The operator tiers are never fit targets;
   wholesale disagreement here means the labels or the calibration need
   a second look.

   ```bash
   venv/bin/python util/calibration/library_crosscheck.py \
       --workers 8 --out _work/calibration/library_crosscheck.md
   ```

   The seed-20260718 campaign's cross-check record is tracked in
   `CAMPAIGN_20260718.md` (this directory): 75 sidecars -- status
   69/75, tier 46/75, offset-within-slack 54/61, zero pipeline
   exceptions -- with per-frame attribution for every flip (the
   dominant confusions are high->medium under the 0.85 high-tier
   boundary and medium->low under the 2.61 px limb floor), the
   W1444747627 single-frame diagnosis, and the CI-gate consequence:
   the sidecar tier expectations predate the recalibration, the
   historical failure-set gate no longer applies as-is, and a sidecar
   re-ratchet is an operator decision.

## Campaign timing baseline

Reference throughput for the collection campaign, measured 2026-07-18 on
the full-truth-axis renderer (campaign seed 20260718; body families
drawing the surface / photometric truth axes and the giant-planet
disc-texture slice):

```bash
source setup.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python util/calibration/collect.py \
    --per-family 600 --workers 14 --out _work/calibration/rows.jsonl
```

Result: 4200 rows, 0 errors, elapsed (real) **7m20-24s** across three
passes (user ~94-95m, sys ~8m).  Machine: i9-13900K with logical CPUs
10-11 excluded by `setup.sh`.  The previous renderer measured 7m13.7s
(2026-07-15) with the same command, so the truth axes cost ~1.5% --
well inside the 2x budget below.

Notes on reproducing the measurement:

- The shell-level `*_NUM_THREADS=1` exports are **required**: `collect.py`
  sets the same variables inside each worker, but the workers inherit the
  parent's already-initialized BLAS thread pools under the fork start
  method, so the in-worker pinning does not take effect.  Unpinned
  BLAS threads oversubscribe the 14 workers and distort the timing.
- Worker CPU affinity on this machine is load-bearing, not cosmetic:
  always `source setup.sh` first so the excluded cores stay excluded.

This baseline is the renderer-throughput budget: a default-stage
4200-scene campaign must stay within **2x this elapsed time** as the
renderer gains fidelity.  Re-measure and update this section whenever the
campaign command, the machine, or the renderer's default stage set
changes materially.

## Structural caps and hard gates

Post-sigmoid caps (BodyBlobNav's 0.4, StarUniqueMatchNav's per-mode
0.7/0.8, StarRefineNav's single-inlier 0.5) encode cross-technique trust
ordering, not per-technique reliability; they are retained, not fitted.
`hard_zero_if` gates likewise stay.

## Caveats

- **Sim-anchored basis.** Every value fitted here is only as real as
  the simulator's match to real images -- quantified per instrument in
  the simulator report's realism-match section, but not yet
  real-anchored.  `confidence_provisional` stays true
  in the metadata until a real-anchored calibration lands.
- The scene families cover the sim's rendering vocabulary; regimes the
  sim cannot render (real PSF wings, saturation bloom on stars,
  calibrated-I/F detector noise) are uncalibrated by this fit.
- The operator-curated image-library tiers are the *plausibility
  cross-check* for this calibration, never fit targets (the curation
  conventions live in `docs/dev_guide/dev_guide_image_library.rst`).
