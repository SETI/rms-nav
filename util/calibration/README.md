# WS-5 confidence calibration tooling (sim-anchored)

Offline tooling for the confidence-calibration workstream (WS-5 of
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`, issue #173), in its
**sim-anchored** regime: the anchors are recovery errors against planted
truth on randomized simulated scenes (WS-2's instrument), because the
real-data anchors (WS-1 per-technique covariance) do not exist yet.  When
WS-1 lands, the same fit reruns against the real anchors and the
sim-anchored values become the fallback for regimes WS-1 cannot reach.

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
5. **`fit_gates.py`** — derives the orchestrator acceptance parameters
   (`config_540_orchestrator.yaml`) from the pass-2 fused rows: tier
   `min_confidence` boundaries at the WS-5 error-percentile targets and
   the final `min_confidence` gate, plus the per-technique sigma coverage
   check.

   ```bash
   venv/bin/python util/calibration/fit_gates.py _work/calibration/rows_v2.jsonl \
       --out-report _work/calibration/gates_v2.md
   ```

## Structural caps and hard gates

Post-sigmoid caps (BodyBlobNav's 0.4, StarUniqueMatchNav's per-mode
0.7/0.8, StarRefineNav's single-inlier 0.5) encode cross-technique trust
ordering, not per-technique reliability; they are retained, not fitted.
`hard_zero_if` gates likewise stay.

## Caveats

- **Sim-anchored basis.** Every value fitted here carries WS-5's
  "sim-anchored" label: it is only as real as the WS-2 realism match,
  which has not been quantified yet.  `confidence_provisional` stays true
  in the metadata until a real-anchored calibration lands.
- The scene families cover the sim's rendering vocabulary; regimes the
  sim cannot render (real PSF wings, saturation bloom on stars,
  calibrated-I/F detector noise) are uncalibrated by this fit.
- The operator-curated image-library tiers are the *plausibility
  cross-check* for this calibration (PHASE10_CURATION), never fit
  targets.
