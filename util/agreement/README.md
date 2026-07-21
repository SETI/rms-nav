# Cross-technique agreement estimator (validation tooling)

Offline tooling that validates the covariance-components estimator the
cross-technique agreement study relies on, on truth-known simulated
scenes (issue #224; the agreement-estimator validation stage of
`plans/VALIDATION_AND_CALIBRATION_PLAN.md`).
The estimator separates per-technique 2x2 error covariances from nothing
but the techniques' pairwise disagreements; before any real-image
per-technique number is trusted, this campaign proves on planted-truth
scenes (a) where that solve is identifiable at all, per scene
composition, and (b) whether the compared techniques stay
bias-independent through their shared preprocessing layer.

Everything here is a repo-checkout script (not part of the distributed
package); generated artifacts go under `_work/agreement/` (gitignored).
The dated campaign record (`CAMPAIGN_20260719.md`, this directory)
preserves the numbers and conclusions that outlive them.

## Pieces

1. **`estimator.py`** — the covariance-components solve itself, the
   component the agreement study consumes.  Truth-free by construction:
   input is per-frame technique offsets plus geometry angles, output is
   per-technique covariance matrices *with an identifiability report*
   (singular spectrum, null space mapped to parameter names,
   per-parameter scores, bootstrap CIs).  Full matrix form throughout:
   rotating anisotropy bases (limb arcs), rank-1 instances (straight
   ring edges), shared parameter groups (same technique on two bodies),
   and declared suspect-pair cross-covariances.
2. **`scene_gen.py`** — seeded scene families whose *estimator
   composition* is the controlled variable: `limb_disc`,
   `limb_disc_ring_fixed` / `_diverse` (frozen vs drawn ring radial
   direction), `limb_ring_aniso_fixed` / `_diverse` (clipped body,
   genuinely anisotropic limb covariance), `multi_body`.  Scenes are
   deliberately clean (smooth ellipsoids, moderate noise): the campaign
   validates the estimator's mathematics, not technique robustness.
3. **`collect.py`** — navigates every scene in-process and writes one
   JSONL row per scene: planted truth, geometry angles, and every
   per-technique result from each configured run (multi-body scenes run
   once per body via `only_models='body_sim:<NAME>'`; ring scenes add a
   blob-only run for a shared-layer-independent extra estimator).

   ```bash
   venv/bin/python util/agreement/collect.py \
       --per-family 400 --workers 8 --out _work/agreement/rows.jsonl
   ```

   `--injection dt_shift` re-runs with the shared gradient / edge-DT
   products translated by a per-scene random bias,
   `--injection noise_scale` with the shared noise-sigma estimate
   scaled, and `--injection reliability_gate` with every per-type
   reliability threshold raised by `--gate-depression` — all as
   harness-level monkeypatches of the orchestrator module (no
   production seam).  The first two shift a surviving technique's
   offset; the gate only admits or drops, so it is the pure selection
   channel (a surviving technique's offset is byte-identical to the
   control pass).  Injected values are recorded per row.

   ```bash
   venv/bin/python util/agreement/collect.py \
       --per-family 400 --families limb_disc_ring_diverse \
       --injection dt_shift --workers 8 \
       --out _work/agreement/rows_dt.jsonl
   ```

4. **`selection.py`** — the survivorship (selection-effect) model for a
   shared layer that *filters* rather than *shifts*.  The reliability
   gate never moves a surviving offset, so its common-mode effect is a
   selection on the cohort, not a bias; a covariance measured downstream
   of it describes the survivor population.  This module quantifies the
   distortion on planted error arrays (truth-based, so exact): it makes
   precise that separate per-technique gates on independent errors do
   not manufacture cross-covariance (joint survival factorizes) and only
   attenuate each marginal variance, while a shared scene latent driving
   both error and admission attenuates the shared cross-covariance toward
   zero.  Its `main` prints the bound grid the campaign record quotes:

   ```bash
   venv/bin/python util/agreement/selection.py --n 200000
   ```

5. **`analyze.py`** — produces the Markdown report: per composition, the
   truth-free solve next to the truth-based reference (empirical error
   covariance against planted offsets), the null-space demonstration for
   degenerate compositions, and — when injected row files are supplied —
   the bias-independence tables (truth-based pair coupling per
   condition, paired per-scene injection response, solve-side detection
   with a declared pair covariance).  `--gate-rows` adds the
   survivorship-stratified selection-effect table (full population vs
   survivor subset), the procedure the agreement study runs on its own
   real cohorts.

   ```bash
   venv/bin/python util/agreement/analyze.py _work/agreement/rows.jsonl \
       --dt-rows _work/agreement/rows_dt.jsonl \
       --noise-rows _work/agreement/rows_noise.jsonl \
       --gate-rows _work/agreement/rows_gate.jsonl \
       --out _work/agreement/report.md
   ```

6. **`tests/`** — pytest suite for the estimator (synthetic-Gaussian
   recovery and degeneracy demonstrations for every composition regime),
   the scene generator (determinism, schema validity, geometry
   invariants), and the selection model (the factorization and
   attenuation facts above).  `util/` is outside the repo's CI
   pytest/ruff/mypy scope; run them directly:

   ```bash
   venv/bin/python -m pytest util/agreement/tests -q
   ruff check util/agreement && ruff format --check util/agreement
   MYPYPATH=src mypy --strict util/agreement/estimator.py \
       util/agreement/scene_gen.py util/agreement/analyze.py \
       util/agreement/collect.py util/agreement/selection.py
   ```

## Reading the estimator output

- **Identifiability score** (per parameter, in [0, 1]): the squared
  projection of that parameter's axis onto the row space of the design
  matrix.  1.0 means the cohort determines it; near 0 means it lies in
  the null space and the returned (minimum-norm) value is arbitrary.
- **Null-space directions**: linear combinations of parameters that can
  be shifted freely without changing any observable — the explicit form
  of "this composition cannot separate these techniques".
- **Pair mean channel**: each pair's differences are centered by a
  fitted mean model before the second moments -- constant (image-frame)
  terms always, plus rotating-frame mean columns for every
  `basis='rotating'` member (reported in `pair_mean_model`).  A bias
  *constant in the image frame* lands here and leaves the covariances
  clean.  A *geometry-locked* bias (constant in a rotating frame) is
  absorbed only when its technique is declared rotating: undeclared, it
  has image-frame mean ~0 over a diverse cohort and aliases into the
  recovered covariance as `C + mu mu^T` -- silently, well-conditioned,
  and with no negative-variance symptom.  Biases locked to geometry the
  model does not carry (e.g. illumination) still alias.  Fitting k mean
  columns per pair deflates the centered second moments by roughly k/n
  (~1-2% at k <= 7, n = 400); an overfit guard falls back to
  constant-only columns below 4 samples per column.
- **Declared pair covariances**: adding `pair_covariances=[(i, j)]`
  turns an assumed-independent pair into a measured one; the cohort must
  be over-determined enough to carry the extra unknowns (check the
  scores).  Model restrictions: a full/full pair's matrix is
  image-frame constant, and a rank1-involving pair's scalar `gamma`
  assumes an axis-independent (isotropic) projected coupling.

## Caveats

- **Truth basis.**  Planted offsets are ground truth by construction, so
  recovered-vs-planted comparisons here are exact statements about the
  estimator and the navigation algorithms on these scenes.  They are
  *not* real-image accuracy claims; the campaign record spells out the
  sim-scope limitation (see the capability-envelope section of
  `docs/dev_guide/dev_guide_simulator.rst`).
- **Information boundary.**  This directory is measurement-layer code
  (like `util/calibration`): it may read planted truth.  Nothing under
  `src/spindoctor/nav_*` touches truth keys, and the injection is a
  harness monkeypatch, not a production hook.
- **Clean-scene basis.**  The campaign's scenes carry no planted model
  error (no mesh relief, pose scatter, or photometric mismatch), so the
  measured per-technique covariances are near-floor values specific to
  this validation cohort — inputs for checking the solve, not shippable
  technique accuracy numbers.
- **Selection effect is bounded, not measured, in-sim.**  The gate's
  distortion of a covariance depends on how strongly feature reliability
  tracks technique error.  In the clean cohort the load-bearing feature
  reliabilities are near-degenerate across scenes, so the gate acts as an
  all-or-nothing cliff rather than a scene-selective filter, and the
  direct in-sim selection effect on the load-bearing covariances is a
  floor value.  The `selection.py` model supplies the bound for the
  error-correlated case the real cohorts may sit in; the actual coupling
  strength is a property of the real reliability scores, outside the
  simulator's envelope.
