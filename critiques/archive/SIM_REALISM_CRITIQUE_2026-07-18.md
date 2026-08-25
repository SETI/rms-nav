# Sim-Realism Non-Circularity Critique

**Date:** 2026-07-18
**Baseline:** `rf_sim_realism` @ a4d9138, all ten phases of the
simulator-realism program merged on the integration branch.
**Question:** does the as-built system meet its goal of being a
non-circular calibration and validation system for the main navigator?
**Method:** fresh adversarial review. Every load-bearing claim below was
verified against the source tree: the boundary machinery was probed at
runtime with concrete leak constructions, the calibration scene generator
and campaign records were read line by line, the information-boundary test
suite was executed, and the shipped documentation was checked against what
the code actually does. This is a frozen snapshot; nothing in it is
maintained.

---

## Verdict

**Yes, with boundaries.** The system genuinely cuts the circularity it was
built to cut: the truth/idealized key partition is complete, machine-read,
and default-deny; the filtered view is the only channel current
navigator-side code reads; the planted-error recovery measurements are
honest by construction; and the shipped documents label the
self-consistency floor, the provisional confidences, and the unverified
instruments with unusual candor. But the non-circularity claim has three
real boundaries. First, the boundary is a *filtered channel*, not a
structural barrier: the full truth dictionary, including the planted
offset, rides on the same observation object every navigator-side model
holds, one attribute access away, with no static enforcement that models
read only the filtered view -- the guarantee is convention plus review,
while the developer guide claims it is structural. Second, the confidence
calibration was not fitted on the realism-anchored renderer configuration:
the realism match vouches for frames rendered with the empirical
instrument chain, but every calibration scene renders with those stages
off (no whole-scene PSF on body and ring scenes, a deliberately
navigator-matched PSF on star scenes, no artifacts, no distortion, no
smear) and every calibration scene emulates a single instrument, Cassini
NAC -- so the realism evidence does not underwrite the shipped
coefficients, and their real-world meaning rests almost entirely on one
75-frame library cross-check that agrees on tier 46 times out of 75.
Third, transfer to real frames is watched by an advisory process whose
regression gate is currently suspended, with no stated criterion for what
would falsify the calibration. Within those boundaries the system is a
large, real improvement over what it replaced; outside them, its numbers
are reproducibility statements wearing accuracy labels that the fine print
mostly -- but not everywhere -- corrects.

---

## 1. The boundary as built versus as claimed

### What holds (verified)

- The key inventory is complete and disjoint: every schema key is
  classified idealized, truth, or test-only, an import-time assertion
  fails on any unclassified or doubly classified key
  (`src/spindoctor/sim/scene_schema.py`), and the filter is default-deny
  (`build_nav_params` in `src/spindoctor/sim/scene.py` copies only
  keys on the idealized allowlists; unknown keys stay behind).
- The 53-test boundary suite passes (run during this review, 1.18 s). It
  exercises every `TRUTH_KEYS` entry with a non-default value and asserts
  none is reachable through `obs.nav_params`; a truth key added without a
  sample fails the completeness check.
- Non-navigable stars and ring features render but never cross: the test
  proves the rendered image changes while the navigator's view is
  identical. `nav_override` overlays the believed geometry and drops the
  true values. All crossing values are deep copies.
- The star detection limit is derived from the *published* instrument
  configuration, never the scene's truth-side noise block, and the test
  proves two scenes differing only in planted noise report the same
  limiting magnitude. A scene that plants unpublished noise gets an
  honestly wrong detection limit -- designed model error, correctly so.
- Current navigator-side code is clean. A sweep of
  `src/spindoctor/nav_model/`, `nav_technique/`, and `nav_orchestrator/`
  finds no read of `sim_params`, `sim_offset_*`, `sim_star_list`,
  `sim_body_models`, or any other renderer output; the simulated models
  read `obs.nav_params` exclusively.

### What does not hold: the structural claim

The developer guide states "The boundary is enforced structurally, not by
convention" and "the navigator side structurally cannot read what is not
there" (`docs/dev_guide/dev_guide_simulator.rst`, information-boundary
section). Both statements are true only of the `nav_params` mapping
itself. `ObsSim.from_file` stores the *full* scene on the snapshot
(`snapshot.sim_params`, `obs_inst_sim.py`), along with the planted offset
as dedicated attributes (`sim_offset_v` / `sim_offset_u`) and the
renderer's truth metadata (`sim_body_models`, `sim_inventory`, body mask
maps). Because `ObsSnapshot` adopts the snapshot's attribute dictionary
wholesale (`self.__dict__ = snapshot.__dict__`,
`src/spindoctor/obs/obs_snapshot.py`), every one of these is a plain
attribute on the very object each navigator-side model holds as
`self.obs`.

Concrete leak construction, executed during this review: instantiate the
star model through the production factory
(`build_models_for_obs(obs)`) and read
`model.obs.sim_params['offset_v']` -- it returns the planted offset
(3.25 in the probe). Nothing raises, nothing warns, no test would notice.
The renderer's output records are equally reachable. The boundary test
asserts only that `obs.sim_star_list` does not exist and that `nav_params`
is clean; it does not and cannot assert that navigator-side *code* never
touches the truth attributes. There is no AST- or import-based check, no
lint rule, and no naming convention (a leading underscore, a wrapper
object raising on access) standing between a future model change and the
truth dictionary.

This matters because the program's own premise is that authorship
discipline is unreliable and enforcement must be mechanical. The
enforcement that exists is mechanical for the channel and conventional for
the consumers. The one historical violation was exactly a consumer
reading a convenient attribute; the attribute surface that made it
possible is still there, now carrying the whole scene instead of a star
list.

### Prospective channels worth recording

- **Correlated error bars in the calibration scenes.** The campaign
  generator draws a ring feature's planted `orbit_error` first and then
  sets `declared_orbit_sigma` to 0.8-1.5x the drawn magnitude
  (`util/calibration/scene_gen.py`). The declared sigma is an idealized
  key the navigator may consume -- and the simulated ring model already
  widens its per-vertex radial sigma with it. Today no technique exploits
  it further, but the standing recommendation for closing the
  orbit-error gap is precisely to consume `sigma_a_px` in the covariance
  or ensemble. If that lands and is then calibrated on scenes built this
  way, the fit will learn that the declared sigma predicts the *realized*
  error to within 50% -- information a real published uncertainty does
  not carry per frame. The sim would overstate the fix's value. Draw the
  planted error *from* an independently chosen declared sigma, not the
  reverse.
- **Scene-authored idealized keys.** `instrument_config` is idealized and
  scene-authored, so a scene can echo truth-side values (e.g. its planted
  noise) into the navigator's published-config view. No machinery can
  prevent an author from doing this; it is worth a validator warning when
  a scene's `instrument_config` duplicates its truth-side `noise` values.
- **Seeds and truth files.** `random_seed` is a truth key and does not
  cross; catalog scene files necessarily contain their planted offsets in
  plain YAML, but only test harnesses read them. No leak found.

## 2. Calibration: where the circle is cut, and where it is relocated

The de-circularization argument is a chain: (1) the boundary makes
per-scene recovery errors honest; (2) the forward model is anchored to
real cohorts through the realism figures of merit, with the
navigator-diagnostics figure of merit excluded from tuning; (3) therefore
confidence formulas fitted on sim campaigns approximate real-frame
error probabilities, provisionally. Link (1) holds (Section 1). Link (2)
holds for what it measured. The problem is that link (3) does not connect
to link (2) as built.

### The calibration cohort is not the realism-verified configuration

The realism match renders its matched frames with
`artifacts: {instrument_defaults: true}` -- the full empirical signal
chain, tuned PSF kernel included
(`tests/integration/sim_realism_scenes.py`). The calibration campaign
does not. Reading `util/calibration/scene_gen.py` end to end:

- The strings `artifacts`, `instrument_defaults`, `spk_error`, `smear`,
  `distortion`, and `atmosphere` do not occur in the file at all.
- Body and ring families carry no `optics` block: their limbs and ring
  edges render with anti-aliasing only, no whole-scene PSF. The realism
  work established that real Cassini limb rises are PSF-shaped (~2.5 px);
  the limb, terminator, ring-edge, disc, and blob confidence formulas
  were fitted on frames without that shaping.
- The two star families set `optics.psf: {match_navigator: true}` -- by
  definition the self-consistency floor form, a pure Gaussian equal to
  the navigator's own model. The star confidence formulas therefore
  price catalog scatter, confounders, and companions, but zero PSF
  mismatch -- on cameras whose real frames are undersampled with measured
  power-law wings.
- The noise model is Poisson plus a log-uniform read-noise draw. No
  banding, no quantization, no hot pixels, no structured telemetry loss,
  none of the artifact catalog the realism match tuned.
- Every scene of all seven families sets `instrument: coiss_nac`. The
  fitted alphas, the model-error floors, and the tier boundaries in
  `config_540_orchestrator.yaml` apply fleet-wide -- Voyager, Galileo,
  LORRI -- from a single-instrument cohort. This scope limit is disclosed
  nowhere: not in `config_510_techniques.yaml`'s otherwise thorough
  provenance header, not in `util/calibration/README.md`, not in
  `CAMPAIGN_20260718.md`.

To be fair about what the campaign *does* contain: it is far from a pure
floor. The body families draw limb relief, non-Lambert photometry,
opposition surge, albedo texture, catalog-scaled mesh error, and pose
error; the ring family draws inclined projections, eccentric and m-mode
orbits, planted per-feature orbit errors, and non-navigable distractors;
the star families draw catalog scatter and confounders. Those axes are
real model error, honestly planted, and fitting on them is exactly what
the program prescribed. What is missing is the entire optics and detector
realism layer -- the layer the realism match exists to vouch for.

The consequence: the chain's central claim -- "the calibration basis is
the sim's realism (quantified per instrument in the simulator report's
realism-match section)", as the `config_510` header puts it -- overstates.
The realism match quantified a renderer configuration the calibration
never ran. The calibration basis is a *different*, unverified-by-design
configuration whose gradients are sharper, whose stars are cleaner, and
whose noise is simpler than the frames the match anchored. The
`util/calibration/README.md` caveat gestures at this but misattributes
it: it says "regimes the sim cannot render (real PSF wings, saturation
bloom on stars, calibrated-I/F detector noise) are uncalibrated by this
fit." The sim *can* render all three now; the campaign chose not to. The
stale phrasing both understates the gap (the whole empirical chain is
out, not three exotica) and misstates its cause (a campaign
configuration choice, not a renderer limitation).

### The mixture prior is authored, not anchored

Even granting per-scene realism, a confidence formula is
P(error <= 1 px | diagnostics) *under the campaign's scene mixture*, and
the tier boundaries are "smallest confidence achieving 0.9 success"
*under that same mixture*. The mixture -- 40% limb relief, 15% orbit
error, 20% ring distractors, family base rates 0.77-0.97 -- is authored
by the same program, with no claim (and no data) that it matches the
operational distribution of real frames. Logistic intercepts and
threshold placements are exactly the quantities that shift under
covariate and base-rate shift. This is not hidden -- the numbers are all
recorded -- but it should be recognized as the place the old circularity
now lives in diluted form: the sim team no longer grades its own
renders, but it still chooses the exam's question mix, and the tier
boundaries inherit that choice.

### The diagnostics firewall is procedural

The exclusion of navigator diagnostics from forward-model tuning is
stated consistently (plan, report, dev guide) and the tuning provenance
trail in `sim/forward/artifacts_catalog.py` is exemplary -- every tuned
value cites its cohort statistic, every retained zero its reason,
including two adoption attempts reverted on figure-of-merit evidence.
But the firewall is a documented practice, not a mechanism: the same
sessions held both sides in context, and nothing structural would have
caught a diagnostics-informed tuning choice. Post hoc, compliance is
unverifiable; the provenance comments are the only audit surface. They
read clean. This is an honest residual risk to acknowledge, not a
finding of violation.

## 3. Validation honesty

### What is labeled well

- The simulator report leads with the realism match as the precondition
  for reading anything else as accuracy, attaches no pass/fail
  thresholds, labels per-instrument support (Galileo "bounded by
  unverified forward-model fidelity", Voyager PSF "unconstrained", LORRI
  per-mode unverified, WAC noise divergent by ~8x one-sided), and
  discloses its own estimator caveats -- including the registration
  asymmetry under which part of the tuned PSF wing may be absorbing
  operator registration error rather than optics. That last admission
  quietly means the anchored kernel itself carries a one-sided bias risk.
- The model-mismatch chapter states plainly that most accuracy numbers
  sit on the self-consistency floor and marks the floor point on every
  mismatch curve. The user guide repeats the distinction in plain
  language. `confidence_provisional: true` is a hard-coded literal in
  every metadata product (`nav_orchestrator/curator.py`).
- The known-gaps list is genuinely adversarial to the system's own
  numbers: stars shining through dark limbs ("simulated star-technique
  success ... is therefore optimistic"), the unmodeled transient share on
  every chain, the quantization-scar floor on calibrated-path noise, the
  per-scene (not per-detector) hot pixels.

### Where reproducibility can still be mistaken for accuracy

- **The confident-wrong pins are green tests.** The expected-outcome
  machinery (`tests/integration/sim_expected.py`) asserts status and
  tier only; it has no offset-error field. `orbit_error_ringlet` pins
  `expected: success / high` for a fused result that is ~3 px wrong at
  0.89 confidence; `titan_crescent_horns` pins `success / low` for a
  ~30 px wrong result at the blob's 0.40 cap, 0.05 above the acceptance
  gate. Both are thoroughly documented as standing ensemble-gap evidence
  -- in YAML comments, the dev guide's simulator chapter, and the
  campaign record -- so they are documented hazards, not silent
  normalization. But the machinery's own docstring says these tests
  exist "so a confident wrong offset is a test failure," which is
  exactly what these two scenes make *not* true: a future regression
  that doubles the wrong offset while keeping status and tier would pass
  CI, and a fix that correctly demotes the ringlet result to conflicted
  would *fail* CI. A pin that asserts "success/high, and the offset
  error stays within [2.5, 4] px" would keep the hazard in view without
  freezing it as correct behavior.
- **The hazards are documented on the wrong side of the boundary.** Both
  confident-wrong behaviors live in simulator-side documentation. The
  ensemble chapter and the user-facing navigation guide -- what an
  operator consuming a high-tier result would read -- do not mention
  that a planted radial catalog error produces a confident high-tier
  wrong answer, or that a high-phase haze frame can pass the gate ~30 px
  wrong. The tier definitions those readers see were calibrated to 0.9
  success; the known counterexamples are three clicks away in a
  different guide.
- **The terminator technique ships on a degenerate fit.** Its campaign
  cohort produced zero usable rows within 1 px out of 116, so the
  formula is an honest ~0.03-0.05 plateau with a 4.32 px floor --
  correctly humble. But the realism verdict for the terminator side was
  never computed (the limb figure of merit was measured on the limb side
  only), and the high-phase strata are where the limb comparison is at
  its worst (sim crescents measurably wider than real, W1/IQR up to
  2.04). The provisional label covers this; the point is that for the
  terminator regime the system currently has neither realism evidence
  nor a non-degenerate calibration -- only the label.

## 4. The navigator-side mirrors

The calibration runs the real technique code (a genuine strength -- the
DT fitters, correlators, and ensemble under test are the production
ones) against hand-written simulated NavModels standing in for the
SPICE-backed models. The program scopes the real models' *geometry* out
(validated by real-image baselines), but calibration transfer also
depends on the mirrors' *emission gates, sigma models, and diagnostics*
matching the real models -- and there the record shows drift is real and
the guard rails are partial:

- Divergences were found late and fixed ad hoc: the highly-irregular
  suppression policy was shared only after a resolved Hyperion mesh was
  caught emitting a terminator the real model would suppress; the sim
  terminator's arc fraction and reliability were recomputed "honestly"
  in the same cleanup; the module docstring's full-parity claim had to
  be corrected to disclose the sigma-model exception.
- The parity mechanism is inconsistent by construction. Some constants
  are imported from the real model (`TERMINATOR_MIN_PHASE_FACTOR`,
  `BODY_BLOB_MIN_DIAMETER_PX`, the shared suppression helper); others
  are hand-duplicated with a comment promising agreement
  (`_LIMB_MAX_PHASE_DEG = 60.0` in `nav_model_body_simulated.py`,
  "matches the catalog body model's limb/blob handoff"). Worse than a
  duplicated constant, the *shape* of the limb emission gate differs:
  the real model gates LIMB_ARC on a derived uncertainty budget
  (`limb_uncertainty_px <= 3.0`, minimum vertices), the sim on hard
  geometric cutoffs (diameter >= 100 px, phase <= 60 deg) -- a
  difference large enough that the realism runner had to bypass the sim
  gates to populate its high-phase strata. Per-vertex sigmas are fixed
  constants in the sim (1.0 / 2.0 px) versus PSF-, softness-, and
  albedo-derived budgets in the real model, disclosed in a docstring.
- There is no parity test asserting the mirrors' gates and sigma
  formulas track the real models, and no single inventory of known
  divergences -- the disclosures are scattered across docstrings, a
  report footnote, and commit messages. A future change to the real
  model's gates silently diverges the calibration cohort's feature mix
  with nothing to catch it. Given that this program cost three fix
  commits to *find* the current divergences, leaving their prevention
  to memory is the identified mechanism by which today's calibration
  quietly rots.

## 5. The real-frame transfer

The one measurement connecting the calibration to reality is the library
cross-check over 75 operator sidecars: status agreement 69/75, tier
46/75, offset-within-slack 54/61, primary technique 25/60. The campaign
record attributes every disagreement to a specific calibration change
and argues most tier flips are the calibration being more honest than
labels that predate it (limb-floor demotions, the raised high boundary).
The per-frame attribution work is excellent forensics. Three caveats
keep it from being validation:

- The attribution is performed by the calibration's own authors against
  labels they simultaneously declare stale. Every disagreement is
  explained; none is treated as potential falsification. There is no
  stated criterion -- before or after the run -- for what level or
  pattern of disagreement would have meant "the calibration does not
  transfer." A 61% tier agreement is consistent with the system's story
  *and* with a miscalibrated tier boundary; nothing shipped
  distinguishes them.
- The regression gate that would watch this over time is suspended: the
  recorded failure set predates the refit, the campaign record says the
  historical gate "no longer applies as-is," and re-ratcheting the
  sidecar expectations is deferred to an operator decision. At the
  moment of merge, the only shipped process watching real-frame behavior
  is inoperative.
- The cross-check surfaced one real-frame failure mode the sim never
  predicts (a star frame whose gates self-flag spurious on operator-
  navigable content), and the diagnostics comparison shows the sim
  consistently optimistic where it is checkable: simulated star inlier
  residuals 3-10x tighter than real, the matched ring render at 0.94
  versus 0.71 on the real frame. Optimism of the sim's diagnostics
  distribution translates directly into overconfident real-frame tiers;
  46/75 is what that looks like.

What would falsify the calibration on real frames -- operator-verified
offsets with measured per-technique errors -- is named in the README as
the future real anchor, but no shipped process collects it.

## 6. Historical residue

The named pre-program violations are actually gone, not relocated:

- `sim_star_list` no longer exists on the observation: the boundary test
  asserts `not hasattr(obs, 'sim_star_list')`, and the only remaining
  uses of the name are internal to the forward renderer's own return
  metadata, which stays on truth-side attributes no navigator code
  reads. The simulated star model builds catalog records from
  `nav_params` (verified in source).
- The ring-epoch defect (renderer honoring `ring_epoch` while the
  navigator always read 0.0) is fixed: `ring_epoch` is classified
  idealized and the simulated ring model reads it from the filtered
  view.
- The campaign generator, which formerly fed dicts straight to the
  renderer, now routes every generated scene through
  `validate_sim_params` at generation time (verified in
  `scene_gen.py`) -- an attack attempted during this review and found
  closed.
- The residue that remains is the attribute surface of Section 1: the
  mechanism that made `sim_star_list` possible -- truth-side convenience
  attributes on the shared observation object -- was not removed, only
  emptied of its most-used member.

---

## What the system should not be trusted for today

- **Confidence values or tiers as real-frame probabilities, on any
  instrument.** The provisional flag says this; believe it. The tier
  boundaries additionally encode an authored scene mixture and a
  single-instrument, floor-optics cohort.
- **Anything on Voyager, Galileo, or LORRI beyond noise/artifact
  incidence.** No realism evidence supports PSF, limb, or ring fidelity
  there (Galileo's cohort is all negative cases; Voyager's PSF
  comparison is unconstrained; LORRI's binned frames cannot check the
  1x1 kernel), and no calibration scene emulated these cameras at all.
- **High-tier results on ring frames with plausible ephemeris error, and
  gate-passing results on high-phase haze frames.** Both confident-wrong
  modes are pinned, reproducible, and unmitigated: a ~2.5 px radial
  catalog error yields success/high ~3 px wrong; a 155-degree haze
  crescent yields a gate-passing ~30 px wrong answer.
- **Terminator-regime accuracy claims.** No terminator-side realism
  verdict exists, the high-phase limb comparison is the match's worst
  stratum, and the technique's calibration cohort contained zero
  sub-pixel successes.
- **Cassini WAC calibrated-path noise behavior** (order-of-magnitude
  divergence, attributed but unresolved), and any regime relying on
  cosmic-ray transients, which every chain currently renders at zero.
- **The permanence of the boundary.** It holds today by test plus
  convention; it is one convenient-attribute read in a future model from
  silently not holding, and no automated check would notice.

## Recommendations, ranked by leverage

1. **Re-run the calibration campaign on the realism-anchored
   configuration.** Enable `instrument_defaults` (empirical PSF and
   detector chain) across the body/ring/star families, keep the truth
   axes, and refit. This is the single change that actually connects
   the realism evidence to the shipped coefficients; until it happens,
   state in the `config_510` header that the calibration basis is the
   floor-optics renderer, not the realism-matched one, and disclose the
   coiss_nac-only cohort in the same header.
2. **Make the boundary structural on the consumer side.** Either move
   truth off the shared attribute surface (a `SimTruth` handle the test
   harness must explicitly request, with `obs.sim_params` gone) or add
   a static check (AST walk over `nav_model/`, `nav_technique/`,
   `nav_orchestrator/`) forbidding the truth attribute names. Cheap,
   and it converts the dev guide's "structurally enforced" claim from
   overstatement to fact.
3. **Give the expected-outcome machinery an offset-error assertion** and
   use it on the confident-wrong pins, so a worsening regression fails
   and a genuine fix fails loudly (prompting a deliberate re-pin) --
   and surface both standing confident-wrong modes in the ensemble and
   user-facing navigation documentation, where their consumers read.
4. **Add a mirror-parity test and a divergence inventory.** Assert the
   sim models' emission gates and shared constants against the real
   models' (importing rather than duplicating where possible), and
   collect the deliberate divergences (sigma models, gate shapes) into
   one documented list with rationale. This is the cheapest insurance
   against silent calibration rot.
5. **Pre-register a transfer criterion and re-arm the gate.** Decide,
   before the next campaign, what library cross-check agreement (and
   what disagreement pattern) is acceptable; re-ratchet the sidecar
   expectations so the real-frame regression gate operates again; and
   start accumulating the named real anchor (operator-verified offsets
   with per-technique errors) so the next fit can be real-anchored
   for the regimes that allow it.
6. **Fix the correlated error-bar draw** in the campaign's ring family
   (draw planted error from an independent declared sigma) before any
   work consumes `declared_orbit_sigma` in the covariance or ensemble.
7. **Correct the two stale/overstated fine-print passages:** the
   `util/calibration/README.md` "cannot render" caveat and the dev
   guide's structural-enforcement claim. The system's credibility is
   carried by the accuracy of exactly this fine print.

## References

- Program plan: `plans/SIM_REALISM_PLAN.md` (issue #227; boundary
  definition, realism figures of merit, diagnostics-firewall rule,
  acceptance criteria).
- Boundary machinery: `src/spindoctor/sim/scene_schema.py`,
  `src/spindoctor/sim/scene.py` (`build_nav_params`),
  `src/spindoctor/obs/obs_inst_sim.py`,
  `src/spindoctor/obs/obs_snapshot.py`,
  `tests/spindoctor/sim/test_information_boundary.py`.
- Calibration: `util/calibration/scene_gen.py`, `collect.py`,
  `README.md`, `CAMPAIGN_20260718.md`;
  `src/spindoctor/config_files/config_510_techniques.yaml`,
  `config_540_orchestrator.yaml`; floors issue #210, terminator
  issue #223.
- Realism match: `tests/integration/sim_realism_scenes.py`,
  `src/spindoctor/sim/forward/artifacts_catalog.py`,
  `docs/simulator_report/simulator_report.rst` (realism section, known
  gaps); diagnostics comparison issue #153.
- Confident-wrong pins:
  `tests/integration/sim_scenes/ring_system/orbit_error_ringlet.yaml`,
  `tests/integration/sim_scenes/atmosphere/titan_crescent_horns.yaml`,
  `tests/integration/sim_expected.py`.
- Mirror divergences: `src/spindoctor/nav_model/nav_model_body_simulated.py`,
  `nav_model_body.py`, `nav_model_rings_simulated.py`; branch commits
  3b424c1, 3ed1a79, 0acf85a.
- Library regression state: issue #288.
