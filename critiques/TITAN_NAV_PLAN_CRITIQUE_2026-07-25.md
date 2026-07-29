<!-- Frozen snapshot 2026-07-25: independent fresh-context adversarial review of
plans/TITAN_NAV_PLAN.md (revision of that date, pre-fix). Findings were
addressed by revising the plan in place; disposition lives in the plan and in
issue #60. This document is not maintained. -->

# Adversarial Review: `plans/TITAN_NAV_PLAN.md`

Reviewed 2026-07-25 against the repo at commit `2a58210` (main). Every named artifact was checked against source. Line numbers cited are current as of this review.

**Verdict — Axis 1 (Opus independence / executability cold): FAIL.** Two blockers (nonexistent ring-occlusion machinery presented as existing; a three-way contradiction about the required status reason) plus several wrong file/API pointers that force an implementer to either guess or deviate silently.

**Verdict — Axis 2 (Unambiguity): CONDITIONAL.** The core method is specified well enough that one careful implementer would build a working fitter, but the two uncertainty formulas, the `c_hat` prose/formula conflict, and several gate semantics would make two reasonable implementers produce materially different code exactly where it feeds the ensemble.

---

## Findings

### BLOCKER

**1. Ring occlusion machinery does not exist, and the plan's cohort makes it load-bearing.**
Section 2.1 item 5: *"built with the same occlusion machinery `NavModelBody` uses for body-body occlusion; rings count as occluders here."*
`NavModelBody`'s occlusion machinery handles sibling **bodies only**: `_compute_occluder_local` (src/spindoctor/nav_model/nav_model_body.py:749) iterates candidate occluder bodies from the inventory ("Every other in-FOV body is a candidate occluder (Titan included...", nav_model_body.py:351). There is no ring-occlusion code anywhere in `src/spindoctor/nav_model/` (grep for ring+occlu: zero hits). "Rings count as occluders here" therefore requires inventing new semantics — which ring radii bound the occluder, what opacity threshold, which backplane — with zero specification. This is not a corner case: the first five entries of the plan's own Phase E cohort (`/seti/all_repos/rms-csmithing/tests/titan_images.txt` lines 1-5) are all "Titan w/edge-on rings occluding". Additionally, the body-body machinery is private instance state inside `NavModelBody.create_model` (`self._occluder_mask`, sibling inventory plumbing via `instances_for_obs`, nav_model_body.py:302, 351-365, 669-680), not a reusable helper `NavModelTitan` can call; "the same machinery" is unactionable without a refactor the plan never mentions.

**2. Three-way contradiction over the status reason for an emitted-feature-but-spurious Titan frame.**
- Section 2.5: technique-side gate failures mean *"the ensemble drops it and the frame resolves exactly as it would today."*
- Phase B: *"confirm a Titan frame with an emitted feature but a spurious technique result still ends `titan_unsupported` (not a generic empty failure) when nothing else is navigable."*
- Acceptance criterion 3: *"Every infeasible/declined Titan frame carries `titan_unsupported`"* (technique-spurious is neither infeasible nor declined).

Against the actual code: `TITAN_UNSUPPORTED` is assigned **only** inside the `if not all_features:` branch (src/spindoctor/nav_orchestrator/orchestrator.py:432-448). Once `NavModelTitan` emits a `TITAN_LIMB` feature, that branch is unreachable for the frame; a Titan-only frame with a spurious result ends `ALL_TECHNIQUES_SPURIOUS` (src/spindoctor/support/status_reason.py:88), and a reliability-gated feature (the 0.30 `reliability_gate` on TITAN_LIMB, config_540_orchestrator.yaml:127) ends `ALL_FEATURES_GATED` (status_reason.py:86). So "resolves exactly as it would today" is impossible (today it resolves `titan_unsupported` via the no-features path, which the feature emission removes), Phase B's "confirm ... still ends titan_unsupported" asks the implementer to confirm something that is false, and the orchestrator change actually required (which failure paths get re-attributed to Titan, and what happens when stars are also present but failed) is unspecified. The "adjust the `_titan_in_models` block only if tests prove it necessary" hedge does not resolve which of the three contradictory behaviors the tests should encode.

### MAJOR

**3. Technique tuning and confidence entries are directed into the wrong config file.**
Section 3 modified-files table: `config_540_orchestrator.yaml` — *"`TitanHazeNav` model-error floor + confidence tuning entries alongside the existing techniques'."* Every existing technique's `model_error_floor_px` and confidence-formula coefficients live in `config_510_techniques.yaml` ("Per-technique confidence-formula coefficients and runtime tunables", config_510_techniques.yaml:1; floors at :122, :242, :372, :491). The loading contract is explicit: `NavTechnique.tuning` is *"loaded from `config_510_techniques.yaml.techniques.<name>.tuning`"* (src/spindoctor/nav_technique/nav_technique.py:272-278), consumed by `load_model_error_floor` (nav_technique.py:370) and `load_confidence_spec` (src/spindoctor/nav_technique/confidence_config.py:36). `config_540_orchestrator.yaml` contains no technique tuning at all. An implementer following the plan literally puts the entries where nothing reads them.

**4. "`TITAN_LIMB: 0.30` ensemble weight already exists" mischaracterizes a reliability gate.**
Section 3: *"the `TITAN_LIMB: 0.30` ensemble weight already exists — leave it."* The entry at config_540_orchestrator.yaml:127 sits under `reliability_gate:` — "Minimum reliability per feature type; features below their type's threshold are gated out before any technique runs" (config_540_orchestrator.yaml:118-121). A parallel copy exists in `DEFAULT_RELIABILITY_THRESHOLDS` (src/spindoctor/feature/reliability.py:37). No per-feature-type ensemble weight exists anywhere. This matters beyond wording: the 0.30 is a **floor the new model's reliability must clear** (interacting directly with finding 5), and an implementer told it is an ensemble weight will neither realize their features can be silently gated nor understand `ALL_FEATURES_GATED` failures during Phase B.

**5. The model's reliability formula is underspecified against a hard gate.**
Phase B: *"reliability = product of a size sigmoid and `(1 - occluded_fraction)`."* No midpoint, steepness, or config key for the size sigmoid appears anywhere — Section 5's schema has `min_envelope_diameter_px` (a decline gate) but no sigmoid parameters. Since the result is compared against the 0.30 reliability gate (finding 4), two reasonable choices of sigmoid produce materially different sets of navigable frames.

**6. Reliability breakdown fields pointed at the wrong file.**
Section 3: `src/spindoctor/feature/reliability.py` — *"Two new optional breakdown fields: `titan_envelope_diameter_px`, `titan_occluded_fraction`."* The breakdown dataclass `NavReliabilityBreakdown` lives in `src/spindoctor/feature/feature.py:31`; `reliability.py` contains only thresholds, `GatedFeatureRecord`, and `FeatureReliabilityGate` (reliability.py:47, 60). An implementer opening the named file finds nothing to extend.

**7. "Reuse its helper path" for the predicted center names a helper that does not exist, next to a trap.**
Section 2.1 item 1: *"the same way `NavModelBody` locates body centers (see `src/spindoctor/nav_model/nav_model_body.py`; reuse its helper path, do not reimplement projection)."* There is no reusable center helper: `NavModelBody`'s geometric center is computed inline in `create_model` as the inventory bbox midpoint plus extfov margin (nav_model_body.py:483-488), entangled with instance state. Worse, the value `NavModelBody` actually stores as its features' *"predicted center"* is the **lit-weighted centroid** (nav_model_body_base.py:303, `_lit_weighted_centroid_vu`), which is phase-biased along the sun direction — precisely the axis this method fits. An implementer who "reuses the helper path" that produces feature centers builds a systematically wrong `p0`.

**8. No sub-solar-point projection exists; the nearest existing API is the wrong quantity.**
Section 2.1 item 3 requires projecting the sub-solar surface point into image `(v, u)`. The repo has no such projection. `NavModelBody` computes only scalar `sub_solar_longitude/latitude` backplane values (nav_model_body.py:429-435). The only image-plane "sub-solar direction" is `_sub_solar_dir_vu` (nav_model_body_base.py:391) — a brightness-centroid heuristic that deliberately collapses to `(0.0, 0.0)` below `_SUB_SOLAR_MIN_OFFSET_PX` at low phase, i.e. exactly where the plan's own risk section (Section 10) still expects a defined axis for the arc fit. The implementer must write new oops projection code with no named API path, and the obvious "existing pattern" is a trap.

**9. Phase D scene schema contradicts the sim package's established convention.**
Phase D: *"a `titan_haze` scene element with truth-side params ... following the established stage/schema conventions of the sim package."* The established convention is the opposite of a new element: haze is an `atmosphere` **block on a body element** — "A body carrying an `atmosphere` block gains an exponential haze layer" (src/spindoctor/sim/forward/atmosphere.py:1-10). Two implementers will materially diverge (new top-level element vs. extending the body atmosphere block), and the plan's extra truth params (interior ramp, north-south asymmetry, cloud blob list) have no placement guidance in either scheme.

**10. Phase A API self-contradiction: `symmetry_scan` has no callback parameter but is required to take one.**
Phase A: `symmetry_scan(grid, grid_valid, r_env_px, window_px, params) -> SymmetryFitResult` *"(includes the angle-refinement loop, which re-calls the resampler through a supplied callback ...)"*. The stated signature has no callback; the stated behavior requires one. The signature and the prose cannot both be implemented.

**11. `TitanHazeFlags` omits `body_name`, breaking feature-level body attribution.**
Section 3 specifies `TitanHazeFlags` with exactly two fields (`surface_window_filter`, `high_phase`). But `NavFeature.body_name` is read off the flags dataclass (`getattr(self.flags, 'body_name', '')`, src/spindoctor/feature/feature.py:230-239), every existing body-feature flags class carries `body_name: str = ''` (flags.py:88, 111, 159, 207, 270), and the orchestrator's source-body logic (`_feature_source_bodies`, orchestrator.py:225-232; `body_names_from_features`, feature.py:242) depends on it. As specced, the Titan feature has empty body attribution; the technique's explicit `source_bodies={'TITAN'}` papers over some paths but not feature-level ones (witness/veto, inventory reporting).

**12. `sigma_cross` and `sigma_along` have no formulas.**
Section 2.2.4: *"`sigma_cross` from the parabola curvature"* — the curvature-to-sigma conversion for a correlation peak is convention-dependent (needs a noise/score-scale model; implementations differ by large factors). Section 2.3.6: *"`sigma_along` from the IRLS normal-equation covariance of `d`"* — requires a residual-variance estimator (weighted RMS? MAD-based? dof correction?) that is never given. These two numbers are the entire covariance the ensemble consumes (Section 2.4), so this is the plan's most consequential unambiguity gap.

**13. `c_hat` prose contradicts its own formula.**
Section 2: *"`c_hat` the unit vector rotated +90 degrees from `a_hat` (in `(v, u)` right-handed image sense)"* vs. Section 2.1.3: *"`c_hat = (cos theta, -sin theta)`."* Rotating `a_hat = (sin theta, cos theta)` by +90 degrees in the ordered `(v, u)` plane gives `(-cos theta, sin theta)` — the negative of the stated formula; the formula is a −90 rotation. The final offset is invariant to the sign flip (c* flips with c_hat), but the Section 7 reviewer instructed to verify Section 2 "line by line" must fail one of the two statements, and the sign of `cross_track_px` in diagnostics/tests differs by implementer depending on which one they trust.

### MINOR

**14. Phase B's "real cached Cassini Titan observation" — no cache, no frame ID.** No cached Titan observation exists in the repo; integration tests fetch from holdings. No frame is named for the Phase B test or the `sd_offset` acceptance run; the only frame source (`titan_images.txt`) is at an external path (`/seti/all_repos/...`), which itself violates the plan's "no briefing beyond CLAUDE.md and the repo" contract (it exists on this machine, but is not the repo).

**15. Feature id `titan_haze:TITAN` violates the documented id format.** `NavFeature.feature_id` format is documented as `<type_lc>:<scope>` (feature.py:93-95), which for `TITAN_LIMB` would be `titan_limb:TITAN`. Not enforced in code, but a convention-checking reviewer will flag it; the plan should state the deviation is deliberate.

**16. Missing `confidence_attributes` and `CURATOR_FIELDS`.** `validate_registered_confidence_specs` requires every spec term to reference a member of the class's `confidence_attributes` ClassVar (nav_technique.py:281-283), and every diagnostics class declares `CURATOR_FIELDS` (diagnostics.py:5). Neither appears in the plan; discoverable via validation failure and pattern-matching, but the plan claims to name everything.

**17. `feature_type.py` docstring goes stale.** feature_type.py:32: TITAN_LIMB is "reserved for Titan navigation; never emitted" — false after Phase B; the file is not in the modified-files table.

**18. Misleading pointer for "which context image".** Phase B points at `nav_technique_body_limb.py` "for which context image/plane conventions apply", but the DT techniques consume `image_edge_dt_ext` / `image_gradient_vu_ext` (nav_technique_body_limb.py:224-249), never the raw image; `TitanHazeNav` needs `NavContext.image_ext` (nav_context.py:77). The pointer teaches extfov coordinates but answers the "which image" question wrongly.

**19. `valid_fraction` denominator ambiguous.** Section 2.2.3: *"used pairs / annulus size"* — pairs vs. samples differ by ~2x and the 0.50 gate sits right where it matters.

**20. Second-peak gate semantics.** "Second/best" is undefined for negative Pearson scores; "separated by more than 2 px" (from the best peak? measured how?) and the no-second-local-max case are unspecified.

**21. "Outer half" of a ray (Section 2.3.2) is undefined** (outer half of the sampled radial range? beyond `r_solid_px`?).

**22. Gradient window can underflow the sampled profile.** Section 2.3.4's window `[r_solid_px - W, r_env_px + W]` extends below the profile start (`0.8 * r_solid_px`, Section 2.3.3) whenever `W > 0.2 * r_solid_px`; clamping is unspecified.

**23. IRLS details.** "Scaled MAD" (1.4826 factor?) unstated; the inner solve for the nonlinear `(d, R)` problem (Gauss-Newton per iteration vs. alternating closed-form) unspecified. Convergent implementations will agree closely, but the code will differ.

**24. Phase D acceptance depends on Phase E's output.** The no-confident-wrong gate uses "confidence >= 0.5" evaluated with Phase-A placeholder anchors, while Phase E is what "set[s] confidence-spec anchors so ... the Phase-D no-confident-wrong bound holds" — circular phase ordering; the plan should say Phase D's confidence-conditioned bound is provisional until re-run after Phase E (Section 7 item 4 partially covers this via the final-revision re-run, but the Phase D acceptance line reads as gating Phase D itself).

**25. Phase E acceptance is not objectively checkable as written.** "Consistency ... within the reported covariance" names no statistical test (chi-square level? 2-sigma per axis?); "non-flagged frames" requires parsing freeform prose annotations ("Titan w/Epimetheus&edge-on rings occluding, Dione", titan_images.txt:1) that the plan calls "flags"; "commanded pointing drift" has no stated data source.

**26. Decline-gate semantics.** `max_occluded_fraction` — fraction of what area (envelope disc? annulus?)? `simulated_unconfigured` — in Phase B the `is_simulated` branch still returns `[]` (nav_model_titan.py:89-90), so no model exists to record that decline until Phase D; which phase owns emitting it is unspecified.

**27. `km_per_px` source ambiguous among three existing conventions.** "Center-of-body resolution oops reports" could be `Backplane.center_resolution` (used in reproj/cartographic_model.py:141), the inventory-derived pixel sizes (nav_model_body.py:489-492), or the per-pixel backplane resolution (nav_model_body.py:664). The plan should name one.

**28. Technique tier and `accepts_feature_types` unstated.** `NavTechnique` requires `tier` ('primary'/'fallback', nav_technique.py:258) and `accepts_feature_types` (:243); defaults make 'primary' the likely accident, but the plan never says which is intended.

---

## Right but fragile

- **Line-number citations** (`nav_technique.py:126`, `:399`) are exact today; any edit to that module silently invalidates them. Symbol names alone would be robust.
- **`TITAN_LIMB: 0.30` exists in two places** — config_540_orchestrator.yaml:127 *and* `DEFAULT_RELIABILITY_THRESHOLDS` (feature/reliability.py:29-38). The plan acknowledges only one; a change to one without the other is a latent inconsistency the implementer isn't warned about.
- **`titan_images.txt` is outside the repo** and outside version control from this project's perspective; the Phase E cohort silently breaks if that tree moves.
- **The `TITAN MODEL` log section already exists** (nav_model_titan.py:104); Phase C's requirement is already half-true and will drift if Phase B rewrites `create_model` without preserving it.
- **`FakeObs` (tests/spindoctor/nav_technique/conftest.py) is the actual synthetic-obs pattern**; the plan's phrase "synthetic `ObsSnapshot`" names a real class (obs/obs_snapshot.py:22) that existing tests do not instantiate — correct-ish, but an implementer who takes the class name literally will fight oops for no reason.
- **`sim/forward/atmosphere.py` already declares the haze "a truth key the navigator never sees"** — the Phase D information-boundary story matches today's code, but adding `NavModelTitanSimulated`'s idealized keys will require touching the boundary whitelist (tests/spindoctor/sim/test_information_boundary.py, test_boundary_static_guard.py); the plan gestures at this ("extend the existing boundary tests ... to the new keys") without noting the whitelist is what enforces it.
- **`config.titan` has zero code consumers today**, so the "complete replacement" of config_060_titan.yaml is safe — but only until anything starts reading `atmosphere_height` between plan-writing and merge. Note also `AttrDict` wraps only the top-level section (config.py:178; attrdict.py has no recursive wrap), so `config.titan.navigation.symmetry` chains stop working after the first level — dict-style access below that, matching how technique `tuning` dicts are consumed.

## Verified correct (spot-check record)

`search_window_for_obs` (nav_technique.py:126, returns `(margin_v, margin_u)`), `add_model_error_floor` (:399, `(covariance, floor_px)`), `NavTechnique._spurious_result` (:324, keyword-only, matches plan usage), `NavFeatureType.TITAN_LIMB` (feature_type.py:46), `NavModelBodySimulated`, `NavModelStarsSimulated`, `validate_registered_confidence_specs` (nav_technique.py:535), `ConfidenceSpec` as ClassVar (:271) mirroring `BodyBlobNav`, `config.titan` property (config.py:503), `util/calibration/library_crosscheck.py`, `scripts/run-all-checks.sh -i` (:166), `sim/forward/atmosphere.py` as the haze anchor, geometry/flags/diagnostics unions living in the files the plan modifies (geometry.py:239, flags.py:274, diagnostics.py:421), tests directory layout including `test_nav_model_titan.py`, boundary tests, `_titan_in_models` (orchestrator.py:215), `TITAN_UNSUPPORTED` in both status files, `titan_images.txt` = 87 lines, OPERATOR_PLAYBOOK Section 4 execution framework, WS-7/#60 mapping in VALIDATION_AND_CALIBRATION_PLAN.md:96, `tukey_biweight_weights` as a pure importable function (dt_fitting/weights.py:25), summary-PNG test pattern (tests/spindoctor/support/test_summary_png.py), and the offset sign convention in Section 2.4 (consistent with actual = predicted + offset given the plan's own grid construction; also invariant to the finding-13 sign flip).
