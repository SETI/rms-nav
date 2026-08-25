<!-- Frozen snapshot 2026-07-25: sixth review round — a second cold-read of
plans/TITAN_NAV_PLAN.md (revision 8) by an Opus-model subagent from the
implementer's seat, with no access to the earlier critiques, instructed to
spot-verify the load-bearing repo claims revision 8 absorbed from the
collateral sweep and to hunt cross-section contradiction drift. Findings
were addressed in the plan's revision 9 the same day, except the structural
consolidation advice in Section 5, which was deliberately handled with
precedence rules instead of restructuring (recorded here for the execution
controller). Not maintained. -->

# Cold-read review: `plans/TITAN_NAV_PLAN.md` (rev 8, second Opus reviewer)

I read the plan start to finish and spot-verified ~35 repo claims against source. The plan is unusually precise and mostly holds up under verification — most of the specific repo claims (config layouts, `stars_in_extfov`/`center_resolution`/`tukey_biweight_weights`/`add_model_error_floor` signatures, the `_titan_in_models` orchestrator branch, `curator.build_metadata_dict` not serializing model metadata, `FakeObs` exposing only `extfov_margin_vu`, the four sim scenes, the two `test_diagnostics.py` parametrize lists, the calibration `TECHNIQUES`/`SIGN_BY_FEATURE` tuples, the 87-line cohort file containing `W1822132529`, the reliability-gate math) are **correct as stated**. Findings below are the exceptions and the friction points.

## 1. AMBIGUOUS

1. **`W` as a scalar vs. per-axis quantity.** Section 2.1 item 6: "`W`: `max(search_window_for_obs(context))`". I confirmed `search_window_for_obs` (`nav_technique.py:126`) returns `tuple[int,int]` per-axis margins (matches `FakeObs.extfov_margin_vu`). The plan then uses the scalar `W` isotropically for the grid half-extent in both `s` and `t`, the offset bounds `|c*|<W`/`|d|<W`, the `not_fully_visible` dilation, and the along-track dilation. Reading A: single scalar `max(margin_v, margin_u)` used everywhere. Reading B: per-axis where the axis is known. **I would pick A (high confidence)** — the text says `max(...)`, and because `c`/`d` are along the rotated `c_hat`/`a_hat` (not `u`/`v`), a per-axis margin cannot be applied cleanly anyway. Worth a one-line note in the plan that the scalar-max is deliberate.

2. **"mask to on-disc pixels" for the min-incidence search** (Section 2.1 item 3). On-disc could mean the solid-body intercept or the envelope disc. Reading A: the solid-body intercept (which is the natural domain of `Backplane.incidence_angle` — off-body is masked). Reading B: everything within `r_env_px`. **I would pick A (high confidence)**: incidence angle is only defined on the surface intercept, so the backplane mask resolves it. But the plan should say "the incidence backplane's valid (surface-intercept) pixels" to remove the doubt.

3. Beyond those two, ambiguity is genuinely low — the algorithm section is written tightly.

## 2. CONFUSING

1. **The mask-shift / dilation / annulus scheme is spread across four places** (Sections 2.1, 2.2.1-2.2.2, 2.3.2, plus the summary table) and I had to cross-reference all four to convince myself they agree (they do). The normative table appears *inside* Section 2.1 but references terms (`c_sub`, "recenter pass", capsule) that are not defined until 2.2-2.3, so a first read forward-references heavily. What would land it: move the table to the top of Section 2.2 (after the terms exist), and state the alignment rule *once* with the table as the single source, deleting the repeated prose in 2.1 and 2.3.

2. **Two different "inner fraction" parameters on two different radii.** The symmetry annulus inner bound is `annulus_inner_fraction * r_env_px` (default 0.55); the arc radial inner bound is `radial_inner_fraction * r_solid_px` (default 0.80). Under time pressure these are easy to conflate (same-looking name, different config block, different base radius). A one-line callout distinguishing them would help.

3. **One concept, three names:** the along-track dilation is called "along-track dilation," "`pass_pad_px`," and "`+-W` (pass 1) / `+-recenter_threshold_px` (recenter)" in different spots. The function signatures use `pass_pad_px`; the table uses "along-track dilation." Consistent naming would reduce re-reads.

4. **`center_pred_vu` breaks the reader's expectation** (see INCORRECT #2) — every other geometry payload I read uses `predicted_center_vu`, so the Titan field name reads as a typo mid-plan.

## 3. INCORRECT

1. **Header self-contradiction.** Line 6 says "revision 2 of the same date, reconciled against the independent review," while line 13 says "Revised through revision 8." Both describe the same document. Non-behavioral, but it is a factual contradiction in the plan's own metadata and a symptom of the duplication drift noted below.

2. **`TitanHazeGeometry.center_pred_vu` contradicts the repo convention it tells me to follow.** The plan defines the field as `center_pred_vu` (Section 4; and again in Section 3: "envelope-circle outline at `r_env_px` around `center_pred_vu`"). But `center_pred_vu` appears **nowhere** in `src/` (I grepped), whereas `predicted_center_vu` is the established name used by `BodyBlobGeometry`, `RingAnnulusGeometry`, `CartographicModelGeometry` (`feature/geometry.py`, 8 occurrences). The plan simultaneously says to add the manual-nav branch "following the `BodyBlobGeometry` branch pattern" — and that branch reads `feature.geometry.predicted_center_vu` (`composition.py:171`). An implementer copying the pattern literally while also naming the field `center_pred_vu` per the plan gets an `AttributeError`. This should be `predicted_center_vu` everywhere.

3. **Imprecise citation for `center_resolution` (minor).** Section 2.1 item 2 cites `reproj/cartographic_model.py` as the "single-axis usage pattern" for `center_resolution(...)`. That file actually calls `bp.center_resolution(body_name).vals` with **no** `axis=` argument (default `'u'`), whereas the plan's own usage passes `axis='u'` and `axis='v'` and averages. The underlying method *does* accept `axis` and return a per-axis scalar (I verified `oops/backplane/resolution.py:35`, `def center_resolution(self, event_key, axis='u')`), so the plan's algorithm is correct — only the cited example doesn't demonstrate the dual-axis call it's cited for.

I found **no** behavioral contradictions in the areas the task flagged: the mask/annulus/pass rules agree across 2.1/2.2/2.3 and the table; the offset assembly (`c_sub_pass2 * c_hat + (d_pass1 + d_pass2) * a_hat`) is self-consistent with the recenter description; gate timing ("final pass only") is stated consistently in 2.2.6, 2.3.7-8, and Phase A test 8; and the Section 3 reliability-gate context matches Section 2/5. The reliability/decline-gate math in Phase B is **verifiably correct**: `sigmoid((D-52)/14)*(1-occ)=0.30` gives D=40.14 at occ=0 and D=42.30 at occ=0.10, exactly the "about 40.1 / about 42.3 px" the plan states, and consistent with `min_envelope_diameter_px=40`.

## 4. MISSING

1. **`NavModelTitan.titan_in_fov` retention is unstated but load-bearing.** The decline -> `TITAN_UNSUPPORTED` path depends entirely on `orchestrator._titan_in_models` doing `getattr(model, 'titan_in_fov', False)` (orchestrator.py:222). Section 2.5 correctly says "no orchestrator logic change," but Phase B rewrites the model (docstrings, `create_model`, `to_features`) and **never says to keep the `titan_in_fov` property**. If an implementer drops it during the rewrite, declines silently degrade to `NO_FEATURES_EXTRACTED` and acceptance criterion 3 breaks with no test necessarily catching it (the plan's status-matrix test would, but only if written to). Should be called out explicitly.

2. **Routing model metadata onto the `NavResult` for serialization.** The plan says Phase B adds a `titan` block to `navigation_result`, and I confirmed `build_metadata_dict` (curator.py:192) currently serializes nothing from model metadata. But `build_metadata_dict(result: NavResult)` takes only the result; the plan doesn't state whether/how the collected `model_metadata` (orchestrator `_collect_model_metadata`, line 430) reaches a field on `NavResult` that the curator can read. The implementer must discover that plumbing before the `titan` block can be serialized.

3. **Shapes of the hand-enumerated lists** the plan says to extend. It names them correctly but not their shapes: `technique_snr_characterization._TECHNIQUES` is a 4-tuple `(label, base_scene_path, technique_name, marker_char)`; `sim_sweep.py` entries and `scene_gen.py` family generators have their own shapes. Discoverable by reading, but a cold implementer needs one round-trip per list.

4. **Numerical guard for `sigma_cross`.** `sigma_cross = cross_sigma_scale * sqrt((1 - s_pk)/(2|a|))` (2.2.4) will produce NaN if the fitted parabola vertex `s_pk` exceeds 1 (Pearson peaks can round-trip above the sampled max after parabolic refinement); NaN does not clamp into `[floor, W]`. Not specified.

5. **Minor:** the config key is `atmosphere_height` (Section 5) but the algorithm refers to `atmosphere_height_km` (2.1 item 2) — units assumed km, key lacks the suffix; and `R_TITAN_km` "from the oops body" doesn't say which radius (Titan is near-spherical, so harmless).

## 5. Overall assessment

**Could I execute this cold, phase by phase, with no other briefing?** For Phases A, B, C — **yes**, with high confidence. Phase A (the fitting library) is the best-specified part: 12 concrete test families with numeric bounds, pure-array signatures, and a clean math spec. Phase B's structure (`TitanGeometryInputs` split, `FakeObs`-vs-direct-construction rule, the exact status matrix) is genuinely executable, modulo MISSING #1-2.

**The phase that worries me most is Phase D.** It is the one place the plan asks me to *invent* code from prose: non-affine hemispheric falloff (`ns_falloff_ratio`), `sector_sharpness_gradient`, `axis_tilt_deg`, etc., as "new rendering math in `atmosphere.py`" (the plan admits the current model is "a single haze profile" — I confirmed it's one exponential-tau profile). On top of that Phase D stacks the information-boundary whitelist trap (the `bodies.atmosphere` atomic-block gotcha — I confirmed `_TRUTH_SAMPLES['bodies.atmosphere']` exists and is treated as one block), the perf-budget gating (confirmed 2s@512 in `test_sim_perf.py`), the three-scene rename + baseline regen, and the `NavModelBodySimulated` routing exclusion. That is five loosely coupled workstreams with acceptance bounds that are only meaningful "if these effects genuinely render" — the highest ratio of judgement-to-spec in the plan. Phase E is also externally gated (operator votes, cohort retrievability) but that is by design, not a spec gap.

**Is the length a defect?** Partly, yes — and the rev-2/rev-8 header contradiction is the tell. The duplication that risks drift:
- The **mask/annulus/pass alignment rule** is stated in full prose in 2.1, restated in 2.2.1-2.2.2, restated in 2.3.2, and tabulated. The plan already concedes "where prose and this table disagree, fix the prose" — which means it *knows* they can diverge. Consolidate to the table + one prose paragraph.
- The **per-file collateral change list** is given once in the Section 3 "Modified files" table and again, per file, in the Phase 6 narratives (and a third time in the Phase F stale-statement checklist). These will drift as revisions accrete. I would make Section 3 the single manifest and have the phases reference it rather than re-describe each file.

Net: a strong, executable plan whose remaining risks are (a) the two naming/metadata gaps that could silently break the decline path and the `center_pred_vu` field, and (b) Phase D's genuine open-ended rendering work.
