<!-- Frozen snapshot 2026-07-25: seventh review round — a third cold-read of
plans/TITAN_NAV_PLAN.md (revision 11) by an Opus-model subagent, with no
access to earlier critiques, directed at the newest redesign (deletion of the
Titan-specific status reason in favor of the uniform reliability-gate path).
The reviewer traced the redesign through the live orchestrator code and
confirmed it works; its three findings (the always-emit/never-raise hole on
clipped frames, the three-file scope of the breakdown serialization, and the
unlisted orchestrator test file) were addressed in the plan's revision 12 the
same day. Not maintained. -->

# Cold-read review: TITAN_NAV_PLAN.md (revision 11, third Opus reviewer)

I read the plan start to finish and verified its load-bearing repo claims against source. Overall it is unusually well-built — most claims check out exactly, including several I expected to be wrong. But the most recent redesign (the `TITAN_UNSUPPORTED` deletion + uniform reliability path) has two genuine holes and one manifest gap, and there is a class of failure the always-emit design does not close.

## HARDEST-SCRUTINY AREA: the reliability-gate redesign

### (a) Does a reliability-gated Titan-only frame actually produce ALL_FEATURES_GATED? — YES, verified.

Traced through `src/spindoctor/nav_orchestrator/orchestrator.py` `_navigate_impl` (lines 420-465) and `src/spindoctor/feature/reliability.py` `FeatureReliabilityGate.apply` (lines 133-160):

- Model always emits -> `all_features` non-empty -> the `if not all_features:` branch (line 432, the one being deleted's parent) is skipped.
- `apply()` gates on `feature.reliability < threshold` (line 149). `0.0 < 0.30` is True -> the feature lands in `gated`, `kept` is empty.
- `if not kept:` (line 457) -> `return self._fail(status_reason=NavStatusReason.ALL_FEATURES_GATED, ...)`.

`DEFAULT_RELIABILITY_THRESHOLDS[TITAN_LIMB] == 0.30` (reliability.py:37) and `config_540_orchestrator.yaml` line 127 `TITAN_LIMB: 0.30` are both present and in sync. The design works as claimed. I also confirmed the reliability crossover math: `sigmoid((40.1-52)/14) = 0.300` at zero occlusion and `D~=42.3` at `occluded_fraction=0.10` — the plan's Phase B numbers are arithmetically correct, and the sigmoid crosses 0.30 just above the `min_envelope_diameter_px=40.0` hard floor as claimed.

### (b) Is deleting `TITAN_UNSUPPORTED` / `_titan_in_models` / `titan_present` safe? — MOSTLY, but the manifest misses one test file.

Full grep of `src/ tests/ util/ docs/ plans/` (excluding generated `docs/_build`). Every consumer the manifest lists is real and correct:
- `status_reason.py:84` (enum) + `:51` (docstring); `status_reason_info.py:57` (template); `orchestrator.py:215-222,438-447`; `test_status_reason.py:20,47` and the `len(...) == 20` assertion at line 38 (plus its docstring "Exactly 20 values" at line 37, which also needs the 20->19 edit — subsumed but easy to miss).
- `nav_model_titan.py:15` (docstring xref) — covered by "docstrings rewritten".
- Only reader of `titan_in_fov` is the orchestrator; deletion is clean.

Deletion-safety claims all verified true:
- `ALLOWED_STATUS_REASONS` in `tests/integration/sidecar.py:94` is `frozenset({reason.value for reason in NavStatusReason})` — derives from the enum, so it self-heals. And no library sidecar references titan (grep of `tests/integration/image_library/` is empty), so no sidecar validation breaks.
- Stats layer stores `status_reason TEXT` (`cli/stats/schema.py:21`) — free text, confirmed.

**Gap:** `tests/spindoctor/nav_orchestrator/test_orchestrator.py` has TWO tests that reference the deleted path — `test_orchestrator_titan_only_yields_titan_unsupported` (line 262, asserts `== TITAN_UNSUPPORTED`) and `test_orchestrator_titan_plus_stars_navigates_normally` (line 274, asserts `!= TITAN_UNSUPPORTED` at line 283), both driving a `_FakeTitanModel`. These will fail to import/run after the enum member is deleted. This file is **not named anywhere in the Section 3 manifest or Phase B**. Per the plan's own rule ("A file a phase touches that is missing here is a plan bug"), this is a manifest bug the implementer will hit at Phase B.

### (c) Do GatedFeatureRecords carry the breakdown, and does anything serialize it to JSON today? — Record carries it (indirectly); JSON serialization does NOT exist, and the fix is bigger than the manifest says.

- `GatedFeatureRecord` (reliability.py:46-56) holds `feature` + a `reason` string. The `feature` carries `reliability_reasons: NavReliabilityBreakdown` (feature.py:139), which will gain the two new Titan fields per the manifest. So the structured breakdown IS reachable through the record. Good.
- **But nothing serializes it into the per-image JSON today.** `build_metadata_dict` (curator.py:192) serializes `result.feature_inventory`, which is `list[NavFeatureSummary]` (nav_result.py:86). `NavFeatureSummary` (feature_summary.py:40-46) carries only scalar `reliability`, `gated`, and the `gate_reason` **string** — NOT `reliability_reasons`. `_curate_feature_summary` (curator.py:144-154) emits exactly those scalar fields. `_format_reliability_breakdown` (orchestrator.py:1111) formats the breakdown only for a DEBUG log line, never JSON. Grep confirms zero `reliability_reasons`/`reliability_breakdown` references in curator.py or feature_summary.py.

So the plan's "verify-then-extend-generically" is honest, and the answer is: the extension is required. **The problem is the manifest under-scopes it.** The Section 3 curator row and Section 2.5 name only `curator.py`. But because `NavResult` retains only summaries (not the raw features/gated records), curator physically cannot reach the breakdown. Making acceptance criterion 3 true requires touching **three** files: add a breakdown field to `NavFeatureSummary` (`feature_summary.py`), populate it in `_summary_from_feature`/`_build_inventory` (`orchestrator.py`), and serialize it in `_curate_feature_summary` (`curator.py`). Two of those three are absent from the manifest. This is the single largest hidden-scope item in Phase B.

### (d) Does the always-emit design have a hole where the old hard-decline was load-bearing? — YES. This is the most serious structural finding.

The old `NavModelTitan.create_model` did essentially nothing (just logged) and `to_features` returned `[]` unconditionally, so it could not throw; attribution came from the orchestrator's `titan_present` branch reading `titan_in_fov`. The new design does heavy oops geometry in `create_model`/`_geometry_from_obs` and **requires it to succeed** to emit the feature.

If `create_model` or `to_features` raises on a pathological frame, the orchestrator's plugin sandbox swallows it: `_build_models` (orchestrator.py:628-638) **drops the model** on any exception ("skipping its features and annotations"); `_extract_features` (orchestrator.py:753-765) treats a raising `to_features` as zero features. Either way `all_features` is empty -> after the redesign deletes the `titan_present` sub-branch, the frame ends **`NO_FEATURES_EXTRACTED` with no `TITAN_LIMB` gate record** — precisely the "unattributable failure" Section 2.5's net invariant promises never happens, and a direct violation of the Phase E acceptance line ("every `off_edge`/`known_bad` frame is type-gated (hard-zero reliability) or fails a named technique gate").

The concrete trigger is real. Section 2.1 item 3 computes `theta` by taking the **minimum-incidence pixel over the incidence backplane's valid (surface-intercept) pixels**. On a badly-clipped frame (inventory bbox half or mostly off-frame), if the backplane meshgrid follows `NavModelBody.create_model` — which **clips the bbox to the extfov** (`nav_model_body.py:456-457`, `obs.clip_extfov`) — the clipped envelope bbox can contain zero surface-intercept pixels (only haze sky, or the corner of the frame). The min-incidence search then operates on an empty set. The plan's only degenerate-case handling (item 3) covers **near-zero phase** (`hypot < axis_min_offset_px -> theta=0`); there is **no specified fallback for an empty valid-pixel set** or for a `map_coordinates`/backplane failure. Note the predicted *center* survives this (it comes from the **unclipped** bbox midpoint — `nav_model_body.py:483-484` uses `u_min_unc`/`u_max_unc` — verified), but `theta` does not.

The plan needs one of: (i) specify the incidence meshgrid is built over the **unclipped** envelope bbox (oops backplanes evaluate fine at off-detector pixel coordinates, so surface intercepts remain available), and/or (ii) mandate that `_geometry_from_obs` never raises — emit a feature with `axis_degenerate=True`, arbitrary `theta`, and reliability that will hard-zero anyway. Without one of these, the always-emit contract silently breaks exactly on the clipped/off-edge frames the hard-zero condition #1 was written to catch.

Relatedly, this exposes an **ambiguity**: Section 2.1 item 3 says build the backplane "the way `NavModelBody.create_model` builds its `restr_bp`" (which clips), but robustness demands the unclipped bbox. The two readings diverge precisely on the frames that matter. I read it as needing unclipped; confidence medium, and the plan should state it outright.

### (e) Leftover plan references assuming the old decline/status flow? — None found.

I scanned the full plan for stale decline language describing the *new* behavior. Every "declines"/"emits no features"/"records the decline" occurrence is explicitly describing the *current* state being changed (nav_model_titan.py docstrings, the orchestrator special case, the dev-guide rewrites in Phase F). The `is_simulated -> []` retention is consistent (stated to stay until Phase D). The redesign reads internally consistent within the plan. Good.

---

## STANDARD SWEEP

### 1. AMBIGUOUS

1. **Incidence meshgrid: clipped vs unclipped envelope bbox** (detailed in (d) above). "the way `NavModelBody.create_model` builds its `restr_bp`" (which clips to extfov) vs. the robustness need for surface intercepts on clipped frames. Pick: unclipped bbox for the backplane. Confidence: medium.

2. **Where `W` and geometry are computed relative to `context`.** Item 6 defines `W = max(search_window_for_obs(context))`, but `search_window_for_obs` (nav_technique.py:126) requires a `NavContext`, and the manifest says `create_model` computes Section 2.1 geometry — `create_model()` takes no `context` (base signature, nav_model.py:117), and Phase B's `_geometry_from_obs(obs, config)` takes `obs`, not `context`. `search_window_for_obs` merely reads `context.obs.extfov_margin_vu`, so the model can read `self.obs.extfov_margin_vu` directly. Two readings: (a) compute geometry lazily in `to_features(context)`; (b) compute in `create_model` reading `obs.extfov_margin_vu` directly and ignore the `search_window_for_obs` reference. Both reach the same `W`. Pick: (b), reading the obs attribute directly. Confidence: high that it's harmless; the plan is just internally loose about `context` vs `obs`.

### 2. CONFUSING

1. **`W` = "search half-window" is a fixed instrument margin, not an uncertainty.** The plan spends a full paragraph (item 6) pre-empting this exact misread, yet then uses `W` for three distinct roles — offset search bound, annulus/mask along-track dilation (`+-W` in pass 1), and grid extent. A reader who internalizes "W = pointing uncertainty" will mis-size everything. The warning is good but the overloading remains a trap.

2. **Two "inner fractions" scaling two different radii.** `annulus_inner_fraction x r_env_px` (symmetry) vs `radial_inner_fraction x r_solid_px` (arc). The plan flags it ("do not conflate"), but it is exactly the kind of thing that gets conflated in a hurried Phase A.

3. **Symbol density in the recenter/double-count logic.** `c`/`c_sub`/`c'`/`d`/`q` plus "pass-2 `c_sub` REPLACES pass-1's while `d` accumulates" (Section 2.4). This is correct and even has a dedicated regression test (Phase A test 12), but Section 2.4's offset assembly is the single densest paragraph and will need re-reading against the table each time.

4. **`config_060_titan.yaml` "complete replacement" while `atmosphere_height` currently has no consumer.** Fine, but the reader must trust the "no code consumer, so safe" claim; I did not exhaustively verify nothing reads the *old* `config_060` shape beyond `atmosphere_height`. Worth a grep at implementation time.

### 3. INCORRECT

No factual repo errors found. I specifically verified and confirmed correct:
- `center_resolution(self, event_key, axis='u')` — signature and default `axis='u'` exact (introspected from oops); cartographic usage is single-axis default (`cartographic_model.py:141`).
- `stars_in_extfov(obs, config, *, catalog_name, mag_min, mag_max, radec_movement=None)` (catalog.py:414) — `mag_min` is keyword-only with no default, exactly as the plan warns.
- `UCAC4_SATURATION_VMAG_LIMIT = 8.0` (stars/saturation.py:91).
- `_compute_occluder_local` exists (nav_model_body.py:749) for extraction; `bodies_in_extfov` is module-level (nav_model_body.py:199).
- `search_window_for_obs -> tuple[int,int]` (nav_technique.py:126); `add_model_error_floor` (:399); `confidence_attributes: ClassVar[frozenset[str]]` (:283); `validate_registered_confidence_specs` (:535); `tukey_biweight_weights` (dt_fitting/weights.py:25).
- `BodyBlobNav` confidence pattern (`confidence_attributes` :703, `self.confidence_spec` consumed :871).
- All 9 diagnostics classes carry `CURATOR_FIELDS`; `test_diagnostics.py` parametrize list is hard-coded (9 entries) — the plan's "add to BOTH lists" is correct.
- `NavModelBodySimulated.instances_for_obs` builds one model per body with no TITAN exclusion (nav_model_body_simulated.py:303+) — Phase D exclusion claim correct.
- The three+one sim scenes exist exactly as named: `titan_haze_limb.yaml`, `titan_crescent_horns{,_noiseless}.yaml` in `sim_scenes/atmosphere/`, `haze_limb_base.yaml` in `sim_scenes/model_mismatch/`.
- `_lit_weighted_centroid_vu`/`_sub_solar_dir_vu` exist in `nav_model_body_base.py` (:372/:391) — the "do not use these" warnings point at real methods.
- Geometry self-consistency: `a_hat=(sin theta, cos theta)`, `theta=atan2(ve-v0, ue-u0)` correctly makes `a_hat` the unit vector toward the min-incidence (sunward) pixel; `c_hat=(cos theta, -sin theta)` is orthogonal (`a_hat . c_hat = 0`); offset assembly and `Sigma = M diag M^T` are dimensionally consistent in `(v,u)`.

### 4. MISSING

1. **`tests/spindoctor/nav_orchestrator/test_orchestrator.py`** — two tests exercise the deleted `titan_present`/`TITAN_UNSUPPORTED` path (see (b)); not in the manifest.
2. **`feature_summary.py` + `orchestrator._summary_from_feature`/`_build_inventory`** — required for the acceptance-criterion-3 breakdown serialization, but the manifest names only `curator.py` (see (c)).
3. **No fallback for degenerate geometry in `create_model`/`_geometry_from_obs`** (empty incidence-valid set / clipped-off-frame / backplane failure). The always-emit invariant depends on this never raising; unspecified (see (d)).
4. **`_FakeTitanModel` / `make_titan_feature` interaction with the deletion.** Phase B adds a `make_titan_feature` factory but the existing `_FakeTitanModel` (in test_orchestrator.py) exposes `titan_in_fov`; the plan doesn't say whether to delete `_FakeTitanModel` or repurpose it. Minor, but tied to #1.
5. **Confirmation that nothing reads the old `config_060` shape** beyond `atmosphere_height` before the "complete replacement." The plan asserts safety for `atmosphere_height` only.

### 5. FRANK OVERALL ASSESSMENT

This is a strong, near-executable plan — materially better than most I cold-read. The method spec (Section 2) is genuinely implementable: the geometry is self-consistent, the pass/mask/dilation table resolves the prose ambiguities, the config schema is complete with defaults, the dataclass signatures are concrete, and the collateral-surface manifest is remarkably thorough (it correctly anticipated the sim-scene TITAN rename, the hand-enumerated diagnostics/calibration/snr-characterization lists, and the reliability-gate sync). My spot-checks of ~20 repo claims found zero factual errors — unusual.

**The redesign has NOT fully converged.** Revision 11's newest change (the `TITAN_UNSUPPORTED` deletion) is where the remaining structural problems cluster, and they are the kind a sixth review round would still surface:
- The always-emit invariant is asserted but not defended against the geometry-computation-can-throw reality of the new heavy `create_model` (finding (d)). This is a *behavioral* gap, not a cosmetic one: it re-opens the "unattributable Titan failure" the whole redesign exists to close, on exactly the off-edge/clipped frames Phase E must handle.
- The JSON-serialization path for the breakdown is under-scoped by two files (finding (c)), and acceptance criterion 3 depends on it.
- One test file that the deletion breaks is unlisted (finding (b)).

**Phase B worries me most** — by a wide margin. It is already the "vertical slice" carrying the model, the technique, the occluder-helper refactor (which must stay bit-identical), all the registrations, and the config. On top of that it silently owns: the geometry-robustness contract (d), the three-file serialization extension (c), and the orchestrator-test rewrite (b) — none fully specified. Phase A (pure array math, twelve well-specified test families) is the safe one; Phase D's new rendering math in `atmosphere.py` is large but honestly budgeted ("this is new rendering math ... not key plumbing"). Phase B is where the plan's optimism and the code's reality collide.

Recommendation before starting Phase B: (1) decide unclipped-bbox for the incidence backplane and add an explicit "geometry-degenerate -> emit reliability-0.0 feature, never raise" clause to Section 2.5; (2) expand the serialization manifest row to name `feature_summary.py` and the orchestrator inventory builders; (3) add `test_orchestrator.py` to the deletion manifest. None of these are redesigns — they are three precise patches to an otherwise-executable plan.
