<!-- Frozen snapshot 2026-07-25: fifth review round on plans/TITAN_NAV_PLAN.md
(revision 7) — a repo-wide collateral-surface sweep by three parallel
fresh-context agents over tests/, util/ + the CLI/statistics layer, and all
documentation, hunting surfaces the Titan work must touch that the plan did
not name. Findings were folded into the plan's revision 8 the same day
(Section 3 table rows, Phase B/C/D/E/F additions, Section 9 items 8-9). The
three reports are preserved verbatim below. Not maintained. -->

# Collateral-surface sweep: tests, utilities, documentation

## Report 1 — tests/ tree

Skimmed plan Sections 3 and 6. The plan already names `tests/integration/sim_sweep.py`, the sim scene catalog, `test_information_boundary.py`, and `test_boundary_static_guard.py`. Everything below is what it must ALSO touch (or can trust to stay silent).

### Category 1 — enumeration / exhaustiveness surfaces

WILL SILENTLY UNDER-COVER (no failure, but the new class goes untested) — must edit:
- `tests/spindoctor/nav_technique/test_diagnostics.py:20-32` and `:40-52` — two `@pytest.mark.parametrize` lists hard-enumerate the 9 diagnostics classes and assert (a) each constructs with defaults and has `CURATOR_FIELDS`, (b) `CURATOR_FIELDS` keys cover every dataclass field. `TitanHazeDiagnostics` is NOT auto-discovered here. Titan work must add it to both lists, or its `CURATOR_FIELDS`-completeness invariant is never checked — a mismatch would only surface later in `curator.assert_diagnostic_fields_present` on a live navigated frame.

Constraints the plan already satisfies (pass automatically once the class is registered/well-formed) — no edit, but load-bearing:
- `tests/spindoctor/nav_technique/test_nav_technique.py:77` `test_registry_contains_only_shipped_techniques` — passes automatically.
- `:134` `test_validate_registered_confidence_specs_passes_for_shipped_techniques` — the exhaustiveness gate for `confidence_spec` vs `confidence_attributes`; a divergence fails process-wide.
- `tests/spindoctor/nav_technique/test_confidence_config.py:19` — the new config_510 `TitanHazeNav` block must merely parse.

Needs NOTHING — verified silence:
- `tests/spindoctor/feature/test_feature_type.py:8-29` — enumerates 9 feature types incl. an exact count; `TITAN_LIMB` is ALREADY member #8; the plan adds no enum value. Passes unchanged.
- `tests/spindoctor/support/test_status_reason.py:8-38` — 20 status reasons incl. `TITAN_UNSUPPORTED`; the plan adds none. Wording-only edits don't touch names.
- `tests/spindoctor/nav_orchestrator/test_status_reason_info.py` — non-empty templates; passes.
- `tests/spindoctor/feature/test_geometry.py` / `test_flags.py` — per-dataclass tests, no exhaustive-union assertion (new variants uncovered unless given their own tests, but nothing breaks).
- `tests/spindoctor/feature/test_feature.py:135` — checks two specific breakdown defaults; the two new fields keep `= None` defaults and it passes untouched.

### Category 2 — curated-library regression machinery

Image library — no Titan today:
- No existing library frame contains Titan (grep of every sidecar). No `expected.*` to protect; the never-edit rule has no live target.
- `tests/integration/sidecar.py:37-57` `DECLARED_SCENE_CLASSES` has NO Titan class; `ALLOWED_STATUS_REASONS` derives from `NavStatusReason` (fine); `Expected.primary_technique` is a free string (fine). Phase E's curated Titan frames must either reuse an existing class or add a new one, or `test_image_library.py:45 test_class_subdirectories_are_subset_of_declared` fails on the new directory.
- `images/README.txt:263-278` calls the deferred technique `TitanNav`; the plan's technique is `TitanHazeNav`. Update in Phase E/F.

Sim curated regression — where "expected.* must not be edited to pass" actually bites:
- Root cause: `nav_model_titan.py:89` returns `[]` for simulated obs, and `nav_model_body_simulated.py:330-347` builds a model for EVERY body with no TITAN exclusion. Today a sim body named `TITAN` is navigated by `NavModelBodySimulated`. When Phase D's `NavModelTitanSimulated` replaces the branch, routing flips to `TitanHazeNav` for:
  - `tests/integration/sim_baselines/titan_haze_limb.json`, `haze_limb_base.json`, `titan_crescent_horns.json` (+ `titan_crescent_horns_noiseless.json`) — `test_sim_baselines.py:52` does EXACT-equal reproduction; these will fail and must be regenerated via `python -m tests.integration.update_sim_baselines` with diff review, not hand-edited.
  - `tests/integration/sim_scenes/atmosphere/titan_haze_limb.yaml` (`expected: success/medium`), `titan_crescent_horns.yaml` (`expected: failed`, rationale comment premised on "BodyBlobNav is the only surviving technique") and `model_mismatch/haze_limb_base.yaml` — `test_sim_expected.py:42` asserts the blocks; the rationale is invalidated once `TitanHazeNav` exists. Re-derive expected blocks + rationale deliberately.
  - The `atmosphere_haze` sweep collision (subtle): `sim_sweeps/atmosphere_haze.yaml` uses `base_scene: model_mismatch/haze_limb_base.yaml`; its PURPOSE is a model-mismatch axis (haze-blind navigator fits a haze-softened limb; `test_sim_sweeps.py:486-497` asserts it). Routing to the haze-AWARE `TitanHazeNav` destroys the axis. Decide: rename the body away from `TITAN` in these fixtures (keeping them as generic-body model-mismatch records), or accept and re-pin the sweep.
- `tests/spindoctor/nav_model/test_sim_model_selection.py` uses non-TITAN names so it won't fail — but there is a real double-model risk with no guard: `NavModelBodySimulated` must be taught to exclude `TITAN` while `NavModelTitanSimulated` claims it; add selection coverage mirroring `test_nav_model_titan.py:69 test_body_model_excludes_titan`.
- New base scene bookkeeping: the Phase D `titan_haze` base scene needs a matching baseline JSON (`test_sim_baselines.py:39 test_every_scene_has_a_baseline`, `:45 test_no_orphan_baselines`); `test_sim_scenes.py` auto-discovers it (`atmosphere` class already declared) but `test_scene_renders` requires the new render math to produce `img.max() > 0`.

### Category 3 — suites/harnesses that run every technique or that new render keys touch

- `tests/integration/test_sim_perf.py:135-170, 324-331` — existing haze-body cold-render budgets (2 s @512, 8 s @1024). The new atmosphere math must be gated behind key presence or it silently regresses the budget.
- `tests/integration/technique_snr_characterization.py:63-72` `_TECHNIQUES` — hard-coded per-technique list feeding the simulator-report response curves (a `__main__` harness). `TitanHazeNav` silently omitted unless added with a Titan base scene.
- `tests/integration/sim_realism.py:114-130` — `_BODY_MODELS` globs and `MODELS_FOR_CLASS` have no `titan:*` / `titan_sim:*`; extend when Phase E library frames land.
- `tests/spindoctor/cli/stats/test_stats.py` — needs NOTHING (technique names/feature types are pure data).
- `test_autonomous_nav.py` — sidecar strings, nothing until Phase E. `test_baselines.py` — the pinned frame is non-Titan. The other sim suites use non-Titan bodies; silence.

### Category 4 — conftest fixtures

- `tests/spindoctor/nav_technique/conftest.py` — feature factories exist for LIMB_ARC/TERMINATOR_ARC/RING_EDGE/STAR only; add a `make_titan_feature` factory. `FakeObs` exposes only `extfov_margin_vu` (consistent with the plan's Phase B statement).
- `nav_orchestrator/conftest.py`, `nav_model/conftest.py` — needs NOTHING.

### Nuance on the two boundary tests the plan already names

The new haze keys live inside `bodies.atmosphere`, a single atomic entry in `TRUTH_KEYS` / `_TRUTH_SAMPLES` (`test_information_boundary.py:84-90`); the completeness check compares at block granularity, so omitting the new sub-keys fails nothing — they would be silently un-exercised for leakage. Add them to `_TRUTH_SAMPLES['bodies.atmosphere']` explicitly. `test_boundary_static_guard.py`'s denylist is about `obs.sim_*` attributes; unaffected.

---

## Report 2 — util/ + CLI/statistics layer

Ground truth first: the metadata JSON, the SQLite stats system, and the confidence-fit gates script are data-driven and absorb a new technique/feature/status automatically. The real surfaces cluster in the calibration fit scripts and the manual-nav composite overlay.

### util/calibration/ — per-technique lists the plan missed

- `util/calibration/fit.py:56-66` `TECHNIQUES` tuple (9 techniques, no Titan); `main()` fits confidence-spec alphas only for these names and writes config_510. Add `TitanHazeNav` or the Phase-B placeholders stay un-recalibrated forever. Companions: `TRANSFORM_OVERRIDES` (86-106) and `SIGN_BY_FEATURE` (115-142) want Titan-term entries (`arc_residual_rms_px: '-'`, quality terms `'+'`).
- `util/calibration/fit_floors.py:28-35` `TECHNIQUES` tuple — where `model_error_floor_px` is actually fitted (2-sigma coverage); add `TitanHazeNav`.
- `util/calibration/scene_gen.py:52-60` `FAMILIES` + `735-743` `_GENERATORS` — one generator per technique regime; no Titan. Add a `titan` family + `gen_titan` or the campaign yields zero Titan rows and both fitters skip the technique.
- Needs nothing (verified data-driven): `fit_gates.py`, `collect.py` (`only_techniques='*'`), `library_crosscheck.py`.

### util/cohort_curation/ — no Titan class anywhere

The 17-class taxonomy lives in three synchronized places (COHORT_CURATION_PLAN table, `sidecar.py` `DECLARED_SCENE_CLASSES`, the structural-invariants test); none has a Titan class; Titan appears only as a radius constant. Phase E deliberately bypasses this pipeline (vendored legacy cohort), so nothing is required by the plan — but a `titan_haze` scene class (taxonomy + scan builders in `scan_stage_a.py` + primary-technique rubric in `build_sidecars.py`) is the natural library-growth follow-up and belongs in Section 9.

### util/agreement/ — concrete shape of the deferred #225 channel

`analyze.py:53-58` `_TECH_TO_INSTANCE` maps only four body/ring techniques (no star technique at all); `_instance_offsets` silently drops unmapped techniques; the rotating-basis and pivotal-pair wiring is hardcoded; `scene_gen.py` `FAMILIES` and `collect.py` `build_runs` have no Titan entries. Correctly deferred by the plan; recorded so Section 9 item 8 can point at the real files.

### Statistics system (src side) — needs NOTHING

`schema.py` stores technique/feature/status as plain TEXT; `ingest.py` has no allow-list; `report_sections.py` sorts status reasons dynamically and classifies a `titan:TITAN` source as `single-body` with no code change; `_CONFIDENCE_TIERS` is ensemble tiers, unaffected.

### Navigation output metadata — one latent guardrail, one real gap

- `curator.py:192-246` `build_metadata_dict` is additive/data-driven for features and techniques; no JSON schema validation exists.
- Load-bearing guardrail: `curator.py:80-104` `assert_diagnostic_fields_present` RAISES at metadata-build time if a diagnostics class lacks a complete `CURATOR_FIELDS` — the plan already requires the field, but every Titan-navigated image crashes ingestion if it's wrong.
- Real gap: model-level metadata (`NavResult.model_metadata`) is NOT serialized by `build_metadata_dict` at all — so the plan's `decline_reason` never reaches the emitted JSON as specced. The plan must decide to serialize it (it wants to: acceptance criterion 3 reads as a product property).

### Manual-nav path — genuine unaddressed gap

- `feature/composition.py:111-184` `compose_dialog_overlay` renders exactly four geometry cases (template, `vertices_vu`, `BodyBlobGeometry`, `StarGeometry`); `TitanHazeGeometry` has none of these — a `TITAN_LIMB` feature is silently skipped.
- `nav_technique_manual.py:110-135` `is_feasible` mirrors the same enumeration; a Titan-only frame returns `feasible=False, 'no_renderable_features_for_manual_nav'` — manual navigation cannot run on a Titan-only frame at all.
- Fix: a `TitanHazeGeometry` branch in both (envelope-circle outline at `r_env_px`, the `BodyBlobGeometry` pattern). Phase C's `to_annotations` covers only the annotation layer, which the dialog renders separately — not the draggable model overlay.
- Needs nothing: `summary_png.py` renders annotations generically; `NavTechniqueManual.accepts_feature_types = frozenset(NavFeatureType)` already includes `TITAN_LIMB`; `ui/library_entry.py` uses operator-filled placeholders.

---

## Report 3 — documentation

Phase F already commits to: user-guide Titan section; dev-guide technique page; API-reference stubs; the simulator report; diagnostics/reliability field docs; and reconciliation of the five plans at named spots. Everything below is what those named spots do NOT cover.

### /seti/newnav/CLAUDE.md
- `:98` — "`NavModelTitan` (a registered placeholder that emits no features)" — false; reword.
- `:99-100` — "Each family has a `NavModel*Simulated` sibling" — becomes fully true after Phase D; confirm Titan intended in the claim.
- `:101-109` — the technique enumeration must add `TitanHazeNav` (tenth autonomous technique).
- `:147` — `config.titan` listing fine; config_060 now carries a full `navigation:` block.

### docs/user_guide/
- `user_guide_navigation.rst:390-394` — "the model emits no features ... fails with `titan_unsupported`" — rewrite (per-frame decline, not permanent).
- `:778-781` — "Three model families ship out of the box" — now four navigating families.
- `:783-785` and `:415-426` — technique enumerations gain `TitanHazeNav`.
- `docs/introduction_configuration.rst:198-202` — `--nav-techniques` name list gains `TitanHazeNav`.
- `user_guide_statistics.rst:218` — feature-provenance family list gains `titan`.
- No user-guide config-key table exists for the `titan` section (star/body/ring each have one); the new `titan.navigation.*` keys need a list-table, not just prose.

### docs/dev_guide/ (the biggest gap — model-side pages beyond the Phase-F "technique page")
- `dev_guide_navigation_models_titan.rst` — entire page false (never-emits, wrong planned method, `titan_in_fov` always-True framing, logs and metadata claims). Full rewrite.
- `dev_guide_navigation_models_titan_simulated.rst` — entire page says the class "is reserved without an implementation" / "no source file". Full rewrite after Phase D.
- `dev_guide_navigation_models_bodies.rst:18-21`, `dev_guide_navigation_models_body.rst:16-19, 331-332`, `dev_guide_navigation_models.rst:38-40` — "records a no-result" clauses false; "builds no NavModelBody" stays true.
- `dev_guide_class_hierarchy.rst:400-402, 424-429, 461-464` — Titan paragraphs false; `:117-126` mermaid diagram gains `TitanHazeNav` node + inheritance edge (~`:236`); `:483-484` geometry-variant enumeration gains `TitanHazeGeometry`; `:487-489` flags enumeration gains `TitanHazeFlags`.
- `dev_guide_annotations.rst:141-143` — "returns an empty collection" false after Phase C.
- `dev_guide_familiarization.rst:284-295` — tour descriptions false; update link text.
- `dev_guide_orchestrator_orchestrator.rst:31, 245-248` — "emits no navigable features by design" -> "when the Titan model declines for this frame".
- `dev_guide_config_and_static_data.rst:68-70` — "single `atmosphere_height` value ... not consumed" — all false; `:149` description reflects expanded schema.
- `dev_guide_techniques_body_titan.rst` — page titled "Titan (unsupported)", names a per-filter-profile planned method; replace with the new technique page and fix the `dev_guide_techniques.rst:64` toctree (family is Titan, not Body).
- `dev_guide_simulator.rst:116-120` capability-envelope row and `:1297-1372` sim-atmosphere section — extend for the new keys; `:1343-1347` "a navigator never learns the haze exists" — caveat that this describes body nav, not `TitanHazeNav`. The `titan_crescent_horns` / `titan_haze_limb` fidelity-record prose stays valid (do not conflate with the new `titan_haze` scene).

### docs/api_reference/
- `api_nav_technique.rst` — add `.. automodule:: spindoctor.nav_technique.nav_technique_titan_haze`.
- `api_nav_model.rst:56` — add `nav_model_titan_simulated` automodule (the other three families list their `*_simulated`).
- `api_feature.rst` etc. — autodoc; flows from source. Needs nothing.

### docs/introduction*.rst and README.md
- `README.md:42` — "Star-based, body-based, and rings-based navigation" — add the Titan/haze family.
- `introduction_overview.rst`, `introduction.rst`, `index.rst` — needs nothing.

### The five plans/ files — beyond the Phase-F named spots
- VALIDATION `:41` — "(e.g. Titan)" as the scope-out example; stale.
- VALIDATION `:809-834` (WS-7) — the "Implement" bullet describes per-filter haze-top altitude profiles + a DT/edge technique — a DIFFERENT method from the French method actually built; rewrite, don't just mark done.
- VALIDATION `:1120` — capability-matrix row 6a resolves to implemented+validated; `:1074-1075` and `:1224-1225` — "decision gates first" framing stale.
- PROGRAM_PLAN `:255-256`, `:282`, `:392` — interim-wording flips (mostly inside named spots; exact sentences flagged).
- ENGINEERING_PLAN `:149-150` — wording flips to shipped.
- OPERATOR_PLAYBOOK `:21-22` — the #60 decision sentence.
- COHORT_CURATION `:268-291` — 17-class/47-image budget table and structural-invariants minima if a Titan class is added; `:288` negative_cases — verify no case is a now-navigable Titan-only frame.
- Deliberate NON-changes verified: the #328 haze-crescent body-nav family references, and #344 (haze brightness is a module constant) — Phase D does not add photometric brightness variation, so #344 remains open and accurate; confirm rather than close.

### Surfaces checked that need NOTHING
`docs/introduction*.rst`, `docs/index.rst`; autodoc-only API pages; `user_guide_backplanes/pds4_bundle/reprojection/simulated_images` + the four instrument appendices; `dev_guide_orchestrator_ensemble.rst` sim-scene reference; README beyond line 42; COHORT_CURATION outside the budget table.
