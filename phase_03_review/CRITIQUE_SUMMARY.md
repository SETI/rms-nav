# Phase 3 Code Review — Executive Summary

## Scope

Phase 3 closed the foundational gaps that should have shipped in Phases 0–2 plus enough per-instrument config wiring so a real image can run through the pipeline without hard-coded defaults misclassifying it. A second round added INFO+DEBUG logging across every NavModel, NavTechnique, and the orchestrator so an operator can read the per-image log and understand what the software is doing, what metadata it derived, and why a given confidence value fell out the way it did.

## What shipped

- Renumbered every shipped `config_NN_*.yaml` → `config_NNN_*.yaml` with three-digit prefixes (`010`/`020`/.../`950`); ring catalogues moved to the `3N0` band, per-instrument blocks to the `4N0` band.
- Converted every radian-valued field in `config_07_bootstrap.yaml` (now `config_070_bootstrap.yaml`) to degrees, with `_deg`-suffixed names.
- Added per-camera `data_units`, `noise:`, `image_quality_thresholds:`, `mag_offset:`, `source_image_filter:`, `fit_camera_rotation:`, `max_rotation_deg:` blocks to every shipped `config_4N0_inst_*.yaml`. Cassini ISS additionally ships a `cassini_iss_calib:` block; the loader picks it when the filename contains `_CALIB` so raw-DN and CALIB I/F products use distinct thresholds.
- Added `nav.nav_orchestrator.instrument_config.instrument_settings_from_obs(obs)` translating the per-instrument block into a typed `InstrumentSettings` dataclass.
- Replaced the hard-coded `_instrument_full_well_dn` constant with a config consumer that reads `obs.inst_config['noise']['saturation_dn']`. Calibrated-IF instruments without preserved raw-saturation flags emit a one-line WARNING and an empty saturation mask.
- Added per-image source-image filter application reading `obs.inst_config.source_image_filter`.
- Wired `STATUS_REASON_INFO_TEMPLATE` through every `NavResult.failed` short-circuit so failure paths emit per-status-reason INFO lines via the new `_fail` helper.
- Populated `Provenance` via the new `collect_provenance_metadata` helper (git SHA, loaded SPICE kernels, sha256 of static-data YAML bytes).
- `NavModelStars.to_features` reads `config.<camera>.mag_offset.{fallback_combo, mag_offset_table}` per star instead of the legacy hard-coded `0.0`.
- Each `NavTechnique` declares `confidence_spec` + `confidence_attributes`; `validate_registered_confidence_specs()` runs at `Config.read_config` time. Validation caught a real bug: `BodyTerminatorNav` referenced `mean_phase_angle_factor` and `mean_albedo_penalty` without declaring them.
- `evaluate_sigmoid_combination(..., return_breakdown=True)` returns `(confidence, ConfidenceBreakdown)` carrying per-term raw / normalized / alpha / contribution plus the sigmoid argument and the `hard_zero` / `hard_cap_applied` flags. `nav.nav_technique.nav_technique.log_confidence_breakdown(logger, breakdown)` logs the breakdown at DEBUG always and at INFO when `confidence <= 0.1`.
- `Config._load_yaml` strips every mapping key starting with `_`; documentation-only `_sources` citation blocks live alongside values in source for human review without bloating runtime config.
- Initial `config_220_body_shape.yaml` with 10 bodies. Per Part 0 §74 anti-hallucination rule, every numeric field is `null` paired with `'PLACEHOLDER — no source found, calibrate in Phase 10'`; the runtime fallback (10 % radius default + reliability cap 0.3) handles `null` values.
- Added INFO+DEBUG logging across every NavModel, NavTechnique, and the orchestrator. `NavModelStars.create_model` emits the legacy-format per-star INFO listing (catalog/name, U+/-move, V+/-move, VMAG/JBMAG/JVMAG, SCLASS, TEMP, CONFLICT) via `_star_short_info`. NavModel section headers via `logger.open` carry the per-instance context (e.g. `CREATE BODY MODEL FOR: MIMAS`) so inline body-name prefixes are dropped inside sections. The rings model logs the visible ring-plane radial range `[min, max] km` immediately after the backplane query, before per-feature filtering.
- Ensemble failure paths now report the actual measured values (combined-confidence vs. min, gap vs. agreement_gap with best/runner-up summed confidences, sigma vs. tier max, all-techniques-spurious technique-name list, unobservable-offset input count) instead of generic "below threshold" messages.
- Fixed two duplicated bugs uncovered while logging surfaced them: `psf_sigma_px` now handles the real `psfmodel.GaussianPSF` per-axis `sigma_x` / `sigma_y` interface (legacy code searched for a single `sigma` attribute that doesn't exist on the shipped class); `nav_model_body._star_psf_sigma` and `detection._psf_sigma` wrappers were deleted in favour of calling `psf_sigma_px` directly.

## Verification

- `ruff check src tests` clean.
- `ruff format --check src tests` clean.
- `mypy src tests` clean — 256 files, 0 issues.
- `pytest -n auto --dist=loadfile` — 1043 fast tests pass; the slow `tests/nav/inst/` integration tests (require remote PDS holdings) pass when run individually.
- `sphinx-build -W -b html docs docs/_build` clean.
- `pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md` clean.
- `./scripts/run-all-checks.sh` green end-to-end.

New tests:
- `tests/nav/config_files/test_body_shape_citations.py` (5)
- `tests/nav/config_files/test_config_load.py` (4)
- `tests/nav/nav_orchestrator/test_instrument_config.py` (10)
- `tests/nav/nav_orchestrator/test_provenance.py` extended (3)
- `tests/nav/nav_technique/test_nav_technique.py` extended (2 — `validate_registered_confidence_specs` shipped + bogus)
- `tests/nav/inst/test_inst_cassini_iss.py` extended (1 — `_CALIB.IMG` selects calibrated-IF block)

## Findings

### Critical

_None._

### High

- **H1 — Per-instrument `noise` / `mag_offset` / image-quality threshold values are PLACEHOLDERs.** Phase 3 wires the schema and consumers but every field that is not a hard-fact ADC saturation DN carries a `# PLACEHOLDER` marker; calibration lands in Phase 10.
- **H2 — `config_220_body_shape.yaml` ships with `null` values across every body.** Per the anti-hallucination rule, citations must be sourced from in-session `WebFetch` lookups; this PR did not perform those fetches. Runtime fallback handles `null` values; Phase 10 calibration replaces them.
- **H3 — `_RING_EDGE_CONFIDENCE_SPEC` saturates to confidence 0 on real Saturn ring fits.** The `per_edge_dt_rms_summed` term (no divisor, no cap) reaches ~150 on multi-edge images and pushes the sigmoid argument to ~ −300 even for clean RMS-0.2-px fits. The `log_confidence_breakdown` helper now surfaces this clearly when it fires; Phase 10 retunes the alpha / divisor or replaces the term with `dt_fit_rms_px` (the path body-limb / body-terminator already use).

### Medium

- **M1 — `data_units == 'calibrated_if'` saturation path emits a WARNING and an empty mask.** Acceptable per the design (no useful raw-saturation signal on calibrated products) but worth documenting in operator-facing release notes.
- **M2 — `Config.read_config` validation is gated on the technique package having been imported.** Validation skips silently when `nav.nav_technique` has not been imported. Phase 4 should ensure the orchestrator entry-point imports the technique package eagerly so this skip cannot mask a real misconfiguration in production.

### Low

- **L1 — `_strip_underscore_keys` walks the entire YAML tree.** O(N) per `_load_yaml` call, dwarfed by YAML parsing time; acceptable.
- **L2 — `_resolve_git_sha` swallows every subprocess error class.** The fallback (return `None`) is the correct behaviour for non-git trees.

## Ready-to-run AI prompts

Each finding above ends in a sentence that names the action. None of the findings are critical/high enough to block phase close. The plan's Phase 10 calibration step explicitly addresses H1/H2/H3.
