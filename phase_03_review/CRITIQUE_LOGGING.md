# Phase 3 Code Review — Logging Best Practices

Review against `.cursor/rules/logging_best_practices.mdc`.

## Strengths

- **`STATUS_REASON_INFO_TEMPLATE` wired through every failure path.** Every `NavResult.failed` site now flows through the new `_fail` helper which emits the per-status-reason INFO lines via `self._logger.info(...)`. Operators see the failure narrative in the per-image log alongside the section header.
- **`%`-as-format-placeholder hazard removed.** The `IMAGE_OVEREXPOSED` and `MISSING_DATA_DOMINANT` templates previously contained literal `%` characters that pdslogger interprets as format placeholders; rewritten as plain prose ("most pixels at full-well DN", "missing-data marker dominates") so the templates can be emitted as-is.
- **No `import logging` introduced anywhere in the new core code.** Every new orchestrator / model / technique log site uses `self._logger` (NavBase) or the module-level `IMAGE_LOGGER` exported from `nav.config.logger`.
- **Per-NavModel and per-NavTechnique section headers via `logger.open(...)`.** `NavModelBody` opens `'CREATE BODY MODEL FOR: <BODY>'` and `'EMIT BODY FEATURES: <BODY>'`; `NavModelRings` opens `'CREATE RINGS MODEL'` and `'EMIT RINGS FEATURES'`; `NavModelStars` opens `'CREATE STARS MODEL'` and `'EMIT STARS FEATURES'`; every `NavTechnique.navigate` body opens `with self.logger.open(f'TECHNIQUE: {self.name}'):`. Inside a section the per-instance prefix is dropped (e.g. no "Body MIMAS:" prefix because the section header already says it).
- **INFO is for the operator-readable narrative + interesting metadata.** Body-model INFO covers subsolar / sub-observer lat-lon, phase angle, subject range, predicted diameter, km/px at limb, visible-lit fraction, silhouette overflow, guaranteed-visible flag. Rings-model INFO covers planet, surviving feature count, km/px radial, subject range, ring-plane radial range visible in image, edge / annulus / straight-edge counts. Stars-model INFO covers dedup'd star count, smear vector + magnitude, body / ring conflict counts, full per-star listing, low-SNR / over-smear skip counts. Technique INFO covers feature consumption, final converged offset / RMS / inliers / confidence, spurious / at_edge flags. Orchestrator INFO covers image-classifier verdict, NavModel build list, reliability-gate kept / gated counts, pass-1 / pass-2 entry markers and result counts, pass-1 prior offset + confidence, final offset + sigma + status + confidence + rank + technique-fusion count.
- **DEBUG is for internal details.** Per-model bbox / size_ok / shape-class hint / backplane oversample factors. Per-technique coarse-NCC offset / vertex sigma range / search window / LM iteration count / supplementary diagnostics. Orchestrator gated-feature breakdown by feature type / per-technique offset / spurious / at_edge.
- **Failure paths report actual values, not just threshold names.** The ensemble logs combined-confidence-below-min, no-tier-earned, conflicted (gap, best-summed, runner-up summed, multipliers), all-techniques-spurious (technique-name list), and unobservable-offset (input count) with the actual measured numbers.
- **Confidence-formula breakdown logger.** `log_confidence_breakdown(self.logger, breakdown)` logs the per-term raw / normalized / alpha / contribution at DEBUG always; promotes the breakdown to INFO when `confidence <= 0.1` so calibration bugs surface in the default operator log. A `hard_zero_if` firing logs `'Confidence forced to 0 by hard_zero_if[%r]=True'` at INFO with the offending attribute name.
- **Star-list emission preserved from the legacy NavModelStars.** `_star_short_info(star)` produces a single grep-friendly line per surviving star with catalog/name, U+/-move, V+/-move, VMAG, JBMAG, JVMAG, SCLASS, TEMP, CONFLICT.

## Notes

- `provenance._resolve_git_sha` and `_resolve_spice_kernels` use `try/except` returning `None` rather than logging errors. This matches the project's convention that provenance metadata is best-effort: a missing git binary or absent SPICE installation should not pollute the per-image log with a WARNING.
- `_build_saturation_mask` emits exactly one WARNING per image when `data_units == 'calibrated_if'` and no `saturation_dn` is configured — the design's expected single-line WARNING.
- The legacy `_FakeMutableStar` test fixture lacks `johnson_mag_b`, `johnson_mag_v`, and `temperature`. `_star_short_info` uses `getattr(..., None) or 0.0` for these so the helper handles minimal fixtures gracefully.

## Open items

- **Phase-4 follow-up:** add `tests/nav/nav_orchestrator/test_log_structure.py` covering the per-status-reason INFO sequence for every status reason. Phase 3's tests cover the wiring; Phase 4 formalises the assertion matrix.
