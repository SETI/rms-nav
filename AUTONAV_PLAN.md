# Autonomous Navigation Overhaul — Design Plan

## Part 0 — Audit corrections (2026-04-26)

These supersede any conflicting text below. Where a Part 1–16 section
disagrees with this list, this list wins; the inline text has been edited
where possible but cross-references still abound.

### Naming (apply uniformly)

Every public new type carries the `Nav` prefix to match the existing
`NavFeature` / `NavTechnique` / `NavModel` / `NavBase`. The renames:

| Old name (in Part 1–16) | New name |
|---|---|
| `FeatureType` | `NavFeatureType` |
| `FeatureGeometry` | `NavFeatureGeometry` |
| `FeatureFlags` | `NavFeatureFlags` |
| `FeatureExtractor` | `NavFeatureExtractor` |
| `FeatureSummary` | `NavFeatureSummary` |
| `FilterSpec` | `NavFilterSpec` |
| `FilterKind` | `NavFilterKind` |
| `FeasibilityReport` | `NavFeasibilityReport` |
| `TechniqueDiagnostics` | `NavTechniqueDiagnostics` |
| `ImageClassifierResult` | `NavImageClassifierResult` |
| `ReliabilityBreakdown` | `NavReliabilityBreakdown` |
| `StatusReason` | `NavStatusReason` |

Field renames inside `NavFeature` to avoid shadowing Python builtins
(Ruff rule `A`):

| Old field | New field |
|---|---|
| `id` | `feature_id` |
| `type` | `feature_type` |
| `range` (subject_range) | `subject_range_km` (units in name) |

Same naming convention applies inside `NavResult`: `status_reason: NavStatusReason`
(typed enum, not `str`).

### Settled decisions from this audit

1. **Pass-1 results in final ensemble.** Pass-1 results are intentionally
   used twice: once to form the prior consumed by pass-2, and once as
   direct inputs to the final ensemble. The information-form combine is
   correct under this; pass-1 is not double-counted in any harmful sense
   because the prior given to pass-2 is a starting point for refinement,
   not a substitute. Document this in Part 4 explicitly so an implementer
   doesn't "fix" it.

2. **`BodyDiscCorrelateNav` compositing = Z-buffer paint.** Sort bodies
   by `subject_range_km` ascending; iterate and paint each body's template
   into the combined image; closer body's nonzero pixels overwrite farther
   body's. Combined mask = OR of per-body masks.

3. **Conflicted-result penalties compose multiplicatively.** When the
   conflicted branch fires, both `disagreement_penalty` (default 0.7) AND
   `conflicted_confidence_multiplier` (default 0.3) apply:
   `final_confidence = combined × 0.7 × 0.3 = combined × 0.21`. Both knobs
   stay; the YAML schema just makes the composition explicit.

4. **`confidence_rank` adds `conflicted` as a 5th rank.** Ranks become
   `{high, medium, low, conflicted, failed}`. Downstream consumers
   (backplanes, PDS4 bundles) refuse `conflicted` independently of `low`.
   Update the JSON schema, `tier_thresholds` YAML, and the rank-derivation
   table accordingly.

5. **Phase-5 placeholder constants get inline comments.** Each placeholder
   value in Phase-1 YAML carries a `# PLACEHOLDER — calibrate in Phase 5`
   comment; PR review enforces that no `PLACEHOLDER` markers ship to
   production. No CI gate is added.

6. **`LimbRefineNav` is dropped.** Remove from Part 8 file list.
   `BodyLimbNav` does its own subpixel refinement after the coarse DT
   match. No separate prior-required limb technique; no Part 13b deferral.

7. **`'noisy'` is a flag, not a class.** Drop `'noisy'` from
   `NavImageClassifierResult.image_class` Literal. `noisy` lives only in
   `flags: list[Literal[..., 'noisy', ...]]`. A noisy-but-clean image is
   `image_class='clean', flags=['noisy']`.

8. **`BodyTerminatorNav` weighting is per-body uniform.** All terminator
   vertices for a given body share `1 / sigma_normal_per_vertex_px²`.
   Drop the "per-pixel albedo-uncertainty" language. `albedo_variation`
   in `config_220_body_shape.yaml` stays per-body scalar.

9. **`only_models` glob negation extends `_filter_models`.** The
   existing `_filter_models` in `src/nav/nav_technique/nav_technique.py:71`
   is extended to parse leading `!` as exclusion, gitignore-style.
   Existing technique callers pass non-negation patterns unchanged.

10. **All-rank-1-same-null-direction → `status='failed'`.** When the
    precision-weighted combine's `W = sum(weights) == 0` (every input
    covariance shares one null direction), return
    `NavResult.failed(status_reason='unobservable_offset')`. New
    `NavStatusReason` value. Add to the table; update the count to 15.

11. **`pipeline_run_iso8601` is excluded from the byte-identical
    contract.** Document in Part 12.1: every Provenance field except
    `pipeline_run_iso8601` is byte-identical for identical inputs;
    regression-baseline comparison strips that field.

12. **`NavModel.create_model` drops `always_create_model`.** Remove
    from the new public contract; if existing implementations use it as
    a private short-circuit, keep that as an internal detail.

13. **`RING_ANNULUS` is one per planet per scene.** Multi-planet scenes
    (rare but real) emit one `RING_ANNULUS` per detectable ring system.
    `RingAnnulusNav.is_feasible` must handle the `len(features) > 1`
    case.

14. **`primary_technique` ties broken alphabetically.** The "highest-
    confidence per-technique result" sort is
    `key=(-confidence, technique_name)`. Deterministic; independent of
    registration order.

15. **`tier_thresholds` are translation-only.** `max_sigma_px` compares
    `max(sigma_dv, sigma_du)` only. Rotation σ is reported in JSON but
    never gates rank.

16. **`evaluate_sigmoid_combination` validates at config-load time.**
    `Config` initialization checks every technique's confidence YAML
    against its `NavTechniqueDiagnostics` dataclass field set; unknown
    fields raise `ValueError` with full diagnostic at startup. A
    config-load test asserts every shipped YAML resolves cleanly.

17. **Regression baselines round to 4 / 3 decimals.** `offset_px` rounded
    to 4 decimals, `confidence` to 3 decimals; comparison is exact-equal
    on rounded values. Stored in `tests/integration/baselines/<image_id>.json`.

18. **`static_data_hashes` are sha256 of raw YAML bytes.** Includes
    comments and whitespace; comment-only edits invalidate baselines
    (correct behavior — comments capture the WHY of value choices).

19. **`oops` backplane queries are named explicitly.** Add a Part 5
    "Backplane query reference" subsection mapping each derived
    quantity to the exact `Backplane.*` method:
    `km_per_pixel_at_limb` ← `Backplane.resolution(body, 'sub_observer')`;
    `km_per_pixel_radial` ← `Backplane.ring_radial_resolution(planet)`;
    `mean_emission_factor` source ← `Backplane.emission_angle(body_or_ring)`;
    `incidence` (per-vertex) ← `Backplane.incidence_angle(body)`;
    per-pixel `cos(incidence)` for predicted DN ← same.

### Factual corrections to the plan body

- The `_combine_models` / `_filter_models` "reuse" reference in Part 4
  points to `src/nav/nav_technique/nav_technique.py:71-86`, NOT
  `nav_master.py`. The functions exist; the file was wrong.
- `obs.inventory` is provided by the `oops` `Observation` superclass; no
  change needed in the Part 8 reuse table.
- `_metadata.json` top-level keys `navigation_techniques`, `offset`,
  `confidence` are added during `NavMaster.navigate()` (around line 395),
  not at metadata-dict initialization. The "preserved as-is" claim in
  Part 4 is true at the read interface; just reword for accuracy.
- `pyproject.toml` `[tool.pytest.ini_options]` has no `testpaths`,
  `markers`, `--strict-markers`, or `filterwarnings = ["error"]` today.
  Phase 1 must add all four (per `critique-test-suite` §19, §22).
- "Three rules" in Part 0 (Cardinal Principles) is actually four; the
  fourth is angles in degrees / radians. Reread Cardinal Principles with
  that count.

### Definitions added by this audit (referenced but never typed in the body)

`NavFeatureGeometry` is a sum type carried on `NavFeature.geometry`. Each
variant lives in `src/nav/feature/geometry.py`:

```python
NavFeatureGeometry = (
    StarGeometry | LimbPolyline | TerminatorPolyline | RingEdgePolyline
    | BodyDiscGeometry | BodyBlobGeometry | RingAnnulusGeometry
    | CartographicModelGeometry
)

@dataclass(frozen=True, eq=False)
class StarGeometry:
    """Single (v, u) point in extfov coords plus the predicted catalog
    position from which it was derived."""
    predicted_vu: tuple[float, float]
    catalog_vu: tuple[float, float]              # same value at extraction
                                                 # time; differs after refinement
    bbox_extfov_vu: tuple[int, int, int, int]    # half-open

@dataclass(frozen=True, eq=False)
class LimbPolyline:
    """Per-vertex limb data after extraction-time cropping."""
    vertices_vu: NDArrayFloat                     # (N, 2) — (v, u) per vertex
    normals_vu: NDArrayFloat                      # (N, 2) — outward normal
    sigma_normal_per_vertex_px: NDArrayFloat      # (N,)
    sigma_tangent_per_vertex_px: NDArrayFloat     # (N,) — typically 0.5
    bbox_extfov_vu: tuple[int, int, int, int]    # bounding box of vertices

# TerminatorPolyline mirrors LimbPolyline (same fields, different
# semantics).

@dataclass(frozen=True, eq=False)
class RingEdgePolyline:
    """Per-vertex ring edge data after extraction-time cropping."""
    vertices_vu: NDArrayFloat                     # (N, 2)
    normals_vu: NDArrayFloat                      # (N, 2) — radial outward
    sigma_radial_per_vertex_px: NDArrayFloat      # (N,)
    sigma_along_edge_per_vertex_px: NDArrayFloat  # (N,)
    is_straight_line: bool                        # rank-1 covariance flag
    bbox_extfov_vu: tuple[int, int, int, int]

@dataclass(frozen=True, eq=False)
class BodyDiscGeometry:
    """Pixel-template payload — geometry is implicit in template_img."""
    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    overflow_fraction: float                      # 0..1, see definitions

@dataclass(frozen=True)
class BodyBlobGeometry:
    """Centroid + extent for blob navigation."""
    predicted_center_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_diameter_px: float

# RingAnnulusGeometry, CartographicModelGeometry similarly small;
# pixel-template payloads with bbox + predicted_center.
```

`NavFeatureFlags` is the sum type already sketched in Part 1; missing
variants get a one-line schema in Part 1 (TerminatorArcFlags, BodyBlobFlags,
StarFlags, RingAnnulusFlags, CartographicModelFlags).

### `at_edge` and `spurious` semantics — per-technique

`at_edge` and `spurious` are set by every technique on its
`NavTechniqueResult`. The semantics:

- **`at_edge`** is True when the technique's solution touches the boundary
  of the search window (`extfov_margin_vu`-derived). Technique-specific
  rules:
  - `BodyDiscCorrelateNav`, `RingAnnulusNav`, `CartographicNav`: the
    sub-pixel-refined NCC peak is within 1 pixel of the search-window
    boundary in either axis.
  - `BodyLimbNav`, `BodyTerminatorNav`, `RingEdgeNav`: the LM-converged
    `(dv, du)` is within 1 pixel of any axis bound.
  - `BodyBlobNav`: the fitted centroid offset is within 1 pixel of any
    axis bound.
  - `StarFieldFromCatalogNav`, `StarUniqueMatchNav`: translation
    component within 1 pixel of any axis bound. Rotation `at_edge` is
    triggered separately when `|dθ| > 0.95 × max_rotation_deg` (config).
  - `StarRefineNav`: same translation rule.
- **`spurious`** is True when the technique's internal sanity check
  fails. Technique-specific rules:
  - NCC-based: `peak_NCC < 0.1`, OR `peak_to_runner_up_ratio < 1.05`,
    OR final RMS DT residual > `5 × sigma_normal_min_px` (limb / ring).
  - `StarFieldFromCatalogNav`: RANSAC found a transform but
    `n_inliers < pattern_match_min_inliers` (default 6).
  - `StarUniqueMatchNav`: detection lies > `extfov_margin` pixels from
    predicted catalog position.
  - `BodyBlobNav`: fitted centroid offset > `extfov_margin_vu`, OR
    `body_snr_inside_predicted_bbox < 1.0`.

A technique that returns `spurious=True` is dropped by the ensemble
unconditionally; `at_edge=True` is dropped only if at least one
non-edge result remains. Both flags are diagnostic; tests assert their
exact value per scenario.

### Additional decisions from this audit (round 2)

These resolve the Critical/High/Medium audit items that weren't covered by
the 19 settled decisions above. Each is binding the same way; where a
plan-body section conflicts, this list wins.

#### Internal-inconsistency fixes

20. **Cutover gating timing (B3).** The 500-image breadth comparison vs
    legacy is a **pre-merge** gate of the Phase 4 change-set, not a
    Phase 7 step. While the change-set is on a feature branch, both
    pipelines coexist; the comparison runs there. After Phase 4 merges,
    legacy is gone and there's nothing to compare against. Phase 7
    cutover-gating is verification (grep sweeps + coverage + docs)
    only.

21. **`'corrupt'` row in image-classifier table (B15).** Add a row:
    `corrupt | image file failed to parse / read; raises in obs construction
    | status='failed', status_reason='image_corrupt'. No technique runs.`
    Detection: any exception during `obs.from_file(...)` or
    `obs.data` access. The classifier never reaches the rest of the table
    in this case.

22. **`notes:` not part of the schema (B14).** Remove the inline `# At
    100 km/px ...` comments from the `config_220_body_shape.yaml` example.
    No `notes:` field; if commentary is needed, it lives in the file's
    YAML comments (`# ...`), not in a key. Schema validator rejects
    unknown top-level keys per body.

#### Specifications added (Section C)

23. **Orchestrator helper methods (C2).** The private helpers behave:
    - `_make_context(obs)`: read `obs.data`; apply per-instrument
      source-image filter; compute `image_noise_sigma` (MAD over
      `sensor_mask_ext`); compute `saturation_mask_ext` (≥
      `saturation_threshold_dn`); compute `cosmic_ray_mask_ext`
      (`raw - median3x3 > 5 × image_noise_sigma`, with `median3x3` using
      `mode='reflect'` at boundaries — see decision §32); run
      `NavImageClassifier` to populate `image_classifier`; compute
      shared `image_gradient_ext` (Sobel-of-Gaussian at
      `image_gradient_sigma_px`); compute `image_edge_dt_ext` (signed DT
      of thresholded gradient at `edge_threshold_k_sigma`); fill in
      `pre_filter_applied` and `spice_provenance`. Returns frozen
      `NavContext`.
    - `_is_hard_failure(image_classifier)`: returns True iff
      `image_class in {'blank', 'fully_overexposed', 'mostly_missing_data',
      'corrupt'}`. (See the classifier table; these are the four
      `failed`-outcome classes.)
    - `_build_image_classifier_failure(context)`: returns
      `NavResult(status='failed', offset_px=None, sigma_px=None,
      sigma_along_unobservable_px=None, confidence_rank='failed',
      confidence=0.0, status_reason=NavStatusReason.from_image_class(...),
      covariance_px2=None, per_technique=[], feature_inventory=[],
      image_classifier=context.image_classifier, model_metadata={},
      annotations=Annotations(), provenance=...)`.
    - `_build_no_features_failure(context)`: same shape; status_reason
      = `'no_features_extracted'` if extractor list is empty,
      `'all_features_gated'` if the gate dropped everything.
    - `_build_no_feasible_techniques_failure(features, context)`:
      same shape; status_reason = `'no_feasible_techniques'`;
      `feature_inventory` populated; per_technique empty.

24. **`NavContext` is frozen; `with_prior` returns a new instance (C3).**
    Add `@dataclass(frozen=True, eq=False)` to `NavContext`. Implement
    `with_prior(*, offset_px, covariance_px2)` as
    `dataclasses.replace(self, prior_offset_px=offset_px,
    prior_covariance_px2=covariance_px2)`. Part 16's
    "frozen unless mutating method explicitly required" rule covers
    this; `with_prior` is non-mutating.

25. **`preferred_filter` derivation per type (C4).** Each extractor
    constructs the `NavFilterSpec` for the features it emits using the
    σ-derivation formulas already in Part 2 ("Per-feature σ derivation
    formulas"). Concretely:
    - `STAR`: `NavFilterSpec(kind=NavFilterKind.NONE)` always.
    - `LIMB_ARC`: `NavFilterSpec(kind=GRADIENT_OF_GAUSSIAN,
      sigma_xy=(σ_normal, σ_tangent), align_axis=normal_at_centroid)`.
    - `TERMINATOR_ARC`: `NavFilterSpec(kind=ANISOTROPIC_GAUSSIAN,
      covariance_px2=diag(σ_normal², σ_tangent²),
      align_axis=normal_at_centroid)`.
    - `RING_EDGE` (curved): `NavFilterSpec(kind=ANISOTROPIC_GAUSSIAN,
      covariance_px2=diag(σ_radial², σ_along_edge²),
      align_axis=radial_at_centroid)`.
    - `RING_EDGE` (`is_straight_line=True`): same shape but with
      `σ_along_edge` set to the `sigma_along_edge_px_for_straight_line`
      sentinel (config; large) so the technique's covariance correctly
      reflects the unobservable axis.
    - `BODY_DISC`: `NavFilterSpec(kind=NavFilterKind.NONE)` —
      `BodyDiscCorrelateNav` chooses RAW vs GRADIENT internally via
      `use_gradient='auto'`.
    - `BODY_BLOB`: `NavFilterSpec(kind=NavFilterKind.NONE)` (no template
      matching).
    - `RING_ANNULUS`: `NavFilterSpec(kind=ISOTROPIC_GAUSSIAN,
      sigma_xy=(σ_mean, σ_mean))` where `σ_mean` is the per-edge
      `σ_radial_px` averaged across constituent edges.
    - `CARTOGRAPHIC_MODEL`: same as `BODY_DISC`.

26. **STAR zero-smear case (C8).** When smear length `L < 0.5 px`, the
    extractor sets `position_cov_px = (σ_PSF / sqrt(SNR))² × I_2` (isotropic
    2×2). The anisotropic formula is used only for `L ≥ 0.5 px`; below
    that, the smear axis is shorter than the PSF and the anisotropy is
    sub-pixel-meaningless. Threshold `0.5 px` lives in
    `config_110_stars.yaml` as `min_anisotropic_smear_px`.

27. **Cosmic-ray boundary handling (C9).** Median-3×3 vs raw uses
    `scipy.ndimage.median_filter(arr, size=3, mode='reflect')`. Boundary
    pixels use mirrored neighbors. No special-casing of corners; reflect
    is the standard choice and avoids edge artifacts.

28. **`_agreement_groups` order independence (C10).** Implementation:
    iterate pairs `(i, j)` with `i < j` over `viable` results in
    enumeration order; for each pair, compute Mahalanobis distance using
    `pinvh(Σ_a + Σ_b, rcond=pinvh_rcond)`; collect "edges" into a
    `scipy.sparse.csgraph.connected_components` call. The connected-
    components result is order-independent by construction (single-link
    clustering = connected components of the threshold graph).

29. **Mahalanobis with rank-deficient sum-covariance (C11).** Use
    `pinvh(Σ_a + Σ_b, rcond=pinvh_rcond)`; `d_M² = δ.T @ pinv @ δ`
    where `δ = μ_a − μ_b`. When the displacement δ has a component in
    the null space of `pinv`, treat it as infinite Mahalanobis (results
    cannot agree along an unobservable axis). Implementation:
    `pinv = scipy.linalg.pinvh(Σ_sum, rcond=...)`;
    `null_proj = δ − Σ_sum @ pinv @ δ`;
    if `np.linalg.norm(null_proj) > 1e-6`, `d_M = float('inf')`; else
    `d_M = sqrt(δ.T @ pinv @ δ)`.

30. **`extractor_diagnostics` aggregation (C15).** When an extractor
    emits zero features, `extractor_diagnostics[name] =
    extractor.zero_emission_reason()` — a method each extractor
    implements returning a single string ("no_predicted_snr_above_threshold",
    "no_body_visible_in_extfov", etc.). When the extractor emits
    features but every one is gated out, the entry is the
    most-common gate reason across that extractor's emissions, computed
    as `Counter(f.gate_reason for f in gated_features).most_common(1)[0][0]`.
    Mixed (some kept, some gated): no diagnostic emitted (extractor
    succeeded).

31. **`extfov_margin_vu` source (C17).** `obs.extfov_margin_vu` is set
    by the per-instrument `Config` block (`config.cassini_iss.nac
    .extfov_margin_vu` etc.) and exposed on `ObsSnapshotInst` via the
    existing per-instrument subclass. New code reads it via
    `obs.extfov_margin_vu`; never re-derived.

32. **`instrument_psf_fwhm` placement (C19).** Stays as-is:
    `obs.star_psf().fwhm()`. The `obs.star_psf()` resolution already
    accounts for camera and filter combo via the existing per-instrument
    config; no plan-level change.

33. **RANSAC tie-breaking (C20).** "Sorted detected-source index
    lexicographically" means: each candidate triplet is identified by
    the tuple `(idx_A, idx_B, idx_C)` sorted ascending. Candidates are
    iterated in `(hash_distance ascending, (idx_A, idx_B, idx_C)
    ascending)` order. Same image → same iteration order → same RANSAC
    winner.

34. **5σ source for cosmic-ray detection (C21).** "5σ" uses
    `image_noise_sigma` (the global MAD on `NavContext`), not a per-pixel
    local σ. Same threshold across the image; deterministic.

35. **`null_filter_threshold_sigma` check-site (C22).** Checked inside
    `apply_filter(arr, spec)`: if `spec.kind` is anisotropic / isotropic
    Gaussian and the largest eigenvalue of `spec.covariance_px2` is
    below `null_filter_threshold_sigma`, return `arr` unchanged. The
    extractor doesn't know to skip; the filter machinery does.

36. **M=30 brightest-source ties (H6).** Detected sources are sorted by
    `(peak_DN descending, v_centroid ascending, u_centroid ascending)`;
    the first 30 are kept. Ties on peak DN broken by lower `(v, u)`
    tuple. Deterministic.

37. **Edge-polarity equality (H4).** "Dot product > 0" means strictly
    greater. `dot product == 0` (orthogonal — vanishingly rare in
    practice) is treated as polarity disagreement; the vertex contributes
    `+∞` to the cost (effectively rejected). Conservative; no false
    "agreement" from numerical zero.

38. **Gauss-Newton termination (H5).** For 3-DoF NCC sub-pixel /
    sub-degree refinement: damping `λ=1e-3`, max 20 iterations,
    terminate on combined step-norm `sqrt(Δdv² + Δdu² + (Δdθ × pivot_dist)²) < 1e-3 px`
    where `pivot_dist` is the rotation pivot's distance from the
    image-center proxy (typical 100–500 px). For DT-based techniques
    (Levenberg-Marquardt): same termination; max 30 iterations in
    3-DoF mode.

39. **Filter-combo canonicalization edge case (H7).** Duplicates
    preserved (`['CL', 'CL'] -> 'CL+CL'`); `None` entries dropped before
    sorting. Single-filter inputs render as `'CL'`. The unit test for
    `canonicalize` covers each rule.

40. **No bracket-interpolation ground-truth workflow.** Spacecraft
    pointing drift between exposures is not linear (thruster firings,
    momentum-wheel desaturations, sub-second jitter) so one image's
    offset cannot be used to derive another image's offset to pixel
    precision.  Cardinal Principle #3 extends to ground-truth
    determination: every test-library image carries an offset that came
    from manually navigating *that* image.  No `nav_bracket_offset`
    helper, no `bracket_interpolated` `source` tier, no warm-start across
    images.

41. **`_combine_precision_weighted` empty-group precondition (H11).**
    The orchestrator caller asserts `len(best_group) > 0` before
    invocation. The combine helper raises `ValueError` ("empty group
    passed to _combine_precision_weighted") if violated; defensive
    assertion only — should never fire from the orchestrator's code
    path (interior filter / spurious filter ensure non-empty before
    grouping).

42. **JSON serialization precision (H9).** Curator rounds every float
    in the `_metadata.json` output: 4 decimals for pixel quantities
    (`offset`, `sigma_*`, `covariance_px2` entries), 3 decimals for
    `confidence` and `confidence_rank`-related thresholds, 6 decimals
    for `image_et`. Scientific notation reserved for `static_data_hashes`
    (already strings) and `pipeline_run_iso8601` (string). Test
    asserts no float in the output JSON has more than its allowed
    precision.

43. **Curator allow-list mechanism (H10).** Each
    `NavTechniqueDiagnostics` dataclass declares a `CURATOR_FIELDS:
    ClassVar[dict[str, str | None]]` mapping each instance attribute to
    its JSON key (or `None` to skip). The curator walks
    `dataclasses.fields(diag)` and asserts every name is in
    `CURATOR_FIELDS`; missing means CI failure. Same for
    `NavTechniqueResult.diagnostics` and `NavResult.model_metadata`.
    Test (`test_curator.py`) does the assertion.

#### Style / structure adjustments (Section D)

44. **Magic constants (D2) lifted out.** Add a `src/nav/feature/constants.py`
    module containing the named constants currently inline in formulas:
    `MAX_INCIDENCE_FACTOR_CAP = 4.76`, `INCIDENCE_FACTOR_ANGLE_CAP_DEG = 80.0`,
    `INCIDENCE_FACTOR_CLIP_DEG = 85.0`, `AGREEMENT_FACTOR_CAP = 1.5`,
    `COMBINED_CONFIDENCE_CAP = 0.99`, `JSON_INF_SENTINEL = 1e9`,
    `MIN_ANISOTROPIC_SMEAR_PX = 0.5`, etc. Each carries a one-line
    docstring with units + intent. Phase 1 implementation populates
    this module; pseudocode in this plan referencing the inline numbers
    is updated to reference the constant name.

45. **Typing aliases (D6).** Add `src/nav/support/types.py` containing
    `NDArrayFloat = np.ndarray  # cast of NDArray[np.floating[Any]]`,
    `NDArrayBool = np.ndarray  # cast of NDArray[np.bool_]`, etc.
    Single source of truth; every new module imports from there. Existing
    code that uses these aliases is migrated to import from the new
    typing module.

46. **`__all__` per package (D7).** Each new package's `__init__.py`
    declares `__all__` listing the public exports:
    - `nav.feature.__all__ = ['NavFeature', 'NavFeatureType',
      'NavFeatureGeometry', 'NavReliabilityBreakdown', 'NavFeatureFlags',
      'StarFlags', 'LimbArcFlags', 'TerminatorArcFlags', 'RingEdgeFlags',
      'BodyDiscFlags', 'BodyBlobFlags', 'RingAnnulusFlags',
      'CartographicModelFlags', 'StarGeometry', 'LimbPolyline',
      'TerminatorPolyline', 'RingEdgePolyline', 'BodyDiscGeometry',
      'BodyBlobGeometry', 'RingAnnulusGeometry',
      'CartographicModelGeometry', 'NavFeatureExtractor',
      'StarFeatureExtractor', 'BodyLimbExtractor', 'BodyTerminatorExtractor',
      'BodyBlobExtractor', 'BodyDiscExtractor', 'RingEdgeExtractor',
      'RingAnnulusExtractor', 'NavCartographicExtractor']`.
    - `nav.nav_technique.__all__ = ['NavTechnique', 'NavFeasibilityReport',
      'NavTechniqueResult', 'NavContext', 'NavTechniqueDiagnostics',
      'BodyDiscDiagnostics', 'BodyLimbDiagnostics',
      'BodyTerminatorDiagnostics', 'BodyBlobDiagnostics',
      'RingEdgeDiagnostics', 'RingAnnulusDiagnostics',
      'StarFieldDiagnostics', 'StarUniqueMatchDiagnostics',
      'StarRefineDiagnostics', 'CartographicDiagnostics',
      'BodyDiscCorrelateNav', 'BodyLimbNav', 'BodyTerminatorNav',
      'BodyBlobNav', 'RingEdgeNav', 'RingAnnulusNav',
      'StarFieldFromCatalogNav', 'StarUniqueMatchNav', 'StarRefineNav',
      'CartographicNav', 'TitanNav', 'evaluate_sigmoid_combination']`.
    - `nav.nav_orchestrator.__all__ = ['NavOrchestrator', 'NavResult',
      'NavFeatureSummary', 'NavImageClassifierResult', 'Provenance',
      'ensemble']`.
    - `nav.support.filters.__all__ = ['NavFilterSpec', 'NavFilterKind',
      'apply_filter']`.
    - `nav.support.status_reason.__all__ = ['NavStatusReason']`.

47. **Pre-emptive module splits (D10).** Three packages get split before
    Phase 4 ships, avoiding > 1000-line files:
    - `nav.nav_orchestrator` already split (`orchestrator.py`,
      `ensemble.py`, `nav_result.py`); also split `nav_result.py` into
      `nav_result.py` (the dataclass), `feature_summary.py`
      (`NavFeatureSummary`), `image_classifier_result.py`,
      `provenance.py`, `curator.py` (`_build_metadata_dict`,
      `assert_diagnostic_fields_present`), `status_reason_info.py`
      (`STATUS_REASON_INFO_TEMPLATE`).
    - `nav.nav_technique.nav_technique_star_field` becomes a sub-package
      (`star_field/__init__.py`): `detection.py` (DAOPHOT-style source
      detection + smeared-kernel matched filter), `triplet_hashing.py`
      (hash construction + KD-tree), `ransac.py` (RANSAC fit + verify),
      `nav.py` (the `StarFieldFromCatalogNav` class, thin orchestrator
      over the three).
    - `nav.feature.body_extractors` becomes a sub-package
      (`body/__init__.py`): `limb_extractor.py`,
      `terminator_extractor.py`, `blob_extractor.py`,
      `disc_extractor.py`, `cartographic_extractor.py`. Already implied
      by the four-extractor list in Part 8; commit to the package.

#### Documentation additions (Section E)

48. **Docstring detail level (E2).** Part 16 says one line; expand:
    every new module/class/function carries a Google-style docstring
    that includes `Parameters:`, `Returns:`, `Raises:`, plus a behavior
    paragraph sufficient to write a black-box test from the docstring
    alone (per `documentation.mdc` §4). Specifically:
    - `NavFeature` — docstring per field describing units, valid range,
      derivation source.
    - Each extractor — exact extraction algorithm including
      preconditions, edge cases, gate-out behavior, and which static-
      data tables it reads.
    - Each technique — accepted feature types, prior requirement,
      output covariance shape, confidence-formula source-of-truth
      pointer (`config_510_techniques.yaml.<technique_key>`).
    - `evaluate_sigmoid_combination` — the canonical math reference;
      every other docstring referring to "the confidence formula" links
      here.

49. **Module index (E4).** Update `docs/index.rst` (the Sphinx
    top-level toctree) to list every new module, sorted by package and
    name. Add to Phase 6 work list.

#### Testing additions (Section F)

50. **`caplog` assertions for WARNING fallbacks (F3).** Every WARNING
    log site listed in Part 12.7 has a corresponding test that runs the
    triggering code path under `caplog.at_level(logging.WARNING)` and
    asserts (a) the expected message substring AND (b) the log level
    `logging.WARNING`. Sites: missing-body fallback in `BodyExtractor`;
    missing-instrument-noise fallback in `StarFeatureExtractor`;
    technique-threw-exception in orchestrator's per-technique try/except;
    all-techniques-spurious in ensemble. Tests in
    `tests/nav/feature/test_logging.py` and
    `tests/nav/nav_orchestrator/test_logging.py`.

51. **Snapshot tests for `_metadata.json` (F4).** Add `syrupy` as a
    dev dependency (`pyproject.toml [project.optional-dependencies] dev`).
    `tests/nav/nav_orchestrator/test_metadata_snapshot.py` runs the
    curator on synthetic `NavResult` objects (one per `NavStatusReason`
    value), asserts the JSON output matches the stored snapshot. Snapshots
    committed under `tests/nav/nav_orchestrator/__snapshots__/`. CI step
    fails if a snapshot needs updating without explicit
    `pytest --snapshot-update` in the same PR.

52. **Test-boundary discipline (F5).** Tests touching
    `NavTechnique._registry` and `STATUS_REASON_INFO_TEMPLATE` document
    in their docstring that they assert internal-state properties.
    `__all__` excludes both names; tests acknowledge they import "for
    testing" via a top-of-file `# noqa: F401  # internal API for testing`.

53. **Property-based test scope expansion (F6).** Add
    `tests/nav/nav_orchestrator/test_ensemble_properties.py` covering:
    - `_agreement_groups(viable)` is order-independent (shuffled inputs
      produce the same partition).
    - `_combine_precision_weighted(group)` is order-independent within
      numerical tolerance.
    - DT fitter produces identity offset (within `0.01 px`) when given
      a planted polyline that already aligns to the image gradient.
    - RANSAC recovers the planted similarity transform with ≥
      `pattern_match_min_inliers` planted inliers under arbitrary noise
      below the centroid σ floor.

54. **Mocking convention (F7) + patch targets (F8).**
    `developer_guide_testing.rst` declares:
    - Default to `mock.patch` for call-assertions / spies; default to
      `monkeypatch` for env vars and module-level state.
    - Patch targets are documented per shared utility:
      `mock.patch('nav.nav_technique.confidence.evaluate_sigmoid_combination')`,
      `mock.patch('scipy.linalg.pinvh')` only when isolating math from
      a higher-level test, etc. Test file headers cite the canonical
      target.

55. **Synthetic-obs fixture realism (F9).** `tests/nav/fixtures/
    synthetic_obs.py` exposes one parametrized base fixture with these
    toggles (defaults in parens):
    `mission` (`'COISS_NAC'`), `image_shape_vu` (`(1024, 1024)`),
    `psf_sigma_px` (`1.0`), `add_noise` (`True`),
    `noise_sigma_dn` (`2.0`), `add_smear` (`False`),
    `smear_vector_vu` (`(0.0, 0.0)`), `add_saturated_pixels`
    (`False`), `add_cosmic_rays` (`False`),
    `add_missing_data_pixels` (`False`), `add_stray_light_gradient`
    (`False`), `add_alternating_lines` (`False`),
    `add_ccd_bloom` (`False`), `body_in_fov` (None or
    `BodyFixture(name='MIMAS', center_vu=(...), radii_km=(...))`),
    `stars_in_fov` (empty list), `ring_system` (None). Combinations
    documented; per-test fixtures override only the toggles they care
    about.

56. **Integration-test holdings dependency (F10).** A single
    `tests/integration/conftest.py` defines:
    `@pytest.fixture(scope='session') def pds3_holdings_dir() -> Path:
    p = os.environ.get('PDS3_HOLDINGS_DIR'); if not p:
    pytest.skip('PDS3_HOLDINGS_DIR unset; integration tests skipped');
    return Path(p)`. Every integration test depends on this fixture; no
    per-test `pytest.skip(...)` calls. Consistent skip behavior.

57. **`xfail` / `skipif` discipline (F12).** Convention in
    `developer_guide_testing.rst`: every `pytest.mark.xfail` carries
    `strict=True` and a comment containing a GH issue link. `skipif`
    with stale conditions (e.g., Python < 3.11) is removed in PR review.
    No xfail / skip lands without ticket reference.

58. **Per-image log assertion (F13).** `tests/nav/nav_orchestrator/
    test_log_structure.py` runs the orchestrator under
    `caplog.at_level(logging.INFO)` for one image-per-status_reason and
    asserts the INFO-line sequence matches the
    `STATUS_REASON_INFO_TEMPLATE` for that reason (substring match per
    expected line, in order). Tests the operator-readable narrative.

59. **Config-load value tests (F14).**
    `tests/nav/config_files/test_config_220_body_shape.py` asserts
    specific values for at least 5 representative bodies (e.g.,
    `config.body_shape['MIMAS'].ellipsoid_rms_residual_km == 1.4`)
    rather than only "loads cleanly". Same pattern for ring catalogs
    and per-camera noise blocks (one assertion per critical value per
    file).

60. **Conformance tests for AI-leeway helpers (F15).**
    `tests/nav/support/test_filters_conformance.py` and
    `tests/nav/nav_technique/test_confidence_conformance.py` lock the
    contract:
    - `apply_filter(arr, NavFilterSpec(kind=NONE))` returns `arr`
      unchanged (identity).
    - `apply_filter(arr, NavFilterSpec(kind=ANISOTROPIC_GAUSSIAN,
      covariance_px2=diag(1e-4, 1e-4)))` returns `arr` unchanged
      (null-filter threshold).
    - `evaluate_sigmoid_combination(spec, diag)` raises `ValueError`
      with a message containing the bad field name when `spec.feature`
      doesn't resolve on the diagnostics dataclass.
    - Order-invariance, idempotency, identity properties for every
      shared helper.

#### Library hygiene (Section G)

61. **`NullHandler` in package `__init__.py` (G1).** Each new top-level
    package (`nav.feature`, `nav.nav_orchestrator`, `nav.nav_technique`,
    `nav.support.filters`, `nav.support.status_reason`) adds
    `logging.getLogger(__name__).addHandler(logging.NullHandler())` at
    import time. Standard library-hygiene boilerplate; one line per
    package.

62. **Explicit `encoding='utf-8'` on every `open()` (G2).** Every new
    `open(...)` call in the new code passes `encoding='utf-8'`. Sidecar
    YAML reader, metadata curator's JSON write, per-image log writer,
    static-data hash compute. Ruff's `PLW` rule (if enabled) would catch
    this; `pyproject.toml` adds `PLW` to the ruff `select` list to make
    enforcement mechanical.

63. **Thread-safety documentation (G3).** Each extractor and the
    orchestrator carry a "Thread safety" docstring section: "Not safe
    for concurrent use on the same `obs` — give each thread/process its
    own instance. The `oops` global precision is mutated by some
    extractors (Part 12.8a)." Cloud-tasks per-process isolation is the
    intended concurrency model.

64. **Custom base exception class (G4).** *No* base exception class is
    introduced. Document explicitly in Part 12.7: "The orchestrator
    captures every per-technique exception and routes it through
    `NavResult(status='failed', status_reason=...)`. No exception
    propagates to callers. Therefore no `NavError` base class exists;
    callers check `nav_result.status` instead of using `try/except`."
    Reviewers who see "no exception hierarchy" should refer here, not
    add one.

#### Section H: AI-leeway sweep

65. **"Future need" sentences swept into Part 13b (H3).** Every "if a
    future need arises" / "tracked in Part 13b" / "to be designed and
    added later" sentence in Parts 1–12 either points to one of the 5
    existing Part 13b items or its referenced item is added. Reading
    pass during Phase 1 catches the rest; for now, the 5 items in
    Part 13b are the canonical deferred-work list.

66. **UI per-technique-result panel layout (H13).** Out of scope for
    the cutover. Filed as a 6th Part 13b item: "UI per-technique panel
    redesign — replace the manual-nav 'Auto' button with a per-technique
    side-by-side panel; ergonomics + interaction TBD; tracker only."
    Phase 4 keeps the existing manual-nav UI working with the new
    `NavResult` (one technique surfaced per the orchestrator's choice);
    the panel redesign happens later.

#### Section I: missing material

67. **Safe YAML loading (I1).** Every YAML load in the new code uses
    `yaml.safe_load(...)` (not `yaml.load(...)`) with explicit
    `encoding='utf-8'` open. No `yaml.UnsafeLoader`. Test asserts the
    sidecar reader rejects YAML with `!!python/object` tags.

68. **Performance budget (I2).** Targets per p50 image (full pipeline,
    one process):
    - COISS NAC, clean frame, `fit_camera_rotation=False`: ≤ 1.5×
      legacy median runtime (translation-only NCC overhead is small).
    - VGISS NA, `fit_camera_rotation=True`: ≤ 3× legacy.
    - Memory ceiling per worker: 800 MB resident set for any of the
      four supported missions at native resolution.
    Phase 5 calibration validates against the library; CI smoke test
    measures one COISS NAC frame and fails if runtime > 5× a stored
    baseline (loose; for catching catastrophic regressions, not
    fine tuning).

69. **CI grep step for deleted symbols (I3).** Phase 7 adds a
    `.github/workflows/ci.yml` step:
    `grep -rE 'NavTechniqueCorrelateAll|NavModelCombined|NavModelResult|NavMaster|weighted_mask|blur_amount|final_offset|final_confidence|use_legacy_pipeline' src tests docs README.md CLAUDE.md && exit 1 || exit 0`
    fails the build if any leftover reference appears. Single sweep;
    run on every PR.

70. **Empty tests directories (I5).** Phase 1 creates
    `tests/nav/feature/__init__.py` (empty),
    `tests/nav/nav_orchestrator/__init__.py` (empty),
    `tests/integration/__init__.py` (empty),
    `tests/integration/conftest.py` (the holdings-dir fixture from
    decision §56). Pytest discovery requires `__init__.py` in each
    new directory.

71. **`.cursor/rules/*.mdc` content (I7).** The three new rule files
    (Part 8) carry binding content:
    - `feature_extractor_conventions.mdc`: "Every NavFeatureExtractor
      subclasses `NavFeatureExtractor`. Per Cardinal Principle #2, no
      per-feature image-side cropping at predicted positions. Reliability
      formulas live in `src/nav/feature/reliability.py`. Tests follow
      the synthetic-obs fixture convention."
    - `nav_technique_conventions.mdc`: "Every NavTechnique subclasses
      `NavTechnique`. `is_feasible` reads feature metadata only (no
      pixels). Confidence formula spec lives in
      `config_510_techniques.yaml.<technique_key>`. Diagnostics dataclass
      declares `CURATOR_FIELDS`."
    - `static_data_conventions.mdc`: "`config_220_body_shape.yaml`
      schema; per-camera `noise:` and `mag_offset:` block schemas;
      fallback rules for missing entries (10% radius default, WARNING
      log, reliability cap 0.3). **Citation requirement (Part 0 §74):
      every numeric value in `config_220_body_shape.yaml` carries a
      sibling `_sources` mapping entry. AI agents drafting these
      values must cite only documents fetched in-session (no
      training-data citations); fabricated citations are a revert-
      in-full PR offense. Human reviewer spot-checks ≥5 citations per
      PR. Validation test
      `tests/nav/config_files/test_body_shape_citations.py` enforces
      schema completeness; citation accuracy is human-verified.**"

72. **`EnsembleReconciler` diagram fix (I10).** Update the architecture
    diagram in Part 0 (head of plan) to show `ensemble()` as a free
    function, matching the implementation. The `EnsembleReconciler` box
    name was a placeholder that drifted from the code shape.

73. **Memory budget (I11).** Per-image working set: the pre-filtered
    image (4 MB at 1024×1024 float32), the gradient (4 MB), the DT
    (4 MB), saturation/cosmic-ray/sensor masks (1 MB each, bool packed),
    plus per-feature templates (BODY_DISC at 256×256 float32 ≈ 0.25 MB
    each, ≤ 5 bodies typical → ≤ 1.5 MB). Total per-image ≈ 25–40 MB.
    Cloud-tasks workers handle one image at a time; concurrent worker
    count × 40 MB sets RAM provisioning.

#### Static-data citations (binding)

74. **Every numeric value in `config_220_body_shape.yaml` requires an
    accurate, non-fabricated citation** — full stop. Navigation trust is
    downstream-safety-critical; an invented uncertainty number propagates
    silently into every per-body extractor decision.

    **Schema.** Each body block carries a sibling `_sources` mapping
    keyed by field name. The runtime YAML loader **ignores** keys
    starting with `_` (so `_sources` is documentation-only and doesn't
    bloat the parsed `Config`); a separate validation step asserts the
    `_sources` block is present and complete. Example:

    ```yaml
    MIMAS:
      radii_km: [207.4, 196.8, 190.6]
      ellipsoid_rms_residual_km: 1.4
      crater_scale_km: 3.0
      albedo_mean: 0.96
      albedo_variation: 0.05
      shape_class_hint: regular
      _sources:
        radii_km: 'Thomas (2010), Icarus 208(1):395-401, Table 3, Mimas row.
                   doi:10.1016/j.icarus.2010.01.025'
        ellipsoid_rms_residual_km: 'Thomas (2010), same paper, Table 3
                                    column "RMS residual".'
        crater_scale_km: 'Schenk & Moore (2007), Cassini ISS Mimas crater
                          census, JGR Planets 112:E12. doi:10.1029/2007JE002942.
                          Median large-crater depth.'
        albedo_mean: 'Verbiscer et al. (2007), Science 315:815, Table 1,
                      Mimas geometric albedo, V band.'
        albedo_variation: 'Buratti & Veverka (1984), Icarus 58:254-264,
                           per-hemisphere albedo std on Mimas.'
        shape_class_hint: 'IAU WGCCRE 2018 report, classification table.'
    ```

    **Citation format requirements.** Each citation string MUST be
    sufficient for a human reviewer to (a) locate the cited document
    and (b) find the specific value within it. Acceptable forms:

    - **Published paper**: `Author(s) (Year), Journal Volume:Pages,
      Table/Figure/§Section, identifying clue. doi:<DOI>` — DOI is
      mandatory when one exists. The "identifying clue" tells the
      reviewer which row/column/equation carries the cited number.
    - **PDS calibration document**: `<Mission> calibration report
      <document_id> v<version>, §<section>, Table <N>` — the document
      ID matches the PDS3/PDS4 holdings.
    - **IAU report**: `IAU WGCCRE <year> report, §<section>`.
    - **Mission-team technical memo**: `<Author or Team> (Year),
      <Title>, <Identifier or URL>` — URL only when no DOI exists.

    **Anti-hallucination procedure.** AI agents that draft body-shape
    entries:
    1. Must only cite documents they have fetched (`WebFetch` /
       `WebSearch`) or read from `oops`-package data files in this
       session. Citing from training-data memory is not allowed.
    2. If a value cannot be sourced from a fetched document, the value
       is left as `null` and the `_sources` entry reads
       `'PLACEHOLDER — no source found, calibrate in Phase 5'`. The
       loader fallback (Part 5: 10% radius default, WARNING log,
       reliability cap 0.3) handles `null` values at runtime.
    3. DOIs and paper titles must verify against a real
       `https://doi.org/<DOI>` lookup; agents do not invent identifiers.
    4. The drafting prompt explicitly forbids fabrication: any draft
       PR that lists a citation an AI agent invented (caught in human
       review) is reverted in full and re-drafted by a different
       process.

    **Human review.** Every PR touching `config_220_body_shape.yaml`
    requires a reviewer to **spot-check at least 5 randomly-selected
    citations** by opening the cited document and verifying the value
    appears at the cited location. The reviewer marks the PR with a
    `cited-values-spot-checked` label; merging is blocked without
    that label. The 5-check minimum is per-PR; a 50-body initial
    population requires the same 5-check rule applied each PR (so an
    initial-population PR is broken into ≤ 10 bodies per PR to keep
    review tractable).

    **Validation test.**
    `tests/nav/config_files/test_body_shape_citations.py` asserts:
    - Every body has a `_sources` mapping.
    - Every non-`null` numeric / list field on a body has a
      corresponding `_sources` entry that is a non-empty string.
    - No `_sources` value contains the substrings `TODO`, `FIXME`,
      or `XXX` (case-insensitive); `PLACEHOLDER` is allowed only when
      the value itself is `null` (matched pair).
    - Test failure messages name the body and the missing/invalid
      field so a human can fix it directly.

    **Same rule applies to other static-data tables** (per-camera
    `noise:` / `mag_offset:` blocks in `config_4N0_inst_*.yaml`, and
    additions to ring catalogs `config_3N0_*_rings.yaml`). Every
    numeric value either inherits its citation from the existing
    catalog comment header or carries a new `_sources` entry. The same
    validation test extends to those files; the same human-review
    spot-check rule applies. Existing values already in the ring
    catalogs are grandfathered (they were curated by orbit-fitting
    astronomers and the catalogs document their pedigree in the file
    header) — only *new* values added by this cutover need explicit
    `_sources` entries.

    **Why this is mandatory and not best-effort.** A wrong
    `ellipsoid_rms_residual_km` value silently distorts every per-image
    σ_normal computation for that body for every navigation run forever.
    There is no downstream cross-check — the orchestrator trusts the
    static data and propagates it directly into reliability scores and
    technique covariances. Citation requirement is the *only* defense
    against fabrication.

#### Smaller advisory items applied silently

The remaining low-severity items (D3 keyword-only after positional, D4
RORO compliance, D5 inline-imports rule, D8 falsy-check rule, D9 getattr,
D11 Annotations carve-out, E3 sphinx clean target, E5 CHANGELOG.md
existence check, E6 README badge audit, E7 sentence spacing, F2 already
addressed under §50–60, G5 CI matrix vs requires-python check, H13 UI
deferred under §66, I6 PyQt6 rationale acknowledgement) are accepted as
implementation-time conventions. Their absence from this section means
"apply the cursor rule as written, no special handling".

---

## Implementation status (snapshot)

This section tracks what has shipped vs. what still needs to be built.
The remainder of the plan (Parts 1–16) describes the *target* design;
this section is the operational checklist.

### Implemented

**Type system, sum types, and dataclasses (Part 1, Part 4, Part 16)**

- `nav.feature.NavFeatureType` enum (9 values).
- `nav.feature.NavFeatureGeometry` sum type and per-type geometry
  payload dataclasses (`StarGeometry`, `LimbPolyline`,
  `TerminatorPolyline`, `RingEdgePolyline`, `BodyDiscGeometry`,
  `BodyBlobGeometry`, `RingAnnulusGeometry`,
  `CartographicModelGeometry`).
- `nav.feature.NavFeatureFlags` sum type and per-type flag
  dataclasses (`StarFlags`, `LimbArcFlags`, `TerminatorArcFlags`,
  `RingEdgeFlags`, `BodyDiscFlags`, `BodyBlobFlags`,
  `RingAnnulusFlags`, `CartographicModelFlags`).
- `nav.feature.NavFeature` (frozen, eq-by-`feature_id`,
  numpy-array-aware) and `NavReliabilityBreakdown`.  ``__post_init__``
  validates ``feature_id``, ``reliability``, ``subject_range_km``
  (>= 0, NaN-rejected, ``inf`` permitted), ``intensity_sigma_rel``
  (finite, ``[0, 1]``), ``feature_type in usable_types``,
  the ``position_cov_px`` shape / finiteness / symmetry / PSD
  invariants, and the ``template_img`` / ``template_mask`` both-or-
  neither + shape-match invariants.
- `nav.feature.constants` module (cap angles, sentinels).
- `nav.feature.compose_template_features` — Z-buffer paint of
  template features into a single ext-FOV image+mask; raises
  ``ValueError`` (with the offending ``feature_id`` and bbox) when
  a template is smaller than its declared bbox.
- `nav.feature.reliability.FeatureReliabilityGate` and the default
  per-type threshold table.  ``__post_init__`` requires the
  ``thresholds`` argument to be a ``Mapping[NavFeatureType, float]``
  with each value finite and in ``[0, 1]``.
- `nav.support.filters` — `NavFilterSpec`, `NavFilterKind`,
  `apply_filter` (NONE, isotropic / anisotropic Gaussian, bandpass
  DoG, distance transform, gradient-of-Gaussian, morph dilate, with
  null-filter short-circuit).
- `nav.support.filter_combo.canonicalize`.
- `nav.support.status_reason.NavStatusReason` (15 values; ``StrEnum``
  on Python 3.11+).
- `nav.support.noise_estimate.estimate_image_noise_sigma`.
- `nav.support.image_quality.saturation_mask` / `cosmic_ray_mask`.
- `nav.support.distance_transform` — `apply_translation`,
  `sample_dt_bilinear`.
- Updated `pyproject.toml` `[tool.pytest.ini_options]` with
  `testpaths`, `--strict-markers`, `--strict-config`,
  `filterwarnings = ["error"]` (with one tolerated rule for
  astropy's FITS-reader leakage), and the `integration` / `slow`
  markers.  Python floor is 3.11; CI matrix covers 3.11 / 3.12 /
  3.13 / 3.14.

**Technique infrastructure (Part 3)**

- `nav.nav_technique.NavTechnique` ABC + `__init_subclass__`
  registry + `filter_technique_names` glob helper (with leading-`!`
  exclusion).
- `nav.nav_technique.NavFeasibilityReport` (validates types and
  the ``feasible=False`` requires non-empty ``reason`` invariant) and
  `NavTechniqueResult` with ``feature_ids: tuple[str, ...]`` (immutable;
  list-typed inputs are converted in ``__post_init__``); covariance
  shape / symmetry / positive-semidefinite validation; the array is
  frozen read-only on construction.
- Per-technique typed diagnostics dataclasses with `CURATOR_FIELDS`
  allow-lists: `BodyDiscDiagnostics`, `BodyLimbDiagnostics`,
  `BodyTerminatorDiagnostics`, `BodyBlobDiagnostics`,
  `RingEdgeDiagnostics`, `RingAnnulusDiagnostics`,
  `StarFieldDiagnostics`, `StarUniqueMatchDiagnostics`,
  `StarRefineDiagnostics`, `CartographicDiagnostics`.
- `nav.nav_technique.confidence` — `evaluate_sigmoid_combination`
  with `ConfidenceSpec` / `ConfidenceTerm`.  Both dataclasses
  validate types, finite numerics, and range invariants in
  ``__post_init__``; ``ConfidenceSpec`` takes a defensive shallow
  copy of ``hard_zero_if`` so the frozen instance cannot be
  mutated through the caller's dict.
- `NavTechniqueManual` — interactive technique opted out of the
  auto-discovery registry.

**NavModel infrastructure (Part 1, Part 8)**

- `nav.nav_model.NavModel` ABC + `__init_subclass__` registry +
  `instances_for_obs` class-method hook + module-level
  `build_models_for_obs(obs)`.
- `nav.nav_model.NavModelBodyBase` and
  `nav.nav_model.NavModelRingsBase` (annotation helpers; abstract).
- `nav.nav_model.NavModelBodySimulated` and
  `NavModelRingsSimulated` adapted to the new contract — emit
  `BODY_DISC` / `RING_ANNULUS` `NavFeature` plus annotations.
- `nav.nav_model.NavModelTitan` — clean stub.
- The `nav.nav_model.rings/` subpackage (`RingFeature`,
  `RingFeatureFilter`, `RingRenderResult`, `RingsRenderContext`,
  `ring_math`, `ring_types`) preserved with restored unit tests.

**Orchestrator + ensemble (Part 4)**

- `nav.nav_orchestrator.NavContext` (frozen, `with_prior`
  non-mutating; the ``with_prior`` boundary validates the prior
  offset and covariance and takes an independent copy + freezes the
  array so the caller cannot mutate it after the fact).
- `Provenance` (``__post_init__`` normalises ``spice_kernels``,
  ``technique_names``, ``extractor_names`` to deterministic sorted
  tuples; derives ``spice_kernel_count``; wraps
  ``static_data_hashes`` with ``MappingProxyType``).
- `NavFeatureSummary` (validates every field including the
  ``gated``/``gate_reason`` invariant) and
  `NavImageClassifierResult` dataclasses.
- `NavImageClassifier` + `ImageQualityThresholds`.  ``classify``
  validates the image is a 2-D ``np.ndarray`` and the optional
  ``sensor_mask`` is a non-empty boolean ndarray with at least one
  True entry and matching shape.
- `NavResult` with `ok` / `failed` / `conflicted` constructors;
  carries an ``annotations: Annotations`` field that the
  orchestrator populates via ``_collect_annotations``, plus the
  ``model_metadata: dict[str, dict[str, Any]]`` map populated by
  ``_collect_model_metadata``.
- `STATUS_REASON_INFO_TEMPLATE` covering every `NavStatusReason`.
- `ensemble.ensemble` — Mahalanobis-distance grouping +
  precision-weighted information-form combine + agreement boost +
  conflict / unobservable-offset handling + tier derivation.
  Rank-deficiency is flagged with a *scale-independent* relative
  test (``eigvals.min() / max(abs(eigvals.max()), eps) < 1e-8``).
- `EnsembleConfig` (default factory deep-copies
  ``DEFAULT_TIER_THRESHOLDS`` so the inner per-rank dicts cannot be
  cross-mutated between instances) and `derive_confidence_rank`.
- `curator.build_metadata_dict` + `assert_diagnostic_fields_present`
  (float-rounding policy: 4 px / 3 conf / 6 ET; ∞ → 1e9 sentinel).
- `NavOrchestrator` two-pass driver
  (preflight → extract → gate → pass-1 → ensemble → pass-2 → final
  ensemble) with `only_models` / `only_techniques` glob filters.
  Per-NavModel ``create_model`` / ``to_features`` /
  ``to_annotations`` and per-NavTechnique ``navigate`` are
  sandboxed in plugin-isolation try/excepts so a misbehaving
  plugin is logged and skipped, never propagated.  Saturation DN
  comes from the named module-level ``DEFAULT_FULL_WELL_DN_12_BIT``
  constant.

**Top-level driver (Part 8)**

- `nav.navigate_image_files.navigate_image_files` — uses
  `build_models_for_obs` + `NavOrchestrator` + `build_metadata_dict`,
  preserves the legacy `(success, metadata)` contract for
  `nav_offset` and `nav_offset_cloud_tasks`.  The
  ``_write_summary_png`` step is an honest no-op that logs at INFO
  until the annotation-rendering pipeline lands; the misleading
  1×1 grey PNG placeholder is gone.

**Manual navigation (Part 12.6)**

- `nav.ui.manual_nav_dialog.ManualNavDialog` adapted: takes a
  composite `model_img_ext` + `model_mask_ext` instead of
  `NavModelCombined`.

**Test coverage**

- Unit and integration tests for every new module under
  `tests/nav/feature/`, `tests/nav/nav_orchestrator/`,
  `tests/nav/nav_technique/`, and the new `tests/nav/support/*`.
  Full suite passes under `pytest -n auto --dist=loadfile`.
  `ruff check`, `ruff format --check`, and `mypy --strict src tests`
  are clean across the cutover-tier source tree.

### Removed without direct replacement

- `NavMaster` and `nav.navigate_image_files`'s previous shape.
- `NavTechniqueCorrelateAll` (the legacy fused-model NCC pipeline).
- `NavModelCombined`, `NavModelResult` (replaced by per-feature
  templates and the ensemble's per-technique combine).
- The legacy `NavModelStars`, `NavModelBody`, `NavModelRings`
  classes — only the `rings/` data-model subpackage and the
  simulated variants survive on the new contract.
- The deleted star-catalog helpers (aberration, proper motion,
  incremental catalog search, body / ring conflict marking) need
  to be carried forward verbatim into a helper module so the new
  `NavModelStars` does not rewrite them; this is tracked under
  "Pending" below.

### Pending

**Concrete NavModels (Part 1, Part 8)**

- Real-scene `NavModelStars` on the new contract — must reuse the
  existing aberration / proper-motion / catalog-reduction /
  body-and-ring-conflict logic from the deleted module rather than
  rewrite it.
- Real-scene `NavModelBody` (per-body limb, terminator, disc, blob,
  cartographic-model emission with the existing shape gates).
- Real-scene `NavModelRings` (per-edge polylines + `RING_ANNULUS`
  fallback, using the preserved `RingFeatureFilter`).

**Concrete NavTechniques (Part 3)**

- `BodyDiscCorrelateNav`, `BodyLimbNav`, `BodyTerminatorNav`,
  `BodyBlobNav`, `RingEdgeNav`, `RingAnnulusNav`,
  `StarFieldFromCatalogNav`, `StarUniqueMatchNav`, `StarRefineNav`,
  `CartographicNav`, `TitanNav`.

**NavContext shared derivatives**

- `image_gradient_ext`, `image_edge_dt_ext`, source-image
  `BANDPASS_DOG` pre-filter — fields are present on `NavContext`
  but the orchestrator does not yet populate them.
- Per-instrument saturation DN read from `config_4N0_inst_*.yaml`
  (the orchestrator currently returns the named module-level
  ``DEFAULT_FULL_WELL_DN_12_BIT = 4095.0`` constant from
  ``_instrument_full_well_dn``; replace with a config loader).

**Provenance population**

- ``Provenance`` dataclass shape is final (sorted-tuple
  normalisation, derived ``spice_kernel_count``,
  ``MappingProxyType``-wrapped ``static_data_hashes``).
- Pending: actually populate ``spice_kernels`` from
  ``spice.ktotal`` / ``spice.kdata``; populate
  ``static_data_hashes`` with sha256 of raw YAML bytes; populate
  ``rms_nav_git_sha`` from ``git rev-parse``.

**Camera rotation correction (Part 5b)**

- Dataclass shape implemented: `NavTechniqueResult.covariance_px2`
  accepts 3×3, and both `NavTechniqueResult` and `NavResult` carry
  `rotation_rad` / `sigma_rotation_rad` fields.
- Pending: per-instrument `fit_camera_rotation` flag wiring; per-
  technique populators that emit the rotation entries; rotation-aware
  ensemble combine math.

**Annotations + summary PNG**

- ``NavResult.annotations: Annotations`` field is implemented; the
  orchestrator's ``_collect_annotations`` helper merges every
  registered NavModel's ``to_annotations(context)`` into it on
  every navigation.
- Pending: replace the honest INFO-level no-op in
  ``navigate_image_files._write_summary_png`` with the real
  annotation-compositing renderer that turns
  ``NavResult.annotations`` plus the source image into a
  ``_summary.png``.

**Static-data files (Part 5)**

- `config_220_body_shape.yaml` (per-body shape, albedo, with
  `_sources` citations per Part 0 §74).
- `config_510_techniques.yaml` (technique tunables + confidence
  formulas).
- `config_520_features.yaml` (per-feature reliability gates).
- `config_530_filters.yaml` (filter clamps + null threshold).
- `config_540_orchestrator.yaml` (ensemble parameters).
- `noise:` and `mag_offset:` blocks in every
  `config_4N0_inst_*.yaml`.
- Renumbering of `config_NN_*.yaml` → `config_NNN_*.yaml`.

**INFO logging cadence (Part 12.7)**

- `STATUS_REASON_INFO_TEMPLATE` exists; the orchestrator does not
  yet emit those INFO lines for each status_reason.
- Per-technique 1-line summary + per-feature reliability score
  breakdown at DEBUG level.

**Image-quality classifier completeness**

- `NavImageClassifier` emits four classes (`clean`, `blank`,
  `fully_overexposed`, `mostly_missing_data`); the caller routes
  image-load exceptions to the fifth (`corrupt`).  The advisory
  flag list also covers `partial_dropout` and `noisy`.  Detection
  for additional content-degradation classes —
  `partial_data_dropout`, `alternating_lines`, `truncated_readout`,
  `ccd_bloom_dominant` (and matching flags) — is not yet
  implemented; the `ImageClass` Literal will gain those values when
  their detection paths land.

**CLI**

- `nav_feature_inspect` (debugging tool that runs only the
  extractor + reliability gate on one image).

**Test image library + integration tests (Part 9, Part 10)**

- ~50 operator-curated images with sidecars + ground-truth offsets
  (manual eye-pick on each image; no cross-image inference).
- `tests/integration/test_image_library.py` (structural invariants).
- `tests/integration/test_autonomous_nav.py` (per-image regression).
- `tests/integration/baselines/<image_id>.json` regression
  baselines.

**Confidence-formula calibration (Part 5)**

- One-time fit of the alpha coefficients in
  `config_510_techniques.yaml` against the curated library.

**Documentation (Part 8)**

- All new Sphinx pages: `developer_guide_autonomous_nav.rst`,
  `developer_guide_features.rst`,
  `developer_guide_techniques.rst`,
  `developer_guide_filters.rst`,
  `developer_guide_uncertainty.rst`,
  `developer_guide_static_data.rst`,
  `developer_guide_orchestrator.rst`,
  `developer_guide_logging.rst`, `developer_guide_cli.rst`,
  `developer_guide_testing.rst`,
  `user_guide_metadata_schema.rst`,
  `user_guide_troubleshooting.rst`,
  `user_guide_image_library.rst`, `user_guide_migration.rst`, plus
  the auto-API pages.
- Updates to `introduction_overview.rst`,
  `user_guide_navigation.rst`,
  `user_guide_configuration.rst`,
  `developer_guide_reprojection.rst`,
  `developer_guide_extending.rst`.
- README + CLAUDE.md + CHANGELOG.md updates.
- The three new `.cursor/rules/*.mdc` files.

**Cleanup (Phase 7)**

- ``sphinx-build -W`` is clean as of stage 0; CI must keep it that
  way.
- CI grep step that fails on residual references to deleted symbols
  (``NavTechniqueCorrelateAll``, ``NavModelCombined``,
  ``NavModelResult``, ``NavMaster``, ``weighted_mask``,
  ``blur_amount``, ``final_offset``, ``final_confidence``,
  ``use_legacy_pipeline``).
- Coverage gate in CI: cutover-tier modules (``nav.feature/``,
  ``nav.nav_orchestrator/``, ``nav.nav_technique/``) at >= 90 %;
  whole-tree percentage gated by per-package thresholds once
  concrete NavModels and NavTechniques arrive (the GUI / mosaic-
  viewer / sim packages pre-date the cutover and are below 90 %
  by design).
- Module size cap.  ``src/nav/ui/manual_nav_dialog.py`` is 1058
  lines, exceeding the project's 1000-line module ceiling.  Split
  into a ``nav.ui.manual_nav/`` subpackage in a dedicated PR with
  GUI-test infrastructure to verify the manual workflow still
  works.

**Deferred (Part 13b)**

- Ring-edge polarity-aware matching.
- Cartographic-model technique testing (production mosaics not
  yet available for the supported missions).
- Atmospheric-body navigation algorithm (`TitanNav` real
  implementation).
- Annotation styling for gated-out features.
- Mixed-instrument batch SPICE-kernel hot path (performance).
- Per-technique side-by-side panel in the manual-nav dialog
  (Part 12.6 redesign).

---

## Foundation cleanup status (stage 0 — complete)

Stage 0 ("foundational support" PR
[#111](https://github.com/SETI/rms-nav/pull/111),
branch ``core_rewrite_foundation`` → ``rf_core_rewrite``) is **complete**.
The 2026-04-27 foundation critique surfaced four severity-tagged
domains of work; every blocking and important item has been resolved.
The original critique reports (`CRITIQUE_PYTHON.md`,
`CRITIQUE_TESTS.md`, `CRITIQUE_DOCS.md`, `CRITIQUE_PLAN.md`,
`CRITIQUE_SUMMARY.md`) and the disposition log
(`CRITIQUE_RESOLUTION.md`) live at the repo root, untracked, as the
historical record of stage-0 cleanup.

### Final stage-0 check matrix

```
$ ruff check src tests           — clean
$ ruff format --check src tests  — 210 files already formatted
$ mypy --strict src tests        — 211 source files, no issues
$ pytest -n auto --dist=loadfile — 719 passed (1 tolerated warning)
$ sphinx-build -W -b html docs docs/_build  — clean
$ pymarkdown scan docs/ .cursor/ README.md CONTRIBUTING.md  — clean
```

### Stage-0 commit lineage (most recent first)

- ``5bb982b`` ci: add Python 3.14 to the supported version matrix
- ``664e092`` refactor: scale-independent rank-deficiency check + defensive ``hard_zero_if`` copy
- ``ae26b54`` refactor: strengthen dataclass ``__post_init__`` validation
- ``1fe6d62`` build: bump Python floor from 3.10 to 3.11
- ``68edecb`` ci: run tests on all pull requests, not just those targeting main
- ``57b70d4`` Stage 0 follow-up: address PR review findings
- ``be61c79`` Stage 0: foundational support for navigation core rewrite

### Architectural skeleton shipped in stage 0

Every package, class, free function, and dataclass listed in
"Implementation status (snapshot) → Implemented" exists at its
documented path with full docstrings, frozen-where-appropriate
dataclasses with strict ``__post_init__`` validation, and complete
unit-test coverage at ≥ 93 % for the cutover-tier modules.

The orchestrator pipeline runs end-to-end: ``NavOrchestrator.navigate``
builds a ``NavContext``, calls ``model.create_model()`` /
``to_features`` / ``to_annotations`` on each registered NavModel
inside a per-plugin sandbox, gates features by reliability, runs
every feasible NavTechnique through its own sandbox, reconciles
per-technique results via the precision-weighted Kalman-style
``ensemble`` free function, and returns a ``NavResult`` carrying
``offset_px`` ± ``sigma_px``, ``confidence_rank``,
``per_technique`` results, the feature inventory, the image-quality
classifier verdict, the merged annotation collection, the
per-NavModel metadata dict, and the reproducibility envelope.

The simulated body / rings NavModels emit on the new contract; the
``NavModelTitan`` stub is registered.  ``NavTechniqueManual`` is
abstract-opted-out of the auto-discovery registry but available for
the GUI driver.  Real-scene NavModels and concrete NavTechniques are
the next stage's work — see "Stage 1 entry points" below.

### Hardening landed in stage 0

Validation-and-immutability hardening across every dataclass and
public function the orchestrator depends on.  An AI continuing the
work can construct any of these and trust the inputs are checked.

- ``NavFeature.__post_init__`` validates: ``feature_id`` non-empty
  string; ``reliability`` in ``[0, 1]``; ``subject_range_km`` >= 0
  and not NaN (``inf`` permitted as the very-far sentinel);
  ``intensity_sigma_rel`` finite and in ``[0, 1]``;
  ``feature_type in usable_types``; ``position_cov_px`` is 2x2
  finite symmetric positive-semidefinite (frozen read-only);
  ``template_img`` and ``template_mask`` are both-or-neither and
  shape-matched (shape check runs before ``setflags(write=False)``
  so rejected inputs leave the caller's array untouched).
- Every per-feature ``Flags`` dataclass with bounded numeric
  fields (``StarFlags.smear_length_px``,
  ``LimbArcFlags.visible_arc_fraction``,
  ``TerminatorArcFlags.{visible_arc_fraction, phase_angle_factor}``,
  ``BodyDiscFlags.overflow_fov_fraction``,
  ``BodyBlobFlags.predicted_diameter_px``,
  ``RingAnnulusFlags.constituent_edge_count``) validates ranges in
  ``__post_init__``.
- ``FeatureReliabilityGate.__post_init__`` requires the thresholds
  argument to be a ``Mapping[NavFeatureType, float]`` with each
  value finite and in ``[0, 1]``.
- ``NavContext.with_prior`` validates ``offset_px`` is a length-2
  finite tuple and ``covariance_px2`` is a 2x2 finite array; takes
  an independent ``copy()`` and ``setflags(write=False)`` so the
  caller cannot mutate the prior covariance after the fact.
- ``NavFeasibilityReport.__post_init__`` validates types of every
  field and requires a non-empty ``reason`` when ``feasible`` is
  False.
- ``NavTechniqueResult`` carries ``feature_ids: tuple[str, ...]``
  (immutable; ``__post_init__`` converts a list-typed input);
  validates 2x2/3x3 covariance shape, symmetry, PSD; freezes the
  array.
- ``NavFeatureSummary.__post_init__`` validates every field
  including the ``gated``/``gate_reason`` invariant (gated features
  must carry a non-empty reason) and the bbox-tuple shape.
- ``ConfidenceTerm.__post_init__`` validates ``feature`` is a
  non-empty string; ``alpha`` / ``offset`` / ``divisor`` are finite
  numerics; ``divisor`` is non-zero; ``cap_at`` (when set) is finite
  and in ``[0, 1]``.
- ``ConfidenceSpec.__post_init__`` validates types of every field
  and takes a defensive shallow copy of ``hard_zero_if`` so the
  frozen dataclass cannot be mutated through the caller's dict.
- ``Provenance.__post_init__`` normalises ``spice_kernels``,
  ``technique_names``, ``extractor_names`` to deterministic sorted
  tuples; derives ``spice_kernel_count`` from
  ``len(spice_kernels)``; wraps ``static_data_hashes`` with
  ``MappingProxyType`` so the frozen dataclass cannot leak a
  mutable mapping.
- ``ImageQualityThresholds`` dropped the redundant
  ``partial_dropout_max_frac`` field (which duplicated
  ``max_missing_frac_clean``); ``NavImageClassifier.classify``
  validates the ``image`` argument is a 2-D ``np.ndarray`` and the
  ``sensor_mask`` (when supplied) is a non-empty boolean ndarray
  with at least one True entry and matching shape.
- ``EnsembleConfig`` default-factory now does a deep copy of
  ``DEFAULT_TIER_THRESHOLDS`` so the inner per-rank dicts can no
  longer be cross-mutated between instances.
- ``ensemble._combine_precision_weighted`` flags rank-deficiency
  with a *scale-independent* relative test
  (``eigvals.min() / max(abs(eigvals.max()), eps) < 1e-8``),
  replacing the previous absolute cutoff that mis-classified
  high-precision combines.
- ``feature.composition.compose_template_features`` raises
  ``ValueError`` (with the offending ``feature_id`` and bbox) when
  a template is smaller than its declared bbox; the silent-skip
  branch is gone.

### API surface published in stage 0

- ``nav.feature``: ``NavFeature``, ``NavFeatureType`` (9 values),
  ``NavFeatureGeometry`` and ``NavFeatureFlags`` sum types,
  ``NavReliabilityBreakdown``, ``compose_template_features``,
  ``FeatureReliabilityGate``, ``GatedFeatureRecord``,
  ``DEFAULT_RELIABILITY_THRESHOLDS``, plus the named constants
  module.
- ``nav.nav_orchestrator``: ``NavOrchestrator``, ``NavContext``,
  ``NavResult``, ``NavFeatureSummary``, ``NavImageClassifier``,
  ``ImageQualityThresholds``, ``NavImageClassifierResult``,
  ``Provenance``, ``EnsembleConfig``, ``ensemble``,
  ``derive_confidence_rank``, ``build_metadata_dict``,
  ``assert_diagnostic_fields_present``,
  ``STATUS_REASON_INFO_TEMPLATE``.
- ``nav.nav_technique``: ``NavTechnique`` ABC with
  ``__init_subclass__`` auto-registry,
  ``filter_technique_names``, ``NavFeasibilityReport``,
  ``NavTechniqueResult``, every per-technique typed
  ``*Diagnostics`` dataclass with ``CURATOR_FIELDS`` allow-lists,
  ``ConfidenceSpec`` / ``ConfidenceTerm`` /
  ``evaluate_sigmoid_combination``, and ``NavTechniqueManual``
  (abstract-opted-out of the registry).
- ``nav.nav_model``: ``NavModel`` ABC with
  ``__init_subclass__`` auto-registry,
  ``build_models_for_obs``, ``NavModelBodyBase``,
  ``NavModelBodySimulated``, ``NavModelRingsBase``,
  ``NavModelRingsSimulated``, ``NavModelTitan`` (registered stub),
  plus the preserved ``nav.nav_model.rings`` data-model subpackage.
- ``nav.support``: ``NavFilterSpec`` / ``NavFilterKind`` /
  ``apply_filter`` (every kind from the plan implemented),
  ``filter_combo.canonicalize``, ``NavStatusReason`` (15 values,
  ``StrEnum`` on Python 3.11+), ``estimate_image_noise_sigma``,
  ``saturation_mask`` / ``cosmic_ray_mask``,
  ``apply_translation`` / ``sample_dt_bilinear``, plus the shared
  ``types`` aliases.
- ``nav.navigate_image_files.navigate_image_files`` is the top-level
  driver that ``nav_offset`` and ``nav_offset_cloud_tasks`` invoke;
  it preserves the legacy ``(success, metadata)`` contract and now
  routes through ``NavOrchestrator``.

### Tooling and conventions established in stage 0 (binding)

- **Python floor** is 3.11.  ``NavStatusReason`` uses ``StrEnum``;
  do not regress to the ``(str, Enum)`` mixin without bumping
  ``requires-python`` first.
- **Stub honesty.**  Code that is not yet implemented either logs
  the deferral and returns an inert value (the summary-PNG path) or
  raises ``NotImplementedError`` naming the deferred work.  No
  silent placeholder values, ever.  See
  ``navigate_image_files._write_summary_png`` for the canonical
  pattern.
- **Magic constants live in module-level ``ALL_CAPS`` constants
  with a one-line docstring** stating units and intent.  See
  ``DEFAULT_FULL_WELL_DN_12_BIT`` for the canonical example.
- **Broad ``except Exception:`` is reserved for the orchestrator's
  plugin-sandbox sites** (per-NavModel ``create_model`` /
  ``to_features`` / ``to_annotations``, per-NavTechnique
  ``navigate``).  Each site carries a docstring explaining why and
  a per-line justification comment.  Other broad catches are
  not acceptable.
- **Pdslogger output is captured with ``capsys``, not ``caplog``.**
  Pdslogger writes through its own stream handler that does not feed
  the standard logging propagation.  Tests that need to verify a
  WARNING / ERROR / EXCEPTION emission read from
  ``capsys.readouterr().out``.
- **Frozen dataclasses validate on construction.**  Public
  dataclasses use ``__post_init__`` to enforce documented
  invariants; ``object.__setattr__`` is the standard escape hatch
  for normalising inputs in a frozen dataclass.
- **Sphinx ``automodule`` for re-exporting packages uses
  ``:no-index:``** to avoid duplicate cross-references with the
  submodule pages (see ``docs/api_reference/api_feature.rst``).
- **``pyproject.toml [tool.pytest.ini_options]``** has
  ``filterwarnings = ["error"]`` plus a single tolerated rule for
  the third-party ``PytestUnraisableExceptionWarning`` raised by
  astropy's FITS reader on integration tests.
- **CI.**  ``.github/workflows/run-tests.yml`` runs on every PR
  (not just those targeting ``main``); the matrix covers Python
  3.11 / 3.12 / 3.13 / 3.14.  ``codecov-action@v6``.

### Stage 1 entry points (the next AI session starts here)

The "Pending" subsection of "Implementation status (snapshot)" is
the canonical work list.  Stage 1 should land **real-scene
NavModels** so concrete NavTechniques have something to consume.
Recommended order:

1. **``NavModelStars``** — the validated star-catalog reduction
   helpers (aberration, proper motion, multi-catalog precedence,
   incremental search, body / ring conflict marking, the
   ``SCLASS_TO_B_MINUS_V`` lookup, smear-aware PSF rendering)
   were preserved in git history under the deleted
   ``src/nav/nav_model/nav_model_stars.py`` (~1004 lines).
   Recover with
   ``git log --diff-filter=D -- src/nav/nav_model/nav_model_stars.py``;
   structure as helper modules under
   ``src/nav/nav_model/stars/`` (``predicted_snr.py``,
   ``detection.py``, ``aberration.py``, ``conflicts.py``, etc.)
   imported by the new ``NavModelStars.to_features``.  Emit
   ``STAR`` features on the new ``NavFeature`` contract; populate
   ``StarGeometry``, ``StarFlags``, predicted-SNR-driven
   ``reliability``, and a ``NavFilterKind.NONE`` filter spec.

2. **``NavModelBody``** — limb-mask extraction, body-silhouette
   computation, and the body-shape lookup logic were preserved in
   the deleted ``src/nav/nav_model/nav_model_body.py`` (~540 lines).
   ``NavModelBodyBase`` (still present) provides the shared
   annotation rendering.  Per resolution / shape / lighting,
   emit a mix of ``LIMB_ARC``, ``TERMINATOR_ARC``, ``BODY_DISC``,
   and ``BODY_BLOB`` features.

3. **``NavModelRings``** — the four-pass ``RingFeatureFilter`` is
   already preserved under
   ``src/nav/nav_model/rings/ring_filter.py``.  The deleted
   ``src/nav/nav_model/nav_model_rings.py`` (~525 lines) carried the
   per-edge polyline rendering that the new ``to_features`` needs
   to call.  Emit ``RING_EDGE`` features per edge, plus a
   ``RING_ANNULUS`` fallback when many edges pack into few pixels.

For each NavModel:

- Implement ``instances_for_obs(cls, obs)`` to construct one
  instance per body / per planet with visible rings / one stars
  model so the registry-based discovery in
  ``build_models_for_obs`` works.
- Populate ``self._metadata`` during ``create_model`` (the curator
  pass-through is already wired into ``NavResult.model_metadata``).
- Build a fresh ``Annotations`` collection in ``to_annotations`` —
  the orchestrator's ``_collect_annotations`` already merges them
  into ``NavResult.annotations``.

After NavModels land, stage 2 should bring up the concrete
NavTechniques (``BodyDiscCorrelateNav``, ``BodyLimbNav``,
``BodyTerminatorNav``, ``BodyBlobNav``, ``RingEdgeNav``,
``RingAnnulusNav``, ``StarFieldFromCatalogNav``,
``StarUniqueMatchNav``, ``StarRefineNav``, ``CartographicNav``).
The technique-class registration mechanism is already in place;
each new subclass auto-registers via ``__init_subclass__``.

End of "Foundation cleanup status (stage 0)" section.

---

## Context

The current navigation pipeline (`NavTechniqueCorrelateAll` + `NavModelCombined`) fuses every available model into one `model_img` and runs a single masked-NCC pyramid against it. That choice hard-codes fusion rules, makes every scene use every feature, and has no mechanism to notice that part of the model is unreliable or that a different algorithm would work better. In the conversation leading to this plan we proved three concrete failure modes:

- A body that overflows the FOV (C0061085400R): raw-NCC plateaus against a nearly-uniform interior and locks onto the search-window edge.
- A body mostly off-frame (C0061084700R): raw-NCC *and* gradient-NCC both fail — gradient drifts to the boundary via the bidirectional overlap-weight ridge; raw picks a bogus peak because the mask-local mean subtraction makes "mostly dark overlap" look positively correlated.
- Highly irregular bodies (Prometheus class): oops has no DSK support, so the predicted limb is an ellipsoid silhouette that doesn't match reality at any resolution that resolves the body.

The eventual goal is autonomous navigation across Cassini, Voyager, Galileo, and New Horizons: any combination of stars, rings, bodies, partial or full FOV visibility, various lighting geometries, no per-image manual fixup, no statistics accumulated across images. The system must say "here is an offset, here is a calibrated confidence" — or say "I can't do this one" and never lie.

This plan describes an architectural shift from "one model, one technique" to **per-feature navigation with filter-aware techniques and an ensemble orchestrator that votes**. The key currency is a `NavFeature` with uncertainty metadata; techniques consume feature subsets; an orchestrator picks viable techniques, runs them, and reconciles.

---

## Cardinal principles

Four rules that everything else in this plan must obey. Violations are bugs.

1. **No backwards compatibility.** This is a complete replacement. `NavTechniqueCorrelateAll`, `NavModelCombined`, `NavMaster.navigate`'s legacy interface, the `weighted_mask` / `blur_amount` / `uncertainty` / `confidence` fields on `NavModelResult`, and the per-image overlay logic that currently consumes the deleted fields all go. Don't preserve them behind flags. The orchestrator becomes the only nav path; `NavMaster` either becomes a thin facade or disappears entirely (preference: disappear; downstream callers move to `NavOrchestrator` directly).

2. **Image-vs-model asymmetry.** Whatever you do to the *image* must be **global** — do not assume any pixel of the image is at a known geometric position, because SPICE pointing is exactly what we're correcting. Image-side allowed operations: global statistics (whole-image MAD-noise estimate, saturation count), global feature detection (all point sources, all gradient edges, regardless of where they should be), one-time bandpass / despeckle filters applied to the entire array. Per-feature image-side cropping (e.g. "look at the postage stamp around predicted Mimas") is **not allowed** — at full SPICE error the postage stamp can miss the body entirely. *Model*-side, in contrast, is where predicted positions live; templates, limb polylines, ring polylines etc. are model artifacts that get *shifted relative to the full image* during matching. NCC pyramid, DT fitting, pattern matching all conform to this asymmetry by construction (the model moves; the image doesn't get cropped to a guessed location).

3. **No cross-image state.** Every image's navigation must depend only on `obs`, `image`, the static data files, and config. Static data is loaded once at process start and never updated by navigation results. There is no learned per-body residual table that grows with experience; the per-body `config_220_body_shape.yaml` is hand-curated literature and stays put. Cloud-tasks parallelism is the natural consequence.

4. **Angles in config and JSON metadata are degrees; angles in code are radians.** Every YAML config field that names an angle uses degrees (e.g. `max_rotation_deg`, `max_incidence_deg`, `lat_resolution_deg`). Every JSON metadata field that names an angle uses degrees (e.g. `phase_deg`, `rotation_deg`, `sigma_rotation_deg`). Internal Python uses radians (consistent with numpy and oops). The config loader converts degrees → radians at load time; the metadata curator converts radians → degrees at write time. **Existing config files (e.g. `config_07_bootstrap.yaml`) that currently store angles in radians are converted to degrees as part of the renumbering cutover.** No backwards compatibility (Cardinal Principle #1).

## High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│  FeatureExtractor[s] (one per type, plugin)                 │
│  observation ──> list[NavFeature]   (with uncertainty + filter │
│                                   hints, reliability, etc.) │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FeatureReliabilityGate                                     │
│  filters features below per-type reliability thresholds     │
│  (drops invisible stars, empty ring edges, body-only-       │
│  terminator etc.)                                           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  TechniqueRegistry                                          │
│  for each registered NavTechnique:                          │
│      if technique.is_feasible(surviving_features):          │
│          result = technique.navigate(subset)                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  ensemble()  (free function in nav.nav_orchestrator.ensemble)│
│  drops results flagged spurious                             │
│  groups agreeing results (within error ellipses)            │
│  picks highest-confidence consensus group                   │
│  emits final offset + calibrated confidence + diagnostics   │
│  OR emits NoResult + reason if nothing agrees               │
└─────────────────────────────────────────────────────────────┘
```

Nothing above this layer (e.g., `nav_offset.py`, the manual-nav dialog) needs to know what features or techniques are in play. The registry is the extensibility hook.

---

## Part 1 — The `NavFeature` layer

### What a Feature is

A `NavFeature` is the smallest independently-navigable piece of the scene. It carries everything a technique needs to know to use or ignore it. Features are **generated per-image from observation + SPICE prediction only**; no cross-image state.

```python
# src/nav/feature/feature.py (new)
@dataclass(frozen=True, eq=False)
class NavFeature:
    # Identity. The feature_id format is '<type_lc>:<scope>'.
    # - STAR: 'star:<catalog>:<unique_number>'        e.g. 'star:UCAC4:144787700'
    # - LIMB_ARC, BODY_DISC, BODY_BLOB, TITAN_LIMB:
    #     '<type_lc>:<BODY_NAME>'                      e.g. 'limb_arc:MIMAS'
    # - TERMINATOR_ARC: 'terminator_arc:<BODY_NAME>'
    # - RING_EDGE: 'ring_edge:<PLANET>:<edge_name>'   e.g. 'ring_edge:SATURN:A_outer'
    # - RING_ANNULUS: 'ring_annulus:<PLANET>'
    # - CARTOGRAPHIC_MODEL: 'cartographic_model:<BODY_NAME>'
    # IDs must be unique within a single NavResult. Two features with the same
    # ID are an extractor bug. The same physical body may emit multiple
    # *different* feature types (LIMB_ARC + BODY_DISC for a fully-in-FOV
    # moon); their IDs differ by type.
    feature_id: str                    # was `id` (shadowed builtin)
    feature_type: NavFeatureType       # was `type` (shadowed builtin); enum:
                                       # STAR, LIMB_ARC, TERMINATOR_ARC,
                                       # BODY_DISC, BODY_BLOB, RING_EDGE,
                                       # RING_ANNULUS, TITAN_LIMB, CARTOGRAPHIC_MODEL
    source_model: str                  # which model generated it
                                       # ('stars','body','rings',...)

    # Geometry in the image (ext-FOV coords)
    geometry: NavFeatureGeometry       # NavFeatureGeometry sum type — see Part 0;
                                       # variants per feature type
    subject_range_km: float            # was `range`/`subject_range`; distance
                                       # to observer in km (for depth-sort)

    # Rendered template (if the technique needs one)
    template_img: NDArrayFloat | None # per-feature rendered model (small postage stamp,
                                      # positioned in ext-FOV coords)
    template_mask: NDArrayBool | None # True where template carries signal

    # Note: the sensor-vs-extfov-padding mask (obs.extfov_data_sensor_mask())
    # is image-wide, identical for every feature, and lives on the NavContext
    # passed to techniques — not on individual NavFeature instances.

    # Uncertainty (image plane, pixel units)
    position_cov_px: np.ndarray       # 2x2; anisotropic; drives anisotropic blur
    intensity_sigma_rel: float        # relative brightness uncertainty (albedo drift etc.)

    # Filter hints
    preferred_filter: NavFilterSpec   # chosen by extractor based on the above (see Part 2)

    # Self-assessment
    reliability: float                # 0..1, single number
    reliability_reasons: NavReliabilityBreakdown  # structured per Part 1 spec; see below
    usable_types: frozenset[NavFeatureType]  # which technique-types are allowed to consume
                                             # this feature (for partial use: a body whose
                                             # limb is reliable but terminator isn't will emit
                                             # only LIMB_ARC, not TERMINATOR_ARC)

    # Optional technique-specific flags — structured per feature type, not a dict-of-Any
    flags: NavFeatureFlags            # union of typed dataclasses keyed by feature type
```

**Structured side types** (defined in `src/nav/feature/feature.py`):

```python
@dataclass(frozen=True)
class NavReliabilityBreakdown:
    """Per-component contributions to a feature's reliability score.

    All fields are optional [0, 1] contributions. Missing fields are 'not
    applicable' for this feature type, not zero.
    """
    predicted_snr: float | None = None        # STAR
    visible_arc_fraction: float | None = None # LIMB_ARC, TERMINATOR_ARC, RING_EDGE
    incidence_factor: float | None = None     # LIMB_ARC
    albedo_penalty: float | None = None       # TERMINATOR_ARC
    shadow_occluded_fraction: float | None = None  # RING_EDGE
    visible_lit_fraction: float | None = None # BODY_DISC
    overflow_fraction: float | None = None    # BODY_DISC
    blob_snr: float | None = None             # BODY_BLOB
    blob_extent_px: float | None = None       # BODY_BLOB
    in_body_silhouette: bool | None = None    # STAR (predicted star inside a body)
    in_saturation_or_cosmic: bool | None = None  # STAR
    smear_length_ok: bool | None = None       # STAR

# Feature flags use a sum type rather than dict-of-Any.
NavFeatureFlags = (
    StarFlags | LimbArcFlags | TerminatorArcFlags | RingEdgeFlags |
    BodyDiscFlags | BodyBlobFlags | RingAnnulusFlags | CartographicModelFlags
)

@dataclass(frozen=True)
class RingEdgeFlags:
    is_straight_line: bool                    # triggers rank-1 covariance handling
    polarity_predictable: bool                # always False; tracked in Part 13b

@dataclass(frozen=True)
class BodyDiscFlags:
    overflow_fov_fraction: float              # see definitions block above

# (other Flag types similarly small; one per feature type)
```

### Feature types

| Type | Produced by | What the reliability captures |
|---|---|---|
| `STAR` | stars extractor | detectability: predicted SNR at this pixel, proximity to bright body edges, saturation risk |
| `LIMB_ARC` | body extractor (one per body) | visible arc length / total limb, limb incidence-angle histogram (glancing limbs are softer), crater scale, ellipsoid residual |
| `TERMINATOR_ARC` | body extractor (one per body) | terminator length, albedo-sensitivity scalar, phase-angle factor |
| `BODY_BLOB` | body extractor | irregular body or body-under-resolution-limit; carries only predicted centroid + size |
| `BODY_DISC` | body extractor | full-disc correlation feasibility: body significantly inside FOV with well-lit interior |
| `RING_EDGE` | rings extractor (one per edge) | edge sharpness (step/soft), radial uncertainty (km → px at this point), projected-curvature scalar (flat vs curved in image), shadow occlusion |
| `RING_ANNULUS` | rings extractor | full-ring-region template when many edges pack into few pixels (low-res encounter with ring system) |

New types are added by adding a `NavFeatureType` enum value and an extractor. The registry discovers them at import time.

### NavFeatureExtractors (plugins)

```python
# src/nav/feature/extractor.py (new)
class NavFeatureExtractor(ABC):
    accepted_subject_types: frozenset[str]   # 'body', 'ring', 'stars'
    @abstractmethod
    def extract(self, obs: ObsSnapshotInst, context: ExtractorContext) -> list[NavFeature]: ...
    @abstractmethod
    def is_applicable(self, obs: ObsSnapshotInst) -> bool: ...
```

One extractor per feature *family*:
- `StarFeatureExtractor` — wraps current `NavModelStars` star catalog; computes predicted-SNR per star.
- `BodyLimbExtractor` — one `LIMB_ARC` per body present in FOV.
- `BodyTerminatorExtractor` — one `TERMINATOR_ARC` per body; may emit nothing if terminator is short or near-invisible.
- `BodyBlobExtractor` — one `BODY_BLOB` per body when shape class is irregular OR resolution is too low.
- `BodyDiscExtractor` — one `BODY_DISC` per body when disc is usable (well-lit, substantially in-FOV).
- `RingEdgeExtractor` — one `RING_EDGE` per configured ring edge for the planet in FOV.
- `RingAnnulusExtractor` — one `RING_ANNULUS` when multiple ring edges compress into a small image area.

Extractors are stateless; they compute everything from `obs` + config + SPICE prediction + static body/ring tables (Part 5). This satisfies the no-cross-image-state constraint.

### NavModel changes — three outputs, no NavModelResult

`NavModelResult` is **deleted entirely** along with `NavModelCombined`. The four obsolete fields (`weighted_mask`, `blur_amount`, `uncertainty`, `confidence`) and the five remaining ones (`model_img`, `model_mask`, `range`, `stretch_regions`, `annotations`) all go. Pixel data, when a technique needs it, lives on `NavFeature.template_img` for `BODY_DISC` / `RING_ANNULUS` / `CARTOGRAPHIC_MODEL` features only — most features (`STAR`, `LIMB_ARC`, `TERMINATOR_ARC`, `RING_EDGE`, `BODY_BLOB`) have `template_img is None` because their geometry alone is sufficient. Render-only callers (manual-nav UI overlay, summary PNG) read from features and from the existing `Annotations` collection (below) instead.

Each `NavModel` exposes **three outputs** consumed by the orchestrator:

```python
class NavModel(NavBase):
    name: str                                    # e.g. 'body:MIMAS', 'rings:SATURN', 'stars'

    # Existing — populates internal state including ._metadata. Already in the tree.
    # Per Part 0 §12: `always_create_model` is dropped from the new public
    # contract. The orchestrator always wants a model to be created when it
    # calls to_features() / to_annotations(). Existing implementations may
    # keep `always_create_model` as a private internal short-circuit.
    def create_model(self) -> None: ...

    # New — produce navigation-ready features. May internally consult render state.
    def to_features(self, context: NavContext) -> list[NavFeature]: ...

    # New — produce per-model annotations (overlays + text labels) for the summary
    # image. Called regardless of whether features end up gated out or whether the
    # technique using them succeeded — annotations summarize the *predicted scene*,
    # not the navigation result.
    def to_annotations(self, context: NavContext) -> Annotations: ...

    # Existing — per-model diagnostic dict (sub-solar geometry, elapsed time, etc.).
    # Continues to populate during create_model() as today.
    metadata: dict[str, Any]
```

The orchestrator's call sequence per model:

```python
for model in selected_models:
    model.create_model()
    nav_features.extend(model.to_features(context))
    summary_annotations.add_annotations(model.to_annotations(context))
    model_metadata[model.name] = model.metadata
```

**Annotations: reuse existing infrastructure, no new classes.** The `src/nav/annotation/` package already provides:

- `Annotation` (`annotation.py:9`) — single annotation: overlay mask + color + thickening + `AnnotationTextInfo` + avoid mask.
- `Annotations` (`annotations.py:16`, `NavBase` subclass) — collection class with `add_annotations()` for merging from multiple sources; owns the rendering / label-placement logic.
- `AnnotationTextInfo` (`annotation_text_info.py:46`) — multi-line text + position + direction.

Each `NavModel.to_annotations()` returns a freshly-constructed `Annotations` collection populated with `Annotation` instances of the existing type. The orchestrator merges per-model collections via `Annotations.add_annotations()` — exactly the existing API. No new annotation classes; no new style/color config; the existing rendering code continues to take its existing inputs. Visual treatment of gated-out features (muted vs same as kept) is tracked in Part 13b as a styling-polish issue.

The orchestrator additionally emits its own `Annotations` for items that have no source NavModel: image name, final offset arrow, confidence label, status banner. These merge into the same collection.

**Per-NavModel changes (no new abstract base classes — subclass methods added concretely):**

- `NavModelStars`: predicted-SNR helper for stars (using `obs.star_psf()` + `NavContext.image_noise_sigma` + per-filter `mag_offset_table`); `to_features()` builds `STAR` features; `to_annotations()` builds star-box overlays + name/magnitude labels.
- `NavModelBody`: helpers returning limb polyline (from existing mask edge) and terminator polyline (from incidence backplane); per-body shape lookup from `config_220_body_shape.yaml`; `to_features()` builds the appropriate body feature types per resolution / shape rules; `to_annotations()` builds body silhouette overlay + name label.
- `NavModelRings`: per-edge polyline + curvature scalar + shadow-occlusion fraction; pulls uncertainty from existing `config_3N0_*_rings.yaml`; `to_features()` builds `RING_EDGE` / `RING_ANNULUS` features via the existing `RingFeatureFilter`; `to_annotations()` builds ring-edge polyline overlays + ring-name labels.

**`RingEdgeExtractor` reuses the existing `RingFeatureFilter`** (`src/nav/nav_model/rings/ring_filter.py`) for the four-pass selection of which ring features to render — date validity, radius-in-FOV, resolvability, fade-conflict. The extractor invokes the filter with observation parameters and emits `RING_EDGE` features only for edges that survive. Do not reimplement filter passes. The extractor's *new* job is the per-edge work the filter doesn't do: polyline geometry, projected curvature, shadow occlusion, projection of `rms` (km) → `position_cov_px`.

---

### Image-quality preprocessing (global, runs once per image)

Three image-level checks happen in the orchestrator before any technique runs. All operate on the full image — Cardinal Principle #2 — and produce masks / statistics that go on `NavContext` for every downstream consumer:

- **Saturation map.** Pixels at or near full-well DN (per-instrument value from the `noise:` block of the matching `config_4N0_inst_*.yaml`) are marked. The global gradient computation excludes them; the star detector treats saturated peaks specially (truncated PSF, unreliable centroid — either reject if isolated or use moments instead of fit).
- **Cosmic-ray / hot-pixel rejection.** Single-pixel spikes 5+σ above their immediate neighbors (median 3×3 vs raw) are masked. This step is global and pre-detection; without it a single hot pixel produces a fake "star" detection that derails pattern matching. Existing GOSSI / Voyager / Cassini pipelines all need this; their archives have many cosmic ray hits per long-exposure image.
- **Smear-aware PSF via the `eval_rect(movement=...)` parameter.** Long exposures with non-zero spacecraft attitude rate smear stars into trails. The existing `psfmodel.GaussianPSF` provides smear support directly: `eval_rect(rect_size, offset, *, movement=(my, mx), movement_granularity=0.1, ...)` integrates the PSF along a line segment of length `sqrt(my² + mx²)` in the `(Y, X)` direction. The internal `_eval_rect_smeared` (PSF base) does the integration; `movement_granularity` is the step size in pixels for the smear-line sampling.

  Approach:
  - Compute smear vector `(my, mx)` (pixels) from SPICE pointing at the start ET and end ET of the observation. Even when the absolute pointing is wrong, the *relative* difference between the two brackets gives the camera rotation during the exposure, which is what produces the smear. Project that small relative rotation through the camera model to a pixel-plane displacement vector.
  - For star *detection* (matched-filter pre-pass): pre-render the smeared template once with `psf.eval_rect(box_size, offset=(0.5, 0.5), movement=(my, mx))` and use it as the cross-correlation kernel against the image.
  - For star *refinement*: the public `PSF.find_position` does **not** currently accept `movement` — its signature is fixed (verified by reading `psf.py` line 581 and confirmed there's no kwargs passthrough). Two options for smear-aware refinement:
    1. Subclass `GaussianPSF` and override `eval_rect` to bake in movement, then `find_position` will use the smeared eval. Lightweight, no upstream change needed.
    2. Open a `psfmodel` upstream change to thread `movement` / `movement_granularity` kwargs through `find_position` to its internal `eval_rect` calls.
    Option 1 is the path the new pipeline takes (zero coordination cost); option 2 is documented as a future upstream improvement.

  Mission applicability:
  - **Cassini**: SPICE-bracket geometry is well-tested elsewhere and the brackets reliably represent the true relative rotation. Enabled by default.
  - **Voyager / Galileo / NHLORRI**: same approach is theoretically applicable but **untested in this project**. Implementation parameterizes uniformly (the smear math doesn't care which mission), but ships with a config flag (`stars.smear_from_spice_brackets: bool`, default `true` for Cassini, `false` for the others until validated). Galileo and Voyager are mostly star-blind anyway.
  - The existing `stars.max_smear: 100` config in `config_110_stars.yaml` is currently dead code (no consumer in today's tree). The new pipeline becomes its first consumer: smear lengths beyond `max_smear` cause the star to be rejected from the star feature list (PSF too elongated to fit reliably even with the smear-aware template). The 100 px value is liberal: it's a "reject above this" hard gate, not a "trust below this" gate. Empirically, smear < 5 px is "minor", 5–20 px is "significant but well-fittable", 20–100 px is "extreme but worth trying once with reduced reliability". Tighten in Phase 5 if the relaxed gate generates too many low-confidence star features.

### Image-quality classification (quick-fail before techniques)

Before any extractor runs, the orchestrator assigns the image to one of these classes by looking at three cheap global statistics on the entire sensor area: `image_noise_sigma`, fraction of pixels at saturation DN, and fraction of pixels equal to the per-instrument "missing-data" marker. Each class drives a deterministic outcome — most "bad" classes never invoke an extractor at all, so a corrupted image fails in milliseconds with a clear reason.

| Class | Detection | Outcome |
|---|---|---|
| `clean` | saturation_frac < threshold, missing_frac < threshold, noise_sigma in normal range | Run full pipeline. |
| `blank` / `dark_frame` | mean DN ≈ 0, max DN < few × noise; noise_sigma very low | `status='failed', status_reason='no_signal_in_image'`. No technique runs. |
| `fully_overexposed` | > 80% of pixels at full-well DN | `status='failed', status_reason='image_overexposed'`. No technique runs. |
| `mostly_missing_data` | > 30% of pixels equal the per-instrument marker (e.g. Voyager 0-fill, Galileo dropout flag) | `status='failed', status_reason='missing_data_dominant'`. |
| `partial_data_dropout` | 5%–30% missing pixels | Pipeline runs, but features overlapping dropouts are gated out and source-image filter is skipped (kernel windows that span the dropout produce artifacts). |
| `alternating_lines` | Voyager-style interlaced readout shows alternating-line structure (FFT spike at Nyquist) | Pipeline runs after a per-line median repair pass that fills the missing field by interpolation. |
| `shortened_lines` / `truncated_readout` | only top-N rows have data | Pipeline runs against the truncated valid region; rest treated as missing data. |
| `ccd_bloom_dominant` | bright vertical streaks from saturated pixels | Mark bloom columns as missing data; otherwise run pipeline. |
| `corrupt` | image file failed to parse / read; `obs.from_file(...)` or `obs.data` access raised an exception | `status='failed', status_reason='image_corrupt'`. No technique runs. (Per Part 0 §21.) |
| (`noisy` is **not** a class; see Part 0 §7) | high read noise = `noise_sigma > per-instrument threshold` | Pipeline runs as `image_class='clean', flags=['noisy']`; per-feature reliability gates use the elevated noise so faint stars are correctly rejected. |

Per-instrument detection thresholds (saturation_dn, marker_value, expected_noise_dn) live in the `noise:` block of each `config_4N0_inst_*.yaml`. The classifier's decision goes into `NavResult.feature_inventory` as a top-level diagnostic so operators can see which images failed for which reason.

### Filter-aware photometry

Every image is taken through a specific filter; both star detection and body model rendering must account for the filter's spectral response. Two distinct concerns:

**Star magnitudes by filter.** Catalog magnitudes are V-band (UCAC4 / YBSC). For predicted-SNR computation in `StarFeatureExtractor`, the catalog magnitude is converted to in-band magnitude using a per-instrument-per-filter color transformation. Sources for the transformations:
- Cassini ISS: published transmission curves × CCD QE (in `inst_calib_data` directory of the CISS calibration release; transformations are tabulated by stellar B-V color).
- GOSSI: similar transmission curves.
- Voyager ISS, NHLORRI: filter response curves from PDS calibration archives.

Implementation: each `config_4N0_inst_*.yaml` carries a per-camera `mag_offset:` block with per-filter-combo `mag_offset_table` keyed by B-V color bin (spectral class is mapped to B-V via the existing `SCLASS_TO_B_MINUS_V` table). The star extractor pulls the offset for each star using its catalog spectral_class / B-V, computes `predicted_mag_in_band = catalog_v_mag + mag_offset`, then proceeds with the existing flux-to-DN computation. Stars with no catalog color get a default offset (with reduced reliability).

**Body / ring model brightness by filter.** The Lambert-shading model normalized brightness range is filter-independent (it's just `cos(incidence)`), but the *match* between the image and the model is filter-dependent: methane-band images of icy moons show enhanced contrast at certain incidence angles, narrow-band UV images can flip the apparent light/dark structure due to surface ice vs silicate response. Two responses:
- For `BodyDiscCorrelateNav` and `RingAnnulusNav`, the `use_gradient='auto'` mode already chooses the better of raw and gradient correlation; in narrow-band filters where raw intensity is unreliable, gradient mode dominates naturally. No filter-specific tuning required.
- For `BodyLimbNav`, the limb position is geometric and filter-independent (modulo Titan-class atmospheric haze, which is its own technique). No filter handling needed.
- The `config_220_body_shape.yaml` `albedo_variation` value is a clear-filter average; for narrow-band images it's a lower bound (variation can be larger). Reliability gate uses it conservatively.

**Per-instrument star PSF.** Each camera/filter combination has its own star PSF. The existing `obs.star_psf()` returns the right PSF for the current obs, sourced from per-instrument config. The new pipeline calls it unchanged — no plan-level work needed beyond making sure the SNR / smear / detection code paths use the obs-supplied PSF, not a global default.

**Star-poor missions.** Galileo SSI and Voyager ISS imaging frames have lower sensitivity than Cassini ISS or NHLORRI; predicted SNR is below threshold for catalogued stars in the majority of *science* frames (long-exposure body or ring imaging). Both archives also contain star-calibration frames where stars are well-detected; those are not the typical case but exist and are exercised by the test library. The implications:
- The orchestrator must not require stars to navigate; body / ring techniques carry the load on routine science frames in these missions.
- `StarFeatureExtractor.is_applicable` returns `False` quickly when predicted-SNR for every star in catalog is below threshold — no detection sweep wasted.
- Test library (Part 10) intentionally over-samples star-only Cassini / NHLORRI scenes (where stars commonly work) and star-absent Galileo / Voyager science scenes (where they don't), so calibration and regression coverage matches reality. Star-cal frames from VGISS / GOSSI are valid library entries when they exist for the right instrument/camera mix.

These three are mandatory; their cost is small (one median-filter pass + one Sobel + comparisons). Skipping them would let single-pixel artifacts dominate every star-pattern match.

### Position covariance per feature type

How `NavFeature.position_cov_px` (or per-vertex equivalents on polyline payloads) is computed. Quadrature combination because error sources are independent.

**LIMB_ARC** — polyline payload (`LimbPolyline`) carries per-vertex anisotropic σ:

```
σ_normal_per_vertex_px = sqrt(
    ellipsoid_residual_km²                     # config_220_body_shape.yaml
  + crater_scale_km²                            # config_220_body_shape.yaml
  + (incidence_factor(i) × limb_softness_km)²  # per-image, PSF-derived
  + spice_orbital_residual_km²                  # ~0.5 km for major moons; see definitions block
) / km_per_pixel_at_vertex                      # from oops backplane

σ_tangent_per_vertex_px = ~0.5                  # polyline-sampling resolution
```

`incidence_factor(i)` ramps from 0 at sub-solar (`i = 0`) to ≈ 1 at moderate-glance (`i = 60°`) and saturates near the terminator. Concretely:

```
incidence_factor(i) = clip( (1/cos(min(i, 85°))) − 1,  0.0,  4.76 )
```

So `i=0° → 0`, `i=60° → 1`, `i=80° → 4.76` (clipped), `i≥80° → 4.76` (clipped). The "−1" gives the desired sub-solar zero; the cap at 4.76 corresponds to `1/cos(80°) − 1` and reflects the empirical observation that limb pixels with incidence > 80° contribute essentially zero useful information about the limb position (the cosine projection is so steep that the limb is effectively perpendicular-to-line-of-sight). Treating any pixel beyond 80° as having the same softness is the right behavior; the cap at exactly the 80° value (4.76) is principled, not arbitrary.

Occlusion / shadow handled at extraction by vertex cropping (no `range` field on the polyline). Technique-level 2×2 covariance comes from the M-estimator's information matrix after LM fitting; long curved limbs aggregate to tight covariance, short straight-ish limbs become rank-1.

**TERMINATOR_ARC** — same shape as LIMB_ARC plus `albedo_variation × terminator_softness_km` and `photometric_model_error_km` quadrature terms; reliability ceiling lower than LIMB_ARC.

**STAR** — scalar payload; `position_cov_px` is the centroid Cramér-Rao Lower Bound (CRLB):

```
σ_along_smear_px  = sqrt(L²/12 + σ_PSF²) / sqrt(SNR_predicted)
σ_across_smear_px = σ_PSF / sqrt(SNR_predicted)
```

where:
- `L` is the smear length in pixels (scalar `sqrt(my² + mx²)`).
- `L² / 12` is the variance of a uniform distribution of length L. **Assumption**: spacecraft attitude rate is approximately constant during the exposure (uniform smear). This holds for almost all imaging — attitude jerk events are rare and typically excluded from imaging campaigns. If the rate is non-uniform during the exposure (e.g., a thruster fired mid-exposure), the smear distribution is non-uniform and `L²/12` is wrong; the practical effect is a wider smear than predicted, which manifests as `position_cov_px` being a *lower bound* rather than the true covariance. Phase 5 calibration flags any library image showing attitude-jerk-during-exposure behavior.
- `σ_PSF` is the per-pixel PSF Gaussian σ.
- `SNR_predicted` is the **integrated** SNR across the full PSF support, computed as `total_signal_DN / sqrt(total_signal_DN + read_noise_DN² × N_pixels_in_aperture)` where the aperture is the smeared-PSF box. *Not* per-pixel SNR. (The denominator under the square root is the variance from shot noise + read noise; the numerator is the total expected signal in DN.)

Anisotropic when smeared (major axis along smear direction). Smear vector `(my, mx)` is computed from the SPICE pointing brackets (camera attitude at start ET vs end ET of the exposure). When the bracket window equals the exposure window, use as-is; when the bracket window is wider than the exposure (e.g. SPICE CK is sampled at a coarser cadence), scale linearly: `(my, mx)_effective = (my, mx)_bracket × (exposure_time / bracket_window)`. Detection uses the existing `psf.eval_rect(..., movement=, movement_granularity=)` path — same API the existing star renderer in `nav_model_stars.py:773` already uses.

**RING_EDGE** — polyline payload (`RingEdgePolyline`) per-vertex:

```
σ_radial_per_vertex_px = sqrt(rms_km² + dynamical_amplitude_km²) / km_per_pixel_radial
σ_along_edge_per_vertex_px = ~0.5
```

`rms_km` from mode-1 entry of existing ring catalog (`config_3N0_*_rings.yaml`); `dynamical_amplitude_km` is sum of higher-mode amplitudes. Shadow/body-occlusion handled at extraction. Polyline sets `is_straight_line` flag when curvature below threshold (drives rank-1 covariance handling).

**BODY_DISC** — pixel-template payload; `NavFeature.position_cov_px = None`. Covariance comes from inverting the local Hessian of the NCC surface at the peak post-fit.

**CARTOGRAPHIC_MODEL** — pixel-template payload; same shape as BODY_DISC; technique-level covariance from peak sharpness.

**BODY_BLOB** — scalar payload:

```
σ_centroid_px = predicted_diameter_px / (2 × sqrt(N_lit_pixels) × SNR_image_in_bbox)
```

Derivation: a brightness-weighted-moment centroid over `N` pixels each with brightness uncertainty `σ_pixel = signal / SNR` and spread by characteristic radius `R` has standard deviation roughly `R / (sqrt(N) × SNR)` — this is the standard error-of-the-mean scaling for a centroid where each pixel contributes its position weighted by its brightness. For a roughly-uniform-brightness disc of diameter `d`, `R ≈ d/2` (the disc radius is the right scale for "characteristic spread"); the formula uses that. Reference: Howell, *Handbook of CCD Astronomy*, 2nd ed., §5 — centroid CRLB for an extended uniform source. The factor of 2 is the radius-vs-diameter conversion; the formula is otherwise dimensionally correct.

Isotropic 2×2 = `σ² · I`. Confidence intrinsically capped at 0.4 in `config_510_techniques.yaml`.

**RING_ANNULUS** — pixel-template payload (multi-ring composite); same shape as BODY_DISC.

**TITAN_LIMB** — placeholder; never produced (TitanNav ships as a stub; the real atmospheric-body algorithm is tracked in Part 13b).

### Reliability — comprehensive

Reliability is a [0, 1] scalar on every emitted `NavFeature`, computed at extraction time from quantities the extractor knows. Per-feature, not per-feature-type; per-image, no cross-image state.

**Distinction from confidence**: reliability lives on a `NavFeature` and answers "how trustworthy is this *as input data*". Confidence lives on a `NavTechniqueResult` and answers "how trustworthy is this *as an offset*". Different objects, different stages.

**Per-type formulas** (sigmoid-of-linear-combination form; α coefficients calibrated in Phase 5):

```
STAR:          sigmoid(α₀ + α₁·(predicted_snr − threshold_snr))
                × not_in_body_silhouette × not_in_saturation_or_cosmic
                × not_too_close_to_image_edge × smear_length_ok
LIMB_ARC:      sigmoid(α₀ + α₁·visible_arc_fraction
                          + α₂·sigmoid(visible_arc_px / 50)
                          − α₃·mean_incidence_factor)
TERMINATOR_ARC: like LIMB_ARC + albedo penalty + phase-angle gate
RING_EDGE:     catalog_default_reliability × visible_arc_fraction
                × (1 − shadow_occluded_fraction)
                × sigmoid(α·mean_emission_factor − β)
RING_ANNULUS:  mean(constituent_reliabilities) × sigmoid(radial_extent_px / 50 − 1)
BODY_DISC:     visible_lit_fraction × (1 − overflow_fraction)
                × sigmoid(body_diameter_px / 30 − 1)
BODY_BLOB:     sigmoid(snr_in_bbox) × sigmoid(extent_px / 8 − 1) × 0.4   # hard cap
CARTOGRAPHIC:  same shape as BODY_DISC with higher ceiling
```

**Gate threshold** in `config_520_features.yaml`, per-type, with per-instrument overrides. Default placeholders, calibrated in Phase 5 from image library.

**Gated-out features**:
1. Not passed to any technique (excluded from technique input lists).
2. Recorded in `NavResult.feature_inventory` with the reliability score and gate-rejection reason.
3. Annotations *still* emitted — the summary image depicts the predicted scene, not the navigation outcome. Visual treatment of dropped features (muted vs same as kept) is tracked in Part 13b as a styling-polish issue.

**Reliability gate vs technique infeasibility**: two distinct gates, composed not duplicated. Reliability filters bad-data features individually; infeasibility checks the surviving feature set against per-technique requirements (e.g., `StarFieldFromCatalogNav` needs ≥3 STAR features). A feature passing reliability is *eligible* to feed a technique; whether the technique runs depends on how many such eligible features remain.

### Definitions used by reliability and feasibility gates

These quantities recur throughout reliability formulas and feasibility checks. Defined here once to avoid AI implementing different definitions in different places:

- **`visible_arc_fraction`** (LIMB_ARC, TERMINATOR_ARC): `arc_length_inside_FOV_km / total_arc_length_km` measured on the *predicted* polyline before any extractor cropping (cropping happens after this measurement). Ranges 0–1.
- **`visible_arc_px`** (LIMB_ARC, TERMINATOR_ARC, RING_EDGE): cumulative arc length of the surviving polyline in pixels, computed as the sum of segment lengths between consecutive vertices after extraction-time cropping. Densified at ~1 vertex per pixel (Part 3) so this approximates vertex count.
- **`overflow_fraction`** (BODY_DISC): `1 − (predicted_disc_area_inside_FOV_px / total_disc_area_px)` where `total_disc_area_px = π × (predicted_disc_radius_px)²` and the inside-FOV area is computed by clipping the predicted ellipse silhouette to the sensor rectangle. Ranges 0 (fully in FOV) to 1 (fully off-frame).
- **`visible_lit_fraction`** (BODY_DISC): fraction of the predicted disc whose `cos(incidence) ≥ 0` (the lit hemisphere) AND that is inside the sensor FOV.
- **`shadow_occluded_fraction`** (RING_EDGE): fraction of polyline vertices dropped because they fell inside the predicted planet shadow before reaching the extractor's emit step.
- **`N_lit_pixels`** (BODY_BLOB): count of pixels inside the predicted-bounding-box that have `predicted_DN > 3 × image_noise_sigma`. Predicted, not measured — Cardinal Principle #2.
- **`SNR_image_in_bbox`** (BODY_BLOB): `mean(predicted_DN_in_lit_pixels) / image_noise_sigma`. Predicted brightness uses Lambert shading: `predicted_DN(pixel) = albedo_mean × cos(incidence(pixel)) × instrument_throughput × exposure_time`, where the per-pixel incidence comes from the existing `oops` `Backplane.incidence_angle()` query. The bbox is the predicted-body bounding rectangle from `obs.inventory_body_in_extfov`. Image-derived noise (`image_noise_sigma`) is the global MAD estimate from `NavContext`. The "lit pixels" are those with `cos(incidence) > 0` AND inside the predicted body silhouette.
- **`brightness_margin_to_next_catalog_star_mag`** (StarUniqueMatchNav): the difference in `predicted_mag_in_band` between the brightest catalog star in extfov and the second-brightest catalog star in extfov, restricted to stars whose `predicted_mag_in_band ≤ detection_mag_threshold + 2`. Larger values mean clearer uniqueness.
- **`instrument_psf_fwhm`** (DAOPHOT shape cuts): `obs.star_psf().fwhm()` — already per-camera-per-filter via the existing `obs.star_psf()` lookup.
- **`mean_emission_factor`** (RING_EDGE reliability): mean of `1/cos(min(emission_angle, 85°))` over the surviving polyline vertices, where `emission_angle` is the angle between the local ring-plane normal and the line of sight. Caps at 11.5 by construction; clipped to 10 like `incidence_factor`.
- **`albedo_penalty`** (TERMINATOR_ARC reliability): `albedo_inflation_factor × albedo_variation` from `config_220_body_shape.yaml`. Same factor used in σ_normal derivation; subtracted from the sigmoid argument so high-albedo-variation bodies have lower terminator reliability.
- **`phase_angle_factor`** (TERMINATOR_ARC reliability): `sin(phase_angle)` — peaks at 90° crescent, near-zero at full or new. Multiplies the sigmoid result; sub-solar-illumination scenes correctly emit weak terminator features.
- **`body_diameter_px`** (BODY_DISC reliability): `2 × max(predicted_disc_semi_major_axis_px, predicted_disc_semi_minor_axis_px)`. The body's larger image-plane axis in pixels.
- **`radial_extent_px`** (RING_ANNULUS reliability): width of the multi-ring region in the radial direction in pixels, measured as the difference between the outermost and innermost edge radii projected to the image plane.
- **`extent_px`** (BODY_BLOB reliability): the longer of the two body-bbox axes in pixels (`max(bbox_height_px, bbox_width_px)`).
- **`snr_in_bbox`** (BODY_BLOB reliability): identical quantity as `SNR_image_in_bbox` above; aliased here because the reliability formula uses the shorter name. The two terms are interchangeable.
- **`spice_orbital_residual_px`** (LIMB_ARC σ_normal contribution; **note: pixel units, not km, despite the name in the formula**): the predicted-vs-actual focal-plane position uncertainty for a major moon, derived from SPK kernel ephemeris uncertainties and the obs's km/px scale. Typically ~0.05 px for major Saturn / Jupiter moons; up to 1 px for irregular satellites with sparse SPK coverage. The formula as written treats it in km (`spice_orbital_residual_km²` summed in quadrature with other km-scale terms then divided by km/px); read the variable name as `spice_orbital_residual_km` for arithmetic clarity even though typical values quoted in pixels are convertible via the at-vertex km/px scale.

**`SCLASS_TO_B_MINUS_V`** lookup table: already present in `src/nav/nav_model/nav_model_stars.py` (existing constant; do not duplicate). When the stars module is split into a package (Part 8), this constant moves to `src/nav/nav_model/stars/predicted_snr.py` and is exported via `__init__.py`.

**`limb_softness_km` clarification**: defined in Part 1 position-covariance section as `star_psf_sigma_px × km_per_pixel_at_limb`. **Use km/px at the limb vertex specifically**, not km/px at FOV center — vertex-by-vertex computation, since limb resolution varies across the body (foreshortening near the silhouette). The km/px at each polyline vertex comes from the existing `oops` Backplane `resolution(...)` query.

## Part 2 — Filter system

### Core abstraction

```python
# src/nav/support/filters.py (new) — names per Part 0 renaming
@dataclass(frozen=True, eq=False)
class NavFilterSpec:
    kind: NavFilterKind               # NONE, ISOTROPIC_GAUSSIAN, ANISOTROPIC_GAUSSIAN,
                                      # BANDPASS_DOG, DISTANCE_TRANSFORM,
                                      # GRADIENT_OF_GAUSSIAN, MORPH_DILATE
    sigma_xy: tuple[float, float]     # for isotropic/gradient-of-gaussian
    covariance_px2: np.ndarray        # for anisotropic gaussian (2x2)
    dt_half_width_px: float           # for distance transform
    bandpass_cutoffs_px: tuple[float, float]  # for DoG
    align_axis: tuple[float, float] | None    # optional direction to align anisotropic filter
                                              # (e.g. limb tangent); if None, axis-aligned

def apply_filter(arr: NDArrayFloat, spec: NavFilterSpec) -> NDArrayFloat: ...
```

### Where filters apply

Three places, distinct purposes, each explicit:

1. **Image-wide pre-filter** (optional, per mission): a DoG bandpass to kill stray-light / scattered-light gradients that bias mask-local NCC. Configured per-instrument in `config_4N0_inst_*.yaml`; disabled by default; enabled for GOSSI and Voyager ISS where stray light is common. Applied once to the image before any feature extraction.

2. **Per-feature filter on the feature's template + its image region**: Each `NavFeature` carries a `preferred_filter` set by its extractor based on its own uncertainty. The feature-consuming technique retrieves the relevant postage-stamp of the image around the feature's bbox, applies the same filter to both image patch and template, *then* runs the matching metric. This is the main user-visible filter path.

3. **Global technique-level preprocessing**: some techniques have intrinsic preprocessing (e.g., limb extraction uses Gradient-of-Gaussian internally). Hidden from the orchestrator.

### Which filter by feature type — decision table

| Feature type | Default filter | Parameters from |
|---|---|---|
| `STAR` | NONE | (stars are PSF-sized already) |
| `LIMB_ARC` (regular body) | GRADIENT_OF_GAUSSIAN, σ from limb uncertainty | body shape table + crater-scale / km_per_px_at_limb |
| `LIMB_ARC` (mild-irregular) | ANISOTROPIC_GAUSSIAN blur on template *and* image, σ_normal ≫ σ_tangent | per-body ellipsoid residual + crater scale |
| `TERMINATOR_ARC` | ANISOTROPIC_GAUSSIAN, larger σ_normal than limb (albedo adds) | body albedo variation table + phase-angle factor |
| `BODY_BLOB` | NONE (blob centroid fit; no template matching) | — |
| `BODY_DISC` | AUTO (see Part 3) between raw and gradient | — |
| `RING_EDGE` (curved) | ANISOTROPIC_GAUSSIAN, σ along radial = ring radial uncertainty in px | ring edge table |
| `RING_EDGE` (flat/straight-line flag) | ANISOTROPIC_GAUSSIAN, σ *across* line = radial uncertainty, σ *along* line = large (offset-along-line is unobservable) → this automatically down-weights the along-line axis in the technique's covariance; techniques recognize the special case | curvature flag in feature |
| `RING_ANNULUS` | ISOTROPIC_GAUSSIAN, σ ~ mean ring-edge uncertainty | aggregated |

### Handling flat / low-curvature rings

When a ring feature's 2-D projection in the image is nearly a straight line, *offset along that line is unobservable*. The `RING_EDGE` feature sets a `straight_line` flag and its `position_cov_px` becomes **rank-deficient** (large eigenvalue along the line, small eigenvalue across).

Geometric note: every ring of a given planet shares the same ring-plane normal, so when the geometry projects them edge-on they all project as **parallel** lines (or near-parallel hyperbolas with negligible curvature). They share one normal, so adding more flat ring edges does not add a new direction of constraint — it just tightens the same 1-D radial constraint. The plan therefore does not pretend "two non-parallel flat ring edges resolve a 2-D offset"; that case doesn't physically arise.

The only way to resolve the unobservable along-ring axis in a flat-ring scene is a feature whose covariance is non-degenerate along that direction — a star, a body limb, or a body blob. The orchestrator's ensemble (Part 4) handles this naturally because covariance is composed properly: a rank-1 ring constraint plus a rank-2 star or body observation produces a fully-resolved 2-D answer; a rank-1 ring constraint alone produces a result with a flagged 1-D-only solution and very large reported uncertainty along the along-ring axis.

The curvature threshold is computed as max deviation of the polyline from a best-fit straight line in px; below (say) 0.5 px the flag fires. Config-tunable.

### Source-image filters

Only applied when the dataset + mission config says so. Stored in `config_4N0_inst_*.yaml`:

```yaml
inst:
  source_image_filter:
    kind: 'BANDPASS_DOG'  # or 'NONE'
    lo_sigma_px: 100       # heavy-blur scale; subtracted to remove low-frequency content
    hi_sigma_px: 0.7       # light-blur scale; preserves anything sharper
    enabled: true
```

Rule: `lo_sigma_px >= 50 × hi_sigma_px` so the bandpass is a true high-pass-with-noise-cap, not a narrow band.

Per-instrument concrete values (placeholders for Phase 5 retuning):

| Instrument | kind | `lo_sigma_px` | `hi_sigma_px` | rationale |
|---|---|---|---|---|
| COISS NAC/WAC | `NONE` | — | — | clean image; PDS-pipeline-flat-fielded |
| GOSSI | `BANDPASS_DOG` | 100 | 0.7 | scattered-light gradient on 800-px image is hundreds of px wide |
| VGISS NA/WA | `BANDPASS_DOG` | 80 | 0.7 | shorter-scale background structure on 800-px frame |
| NHLORRI | `NONE` | — | — | clean wide-aperture image |

### Filter parameters: floors, ceilings, formulas

Per-feature filter σ scales with per-image physical uncertainty (km uncertainty divided by km/px at the feature). The filter config supplies the **clamps and the kernel-truncation rule**, not raw σ values — those are derived per image. Clamps are uniform across instruments because the σ formula is already physically scaled. Adding a per-mission override slot is a normal config schema change if a future tuning requires it; no slot is reserved up front.

Shared derivatives (`config_540_orchestrator.yaml`):

```yaml
context:
  image_gradient_sigma_px: 1.2     # Sobel-of-Gaussian σ for the shared image gradient.
                                   # Used by limb / terminator / ring-edge DT. PSF-sized;
                                   # below 1 px is noise-dominated, above 1.5 px sharp
                                   # limbs blur out.
  edge_threshold_k_sigma: 4.0      # threshold for binarizing the gradient before DT
                                   # (units: noise σ from MAD).
```

Per-feature filter clamps (`config_530_filters.yaml`):

```yaml
common:
  kernel_truncation_sigma: 4.0      # kernel half-width = ceil(4σ); odd-sized kernels.
                                    # 4× (vs the more standard 3×) so anisotropic kernels
                                    # with σ_tangent = 0.5 still have a well-formed
                                    # footprint along the small axis.
  null_filter_threshold_sigma: 0.4  # below this σ, filter is replaced by NONE.

limb_arc:
  sigma_normal_min_px: 0.5
  sigma_normal_max_px: 5.0          # above this, the body extractor emits BODY_BLOB
                                    # instead of LIMB_ARC.
  sigma_tangent_px: 0.5             # always anisotropic; tangent σ never grows.

terminator_arc:
  sigma_normal_min_px: 1.0          # albedo always softens the terminator.
  sigma_normal_max_px: 8.0
  sigma_tangent_px: 0.5
  albedo_inflation_factor: 3.0      # σ_normal *= (1 + factor × albedo_variation)

ring_edge:
  sigma_radial_min_px: 0.5
  sigma_radial_max_px: 3.0          # above this, extractor degenerates to RING_ANNULUS.
  sigma_along_edge_px: 0.5
  flat_curvature_threshold_px: 0.5  # max polyline deviation from straight-line fit;
                                    # triggers `straight_line` flag + rank-1 covariance.

ring_annulus:
  sigma_isotropic_min_px: 1.0
  sigma_isotropic_max_px: 4.0
```

Per-feature σ derivation formulas (executed by extractors per image):

```
LIMB_ARC.sigma_normal_px = clip(
    sqrt(ellipsoid_residual_km² + crater_scale_km² + (incidence_factor × limb_softness_km)²)
    / km_per_pixel_at_limb,
    limb_arc.sigma_normal_min_px,
    limb_arc.sigma_normal_max_px)

RING_EDGE.sigma_radial_px = clip(
    sqrt(rms_km² + dynamical_amplitude_km²) / km_per_pixel_radial,
    ring_edge.sigma_radial_min_px,
    ring_edge.sigma_radial_max_px)

TERMINATOR_ARC.sigma_normal_px = clip(
    LIMB_ARC.sigma_normal_px × (1 + albedo_inflation_factor × albedo_variation),
    terminator_arc.sigma_normal_min_px,
    terminator_arc.sigma_normal_max_px)
```

`limb_softness_km = star_psf_sigma_px × km_per_pixel_at_limb` — the projected PSF width at the limb, computed per image; no new per-instrument constant needed.

---

## Part 3 — Technique system

### NavTechnique contract

```python
# src/nav/nav_technique/nav_technique.py (modified) — names per Part 0
class NavTechnique(ABC):
    name: str
    accepts_feature_types: frozenset[NavFeatureType]

    @abstractmethod
    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport: ...
    @abstractmethod
    def navigate(self, features: list[NavFeature], image: NDArrayFloat,
                 context: NavContext) -> NavTechniqueResult: ...

# NavContext carries everything image-wide that techniques need but that
# isn't per-feature: the observation, the (possibly pre-filtered) image,
# globally-derived image statistics, shared image-side derivatives, and
# provenance. Per Cardinal Principle #2, every member here is computed without
# knowing where features are in the image.
@dataclass
class NavContext:
    obs: ObsSnapshotInst
    image_ext: NDArrayFloat              # the extended-FOV image (post source-image filter)
    sensor_mask_ext: NDArrayBool         # obs.extfov_data_sensor_mask() — sensor vs padding;
                                         # image-vs-zero-pad layout, no navigation needed.

    # Global image statistics (computed across the whole sensor area, not
    # over any predicted-feature region).
    image_noise_sigma: float             # MAD over the entire image inside sensor_mask_ext.
                                         # Robust to bright body / star content because MAD
                                         # is dominated by background pixels (which are most
                                         # of the image) regardless of where features are.
    saturation_mask_ext: NDArrayBool     # True where pixel >= full_well_dn (per-instrument
                                         # static value); used to mask saturated pixels out
                                         # of star detection and gradient computation.
    cosmic_ray_mask_ext: NDArrayBool     # True where a single-pixel spike was detected by
                                         # global despeckle (median-filter difference).

    # Shared image-side derivatives (one-time per image; reused across
    # techniques — image gradient is needed by limb fit, terminator fit, ring
    # fit, and ring-annulus correlation).
    image_gradient_ext: NDArrayFloat | None    # Sobel-of-Gaussian magnitude, σ from config
    image_edge_dt_ext: NDArrayFloat | None     # signed distance transform of thresholded
                                               # gradient image; built once, reused by every
                                               # DT-based technique.

    # Prior offset from pass-1 (set by orchestrator between passes; None on
    # the first pass).
    prior_offset_px: tuple[float, float] | None
    prior_covariance_px2: np.ndarray | None

    # Filter spec applied to the source image, for diagnostic provenance.
    pre_filter_applied: NavFilterSpec | None
    spice_provenance: dict[str, str]     # kernel ID → version

@dataclass(frozen=True, eq=False)
class NavTechniqueResult:
    technique_name: str
    feature_ids: list[str]              # which features were actually used
    offset_px: tuple[float, float]      # (dv, du)
    covariance_px2: np.ndarray          # 2x2 (or 3x3 with rotation enabled)
    confidence: float                   # 0..1, self-assessed, calibrated
    spurious: bool                      # hard-reject flag (Part 0 "at_edge/spurious"
                                        # subsection — per-technique semantics defined)
    at_edge: bool                       # peak near search-window boundary
    diagnostics: NavTechniqueDiagnostics   # structured per Part 1 spec; see below

# Per-technique diagnostic dataclasses — structured, not dict-of-Any.
NavTechniqueDiagnostics = (
    BodyDiscDiagnostics | BodyLimbDiagnostics | BodyTerminatorDiagnostics |
    BodyBlobDiagnostics | RingEdgeDiagnostics | RingAnnulusDiagnostics |
    StarFieldDiagnostics | StarUniqueMatchDiagnostics | StarRefineDiagnostics |
    CartographicDiagnostics
)
# Each carries technique-specific debug fields (e.g. NCC peak ratios for
# BodyDiscDiagnostics; LM iteration count + Tukey inlier count for
# BodyLimbDiagnostics; n_inliers + median_residual for StarFieldDiagnostics).
# Concrete schemas live next to each technique implementation; the curator
# (Part 4) knows which fields to copy to JSON per technique.
```

### Techniques

Techniques split into two groups by their dependence on a prior offset.

**Prior-free techniques** (run first; produce an offset from raw SPICE prediction). Every technique consumes a **list** of features of its accepted types and produces **one** offset. When multiple bodies (or multiple ring edges) are visible, they go into the *same* technique invocation and constrain the offset jointly — they're additional constraint, not separate votes. This eliminates the "which moon's correlation peak wins when Rhea and Dione look alike" failure mode: the combined-template peak unambiguously identifies the alignment that puts both moons in their right places simultaneously, even if either moon alone would be ambiguous.

| Technique | Accepts | Notes |
|---|---|---|
| `BodyDiscCorrelateNav` | list of `BODY_DISC` | Builds a combined template by **Z-buffer paint** (per Part 0 §2): sort bodies by `subject_range_km` ascending, iterate and paint each body's template into the combined image; closer body's nonzero pixels overwrite farther body's. Combined mask = OR of per-body masks. Runs `navigate_with_pyramid_kpeaks` once on this combined template. Uses each feature's preferred filter; `use_gradient='auto'` self-selects RAW vs GRADIENT. With N bodies the SNR of the combined correlation peak grows roughly as √N if backgrounds are independent, and ambiguous-look-alike pairs are disambiguated geometrically. |
| `BodyLimbNav` | list of `LIMB_ARC` | DT-based limb fit: union of all visible limb polylines into one DT. Image-side limb extraction is shared (one gradient computation, all limbs detected at once). One 2-D shift minimizes summed squared distances across all bodies' limbs simultaneously. |
| `BodyTerminatorNav` | list of `TERMINATOR_ARC` | Same as `BodyLimbNav` but on terminator polylines, with **per-body uniform weight** (Part 0 §8): `1 / sigma_normal_per_vertex_px²` shared across all vertices of a given body, derived from `albedo_variation` (a per-body scalar in `config_220_body_shape.yaml`). |
| `BodyBlobNav` | list of `BODY_BLOB` | Detects the brightness-weighted centroid of each body's bright region and fits the offset that maps predicted centroids to detected centroids in least-squares (one offset for all bodies). With ≥2 blobs the geometry is over-determined, which dramatically improves robustness compared to a single-blob fit. |
| `RingEdgeNav` | list of `RING_EDGE` | DT match of all available ring edge polylines together. Already list-based in the original design; keeps rank-1 covariance for flat-ring-only scenes. |
| `RingAnnulusNav` | list of `RING_ANNULUS` (one per planet per scene; Part 0 §13) | Template correlation of the multi-ring region as a whole. With multiple planets visible, runs one correlation per planet annulus and combines their results in the same precision-weighted way as multi-body limb fits — each annulus contributes its own translational constraint, and `is_feasible` is True iff at least one `RING_ANNULUS` is present. |
| `StarFieldFromCatalogNav` | list of `STAR` features that pass the reliability gate (≥3, typically ≥4) | True from-scratch pattern match. **Feasibility N is by reliability-passing STAR features, not detection count** — detection happens inside the technique using a global sweep, not from the input feature list (the input list specifies which catalog stars are predicted-detectable; the actual detection finds image-plane sources independently). Detects bright point sources in the image, computes geometry-invariant hashes (relative-distance ratios and angles for star triplets / quads, astrometry.net-style), matches against the same hashes computed from catalog stars in a search radius around the predicted FOV. Position-independent: works for *any* SPICE pointing error because the matching is on relative geometry, not absolute position. Confidence drops when fewer than ~4 unambiguous bright stars are present (sparse fields, low-SNR fields), or when an unusually rich field produces too many candidate matches (the technique must verify a consistent transformation across multiple triplets). Reports infeasibility when no consistent transformation is found. |
| `StarUniqueMatchNav` | list of `STAR` (1 or 2 reliability-passing) | Direct catalog-uniqueness match. Used when `stars_list_for_obs()` predicts exactly 1 or 2 catalog stars in the extended FOV bright enough to detect, with the brightest at least `unique_match_brightness_margin_mag` brighter than the next-brightest predictable source. **Config**: `unique_match_brightness_margin_mag` (default 1.5) lives under `star_unique_match` in `config_510_techniques.yaml` alongside the technique's confidence formula. The catalog itself supplies uniqueness; no triplet hash needed. 1-star: assign the brightest detection to the unique star (sanity-check that detection lies within `extfov_margin` of predicted catalog position); offset = detection − prediction. Rotation unconstrained on `fit_camera_rotation=True` instruments — the 3×3 covariance is rank-2 (translation observable, rotation unobservable) per Part 5b's rank-deficient-handling rule. 2-star: try both detection-to-catalog assignments, pick smaller-residual fit. Mutually exclusive with `StarFieldFromCatalogNav` at feasibility time. **This makes "one bright star, no body" a primary navigation mode**, not a fallback. |
| `TitanNav` | (placeholder — stub for now) | Titan and other bodies with thick opaque atmospheres need a fundamentally different algorithm than ellipsoid-limb fitting: the visible "limb" is the haze top, varies with wavelength, and the surface inside is invisible. **In this plan, `TitanNav` ships as a registered-but-stub technique** that emits an infeasibility report for any input. Bodies flagged `atmospheric: true` in `config_220_body_shape.yaml` (currently just Titan; potentially Venus / Triton later) skip the standard limb / disc / blob extraction and instead emit no body-derived features, so the orchestrator falls through to other features (stars / rings) on those scenes. The real Titan algorithm is to be designed and added later — not in scope for this plan. |
| `CartographicNav` | list of `CARTOGRAPHIC_MODEL` features | Bootstrap-style navigation against a previously-built mosaic of a body. The body extractor checks for an available cartographic model (under `bootstrap_results_root`, configured per `config_150_bootstrap.yaml`) and emits a `CARTOGRAPHIC_MODEL` feature instead of (or in addition to) a `BODY_DISC` for that body when one is available. The technique reprojects the mosaic onto the predicted body silhouette (using `nav.reproj.cartographic_model.create_cartographic_model`) and correlates that detail-rich rendering against the image — far more discriminating than the smooth Lambert disc. |

**Prior-required techniques** (run after, given the best pass-1 offset as starting point):

| Technique | Accepts | Notes |
|---|---|---|
| `StarRefineNav` | `STAR` (≥1) | The existing per-star PSF-fit refinement, generalized into a standalone technique. Requires a prior offset accurate to ~1–2 px (one PSF radius). Refines each predicted star with `psf.find_position()`, accepts those that pass quality gates, returns the median-of-survivors offset and a covariance from the per-star residual scatter. **Cannot run from a zero prior on missions with poor SPICE pointing reconstruction** — that's why it's prior-required. When a prior-free technique provides the offset, this technique can sharpen it from ~1 px to ~0.05 px on a clean star field. |
(`LimbRefineNav` was considered and dropped — see Part 0 §6. `BodyLimbNav` does its own LM subpixel refinement after the coarse DT match; no separate prior-required limb technique exists.)

The orchestrator (Part 4) runs the prior-free group first, picks the best result by the standard rule, then sets `NavContext.prior_offset` and runs the prior-required group. Both groups' results go into the ensemble, so `StarRefineNav` can sharpen the body offset OR cross-validate it (disagreement triggers the conflicted branch).

**Two-pass priority order — what becomes the prior?** When multiple prior-free techniques succeed, the prior fed to pass 2 is the **ensemble's pass-1 result** (after non-spurious / not-at-edge / quality tiers + precision-weighted combination). It is *not* the highest-quality single technique. Reason: the ensemble has already cross-validated and combined; it has lower variance and fewer mode-failures than any single technique. Pass 2 then refines that prior; the pass-2 result is added to the pass-1 results, the ensemble runs again over the union, and that's the final answer. If no prior-free technique succeeds, the prior-required group is skipped and `NavResult.status='failed'`.

**Pass-1 results enter the final ensemble twice — intentional (Part 0 §1).** Once via the prior given to pass 2 (a starting point for refinement), once as direct inputs to the final ensemble. The information-form Kalman combine is correct under this: the prior is a compact summary used for refinement, not a substitute for the per-technique pass-1 results, which carry their own evidence into the final combine. Implementer must not "fix" this by deduplicating pass-1 inputs.

| `ManualNav` | all | Existing UI, refactored to present per-technique results side-by-side and let user pick. Outside the autonomous flow. |

### Polyline density convention

Limb / ring / terminator polylines are sub-sampled at **~1 vertex per pixel of arc length** so vertex density doesn't bottleneck the bilinear-DT sub-pixel precision. Sparser polylines (e.g. 5 px between vertices) cause the LM optimizer to interpolate between widely-spaced normals and lose precision; denser is small additional memory and tightens accuracy. The extractor performs this densification via spline interpolation of the analytic (mass-mode + harmonic) edge model. Same convention for both LIMB_ARC and RING_EDGE.

### Determinism in RANSAC

Pattern matching's RANSAC step iterates over triplet-correspondence candidates in a **deterministic order** — sorted first by hash distance, then by a canonical tie-breaker (sorted detected-source index lexicographically). With M=30 brightest detected sources capped, the candidate list is bounded (~30³ = 27K triplets, far fewer surviving the hash-distance filter), so exhaustive evaluation is fast and avoids the non-determinism of random sampling. Same image → same matching attempts → same winner. Per Cardinal Principle #3.

### Multi-feature joint fitting — the architectural principle

Combined-feature techniques (`BodyLimbNav`, `BodyTerminatorNav`, `RingEdgeNav`) parameterize their cost function over a **single rigid 2-D translation `(dv, du)`** (or 3-DoF `(dv, du, dθ)` when rotation correction is enabled, see Part 5b) applied to the **concatenation of all input polylines**. Per-feature relative geometry is preserved by construction — there are no per-feature offset DoF, and SPICE orbital relative positions (accurate to ~0.05 px on the focal plane for major moons) are inherited as-is from the input.

This is what makes multi-body navigation correct without explicit body-to-blob disambiguation: the optimizer cannot represent "swap moon assignments" because that would require independent per-body offsets, which the 2-DoF (or 3-DoF) parameterization doesn't admit. The cost function is summed over all vertices from all bodies; every vertex sees the same `(dv, du)` shift; the unique minimum places every body at its correct image position simultaneously. If predicted relative positions disagree with actual relative positions (rare, would indicate orbit-prediction error rather than attitude error), the combined fit reports high residual and low confidence — honest failure rather than confident wrong answer.

The same principle applies to multi-edge ring fits and multi-star pattern matching: a single transform constrains all features jointly. **Single-feature solo navigation is not architecturally admitted for any feature type that occurs in plurals** — it would reintroduce the disambiguation problem with no benefit.

Implementation detail for `BodyLimbNav` specifically: the polylines from every body's LIMB_ARC feature are *concatenated* into one `(N_total, 2)` array of points + matching `(N_total, 2)` normals + per-vertex weights `w_v = 1 / σ_normal²`. The optimizer doesn't need to know which body a vertex came from; it's just weighted points. The DT-based coarse search uses the *combined* edge-mask NCC against the image gradient; no per-body coarse search.

### Algorithm specifics (no more hand-waving)

**Occlusion handled at extraction time, uniformly.** Limb polylines, terminator polylines, and ring polylines are all cropped at extraction by the producer model, not by the technique. The body extractor walks polyline vertices and drops each vertex where another body's silhouette is closer to the observer; the ring extractor drops vertices behind the planet's shadow or behind a closer body. After this cropping, polyline features carry only visible vertices and no per-vertex range field is needed downstream. `BodyDiscCorrelateNav` still depth-sorts its template input (a fully-in-FOV closer body's disc paints over a partially-occluded farther body's disc when building the combined template), but that depth-sort uses the per-feature scalar `subject_range` carried on `BODY_DISC` payloads — not per-vertex range, which the polyline payloads no longer carry.

**DT-based fitting** (used by `BodyLimbNav`, `BodyTerminatorNav`, `RingEdgeNav`):

1. *Image-side limb / edge detection* is global. Compute Sobel-of-Gaussian magnitude across the whole image (reused from `NavContext.image_gradient_ext`); threshold + non-maximum suppression to extract a thin-edge map; build the signed distance transform once into `NavContext.image_edge_dt_ext`. Three techniques share this output.
2. *Coarse search* by 2-D NCC of model-edge-mask against image-edge-mask, restricted to `extfov_margin_vu` window. Produces an integer-pixel offset within the basin of attraction.
3. *Subpixel refinement* by Levenberg-Marquardt minimization of `Σ_i DT(image_edge, model_polyline_i + (dv, du))²` with bilinear DT interpolation. Robust loss (Tukey biweight) downweights outlier polyline points. Converges from the coarse-search basin in < 10 iterations.
4. *Edge polarity*. The model carries which side of the polyline is "bright": gradient direction at each model polyline point. The image's gradient direction at the matched image-edge pixel must agree (dot product > 0); pixels with the wrong polarity contribute `+∞` to the cost (effectively rejected). This is what stops the limb fit from latching onto a bright-to-dark image edge that goes the wrong way (e.g., shadow of a foreground body).
5. *Confidence* from final RMS DT residual, number of polyline points used (after polarity / Tukey rejection), and inferred 2-D covariance from the M-estimator's information matrix.

**`StarFieldFromCatalogNav` pattern matching:**

1. *Source detection*. DAOPHOT-style: convolve image with the obs-supplied PSF kernel (smear-aware when applicable), threshold at `k * NavContext.image_noise_sigma`, find local maxima, fit a Gaussian to each to get sub-pixel centroids and intensity.
   - **Saturated stars** are not rejected outright. A bright star whose peak is clipped at full-well still has a usable centroid via the wings; for those, the detector switches from peak-Gaussian-fit to brightness-weighted-moment centroid over an annular region that excludes the saturated core. Saturated star centroids carry larger reliability and feed the RANSAC matching with appropriate weight.
   - **CCD bloom along columns** from very-bright stars (saturated charge bleeding vertically) is detected as a saturated-pixel run extending well outside the PSF box. The bloom column gets masked from the gradient computation (so it doesn't masquerade as a limb edge) but the parent star is still kept (with bloom-aware centroid).
   - Reject centroids that fail Gaussian-shape tests (cosmic-ray rejection at the detection level), centroids inside `cosmic_ray_mask_ext`, and centroids in pixels marked missing-data by the image classifier.
2. *Catalog reduction*. Reuse `NavModelStars.stars_list_for_obs()` from `src/nav/nav_model/nav_model_stars.py` — its `obs.ra_dec_limits_ext()`-bounded query is already the right shape: it covers the full extended FOV (which encodes the per-instrument pointing-error budget via `extfov_margin_vu`). Don't reimplement aberration / proper motion / multi-catalog precedence / magnitude binning / star-vs-star conflicts — they're all there and validated. Don't dilate body silhouettes for source rejection; for high-pointing-error instruments the predicted silhouettes are wrong by hundreds of pixels and excluding image stars near them would lose real matches. Per-instrument flag `stars.use_body_conflict_in_catalog_reduction: bool` (default true for Cassini/NHLORRI, false for VGISS/GOSSI) governs whether the existing catalog-side body-conflict marking is applied at all.
3. *Triplet hashing*. The hash is **similarity-invariant** (translation + rotation + uniform scale): it uses two distance ratios and one angle, all of which survive arbitrary similarity transforms. For each unordered triplet of bright detected sources `{A, B, C}` with `A` the brightest of the three, compute:

```
h = ( d_AB / d_AC, d_BC / d_AC, ∠BAC )       # 3-vector
```

The "A is brightest" rule canonicalizes the ordering so each unordered triplet produces exactly one hash (not 6 from permutations). Compute the same hash for catalog triplets, with the catalog star's `predicted_mag_in_band` used as the brightness ranking.

**Distance metric for KD-tree**: the hash space mixes ratios and an angle. Use a **weighted Euclidean metric** with the angle in radians: `d² = w_ratio · ((Δr₁)² + (Δr₂)²) + w_angle · (Δθ)²` with `w_ratio = 1.0`, `w_angle = 1.0` (radians and dimensionless ratios are roughly comparable in magnitude over the typical triplet shapes; a 0.1 rad ≈ 5.7° error is comparable to a 0.1 ratio error). Tunable in `config_510_techniques.yaml` under `star_field`.

**Match radius**: the KD-tree query returns the K nearest catalog hashes within radius `hash_match_tolerance` (default 0.05 in the weighted metric — empirically the right scale for centroid σ ≈ 0.1 px against typical triplet scales). Tunable.

KD-tree the catalog hashes; for each detection-triplet hash, query nearest catalog neighbors within radius. Each surviving (detection-triplet, catalog-triplet) candidate proceeds to RANSAC.
4. *RANSAC*. Each candidate triplet correspondence proposes a 2-D similarity transform (translation + small rotation; nominally rotation = 0 because the SPICE camera frame is good — even when pointing direction is bad, the camera-twist is small). Score by counting inlier detections within a couple of pixels of their transformed catalog position. Best-scoring transform with ≥ `pattern_match_min_inliers` (config, default 6) wins.
5. *Verify and refine*. With the inlier set, refit the transform by least squares; report (offset, covariance, n_inliers).

If RANSAC fails (no transform with enough inliers), the technique reports infeasibility and the orchestrator continues without star-from-scratch contribution. Failure costs nothing downstream — `is_feasible` is consulted before invoking the technique, so this is the orchestrator's "tried, didn't work" path.

**Detection-step internals (concrete parameters):**

- **Threshold**: `threshold_k_sigma` per-camera in each `config_4N0_inst_*.yaml`. Defaults: COISS NAC/WAC and NHLORRI = 4.0; GOSSI and VGISS NA/WA = 5.0 (higher noise / stray light → more conservative).
- **Cap**: `max_sources = 30` uniform across all instruments. Triplet hashing is M³ in candidate count; 30 keeps it bounded (~27K triplets) and is plenty for ~6-inlier matching.
- **Smeared kernel**: built once per image via `obs.star_psf().eval_rect((H, H), offset=(0.5, 0.5), scale=1.0, movement=(my, mx), movement_granularity=max(0.1, smear_length / 50))` with `H` = the largest entry in `star_psf_sizes`; normalized for matched-filter convolution. Smear vector from SPICE start-ET vs end-ET brackets. If `smear_length > stars.max_smear` (default 100 px) the extractor emits no STAR features.
- **Per-detection refinement, two paths by saturation**:
  - *Unsaturated* (`peak_DN < saturation_threshold_dn`): 2-D Gaussian fit on a `2.5 × psf_fwhm` box; cuts `fitted_fwhm ∈ [0.7, 1.5] × instrument_psf_fwhm`, `roundness < 0.25` (skipped for smeared images), `sharpness > 0.4`. Hot-pixel sanity: reject if `peak_DN > 5 × max(8 surrounding pixels)`.
  - *Saturated* (`peak_DN ≥ saturation_threshold_dn`): annular brightness-weighted moment centroid over `r ∈ [0.5, 1.5] × psf_fwhm`, excluding `cosmic_ray_mask_ext`, `saturation_mask_ext`, bloom columns, missing-data pixels. Reject if fewer than 8 valid annulus pixels remain. σ_centroid floor 0.2 px; `saturated=True` tag downweights triplet weight.
- **Bloom detection**: vertical (per `bloom_orientation`) saturated runs > 1.5 × psf_fwhm at a saturated peak mark a bloom column; column + 1 px each side merged into `saturation_mask_ext`. Parent star kept (centroided via annulus).
- **Implementation choice**: DAOPHOT-style sharpness/roundness cuts are hand-rolled in numpy / scipy (already in deps). Modern alternatives (`photutils.detection.StarFinder`, `sep`, `scikit-image.feature`) cost extra dependencies; `photutils` would be the cheapest fallback if the hand-roll proves troublesome in Phase 3.

**Single-bright-star scenes** are a **primary navigation mode** via `StarUniqueMatchNav`, not a failure. When `stars_list_for_obs()` predicts 1–2 catalog stars in extended FOV and the brightest is ≥ `unique_match_brightness_margin_mag` (default 1.5 mag) brighter than the next-brightest predictable source, the catalog itself supplies uniqueness — no triplet hash needed. The brightest detection assigns to the unique catalog star; offset = detection − prediction; sanity check is that the detection lies within `extfov_margin` of predicted position (otherwise the brightest detection is an asteroid / surviving cosmic ray / rare transient, and the technique reports infeasibility). 2-star case fits a translation (or translation + rotation) via two-assignment trial. `StarFieldFromCatalogNav` and `StarUniqueMatchNav` are mutually exclusive at feasibility time: ≥3 unambiguous catalog stars → triplet matcher runs, unique matcher returns `'enough_stars_for_triplet_match'`; 1–2 → unique matcher runs, triplet returns `'fewer_than_3_detected_sources'`. The single-star STAR feature is also still consumed by `StarRefineNav` in pass 2 when a body / ring prior exists — multiple paths benefit from the lone bright star.

### Correlation type — RAW vs GRADIENT vs AUTO

The decision lives **inside** each correlation-based technique (`BodyDiscCorrelateNav`, `RingAnnulusNav`) because the right choice depends on the feature at hand. The already-implemented `use_gradient='auto'` behavior is the default for these techniques: run both, pick by non-spurious > not-at-edge > higher-quality.

`RingAnnulusNav` benefits from the same auto-mode logic as bodies: a multi-ring composite template often has both broad brightness gradients (raw mode wins on low-resolution Saturn rings where the C-ring is uniformly dim) and sharp edges (gradient mode wins on high-resolution scenes where individual ringlet edges dominate). Auto-mode picks correctly per-image.

For `BodyLimbNav`, `BodyTerminatorNav`, `RingEdgeNav`, `StarFieldFromCatalogNav`, `StarUniqueMatchNav`, `BodyBlobNav`: no NCC, so no gradient-vs-raw decision.

### Feasibility reports

```python
@dataclass
class NavFeasibilityReport:
    feasible: bool
    reason: str                  # human-readable when not feasible
    consumed_feature_count: int
```

Examples:
- `StarFieldFromCatalogNav.is_feasible`: True iff ≥3 features pass `reliability ≥ star_reliability_gate`.
- `BodyLimbNav.is_feasible`: True iff at least one `LIMB_ARC` has `visible_arc_px ≥ limb_min_arc_px` (config).
- `RingEdgeNav.is_feasible`: True iff at least one ring edge feature is present. The technique always runs and always returns its honest covariance — full 2-D when at least one curved edge is present, rank-1 (along-ring axis unobservable) when all available edges are straight-line. The ensemble (Part 4) is responsible for fusing rank-1 ring constraints with stars or body features that supply the missing axis; if no such other feature is present, the final 2-D confidence is correctly low because the covariance reflects an unbounded along-ring uncertainty.

  **Per-planet detectability defaults** in the ring catalog YAMLs determine which edges the extractor even attempts:
  - **Saturn**: rings are bright and abundant; nearly every named edge is detectable when geometrically in the FOV. Default reliability high.
  - **Uranus**: rings are dark and narrow; only a few of the strongest edges (epsilon, delta) are reliably detectable, and only at high resolution / favorable lighting. Most catalog entries should set `default_reliability` low so the extractor doesn't emit features that won't survive matching.
  - **Neptune**: rings are partial *arcs* (incomplete azimuthally — Le Verrier, Galle, Adams). The extractor honors the arc geometry: only the in-arc longitude range produces a `RING_EDGE` feature; outside the arc the edge isn't present in the image even when the ring radius is in the FOV.
  - **Jupiter**: rings are extremely dim and only detected in dedicated long-exposure mosaics. Default reliability very low; expect zero ring features in most Jupiter-region images.

  These defaults live in the existing `config_3N0_*_rings.yaml` catalogs; per-feature overrides for unusual edge behavior remain available.

Infeasible techniques are skipped with a diagnostic; not a failure.

---

## Part 4 — Orchestrator & ensemble

### Decision flow

```python
# src/nav/nav_orchestrator/orchestrator.py (new)
class NavOrchestrator(NavBase):
    """Orchestrates feature extraction, technique execution, and ensemble.

    Parameters:
        config: Loaded Config (defaults to DEFAULT_CONFIG).
        only_models: Glob-pattern string or list (default '*'). See Part 4
            'Filtering for debugging'.
        only_techniques: Glob-pattern string or list (default '*'). See Part 4.
    """
    def __init__(self, config: Config | None = None, *,
                 only_models: str | list[str] = '*',
                 only_techniques: str | list[str] = '*') -> None:
        ...

    def navigate(self, obs: ObsSnapshotInst) -> NavResult:
        """Run the full pipeline on one observation.

        The image is read from obs.data internally; not passed separately.
        Two-pass: prior-free techniques first, then prior-required techniques
        with the pass-1 ensemble result as their prior.
        """
        # Image-quality preflight — runs before any extraction.
        context = self._make_context(obs)
        if context.image_classifier.image_class != 'clean' \
                and self._is_hard_failure(context.image_classifier):
            return self._build_image_classifier_failure(context)

        features = self._extract_features(obs, context)
        features = self._gate_features_by_reliability(features)

        if not features:
            return self._build_no_features_failure(context)

        # Pass 1: prior-free techniques.
        pass1_results = self._run_techniques_for_pass(
            features, context, requires_prior=False)
        if not pass1_results:
            return self._build_no_feasible_techniques_failure(features, context)

        # Combine pass 1 to get prior for pass 2.
        pass1_ensemble = self._ensemble(pass1_results, features, context)
        if pass1_ensemble.status == 'failed':
            return pass1_ensemble

        # Pass 2: prior-required techniques (StarRefineNav, etc.).
        context_with_prior = context.with_prior(
            offset_px=pass1_ensemble.offset_px,
            covariance_px2=pass1_ensemble.covariance_px2,
        )
        pass2_results = self._run_techniques_for_pass(
            features, context_with_prior, requires_prior=True)

        # Final ensemble over the union of both passes.
        return self._ensemble(pass1_results + pass2_results, features, context)

    def _run_techniques_for_pass(
        self,
        features: list[NavFeature],
        context: NavContext,
        *,
        requires_prior: bool,
    ) -> list[NavTechniqueResult]:
        results: list[NavTechniqueResult] = []
        for technique in self._techniques(requires_prior=requires_prior):
            if not technique.accepts_feature_types & {f.type for f in features}:
                continue
            feasibility = technique.is_feasible(features)
            if not feasibility.feasible:
                continue
            try:
                subset = [f for f in features
                          if f.type in technique.accepts_feature_types]
                result = technique.navigate(subset, context)
                results.append(result)
            except Exception as exc:
                self._logger.exception(
                    'technique %s failed: %s', technique.name, exc)
        return results
```

**`_extract_features` and `_gate_features_by_reliability`** are separate steps. The first invokes every registered `FeatureExtractor.extract()`; the second applies the per-feature-type reliability gate from `config_520_features.yaml`. Splitting them lets `nav_feature_inspect` (Part 8) report both pre- and post-gate features, and lets the per-status_reason `no_features_extracted` vs `all_features_gated` distinction be drawn cleanly.

**Technique ordering**: `NavTechnique` declares a `requires_prior: ClassVar[bool] = False` attribute (overridden to `True` by `StarRefineNav` and any future `LimbRefineNav`). `_techniques(requires_prior=...)` filters the registry by this flag, then applies the `only_techniques` glob filter. Within each pass, ordering among techniques is arbitrary (results are ensemble-combined; per-technique order doesn't affect the final answer because the ensemble is order-invariant).

**Registry mechanism**: `NavTechnique` and `NavFeatureExtractor` use Python's `__init_subclass__` hook — every concrete subclass auto-registers with the central registry at import time. Imports happen via the existing `nav.nav_technique.__init__.py` and `nav.feature.__init__.py` which explicitly import every module. This is preferred over entry-points (no installation step) and over a decorator (no boilerplate).

```python
# src/nav/nav_technique/nav_technique.py (sketch)
class NavTechnique(NavBase):
    _registry: ClassVar[list[type['NavTechnique']]] = []
    _abstract: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Always call super() — required so cooperative __init_subclass__
        # chains work (mixins / abstract base classes). Any subclass that
        # overrides __init_subclass__ must also call super().__init_subclass__().
        # Test test_registry.py asserts every concrete subclass appears in the
        # registry; a subclass that breaks the chain will fail that test.
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get('_abstract', False):
            NavTechnique._registry.append(cls)
        # NOTE: do not call self._logger or any logger at class-creation time —
        # __init_subclass__ runs at import time, before logging is configured.
        # Registry side effects only.
```

`NavOrchestrator._techniques()` filters `NavTechnique._registry` by `only_techniques`, applies the same `_filter_models`-style glob match (reuses the existing utility on `NavTechnique` itself in `src/nav/nav_technique/nav_technique.py:71-86`, **extended to parse a leading `!` as exclusion** per Part 0 §9 — gitignore-style. Existing technique callers pass non-negation patterns and behave unchanged).

### Filtering for debugging — two glob-pattern tiers

The orchestrator accepts two independent filter parameters, both glob-pattern strings (or lists of strings, OR semantics) following the existing `_combine_models(['*'])` / `_filter_models([...])` conventions in `nav_master.py`:

1. **`only_models`** — selects which `NavModel` instances run at all. Examples:
   - `'*'` (default) — every model runs.
   - `'body:enceladus'` — only the Enceladus body NavModel; no stars, rings, or other bodies.
   - `'rings'` — only ring NavModels.
   - `['body:*', 'stars']` — every body NavModel plus the stars NavModel; no rings.
   - `'!body:*'` — every model except body NavModels (negation supported).

2. **`only_techniques`** — selects which `NavTechnique` instances run after features are extracted. Examples:
   - `'*'` (default) — every feasible technique runs.
   - `'BodyLimbNav'` — only limb-fitting; no disc correlation, no blob, no stars.
   - `['BodyLimbNav', 'BodyTerminatorNav']` — limb and terminator only.
   - `'!StarFieldFromCatalogNav'` — every technique except star-field pattern matching (negation per Part 0 §9).
   - `'StarFieldFromCatalogNav'` vs `'StarRefineNav'` — distinguishes the two STAR-consuming techniques. (This is the case where technique-level filtering is strictly more expressive than feature-type filtering would be — STAR features feed both techniques, but you can disable one independently.)

The two filters compose: model filter shrinks the feature pool (fewer features in play); technique filter shrinks the algorithm pool (fewer techniques run on whatever features remain). Both default to `'*'`; both are CLI flags (`--only-models`, `--only-techniques`) and config keys.

**No feature-type filter.** Technique-name filtering covers the same ground — every feature type is consumed by exactly one technique except STAR, which is consumed by two (and technique-name filtering distinguishes them). If a future technique consumes multiple feature types and you want to feed it only one, a feature-type filter can be added then; the orchestrator design leaves room.

### Confidence calibration (no statistics across images)

Every `NavTechniqueResult.confidence` is a calibrated 0..1 score derived **from that run's self-evidence only**. No cross-image learning. Each technique uses a sigmoid-of-linear-combination form so the constants are easy to tune from the image library:

```
confidence = sigmoid(α₀ + Σᵢ αᵢ * normalized_feature_iᵢ)
```

with `sigmoid(x) = 1 / (1 + exp(-x))`. Starting constants — to be retuned in Phase 5 against the library; what matters now is that they're *concrete* enough to implement and calibrate against. **The example confidence values listed alongside each formula below are illustrative targets, not arithmetic outputs of the listed coefficients** — coefficients are calibrated in Phase 5 to produce those targets, not the other way round. Implementations should not assert on the example values; they should evaluate the formula as written and accept whatever confidence emerges.

- **StarFieldFromCatalogNav**: `α₀ = -2`, `α(n_inliers, capped at 12) = 0.6`, `α(median_residual_px) = -2.5`. With 6 inliers and 0.4 px median residual, confidence ≈ 0.85.
- **StarUniqueMatchNav (1-star)**: `α₀ = -1.5`, `α(predicted_snr_capped_at_50) = 0.06`, `α(brightness_margin_mag − 1.5, capped at 3) = 0.5`. With SNR=20 and 2-mag margin, confidence ≈ 0.6. Hard-capped at 0.7 (no internal cross-check is possible from one star).
- **StarUniqueMatchNav (2-star)**: `α₀ = -1`, `α(min_snr_capped_at_30) = 0.07`, `α(2-star residual_px) = -2`. With both SNR ≥10 and 0.3 px residual, confidence ≈ 0.8.
- **StarRefineNav**: `α₀ = -3`, `α(n_stars_used, capped at 8) = 0.5`, `α(median_pos_err_px) = -3`. With 5 stars and 0.1 px residual, confidence ≈ 0.92.
- **BodyLimbNav**: `α₀ = -1`, `α(visible_limb_arc_fraction) = 3`, `α(DT_fit_rms_px) = -1.5`, `α(visible_arc_px / 100, capped at 5) = 0.4`. Long sharp limb fit to 0.3 px RMS gives ≈ 0.9; short noisy limb < 0.4.
- **BodyDiscCorrelateNav**: `α₀ = -2`, `α(NCC_peak) = 4`, `α(peak_to_runner_up_ratio - 1) = 1`, `α(consistency_px) = -0.4`, hard zero if `at_edge`. **`peak_to_runner_up_ratio` definition**: `peak_NCC / second_peak_NCC` where `second_peak_NCC` is the highest NCC value at any pixel **outside a 3 × kernel-half-width exclusion radius around the peak**. The exclusion radius prevents NCC sidelobes from being treated as runner-up peaks (they are spatially correlated with the main peak, not independent alternatives). **`consistency_px`** is the mean per-axis disagreement between the peak's coarse-pyramid-level location and its full-resolution location after sub-pixel refinement; large values signal a peak that drifts under refinement (a sign of an unstable correlation surface).
- **BodyBlobNav**: `α₀ = -1.5`, `α(body_snr_inside_predicted_bbox) = 0.5`, `α(body_extent_px / 30, capped at 1) = 0.5`, **clamped to ≤ 0.4** regardless (precision is intrinsically limited).
- **RingEdgeNav**: `α₀ = -1`, `α(total_edge_length_px / 200, capped at 3) = 1`, `α(per_edge_DT_rms summed, normalized) = -2`. Rank-1 result from straight-line-only edges: confidence reported per axis; the ensemble multiplies the perpendicular axis by 1.0 and the unobservable axis by 0.0, naturally producing low 2-D combined confidence.
- **CartographicNav**: `α₀ = -2`, `α(NCC_peak) = 5`, `α(visible_body_area_px / 5000, capped at 1) = 1`. Cartographic models contain enough surface detail to distinguish small offsets; well-fit confidence reaches 0.95.

**Confidence calibration YAML schema** (in `config_510_techniques.yaml`):

```yaml
techniques:
  body_limb_nav:
    confidence:
      alpha0: -1.0
      terms:
        - feature: visible_limb_arc_fraction      # name from technique's diagnostics
          alpha: 3.0
          cap_at: null                            # no cap (already in [0,1])
        - feature: dt_fit_rms_px
          alpha: -1.5
          cap_at: null
        - feature: visible_arc_px
          alpha: 0.4
          divisor: 100.0                          # x_normalized = x / divisor
          cap_at: 5.0                             # x_normalized clipped to [0, cap_at]
      hard_zero_if:                               # optional gates that force confidence=0
        - at_edge: true
      hard_cap: null                              # optional ceiling (e.g. 0.7 for unique 1-star)
  body_blob_nav:
    confidence:
      alpha0: -1.5
      terms:
        - feature: body_snr_inside_predicted_bbox
          alpha: 0.5
        - feature: body_extent_px
          alpha: 0.5
          divisor: 30.0
          cap_at: 1.0
      hard_cap: 0.4                               # BLOB precision is intrinsically limited
  star_unique_match_nav_one_star:
    confidence:
      alpha0: -1.5
      terms:
        - feature: predicted_snr
          alpha: 0.06
          cap_at: 50.0
        - feature: brightness_margin_mag
          alpha: 0.5
          offset: -1.5                            # x_normalized = x − offset
          cap_at: 3.0
      hard_cap: 0.7
  # ... one block per technique
tier_thresholds:                                  # used to derive confidence_rank
  # Both conditions must hold (AND) for a tier to apply. confidence_rank is
  # the highest tier whose AND-gate passes; if none pass, the rank is 'failed'.
  # max_sigma_px = null means no sigma constraint (only confidence matters).
  # The check uses max(sigma_dv, sigma_du) for the comparison.
  high:    {min_confidence: 0.8, max_sigma_px: 0.5}
  medium:  {min_confidence: 0.5, max_sigma_px: 2.0}
  low:     {min_confidence: 0.2, max_sigma_px: null}
```

A shared `evaluate_sigmoid_combination(spec, diagnostics)` helper in `src/nav/nav_technique/confidence.py` (alongside the techniques that use it, not in nav_orchestrator) reads the spec and the technique's diagnostics object, applies normalize (`offset` → `divisor` → `cap_at`) → linear-combine → sigmoid → optional hard-cap, returns the float. Tested in isolation; every technique's confidence path uses the helper.

**Config-load validation (Part 0 §16).** `Config` initialization walks every technique's confidence YAML and asserts each `feature:` name resolves to a real attribute on the technique's `NavTechniqueDiagnostics` dataclass (e.g. `BodyLimbDiagnostics`). Unknown field names raise `ValueError` at startup with the offending technique name, the bad field, and the dataclass's actual fields listed. A test (`tests/nav/config_files/test_confidence_specs.py`) loads each shipped YAML and asserts no errors. This means a refactor renaming a diagnostic field can never silently degrade a confidence formula.

Adding the autonomous-nav settings is the right moment to **renumber every config file from 2-digit to 3-digit prefixes**, expanding the slot namespace and giving room to group related configs without crowding. The merge order remains numeric (sorted by name), so behavior is unchanged.

Renumbering scheme — keeps groupings logical and leaves gaps for additions:

| New name | Old name | Group |
|---|---|---|
| `config_010_general.yaml` | `config_01_general.yaml` | Core |
| `config_020_offset.yaml` | `config_02_offset.yaml` | Core |
| `config_110_stars.yaml` | `config_03_stars.yaml` | Per-feature defaults |
| `config_120_bodies.yaml` | `config_04_bodies.yaml` | Per-feature defaults |
| `config_130_rings.yaml` | `config_05_rings.yaml` | Per-feature defaults |
| `config_140_titan.yaml` | `config_06_titan.yaml` | Per-feature defaults |
| `config_150_bootstrap.yaml` | `config_150_bootstrap.yaml` | Per-feature defaults |
| `config_210_satellites.yaml` | `config_10_satellites.yaml` | Per-planet body data |
| `config_220_body_shape.yaml` | (new) | Per-planet body data |
| `config_310_jupiter_rings.yaml` | `config_20_jupiter_rings.yaml` | Per-planet ring data |
| `config_320_saturn_rings.yaml` | `config_21_saturn_rings.yaml` | Per-planet ring data |
| `config_330_uranus_rings.yaml` | `config_22_uranus_rings.yaml` | Per-planet ring data |
| `config_340_neptune_rings.yaml` | `config_23_neptune_rings.yaml` | Per-planet ring data |
| `config_410_inst_coiss.yaml` | `config_30_inst_coiss.yaml` | Per-instrument |
| `config_420_inst_gossi.yaml` | `config_31_inst_gossi.yaml` | Per-instrument |
| `config_430_inst_nhlorri.yaml` | `config_32_inst_nhlorri.yaml` | Per-instrument |
| `config_440_inst_vgiss.yaml` | `config_33_inst_vgiss.yaml` | Per-instrument |
| `config_510_techniques.yaml` | (new) | Orchestration |
| `config_520_features.yaml` | (new) | Orchestration |
| `config_530_filters.yaml` | (new) | Orchestration |
| `config_540_orchestrator.yaml` | (new) | Orchestration |
| `config_810_sim.yaml` | `config_40_sim.yaml` | Sim / late stages |
| `config_910_backplanes.yaml` | `config_90_backplanes.yaml` | Late stages |
| `config_920_pds4.yaml` | `config_95_pds4.yaml` | Late stages |

The renumbering is a single mechanical mass-rename; the loader's "sorted" pass handles it transparently. Update `pyproject.toml` if any of the existing CLI scripts include literal filenames (they don't currently; the loader globs `config_*.yaml`).

Tier thresholds (`high` / `medium` / `low`) live alongside the formulas in `config_510_techniques.yaml`.

### Reconciliation

```python
# src/nav/nav_orchestrator/ensemble.py (new)
def ensemble(results: list[NavTechniqueResult],
             features: list[NavFeature]) -> NavResult:
    # Drop spurious.
    viable = [r for r in results if not r.spurious]
    if not viable:
        return NavResult.failed(reason='all_techniques_spurious', results=results)

    # Drop at-edge unless it's the only one left.
    interior = [r for r in viable if not r.at_edge]
    if interior:
        viable = interior

    # Group results that agree within combined 2-sigma covariance (Mahalanobis distance).
    # Single-link clustering: results A and B are in the same group iff their
    # pairwise Mahalanobis distance is below sigma_threshold. Transitive closure
    # builds final groups. Single-link is correct here because compatible
    # observations form a connected component in the "agreement" graph; a
    # complete-link variant would split chains of compatible-but-not-pairwise-tight
    # results, hiding genuine consensus.
    groups = _agreement_groups(viable, sigma_threshold=config.agreement_sigma)

    # Rank groups by summed confidence; pick the highest.
    ranked = sorted(groups, key=lambda g: sum(r.confidence for r in g), reverse=True)
    best_group = ranked[0]
    best_group_confidence = sum(r.confidence for r in best_group)

    # Precision-weighted combine within the best group (proper Kalman-style combination
    # handles rank-deficient covariances from flat rings correctly).
    offset, cov = _combine_precision_weighted(best_group)
    combined_confidence = _combine_confidence(best_group,
                                              disagreement_penalty=len(groups) > 1)

    if len(ranked) > 1:
        runner_up_confidence = sum(r.confidence for r in ranked[1])
        # Confidence gap between best and runner-up groups (zero or positive).
        confidence_gap = best_group_confidence - runner_up_confidence
        if confidence_gap < config.agreement_gap:
            # Two comparably-confident answers disagree — refuse rather than lie.
            return NavResult.conflicted(groups=groups, offset=offset, cov=cov,
                                        confidence=combined_confidence * 0.3)

    return NavResult.ok(offset_px=offset, covariance_px2=cov,
                        confidence=combined_confidence,
                        technique_results=results,
                        features=features)
```

**`_combine_precision_weighted` — exact algorithm.** Given a list of results `[(μ_i, Σ_i)]` where `μ_i ∈ R^D` and `Σ_i ∈ R^{D×D}` (D = 2 without rotation, 3 with), the standard Kalman-style information-form combination:

```
Σ_combined = pinvh( Σ_i pinvh(Σ_i, rcond=1e-9), rcond=1e-9 )
μ_combined = Σ_combined · Σ_i ( pinvh(Σ_i, rcond=1e-9) · μ_i )
```

`pinvh` (Hermitian pseudoinverse via `scipy.linalg.pinvh`) is mandatory — *not* `inv` — because rank-deficient inputs are routine in this pipeline (flat-ring scenes, 1-star unique match without rotation info). For a rank-1 covariance, `pinvh` projects onto the column space and treats the null direction as zero information; the combination then correctly inherits constraint from whichever other result covers that direction. The outer pseudoinverse handles the case where every input shares a null direction — the combined result legitimately remains rank-deficient and `sigma_along_unobservable_px` is set per Part 4.

**`rcond=1e-9` is mandatory** — `pinvh`'s default `rcond` is `dimension × eps_float64 ≈ 7e-16`, far too tight for our scale (per-pixel covariances of 0.1–1.0 routinely produce condition numbers up to ~10¹⁰ from the addition of small spice-orbital-residual terms). With the default, near-rank-deficient matrices are silently treated as full-rank with garbage inverse entries. `1e-9` is liberal enough to fold near-singular directions into the null space, conservative enough to preserve genuine 2-D / 3-D constraints. Tunable via `config_540_orchestrator.yaml`.

**`_combine_confidence` — exact algorithm.** Naive probabilistic-OR (`1 − Π(1 − cᵢ)`) overstates combined confidence because techniques operate on the **same image** with the **same SPICE prior** — they are not independent witnesses. Use a **precision-weighted average** of the individual confidences, weighted by each result's information content (trace of its precision matrix `pinvh(Σᵢ)`).

**Zero-W refusal (Part 0 §10).** When `W = sum(weights) == 0` (every input covariance shares one null direction — e.g. flat-ring-only scene with no orthogonal feature), the precision-weighted combine cannot proceed. The orchestrator returns `NavResult.failed(status_reason='unobservable_offset')` with no offset reported. This is honest "we can't navigate this from these features" behavior; never fall back to arithmetic-mean confidence on a degenerate covariance set.

```
weights = [ trace(pinvh(Σ_i, rcond=1e-9))   for r_i in best_group ]
W = sum(weights)
combined_confidence_raw = sum(w_i × r_i.confidence) / W   # convex average

# Boost by ≤ 1.5× when multiple precision-weighted-significant results agree —
# captures "more agreeing evidence is somewhat better" without overstating
# independence.
n_significant = sum(1 for w in weights if w > 0.1 × max(weights))
agreement_factor = 1.0 + 0.5 × max(0, log2(n_significant))   # 1, 1.5, 1.95, 2.45 for n=1,2,4,8
agreement_factor = min(agreement_factor, 1.5)               # cap

combined_confidence = min(combined_confidence_raw × agreement_factor, 0.99)

if disagreement_penalty:
    combined_confidence *= config.disagreement_penalty       # default 0.7
```

This captures both intuitions: agreement matters, but the techniques aren't independent so the boost is bounded. With one result at confidence 0.7 and information-weight 1.0, combined = 0.7. With two results at 0.7 each (similar weights), combined ≈ 0.7 × 1.5 = 0.99 (capped) — the cap is the operationally honest answer for "two correlated estimators agree". A single high-confidence result (StarFieldFromCatalog at 0.9) plus a low-confidence ring (RingEdgeNav at 0.3) is dominated by the higher-information result, ending around 0.85.

The disagreement penalty (default 0.7, in `config_540_orchestrator.yaml`) reduces the combined confidence when other groups existed but were rejected. **Composition with `conflicted_confidence_multiplier` (Part 0 §3): multiplicative.** When a result lands in the conflicted branch, both penalties apply: `final_confidence = combined × disagreement_penalty × conflicted_confidence_multiplier = combined × 0.7 × 0.3 = combined × 0.21`. When a result is *not* conflicted but multiple groups existed, only `disagreement_penalty` applies. Both gates check `len(groups_after_interior_filter) > 1` independently. The plan code's `disagreement_penalty=len(groups) > 1` matches this — `groups` is from `_agreement_groups(viable)` which already filtered.

**Key properties:**
1. **No majority vote by count**; agreement is confidence-weighted. One high-confidence technique beats two low-confidence techniques.
2. **Covariance is honored**. Flat-ring "constrains in one direction" naturally composes with another technique constraining in the other direction via `pinvh`.
3. **Disagreement is surfaced, not hidden**. If two independent techniques strongly disagree, the final confidence is knocked down sharply and the result includes both for inspection.
4. **Final-result spurious**: if `confidence < config.min_confidence`, orchestrator emits `NavResult.failed` instead of a bad offset.

**Two Mahalanobis thresholds, distinct purposes:**

- **Grouping threshold (`agreement_sigma`, default 2.0)**: results are placed in the same group iff their pairwise Mahalanobis distance is below this threshold using the *sum* of their covariances. This decides "are these compatible enough to combine".
- **Conflict threshold (implicit; emerges from `agreement_gap`)**: when ≥ 2 groups exist, conflict declaration is gated by the *summed-confidence gap* between the best and runner-up groups, not by their Mahalanobis distance. The Mahalanobis distance between groups is whatever it is; what matters is whether one group is decisively more confident than the others. This is the right question for "should I refuse rather than guess".

Both thresholds live in `config_540_orchestrator.yaml`. Full schema:

```yaml
orchestrator:
  agreement_sigma: 2.0                        # grouping threshold (Mahalanobis)
  agreement_gap: 0.5                          # min summed-confidence gap to declare
                                              # a winner; below this, status='conflicted'
  disagreement_penalty: 0.7                   # multiplier when >1 group existed
  conflicted_confidence_multiplier: 0.3       # multiplier on confidence when conflicted
  min_confidence: 0.2                         # below this final confidence, status='failed'
  pinvh_rcond: 1.0e-9                         # rank-deficient threshold for covariance
                                              # pseudoinverse (Part 4)

context:
  image_gradient_sigma_px: 1.2                # Sobel-of-Gaussian σ
  edge_threshold_k_sigma: 4.0                 # gradient threshold (units: image noise σ)

logging:
  per_image_log_dir: '{nav_results_root}/logs'   # absolute or relative; placeholders
                                                 # '{nav_results_root}' substituted at
                                                 # init from config.environment.
  per_image_log_filename: '{image_id}.log'
```

### Final success estimate

The whole `NavResult` is designed so the user (and downstream code) can read a single number-pair "offset ± uncertainty" and a single confidence rank — without needing to inspect individual technique results.

`NavResult` is the orchestrator's full in-memory output for one image. It is **not** directly JSON-serialized; a separate curator function (below) builds a curated metadata dict from it. So the dataclass can be heavy — full Annotations, per-technique state, diagnostic arrays — without bloating the on-disk JSON.

```python
@dataclass(frozen=True)
class NavResult:
    # ─── Headline (the "offset ± uncertainty" the user reads) ───────────────
    status: Literal['ok', 'failed', 'conflicted']
    offset_px: tuple[float, float] | None         # (dv, du), None on failure
    sigma_px: tuple[float, float] | None          # (sigma_dv, sigma_du), 1σ marginal
                                                  # uncertainties from the covariance
                                                  # diagonal — what users print
    sigma_along_unobservable_px: float | None     # filled when covariance is rank-1
                                                  # (e.g. flat-ring-only scenes); None
                                                  # when covariance is full-rank
    confidence_rank: Literal['high', 'medium', 'low', 'conflicted', 'failed']
    confidence: float                              # 0..1, the underlying calibrated score
    status_reason: NavStatusReason                 # typed enum (Part 0 rename); values
                                                   # include 'ok', 'no_feasible_techniques',
                                                   # 'conflicted_techniques', 'rank_1_only',
                                                   # 'unobservable_offset', etc.

    # ─── Diagnostics (curated subset → JSON; full payload here) ─────────────
    covariance_px2: np.ndarray | None              # 2×2 full covariance (sigma_px is its
                                                   # diagonal sqrt; cross terms here)
    per_technique: list[NavTechniqueResult]        # every technique that ran
    feature_inventory: list[NavFeatureSummary]     # what was extracted, gated, why
                                                   # (NavFeatureSummary defined below)
    image_classifier: NavImageClassifierResult     # always populated; flags list may be empty

    # ─── Per-NavModel diagnostic dicts (existing pattern) ───────────────────
    model_metadata: dict[str, dict[str, Any]]      # keyed by NavModel name
                                                   # (e.g. 'body:MIMAS', 'stars').
                                                   # Pass-through into the existing
                                                   # `models:` JSON block.

    # ─── Summary-image data (NOT serialized to JSON) ────────────────────────
    annotations: Annotations                        # merged from every NavModel's
                                                    # to_annotations() + the
                                                    # orchestrator's own additions
                                                    # (image name, offset arrow,
                                                    # confidence label).

    # ─── Provenance ─────────────────────────────────────────────────────────
    provenance: Provenance                         # SPICE kernel ids, code versions,
                                                   # static-data file hashes
```

**`FeatureSummary` and `ImageClassifierResult`** (in `src/nav/nav_orchestrator/nav_result.py`):

```python
@dataclass(frozen=True)
class NavFeatureSummary:
    """One entry in NavResult.feature_inventory — what an extractor produced.

    Renamed from FeatureSummary per Part 0 naming.
    """
    feature_id: str                       # 'star:UCAC4:144787700' etc.
    feature_type: NavFeatureType
    source_model: str                     # 'stars', 'body:MIMAS', 'rings:SATURN'
    reliability: float                    # [0,1]
    gated: bool                           # True if the reliability gate dropped it
    gate_reason: str | None               # human-readable when gated; e.g.
                                          # 'predicted_snr_below_threshold'
    bbox_extfov_vu: tuple[int, int, int, int]  # (v_min, u_min, v_max, u_max),
                                                # half-open (numpy slicing convention)

@dataclass(frozen=True)
class NavImageClassifierResult:
    """Output of the image-quality classifier (Part 1).

    Per Part 0 §7: 'noisy' is a flag, not a class. A noisy-but-clean image
    is `image_class='clean', flags=['noisy']`.
    """
    image_class: Literal['clean', 'blank', 'fully_overexposed',
                         'mostly_missing_data', 'partial_data_dropout',
                         'alternating_lines', 'truncated_readout',
                         'ccd_bloom_dominant', 'corrupt']
    saturation_frac: float
    missing_frac: float
    noise_sigma: float
    max_dn: float
    flags: list[Literal['partial_dropout', 'noisy', 'alternating_lines',
                        'truncated_readout', 'ccd_bloom_present']]
```

**Three artifacts come out of one `NavResult`, each with its own curation step:**

1. **Curated metadata dict** — `_build_metadata_dict(nav_result, obs) -> dict`. Picks JSON-friendly fields, formats numbers to fixed precision, drops large arrays, resolves cross-references. Output merges into the existing `_metadata.json` as the additive `navigation_result:` block plus a pass-through population of the existing `models:` block from `nav_result.model_metadata`. The curator has a strict allow-list of fields it copies; an explicit test asserts every `NavTechniqueResult` and per-model-metadata field is either explicitly included or explicitly skipped (with a comment) so that adding a diagnostic to a technique without updating the curator is caught in CI.

2. **Summary PNG** — `render_summary_image(image, nav_result.annotations) -> ndarray`. Calls into the existing `Annotations` rendering machinery (label avoidance, overlay compositing). No JSON involvement. The `annotations` field on `NavResult` exists for exactly this purpose; it's never serialized.

3. **Image log file** — already running via `pdslogger`; the orchestrator + techniques write structured INFO/DEBUG lines as they go. The text log is the human-readable companion to the JSON metadata.

**`offset_px ± sigma_px`** is the user-visible answer. `sigma_px` is the per-axis 1σ marginal uncertainty — the square-root of the diagonal of the 2×2 covariance. For most images this is the only thing that gets printed in the metadata file (e.g. `"offset_dv = 298.96 ± 0.4 px, offset_du = -130.71 ± 0.5 px"`). It's a single-number-per-axis quantity that aggregates all the contributions: per-technique measurement uncertainty, per-feature position uncertainty, and ensemble combination uncertainty — already composed by the orchestrator's precision-weighted Kalman-style merge.

**`sigma_along_unobservable_px`** handles the rank-1 case explicitly. When the only feasible features are flat ring edges and there's no star or body to supply the orthogonal axis, the covariance is rank-1; in that case `sigma_px` is the perpendicular-direction uncertainty (small), and `sigma_along_unobservable_px` is set to `float('inf')` in the in-memory `NavResult`. `status_reason` is `'rank_1_only'`. The metadata curator (which produces JSON) clamps to a finite sentinel (`1e9`) for JSON serialization since `Infinity` is not strictly JSON. Downstream consumers reading the JSON should treat any value ≥ `1e8` as "axis unconstrained". The in-memory representation keeps `inf` so internal arithmetic produces correct unbounded propagation.

**`confidence_rank`** is the simplified five-bucket rank derived from `confidence` and `status`, intended for users who want a yes/no/maybe rather than a float. The five-bucket form (per Part 0 §4) lets downstream consumers refuse `conflicted` results independently of merely-low-confidence successes. The mapping (config-tunable via `config_510_techniques.yaml`):

| rank | when |
|---|---|
| `high` | status='ok', confidence > 0.8, expected offset error < 0.5 px (calibrated on image library) |
| `medium` | status='ok', confidence ∈ (0.5, 0.8], expected offset error < 2 px |
| `low` | status='ok', confidence ∈ (0.2, 0.5]; offset is reported but should not be used downstream without review |
| `conflicted` | status='conflicted'; offset reported (best-group precision-weighted combine) but flagged as untrusted because ≥2 high-confidence groups disagreed |
| `failed` | status='failed' or confidence ≤ 0.2 (no usable offset) |

The thresholds (0.8 / 0.5 / 0.2 and the px tolerances) are declared in `config_510_techniques.yaml`, not hard-coded, so an operator can tighten them for a high-stakes pipeline (e.g., "high requires confidence > 0.9 and sigma_px < 0.3").

**Metadata format — additive, not replacing.** The existing per-image `_metadata.json` schema (top-level keys `status`, `observation`, `spice_kernels`, `models`, `navigation_techniques`, `offset`, `confidence`) is preserved as-is for downstream readers that already parse it. The new pipeline writes the existing fields with their existing semantics (`offset` = `(dv, du)` tuple or null; `confidence` = float; `navigation_techniques` = dict keyed by technique name) and **adds** a `navigation_result` sub-block alongside them with the richer information:

```json
{
  "status": "success",
  "observation": { ... },
  "spice_kernels": [ ... ],
  "models": { ... },
  "navigation_techniques": { ... },
  "offset": [298.96, -130.71],
  "confidence": 0.87,
  "navigation_result": {
    "sigma_dv_px": 0.41,
    "sigma_du_px": 0.52,
    "sigma_along_unobservable_px": null,
    "confidence_rank": "high",
    "status_reason": "ok",
    "covariance_px2": [[0.168, 0.012], [0.012, 0.270]],
    "techniques_used": ["BodyDiscCorrelateNav", "StarRefineNav"],
    "feature_count_by_type": {"BODY_DISC": 1, "LIMB_ARC": 1, "STAR": 5},
    "per_technique": [ ... ],
    "feature_inventory": [ ... ],
    "provenance": { ... }
  }
}
```

Existing readers that consume `offset` / `confidence` keep working unchanged. New readers can opt into the richer `navigation_result` block.

### Failure-mode taxonomy

A single `NavStatusReason(StrEnum)` in `src/nav/support/status_reason.py` carries every value, importable so call sites can't drift on string spellings. Located under `nav.support` (not `nav_orchestrator`) because it includes failure modes that are not strictly navigation outcomes — image-load errors, missing kernels, instrument-not-configured — and is consumed by code outside the orchestrator (image-quality classifier, kernel-loading shim, dataset enumeration). 15 values across five stages (the `unobservable_offset` value below was added per Part 0 §10 to bring the table to its declared 15).

| `status_reason` | `status` | `confidence_rank` | offset reported? | When |
|---|---|---|---|---|
| `ok` | ok | high / medium / low | yes | normal success |
| `rank_1_only` | ok | low or medium | yes | flat-ring-only scene; 1 axis unobservable, `sigma_along_unobservable_px` set |
| `conflicted_techniques` | conflicted | low | yes (combined, confidence × 0.3) | ≥2 groups exist *after* 2σ Mahalanobis grouping, and best-vs-runner-up summed-confidence gap < `agreement_gap` (default 0.5) |
| `no_signal_in_image` | failed | failed | no | image classifier: blank / dark frame |
| `image_overexposed` | failed | failed | no | image classifier: > 80% pixels at full-well DN |
| `missing_data_dominant` | failed | failed | no | image classifier: > 30% pixels at missing-data marker |
| `image_corrupt` | failed | failed | no | image file failed to parse / read |
| `kernels_unavailable` | failed | failed | no | SPICE coverage missing for the image ET |
| `instrument_not_configured` | failed | failed | no | no `config_4N0_inst_*.yaml` entry for this camera |
| `no_features_extracted` | failed | failed | no | every extractor returned empty |
| `all_features_gated` | failed | failed | no | features extracted but all fell below reliability gate |
| `no_feasible_techniques` | failed | failed | no | features pass gate but no technique's `is_feasible` returns true |
| `all_techniques_spurious` | failed | failed | no | every technique returned `spurious=True` |
| `final_confidence_below_threshold` | failed | failed | no | ensemble combined confidence < `config.min_confidence` |
| `unobservable_offset` | failed | failed | no | every input covariance shares one null direction (e.g. all-flat-rings, no orthogonal feature). `_combine_precision_weighted` cannot proceed; refusing rather than guessing. Per Part 0 §10. |

`no_features_extracted` vs `all_features_gated` are kept distinct because the diagnostic owner is different — the former points at extractors (predicted nothing was visible); the latter points at the reliability formulas (predicted things but rated them all unusable).

`conflicted_techniques` reports the best-group precision-weighted offset with `confidence × 0.3` and `confidence_rank='low'`; backplanes / PDS4 bundles must check `confidence_rank` and refuse to consume a `low` result by default.

**Per-status_reason mandatory diagnostic fields** (added to the standard `navigation_result` block):

```jsonc
// "ok" / "rank_1_only" — no extra fields beyond standard set

// "conflicted_techniques"
"conflicted_groups": [
  {"technique_names": [...], "offset_px": [...], "confidence": ...}, ...
],
"pairwise_mahalanobis_max": 4.7

// "no_signal_in_image" / "image_overexposed" / "missing_data_dominant" / "image_corrupt"
"image_classifier": {
  "class": "blank" | "fully_overexposed" | "mostly_missing_data" | "corrupt",
  "saturation_frac": 0.04,
  "missing_frac": 0.42,
  "noise_sigma": 1.3,
  "max_dn": 18.5,
  "flags": []                      // see below
}

// "kernels_unavailable"
"missing_kernels": [...],
"image_et": 414504000.0

// "instrument_not_configured"
"instrument_name": "...",
"camera": "..."

// "no_features_extracted" / "all_features_gated"
// (feature_inventory is already standard)
"extractor_diagnostics": {
  "StarFeatureExtractor": "no_predicted_snr_above_threshold",
  ...
}

// "no_feasible_techniques"
"technique_feasibility": {
  "BodyDiscCorrelateNav": "no BODY_DISC features",
  "StarFieldFromCatalogNav": "only 2 STAR features (need 3)",
  ...
}

// "all_techniques_spurious" / "final_confidence_below_threshold"
// (per_technique is already standard)
"ensemble_diagnostics": {
  "best_group_summed_confidence": 0.18,
  "min_confidence_threshold": 0.20,
  "all_results_spurious": true,
  "all_results_at_edge": false
}
```

**Standard `image_classifier` block on every result**, success or fail, with a `flags` list for caveats that don't promote to a status_reason. Flag values: `partial_dropout` (5–30% missing pixels), `noisy` (noise_sigma above instrument threshold), `alternating_lines` (Voyager-style interlace repaired), `truncated_readout` (only top-N rows have data), `ccd_bloom_present` (saturated columns masked). Downstream consumers can assert "this image was correctly noted as noisy" without parsing thresholds.

**Operator INFO line per status_reason**: a static `STATUS_REASON_INFO_TEMPLATE` dict on `NavResult`; tests assert message format for every value. The status_reason → required-extra-fields map is enforced by `assert_diagnostic_fields_present(result)` in the metadata curator; missing required field on a known status_reason is a CI failure.

---

## Part 5 — Static data tables (the no-statistics substitute)

Because no cross-image learning is allowed, all numeric uncertainties must be **static, per-object** data the extractors read at render time.

### Static-data sources (mix of existing + new)

**1. Existing ring-feature catalogs — reuse and minimally extend.**

`src/nav/config_files/config_20_jupiter_rings.yaml`, `config_21_saturn_rings.yaml`, `config_22_uranus_rings.yaml`, `config_23_neptune_rings.yaml` already contain per-ring-feature data with everything the `RingEdgeExtractor` needs:

- `feature_type` (`GAP` | `RINGLET`) — defines which side of the feature the radial transition is on.
- `inner_data[]` / `outer_data[]` mode lists. The mode-1 entry carries semi-major axis `a` (radius in km), `rms` (radial RMS precision — the primary radial uncertainty), and eccentric-orbit parameters (`ae`, `long_peri`, `rate_peri`). Higher-mode entries carry amplitude / phase / pattern-speed terms that describe non-circular / normal-mode distortions (captured analytically, not as uncertainty).
- Some features are time-versioned (e.g., `keeler_a_ring_oe_*` with `start_date` / `end_date`); extractor already filters by observation time.

The **radial uncertainty** fed into the `RING_EDGE` feature is the mode-1 `rms` value (km), projected to pixels at the observation resolution. The **dynamical variation** (e.g. amplitude of mode-2 oscillations) is rendered into the *template* at the feature's observation-time longitude, not smeared as uncertainty — so it sharpens the matchable signal rather than fuzzes it.

Minimal extensions to these YAMLs (adding keys, not schema changes):

- Optional per-feature `default_reliability: float` — operator override when a named feature is known to be unreliable in particular regimes (e.g. F-ring core when the clumps are active). If absent, reliability is derived from `rms`: `max(0.3, 1 - rms_km / 10)`.
- Optional `sharpness: 'sharp' | 'soft'` — governs whether a gradient-based or distance-transform-based matching metric is used. Default (`sharp` if `rms < 2 km`, `soft` otherwise).

The reusability is a real cost saving: no new Saturn / Uranus / Neptune / Jupiter catalog to populate — the astronomers' decades of ring-orbit fitting is already encoded.

**2. New: `config_220_body_shape.yaml`** (in `src/nav/config_files/`, loaded by the existing `Config` machinery — accessed as `config.body_shape['MIMAS']`).

Scope per operator decision (§14.1): **all relatively large satellites of the four giants + Mars moons + Earth's Moon + any planet that might appear in images.** No comets, no small asteroids (their images are rare and mostly appear via dedicated mission datasets, not as bystanders).

Key observation from the operator: *"Even irregular satellites can be navigated against if you are sufficiently far away so the shape isn't obvious."* Irregularity isn't a shape-class-alone gate — it's irregularity **projected to pixels at the current observation resolution**. The body extractor therefore computes `limb_uncertainty_px = ellipsoid_residual_km / km_per_px_at_limb` per image and uses *that* as the gate; a `shape_class` tag is only a default starting point.

```yaml
MIMAS:
  radii_km: [207.4, 196.8, 190.6]
  ellipsoid_rms_residual_km: 1.4
  crater_scale_km: 3.0
  albedo_mean: 0.96
  albedo_variation: 0.05                # relative albedo std on visible hemisphere
  shape_class_hint: regular             # informational only — not the gate
  _sources:                             # Per Part 0 §74 — every numeric value cited.
                                        # Loader ignores keys starting with `_`.
    radii_km: 'Thomas (2010), Icarus 208(1):395-401, Table 3 Mimas row.
               doi:10.1016/j.icarus.2010.01.025'
    ellipsoid_rms_residual_km: 'Thomas (2010), same paper, Table 3
                                "RMS residual" column.'
    crater_scale_km: 'Schenk & Moore (2007), JGR Planets 112:E12.
                      doi:10.1029/2007JE002942. Median large-crater depth.'
    albedo_mean: 'Verbiscer et al. (2007), Science 315:815, Table 1
                  geometric albedo V band.'
    albedo_variation: 'Buratti & Veverka (1984), Icarus 58:254-264,
                       per-hemisphere albedo std.'
    shape_class_hint: 'IAU WGCCRE 2018 report, classification table.'

PROMETHEUS:
  radii_km: [68.2, 41.6, 28.3]
  ellipsoid_rms_residual_km: 8.0
  crater_scale_km: 2.0
  albedo_mean: 0.6
  albedo_variation: 0.20
  shape_class_hint: highly_irregular
  _sources:
    radii_km: 'Thomas et al. (2010), Icarus 208:395-401, Table 3
               Prometheus row. doi:10.1016/j.icarus.2010.01.025'
    ellipsoid_rms_residual_km: 'Thomas et al. (2010), same table; large
                                value reflects irregular shape.'
    crater_scale_km: 'Porco et al. (2007), Science 318:1602, ISS imaging
                      of Prometheus, surface roughness estimate.'
    albedo_mean: 'Buratti et al. (2010), Icarus 206:524, Table 2
                  Prometheus geometric albedo.'
    albedo_variation: 'Buratti et al. (2010), same paper, std across
                       observed phase angles.'
    shape_class_hint: 'IAU WGCCRE 2018 report, irregular satellites
                       classification.'
```

The example citations above are illustrative for the schema; the actual
populated YAML uses citations whose accuracy has been spot-checked per
Part 0 §74. Sample values (e.g., `1.4` for MIMAS `ellipsoid_rms_residual_km`)
must be re-verified by an operator before they ship.

(Per Part 0 §22: no `notes:` field is part of the schema; per-body
commentary lives in YAML `# ...` comments. The schema validator rejects
unknown top-level keys per body. As a worked numerical example for
PROMETHEUS — at 100 km/px `limb_uncertainty = 0.08 px → LIMB_ARC` is
emitted; at 1 km/px `limb_uncertainty = 8 px → BODY_BLOB` is emitted
instead.)

Extractor rules (per-image, per-body):

- If `limb_uncertainty_px ≤ limb_uncertainty_px_max_for_arc` (config default 3 px): emit `LIMB_ARC` with filter σ = `limb_uncertainty_px`.
- Else if body diameter in image ≥ `body_blob_min_px` (config default 8 px): emit `BODY_BLOB`.
- Else: no body features; body is below the useful-resolution threshold for this image.

So Prometheus at approach is a `BODY_BLOB` (maybe with confidence-cap 0.3); Prometheus in a wide F-ring mosaic (100+ km/px) is a perfectly usable `LIMB_ARC`. Same entry, different per-image outcomes.

**3. Extensions to existing `config_4N0_inst_*.yaml`** — per-camera `noise:` and `mag_offset:` blocks added under each existing camera section. The instrument noise data, mag-offset tables, and image-quality classifier thresholds all live here, alongside the existing `extfov_margin_vu` / `star_psf_sigma` fields. Schema in the proposal section below; no new file. Loaded by the existing `Config` machinery — accessed as `config.cassini_iss.nac.noise.read_noise_dn` and `config.cassini_iss.nac.mag_offset.filter_combos['CL1+CL2']`.

Each entry is loaded once at config init and cached. Missing body entries fall back to `shape_class_hint: unknown` with conservative defaults (residual ≈ 10% of mean radius). Missing instrument-noise blocks log a warning and force stars to `reliability ≤ 0.3`.

### How these substitute for cross-image statistics

- Per-body ellipsoid residuals from literature once, captured permanently.
- Per-ring radial uncertainties from the existing `config_3N0_*_rings.yaml` catalogs.
- Per-instrument noise from radiometric characterization.
- All image-specific uncertainty (km_per_pixel at the feature, projected curvature, phase angle, limb-uncertainty-in-px) is derived per-image from SPICE + image size.

No image's result influences any other image's defaults. Parallel execution is natively supported.

### Backplane query reference (per Part 0 §19)

Every quantity the new pipeline derives from `oops` backplanes is named
explicitly here so two implementers don't pick different methods. All queries
go through `obs.ext_bp` (the existing `Backplane` accessor on `ObsSnapshotInst`).

| Quantity | Backplane query | Notes |
|---|---|---|
| `km_per_pixel_at_limb` (per-vertex) | `Backplane.resolution(body, 'sub_observer')` evaluated at the vertex pixel | Per-vertex; foreshortening varies across the silhouette |
| `km_per_pixel_at_FOV_center` (legacy, used in non-vertex contexts) | `Backplane.center_resolution(body)` | Single scalar per body per image |
| `km_per_pixel_radial` (ring) | `Backplane.ring_radial_resolution(planet)` | Per-edge; for projection of `rms_km` → px |
| `incidence_angle_per_pixel` (body, lit/dark check) | `Backplane.incidence_angle(body)` | Used by `cos(incidence)` in predicted-DN computation |
| `emission_angle_per_pixel` (body or ring) | `Backplane.emission_angle(body_or_ring)` | Source for `mean_emission_factor` (RING_EDGE reliability) |
| `phase_angle` (per-body) | `Backplane.phase_angle(body)` | Source for `phase_angle_factor` (TERMINATOR_ARC reliability) |
| `predicted_disc_silhouette_mask` | `Backplane.where_intercepted(body)` | For `overflow_fraction` and Z-buffer paint silhouette |
| `predicted_ring_radius_per_pixel` | `Backplane.ring_radius(planet)` | For ring-edge polyline densification + clipping |
| `predicted_ring_longitude_per_pixel` | `Backplane.ring_longitude(planet)` | For Neptune arc geometry + ring-feature longitude |
| `body_distance` (subject_range_km) | `Backplane.distance(body)` evaluated at the body center | Scalar per body; used for Z-buffer sort |

Implementers add new queries by extending this table, not by inventing
ad-hoc names. Tests in `tests/nav/feature/test_backplane_queries.py` assert
each named query returns finite values for the synthetic-obs fixtures.

---

## Part 5b — Camera rotation correction (per instrument)

Some missions have rotation errors as well as offset errors in their reconstructed attitude — Cassini ISS has near-perfect rotation, but VGISS and GOSSI do not, and the rotation residual is not consistent image-to-image so it can't be calibrated out per-mission. The new architecture supports fitting a small camera rotation as part of the navigation solution.

**Per-instrument flag (config):**

```yaml
inst:
  fit_camera_rotation: false      # Cassini, NHLORRI default — known-accurate rotation
  max_rotation_deg: 5.0           # applies when fit_camera_rotation = true
```

Stored in `config_4N0_inst_*.yaml`. Default `false`; flip on for VGISS / GOSSI based on observed residuals during Phase 5 calibration.

**Behavior when enabled:**

- Each technique's cost function gains a third parameter `dθ` (radians internally; bounded `±deg_to_rad(max_rotation_deg)`):
  ```
  cost(dv, du, dθ) = Σ DT(image, R(dθ) · (vertex − pivot) + pivot + (dv, du))²
  ```
- The rotation pivot is the natural geometric center for the technique:
  - `BodyLimbNav`: predicted body center (single body) or centroid-of-body-centers (multi-body).
  - `BodyDiscCorrelateNav`: same as above; pyramid extends to a 3-D (dv, du, dθ) coarse-search grid.
  - `RingEdgeNav`: predicted planet center.
  - `BodyBlobNav`: 2 DoF only (a centroid is rotation-invariant).
  - `StarFieldFromCatalogNav`: 3 DoF natively (Procrustes already fits rotation). Pivot is the centroid of inlier matched points.
  - `StarRefineNav`: 3 DoF when pass-1 produced a rotation; 2 DoF otherwise.
- All techniques on the same image use the same parameterization shape — driven by the per-instrument flag (2-DoF or 3-DoF), never mixed across techniques. Within a 3-DoF run, a single technique may legitimately produce a *rank-deficient* result when geometry doesn't constrain a dimension (e.g. `BodyBlobNav` carries zero rotation information, `RingEdgeNav` on flat rings carries one fewer translation dimension, `StarUniqueMatchNav` 1-star carries zero rotation information). The 3×3 covariance reflects this directly; the precision-weighted ensemble combines such rank-deficient results correctly without special cases.

**`NavResult` gains:**

```python
rotation_rad: float | None        # internal representation
sigma_rotation_rad: float | None
```

The metadata curator converts to `rotation_deg` and `sigma_rotation_deg` for the JSON, and omits both fields entirely when `fit_camera_rotation` is False (cleaner than serializing nulls).

**Ensemble combination:** when rotation is enabled, agreement / precision-weighted-combine math operates in 3-D (offset_v, offset_u, rotation) instead of 2-D. Covariance is 3×3. Mahalanobis agreement check is in 3-D. Rank-deficient handling extends naturally — if a technique's rotation eigenvalue is at the noise floor (e.g. `BodyBlobNav` providing zero rotation information, or `RingEdgeNav` on flat rings being doubly-rank-deficient), the precision-weighted combine treats that direction as no-information.

**Sub-decisions / pessimism:**

- **3-DoF NCC pyramid is more compute.** Concrete sample schedule for body disc correlation, mirroring the existing `navigate_with_pyramid_kpeaks` 4-level structure:
  - Level 0 (coarsest, 1/8 resolution): 11 rotation samples across `±max_rotation_deg` in 1° steps (centered on 0°).
  - Level 1 (1/4 resolution): 5 rotation samples in 0.5° steps centered on the level-0 winner.
  - Level 2 (1/2 resolution): 3 rotation samples in 0.25° steps centered on the level-1 winner.
  - Level 3 (full resolution): 1 sample (the level-2 winner); sub-pixel + sub-degree refinement via local Gauss-Newton on the (dv, du, dθ) cost surface.

  Total rotation evaluations: 11 + 5 + 3 + 1 = 20. With the existing 5 spatial pyramid levels each costing roughly the same, total NCC cost is ~20× the 2-DoF baseline at the *coarsest* level only (most cost is at full resolution where rotation is already fixed). Net overhead vs 2-DoF: roughly 2-3× total compute, dominated by level 0. Cassini and NHLORRI pay no overhead (flag off); VGISS and GOSSI accept the cost.

  For DT-based techniques (BodyLimbNav, BodyTerminatorNav, RingEdgeNav), rotation is a third optimizer parameter rather than an outer-loop search — Levenberg-Marquardt converges from the 2-D coarse-search basin in < 15 iterations including rotation. Compute overhead < 50%.
- **Rotation ambiguity on partial-overlap geometry.** A short visible limb arc + free rotation has correlated (rotation, translation) ambiguity — covariance becomes ill-conditioned. The 3×3 covariance reflects this directly; orchestrator's rank-deficient handling produces low-confidence results without needing special cases.
- **`max_rotation_deg` is a knob.** Default 5° per the user's "<5°" guidance; per-instrument tuning in Phase 5.

## Part 6 — Scenario handling

### Scenarios this design explicitly addresses

| Scenario | How it's handled |
|---|---|
| Body fills FOV, small overflow | `BODY_DISC` feature → `BodyDiscCorrelateNav` with `use_gradient='auto'`. Gradient wins. |
| Body 80% off-frame (C0061084700R-class) | `LIMB_ARC` feature with visible arc > threshold → `BodyLimbNav` (DT matching). Ellipsoid fine for Moon. |
| Highly irregular body (Prometheus) at close range | `BodyLimbExtractor` computes `limb_uncertainty_px = ellipsoid_residual_km / km_per_px_at_limb`. Close-in, that's > 3 px; extractor emits `BODY_BLOB`. `BodyBlobNav` returns centroid-based offset, confidence cap ~0.3. |
| Highly irregular body (Prometheus) at distance | Same extractor, same formula. Far away, `limb_uncertainty_px < 3`; extractor emits `LIMB_ARC` as normal. Ellipsoid model works because shape-deviation doesn't project to a resolvable offset. |
| Multiple bodies in FOV | All bodies produce features independently; the body techniques (`BodyDiscCorrelateNav`, `BodyLimbNav`, `BodyBlobNav`) consume the full list and produce one offset that constrains all bodies jointly. Combined template/limb/blob fit disambiguates "which moon is which" even when individual moons look alike (Rhea / Dione, Atlas / Pan, Tethys / Mimas at low-res). |
| Ring-only scene with straight-edge rings | Each ring edge contributes a rank-1 (radial-direction-only) constraint. All edges of one planet share the same projected normal, so multiple flat edges give the same 1-D direction, not a 2-D offset. Ensemble returns a result with rank-1 covariance and very large reported uncertainty along the unobservable axis; final 2-D confidence is low. The ensemble *does* resolve to a full 2-D offset when at least one star or body feature is also present (their covariance covers the orthogonal axis). |
| Faint star field (sensor-limited) | Star extractor predicts SNR per star. If no star has predicted SNR > threshold (say 3), extractor emits zero features and flags `starfield_unusable`. Both `StarFieldFromCatalogNav` and `StarUniqueMatchNav` silently skip. |
| One bright star, no body, nothing else | `StarUniqueMatchNav` runs: catalog reduction yields a unique brightest predictable star; the brightest detection is matched to it; offset = detection − prediction. Confidence capped at 0.7 (no internal cross-check). On `fit_camera_rotation = True` instruments the 3×3 covariance is rank-2 (translation observable, rotation unobservable); ensemble combine handles via `pinvh`. |
| Two bright stars, no body | `StarUniqueMatchNav` runs in 2-star mode: tries both detection-to-catalog assignments, picks smaller-residual fit; confidence ~0.8 because the fit residual cross-checks the assignment. Fits rotation when enabled. |
| Terminator-only body (high phase, no limb) | `BodyLimbExtractor` sees no limb-incidence-angle pixels in FOV; emits only `TERMINATOR_ARC`. Orchestrator relies on `BodyTerminatorNav` + other features. |
| Scattered-light contaminated image (GOSSI / VGISS) | Source-image DoG bandpass removes low-frequency stray light. Per-feature filters unchanged. |
| Large body at low resolution (~10 px) | Resolution gate in body extractor: emit `BODY_BLOB` regardless of shape class. |
| No usable features at all | Orchestrator returns `NavResult.failed(status_reason='no_feasible_techniques')`. |
| Two techniques agree; one disagrees | Ensemble's agreement-grouping picks the larger-confidence group; disagreeing one is listed in diagnostics. |
| Two techniques both confident but disagree | `NavResult.conflicted` with lowered confidence; operator-review flag. |

### Non-visible rings

If a scene contains a planet but predicted ring radii are either (a) behind the planet (occlusion check), (b) in the planet's shadow (shadow geometry), or (c) at < 1 px/radius resolution for the farthest edge, the ring extractor emits zero ring features (or a single `RING_ANNULUS` when individual edges can't be separated).

### Unusable stars

`StarFeatureExtractor`:
- For each predicted star in ext-FOV, compute expected peak DN using star's catalog magnitude + instrument gain + exposure + PSF.
- Estimate image noise via `mad_std(image[outside-body-mask])`.
- predicted_SNR = peak_DN / noise_DN.
- Emit `STAR` feature iff predicted_SNR > `stars.min_predicted_snr` (config, default 3.0) AND star is not inside a body mask AND star is > `stars.min_pixels_from_edge` from image edge.
- If no features emitted, write diagnostic `starfield_failure_reason` with the breakdown.

---

## Part 7 — Fully-autonomous operation constraints

Every design decision checked against these:

1. **No cross-image state** — extractors and techniques only consume `obs`, `image`, static tables, and config. The orchestrator initializes a fresh state per image. Enforced structurally: extractors/techniques are stateless classes instantiated per-image.
2. **Parallelizable** — no shared mutable state. Each image's orchestrator run is independent. Cloud-tasks worker launches N parallel navigations.
3. **No manual validation step** — the pipeline emits a `NavResult` with a confidence; downstream filters on confidence. No human-in-the-loop required.
4. **Deterministic (given fixed SPICE + image)** — techniques use fixed random seeds where relevant (none should need randomness). Same inputs ⇒ same outputs.
5. **Calibrated, not accumulated** — confidence formulas are fixed in config; they don't learn.

---

## Part 8 — File list

### Existing code to **reuse**, not reimplement

Validated routines that the new pipeline depends on. Do not duplicate or paraphrase them; import directly.

| Existing component | File | What we use |
|---|---|---|
| `NavModelStars.stars_list_for_obs()` | `src/nav/nav_model/nav_model_stars.py:437` | Catalog reduction (UCAC4 / Tycho2 / YBSC), aberration via `_aberrate_star`, proper motion, magnitude binning, multi-catalog precedence, star-vs-star + star-vs-body conflicts via `_mark_conflicts_obj`, per-star PSF size, smear via `move_u`/`move_v`. Fully replaces my proposed pattern-matcher catalog reduction. |
| `RingFeatureFilter` | `src/nav/nav_model/rings/ring_filter.py:44` | Four-pass selection (date / radius-in-FOV / resolvability / fade-conflict). Reused by `RingEdgeExtractor`. |
| `Annotation`, `Annotations`, `AnnotationTextInfo` | `src/nav/annotation/{annotation,annotations,annotation_text_info}.py` | Summary-image rendering. Each `NavModel.to_annotations()` returns an `Annotations` collection; orchestrator merges via `add_annotations()`. No new annotation classes. |
| `nav.reproj.cartographic_model.create_cartographic_model` | `src/nav/reproj/cartographic_model.py:43` | Cartographic-model technique's template render. |
| `psfmodel.GaussianPSF.eval_rect(..., movement=, movement_granularity=)` | `psfmodel.gaussian:689` | Smear-aware PSF render for star detection kernel + per-star template; matches the `(star.move_v, star.move_u)` parameters already used by the existing renderer at `nav_model_stars.py:773`. |
| `obs.extfov_data_sensor_mask`, `.inventory`, `.inventory_body_in_fov`, `.inventory_body_in_extfov`, `.ext_bp`, `.star_psf`, `.star_psf_size`, `.ra_dec_limits_ext` | `src/nav/obs/obs_snapshot.py` | Image-side scaffolding consumed by the orchestrator and extractors. None reimplemented. |
| `NavBase` | `src/nav/support/nav_base.py:8` | Base class for all new orchestrator / extractor / technique classes — provides `_logger` + `_config` consistently. |
| Existing stats consumer reference | `/seti/nav/rms-csmithing/navigation/nav_main_stats.py` | Schema reference for what the future stats program will track (per-technique offsets, per-NavModel metadata blocks, scene-class numerics like `radial_gradient` / `curvature` / `emission` / `num_features`). The redesigned JSON metadata preserves this information set in the new shape. |

### New files (code)

```
src/nav/feature/
  __init__.py                  # declares __all__ per Part 0 §46;
                               # NullHandler per Part 0 §61
  feature.py                   # NavFeature dataclass + flag dataclasses
  feature_type.py              # NavFeatureType enum (single canonical list of 9)
  geometry.py                  # NavFeatureGeometry sum type + variants per Part 0
  extractor.py                 # NavFeatureExtractor ABC + registry
  star_extractor.py            # StarFeatureExtractor
  body/                        # split per Part 0 §47
    __init__.py
    limb_extractor.py          # BodyLimbExtractor
    terminator_extractor.py    # BodyTerminatorExtractor
    blob_extractor.py          # BodyBlobExtractor
    disc_extractor.py          # BodyDiscExtractor
    cartographic_extractor.py  # NavCartographicExtractor
  ring_extractors.py           # RingEdgeExtractor, RingAnnulusExtractor
  reliability.py               # FeatureReliabilityGate, scoring functions
  constants.py                 # named constants per Part 0 §44 (
                               # MAX_INCIDENCE_FACTOR_CAP = 4.76, etc.)

src/nav/support/
  filters.py                   # NavFilterSpec, apply_filter, all NavFilterKind kinds
  distance_transform.py        # DT computation + chamfer matching helpers
  noise_estimate.py            # image noise estimator used across extractors
  status_reason.py             # NavStatusReason(StrEnum) — used by orchestrator,
                               # image classifier, kernel-loading shim,
                               # dataset enumeration; placed under support
                               # because it includes non-navigation failure
                               # modes (kernels missing, file not found, etc.)
  types.py                     # NDArrayFloat, NDArrayBool typing aliases
                               # (single source of truth, per Part 0 §45)
  glob_filter.py               # NOT a new file — glob negation is added to
                               # nav.nav_technique.NavTechnique._filter_models
                               # itself (Part 0 §9). Documenting the absence
                               # here so an implementer doesn't re-add it.
  backplane_queries.py         # named wrappers from Part 5 "Backplane query
                               # reference" (km_per_pixel_at_limb, ...).
                               # Tests in test_backplane_queries.py.
  filter_combo.py              # canonicalize obs.filters → canonical string;
                               # shared by mag_offset lookup + sidecar schema.
                               # Spec: canonicalize(Sequence[str|None]) -> str
                               #   []           -> 'NONE'
                               #   ['CL1']      -> 'CL1'           # 1 filter
                               #   ['CL2','CL1']-> 'CL1+CL2'       # sorted-joined
                               #   ['F1',F2,F3] -> 'F1+F2+F3'      # any N
                               #   None entries dropped before sorting

src/nav/nav_technique/
  nav_technique.py                # NavTechnique ABC + __init_subclass__ registry
                                  # (the registry is a ClassVar on NavTechnique
                                  # itself; no separate module needed — see
                                  # registry-mechanism block in Part 4).
                                  # Per Part 0 §9: _filter_models is extended
                                  # to parse leading '!' as exclusion (gitignore-style).
  confidence.py                   # evaluate_sigmoid_combination + load-time
                                  # validator (Part 0 §16)
  star_field/                     # split per Part 0 §47
    __init__.py
    detection.py                  # DAOPHOT-style source detection
    triplet_hashing.py            # hash construction + KD-tree
    ransac.py                     # RANSAC fit + verify
    nav.py                        # StarFieldFromCatalogNav (thin orchestrator)
  nav_technique_star_unique.py    # 1-or-2-star catalog-uniqueness match
  nav_technique_star_refine.py    # prior-required PSF refinement
  nav_technique_body_disc.py      # single-body disc correlation
  nav_technique_body_limb.py      # DT-based limb nav (NEW)
  nav_technique_body_terminator.py
  nav_technique_body_blob.py      # centroid matching
  nav_technique_ring_edge.py      # DT-based ring edge nav
  nav_technique_ring_annulus.py
  nav_technique_cartographic.py   # cartographic-model correlation
  nav_technique_titan.py          # stub; reports infeasible always
  # NOTE per Part 0 §6: LimbRefineNav was considered and dropped.
  # BodyLimbNav does its own LM subpixel refinement after coarse DT match;
  # no separate prior-required limb technique exists.

src/nav/nav_orchestrator/
  __init__.py                  # declares __all__ per Part 0 §46;
                               # NullHandler per Part 0 §61
  orchestrator.py              # NavOrchestrator (helper bodies per Part 0 §23)
  ensemble.py                  # result reconciliation (free function `ensemble`)
  nav_result.py                # NavResult dataclass
  feature_summary.py           # NavFeatureSummary (split per Part 0 §47)
  image_classifier_result.py   # NavImageClassifierResult (split)
  provenance.py                # Provenance dataclass (split)
  curator.py                   # _build_metadata_dict, assert_diagnostic_fields_present
  status_reason_info.py        # STATUS_REASON_INFO_TEMPLATE dict

src/main/
  nav_feature_inspect.py       # NEW CLI: dump feature inventory for a single image
                               # (debugging tool, no navigation done)
```

### Modified files

```
src/nav/nav_master/nav_master.py
  - DELETED. Callers move to NavOrchestrator directly (per Part 12.5).
  - If a transitional facade is unavoidable, it lives in src/nav/nav_orchestrator/
    and forwards a single .navigate() call; no flag-gated legacy path.

src/nav/nav_model/nav_model.py
  - add to_features(context) default implementation (returns [])
  - add to_annotations(context) default implementation (returns empty Annotations)

src/nav/nav_model/nav_model_result.py
  - DELETED. NavModelResult is gone (per Part 1). Pixel templates live on
    NavFeature.template_img for BODY_DISC / RING_ANNULUS / CARTOGRAPHIC_MODEL;
    polyline payloads live on NavFeature.geometry for LIMB_ARC / TERMINATOR_ARC /
    RING_EDGE; per-model diagnostic state is the existing NavModel.metadata dict.

src/nav/nav_model/nav_model_stars.py  → SPLIT into a package:
  src/nav/nav_model/stars/
    __init__.py             # re-export NavModelStars and the helpers used by tests
    nav_model_stars.py      # the NavModelStars class + create_model() (existing logic)
    predicted_snr.py        # per-star predicted-SNR / mag-offset / B-V lookup (new)
    detection.py            # DAOPHOT-style source detection (matched filter, shape cuts,
                            # saturated-star annular moment, bloom detection)
    features.py             # to_features() — emits STAR features
    annotations.py          # to_annotations() — star-box overlays + name labels
  Reason: existing module is ~890 lines; adding to_features / detection / SNR
  pushes past the 1000-line cap (cursor rule §2). Split before merging.

src/nav/nav_model/nav_model_body.py
  - add limb / terminator polyline extraction
  - add shape-class lookup from config_220_body_shape.yaml
  - to_features() emits LIMB_ARC / TERMINATOR_ARC / BODY_DISC / BODY_BLOB per
    shape-class + resolution logic

src/nav/nav_model/nav_model_rings.py
  - add per-edge polyline + straight-line detection
  - to_features() emits RING_EDGE (one per edge) or RING_ANNULUS

src/nav/support/correlate.py
  - keep as a library; BodyDiscCorrelateNav uses navigate_with_pyramid_kpeaks
  - retain use_gradient='auto' behavior unchanged
  - move at-edge / spurious computation to shared helpers so techniques other
    than body-disc can reuse

src/nav/nav_technique/nav_technique_correlate_all.py
  - DELETED in the cutover (Cardinal Principle #1). NavTechniqueCorrelateAll
    and NavModelCombined go in the same change-set that lands the orchestrator;
    no transitional flag.

src/nav/nav_technique/__init__.py
  - register new techniques
  - TechniqueRegistry discovers subclasses at import time

src/nav/obs/obs_snapshot.py
  - add overflow_fraction_vu(inventory_entry) helper — returns
    fraction of body bounding box outside the sensor, computed from
    u/v_min/_max_unclipped vs data_shape_vu

src/nav/config_files/
  RENAME ALL existing config_NN_*.yaml to config_NNN_*.yaml per the table in
  Part 5. Single mechanical mass-rename; loader handles transparently.
  new config_220_body_shape.yaml    # per-body shape / albedo / atmospheric flags
  new config_510_techniques.yaml    # technique tunables, confidence calibration
  new config_520_features.yaml      # per-feature-type reliability gates
  new config_530_filters.yaml       # default filter parameters
  new config_540_orchestrator.yaml  # ensemble parameters
  config_4N0_inst_*.yaml            # add source_image_filter, noise, and
                                    # mag_offset blocks per-camera

pyproject.toml
  [project.scripts] — existing entries (nav_offset, nav_backplanes,
  nav_create_bundle, nav_mosaic_*, nav_*_cloud_tasks variants,
  nav_create_simulated_image, nav_backplane_viewer) all remain unchanged.
  One new entry:
    nav_feature_inspect = "main.nav_feature_inspect:main"
```

### Documentation to write or update

```
docs/developer_guide_autonomous_nav.rst    NEW: architectural overview, pipeline
                                            from observation to NavResult, worked
                                            examples of feature extraction, sample
                                            decision traces

docs/developer_guide_features.rst          NEW: writing a new FeatureExtractor,
                                            the NavFeature schema, registering a type

docs/developer_guide_techniques.rst        NEW: writing a new NavTechnique,
                                            confidence calibration guidance,
                                            examples

docs/developer_guide_filters.rst           NEW: filter catalog, when to use which.
                                            Mandatory content: when raw-NCC vs
                                            gradient-NCC vs DT-matching is the
                                            right metric (with worked examples);
                                            how the σ clamps in
                                            config_530_filters.yaml interact
                                            with per-image σ derivation; the
                                            three filter-application sites
                                            (source-image pre-filter, per-feature
                                            template+image filter, technique-
                                            internal preprocessing) and their
                                            distinct purposes.

docs/developer_guide_uncertainty.rst       NEW: how uncertainty propagates from
                                            static tables through features through
                                            techniques to NavResult covariance;
                                            rank-deficient cases; combination rules

docs/developer_guide_static_data.rst       NEW: config_220_body_shape and the
                                            per-camera noise / mag-offset
                                            schemas (in config_4N0_inst_*),
                                            how to populate from literature

docs/user_guide_navigation.rst             UPDATE: describe confidence thresholds,
                                            NavResult interpretation, failure
                                            modes and diagnostics

docs/user_guide_metadata_schema.rst        NEW: full _metadata.json schema
                                            including the additive
                                            navigation_result block, the
                                            image_classifier block, the
                                            feature_inventory schema, the
                                            conditional rotation_deg field,
                                            and the per-status_reason
                                            mandatory diagnostic fields.
                                            This is the API contract for every
                                            downstream metadata consumer.
                                            Reproduce the JSON example from
                                            Part 4 verbatim as the canonical
                                            schema reference; doc must not
                                            paraphrase the example.

docs/user_guide_troubleshooting.rst        NEW: maps each StatusReason value
                                            to "what happened" and "what to
                                            do"; maps each image_classifier
                                            flag (partial_dropout, noisy,
                                            alternating_lines, truncated_
                                            readout, ccd_bloom_present) to
                                            its cause and operator action.

docs/user_guide_image_library.rst          NEW: how to add a library entry,
                                            how to set tolerances, the
                                            manual-nav save-as-library
                                            workflow, sidecar schema
                                            reference.

docs/user_guide_migration.rst              NEW: migration guide for
                                            downstream consumers.
                                            _metadata.json field migration
                                            table — what was removed,
                                            added, changed. Critical for
                                            backplane / PDS4 / stats tools.

docs/user_guide_configuration.rst          UPDATE: document the new config
                                            files (220_body_shape, 510_
                                            techniques, 520_features, 530_
                                            filters, 540_orchestrator) and
                                            the per-camera noise: / mag_
                                            offset: blocks added to each
                                            config_4N0_inst_*.yaml.

docs/introduction_overview.rst             UPDATE: rewrite "Single-image pipeline"
                                            section to describe feature+technique+
                                            ensemble architecture; embed a Sphinx-
                                            rendered architecture diagram (mermaid
                                            or graphviz; ASCII text-art is the
                                            fallback, not the source of truth)
                                            showing observation → context →
                                            extractors → reliability gate →
                                            techniques → ensemble → NavResult

docs/developer_guide_reprojection.rst      UPDATE: note that reprojection now
                                            consumes NavResult, not just offset

docs/developer_guide_extending.rst         UPDATE: point new-feature-type and
                                            new-technique authors at the
                                            developer_guide_features /
                                            _techniques pages

docs/developer_guide_orchestrator.rst      NEW: orchestrator decision flow,
                                            two-pass logic, ensemble math
                                            (precision-weighted Kalman,
                                            Mahalanobis grouping at 2σ vs
                                            conflict at low gap, rank-deficient
                                            combine), StatusReason enum.
                                            Heart of the new pipeline.

docs/developer_guide_logging.rst           NEW: INFO/DEBUG/WARNING/ERROR
                                            conventions per Part 12.7;
                                            per-image log structure;
                                            how to add new log lines without
                                            polluting INFO.

docs/developer_guide_cli.rst               NEW: nav_offset, nav_feature_inspect,
                                            nav_backplanes, nav_create_bundle,
                                            nav_mosaic_* and their _cloud_tasks
                                            variants — invocation, flags, exit
                                            codes.

docs/developer_guide_testing.rst           NEW: how to use the image library, how
                                            to add tests for new features or
                                            techniques, regression-baseline
                                            procedure, synthetic-obs fixture
                                            conventions

docs/api/                                   NEW Sphinx automodule directory.
                                            One .rst file per new package:
                                            nav.feature, nav.nav_orchestrator,
                                            nav.nav_technique (overall package
                                            doc), and the new modules under
                                            nav.support (status_reason,
                                            filter_combo, filters, distance_
                                            transform, noise_estimate). Each
                                            uses .. automodule:: with
                                            :members: :undoc-members:
                                            :show-inheritance:.

CHANGELOG.md                                NEW or UPDATE (existing repo
                                            convention checked at impl time):
                                            prominent entry for the cutover
                                            describing the architecture shift,
                                            the deleted classes, the new
                                            metadata schema, and a link to the
                                            user_guide_migration.rst.

README.md                                  UPDATE: new architecture bullet in
                                            overview; new CLI listed.

CLAUDE.md                                  UPDATE the project root file:
                                            - Replace the "Pipeline (single
                                              image)" section with the new
                                              feature → technique → ensemble
                                              flow.
                                            - Add the registry mechanism
                                              (__init_subclass__) and
                                              extension points.
                                            - Add the static-data file
                                              locations (config_220_body_
                                              shape.yaml, per-camera noise:
                                              and mag_offset: blocks).
                                            - Add the StatusReason enum
                                              location and the curator/
                                              metadata schema reference.
                                            - Update the 'Adding a new
                                              instrument' section with the
                                              new noise: / mag_offset:
                                              requirements.

.cursor/rules/feature_extractor_conventions.mdc      NEW: how to write a
                                            NavFeatureExtractor — registration,
                                            NavReliabilityBreakdown,
                                            NavFeatureFlags sum types, what NOT
                                            to do (per-feature image cropping
                                            per Cardinal Principle #2).
                                            Content per Part 0 §71.

.cursor/rules/nav_technique_conventions.mdc          NEW: how to write a
                                            NavTechnique — feasibility report,
                                            confidence calibration spec,
                                            structured NavTechniqueDiagnostics,
                                            CURATOR_FIELDS declaration,
                                            interaction with two-pass
                                            orchestrator. Content per Part 0 §71.

.cursor/rules/static_data_conventions.mdc            NEW: schema reference for
                                            config_220_body_shape.yaml,
                                            per-camera noise: / mag_offset:
                                            blocks, and the fallback rules
                                            for missing entries (10% radius
                                            default, WARNING log, reliability
                                            cap 0.3). Content per Part 0 §71.

Documentation organization: the existing docs follow a guide structure (user_guide_*
for operators / downstream users; developer_guide_* for contributors). New pages
fit the same convention — user_guide_* are reference + how-to for people running
the pipeline; developer_guide_* are explanation + reference for people changing
the code. No tutorial pages added in the cutover; tutorials are out of scope but
the existing introduction_overview.rst update covers orientation.
```

---

## Part 9 — Test strategy

### Methodology — TDD throughout

Every phase is executed test-first per `.cursor/rules/python_best_practices.mdc`:

1. Read the spec (this plan + the docstring being implemented).
2. Write the test. Required test-construction discipline:
   - Type-annotated `def test_xxx(...) -> None:` signatures.
   - Exact-value assertions wherever the expected value is known.
   - **Parametrize** for inputs that exercise the same code path with different data; each "bullet" in the per-module test lists below is a *parametrize case*, not a separate test function.
   - **Combine** tests that hit the same code path but assert on different parts of the result; one test per code path.
   - **Every** `pytest.raises(..., match=...)` asserts on the exception **message content**, not just the type. Apply uniformly across all tests, not only technique tests.
   - One logical assertion per `assert` (no `and` in assertions).
   - All tests independent — runnable in any order under `pytest -n auto --dist=loadfile` (the project's `--dist=loadfile` is required because PyQt6 workers crash when split by-test).
   - Test fixtures restore any mutated global state (config singleton, SPICE kernel pool, matplotlib state) via `try` / `finally` or `yield`-and-teardown fixtures.
3. **Inner loop**: run the failing test. Then implement minimum code to pass. Then run `mypy <changed files>` and `ruff check <changed files>` *before* re-running the test (cursor rule §5: "ALWAYS run `mypy` on the full codebase (including tests) after changes"). Then run the test green. Then refactor. Then re-run mypy / ruff / pytest.
4. After the test passes, run `.cursor/skills/python-codebase-analysis/SKILL.md` and `.cursor/skills/critique-test-suite/SKILL.md` over the new code and tests respectively. Adjust before moving on. Specifically check:
   - Module size < 1000 lines; split if larger.
   - Tests parametrized rather than copy-pasted across similar inputs.
   - Exception tests assert on message content, not just type.
   - Assertions are exact-value where the value is known.
   - No global mutable state in tests; fixtures clean up.

**Coverage target**: at least 90% line coverage measured over the *full* test suite (not a subset), per cursor rule §7. CI fails the merge if coverage drops below 90%. Skipping rare exception paths is acceptable; skipping a meaningful branch is not.

**No inline imports in new code**: cursor rule §2 forbids them except for heavy optional GUI dependencies. The PyQt6 manual-nav dialog already does this; new feature / extractor / technique modules use top-of-file imports only.

This methodology is mandatory per the cursor rules; it isn't optional polish at the end of each phase. New behavior added without a failing test first is a process bug.

### Unit test coverage (per-module, in `tests/nav/...`)

**Features and extractors** (`tests/nav/feature/`):
- `test_feature.py`: NavFeature dataclass invariants, covariance positivity, reliability in [0,1].
- `test_star_extractor.py`: Predicted-SNR computation on synthetic known-magnitude stars at known positions; zero-features case when stars predicted below threshold; exclusion of stars behind bodies.
- `test_body_extractors.py`:
  - Regular body fully in FOV → `BODY_DISC` + `LIMB_ARC` + optional `TERMINATOR_ARC`.
  - Regular body partially off-frame → `LIMB_ARC` with correct visible-arc fraction.
  - Irregular body → `BODY_BLOB`, no limb.
  - Body below resolution threshold → `BODY_BLOB`.
  - Per-body uncertainty pulled from `config_220_body_shape.yaml`; fallback to category default when body missing.
- `test_ring_extractors.py`:
  - All Saturn edges visible → per-edge `RING_EDGE`.
  - Flat-projection ring → `straight_line` flag set, rank-deficient covariance.
  - Low-resolution ring system → `RING_ANNULUS`.
  - Planet shadow occludes part of ring → reduced visible-arc, not rejected.
- `test_reliability.py`: Reliability score composition; gating behavior at thresholds.

**Filters** (`tests/nav/support/test_filters.py`):
- Each filter kind: identity inputs, known-output synthetic inputs (step, delta, gaussian).
- Anisotropic gaussian with aligned axis: verify blur extents along principal axes.
- Distance transform: planted edge points, verify DT values at known distances.
- Apply filter + inverse regenerates approximately the input for invertible filters.

**Techniques** (`tests/nav/nav_technique/`):
- `test_star_field_nav.py`: 3 / 5 / 10 stars with planted offsets; confidence scaling.
- `test_body_disc_correlate.py`: replicates and extends current pyramid tests.
- `test_body_limb_nav.py`:
  - Full limb visible → recovers offset.
  - Half limb visible (body partly off-frame, C0061084700R synthetic analog) → recovers offset.
  - Straight-line "limb" (degenerate) → high covariance along the line, technique emits low confidence.
- `test_body_blob_nav.py`: planted centroid, recovered ≤ expected precision.
- `test_ring_edge_nav.py`:
  - Single curved edge → correct offset.
  - All-flat ring edges (parallel to each other, since they share one ring-plane normal) → rank-1 covariance reflecting the 1-D radial constraint; the test asserts `sigma_along_unobservable_px` is set and the result is consistent with the radial-only resolution.
  - One curved edge → full-rank covariance, 2-D constraint.
- `test_ring_annulus_nav.py`: multi-ring region correlation.

**Orchestrator + ensemble** (`tests/nav/nav_orchestrator/`):
- `test_feasibility.py`: feasibility decisions across feature-set permutations.
- `test_ensemble.py`:
  - Agreement: 2 techniques agree within covariance → combined confidence > individuals.
  - Disagreement: 2 high-confidence results disagree → returns 'conflicted'.
  - Spurious filtering: spurious results never chosen.
  - Empty input: returns 'failed' with reason.
  - Rank-deficient + full-rank combination: covariances compose correctly. Specific properties checked: (a) `rank(combined_cov) ≥ max(ranks)`; (b) eigenvectors of `combined_cov` cover the union of input column spaces; (c) precision-weighted-mean position lies on the constraint manifold of every contributing rank-deficient input.
  - `pinvh` rcond behavior: synthetic 2×2 with condition number 1e10 → combined output's null direction has σ ≥ 1e4 (correctly rank-deficient); same input with rcond=1e-15 → silent corruption (negative test).
  - Two Mahalanobis thresholds: synthetic groups at distance 1.5σ end up grouped; at 3.0σ end up separate; distinct `agreement_gap` produces distinct conflicted vs ok behavior.
- `test_confidence_calibration.py`:
  - `evaluate_sigmoid_combination` over every supported normalization (cap, divisor, offset), every term combination, hard-zero gates, hard-cap.
  - Parametrized over each technique's spec from `config_510_techniques.yaml`; for each technique, a synthetic diagnostics object yields a confidence in [0, 1].
- `test_curator.py`:
  - For every `StatusReason` value, build a synthetic `NavResult` and assert the curator emits exactly the mandatory diagnostic fields (failure modes that promise `image_classifier` block, etc.). This is the test that enforces the "diagnostic fields per status_reason" contract — driven from a single source-of-truth dict in `nav_result.py`.
  - Backwards-compat test: synthetic `NavResult` → curator → JSON has top-level `status`, `observation`, `spice_kernels`, `models`, `navigation_techniques`, `offset`, `confidence` keys with their pre-cutover semantics.
  - `image_classifier` block always present on every result regardless of status (asserted across all 15 status_reasons).
  - `provenance.static_data_hashes` present, contains hashes for every `config_220_body_shape.yaml` + `config_3N0_*_rings.yaml` + `config_4N0_inst_*.yaml`.
- `test_orchestrator.py`: end-to-end with mocked obs.
- `test_registry.py`:
  - Asserts `NavTechnique._registry` contains every concrete subclass after importing `nav.nav_technique`. Iterates a hardcoded expected-name list (`StarFieldFromCatalogNav`, `StarUniqueMatchNav`, `StarRefineNav`, `BodyDiscCorrelateNav`, `BodyLimbNav`, `BodyTerminatorNav`, `BodyBlobNav`, `RingEdgeNav`, `RingAnnulusNav`, `CartographicNav`, `TitanNav`); a missing name means a module wasn't imported in `__init__.py` — the registry "found nothing" failure mode.
  - Same for `FeatureExtractor._registry`.
- `test_status_reason_completeness.py`:
  - Iterates every `StatusReason` value and asserts (a) it has an entry in the orchestrator's `STATUS_REASON_INFO_TEMPLATE` dict; (b) it has an entry in `assert_diagnostic_fields_present`'s mapping. New `StatusReason` values without templates / field maps fail the test.

**Property-based tests** (`tests/nav/feature/test_feature_properties.py`):
- Using `hypothesis`: `NavFeature.position_cov_px` must be 2×2 (or 3×3 with rotation) symmetric positive-semidefinite. Test generates random covariances and asserts the constructor accepts only valid ones (and `pytest.raises(ValueError, match=...)` for invalid).
- `evaluate_sigmoid_combination` output is in [0, 1] for arbitrary input (no NaN, no out-of-range).
- `_combine_precision_weighted` is order-invariant: shuffling the input list produces an output within numerical tolerance.

**Synthetic obs fixture stability** (`tests/nav/fixtures/test_synthetic_obs_stability.py`):
- Each fixture returns the same `obs.data` array every call. Hash-stability check on the array data; SPICE-pool-state check (no kernels furnished).

**Static data** (`tests/nav/config_files/`):
- `config_220_body_shape.yaml` and the new `noise:` / `mag_offset:` blocks under each `config_4N0_inst_*.yaml` load successfully; all bodies named in `config_210_satellites.yaml` have a `config_220_body_shape.yaml` entry or fallback is chosen; all configured ring edges have entries.
- Per Part 0 §59 (`test_config_220_body_shape.py`, etc.): assert specific values for ≥5 representative bodies (`config.body_shape['MIMAS'].ellipsoid_rms_residual_km == 1.4`), specific ring-edge `rms_km` values for ≥3 ring features per planet, specific per-camera `noise.read_noise_dn` values per instrument. Loud failure when an edit mis-types a key.
- Per Part 0 §16 (`test_confidence_specs.py`): every shipped `techniques: <key>: confidence:` block resolves to real fields on the corresponding `*Diagnostics` dataclass; unknown fields raise `ValueError` at config-load time.
- Per Part 0 §74 (`test_body_shape_citations.py`): every body in `config_220_body_shape.yaml` has a `_sources` mapping; every non-`null` numeric / list field has a non-empty `_sources` entry; no entry contains `TODO` / `FIXME` / `XXX` (case-insensitive); `PLACEHOLDER` is allowed only as a matched pair with a `null` value. Failure messages identify body and field. Same test extends to per-camera `noise:` / `mag_offset:` blocks in `config_4N0_inst_*.yaml` and any new entries in `config_3N0_*_rings.yaml` (existing ring catalog values grandfathered).

**Logging assertions** (`tests/nav/feature/test_logging.py`,
`tests/nav/nav_orchestrator/test_logging.py`) — per Part 0 §50:
- Missing-body fallback in `BodyExtractor` emits WARNING with body name + reason.
- Missing-instrument-noise fallback in `StarFeatureExtractor` emits WARNING.
- Per-technique exception in orchestrator's try/except emits WARNING with technique name and the original exception type.
- All-techniques-spurious in ensemble emits WARNING.
Each assertion checks (a) message substring AND (b) `record.levelno == logging.WARNING` (separate asserts per Part 9 "one logical assertion per assert").

**Snapshot tests** (`tests/nav/nav_orchestrator/test_metadata_snapshot.py`)
— per Part 0 §51: `syrupy` snapshots of the curator's JSON output, one per
`NavStatusReason` value. Snapshots committed under `__snapshots__/`. CI
fails on stale snapshots without `pytest --snapshot-update` in the same
PR.

**Log-structure assertion** (`tests/nav/nav_orchestrator/test_log_structure.py`)
— per Part 0 §58: orchestrator runs under `caplog.at_level(logging.INFO)`
once per status_reason; the captured INFO-line sequence matches
`STATUS_REASON_INFO_TEMPLATE[status_reason]` line-by-line (substring match
in declared order).

**Conformance tests** (`tests/nav/support/test_filters_conformance.py`,
`tests/nav/nav_technique/test_confidence_conformance.py`) — per Part 0 §60:
locks the spec for every shared helper. `apply_filter` identity / null
behavior; `evaluate_sigmoid_combination` `ValueError` with bad field name;
order-invariance / idempotency / identity properties.

**Property-based tests** (extended per Part 0 §53): in addition to
`test_feature_properties.py`, add `tests/nav/nav_orchestrator/test_ensemble_properties.py`
covering `_agreement_groups` order-independence, `_combine_precision_weighted`
order-independence within tolerance, DT fitter identity-offset
recovery, and RANSAC inlier recovery under noise.

**Backplane query smoke tests** (`tests/nav/feature/test_backplane_queries.py`)
— per Part 0 §19: each named query returns finite values on the
synthetic-obs fixtures; no `NaN`, no `inf`. Verifies the Part 5
"Backplane query reference" mapping is correctly wired.

**Safe-YAML test** (`tests/nav/support/test_yaml_safe.py`) — per Part 0 §67:
the sidecar reader rejects YAML containing `!!python/object` tags with a
`yaml.constructor.ConstructorError` (i.e., uses `yaml.safe_load`, not
`yaml.load`).

### Integration tests (against the image library — Part 10)

`tests/integration/test_autonomous_nav.py`:
- Parametrized per library image; asserts the reported offset lies within each image's annotated tolerance; asserts confidence is at or above the expected tier.
- Asserts that no-result cases (e.g., irregular-body-only with no other features) return `failed` rather than silently bogus offset.

### Synthetic obs fixtures

Unit tests for extractors and techniques rely on synthetic `ObsSnapshotInst` instances rather than real PDS3 images, so they run without `PDS3_HOLDINGS_DIR`. Conventions:

- **Location**: `tests/nav/fixtures/synthetic_obs.py`. Every fixture is a `pytest.fixture` returning a fully-populated `ObsSnapshotInst` subclass (e.g. `ObsCassiniISS`).
- **Construction**: built from oops simulator primitives (`oops.obs.Snapshot` plus a synthetic camera FOV) — no SPICE coverage required because the simulator can compose the required fields directly. Parametrize over the four supported instruments.
- **Image content**: a numpy array constructed by the fixture; deterministic per fixture so tests are reproducible. Saturation peaks, missing-data markers, cosmic-ray spikes, and stray-light gradients are toggleable by fixture parameters.
- **Toggle inventory (per Part 0 §55)**: the parametrized base fixture exposes
  `mission` (default `'COISS_NAC'`), `image_shape_vu` (`(1024, 1024)`),
  `psf_sigma_px` (`1.0`), `add_noise` (`True`), `noise_sigma_dn` (`2.0`),
  `add_smear` (`False`), `smear_vector_vu` (`(0.0, 0.0)`),
  `add_saturated_pixels` (`False`), `add_cosmic_rays` (`False`),
  `add_missing_data_pixels` (`False`), `add_stray_light_gradient` (`False`),
  `add_alternating_lines` (`False`), `add_ccd_bloom` (`False`),
  `body_in_fov` (None or `BodyFixture(name=..., center_vu=..., radii_km=...)`),
  `stars_in_fov` (empty list), `ring_system` (None). Per-test fixtures
  override only the toggles they care about.
- **Cleanup**: each fixture is `yield`-and-teardown style. Mutated globals (config singleton, `oops` global precision, matplotlib state if the test renders) are reset in the teardown clause. Tests must not leave `kernel_furnsh` state behind — no real kernels are loaded, but if a test mock-furnishes a kernel for SPICE-API testing it must `kernel_unload` in teardown.
- **Reuse**: fixtures composed via `pytest.fixture` dependencies; small fixtures combine into scenario-specific fixtures (e.g. `obs_with_one_bright_star`, `obs_with_full_fov_body`).

Tests that exercise real holdings-dependent code paths (catalog lookup over the network, real kernel data) are integration tests under `tests/integration/`, not unit tests, and respect the `PDS3_HOLDINGS_DIR` skip rule via the single `pds3_holdings_dir` session fixture (Part 0 §56).

### Regression baseline

Baseline JSON per library image at `tests/integration/baselines/<image_id>.json` containing `(offset, confidence, per_technique_confidences, chosen_feature_ids, provenance.static_data_hashes)`; CI rejects PRs that change these without an explicit baseline update. Prevents silent quality drift. The `static_data_hashes` field makes the comparison meaningful — when static data changes, the test reports "static data changed; rebaseline required" rather than a misleading regression failure.

**Float precision policy (Part 0 §17).** Baseline files store `round(offset_dv_px, 4)`, `round(offset_du_px, 4)`, `round(confidence, 3)`, and `round(per_technique_confidences[name], 3)`; comparison is exact-equal on the rounded values. 4 decimals (~0.0001 px) sits well under the per-image tolerance budget; 3 decimals on confidence matches tier-boundary granularity. `provenance.pipeline_run_iso8601` is **stripped** from both sides of the comparison (Part 0 §11) since wall-clock time is not part of the byte-identical contract.

---

## Part 10 — Test image library

### Design goal

Small (~50 images total), high-information-density, covering the space of scene types. Each image ships with:
- The PDS3 .IMG (or a reference to it via PDS3_HOLDINGS_DIR).
- A YAML sidecar with metadata (see below).
- An operator-verified ground-truth offset.

### Library structure (under `tests/integration/image_library/`)

```
image_library/
  images/                             # directory tree IS the registry; no manifest file
    star_dominated/                   # 4: StarFieldFromCatalogNav primary
    body_full_fov/                    # 3: BodyDiscCorrelateNav primary
    body_partial_overflow/            # 3: BodyDiscCorrelateNav (gradient mode)
    body_mostly_offscreen/            # 4: BodyLimbNav primary (C0061084700R-class)
    body_irregular/                   # 3: BodyBlobNav primary
    multi_body/                       # 3: multi-feature joint fit
    ring_only_curved/                 # 3: RingEdgeNav full 2-D
    ring_only_flat/                   # 3: RingEdgeNav rank-1 → rank_1_only status
    ring_plus_body/                   # 3: ensemble (rings + body)
    stars_plus_body/                  # 3: ensemble (body + 3+ stars)
    one_bright_star_no_body/          # 2: StarUniqueMatchNav 1-star primary
    two_bright_stars_no_body/         # 2: StarUniqueMatchNav 2-star primary
    faint_stars/                      # 2: stars below usable SNR
    scattered_light/                  # 2: GOSSI/VGISS stray-light → tests DoG bandpass
    high_phase_terminator/            # 2: BodyTerminatorNav primary
    below_resolution_body/            # 2: BodyBlobNav (resolution gate)
    negative_cases/                   # 3: expected.status='failed'
                                      # Total: 49.
```

### Coverage matrix — every technique exercises on ≥1 image

| Technique | Wins primary on | Min images |
|---|---|---|
| `StarFieldFromCatalogNav` | `star_dominated`, `stars_plus_body` | 4+ |
| `StarUniqueMatchNav` (1-star) | `one_bright_star_no_body`; also contributes when 1 star is in `stars_plus_body` | 2+ |
| `StarUniqueMatchNav` (2-star) | `two_bright_stars_no_body` | 2+ |
| `StarRefineNav` (pass 2) | refines on top of body/ring prior in `stars_plus_body` | (overlap) |
| `BodyDiscCorrelateNav` | `body_full_fov`, `body_partial_overflow` | 6+ |
| `BodyLimbNav` | `body_mostly_offscreen`, `ring_plus_body` (partial body) | 4+ |
| `BodyTerminatorNav` | `high_phase_terminator` | 2+ |
| `BodyBlobNav` | `body_irregular` (close), `below_resolution_body` | 3+ |
| `RingEdgeNav` | `ring_only_curved`, `ring_only_flat`, `ring_plus_body` | 4+ |
| `RingAnnulusNav` | `ring_only_flat` (low-res), distant ring scenes | 2+ |
| `CartographicNav` | (Part 13b — calibration deferred) | 0 |
| `TitanNav` | (Part 13b — algorithm deferred) | 0 |

### Per-class selection criteria

| Class | Geometric requirement | Mission spread | Common gotchas |
|---|---|---|---|
| `star_dominated` | ≥3 catalog stars predicted detectable in extfov; no body silhouette | ≥1 Cassini + ≥1 NHLORRI | smear > 30 px; saturated bloom across frame |
| `body_full_fov` | regular body ≥ 70% of FOV; limb fully in frame; ≥30% lit | COISS, GOSSI, NHLORRI | terminator-heavy crescent (use `high_phase_terminator`) |
| `body_partial_overflow` | body 70–90% in frame; limb arc visible >30% | Cassini close-encounter, Galileo flyby | <50% in-frame (use `body_mostly_offscreen`) |
| `body_mostly_offscreen` | body 50–90% off-frame; limb arc ≥10% in FOV | Cassini, Galileo | no limb at all in FOV |
| `body_irregular` | irregular body at range where `limb_uncertainty_px > 3` (BLOB regime) | Cassini close approaches | body so close that BLOB centroid is also ambiguous |
| `multi_body` | ≥2 separable bodies in FOV; not occluding | Cassini Saturn, Galileo Jupiter | bodies overlapping (occlusion tested separately) |
| `ring_only_curved` | polyline max-deviation > 0.5 px from straight-line; no bodies | Cassini Saturn | bodies in FOV (use `ring_plus_body`) |
| `ring_only_flat` | polyline curvature < 0.5 px | Cassini Saturn | curved enough to hit full-rank (defeats rank-1 test) |
| `ring_plus_body` | rings + ≥1 moon in FOV | Cassini | edge-on rings + body (harder to characterize) |
| `stars_plus_body` | body + ≥3 visible stars | Cassini, NHLORRI | accidentally also a `multi_body` |
| `one_bright_star_no_body` | catalog yields 1 unambiguous star (next-brightest ≥ 1.5 mag fainter); no body, no rings | Cassini, NHLORRI star-cal frames | next-brightest within 1.5 mag |
| `two_bright_stars_no_body` | catalog yields 2 unambiguous stars; no body, no rings | Cassini, NHLORRI | saturated/faint pair → assignment ambiguous |
| `faint_stars` | predicted SNR < 3.0 for every catalog star in FOV | GOSSI / VGISS | accidentally-clean frame where stars do show |
| `scattered_light` | GOSSI/VGISS with visible stray-light gradient | GOSSI MOON, VGISS Saturn-encounter outer-leg | already-flat frame |
| `high_phase_terminator` | crescent body, phase > 90° | Cassini, Galileo | crescent so thin no terminator pixels above noise |
| `below_resolution_body` | body diameter < 15 px | any (distant body) | body so distant BLOB centroid is sub-noise |
| `negative_cases` | unnavigable: distant tiny body + sensor-limited stars; empty interplanetary frame; majority-dropout image | spread across missions | scene that *barely* navigates (use `low` confidence_tier instead of `failed`) |

Scene-class boundaries are fuzzy by design. When a candidate sits between two classes, pick the one that exercises the *expected primary technique*. Document the judgment in the sidecar `notes`.

### Per-image sidecar schema (schema_version 1)

```yaml
# images/body_partial_overflow/C0061085400R.yaml
schema_version: 1
image_id: C0061085400R
mission: GOSSI                        # CASSINI_ISS | VOYAGER_ISS | GOSSI | NHLORRI
camera: SSI                           # NAC | WAC | SSI | NA | WA | LORRI
filter_combo: 'CL+CL'                 # canonicalized: filters sorted, '+'-joined
                                      # (same canonicalization used by mag_offset lookup)
image_url: pds3://volumes/GO_0xxx/GO_0004/MOON/C0061085400R.IMG
                                      # opaque URL; resolved via PDS3_HOLDINGS_DIR

scene_tags: [body_partial_overflow, moon, uniform_lit]
                                      # first tag is primary class; must match
                                      # the directory the file lives in.

ground_truth:
  offset_dv_px: 299.0
  offset_du_px: -131.0
  offset_uncertainty_px: 2.0          # 1σ marginal; the test's tolerance budget
  source: operator_verified           # operator_verified — every sidecar's
                                      # offset comes from manually navigating
                                      # *that* image; cross-image inference is
                                      # forbidden (Part 0 §40).
  operator: rfrench                   # provenance — required
  verified_date: 2026-04-26
  ui_version: 'rms-nav 0.X.Y'
  notes: |
    Moon fills FOV, top ~90 px off frame; verified by overlaying limb fit.

expected:
  status: ok                          # ok | failed | conflicted
  confidence_tier: high               # high | medium | low | failed
  primary_technique: BodyDiscCorrelateNav
  techniques_must_run: [BodyDiscCorrelateNav, BodyLimbNav]
                                      # techniques expected feasible — losing one is a
                                      # regression in feasibility logic.
  techniques_must_skip: [StarFieldFromCatalogNav]
                                      # techniques expected infeasible — emitting a
                                      # result is also a regression.

camera_rotation_expected:             # OPTIONAL; only when fit_camera_rotation is on
  rotation_deg: null
  uncertainty_deg: null
```

Schema validated at sidecar load (pydantic-style) so malformed entries break the build.

**Tolerance regimes:**

| Source | Typical `offset_uncertainty_px` | Use when |
|---|---|---|
| `operator_verified`, sharp limb / bright stars | 1.0 | majority of cases |
| `operator_verified`, soft features / star-poor | 2.0 | high-phase, soft-edge ring, faint-star fields |

CI test tolerance = `offset_uncertainty_px + 0.5 px` slack. Slack absorbs algorithm-version-level pixel jitter without false-failing. `confidence_tier` mismatches always fail (no slack — tier is part of the calibration target).

**Ground-truth update rules**: any change to `ground_truth.offset_*` re-stamps `verified_date` + `operator`. Sidecar values are not auto-updated by pipeline runs; widening tolerance or moving ground truth requires explicit operator review on the PR.

**Cross-image inference is forbidden** (Part 0 §40 / Cardinal Principle #3): there is no helper that derives one image's ground truth from another's offset.  Spacecraft attitude drift is non-linear at the time-scales between exposures (thruster firings, momentum-wheel desaturations, sub-second jitter), so any "between two anchors" interpolation is unsafe at pixel precision.  Every sidecar's `offset_dv_px` / `offset_du_px` is the result of an operator manually navigating *that* image.

**CI integration**: two test files, both under `tests/integration/`.

- `test_autonomous_nav.py` — per-image regression. `pytest_generate_tests` hook discovers sidecars by directory glob (`image_library/images/*/*.yaml`), parametrizes one test per sidecar with the `image_id` as the test ID. Each test runs the orchestrator, asserts (a) `status` exact-match, (b) `confidence_rank` exact-match (no slack), (c) `offset_px` within `offset_uncertainty_px + 0.5 px` slack for `ok` results, (d) `expected.primary_technique` is the highest-confidence per-technique result with **ties broken by `(−confidence, technique_name)` ascending** (Part 0 §14; deterministic, registration-order-independent), (e) every name in `techniques_must_run` ran, every name in `techniques_must_skip` did not. Skipped uniformly when `PDS3_HOLDINGS_DIR` is unset.
- `test_image_library.py` — library structural invariants (does not touch holdings; runs in every dev env). Asserts: every sidecar passes pydantic validation; primary `scene_tag` matches containing directory; no duplicate `image_id`s; every non-deferred technique appears as `primary_technique` on ≥1 sidecar (coverage matrix); per-class image count meets the Part 10 minimum; the *set* of class subdirectories under `image_library/images/` exactly equals the declared scene-class list (catches typos like `body_overflow` vs `body_partial_overflow` immediately, before any per-sidecar test runs with a confusing error).

**No `manifest.yaml`** — the directory tree IS the registry. Adding a library image = creating a sidecar in the right class directory; CI picks it up automatically. Removing a sidecar stops its test. Drift between manifest and disk is impossible because no manifest exists.

**Regression baselines** (per Part 9) live in a separate `tests/integration/baselines/<image_id>.json` per image and run as a third test layer; baseline updates require explicit PR review (Phase 5 work).

### Library curation policy

- Start with ~50 images selected by the operator from existing mission archives.
- Each scene class gets at least 2 images to detect accidental overfitting to one sample.
- Ground-truth offsets verified by manual navigation in the existing UI.
- Library is read-only once in; updates only when new scene classes emerge.
- Library is kept under `tests/integration/image_library/` but images themselves stay on mission holdings (URLs only); CI downloads via the existing `PDS3_HOLDINGS_DIR` cache.

### Confidence calibration

The confidence formulas in each technique are tuned once on this library: the constants are chosen so that the per-tier mappings in Part 4 hold empirically on the library. This is the single calibration pass; after that, the constants are frozen. Re-calibration happens only if technique code changes substantially.

---

## Part 11 — Manual research vs. AI-automated research

### Manual research (one-time, by a human)

- **Body shape table (`config_220_body_shape.yaml`)**: ellipsoid residuals, crater scales, albedo values for **all relatively large satellites of the four giants + Mars moons + Earth's Moon + Pluto system + planets** (~55 bodies). No comets / small asteroids. Sources: IAU working group on cartographic coordinates; Thomas et al. shape papers; JPL planetary physical parameters. **Per Part 0 §74, every numeric value carries an accurate non-fabricated `_sources` citation** with DOI where available; AI agents drafting this YAML cite only documents they have actually fetched in-session, never from training-data memory. Each PR adding ≤10 bodies; reviewer spot-checks ≥5 citations per PR. Estimated effort: 2–3 days (the citation-verification burden roughly doubles the original 1–2 day estimate, but the resulting trust is what makes the table usable downstream).
- **Ring edge catalog — *mostly already done***: the existing per-planet `config_3N0_*_rings.yaml` catalogs already carry per-feature `rms` (radial uncertainty), orbital-mode decomposition, and feature type. Only two minimal additions needed: optional `default_reliability` override per feature and optional `sharpness` tag where the default (from `rms`) is wrong. Estimated effort: ≤ 1 day of review; no new data to hunt down.
- **Instrument noise models** (`noise:` block under each camera section of `config_4N0_inst_*.yaml`): radiometric characterization values (read noise, dark current, full well, gain) for every mission/camera. Sources: PDS calibration reports per mission. Estimated effort: 1 day. **Per-instrument-per-filter `mag_offset_table`** (`mag_offset:` block under each camera section): V-to-in-band magnitude offset keyed by B-V color bin per filter combo. Sources: CISS Calibration Report, GOSSI / VGISS / NHLORRI calibration documentation. Estimated effort: 2–3 days (long pole; AI-drafted, human-reviewed).
- **Image library curation**: ~50 operator-verified images with ground-truth offsets. Estimated effort: 2–3 days (most of it is the manual nav for ground truth, which is the already-existing UI).

### AI-automatable research (with appropriate guardrails)

- **Initial population of the body/ring tables from PDS / literature**: an agent can fetch the relevant tables from NASA PDS or published papers, extract tabular data, and produce the YAML. Per Part 0 §74, the agent **must cite the exact fetched document for every value** (DOI, table number, identifying clue); citations from training-data memory are forbidden, and any PR found to contain a fabricated citation is reverted in full. Must be **human-reviewed** before merge — uncertainty values directly affect navigation trust; reviewer spot-checks ≥5 citations per PR by opening the cited document.
- **Scene classification for the image library**: given a candidate image + SPICE prediction, an agent can propose a scene_tags list. Human confirms.
- **Confidence formula tuning**: an agent can do the regression on library ground-truth to propose formula constants. Human checks the calibration fits the declared tiers.
- **Documentation generation**: new .rst pages can be drafted by an agent from the code-level docstrings and this plan; human edits for clarity.
- **Test scaffolding**: per-technique test skeletons + parametrized cases drafted by an agent; human writes the tricky numerical assertions.

### What must stay human

- Ground-truth offsets for the library (agents can't verify nav correctness; a human eye verifying the overlay is the only reliable source).
- Acceptance of any change to uncertainty defaults (downstream safety-critical).
- Acceptance of confidence-tier thresholds (downstream users of `NavResult` rely on them).

---

## Part 12 — Other critical design aspects

### 12.1 — Deterministic invocation (for reproducibility + debugging)

Each `NavResult` includes a `provenance` block. Two navigations with identical inputs produce byte-identical `NavResult.provenance` **except for `pipeline_run_iso8601`** (Part 0 §11), which is wall-clock by construction; regression-baseline comparison strips that field before comparing. Concrete schema:

```python
# src/nav/nav_orchestrator/nav_result.py
@dataclass(frozen=True)
class Provenance:
    rms_nav_version: str                  # __version__ string (e.g. '0.5.2')
    rms_nav_git_sha: str | None           # 'dirty' or short SHA; from git or pyproject
    spice_kernels: list[str]              # sorted kernel filenames actually loaded
                                          # (from spice.ktotal()/kdata())
    spice_kernel_count: int               # convenience; equals len(spice_kernels)
    # NOTE per Part 0 §11: `pipeline_run_iso8601` below is EXCLUDED from the
    # byte-identical-for-identical-inputs guarantee. Regression-baseline
    # comparison strips that field before comparing.
    static_data_hashes: dict[str, str]    # filename → sha256 of RAW yaml file
                                          # bytes (Part 0 §18; comments and
                                          # whitespace included) for
                                          # config_220_body_shape.yaml, every
                                          # config_3N0_*_rings.yaml, every
                                          # config_4N0_inst_*.yaml. Used by
                                          # regression tests to detect "static
                                          # data changed; baseline must update".
                                          # Implementation: hashlib.sha256(
                                          #   path.read_bytes()).hexdigest().
    technique_names: list[str]            # registered techniques (sorted) — implicit
                                          # version tag via name change on incompatible
                                          # behavior change
    extractor_names: list[str]            # registered extractors (sorted)
    image_et: float                       # the obs's ET; placed here for grep-ability
    pipeline_run_iso8601: str             # UTC; populated by the orchestrator
```

The `static_data_hashes` field is what makes the per-image regression baselines (Part 9) self-validating: a baseline-vs-fresh-result comparison is meaningful only when the inputs match; if hashes differ, the test reports "static data changed; rebaseline" rather than "regression detected".

### 12.2 — Error budget accounting

`NavResult.covariance_px2` is the **total** covariance — it includes:
- Per-technique measurement uncertainty (from the matching algorithm's intrinsic precision).
- Per-feature position uncertainty (from static tables, image-plane-projected).
- Combination uncertainty (from ensemble disagreement if any).

The orchestrator does not silently inflate or deflate confidence. Every contribution is diagnostic-traceable.

### 12.3 — Cost control

Running every technique on every image is expensive. Mitigations:
- **Feasibility gates are cheap**: they read feature metadata, not pixels. Infeasible techniques skip before heavy compute.
- **Resolution gates** force body features to blob-only below ~15 px, pre-empting expensive correlation / DT on under-sampled data.
- **Early termination**: orchestrator can short-circuit if `StarFieldFromCatalogNav` returns confidence > 0.95 — no need to run body techniques when stars pin it to 0.1 px. Config-flagged because it breaks ensemble cross-validation; off by default.

### 12.4 — SPICE assumptions

The whole purpose of this pipeline is to correct the spacecraft attitude (pointing), so the design **must not** assume SPICE pointing is reliable. Every technique either (a) is position-invariant (pattern-matching), (b) gets a prior offset from a position-invariant technique, or (c) succeeds at any pointing error within the configured search window (`obs.extfov_margin_vu`). Body and ring *world* positions from SPICE are normally good — those are the SPK/PCK kernels — but the spacecraft attitude in CK kernels is the source of truth we are *correcting*, not consuming. Predicted feature pixel positions in the image are therefore treated only as the center of a search window of size `extfov_margin`, not as a tight prior, and can be wrong by the entire margin. The image library (Part 10) deliberately includes images where the SPICE-predicted body position and the actual image position differ by hundreds of pixels.

### 12.5 — Cutover (no backwards compatibility)

Per Cardinal Principle #1: the new pipeline is a complete replacement.

- `NavMaster` either becomes a thin facade that immediately delegates to `NavOrchestrator`, or is deleted in favor of callers using `NavOrchestrator` directly. Strong preference for delete.
- `NavTechniqueCorrelateAll` is deleted.
- `NavModelCombined` is deleted.
- `NavModelResult` loses `weighted_mask`, `blur_amount`, `uncertainty`, `confidence` (no deprecation period).
- The `use_legacy_pipeline` flag does not exist.
- Downstream callers (`nav_offset.py`, the cloud_tasks workers, the manual nav UI) are updated in the same change-set that lands the new orchestrator. There is no "release where both pipelines coexist". Any user holding a pinned old version stays on the old version permanently; the new version is the new pipeline only.

What this protects: it keeps the tree from growing two parallel nav systems where both rot. The image library (Part 10) is the safety net — if a regression turns up, fix it in the new pipeline; don't fall back to the old one.

### 12.6 — UI side effects

- Manual-nav dialog: reuse, but replace the current single-"Auto" button with a per-technique-result panel. Each technique's proposal, confidence, and overlay are displayed; operator picks one (or accepts the orchestrator's choice). No correctness cost; big usability win.
- Summary PNG: annotate with contributing features + per-technique offsets, not just the final one.

### 12.7 — Observability and logging conventions

Logged information must match the level it's logged at. The existing pipeline uses `pdslogger.PdsLogger` via `NavBase`; new code follows the same pattern with these conventions:

**INFO level — what an operator running the pipeline wants to see for any image.** One short line per significant event. Concretely:
- "Navigating <image_name>"
- "Extracted <N> features: 5 STAR, 1 BODY_DISC, 3 RING_EDGE" (one summary line per image)
- "Pass 1 ran 4 techniques; best: BodyLimbNav (confidence 0.87)"
- "Pass 2 refined via StarRefineNav: offset (298.96, -130.71) ± (0.41, 0.52)"
- "Final: status=ok, confidence_rank=high, sigma=(0.41, 0.52)"
- Per-technique 1-line summary: "BodyDiscCorrelateNav: offset (300, -132) ± (0.6, 0.7), confidence 0.81, RAW mode, 1 BODY_DISC consumed"

INFO output is what gets archived in the per-image log file alongside the metadata; downstream operators read this when triaging a batch.

**DEBUG level — everything an engineer would need to reproduce a decision.** Verbose; per-feature, per-pyramid-level, per-iteration. Concretely:
- Per-feature reliability score breakdown (which `reliability_reasons` contributed what).
- Per-technique feasibility report and rejection reason.
- For every NCC pyramid level: integer-arg-max location, sub-pixel refinement steps, peak-to-runner-up ratio, edge-of-window flag.
- For every DT iteration: residual statistics, inlier count after Tukey, Levenberg-Marquardt step.
- For ensemble: agreement-group composition, per-group total confidence, pairwise Mahalanobis distances.
- For pattern matcher: number of detected sources, number of triplets hashed, RANSAC inlier counts per candidate transform.

**WARNING level** — recoverable problem worth flagging:
- Static data fallback used (body not in `config_220_body_shape.yaml`, instrument camera missing its `noise:` block in `config_4N0_inst_*.yaml`).
- Technique threw an exception and was skipped.
- All techniques returned spurious; final result is `failed`.

**ERROR level** — exceptional faults that prevent navigation entirely (corrupt image, missing kernel). The orchestrator never raises through to the caller; everything is captured as a structured `NavResult`.

**No custom base exception class** (per Part 0 §64). Because every per-technique exception is captured by the orchestrator's per-technique try/except and routed through `NavResult(status='failed', status_reason=...)`, no exception propagates to callers. Therefore no `NavError` base class exists; callers check `nav_result.status` and `nav_result.status_reason` instead of using `try/except`. Reviewers who notice the absence of an exception hierarchy should refer to this paragraph rather than introduce one.

Per-image log file output uses the existing `IMAGE_LOGGER.open(...)` context manager pattern; the orchestrator is the one image-level scope, with techniques as nested `with self.logger.open('TECHNIQUE_NAME')` blocks so the section structure in the log matches the technique structure in the result. Every new class inherits from `NavBase` to get the `_logger` / `_config` plumbing for free.

`NavResult` JSON written to `_metadata.json` is the structured log; the human-readable log file is the textual log. Both must be present.

`nav_feature_inspect` (Part 8) is a diagnostic CLI: runs only the extractor + reliability gate, dumps features + reasons to stdout. Useful when triaging "why didn't technique X get invoked on image Y" without paying for full navigation.

### 12.8a — `oops` global state and concurrency

Per `CLAUDE.md`, `RingMosaic.reproject()` mutates `oops` global precision via `_reduced_oops_precision`. The new pipeline introduces additional `oops`-touching paths (every `Backplane` query in extractors). Concurrency rules:

- **Per-image orchestrator runs are single-threaded by default.** No internal threading inside `NavOrchestrator.navigate()`. `oops` mutations within one image's run are sequential and observable only by that run.
- **Cloud-tasks parallelism is multi-process, not multi-threaded.** Each worker is a separate Python process with its own `oops` global state. Cross-process isolation is automatic (Cardinal Principle #3 satisfied structurally).
- **Within-process parallelism (e.g. `concurrent.futures` in a single `nav_offset` invocation handling N images)** is *not* supported by the new pipeline because of the shared `oops` global. If a future need arises, the right fix is to subprocess-isolate per image, not thread-isolate.
- Tests must not invoke navigation in worker threads of the same process. The synthetic-obs fixtures don't touch the real `oops` global so are safe; integration tests with real obs use `pytest -n auto` which is process-parallel via `pytest-xdist` and is therefore safe.

### 12.8 — Safety on unknown scenes

If an image shows a body that's not in `config_220_body_shape.yaml`, extractor uses the most conservative category default (residual ≈ 10% of r). If a ring's planet is unknown, rings extractor emits zero features. The system *never* invents uncertainty numbers out of the air.

---

## Part 13 — Phasing (implementation order)

Order matters — each phase is self-contained, follows TDD throughout (Part 9), and leaves the tree green. Each phase ends with `ruff check && ruff format --check && mypy && pytest -n auto --dist=loadfile` clean before merging.

**Phase 1: Foundations (~1 week).** Types and plumbing only; no behavior change.
- *Tests first* for: `NavFeature` invariants, `NavFilterSpec` construction, `NavTechniqueResult` round-trip, `NavContext` field requirements, static-data loader for missing-body fallback and missing-instrument warning. Use parametrize for the per-type variants.
- Implement: `NavFeature`, `NavFilterSpec`, `NavTechniqueResult`, `NavContext`, `NavResult`, `NavFeatureExtractor` ABC + registry. Static-data loader. Implement payload sum-type in `src/nav/feature/geometry.py` (`NavFeatureGeometry` per Part 0).
- **`pyproject.toml` `[tool.pytest.ini_options]` updates (mandatory)**: add `testpaths = ["tests"]`, `--strict-markers` and `--strict-config` to `addopts`, register `markers = ["integration: requires PDS3_HOLDINGS_DIR", "slow: per-image library regression"]`, and set `filterwarnings = ["error"]` so unexpected warnings fail the build (per `critique-test-suite` §19, §22; verified absent in baseline pyproject.toml).
- **Manual artifacts needed at the start of Phase 1**: `config_220_body_shape.yaml`, the `noise:` / `mag_offset:` block additions to each `config_4N0_inst_*.yaml`, and the `default_reliability` / `sharpness` extensions to the existing ring catalogs. Without them the loader has nothing to load. Required scope: ~55 body entries (Part 5 list); all four mission instrument-noise blocks; `mag_offset:` populated for the dominant filter combos per camera, with `fallback_combo` for the long tail. **YAML population workflow (per Part 0 §74)**: an AI agent drafts entries from PDS calibration reports + Thomas et al. shape papers + IAU physical-parameters tables, **citing only documents fetched in-session via `WebFetch` / `WebSearch`** (never from training-data memory) — every numeric value carries a sibling `_sources` entry with DOI where available. PRs are split into ≤10 bodies each so the human reviewer can spot-check ≥5 citations per PR by opening the cited document and verifying the value appears at the cited location. Fabricated citations are a revert-in-full offense. The `test_body_shape_citations.py` validation test (Part 9) blocks merge on schema gaps; citation accuracy is human-verified.
- No old code touched in this phase.

**Phase 2: Image preprocessing + extractors (~1 week).**
- *Tests first* for: cosmic-ray despeckle (planted hot pixels), saturation mask, smear-aware PSF size, global noise estimate convergence. Then per-extractor tests using synthetic obs fixtures (already covered in Part 9).
- Implement: image-quality preprocessing, `StarFeatureExtractor`, body extractors, ring extractors, cartographic extractor. Reuse existing `RingFeatureFilter`.
- New CLI `nav_feature_inspect` for visual eyeballing — useful for confirming what the extractors produce on real images without running navigation.
- Old pipeline still the navigation path.

**Phase 3: Techniques (~3–4 weeks; pattern matching is the long pole).**
- *Tests first* per technique. For each technique, write tests that: (a) recover a planted offset on a synthetic input; (b) report appropriate confidence at boundary cases (low signal, partial overlap, etc.); (c) detect infeasibility cleanly; (d) raise on invalid input with assert-on-message.
- Implement in this order: `BodyDiscCorrelateNav` (port existing correlation; well-tested baseline), `BodyBlobNav` (cheap), `BodyLimbNav` (DT-based; new heavy algorithm), `BodyTerminatorNav` (small variant of limb), `RingEdgeNav`, `RingAnnulusNav`, `StarRefineNav` (port existing refinement), `CartographicNav` (port existing bootstrap), `TitanNav` (stub — registered but always reports infeasible; real algorithm is a separate future work item).
- `StarFieldFromCatalogNav` (pattern matching from scratch) is its own sub-phase: source detection → triplet hashing → RANSAC → verification. Each piece TDD'd separately.
- After each technique, run the codebase-analysis + critique-test-suite skills; revise before moving on.

**Phase 4: Orchestrator + ensemble (~1 week).**
- *Tests first* for: feasibility filtering, two-pass priority, agreement grouping, conflicted handling, rank-deficient combination. Use mocked `NavTechniqueResult`s so tests don't depend on technique implementations.
- Implement `NavOrchestrator`, `ensemble()`, two-pass driver, `NavResult` JSON serialization (extending the existing `_metadata.json` additively).
- **CLI cutover** (every script in `src/main/` that invokes navigation):
  - `nav_offset.py` → `NavOrchestrator(...).navigate(obs)`.
  - `nav_offset_cloud_tasks.py` → same; per-image task body unchanged in shape.
  - `nav_backplanes.py` and `nav_backplanes_cloud_tasks.py` → read `offset` from the new metadata (already present in additive form).
  - `nav_create_bundle.py` and `nav_create_bundle_cloud_tasks.py` → read `confidence_rank` and refuse to bundle `low` results unless `--include-low-confidence` is set.
  - `nav_create_simulated_image.py` → no change (writes images, doesn't read nav metadata).
  - `nav_mosaic_*.py` and `nav_mosaic_*_cloud_tasks.py` → read `offset` and `confidence_rank` per Part 12.6.
- **CI workflow**: `.github/workflows/run-tests.yml` already sets `PDS3_HOLDINGS_DIR`, `PDS4_HOLDINGS_DIR`, `OOPS_RESOURCES`, `UCAC4_PATH`, `YBSC_PATH`. No new env vars. Add a new job step for `pytest tests/integration/test_image_library.py` (structural invariants — runs without holdings) and `pytest tests/integration/test_autonomous_nav.py -n auto` (per-image regression — runs with holdings).
- Legacy `NavTechniqueCorrelateAll` / `NavModelCombined` / `NavModelResult` are **deleted** in this same change-set (Cardinal Principle #1).

**Phase 5: Image library + calibration (~2 weeks).** *This is when the manually-supplied test images are required.*
- **UI work**: add a "Save as library entry" button to the manual-nav dialog that writes a sidecar with provenance pre-filled. Without it, hand-curating 50 sidecars is a 50-paste ceremony. Small but real Phase 5 work; sits alongside the per-technique-result panel work in Part 12.6.
- **Manual artifact needed**: ~50 PDS3 image identifiers chosen by the operator covering all scene classes (Part 10), plus operator-verified ground-truth offsets via the manual-nav UI. Library cannot be drafted by an AI; ground truth must be a human eye matching the overlay.
- Tune every confidence formula (Part 4) against library ground-truth. The model is `confidence = sigmoid(α₀ + Σ αᵢ × xᵢ)` (logistic in the αᵢ); fitting is via `scipy.optimize.curve_fit` with the sigmoid model, optimizing the αᵢ to minimize `Σ_image (predicted_confidence − target_tier_midpoint)²` where `target_tier_midpoint` is `0.9` for images expected to land in `high`, `0.65` for `medium`, `0.35` for `low`, `0.1` for `failed`. (Equivalent to a logistic regression with continuous targets; not strictly OLS because the model is nonlinear, hence `curve_fit` not `lstsq`.) The placeholder coefficients in Part 4 are arithmetically illustrative only — they are *not* claimed to produce the example confidence values stated alongside them; calibration replaces them. This is a one-time calibration; `config_510_techniques.yaml` is checked in afterward and never updated by per-image runs.
- Set the regression baselines in `tests/integration/baselines/`.

**Phase 6: Documentation (~1 week).**
- All new `.rst` pages (Part 8 file list). Existing `user_guide_*` and `introduction_overview` updated.
- AI agent drafts pages from code docstrings + this plan; human edits for clarity and accuracy.
- Per Part 0 §49: update `docs/index.rst` Sphinx toctree to list every new module sorted by package and name.
- Per Part 0 §48: every new module/class/function carries a Google-style docstring (Parameters / Returns / Raises) detailed enough to write a black-box test from the docstring alone (per `documentation.mdc` §4). Each technique's docstring links to its confidence-formula source-of-truth (`config_510_techniques.yaml.<technique_key>`).
- Per Part 0 §63: thread-safety section added to each extractor and the orchestrator docstrings ("Not safe for concurrent use on the same `obs`...").
- Per Part 0 §54: `developer_guide_testing.rst` declares mocking conventions (`mock.patch` for spies; `monkeypatch` for env / module state) and lists canonical patch targets per shared utility.
- Per Part 0 §57: `developer_guide_testing.rst` declares the `xfail` / `skipif` discipline.

**Phase 7: Cleanup (~3 days).** All deletions happen in Phase 4 (Cardinal Principle #1); Phase 7 is verification only. Per Part 0 §20, the breadth comparison vs legacy already ran pre-merge in Phase 4 — Phase 7 is post-merge cleanup, not gating.
- `grep` sweeps to verify nothing references deleted symbols (`NavTechniqueCorrelateAll`, `NavModelCombined`, `NavModelResult`, `NavMaster`, `weighted_mask`, `blur_amount`, `final_offset`, `final_confidence`, `use_legacy_pipeline`). Per Part 0 §69: implemented as a single-line CI step that fails the build on any match.
- Final regression run on the full library; the 500-image breadth comparison vs legacy already ran pre-merge in Phase 4 (Part 0 §20).
- Coverage report ≥ 90% on the new code.
- `sphinx-build -W -b html` clean.
- Per Part 0 §69: new CI step (single grep line in `.github/workflows/ci.yml`) fails the build on any leftover reference to deleted symbols.
- If the grep sweeps find any leftover references, fix in Phase 7 — but Phase 4 should have left no leftovers.

**Cutover gating — what blocks the legacy delete (per Part 0 §20):**

The legacy code is deleted in Phase 4's change-set per Cardinal Principle #1. **Pre-merge gates run on the feature branch while both pipelines coexist** — the breadth comparison cannot run after merge because legacy is gone by then. Gates that must be green before merging that change-set:

1. Every image in the Phase 5 library navigates to `expected.status` and `expected.confidence_tier`, with offset within `offset_uncertainty_px + 0.5 px` slack.
2. The 500-image breadth comparison (Part 15) shows the new pipeline equal-or-better than the legacy on every aggregate metric (% ok, P50 / P95 offset error, % conflicted, % failed). "Equal-or-better" is per-class, not just overall — a regression on one scene class is a blocker even when overall metrics improve.
3. Coverage ≥ 90% on the new code (cursor rule §7).
4. Every doc page listed in Part 8 written and `sphinx-build -W -b html` clean.

If a Phase 5 image regresses, the response is **fix the new pipeline**, not retain the legacy as a fallback. The image library plus the breadth comparison is the safety net. There is no rollback to the legacy pipeline post-merge (per Cardinal Principle #1); pre-merge, the change-set sits behind these gates.

Total estimated: 2–3 months of focused work, including iteration. The design is structured so that each phase ships useful progress even if the overall project pauses.

---

## Part 13b — Deferred work to file as GitHub issues

When implementation begins, file these as separate tracking issues so they
don't get lost in the cutover. They are out of scope here but the
architecture leaves room for them.

1. **Ring-edge polarity-aware matching.** `RingEdgeNav` uses
   gradient-magnitude-only DT matching, which is correct for any combination
   of gap/ringlet, emission angle, and dust content but loses some robustness
   when polarity *is* known (e.g. A-ring outer at low emission is reliably
   bright-inside / dark-outside). Track: add a `polarity_predictable: bool`
   flag per edge in `config_3N0_*_rings.yaml`, plus an optional dust-content
   indicator. When the flag fires, `RingEdgeNav` switches to signed-gradient
   matching with a per-vertex polarity check. Lay out which edges qualify;
   add a regression test set.

2. **Cartographic-model technique testing.** The architecture supports
   `CARTOGRAPHIC_MODEL` features and `CartographicNav` and the technique is
   implemented + unit-tested against synthetic mosaics in the cutover, but
   the Phase 5 image library does not include cartographic-model test images
   (no production cartographic mosaics exist for the supported missions
   yet). Track: when production mosaics become available, add representative
   cartographic test images to the library, recalibrate the
   `CartographicNav` confidence formula against ground truth, and add the
   technique to the regression baseline.

3. **Atmospheric-body navigation algorithm.** `TitanNav` ships as a stub
   that always returns infeasibility; bodies flagged `atmospheric: true` in
   `config_220_body_shape.yaml` (TITAN, VENUS, TRITON, EARTH) emit no body
   features, so the orchestrator falls through to other features on those
   scenes. The real algorithm needs to handle haze-top "limb" that varies
   with wavelength, surface-invisible interior, and per-filter transmission
   differences. Track: design a haze-aware limb-fit technique with the
   appropriate per-filter haze profile; potentially a separate
   `AtmosphericLimbNav` rather than reusing `BodyLimbNav`. Out of scope
   here because no current production traffic requires it (the four
   supported missions have either no atmospheric bystanders, or treat
   Titan as a primary target with mission-specific manual workflows).

4. **Annotation styling for gated-out features.** Features dropped by the
   reliability gate are still rendered onto the summary PNG (the predicted
   scene is the truthful annotation) but currently rendered identically to
   kept features. Track: decide whether dropped features should render
   muted (lower alpha, dashed outline, gray label) or carry an explicit
   "rejected: <reason>" badge. UI / styling pass; no algorithmic impact.

5. **Mixed-instrument batch SPICE-kernel hot path.** Cloud-tasks workers
   serving images of different instruments back-to-back may pay repeated
   `kernel_furnsh` cost as one mission's kernel set is loaded then
   another's. Performance only — correctness is unaffected. Track: profile
   a real mixed batch and decide whether per-worker mission affinity (route
   like-instrument images to the same worker) is worth the scheduling
   complexity, or whether the SPICE pool can keep multiple mission's CK
   chains resident concurrently without conflict.

6. **UI per-technique-result panel redesign (Part 0 §66).** The
   manual-nav dialog currently surfaces a single "Auto" button; the
   target end-state is a per-technique side-by-side panel showing each
   technique's proposal, confidence, and overlay (operator picks one or
   accepts the orchestrator's choice). Out of scope for the cutover;
   Phase 4 keeps the existing UI working with the new `NavResult` (the
   orchestrator's chosen offset is what gets surfaced). Track:
   ergonomics, interaction model, layout, color coding for confidence
   tiers.

## Part 14 — Open questions — operator decisions

Resolved during review:

1. **Static-data coverage for `config_220_body_shape.yaml`** — *all relatively large satellites* of the four giants + Mars moons + Earth's Moon + planets. Comets and small asteroids are skipped (revisit later if dedicated mission data requires). Irregular satellites **are still included**: they remain navigable at sufficient distance where irregularity projects to < 3 px, so the extractor's per-image uncertainty gate (not a shape class) decides whether a `LIMB_ARC` or `BODY_BLOB` feature is produced.
2. **Ring-edge catalog scope** — *reuse the existing per-planet ring catalogs* (`config_20_jupiter_rings.yaml`, `config_21_saturn_rings.yaml`, `config_22_uranus_rings.yaml`, `config_23_neptune_rings.yaml`). Each per-feature `rms` already encodes radial uncertainty; higher modes already encode dynamical amplitude. Only optional per-feature `default_reliability` / `sharpness` keys added.
3. **Confidence tier thresholds** — 0.8 / 0.5 / 0.2 starting point accepted; re-tuned on the image library during calibration.
4. **"Conflicted" strictness** — groups form at 2σ Mahalanobis (`agreement_sigma`); `conflicted` fires when ≥2 groups exist and the summed-confidence gap between best and runner-up is below `agreement_gap` (default 0.5). Both thresholds live in `config_540_orchestrator.yaml`. See Part 4.
5. **Cloud-tasks parallelism** — the cloud-tasks environment is already multi-process; the navigation pipeline stays single-image-per-task and parallelism is the harness's job. No pipeline changes required.
6. **Legacy retention** — none. `NavTechniqueCorrelateAll`, `NavModelCombined`, `NavModelResult`, `NavMaster` are deleted in the Phase 4 cutover change-set. No `use_legacy_pipeline` flag. Pre-merge gates (Phase 7) protect against regression; no post-merge rollback.

No questions outstanding — ready to implement when scheduled.

---

## Part 15 — Verification

How we know it works:

1. **Unit tests**: ~150 new tests across extractors, filters, techniques, ensemble; existing ~270 tests still pass.
2. **Image library regression**: all ~50 library images navigate within their declared tolerances; confidence tiers match `expected.confidence_tier`.
3. **End-to-end**: run `nav_offset` over a curated 500-image selection (5 missions × 100 images covering scene diversity) and measure:
   - % images with `status: ok`.
   - Per-tier offset-error distribution (median, P95).
   - % `conflicted` results (should be small).
   - Comparison to legacy pipeline on the same 500 images — gate cutover on "new pipeline ≥ legacy on every metric".
4. **Typecheck / lint**: `ruff check src tests`, `ruff format --check`, `mypy src tests` all clean.
5. **Docs build**: `sphinx-build -W -b html docs docs/_build` clean.

---

## Appendix A — Key interface signatures (reference)

```python
# src/nav/feature/feature.py — names per Part 0 renaming
class NavFeatureType(Enum):
    STAR, LIMB_ARC, TERMINATOR_ARC, BODY_DISC, BODY_BLOB,
    RING_EDGE, RING_ANNULUS, TITAN_LIMB, CARTOGRAPHIC_MODEL = ...
    # 9 values total. TITAN_LIMB is reserved for the deferred atmospheric-
    # body work (Part 13b §3); the rest are produced in v1.

class NavFeature: ...  # see Part 1

# src/nav/feature/extractor.py
class NavFeatureExtractor(ABC):
    accepted_subject_types: frozenset[str]
    @abstractmethod
    def is_applicable(self, obs: ObsSnapshotInst) -> bool: ...
    @abstractmethod
    def extract(self, obs: ObsSnapshotInst, context: ExtractorContext) -> list[NavFeature]: ...

# src/nav/nav_technique/nav_technique.py
class NavTechnique(ABC):
    name: str
    accepts_feature_types: frozenset[NavFeatureType]
    @abstractmethod
    def is_feasible(self, features: list[NavFeature]) -> NavFeasibilityReport: ...
    @abstractmethod
    def navigate(self, features: list[NavFeature], image: NDArrayFloat,
                 context: NavContext) -> NavTechniqueResult: ...

# src/nav/nav_orchestrator/nav_result.py
@dataclass(frozen=True)
class NavResult:
    # Headline (offset ± uncertainty + simple rank for the user)
    status: Literal['ok', 'failed', 'conflicted']
    offset_px: tuple[float, float] | None
    sigma_px: tuple[float, float] | None              # 1σ marginal per axis
    sigma_along_unobservable_px: float | None         # filled for rank-1 results
    confidence_rank: Literal['high', 'medium', 'low', 'conflicted', 'failed']
    confidence: float                                  # 0..1
    status_reason: str

    # Diagnostics
    covariance_px2: np.ndarray | None                  # full 2×2 (cross-terms)
    per_technique: list[NavTechniqueResult]
    feature_inventory: list[NavFeatureSummary]
    provenance: Provenance
```

End of plan.

---

## Part 16 — Cross-cutting style conventions

These apply uniformly across every section above; called out here so an implementer doesn't need to re-derive them per file.

- **Pessimism subsections**: every Part listing parameters or formulas includes a `### Pessimism` paragraph stating known weaknesses, calibration unknowns, and edge cases that fail. When implementing, treat the pessimism notes as preserved-context, not optional commentary.
- **Frozen dataclasses**: every dataclass shown in this plan is `@dataclass(frozen=True)` unless a mutating method is explicitly required (e.g., orchestrator's intermediate state). **Carve-out for numpy-array fields**: dataclasses whose fields include `np.ndarray` (e.g. `NavFeature`, `NavTechniqueResult`, `NavResult` with `position_cov_px` / `covariance_px2`) cannot be hashed because numpy arrays are unhashable. For these, `@dataclass(frozen=True, eq=False)` is used (`eq=False` disables auto-`__hash__` generation that would otherwise break) and the array is made read-only at construction (`__post_init__` calls `arr.setflags(write=False)`). This preserves the immutability contract without breaking on the unhashable array. Equality comparison uses `np.array_equal` via a custom `__eq__` when needed; most call sites don't compare these dataclasses for equality, so the absence of auto-equality is acceptable.
- **Internal helper signatures**: orchestrator-private methods that may grow new parameters over time use keyword-only after the obvious positional argument: `_make_context(self, obs: ObsSnapshotInst, *, ...)`, `_run_techniques_for_pass(self, features, context, *, requires_prior)`. Cursor rule §2 caps positional args at 3; this convention keeps internal APIs from drifting past the cap as features accrete.
- **Structured types over dict-of-Any**: `dict[str, Any]` appears only where the contents are user-supplied YAML config not otherwise typed. Internal data structures (`ReliabilityBreakdown`, `FeatureFlags`, `TechniqueDiagnostics`, `Provenance`, `ImageClassifierResult`) are typed dataclasses or frozen sum types.
- **Public API**: every new module declares `__all__` per cursor rule §3. The `nav` package's existing `py.typed` marker covers new code; verify it's still present after the new packages are created.
- **Docstrings**: every module, class, function, method gets a Google-style docstring with `Parameters:`, `Returns:`, `Raises:` per cursor rule §6. Wrap at 90 chars. The code snippets in this plan omit docstrings for brevity; implementations always include them.
- **Imports**: top-of-file, three groups (stdlib / third-party / local), alphabetized within group. No inline imports in new code (cursor rule §2).
- **Logging**: every new class inherits from `NavBase` (cursor rule §2 + observability conventions in Part 12.7). Never use bare `print()`.
- **Error handling**: catch exceptions at the smallest granularity possible (cursor rule §2). The orchestrator's per-technique `try` is the only broad catch; everything below catches specific exception types and either handles or `raise ... from`. The orchestrator never raises through to the caller — failures are always captured as `NavResult.failed` with a `StatusReason`.
- **No magic constants**: every numeric tunable lives in a config YAML or as a module-level `ALL_CAPS` constant. No hard-coded magic numbers in extractor / technique / orchestrator code.

---

# Resumption notes — read these first when picking up the discussion

This plan is the product of an extended interview-style design discussion. If you're picking it up in a fresh context, read this section before proposing changes — it preserves the working state, the user's communication style, and the load-bearing decisions that have already been settled.

## What this is

A complete-replacement design for the autonomous image-navigation pipeline in `rms-nav`. The current pipeline (`NavTechniqueCorrelateAll` + `NavModelCombined`) gets deleted; a feature-extraction + ensemble-orchestration design takes its place. Goal: navigate Cassini ISS, Voyager ISS, Galileo SSI, and New Horizons LORRI images autonomously, producing a calibrated `(offset, sigma, confidence_rank)` per image with no manual intervention and no cross-image state.

The plan file lives at `AUTONAV_PLAN.md` in the repo root (not the default plan-mode location) per the user's explicit direction.

## User communication style and preferences

- **Direct, concise, technically rigorous.** Don't pad. Don't hedge with weasel words. State decisions confidently or admit uncertainty cleanly; "probably", "maybe", "I think" should be reserved for actual uncertainty, not as politeness padding.
- **Honest about hallucinations.** The user has caught API hallucinations multiple times (`PSFModel.smear()`, `find_position` with movement, anisotropic-Gaussian smear). When unsure, **read the code** — paths like `/seti/all_repos/rms-psfmodel/src/psfmodel/` are valid; oops/numpy APIs should be verified, not assumed. The user explicitly said "Don't make things up. Look at the explicit API." This applies broadly.
- **Don't defer.** "If we're going to do it then do it all and do it right the first time right now." Anything that might once have been "v2 deferred" must either be implemented in v1 or filed as an explicit GitHub-issue tracker (Part 13b).
- **No backwards compatibility.** Cardinal Principle #1.
- **Walk-the-tree interview style for design.** The user prefers single-focused-question turns over multi-part questionnaires. They sometimes prefer free-form discussion to multiple-choice via `AskUserQuestion`. They occasionally reject `AskUserQuestion` calls entirely with "the user wants to clarify these questions" and want a more conversational follow-up. **Pattern: state the proposal concretely, list pessimism honestly, ask one focused question at the end.**
- **Likes concrete proposals**, not enumerated alternatives. When asked something they "don't fully understand", they want me to *propose the right answer with reasoning*, not lay out options for them to pick.
- **Doesn't do plan-approval prematurely.** When the user says "interview me relentlessly until aligned", don't end turns with `ExitPlanMode`; the plan-mode workflow is being driven by user intent, not the system's default 5-phase model.
- **Auto-mode + plan-mode tension is normal.** System reminders sometimes flip plan mode and auto mode on simultaneously. Resolution: follow the user's explicit instruction in their latest message; the plan file (`AUTONAV_PLAN.md`) is the only file edited; everything else is read-only.

## Cardinal principles (load-bearing)

1. **No backwards compatibility.** Delete legacy code outright. No dual-pipeline flags.
2. **Image-vs-model asymmetry.** Image-side operations are global only (whole-image MAD-noise, full-frame source detection, full-frame gradients). Per-feature image-side cropping at predicted positions is forbidden because SPICE attitude is what we're correcting. Model-side is where predicted positions live.
3. **No cross-image state.** Static data + config + per-image obs only. Cloud-tasks parallelism is a natural consequence.
4. **Angles in degrees in YAML config and JSON metadata; radians in code.** Loader converts at boundaries. Existing radian-based configs (e.g., bootstrap) are converted in the cutover.

## Decisions already settled

The body sections (Parts 1–16) are the source of truth. Read them directly. Part 13b enumerates the five tracked deferred items.

## Branches walked vs branches still open

**Walked** (closed):
- Branch 1 — User experience / CLI surface (operator + cloud-tasks worker; no manual retry; metadata schema is local-only)
- Branch 2 — Metadata content (redesigned schema; backplane reads only `offset`; stats program tracks technique/model/feature contributions; existing `nav_main_stats.py` is the schema reference)
- Branch 3 — Position covariance per feature type (LIMB_ARC, STAR, RING_EDGE, BODY_DISC, BODY_BLOB, TERMINATOR_ARC, CARTOGRAPHIC_MODEL)
- Branch 4 — Two-pass orchestration + ensemble combination (Mahalanobis 2σ agreement, 3σ conflict, precision-weighted Kalman)
- Branch 5 — Pattern matching (StarFieldFromCatalogNav)
- Reliability semantics + gates
- Multi-feature joint fitting + sub-pixel precision
- Camera rotation correction (Part 5b)

**Still to walk**: nothing in the original priority list remains. New branches will appear as the user surfaces them; no architectural decisions outstanding from the original walk.

## Open questions parked

None outstanding from the operator (Part 14 records resolved decisions). The user has not asked for plan approval; the discussion is mid-walk.

## What to do when continuing

1. Read this whole resumption-notes section, then read Cardinal Principles in the plan body, then skim the per-Part headings.
2. Pick the next branch from "Still to walk" (or ask the user which one).
3. Start with a focused proposal — concrete answer with reasoning, not options.
4. End the turn with one focused question (or `AskUserQuestion` with 3–4 options, recommended option marked first). Don't end with `ExitPlanMode` until the user explicitly says we're aligned.
5. When the user gives a corrective instruction ("don't dilate body silhouettes", "use existing nav_model_stars routines", "angles in degrees"), apply it immediately to both this turn's proposal and to any plan text affected. Don't park corrections.
6. Verify any API claim against actual code before committing it to the plan. Read paths under `/seti/newnav/rms-nav/src/`, `/seti/all_repos/rms-psfmodel/src/`, `/seti/nav/rms-csmithing/`, etc.

## Files and paths the user has referenced

- `/seti/newnav/rms-nav/AUTONAV_PLAN.md` — this plan file.
- `/seti/newnav/rms-nav/src/nav/nav_model/nav_model_stars.py` — the validated star routines to reuse.
- `/seti/newnav/rms-nav/src/nav/nav_model/rings/ring_filter.py` — `RingFeatureFilter` to reuse.
- `/seti/newnav/rms-nav/src/nav/annotation/` — annotation classes to reuse.
- `/seti/newnav/rms-nav/src/nav/config_files/config_21_saturn_rings.yaml` — existing ring catalog example (per-feature `rms` already encodes radial uncertainty).
- `/seti/newnav/rms-nav/src/nav/config_files/config_07_bootstrap.yaml` — existing bootstrap config; angles currently in radians, will be converted to degrees during the renumbering cutover.
- `/seti/all_repos/rms-psfmodel/src/psfmodel/` — actual PSFModel source (the canonical place to verify API; `eval_rect` accepts `movement=(my, mx)`, `movement_granularity=` per `gaussian.py:689`; `find_position` does **not** accept movement, so smear-aware refinement uses a subclass-with-baked-movement pattern).
- `/seti/nav/rms-csmithing/navigation/nav_main_stats.py` — schema reference for the future stats program.

## Hallucinations the user has caught (avoid repeating)

- `PSFModel.smear()` method — does not exist.
- Anisotropic-Gaussian construction for smear — wrong; `eval_rect` takes `movement` directly.
- `find_position(..., movement=...)` — does not exist; `find_position` does not have `movement` in its signature.
- "5× FOV catalog search radius" for Voyager/Galileo — wrong; use `obs.ra_dec_limits_ext()` which is already extfov-bounded.
- "Reliable SPICE for Cassini and NHLORRI" qualifier — wrong; SPICE attitude is *what we're correcting*, never assumed reliable.
- Body-silhouette dilation by `extfov_margin_vu` for source-rejection — wrong; high-pointing-error instruments would expand the silhouette to fill the frame.
- "Two non-parallel flat ring edges resolve a 2-D offset" — wrong; all ring edges of one planet share a normal; flat-ring-only scenes are rank-1.
- Polarity check for ring-edge matching — wrong; v1 uses gradient-magnitude only because emission-angle-vs-dust-content can flip sign.
- `range` field on `LimbPolyline` — wrong; occlusion is handled at extraction time by vertex cropping, just like ring shadows.
- Treating `NavResult` as a direct JSON dump — wrong; a curator function builds a curated subset.
- "One bright star, no body, can't navigate from scratch" — wrong; `StarUniqueMatchNav` exploits catalog uniqueness (when the brightest predicted catalog star is ≥1.5 mag brighter than the next-brightest predictable source, the catalog itself supplies the assignment, no triplet hash needed). 1-star and 2-star scenes are primary navigation modes, not failures.

## Active style conventions

- Per-feature data is on `NavFeature` with optional pixel template.
- Per-NavModel diagnostic dict (existing `metadata` property) flows to JSON `models:` block via curator pass-through.
- Per-image annotations flow through `NavResult.annotations` to summary PNG, never to JSON.
- Three-tuple uncertainty `(sigma_major_px, sigma_minor_px, angle_deg)` in JSON; full 2×2 (or 3×3 with rotation) covariance in `NavResult`.
- Angles in degrees in YAML and JSON; radians in code.
- File numbering: 3-digit, grouped (010-020 core, 100-150 per-feature defaults, 200-300s per-planet/instrument, 500s orchestration, 800-920 late stages).
- Logging: INFO is operator-readable summary; DEBUG is engineer-readable trace; WARNING for fallback / skip; ERROR for orchestrator-internal faults.

End of resumption notes.

---

## Continuation notes for the implementation (binding)

Hard-won decisions made during the cutover that the plan body does not
spell out.  Anyone resuming the implementation must respect these or the
codebase regresses to its previous shape.

### Code style — no plan references

- Source-tree code, comments, docstrings, and tests **must not** mention
  the plan.  No "Per Part X §Y", no "Phase N", no "Cardinal Principle
  #N", no "AUTONAV_PLAN".  The implementation reads as if it were
  written from scratch with the design in mind.  The plan is the design
  document; the code is the artifact.
- No legacy code, no backwards-compat shims, no "DEPRECATED: removed
  in vNext" comments.  When something is replaced, the old form is
  deleted; the new form is the only form.

### Git as the source of preserved-but-deleted code

Several modules were deleted during the cutover whose internal
*algorithms* must be preserved verbatim when their replacements are
written.  The recovery path is git history; do not rewrite from
scratch.

- **`src/nav/nav_model/nav_model_stars.py`** (deleted; ~1004 lines).
  Contains the validated star-catalog reduction the new
  `NavModelStars` must reuse:
  - Aberration via `_aberrate_star`.
  - Proper-motion application.
  - Multi-catalog precedence (UCAC4 / Tycho-2 / YBSC).
  - Magnitude binning + incremental catalog search for performance.
  - `stars_list_for_obs(...)` ext-FOV catalog reduction.
  - Star-vs-star + star-vs-body + star-vs-ring conflict marking via
    `_mark_conflicts_obj`.
  - `SCLASS_TO_B_MINUS_V` lookup.
  - Smear-aware PSF rendering through `psf.eval_rect(movement=...)`.
  Recover with `git log --diff-filter=D -- src/nav/nav_model/nav_model_stars.py`
  and pull the file from the appropriate commit; structure as helper
  modules under `src/nav/nav_model/stars/` (`predicted_snr.py`,
  `detection.py`, `aberration.py`, `conflicts.py`, etc.) imported by
  the new `NavModelStars.to_features`.
- **`src/nav/nav_model/nav_model_body.py`** (deleted; ~540 lines).
  Limb-mask extraction, body-silhouette computation, and the
  body-shape lookup logic should be lifted into the new
  `NavModelBody`.  `NavModelBodyBase` (still present) provides
  shared annotation rendering.
- **`src/nav/nav_model/nav_model_rings.py`** (deleted; ~525 lines).
  The four-pass `RingFeatureFilter` is preserved under
  `src/nav/nav_model/rings/ring_filter.py`.  The deleted
  `NavModelRings` carried the per-edge polyline rendering that the
  new `NavModelRings.to_features` needs to call.
- **`src/nav/nav_master/nav_master.py`** (deleted; ~500+ lines).
  Mostly subsumed by `NavOrchestrator`; lifted code is *not*
  required.  Recover only if a specific helper turns out to be
  reusable.
- **`tests/nav/nav_model/test_nav_model_*.py`** (deleted, except the
  `test_ring_*.py` files which were restored).  These tests
  exercised the deleted concrete classes; rewrite against the new
  contract — do not restore the deleted ones verbatim.

### Implementation conventions established during the cutover

- **`NavModel._abstract` and `NavTechnique._abstract`** are class
  attributes that, when ``True``, opt a class out of the auto-discovery
  registry.  Used by `NavModelBodyBase`, `NavModelRingsBase`, and
  `NavTechniqueManual`.  Concrete subclasses do not set this; the
  default ``False`` registers them.
- **`NavModel.instances_for_obs(cls, obs)`** is the per-class hook that
  `build_models_for_obs` iterates.  Default returns ``[]``.  Real-scene
  concrete subclasses override to return one instance per body / one
  instance for stars / one instance per planet with visible rings.
  Simulated subclasses inherit the empty default; the GUI flow
  constructs them directly.
- **`NavTechniqueManual` is not in the auto-registry.**  An
  interactive driver (a `nav_manual` script, when one is written, or
  the simulated-image GUI) instantiates it directly.  Setting
  `_abstract = True` keeps it out of background runs.
- **`compose_template_features(features, extfov_shape_vu)`** is the
  canonical bridge from `NavFeature` lists back to a single
  ext-FOV image+mask.  Used by the manual-nav dialog and (eventually)
  by the summary-PNG renderer.  Any future technique that wants a
  composite scene image should call this rather than rolling its own.
- **`navigate_image_files` does not accept a `model_factory`
  argument.**  It calls `build_models_for_obs(snapshot)` itself.  The
  caller supplies only `nav_models` / `nav_techniques` glob patterns.
- **Float-rounding in JSON output** is centralised in
  `nav.nav_orchestrator.curator`: 4 decimals for pixel quantities, 3
  for confidence, 6 for ET; ``inf`` becomes the
  `JSON_INF_SENTINEL = 1e9` finite sentinel.  Anywhere else in the
  codebase that writes JSON should reuse the curator's helpers.
- **Manual-nav offset uncertainty** is currently a constant
  ``_MANUAL_OFFSET_SIGMA_PX = 1.0`` per axis in
  `nav_technique_manual.py`.  Change it if operator precision is ever
  characterised against the library; do not let it become a config
  knob without a real reason.
- **`NavStatusReason` uses `StrEnum`.**  The minimum Python version is
  3.11; `StrEnum` is available natively.  Do not regress to a
  ``(str, Enum)`` mixin.
- **Stub honesty.**  Code that is not yet implemented either logs the
  deferral and returns an inert value (e.g. ``_write_summary_png``
  which logs at INFO and returns) or raises ``NotImplementedError``
  with a message naming the deferred work.  Silent placeholder
  values (1×1 PNGs, all-zero diagnostics, hard-coded magic
  saturation DNs, etc.) are not acceptable — the previous "first
  pass" approach was the foundation-cleanup origin story.
- **Magic constants live in module-level ALL_CAPS constants** with a
  one-line docstring stating units and intent (e.g.
  ``DEFAULT_FULL_WELL_DN_12_BIT`` in ``nav.nav_orchestrator.orchestrator``).
  No hard-coded saturation thresholds, magic factors, or sentinels in
  function bodies.
- **Broad `except Exception:` is reserved for the orchestrator's
  plugin-sandbox sites** (per-NavModel ``to_features`` / ``to_annotations``,
  per-NavTechnique ``navigate``).  Each site carries a docstring
  explaining the broad catch is intentional and a per-line
  justification comment.  Other broad catches are not acceptable.
- **Pdslogger output is captured via `capsys`, not `caplog`.**
  Pdslogger writes through its own stream handler and does not feed
  the standard logging propagation; tests that need to verify a
  WARNING / ERROR / EXCEPTION line read from
  ``capsys.readouterr().out``.
- **`NavResult.annotations: Annotations`** is the canonical
  collection slot for the summary PNG.  The orchestrator's
  ``_collect_annotations`` helper merges every NavModel's
  ``to_annotations`` into a single ``Annotations`` instance and
  threads it through every ``NavResult`` constructor.  The summary-PNG
  renderer is *not yet implemented*; ``_write_summary_png`` honestly
  logs and returns.  Replacing it requires (a) the rendering path
  itself, (b) updating the test harness to expect a real PNG, and
  (c) removing the current INFO-level skip.
- **`HARD_FAILURE_CLASSES`** as a separate frozen-string set was
  removed; the `_HARD_FAILURE_TO_REASON` dict (typed
  `dict[ImageClass, NavStatusReason]`) is the single source of
  truth for which classifier classes short-circuit before any
  technique runs.
- **`obs_snapshot.py` carries a scoped
  ``# type: ignore[misc]  # oops.Snapshot has no type stubs``** so
  ``mypy --strict`` is clean.  Remove the ignore once oops ships
  type stubs.

### Things that already work and should not be re-litigated

- The 700-test suite is green.  Adding code that breaks existing tests
  is a regression; either fix the regression or delete the obsolete
  test (with justification).
- `ruff check`, `ruff format --check`, and `mypy --strict` are clean
  on every committed file.  CI must stay clean.
- The `nav.nav_model.rings/` data-model subpackage is preserved
  verbatim from the original codebase; its 142 tests pass against
  it.  Do not refactor it without a concrete reason.
- The simulated body and ring NavModels (`NavModelBodySimulated`,
  `NavModelRingsSimulated`) are working on the new contract; the
  simulated-image GUI workflow continues through them.

End of continuation notes.
