# Phase 7 + 8 — image library seed (operator instructions)

This file walks an operator through curating the additional library
entries Phases 7 and 8 call for. The new techniques —
`StarUniqueMatchNav` (1- or 2-star unique match), `StarRefineNav`
(pass-2 refinement on the pass-1 prior), and `StarFieldFromCatalogNav`
(multi-star RANSAC pattern matcher) — ship with end-to-end unit tests
against synthetic inputs, but the integration regression suite needs
new sidecars on real images that exercise each star technique against
real spacecraft data.

The Phase 4 runbook (`PHASE4_LIBRARY_SEED.md`) covers the manual-nav
workflow, the `Save as Library Entry…` button, the `pds3://` URL
convention, and the `<image_id>` filename rule. **Read it first.** This
file only documents the Phase 7 / 8 scenario picks plus the
technique-specific gotchas for each new feature.

## What's new in Phases 7 + 8

- `StarUniqueMatchNav` consumes 1 or 2 STAR features. The 1-star path
  fires when the brightest predictable star is at least
  `brightness_margin_to_next_catalog_star_mag` (default 1.5 mag)
  brighter than the next-brightest predictable star inside extfov.
  The 2-star path fires when two predictable stars are present; the
  technique tries both detection-to-prediction assignments and picks
  the one whose joint residual is smaller. Confidence is capped at
  0.7 (1-star) or 0.8 (2-star).
- `StarRefineNav` runs in pass 2 with the pass-1 ensemble's prior
  offset attached to `NavContext`. For each predicted catalog star it
  (i) shifts the prediction by the prior, (ii) finds the brightest
  peak in a small refinement window, and (iii) returns the
  inverse-variance-weighted mean of per-star residuals. Stars whose
  detection sits more than `max_per_star_residual_px` (default 4 px)
  from the shifted prediction are dropped before the joint average.
- `StarFieldFromCatalogNav` requires ≥ 3 STAR features. It detects
  bright sources globally, builds similarity-invariant triplet hashes
  for both the detection and catalog cohorts, and iterates
  (det_triplet, cat_triplet) candidate pairs in deterministic order
  (hash distance ascending → sorted detection-source indices
  ascending). Each candidate proposes a translation; the winner is the
  one that scores the most inliers under a per-correspondence
  tolerance. With the inlier set, the technique refits translation by
  Tukey-biweight-reweighted least squares.

## What images do these techniques need?

Each technique has a different feasibility envelope, so the curated
sidecars span four canonical scenarios.

| Scenario | Primary technique | Stars in extfov | Other features | Why |
|---|---|---|---|---|
| **A** — Bright single star | `StarUniqueMatchNav` (1-star) | 1 unique-bright (≥ 1.5 mag margin) | None | Simplest path; should hit confidence cap 0.7. |
| **B** *(optional)* — Two-star translation | `StarUniqueMatchNav` (2-star) | 2 unambiguous | None | Cross-checks the assignment via residual; confidence cap 0.8. |
| **C** — Star-rich field | `StarFieldFromCatalogNav` | ≥ 6 detectable | None | Triplet matcher should converge to ≥ 6 inliers. |
| **D** — Stars + body | `StarRefineNav` (pass 2) | ≥ 1 detectable | Body limb (or disc) | A body-fed pass-1 prior lets the refiner sharpen on the lone star. |

**Required sidecars: A, C, D** (one per technique that has a unique
feasibility envelope).  Scenario A is the cheapest win; Scenario C is
the highest-value pick because `StarFieldFromCatalogNav` cannot be
exercised by either of the other two star techniques.

**Scenario B is optional** — see the rationale at the head of its
section below.  Skipping it does not weaken the regression suite; the
2-star path is structurally identical to the 1-star path with a
residual cross-check, and the unit test
``test_star_unique_match_two_star_recovers_planted_offset`` exercises
the algorithm end-to-end against a synthetic two-Gaussian fixture.

### Scenario A — Bright single star (`StarUniqueMatchNav` 1-star path)

A Cassini ISS or NHLORRI star-cal frame in which exactly one bright
catalog star (V ~ 5–7 mag) sits inside the FOV with no body or ring
in sight. The pointing-error envelope must be small enough that the
predicted star's `(v, u)` falls within the per-instrument
`search_window_px` of its planted position (default 30 px on the
unique-match technique).

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS NAC; NHLORRI; VGISS (where pointing reconstruction allows) |
| Star count in FOV | Exactly 1 detectable (catalog-predicted), ≥ 1.5 mag brighter than the next-brightest predictable star |
| Subject | Star calibration target — Cassini "STAR_CAL" or "BORESIGHT" sequence frames are ideal |
| Filter | Clear preferred (CL1+CL2 on Cassini); broadband filter on other instruments |

**Sidecar location**:
`tests/integration/image_library/images/one_bright_star_no_body/<IMAGE_ID>.yaml`

**Expected behavior**:

- `expected.status: ok`
- `expected.confidence_tier: medium` (placeholder coefficients clamp
  the 1-star path at 0.7 by design)
- `expected.primary_technique: StarUniqueMatchNav`
- `expected.techniques_must_run: [StarUniqueMatchNav]`
- `expected.techniques_must_skip:` every body / ring / star-field
  technique on a body-free, single-star scene

### Scenario B *(optional)* — Two-star translation (`StarUniqueMatchNav` 2-star path)

**Why this is optional.**  Real spacecraft frames with **exactly two**
detectable catalog stars in extfov (and no third predictable star
within reach of the SNR floor) are surprisingly rare:

- Cassini ISS NAC has a 0.35° FOV; star-cal targets are deliberately
  one bright star (Vega, Canopus, Spica) so the field around them is
  usually too sparse to find a second predictable star at all.
- Cassini ISS WAC has a 3.5° FOV; fields with 2 catalog stars almost
  always have a third within reach, which kicks the orchestrator over
  to ``StarFieldFromCatalogNav`` and makes the scene a Scenario C
  rather than Scenario B.
- NHLORRI and VGISS narrow-angle deep-sky frames are better hunting
  grounds, but the search is still mostly manual.

The 2-star path is structurally **identical to the 1-star path with a
residual cross-check**.  The unit test
``test_star_unique_match_two_star_recovers_planted_offset`` plants
two synthetic Gaussian PSFs at known positions, runs the technique,
and verifies the planted offset is recovered to within 0.4 px.  An
integration sidecar would re-test the same algorithm against a real
image — useful for breadth, but not essential when synthetic-fixture
coverage already exists and the placeholder confidence coefficients
are a Phase-10 retune target anyway.

**If you do find a real 2-star frame**, the rest of this section
documents the sidecar shape.  If not, skip it; Scenarios A, C, and D
are sufficient for the Phase 7 + 8 library seed.

Same posture as Scenario A but with two detectable catalog stars
inside extfov.  The 2-star path tries both detection-to-prediction
assignments and picks the smaller-residual fit.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS NAC star calibration; NHLORRI deep-sky pointing |
| Star count in FOV | Exactly 2 detectable; SNRs need not be equal but both must clear the per-instrument detection threshold |
| Subject | A short or pre-encounter star-cal frame |
| Filter | Clear |

**Sidecar location**:
`tests/integration/image_library/images/two_bright_stars_no_body/<IMAGE_ID>.yaml`

**Expected behavior**:

- `expected.status: ok`
- `expected.confidence_tier: medium` (2-star cap at 0.8)
- `expected.primary_technique: StarUniqueMatchNav`
- `expected.techniques_must_run: [StarUniqueMatchNav]`
- `expected.techniques_must_skip: [StarFieldFromCatalogNav, ...]`
  (`StarFieldFromCatalogNav` is mutually exclusive at feasibility time;
  it returns infeasible for fewer than 3 predictable stars).

### Scenario C — Star-rich field (`StarFieldFromCatalogNav`)

A dense star field — typically a Cassini ISS NAC star calibration
mosaic frame near the galactic plane, or an NHLORRI deep-sky frame
on the way to Pluto. The matcher needs ≥ 6 inlier correspondences for
a clean `expected.status: ok`.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS NAC (star-cal); NHLORRI; long-exposure VGISS NA |
| Detectable catalog stars | ≥ 6 inside extfov, with predicted SNRs that clear the detection threshold |
| Subject | Pure star field — no body or ring visible (a faint distant moon is acceptable but adds a competing technique) |
| Background | Dark sky |
| Filter | Clear preferred; long-exposure pointing-cal frames work well |

**Sidecar location**:
`tests/integration/image_library/images/star_dominated/<IMAGE_ID>.yaml`

(``star_dominated`` is the dense-star-field scene class.  Use
``faint_stars`` instead if the scene is dominated by 6+ faint
catalog stars rather than a smaller cohort of bright ones.)

**Expected behavior**:

- `expected.status: ok`
- `expected.confidence_tier: medium` to `high` (the matcher's
  `n_inliers` term saturates at 6 inliers in the placeholder
  coefficients)
- `expected.primary_technique: StarFieldFromCatalogNav`
- `expected.techniques_must_run: [StarFieldFromCatalogNav]`
- `expected.techniques_must_skip: [StarUniqueMatchNav, ...]`
  (`StarUniqueMatchNav` is mutually exclusive: with ≥ 3 unambiguous
  catalog stars the brightness-margin gate fails and the technique
  reports `enough_stars_for_triplet_match`-equivalent infeasibility.)

### Scenario D — Stars + body (`StarRefineNav` pass-2 refinement)

A stars-plus-body scene in which the body fit (limb / terminator /
disc) supplies the pass-1 prior and `StarRefineNav` sharpens it on the
1–2 predictable catalog stars also in the FOV. This is the prior-
required path; `StarRefineNav` cannot run from a zero prior.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS NAC (Saturn approach phase frames with a star nearby); NHLORRI Pluto encounter approach |
| Body | One body in FOV with a fittable limb or disc (favor a sharp lit limb) |
| Star count | ≥ 1 catalog-predicted star inside extfov, NOT inside the body silhouette |
| Filter | Whatever the body fit prefers (typically Clear) |

**Sidecar location**: `tests/integration/image_library/images/<scene_class>/<IMAGE_ID>.yaml`

Where `<scene_class>` is the body-side scene class (e.g.
`body_only_limb_curved`); the sidecar's `expected.techniques_must_run`
list pins both the body technique AND `StarRefineNav` so the
regression test verifies pass 2 fires.

**Expected behavior**:

- `expected.status: ok` (the body technique typically wins on
  confidence; `StarRefineNav` enters the ensemble in pass 2 and
  cross-validates the body fit).
- `expected.confidence_tier: medium` to `high`
- `expected.primary_technique:` whichever body technique wins (limb /
  terminator / disc)
- `expected.techniques_must_run:` list both the body technique AND
  `StarRefineNav` — this is the only sidecar that pins pass 2
  invocation.
- `expected.techniques_must_skip:` every technique that does not
  feasibly fire (e.g. `RingAnnulusNav` on a body-only scene).

## Workflow per image

Same as the Phase 4 runbook (steps 1–7) — load the image with
`nav_offset --manual <image_list_file>`, align by hand in the
manual-nav dialog, click `Save as Library Entry…`, edit the
`TODO_REPLACE_*` placeholders, and drop the file under the correct
scene-class directory.

For Phases 7 / 8 specifically:

- **Star-only scenes (A, B, C).** The dialog overlay paints predicted
  catalog stars as small crosses; verify visually that each predicted
  cross sits within ~1 px of a detectable bright pixel before
  accepting the offset.
- **Stars + body scene (D).** The dialog overlays both the body
  feature (limb polyline / disc template) and the predicted star
  cross. Align the body first (the body fit dominates the operator's
  perception of "right"); the predicted star should fall within a
  pixel of its image position once the body offset is right.

After the sidecar lands, run:

```bash
pytest tests/integration/test_image_library.py -v
pytest tests/integration/test_autonomous_nav.py -v -k <IMAGE_ID>
```

Both must pass before you commit.

## Sidecar template

The `Save as Library Entry…` button writes a stub like the one below;
fill the `TODO_REPLACE_*` slots from the dialog's reported values.
The example below is a Scenario C star-rich field; adjust
`expected.primary_technique` / `expected.techniques_must_run` to match
the scenario.

```yaml
schema_version: 1
image_id: <IMAGE_ID>                       # e.g. N1450122031_1_CALIB
mission: CASSINI_ISS                       # CASSINI_ISS | VOYAGER_ISS | GOSSI | NHLORRI
camera: NAC                                # NAC | WAC | SSI | NA | WA | LORRI
filter_combo: 'CL1+CL2'                    # canonicalized: filters sorted, '+'-joined
image_url: 'pds3://...'                    # path under PDS3_HOLDINGS_DIR

scene_tags:
  - star_dominated                         # primary scene class — must match the directory

ground_truth:
  offset_dv_px: TODO_REPLACE_DV            # operator-verified, in extfov px
  offset_du_px: TODO_REPLACE_DU
  offset_uncertainty_px: 1.0               # 1sigma; tighten for star-rich fields
  source: operator_verified
  operator: <username>
  verified_date: 2026-04-30                # YYYY-MM-DD
  ui_version: 'rms-nav 0.1.devXX'
  notes: |
    Star calibration field. Eight detectable catalog stars inside
    extfov; StarFieldFromCatalogNav recovers the planted offset by
    triplet pattern matching. Phase-8 status: matcher converges to
    (TODO, TODO), within TODO px of the operator's ground truth.
    Confidence coefficients are placeholders pending Phase 10
    calibration; the expected.confidence_tier below pins today's
    behavior.

expected:
  status: ok                               # ok | failed | conflicted
  confidence_tier: medium                  # high | medium | low | failed | conflicted
  primary_technique: StarFieldFromCatalogNav
  techniques_must_run: [StarFieldFromCatalogNav]
  techniques_must_skip:
    - BodyDiscCorrelateNav
    - BodyBlobNav
    - BodyLimbNav
    - BodyTerminatorNav
    - RingAnnulusNav
    - RingEdgeNav
    - StarUniqueMatchNav
```

### Field-by-field guidance

- **`offset_dv_px` / `offset_du_px`**: The dialog reports the operator's
  picked offset as `(dv, du)` in extfov pixels. Convention: predicted
  position `(v, u)` means actual position is `(v + dv, u + du)`. Round
  to 4 decimal places (the regression-test baseline rule).
- **`offset_uncertainty_px`**: 1 sigma marginal in pixels. Star-rich
  scenes typically support 0.3–0.5 px uncertainty (per-star centroid
  uncertainty is ~0.1 px and the joint translation averages many
  of those); single- or two-star scenes drop to 0.5–1.0 px because
  the centroid noise floor is larger relative to the constraint.
- **`expected.status`**: `ok` if the orchestrator returns a usable
  offset (combined confidence above `min_confidence`); `failed` if it
  reports `status=failed` with the right reason (the placeholder
  confidence formula often pushes the headline below
  `min_confidence` even when the answer is correct — record honestly,
  exactly as Phase 4 did with `ring_only_curved/N1492091163`). If
  `status: failed` is recorded, set `confidence_tier: failed` to
  satisfy the schema's cross-check.
- **`techniques_must_skip`**: List every technique you expect *not* to
  run on this scene — typically every body / ring technique on a
  body-free star scene, plus the mutually-exclusive star technique
  (`StarUniqueMatchNav` is infeasible when ≥ 3 unambiguous catalog
  stars are present; `StarFieldFromCatalogNav` is infeasible when <
  3 predictable stars are inside extfov).

## Confidence-formula calibration deferred

Phases 7 and 8 ship placeholder coefficients on every star
technique's confidence spec (now living in
`config_510_techniques.yaml.techniques.<TechniqueName>`, not the old
per-module Python constants). Phase 10 (image-library expansion +
confidence calibration) is when the alphas get retuned against the
full ~50-image library. Until then, expect the new sidecars to need
conservative `expected.confidence_tier` values:

- `StarUniqueMatchNav` 1-star: capped at 0.7 by design — record `medium`.
- `StarUniqueMatchNav` 2-star: capped at 0.8 by design — record `medium`.
- `StarRefineNav`: dominated by `n_stars_used` and `residual_scatter_px`.
  A clean refinement on 2–3 inliers should land in the `medium` tier;
  a single-inlier refinement on a soft star may fall to `low`.
- `StarFieldFromCatalogNav`: dominated by `n_inliers`. The placeholder
  coefficients saturate the term at 6 inliers; expect `medium` for
  6–8 inliers and `high` once the technique sees a 10+ inlier match.

Record the actual orchestrator-reported tier in your sidecar's
`expected.confidence_tier` and let the regression test pin today's
behavior. When Phase 10 retunes the coefficients in
`config_510_techniques.yaml` the sidecars will need a corresponding
`expected.confidence_tier` bump; that's a bookkeeping pass, not a
re-curation.

## After the seed lands

- The new sidecars enter the regression suite the moment they are
  committed; CI runs them on every PR via the `integration` mark
  (gated on `PDS3_HOLDINGS_DIR` etc.).
- For each `expected.status: ok` sidecar, seed a baseline JSON under
  `tests/integration/baselines/<IMAGE_ID>.json` so any orchestrator
  drift trips the byte-level regression test.
- Phase 7 / 8 follow-ups uncovered during integration runs (e.g. a
  per-instrument `psf_sigma_px` value that calibration will need to
  retune, or a `pattern_match_min_inliers` threshold that needs
  bumping per instrument) belong in `AUTONAV_PLAN.md` under the
  Phase 7 / 8 follow-ups subsection.
