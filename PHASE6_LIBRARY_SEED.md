# Phase 6 — image library seed (operator instructions)

This file walks an operator through curating the 1–2 additional library
entries Phase 6 calls for. The new technique — `RingAnnulusNav` — ships
with end-to-end unit tests against synthetic inputs, but the integration
regression suite needs new sidecars on real images that exercise the
ring-annulus path.

The Phase 4 runbook (`PHASE4_LIBRARY_SEED.md`) covers the manual-nav
workflow, the `Save as Library Entry…` button, the `pds3://` URL
convention, and the `<image_id>` filename rule. **Read it first.** This
file only documents the Phase 6 scenario picks plus the
technique-specific gotchas for the new feature.

## What's new in Phase 6

- `RingAnnulusNav` consumes per-planet `RING_ANNULUS` features. The
  rings model emits a `RING_ANNULUS` instead of the per-edge
  `RING_EDGE` polylines whenever the surviving edge polyline compresses
  radially below `RING_ANNULUS_MAX_RADIAL_PX = 5 px`, i.e. the
  individual ringlets are no longer separable at the image scale.
  Each `RING_ANNULUS` feature carries a multi-ring composite template
  (`template_img`) plus a boolean mask; the technique fuses every
  per-planet template via Z-buffer paint and runs one joint
  pyramid-NCC against the composite.
- Multi-planet scenes (rare but real — Cassini approach phase imaged
  Jupiter and Saturn together; New Horizons imaged Jupiter from Pluto
  distance) emit one `RING_ANNULUS` per detectable ring system; the
  technique handles `len(features) > 1` by Z-buffer painting all
  annuli into the same composite.
- `use_gradient='auto'` self-selects raw vs gradient mode per image:
  raw wins on broad-brightness-gradient ring geometries (low-resolution
  Saturn rings where the C-ring is uniformly dim); gradient wins when
  sharp ringlet edges still dominate the composite.

## When does the rings model emit `RING_ANNULUS` instead of `RING_EDGE`?

The rings model walks every catalog ring feature in the FOV, renders it
to a per-edge mask, samples the mask into a polyline, and then decides
per polyline:

```
radial_extent_px = max - min projection of the polyline onto its mean radial normal
straight = polyline's max deviation from best-fit straight line < FLAT_CURVATURE_THRESHOLD_PX

if radial_extent_px <= RING_ANNULUS_MAX_RADIAL_PX (5 px) and not straight:
    emit RING_ANNULUS  (carrying the rendered template)
else:
    emit RING_EDGE     (carrying the per-vertex polyline)
```

So the gate fires on *low-resolution* ring scenes where two or more
adjacent rings have collapsed within 5 pixels of each other in the
image plane, AND where the ring still shows curvature (a flat-edge ring
goes through `RING_EDGE` with the rank-1 covariance path instead).

## Picking Phase 6 candidates

Aim for **1–2 sidecars**, both Cassini ISS distant ring views.
Scenario **A** is the cheap win: one Cassini distant Saturn frame in
which the rings span only ~50 px radially across the entire span A→C
ring. Scenario **B** is the higher-value pick: two ring systems in one
frame (Saturn approach plus a galaxy-glimpse Jupiter, or NHLORRI Pluto
plus Charon if either has a detectable ring system). Scenario **B** is
optional — ship the easy one first.

### Scenario A — Distant Saturn ring view (`RingAnnulusNav` single-planet)

A Cassini distant ring view in which the Saturn ring system has
compressed radially below 5 px between adjacent ringlets. Every
catalog ring polyline goes through the annulus path; the rings model
emits one `RING_ANNULUS` feature per Saturn ring; the technique fuses
them and runs one joint NCC.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC or WAC |
| Ring radial span in FOV | < 200 px from C-ring inner edge to A-ring outer edge |
| Subject range | Distant — typically > 5e6 km from Saturn (the approach phase, before the encounter; or the late mission post-Grand-Finale departure frames) |
| Ring tilt | Any (the technique handles edge-on through fully-open) |
| Bodies | None visible inside FOV preferred (a distant moon adds a competing technique that complicates the regression check) |
| Background | Dark sky preferred |

**Sidecar location**:
`tests/integration/image_library/images/ring_only_curved/<IMAGE_ID>.yaml`

(Use `ring_only_curved` — the existing scene class. The schema does
not have a separate `ring_annulus` class because the operator's intent
is "ring-only scene, curved enough that the technique sees a usable
geometry"; whether the rings model emits `RING_EDGE` or `RING_ANNULUS`
is downstream of the operator's intent.)

**Expected behavior**:

- `expected.status: ok` (or `failed` if the placeholder confidence
  formula puts the result below `min_confidence` — record honestly,
  exactly as Phase 4's `ring_only_curved/N1492091163_1_CALIB.yaml` did
  for `RingEdgeNav`).
- `expected.confidence_tier: low` to `medium` (placeholder
  coefficients; Phase 10 retunes).
- `expected.primary_technique: RingAnnulusNav`
- `expected.techniques_must_run: [RingAnnulusNav]`
- `expected.techniques_must_skip: [BodyDiscCorrelateNav, BodyBlobNav, BodyLimbNav, BodyTerminatorNav]`
  (and `RingEdgeNav` if no surviving ring polyline crosses the
  `RING_ANNULUS_MAX_RADIAL_PX` threshold the other way).

### Scenario B — Multi-planet ring composite (`RingAnnulusNav` joint fit)

Optional: a single frame with two visible ring systems. Real
candidates are limited — the obvious picks are:

- A Cassini approach-phase frame with both Saturn and Jupiter in the
  FOV (Cassini's 2000-12 Jupiter encounter while inbound to Saturn
  produced a few of these).
- A New Horizons LORRI frame at Pluto distance that captures a
  detectable Jupiter ring system, if the geometry happens to put both
  in the same FOV.

The real Cassini case is the one to chase first. NHLORRI Pluto +
Charon does *not* qualify because Pluto and Charon do not have
catalog ring systems — there is no `NavModelRings` instance per body
without a ring catalog entry.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC (the Jupiter-and-Saturn encounter frames) |
| Ring systems in FOV | Two — both with detectable annuli |
| Subject ranges | Different (so the Z-buffer ordering by `subject_range_km` is well-defined) |
| Bodies | None preferred (same reason as Scenario A) |

**Sidecar location**:
`tests/integration/image_library/images/ring_only_curved/<IMAGE_ID>.yaml`

**Expected behavior**:

- `expected.status: ok` (joint NCC across two planets is more
  constrained than one — the planted-offset unit test plants the
  same offset against two separate annuli and recovers it within
  1 px).
- `expected.confidence_tier: low` to `medium` (still placeholder
  coefficients).
- `expected.primary_technique: RingAnnulusNav`
- `expected.techniques_must_run: [RingAnnulusNav]`
- `expected.techniques_must_skip:` whatever does not fire on this
  scene; expect at least the body techniques.

## Workflow per image

Same as the Phase 4 runbook (steps 1–7) — load the image with
`nav_offset --manual <image_list_file>`, align by hand in the
manual-nav dialog, click `Save as Library Entry…`, edit the
`TODO_REPLACE_*` placeholders, and drop the file under the correct
scene-class directory.

For Phase 6 specifically, the dialog overlay paints `RING_ANNULUS`
features as their template (the multi-ring band) rather than per-edge
polylines.  Verify visually that the predicted band overlays the
observed bright ring region within ~1 px before accepting the offset.

After the sidecar lands, run:

```bash
pytest tests/integration/test_image_library.py -v
pytest tests/integration/test_autonomous_nav.py -v -k <IMAGE_ID>
```

Both must pass before you commit.

## Sidecar template

The `Save as Library Entry…` button writes a stub like the one below;
fill the `TODO_REPLACE_*` slots from the dialog's reported values.

```yaml
schema_version: 1
image_id: <IMAGE_ID>                       # e.g. N1450122031_1_CALIB
mission: CASSINI_ISS                       # CASSINI_ISS | VOYAGER_ISS | GOSSI | NHLORRI
camera: NAC                                # NAC | WAC | SSI | NA | WA | LORRI
filter_combo: 'CL1+CL2'                    # canonicalized: filters sorted, '+'-joined
image_url: 'pds3://...'                    # path under PDS3_HOLDINGS_DIR

scene_tags:
  - ring_only_curved                       # primary scene class — must match the directory

ground_truth:
  offset_dv_px: TODO_REPLACE_DV            # operator-verified, in extfov px
  offset_du_px: TODO_REPLACE_DU
  offset_uncertainty_px: 1.0               # 1sigma; tighten for sharp annuli
  source: operator_verified
  operator: <username>
  verified_date: 2026-04-29                # YYYY-MM-DD
  ui_version: 'rms-nav 0.1.devXX'
  notes: |
    Distant Saturn ring view. The catalog A/B/C edges all collapse
    radially below RING_ANNULUS_MAX_RADIAL_PX so the rings model
    emits one RING_ANNULUS feature for the Saturn ring system; the
    technique runs one joint NCC against the composite annulus
    template.

    Phase-6 status: RingAnnulusNav converges to (TODO, TODO),
    within TODO px of the operator's ground truth. Confidence
    coefficients are placeholders pending Phase 10 calibration; the
    expected.confidence_tier below pins today's behavior.

expected:
  status: ok                               # ok | failed | conflicted
  confidence_tier: low                     # high | medium | low | failed | conflicted
  primary_technique: RingAnnulusNav
  techniques_must_run: [RingAnnulusNav]
  techniques_must_skip:
    - BodyDiscCorrelateNav
    - BodyBlobNav
    - BodyLimbNav
    - BodyTerminatorNav
```

### Field-by-field guidance

- **`offset_dv_px` / `offset_du_px`**: The dialog reports the operator's
  picked offset as `(dv, du)` in extfov pixels.  Convention:
  predicted position `(v, u)` means actual position is
  `(v + dv, u + du)`.  Round to 4 decimal places (the regression-test
  baseline rule).
- **`offset_uncertainty_px`**: 1 sigma marginal in pixels.  The
  regression test gates the orchestrator's recovered offset against
  `offset_uncertainty_px + 0.5` per axis, so this number sets the slack
  for an `expected.status: ok` sidecar.  For a sharp distant-ring
  scene, 1.0 px is a sensible default; bump to 1.5 or 2.0 if the
  annulus is heavily smeared.
- **`expected.status`**: `ok` if the orchestrator returns a usable
  offset (combined confidence above `min_confidence`); `failed` if it
  reports `status=failed` with the right reason (the placeholder
  confidence formula often pushes the headline below
  `min_confidence` even when the answer is correct — record honestly,
  exactly as Phase 4 did with `ring_only_curved/N1492091163`).  If
  `status: failed` is recorded, set `confidence_tier: failed` to
  satisfy the schema's cross-check.
- **`techniques_must_skip`**: List every technique you expect *not* to
  run — typically all four body techniques on a body-free ring scene.
  Listing extra techniques here is harmless if they genuinely don't
  fire; missing one that *does* fire trips the regression test.

## Confidence-formula calibration deferred

Phase 6 ships placeholder coefficients on `RingAnnulusNav`'s confidence
spec (now living in `config_510_techniques.yaml.techniques.RingAnnulusNav`,
not the old per-module Python constant).  Phase 10 (image-library
expansion + confidence calibration) is when the alphas get retuned
against the full ~50-image library.  Until then, expect the new
sidecars to need conservative `expected.confidence_tier` values:

- Single-annulus scenes: `low` (the `annulus_count` term contributes
  only `0.4 * (1/2) = 0.2` to the sigmoid argument with placeholder
  alphas).
- Multi-annulus scenes: `medium` (saturated `annulus_count` term
  contribution).

Record the actual orchestrator-reported tier in your sidecar's
`expected.confidence_tier` and let the regression test pin today's
behavior.  When Phase 10 retunes the coefficients in
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
- Phase 6 follow-ups uncovered during integration runs (e.g. a
  `RingAnnulusNav` consistency value that calibration will need to
  retune, or a `RING_ANNULUS_MAX_RADIAL_PX` threshold that needs
  bumping per instrument) belong in `AUTONAV_PLAN.md` under the
  Phase 6 follow-ups subsection.
