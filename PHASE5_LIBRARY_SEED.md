# Phase 5 — image library seed (operator instructions)

This file walks an operator through curating the 2–4 additional library
entries Phase 5 calls for. The two new techniques —
`BodyDiscCorrelateNav` and `BodyBlobNav` — both ship with end-to-end
unit tests against synthetic inputs, but the integration regression
suite needs new sidecars on real images that exercise their happy and
boundary paths.

The Phase 4 runbook (`PHASE4_LIBRARY_SEED.md`) covers the manual-nav
workflow, the `Save as Library Entry…` button, the `pds3://` URL
convention, and the `<image_id>` filename rule. **Read it first.** This
file only documents the Phase 5 scenario picks plus the
technique-specific gotchas for the two new features.

## What's new in Phase 5

- `BodyDiscCorrelateNav` consumes per-body `BODY_DISC` features (the
  full-disc Lambert-shaded postage-stamp template). It is the right
  technique for scenes where the body fits inside the FOV with a
  well-lit, mostly visible disc — *not* the partial-overflow scenes the
  Phase 4 limb runbook targeted.
- `BodyBlobNav` consumes `BODY_BLOB` features. The body extractor emits
  `BODY_BLOB` instead of `LIMB_ARC` whenever
  `limb_uncertainty_px > LIMB_ARC_MAX_UNCERTAINTY_PX` (default 3 px),
  which fires on close-range irregular satellites and on under-resolved
  bodies. So a Prometheus-at-approach scene goes through `BodyBlobNav`,
  while Prometheus in a wide F-ring mosaic goes through `BodyLimbNav` —
  same body, different per-image path.

## Body extractor emission gate (recap)

The body NavModel decides per-image, per-body which feature(s) to emit:

```
if limb_uncertainty_px <= 3 px:           # LIMB_ARC_MAX_UNCERTAINTY_PX
    emit LIMB_ARC
    if visible_lit_fraction >= 0.4 and overflow_fraction <= 0.3:
        also emit BODY_DISC
elif predicted_diameter_px >= 8 px:       # BODY_BLOB_MIN_DIAMETER_PX
    emit BODY_BLOB
else:
    emit nothing
```

The `BODY_DISC` companion gates ensure we only run NCC pyramids when the
body actually fills enough of the FOV for the correlation peak to be
sharp. A body that overflows by > 30 % falls back to LIMB_ARC alone.

## Picking Phase 5 candidates

Aim for **2–4 sidecars**, choosing whichever subset of the four
scenarios below your holdings cover. Prioritise scenario **A** because
it exercises both Phase-5 techniques back-to-back (limb + disc on the
same body); scenario **C** is the cheap "blob-only" win on a small or
distant moon. Scenario **D** stretches the multi-body Z-buffer paint
path.

### Scenario A — Body fills FOV with small overflow (`BodyDiscCorrelateNav` + `BodyLimbNav`)

A scene where one regular moon dominates the frame with `<= 30 %`
overflow and `>= 40 %` of the lit hemisphere visible. Both `BODY_DISC`
and `LIMB_ARC` get emitted; the ensemble combines them.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Body | Mimas, Enceladus, Tethys, Dione, Rhea, Iapetus |
| Body diameter | 800–1024 px (NAC frame is 1024 px) |
| Visible lit fraction | `>= 40 %` of the disc area |
| Overflow fraction | `<= 30 %` |
| Phase angle | < 60° (most of the disc is bright) |
| Other features | None preferred |

**Sidecar location**:
`tests/integration/image_library/images/body_full_fov/<IMAGE_ID>.yaml`

**Expected behavior**:
- `expected.status: ok`
- `expected.confidence_tier: high`
- `expected.primary_technique: BodyDiscCorrelateNav` (NCC against the
  full disc usually wins on confidence; `BodyLimbNav` is the runner-up)
- `expected.techniques_must_run: [BodyDiscCorrelateNav, BodyLimbNav]`
- `expected.techniques_must_skip: [RingEdgeNav, StarFieldFromCatalogNav]`

### Scenario B — Body partial overflow (`BodyLimbNav` only, no disc)

A scene where the body overflows the FOV by > 30 % but the limb is
sharp enough for `LIMB_ARC` to fire. The body extractor emits
`LIMB_ARC` only — the disc gate fails on overflow. This complements
the Phase 4 Tethys scene (`body_mostly_offscreen` was the
`BodyLimbNav`-LM-divergence test); pick a different body to broaden
the calibration sample.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Body | Different from the Phase 4 Tethys (Rhea / Dione / Mimas) |
| Body fraction in FOV | 50–80 % (overflow_fraction 0.2–0.5) |
| Limb visible | Long bright arc (`>= 60 px`) |
| Other features | None |

**Sidecar location**:
`tests/integration/image_library/images/body_partial_overflow/<IMAGE_ID>.yaml`

**Expected behavior**:
- `expected.status: ok` (or `failed` if the LM divergence pattern
  re-fires on a heavily-cratered body — record honestly)
- `expected.primary_technique: BodyLimbNav`
- `expected.techniques_must_run: [BodyLimbNav]`
- `expected.techniques_must_skip: [BodyDiscCorrelateNav]`

### Scenario C — Below-resolution / irregular body (`BodyBlobNav`)

A scene where a small or irregular moon is too unresolved or
shape-irregular for the limb fit. The body extractor emits `BODY_BLOB`
only.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Body | Prometheus, Pandora, Atlas, Pan, Hyperion at close range, **or** any regular moon at a distance where `predicted_diameter_px` is 10–30 px |
| Body diameter in FOV | 8–30 px |
| Other bright sources | None inside the predicted bbox (otherwise the centroid is biased) |
| Background | Dark sky preferred |

**Sidecar location**:
- Irregular body: `tests/integration/image_library/images/body_irregular/<IMAGE_ID>.yaml`
- Distant regular body: `tests/integration/image_library/images/below_resolution_body/<IMAGE_ID>.yaml`

**Expected behavior**:
- `expected.status: ok` (or `failed` if calibration discovers the
  centroid is too soft on the chosen frame; the confidence cap of 0.4
  may push the orchestrator below `min_confidence` — record honestly)
- `expected.confidence_tier: low` or `medium` (cap is 0.4 — never
  `high`)
- `expected.primary_technique: BodyBlobNav`
- `expected.techniques_must_run: [BodyBlobNav]`
- `expected.techniques_must_skip: [BodyLimbNav, BodyDiscCorrelateNav]`

### Scenario D — Multi-body Z-buffer paint (`BodyDiscCorrelateNav` joint fit)

Optional but high-value: a single Cassini frame with two visible
moons. The Z-buffer paint composes both per-body `BODY_DISC` templates
into one composite, and the joint NCC removes the swap-moon
ambiguity that solo correlation suffers from. The trickiest part is
finding a real Cassini frame with two non-trivially-sized moons in the
same FOV; mosaic frames around saturnian conjunctions are the natural
hunting ground.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC or WAC |
| Bodies | Two regular moons, both `>= 30 px` diameter, both inside FOV |
| Subject ranges | Different (so depth ordering is well-defined) |
| Other features | Saturn rings OK; star field OK |

**Sidecar location**:
`tests/integration/image_library/images/multi_body/<IMAGE_ID>.yaml`

**Expected behavior**:
- `expected.status: ok`
- `expected.primary_technique: BodyDiscCorrelateNav` (joint NCC)
- `expected.techniques_must_run: [BodyDiscCorrelateNav]`
- `expected.techniques_must_skip: []` (limb / blob may also fire — list
  only the ones the orchestrator must NOT run)

## Workflow per image

Same as the Phase 4 runbook (steps 1–7). The `Save as Library Entry…`
button writes a sidecar pre-filled with auto-derivable fields plus
`TODO_REPLACE_*` placeholders for the operator-curated entries.
Edit those, drop the file under the correct scene-class directory, and
run:

```bash
pytest tests/integration/test_image_library.py -v
pytest tests/integration/test_autonomous_nav.py -v -k <IMAGE_ID>
```

Both must pass before you commit.

## Confidence-formula calibration deferred

Phase 5 ships placeholder coefficients on both new techniques' confidence
specs. Phase 10 (image-library expansion + confidence calibration) is
when those alphas get retuned against the full ~50-image library. Until
then, expect the new sidecars to need conservative
`expected.confidence_tier` values:

- `BodyDiscCorrelateNav`: `low` to `medium` even on textbook-clean
  scenes.
- `BodyBlobNav`: `low` (the 0.4 hard cap is permanent — even Phase 10
  calibration cannot push past it).

Record the actual orchestrator-reported tier in your sidecar's
`expected.confidence_tier` and let the regression test pin today's
behavior. When Phase 10 retunes the coefficients the sidecars will
need a corresponding `expected.confidence_tier` bump; that's a
bookkeeping pass, not a re-curation.

## After the seed lands

- The new sidecars enter the regression suite the moment they are
  committed; CI runs them on every PR via the `integration` mark
  (gated on `PDS3_HOLDINGS_DIR` etc.).
- For each `expected.status: ok` sidecar, the operator may also seed a
  baseline JSON under `tests/integration/baselines/<IMAGE_ID>.json` so
  any orchestrator drift trips the byte-level regression test. Use
  `python -c "from tests.integration.baseline import seed_baseline; …"`
  or the helper Phase 4 documented.
- Phase 5 follow-ups uncovered during integration runs (e.g. a
  `BodyDiscCorrelateNav` consistency value that calibration will need
  to retune) belong in `AUTONAV_PLAN.md` under the new
  "Phase 6 / Phase 10 follow-ups uncovered by Phase 5 integration"
  subsection.
