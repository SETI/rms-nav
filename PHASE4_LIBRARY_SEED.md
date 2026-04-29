# Phase 4 — image library seed (operator instructions)

This file walks an operator through curating the 1–3 initial entries for the test image library. All the scaffolding (dialog button, schema, tests, baselines) is already in place — what you do here is pick the images, navigate them by hand, and commit the sidecars.

## Goal

Pick **1–3 images that the DT techniques (`BodyLimbNav`, `BodyTerminatorNav`, `RingEdgeNav`) are expected to succeed on**, drop sidecars under `tests/integration/image_library/images/<scene_class>/<image_id>.yaml`, and verify they pass the structural-invariants test (`tests/integration/test_image_library.py`). These are the seed entries for the autonomous-nav regression suite.

The integration regression test (`tests/integration/test_autonomous_nav.py`) will start scoring the orchestrator end-to-end against each sidecar as soon as it lands, provided `PDS3_HOLDINGS_DIR` is set.

## Prereqs

```bash
export PDS3_HOLDINGS_DIR=https://pds-rings.seti.org/holdings   # or local mount
export SPICE_PATH=/path/to/your/spice/kernels                  # required
pip install -e ".[dev]"
```

## Pick one image per scenario

You only need 1–3 entries — pick whichever subset of the three scenarios below you can find good candidates for. All three recommendations target Cassini ISS because it has the densest holdings and the most consistent calibration.

### Scenario A — `BodyLimbNav` (priority 1)

A scene where a single body fills a large fraction of the FOV with a long lit limb visible. This is the easiest DT win.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Body | Mimas, Enceladus, Tethys, Dione, or Rhea (regular moons) |
| Body diameter in FOV | 200–800 px (so >= ~30% of NAC frame) |
| Visible lit-limb fraction | >= 30% of the body's silhouette |
| Phase angle | < 90° (dayside hemisphere visible) |
| Other features in FOV | None preferred (no rings, no second body) |
| Filter | CL+CL ideal; otherwise broadband |
| Exposure | Not saturated, not blank |

**Sidecar location**: `tests/integration/image_library/images/body_mostly_offscreen/<IMAGE_ID>.yaml` (use `body_mostly_offscreen` even for partial-overflow scenes — it is the BodyLimbNav home class per the plan.)

### Scenario B — `RingEdgeNav` (priority 2)

A clean Saturn-rings scene with a *curved* ring edge visible in the frame. The curvature matters: a perfectly straight edge falls into the `ring_only_flat` class which is rank-1 (deferred to later phases).

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Subject | Saturn's rings; one named edge (e.g. A-ring outer edge) |
| Edge curvature | Polyline max-deviation > 0.5 px from a straight line |
| Bodies in FOV | None (no shepherd moons, no Saturn limb) |
| Filter | CL+CL ideal |

**Sidecar location**: `tests/integration/image_library/images/ring_only_curved/<IMAGE_ID>.yaml`

### Scenario C — `BodyTerminatorNav` (optional, hard)

A "terminator-only" scene is essentially impossible for a regular spheroidal moon: any phase angle high enough to give a long terminator (>90°) also leaves the bright lit limb of the crescent in frame, and the lit limb dominates confidence. So this scenario is genuinely optional for the Phase 4 seed — skip it if Scenarios A and B cover you.

If you do want a `BodyTerminatorNav` exercise in the seed, two paths work:

1. **Both-feature scene, primary = `BodyLimbNav`.** Pick a high-phase crescent (phase 100–140°) of the same body set as Scenario A. The sidecar's `expected.primary_technique` is `BodyLimbNav` (it almost always wins on confidence) but `expected.techniques_must_run` includes both `[BodyLimbNav, BodyTerminatorNav]`, so the regression test verifies the terminator path also fires. This is the recommended option.
2. **Extreme thin crescent, primary = `BodyTerminatorNav`.** Push to phase > 150° on a body where the lit limb shrinks below `LIMB_ARC` feasibility (< 30 surviving polyline vertices, per the Body NavModel) so `BodyTerminatorNav` becomes the primary. These scenes are rare and the SNR is bad; only do this if you have a hand-picked candidate already.

| Field | What to look for |
|---|---|
| Mission / camera | Cassini ISS, NAC |
| Body | Same set as Scenario A |
| Phase angle | 100–140° (option 1); > 150° (option 2) |
| Terminator | Long visible terminator arc through bright lit terrain |
| Filter | CL+CL |

**Sidecar location**: `tests/integration/image_library/images/high_phase_terminator/<IMAGE_ID>.yaml`

### Multi-feature scenes are normal

Most Cassini scenes have multiple feature kinds in the FOV at once (body + rings + stars). The DT techniques each navigate *independently* on whichever features they consume; the orchestrator's ensemble fuses all the results. So:

- A Scenario A "BodyLimbNav primary" image often *also* fires `BodyTerminatorNav` if the phase angle is non-trivial — that's fine. Set `expected.primary_technique` to whichever wins on confidence (usually `BodyLimbNav`); add the secondary technique to `expected.techniques_must_run` to verify it fires.
- A Scenario B "RingEdgeNav primary" image with a small moon also in frame fires `BodyLimbNav` or `BodyBlobNav` — same rule: list both in `techniques_must_run`.

The sidecar schema's `expected.techniques_must_run` / `techniques_must_skip` are exactly for capturing these multi-technique expectations.

### How to find candidates

The OPUS search interface at <https://opus.pds-rings.seti.org/> is the fastest way:

- Cassini ISS NAC, target = Mimas / Enceladus / Tethys / Dione / Rhea, phase angle 30–80°, target_distance ≥ 85,000 km → Scenario A candidates. (The distance floor is so the limb-uncertainty gate accepts the image; closer than ~85,000 km the autonomous pipeline will refuse to emit `LIMB_ARC` for the regular Saturn moons. See the "Resolution gate for `BodyLimbNav`" callout below.)
- Cassini ISS NAC, target = Saturn rings, no satellites in FOV → Scenario B.
- Same as A but phase angle 100–140° → Scenario C option 1.

For each candidate, copy the `Primary File Spec` (e.g. `COISS_2021/data/1521584844_1521609901/N1521598221_1.IMG`). The sidecar's `image_url` becomes `pds3://volumes/<that file spec>` — see the worked example below.

### Resolution gate for `BodyLimbNav`

The autonomous limb-emit gate refuses scenes where the per-pixel resolution is finer than the body's shape uncertainty: `limb_uncertainty_px = ellipsoid_residual_km / km_per_pixel_at_limb` must be `<= 2.0 px`. For Saturn's regular moons `ellipsoid_residual_km = 1.0` (per the body-shape table), so:

- Cassini NAC IFOV is 5.96 µrad/px.
- `km_per_pixel_at_limb >= 0.5` ↔ `subject_range >= ~84,000 km`.

A Mimas/Tethys/Dione/Rhea image *closer* than ~84,000 km from spacecraft will fail to emit `LIMB_ARC` (you'll see `Emitted N feature(s) [BODY_BLOB, ...]` in the log without any `LIMB_ARC`), and `BodyLimbNav` won't fire on it autonomously. Pick images at greater range for the BodyLimbNav seed.

## How accurate does the offset need to be?

Two thresholds matter, and they are not the same:

1. **Your eye's accuracy** at the dialog. At zoom 4–8× a sharp lit limb or a bright ring edge can be aligned to **~0.5 px**; that is the practical floor for `operator_verified` ground truth. Soft terminators, star-poor scenes, and faint ring edges drop to **~1–2 px**. Don't push past your eye's confidence — record the honest precision in `offset_uncertainty_px`.

2. **The CI tolerance** the regression test uses. It is `offset_uncertainty_px + 0.5 px` slack on each axis; the slack absorbs algorithm-version-level pixel jitter without false-failing. So the *test* will accept any orchestrator answer within that envelope of *your* answer. Tightening `offset_uncertainty_px` makes the test stricter; loosening it makes it more permissive.

### Setting `offset_uncertainty_px`

| Feature | Recommended `offset_uncertainty_px` | CI slack on each axis |
|---|---|---|
| Sharp lit limb (Mimas/Tethys/Rhea, low phase) | **1.0** px | ±1.5 px |
| Sharp ring edge (Cassini A-ring, B-ring) | **1.0** px | ±1.5 px |
| High-phase terminator | **2.0** px | ±2.5 px |
| Soft / saturated / faint feature | **2.0** px | ±2.5 px |

**Default to 1.0 px for the Phase 4 seed.** All three recommended scenarios should hit sharp-limb / sharp-edge precision when you pick clean candidates.

### How to verify your accuracy in the dialog

1. Pick the offset. Zoom to **4× minimum** before claiming you're done.
2. Walk along the visible limb / edge with your eye. The model polyline (green) should sit *on* the image edge, not parallel beside it, for the full visible extent. A 1-pixel parallel offset is your precision limit at that zoom.
3. Toggle the model overlay on/off (the dialog has a checkbox) and confirm the overlay snaps to the same pixels each time.
4. If the model edge is ambiguous (e.g. very soft terminator), bump `offset_uncertainty_px` to 2.0 and add a one-line note explaining why in `ground_truth.notes`.

### Cross-image inference is forbidden

Per the plan (Part 0 §40), every sidecar's offset must come from manually navigating *that* image. Even if the next frame in a sequence is 0.05 s later and "obviously" has nearly the same offset, do not copy it — spacecraft attitude drifts non-linearly at sub-second scales (thruster firings, momentum-wheel desats, jitter), so any between-anchor interpolation is unsafe at pixel precision.

## Workflow per image

1. **Run the manual-nav dialog**:

   ```bash
   nav_offset coiss --manual --image-filespec <IMAGE_NAME_NO_EXT>
   ```

   For a Cassini Rhea NAC frame:

   ```bash
   nav_offset coiss --manual --image-filespec N1521598221_1
   ```

   The dialog will load, show the source image plus the predicted model overlay, and let you drag the offset by hand.

2. **Pick the offset.** Hit **Auto** for a starting point if you like; refine by dragging or by using the dV / dU spinners. For Phase 4 seed images, sub-pixel accuracy matters — zoom in (+ button or mouse wheel) and align edges visually.

3. **Save the sidecar.** Click **Save as Library Entry...**. The file-save dialog suggests `<image_id>.yaml`; navigate to the right scene-class directory under `tests/integration/image_library/images/<scene_class>/` and save. The `<image_id>` must match the sidecar's `image_id` field — the `Save as Library Entry...` button pre-fills both, so just keep the suggested filename.

4. **Edit the YAML.** Open the file you just wrote and replace every `TODO_REPLACE_*` placeholder. An unedited template fails the structural-invariants test, so this is the safety net.

   The mandatory edits are:
   - `scene_tags`: replace `TODO_REPLACE_PRIMARY_CLASS` with the containing directory name (e.g. `body_mostly_offscreen`). Add secondary tags like the body name (`mimas`, `rhea`).
   - `expected.primary_technique`: `BodyLimbNav` / `BodyTerminatorNav` / `RingEdgeNav` to match the scenario.
   - `ground_truth.notes`: one line on what's in the scene and how you verified the offset.

   You may also tighten `ground_truth.offset_uncertainty_px` (default `1.0`) if the limb / edge is sharp and bright; loosen to `2.0` for soft features.

5. **Validate locally** (no holdings access needed):

   ```bash
   pytest tests/integration/test_image_library.py -v
   ```

   Every test must pass. Validation errors point at the exact field that needs editing.

6. **Run the live regression** (needs `PDS3_HOLDINGS_DIR` + SPICE):

   ```bash
   pytest tests/integration/test_autonomous_nav.py::test_one_library_image -v -k <IMAGE_ID>
   ```

   This will run the full orchestrator on the image and compare the computed offset to your hand-picked one. If it fails on offset tolerance, double-check your manual offset by re-running `nav_offset --manual` on the same image; if the algorithm and your eye still disagree by more than a couple of pixels, that is itself the data point — record it in `ground_truth.notes` and either widen the uncertainty or pick a cleaner candidate.

7. **Commit.** Add the sidecar to git. No baseline JSON yet — the regression-baseline seed is a separate small follow-up after the sidecars exist.

## Worked example

Hypothetical Mimas BodyLimbNav seed:

```text
tests/integration/image_library/images/body_mostly_offscreen/N1644931625_2.yaml
```

After the dialog writes the template and you fill in the `TODO`s:

```yaml
schema_version: 1
image_id: N1644931625_2
mission: CASSINI_ISS
camera: NAC
filter_combo: 'CL+CL'
image_url: 'pds3://volumes/COISS_2xxx/COISS_2068/data/1644892296_1644968375/N1644931625_2.IMG'

scene_tags:
  - body_mostly_offscreen
  - mimas

ground_truth:
  offset_dv_px: -3.4500
  offset_du_px: 1.2700
  offset_uncertainty_px: 1.0
  source: operator_verified
  operator: rfrench
  verified_date: 2026-04-29
  ui_version: 'rms-nav 0.1.dev26'
  notes: |
    Mimas fills ~60% of the NAC FOV; long lit limb on the eastern edge. Verified by overlaying the model limb polyline at zoom = 4x.

expected:
  status: ok
  confidence_tier: high
  primary_technique: BodyLimbNav
  techniques_must_run: [BodyLimbNav]
  techniques_must_skip: [StarFieldFromCatalogNav, RingEdgeNav]
```

Replace `image_id`, `image_url`, `offset_dv_px` / `offset_du_px`, `operator`, `verified_date`, `notes`, and the technique lists with the real values for your chosen frame.

## After the seed lands

Phase 10 grows the library to ~50 images and adds the regression baselines (`tests/integration/baselines/<image_id>.json`). Phase 4 only needs the seed; everything else is in place to grow into.
