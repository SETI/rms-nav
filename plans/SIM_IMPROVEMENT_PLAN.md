# Simulator improvement plan — backend, GUI, and calibration test layer

This document is a self-contained plan for upgrading the rms-nav
simulator system so it can serve as a useful adjunct to the
operator-curated real-image library for navigation calibration and
verification work.  It is written so that an AI or developer with no
prior context can read this single file and understand both what to
build and why.

---

## Reading-order guide

If you are picking this up cold, read sections in this order:

1. **Context and problem statement** — what motivates this plan.
2. **System inventory** — what exists today.
3. **Cardinal principles** — non-negotiable design constraints.
4. **Backend improvement plan** — what to change in
   `src/nav/sim/`, in dependency order.
5. **GUI improvement plan** — how the simulator GUI in
   `src/main/nav_create_simulated_image.py` follows the backend.
6. **Calibration test strategy** — how the upgraded sim feeds into
   confidence-formula calibration and regression coverage.
7. **Phase ordering** — which work unblocks which.
8. **Out of scope / explicit non-goals** — things that look good but
   aren't worth doing for the calibration mission.
9. **Acceptance criteria** — how to know each phase is done.

The plan is intentionally not prescriptive about the exact PR
boundaries; it gives a phase ordering with clean handoffs and lets
the implementer batch them.

---

## 1. Context and problem statement

### 1.1 What this plan is for

The rms-nav navigation pipeline (`src/nav/nav_orchestrator/`) maps
real spacecraft images to per-image pixel-offset corrections plus a
calibrated `(offset, sigma, confidence_tier)` triple.  The
confidence formula for each technique is `sigmoid(α₀ + Σ αᵢ × xᵢ)`
where the `αᵢ` are coefficients tuned once against an
operator-curated library of ~50 real-holdings images
(`tests/integration/image_library/images/<scene_class>/<image_id>.yaml`,
documented in `tests/integration/image_library/images/README.txt` and
`PHASE10_CURATION.md`).

The Phase 10 curation work is operator-bottlenecked.  Finding real
images with specific scene attributes (a body at a specific phase
angle, a body of a specific irregularity, a frame with exactly N
catalog stars, a frame with stray-light contamination) is hard.
Manually navigating each found image to set the ground-truth offset
is labour-intensive.  The set of scenes that *exist* in the real
archives is fixed by what spacecraft happened to take during their
missions; there are gaps that no amount of curation can fill.

The simulator system at `src/nav/sim/` can synthesize images on
demand for arbitrary geometry.  The natural question — and the
question that motivates this plan — is *can we use the simulator
to make the calibration work easier or more thorough?*

### 1.2 The cardinal answer: augment, do not substitute

A separate analysis (preserved here so this document is
self-contained) concluded:

- **Sim cannot substitute for real-data calibration.** The sim has
  its own model assumptions (ellipsoidal bodies, Gaussian PSF,
  additive Gaussian noise, no scattered light, no instrument
  artifacts).  Calibrating the per-technique α coefficients against
  sim-image diagnostic distributions tunes the formula for the sim's
  idealized world — and on real images the same diagnostics
  distribute differently, so the calibrated tiers underflow or
  overflow systematically.  The whole class of failure modes the
  operator-curated library exists to characterize (irregular bodies,
  real PSF wings, real noise structure, real scattered light) is
  fundamentally absent from sim today.
- **Sim can augment the verification, sensitivity, and
  regression layers.** Once calibrated against real data, the
  sim is the right tool for: single-variable sensitivity analysis
  (sweep phase angle continuously, verify confidence response is
  smooth and monotonic), unit testing the technique algorithms
  (planted-offset recovery), regression coverage for combinations
  that don't exist in real archives, and bootstrap pre-tuning to
  give the curve-fit a sensible α starting point.
- **The gap between sim and real is closeable.** Specific
  improvements to the sim — realistic noise, per-instrument PSF
  coupling, non-ellipsoidal bodies, smear as a true convolution —
  bring sim diagnostics close enough to real that the sim's
  verification layer becomes load-bearing.  This plan enumerates
  those improvements.

The plan is therefore organized around two goals:

1. **Close the realism gap** between sim and real, in priority order
   of which gaps most distort the diagnostics the calibration cares
   about.
2. **Build the test infrastructure** that lets the sim carry weight
   in regression / sensitivity / bootstrap work without having to
   become the calibration target itself.

### 1.3 Why both backend and GUI must move together

The sim GUI (`src/main/nav_create_simulated_image.py`, a single ~2500
LOC PyQt6 application) is operator-facing: it lets a human dial in
geometry parameters and watch the rendered image update live.  Every
backend parameter that the GUI doesn't expose is invisible to the
operator and effectively dead.  Conversely, every GUI control that
isn't backed by a meaningful backend parameter is a lie.

The plan therefore phases the backend and GUI together: each backend
addition lands with a corresponding GUI tab / control / preview pane,
and each GUI affordance is backed by a real backend parameter that is
also reachable from a YAML scene spec (so the GUI is one of three
peers — Python API, YAML, GUI — not the sole control surface).

---

## 2. System inventory (what exists today)

This is the state of the world at the time this plan was written.
Everything below is grep-verifiable in the repo.

### 2.1 Backend: `src/nav/sim/`

Three Python modules, ~1800 LOC total:

- **`render.py`** (~840 LOC) — top-level scene composition.
  - `render_combined_model(...)` is the entry point that composes
    bodies, rings, stars, noise, and background-stars into a single
    image array.
  - `render_stars(...)` rasterizes a list of stars with a single
    `psfmodel.GaussianPSF` (configurable `psf_sigma`).  Smear is
    implemented as a per-star translation `(move_v, move_u)` —
    **not** a true line-integral convolution, so smeared stars are
    rendered as translated points rather than line segments.
  - `render_background_noise(img, noise_level, seed)` adds **purely
    additive Gaussian noise** with `np.random.default_rng(seed)`.
    There is no Poisson term, no read-noise floor, no cosmic-ray
    injection, no missing-data marker injection.
  - `render_background_stars(...)` synthesizes a uniform random
    background-star field (not from a real catalog).
  - Caching is via `functools.lru_cache` on parameter hashes; same
    inputs → same outputs in-process, but cross-process determinism
    relies on the seed parameter being passed explicitly.

- **`sim_body.py`** (~480 LOC) — single-body silhouette renderer.
  - `create_simulated_body(axis1, axis2, ...)` renders an
    **elliptical** silhouette (two axes only — there is no third
    axis or 3D shape).  No polyhedral mesh support, no DSK
    integration, no actual irregular silhouette.
  - Lambertian shading via `_lambertian_shading` with optional
    crater overlay via `_add_craters_and_shading` (random craters
    drawn into the brightness map; craters do *not* perturb the
    silhouette).
  - The body is positioned by the caller via `(dv, du)` translation
    of the rendered patch.

- **`sim_ring.py`** (~440 LOC) — geometric ring rendering with
  per-pixel anti-aliasing of the radial edge.  Computes ring-edge
  radii from mode-1 / mode-N analytical models.  Independent of the
  per-planet ring catalog the navigator consumes.

### 2.2 GUI: `src/main/nav_create_simulated_image.py`

Single 2509-LOC PyQt6 application:

- `CreateSimulatedImageModel` is a `QMainWindow` with a tabbed
  parameter pane (General + per-body + per-ring + a `'+'` add-tab)
  and a live preview panel with zoom/pan.
- General tab carries: image size (V/U), offset, random seed,
  closest planet (for ring geometry), time + epoch (TDB seconds),
  background noise intensity (single Gaussian-amplitude slider),
  background star count + PSF sigma + radial-distribution exponent.
- Per-body tabs (added via `_add_body_tab`) carry the
  `create_simulated_body` parameters: axis1/axis2, position, crater
  density, albedo, etc.
- Per-ring tabs (added via `_add_ring_tab`) carry mode amplitudes,
  inner/outer radii, etc.
- The GUI saves a model spec as JSON plus the rendered image as a
  PNG / FITS pair via `nav_create_simulated_image` from the CLI
  side.

### 2.3 Navigator-side glue

- `src/nav/obs/obs_inst_sim.py` — `ObsSim` subclass of
  `ObsSnapshotInst`; loads a sim image and presents it through the
  same interface as Cassini / Voyager / Galileo / NHLORRI obs.
- `src/nav/dataset/dataset_sim.py` — `DataSetSim` registered under
  the `'sim'` dataset name.
- `src/nav/nav_model/nav_model_body_simulated.py` — body NavModel
  variant that renders the predicted body silhouette using the same
  sim renderer; used so the navigator's body overlay matches the
  sim image when both are derived from the same parameters.
- `src/nav/nav_model/nav_model_rings_simulated.py` — analogous for
  rings.

### 2.4 Configuration

- `src/nav/config_files/config_440_sim.yaml` (32 lines).  This is a
  *separate* config from the per-instrument configs
  (`config_400_inst_coiss.yaml`, `config_410_inst_gossi.yaml`,
  `config_420_inst_nhlorri.yaml`, `config_430_inst_vgiss.yaml`).
  The sim's noise model, PSF, ADC ceiling, etc. are sim-specific
  rather than mirrored from the instrument it's pretending to be.

### 2.5 Test coverage

A handful of unit tests in `tests/nav/nav_model/test_nav_model_body_simulated.py`
and similar for rings.  No scene-spec catalog, no rendered-image
regression baselines, no sensitivity-sweep harness, no algorithmic-
invariant test set keyed off planted offsets.

### 2.6 Known limitations relative to real holdings

Direct consequences of section 2.1:

| Real-image phenomenon | Sim today | Gap impact on calibration |
|---|---|---|
| Body silhouette deviates from ellipsoid | Always ellipsoidal | `phase_irregularity_factor` always 0; can't exercise irregular-body regime |
| Body rotational pose | Driven by SPICE pose at midtime | Chaotic rotators (Hyperion) have wrong orientation in sim too — same problem as real |
| Star PSF | Single Gaussian | Wings, diffraction spikes absent |
| Star centroid offset from catalog | Zero by construction | Can't exercise unique-match assignment ambiguity |
| Per-pixel noise | Additive Gaussian, signal-independent | No Poisson, read floor, cosmic rays, dropouts; MAD-noise estimator overfits to a regime that doesn't exist in real frames |
| Smear on long exposure | Per-star translation | Real smear is a line integral; centroid bias and SNR distribute differently |
| Saturation / bloom | Not modeled | Saturated bright stars don't get flagged in sim; no full-well clipping |
| Stray-light gradient | Not modeled | `scattered_light` scene class can't be sim'd |
| Missing-data markers | Not modeled | Partial-dropout classifier can't be exercised |

---

## 3. Cardinal principles

These principles are non-negotiable for the rest of the plan.

### 3.1 Sim is verification, real data is calibration

The `αᵢ` coefficients in `config_510_techniques.yaml` are tuned
*only* against the operator-curated real-holdings library.  Sim
images are used to *verify* that calibration (sensitivity sweeps,
algorithmic invariants, regression coverage) and to *bootstrap*
α to a sensible starting point — never to determine the final
calibrated values.  Any phase of this plan that crosses this line is
out of scope.

### 3.2 Per-instrument coupling

Today the sim has its own
`config_440_sim.yaml`, separate from the per-instrument configs.
This means a sim image isn't a "Cassini NAC raw frame", it's a "sim
frame".  The plan converges on **the sim consuming the same
per-instrument configs the navigator uses**, so a "sim COISS NAC raw"
frame goes through the same noise-sigma branch, the same
`signal_dn_to_image_unit_scale` path, and the same predicted-SNR
formula as a real CISS frame.  Without this, sim sensitivity analysis
isn't transferable to real-image interpretation.

### 3.3 Determinism

Every random call (noise, cosmic rays, star jitter, dropout markers)
takes a single per-scene seed and produces byte-identical output for
identical inputs.  No module-level RNG state, no `time.time()`
seeding fallbacks.  Required so sim images can participate in the
rounded-baseline regression layer.

### 3.4 Three peers, not one

The sim's parameter surface has **three equally-valid entry points**:

1. **Python API** (`render_combined_model`, etc.).
2. **YAML scene specs** (under
   `tests/integration/sim_scenes/<class>/<name>.yaml`, mirroring the
   real-image library's structure).
3. **GUI** (`nav_create_simulated_image`).

Every backend parameter must be reachable from all three.  Adding a
new physical effect means adding the parameter to the rendering
function, the YAML schema, *and* a GUI control.  The GUI is the
operator-friendly view; the YAML is the durable catalog artifact;
the API is what tests call.

### 3.5 Phase backend before GUI

GUI controls that aren't backed by real backend parameters are lies.
Backend parameters that the GUI doesn't expose are operator-invisible.
Each phase pairs a backend addition with the corresponding GUI work,
and the GUI work cannot start until the backend lands (otherwise
the controls have nothing real to drive).

### 3.6 No backwards-compat shims

Per the project's cardinal principles (Part 0 of `AUTONAV_PLAN.md`):
when sim parameters are renamed or restructured, the YAML / GUI / API
all change in the same PR.  No `if old_param: ... elif new_param:`
adapters.  Existing sim scenes either get migrated in the same PR or
are deleted.

---

## 4. Backend improvement plan

Phases are listed in *dependency order* — earlier phases unblock
later ones — not in priority order.  Within each phase the work is
small enough to land as one or two PRs.

### Phase B0: Determinism and seed audit (prerequisite for everything)

**Goal:** establish that every random call in the sim is seeded
through a single per-scene seed parameter and produces byte-identical
output for identical inputs.

**Scope:**
- Audit every `np.random.*`, `random.*`, and `default_rng(...)` call
  in `src/nav/sim/`.  Any that doesn't take a seed traceable to the
  scene's `random_seed` parameter is a determinism bug; fix it.
- Add a unit test (`tests/nav/sim/test_sim_determinism.py`) that
  renders the same scene twice and asserts byte-identical output.
- Add a per-effect sub-seed derivation:
  `noise_seed = hash((scene_seed, 'noise'))`, etc., so adding a new
  randomized effect (cosmic rays in B1) doesn't perturb the noise
  output of pre-existing scenes.

**Why first:** every later regression test relies on this.  Without
it, "the sim image changed" is ambiguous — was it the seed walking,
or did B5's PSF change really affect this scene?

**Files touched:** `src/nav/sim/render.py`, `src/nav/sim/sim_body.py`,
`src/nav/sim/sim_ring.py`, new `tests/nav/sim/test_sim_determinism.py`.

### Phase B1: Realistic noise model

**Goal:** replace pure additive Gaussian noise with the real
camera-noise structure.

**Scope:**

- **Poisson shot noise on the signal.** After all bodies / rings /
  stars are composed, sample each pixel from a Poisson distribution
  with mean equal to the noise-free signal in DN.  Required so that
  noise grows with brightness — the regime real cameras have.  Use
  `numpy.random.Generator.poisson` with the per-effect seed from B0.
- **Gaussian read-noise floor** added on top of Poisson at
  `read_noise_dn` from the per-instrument config (B2's coupling makes
  this pull from the right config block).  Until B2 lands, default
  to `config_440_sim.yaml`'s value.
- **Cosmic-ray hits** as sparse single-pixel spikes at a configurable
  fluence (events / cm² / sec × exposure_sec × pixel area).  Use a
  Poisson-distributed count of hits, each placed at a uniformly
  random pixel with a single-pixel intensity drawn from a long-tail
  distribution (Pareto or log-normal).  Spike intensity exceeds
  saturation by design — this exercises the orchestrator's
  `cosmic_ray_mask` path.
- **Missing-data markers** at the instrument's marker value (0 for
  raw CISS, NaN for CALIB) at a configurable rate.  Spread as
  isolated pixels (most common in real archives) plus optionally a
  contiguous-block "row dropout" mode for the partial-dropout
  classifier.

**Backend interface:**

```python
def render_background_noise(
    img: NDArrayFloatType,
    *,
    read_noise_dn: float,         # additive Gaussian sigma
    cosmic_ray_rate_per_sec: float = 0.0,  # events / cm² / sec
    exposure_sec: float = 1.0,
    pixel_area_cm2: float = 1.0,
    missing_data_marker_dn: float = 0.0,
    missing_data_rate: float = 0.0,
    seed: int,
) -> None:
    ...
```

The Poisson term is applied implicitly to the signal already in
`img` (no separate parameter — Poisson is always on whenever signal
is present).

**Why now:** this is the single biggest lever in the plan.  Without
it, every per-feature SNR diagnostic is in the wrong regime, and
sim-derived sensitivity analysis doesn't transfer to real frames.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/config_files/config_440_sim.yaml`, new
`tests/nav/sim/test_sim_noise.py` (asserts the noise distribution
matches the parameters).

### Phase B2: Per-instrument coupling

**Goal:** the sim consumes the same `config_4N0_inst_*.yaml` files
the navigator uses, so a "sim COISS NAC raw" frame goes through the
same code paths a real CISS NAC raw frame does.

**Scope:**

- The sim takes an `instrument` parameter (string: `'coiss_nac'`,
  `'coiss_wac'`, `'coiss_calib_nac'`, `'coiss_calib_wac'`,
  `'gossi'`, `'nhlorri'`, `'vgiss'`).  This selects which
  per-instrument config block drives the rendering: noise sigma /
  read-noise / saturation_dn / marker_value / data_units / star
  PSF / mag_offset / extfov_margin.
- `config_440_sim.yaml` becomes a small wrapper that defines the
  *defaults* used when no instrument is specified, plus
  sim-specific knobs that don't belong on real instruments (e.g.
  `closest_planet` for ring geometry).  All physical parameters
  delegate to the per-instrument config.
- `obs_inst_sim.py` exposes `inst_config` populated from the
  selected per-instrument block, so the orchestrator's
  `instrument_settings_from_obs(obs)` returns the right
  `InstrumentSettings` (noise, saturation_dn, signal scale,
  rotation flag).

**Why now:** every later phase that adds physics (smear, saturation,
PSF, etc.) needs to look up the right per-instrument value.  Doing
B2 first means each later phase's parameter wiring is a one-line
config lookup, not a duplicate sim-side knob.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/sim/sim_body.py`, `src/nav/obs/obs_inst_sim.py`,
`src/nav/dataset/dataset_sim.py`, `src/nav/config_files/config_440_sim.yaml`,
new `tests/nav/sim/test_sim_instrument_coupling.py`.

### Phase B3: Smear as a true convolution

**Goal:** rendered smeared stars become line integrals of the PSF,
matching the navigator's `smeared_psf.py` model.

**Scope:**

- `render_stars` no longer translates the PSF by `(move_v, move_u)`.
  Instead, build the smeared kernel via the same
  `nav.nav_model.stars.smeared_psf.compute_smeared_psf(psf, move_v,
  move_u)` the navigator uses, and rasterize that.
- The kernel's total flux is preserved (peak DN drops as smear
  length grows), matching the real photometry.
- Smear length is per-star (different stars in the FOV have
  different smear vectors when the camera rotates during the
  exposure).  Today the GUI applies one global smear; backend should
  accept per-star.

**Why now:** unblocks honest calibration of `StarUniqueMatchNav` and
`StarFieldFromCatalogNav` against scenes with non-trivial smear.
Today's translated-PSF approximation gives star centroids the wrong
sub-pixel position and the wrong SNR, which would systematically
mistune the star-related α coefficients if sim were used in
calibration.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_smear.py` (asserts a smeared star's
brightness-weighted centroid lies along the smear track).

### Phase B4: Saturation + bloom

**Goal:** clip pixels at the per-instrument full-well DN, with
column-direction bloom on cameras that have it.

**Scope:**

- After all noise is added, clip pixel values at
  `saturation_dn` from the per-instrument config (B2's coupling).
- For cameras with documented column-bloom behavior (Cassini NAC at
  high SNR), spread saturated-pixel excess flux into the column
  direction up to a configurable bloom-length.
- Saturated pixels should land on `saturation_mask_ext` when the
  navigator processes the sim image — the orchestrator's
  `_build_saturation_mask` already does the comparison; this phase
  just makes sure the pixels actually hit the threshold.

**Why now:** unblocks the saturated-bright-star reliability gate
(STAR features whose predicted pixel is in the saturation mask get
zeroed reliability per `_reliability_from_snr`).  Without B4, sim
star-cal scenes always navigate cleanly because no star ever
saturates.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_saturation.py`.

### Phase B5: Realistic per-instrument PSF

**Goal:** sim stars use the same `psfmodel.PSF` instance the
navigator uses, with instrument-specific wings.

**Scope:**

- Replace `GaussianPSF(sigma=...)` in `render.py` with the PSF
  returned by `obs.star_psf()` for the selected instrument (B2 makes
  this lookup possible).  For Cassini NAC this means a real
  measured-from-data PSF if one is in `psfmodel`; for other
  cameras, the configured `star_psf_sigma` parameterized PSF.
- Optional: diffraction spikes for very-bright stars on Cassini NAC
  (cross-shaped spikes at SNR > some threshold).  Configurable
  on/off so other cameras can opt out.

**Why now:** brings star-centroid diagnostics into the same
distribution as real CISS frames.  Less load-bearing than B1 and B2,
but matters for `StarFieldFromCatalogNav` calibration verification.

**Files touched:** `src/nav/sim/render.py`, new
`tests/nav/sim/test_sim_psf.py`.

### Phase B6: Stray-light gradient

**Goal:** optional per-frame stray-light contribution that exercises
the BANDPASS_DOG source-image filter.

**Scope:**

- A new optional `stray_light` scene parameter taking
  `(amplitude, gradient_direction_deg, model='linear' | 'radial')`.
  Linear = a brightness ramp across the frame; radial = a brightness
  bump centered at a configurable point (mimics a sun-near-FOV
  flare).
- Applied multiplicatively to the noise-free image before noise is
  added.

**Why now:** the only way to test that the navigator's
source-image filter (`config_4N0_inst_*.yaml` `source_image_filter:
BANDPASS_DOG`) actually mitigates scattered light without finding a
real Galileo or Voyager outer-leg frame.

**Files touched:** `src/nav/sim/render.py`,
`src/nav/config_files/config_440_sim.yaml`, new
`tests/nav/sim/test_sim_stray_light.py`.

### Phase B7: Non-ellipsoidal bodies

**Goal:** render at least one canonical irregular body
(Hyperion-like, Phoebe-like) from a polyhedral mesh rather than as
an ellipsoid silhouette.

**Scope:**

This phase is paired with `AUTONAV_PLAN.md` Part 13b §7 (resolved
irregular-body LIMB_ARC via real shape models).  Two deployment
paths:

- **DSK-via-oops path.** When `oops` adds DSK kernel support, the
  sim's `sim_body.py` consumes the same DSK and renders the actual
  silhouette.  At the same time the navigator's body NavModel
  consumes the same DSK and predicts the actual silhouette.  Both
  sides are then ellipsoid-free for that body.
- **Standalone polyhedral renderer fallback.** Until DSK lands, the
  sim grows a small polyhedral renderer (project triangle vertices
  through the `oops`-supplied pose into image space, draw the
  silhouette).  Mesh files live under
  `src/nav/sim/shape_meshes/<BODY>.obj` (or .ply).  This is sim-only
  — the navigator still uses the ellipsoidal model — and is
  documented as a *deliberate* shape-mismatch test fixture: "render
  the actual Hyperion shape so we can measure how badly the
  navigator's ellipsoidal silhouette mis-fits".  This is the
  scenario that exercises `phase_irregularity_factor`.

**Why now:** the `phase_irregularity_factor` term added in Phase 10
§F is identically zero on every sim frame today.  Without B7, that
term cannot be calibrated, sensitivity-tested, or regression-covered
on sim.

**Files touched:** new `src/nav/sim/sim_body_polyhedral.py`,
`src/nav/sim/render.py`, `src/nav/sim/shape_meshes/`, new
`tests/nav/sim/test_sim_irregular_body.py`.

### Phase B8: Diffraction spikes (optional)

**Goal:** Cassini NAC's cross-shaped diffraction pattern on very
bright stars.

**Scope:** configurable on/off per camera; only enabled for cameras
with documented spike behavior.  Spike length scales with star
brightness above some threshold.

**Why now:** lowest-leverage of the phases.  Nice-to-have for
visualization; doesn't shift any calibration diagnostic meaningfully.

---

## 5. GUI improvement plan

Each GUI phase consumes one or more backend phases.  The phasing
mirrors the backend phasing — never start a GUI phase before its
backend is in.

### Phase G0: Three-peer audit

**Goal:** confirm every existing GUI control is also reachable from
the YAML scene-spec schema (Phase T1 below) and the Python API.
Where there's drift, fix it.

**Scope:** for each control in `nav_create_simulated_image.py`
(general tab, per-body tab, per-ring tab), verify the corresponding
field exists in the YAML schema and in the API.  Where it doesn't,
add it.

**Why now:** keeps the three peers from drifting out of sync as the
later phases land.  Cheap to do once; expensive to retrofit later.

**Files touched:** `src/main/nav_create_simulated_image.py`, scene
schema docs.

### Phase G1: Instrument selector

**Goal:** add a top-level "Instrument" combo box that drives the
per-instrument config used for the rest of the rendering.

**Scope:**

- New combo with options `coiss_nac`, `coiss_wac`,
  `coiss_calib_nac`, `coiss_calib_wac`, `gossi`, `nhlorri`,
  `vgiss`, `(generic)`.
- Selecting an instrument updates the General tab's defaults to
  that instrument's PSF sigma, image dimensions, ADC bits,
  saturation DN, etc.
- Existing per-tab parameters that the instrument doesn't override
  (random seed, body / ring tabs) stay as the operator left them.

**Backend dependency:** B2 must land first.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G2: Noise-model controls

**Goal:** expose the B1 noise model in the GUI.

**Scope:** the General tab's single "Background noise intensity"
slider expands into a small sub-panel:

- Poisson shot noise (boolean — usually on)
- Read-noise floor (DN, slider 0–20)
- Cosmic-ray rate (events / sec, slider 0–10; auto-derived from
  exposure_sec)
- Missing-data rate (fraction, slider 0–0.30)
- Missing-data block size (single pixels vs. row blocks)

When G1's instrument selector changes, these populate from the
selected instrument's defaults.

**Backend dependency:** B1, B2.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G3: Smear controls

**Goal:** expose B3's per-star smear correctly.

**Scope:** today the "Background stars" tab has a single PSF sigma
spin and no smear control (smear is only applied via the body's
motion vector).  Add:

- "Camera smear (global)" two-spinner row: smear vector v / u in px.
  Applied to every star and to the catalog-star overlay.
- For per-body tabs, the existing motion vector becomes a true
  smear (per B3) rather than a translation.
- Live preview shows the smear track on a representative star.

**Backend dependency:** B3.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G4: Saturation visualization

**Goal:** show the operator which pixels saturate.

**Scope:**

- New checkbox in the General tab: "Show saturation overlay".
  When checked, pixels at or above the per-instrument
  `saturation_dn` are highlighted in red on the live preview.
- The status bar shows the per-image saturation fraction.

**Backend dependency:** B4.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G5: PSF preview pane

**Goal:** let the operator see what PSF the sim is using.

**Scope:**

- New collapsible pane (initially closed) showing the rendered
  current per-instrument PSF as a small inset image, with the FWHM
  / sigma annotated.  Updates when the instrument selector changes.
- Optional toggle to also show the smeared PSF for the current
  global smear setting (B3 + B5).

**Backend dependency:** B5.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G6: Stray-light controls

**Goal:** expose B6's stray-light gradient.

**Scope:**

- New "Stray light" panel in the General tab: amplitude slider,
  gradient direction (dial widget), model dropdown (linear /
  radial).
- Defaults to off.

**Backend dependency:** B6.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G7: Shape-file selector for irregular bodies

**Goal:** let the operator pick "Hyperion (polyhedral mesh)"
instead of just "Hyperion (ellipsoid)" for a per-body tab.

**Scope:**

- Per-body tab gets a new dropdown: "Shape model" with options
  `ellipsoid`, `polyhedral_mesh: HYPERION`, `polyhedral_mesh:
  PHOEBE`, etc. — populated from the meshes available under
  `src/nav/sim/shape_meshes/`.
- When `polyhedral_mesh` is selected, the axis-1 / axis-2 sliders
  are greyed out (the mesh determines the silhouette); pose comes
  from `oops` at the configured midtime as before.

**Backend dependency:** B7.

**Files touched:** `src/main/nav_create_simulated_image.py`.

### Phase G8: Scene catalog browser

**Goal:** the GUI becomes a first-class editor for the YAML scene
catalog (Phase T1 below).

**Scope:**

- New "File" menu items: "Open Scene…" (browses
  `tests/integration/sim_scenes/`), "Save As…" (writes a YAML scene
  spec), "Save & Render" (renders the saved scene).
- Loading a YAML scene populates every GUI tab from the YAML.
- Saving from the GUI writes a YAML spec the test harness can
  consume.

This is the phase that makes the GUI a peer of the YAML, not just a
control surface.  After G8, an operator who renders an interesting
scene in the GUI can save it as a regression artifact with one click.

**Backend dependency:** T1 (the YAML schema must exist first).

**Files touched:** `src/main/nav_create_simulated_image.py`, plus
the YAML schema introduced by T1.

---

## 6. Calibration test strategy

This is the section that turns the upgraded sim into useful
calibration / verification infrastructure.  Phases T1–T7.

### Phase T1: Scene-spec YAML catalog

**Goal:** durable artifact directory for sim scenes, mirroring the
real-image library's layout.

**Scope:**

- New directory `tests/integration/sim_scenes/`.
- Subdirectories per scene class:
  `tests/integration/sim_scenes/<scene_class>/<scene_name>.yaml`.
  Scene classes are scoped to "what we're calibration-testing", not
  the real-image library's `DECLARED_SCENE_CLASSES`.  Initial set:
  - `phase_sweep_regular_body/` — same body at increasing phase
    angles (0, 30, 60, 90, 120, 150°).
  - `phase_sweep_irregular_body/` — same Hyperion-mesh body at
    increasing phase angles.
  - `noise_sweep/` — same scene at increasing read-noise levels.
  - `smear_sweep/` — same scene at increasing smear lengths.
  - `range_sweep/` — same body at increasing distances (resolves
    the BLOB threshold transition).
  - `multi_body_geometry/` — controlled multi-body arrangements.
  - `algorithmic_invariants/` — clean planted-offset scenes for
    technique unit tests.
- YAML schema documented in
  `tests/integration/sim_scenes/README.txt` (analogous to
  `images/README.txt` for the real library).  Key fields:

  ```yaml
  schema_version: 1
  scene_name: <stem must match filename>
  instrument: coiss_nac      # see G1
  random_seed: 42
  midtime_utc: '2010-01-01T00:00:00Z'
  exposure_sec: 1.0

  bodies:
    - name: HYPERION
      shape_model: polyhedral_mesh   # or 'ellipsoid'
      ...
  rings: [...]
  stars:
    catalog: ucac4
    smear_v_px: 0.0
    smear_u_px: 0.0
  noise:
    poisson: true
    read_noise_dn: 4.0
    cosmic_ray_rate_per_sec: 0.0
    missing_data_rate: 0.0
  stray_light:
    amplitude: 0.0
    direction_deg: 0.0
    model: linear

  ground_truth:
    planted_offset_dv_px: 0.0   # see T4
    planted_offset_du_px: 0.0
    planted_rotation_deg: 0.0
  ```

- Validator at `tests/integration/sim_scene.py` (parallel to
  `tests/integration/sidecar.py`) that loads + validates each YAML.
- Structural-invariants test
  (`tests/integration/test_sim_scenes.py`) that asserts every
  YAML validates and every directory name is a known scene class.

**Why now:** turns sim scenes from one-off Python fixtures into
catalog artifacts that the GUI (G8), the test harness (T2–T7), and
human reviewers can all consume.

**Files touched:** new `tests/integration/sim_scenes/`, new
`tests/integration/sim_scene.py`, new
`tests/integration/test_sim_scenes.py`.

### Phase T2: Deterministic regression baselines

**Goal:** sim scenes get a regression-baseline layer parallel to
the real-image library's `tests/integration/baselines/`.

**Scope:**

- New directory `tests/integration/sim_baselines/<scene_name>.json`.
- Each baseline records the rounded
  `(offset_dv_px, offset_du_px, confidence)` the orchestrator
  produces when navigating the sim'd scene.
- Test layer
  `tests/integration/test_sim_baselines.py` parametrized one case
  per `(scene, baseline)` pair; renders the scene, runs the
  navigator, asserts exact-equal rounded match.
- Updater under
  `tests/integration/update_sim_baselines.py` (parallel to
  `tests/integration/update_baselines.py` from earlier work);
  invoked as `python -m tests.integration.update_sim_baselines …`.

**Why now:** a sim scene catalog without baselines doesn't gate
regressions — anyone can rerun and get any result.  Baselines turn
"this scene navigates to X" into a tripwire.

**Files touched:** new
`tests/integration/sim_baselines/`, new
`tests/integration/test_sim_baselines.py`, new
`tests/integration/update_sim_baselines.py`.

### Phase T3: Single-variable parameter sweeps

**Goal:** harness that renders a sweep of scenes varying one
parameter, navigates each, and emits a per-parameter response curve.

**Scope:**

- Sweep specs are themselves YAML files at
  `tests/integration/sim_sweeps/<sweep_name>.yaml`.  Each sweep
  declares a base scene (referencing a `sim_scenes/` template), a
  parameter name, and a list of values.
- Harness at
  `tests/integration/sim_sweep_runner.py` (also runnable as
  `python -m`) that renders the sweep, navigates each frame, and
  emits a JSON or CSV of `(parameter_value, offset_error_px,
  confidence, primary_technique)` per row.
- Test layer
  `tests/integration/test_sim_sweeps.py` that asserts invariants
  per sweep.  Examples:
  - `phase_sweep_regular_body` — confidence should be roughly flat
    across phase 0–60° and decrease modestly above 60°.
  - `phase_sweep_irregular_body` — confidence should drop more
    sharply with phase (the `phase_irregularity_factor` term is
    doing its job).
  - `noise_sweep` — confidence monotonically decreases with
    `read_noise_dn`.
  - `range_sweep` — primary technique transitions from BodyDisc to
    BodyLimb to BodyBlob at the right diameter thresholds.

**Why now:** this is the verification layer the calibration sweep
relies on.  After fitting α against the real-image library, run
these sweeps and assert the calibrated formulas respond smoothly to
single-parameter changes.  A non-monotonic or jagged response
signals a calibration bug.

**Files touched:** new `tests/integration/sim_sweeps/`, new
`tests/integration/sim_sweep_runner.py`, new
`tests/integration/test_sim_sweeps.py`.

### Phase T4: Algorithmic invariants (planted-offset / planted-rotation)

**Goal:** unit-test the techniques against ground truth that's
correct *by construction*.

**Scope:**

- Scene specs in `sim_scenes/algorithmic_invariants/` carry
  `ground_truth.planted_offset_dv_px` / `planted_offset_du_px` /
  `planted_rotation_deg`.
- The sim renders at
  `(predicted_pose ± planted_offset, predicted_rotation ± planted_rotation)`.
- The test asserts the navigator recovers the planted values to
  within a tight tolerance (e.g. 0.1 px, 0.01°).
- Coverage targets:
  - `BodyDiscCorrelateNav` — clean disc, planted (dv, du), assert
    recovery to <0.1 px.
  - `BodyLimbNav` — limb fit on planted offset.
  - `BodyBlobNav` — BLOB on planted offset; assert lit-weighted
    centroid recovers correctly even at high phase.
  - `RingEdgeNav` — curved-ring planted offset.
  - `StarFieldFromCatalogNav` — N planted stars + planted rotation,
    assert both translation and rotation recovered.
  - `StarUniqueMatchNav` — single bright star, planted offset.
  - `StarRefineNav` — pre-existing prior + small planted refinement.

**Why now:** these tests replace many of the inline-constructed
fixtures in `tests/nav/nav_technique/` with catalog-driven scenes,
and they grow the test suite faster than operator labour can grow
the real library.

**Files touched:** new
`tests/integration/sim_scenes/algorithmic_invariants/*.yaml`, new
`tests/integration/test_sim_algorithmic_invariants.py`.

### Phase T5: α-bootstrap pre-fit

**Goal:** use sim-derived diagnostic distributions to give the
real-data calibration sweep a sensible starting point for the α
optimization.

**Scope:**

- Run a representative set of sim scenes through the navigator.
- For each scene, record the per-technique diagnostics.
- Use these to pre-fit the α coefficients via
  `scipy.optimize.curve_fit` against an *operator-supplied
  approximate target* per scene (e.g. "this clean Mimas should land
  high; this irregular Hyperion should land low").  The targets are
  rough by design — the sim can't supply real-world tier targets, but
  it can rule out obviously-wrong α regions.
- The output α is **not** committed to
  `config_510_techniques.yaml`.  It's used as the initial guess for
  the real-data calibration in Phase 10 §C, replacing the current
  arithmetic-illustrative placeholder coefficients.
- Documented as a separate one-off helper script under
  `tests/integration/sim_alpha_bootstrap.py`.

**Why now:** shrinks the optimization basin for the real-data
calibration without committing to sim's diagnostic distributions.
Optional but recommended; skip if the operator finds the placeholder
coefficients already close enough.

**Files touched:** new
`tests/integration/sim_alpha_bootstrap.py`.

### Phase T6: Real-vs-sim diagnostic distribution comparison

**Goal:** quantify how close sim has come to real, per technique.

**Scope:**

- For each technique, harvest the diagnostic values reported on:
  - Every real-image library sidecar (run the navigator over the
    real library).
  - Every sim-scene catalog entry (run the navigator over the sim
    scenes).
- Plot per-diagnostic histograms side-by-side (real vs. sim).
- Report Kolmogorov–Smirnov distances per diagnostic.
- The output is a Markdown report at
  `tests/integration/sim_vs_real_report.md` regenerable on demand.

**Why now:** this is the empirical answer to "how good is the sim
*now*".  Before any backend phase lands, run T6 to get a baseline.
After each backend phase lands, re-run T6 and report the change.
The KS distance per diagnostic measures whether a backend
improvement actually closed the gap it was supposed to.

**Files touched:** new
`tests/integration/sim_vs_real_diagnostics.py`,
`tests/integration/sim_vs_real_report.md`.

### Phase T7: Calibration validation harness

**Goal:** after the real-data Phase 10 §C calibration sweep lands,
verify the calibrated formulas pass the sim sweeps from T3.

**Scope:**

- Same sweep infrastructure as T3.
- Assertions tightened:
  - Monotonicity in the expected direction for each parameter.
  - Smoothness (no per-sample jumps > some configurable epsilon).
  - The three regimes (high / medium / low confidence) appear in
    the right portions of each sweep — e.g. the noise sweep should
    cross from high to medium at the noise level corresponding to
    the threshold, not somewhere arbitrary.
- Failures are *advisory* (informational on the PR, not
  CI-blocking) until the operator certifies the calibration is
  stable enough that sim sensitivity tests can gate.

**Why now:** turns the sim sweeps from "useful diagnostic" into
"calibration-validation tripwire".  Catches calibration
regressions that don't show up on the real-image library because
the real library doesn't sample the parameter space densely enough.

**Files touched:** extends
`tests/integration/test_sim_sweeps.py`.

---

## 7. Phase ordering and dependencies

The phases map onto a dependency graph.  Earlier phases unblock
later ones:

```
B0 (determinism)
 │
 ├──→ B1 (noise model)
 │     │
 │     └──→ B4 (saturation), B5 (PSF), B6 (stray light), B7 (irregular)
 │
 └──→ B2 (per-instrument coupling)
       │
       └──→ B3 (smear), B4, B5, B6, B7

T1 (scene catalog)
 │
 ├──→ T2 (regression baselines)
 ├──→ T3 (sweeps)
 ├──→ T4 (algorithmic invariants)
 │
 └──→ G8 (GUI catalog browser)

G0 (3-peer audit) before any GUI phase
G1 follows B2
G2 follows B1, B2
G3 follows B3
G4 follows B4
G5 follows B5
G6 follows B6
G7 follows B7

T5 (α bootstrap) — anytime after B1, B2, T1
T6 (real-vs-sim) — anytime after T1; ideally after each B phase
T7 (calibration validation) — last; depends on real-data calibration landing
```

### Minimum viable cut

If the work has to be sequenced over multiple releases, the smallest
useful self-contained slice is:

1. **B0** (determinism)
2. **B1** (noise model)
3. **B2** (per-instrument coupling)
4. **G0** (3-peer audit)
5. **G1, G2** (instrument selector + noise GUI)
6. **T1** (scene catalog)
7. **T2** (regression baselines)
8. **T4** (planted-offset invariants)
9. **T6** (real-vs-sim baseline report)

With this slice, the sim becomes a deterministic, instrument-aware,
realistic-noise simulator with a YAML-catalog-driven test layer that
can grow algorithmic-invariant coverage without operator labour.
Items B3 / B4 / B5 / B6 / B7 / T3 / T5 / T7 / G3–G8 each add a
specific incremental capability that can be deferred.

### Recommended interleaving

Within a single release the natural interleaving alternates one
backend phase with its matching GUI phase:

1. B0 → tests pass, no GUI change
2. B1 → G2 (noise controls)
3. B2 → G0 + G1 (3-peer audit + instrument selector)
4. T1 → G8 (catalog browser)
5. T2, T4, T6 (test infrastructure; no GUI change)
6. B3 → G3 (smear)
7. B4 → G4 (saturation)
8. B5 → G5 (PSF preview)
9. B6 → G6 (stray light)
10. B7 → G7 (irregular bodies)
11. T3, T5, T7 (sweeps, bootstrap, validation)
12. B8 (diffraction; optional polish)

---

## 8. Out of scope (and why)

These items sound like they belong in a sim-improvement plan but
explicitly do *not* move the calibration needle, so they are
deferred indefinitely:

- **Hapke / Lommel-Seeliger / Akimov photometric models.** The
  navigator works on gradients and silhouettes; replacing Lambert
  with a more accurate photometric function gives better-looking
  pictures but doesn't shift any diagnostic the calibration depends
  on.  Polish, not load-bearing.
- **Fixed-pattern noise / detector banding.** Captured by the
  noise-sigma estimator as elevated noise; the calibration
  response is essentially the same as B1's contribution.
- **JPEG / lossy compression artifacts.** Real archives don't
  ship JPEGs; this is moot.
- **Color rendering.** The navigator ignores color (per-camera
  `mag_offset` already absorbs band conversion).  Color is
  presentation polish.
- **Multi-frame trajectory rendering.** Render a sequence of frames
  along an encounter trajectory.  Useful for testing batch behavior
  (worker pool, cloud_tasks queue dynamics) but not for per-image
  calibration.  Deferred to a separate plan if the cloud-tasks layer
  needs it.
- **Atmospheric body navigation (TitanNav stub).** Tracked in
  `AUTONAV_PLAN.md` Part 13b §3 and won't be exercised by sim until
  the navigator side is implemented.

---

## 9. Acceptance criteria

A phase is considered done when **all** of these are true:

1. Code changes land in the project's standard format
   (`ruff check`, `ruff format`, `mypy`, `sphinx-build -W` all
   clean).
2. New unit tests cover the new functionality with the same density
   as adjacent code (target ≥ 90% line coverage on new files).
3. Backend phases: `tests/nav/sim/test_sim_*` covers the
   determinism guarantee, the parameter ranges, and at least one
   "renders without exception across realistic input ranges" smoke
   test.
4. GUI phases: at least one UI smoke test that constructs the
   `CreateSimulatedImageModel` and exercises the new control's
   change-handler.  GUI tests are inherently fragile; the bar is
   "the control wires to the right backend parameter", not "the
   pixmap looks right".
5. Test phases: the harness runs as `python -m
   tests.integration.<module>` from a clean checkout and emits the
   expected JSON / Markdown / per-test summary.
6. Documentation:
   - Backend phases update
     `tests/integration/sim_scenes/README.txt` (after T1) with the
     new YAML field if applicable.
   - GUI phases update screenshots in the developer guide if one
     exists; otherwise leave a one-line note in the GUI's docstring.
   - This `SIM_IMPROVEMENT_PLAN.md` file gets a checkmark next to
     the completed phase in section 7.
7. The phase's output is consumable by the next phase that depends
   on it (verify by spot-running one of the dependent phase's
   acceptance tests against the new state).

---

## 10. Concrete file additions / changes summary

For a quick scan of what this plan touches:

### New files
```
tests/integration/sim_scenes/                            (T1)
tests/integration/sim_scenes/README.txt                  (T1)
tests/integration/sim_scenes/<class>/<name>.yaml         (T1+)
tests/integration/sim_baselines/                         (T2)
tests/integration/sim_baselines/<name>.json              (T2+)
tests/integration/sim_sweeps/                            (T3)
tests/integration/sim_sweeps/<name>.yaml                 (T3+)
tests/integration/sim_scene.py                           (T1)
tests/integration/sim_sweep_runner.py                    (T3)
tests/integration/sim_alpha_bootstrap.py                 (T5)
tests/integration/sim_vs_real_diagnostics.py             (T6)
tests/integration/sim_vs_real_report.md                  (T6 output)
tests/integration/test_sim_scenes.py                     (T1)
tests/integration/test_sim_baselines.py                  (T2)
tests/integration/test_sim_sweeps.py                     (T3, T7)
tests/integration/test_sim_algorithmic_invariants.py     (T4)
tests/integration/update_sim_baselines.py                (T2)
tests/nav/sim/test_sim_determinism.py                    (B0)
tests/nav/sim/test_sim_noise.py                          (B1)
tests/nav/sim/test_sim_instrument_coupling.py            (B2)
tests/nav/sim/test_sim_smear.py                          (B3)
tests/nav/sim/test_sim_saturation.py                     (B4)
tests/nav/sim/test_sim_psf.py                            (B5)
tests/nav/sim/test_sim_stray_light.py                    (B6)
tests/nav/sim/test_sim_irregular_body.py                 (B7)
src/nav/sim/sim_body_polyhedral.py                       (B7)
src/nav/sim/shape_meshes/                                (B7)
src/nav/sim/shape_meshes/<BODY>.obj                      (B7+)
```

### Modified files
```
src/nav/sim/render.py              (B0, B1, B3, B4, B5, B6, B7, B8)
src/nav/sim/sim_body.py            (B0, B2, B7)
src/nav/sim/sim_ring.py            (B0)
src/nav/obs/obs_inst_sim.py        (B2)
src/nav/dataset/dataset_sim.py     (B2)
src/nav/config_files/config_440_sim.yaml  (B1, B2, B6)
src/main/nav_create_simulated_image.py  (G0, G1, G2, G3, G4, G5, G6, G7, G8)
```

### No changes
```
The operator-curated real-image library at
tests/integration/image_library/ — by cardinal principle 3.1, the
sim work does not modify or replace the real library.  The
calibration target stays anchored to real data.
```

---

## 11. Discussion notes preserved for context

This section captures the reasoning that led to the plan above,
including dead-ends and explicit rejections, so a future reader can
see why specific decisions were made and which alternatives were
considered.

### 11.1 Why not just calibrate against sim?

The thought experiment that motivated this plan was: "the sim can
produce hundreds of controlled images; why curate real ones at
all?"  The rejection rests on three observations:

1. The α coefficients in the confidence formula encode "what
   diagnostic value `x` corresponds to what tier".  If sim
   diagnostics distribute differently from real (and they do —
   ellipsoidal bodies, perfect-Gaussian PSFs, additive Gaussian
   noise, no irregularity, no scattered light), α tuned against sim
   gives the wrong answer on real.
2. Verifying that "α tuned against sim works on real" requires the
   real-image library anyway.  So sim doesn't replace the labour,
   it just front-loads it.
3. The tier labels themselves (`high` / `medium` / `low` / `failed`)
   are judgments about real-world performance.  What does `high`
   mean on a sim'd Hyperion?  The sim's idealized-Hyperion-as-an-
   ellipsoid would always navigate `high`; the real-Hyperion is
   `low`.  Tier targets assigned to sim scenes don't reflect the
   real-world performance the calibration is supposed to map onto.

The compromise is the augment-not-substitute architecture in this
plan.  Sim covers what it's good at (controlled sweeps,
algorithmic invariants, regression coverage); the real library
remains the calibration target.

### 11.2 Why a YAML scene catalog instead of inline Python?

Today's sim tests construct scenes inline.  The reasons to move to
YAML:

- **Reviewability.**  A reviewer can read a YAML scene and
  understand it without reading Python.  Inline-Python scenes hide
  parameters in helper-function defaults.
- **GUI/test/library parity.**  The same YAML file can be loaded by
  the GUI for visualization, by the test harness for assertions,
  and by the regression layer for baselining.  Inline Python forces
  the GUI to re-implement the scene by hand.
- **Catalog growth.**  The real-image library's directory-as-registry
  pattern works well; the sim catalog mirrors it.  Adding a scene
  is one YAML file.
- **Versioning.**  YAML scenes are diffable in git.  Inline Python
  scene constructors couple to refactors of the helper functions.

### 11.3 Why instrument-coupling (B2) is so high in the order

The original draft had B2 much later — "nice-to-have polish".  On
review, every later phase that adds physics needs to look up the
right per-instrument value (noise sigma, PSF, full-well, marker
value, signal scale).  Without B2 each later phase grows a
duplicate sim-side knob, then a second pass has to reconcile the
two.  Doing B2 second (after determinism) means the later phases
each pull from one place.

### 11.4 Why determinism (B0) is the first phase

Anything that adds randomness later (cosmic rays in B1, stray-light
phase noise) silently changes the output of every previously-
existing test.  If the determinism contract isn't established
first, B0 becomes a forensic exercise: "this baseline drifted —
why?".  Doing it first makes B1 and beyond auditable: a baseline
change is either explainable by a documented physics addition or
it's a bug.

### 11.5 Why the GUI work is mandatory rather than optional

The GUI exists because operators want to dial parameters and see
the result live.  Backend additions that the GUI can't show are
operator-invisible.  If we land B1 (realistic noise) without G2,
the only way an operator sees the new noise model is to write a
Python script — which they won't.  Bundling the GUI work with the
backend work means each release lands a *complete* operator-visible
capability, not a backend half-feature.

The cost of this discipline is real: every backend PR also touches
the GUI file, which is large and PyQt-heavy.  The alternative —
stockpile backend PRs and do the GUI later — looks attractive
short-term but produces a GUI file that's permanently out of sync.

### 11.6 Why the irregular-body phase (B7) is so far down

Three reasons:

1. **It's gated on `oops` DSK support, which is out of our control.**
   The B7 fallback (standalone polyhedral renderer) is doable but
   significant: ~500 LOC of mesh loading, projection, silhouette
   extraction, anti-aliasing.
2. **The `phase_irregularity_factor` term it would exercise is
   already in place in the navigator** (Phase 10 §F's BLOB work).
   It's calibration-relevant but the term defaults to 0 in
   sim-ellipsoid scenes, which means existing calibration work isn't
   blocked on B7 — it just doesn't get to *test* the irregular path
   on sim until B7 lands.
3. **One real Cassini Phoebe / Hyperion close-flyby image in the
   real-image library would cover the irregular-body case for
   calibration purposes.**  B7 is the right answer when the gap
   between "what the real library covers" and "what we want to
   sweep with sim" specifically includes irregular-body
   sensitivity, which is a Phase 11+ concern.

If irregular-body sensitivity becomes a hot calibration question
sooner, B7 can be promoted up the order.

### 11.7 Why diffraction spikes (B8) are explicitly last

They look impressive in the rendered image but don't shift any
diagnostic the calibration cares about.  Pure polish.  Listed in
the plan only so they don't get accidentally promoted later
("we should add diffraction spikes — should we?  no, see B8").

### 11.8 What about the existing simulator users?

`nav_create_simulated_image` today is used (rarely) for one-off
illustrative renders.  The plan's GUI changes are all additive —
existing controls stay where they are, new tabs / panels appear
alongside.  The biggest behavior change for existing users is the
instrument selector (G1), which defaults to `(generic)` so existing
workflows are untouched unless the operator opts in.

### 11.9 Estimated effort

Rough order-of-magnitude scoping (single full-time developer,
including tests + docs + reviews):

| Phase | Estimated effort |
|---|---|
| B0 | 1 day |
| B1 | 3 days |
| B2 | 4 days |
| B3 | 1 day |
| B4 | 1 day |
| B5 | 2 days |
| B6 | 1 day |
| B7 | 2 weeks |
| B8 | 1 day |
| G0 | 1 day |
| G1 | 1 day |
| G2 | 2 days |
| G3 | 1 day |
| G4 | 1 day |
| G5 | 2 days |
| G6 | 1 day |
| G7 | 1 day |
| G8 | 3 days |
| T1 | 2 days |
| T2 | 2 days |
| T3 | 3 days |
| T4 | 3 days |
| T5 | 2 days |
| T6 | 3 days |
| T7 | 2 days |

**Minimum viable cut total**: ~16 days.
**Full plan total**: ~50 days, dominated by B7 (irregular-body
mesh renderer).  Skipping B7 brings the full plan to ~36 days.

These are rough — the actual numbers depend heavily on how
ergonomic the existing code is to extend, how hard `oops`
integration turns out to be for B2, and how much new GUI design
work G2 / G8 require.  Use as a relative-ranking guide, not a
schedule commitment.

---

## 12. Open questions — operator decisions before starting

Items the implementer should get explicit answers on before
starting:

1. **Cardinal-principle confirmation.** Does the
   "augment-not-substitute" stance (section 3.1) reflect the
   project's intent?  If the operator wants sim to be the primary
   calibration target, the plan needs to be reframed entirely.
2. **Per-instrument config split** (B2).  The plan assumes the
   `config_440_sim.yaml` block shrinks to a wrapper.  An alternative
   is to keep `config_440_sim.yaml` as the sole source of truth for
   sim parameters and *copy* the per-instrument values across at
   load time.  The first option keeps the configs in sync
   automatically; the second isolates the sim from instrument-config
   changes.  Operator preference?
3. **Polyhedral mesh source for B7.**  Per-body ``.obj`` files
   bundled in the repo, or fetched on demand from a per-mission
   shape archive?  Bundled is reproducible but adds repo size
   (Hyperion mesh is ~5 MB at typical resolution); on-demand
   requires network access during test runs.
4. **Sim baseline policy** (T2).  Should sim baselines be
   regenerated automatically when the sim's noise model changes
   (B1), or should each backend change explicitly call out which
   sim baselines need re-blessing?  The latter is more
   conservative; the former is faster.
5. **Sweep failure policy** (T3, T7).  Should sweep-invariant
   failures CI-block immediately, or be advisory until the
   real-data calibration is also stable?  The plan assumes
   advisory-then-promote-to-blocking; operator can flip this.

When these are decided, the corresponding sections of this plan
should be updated with the operator's chosen direction so the
implementer has a clear contract.
