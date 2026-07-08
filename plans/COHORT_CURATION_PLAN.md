# Cohort Curation Plan: metadata-driven image discovery for validation and calibration

**Audience.** A fresh AI session with no prior context, working in this repo
with an operator (rfrench) available only for brief review votes. Read
`/seti/newnav/CLAUDE.md` first, then `plans/VALIDATION_AND_CALIBRATION_PLAN.md`
(the methodology this plan feeds) and `plans/PHASE10_CURATION.md` (the sidecar
mechanics this plan automates the front half of).

**Goal.** Populate every image cohort the validation and calibration program
needs — the 49-image curated library (#172), its WS-3 growth to >=20 per
instrument / >=120 total, and the WS-1 agreement-study cohorts — by
**automated search over the published PDS geometry metadata**, so the operator
does nothing but look at overlay PNGs and vote yes/no. Arbitrary or convenient
images are not acceptable: every cohort must span a **wide variety** of moons,
rings, planets, illumination, viewing geometry, resolution, and filters
(quotas below), because the validation math bins by exactly those axes and a
narrow cohort silently invalidates it.

---

## 1. Data sources (all local, no network needed)

| Resource | Path |
|---|---|
| PDS3 holdings (images) | `/mnt/ganymede/PDS/holdings` (set `PDS3_HOLDINGS_DIR` to this) |
| Geometry metadata tables | `/mnt/ganymede/PDS/holdings/metadata/<VOLSET>/<VOLUME>/` |
| SPICE kernels | `/mnt/ganymede/SPICE` (`SPICE_PATH`) |
| UCAC4 star catalog | `/mnt/ganymede/UCAC4` (`UCAC4_PATH`) |
| OOPS resources | set `OOPS_RESOURCES` per CI workflow if no local mirror |

Volume sets by instrument: `COISS_1xxx` (Cassini Jupiter leg) and `COISS_2xxx`
(Cassini Saturn) for COISS; `VGISS_5xxx`/`6xxx`/`7xxx`/`8xxx` (Voyager
Jupiter/Saturn/Uranus/Neptune); `GO_0xxx` (Galileo SSI); `NHxxLO_xxxx`
(New Horizons LORRI). Use the unversioned directories (e.g. `COISS_2xxx`, not
`COISS_2xxx_v1.0`).

Each volume directory carries five table pairs (`.lbl` describes columns,
`.tab`/`.csv` is the data). **Always parse the `.lbl` to get authoritative
column positions per volume set** — layouts differ between missions. Verified
for COISS_2xxx:

- `*_moon_summary.tab` — one row per (image, moon). Key columns (1-based):
  2 `FILE_SPECIFICATION_NAME`, 4 `TARGET_NAME`, 5-10 planetocentric/graphic
  lat and IAU lon min/max, 15-18 surface resolution min/max, 21-22 phase
  angle min/max, 23-26 incidence/emission min/max, 33 `CENTER_RESOLUTION`
  (km/px), 34 `CENTER_DISTANCE`, 35 `CENTER_PHASE_ANGLE`. **`-999` means the
  quantity is undefined for that image (typically: the body is in the FOV
  inventory but not resolved/on-disk)** — filter on it deliberately: lat/lon
  columns at -999 with a valid center distance means an unresolved point
  target; valid lat/lon ranges mean resolved surface in frame.
- `*_ring_summary.tab` — one row per image with rings in FOV. Key columns:
  RA/dec min/max (4-7), `MINIMUM/MAXIMUM_RING_RADIUS` (8-9), radial
  resolution, ring longitude min/max, phase/incidence/emission, and the
  solar/observer **ring opening angles** (last columns).
- `*_saturn_summary.tab` (or the mission's planet summary) — same shape as
  moon rows, for the planet disc.
- `*_inventory.csv` — per image, the full list of bodies in the FOV: the
  fastest way to require or exclude bodies (multi-body scenes, "no body"
  star frames).
- `*_index.tab` — per image: exposure, filters, camera, pointing RA/dec.
  Use it for filter/exposure diversity and for star-frame pointing.

Image path construction: `FILE_SPECIFICATION_NAME` gives
`data/.../N1454725799_1.LBL`; the raw image swaps `.LBL` for `.IMG` under
`volumes/<VOLSET>/<VOLUME>/`, and the Cassini calibrated variant lives under
`calibrated/<VOLSET>/<VOLUME>/...<name>_CALIB.IMG`. Sidecar `image_url` uses
the `pds3://` form (see any existing sidecar in
`tests/integration/image_library/images/`). Voyager must use the geometrically
corrected (`GEOMED`/calibrated) products — see PHASE10_CURATION's
mission-specific hints.

Body radii for apparent-size computation: use the values already in
`src/spindoctor/config_files/config_220_body_shape.yaml` and oops, not a
hand-typed table. Apparent diameter in pixels = 2 * R_body /
`CENTER_RESOLUTION`.

Ring edge radii for "does an edge cross this frame": read the per-planet ring
catalogs `src/spindoctor/config_files/config_3N0_*_rings.yaml` — do not
hardcode edge radii in the search script.

**Proven feasible:** a 20-line Python scan of 35 COISS volumes' moon summaries
found 2,223 `high_phase_terminator` candidates (phase 110-155 deg, apparent
diameter 150-800 px) in seconds. The same pattern covers ~15 of the 17 scene
classes; the star classes additionally need a UCAC4 count at the index-table
pointing (predicted detectable stars given the instrument magnitude limit —
reuse `spindoctor.nav_model.stars` catalog plumbing rather than reimplementing).

---

## 2. Cohorts to build and what each feeds

One physical library (`tests/integration/image_library/`), tagged per image;
cohorts are queries over the tags, not separate directories beyond the
existing scene classes. Consult VALIDATION_AND_CALIBRATION_PLAN WS-0/WS-1/WS-3
for the full rationale.

| Cohort | Definition (metadata query sketch) | Feeds |
|---|---|---|
| **Scene-class library, 49 images** | the 17 classes in PHASE10_CURATION's budget table, found per class as in section 3 | #172 regression seed, #173 calibration diagnostics, #174 baselines |
| **WS-3 growth** | same classes continued to >=20/instrument, >=120 total | WS-1 statistics, WS-4 CI tiers |
| **Route 1: intra-body** | one resolved moon, apparent diameter 150-900 px, full limb or >=30% arc, phase < 90 for limb+disc pairs and > 90 to add terminator; **restricted to round, photometrically bland bodies** per the config_220 shape gate (`ellipsoid_rms_residual_km`, `crater_scale_km`, `albedo_variation`) | WS-1 Route 1 (technique pairs on one body; SPICE cancels) |
| **Route 2: body + ring** | inventory has >=1 moon AND ring_summary row with a catalog edge radius inside [MIN,MAX]_RING_RADIUS; the bulk cohort — collect widely | WS-1 Route 2 (ring-radial axis) |
| **Route 3: star tie-points** | star count >= 3 at pointing AND (resolved moon OR ring edge) in frame; scarce — sweep exhaustively, keep every hit | WS-1 Route 3 (absolute attitude, ephemeris probe) |
| **Multi-body** | inventory has >=2 resolved moons (each diameter > 50 px), non-occluding | WS-1 orthogonal axis, inter-moon ephemeris |
| **Over-determined** | >=2 resolved moons AND a ring edge in one frame (rarest; sweep everything, keep every hit) | WS-1 closure test — the only real-photon assumption check |
| **WS-1b sequences** | 5-20 consecutive frames of one body with overlapping footprints (sort moon_summary by time within a volume; same target, overlapping lat/lon boxes), no second fiducial needed | WS-1b reprojection consistency |
| **WS-17 star fields** | >=20 predicted stars across the FOV, no body/ring (empty inventory); Cassini (and LORRI if counts allow) only | WS-17 distortion validation via plate-solve residuals |
| **WS-2 realism set** | any representative real frames per instrument spanning exposure/filter/noise regimes (no geometry requirement — it is a statistics match) | WS-2 sim-realism validation |

## 3. Diversity requirements (mandatory, not aspirational)

The WS-1 solve bins by resolution / phase / lit-fraction /
limb-orientation-vs-radial, assumes within-bin stationarity, and needs
populated bins on every axis. A cohort of thirty Enceladus frames from one
flyby is worthless. Enforce at query time:

- **Targets:** every resolved moon the archive offers per mission (Saturn
  system: at minimum Mimas, Enceladus, Tethys, Dione, Rhea, Iapetus; plus
  irregulars Phoebe/Hyperion/Janus/Epimetheus for the shape-contaminated
  bucket — kept separate, never in Route 1). Jupiter: Galilean four. Include
  the planet discs. Cover **all four ring systems** where the mission saw
  them (Saturn primary; Jupiter/Uranus/Neptune rings are Phase-2 models —
  collect candidates now, navigate later).
- **Illumination:** phase bins at least {<30, 30-60, 60-90, 90-120, >120}
  deg with several images per populated bin per target class; both
  low-incidence and terminator-dominated lighting.
- **Viewing geometry:** for rings, spread ring opening angle (near edge-on
  through wide-open) and ring longitude; for moons, spread
  limb-orientation-vs-ring-radial (compute from pointing + geometry at query
  time or post-hoc) because WS-1 bins on it.
- **Resolution:** at least three decades (e.g. <1 km/px close flybys,
  1-10 km/px mid-range, >10 km/px distant) per mission where available.
- **Filters/exposure:** do not let one filter dominate; include the CALIB and
  RAW Cassini variants per the PHASE10 saturation policy; include long and
  short exposures for the star classes.
- **Time:** spread over the mission (different SPK/CK eras), not one
  encounter.
- **Stratified sampling:** when a query returns thousands (it will — 2,223
  hits for one class in 35 volumes), sample the strata (target x phase-bin x
  resolution-decade x filter), never take the top-N rows. Seed any random
  sampling and record the seed.
- **Provenance:** every sidecar's `notes` records the query criteria and
  metadata values that selected it, so cohort composition is reproducible.

## 4. Workflow: minimal operator interaction

The operator's entire job is: look at a PNG, vote yes/no, optionally comment.
Target under 30 seconds per image, in batches of 20-30. Everything else is
automated.

**Stage A — query.** Scripted scan of the metadata tables per cohort
(section 2), stratified sampling (section 3), producing a candidate manifest
(CSV/YAML: image id, path, class, selecting metadata values).

**Stage B — automated triage (no operator).** For each candidate, run the
autonomous pipeline locally (`sd_offset` with the local mounts; ~35 s/frame,
so batch overnight if needed). Auto-drop, without operator review: frames the
feasibility gates reject, majority-missing-data frames, saturated-bloom
frames, and frames whose scene class the actual geometry contradicts. Keep
the per-frame `_metadata.json` and the summary PNG. A candidate is only
promoted to review when the pipeline produced a proposed offset AND the
overlay looks self-consistent by machine checks (fit residuals, technique
agreement) — or when the frame is scarce (Route 3 / over-determined), in
which case promote it even on failure and flag it for manual navigation.

**Stage C — operator review.** Produce one review batch at a time:
`_work/cohort_review/batch_NNN/` containing, per image, a single composite
PNG (red image / green model overlay AT the proposed offset — the same
rendering the manual-nav "Save as Library Entry" writes, plus the proposed
`(dv, du)`, class, and any triage warnings burned into the image margin),
plus a pre-filled `votes.yaml` listing every image with `vote: null` and an
empty `comment:`. The operator opens the PNGs, edits `votes.yaml` to `y`/`n`
(+ optional comment), and hands it back — or simply votes in chat
("1-14 y; 15 n, limb misaligned"). Never ask the operator to run tools, drag
overlays, or fill schemas; that is the AI's job. Reserve `sd_offset --manual`
for the scarce-cohort frames flagged in Stage B, and queue them so the
operator does all manual work in one sitting.

**Stage D — sidecar generation (no operator).** For each `y`: write the
sidecar into the right scene-class directory with `ground_truth` from the
reviewed offset (`source: operator_verified`, the vote date, the reviewed
PNG kept beside the YAML), auto-filled `expected.*` per the PHASE10 rubric
(conservative tier — `medium` when unsure; tier labels are plausibility
cross-checks, never calibration fit targets), and the selection provenance in
`notes`. Run `pytest tests/integration/test_image_library.py -m "" -k <id>`
until structurally clean. Submit in PRs of 5-10 sidecars (reviewer cost
rubric in PHASE10_CURATION).

**Stage E — baselines and consumption.** After sidecar PRs merge, seed
regression baselines (`python -m tests.integration.update_baselines`), then
hand off per consumer: calibration diagnostics collection for #173 (WS-5
methodology — reliability diagrams against measured error anchors, never
tier-midpoint fitting), agreement-study runs per WS-1's harness plan, WS-17
plate solves on the star-field cohort. CI stays tiered per WS-4's
"Library consumers and CI tiers" note: the full library and all offline
analyses never run per-PR.

## 5. Order of work

1. Fill the eight empty scene classes of the 49-image budget first (list in
   PHASE10_CURATION "Concrete next steps"), one review batch.
2. Top up all classes to the per-class minima; verify mission spread
   (>=1 image from each of the four missions; >=1 Cassini `_CALIB`).
3. Sweep for the scarce cohorts (Route 3 star tie-points, over-determined
   frames) across ALL volumes — these are kept whenever found, and they gate
   the most valuable science (absolute attitude, closure test).
4. Build Route 1 / Route 2 / multi-body / WS-1b cohorts to the WS-3 target
   (>=20/instrument, >=120 total), stratified per section 3.
5. WS-17 star fields and the WS-2 realism set (no ground-truth votes needed
   for WS-2 — it is a distributional match, so it can be fully automated).

Checkpoint with the operator between numbered steps, not within them.

## 6. Known traps

- `-999` sentinels in every summary table; never treat them as values.
- A body in `inventory.csv` may be unresolved or behind the planet; always
  cross-check the moon_summary row before classifying.
- Metadata tables describe the PREDICTED geometry from archived SPICE — the
  same kernels the navigator corrects. Fine for finding scenes; never treat
  metadata geometry as ground truth for offsets.
- Scene classes are decided by the expected PRIMARY technique
  (PHASE10_CURATION per-class table), not by what happens to be in frame;
  when a frame straddles classes, pick the class that exercises the primary
  technique and record the judgment.
- Do not let the autonomous proposal bias ground truth on frames where it is
  systematically wrong (the known ~0.1 px limb bias, WS-10): the vote
  verifies pixel-level alignment, and sub-pixel systematics are exactly what
  WS-1/WS-2 measure — record `offset_uncertainty_px` honestly (1.0 px
  default, 2.0 px for soft features), never 0.1 px.
- Voyager: only geometrically corrected products; Galileo/Voyager carry
  camera-rotation fitting (slow) — budget triage time accordingly.
