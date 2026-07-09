# Cohort curation tooling

Automation for `plans/COHORT_CURATION_PLAN.md`: metadata-driven discovery of
image-library candidates, autonomous-pipeline triage, and operator review
batches. These scripts are workflow tooling, not part of the `spindoctor`
package; they are excluded from the packaged distribution and from the
`src`/`tests` lint targets.

All generated artifacts (candidate manifests, triage results, review
batches, votes) are written under `_work/`, which is gitignored. Only the
scripts and the static support file `body_radii.json` are tracked.

## Environment

Source `/seti/newnav/setup.sh` first (holdings, OOPS resources, star
catalogs, venv). The scripts hardcode the same paths for subprocess use.

## Pipeline (one review batch)

```bash
venv/bin/python util/cohort_curation/scan_stage_a.py
venv/bin/python util/cohort_curation/triage_stage_b.py --workers 3
venv/bin/python util/cohort_curation/build_review_batch.py --batch N
```

- `scan_stage_a.py` — Stage A: scans the PDS geometry metadata tables
  (`$PDS3_HOLDINGS_DIR/metadata/<VOLSET>/<VOLUME>/`) plus UCAC4 star counts
  at the index pointings, and writes a stratified, seeded candidate manifest
  to `_work/cohort_curation/candidates_batch001.yaml`.
- `triage_stage_b.py` — Stage B: runs `sd_offset` on every candidate,
  applies the machine accept/drop rules (missing data, saturation,
  feasibility, technique agreement; negative cases promote on clean
  failure), and writes `_work/cohort_curation/triage_report.yaml`.
  Already-triaged frames are reused unless `--force`.
- `build_review_batch.py` — Stage C: composes per-image annotated review
  PNGs and a pre-filled `votes.yaml` under
  `_work/cohort_review/batch_NNN/`. Batch size is capped at 100 images
  (per-class caps in `CLASS_CAPS`).

Support files:

- `pdsmeta.py` — minimal PDS3 label/table reader (ITEMS-aware column
  mapping; `.tab`/`.csv`; `-999`/`-1e32`/`-99.9999` sentinels).
- `body_radii.json` — mean body radii (km) dumped from oops/SPICE
  (`oops.Body.define_solar_system`); regenerate with any SPICE-enabled
  session if bodies are added.

## Gotchas encoded in the scripts

- The per-volume index uses `.IMG` filespecs while the summary tables use
  `.LBL`; joins strip the extension.
- Vector columns (`ITEMS = n`, e.g. `SC_*_POSITION_VECTOR`) occupy n CSV
  fields; `COLUMN_NUMBER` counts columns, not fields.
- A `ring_summary` row exists for nearly every frame (the ring PLANE
  crosses the FOV); visible main rings require the radius range to overlap
  the main-ring span.
- Cumulative `*_?999` pseudo-volumes duplicate every real volume's rows and
  are skipped.
- `sd_offset` image-name selection is `index_name.startswith(arg)` where
  the index name has no product/version suffix: pass `N1828132857`, not
  `N1828132857_1`.
