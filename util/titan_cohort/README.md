# Titan real-frame validation cohort

The Cassini frames the haze navigator's accuracy claim is measured on, and
the tooling that navigates them and reports what they proved.

`titan_images.csv` is the cohort itself: 82 frames vendored from a legacy
annotated test list, one row each, carrying the flags this repo reads off the
original freeform annotation (`rings_occluding`, `moon_occluding`,
`high_phase`, `near_edge`, `off_edge`, `known_bad`, `clean`) with that
annotation preserved verbatim. Frames the annotation warned about are as much
of the point as the clean ones: an occulted or clipped Titan must refuse to
navigate, and the cohort is what shows it does.

## Running a campaign

```bash
source /seti/newnav/setup.sh
python util/titan_cohort/cohort.py                       # resolve + self-check
python util/titan_cohort/collect.py --workers 10 \
    --campaign-dir _work/titan_cohort/run1
python util/titan_cohort/analyze.py _work/titan_cohort/run1/rows.jsonl \
    --out _work/titan_cohort/run1/report.md
```

The whole cohort takes about ten minutes on ten workers. `collect.py` runs
the full autonomous pipeline -- every model, every technique -- so each frame
writes the same `*_metadata.json`, `*_summary.png`, and per-image log a
production run writes, plus one distilled JSON line per frame for the
analyzer. Those outputs are large and are not committed; the campaign record
and the report are.

`cohort.py` resolves each image id to its holdings path and epoch by scanning
the per-volume PDS3 index tables, so nothing in the cohort file has to record
a path that could go stale.

## What the analyzer measures

Offsets are measured against each frame's own SPICE prediction, so a
commanded pointing difference between two frames is already removed and two
frames of the same scene should agree.

- **Tier (a), star-anchored.** A prior-free star technique locking on the
  same frame is an independent measurement of the same scene-wide
  translation. The per-axis 2-sigma test between it and the haze offset is
  the strongest per-frame truth available without an operator eyeball.
  `StarRefineNav` is excluded: it is seeded by the pass-1 prior, which on a
  Titan frame is usually the haze answer itself.
- **Tier (b), within-sequence.** Clean frames of the same target within 30
  minutes.
- **Tier (c), cross-filter.** Near-simultaneous frames through different
  filters. This is the direct test of the method's filter-independence
  claim, and it is the tier that found the wavelength-dependent haze top.
- **Companion-body witnesses**, reported separately and not counted in the
  acceptance fractions: a body technique locking on another moon in the same
  frame is the same physics as the star anchor, but the validation plan
  states its bound over star anchors.

## The operator review batch

```bash
python util/titan_cohort/build_review_batch.py \
    _work/titan_cohort/run1/rows.jsonl --campaign-dir _work/titan_cohort/run1
```

Writes `review_batch/`: a stratified sample spread over filter combinations
and phase bins, one annotated preview per frame, a manifest CSV, and a
`votes.yaml` whose votes are null. Nothing here fills those votes in.

## Library nominations

```bash
python util/titan_cohort/build_nominations.py
```

Writes `nominations/`: one draft sidecar per candidate frame, built from the
calibrated product the library holds. They stay here rather than under
`tests/integration/image_library/images/` because a library sidecar's ground
truth must be operator-verified and these carry an autonomous fix. Promoting
one means verifying the offset, choosing the scene class, and moving the
file.

## What is committed

`titan_images.csv` (the cohort), the tooling, `README.md`,
`CAMPAIGN_20260726.md` (the campaign record: every sweep, every knob it
moved, and every open question), `final_run_summary.csv` (one row per frame
of the shipped configuration's run, so a past result is readable without
re-running it), and the two pending-operator directories. Campaign
directories themselves are not.
