Navigation Statistics
=====================

The statistics system turns the per-image metadata JSON files written by the
navigation pipeline into a local SQLite database and a deterministic report
(Markdown text plus PNG charts). It is the standing quality check on a
production run: success and failure rates, which techniques and models carry
the load, offset distributions, how well the techniques agree with one
another, and whether the confidence tiers behave as designed.

Two commands cooperate over the database and are cron-friendly (each is a
single non-interactive invocation):

.. code-block:: bash

    # Ingest one or more navigation-results roots into ./nav_stats.sqlite3
    sd_stats_ingest /data/nav-offset-results --db nav_stats.sqlite3

    # Generate report.md + charts for any slice of the database
    sd_stats_report --db nav_stats.sqlite3 --output-dir stats_report
    sd_stats_report --db nav_stats.sqlite3 --instrument coiss \
        --start-date 2005-03-01 --end-date 2005-03-01 --output-dir day_report

Ingestion
---------

``sd_stats_ingest ROOT [ROOT ...] [--db PATH]`` scans each root recursively
for ``*_metadata.json`` files (the documents ``navigate_image_files`` writes
under ``nav_results_root``) and loads them into the database. Roots may be
local directories or any URL the project's ``filecache`` layer accepts, so
cloud-hosted results ingest the same way as local ones.

Ingestion is idempotent: the database holds one row per image (keyed by image
name), and re-ingesting the same or an updated metadata file replaces that
image's row and its child rows rather than duplicating them. Malformed files
are logged and skipped; the exit status is nonzero only when nothing at all
was ingested.

The schema has three tables:

- ``images`` — one row per image: instrument (from the metadata document's
  ``observation.instrument`` field, falling back to filename-shape
  classification for documents that lack the field), UTC image date (derived
  from the SPICE epoch), status and failure
  reason, fused offset / sigma / confidence / tier, ensemble consensus
  exclusions, image-classifier verdict, and run provenance (config hash, git
  SHA, pipeline timestamp).
- ``techniques`` — one row per technique result: offset, per-axis sigma,
  confidence, spurious / at-edge flags, the body or ring or catalog names it
  used, and the full diagnostics dict as JSON.
- ``feature_sources`` — per image, the feature counts by (feature type,
  source model, source name), with how many were gated, so body / ring /
  star-catalog usage can be aggregated.

Reporting
---------

``sd_stats_report [--db PATH] [--output-dir DIR] [--instrument NAME]
[--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]`` writes ``report.md`` and
its charts into the output directory. All filters combine; dates are
inclusive UTC image dates, so a single day's run is
``--start-date D --end-date D``. The same inputs always produce the same
numbers and the same charts.

The report contains:

- **Success / failure counts** with a breakdown of failure reasons.
- **Technique usage** — runs, non-spurious runs, and mean confidence per
  technique.
- **Model and source usage** — which bodies, rings, and star catalogs
  appeared, in how many images, and how many of their features survived the
  reliability gate.
- **Offset statistics** — mean, median, standard deviation, minimum, and
  maximum of the fused V and U offsets over successful images, with
  histograms.
- **Cross-technique agreement** — for every technique pair, the median and
  95th-percentile Euclidean distance between their offsets on images where
  both produced non-spurious results.
- **Confidence calibration** — per confidence tier, the distribution of each
  image's maximum cross-technique disagreement. Without ground truth,
  agreement between independent techniques is the production proxy for
  accuracy (the calibrated anchor is the simulation campaign; see the
  developer guide's calibration documentation): a healthy pipeline shows
  high-tier images agreeing tightly and disagreement growing toward the low
  tier.
- **Ensemble outlier exclusions** — how often the ensemble excluded a
  technique from the consensus, and which techniques.
